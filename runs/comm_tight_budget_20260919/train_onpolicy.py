"""The pilot's PPO, with the probability it samples from and the probability it logs made one.

Round 2 collected rollouts through ``mu(a|s) = 0.9*pi(a|s) + 0.1*Uniform(a)`` and stored
``log pi(a|s)``; round 3's protocol registered that as a live defect and trained on regardless,
because that round was permitted to change only duration. This file is the other half: the same
network, the same reward, the same budget and silence bound, with the behaviour policy replaced by
the policy being optimised.

What is deliberately *not* changed here, because this round must attribute an effect to one cause:

* the network (19 -> 64 -> 64, per-candidate send/hold head, pooled value head);
* the reward (one credit per target completed before its deadline, booked on its step);
* the clip, GAE, advantage scaling, epochs, minibatch size, learning rate and gradient clip;
* the entropy bonus, which is now the only exploration term and is why removing the epsilon
  overlay does not mean training greedily.

What is added is the bookkeeping the protocol asks for: the actionable set per decision, the
optimiser steps, free-candidate samples, and the drawn-versus-logged check that would have caught
the original defect in one line.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

import tight_common as common

import pilot_env
import train_pilot
from onpolicy import SEND, OnPolicyCollector, action_entropy, joint_log_prob, restore_rng, rng_state

FEATURE_DIM = train_pilot.FEATURE_DIM
GAMMA = train_pilot.GAMMA
GAE_LAMBDA = train_pilot.GAE_LAMBDA
CLIP = train_pilot.CLIP
VALUE_COEFFICIENT = train_pilot.VALUE_COEFFICIENT
ENTROPY_BONUS = train_pilot.ENTROPY_BONUS
MAX_GRAD_NORM = train_pilot.MAX_GRAD_NORM
LEARNING_RATE = train_pilot.LEARNING_RATE
EPISODES_PER_UPDATE = train_pilot.EPISODES_PER_UPDATE
EPOCHS_PER_BATCH = train_pilot.EPOCHS_PER_BATCH
MINIBATCH_STEPS = train_pilot.MINIBATCH_STEPS
EVALUATE_EVERY_STEPS = train_pilot.EVALUATE_EVERY_STEPS
STEPS_PER_EPISODE = common.STREAM_STEPS
#: Milestone checkpoints for a run that starts from scratch: every 100k environment steps.
CHECKPOINT_STEPS = (100_000, 200_000, 300_000, 400_000)


def collector_for(controller: Any, *, greedy: bool = False) -> OnPolicyCollector:
    """The one collector constructor this round uses, so no arm can pick a different sampler.

    ``forced`` is the pilot's own bound predicate, read from the frozen module rather than
    restated here: if the bound's numbers ever move, this moves with them instead of quietly
    training a controller whose mask no longer matches the transport's.
    """
    return OnPolicyCollector(
        controller, forced=train_pilot.forced, mask=True, greedy=greedy
    )


def drawn_versus_logged(
    controller: train_pilot.Controller, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Does the number in the tape describe the draw that produced the action?

    Recomputes the joint log-probability from the recorded features, actions and actionable set.
    A sampler that drew from a different distribution than the one it logged shows up here as a
    gap on the very first minibatch, before any weight moved -- which is the check round 2 lacked.
    """
    if not records:
        return {"records": 0}
    gaps: list[float] = []
    for record in records:
        logits, _values = controller.forward(
            torch.as_tensor(np.asarray(record["features"], dtype=np.float32))
        )
        with torch.no_grad():
            recomputed = float(
                joint_log_prob(
                    logits,
                    torch.tensor(
                        [record["actions"][target] for target in record["targets"]], dtype=torch.long
                    ),
                    torch.tensor(
                        [target in set(record["free"]) for target in record["targets"]],
                        dtype=torch.float32,
                    ),
                ).item()
            )
        gaps.append(recomputed - float(record["log_prob"]))
    return {
        "records": len(records),
        "mean_gap": round(float(np.mean(gaps)), 8),
        "max_abs_gap": round(float(np.max(np.abs(gaps))), 6),
    }


def gae(rewards: list[float], values: list[float]) -> tuple[list[float], list[float]]:
    """The pilot's advantage estimator, unchanged, kept here so the tape alignment is local."""
    return train_pilot.gae(rewards, values)


def flatten(episodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One row per decision, with its advantage, and the per-episode diagnostics beside it.

    Returns ``(flat, alignment_checks)``. The alignment is a per-tape recomputation, so a
    mismatch between the sampler and the likelihood fails the update rather than biasing it.
    """
    flat: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes):
        records = episode["decisions"]
        checks.append(drawn_versus_logged(episode["controller"], records))
        values = [record["value"] for record in records]
        if not values:
            continue
        advantages, returns = gae(episode["rewards"], values)
        scale = max(1e-6, float(np.std(advantages)))
        for record, advantage, target_return in zip(records, advantages, returns):
            flat.append(
                {
                    "features": np.asarray(record["features"], dtype=np.float32),
                    "targets": record["targets"],
                    "actions": record["actions"],
                    "free": set(record["free"]),
                    "log_prob": record["log_prob"],
                    "behaviour_value": float(record["value"]),
                    "advantage": advantage / scale,
                    "return": target_return,
                }
            )
    return flat, checks


def update(
    controller: train_pilot.Controller,
    optimizer: torch.optim.Optimizer,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """PPO over joint per-step actions, with the likelihood helpers the collector also uses."""
    flat, checks = flatten(episodes)
    alignment = {
        "max_abs_gap": max((int(item.get("records", 0)) and float(item["max_abs_gap"])) for item in checks)
        if any(int(item.get("records", 0)) for item in checks)
        else 0.0,
        "tapes": sum(int(item.get("records", 0)) for item in checks),
    }
    if alignment["max_abs_gap"] > 1e-4:
        raise RuntimeError(
            "the rollout's logged probability does not describe the action that was drawn: "
            f"{alignment}"
        )
    if not flat:
        return {"samples": 0, "minibatches": 0, "optimizer_steps": 0, "likelihood_alignment": alignment}
    returns = np.asarray([item["return"] for item in flat], dtype=float)
    behaviour_values = np.asarray([item["behaviour_value"] for item in flat], dtype=float)
    explained_variance = (
        1.0 - float(np.var(returns - behaviour_values) / max(1e-9, np.var(returns)))
        if len(returns) > 1 and float(np.var(returns)) > 1e-9
        else None
    )
    free_candidates = sum(len(item["free"]) for item in flat)
    order = list(range(len(flat)))
    totals = {"surrogate": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0, "clip": 0.0, "grad": 0.0}
    minibatches = 0
    for _ in range(EPOCHS_PER_BATCH):
        random.shuffle(order)
        for start in range(0, len(order), MINIBATCH_STEPS):
            chunk = [flat[index] for index in order[start : start + MINIBATCH_STEPS]]
            stacked = torch.as_tensor(np.concatenate([item["features"] for item in chunk], axis=0))
            logits, candidate_values = controller.forward(stacked)
            row = 0
            new_log_prob: list[torch.Tensor] = []
            entropy: list[torch.Tensor] = []
            pooled: list[torch.Tensor] = []
            for item in chunk:
                targets = item["targets"]
                rows = torch.arange(row, row + len(targets))
                row += len(targets)
                picked = torch.tensor([item["actions"][target] for target in targets], dtype=torch.long)
                usable = torch.tensor(
                    [target in item["free"] for target in targets], dtype=torch.float32
                )
                new_log_prob.append(joint_log_prob(logits[rows], picked, usable))
                entropy.append(action_entropy(logits[rows], usable))
                pooled.append(candidate_values[rows].mean())
            new_log_prob_t = torch.stack(new_log_prob)
            entropy_t = torch.stack(entropy)
            old_log_prob = torch.tensor([item["log_prob"] for item in chunk], dtype=torch.float32)
            advantage = torch.tensor([item["advantage"] for item in chunk], dtype=torch.float32)
            target_returns = torch.tensor([item["return"] for item in chunk], dtype=torch.float32)
            ratio = torch.exp(new_log_prob_t - old_log_prob)
            surrogate = torch.min(ratio * advantage, torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * advantage)
            value_loss = ((torch.stack(pooled) - target_returns) ** 2).mean()
            loss = -surrogate.mean() + VALUE_COEFFICIENT * value_loss - ENTROPY_BONUS * entropy_t.mean()
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(controller.parameters, MAX_GRAD_NORM)
            optimizer.step()
            totals["surrogate"] += float(surrogate.mean().item())
            totals["value"] += float(value_loss.item())
            totals["entropy"] += float(entropy_t.mean().item())
            totals["kl"] += float((old_log_prob - new_log_prob_t).mean().item())
            totals["clip"] += float(((ratio - 1.0).abs() > CLIP).float().mean().item())
            totals["grad"] += float(norm)
            minibatches += 1
    return {
        "samples": len(flat),
        "free_candidate_samples": int(free_candidates),
        "minibatches": minibatches,
        "optimizer_steps": minibatches,
        "epochs": EPOCHS_PER_BATCH,
        "policy_surrogate": round(totals["surrogate"] / minibatches, 5),
        "value_loss": round(totals["value"] / minibatches, 5),
        "entropy": round(totals["entropy"] / minibatches, 5),
        "approx_kl": round(totals["kl"] / minibatches, 6),
        "clip_fraction": round(totals["clip"] / minibatches, 4),
        "gradient_norm_mean": round(totals["grad"] / minibatches, 4),
        "value_explained_variance": round(explained_variance, 4) if explained_variance is not None else None,
        "likelihood_alignment": alignment,
    }


def run_episode(controller: train_pilot.Controller, seed: int, *, budget_bytes: int, directory: Path, label: str) -> dict[str, Any]:
    """One rollout episode: the collector is the only sender, so the tape is exactly the decisions."""
    collector = collector_for(controller)
    episode = train_pilot.run_one(
        seed,
        collector=collector,
        budget_bytes=budget_bytes,
        directory=directory,
        label=label,
    )
    episode["controller"] = controller
    episode["timing"] = collector.timing()
    episode["collector"] = collector
    return episode


def collect_batch(
    controller: train_pilot.Controller, batch_seeds: list[int], *, budget_bytes: int, directory: Path, training_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes, details = [], []
    for seed_value in batch_seeds:
        episode = run_episode(
            controller, seed_value, budget_bytes=budget_bytes, directory=directory,
            label=f"tight_train{training_seed}",
        )
        episodes.append(episode)
        details.append(episode_diagnostics(episode["collector"], episode["row"]))
    return episodes, details


def episode_diagnostics(collector: OnPolicyCollector, row: dict[str, Any]) -> dict[str, Any]:
    ledger = row["ledger"]
    timing = collector.timing()
    return {
        "decisions": int(collector.decisions),
        "candidates": int(collector.candidates),
        "free_candidates": int(timing["free_candidates"]),
        "actionable_share": round(timing["free_candidates"] / max(1, collector.candidates), 4),
        "drawn_send_rate": timing["voluntary_send_rate"],
        "masked_by_bound": int(collector.forced_sends),
        "emissions": int(ledger["emissions"]),
        "emissions_by_policy": int(ledger["emissions_policy"]),
        "emissions_forced": int(ledger["emissions_forced_maxwait"]),
        "refused_by_policy": int(ledger["refused_policy"]),
        "refused_forced": int(ledger["refused_forced_maxwait"]),
        "bytes": int(row["spent_bytes"]),
        "on_time": int(row["task"]["on_time"]),
        "targets": int(row["task"]["targets"]),
        "seconds": float(row["seconds"]),
        "steps_run": actual_steps(row["artefacts"]["lifecycle_log"]),
    }


def actual_steps(lifecycle_log: str) -> int:
    from integrated_sar.lifecycle_metrics import load_records

    records = load_records(Path(lifecycle_log))
    closing = [float(record.get("timestamp") or 0.0) for record in records if record.get("event_type") == "final"]
    if not closing:
        raise RuntimeError(f"{lifecycle_log}: no closing record, so the episode length is unknown")
    return int(max(closing))


def save(controller: train_pilot.Controller, optimizer: torch.optim.Optimizer, state: dict[str, Any], path: Path) -> None:
    torch.save(
        {
            "controller": controller.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": state,
            "trainer": "onpolicy",
        },
        path,
    )


def load(path: Path, training_seed: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("trainer") != "onpolicy":
        raise RuntimeError(
            f"{path} came from the epsilon-greedy collector; a fresh run must start from initial "
            "weights, and resuming it here would mix the two behaviour policies"
        )
    return payload


def train(args: argparse.Namespace) -> int:
    training_seed = int(args.seed)
    output_root = Path(getattr(args, "output_root", None) or common.TRAIN_OUT)
    directory = output_root / f"{args.tier}_seed{training_seed}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"tight_{args.tier}_seed{training_seed}.pt"
    torch.set_num_threads(max(1, int(args.threads)))
    random.seed(training_seed)
    np.random.seed(training_seed % 2**31)
    controller = train_pilot.Controller(training_seed)
    optimizer = torch.optim.Adam(controller.parameters, lr=LEARNING_RATE)
    state: dict[str, Any] = {
        "seed": training_seed,
        "tier": args.tier,
        "budget_bytes": args.budget_bytes,
        "environment_steps": 0,
        "episodes": 0,
        "cursor": 0,
        "updates": [],
        "evaluations": [],
        "collector": "onpolicy",
        "initial_weights_fingerprint": common.weight_fingerprint(controller),
        "scenario_order": list(common.TRAIN_SEEDS),
    }
    if path.exists():
        payload = load(path, training_seed)
        controller.load_state_dict(payload["controller"])
        optimizer.load_state_dict(payload["optimizer"])
        state.update(payload["state"])
        restored = restore_rng(state.get("rng"))
        print(
            f"[{args.tier} seed {training_seed}] resumed at {state['environment_steps']} steps "
            f"(rng restored: {restored})",
            flush=True,
        )
    started = time.time()
    while state["environment_steps"] < args.steps:
        batch_seeds = [
            common.TRAIN_SEEDS[(state["cursor"] + index) % len(common.TRAIN_SEEDS)]
            for index in range(EPISODES_PER_UPDATE)
        ]
        state["cursor"] = (state["cursor"] + EPISODES_PER_UPDATE) % len(common.TRAIN_SEEDS)
        episodes, details = collect_batch(
            controller, batch_seeds, budget_bytes=args.budget_bytes,
            directory=directory / "rollout", training_seed=training_seed,
        )
        report = update(controller, optimizer, episodes)
        measured = sorted({int(item["steps_run"]) for item in details})
        if measured != [STEPS_PER_EPISODE]:
            raise RuntimeError(
                f"the environment ran {measured} steps per episode, not the configured "
                f"{STEPS_PER_EPISODE}; the step budget would be a different quantity"
            )
        spent = sum(int(item["row"]["spent_bytes"]) for item in episodes)
        report.update(
            {
                "at_steps": state["environment_steps"] + STEPS_PER_EPISODE * len(batch_seeds),
                "batch_seeds": batch_seeds,
                "batch_on_time": sum(int(item["row"]["task"]["on_time"]) for item in episodes),
                "batch_targets": sum(int(item["row"]["task"]["targets"]) for item in episodes),
                "batch_budget_use": round(spent / (args.budget_bytes * len(batch_seeds)), 4),
                "batch_bytes": spent,
                "decisions": sum(item["decisions"] for item in details),
                "candidates": sum(item["candidates"] for item in details),
                "actionable_candidates": sum(item["free_candidates"] for item in details),
                "drawn_send_rate": round(
                    float(np.mean([item["drawn_send_rate"] for item in details])), 4
                ),
                "emissions": sum(item["emissions"] for item in details),
                "emissions_by_policy": sum(item["emissions_by_policy"] for item in details),
                "emissions_forced": sum(item["emissions_forced"] for item in details),
                "refused": sum(item["refused_by_policy"] + item["refused_forced"] for item in details),
                "seconds": round(sum(item["seconds"] for item in details), 2),
            }
        )
        state["environment_steps"] = report["at_steps"]
        state["episodes"] += len(batch_seeds)
        state["updates"].append(report)
        state["rng"] = rng_state()
        _guard(report, controller)
        for milestone in CHECKPOINT_STEPS:
            if report["at_steps"] >= milestone and milestone not in state.setdefault("milestones", {}):
                state["milestones"][milestone] = report["at_steps"]
        save(controller, optimizer, state, path)
        for milestone, landed in sorted(state.get("milestones", {}).items()):
            if landed == report["at_steps"]:
                save(controller, optimizer, state, directory / f"{path.stem}_at{milestone}.pt")
                print(
                    f"[{args.tier} seed {training_seed}] {milestone}-step milestone landed at "
                    f"{report['at_steps']}",
                    flush=True,
                )
        print(
            f"[{args.tier} seed {training_seed}] steps={state['environment_steps']:>6} "
            f"on_time={report['batch_on_time']}/{report['batch_targets']} bytes={report['batch_bytes']} "
            f"use={report['batch_budget_use']} auton={report['emissions_by_policy']}/{report['emissions']} "
            f"send_rate={report['drawn_send_rate']} samples={report['samples']} opt={report['optimizer_steps']} "
            f"kl={report['approx_kl']} clip={report['clip_fraction']} ent={report['entropy']} "
            f"ev={report['value_explained_variance']} ({time.time() - started:.0f}s)",
            flush=True,
        )
        crossed = (
            state["environment_steps"] // EVALUATE_EVERY_STEPS
            > (state["environment_steps"] - STEPS_PER_EPISODE * len(batch_seeds)) // EVALUATE_EVERY_STEPS
        )
        if crossed and args.validation_episodes > 0:
            result = common.evaluate_controller(
                controller, budget_bytes=args.budget_bytes, episodes=args.validation_episodes,
                directory=directory / "eval", label=f"tight_{args.tier}_seed{training_seed}",
            )
            result["at_steps"] = state["environment_steps"]
            state["evaluations"].append(result)
            save(controller, optimizer, state, path)
            print(f"[{args.tier} seed {training_seed}] validation {json.dumps(result, ensure_ascii=False)}", flush=True)
    summary = {
        key: state[key]
        for key in ("seed", "tier", "budget_bytes", "environment_steps", "episodes", "collector", "initial_weights_fingerprint")
    }
    summary["updates"] = len(state["updates"])
    summary["optimizer_steps_cumulative"] = sum(int(item.get("optimizer_steps", 0)) for item in state["updates"])
    summary["decision_samples_recorded"] = sum(int(item.get("samples", 0)) for item in state["updates"])
    summary["likelihood_max_abs_gap"] = max(
        (float(item["likelihood_alignment"]["max_abs_gap"]) for item in state["updates"] if "likelihood_alignment" in item),
        default=0.0,
    )
    counted = [item for item in state["updates"] if "batch_seeds" in item]
    summary["sampling_frequency"] = dict(sorted(Counter(seed for item in counted for seed in item["batch_seeds"]).items()))
    summary["final_weights_fingerprint"] = common.weight_fingerprint(controller)
    summary["last_evaluation"] = state["evaluations"][-1] if state["evaluations"] else None
    common.write_json(directory / f"{path.stem}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    return 0


def _guard(report: dict[str, Any], controller: train_pilot.Controller) -> None:
    if any(not torch.isfinite(tensor).all() for tensor in common.tensors(controller.state_dict())):
        raise FloatingPointError(f"NaN or inf in the weights at {report['at_steps']} steps")


def throughput(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(1, int(args.threads)))
    directory = common.HERE / "throughput"
    directory.mkdir(parents=True, exist_ok=True)
    controller = train_pilot.Controller(args.seed)
    started = time.time()
    episodes = [
        run_episode(controller, seed_value, budget_bytes=args.budget_bytes, directory=directory, label="throughput")
        for seed_value in common.TRAIN_SEEDS[: args.episodes]
    ]
    seconds = time.time() - started
    per_episode = seconds / len(episodes)
    payload = {
        "threads": int(args.threads),
        "episodes": len(episodes),
        "seconds": round(seconds, 2),
        "seconds_per_episode": round(per_episode, 3),
        "environment_steps_per_second": round(STEPS_PER_EPISODE / per_episode, 1),
        "seconds_per_update_4_episodes": round(per_episode * EPISODES_PER_UPDATE, 1),
        "collector": "onpolicy",
        "projection": {
            "one_run_to_400k_hours": round(400_000 / STEPS_PER_EPISODE * per_episode / 3600, 2),
            "six_runs_to_400k_serial_hours": round(
                6 * 400_000 / STEPS_PER_EPISODE * per_episode / 3600, 2
            ),
            "runs": 6,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    if args.record:
        common.write_json(common.json_path("COMM_TIGHT_BUDGET_THROUGHPUT.json"), payload)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "throughput"), default="train")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tier", default="75pct", choices=tuple(common.BUDGETS_BYTES))
    parser.add_argument("--budget-bytes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument(
        "--output-root", default=None,
        help="where checkpoints go; one tree per run, so no two runs can read each other's state",
    )
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args(argv)
    args.budget_bytes = args.budget_bytes or common.locked_budget(args.tier)
    if args.budget_bytes != common.locked_budget(args.tier):
        raise SystemExit(
            f"--budget-bytes {args.budget_bytes} is not the locked {args.tier} tier "
            f"({common.locked_budget(args.tier)}); the tiers are calibration output, not a dial"
        )
    if args.mode == "throughput":
        return throughput(args)
    return train(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
