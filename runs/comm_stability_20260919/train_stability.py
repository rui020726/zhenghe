"""§4 and §5: continue the pilot's three runs, with the training volume stated in real units.

Two things were wrong with how round 2 described its own effort. 100,000 environment steps is a
count of *simulator ticks* (100 episodes x 1000 steps), and the learning in it was roughly 640
policy samples per episode and about 1,200 gradient steps in total -- which is a different, and
much smaller, quantity. And the inference cost published with the result was ~200 ms per
decision, which turned out to be three trainers and their thread pools contending for the same
cores; measured on one thread it is ~0.2 ms. Both are corrected here, in the open, with the
numbers that produced them.

The training itself is deliberately not improved. Rollout stays at 4 episodes per update, the
network, the reward, the budget and the silence bound are the pilot's, and the only main
variable is duration: 100k -> 200k, and no further unless §5's continuation conditions hold.
What is added is measurement -- clip fraction, explained variance, gradient norm, optimizer
steps, voluntary versus bound-forced sends, and the sampling frequency of the 48 training
streams -- plus the state needed to resume honestly:

* actor, critic and optimizer moments come from the pilot checkpoint, exactly;
* the pilot saved no RNG state, so the Python, NumPy and Torch streams restart. That makes this
  a **warm start**, not a seamless continuation, and it is recorded as such in the checkpoint.
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

import stability_common as common

import pilot_env
import train_pilot
from train_pilot import FEATURE_DIM, GAMMA, GAE_LAMBDA, CLIP, VALUE_COEFFICIENT, ENTROPY_BONUS, MAX_GRAD_NORM

EPISODES_PER_UPDATE = train_pilot.EPISODES_PER_UPDATE
EPOCHS_PER_BATCH = train_pilot.EPOCHS_PER_BATCH
MINIBATCH_STEPS = train_pilot.MINIBATCH_STEPS
EXPLORATION_EPSILON = train_pilot.EXPLORATION_EPSILON
EVALUATE_EVERY_STEPS = train_pilot.EVALUATE_EVERY_STEPS
STEPS_PER_EPISODE = common.STREAM_STEPS
PILOT_STEPS = 100_000
#: The milestones round 3 keeps a checkpoint at, decided before the first update was run.
CHECKPOINT_STEPS = (150_000, 200_000, 300_000, 400_000)
#: What the extension beyond 200k requires, written before any 200k number existed.
CONTINUATION_CONDITION = {
    "train_and_validation_both_stable": (
        "no NaN, no budget violation, and the last three validation points inside one "
        "completion of each other"
    ),
    "weaker_seeds_still_improving": (
        "a seed whose mean of the last two validation points exceeds its mean of the two "
        "before them by more than 0.05 completions"
    ),
    "rule": (
        "extend all three seeds together to at most 400,000 steps only if both hold; never "
        "extend the best seed alone; never raise the cap, the reward or the action space"
    ),
}


def rng_state() -> dict[str, Any]:
    """Everything a resampled coin, a shuffled batch and a network initialisation depend on."""
    return {
        "python": [state[0], list(state[1]), state[2]] if (state := random.getstate()) else None,
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]]
        if (state := np.random.get_state()) is not None
        else None,
        "torch": torch.get_rng_state().tolist(),
    }


def restore_rng(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    random.setstate((payload["python"][0], tuple(payload["python"][1]), payload["python"][2]))
    numpy_state = payload["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(torch.tensor(payload["torch"], dtype=torch.uint8))
    return True


def update_with_diagnostics(
    controller: train_pilot.Controller, optimizer: torch.optim.Optimizer, episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Round 2's PPO update, with the statistics §4 asks for and round 2 never published.

    The maths is unchanged -- same surrogate, same clip, same minibatch size, same advantage
    scaling -- so a curve that crosses into this file is comparable with the pilot's. What is
    new is only what gets measured.
    """
    flat: list[dict[str, Any]] = []
    for episode in episodes:
        values = [record["value"] for record in episode["decisions"]]
        rewards = episode["rewards"]
        if not values:
            continue
        advantages, returns = train_pilot.gae(rewards, values)
        scale = max(1e-6, float(np.std(advantages)))
        for record, advantage, target_return in zip(episode["decisions"], advantages, returns):
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
    if not flat:
        return {"samples": 0, "minibatches": 0, "optimizer_steps": 0}
    returns = np.asarray([item["return"] for item in flat], dtype=float)
    behaviour_values = np.asarray([item["behaviour_value"] for item in flat], dtype=float)
    explained_variance = (
        1.0 - float(np.var(returns - behaviour_values) / max(1e-9, np.var(returns)))
        if len(returns) > 1 and float(np.var(returns)) > 1e-9
        else None
    )
    order = list(range(len(flat)))
    totals = {"surrogate": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0, "clip": 0.0, "grad": 0.0}
    minibatches = 0
    for _ in range(EPOCHS_PER_BATCH):
        random.shuffle(order)
        for start in range(0, len(order), MINIBATCH_STEPS):
            chunk = [flat[index] for index in order[start : start + MINIBATCH_STEPS]]
            stacked = torch.as_tensor(np.concatenate([item["features"] for item in chunk], axis=0))
            logits, candidate_values = controller.forward(stacked)
            probabilities = torch.softmax(logits, dim=-1)
            log_probabilities = torch.log_softmax(logits, dim=-1)
            row = 0
            new_log_prob: list[torch.Tensor] = []
            entropy: list[torch.Tensor] = []
            pooled: list[torch.Tensor] = []
            for item in chunk:
                count = len(item["targets"])
                rows = torch.arange(row, row + count)
                row += count
                picked = torch.tensor([item["actions"][target] for target in item["targets"]], dtype=torch.long)
                usable = torch.tensor(
                    [target in item["free"] for target in item["targets"]], dtype=torch.float32
                )
                new_log_prob.append((log_probabilities[rows, picked] * usable).sum())
                entropy.append((-(probabilities[rows] * log_probabilities[rows]).sum(dim=-1) * usable).sum())
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
    }


def episode_diagnostics(collector: train_pilot.Collector, row: dict[str, Any]) -> dict[str, Any]:
    """How much of an episode was a decision, and how much of it was the rule."""
    ledger = row["ledger"]
    decisions = max(1, collector.decisions)
    voluntary = sum(
        1
        for record in collector.records
        for target, action in record["actions"].items()
        if action == train_pilot.SEND and target in record["free"]
    )
    return {
        "decisions": int(collector.decisions),
        "candidates": int(collector.candidates),
        "candidates_per_decision": round(collector.candidates / decisions, 3),
        "masked_by_bound": int(collector.forced_sends),
        "voluntary_sends_requested": int(voluntary),
        "voluntary_sends_share": round(voluntary / max(1, collector.candidates), 4),
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
    """The clock the episode really reached. 1000 by configuration; measured, not assumed."""
    from integrated_sar.lifecycle_metrics import load_records

    records = load_records(Path(lifecycle_log))
    closing = [float(record.get("timestamp") or 0.0) for record in records if record.get("event_type") == "final"]
    if not closing:
        raise RuntimeError(f"{lifecycle_log}: no closing record, so the episode length is unknown")
    return int(max(closing))


def collect_batch(
    controller: train_pilot.Controller, batch_seeds: list[int], *, budget_bytes: int, directory: Path, training_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes, details = [], []
    for seed_value in batch_seeds:
        collector = common.transport_bounded_collector(controller, epsilon=EXPLORATION_EPSILON)
        episode = train_pilot.run_one(
            seed_value,
            collector=collector,
            budget_bytes=budget_bytes,
            directory=directory,
            label=f"stability_train{training_seed}",
        )
        episodes.append(episode)
        details.append(episode_diagnostics(collector, episode["row"]))
    return episodes, details


def train(args: argparse.Namespace) -> int:
    training_seed = int(args.seed)
    directory = common.TRAIN_OUT / f"seed{training_seed}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"stability_seed{training_seed}.pt"
    source = Path(args.resume_from) if args.resume_from else common.PILOT_CHECKPOINTS.get(training_seed)
    torch.set_num_threads(max(1, int(args.threads)))
    random.seed(training_seed)
    np.random.seed(training_seed % 2**31)
    controller = train_pilot.Controller(training_seed)
    optimizer = torch.optim.Adam(controller.parameters, lr=train_pilot.LEARNING_RATE)
    state: dict[str, Any] = {
        "seed": training_seed,
        "budget_bytes": args.budget_bytes,
        "environment_steps": 0,
        "episodes": 0,
        "cursor": 0,
        "updates": [],
        "evaluations": [],
        "collector": "transport_bounded",
        "warm_start": False,
        "rng": None,
    }
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        controller.load_state_dict(payload["controller"])
        optimizer.load_state_dict(payload["optimizer"])
        state.update(payload["state"])
        restored = restore_rng(state.get("rng"))
        print(
            f"[seed {training_seed}] resumed at {state['environment_steps']} environment steps "
            f"(rng restored: {restored})",
            flush=True,
        )
    elif source is not None and Path(source).exists():
        payload = torch.load(source, map_location="cpu", weights_only=False)
        controller.load_state_dict(payload["controller"])
        optimizer.load_state_dict(payload["optimizer"])
        state.update(
            {
                key: payload["state"][key]
                for key in ("environment_steps", "episodes", "cursor", "updates", "evaluations")
            }
        )
        # The pilot stored no RNG state, and its streams cannot be rewound after the fact.
        state["warm_start"] = True
        state["resumed_from"] = str(source)
        print(
            f"[seed {training_seed}] warm start from {source}: weights and optimizer moments are "
            "exact, the Python/NumPy/Torch streams restart",
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
            controller, batch_seeds, budget_bytes=args.budget_bytes, directory=directory / "rollout", training_seed=training_seed
        )
        report = update_with_diagnostics(controller, optimizer, episodes)
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
                "masked_by_bound": sum(item["masked_by_bound"] for item in details),
                "emissions": sum(item["emissions"] for item in details),
                "emissions_by_policy": sum(item["emissions_by_policy"] for item in details),
                "emissions_forced": sum(item["emissions_forced"] for item in details),
                "refused_by_policy": sum(item["refused_by_policy"] for item in details),
                "refused_forced": sum(item["refused_forced"] for item in details),
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
        _save(controller, optimizer, state, path)
        for milestone, landed in sorted(state.get("milestones", {}).items()):
            if landed == report["at_steps"]:
                _save(
                    controller,
                    optimizer,
                    state,
                    directory / f"stability_seed{training_seed}_at{milestone}.pt",
                )
                print(
                    f"[seed {training_seed}] checkpoint for the {milestone}-step milestone "
                    f"landed at {report['at_steps']}",
                    flush=True,
                )
        print(
            f"[seed {training_seed}] steps={state['environment_steps']:>6} episodes={state['episodes']:>4} "
            f"on_time={report['batch_on_time']}/{report['batch_targets']} bytes={report['batch_bytes']} "
            f"auton={report['emissions_by_policy']}/{report['emissions']} "
            f"samples={report['samples']} opt={report['optimizer_steps']} kl={report['approx_kl']} "
            f"clip={report['clip_fraction']} ent={report['entropy']} ev={report['value_explained_variance']} "
            f"({report['seconds']}s, {time.time() - started:.0f}s elapsed)",
            flush=True,
        )
        crossed = (
            state["environment_steps"] // EVALUATE_EVERY_STEPS
            > (state["environment_steps"] - STEPS_PER_EPISODE * len(batch_seeds)) // EVALUATE_EVERY_STEPS
        )
        if crossed:
            result = _evaluate(controller, args, directory, state["environment_steps"])
            result["at_steps"] = state["environment_steps"]
            state["evaluations"].append(result)
            _save(controller, optimizer, state, path)
            print(f"[seed {training_seed}] validation {json.dumps(result, ensure_ascii=False)}", flush=True)
    summary = {key: state[key] for key in ("seed", "budget_bytes", "environment_steps", "episodes", "warm_start", "collector")}
    summary["updates"] = len(state["updates"])
    # The pilot's update reports called this "minibatches"; one minibatch was one optimizer.step,
    # so the two names add up into one cumulative figure rather than needing a footnote.
    summary["optimizer_steps_cumulative"] = sum(
        int(item.get("optimizer_steps", item.get("minibatches", 0))) for item in state["updates"]
    )
    summary["decision_samples_recorded"] = sum(int(item.get("samples", 0)) for item in state["updates"])
    counted = [item for item in state["updates"] if "batch_seeds" in item]
    summary["sampling_frequency"] = dict(
        sorted(Counter(seed for item in counted for seed in item["batch_seeds"]).items())
    )
    summary["rounds_without_recorded_batch"] = len(state["updates"]) - len(counted)
    summary["last_evaluation"] = state["evaluations"][-1] if state["evaluations"] else None
    common.write_json(directory / f"stability_seed{training_seed}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    return 0


def _evaluate(controller: train_pilot.Controller, args: argparse.Namespace, directory: Path, at_steps: int) -> dict[str, Any]:
    """The pilot's own validation read -- same 8 streams, same greedy rollout, same interval.

    Measured through ``attribution.run_arm`` rather than round 2's evaluator, so a point on this
    curve and a row of the attribution table are the same kind of number: same collector, same
    units, same split between what the model asked for and what the bound made it say.
    """
    import attribution

    spec = {"arm": f"stability_at{at_steps}", "kind": "checkpoint", "policy": "learned", "params": ""}
    rows = [
        attribution.run_arm(
            spec,
            seed,
            controller=controller,
            provenance={"controller": "live", "at_steps": at_steps},
            directory=directory / "eval",
            budget=args.budget_bytes,
        )
        for seed in list(common.VALIDATION_SEEDS)[: args.validation_episodes]
    ]

    def mean(key: str) -> float:
        return round(float(np.mean([float(row[key]) for row in rows])), 4)

    return {
        "episodes": len(rows),
        "mean_on_time": mean("on_time_targets_per_episode"),
        "mean_on_time_rate_per_episode": mean("on_time_rate_within_detected_this_episode"),
        "mean_expired": mean("expired_targets_per_episode"),
        "mean_airtime_bytes": mean("airtime_bytes"),
        "mean_emissions": mean("emissions"),
        "mean_emissions_by_policy": round(
            sum(int(row["rollup"]["by_reason"]["policy"]["emissions"]) for row in rows) / len(rows), 3
        ),
        "mean_emissions_forced": round(
            sum(int(row["rollup"]["by_reason"]["forced_maxwait"]["emissions"]) for row in rows) / len(rows), 3
        ),
        "mean_refusals": round(
            sum(
                sum(bucket["refused"] for bucket in row["rollup"]["by_reason"].values())
                for row in rows
            )
            / len(rows),
            3,
        ),
        "autonomous_share_of_bytes": round(
            float(np.mean([float(row["autonomous_share_of_bytes"] or 0.0) for row in rows])), 4
        ),
        "mean_applied": mean("deliveries_applied"),
        "mean_belief_age": mean("belief_age_steps"),
        "inference_milliseconds_per_decision": mean("inference_milliseconds_per_decision"),
    }


def _save(controller: train_pilot.Controller, optimizer: torch.optim.Optimizer, state: dict[str, Any], path: Path) -> None:
    torch.save({"controller": controller.state_dict(), "optimizer": optimizer.state_dict(), "state": state}, path)


def _guard(report: dict[str, Any], controller: train_pilot.Controller) -> None:
    """Stop on the two things that would make the rest of the run meaningless.

    A budget violation already raises inside the episode runner, so it cannot reach here.
    """
    if any(not torch.isfinite(tensor).all() for tensor in _tensors(controller.state_dict())):
        raise FloatingPointError(f"NaN or inf in the weights at {report['at_steps']} steps")


def _tensors(payload: Any) -> list[Any]:
    found: list[Any] = []
    for value in payload.values() if isinstance(payload, dict) else payload:
        if isinstance(value, dict):
            found.extend(_tensors(value))
        else:
            found.append(value)
    return found


def throughput(args: argparse.Namespace) -> int:
    """Measure before promising: the pilot's time estimate was never checked, so neither is this one."""
    torch.set_num_threads(max(1, int(args.threads)))
    directory = common.TRAIN_OUT / "throughput"
    directory.mkdir(parents=True, exist_ok=True)
    controller = train_pilot.Controller(args.seed)
    started = time.time()
    episodes = [
        train_pilot.run_one(
            seed_value,
            collector=common.transport_bounded_collector(controller, epsilon=EXPLORATION_EPSILON),
            budget_bytes=args.budget_bytes,
            directory=directory,
            label=f"throughput{args.seed}",
        )
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
        "projection": {
            "to_200k_per_seed": {
                "updates": (200_000 - PILOT_STEPS) // (STEPS_PER_EPISODE * EPISODES_PER_UPDATE),
                "hours_serial": round((200_000 - PILOT_STEPS) / STEPS_PER_EPISODE * per_episode / 3600, 2),
            },
            "to_400k_per_seed": {
                "updates": (400_000 - PILOT_STEPS) // (STEPS_PER_EPISODE * EPISODES_PER_UPDATE),
                "hours_serial": round((400_000 - PILOT_STEPS) / STEPS_PER_EPISODE * per_episode / 3600, 2),
            },
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    if args.record:
        common.write_json(common.json_path("COMM_TRAINING_THROUGHPUT.json"), payload)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "throughput"), default="train")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--budget-bytes", type=int, default=common.BUDGET_BYTES)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "throughput":
        return throughput(args)
    return train(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
