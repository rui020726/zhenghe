"""Bounded PPO training under ``policy_owned_timing``: two credits, three seeds, 400k steps each.

This is round 4's corrected trainer -- the sampler, the shared likelihood helpers and the update
that refuses a mismatched tape are imported from it, not re-derived -- with one thing moved and one
thing added. What moves is the action mask: under the new protocol no candidate is taken over by
the transport, so no candidate may be dropped from the likelihood or the entropy bonus either. What
is added is the bookkeeping §5 of the brief asks for, because the optimisation load is genuinely
different now and a reader has to be able to see by how much: free decision samples, optimiser
steps, per-step and per-candidate entropy reported separately, KL, gradient norm, clip fraction,
value loss, when the credit ran out, and which reasons produced the emissions.

Nothing adapts to a result here: the step ceiling, the milestones, the evaluation cadence and the
hyperparameters are the frozen ones, the curve evaluations are reported beside the end-of-run
checkpoint rather than used to pick a winner, and a run that fails its gate is not restarted.

    python train_protocol.py --mode throughput --tier 25pct --seed 1 --episodes 8 --threads 1
    python train_protocol.py --mode train --tier 25pct --seed 1 --steps 400000 --threads 1
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

import protocol_common as common
import protocol_env

import train_pilot
from onpolicy import OnPolicyCollector, action_entropy, joint_log_prob, restore_rng, rng_state
from train_onpolicy import drawn_versus_logged, update

TRAINER_TAG = "onpolicy-v2"
FEATURE_DIM = train_pilot.FEATURE_DIM
EPISODES_PER_UPDATE = train_pilot.EPISODES_PER_UPDATE
EVALUATE_EVERY_STEPS = train_pilot.EVALUATE_EVERY_STEPS
STEPS_PER_EPISODE = common.STREAM_STEPS
#: Initialisation plus every 100k. The initial checkpoint is saved because §5 of the brief makes
#: "compared with its own untrained weights" a required read, and round 2 could only reconstruct it.
CHECKPOINT_STEPS = (0, 100_000, 200_000, 300_000, 400_000)
#: The protocol this trainer speaks. A checkpoint from the other one is a different experiment.
TIMING_MODE = common.POLICY_MODE


def nothing_forced(item: Any) -> bool:
    """No candidate is the transport's decision under ``policy_owned_timing``.

    Kept as a named predicate rather than inlined as ``mask=False`` so the claim is visible where
    the sampler is built: here, every candidate the relay offers is the arm's own choice, and
    therefore every one of them belongs in the likelihood and the entropy.
    """
    return False


def collector_for(controller: Any, *, greedy: bool = False) -> OnPolicyCollector:
    return OnPolicyCollector(
        controller, forced=nothing_forced, mask=False, greedy=greedy
    )


def run_episode(controller: Any, seed: int, *, budget_bytes: int, directory: Path, label: str) -> dict[str, Any]:
    collector = collector_for(controller)
    episode = protocol_env.rollout_episode(
        controller,
        seed,
        collector=collector,
        budget_bytes=budget_bytes,
        directory=directory,
        label=label,
        timing_mode=TIMING_MODE,
    )
    episode["controller"] = controller
    episode["collector"] = collector
    return episode


def entropy_probe(controller: Any, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Post-update entropy, reported two ways so a wider choice surface cannot read as exploration.

    Round 4 caught itself on exactly this: the per-step number is a *sum over free candidates*, so
    it rises when an arm gains more real choices even while each individual choice sharpens. Both
    readings are produced here from the same ``action_entropy`` helper the loss uses, over the same
    tapes, with no gradient.
    """
    per_step: list[float] = []
    terms = 0
    for episode in episodes:
        for record in episode["decisions"]:
            if not record["free"]:
                continue
            logits, _values = controller.forward(
                torch.as_tensor(np.asarray(record["features"], dtype=np.float32))
            )
            targets = record["targets"]
            free = set(record["free"])
            usable = torch.tensor(
                [target in free for target in targets], dtype=torch.float32
            )
            with torch.no_grad():
                per_step.append(float(action_entropy(logits, usable).item()))
            terms += len(free)
    if not per_step:
        return {"entropy_probe_steps": 0}
    return {
        "entropy_probe_steps": len(per_step),
        "entropy_sum_per_step_mean": round(float(np.mean(per_step)), 5),
        "entropy_per_free_candidate_mean": round(float(np.sum(per_step)) / max(1, terms), 5),
    }


def collect_batch(
    controller: Any, batch_seeds: list[int], *, budget_bytes: int, directory: Path, training_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for seed_value in batch_seeds:
        episode = run_episode(
            controller,
            seed_value,
            budget_bytes=budget_bytes,
            directory=directory,
            label=f"v2_train{training_seed}",
        )
        episodes.append(episode)
        details.append(episode_diagnostics(episode["collector"], episode["row"]))
    return episodes, details


def episode_diagnostics(collector: OnPolicyCollector, row: dict[str, Any]) -> dict[str, Any]:
    ledger = row["ledger"]
    timing = collector.timing()
    records = common.metrics.load_transcript(row["artefacts"]["transcript"])
    refusals = sorted(
        {int(record["step"]) for record in records if record["kind"] == "refused"}
    )
    sends = [record for record in records if record["kind"] == "send"]
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
        "first_refusal_step": refusals[0] if refusals else None,
        "refusals": len(refusals),
        "longest_silence_steps": int(ledger["max_backlog_age"]),
        "timer_driven_refreshes": sum(
            1
            for record in sends
            if int(record["step"]) - int(record["ready_step"])
            >= (
                common.BOUND_URGENT_WAIT_STEPS
                if 0.0 <= float(record["deadline"]) - float(record["now"]) <= common.BOUND_URGENCY_TIME
                else common.BOUND_MAX_WAIT_STEPS
            )
            and not bool(record.get("first"))
        ),
        "on_time": int(row["task"]["on_time"]),
        "targets": int(row["task"]["targets"]),
        "seconds": float(row["seconds"]),
        "steps_run": actual_steps(row["artefacts"]["lifecycle_log"]),
    }


def actual_steps(lifecycle_log: str) -> int:
    from integrated_sar.lifecycle_metrics import load_records

    records = load_records(Path(lifecycle_log))
    closing = [
        float(record.get("timestamp") or 0.0) for record in records if record.get("event_type") == "final"
    ]
    if not closing:
        raise RuntimeError(f"{lifecycle_log}: no closing record, so the episode length is unknown")
    return int(max(closing))


def save(
    controller: Any,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    path: Path,
) -> None:
    torch.save(
        {
            "controller": controller.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": state,
            "trainer": TRAINER_TAG,
            # The protocol is part of the checkpoint's identity: resuming a policy-owned run from
            # a legacy-bound checkpoint would train an arm whose mask no longer exists.
            "timing_mode": TIMING_MODE,
            "feature_layout": train_pilot.FEATURE_LAYOUT_NAME,
        },
        path,
    )


def load(path: Path, *, tier: str, training_seed: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("trainer") != TRAINER_TAG:
        raise RuntimeError(
            f"{path} came from trainer {payload.get('trainer')!r}, not {TRAINER_TAG!r}; this round "
            "starts from initial weights under its own protocol and must not inherit the other one"
        )
    if payload.get("timing_mode") != TIMING_MODE:
        raise RuntimeError(
            f"{path} was trained under timing mode {payload.get('timing_mode')!r}; refusing to "
            f"continue it as {TIMING_MODE!r}"
        )
    state = payload.get("state") or {}
    if int(state.get("seed", -1)) != int(training_seed) or state.get("tier") != tier:
        raise RuntimeError(
            f"{path} is seed={state.get('seed')} tier={state.get('tier')}, asked for "
            f"seed={training_seed} tier={tier}"
        )
    return payload


def train(args: argparse.Namespace) -> int:
    training_seed = int(args.seed)
    output_root = Path(args.output_root) if args.output_root else common.TRAIN_OUT
    directory = output_root / f"{args.tier}_seed{training_seed}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"v2_{args.tier}_seed{training_seed}.pt"
    torch.set_num_threads(max(1, int(args.threads)))
    random.seed(training_seed)
    np.random.seed(training_seed % 2**31)
    controller = train_pilot.Controller(training_seed)
    optimizer = torch.optim.Adam(controller.parameters, lr=train_pilot.LEARNING_RATE)
    state: dict[str, Any] = {
        "seed": training_seed,
        "tier": args.tier,
        "budget_bytes": args.budget_bytes,
        "timing_mode": TIMING_MODE,
        "environment_steps": 0,
        "episodes": 0,
        "cursor": 0,
        "updates": [],
        "evaluations": [],
        "collector": TRAINER_TAG,
        "initial_weights_fingerprint": common.weight_fingerprint(controller),
        "scenario_order": list(common.TRAIN_SEEDS),
    }
    if path.exists():
        payload = load(path, tier=args.tier, training_seed=training_seed)
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
            controller,
            batch_seeds,
            budget_bytes=args.budget_bytes,
            directory=directory / "rollout",
            training_seed=training_seed,
        )
        report = update(controller, optimizer, episodes)
        report.update(entropy_probe(controller, episodes))
        measured = sorted({int(item["steps_run"]) for item in details})
        if measured != [STEPS_PER_EPISODE]:
            raise RuntimeError(
                f"the environment ran {measured} steps per episode, not the configured "
                f"{STEPS_PER_EPISODE}; the step budget would be a different quantity"
            )
        masked = sum(int(item["masked_by_bound"]) for item in details)
        if masked:
            raise RuntimeError(
                f"the sampler masked away {masked} candidates under {TIMING_MODE}: the transport "
                "is still deciding for the arm, which is the thing this round removed"
            )
        spent = sum(int(item["row"]["spent_bytes"]) for item in episodes)
        report.update(
            {
                "at_steps": state["environment_steps"] + STEPS_PER_EPISODE * len(batch_seeds),
                "batch_seeds": batch_seeds,
                "batch_on_time": sum(int(item["on_time"]) for item in details),
                "batch_targets": sum(int(item["targets"]) for item in details),
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
                "timer_driven_refreshes": sum(item["timer_driven_refreshes"] for item in details),
                "longest_silence_steps": max(int(item["longest_silence_steps"]) for item in details),
                "median_first_refusal_step": _median(
                    [item["first_refusal_step"] for item in details if item["first_refusal_step"]]
                ),
                "seconds": round(sum(item["seconds"] for item in details), 2),
            }
        )
        state["environment_steps"] = report["at_steps"]
        state["episodes"] += len(batch_seeds)
        state["updates"].append(report)
        state["rng"] = rng_state()
        _guard(report, controller)
        for milestone in CHECKPOINT_STEPS:
            milestones = state.setdefault("milestones", {})
            if report["at_steps"] >= milestone and milestone not in milestones:
                milestones[milestone] = report["at_steps"]
        save(controller, optimizer, state, path)
        for milestone, landed in sorted(state["milestones"].items()):
            if landed == report["at_steps"] or (milestone == 0 and not state.get("saved_initial")):
                save(controller, optimizer, state, directory / f"v2_{args.tier}_seed{training_seed}_at{milestone}.pt")
                if milestone == 0:
                    state["saved_initial"] = True
                print(
                    f"[{args.tier} seed {training_seed}] checkpoint {milestone} landed at "
                    f"{report['at_steps']} steps",
                    flush=True,
                )
        print(
            f"[{args.tier} seed {training_seed}] steps={state['environment_steps']:>6} "
            f"on_time={report['batch_on_time']}/{report['batch_targets']} bytes={report['batch_bytes']} "
            f"use={report['batch_budget_use']} free={report['actionable_candidates']}/{report['candidates']} "
            f"send_rate={report['drawn_send_rate']} samples={report['samples']} opt={report['optimizer_steps']} "
            f"kl={report['approx_kl']} clip={report['clip_fraction']} ent={report['entropy']} "
            f"ent_free={report.get('entropy_per_free_candidate_mean')} ev={report['value_explained_variance']} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )
        crossed = (
            state["environment_steps"] // EVALUATE_EVERY_STEPS
            > (state["environment_steps"] - STEPS_PER_EPISODE * len(batch_seeds))
            // EVALUATE_EVERY_STEPS
        )
        if crossed and args.validation_episodes > 0:
            result = evaluate_controller(
                controller,
                budget_bytes=args.budget_bytes,
                episodes=args.validation_episodes,
                directory=directory / "eval",
                label=f"v2_{args.tier}_seed{training_seed}",
            )
            result["at_steps"] = state["environment_steps"]
            state["evaluations"].append(result)
            save(controller, optimizer, state, path)
            print(
                f"[{args.tier} seed {training_seed}] validation "
                f"{json.dumps(result, ensure_ascii=False)}",
                flush=True,
            )
    summary = {
        key: state[key]
        for key in (
            "seed",
            "tier",
            "budget_bytes",
            "timing_mode",
            "environment_steps",
            "episodes",
            "collector",
            "initial_weights_fingerprint",
        )
    }
    summary["updates"] = len(state["updates"])
    summary["optimizer_steps_cumulative"] = sum(
        int(item.get("optimizer_steps", 0)) for item in state["updates"]
    )
    summary["decision_samples_recorded"] = sum(int(item.get("samples", 0)) for item in state["updates"])
    summary["free_decision_terms_recorded"] = sum(
        int(item.get("free_candidate_samples", 0)) for item in state["updates"]
    )
    summary["likelihood_max_abs_gap"] = max(
        (
            float(item["likelihood_alignment"]["max_abs_gap"])
            for item in state["updates"]
            if "likelihood_alignment" in item
        ),
        default=0.0,
    )
    counted = [item for item in state["updates"] if "batch_seeds" in item]
    summary["sampling_frequency"] = dict(
        sorted(Counter(seed for item in counted for seed in item["batch_seeds"]).items())
    )
    summary["final_weights_fingerprint"] = common.weight_fingerprint(controller)
    summary["last_evaluation"] = state["evaluations"][-1] if state["evaluations"] else None
    summary["selected_by_rule"] = select_checkpoint(state["evaluations"])
    common.write_json(directory / f"v2_{args.tier}_seed{training_seed}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    return 0


def _median(values: list[Any]) -> Any:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return None
    middle = len(numbers) // 2
    return numbers[middle] if len(numbers) % 2 else round((numbers[middle - 1] + numbers[middle]) / 2.0, 1)


def select_checkpoint(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Round 4's selection rule, unchanged, and reported as an additional curve point only.

    Latest evaluation wins, ties broken by fewer bytes. The primary judgement of this round is the
    400k end-of-run checkpoint, so a selected point never replaces it.
    """
    best: dict[str, Any] | None = None
    for evaluation in evaluations:
        if best is None:
            best = evaluation
            continue
        if float(evaluation["mean_on_time"]) > float(best["mean_on_time"]):
            best = evaluation
        elif (
            float(evaluation["mean_on_time"]) == float(best["mean_on_time"])
            and float(evaluation["mean_airtime_bytes"]) < float(best["mean_airtime_bytes"])
        ):
            best = evaluation
    return None if best is None else {"at_steps": best["at_steps"], "mean_on_time": best["mean_on_time"]}


def evaluate_controller(
    controller: Any,
    *,
    budget_bytes: int,
    episodes: int,
    directory: Path,
    label: str,
    information: str = "isolated",
    delivery_probability: float = 1.0,
) -> dict[str, Any]:
    """Greedy deployment reads on the fixed validation streams, under this round's protocol."""
    seeds = list(common.VALIDATION_SEEDS)[:episodes]
    rows: list[dict[str, Any]] = []
    timings: list[dict[str, float]] = []
    for seed in seeds:
        collector = collector_for(controller, greedy=True)
        row = protocol_env.run_episode(
            seed,
            directory=directory,
            label=label,
            policy="learned",
            budget_bytes=budget_bytes,
            controller=collector.policy(),
            information=information,
            delivery_probability=delivery_probability,
            timing_mode=TIMING_MODE,
            max_wait_steps=None,
            urgent_wait_steps=None,
        )
        records = common.metrics.load_transcript(row["artefacts"]["transcript"])
        rollup = common.attribution_metrics._per_reason_rollup(records)
        rows.append(
            {
                "on_time_targets_per_episode": int(row["task"]["on_time"]),
                "detected": int(row["task"]["targets"]),
                "airtime_bytes": int(row["spent_bytes"]),
                "emissions": int(row["ledger"]["emissions"]),
                "emissions_by_policy": int(row["ledger"]["emissions_policy"]),
                "emissions_forced": int(row["ledger"]["emissions_forced_maxwait"]),
                "redundant_emissions": int(row["ledger"]["redundant_emissions"]),
                "refusals": int(row["ledger"]["budget_refused_sends"]),
                "autonomous_share_of_bytes": float(rollup["autonomous_share_of_bytes"] or 0.0),
                "applied": int(row["ledger"]["deliveries_applied"]),
                "belief_age": float(row["belief"].get("belief_age_steps_mean") or 0.0),
                "max_backlog_age": int(row["ledger"]["max_backlog_age"]),
                "seconds": float(row["seconds"]),
            }
        )
        timings.append(collector.timing())

    def mean(key: str) -> float:
        return round(float(np.mean([float(row[key] or 0.0) for row in rows])), 4)

    return {
        "episodes": len(rows),
        "information": information,
        "delivery_probability": delivery_probability,
        "timing_mode": TIMING_MODE,
        "budget_bytes": budget_bytes,
        "mean_on_time": mean("on_time_targets_per_episode"),
        "mean_airtime_bytes": mean("airtime_bytes"),
        "mean_emissions": mean("emissions"),
        "mean_emissions_by_policy": mean("emissions_by_policy"),
        "mean_emissions_forced": mean("emissions_forced"),
        "mean_redundant_emissions": mean("redundant_emissions"),
        "mean_refusals": mean("refusals"),
        "autonomous_share_of_bytes": mean("autonomous_share_of_bytes"),
        "mean_applied": mean("applied"),
        "mean_belief_age": mean("belief_age"),
        "mean_max_backlog_age": mean("max_backlog_age"),
        "free_candidate_share_mean": round(
            float(np.mean([item["free_candidates"] / max(1, item["candidates_seen"]) for item in timings])),
            4,
        ),
        "inference_milliseconds_per_decision": round(
            sum(item["decide_milliseconds_per_call"] * item["decisions"] for item in timings)
            / max(1, sum(item["decisions"] for item in timings)),
            4,
        ),
        "per_episode": rows,
    }


def _guard(report: dict[str, Any], controller: Any) -> None:
    if any(not torch.isfinite(tensor).all() for tensor in common.tensors(controller.state_dict())):
        raise FloatingPointError(f"NaN or inf in the weights at {report['at_steps']} steps")


def throughput(args: argparse.Namespace) -> int:
    """Measure before estimating: the free-candidate set is larger now, so the old seconds/update
    is not this round's number."""
    torch.set_num_threads(max(1, int(args.threads)))
    directory = common.HERE / "throughput"
    directory.mkdir(parents=True, exist_ok=True)
    controller = train_pilot.Controller(args.seed)
    started = time.time()
    episodes = [
        run_episode(
            controller, seed_value, budget_bytes=args.budget_bytes, directory=directory, label="throughput"
        )
        for seed_value in common.TRAIN_SEEDS[: args.episodes]
    ]
    seconds = time.time() - started
    per_episode = seconds / len(episodes)
    payload = {
        "threads": int(args.threads),
        "timing_mode": TIMING_MODE,
        "tier": args.tier,
        "budget_bytes": args.budget_bytes,
        "episodes": len(episodes),
        "seconds": round(seconds, 2),
        "seconds_per_episode": round(per_episode, 3),
        "environment_steps_per_second": round(STEPS_PER_EPISODE / per_episode, 1),
        "seconds_per_update_4_episodes": round(per_episode * EPISODES_PER_UPDATE, 1),
        "decisions_per_episode": round(
            float(np.mean([item["collector"].decisions for item in episodes])), 2
        ),
        "candidates_per_decision": round(
            float(
                np.mean(
                    [
                        item["collector"].candidates / max(1, item["collector"].decisions)
                        for item in episodes
                    ]
                )
            ),
            3,
        ),
        "free_candidates_per_decision": round(
            float(
                np.mean(
                    [
                        item["collector"].timing()["free_candidates"]
                        / max(1, item["collector"].decisions)
                        for item in episodes
                    ]
                )
            ),
            3,
        ),
        "projection": {
            "one_run_to_400k_hours": round(400_000 / STEPS_PER_EPISODE * per_episode / 3600, 2),
            "six_runs_to_400k_serial_hours": round(
                6 * 400_000 / STEPS_PER_EPISODE * per_episode / 3600, 2
            ),
            "runs": 6,
        },
        "round4_reference": {
            "mean_seconds_per_update_six_runs_in_parallel": 24.1,
            "environment_steps_per_second": 167.0,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    if args.record:
        common.write_json(common.json_path("COMM_PROTOCOL_V2_THROUGHPUT.json"), payload)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("train", "throughput"), default="train")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tier", default=common.PRIMARY_TIER, choices=list(common.BUDGET_TIERS))
    parser.add_argument("--budget-bytes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument(
        "--output-root",
        default=None,
        help="where checkpoints go; one tree per run, so no two runs can read each other's state",
    )
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args(argv)
    if args.steps > 400_000:
        raise SystemExit(
            f"--steps {args.steps} exceeds the 400 000 per-run ceiling frozen in the brief; this "
            "round does not extend itself"
        )
    args.budget_bytes = args.budget_bytes or common.tier_bytes(args.tier)
    if args.budget_bytes != common.tier_bytes(args.tier):
        raise SystemExit(
            f"--budget-bytes {args.budget_bytes} is not the locked {args.tier} tier "
            f"({common.tier_bytes(args.tier)}); the tiers are calibration output, not a dial"
        )
    if args.mode == "throughput":
        return throughput(args)
    return train(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
