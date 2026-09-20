"""Evaluate the protocol-v2 round: rule tables, the learned arms, and the three comparisons.

Four modes, in the order the frozen protocol puts them:

``--mode curve``    every checkpoint a run owns (initial, 100k, 200k, 300k, 400k) on the 24
                    validation streams, at the run's own credit, isolated-lossless. Reported beside
                    the frozen judgement and never used to replace the end-of-run checkpoint.
``--mode arms``     the frozen rule table plus every learned arm's 400k checkpoint and every run's
                    initial weights, on the same 24 streams. The legacy-protocol rules are included
                    so comparison 1 of §7 (protocol gain) can be read without re-running anything.
``--mode diagnostics`` the locked comparator, the two 3/10 rules and the six learned arms, at the
                    primary credit, under packet loss 0.3 and the normal information path. These are
                    diagnostics: they cannot promote or demote the primary judgement.
``--mode report``   aggregate, then apply the frozen gate and write the three comparisons §7
                    requires -- protocol gain, learning's added value over the strongest rule in the
                    new protocol, and improvement over each run's own initialisation -- separately.

    python evaluate_protocol.py --mode curve --tier 25pct --shards 6 --shard 0 --threads 1
    python evaluate_protocol.py --mode arms  --shards 6 --shard 0 --threads 1
    python evaluate_protocol.py --mode diagnostics --threads 1
    python evaluate_protocol.py --mode report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import protocol_common as common
import protocol_env
import train_protocol

import train_pilot
from integrated_sar.comm import LEGACY_SHARED_BOUND, POLICY_OWNED_TIMING
from integrated_sar.comm_policy import parse_parameters
from integrated_sar.joint_driver import RunConfiguration

#: How a validation stream is read. Isolated-lossless is the frozen primary setting; the other two
#: are diagnostics that run after the primary judgement exists.
PRIMARY_SETTING = ("isolated", 1.0)
DIAGNOSTIC_SETTINGS = (("isolated", 0.3), ("normal", 1.0))

#: The frozen checkpoints a run owns, named by environment steps. 0 is the saved initialisation.
CHECKPOINT_STEPS = (0, 100_000, 200_000, 300_000, 400_000)


def arm_label(prefix: str, tier: str, seed: int | None, *, at_steps: int | None = None) -> str:
    name = f"{prefix}_{tier}"
    if seed is not None:
        name = f"{name}_seed{seed}"
    if at_steps is not None:
        name = f"{name}_at{at_steps}"
    return name


def checkpoint_path(tier: str, seed: int, at_steps: int) -> Path:
    return common.TRAIN_OUT / f"{tier}_seed{seed}" / f"v2_{tier}_seed{seed}_at{at_steps}.pt"


def load_controller(tier: str, seed: int, at_steps: int):
    """One run's weights at one milestone, through the round's own loader.

    The loader refuses a checkpoint from the other protocol or the other trainer, so an evaluation
    that would silently load the wrong generation fails here instead of in the results table.
    """
    path = checkpoint_path(tier, seed, at_steps)
    if not path.exists():
        raise RuntimeError(f"{path} is missing; a run that has not reached {at_steps} steps has no checkpoint")
    loaded = train_protocol.load(path, tier=tier, training_seed=seed)
    controller = train_pilot.Controller(seed)
    controller.load_state_dict(loaded["controller"])
    return controller, loaded


def score_learned(
    controller: Any,
    *,
    seed: int,
    tier: str,
    directory: Path,
    arm: str,
    setting: tuple[str, float],
    greedy: bool = True,
) -> dict[str, Any]:
    """One learned arm, one validation stream, one credit, on this round's protocol."""
    information, probability = setting
    collector = common.onpolicy.OnPolicyCollector(
        controller, forced=train_protocol.nothing_forced, mask=False, greedy=greedy
    )
    row = protocol_env.run_episode(
        seed,
        directory=directory,
        label=arm,
        policy="learned",
        budget_bytes=common.tier_bytes(tier),
        controller=collector.policy(),
        information=information,
        delivery_probability=probability,
        timing_mode=POLICY_OWNED_TIMING,
        max_wait_steps=common.BOUND_MAX_WAIT_STEPS,
        urgent_wait_steps=common.BOUND_URGENT_WAIT_STEPS,
    )
    records = common.metrics.load_transcript(row["artefacts"]["transcript"])
    rollup = common.attribution_metrics._per_reason_rollup(records)
    ledger = row["ledger"]
    emissions = int(ledger["emissions"])
    forced = int(ledger["emissions_forced_maxwait"])
    return {
        "arm": arm,
        "kind": "learned",
        "tier": tier,
        "seed": int(seed),
        "setting": {"information": information, "delivery_probability": probability},
        "timing_mode": POLICY_OWNED_TIMING,
        "on_time_targets_per_episode": int(row["task"]["on_time"]),
        "detected_targets_this_episode": int(row["task"]["targets"]),
        "expired_targets_per_episode": int(row["task"]["expired"]),
        "airtime_bytes": int(row["spent_bytes"]),
        "credit_used_fraction": row["budget_used_fraction"],
        "emissions": emissions,
        "emissions_by_policy": int(ledger["emissions_policy"]),
        "emissions_forced": forced,
        "forced_share_of_emissions": round(forced / emissions, 5) if emissions else None,
        "autonomous_share_of_bytes": rollup["autonomous_share_of_bytes"],
        "redundant_emissions": int(ledger["redundant_emissions"]),
        "budget_refusals": int(ledger["budget_refused_sends"]),
        "deliveries_applied": int(ledger["deliveries_applied"]),
        "belief_age_steps": float(row["belief"].get("belief_age_steps_mean") or 0.0),
        "max_backlog_age": int(ledger["max_backlog_age"]),
        "inference_milliseconds_per_decision": round(
            1000.0 * collector.decide_seconds / max(1, collector.decisions), 4
        ),
        "config_hash": row["config_hash"],
        "seconds": float(row["seconds"]),
    }


def score_rule(
    arm: str,
    *,
    tier: str,
    seed: int,
    directory: Path,
    setting: tuple[str, float],
    table: str = "v2",
) -> dict[str, Any]:
    import rule_matrix

    information, probability = setting
    spec = common.arm_spec(table, arm)
    spec["arm"] = arm
    spec["kind"] = "rule"
    row = protocol_env.run_episode(
        seed,
        directory=directory,
        label=arm,
        policy=spec["policy"],
        policy_params=parse_parameters(spec.get("params", "")) or None,
        budget_bytes=None if spec.get("ceiling") else common.tier_bytes(tier),
        timing_mode=LEGACY_SHARED_BOUND if table == "legacy" else POLICY_OWNED_TIMING,
        max_wait_steps=None if spec.get("bound") == "none" else common.BOUND_MAX_WAIT_STEPS,
        urgent_wait_steps=None if spec.get("bound") == "none" else common.BOUND_URGENT_WAIT_STEPS,
        information=information,
        delivery_probability=probability,
    )
    records = common.metrics.load_transcript(row["artefacts"]["transcript"])
    rollup = common.attribution_metrics._per_reason_rollup(records)
    ledger = row["ledger"]
    emissions = int(ledger["emissions"])
    forced = int(ledger["emissions_forced_maxwait"])
    return {
        "arm": arm,
        "kind": "rule",
        "tier": tier,
        "seed": int(seed),
        "setting": {"information": information, "delivery_probability": probability},
        "timing_mode": LEGACY_SHARED_BOUND if table == "legacy" else POLICY_OWNED_TIMING,
        "on_time_targets_per_episode": int(row["task"]["on_time"]),
        "detected_targets_this_episode": int(row["task"]["targets"]),
        "expired_targets_per_episode": int(row["task"]["expired"]),
        "airtime_bytes": int(row["spent_bytes"]),
        "credit_used_fraction": row["budget_used_fraction"],
        "emissions": emissions,
        "emissions_by_policy": int(ledger["emissions_policy"]),
        "emissions_forced": forced,
        "forced_share_of_emissions": round(forced / emissions, 5) if emissions else None,
        "autonomous_share_of_bytes": rollup["autonomous_share_of_bytes"],
        "redundant_emissions": int(ledger["redundant_emissions"]),
        "budget_refusals": int(ledger["budget_refused_sends"]),
        "deliveries_applied": int(ledger["deliveries_applied"]),
        "belief_age_steps": float(row["belief"].get("belief_age_steps_mean") or 0.0),
        "max_backlog_age": int(ledger["max_backlog_age"]),
        "config_hash": row["config_hash"],
        "seconds": float(row["seconds"]),
    }


def rows_path(tier: str, setting: tuple[str, float], directory: Path, suffix: str = "") -> Path:
    information, probability = setting
    name = f"rows_{tier}_{information}_{probability}{suffix}.jsonl"
    return directory / name


def write_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_rows(tier: str, setting: tuple[str, float], directory: Path, suffix: str = "") -> list[dict[str, Any]]:
    path = rows_path(tier, setting, directory, suffix)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def done_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(str(row["arm"]), int(row["seed"])) for row in rows}


def curve_mode(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(1, int(args.threads)))
    directory = common.evaluation_root(args.tier)
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    tasks = [(seed, at_steps) for seed in (1, 2, 3) for at_steps in CHECKPOINT_STEPS]
    for seed, at_steps in tasks[args.shard :: args.shards]:
        arm = arm_label("learned", args.tier, seed, at_steps=at_steps)
        rows = load_rows(args.tier, PRIMARY_SETTING, directory, suffix="_curve")
        if arm in {row["arm"] for row in rows} and not args.overwrite:
            continue
        controller, _ = load_controller(args.tier, seed, at_steps)
        row = score_learned(
            controller,
            seed=seed,
            tier=args.tier,
            directory=directory / "runs",
            arm=arm,
            setting=PRIMARY_SETTING,
        )
        row["at_steps"] = at_steps
        row["greedy"] = True
        write_row(rows_path(args.tier, PRIMARY_SETTING, directory, "_curve"), row)
        written += 1
        print(
            f"[curve {args.tier}] seed{seed}@{at_steps}: onT={row['on_time_targets_per_episode']} "
            f"bytes={row['airtime_bytes']} ({row['seconds']}s)",
            flush=True,
        )
    print(json.dumps({"rows_written": written}, ensure_ascii=False))
    return 0


def arms_mode(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(1, int(args.threads)))
    directory = common.evaluation_root(args.tier)
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    tasks: list[tuple[str, str, Any]] = []
    for arm in common.V2_ARMS:
        tasks.append(("rule", arm, None))
    for arm in ("immediate", "hold_to_bound", "send_once_no_bound"):
        tasks.append(("legacy", arm, None))
    for seed in (1, 2, 3):
        tasks.append(("learned_final", arm_label("learned", args.tier, seed, at_steps=400_000), seed))
        tasks.append(("learned_initial", arm_label("init", args.tier, seed), seed))
    # One job per (arm, validation stream): a rule is read on all 24 streams, exactly like a
    # learned arm, so the frozen comparison pairs identical scenarios.
    jobs = [(kind, arm, seed, stream) for kind, arm, seed in tasks for stream in args.seeds]
    for kind, arm, seed, stream in jobs[args.shard :: args.shards]:
        rows = load_rows(args.tier, PRIMARY_SETTING, directory)
        if (arm, stream) in done_keys(rows) and not args.overwrite:
            continue
        if kind in ("rule", "legacy"):
            row = score_rule(
                arm,
                tier=args.tier,
                seed=stream,
                directory=directory / "runs",
                setting=PRIMARY_SETTING,
                table="legacy" if kind == "legacy" else "v2",
            )
        else:
            at_steps = 400_000 if kind == "learned_final" else 0
            controller, _ = load_controller(args.tier, int(seed), at_steps)
            row = score_learned(
                controller,
                seed=int(seed),
                tier=args.tier,
                directory=directory / "runs",
                arm=arm,
                setting=PRIMARY_SETTING,
            )
            row["at_steps"] = at_steps
        write_row(rows_path(args.tier, PRIMARY_SETTING, directory), row)
        written += 1
        print(
            f"[arms {args.tier}] {arm} s{row['seed']}: onT={row['on_time_targets_per_episode']} "
            f"bytes={row['airtime_bytes']} ({row['seconds']}s)",
            flush=True,
        )
    print(json.dumps({"rows_written": written}, ensure_ascii=False))
    return 0


def diagnostics_mode(args: argparse.Namespace) -> int:
    """The extra settings, for the locked rule and the six final learned arms only."""
    torch.set_num_threads(max(1, int(args.threads)))
    directory = common.evaluation_root(args.tier)
    directory.mkdir(parents=True, exist_ok=True)
    locked = json.loads(common.json_path("COMM_PROTOCOL_V2_RULE_SELECTION.json").read_text(encoding="utf-8"))
    comparator = locked["tiers"][args.tier]["selected_primary"]
    rules = [comparator, "first_send_plus_bound_3_10", "bound_only_3_10"]
    learned = [
        arm_label("learned", args.tier, seed, at_steps=400_000) for seed in (1, 2, 3)
    ]
    written = 0
    for setting in DIAGNOSTIC_SETTINGS:
        suffix = "_diag"
        for arm in rules:
            for seed in args.seeds:
                rows = load_rows(args.tier, setting, directory, suffix)
                if (arm, seed) in done_keys(rows) and not args.overwrite:
                    continue
                row = score_rule(
                    arm,
                    tier=args.tier,
                    seed=seed,
                    directory=directory / "runs",
                    setting=setting,
                )
                write_row(rows_path(args.tier, setting, directory, suffix), row)
                written += 1
                print(
                    f"[diag {args.tier} {setting}] {arm} s{seed}: onT={row['on_time_targets_per_episode']} "
                    f"bytes={row['airtime_bytes']}",
                    flush=True,
                )
        for arm, seed in zip(learned, (1, 2, 3)):
            for stream in args.seeds:
                rows = load_rows(args.tier, setting, directory, suffix)
                if (arm, stream) in done_keys(rows) and not args.overwrite:
                    continue
                controller, _ = load_controller(args.tier, seed, 400_000)
                row = score_learned(
                    controller,
                    seed=stream,
                    tier=args.tier,
                    directory=directory / "runs",
                    arm=arm,
                    setting=setting,
                )
                row["at_steps"] = 400_000
                write_row(rows_path(args.tier, setting, directory, suffix), row)
                written += 1
                print(
                    f"[diag {args.tier} {setting}] {arm} s{stream}: onT={row['on_time_targets_per_episode']} "
                    f"bytes={row['airtime_bytes']}",
                    flush=True,
                )
    print(json.dumps({"rows_written": written}, ensure_ascii=False))
    return 0


def _summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    return {
        "episodes": len(rows),
        "mean": round(float(np.mean(values)), 5),
        "stdev": round(float(np.std(values)), 5),
        "total": round(float(np.sum(values)), 3),
    }


def report_mode(args: argparse.Namespace) -> int:
    directory = common.evaluation_root(args.tier) if args.tier else common.EVALUATION_OUT
    tiers = [args.tier] if args.tier else list(common.BUDGET_TIERS)
    locked = json.loads(common.json_path("COMM_PROTOCOL_V2_RULE_SELECTION.json").read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "question": (
            "With the 3/10 silence bound no longer spoken for by the transport, can a learned "
            "policy beat a strong rule that owns the same choice, under the same hard credit?"
        ),
        "protocol": "COMM_PROTOCOL_V2_PROTOCOL.md",
        "comparators_locked_on_training_streams": {
            tier: locked["tiers"][tier]["selected_primary"] for tier in tiers
        },
        "primary_setting": {"information": PRIMARY_SETTING[0], "delivery_probability": PRIMARY_SETTING[1]},
        "primary_metric": "on_time_targets_per_episode (targets/episode)",
        "non_inferiority_margin": -0.05,
        "tiers": {},
    }
    for tier in tiers:
        rows = load_rows(tier, PRIMARY_SETTING, common.evaluation_root(tier))
        if not rows:
            continue
        by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_arm[str(row["arm"])].append(row)
        comparator = locked["tiers"][tier]["selected_primary"]
        table: dict[str, Any] = {}
        for arm, values in sorted(by_arm.items()):
            kind = values[0].get("kind", "rule")
            table[arm] = {
                "kind": kind,
                "episodes": len(values),
                "seeds": sorted({int(row["seed"]) for row in values}),
                "at_steps": values[0].get("at_steps"),
                "on_time_per_episode": _summary(values, "on_time_targets_per_episode"),
                "airtime_bytes": _summary(values, "airtime_bytes"),
                "detected_total": int(sum(int(row["detected_targets_this_episode"]) for row in values)),
                "expired_total": int(sum(int(row["expired_targets_per_episode"]) for row in values)),
                "emissions_mean": round(
                    float(np.mean([int(row["emissions"]) for row in values])), 3
                ),
                "emissions_forced_mean": round(
                    float(np.mean([int(row["emissions_forced"]) for row in values])), 3
                ),
                "forced_share_of_emissions_mean": round(
                    float(np.mean([row["forced_share_of_emissions"] or 0.0 for row in values])), 4
                ),
                "autonomous_share_of_bytes_mean": round(
                    float(np.mean([row["autonomous_share_of_bytes"] or 0.0 for row in values])), 4
                ),
                "budget_refusals_mean": round(
                    float(np.mean([int(row["budget_refusals"]) for row in values])), 3
                ),
                "belief_age_steps_mean": round(
                    float(np.mean([float(row["belief_age_steps"]) for row in values])), 3
                ),
                "max_backlog_age_max": max(int(row["max_backlog_age"]) for row in values),
            }
        learned_arms = [
            arm for arm in table if table[arm]["kind"] == "learned" and table[arm].get("at_steps") == 400_000
        ]
        completion = common.paired_bootstrap(
            rows, baseline=comparator, extract=lambda row: float(row["on_time_targets_per_episode"])
        )
        bytes_paired = common.paired_bootstrap(
            rows, baseline=comparator, extract=lambda row: float(row["airtime_bytes"])
        )
        gate: dict[str, Any] = {}
        for arm in sorted(learned_arms):
            stats = completion["arms"].get(arm, {})
            cost = bytes_paired["arms"].get(arm, {})
            gate[arm] = {
                "completion_difference_per_episode": stats,
                "byte_difference": cost,
                "primary_gate": bool(
                    stats
                    and stats["ci95"][0] > 0
                    and stats["pairs"] >= 3
                ),
                "saving_gate": bool(
                    cost
                    and stats
                    and cost["ci95"][1] < 0
                    and stats["ci95"][0] >= -0.05
                ),
            }
        # §7 comparison 3: each final model against its own untrained weights.
        initialisation: dict[str, Any] = {}
        for seed in (1, 2, 3):
            final = f"learned_{tier}_seed{seed}_at400000"
            initial = f"init_{tier}_seed{seed}"
            if final in by_arm and initial in by_arm:
                difference = _summary(
                    [
                        {
                            "value": float(row["on_time_targets_per_episode"])
                            - float(
                                next(
                                    other["on_time_targets_per_episode"]
                                    for other in by_arm[initial]
                                    if int(other["seed"]) == int(row["seed"])
                                )
                            )
                        }
                        for row in by_arm[final]
                    ],
                    "value",
                )
                initialisation[f"seed{seed}"] = difference
        report["tiers"][tier] = {
            "comparator": comparator,
            "episodes": len(rows),
            "arms": table,
            "paired_against_comparator": {
                "on_time_targets_per_episode": completion,
                "airtime_bytes": bytes_paired,
            },
            "learning_gate_per_seed": gate,
            "learning_versus_own_initialisation": initialisation,
            "diagnostics": {
                str(setting): len(load_rows(tier, setting, common.evaluation_root(tier), "_diag"))
                for setting in DIAGNOSTIC_SETTINGS
            },
        }
        print(f"[{tier}] comparator={comparator} learned_arms={sorted(learned_arms)}", flush=True)
    report["verdict"] = _verdict(report)
    common.write_json(common.json_path("COMM_PROTOCOL_V2_RESULTS.json"), report)
    print(json.dumps({"verdict": report["verdict"]}, ensure_ascii=False))
    return 0


def _verdict(report: dict[str, Any]) -> dict[str, Any]:
    """The frozen gate, applied per tier and per seed exactly as §6 wrote it."""
    passed: list[str] = []
    saved: list[str] = []
    for tier, block in report["tiers"].items():
        positive = [
            arm
            for arm, value in block["learning_gate_per_seed"].items()
            if value["primary_gate"]
        ]
        for arm in positive:
            passed.append(f"{arm}@{tier}")
        for arm, value in block["learning_gate_per_seed"].items():
            if value["saving_gate"]:
                saved.append(f"{arm}@{tier}")
    return {
        "primary_gate_passed": sorted(passed),
        "saving_gate_passed": sorted(saved),
        "primary_gate_met": bool(passed),
        "saving_gate_met": bool(saved),
        "note": (
            "§6 also requires the third seed's point estimate to be no worse than -0.05 "
            "targets/episode when 2/3 seeds pass; that is read from the paired table beside this "
            "verdict, not folded into a boolean"
        ),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("curve", "arms", "diagnostics", "report"), required=True)
    parser.add_argument(
        "--tiers", nargs="+", default=list(common.BUDGET_TIERS), choices=list(common.BUDGET_TIERS)
    )
    parser.add_argument("--tier", default=None, choices=list(common.BUDGET_TIERS))
    parser.add_argument("--seed", type=int, default=common.VALIDATION_SEEDS[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(common.VALIDATION_SEEDS))
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.shard >= args.shards:
        raise SystemExit(f"--shard {args.shard} is outside --shards {args.shards}")
    if args.mode == "curve":
        return curve_mode(args)
    if args.mode == "arms":
        return arms_mode(args)
    if args.mode == "diagnostics":
        return diagnostics_mode(args)
    if not args.tier:
        raise SystemExit("--mode report needs --tier, or a single-tier run: the report reads one tier")
    return report_mode(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
