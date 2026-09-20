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
        "transcript": row["artefacts"]["transcript"],
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
        "transcript": row["artefacts"]["transcript"],
        "seconds": float(row["seconds"]),
    }


def rows_path(
    tier: str,
    setting: tuple[str, float],
    directory: Path,
    suffix: str = "",
    shard: int | None = None,
) -> Path:
    information, probability = setting
    name = f"rows_{tier}_{information}_{probability}{suffix}"
    if shard is not None:
        # One file per shard: two processes appending to one JSONL can interleave a half-written
        # line, and a reader that trips over it should not have to guess which half it got.
        name = f"{name}_s{shard}"
    return directory / f"{name}.jsonl"


def write_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_rows(
    tier: str, setting: tuple[str, float], directory: Path, suffix: str = ""
) -> list[dict[str, Any]]:
    """Every row for one tier, setting and suffix, from the sharded files as well as the shared one.

    The unsuffixed read is an exact filename rather than a glob: a glob would also pick up the
    curve and diagnostic files, which are different readings and must never be pooled with it.
    """
    information, probability = setting
    if not suffix:
        paths = [directory / f"rows_{tier}_{information}_{probability}.jsonl"]
    else:
        paths = sorted(directory.glob(f"rows_{tier}_{information}_{probability}{suffix}*.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A trailing line cut by a concurrent writer is not a row. Anything else is a real
                # defect, and it will surface as a coverage gap in the report rather than silently.
                continue
    # A cell is one (arm, stream) pair. An interrupted shard can redo a cell it had already
    # written, and a duplicated row would silently double that scenario's weight in every mean, so
    # the last write for a cell wins.
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        unique[(str(row["arm"]), int(row["seed"]))] = row
    return list(unique.values())


def done_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(str(row["arm"]), int(row["seed"])) for row in rows}


def curve_mode(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(1, int(args.threads)))
    directory = common.evaluation_root(args.tier)
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    # One job per (run, checkpoint, validation stream): the training seed names the weights and the
    # stream names the scenario, which are different things and must not share a variable.
    tasks = [
        (seed, at_steps, stream)
        for seed in (1, 2, 3)
        for at_steps in CHECKPOINT_STEPS
        for stream in args.seeds
    ]
    for seed, at_steps, stream in tasks[args.shard :: args.shards]:
        arm = arm_label("learned", args.tier, seed, at_steps=at_steps)
        rows = load_rows(args.tier, PRIMARY_SETTING, directory, suffix="_curve")
        if (arm, stream) in done_keys(rows) and not args.overwrite:
            continue
        controller, _ = load_controller(args.tier, seed, at_steps)
        row = score_learned(
            controller,
            seed=stream,
            tier=args.tier,
            directory=directory / f"runs_s{args.shard}",
            arm=arm,
            setting=PRIMARY_SETTING,
        )
        row["at_steps"] = at_steps
        row["greedy"] = True
        write_row(
            rows_path(args.tier, PRIMARY_SETTING, directory, "_curve", shard=args.shard), row
        )
        written += 1
        print(
            f"[curve {args.tier}] run{seed}@{at_steps} s{stream}: "
            f"onT={row['on_time_targets_per_episode']} bytes={row['airtime_bytes']} ({row['seconds']}s)",
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
                directory=directory / f"runs_s{args.shard}",
                setting=PRIMARY_SETTING,
                table="legacy" if kind == "legacy" else "v2",
            )
        else:
            at_steps = 400_000 if kind == "learned_final" else 0
            controller, _ = load_controller(args.tier, int(seed), at_steps)
            row = score_learned(
                controller,
                seed=stream,
                tier=args.tier,
                directory=directory / f"runs_s{args.shard}",
                arm=arm,
                setting=PRIMARY_SETTING,
            )
            row["at_steps"] = at_steps
        write_row(
            rows_path(args.tier, PRIMARY_SETTING, directory, shard=args.shard), row
        )
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
                    directory=directory / f"runs_s{args.shard}",
                    setting=setting,
                )
                write_row(
                    rows_path(args.tier, setting, directory, suffix, shard=args.shard), row
                )
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
                    directory=directory / f"runs_s{args.shard}",
                    arm=arm,
                    setting=setting,
                )
                row["at_steps"] = 400_000
                write_row(
                    rows_path(args.tier, setting, directory, suffix, shard=args.shard), row
                )
                written += 1
                print(
                    f"[diag {args.tier} {setting}] {arm} s{stream}: onT={row['on_time_targets_per_episode']} "
                    f"bytes={row['airtime_bytes']}",
                    flush=True,
                )
    print(json.dumps({"rows_written": written}, ensure_ascii=False))
    return 0


def sequence_mode(args: argparse.Namespace) -> int:
    """Section 6's "look at the action and message sequences, not just the completions".

    The locked comparator, the two 3/10 equivalents and every final learned arm, on the same
    validation streams, with each episode's transport transcript kept: what went on the air and
    when, how long a change waited, how long a discovery waited for its first report, and whether
    two arms that finish the same tasks did it with the same packets. Equal completions are not
    equal behaviour -- and "this policy converged onto that rule" is a claim about sequences, so it
    has to be able to say which of the two it is.
    """
    torch.set_num_threads(max(1, int(args.threads)))
    directory = common.evaluation_root(args.tier)
    directory.mkdir(parents=True, exist_ok=True)
    locked = json.loads(common.json_path("COMM_PROTOCOL_V2_RULE_SELECTION.json").read_text(encoding="utf-8"))
    comparator = locked["tiers"][args.tier]["selected_primary"]
    arms: list[tuple[str, Any]] = [(comparator, None)]
    for arm in ("first_send_plus_bound_3_10", "bound_only_3_10"):
        arms.append((arm, None))
    for seed in (1, 2, 3):
        controller, _ = load_controller(args.tier, seed, 400_000)
        arms.append((arm_label("learned", args.tier, seed, at_steps=400_000), controller))
    report: dict[str, Any] = {
        "tier": args.tier,
        "comparator": comparator,
        "budget_bytes": common.tier_bytes(args.tier),
        "setting": {"information": PRIMARY_SETTING[0], "delivery_probability": PRIMARY_SETTING[1]},
        "streams": len(args.seeds),
        "arms": {},
    }
    sequences: dict[str, dict[int, list[list[int]]]] = {}
    for arm, controller in arms:
        rows: list[dict[str, Any]] = []
        sequences[arm] = {}
        waits: list[int] = []
        for stream in args.seeds:
            row = (
                score_rule(
                    arm, tier=args.tier, seed=stream, directory=directory / f"seq_s{args.shard}", setting=PRIMARY_SETTING
                )
                if controller is None
                else score_learned(
                    controller,
                    seed=stream,
                    tier=args.tier,
                    directory=directory / f"seq_s{args.shard}",
                    arm=arm,
                    setting=PRIMARY_SETTING,
                )
            )
            row["at_steps"] = 400_000 if controller is not None else None
            write_row(
                rows_path(args.tier, PRIMARY_SETTING, directory, "_sequence", shard=args.shard), row
            )
            rows.append(row)
            records = common.metrics.load_transcript(row["transcript"])
            sequences[arm][stream] = _send_sequence(records)
            waits.extend(
                int(record["step"]) - int(record["ready_step"])
                for record in records
                if record["kind"] == "send"
            )
        report["arms"][arm] = {
            "episodes": len(rows),
            "on_time_per_episode": _summary(rows, "on_time_targets_per_episode"),
            "airtime_bytes": _summary(rows, "airtime_bytes"),
            "emissions_mean": round(float(np.mean([int(r["emissions"]) for r in rows])), 3),
            "forced_share_of_emissions_mean": round(
                float(np.mean([r["forced_share_of_emissions"] or 0.0 for r in rows])), 4
            ),
            "autonomous_share_of_bytes_mean": round(
                float(np.mean([r["autonomous_share_of_bytes"] or 0.0 for r in rows])), 4
            ),
            "longest_silence_max": max(int(r["max_backlog_age"]) for r in rows),
            "belief_age_steps_mean": round(float(np.mean([float(r["belief_age_steps"]) for r in rows])), 3),
            "mean_wait_of_a_change_steps": round(float(np.mean(waits)), 3) if waits else None,
            "max_wait_of_a_change_steps": max(waits) if waits else None,
        }
        print(
            f"[sequence {args.tier}] {arm}: onT={report['arms'][arm]['on_time_per_episode']['mean']} "
            f"bytes={report['arms'][arm]['airtime_bytes']['mean']} "
            f"em={report['arms'][arm]['emissions_mean']} "
            f"longest_silence={report['arms'][arm]['longest_silence_max']}",
            flush=True,
        )
    report["sequence_comparison"] = _sequence_comparison(sequences, comparator)
    common.write_json(common.json_path("COMM_PROTOCOL_V2_SEQUENCES.json"), report)
    print(json.dumps({"arms": len(report["arms"])}, ensure_ascii=False))
    return 0


def _send_sequence(records: list[dict[str, Any]]) -> list[list[Any]]:
    """Every emission as (step, target, position within the step), in the order taken."""
    positions: dict[int, int] = defaultdict(int)
    sequence: list[list[Any]] = []
    for record in records:
        if record["kind"] != "send":
            continue
        step = int(record["step"])
        sequence.append([step, str(record["target"]), positions[step]])
        positions[step] += 1
    return sequence


def _sequence_comparison(
    sequences: dict[str, dict[int, list[list[Any]]]], comparator: str
) -> dict[str, Any]:
    """Per arm, how often it emitted the same packets at the same steps as the comparator.

    Three ways to differ, kept apart because they mean different things: a packet the arm sent and
    the comparator did not, a packet the comparator sent and the arm did not, and the same packets
    in a different order or at different steps. "Same completions" and "same behaviour" are only
    the same sentence when all three counts are zero.
    """
    report: dict[str, Any] = {}
    for arm, per_stream in sorted(sequences.items()):
        if arm == comparator:
            continue
        identical = 0
        extra = 0
        missing = 0
        retimed = 0
        for stream, sequence in per_stream.items():
            other = sequences[comparator].get(stream, [])
            if sequence == other:
                identical += 1
                continue
            left = {(step, target) for step, target, _ in sequence}
            right = {(step, target) for step, target, _ in other}
            extra += len(left - right)
            missing += len(right - left)
            if left == right:
                retimed += 1
        report[arm] = {
            "streams_identical": identical,
            "streams_compared": len(per_stream),
            "packets_extra": extra,
            "packets_missing": missing,
            "streams_same_packets_different_timing": retimed,
        }
    return report


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
        curve = load_rows(tier, PRIMARY_SETTING, common.evaluation_root(tier), "_curve")
        curve_table: dict[str, Any] = {}
        for arm in sorted({str(row["arm"]) for row in curve}):
            values = [row for row in curve if row["arm"] == arm]
            curve_table[arm] = {
                "episodes": len(values),
                "at_steps": values[0].get("at_steps"),
                "on_time_per_episode": _summary(values, "on_time_targets_per_episode"),
                "airtime_bytes": _summary(values, "airtime_bytes"),
            }
        completion = common.paired_bootstrap(
            rows, baseline=comparator, key="on_time_targets_per_episode"
        )
        bytes_paired = common.paired_bootstrap(rows, baseline=comparator, key="airtime_bytes")
        # One arm is one *training seed*. The frozen gate counts training seeds, so each arm's CI
        # is computed against the comparator over that arm's own 24 scenario pairs -- pooling all
        # three would treat 3 x 24 as 72 independent scenarios, which the protocol forbids.
        per_seed: dict[str, Any] = {}
        for arm in sorted(
            arm for arm in by_arm if arm in learned_arms or arm.startswith("init_")
        ):
            subset = [row for row in rows if row["arm"] in {arm, comparator}]
            own_completion = common.paired_bootstrap(
                subset, baseline=comparator, key="on_time_targets_per_episode"
            )["arms"].get(arm, {})
            own_bytes = common.paired_bootstrap(
                subset, baseline=comparator, key="airtime_bytes"
            )["arms"].get(arm, {})
            per_seed[arm] = {
                "completion_difference_per_episode": own_completion,
                "byte_difference": own_bytes,
                "primary_gate_seed_passes": bool(own_completion and own_completion["ci95"][0] > 0),
                "saving_gate_seed_passes": bool(
                    own_completion
                    and own_bytes
                    and own_bytes["ci95"][1] < 0
                    and own_completion["ci95"][0] >= -0.05
                ),
            }
        learned_seed_arms = [arm for arm in sorted(learned_arms)]
        # §7 comparison 3: each final model against its own untrained weights, per stream.
        initialisation: dict[str, Any] = {}
        for arm in learned_seed_arms:
            initial = f"init_{tier}_seed{arm.rsplit('_seed', 1)[1].split('_at', 1)[0]}"
            if initial not in by_arm:
                continue
            baseline_by_stream = {
                int(row["seed"]): float(row["on_time_targets_per_episode"])
                for row in by_arm[initial]
            }
            initialisation[arm] = _summary(
                [
                    {
                        "value": float(row["on_time_targets_per_episode"])
                        - baseline_by_stream[int(row["seed"])]
                    }
                    for row in by_arm[arm]
                    if int(row["seed"]) in baseline_by_stream
                ],
                "value",
            )
        passing = [arm for arm in learned_seed_arms if per_seed[arm]["primary_gate_seed_passes"]]
        remaining = [arm for arm in learned_seed_arms if arm not in passing]
        report["tiers"][tier] = {
            "comparator": comparator,
            "episodes": len(rows),
            "arms": table,
            # The curve is reported beside the frozen judgement, never instead of it.
            "learning_curves": curve_table,
            "pooled_for_reference_only": {
                "on_time_targets_per_episode": completion,
                "airtime_bytes": bytes_paired,
                "warning": (
                    "pooled across training seeds, so its CI is optimistic: the frozen judgement "
                    "reads per_seed below"
                ),
            },
            "per_training_seed": per_seed,
            "learning_gate_per_seed": {
                arm: {
                    "primary_gate_seed_passes": per_seed[arm]["primary_gate_seed_passes"],
                    "saving_gate_seed_passes": per_seed[arm]["saving_gate_seed_passes"],
                    "completion_difference_per_episode": per_seed[arm]["completion_difference_per_episode"],
                    "byte_difference": per_seed[arm]["byte_difference"],
                }
                for arm in learned_seed_arms
            },
            "gate_verdict": {
                "seeds_passing": passing,
                "seeds_not_passing": remaining,
                "third_seed_point_estimate": {
                    arm: per_seed[arm]["completion_difference_per_episode"].get("mean_difference")
                    for arm in remaining
                },
                "third_seed_within_margin": all(
                    (per_seed[arm]["completion_difference_per_episode"].get("mean_difference") or 0.0)
                    >= -0.05
                    for arm in remaining
                ),
                "primary_gate_met": len(passing) >= 2
                and all(
                    (per_seed[arm]["completion_difference_per_episode"].get("mean_difference") or 0.0)
                    >= -0.05
                    for arm in remaining
                ),
            },
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
    """The frozen gate, applied per tier and per training seed exactly as §6 wrote it.

    A tier's primary gate is met when at least two of the three training seeds show a positive 95 %
    CI lower bound against the locked comparator and the remaining seed's point estimate is no
    worse than -0.05 targets/episode. Nothing here pools seeds, and nothing here relaxes the
    margin because a result was close.
    """
    passed: list[str] = []
    saved: list[str] = []
    for tier, block in report["tiers"].items():
        verdict = block["gate_verdict"]
        if verdict["primary_gate_met"]:
            passed.append(f"{tier}: {','.join(verdict['seeds_passing'])}")
        for arm, value in block["learning_gate_per_seed"].items():
            if value["saving_gate_seed_passes"]:
                saved.append(f"{arm}@{tier}")
    return {
        "primary_gate_passed": passed,
        "saving_gate_passed": sorted(saved),
        "primary_gate_met": bool(passed),
        "saving_gate_met": bool(saved),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode", choices=("curve", "arms", "diagnostics", "sequence", "report"), required=True
    )
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
    if args.mode == "sequence":
        if not args.tier:
            raise SystemExit("--mode sequence needs --tier")
        return sequence_mode(args)
    # With or without --tier, one payload holds every tier that was asked for: writing the file
    # once per tier would leave the last tier's results standing in for both, which is exactly the
    # substitution the frozen protocol forbids.
    return report_mode(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
