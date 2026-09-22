"""Record one pressure layer's 16 detection streams, then register what was actually placed.

The recording configuration is the frozen block's own constant with ``num_goals`` set to the
layer's value -- :func:`pressure_common.assert_only_num_goals_changed` proves no other field moved
before a single stream is written. Each stream is recorded once and replayed by all nine arms, so
a difference between arms cannot be a difference in what was searched.

The recorder is idempotent: an existing stream is reported, never rewritten. A re-run needs a
pre-registered reason and moves the old file to ``stream_reruns/`` instead of overwriting it,
because a seed may be recorded twice but never silently replaced, and a hard seed may never be
dropped for a prettier one.

    python record_layer.py --goals 6 --shards 4 --shard 0
    python record_layer.py --goals 6 --mode merge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pressure_common as common

from integrated_sar.detection_replay import scenario_map_sha256
from integrated_sar.joint_driver import RunConfiguration, build_joint_run
from integrated_sar.scenarios import build_scenario

#: Modules whose git-blob identity is frozen into the registration: the code that decided where
#: targets are, how they are found and how the world books a task.
RECORDED_MODULES = (
    "integrated_sar/joint_driver.py",
    "integrated_sar/contracts.py",
    "integrated_sar/task_registry.py",
    "integrated_sar/orchestrator.py",
    "integrated_sar/adapters/task03.py",
    "integrated_sar/adapters/piloc.py",
    "integrated_sar/comm.py",
    "integrated_sar/scenarios.py",
    "integrated_sar/piloc_runtime.py",
    "integrated_sar/search_runtime.py",
    "integrated_sar/lifecycle_log.py",
    "integrated_sar/detection_replay.py",
    "integrated_sar/expired_attribution.py",
)

RERUN_REASONS = ("process_killed", "disk_full", "checkpoint_load_failure")


def require_clean_audit() -> dict[str, Any]:
    path = common.json_path("SEED_COLLISION_AUDIT.json")
    if not path.exists():
        raise SystemExit(
            "SEED_COLLISION_AUDIT.json does not exist; the four-source audit runs before any "
            "stream is recorded"
        )
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not audit.get("clean"):
        raise SystemExit(f"the seed audit is not clean; recording is not allowed: {audit.get('verdict')}")
    for goals, block in audit["blocks_audited"].items():
        seeds = common.LAYERS[int(goals)]
        if [seeds[0], seeds[-1]] != list(block["seeds"]):
            raise SystemExit(
                f"the audit covers {block['seeds']} for goals={goals} but this round registered "
                f"{[seeds[0], seeds[-1]]}; the block moved after the audit"
            )
    return audit


def code_identity() -> dict[str, Any]:
    modules = {relative: common.blob_identity(relative) for relative in RECORDED_MODULES}
    return {
        "branch": common.git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": common.git("rev-parse", "HEAD"),
        "comparison": "git blob identity (line-ending filters applied)",
        "modules": modules,
        "all_modules_match_head": all(value["matches_head"] for value in modules.values()),
    }


def scenario_facts(goals: int, seed: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the world from its seed and prove the recorded stream is that world."""
    scenario = build_scenario(
        seed=int(seed),
        num_agents=int(common.RECORD_CONFIGURATION["search_agents"]),
        num_goals=int(goals),
        size=int(common.RECORD_CONFIGURATION["grid_size"]),
        obstacle_density=float(common.RECORD_CONFIGURATION["obstacle_density"]),
    )
    recorded = payload.get("scenario", {})
    goals_cells = [list(cell) for cell in recorded.get("goal_positions", [])]
    agent_cells = [[int(cell.row), int(cell.col)] for cell in scenario.agent_positions]
    distinct = len({tuple(cell) for cell in goals_cells}) == len(goals_cells)
    collisions = sorted({tuple(cell) for cell in goals_cells} & {tuple(cell) for cell in agent_cells})
    detections = {
        str(detection.get("target_id", detection.get("target")))
        for frame in payload.get("frames", [])
        for detection in frame.get("detections", [])
    }
    first_detection: dict[str, float] = {}
    for frame in payload.get("frames", []):
        for detection in frame.get("detections", []):
            target = str(detection.get("target_id", detection.get("target")))
            moment = float(detection.get("detected_at", frame.get("step", 0.0)))
            first_detection[target] = min(first_detection.get(target, moment), moment)
    return {
        "placed": int(recorded.get("num_goals_placed", -1)),
        "configured_goals": int(goals),
        "goal_cells": goals_cells,
        "goal_cells_distinct": distinct,
        "goal_cells_on_agent_start": [list(cell) for cell in collisions],
        "map_sha256_reproduced": scenario_map_sha256(scenario) == str(payload.get("map_sha256")),
        "detected": len(detections),
        "detected_ratio_of_placed": (
            round(len(detections) / float(recorded.get("num_goals_placed") or 1), 5)
            if recorded.get("num_goals_placed")
            else None
        ),
        "detection_times": sorted(round(value, 3) for value in first_detection.values()),
        "last_detection_step": max(first_detection.values(), default=None),
        "steps_recorded": int(payload.get("steps_recorded", 0)),
        "frames": len(payload.get("frames", [])),
    }


def legality(facts: dict[str, Any]) -> list[str]:
    """What L0 calls a legal placed set, written as a function so the report can quote it."""
    problems: list[str] = []
    if facts["placed"] != facts["configured_goals"]:
        problems.append(f"placed {facts['placed']} != configured {facts['configured_goals']}")
    if not facts["goal_cells_distinct"]:
        problems.append("two targets share a cell")
    if facts["goal_cells_on_agent_start"]:
        problems.append(f"a target sits on a search-robot start: {facts['goal_cells_on_agent_start']}")
    if facts["detected"] > facts["placed"]:
        problems.append("more detections than placed targets")
    if facts["map_sha256_reproduced"] is not True:
        problems.append("the obstacle world does not reproduce from its seed")
    if facts["steps_recorded"] < common.STREAM_STEPS:
        problems.append(f"only {facts['steps_recorded']} steps recorded")
    return problems


def record(goals: int, seed: int, *, reason: str | None = None) -> dict[str, Any]:
    stream = common.stream_path(goals, seed)
    if stream.exists() and reason is None:
        payload = json.loads(stream.read_text(encoding="utf-8"))
        facts = scenario_facts(goals, seed, payload)
        return {
            "seed": seed,
            "skipped": True,
            "sha256": common.sha256(stream),
            **facts,
            "illegal": legality(facts),
            "note": "already recorded; the recorder never rewrites a stream in place",
        }
    if stream.exists() and reason is not None:
        if reason not in RERUN_REASONS:
            raise RuntimeError(
                f"seed {seed}: re-run reason {reason!r} is not one of {RERUN_REASONS}; stop and "
                "report instead of inventing a reason"
            )
        quarantine = common.HERE / "stream_reruns" / f"g{goals}" / f"s{seed}_{reason}"
        quarantine.mkdir(parents=True, exist_ok=True)
        stream.replace(quarantine / stream.name)

    logs = common.HERE / "record_logs" / f"g{goals}"
    logs.mkdir(parents=True, exist_ok=True)
    configuration = RunConfiguration(
        seed=seed,
        output=str(logs / f"rec_g{goals}_s{seed}.jsonl"),
        decision_trace=str(logs / f"rec_tr_g{goals}_s{seed}.jsonl"),
        record_stream=str(stream),
        **common.recording_configuration(goals),
    )
    started = time.time()
    summary = build_joint_run(configuration).run()
    if not stream.exists():
        raise AssertionError(f"g{goals} s{seed}: the recorder produced no stream")
    payload = json.loads(stream.read_text(encoding="utf-8"))
    facts = scenario_facts(goals, seed, payload)
    return {
        "seed": seed,
        "skipped": False,
        "rerun_reason": reason,
        "sha256": common.sha256(stream),
        **facts,
        "illegal": legality(facts),
        "run_steps": int(summary.get("steps", 0)),
        "targets_discovered_by_registry": int(summary.get("targets_discovered", 0)),
        "seconds": round(time.time() - started, 1),
    }


def fold(rows: list[dict[str, Any]], goals: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    total = 0
    for row in sorted(rows, key=lambda item: int(item["seed"])):
        payload = common.stream_path(goals, int(row["seed"])).read_bytes()
        total += len(payload)
        digest.update(str(int(row["seed"])).encode("utf-8"))
        digest.update(payload)
    return {"streams": len(rows), "bytes": total, "sha256_fold": digest.hexdigest()}


def registration(goals: int, rows: list[dict[str, Any]], shard: int | None) -> dict[str, Any]:
    audit = require_clean_audit()
    payload = {
        "goals": int(goals),
        "block": [common.LAYERS[goals][0], common.LAYERS[goals][-1]],
        "size": len(common.LAYERS[goals]),
        "purpose": (
            "exploratory calibration data for the task-pressure learnability gate; never an "
            "independent test set for a learned model"
        ),
        "seed_audit": {
            "file": "SEED_COLLISION_AUDIT.json",
            "clean": audit["clean"],
            "sha256": common.sha256(common.json_path("SEED_COLLISION_AUDIT.json")),
        },
        "recording_configuration": common.recording_configuration(goals),
        "only_variable_check": common.assert_only_num_goals_changed(goals),
        "executed_code": code_identity(),
        "rerun_policy": {
            "allowed_reasons": list(RERUN_REASONS),
            "rule": "same seed only, for a pre-registered reason; no replacement sample",
        },
        "streams": rows,
        "fold": fold(rows, goals),
        "illegal_streams": {
            str(int(row["seed"])): row["illegal"] for row in rows if row.get("illegal")
        },
    }
    if shard is not None:
        payload["shard"] = shard
    return payload


def merge(goals: int) -> dict[str, Any]:
    shards = sorted(common.HERE.glob(f"STREAM_MANIFEST_g{goals}_shard*.json"))
    if not shards:
        raise SystemExit(f"no shard registrations for layer {goals}; record it first")
    rows: list[dict[str, Any]] = []
    for path in shards:
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["streams"])
    by_seed: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row["seed"])
        if seed in by_seed:
            raise SystemExit(f"g{goals} s{seed} registered twice; the layer was recorded twice")
        by_seed[seed] = row
    expected = set(common.LAYERS[goals])
    if set(by_seed) != expected:
        raise SystemExit(
            f"g{goals}: missing={sorted(expected - set(by_seed))} extra={sorted(set(by_seed) - expected)}"
        )
    for seed in sorted(by_seed):
        observed = common.sha256(common.stream_path(goals, seed))
        if observed != by_seed[seed]["sha256"]:
            raise SystemExit(
                f"g{goals} s{seed}: bytes moved after registration ({observed[:10]} vs "
                f"{str(by_seed[seed]['sha256'])[:10]})"
            )
        recomputed = scenario_facts(goals, seed, json.loads(common.stream_path(goals, seed).read_text(encoding="utf-8")))
        for key in ("placed", "detected", "map_sha256_reproduced", "steps_recorded"):
            if recomputed[key] != by_seed[seed][key]:
                raise SystemExit(
                    f"g{goals} s{seed}: {key} reads {recomputed[key]} on re-check, "
                    f"{by_seed[seed][key]} at registration"
                )
    ordered = [by_seed[seed] for seed in sorted(by_seed)]
    for row in ordered:
        row.pop("skipped", None)
        row.pop("note", None)
        row.pop("seconds", None)
    payload = registration(goals, ordered, shard=None)
    payload["shards"] = [path.name for path in shards]
    common.write_json(common.json_path(f"STREAM_MANIFEST_g{goals}.json"), payload)
    return payload


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", type=int, required=True, choices=sorted(common.LAYERS))
    parser.add_argument("--mode", choices=("record", "merge"), default="record")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--rerun-seed", type=int, default=None)
    parser.add_argument("--rerun-reason", default=None)
    args = parser.parse_args()

    if args.mode == "merge":
        payload = merge(args.goals)
        print(
            json.dumps(
                {
                    "goals": args.goals,
                    "streams": len(payload["streams"]),
                    "fold": payload["fold"],
                    "illegal_streams": payload["illegal_streams"],
                    "mean_detected_ratio": round(
                        sum(row["detected"] / row["placed"] for row in payload["streams"])
                        / len(payload["streams"]),
                        5,
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0 if not payload["illegal_streams"] else 1

    require_clean_audit()
    common.layer_streams(args.goals).mkdir(parents=True, exist_ok=True)
    if (args.rerun_seed is None) != (args.rerun_reason is None):
        parser.error("--rerun-seed and --rerun-reason go together")

    seeds = list(common.LAYERS[args.goals])
    if args.rerun_seed is not None:
        seeds = [args.rerun_seed]
    else:
        seeds = common.shard_of(seeds, args.shards, args.shard)

    ledger = common.LaunchLedger(f"LAUNCH_LEDGER_record_g{args.goals}_p{os.getpid()}.json")
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        # A reservation is a promise that an episode is about to run. Re-registering a stream that
        # is already on disk runs nothing, so it spends nothing -- but it is still recorded, so a
        # reader can tell "re-checked" from "never looked at".
        if not common.stream_path(args.goals, seed).exists() or args.rerun_seed == seed:
            ledger.reserve("record", args.goals, seed, arm="record")
        try:
            row = record(
                args.goals,
                seed,
                reason=args.rerun_reason if args.rerun_seed == seed else None,
            )
        except Exception as error:  # noqa: BLE001 - a failed recording still spends the launch
            ledger.record("record", args.goals, seed, False, f"{type(error).__name__}: {error}"[:300])
            raise
        ledger.record(
            "record",
            args.goals,
            seed,
            not row["illegal"],
            ("re-registered without an episode: " if row.get("skipped") else "")
            + f"placed={row['placed']} detected={row['detected']} illegal={row['illegal']}",
        )
        rows.append(row)
        print(
            f"[record g{args.goals} s{seed}] placed={row['placed']} detected={row['detected']} "
            f"sha={str(row['sha256'])[:10]}"
            + (" (already there)" if row.get("skipped") else "")
            + (f" ILLEGAL={row['illegal']}" if row["illegal"] else ""),
            flush=True,
        )

    payload = registration(args.goals, rows, shard=args.shard if args.shards > 1 else None)
    name = (
        f"STREAM_MANIFEST_g{args.goals}_shard{args.shard}.json"
        if args.shards > 1
        else f"STREAM_MANIFEST_g{args.goals}_shard0.json"
    )
    common.write_json(common.HERE / name, payload)
    print(
        json.dumps(
            {
                "goals": args.goals,
                "recorded": sum(1 for row in rows if not row.get("skipped")),
                "skipped": sum(1 for row in rows if row.get("skipped")),
                "illegal": payload["illegal_streams"],
                "fold": payload["fold"],
                "output": name,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if not payload["illegal_streams"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
