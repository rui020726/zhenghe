#!/usr/bin/env python
"""ROUND68 independent compact projection (zero new simulation / forward / optimizer).

Builds a scene-sorted, machine-readable projection of every QUALIFIED cell actually used by the
ROUND68 statistics, straight from the original row JSON. Two deliberate properties:

  * admissibility is still enforced: the only rows read are those NAMED BY a complete claim whose
    result_receipt re-proves, resolved through r68_eval._qualified_row_paths;
  * the NUMBERS are not produced by the audited statistics functions. on_time / spent_bytes are
    copied verbatim from the row; the on-time target-ID set is rebuilt here from the row's own
    task_reward_events under the same evidentiary rule; the bootstrap index is REGENERATED from
    its documented recipe and its SHA is asserted against the one recorded in the frozen
    statistics. Means and quantiles are computed here with numpy.

Afterwards the projection is cross-checked against ROUND68_STATISTICS.json and any disagreement
is reported as a field, never silently reconciled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r68_eval as ev      # noqa: E402  only for claim-admissibility resolution + paths
import r68_execute as base  # noqa: E402

TARGET_SLOTS = tuple(f"piloc-e1-target-{i}" for i in range(6))
SCENES = tuple(range(1_800_000, 1_800_032))
FOCAL_SCENE = 1_700_027
SEEDS = (11, 22, 33)
MODES = ("greedy", "sampled")
MODEL_ARMS = ("INIT_ALPHA", "TRAINED_ALPHA")
RULE_ARMS = ("IMMEDIATE", "LATE_DISCOVERY_GATE", "ALWAYS_HOLD")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def on_time_ids_from_events(row: dict, scene: int) -> list:
    """Independent rebuild of the evidenced on-time target set from the row's own events."""
    on_time = row.get("on_time")
    if not isinstance(on_time, int) or isinstance(on_time, bool):
        raise RuntimeError(f"scene {scene}: no integer on_time")
    events = row.get("task_reward_events")
    if events is None:
        events = []
    if not isinstance(events, list):
        raise RuntimeError(f"scene {scene}: task_reward_events is {type(events).__name__}")
    if not events:
        if int(on_time) != 0:
            raise RuntimeError(f"scene {scene}: on_time={on_time} but no events")
        return []
    ids, n_ev = set(), 0
    for e in events:
        if not isinstance(e, dict):
            raise RuntimeError(f"scene {scene}: malformed event")
        if e.get("on_time") is not True:
            continue
        n_ev += 1
        tid = e.get("task_id")
        if not tid:
            raise RuntimeError(f"scene {scene}: on-time event without task_id")
        ids.add(str(tid))
    if len(ids) != int(on_time) or n_ev != int(on_time):
        raise RuntimeError(f"scene {scene}: {len(ids)} ids / {n_ev} events but on_time={on_time}")
    return sorted(ids)


def mask_of(ids: list) -> dict:
    """6-bit mask over piloc-e1-target-0..5. Asserts no duplicate ids and popcount == on_time."""
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"duplicate target ids: {ids}")
    unknown = [t for t in ids if t not in TARGET_SLOTS]
    if unknown:
        raise RuntimeError(f"target ids outside the frozen 6 slots: {unknown}")
    m = 0
    for t in ids:
        m |= 1 << TARGET_SLOTS.index(t)
    return {"mask": m, "mask_bin": format(m, "06b"), "popcount": bin(m).count("1"),
            "ids": list(ids)}


def cell_from_row(row: dict, scene: int, row_path: str, logical: str) -> dict:
    on_time = int(row["on_time"])
    ids = on_time_ids_from_events(row, scene)
    mk = mask_of(ids)
    if mk["popcount"] != on_time:
        raise RuntimeError(f"scene {scene}: popcount {mk['popcount']} != on_time {on_time}")
    return {"scene": scene, "on_time": on_time, "spent_bytes": int(row["spent_bytes"]),
            "mask": mk["mask"], "mask_bin": mk["mask_bin"], "ids": mk["ids"],
            "row_path": row_path, "row_sha256": _sha256_file(Path(row_path)),
            "logical_job": logical}


def bootstrap_index():
    gen = np.random.Generator(np.random.PCG64(20261012))
    idx = gen.integers(0, 32, size=(10000, 32))
    return idx, hashlib.sha256(np.ascontiguousarray(idx).tobytes()).hexdigest()


def paired(a_cells, b_cells, scenes, idx, key):
    """Independent paired delta: mean, bootstrap CI, lost/gained target sets."""
    dt = np.array([a_cells[s][key] - b_cells[s][key] for s in scenes], dtype=np.float64)
    means = dt[idx].mean(axis=1)
    lost, gained = [], []
    for s in scenes:
        for t in sorted(set(b_cells[s]["ids"]) - set(a_cells[s]["ids"])):
            lost.append({"scene": s, "target_id": t})
        for t in sorted(set(a_cells[s]["ids"]) - set(b_cells[s]["ids"])):
            gained.append({"scene": s, "target_id": t})
    return {"mean": float(dt.mean()),
            "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "lost": lost, "lost_count": len(lost),
            "gained": gained, "gained_count": len(gained)}


def build(raw_root: Path) -> dict:
    raw_root = Path(raw_root).resolve()
    idx, idx_sha = bootstrap_index()

    q68 = ev._qualified_row_paths(raw_root, "r68eval")
    q66 = ev._qualified_row_paths(ev.base.R66_DIR / "raw", "r66eval2")

    model: dict[tuple, dict] = {}
    focal: dict[tuple, dict] = {}
    rule: dict[tuple, dict] = {}
    for logical, paths in sorted(q68.items()):
        if len(paths) != 1:
            raise RuntimeError(f"{logical}: {len(paths)} receipts")
        row = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
        arm = str(row["model_arm"]).upper()
        scene = int(row["scenario_seed"])
        if logical.startswith("r68eval_model_"):
            key = (arm, int(row["training_seed"]), str(row["eval_mode"]), scene)
            if key in model:
                raise RuntimeError(f"duplicate model cell {key}")
            model[key] = cell_from_row(row, scene, paths[0], logical)
        elif logical.startswith("r68eval_focal_"):
            key = (arm, int(row["training_seed"]))
            focal[key] = cell_from_row(row, scene, paths[0], logical)
    for logical, paths in sorted(q66.items()):
        if not logical.startswith("r66eval2_rule_"):
            continue
        if len(paths) != 1:
            raise RuntimeError(f"{logical}: {len(paths)} receipts")
        row = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
        arm = str(row["model_arm"]).upper()
        scene = int(row["scenario_seed"])
        if arm not in RULE_ARMS or scene not in SCENES:
            continue
        rule[(arm, scene)] = cell_from_row(row, scene, paths[0], logical)

    scenes = list(SCENES)
    # completeness
    missing_model = [(a, s, m, sc) for a in MODEL_ARMS for s in SEEDS for m in MODES
                     for sc in scenes if (a, s, m, sc) not in model]
    if missing_model:
        raise RuntimeError(f"missing {len(missing_model)} model cells, e.g. {missing_model[:3]}")
    missing_rule = [(a, sc) for a in RULE_ARMS for sc in scenes if (a, sc) not in rule]
    if missing_rule:
        raise RuntimeError(f"missing {len(missing_rule)} rule cells, e.g. {missing_rule[:3]}")

    primary = {}
    for seed in SEEDS:
        for mode in MODES:
            A = {s: model[("TRAINED_ALPHA", seed, mode, s)] for s in scenes}
            B = {s: model[("INIT_ALPHA", seed, mode, s)] for s in scenes}
            primary[f"{seed}/{mode}"] = {
                "task_delta": paired(A, B, scenes, idx, "on_time"),
                "cost_delta_B": paired(A, B, scenes, idx, "spent_bytes"),
                "trained_on_time_total": sum(A[s]["on_time"] for s in scenes),
                "init_on_time_total": sum(B[s]["on_time"] for s in scenes),
                "trained_bytes_per_scene": sum(A[s]["spent_bytes"] for s in scenes) / len(scenes),
                "init_bytes_per_scene": sum(B[s]["spent_bytes"] for s in scenes) / len(scenes),
            }

    rule_abs, rule_cmp = {}, {}
    for arm in RULE_ARMS:
        cells = {s: rule[(arm, s)] for s in scenes}
        rule_abs[arm] = {
            "on_time_total": sum(cells[s]["on_time"] for s in scenes),
            "spent_bytes_total": sum(cells[s]["spent_bytes"] for s in scenes),
            "spent_bytes_per_scene": sum(cells[s]["spent_bytes"] for s in scenes) / len(scenes),
            "union_ids": sorted({t for s in scenes for t in cells[s]["ids"]}),
            "union_count": len({t for s in scenes for t in cells[s]["ids"]}),
        }
        for seed in SEEDS:
            for mode in MODES:
                for marm in MODEL_ARMS:
                    A = {s: model[(marm, seed, mode, s)] for s in scenes}
                    rule_cmp[f"{marm}_s{seed}_{mode}_minus_{arm}"] = {
                        "task_delta": paired(A, cells, scenes, idx, "on_time"),
                        "cost_delta_B": paired(A, cells, scenes, idx, "spent_bytes"),
                    }

    guard = json.loads((raw_root / "runtime" / "ROUND68_EVAL_QUOTA_GUARD.json")
                       .read_text(encoding="utf-8"))
    ppo = sum(1 for _ in open(raw_root / "runtime" / "ppo_update_calls.jsonl", encoding="utf-8"))
    opt = sum(1 for _ in open(raw_root / "runtime" / "optimizer_step_calls.jsonl", encoding="utf-8"))

    out = {
        "kind": "round68-independent-projection/1",
        "task_id": base.TASK_ID,
        "branch": "exp/round68-late-initial-confidence-ablation",
        "purpose": "scene-sorted projection of the qualified cells actually used by the statistics",
        "block": {"scenes": [SCENES[0], SCENES[-1]], "status": "already-used development block"},
        "target_slots": list(TARGET_SLOTS),
        "bootstrap": {"recipe": "PCG64(20261012).integers(0,32,size=(10000,32))",
                      "index_sha256": idx_sha, "quantile": "numpy linear 0.025/0.975"},
        "source_audit": {
            "r68_raw_root": str(raw_root),
            "r68_qualified_logical_jobs": len(q68),
            "r66_raw_root": str(ev.base.R66_DIR / "raw"),
            "r66_qualified_logical_jobs": len(q66),
            "admission": "every row read is named by a complete claim whose receipt re-proves",
        },
        "model_cells": {f"{a}|{s}|{m}|{sc}": model[(a, s, m, sc)]
                        for a in MODEL_ARMS for s in SEEDS for m in MODES for sc in scenes},
        "rule_cells": {f"{a}|{sc}": rule[(a, sc)] for a in RULE_ARMS for sc in scenes},
        "focal_cells": {f"{a}|{s}": focal[(a, s)] for a in MODEL_ARMS for s in SEEDS},
        "primary_trained_minus_init": primary,
        "rule_absolute": rule_abs,
        "arm_minus_rule": rule_cmp,
        "budget": {
            "booked_normal": guard.get("booked_normal"),
            "booked_reserve": guard.get("booked_reserve"),
            "actual_environment_constructions": guard.get("actual_environment_constructions"),
            "hard_cap": guard.get("hard_cap"),
            "hard_headroom": guard.get("hard_headroom"),
            "historical_non_plan_probe": guard.get("historical_non_plan_probe"),
            "unregistered_extra_environment_constructions":
                guard.get("unregistered_extra_environment_constructions"),
        },
        "training_journals": {"ppo_update_calls": ppo, "optimizer_step_calls": opt},
        "environment_constructions": 0, "model_forwards": 0, "ppo_calls": 0,
        "optimizer_steps": 0, "checkpoint_writes": 0,
    }

    # cross-check against the frozen statistics (reported, never used to build the numbers)
    try:
        st = json.loads((raw_root / "ROUND68_STATISTICS.json").read_text(encoding="utf-8"))
        x = {}
        for key, mine in primary.items():
            seed, mode = key.split("/")
            seed = int(seed)
            theirs = st["per_seed_mode"][key]["primary"]
            x[key] = {
                "task_mean_match": abs(mine["task_delta"]["mean"]
                                       - theirs["task_delta_TRAINED_ALPHA_minus_INIT_ALPHA_mean"]) < 1e-9,
                "task_ci_match": (mine["task_delta"]["ci95"] == theirs[
                    "task_ci95_TRAINED_ALPHA_minus_INIT_ALPHA"]),
                "cost_mean_match": abs(mine["cost_delta_B"]["mean"] - theirs[
                    "cost_delta_TRAINED_ALPHA_minus_INIT_ALPHA_B_per_scene"]) < 1e-6,
                "cost_ci_match": (mine["cost_delta_B"]["ci95"] == theirs[
                    "cost_ci95_TRAINED_ALPHA_minus_INIT_ALPHA_B_per_scene"]),
                "lost_count_match": (mine["task_delta"]["lost_count"] == theirs[
                    "targets_lost_TRAINED_ALPHA_vs_INIT_ALPHA_count"]),
                "gained_count_match": (mine["task_delta"]["gained_count"] == theirs[
                    "targets_gained_TRAINED_ALPHA_vs_INIT_ALPHA_count"]),
            }
        out["cross_check_vs_frozen_statistics"] = x
        out["cross_check_all_match"] = all(all(v.values()) for v in x.values())
        out["bootstrap_index_sha_match"] = (idx_sha == st["bootstrap"]["index_sha256"])
    except Exception as exc:
        out["cross_check_vs_frozen_statistics"] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND68 independent compact projection")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    res = build(Path(args.raw_root).resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    tmp.replace(args.out)
    slim = {k: res[k] for k in ("kind", "bootstrap", "budget", "training_journals",
                                "cross_check_vs_frozen_statistics",
                                "cross_check_all_match", "bootstrap_index_sha_match",
                                "rule_absolute", "primary_trained_minus_init")}
    print(json.dumps(slim, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())