"""ROUND66 corrected evaluation adapter (private, append-only).

WHY THIS FILE EXISTS
--------------------
ROUND66_FREEZE.json at commit 7267fc7 recorded an evaluation matrix (model_eval = NEW 192 + BC 192,
rule_eval = 288 with extra rule arms) that does not match the decision-side plan. The decision-side
C block is:

    NEW / R60ep128 / BC  x 3 seeds x 2 modes x 32 scenes  = 576
    immediate / late147 / always_hold  x 32 scenes        =  96
    focal diagnostics (unchanged block D)                 =   6

This module is the corrected evaluation entry. It does NOT re-run or re-plan training, it shares the
same R66 resource ledger, raw root, claim/slot roots and 1110/40/1150 quota as r66_execute.py, and
it carries its own evaluation identity so the corrected matrix is auditable separately.

Arms:
    NEW = R66 ep128 (frozen early path intact, read-only load)
    OLD = compliant R60 ep128, read-only, NOT frozen (runs exactly as R60 had it)
    BC  = compliant R60 ep0 initial value, read-only, NOT frozen

No training, no optimizer step, no checkpoint write, no RNG restore happens here.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r66_frozen as r66
import r66_execute as base

SEEDS = (11, 22, 33)
MODES = ("greedy", "sampled")
SCENES_NEW = tuple(range(1_800_000, 1_800_032))
FOCAL_SCENE = 1_700_027
MODEL_ARMS = ("NEW", "OLD", "BC")
RULE_ARMS = ("immediate", "late_discovery_gate", "always_hold")
EVAL_VERSION = r66.R66_TRAINING_VERSION + "+r66-eval-matrix-correction/1"
NORMAL_CAP, RESERVE_CAP, HARD_CAP = base.NORMAL_CAP, base.RESERVE_CAP, base.HARD_CAP


def build_eval_plan() -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    slot = 432  # 48 block-A references + 384 training episodes are already numbered
    for arm in MODEL_ARMS:
        for seed in SEEDS:
            for mode in MODES:
                for scene in SCENES_NEW:
                    slot += 1
                    jobs.append({"scope": "model_eval", "arm": arm, "seed": seed, "mode": mode,
                                 "scene": int(scene), "logical_job": f"r66eval_model_{arm.lower()}_s{seed}_{mode}_scene{scene}",
                                 "normal_slot": slot})
    for arm in RULE_ARMS:
        for scene in SCENES_NEW:
            slot += 1
            jobs.append({"scope": "rule_eval", "arm": arm.upper(), "mode": "greedy",
                         "scene": int(scene), "logical_job": f"r66eval_rule_{arm}_greedy_scene{scene}",
                         "normal_slot": slot})
    for seed in SEEDS:
        for arm in ("NEW", "BC"):
            slot += 1
            jobs.append({"scope": "focal", "arm": arm, "seed": seed, "mode": "greedy",
                         "scene": FOCAL_SCENE, "logical_job": f"r66eval_focal_{arm.lower()}_s{seed}_scene{FOCAL_SCENE}",
                         "normal_slot": slot})
    counts = {}
    for j in jobs:
        counts[j["scope"]] = counts.get(j["scope"], 0) + 1
    if counts != {"model_eval": 576, "rule_eval": 96, "focal": 6} or len(jobs) != 678:
        raise AssertionError(f"corrected eval plan is not 576+96+6=678 unique jobs: {counts}")
    if len({j["logical_job"] for j in jobs}) != 678:
        raise AssertionError("corrected eval plan contains duplicate logical jobs")
    return {"kind": "round66-corrected-eval-plan/1", "task_id": base.TASK_ID,
            "branch": base.BRANCH, "eval_version": EVAL_VERSION,
            "supersedes": ("the model_eval/rule_eval scopes of raw/ROUND66_FIXED_PLAN.json written "
                           "from the erroneous freeze at 7267fc7; that file is preserved unchanged"),
            "model_eval_count": 576, "rule_eval_count": 96, "focal_count": 6, "eval_total": 678,
            "model_arms": list(MODEL_ARMS), "rule_arms": list(RULE_ARMS),
            "eval_scenes": list(SCENES_NEW), "focal_scene": FOCAL_SCENE,
            "normal_slot_range": [jobs[0]["normal_slot"], jobs[-1]["normal_slot"]],
            "quota": {"normal_cap": NORMAL_CAP, "reserve_cap": RESERVE_CAP, "hard_cap": HARD_CAP,
                      "note": ("shared with the R66 training ledger; reserve use is capped at 39 so "
                               "that actual constructions (slots + the D1 reaped partial) stay <= 1150")},
            "jobs": jobs}


def _load_arm(cfg, seed: int, arm: str):
    """Read-only arm controllers. NEW keeps its frozen mask; OLD and BC run exactly as R60 had them."""
    import r42_one_shot  # noqa: F401  (ensures the frozen collector module is importable)
    common = cfg["common"]
    if arm == "NEW":
        ep128 = Path(common.CHECKPOINTS) / f"v2_train_s{seed}_ep128.pt"
        if not ep128.is_file():
            raise FileNotFoundError(f"seed {seed}: R66 ep128 checkpoint missing")
        controller, state, identity = r66.load_checkpoint_read_only(ep128, freeze=base.R66_FREEZE)
        if int(state.get("episodes_done", -1)) != base.EPISODES_PER_SEED:
            raise RuntimeError(f"seed {seed}: R66 ep128 is not {base.EPISODES_PER_SEED} episodes")
        if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
            raise RuntimeError(f"seed {seed}: NEW read-only eval restored state")
        if any(p.requires_grad for p in r66.frozen_parameters(controller)):
            raise RuntimeError(f"seed {seed}: NEW ep128 lost its frozen mask")
        return controller, identity
    role = "ep128" if arm == "OLD" else "ep0"
    return r66.load_r60_checkpoint_read_only(seed, role)


def eval_model_arms(cfg, ledger, plan) -> dict:
    common = cfg["common"]; runner = cfg["runner"]
    import r42_one_shot
    for seed in SEEDS:
        arms = {}
        for arm in MODEL_ARMS:
            controller, identity = _load_arm(cfg, seed, arm)
            arms[arm] = (controller, identity)
        for arm in MODEL_ARMS:
            controller, identity = arms[arm]
            for mode in MODES:
                for scene in SCENES_NEW:
                    logical = f"r66eval_model_{arm.lower()}_s{seed}_{mode}_scene{scene}"
                    if ledger.is_complete(logical):
                        continue
                    collector = r42_one_shot.OneShotSendSkipCollector(controller, greedy=(mode == "greedy"))
                    tag = f"r66eval_{arm.lower()}_s{seed}_{mode}"
                    metadata = {"arm": arm, "training_seed": int(seed), "mode": mode,
                                "scenario_seed": int(scene), "eval_version": EVAL_VERSION}
                    def _do(controller=controller, collector=collector, tag=tag, scene=scene,
                            mode=mode, arm=arm, identity=identity, seed=seed):
                        row = runner._run_episode(tag, "eval", int(scene), policy="learned",
                                                  attach=collector.policy(), controller=controller,
                                                  collector=collector,
                                                  action_rng_seed=runner.eval_action_seed(int(seed), int(scene), mode),
                                                  tag=tag, checkpoint_load_identity=identity)
                        row["model_arm"] = arm; row["eval_mode"] = mode; row["training_seed"] = int(seed)
                        return row
                    row = base._atomic_run(cfg, logical, metadata, _do,
                                           detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}")
                    base._write_row(cfg, row)
            ledger.write_ledger()
        print(f"R66 corrected eval seed {seed}: arms {list(MODEL_ARMS)} done", flush=True)
    return {"status": "model_arms_eval_complete", "arms": list(MODEL_ARMS)}


def eval_rules(cfg, ledger, plan) -> dict:
    import v2_rules
    runner = cfg["runner"]
    for arm in RULE_ARMS:
        for scene in SCENES_NEW:
            logical = f"r66eval_rule_{arm}_greedy_scene{scene}"
            if ledger.is_complete(logical):
                continue
            tag = f"r66eval_rule_{arm}"
            metadata = {"arm": arm.upper(), "mode": "greedy", "scenario_seed": int(scene),
                        "eval_version": EVAL_VERSION}
            def _do(arm=arm, scene=scene, tag=tag):
                if arm == "immediate":
                    row = runner._run_episode(tag, "rule", int(scene), policy="immediate", tag=tag)
                else:
                    policy = v2_rules.build_rule(arm, int(scene))
                    row = runner._run_episode(tag, "rule", int(scene), policy="learned",
                                              attach=policy, tag=tag)
                    row["rule_identity"] = v2_rules.rule_identity(arm)
                row["model_arm"] = arm.upper(); row["eval_mode"] = "greedy"
                row["scenario_seed"] = int(scene)
                return row
            row = base._atomic_run(cfg, logical, metadata, _do,
                                   detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}")
            base._write_row(cfg, row)
        ledger.write_ledger()
    return {"status": "rules_eval_complete", "arms": list(RULE_ARMS)}


def eval_focal(cfg, ledger, plan) -> dict:
    common = cfg["common"]; runner = cfg["runner"]
    import r42_one_shot
    for seed in SEEDS:
        for arm in ("NEW", "BC"):
            controller, identity = _load_arm(cfg, seed, arm)
            logical = f"r66eval_focal_{arm.lower()}_s{seed}_scene{FOCAL_SCENE}"
            if ledger.is_complete(logical):
                continue
            collector = r42_one_shot.OneShotSendSkipCollector(controller, greedy=True)
            tag = f"r66eval_focal_{arm.lower()}_s{seed}"
            metadata = {"arm": arm, "training_seed": int(seed), "mode": "greedy",
                        "scenario_seed": FOCAL_SCENE, "block": "D", "eval_version": EVAL_VERSION}
            def _do(controller=controller, collector=collector, tag=tag, arm=arm, identity=identity, seed=seed):
                row = runner._run_episode(tag, "eval", FOCAL_SCENE, policy="learned",
                                          attach=collector.policy(), controller=controller,
                                          collector=collector,
                                          action_rng_seed=runner.eval_action_seed(int(seed), FOCAL_SCENE, "greedy"),
                                          tag=tag, checkpoint_load_identity=identity)
                row["model_arm"] = arm; row["eval_mode"] = "greedy"; row["training_seed"] = int(seed)
                row["focal"] = True
                return row
            row = base._atomic_run(cfg, logical, metadata, _do,
                                   detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}")
            base._write_row(cfg, row)
        ledger.write_ledger()
    return {"status": "focal_complete"}


# ------------------------------------------------------------------ statistics ---------------
def _bootstrap_index():
    import hashlib
    gen = np.random.Generator(np.random.PCG64(20261012))
    idx = gen.integers(0, 32, size=(10000, 32))
    return idx, hashlib.sha256(np.ascontiguousarray(idx).tobytes()).hexdigest()


def _load_model_rows(root: Path) -> dict:
    out = {}
    for p in Path(root).glob("v2_r66eval_model_*.json"):
        row = base._read_json(p)
        key = (int(row["training_seed"]), row["eval_mode"], int(row["scenario_seed"]))
        out[(row["model_arm"],) + key] = {
            "on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
            "on_time_ids": sorted(row.get("on_time_target_ids") or row.get("on_time_ids") or []),
            "row_path": str(p)}
    return out


def _load_rule_rows(root: Path) -> dict:
    out = {}
    for p in Path(root).glob("v2_r66eval_rule_*.json"):
        row = base._read_json(p)
        out[((row.get("model_arm") or "").lower(), row.get("eval_mode"), int(row["scenario_seed"]))] = {
            "on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]), "row_path": str(p)}
    return out


def compute_stats(raw_root: Path) -> dict:
    idx, idx_sha = _bootstrap_index()
    rows = _load_model_rows(Path(raw_root) / "rows")
    rules = _load_rule_rows(Path(raw_root) / "rows")
    scenes = list(SCENES_NEW)
    per: dict[str, Any] = {}
    for seed in SEEDS:
        for mode in MODES:
            def cell(arm, scene):
                return rows.get((arm, seed, mode, scene))
            missing = [(a, s) for a in MODEL_ARMS for s in scenes if cell(a, s) is None]
            if missing:
                raise RuntimeError(f"missing paired cells seed={seed} mode={mode}: {missing[:4]} ({len(missing)})")
            entry: dict[str, Any] = {}
            for arm in MODEL_ARMS:
                entry[f"{arm}_on_time_total"] = int(sum(cell(arm, s)["on_time"] for s in scenes))
                entry[f"{arm}_bytes_per_scene"] = float(np.mean([cell(arm, s)["spent_bytes"] for s in scenes]))
            for other in ("OLD", "BC"):
                d_task = np.array([cell("NEW", s)["on_time"] - cell(other, s)["on_time"] for s in scenes], dtype=np.float64)
                d_cost = np.array([cell("NEW", s)["spent_bytes"] - cell(other, s)["spent_bytes"] for s in scenes], dtype=np.float64)
                entry[f"task_delta_NEW_minus_{other}_mean"] = float(d_task.mean())
                entry[f"task_ci95_NEW_minus_{other}"] = [float(np.quantile(d_task[idx].mean(axis=1), 0.025)),
                                                          float(np.quantile(d_task[idx].mean(axis=1), 0.975))]
                entry[f"cost_delta_NEW_minus_{other}_B_per_scene"] = float(d_cost.mean())
                entry[f"cost_ci95_NEW_minus_{other}_B_per_scene"] = [float(np.quantile(d_cost[idx].mean(axis=1), 0.025)),
                                                                      float(np.quantile(d_cost[idx].mean(axis=1), 0.975))]
                entry[f"scenes_NEW_better_task_than_{other}"] = int((d_task > 0).sum())
                entry[f"scenes_NEW_worse_task_than_{other}"] = int((d_task < 0).sum())
                entry[f"scenes_NEW_cheaper_than_{other}"] = int((d_cost < 0).sum())
                lost = sorted({t for s in scenes for t in cell(other, s)["on_time_ids"]} -
                              {t for s in scenes for t in cell("NEW", s)["on_time_ids"]})
                gained = sorted({t for s in scenes for t in cell("NEW", s)["on_time_ids"]} -
                                {t for s in scenes for t in cell(other, s)["on_time_ids"]})
                entry[f"targets_lost_NEW_vs_{other}"] = lost
                entry[f"targets_gained_NEW_vs_{other}"] = gained
            for arm in RULE_ARMS:
                have = [rules.get((arm, "greedy", s)) for s in scenes]
                if all(h is not None for h in have):
                    entry[f"{arm}_on_time_total"] = int(sum(h["on_time"] for h in have))
                    entry[f"{arm}_bytes_per_scene"] = float(np.mean([h["spent_bytes"] for h in have]))
                    entry[f"NEW_minus_{arm}_bytes_B_per_scene"] = float(
                        np.mean([cell("NEW", s)["spent_bytes"] for s in scenes]) - entry[f"{arm}_bytes_per_scene"])
            per[f"{seed}/{mode}"] = entry
    return {"kind": "round66-corrected-statistics/1", "eval_version": EVAL_VERSION,
            "primary_comparison": "NEW_minus_OLD", "secondary_comparison": "NEW_minus_BC",
            "greedy_is_primary": True, "sampled_reported_separately": True,
            "scene_unit": 32, "seeds_not_pooled": True,
            "bootstrap": {"rng": "PCG64", "seed": 20261012, "draws": 10000, "index_sha256": idx_sha,
                          "quantile": "numpy linear 0.025/0.975"},
            "per_seed_mode": per, "cells": len(rows),
            "note": ("NEW=R66 ep128 frozen-early-path, OLD=compliant R60 ep128, BC=compliant R60 ep0 "
                     "initial value; all read-only on the fresh 32-scene block. CI containing 0 is "
                     "not equivalence. Restricted policy family, single development block, three "
                     "seeds not pooled.")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND66 corrected evaluation adapter")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--phase", choices=("plan", "eval", "stats"), required=True)
    args = ap.parse_args(argv)
    raw_root = Path(args.raw_root).resolve()
    if args.phase == "plan":
        plan = build_eval_plan()
        base._write_json(raw_root / "ROUND66_EVAL_PLAN_CORRECTED.json", plan)
        print(json.dumps({k: v for k, v in plan.items() if k != "jobs"}, indent=1))
        return 0
    cfg = base.configure_and_inject(raw_root)
    ledger = base.R66Ledger(raw_root)
    plan = base._read_json(raw_root / "ROUND66_EVAL_PLAN_CORRECTED.json")
    if args.phase == "eval":
        print(json.dumps(eval_rules(cfg, ledger, plan), indent=1), flush=True)
        print(json.dumps(eval_model_arms(cfg, ledger, plan), indent=1), flush=True)
        print(json.dumps(eval_focal(cfg, ledger, plan), indent=1), flush=True)
        ledger.write_ledger()
        return 0
    stats = compute_stats(raw_root)
    base._write_json(raw_root / "ROUND66_CORRECTED_STATISTICS.json", stats)
    print(json.dumps(stats, indent=1))
    ledger.write_ledger()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
