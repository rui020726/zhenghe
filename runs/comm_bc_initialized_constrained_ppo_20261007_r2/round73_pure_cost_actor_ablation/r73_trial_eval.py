#!/usr/bin/env python
"""ROUND73 approved fixed trial: evaluation adapter (192 model cells + 3 focal cells).

A faithful adaptation of the verified ROUND71 evaluation seam with an independent ROUND73
identity and the ROUND73 comparison contract:

  * arm: NEW_EP128 (this run's trained ep128 under the pure-cost actor objective, read-only);
    the R68 fresh factory and the ROUND71 training checkpoints are never used as an R73 arm;
  * booking goes through the SAME claimed-but-started-safe helper and construction-permit
    regime as training (``r68_execute._r68_atomic_run``), so eval and train cannot drift apart;
    every complete claim carries a receipt that re-proves from disk;
  * PRIMARY comparison: NEW_EP128 minus the ROUND71 ep128 control, per (seed, mode) over the 32
    frozen scenes 1800000..1800031 (greedy primary, sampled separate, seeds not pooled), with the
    ROUND71 rows reused READ-ONLY;
  * two-column task/cost context against the IMMEDIATE, LATE_DISCOVERY_GATE and ALWAYS_HOLD rule
    arms from the frozen ROUND68 rule-context file (read-only);
  * focal scene 1700027 detail (the early key target piloc-e1-target-5) per seed, against the
    ROUND71 focal rows reused read-only;
  * first-report SEND/SKIP sets split early (ready_step <= 147) / late, from the eval tapes.

Budget: 192 model cells + 3 focal cells = 195 of the 579 approved normal launches; the remaining
384 belong to training.  Nothing here re-simulates a rule or a control arm.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN_ROOT = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
R68_DIR = RUN_ROOT / "round68_late_initial_confidence_ablation"
R71_DIR = RUN_ROOT / "round71_critic_initialization_calibration"
READINESS_DIR = REPO / "analysis" / "round71-critic-calibration-readiness"
for _d in (READINESS_DIR, str(R68_DIR), str(HERE)):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import r73_trial_binding as trial                                       # noqa: E402
import r73_checkpoint as ckpt                                           # noqa: E402
import r73_trial_execute as execute                                      # noqa: E402
import r71_runtime_binding as readiness                                  # noqa: E402
import r68_execute as r68x                                               # noqa: E402
import r68_eval as r68eval                                               # noqa: E402
import r68_frozen as r68                                                 # noqa: E402

SEEDS = execute.SEEDS
MODES = execute.MODES
SCENES_NEW = execute.SCENES_NEW
FOCAL_SCENE = execute.FOCAL_SCENE
MODEL_ARMS = trial.MODEL_ARMS                    # ("NEW_EP128",)
MODEL_EVAL_ARM = execute.MODEL_EVAL_ARM          # the 192-cell block evaluates this arm only
ROW_TAG = trial.EVAL_ROW_TAG                     # r73eval
R71_ROW_TAG = "r71eval"                          # read-only control reuse (ROUND71 ep128)
R68_ROW_TAG = r68eval.ROW_TAG                    # r68eval
MODEL_EVAL_COUNT = 192
FOCAL_COUNT = 3
EVAL_CELLS = MODEL_EVAL_COUNT + FOCAL_COUNT
EVAL_VERSION = ckpt.R73_TRAINING_VERSION + "+r73-eval/1"
RULE_CONTEXT_PATH = R68_DIR / "raw" / "ROUND68_RULE_CONTEXT.json"
RULE_ARMS = ("IMMEDIATE", "LATE_DISCOVERY_GATE", "ALWAYS_HOLD")
CONTROL_ARM = "NEW_EP128"                        # the ROUND71 control rows carry this arm label
CONTROL_LABEL = "R71_EP128"
READY_CUT = 147.0
FOCAL_TARGET = "piloc-e1-target-5"

_sha256_file = r68x._sha256_file
_read_json = r68x._read_json
_write_json = r68x._write_json
VISITED: list = []


# ------------------------------------------------------------------ plan / identity ------
def eval_job_order():
    """Every ROUND73 evaluation cell in the exact order the executor requests it."""
    for seed in SEEDS:
        for mode in MODES:
            for scene in SCENES_NEW:
                yield {"scope": "model_eval", "arm": MODEL_EVAL_ARM, "seed": int(seed),
                       "mode": mode, "scene": int(scene)}
    for seed in SEEDS:
        for arm in MODEL_ARMS:
            yield {"scope": "focal", "arm": arm, "seed": int(seed), "mode": "greedy",
                   "scene": FOCAL_SCENE}


def logical_job(cell: dict[str, Any]) -> str:
    if cell["scope"] == "model_eval":
        return (f"{ROW_TAG}_model_{cell['arm'].lower()}_s{cell['seed']}_{cell['mode']}"
                f"_scene{cell['scene']}")
    return f"{ROW_TAG}_focal_{cell['arm'].lower()}_s{cell['seed']}_scene{cell['scene']}"


def build_eval_plan() -> dict[str, Any]:
    cells = list(eval_job_order())
    jobs = []
    for index, cell in enumerate(cells):
        jobs.append({"scope": cell["scope"], "arm": cell["arm"], "seed": cell["seed"],
                     "mode": cell["mode"], "scene": cell["scene"],
                     "logical_job": logical_job(cell), "planned_normal_slot": 385 + index})
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j["scope"]] = counts.get(j["scope"], 0) + 1
    if counts != {"model_eval": MODEL_EVAL_COUNT, "focal": FOCAL_COUNT} or len(jobs) != EVAL_CELLS:
        raise AssertionError(f"ROUND73 eval plan is not {MODEL_EVAL_COUNT}+{FOCAL_COUNT}="
                             f"{EVAL_CELLS} unique jobs: {counts}")
    if len({j["logical_job"] for j in jobs}) != EVAL_CELLS:
        raise AssertionError("ROUND73 eval plan contains duplicate logical jobs")
    planned = [j["planned_normal_slot"] for j in jobs]
    if planned != list(range(385, 385 + EVAL_CELLS)):
        raise AssertionError("planned normal slots are not the contiguous range 385..579")
    if execute.TRAINING_COUNT + EVAL_CELLS != execute.NORMAL_CAP:
        raise AssertionError("384 training + 195 eval cells != the 579 normal cap")
    return {"kind": "round73-eval-plan/1", "task_id": trial.R73_TRIAL_TASK_ID,
            "branch": trial.R73_TRIAL_BRANCH, "run_id": trial.TRIAL_RUN_ID,
            "eval_version": EVAL_VERSION, "training_version": ckpt.R73_TRAINING_VERSION,
            "actor_objective": ckpt.R73_ACTOR_OBJECTIVE,
            "matrix": {"model_eval": MODEL_EVAL_COUNT, "focal": FOCAL_COUNT, "total": EVAL_CELLS},
            "model_arms": list(MODEL_ARMS), "model_eval_arm": MODEL_EVAL_ARM,
            "primary_comparison": "NEW_EP128_minus_R71_EP128",
            "primary_meaning": ("this run's pure-cost-objective-trained ep128 minus the ROUND71 "
                                "ep128 control (same INIT, same training data and order; only "
                                "the actor objective differs).  ROUND71 rows are reused "
                                "READ-ONLY and are not new constructions."),
            "context_comparisons": {
                "arms_vs_rules": "two-column task/cost against IMMEDIATE, LATE_DISCOVERY_GATE "
                                 "and ALWAYS_HOLD (read-only ROUND68 rule context)",
                "focal_vs_r71_focal": "scene 1700027 / piloc-e1-target-5 detail against the "
                                      "read-only ROUND71 focal rows"},
            "greedy_is_primary": True, "sampled_reported_separately": True,
            "scene_unit": 32, "seeds_not_pooled": True,
            "eval_block": "1800000..1800031 (already-used development block, NOT a confirmation set)",
            "eval_scenes": list(SCENES_NEW), "focal_scene": FOCAL_SCENE,
            "focal_target": FOCAL_TARGET,
            "execution_order": ["model_eval", "focal"],
            "row_namespace": {"tag_prefix": ROW_TAG,
                              "row_globs": [f"v2_{ROW_TAG}_model_*.json", f"v2_{ROW_TAG}_focal_*.json"],
                              "disjoint_from": [f"v2_{R71_ROW_TAG}_*.json", f"v2_{R68_ROW_TAG}_*.json"]},
            "normal_cap": execute.NORMAL_CAP, "reserve_cap": execute.RESERVE_CAP,
            "hard_cap": execute.HARD_CAP, "jobs": jobs}


# ------------------------------------------------------------------ configure / guard ----
def configure_r73_trial_eval(raw_root: Path) -> dict[str, Any]:
    """Bind the live ROUND73 trial runtime (quota 579/21/600 on the live modules) before any
    builder can run."""
    cfg = execute.configure_r73_trial(
        Path(raw_root), trial.trial_gate(), trial.trial_run_identity(),
        model_read_approval=trial.MODEL_READ_APPROVAL)
    cfg["r73_eval_quota_guard"] = quota_guard(cfg)
    return cfg


def quota_guard(cfg: dict[str, Any]) -> dict:
    """The real hard-cap judgement: booked slots <= the ROUND73 caps."""
    r40 = cfg["runtime"]["r40_common"]
    snapshot = r40.ledger_snapshot()          # itself raises if a live cap is already exceeded
    normal_used = int(snapshot["normal_used"])
    reserve_used = int(snapshot["reserve_used"])
    booked = normal_used + reserve_used
    hard = int(r40.HARD_CAP)
    if normal_used > execute.NORMAL_CAP:
        raise RuntimeError(f"ROUND73-EVAL: booked normal {normal_used} exceeds the normal cap "
                           f"{execute.NORMAL_CAP}")
    if reserve_used > execute.RESERVE_CAP:
        raise RuntimeError(f"ROUND73-EVAL: booked reserve {reserve_used} exceeds the ceiling "
                           f"{execute.RESERVE_CAP}")
    if booked > hard:
        raise RuntimeError(f"ROUND73-EVAL: booked constructions {booked} exceed the hard cap {hard}")
    return {"kind": "round73-eval-quota-guard/1", "eval_version": EVAL_VERSION,
            "booked_normal": normal_used, "booked_reserve": reserve_used, "booked_total": booked,
            "actual_environment_constructions": booked,
            "normal_cap": execute.NORMAL_CAP, "reserve_cap": execute.RESERVE_CAP,
            "hard_cap": hard, "hard_headroom": hard - booked,
            "live_bindings": {"r40_NORMAL_CAP": int(r40.NORMAL_CAP),
                              "r40_RESERVE_CAP": int(r40.RESERVE_CAP),
                              "r40_HARD_CAP": hard,
                              "common_NORMAL_LAUNCHES": int(cfg["common"].NORMAL_LAUNCHES),
                              "common_RESERVE_LAUNCHES": int(cfg["common"].RESERVE_LAUNCHES),
                              "common_LAUNCH_CAP": int(cfg["common"].LAUNCH_CAP)}}

class AlreadyComplete(RuntimeError):
    pass


ARTEFACT_KEYS = ("row", "lifecycle", "transcript", "tape")


def persist_result(cfg, row, run_tag: str) -> dict:
    """Write the result row and prove the durable artefacts exist before anything is committed."""
    r68x._write_row(cfg, row)
    artefacts = row.get("artefacts") or {}
    if not artefacts.get("row"):
        raise RuntimeError(f"{run_tag}: result row declares no row artefact")
    row_path = Path(artefacts["row"])
    if not row_path.is_file() or row_path.stat().st_size == 0:
        raise RuntimeError(f"{run_tag}: result row is missing or empty at {row_path}")
    first = _sha256_file(row_path)
    reread = _read_json(row_path)
    if _sha256_file(row_path) != first:
        raise RuntimeError(f"{run_tag}: the result row changed while it was being hashed")
    receipt = {"row_path": str(row_path), "row_sha256": first,
               "row_bytes": int(row_path.stat().st_size),
               "on_time": int(reread.get("on_time", -1)),
               "spent_bytes": int(reread.get("spent_bytes", -1)),
               "steps": int(reread.get("steps", -1)),
               "optimizer_steps": int(reread.get("optimizer_steps", -1)),
               "eval_version": EVAL_VERSION}
    for key in ARTEFACT_KEYS[1:]:
        path = artefacts.get(key)
        if path:
            p = Path(path)
            if not p.is_file():
                raise RuntimeError(f"{run_tag}: declared artefact {key} is missing at {p}")
    return receipt


def _safe_atomic_run(cfg, logical, metadata, run_fn, *, detail, tag, persist=None, cell=None):
    """The ONE booking path: claim/slot/permit first, run second, receipt third."""
    r40 = cfg["runtime"]["r40_common"]
    holder = {"attempt": None, "claim_path": None, "run_tag": tag}

    def _on_claim(claim_path, attempt):
        holder["claim_path"] = claim_path
        holder["attempt"] = int(attempt)
        r40._update_claim(claim_path, row_tag=holder["run_tag"], eval_version=EVAL_VERSION)

    def _wrapped():
        row = run_fn(holder["run_tag"])
        if persist is not None:
            receipt = persist(row, holder["run_tag"])
            r40._update_claim(holder["claim_path"], result_receipt=receipt)
        if isinstance(row, dict):
            row["claim_attempt"] = int(holder["attempt"])
            row["claim_path"] = str(holder["claim_path"])
        return row

    return r68x._r68_atomic_run(cfg, logical, metadata, _wrapped, detail=detail,
                                on_claim=_on_claim, allow_complete_retry=True)


def _cell_complete(cfg: dict[str, Any], logical: str, cell: dict | None = None) -> bool:
    r40 = cfg["runtime"]["r40_common"]
    directory = r40.ROUND_DIR / "launch_claims" / r40._slug(logical)
    if not directory.is_dir():
        return False
    for path in sorted(directory.glob("attempt_*.json")):
        try:
            claim = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if claim.get("status") == "complete" and r68eval._receipt_is_sound(claim, cell):
            return True
    return False


# ------------------------------------------------------------------ arms / execution -----
def _load_arm(cfg, seed: int, arm: str):
    """Read-only ROUND73 arm controller: this run's pure-cost-objective-trained ep128."""
    import r42_one_shot  # noqa: F401  (ensures the frozen collector module is importable)
    common = cfg["common"]
    seed = int(seed)
    paths = execute.init_paths(cfg, seed)
    if arm != "NEW_EP128":
        raise ValueError(f"unknown ROUND73 arm {arm}")
    if not paths["ep128"].is_file():
        raise FileNotFoundError(f"seed {seed}: ROUND73 ep128 checkpoint missing")
    controller, state, identity = ckpt.r73_load_checkpoint_read_only(paths["ep128"])
    if int(state.get("episodes_done", -1)) != execute.EPISODES_PER_SEED:
        raise RuntimeError(f"seed {seed}: ROUND73 ep128 is not {execute.EPISODES_PER_SEED} episodes")
    if int(state.get("updates", -1)) != execute.EPISODES_PER_SEED // execute.EPISODES_PER_UPDATE:
        raise RuntimeError(f"seed {seed}: ROUND73 ep128 did not run 32 updates")
    if state.get("actor_objective") != ckpt.R73_ACTOR_OBJECTIVE:
        raise RuntimeError(f"seed {seed}: the ep128 training state is not the pure-cost objective")
    if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
        raise RuntimeError(f"seed {seed}: NEW_EP128 read-only eval restored training state")
    if any(p.requires_grad for p in r68.frozen_parameters(controller)):
        raise RuntimeError(f"seed {seed}: NEW_EP128 lost its frozen mask")
    return controller, identity


def eval_model_arms(cfg, plan) -> dict:
    common = cfg["common"]; runner = cfg["runner"]
    import r42_one_shot
    for cell in eval_job_order():
        if cell["scope"] != "model_eval":
            continue
        seed, arm, mode, scene = int(cell["seed"]), cell["arm"], cell["mode"], int(cell["scene"])
        logical = logical_job(cell)
        VISITED.append(logical)
        if _cell_complete(cfg, logical, cell):
            continue
        controller, identity = _load_arm(cfg, seed, arm)
        collector = r42_one_shot.OneShotSendSkipCollector(controller, greedy=(mode == "greedy"))
        tag = f"{ROW_TAG}_model_{arm.lower()}_s{seed}_{mode}"
        metadata = {"arm": arm, "training_seed": seed, "mode": mode, "scenario_seed": scene,
                    "eval_version": EVAL_VERSION, "actor_objective": ckpt.R73_ACTOR_OBJECTIVE}

        def _do(run_tag, controller=controller, collector=collector, tag=tag, scene=scene,
                mode=mode, arm=arm, identity=identity, seed=seed):
            row = runner._run_episode(run_tag, "eval", scene, policy="learned",
                                      attach=collector.policy(), controller=controller,
                                      collector=collector,
                                      action_rng_seed=runner.eval_action_seed(seed, scene, mode),
                                      tag=run_tag, checkpoint_load_identity=identity)
            row["model_arm"] = arm; row["eval_mode"] = mode; row["training_seed"] = seed
            return row

        _safe_atomic_run(cfg, logical, metadata, _do,
                         detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}",
                         tag=tag, persist=lambda r, t: persist_result(cfg, r, t), cell=cell)
    return {"status": "model_arms_eval_complete", "arms": list(MODEL_ARMS),
            "cells": MODEL_EVAL_COUNT}


def eval_focal(cfg, plan) -> dict:
    runner = cfg["runner"]
    import r42_one_shot
    for cell in eval_job_order():
        if cell["scope"] != "focal":
            continue
        seed, arm = int(cell["seed"]), cell["arm"]
        logical = logical_job(cell)
        VISITED.append(logical)
        if _cell_complete(cfg, logical, cell):
            continue
        controller, identity = _load_arm(cfg, seed, arm)
        collector = r42_one_shot.OneShotSendSkipCollector(controller, greedy=True)
        tag = f"{ROW_TAG}_focal_{arm.lower()}_s{seed}"
        metadata = {"arm": arm, "training_seed": seed, "mode": "greedy",
                    "scenario_seed": FOCAL_SCENE, "block": "focal", "eval_version": EVAL_VERSION,
                    "actor_objective": ckpt.R73_ACTOR_OBJECTIVE}

        def _do(run_tag, controller=controller, collector=collector, tag=tag, arm=arm,
                identity=identity, seed=seed):
            row = runner._run_episode(run_tag, "eval", FOCAL_SCENE, policy="learned",
                                      attach=collector.policy(), controller=controller,
                                      collector=collector,
                                      action_rng_seed=runner.eval_action_seed(seed, FOCAL_SCENE,
                                                                              "greedy"),
                                      tag=run_tag, checkpoint_load_identity=identity)
            row["model_arm"] = arm; row["eval_mode"] = "greedy"; row["training_seed"] = seed
            row["focal"] = True
            return row

        _safe_atomic_run(cfg, logical, metadata, _do,
                         detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}",
                         tag=tag, persist=lambda r, t: persist_result(cfg, r, t), cell=cell)
    return {"status": "focal_complete", "cells": FOCAL_COUNT}


def _assert_visited_order(plan) -> dict:
    expected = [j["logical_job"] for j in plan["jobs"]]
    if VISITED != expected:
        for i, (a, b) in enumerate(zip(VISITED, expected)):
            if a != b:
                raise AssertionError(f"execution order diverges from the plan at cell {i}: "
                                     f"ran {a}, plan {b}")
        raise AssertionError(f"visited {len(VISITED)} cells, plan has {len(expected)}")
    return {"kind": "round73-eval-order-check/1", "status": "passed", "cells": len(VISITED),
            "note": "the executor visited every planned cell in exactly the planned order"}

# ------------------------------------------------------------------ statistics -----------
def first_report_sets(row_path) -> list:
    """Per-target first free SEND/SKIP decisions from one eval row's tape, split early/late.

    The first report of a target is its first decision step in which it is free (actionable);
    the region comes from the SAME legal 19-D input the policy saw (ready_step =
    round(f[18]*1000) - round(f[0]), early when <= 147).
    """
    row = _read_json(row_path)
    tape = Path(row["artefacts"]["tape"])
    records = [json.loads(line) for line in tape.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    out, seen = [], set()
    for rec in sorted(records, key=lambda r: int(r["step"])):
        targets = list(rec.get("targets") or [])
        actionable = set(rec.get("actionable") or [])
        actions = dict(rec.get("actions") or {})
        features = rec.get("features") or []
        for index, target in enumerate(targets):
            if target in seen or target not in actionable:
                continue
            seen.add(target)
            action = actions.get(target)
            ready = None
            if index < len(features):
                vector = features[index]
                if vector is not None and len(vector) >= 19:
                    ready = int(round(float(vector[18]) * 1000.0) - round(float(vector[0])))
            out.append({"target": str(target), "step": int(rec["step"]), "action": action,
                        "ready_step": ready,
                        "region": ("early" if ready is not None and ready <= READY_CUT
                                   else "late")})
    return out


def _first_report_summary(rows_by_scene) -> dict:
    sends = {"early": 0, "late": 0}
    skips = {"early": 0, "late": 0}
    unknown = 0
    events = []
    for scene in sorted(rows_by_scene):
        for event in first_report_sets(rows_by_scene[scene]["row_path"]):
            event = dict(event, scene=int(scene))
            events.append(event)
            if event["region"] not in sends:
                unknown += 1
            elif event["action"] == 1:
                sends[event["region"]] += 1
            elif event["action"] == 0:
                skips[event["region"]] += 1
            else:
                unknown += 1
    return {"sends_early": sends["early"], "skips_early": skips["early"],
            "sends_late": sends["late"], "skips_late": skips["late"],
            "unknown_region_or_action": unknown, "events": events}


def _first_report_action_diff(rows_new, rows_control, scenes) -> dict:
    """Per-(scene, target) first-report action differences, split early/late."""
    def index(rows):
        out = {}
        for scene in scenes:
            for event in first_report_sets(rows[scene]["row_path"]):
                out[(int(scene), event["target"])] = event
        return out
    a, b = index(rows_new), index(rows_control)
    flips = []
    for key in sorted(set(a) & set(b)):
        if a[key]["action"] != b[key]["action"]:
            flips.append({"scene": key[0], "target": key[1], "region": a[key]["region"],
                          "new_action": a[key]["action"], "control_action": b[key]["action"],
                          "ready_step": a[key]["ready_step"]})
    only_new = sorted(set(a) - set(b))
    only_control = sorted(set(b) - set(a))
    return {"action_flips_new_vs_control": flips,
            "flips_early": [f for f in flips if f["region"] == "early"],
            "flips_late": [f for f in flips if f["region"] == "late"],
            "first_reports_only_in_new": [{"scene": s, "target": t} for s, t in only_new],
            "first_reports_only_in_control": [{"scene": s, "target": t} for s, t in only_control]}


def _rule_context() -> dict:
    data = _read_json(RULE_CONTEXT_PATH)
    out = {}
    for arm in RULE_ARMS:
        block = data["rule_absolute"][arm]
        out[arm] = {"on_time_total": int(block["on_time_total"]),
                    "bytes_per_scene": float(block["spent_bytes_per_scene_mean"]),
                    "on_time_ids_per_scene": {int(k): sorted(v) for k, v
                                              in block["on_time_target_ids_per_scene"].items()}}
    out["_audit"] = {"qualified_rule_cells": data["rule_audit"]["qualified_rule_cells"],
                     "source_root": data["rule_audit"]["source_root"],
                     "environment_constructions": data["environment_constructions"],
                     "new_rule_simulations": data["new_rule_simulations"]}
    return out


def _load_focal_rows(raw_root: Path, row_tag: str, arm_filter=None) -> dict:
    qualified = r68eval._qualified_row_paths(Path(raw_root), row_tag)
    out = {}
    for logical in sorted(qualified):
        if not logical.startswith(f"{row_tag}_focal_"):
            continue
        paths = qualified[logical]
        if len(paths) != 1:
            raise RuntimeError(f"{logical}: {len(paths)} sound complete receipts name different "
                               f"rows; refusing to choose one by order")
        row = _read_json(paths[0])
        arm = str(row["model_arm"]).upper()
        if arm_filter is not None and arm not in arm_filter:
            continue
        key = (arm, int(row["training_seed"]))
        out[key] = {"on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
                    "on_time_ids": r68eval.on_time_target_ids(row, FOCAL_SCENE),
                    "row_path": paths[0], "logical_job": logical,
                    "task_reward_events": row.get("task_reward_events") or []}
    return out


def _focal_detail(focal_rows: dict, label: str) -> dict:
    out = {}
    for (arm, seed), cell in sorted(focal_rows.items()):
        target_events = [e for e in cell["task_reward_events"]
                         if str(e.get("task_id")) == FOCAL_TARGET]
        first_reports = [e for e in first_report_sets(cell["row_path"])
                         if e["target"] == FOCAL_TARGET]
        out[f"{label}_s{seed}"] = {
            "on_time": cell["on_time"], "spent_bytes": cell["spent_bytes"],
            "on_time_target_ids": cell["on_time_ids"],
            "target5_on_time": FOCAL_TARGET in cell["on_time_ids"],
            "target5_events": target_events, "target5_first_reports": first_reports,
            "row_path": cell["row_path"]}
    return out


def _focal_raw_delta(a, b, label_a: str, label_b: str) -> dict:
    """Raw paired difference on ONE scene: no bootstrap CI (n=1), no significance claim."""
    if a is None or b is None:
        return {"unavailable": f"missing focal cell for {label_a} or {label_b}"}
    a_ids, b_ids = set(a["on_time_ids"]), set(b["on_time_ids"])
    return {f"task_delta_{label_a}_minus_{label_b}": int(a["on_time"] - b["on_time"]),
            f"cost_delta_{label_a}_minus_{label_b}_B": int(a["spent_bytes"] - b["spent_bytes"]),
            f"targets_lost_{label_a}_vs_{label_b}": sorted(b_ids - a_ids),
            f"targets_gained_{label_a}_vs_{label_b}": sorted(a_ids - b_ids),
            f"on_time_ids_{label_a}": sorted(a_ids),
            f"on_time_ids_{label_b}": sorted(b_ids),
            "target5_on_time_a": FOCAL_TARGET in a_ids,
            "target5_on_time_b": FOCAL_TARGET in b_ids,
            "note": "raw single-scene difference; no CI, no significance claim"}


def compute_stats(raw_root: Path) -> dict:
    idx, idx_sha = r68eval._bootstrap_index()
    raw_root = Path(raw_root)
    new_rows, new_audit = r68eval._load_model_rows(raw_root, ROW_TAG)
    r71_rows, r71_audit = r68eval._load_model_rows(R71_DIR / "raw", R71_ROW_TAG)
    control = {k: v for k, v in r71_rows.items() if k[0] == CONTROL_ARM}
    focal_new = _load_focal_rows(raw_root, ROW_TAG)
    focal_r71 = _load_focal_rows(R71_DIR / "raw", R71_ROW_TAG)
    scenes = list(SCENES_NEW)
    rules = _rule_context()
    per: dict[str, Any] = {}
    for seed in SEEDS:
        for mode in MODES:
            def cell(store, arm, s, seed=seed, mode=mode):
                return store.get((arm, seed, mode, s))

            missing = [(MODEL_EVAL_ARM, s) for s in scenes
                       if cell(new_rows, MODEL_EVAL_ARM, s) is None]
            missing += [(CONTROL_ARM, s) for s in scenes if cell(control, CONTROL_ARM, s) is None]
            if missing:
                raise RuntimeError(f"missing ROUND73 paired cells seed={seed} mode={mode}: "
                                   f"{missing[:4]} ({len(missing)})")
            entry: dict[str, Any] = {}
            entry[f"{MODEL_EVAL_ARM}_on_time_total"] = int(
                sum(cell(new_rows, MODEL_EVAL_ARM, s)["on_time"] for s in scenes))
            entry[f"{MODEL_EVAL_ARM}_bytes_per_scene"] = float(np.mean(
                [cell(new_rows, MODEL_EVAL_ARM, s)["spent_bytes"] for s in scenes]))
            entry[f"{CONTROL_LABEL}_on_time_total"] = int(
                sum(cell(control, CONTROL_ARM, s)["on_time"] for s in scenes))
            entry[f"{CONTROL_LABEL}_bytes_per_scene"] = float(np.mean(
                [cell(control, CONTROL_ARM, s)["spent_bytes"] for s in scenes]))
            # PRIMARY: the pure-cost-trained ep128 vs the ROUND71 ep128 control
            entry["primary"] = r68eval._paired_delta(
                lambda s: cell(new_rows, MODEL_EVAL_ARM, s),
                lambda s: cell(control, CONTROL_ARM, s), scenes, idx,
                MODEL_EVAL_ARM, CONTROL_LABEL)
            # rule context columns (read-only; the same 32 scenes)
            rule_columns = {}
            for arm in RULE_ARMS:
                rule_columns[arm] = {
                    "on_time_total": rules[arm]["on_time_total"],
                    "bytes_per_scene": rules[arm]["bytes_per_scene"],
                    "task_delta_new_minus_rule": int(
                        entry[f"{MODEL_EVAL_ARM}_on_time_total"] - rules[arm]["on_time_total"]),
                    "cost_delta_new_minus_rule_B_per_scene": float(
                        entry[f"{MODEL_EVAL_ARM}_bytes_per_scene"] - rules[arm]["bytes_per_scene"]),
                }
            entry["arms_vs_rules"] = rule_columns
            # first reports, split early/late: NEW and the ROUND71 control
            first = _first_report_summary(
                {s: cell(new_rows, MODEL_EVAL_ARM, s) for s in scenes})
            control_first = _first_report_summary(
                {s: cell(control, CONTROL_ARM, s) for s in scenes})
            entry["first_reports"] = {
                MODEL_EVAL_ARM: {k: v for k, v in first.items() if k != "events"},
                CONTROL_LABEL: {k: v for k, v in control_first.items() if k != "events"},
                "action_diff_new_vs_control": _first_report_action_diff(
                    {s: cell(new_rows, MODEL_EVAL_ARM, s) for s in scenes},
                    {s: cell(control, CONTROL_ARM, s) for s in scenes}, scenes)}
            per[f"s{seed}_{mode}"] = entry
    focal_per_seed = {}
    for seed in SEEDS:
        a = focal_new.get((MODEL_EVAL_ARM, seed))
        b = focal_r71.get((CONTROL_ARM, seed))
        focal_per_seed[f"s{seed}"] = _focal_raw_delta(a, b, "NEW_EP128", "R71_EP128")
    stats = {"kind": "round73-statistics/1", "task_id": trial.R73_TRIAL_TASK_ID,
             "branch": trial.R73_TRIAL_BRANCH, "run_id": trial.TRIAL_RUN_ID,
             "training_version": ckpt.R73_TRAINING_VERSION,
             "actor_objective": ckpt.R73_ACTOR_OBJECTIVE,
             "eval_version": EVAL_VERSION,
             "primary_comparison": "NEW_EP128_minus_R71_EP128",
             "greedy_primary": True, "sampled_separate": True, "seeds_not_pooled": True,
             "scene_unit": 32, "bootstrap_index_sha256": idx_sha,
             "bootstrap_index_source": "inherited from the ROUND68/R71 chain (PCG64 20261012)",
             "row_audit": {"round73": new_audit, "round71_control": r71_audit},
             "rule_audit": rules["_audit"],
             "per_seed_mode": per,
             "focal": {"scene": FOCAL_SCENE, "target": FOCAL_TARGET,
                       "per_seed": focal_per_seed,
                       "new_detail": _focal_detail(focal_new, "NEW_EP128"),
                       "r71_control_detail": _focal_detail(focal_r71, "R71_EP128")}}
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND73 evaluation adapter")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--phase", choices=("plan", "eval", "stats"), required=True)
    args = ap.parse_args(argv)
    raw_root = Path(args.raw_root).resolve()
    if args.phase == "plan":
        plan = build_eval_plan()
        _write_json(raw_root / "ROUND73_EVAL_PLAN.json", plan)
        print(json.dumps({k: v for k, v in plan.items() if k != "jobs"}, indent=1))
        return 0
    cfg = configure_r73_trial_eval(raw_root)
    _write_json(raw_root / "runtime" / "ROUND73_EVAL_QUOTA_GUARD.json",
                cfg["r73_eval_quota_guard"])
    if args.phase == "eval":
        plan = _read_json(raw_root / "ROUND73_EVAL_PLAN.json")
        VISITED.clear()
        print(json.dumps(eval_model_arms(cfg, plan), indent=1), flush=True)
        print(json.dumps(eval_focal(cfg, plan), indent=1), flush=True)
        order = _assert_visited_order(plan)
        _write_json(raw_root / "runtime" / "ROUND73_EVAL_ORDER_CHECK.json", order)
        print(json.dumps(order, indent=1), flush=True)
        _write_json(raw_root / "runtime" / "ROUND73_EVAL_QUOTA_GUARD.json", quota_guard(cfg))
        return 0
    stats = compute_stats(raw_root)
    _write_json(raw_root / "ROUND73_STATISTICS.json", stats)
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("per_seed_mode", "focal")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())