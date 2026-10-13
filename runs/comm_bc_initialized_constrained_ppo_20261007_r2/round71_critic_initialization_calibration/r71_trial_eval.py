#!/usr/bin/env python
"""ROUND71 approved fixed trial: evaluation adapter (192 model cells + 6 focal cells).

A faithful adaptation of the verified ROUND68 evaluation seam with an independent ROUND71
identity and the ROUND71 comparison contract:

  * arms: NEW_INIT (this run's calibrated INIT checkpoint, read-only) and NEW_EP128 (this
    run's trained ep128, read-only); the R68 fresh factory is never used for an R71 arm;
  * booking goes through the SAME claimed-but-started-safe helper and construction-permit
    regime as training (``r68_execute._r68_atomic_run``), so eval and train cannot drift
    apart; every complete claim carries a receipt that re-proves from disk;
  * PRIMARY comparison: NEW_EP128 minus the ROUND68 TRAINED_ALPHA control, per (seed, mode)
    over the 32 frozen scenes 1800000..1800031 (greedy primary, sampled separate, seeds not
    pooled), with the ROUND68 rows reused READ-ONLY;
  * CONTEXT: NEW_INIT minus ROUND68 INIT_ALPHA (the calibration must not move the initial
    policy at all: only the critic head VALUES differ, the actor path is untouched) and
    NEW_EP128 minus NEW_INIT (the training effect from the calibrated init); both are
    measured on the focal scene, where both arms run (NEW_INIT is not re-evaluated on the
    192-cell block);
  * two-column task/cost context against the IMMEDIATE and LATE_DISCOVERY_GATE rules from
    the frozen ROUND68 rule-context file (read-only);
  * focal scene 1700027 detail (the early key target piloc-e1-target-5) per seed and arm;
  * first-report SEND/SKIP sets split early (ready_step <= 147) / late, from the eval tapes.

Budget: 192 model cells + 6 focal cells = 198 of the 582 approved normal launches; the
remaining 384 belong to training.  Nothing here re-simulates a rule or a control arm.
"""
from __future__ import annotations

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
READINESS_DIR = REPO / "analysis" / "round71-critic-calibration-readiness"
for _d in (READINESS_DIR, str(R68_DIR), str(HERE)):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import r71_trial_binding as trial                                       # noqa: E402
import r71_runtime_binding as readiness                                  # noqa: E402
import r71_trial_execute as execute                                      # noqa: E402
import r68_execute as r68x                                               # noqa: E402
import r68_eval as r68eval                                               # noqa: E402
import r68_frozen as r68                                                 # noqa: E402

SEEDS = execute.SEEDS
MODES = execute.MODES
SCENES_NEW = execute.SCENES_NEW
FOCAL_SCENE = execute.FOCAL_SCENE
MODEL_ARMS = trial.MODEL_ARMS                    # ("NEW_INIT", "NEW_EP128")
MODEL_EVAL_ARM = execute.MODEL_EVAL_ARM        # the 192-cell block evaluates this arm only
ROW_TAG = trial.EVAL_ROW_TAG                    # r71eval
R68_ROW_TAG = r68eval.ROW_TAG                   # r68eval (read-only control reuse)
MODEL_EVAL_COUNT = 192
FOCAL_COUNT = 6
EVAL_CELLS = MODEL_EVAL_COUNT + FOCAL_COUNT
EVAL_VERSION = readiness.R71_TRAINING_VERSION + "+r71-eval/1"
RULE_CONTEXT_PATH = R68_DIR / "raw" / "ROUND68_RULE_CONTEXT.json"
RULE_ARMS = ("IMMEDIATE", "LATE_DISCOVERY_GATE")
READY_CUT = 147.0

_sha256_file = r68x._sha256_file
_read_json = r68x._read_json
_write_json = r68x._write_json
VISITED: list = []


# ------------------------------------------------------------------ plan / identity ------
def eval_job_order():
    """Every ROUND71 evaluation cell in the exact order the executor requests it.

    The 192-cell block carries the TRAINED arm only (the licence allots 192 evaluation
    launches to 3 seeds x 2 modes x 32 scenes); both arms run on the 6 focal cells.
    """
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
        raise AssertionError(f"ROUND71 eval plan is not {MODEL_EVAL_COUNT}+{FOCAL_COUNT}="
                             f"{EVAL_CELLS} unique jobs: {counts}")
    if len({j["logical_job"] for j in jobs}) != EVAL_CELLS:
        raise AssertionError("ROUND71 eval plan contains duplicate logical jobs")
    planned = [j["planned_normal_slot"] for j in jobs]
    if planned != list(range(385, 385 + EVAL_CELLS)):
        raise AssertionError("planned normal slots are not the contiguous range 385..582")
    if execute.TRAINING_COUNT + EVAL_CELLS != execute.NORMAL_CAP:
        raise AssertionError("384 training + 198 eval cells != the 582 normal cap")
    return {"kind": "round71-eval-plan/1", "task_id": trial.R71_TRIAL_TASK_ID,
            "branch": trial.R71_TRIAL_BRANCH, "run_id": trial.TRIAL_RUN_ID,
            "eval_version": EVAL_VERSION, "training_version": readiness.R71_TRAINING_VERSION,
            "matrix": {"model_eval": MODEL_EVAL_COUNT, "focal": FOCAL_COUNT, "total": EVAL_CELLS},
            "model_arms": list(MODEL_ARMS), "model_eval_arm": MODEL_EVAL_ARM,
            "arm_allocation": ("the 192-cell block evaluates NEW_EP128 only; NEW_INIT runs on "
                               "the 6 focal cells (scene 1700027, greedy, 3 seeds)"),
            "primary_comparison": "NEW_EP128_minus_R68_TRAINED_ALPHA",
            "primary_meaning": ("this run's calibrated-init-trained ep128 minus the ROUND68 "
                                "trained control (same trunk/actor init, same training data and "
                                "order; only the critic head INIT values differ). ROUND68 rows "
                                "are reused READ-ONLY and are not new constructions."),
            "context_comparisons": {
                "NEW_INIT_minus_R68_INIT_ALPHA": ("the calibration must not move the initial "
                                                  "policy: identical actors and trunks, only the "
                                                  "critic head values differ, so this context is "
                                                  "expected to be exactly zero. NEW_INIT is not "
                                                  "re-evaluated on the 192-cell block; this "
                                                  "context is measured on the focal scene"),
                "NEW_EP128_minus_NEW_INIT": ("the training effect from the calibrated init, "
                                             "measured on the focal scene"),
                "arms_vs_rules": "two-column task/cost against IMMEDIATE and LATE_DISCOVERY_GATE"},
            "greedy_is_primary": True, "sampled_reported_separately": True,
            "scene_unit": 32, "seeds_not_pooled": True,
            "eval_block": "1800000..1800031 (already-used development block, NOT a confirmation set)",
            "eval_scenes": list(SCENES_NEW), "focal_scene": FOCAL_SCENE,
            "focal_target": "piloc-e1-target-5",
            "execution_order": ["model_eval", "focal"],
            "row_namespace": {"tag_prefix": ROW_TAG,
                              "row_globs": [f"v2_{ROW_TAG}_model_*.json", f"v2_{ROW_TAG}_focal_*.json"],
                              "disjoint_from": [f"v2_{R68_ROW_TAG}_*.json"]},
            "normal_cap": execute.NORMAL_CAP, "reserve_cap": execute.RESERVE_CAP,
            "hard_cap": execute.HARD_CAP, "jobs": jobs}


# ------------------------------------------------------------------ configure / guard ----
def configure_r71_trial_eval(raw_root: Path) -> dict[str, Any]:
    """Bind the live ROUND71 trial runtime (quota 582/18/600 on the live modules) before any
    builder can run."""
    cfg = execute.configure_r71_trial(
        Path(raw_root), trial.trial_gate(), trial.trial_run_identity(),
        model_read_approval=trial.MODEL_READ_APPROVAL)
    cfg["r71_eval_quota_guard"] = quota_guard(cfg)
    return cfg


def quota_guard(cfg: dict[str, Any]) -> dict:
    """The real hard-cap judgement: booked slots <= the ROUND71 caps."""
    r40 = cfg["runtime"]["r40_common"]
    snapshot = r40.ledger_snapshot()          # itself raises if a live cap is already exceeded
    normal_used = int(snapshot["normal_used"])
    reserve_used = int(snapshot["reserve_used"])
    booked = normal_used + reserve_used
    hard = int(r40.HARD_CAP)
    if normal_used > execute.NORMAL_CAP:
        raise RuntimeError(f"ROUND71-EVAL: booked normal {normal_used} exceeds the normal cap "
                           f"{execute.NORMAL_CAP}")
    if reserve_used > execute.RESERVE_CAP:
        raise RuntimeError(f"ROUND71-EVAL: booked reserve {reserve_used} exceeds the ceiling "
                           f"{execute.RESERVE_CAP}")
    if booked > hard:
        raise RuntimeError(f"ROUND71-EVAL: booked constructions {booked} exceed the hard cap {hard}")
    return {"kind": "round71-eval-quota-guard/1", "eval_version": EVAL_VERSION,
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
    if int(reread.get("steps", -1)) <= 0:
        raise RuntimeError(f"{run_tag}: persisted row records no steps")
    if int(reread.get("on_time", -1)) != int(row.get("on_time", -2)):
        raise RuntimeError(f"{run_tag}: persisted row does not read back consistently")
    if _sha256_file(row_path) != first:
        raise RuntimeError(f"{run_tag}: persisted row hash is not stable across re-reads")
    companions, not_produced, present = {}, [], []
    for key in ARTEFACT_KEYS:
        if key == "row":
            continue
        declared = artefacts.get(key)
        if not declared:
            not_produced.append({"key": key, "reason": "not declared by this row"})
            continue
        path = Path(declared)
        if not path.is_file():
            not_produced.append({"key": key, "path": str(path),
                                 "reason": "declared by the runner but never written: this run "
                                           "kind attached no collector"})
            continue
        if path.stat().st_size == 0:
            raise RuntimeError(f"{run_tag}: artefact {key} is empty at {path}")
        companions[key] = {"path": str(path), "sha256": _sha256_file(path),
                           "bytes": int(path.stat().st_size), "status": "present"}
        present.append(key)
    if not present:
        raise RuntimeError(f"{run_tag}: no companion artefact was produced at all")
    return {"row_path": str(row_path), "row_sha256": first,
            "row_bytes": int(row_path.stat().st_size),
            "companions": companions, "declared_but_not_produced": not_produced,
            "run_tag": run_tag, "persisted_at_unix": time.time()}


def _safe_atomic_run(cfg, logical, metadata, run_fn, *, detail, tag, persist=None, cell=None):
    """ROUND71 evaluation booking: the SHARED claimed-but-started-safe helper, one path only."""
    r40 = cfg["runtime"]["r40_common"]
    r42_run = cfg["r42_run"]
    if r42_run._ACTIVE_CLAIM is not None:
        raise RuntimeError("nested ROUND71 evaluation claims are not allowed")
    directory = r40.ROUND_DIR / "launch_claims" / r40._slug(logical)
    prior = []
    if directory.is_dir():
        for path in sorted(directory.glob("attempt_*.json")):
            try:
                prior.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                prior.append({"attempt": None, "status": "unreadable"})
    if any(c.get("status") == "complete" and r68eval._receipt_is_sound(c, cell) for c in prior):
        raise AlreadyComplete(f"{logical}: a complete attempt with a sound receipt already exists")
    holder = {"run_tag": f"{tag}_a1", "claim_path": None, "attempt": None}

    def _on_claim(claim_path, attempt):
        holder["run_tag"] = f"{tag}_a{int(attempt)}"
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
    """Read-only ROUND71 arm controllers.

    NEW_INIT: this run's calibrated INIT checkpoint (five modules model-only, no optimizer,
    no lambda, no training RNG); its parameter identity must equal the registered calibrated
    init identity written by the offline-fit phase.
    NEW_EP128: this run's trained ep128 checkpoint, read-only.
    """
    import r42_one_shot  # noqa: F401  (ensures the frozen collector module is importable)
    common = cfg["common"]
    seed = int(seed)
    paths = execute.init_paths(cfg, seed)
    if arm == "NEW_INIT":
        if not paths["init"].is_file():
            raise FileNotFoundError(f"seed {seed}: ROUND71 calibrated INIT checkpoint missing")
        controller, state, identity = readiness.r71_load_checkpoint_read_only(paths["init"])
        if int(state.get("episodes_done", -1)) != 0 or int(state.get("updates", -1)) != 0:
            raise RuntimeError(f"seed {seed}: NEW_INIT is not an untrained initialisation")
        if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
            raise RuntimeError(f"seed {seed}: NEW_INIT read-only eval restored training state")
        registered = _read_json(paths["identity"])
        live = controller.parameter_summary()["parameter_sha256"]
        if live != registered["init_parameter_sha256"]:
            raise RuntimeError(f"seed {seed}: NEW_INIT parameters != the registered calibrated "
                               f"identity")
        if any(p.requires_grad for p in r68.frozen_parameters(controller)):
            raise RuntimeError(f"seed {seed}: NEW_INIT lost its frozen mask")
        identity["registered_init_identity"] = registered
        return controller, identity
    if arm == "NEW_EP128":
        if not paths["ep128"].is_file():
            raise FileNotFoundError(f"seed {seed}: ROUND71 ep128 checkpoint missing")
        controller, state, identity = readiness.r71_load_checkpoint_read_only(paths["ep128"])
        if int(state.get("episodes_done", -1)) != execute.EPISODES_PER_SEED:
            raise RuntimeError(f"seed {seed}: ROUND71 ep128 is not "
                               f"{execute.EPISODES_PER_SEED} episodes")
        if int(state.get("updates", -1)) != execute.EPISODES_PER_SEED // execute.EPISODES_PER_UPDATE:
            raise RuntimeError(f"seed {seed}: ROUND71 ep128 did not run 32 updates")
        if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
            raise RuntimeError(f"seed {seed}: NEW_EP128 read-only eval restored training state")
        if any(p.requires_grad for p in r68.frozen_parameters(controller)):
            raise RuntimeError(f"seed {seed}: NEW_EP128 lost its frozen mask")
        return controller, identity
    raise ValueError(f"unknown ROUND71 arm {arm}")


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
                    "eval_version": EVAL_VERSION}

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
                    "scenario_seed": FOCAL_SCENE, "block": "focal", "eval_version": EVAL_VERSION}

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
    return {"kind": "round71-eval-order-check/1", "status": "passed", "cells": len(VISITED),
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
            "unknown_actions": unknown, "events": events}


def _rule_context() -> dict:
    data = _read_json(RULE_CONTEXT_PATH)
    out = {}
    for arm in RULE_ARMS:
        block = data["rule_absolute"][arm]
        out[arm] = {"on_time_total": int(block["on_time_total"]),
                    "bytes_per_scene": float(block["spent_bytes_per_scene_mean"]),
                    "on_time_ids_per_scene": {int(k): sorted(v) for k, v
                                              in block["on_time_target_ids_per_scene"].items()}}
    return out


def _load_focal_rows(raw_root: Path, row_tag: str) -> dict:
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
        key = (str(row["model_arm"]).upper(), int(row["training_seed"]))
        out[key] = {"on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
                    "on_time_ids": r68eval.on_time_target_ids(row, FOCAL_SCENE),
                    "row_path": paths[0], "logical_job": logical,
                    "task_reward_events": row.get("task_reward_events") or []}
    return out


def _focal_detail(focal_rows: dict) -> dict:
    out = {}
    for (arm, seed), cell in sorted(focal_rows.items()):
        target_events = [e for e in cell["task_reward_events"]
                         if str(e.get("task_id")) == "piloc-e1-target-5"]
        first_reports = [e for e in first_report_sets(cell["row_path"])
                         if e["target"] == "piloc-e1-target-5"]
        out[f"{arm}_s{seed}"] = {
            "on_time": cell["on_time"], "spent_bytes": cell["spent_bytes"],
            "on_time_target_ids": cell["on_time_ids"],
            "target5_on_time": "piloc-e1-target-5" in cell["on_time_ids"],
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
            "note": "raw single-scene difference; no CI, no significance claim"}


def compute_stats(raw_root: Path) -> dict:
    idx, idx_sha = r68eval._bootstrap_index()
    raw_root = Path(raw_root)
    r71_rows, r71_audit = r68eval._load_model_rows(raw_root, ROW_TAG)
    r68_rows, r68_audit = r68eval._load_model_rows(R68_DIR / "raw", R68_ROW_TAG)
    r68_trained = {k: v for k, v in r68_rows.items() if k[0] == "TRAINED_ALPHA"}
    r68_init = {k: v for k, v in r68_rows.items() if k[0] == "INIT_ALPHA"}
    focal_rows = _load_focal_rows(raw_root, ROW_TAG)
    scenes = list(SCENES_NEW)
    rules = _rule_context()
    per: dict[str, Any] = {}
    for seed in SEEDS:
        for mode in MODES:
            def cell(store, arm, s, seed=seed, mode=mode):
                return store.get((arm, seed, mode, s))

            # the 192-cell block carries the TRAINED arm only (licence allocation); NEW_INIT
            # runs on the focal scene, so its contexts are computed in the focal section.
            missing = [(MODEL_EVAL_ARM, s) for s in scenes
                       if cell(r71_rows, MODEL_EVAL_ARM, s) is None]
            missing += [("R68_TRAINED_ALPHA", s) for s in scenes
                        if cell(r68_trained, "TRAINED_ALPHA", s) is None]
            if missing:
                raise RuntimeError(f"missing ROUND71 paired cells seed={seed} mode={mode}: "
                                   f"{missing[:4]} ({len(missing)})")
            entry: dict[str, Any] = {}
            entry[f"{MODEL_EVAL_ARM}_on_time_total"] = int(
                sum(cell(r71_rows, MODEL_EVAL_ARM, s)["on_time"] for s in scenes))
            entry[f"{MODEL_EVAL_ARM}_bytes_per_scene"] = float(np.mean(
                [cell(r71_rows, MODEL_EVAL_ARM, s)["spent_bytes"] for s in scenes]))
            # PRIMARY: the calibrated-init-trained ep128 vs the ROUND68 trained control
            entry["primary"] = r68eval._paired_delta(
                lambda s: cell(r71_rows, MODEL_EVAL_ARM, s),
                lambda s: cell(r68_trained, "TRAINED_ALPHA", s), scenes, idx,
                MODEL_EVAL_ARM, "R68_TRAINED_ALPHA")
            # CONTEXT placeholders on the block: NEW_INIT is not evaluated here
            entry["context_new_init_minus_r68_init_alpha"] = {
                "unavailable_on_the_192_cell_block":
                    "NEW_INIT runs on the focal scene only; see focal.per_seed"}
            entry["training_effect_new_ep128_minus_new_init"] = {
                "unavailable_on_the_192_cell_block":
                    "NEW_INIT runs on the focal scene only; see focal.per_seed"}
            # first reports, split early/late: the trained arm and the ROUND68 control
            first = _first_report_summary(
                {s: cell(r71_rows, MODEL_EVAL_ARM, s) for s in scenes})
            control_first = _first_report_summary(
                {s: cell(r68_trained, "TRAINED_ALPHA", s) for s in scenes})
            entry["first_reports"] = {
                MODEL_EVAL_ARM: {k: v for k, v in first.items() if k != "events"},
                "R68_TRAINED_ALPHA": {k: v for k, v in control_first.items() if k != "events"}}
            changed = []
            for a, b in zip(first["events"], control_first["events"]):
                if a != b:
                    changed.append({MODEL_EVAL_ARM: a, "r68_trained_alpha": b})
            entry["first_reports"][f"{MODEL_EVAL_ARM}_vs_r68_trained_changed_events"] = changed
            # two-column task/cost context against the frozen rules (trained arm)
            vs_rules = {}
            for rule in RULE_ARMS:
                task_delta = entry[f"{MODEL_EVAL_ARM}_on_time_total"] - rules[rule]["on_time_total"]
                cost_delta = (entry[f"{MODEL_EVAL_ARM}_bytes_per_scene"]
                              - rules[rule]["bytes_per_scene"])
                lost, gained = [], []
                for scene in scenes:
                    arm_ids = set(cell(r71_rows, MODEL_EVAL_ARM, scene)["on_time_ids"])
                    rule_ids = set(rules[rule]["on_time_ids_per_scene"].get(scene, []))
                    lost += [{"scene": scene, "target_id": t}
                             for t in sorted(rule_ids - arm_ids)]
                    gained += [{"scene": scene, "target_id": t}
                               for t in sorted(arm_ids - rule_ids)]
                vs_rules[f"{MODEL_EVAL_ARM}_vs_{rule}"] = {
                    "task_column_on_time_total": entry[f"{MODEL_EVAL_ARM}_on_time_total"],
                    "rule_on_time_total": rules[rule]["on_time_total"],
                    "task_delta_minus_rule": int(task_delta),
                    "cost_column_bytes_per_scene": entry[f"{MODEL_EVAL_ARM}_bytes_per_scene"],
                    "rule_bytes_per_scene": rules[rule]["bytes_per_scene"],
                    "cost_delta_minus_rule_B_per_scene": float(cost_delta),
                    "targets_lost_vs_rule": lost, "targets_gained_vs_rule": gained}
            entry["vs_rules"] = vs_rules
            per[f"{seed}/{mode}"] = entry

    # ---- focal scene 1700027: both ROUND71 arms and both ROUND68 focal controls ------------
    # n=1 per cell: raw paired differences only, no bootstrap CI, no significance claim.
    r68_focal = _load_focal_rows(R68_DIR / "raw", R68_ROW_TAG)
    focal_section = {
        "scene": FOCAL_SCENE, "target": "piloc-e1-target-5",
        "note": ("single scene, one random realisation per cell: raw paired differences "
                 "only, no bootstrap CI and no significance claim; R68 focal controls are "
                 "read-only reuse")}
    for seed in SEEDS:
        cells = {"NEW_INIT": focal_rows.get(("NEW_INIT", seed)),
                 "NEW_EP128": focal_rows.get(("NEW_EP128", seed)),
                 "R68_INIT_ALPHA": r68_focal.get(("INIT_ALPHA", seed)),
                 "R68_TRAINED_ALPHA": r68_focal.get(("TRAINED_ALPHA", seed))}
        entry_f: dict[str, Any] = {"cells_present": {k: v is not None for k, v in cells.items()}}
        for a, b in (("NEW_INIT", "R68_INIT_ALPHA"),
                     ("NEW_EP128", "R68_TRAINED_ALPHA"),
                     ("NEW_EP128", "NEW_INIT")):
            entry_f[f"{a}_minus_{b}"] = _focal_raw_delta(cells[a], cells[b], a, b)
        entry_f["first_reports"] = {
            arm: _first_report_summary({FOCAL_SCENE: cells[arm]})
            for arm in ("NEW_INIT", "NEW_EP128")}
        focal_section[f"seed{seed}"] = entry_f
    return {"kind": "round71-statistics/1", "eval_version": EVAL_VERSION,
            "training_version": readiness.R71_TRAINING_VERSION,
            "task_id": trial.R71_TRIAL_TASK_ID, "branch": trial.R71_TRIAL_BRANCH,
            "primary_comparison": "NEW_EP128_minus_R68_TRAINED_ALPHA",
            "primary_meaning": ("does replacing the two critic heads' INIT values (per-seed "
                                "dual-output ridge on that seed's own ROUND68 teacher data) "
                                "change the trained outcome vs the ROUND68 control with the "
                                "identical trunk/actor init, data and order? NOT an "
                                "added-learning gain claim and NOT a same-budget race: the "
                                "candidate additionally exposes offline teacher data."),
            "context_comparisons": {
                "NEW_INIT_minus_R68_INIT_ALPHA": ("must be exactly zero: identical actors and "
                                                  "trunks, only the critic head values differ; "
                                                  "measured on the focal scene, where NEW_INIT "
                                                  "runs"),
                "NEW_EP128_minus_NEW_INIT": ("the training effect from the calibrated init; "
                                             "measured on the focal scene"),
                "arms_vs_rules": "aggregate two-column task/cost context vs IMMEDIATE and "
                                 "LATE_DISCOVERY_GATE (rule cost is the frozen per-scene mean)"},
            "greedy_is_primary": True, "sampled_reported_separately": True,
            "scene_unit": 32, "seeds_not_pooled": True,
            "eval_block": "1800000..1800031 (already-used development block, NOT a confirmation set)",
            "focal": {"scene": FOCAL_SCENE, "target": "piloc-e1-target-5",
                      "per_seed": focal_section, "detail": _focal_detail(focal_rows)},
            "bootstrap": {"rng": "PCG64", "seed": 20261012, "draws": 10000,
                          "index_sha256": idx_sha, "quantile": "numpy linear 0.025/0.975"},
            "row_source_audit": {"r71_model": r71_audit, "r68_control_reuse": r68_audit,
                                 "admission_rule": "every admitted row is named by a complete "
                                                   "claim whose result_receipt re-proves from "
                                                   "disk; the ROUND68 rows are read-only reuse"},
            "per_seed_mode": per,
            "note": ("CI containing 0 is not equivalence. Three seeds not pooled, one random "
                     "realisation per cell, single development block. Results support only the "
                     "whole critic-initialisation-calibration structure on this block; they do "
                     "not prove the uncalibrated critic init was the only root cause and do not "
                     "support deployment.")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND71 evaluation adapter")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--phase", choices=("plan", "eval", "stats"), required=True)
    args = ap.parse_args(argv)
    raw_root = Path(args.raw_root).resolve()
    if args.phase == "plan":
        plan = build_eval_plan()
        _write_json(raw_root / "ROUND71_EVAL_PLAN.json", plan)
        print(json.dumps({k: v for k, v in plan.items() if k != "jobs"}, indent=1))
        return 0
    cfg = configure_r71_trial_eval(raw_root)
    _write_json(raw_root / "runtime" / "ROUND71_EVAL_QUOTA_GUARD.json",
                cfg["r71_eval_quota_guard"])
    if args.phase == "eval":
        plan = _read_json(raw_root / "ROUND71_EVAL_PLAN.json")
        VISITED.clear()
        print(json.dumps(eval_model_arms(cfg, plan), indent=1), flush=True)
        print(json.dumps(eval_focal(cfg, plan), indent=1), flush=True)
        order = _assert_visited_order(plan)
        _write_json(raw_root / "runtime" / "ROUND71_EVAL_ORDER_CHECK.json", order)
        print(json.dumps(order, indent=1), flush=True)
        _write_json(raw_root / "runtime" / "ROUND71_EVAL_QUOTA_GUARD.json", quota_guard(cfg))
        return 0
    stats = compute_stats(raw_root)
    _write_json(raw_root / "ROUND71_STATISTICS.json", stats)
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("per_seed_mode", "focal")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())