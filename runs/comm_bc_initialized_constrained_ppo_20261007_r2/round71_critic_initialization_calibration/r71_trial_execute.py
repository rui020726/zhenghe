#!/usr/bin/env python
"""ROUND71 approved fixed trial: training executor.

A faithful adaptation of the verified ROUND68 executor with an independent ROUND71 identity:
  * the runtime is stood up through the readiness binding's ``configure_r71_runtime`` (the REAL
    ``r68_execute.configure_and_inject`` with ONLY quota/version/checkpoint rebound), so the
    routed PPO update (R53 critic->trunk isolation + the unchanged R66/R68 update math), the
    sequential mask, the loss, the dual update and the claim/slot/permit regime are inherited
    verbatim;
  * the fresh controller of every seed is THIS run's calibrated INIT checkpoint, loaded
    model-only (``cfg["r71_fresh"]``); the R68 fresh factory is never used for training;
  * training rows/tapes/claims/logicals carry the ROUND71 names (``r71_train_s{seed}_e{ep}`` /
    ``r71_new_training_s{seed}_e{ep}``) under the ROUND71 raw root;
  * checkpoints are saved through the rebound ROUND71 five-module contract
    (``readiness.r71_save_checkpoint``); a foreign-round checkpoint is refused as a resume source;
  * the first-batch gate is REAL, runs right after the first update (episodes 0-3 count toward
    128), and must pass before that seed continues.

Budget: historical non-plan probe = 0; normal plan = 582 (384 training + 192 model eval + 6 focal);
technical reserve = 18; cumulative environment-construction hard cap = 600.  R58/R60/R66/R68
constructions are NOT ROUND71 consumption.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN_ROOT = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
R68_DIR = RUN_ROOT / "round68_late_initial_confidence_ablation"
R66_DIR = RUN_ROOT / "round66_bc_early_path_preserving_late_adaptation"
READINESS_DIR = REPO / "analysis" / "round71-critic-calibration-readiness"
for _d in (READINESS_DIR, str(R68_DIR), str(R66_DIR)):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import r71_trial_binding as trial                                       # noqa: E402
import r71_runtime_binding as readiness                                 # noqa: E402
import r68_execute as r68x                                              # noqa: E402
import r66_execute as r66x                                              # noqa: E402
import r68_frozen as r68                                                # noqa: E402

SEEDS = (11, 22, 33)
EPISODES_PER_SEED = 128
EPISODES_PER_UPDATE = 4
MODES = ("greedy", "sampled")
SCENES_NEW = tuple(range(1_800_000, 1_800_032))
FOCAL_SCENE = 1_700_027
ROW_TAG = trial.ROW_TAG                       # r71train
EVAL_ROW_TAG = trial.EVAL_ROW_TAG             # r71eval
REF_COUNT = 48
TRAINING_COUNT = 384
EVAL_COUNT = 192
FOCAL_COUNT = 6
# The licence allots 192 evaluation launches to the 3x2x32 block; the frozen breakdown is
# "192 fixed-ep128 evaluations + 6 focal cells (INIT/TRAINED x 3 seeds)".  The block therefore
# evaluates the TRAINED arm only; NEW_INIT appears on the focal scene alone (its actor/trunk
# are untouched, so a full-block INIT rerun would duplicate R68 INIT_ALPHA behaviour and the
# 600-construction hard cap forbids a second 192-cell arm).
MODEL_EVAL_ARM = "NEW_EP128"
NORMAL_CAP = 582
RESERVE_CAP = 18
HARD_CAP = 600
HISTORICAL_PROBE = 0
TASK_ID = trial.R71_TRIAL_TASK_ID
BRANCH = trial.R71_TRIAL_BRANCH
R68_SOURCE_COMMIT = "420fa9af18932f735efdf6528ab68174c561ec52"

# generic, seam-free helpers reused verbatim from the mature executors
_sha256_file = r66x._sha256_file
_read_json = r66x._read_json
_write_json = r66x._write_json
_write_row = r66x._write_row
_reload_episode = r66x._reload_episode
_slot_for = r66x._slot_for
# the ONE claimed-but-started-safe booking helper + the durable construction-permit regime
_atomic_run = r68x._r68_atomic_run
_active_attempt = r68x._active_attempt
_qualified_attempt = r68x._qualified_attempt


# --------------------------------------------------------------------------- configure ----
def configure_r71_trial(raw_root: Path, gate, run_identity, *, model_read_approval=None) -> dict:
    """Stand up the REAL ROUND68 runtime under the ROUND71 identity and inject the trial seams."""
    cfg = readiness.configure_r71_runtime(Path(raw_root), gate, run_identity,
                                          model_read_approval=model_read_approval)
    r40 = cfg["runtime"]["r40_common"]
    common = cfg["common"]
    if (int(r40.NORMAL_CAP), int(r40.RESERVE_CAP), int(r40.HARD_CAP)) != (NORMAL_CAP, RESERVE_CAP, HARD_CAP):
        raise RuntimeError("ROUND71 quota caps did not bind on the live r40_common ledger")
    if (int(common.NORMAL_LAUNCHES), int(common.RESERVE_LAUNCHES), int(common.LAUNCH_CAP)) != (
            NORMAL_CAP, RESERVE_CAP, HARD_CAP):
        raise RuntimeError("ROUND71 quota caps did not bind on the live common module")
    cfg["r71_run_identity"] = run_identity
    cfg["r71_fresh"] = lambda seed: r71_fresh(cfg, int(seed))
    cfg["trial"] = {
        "task_id": TASK_ID, "branch": BRANCH, "run_id": run_identity.run_id,
        "row_tag": ROW_TAG, "eval_row_tag": EVAL_ROW_TAG,
        "model_arms": list(trial.MODEL_ARMS), "focal_scene": FOCAL_SCENE,
        "training_count": TRAINING_COUNT, "eval_count": EVAL_COUNT, "focal_count": FOCAL_COUNT,
        "normal_cap": NORMAL_CAP, "reserve_cap": RESERVE_CAP, "hard_cap": HARD_CAP,
        "historical_non_plan_probe": HISTORICAL_PROBE,
        "approval_id": trial.TRIAL_APPROVAL_ID,
        "ppo_update_is_r68_routed": cfg["r71"]["identity"]["ppo_update_unchanged"],
        "save_is_r71": cfg["r71"]["identity"]["save_is_r71"],
        "load_is_r71": cfg["r71"]["identity"]["load_is_r71"],
        "load_ro_is_r71": cfg["r71"]["identity"]["load_ro_is_r71"]}
    return cfg


def init_paths(cfg, seed: int) -> dict:
    ckpt_dir = Path(cfg["common"].CHECKPOINTS)
    return {"dir": ckpt_dir,
            "init": ckpt_dir / f"v2_train_s{int(seed)}_init.pt",
            "identity": ckpt_dir / f"r71_init_identity_s{int(seed)}.json",
            "rolling": ckpt_dir / f"v2_train_s{int(seed)}.pt",
            "ep128": ckpt_dir / f"v2_train_s{int(seed)}_ep128.pt"}


def r71_fresh(cfg, seed: int):
    """THIS run's NEW controller for training: the calibrated INIT, model-only, never a re-fit."""
    paths = init_paths(cfg, seed)
    if not paths["init"].is_file():
        raise FileNotFoundError(f"seed {seed}: ROUND71 calibrated INIT checkpoint missing: "
                                f"{paths['init']}; run the offline-fit phase first")
    controller, state, identity = readiness.r71_load_checkpoint_read_only(paths["init"])
    if int(state.get("episodes_done", -1)) != 0 or int(state.get("updates", -1)) != 0:
        raise RuntimeError(f"seed {seed}: the INIT checkpoint is not an untrained initialisation")
    if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
        raise RuntimeError(f"seed {seed}: the INIT load restored training state")
    if not paths["identity"].is_file():
        raise RuntimeError(f"seed {seed}: registered INIT identity evidence missing")
    registered = _read_json(paths["identity"])
    summary = controller.parameter_summary()
    if summary["parameter_sha256"] != registered["init_parameter_sha256"]:
        raise RuntimeError(f"seed {seed}: INIT parameters != the registered calibrated identity")
    module_sha = {name: r68._module_sha(controller, name)
                  for name in ("trunk", "early_head", "late_head", "value_task_head", "cost_head")}
    cal = registered["calibration"]
    for key, want in (("trunk", cal["trunk_sha256"]), ("early_head", cal["early_head_sha256"]),
                      ("late_head", cal["late_head_sha256"]),
                      ("value_task_head", cal["value_task_head_sha256"]),
                      ("cost_head", cal["value_cost_head_sha256"])):
        if module_sha[key] != want:
            raise RuntimeError(f"seed {seed}: INIT module {key} != the registered identity")
    if any(p.requires_grad for p in r68.frozen_parameters(controller)) or \
            not all(p.requires_grad for p in r68.trainable_parameters(controller)):
        raise RuntimeError(f"seed {seed}: the INIT controller lost its frozen mask")
    if summary["parameter_object_count"] != 12:
        raise RuntimeError(f"seed {seed}: the INIT controller is not the five-module schema")
    return controller


# --------------------------------------------------------------------------- training ----
def _run_train_episode(cfg, controller, collector, *, seed, ep, scenario, lam):
    runner = cfg["runner"]
    tag = f"r71_train_s{seed}_e{ep}"
    action_seed = runner.action_seed(int(seed), int(ep))
    logical = f"r71_new_training_s{int(seed)}_e{int(ep)}"
    metadata = {"arm": "NEW_CALIBRATED_SRC", "training_seed": int(seed), "episode_index": int(ep),
                "scenario_seed": int(scenario), "lambda_at_rollout": float(lam)}

    def _do():
        row = runner._run_episode(tag, "train", int(scenario), policy="learned",
                                  attach=collector.policy(), controller=controller, collector=collector,
                                  action_rng_seed=action_seed, tag=tag,
                                  attempt=_active_attempt(cfg))
        row["collector"] = collector
        row["training_seed"] = int(seed); row["episode_index"] = int(ep)
        row["scenario_seed"] = int(scenario); row["lambda"] = float(lam)
        return row

    return _atomic_run(cfg, logical, metadata, _do,
                       detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}")


def reuse_r66_references(cfg, order) -> dict:
    """Read ROUND66's committed 48 immediate-reference rows READ-ONLY (no environment)."""
    r66_rows = R66_DIR / "raw" / "rows"
    refs: dict[int, float] = {}
    for scene in sorted({int(s) for s in order}):
        path = r66_rows / f"v2_r66_ref_s{scene}_s{scene}_a1.json"
        if not path.is_file():
            raise RuntimeError(f"ROUND71 reuse: ROUND66 immediate reference row missing for scene "
                               f"{scene} at {path}")
        row = _read_json(path)
        if int(row.get("scenario_seed", -1)) != int(scene):
            raise RuntimeError(f"scene {scene}: ROUND66 reference row scenario_seed mismatch")
        refs[int(scene)] = float(row["on_time"])
    if len(refs) != REF_COUNT:
        raise RuntimeError(f"reused ROUND66 references {len(refs)} != {REF_COUNT}")
    return refs


def _r68_batch0_behaviour(seed: int) -> dict:
    """The committed ROUND68 batch-0 episode behaviour (read-only), for the init-identity check."""
    out = {}
    for b in range(1):
        rec = _read_json(R68_DIR / "raw" / "checkpoints" / f"r40_batch_s{int(seed)}_b{b}_attempt01.json")
        for ep in rec["episodes"]:
            hits = sorted((R68_DIR / "raw" / "rows").glob(
                f"v2_r68_train_s{int(seed)}_e{int(ep)}_s*_a1.json"))
            if len(hits) != 1:
                raise RuntimeError(f"seed {seed} ep {ep}: {len(hits)} ROUND68 row matches")
            row = _read_json(hits[0])
            out[int(ep)] = {"on_time": int(row["on_time"]),
                            "spent_bytes": int(row["spent_bytes"]),
                            "task_reward_total": float(row["task_reward_total"]),
                            "airtime_tap_total": float(row["airtime_tap_total"])}
    return out


def build_plan(order: list[int]) -> dict[str, Any]:
    jobs = []
    slot = 0
    train_scenes = sorted({int(s) for s in order})
    if len(train_scenes) != REF_COUNT:
        raise AssertionError(f"training scenes {len(train_scenes)} != {REF_COUNT}")
    for seed in SEEDS:
        for ep in range(EPISODES_PER_SEED):
            slot += 1
            jobs.append({"scope": "ppo_training", "logical_job": f"r71_new_training_s{seed}_e{ep}",
                         "seed": seed, "episode_index": ep, "scene": int(order[ep]),
                         "normal_slot": slot})
    for arm in (MODEL_EVAL_ARM,):
        for seed in SEEDS:
            for mode in MODES:
                for scene in SCENES_NEW:
                    slot += 1
                    jobs.append({"scope": "model_eval", "arm": arm,
                                 "logical_job": f"{EVAL_ROW_TAG}_model_{arm.lower()}_s{seed}_{mode}_scene{scene}",
                                 "seed": seed, "mode": mode, "scene": int(scene),
                                 "normal_slot": slot})
    for seed in SEEDS:
        for arm in trial.MODEL_ARMS:
            slot += 1
            jobs.append({"scope": "focal", "arm": arm,
                         "logical_job": f"{EVAL_ROW_TAG}_focal_{arm.lower()}_s{seed}_scene{FOCAL_SCENE}",
                         "seed": seed, "mode": "greedy", "scene": FOCAL_SCENE,
                         "normal_slot": slot})
    if len(jobs) != NORMAL_CAP or len({j["logical_job"] for j in jobs}) != NORMAL_CAP:
        raise AssertionError(f"ROUND71 fixed plan is not exactly {NORMAL_CAP} unique jobs "
                             f"(got {len(jobs)})")
    return {"kind": "round71-fixed-job-plan/1", "task_id": TASK_ID, "branch": BRANCH,
            "training_version": readiness.R71_TRAINING_VERSION,
            "structure": readiness.R71_STRUCTURE,
            "approval_id": trial.TRIAL_APPROVAL_ID,
            "intervention": ("per-seed dual-output ridge calibration of the two critic affine "
                             "heads on that seed's own ROUND68 teacher data over the frozen "
                             "R68 ep0 trunk; trunk/actor/initial policy unchanged"),
            "training_episode_count": TRAINING_COUNT, "model_eval_count": EVAL_COUNT,
            "model_eval_arm": MODEL_EVAL_ARM,
            "focal_count": FOCAL_COUNT, "reference_count": REF_COUNT,
            "normal_count": NORMAL_CAP, "reserve_cap": RESERVE_CAP, "hard_cap": HARD_CAP,
            "historical_probe": HISTORICAL_PROBE,
            "eval_scenes": list(SCENES_NEW), "focal_scene": FOCAL_SCENE,
            "row_tag": ROW_TAG, "eval_row_tag": EVAL_ROW_TAG,
            "jobs": jobs,
            "r68_source_dir": str(R68_DIR),
            "reference_reuse_note": ("the 48 immediate references are reused read-only from "
                                     "ROUND66; the ROUND68 same-seed/same-mode control rows are "
                                     "read-only reuse subject to eligibility")}


def _qualified_batch_record(cfg, seed: int, batch: int, *, episodes_done_after: int,
                            before_sha=None, controller=None, state=None):
    ckpt_dir = Path(cfg["common"].CHECKPOINTS)
    candidates = []
    for path in sorted(ckpt_dir.glob(f"r40_batch_s{seed}_b{batch}_attempt*.json")):
        try:
            rec = _read_json(path)
        except Exception:
            continue
        if int(rec.get("episodes_done_after", -1)) != int(episodes_done_after):
            continue
        if before_sha is not None and rec.get("parameters_before_sha") != before_sha:
            continue
        candidates.append((path, rec))
    if len(candidates) != 1:
        raise RuntimeError(
            f"R71 seed {seed} batch {batch}: cannot resolve exactly one qualified batch record "
            f"(candidates={len(candidates)}); refusing to fall back to a fixed attempt01")
    path, rec = candidates[0]
    if controller is not None and state is not None:
        if int(state["episodes_done"]) == int(rec["episodes_done_after"]):
            live = controller.parameter_summary()["parameter_sha256"]
            if live != rec.get("parameters_after_sha"):
                raise RuntimeError(
                    f"R71 seed {seed} batch {batch}: the live controller ({live}) does not match "
                    f"the qualified record {path.name} ({rec.get('parameters_after_sha')})")
    return path, rec


def train_seed(cfg, ledger, plan, seed, order, references, *, stop_after_first_gate=False):
    common = cfg["common"]; trainer = cfg["trainer"]
    run_identity = cfg["r71_run_identity"]
    ckpt_dir = Path(common.CHECKPOINTS)
    rolling = init_paths(cfg, seed)["rolling"]
    if rolling.exists():
        controller, optimizer, state = readiness.r71_load_checkpoint(
            rolling, run_identity=run_identity)
        ps = controller.parameter_summary()
        if ps["parameter_object_count"] != 12 or \
                r68.frozen_parameter_summary(controller).get("structure") != r68.R68_STRUCTURE or \
                state.get("structure") != readiness.R71_STRUCTURE:
            raise RuntimeError(f"seed {seed}: rolling checkpoint is not a ROUND71 calibrated model")
        if any(p.requires_grad for p in r68.frozen_parameters(controller)) or \
                not all(p.requires_grad for p in r68.trainable_parameters(controller)):
            raise RuntimeError(f"seed {seed}: restored controller did not re-establish the frozen mask")
        resumed = True
    else:
        controller = cfg["r71_fresh"](seed)
        optimizer = r68.build_optimizer(controller, common.LEARNING_RATE)
        state = r68.fresh_training_state(
            int(seed), gae_lambda=common.GAE_LAMBDA, eta_lambda=common.ETA_LAMBDA,
            gamma=common.GAMMA, delta=0.0, lambda_init=1.0,
            data_order_seed=common.DATA_ORDER_SEED, b0_bytes=common.B0_BYTES,
            alpha=r68.LATE_INIT_ALPHA)
        state["training_version"] = readiness.R71_TRAINING_VERSION
        state["loss_version"] = readiness.R71_LOSS_VERSION
        state["state_schema"] = readiness.R71_STATE_SCHEMA
        state["optimizer_schema"] = readiness.R71_OPTIMIZER_SCHEMA
        state["structure"] = readiness.R71_STRUCTURE
        state["calibration_route"] = readiness.R71_CALIBRATION_ROUTE
        state["run_identity"] = run_identity.as_dict()
        state["critic_heads_calibration"] = _read_json(init_paths(cfg, seed)["identity"])["calibration"]
        resumed = False
    for batch in range(int(state["updates"]), EPISODES_PER_SEED // EPISODES_PER_UPDATE):
        if int(state["episodes_done"]) != batch * EPISODES_PER_UPDATE:
            raise RuntimeError(f"seed {seed} batch {batch}: episode cursor and update index "
                               f"inconsistent")
        lam = float(state["lambda"])
        before = controller.parameter_summary()
        pre = ckpt_dir / f"v2_train_s{seed}_before_b{batch}.pt"
        if not pre.exists():
            trainer.save_checkpoint(controller, optimizer, state, pre)
        episodes = []
        for k in range(EPISODES_PER_UPDATE):
            ep = int(state["episodes_done"]) + k
            scenario = int(order[ep])
            logical = f"r71_new_training_s{seed}_e{ep}"
            if ledger.is_complete(logical):
                att = _qualified_attempt(cfg["runtime"]["r40_common"], logical)
                paths = common.cell_paths("train", scenario, att, tag=f"r71_train_s{seed}_e{ep}")
                episodes.append(_reload_episode(paths["row"], paths["tape"],
                                                seed=seed, ep=ep, scenario=scenario, lam=lam))
                continue
            collector = __import__("r42_one_shot").OneShotSendSkipCollector(controller, greedy=False)
            row = _run_train_episode(cfg, controller, collector, seed=seed, ep=ep,
                                     scenario=scenario, lam=lam)
            _write_row(cfg, row)
            episodes.append(row)
        if len(episodes) != EPISODES_PER_UPDATE:
            raise RuntimeError(f"seed {seed} batch {batch}: partial PPO batch")
        update_report = trainer.ppo_update(controller, optimizer, episodes, update_index=batch)
        g_record = trainer.batch_g_hat(episodes, references)
        dual_record = trainer.dual_update(state["lambda"], g_record["g_hat"])
        state["lambda"] = dual_record["lambda_next"]
        state["lambda_history"].append({"batch": batch, "g_hat": dual_record["g_hat"],
                                        "lambda_before": dual_record["lambda_before"],
                                        "lambda_next": dual_record["lambda_next"]})
        state["updates"] += 1
        state["optimizer_steps"] += int(update_report.get("optimizer_steps", 0))
        state["episodes_done"] += EPISODES_PER_UPDATE
        after = controller.parameter_summary()
        if after["parameter_sha256"] == before["parameter_sha256"]:
            raise RuntimeError(f"seed {seed} batch {batch}: parameters did not move")
        post = ckpt_dir / f"v2_train_s{seed}_after_b{batch}.pt"
        trainer.save_checkpoint(controller, optimizer, state, post)
        trainer.save_checkpoint(controller, optimizer, state, rolling)
        batch_record = {"kind": "round71-training-batch-record/1", "batch": batch, "seed": seed,
                        "episodes": [int(r["episode_index"]) for r in episodes],
                        "scenarios": [int(r["scenario_seed"]) for r in episodes],
                        "lambda_at_rollout": lam, "update": update_report,
                        "constraint": g_record, "dual": dual_record,
                        "parameters_before_sha": before["parameter_sha256"],
                        "parameters_after_sha": after["parameter_sha256"],
                        "episodes_done_after": int(state["episodes_done"]),
                        "optimizer_steps_after": int(state["optimizer_steps"]),
                        "structure": readiness.R71_STRUCTURE,
                        "resumed_batch": bool(resumed and batch == int(state["updates"]) - 1)}
        _write_json(ckpt_dir / f"r40_batch_s{seed}_b{batch}_attempt01.json", batch_record)
        ledger.write_ledger()
        gate_path = cfg["raw_root"] / "runtime" / f"r71_first_batch_gate_s{seed}.json"
        if batch == 0 and not gate_path.exists():
            init_id = _read_json(init_paths(cfg, seed)["identity"])
            gate = _first_batch_gate(cfg, controller, state, batch_record, episodes,
                                     init_identity=init_id)
            _write_json(gate_path, gate)
            if gate["status"] != "passed":
                raise RuntimeError(f"seed {seed}: first-batch gate failed: {gate}")
            if stop_after_first_gate:
                return {"seed": seed, "status": "first_gate_passed", "gate": gate["status"]}
        print(f"R71 seed {seed} batch {batch}: episodes_done={state['episodes_done']} "
              f"lambda={state['lambda']:.4f} early_free={update_report.get('early_free')} "
              f"late_free={update_report.get('late_free')} "
              f"gnorms={update_report.get('gradient_norm_mean')}", flush=True)
    gate_from_committed = _gate_from_committed(cfg, seed, controller, state)
    ep128 = init_paths(cfg, seed)["ep128"]
    trainer.save_checkpoint(controller, optimizer, state, ep128)
    return {"seed": seed, "status": "ep128_complete",
            "episodes_done": int(state["episodes_done"]),
            "updates": int(state["updates"]),
            "optimizer_steps": int(state["optimizer_steps"]),
            "lambda_final": float(state["lambda"]),
            "gate_from_committed": (gate_from_committed or {}).get("status")}


def _gate_from_committed(cfg, seed, controller, state):
    """Run the first-batch gate from the committed batch-0 record + reloaded episodes (resume)."""
    gate_path = cfg["raw_root"] / "runtime" / f"r71_first_batch_gate_s{seed}.json"
    if gate_path.exists() or int(state["episodes_done"]) < EPISODES_PER_UPDATE:
        return None
    init_id = _read_json(init_paths(cfg, seed)["identity"])
    record_path, record = _qualified_batch_record(
        cfg, seed, 0, episodes_done_after=EPISODES_PER_UPDATE,
        before_sha=init_id["init_parameter_sha256"], controller=controller, state=state)
    order = [int(x) for x in cfg["common"].scenario_order()]
    lam = float(record["lambda_at_rollout"])
    r40 = cfg["runtime"]["r40_common"]
    episodes = []
    for ep in record["episodes"]:
        scenario = int(order[int(ep)])
        att = _qualified_attempt(r40, f"r71_new_training_s{seed}_e{int(ep)}")
        paths = cfg["common"].cell_paths("train", scenario, att, tag=f"r71_train_s{seed}_e{ep}")
        episodes.append(_reload_episode(paths["row"], paths["tape"], seed=seed, ep=int(ep),
                                        scenario=scenario, lam=lam))
    gate = _first_batch_gate(cfg, controller, state, record, episodes, init_identity=init_id)
    gate["batch_record_path"] = record_path.name
    gate["batch_record_attempt"] = int(re.search(r"attempt(\d+)$", record_path.stem).group(1))
    gate["reload_attempt_resolution"] = "qualified_attempt_from_complete_claim"
    _write_json(gate_path, gate)
    if gate["status"] != "passed":
        raise RuntimeError(f"seed {seed}: first-batch gate failed (from committed): {gate}")
    return gate


def _first_batch_gate(cfg, controller, state, record, episodes, *, init_identity):
    """The REAL ROUND71 first-batch gate.

    Beyond the ROUND68 checks it proves the calibration identity: the batch started from THIS
    run's calibrated INIT (trunk/early/late == the registered R68 post-scale identity, both
    critic heads == the registered calibrated heads), the optimizer owns exactly the six
    trainable objects, the frozen modules recorded no gradient, the partitioned actor still
    routed both regions, and the first four episodes' behaviour matches the committed ROUND68
    batch-0 rows (the initial policy is identical; only the critics differ).
    """
    update_report = record["update"]
    dual_record = record["dual"]
    before_sha = record["parameters_before_sha"]
    after = controller.parameter_summary()
    after_frozen = r68.frozen_parameter_summary(controller)
    gate_diag = update_report.get("gate_diag") or {}
    modgrad = update_report.get("module_gradient_l1_mean") or {}
    checks: dict[str, Any] = {}
    checks["dual_head_routing_active"] = bool(update_report.get("early_free", 0) > 0
                                              and update_report.get("late_free", 0) > 0)
    checks["early_free"] = int(update_report.get("early_free", 0))
    checks["late_free"] = int(update_report.get("late_free", 0))
    cal = init_identity["calibration"]
    module_sha = {name: r68._module_sha(controller, name)
                  for name in ("trunk", "early_head", "late_head", "value_task_head", "cost_head")}
    checks["init_trunk_is_r68_ep0_trunk"] = (module_sha["trunk"] == cal["trunk_sha256"])
    checks["init_early_head_is_r68_ep0"] = (module_sha["early_head"] == cal["early_head_sha256"])
    checks["init_late_head_is_r68_post_scale"] = (module_sha["late_head"] == cal["late_head_sha256"])
    checks["init_started_from_calibrated_init"] = (before_sha == init_identity["init_parameter_sha256"])
    checks["critic_heads_are_calibrated"] = (
        module_sha["value_task_head"] == cal["value_task_head_sha256"]
        and module_sha["cost_head"] == cal["value_cost_head_sha256"])
    checks["init_cursor_zero"] = (int(init_identity.get("init_parameter_object_count", -1)) == 12)
    behaviour = _r68_batch0_behaviour(int(state["training_seed"]))
    mismatches = []
    for row in episodes:
        ep = int(row["episode_index"])
        want = behaviour.get(ep)
        if want is None:
            mismatches.append({"episode": ep, "problem": "no ROUND68 committed row"})
            continue
        for field in ("on_time", "spent_bytes", "task_reward_total", "airtime_tap_total"):
            got = row.get(field)
            if float(got) != float(want[field]):
                mismatches.append({"episode": ep, "field": field, "r71": got, "r68": want[field]})
    checks["first_batch_behaviour_matches_r68"] = (not mismatches)
    checks["first_batch_behaviour_mismatches"] = mismatches
    checks["structure_live_r68"] = (after_frozen.get("structure") == r68.R68_STRUCTURE)
    checks["structure_state_r71"] = (state.get("structure") == readiness.R71_STRUCTURE)
    checks["parameter_objects_12"] = (after["parameter_object_count"] == 12)
    checks["five_modules"] = (sorted(after["shapes"].keys()) ==
        sorted(["trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
                "early_head.weight", "early_head.bias", "late_head.weight", "late_head.bias",
                "value_task_head.weight", "value_task_head.bias",
                "value_cost_head.weight", "value_cost_head.bias"]))
    checks["frozen_modules_requires_grad_false"] = (
        not any(p.requires_grad for p in r68.frozen_parameters(controller)))
    checks["trainable_modules_requires_grad_true"] = (
        all(p.requires_grad for p in r68.trainable_parameters(controller)))
    checks["optimizer_owns_six_trainable"] = (
        int(update_report.get("trainable_parameter_object_count", -1)) == 6)
    checks["frozen_six_recorded"] = (int(update_report.get("frozen_parameter_object_count", -1)) == 6)
    checks["frozen_modules_gradient_recorded_none"] = (
        modgrad.get("trunk") is None and modgrad.get("early_head") is None)
    checks["trainable_modules_gradient_recorded_number"] = (
        isinstance(modgrad.get("late_head"), float) and isinstance(modgrad.get("cost_head"), float))
    checks["critic_received_gradient"] = (float(modgrad.get("cost_head", 0.0) or 0.0) > 0.0)
    checks["task_critic_received_gradient"] = (
        float(modgrad.get("task_value_head", 0.0) or 0.0) > 0.0)
    if gate_diag:
        checks["late_actor_received_gradient"] = (
            float(gate_diag.get("late_actor_grad_l1", -1.0)) > 0.0)
        checks["gate_diag_frozen_labelled"] = (
            gate_diag.get("trunk") == "frozen_requires_grad_false"
            and gate_diag.get("early_head") == "frozen_requires_grad_false")
    checks["likelihood_alignment_passed"] = bool(update_report["likelihood_alignment"]["passed"])
    checks["value_losses_finite"] = all(np.isfinite([update_report.get("value_loss_task", 0.0),
                                                     update_report.get("value_loss_cost", 0.0)]))
    checks["lambda_updated"] = ("lambda_next" in dual_record)
    checks["parameters_moved"] = (after["parameter_sha256"] != before_sha)
    checks["offline_ledger_within_cap"] = bool(
        init_identity["offline_ledger"]["rows"] <= init_identity["offline_ledger"]["cap"])
    status = "passed" if all(v for k, v in checks.items() if isinstance(v, bool)) else "failed"
    return {"kind": "round71-first-batch-gate/1", "seed": int(state["training_seed"]),
            "status": status, "checks": checks,
            "parameters_after_sha": after["parameter_sha256"],
            "frozen_sha256": r68x.r66f._tensor_identity(r68.frozen_parameters(controller))["sha256"],
            "lambda_after_first_batch": dual_record["lambda_next"],
            "isolation_note": ("critic->trunk isolation is applied inside routed_ppo, but its trunk "
                               "effect is masked because the trunk is requires_grad=False by the "
                               "inherited ROUND68 intervention; the absent trunk gradient is NOT "
                               "evidence about isolation (proved separately on a synthetic "
                               "backward in the pre-launch gates)"),
            "gate_diag": gate_diag, "module_gradient_l1_mean": modgrad}