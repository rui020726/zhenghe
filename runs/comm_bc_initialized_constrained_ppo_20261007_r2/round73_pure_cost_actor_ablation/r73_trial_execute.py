#!/usr/bin/env python
"""ROUND73 approved fixed trial: training executor (pure-cost actor objective).

A faithful adaptation of the verified ROUND71 executor with an independent ROUND73 identity:

  * the runtime is stood up through the readiness binding's ``configure_r71_runtime`` (the REAL
    ``r68_execute.configure_and_inject`` with quota/version/checkpoint rebound), so the routed
    PPO update (R53 critic->trunk isolation + the unchanged R66/R68 update math), the sequential
    mask, the loss, the dual update and the claim/slot/permit regime are inherited verbatim;
  * the ONE structural intervention is installed on the live trainer:
    ``install_pure_cost_objective`` scopes ``cost_math.actor_advantage`` to ``-A_cost`` for the
    duration of every ``batch_samples`` call, so the actor surrogate (and only it) sees
    ``A = -A_cost``; the dual critics, GAE, reward timing, mask, entropy, clip and the
    dual-lambda update are untouched;
  * every seed starts from THIS run's staged INIT: the ROUND71 calibrated INIT file copied
    byte-exact and loaded model-only (five modules; fresh optimizer / lambda / training RNG);
    no offline fit is run and no ROUND71 trained weights, Adam state or lambda are inherited;
  * training rows/tapes/claims/logicals carry the ROUND73 names under the ROUND73 raw root;
  * checkpoints are saved through the ROUND73 five-module contract (``r73_checkpoint``); a
    ROUND60/66/68/71 checkpoint is refused as a ROUND73 resume source;
  * the first-batch gate is REAL, runs right after the first update (episodes 0-3 count toward
    128), and must pass before that seed continues.

Budget: historical non-plan probe = 0; normal plan = 579 (384 training + 192 model eval + 3
focal); technical reserve = 21; cumulative environment-construction hard cap = 600.  R58/R60/
R66/R68/R71 constructions are NOT ROUND73 consumption.
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
R71_DIR = RUN_ROOT / "round71_critic_initialization_calibration"
READINESS_DIR = REPO / "analysis" / "round71-critic-calibration-readiness"
for _d in (READINESS_DIR, str(R68_DIR), str(R66_DIR), str(HERE)):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import r73_trial_binding as trial                                       # noqa: E402
import r73_checkpoint as ckpt                                           # noqa: E402
import r73_pure_cost_objective as r73                                   # noqa: E402
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
ROW_TAG = trial.ROW_TAG                       # r73train
EVAL_ROW_TAG = trial.EVAL_ROW_TAG             # r73eval
REF_COUNT = 48
TRAINING_COUNT = 384
EVAL_COUNT = 192
FOCAL_COUNT = 3
MODEL_ARMS = trial.MODEL_ARMS                 # ("NEW_EP128",)
MODEL_EVAL_ARM = "NEW_EP128"
NORMAL_CAP = 579
RESERVE_CAP = 21
HARD_CAP = 600
HISTORICAL_PROBE = 0
TASK_ID = trial.R73_TRIAL_TASK_ID
BRANCH = trial.R73_TRIAL_BRANCH
R71_INDEX_PATH = R71_DIR / "ROUND71_CHECKPOINT_INDEX.json"
# the byte-frozen canonical trainer copy the zero-environment gates exercise; the LIVE trainer
# the runtime stands up must be the same bytes, so a gate result transfers to the real run
FROZEN_SOURCES = (RUN_ROOT / "round51_frozen_source_gradient_close" / "frozen_sources"
                  / "canonical_v2_trainer.py")
COST_MATH_SOURCE = REPO / "runs" / "comm_cost_controller_stage_a_20260927" / "cost_math.py"
R71_CHECKPOINTS = R71_DIR / "raw" / "checkpoints"
R71_ROWS = R71_DIR / "raw" / "rows"

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
def configure_r73_trial(raw_root: Path, gate, run_identity, *, model_read_approval=None) -> dict:
    """Stand up the REAL ROUND71 runtime under the ROUND73 identity and inject the ONE intervention."""
    cfg = readiness.configure_r71_runtime(Path(raw_root), gate, run_identity,
                                          model_read_approval=model_read_approval)
    r40 = cfg["runtime"]["r40_common"]
    common = cfg["common"]
    trainer = cfg["trainer"]
    # quota: the ROUND71 configure binds the ROUND71 caps first; rebind them to ROUND73 here
    if (int(r40.NORMAL_CAP), int(r40.RESERVE_CAP), int(r40.HARD_CAP)) != (
            readiness.R71_NORMAL_CAP, readiness.R71_RESERVE_CAP, readiness.R71_HARD_CAP):
        raise RuntimeError("the ROUND71 runtime did not bind its own caps before the ROUND73 rebind")
    r40.NORMAL_CAP = NORMAL_CAP
    r40.RESERVE_CAP = RESERVE_CAP
    r40.HARD_CAP = HARD_CAP
    common.NORMAL_LAUNCHES = NORMAL_CAP
    common.RESERVE_LAUNCHES = RESERVE_CAP
    common.LAUNCH_CAP = HARD_CAP
    r40.plan = lambda: {"normal_total": NORMAL_CAP, "technical_reserve_max": RESERVE_CAP,
                        "hard_cap": HARD_CAP, "task_id": TASK_ID}
    r40.ROUND_DIR = Path(raw_root).resolve()
    # version: the ROUND73 training-version literal on the live versions module
    try:
        cfg["runtime"]["r42_versions"].TRAINING_VERSION = ckpt.R73_TRAINING_VERSION
    except Exception as exc:                     # pragma: no cover - identity must never drift
        raise readiness.SourceIdentityError(
            f"could not rebind the live training version: {exc}") from exc
    # checkpoint: the ROUND73 five-module contract (foreign resumes still refused)
    def r73_save(controller, optimizer, state, path, *, freeze=None):
        return ckpt.r73_save_checkpoint(controller, optimizer, state, path)

    def r73_load(path, *, freeze=None):
        return ckpt.r73_load_checkpoint(path, run_identity=cfg["r73_run_identity"])

    def r73_load_ro(path, *, freeze=None):
        return ckpt.r73_load_checkpoint_read_only(path)

    ppo_update_before = trainer.ppo_update
    trainer.save_checkpoint = r73_save
    trainer.load_checkpoint = r73_load
    trainer.load_checkpoint_read_only = r73_load_ro
    if trainer.ppo_update is not ppo_update_before:
        raise readiness.SourceIdentityError(
            "the routed PPO update must stay the verified ROUND68 update (critic->trunk "
            "isolation + frozen-early-path math); ROUND73 never rebinds it")
    # the ONE structural intervention, on the live trainer and its own cost_math
    if getattr(trainer, "R73_ACTOR_OBJECTIVE", None) != ckpt.R73_ACTOR_OBJECTIVE:
        install = r73.install_pure_cost_objective(trainer, cm=trainer.cm, freeze=ckpt.R73_FREEZE)
    else:
        install = {"actor_objective": ckpt.R73_ACTOR_OBJECTIVE,
                   "objective_version": ckpt.R73_OBJECTIVE_VERSION,
                   "training_version": ckpt.R73_TRAINING_VERSION,
                   "patched": "trainer.batch_samples (already installed in this process)"}
    captured = readiness.capture_live_config(cfg)
    identity = dict(cfg["r71"]["identity"])
    identity.update({
        "r73_training_version": ckpt.R73_TRAINING_VERSION,
        "r73_quota": {"normal": NORMAL_CAP, "reserve": RESERVE_CAP, "hard": HARD_CAP},
        "actor_objective": ckpt.R73_ACTOR_OBJECTIVE,
        "objective_version": ckpt.R73_OBJECTIVE_VERSION,
        "objective_install": install,
        "ppo_update_unchanged": trainer.ppo_update is ppo_update_before,
        "save_is_r73": trainer.save_checkpoint is r73_save,
        "load_is_r73": trainer.load_checkpoint is r73_load,
        "load_ro_is_r73": trainer.load_checkpoint_read_only is r73_load_ro,
        "live_config": captured,
        "run_identity": run_identity.as_dict(),
        "model_read_approval": model_read_approval,
        "cost_math_module": str(Path(trainer.cm.__file__).resolve()),
        "trainer_batch_samples_is_scoped": getattr(trainer.batch_samples, "__name__", "") == "_scoped_batch_samples",
    })
    cfg["r73"] = {"identity": identity, "run_identity": run_identity.as_dict(),
                  "freeze": ckpt.R73_FREEZE}
    cfg["r73_run_identity"] = run_identity
    return cfg


# --------------------------------------------------------------------------- init ---------
def _registered_init_identity(seed: int) -> dict:
    """The ROUND73 init identity of one seed, taken from the ROUND71 checkpoint index (read-only)."""
    if not R71_INDEX_PATH.is_file():
        raise FileNotFoundError(f"the ROUND71 checkpoint index is missing: {R71_INDEX_PATH}")
    index_sha = _sha256_file(R71_INDEX_PATH)
    index = _read_json(R71_INDEX_PATH)
    node = index["seeds"][f"s{int(seed)}"]
    init = node["calibrated_init"]
    ident = node["init_identity"]
    calibration = {
        "trunk_sha256": ident["digest_after"]["trunk"],
        "early_head_sha256": ident["digest_after"]["early_head"],
        "late_head_sha256": ident["digest_after"]["late_head"],
        "value_task_head_sha256": ident["digest_after"]["value_task_head"],
        "value_cost_head_sha256": ident["digest_after"]["value_cost_head"],
        "changed_modules": ident["changed_modules"],
        "fit": ident["fit"],
        "rng_unchanged_by_calibration": ident["rng_unchanged_by_calibration"],
        "optimizer_fresh": ident["optimizer_fresh"],
        "lambda_init": ident["lambda_init"],
        "frozen_modules": ident["frozen_modules"],
        "trainable_modules": ident["trainable_modules"],
        "frozen_scalar_count": ident["frozen_scalar_count"],
        "trainable_scalar_count": ident["trainable_scalar_count"],
    }
    return {
        "kind": "round73-init-identity/1",
        "seed": int(seed),
        "task_id": TASK_ID,
        "source": "ROUND71 calibrated INIT (model-only, byte-exact copy; no offline fit in R73)",
        "source_index": R71_INDEX_PATH.name,
        "source_index_sha256": index_sha,
        "source_file": init["file"],
        "file": init["file"],
        "file_sha256": init["file_sha256"],
        "bytes": init["bytes"],
        "parameter_sha256": init["parameter_sha256"],
        "parameter_count": init["parameter_count"],
        "parameter_object_count": init["parameter_object_count"],
        "init_parameter_sha256": init["parameter_sha256"],
        "pre_scale_equals_r60_ep0": ident.get("pre_scale_equals_r60_ep0"),
        "source_r60_ep0_parameter_sha256": ident.get("source_r60_ep0_parameter_sha256"),
        "late_init_alpha": ident.get("late_init_alpha"),
        "calibration": calibration,
    }


def stage_init(cfg) -> dict:
    """Copy each seed's ROUND71 calibrated INIT into the ROUND73 checkpoint dir and register it.

    A byte-exact file copy plus an identity JSON: no environment construction, no model read,
    no forward, no optimizer.  The copy is verified against the ROUND71 checkpoint index.
    """
    import os
    out = {}
    ckpt_dir = Path(cfg["common"].CHECKPOINTS)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        registered = _registered_init_identity(seed)
        src = R71_CHECKPOINTS / registered["source_file"]
        if not src.is_file():
            raise FileNotFoundError(f"seed {seed}: the ROUND71 INIT file is missing: {src}")
        if _sha256_file(src) != registered["file_sha256"]:
            raise RuntimeError(f"seed {seed}: the ROUND71 INIT file sha256 drifted from the index")
        dst = ckpt_dir / registered["file"]
        if dst.is_file():
            if _sha256_file(dst) != registered["file_sha256"]:
                raise RuntimeError(f"seed {seed}: the staged ROUND73 INIT already exists and differs")
        else:
            tmp = dst.with_suffix(dst.suffix + f".{os.getpid()}.tmp")
            tmp.write_bytes(src.read_bytes())
            os.replace(tmp, dst)
        if _sha256_file(dst) != registered["file_sha256"]:
            raise RuntimeError(f"seed {seed}: the staged ROUND73 INIT copy is not byte-exact")
        identity_path = ckpt_dir / f"r73_init_identity_s{int(seed)}.json"
        _write_json(identity_path, registered)
        out[f"s{int(seed)}"] = {"file": str(dst), "file_sha256": registered["file_sha256"],
                                "parameter_sha256": registered["parameter_sha256"],
                                "identity": str(identity_path)}
    return {"kind": "round73-init-staging/1", "seeds": out, "environment_constructions": 0}


def init_paths(cfg, seed: int) -> dict:
    ckpt_dir = Path(cfg["common"].CHECKPOINTS)
    return {"dir": ckpt_dir,
            "init": ckpt_dir / f"v2_train_s{int(seed)}_init.pt",
            "identity": ckpt_dir / f"r73_init_identity_s{int(seed)}.json",
            "rolling": ckpt_dir / f"v2_train_s{int(seed)}.pt",
            "ep128": ckpt_dir / f"v2_train_s{int(seed)}_ep128.pt"}


def r73_fresh(cfg, seed: int):
    """THIS run's NEW controller for training: the staged ROUND73 INIT, model-only, never a re-fit."""
    paths = init_paths(cfg, seed)
    if not paths["init"].is_file():
        raise FileNotFoundError(f"seed {seed}: the ROUND73 INIT checkpoint is missing: "
                                f"{paths['init']}; run the init-staging step first")
    if not paths["identity"].is_file():
        raise RuntimeError(f"seed {seed}: the registered ROUND73 INIT identity evidence is missing")
    registered = _read_json(paths["identity"])
    controller, state, identity = ckpt.r73_load_init_read_only(paths["init"], registered=registered)
    if int(state.get("episodes_done", -1)) != 0 or int(state.get("updates", -1)) != 0:
        raise RuntimeError(f"seed {seed}: the INIT checkpoint is not an untrained initialisation")
    if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
        raise RuntimeError(f"seed {seed}: the INIT load restored training state")
    summary = controller.parameter_summary()
    if summary["parameter_sha256"] != registered["init_parameter_sha256"]:
        raise RuntimeError(f"seed {seed}: INIT parameters != the registered calibrated identity")
    if any(p.requires_grad for p in r68.frozen_parameters(controller)) or \
            not all(p.requires_grad for p in r68.trainable_parameters(controller)):
        raise RuntimeError(f"seed {seed}: the INIT controller lost its frozen mask")
    if summary["parameter_object_count"] != 12:
        raise RuntimeError(f"seed {seed}: the INIT controller is not the five-module schema")
    return controller

# --------------------------------------------------------------------------- training ----
def _run_train_episode(cfg, controller, collector, *, seed, ep, scenario, lam):
    runner = cfg["runner"]
    tag = f"r73_train_s{seed}_e{ep}"
    action_seed = runner.action_seed(int(seed), int(ep))
    logical = f"r73_new_training_s{int(seed)}_e{int(ep)}"
    metadata = {"arm": "NEW_PURE_COST_SRC", "training_seed": int(seed), "episode_index": int(ep),
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
            raise RuntimeError(f"ROUND73 reuse: ROUND66 immediate reference row missing for scene "
                               f"{scene} at {path}")
        row = _read_json(path)
        if int(row.get("scenario_seed", -1)) != int(scene):
            raise RuntimeError(f"scene {scene}: ROUND66 reference row scenario_seed mismatch")
        refs[int(scene)] = float(row["on_time"])
    if len(refs) != REF_COUNT:
        raise RuntimeError(f"reused ROUND66 references {len(refs)} != {REF_COUNT}")
    return refs


def _r71_batch0_behaviour(seed: int) -> dict:
    """The committed ROUND71 batch-0 episode behaviour (read-only), for the init-identity check.

    ROUND73 starts from the same per-seed INIT with the same fresh lambda and the same action
    seeds, and the actor objective enters only at the first UPDATE (after the rollouts), so the
    first four episodes must reproduce the ROUND71 batch-0 behaviour field for field.
    """
    rec = _read_json(R71_CHECKPOINTS / f"r40_batch_s{int(seed)}_b0_attempt01.json")
    out = {}
    for ep in rec["episodes"]:
        hits = sorted(R71_ROWS.glob(f"v2_r71_train_s{int(seed)}_e{int(ep)}_s*_a*.json"))
        if len(hits) != 1:
            raise RuntimeError(f"seed {seed} ep {ep}: {len(hits)} ROUND71 row matches")
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
            jobs.append({"scope": "ppo_training", "logical_job": f"r73_new_training_s{seed}_e{ep}",
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
        for arm in MODEL_ARMS:
            slot += 1
            jobs.append({"scope": "focal", "arm": arm,
                         "logical_job": f"{EVAL_ROW_TAG}_focal_{arm.lower()}_s{seed}_scene{FOCAL_SCENE}",
                         "seed": seed, "mode": "greedy", "scene": FOCAL_SCENE,
                         "normal_slot": slot})
    if len(jobs) != NORMAL_CAP or len({j["logical_job"] for j in jobs}) != NORMAL_CAP:
        raise AssertionError(f"ROUND73 fixed plan is not exactly {NORMAL_CAP} unique jobs "
                             f"(got {len(jobs)})")
    return {"kind": "round73-fixed-job-plan/1", "task_id": TASK_ID, "branch": BRANCH,
            "training_version": ckpt.R73_TRAINING_VERSION,
            "structure": ckpt.R73_STRUCTURE,
            "actor_objective": ckpt.R73_ACTOR_OBJECTIVE,
            "approval_id": trial.TRIAL_APPROVAL_ID,
            "intervention": ("the combined actor advantage in canonical_v2_trainer.batch_samples "
                             "becomes -A_cost (task term and lambda dropped); dual critics, GAE, "
                             "reward timing, mask, entropy, clip and the dual update unchanged"),
            "training_episode_count": TRAINING_COUNT, "model_eval_count": EVAL_COUNT,
            "model_eval_arm": MODEL_EVAL_ARM,
            "focal_count": FOCAL_COUNT, "reference_count": REF_COUNT,
            "normal_count": NORMAL_CAP, "reserve_cap": RESERVE_CAP, "hard_cap": HARD_CAP,
            "historical_probe": HISTORICAL_PROBE,
            "eval_scenes": list(SCENES_NEW), "focal_scene": FOCAL_SCENE,
            "row_tag": ROW_TAG, "eval_row_tag": EVAL_ROW_TAG,
            "jobs": jobs,
            "r71_source_dir": str(R71_DIR),
            "reference_reuse_note": ("the 48 immediate references are reused read-only from "
                                     "ROUND66; the ROUND71 same-seed/same-mode ep128 control rows "
                                     "are read-only reuse subject to eligibility")}


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
            f"R73 seed {seed} batch {batch}: cannot resolve exactly one qualified batch record "
            f"(candidates={len(candidates)}); refusing to fall back to a fixed attempt01")
    path, rec = candidates[0]
    if controller is not None and state is not None:
        if int(state["episodes_done"]) == int(rec["episodes_done_after"]):
            live = controller.parameter_summary()["parameter_sha256"]
            if live != rec.get("parameters_after_sha"):
                raise RuntimeError(
                    f"R73 seed {seed} batch {batch}: the live controller ({live}) does not match "
                    f"the qualified record {path.name} ({rec.get('parameters_after_sha')})")
    return path, rec


def train_seed(cfg, ledger, plan, seed, order, references, *, stop_after_first_gate=False):
    common = cfg["common"]; trainer = cfg["trainer"]
    run_identity = cfg["r73_run_identity"]
    ckpt_dir = Path(common.CHECKPOINTS)
    rolling = init_paths(cfg, seed)["rolling"]
    if rolling.exists():
        controller, optimizer, state = ckpt.r73_load_checkpoint(
            rolling, run_identity=run_identity)
        ps = controller.parameter_summary()
        if ps["parameter_object_count"] != 12 or \
                r68.frozen_parameter_summary(controller).get("structure") != r68.R68_STRUCTURE or \
                state.get("structure") != ckpt.R73_STRUCTURE or \
                state.get("actor_objective") != ckpt.R73_ACTOR_OBJECTIVE:
            raise RuntimeError(f"seed {seed}: rolling checkpoint is not a ROUND73 pure-cost model")
        if any(p.requires_grad for p in r68.frozen_parameters(controller)) or \
                not all(p.requires_grad for p in r68.trainable_parameters(controller)):
            raise RuntimeError(f"seed {seed}: restored controller did not re-establish the frozen mask")
        resumed = True
    else:
        controller = r73_fresh(cfg, seed)
        optimizer = r68.build_optimizer(controller, common.LEARNING_RATE)
        state = r68.fresh_training_state(
            int(seed), gae_lambda=common.GAE_LAMBDA, eta_lambda=common.ETA_LAMBDA,
            gamma=common.GAMMA, delta=0.0, lambda_init=1.0,
            data_order_seed=common.DATA_ORDER_SEED, b0_bytes=common.B0_BYTES,
            alpha=r68.LATE_INIT_ALPHA)
        state["training_version"] = ckpt.R73_TRAINING_VERSION
        state["loss_version"] = ckpt.R73_LOSS_VERSION
        state["state_schema"] = ckpt.R73_STATE_SCHEMA
        state["optimizer_schema"] = ckpt.R73_OPTIMIZER_SCHEMA
        state["structure"] = ckpt.R73_STRUCTURE
        state["actor_objective"] = ckpt.R73_ACTOR_OBJECTIVE
        state["objective_version"] = ckpt.R73_OBJECTIVE_VERSION
        state["run_identity"] = run_identity.as_dict()
        state["init_source"] = _read_json(init_paths(cfg, seed)["identity"])
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
            logical = f"r73_new_training_s{seed}_e{ep}"
            if ledger.is_complete(logical):
                att = _qualified_attempt(cfg["runtime"]["r40_common"], logical)
                paths = common.cell_paths("train", scenario, att, tag=f"r73_train_s{seed}_e{ep}")
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
        batch_record = {"kind": "round73-training-batch-record/1", "batch": batch, "seed": seed,
                        "episodes": [int(r["episode_index"]) for r in episodes],
                        "scenarios": [int(r["scenario_seed"]) for r in episodes],
                        "lambda_at_rollout": lam, "update": update_report,
                        "constraint": g_record, "dual": dual_record,
                        "parameters_before_sha": before["parameter_sha256"],
                        "parameters_after_sha": after["parameter_sha256"],
                        "episodes_done_after": int(state["episodes_done"]),
                        "optimizer_steps_after": int(state["optimizer_steps"]),
                        "structure": ckpt.R73_STRUCTURE,
                        "actor_objective": ckpt.R73_ACTOR_OBJECTIVE,
                        "resumed_batch": bool(resumed and batch == int(state["updates"]) - 1)}
        _write_json(ckpt_dir / f"r40_batch_s{seed}_b{batch}_attempt01.json", batch_record)
        ledger.write_ledger()
        gate_path = cfg["raw_root"] / "runtime" / f"r73_first_batch_gate_s{seed}.json"
        if batch == 0 and not gate_path.exists():
            init_id = _read_json(init_paths(cfg, seed)["identity"])
            gate = _first_batch_gate(cfg, controller, state, batch_record, episodes,
                                     init_identity=init_id)
            _write_json(gate_path, gate)
            if gate["status"] != "passed":
                raise RuntimeError(f"seed {seed}: first-batch gate failed: {gate}")
            if stop_after_first_gate:
                return {"seed": seed, "status": "first_gate_passed", "gate": gate["status"]}
        print(f"R73 seed {seed} batch {batch}: episodes_done={state['episodes_done']} "
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
    gate_path = cfg["raw_root"] / "runtime" / f"r73_first_batch_gate_s{seed}.json"
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
        att = _qualified_attempt(r40, f"r73_new_training_s{seed}_e{int(ep)}")
        paths = cfg["common"].cell_paths("train", scenario, att, tag=f"r73_train_s{seed}_e{ep}")
        episodes.append(_reload_episode(paths["row"], paths["tape"], seed=seed, ep=int(ep),
                                        scenario=scenario, lam=lam))
    gate = _first_batch_gate(cfg, controller, state, record, episodes, init_identity=init_id)
    gate["reload_attempt_resolution"] = "qualified_attempt_from_complete_claim"
    _write_json(gate_path, gate)
    if gate["status"] != "passed":
        raise RuntimeError(f"seed {seed}: first-batch gate failed (from committed): {gate}")
    return gate


def _first_batch_gate(cfg, controller, state, record, episodes, *, init_identity):
    """The REAL ROUND73 first-batch gate.

    Beyond the inherited ROUND71 checks it proves, on the committed batch itself:

      * the live trainer carries the ROUND73 objective and the scoped ``batch_samples`` patch;
      * re-forming the batch through that live path yields ``advantage == -(return_cost -
        behaviour_value_cost)`` for EVERY sample, with A_task genuinely non-zero, so the actor
        surrogate the update consumed was the pure-cost one and the CONTROL combination would
        have differed (a pure recomputation: no environment, no optimizer step);
      * ``cost_math.actor_advantage`` is restored outside the scoped call, so nothing else in
        the runtime saw the override;
      * the batch started from THIS run's staged INIT (module digests vs the registered
        ROUND73 init identity) and the first four episodes reproduce the committed ROUND71
        batch-0 behaviour field for field (the initial policy is identical; the objective
        enters only at the update).
    """
    trainer = cfg["trainer"]
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
    # The init-identity checks read the PRE-update INIT module shas (late_head and both critic
    # heads are trainable and move during the first update); the load is parameter-hashing only.
    _init_ctrl, _init_state, _init_id = ckpt.r73_load_init_read_only(
        init_paths(cfg, int(state["training_seed"]))["init"], registered=init_identity)
    module_sha = ckpt._registered_module_sha(_init_ctrl)
    checks["init_trunk_is_r71_ep0_trunk"] = (module_sha["trunk"] == cal["trunk_sha256"])
    checks["init_early_head_is_r71_ep0"] = (module_sha["early_head"] == cal["early_head_sha256"])
    checks["init_late_head_is_r71_post_scale"] = (module_sha["late_head"] == cal["late_head_sha256"])
    checks["init_started_from_r73_staged_init"] = (
        before_sha == init_identity["init_parameter_sha256"])
    checks["critic_heads_are_calibrated"] = (
        module_sha["value_task_head"] == cal["value_task_head_sha256"]
        and module_sha["value_cost_head"] == cal["value_cost_head_sha256"])
    checks["init_cursor_zero"] = (int(init_identity.get("parameter_object_count", -1)) == 12)
    # ---- the ONE intervention, proven on the committed batch through the live path ----
    checks["actor_objective_live"] = (getattr(trainer, "R73_ACTOR_OBJECTIVE", None)
                                      == ckpt.R73_ACTOR_OBJECTIVE)
    checks["batch_samples_is_scoped_patch"] = (
        getattr(trainer.batch_samples, "__name__", "") == "_scoped_batch_samples")
    samples = trainer.batch_samples(episodes)          # pure recomputation; no env, no optimizer
    lam = float(record["lambda_at_rollout"])
    pure, task_nonzero, control_differs = True, False, False
    for s in samples:
        a_cost = float(s["return_cost"]) - float(s["behaviour_value_cost"])
        a_task = float(s["return_task"]) - float(s["behaviour_value_task"])
        if abs(float(s["advantage"]) - (-a_cost)) > 1e-9:
            pure = False
        if abs(a_task) > 1e-9:
            task_nonzero = True
        if abs(float(s["advantage"]) - (lam * a_task - a_cost)) > 1e-9:
            control_differs = True
    checks["actor_advantage_is_minus_a_cost"] = pure and len(samples) > 0
    checks["a_task_nonzero_in_batch"] = task_nonzero
    checks["control_objective_would_differ"] = control_differs
    checks["actor_advantage_restored_outside_scope"] = (
        getattr(trainer.cm.actor_advantage, "__name__", "") != "pure_cost_actor_advantage")
    # ---- inherited ROUND71 checks ----
    behaviour = _r71_batch0_behaviour(int(state["training_seed"]))
    mismatches = []
    for row in episodes:
        ep = int(row["episode_index"])
        want = behaviour.get(ep)
        if want is None:
            mismatches.append({"episode": ep, "problem": "no ROUND71 committed row"})
            continue
        for field in ("on_time", "spent_bytes", "task_reward_total", "airtime_tap_total"):
            got = row.get(field)
            if float(got) != float(want[field]):
                mismatches.append({"episode": ep, "field": field, "r73": got, "r71": want[field]})
    checks["first_batch_behaviour_matches_r71"] = (not mismatches)
    checks["first_batch_behaviour_mismatches"] = mismatches
    checks["structure_live_r68"] = (after_frozen.get("structure") == r68.R68_STRUCTURE)
    checks["structure_state_r73"] = (state.get("structure") == ckpt.R73_STRUCTURE)
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
    status = "passed" if all(v for k, v in checks.items() if isinstance(v, bool)) else "failed"
    return {"kind": "round73-first-batch-gate/1", "seed": int(state["training_seed"]),
            "status": status, "checks": checks,
            "parameters_after_sha": after["parameter_sha256"],
            "frozen_sha256": r68x.r66f._tensor_identity(r68.frozen_parameters(controller))["sha256"],
            "lambda_after_first_batch": dual_record["lambda_next"],
            "actor_objective": ckpt.R73_ACTOR_OBJECTIVE,
            "objective_version": ckpt.R73_OBJECTIVE_VERSION,
            "isolation_note": ("critic->trunk isolation is applied inside routed_ppo, but its trunk "
                               "effect is masked because the trunk is requires_grad=False by the "
                               "inherited ROUND68 intervention; the absent trunk gradient is NOT "
                               "evidence about isolation (proved separately on a synthetic "
                               "backward in the pre-launch gates)"),
            "gate_diag": gate_diag, "module_gradient_l1_mean": modgrad}


# ------------------------------------------------------------------ zero-env acceptance ----
def zero_env_cli_acceptance(cfg) -> dict:
    """Zero-environment CLI acceptance of the live ROUND73 wiring, stopped before any episode.

    Proves the real CLI chain (configure/bind -> model construction -> gate reads) is ONE ROUND73
    object: the live module paths, the ROUND73 parameter set and counts, the checkpoint routing,
    the installed objective and the live quota bindings.  No environment is constructed, no
    episode provider is called, no model is forwarded.
    """
    trainer = cfg["trainer"]; common = cfg["common"]; r40 = cfg["runtime"]["r40_common"]
    identity = cfg["r73"]["identity"]
    checks = {
        "task_id": identity.get("task_id") == TASK_ID or cfg["r73"]["run_identity"]["task_id"] == TASK_ID,
        "branch": cfg["r73"]["run_identity"]["branch"] == BRANCH,
        "training_version": cfg["runtime"]["r42_versions"].TRAINING_VERSION == ckpt.R73_TRAINING_VERSION,
        "quota_normal": int(r40.NORMAL_CAP) == NORMAL_CAP and int(common.NORMAL_LAUNCHES) == NORMAL_CAP,
        "quota_reserve": int(r40.RESERVE_CAP) == RESERVE_CAP and int(common.RESERVE_LAUNCHES) == RESERVE_CAP,
        "quota_hard": int(r40.HARD_CAP) == HARD_CAP and int(common.LAUNCH_CAP) == HARD_CAP,
        "save_is_r73": trainer.save_checkpoint.__name__ == "r73_save",
        "load_is_r73": trainer.load_checkpoint.__name__ == "r73_load",
        "load_ro_is_r73": trainer.load_checkpoint_read_only.__name__ == "r73_load_ro",
        "ppo_update_is_r68_routed": identity.get("ppo_update_unchanged") is True,
        "actor_objective_installed": getattr(trainer, "R73_ACTOR_OBJECTIVE", None) == ckpt.R73_ACTOR_OBJECTIVE,
        "batch_samples_scoped": getattr(trainer.batch_samples, "__name__", "") == "_scoped_batch_samples",
        "cost_math_restored": getattr(trainer.cm.actor_advantage, "__name__", "") != "pure_cost_actor_advantage",
        "raw_root_is_round73": (Path(cfg["raw_root"]).parent.name
                                == "round73_pure_cost_actor_ablation"),
        "live_gae_lambda": float(common.GAE_LAMBDA) == readiness._EXPECTED_LIVE["common.GAE_LAMBDA"],
        "live_eta_lambda": float(common.ETA_LAMBDA) == readiness._EXPECTED_LIVE["common.ETA_LAMBDA"],
        "live_gamma": float(common.GAMMA) == 1.0,
        "live_trainer_is_gated_module": (
            _sha256_file(Path(trainer.__file__)) == _sha256_file(FROZEN_SOURCES)),
        "live_cost_math_is_gated_module": (
            _sha256_file(Path(trainer.cm.__file__)) == _sha256_file(COST_MATH_SOURCE)),
    }
    return {"kind": "round73-zero-env-cli-acceptance/1", "task_id": TASK_ID,
            "status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "trainer_module": identity.get("trainer_module"),
            "controller_module": identity.get("controller_module"),
            "common_module": identity.get("common_module"),
            "runner_module": identity.get("runner_module"),
            "cost_math_module": identity.get("cost_math_module"),
            "gated_module_identity": {
                "live_trainer": str(Path(trainer.__file__).resolve()),
                "live_trainer_sha256": _sha256_file(Path(trainer.__file__)),
                "gated_trainer_copy": str(FROZEN_SOURCES),
                "gated_trainer_copy_sha256": _sha256_file(FROZEN_SOURCES),
                "live_cost_math": str(Path(trainer.cm.__file__).resolve()),
                "live_cost_math_sha256": _sha256_file(Path(trainer.cm.__file__))},
            "consumption": {"environment_constructions": 0, "episodes": 0, "model_forwards": 0,
                            "optimizer_steps": 0, "ppo_updates": 0}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND73 training executor")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--phase", choices=("acceptance", "init", "train"), required=True)
    ap.add_argument("--seed", type=int, action="append", choices=(11, 22, 33), default=None)
    ap.add_argument("--first-gate-only", action="store_true")
    args = ap.parse_args(argv)
    raw_root = Path(args.raw_root).resolve()
    gate = trial.trial_gate()
    run_identity = trial.trial_run_identity()
    cfg = configure_r73_trial(raw_root, gate, run_identity,
                              model_read_approval=trial.MODEL_READ_APPROVAL)
    if args.phase == "acceptance":
        receipt = zero_env_cli_acceptance(cfg)
        _write_json(raw_root / "runtime" / "ROUND73_ZERO_ENV_ACCEPTANCE.json", receipt)
        print(json.dumps(receipt, indent=1))
        return 0 if receipt["status"] == "passed" else 1
    if args.phase == "init":
        receipt = stage_init(cfg)
        _write_json(raw_root / "runtime" / "ROUND73_INIT_STAGING.json", receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k != "seeds"}, indent=1))
        return 0
    import r73_run_trial as runner
    return runner.phase_train_cli(cfg, raw_root, args)


if __name__ == "__main__":
    raise SystemExit(main())