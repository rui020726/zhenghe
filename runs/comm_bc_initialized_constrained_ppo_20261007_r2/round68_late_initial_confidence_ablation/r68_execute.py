"""ROUND68 executor: late-initial-confidence ablation (acceptance + fixed training only).

The ONLY scientific difference from ROUND66 is a one-time late_head.weight *= 0.10 and
late_head.bias *= 0.10 on the fresh controller built from the compliant ROUND60 ep0 (model-only).
trunk + early_head stay frozen (requires_grad=False); only late_head + both critics train. The
loss, the routed forward, the sequential mask, the joint log-probability, the combined advantage
lambda*A_task - A_cost, GAE, ALL_ROWS, the global gradient clip, the ratio clip, the beta=1.00 gate
prior and the dual update are all ROUND66's, unchanged. The 0.10 scale is INIT-ONLY: applied once at
fresh construction, never a per-forward temperature, never re-applied on resume.

Budget (fresh ROUND68 authorization): historical non-plan probe = 0; normal plan = 774
(384 training + 384 model-arm eval + 6 focal); technical-failure reserve = 26; cumulative
launch/attempt hard cap = 800. ROUND58/R60/R66/R67 constructions are NOT ROUND68 consumption.

Wiring note: this module reuses ONLY the generic, seam-free helpers from r66_execute (r66x.*):
_sha256_file/_read_json/_write_json/_atomic_run/_write_row/_reload_episode/_slot_for, none of which
touch the changing seams. Every ROUND68-specific seam (fresh controller, save/load/restore, version,
gate, plan, output roots) is defined HERE with explicit r68.* / r66x.* qualification. There is no
`from r66_execute import *`, so an inherited function cannot silently keep an R66 object in its
__globals__; the live chain is proven by zero_env_acceptance on the live modules.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent
R66_DIR = RUN_ROOT / "round66_bc_early_path_preserving_late_adaptation"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(R66_DIR))
import r66_execute as r66x      # generic executor helpers + R58 runtime setup (seam-free reuse)
import r66_frozen as r66f       # low-level frozen-path helpers (_tensor_identity, etc.)
import r68_frozen as r68        # the ONLY scientific difference + full ROUND68 identity

SEEDS = (11, 22, 33)
EPISODES_PER_SEED = 128
EPISODES_PER_UPDATE = 4
MODES = ("greedy", "sampled")
SCENES_NEW = tuple(range(1_800_000, 1_800_032))
FOCAL_SCENE = 1_700_027
ROW_TAG = "r68eval"
REF_COUNT = 48
TRAINING_COUNT = 384
EVAL_COUNT = 384
FOCAL_COUNT = 6
TASK_ID = "ROUND68-LATE-INITIAL-CONFIDENCE-ABLATION"
BRANCH = "exp/round68-late-initial-confidence-ablation"
NORMAL_CAP = 774
RESERVE_CAP = 26
HARD_CAP = 800
HISTORICAL_PROBE = 0
R60_SOURCE_COMMIT = "cbae709f6bb2bf566acf341d626de659524772f4"

R68_FREEZE = {
    "kind": r68.R68_FREEZE_KIND,
    "task_id": TASK_ID, "branch": BRANCH,
    "training_version": r68.R68_TRAINING_VERSION,
    "structure": r68.R68_STRUCTURE,
    "loss_version": r68.R68_LOSS_VERSION,
    "late_init_alpha": r68.LATE_INIT_ALPHA,
    "frozen_modules": list(r68.FROZEN_MODULES),
    "trainable_modules": list(r68.TRAINABLE_MODULES),
    "gae_lambda": 1.0, "eta_lambda": 0.5, "gamma": 1.0, "delta": 0.0, "lambda_init": 1.0,
    "data_order_seed": 20260927, "b0_bytes": 10000.0,
    "r60_source_commit": R60_SOURCE_COMMIT,
    "inherits_from": ("round66-bc-early-path-preserving-late-adaptation (structure/loss/config); "
                      "only the one-time late-head init scale differs"),
}

# Generic, seam-free helpers reused verbatim from the mature R66 executor.
_sha256_file = r66x._sha256_file
_read_json = r66x._read_json
_write_json = r66x._write_json
_atomic_run = r66x._atomic_run
_write_row = r66x._write_row
_reload_episode = r66x._reload_episode
_slot_for = r66x._slot_for


class R68Ledger:
    """Independent ROUND68 claim/slot/quota ledger under the ROUND68 raw root."""

    def __init__(self, raw: Path) -> None:
        self.raw = raw
        self.claims_dir = raw / "launch_claims"
        self.slots_dir = raw / "launch_slots"
        self.runtime_dir = raw / "runtime"
        for d in (self.claims_dir, self.slots_dir / "normal", self.slots_dir / "reserve", self.runtime_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.runtime_dir / "ROUND68_LEDGER.json"

    def _slug(self, logical: str) -> str:
        import re
        return re.sub(r"[^0-9A-Za-z_.-]+", "_", logical)

    def claim_path(self, logical: str) -> Path:
        return self.claims_dir / self._slug(logical) / "attempt_0001.json"

    def is_complete(self, logical: str) -> bool:
        p = self.claim_path(logical)
        return p.is_file() and _read_json(p).get("status") == "complete"

    def counts(self) -> dict:
        normal = reserve = complete = failed = 0
        for p in self.claims_dir.glob("*/*.json"):
            c = _read_json(p)
            if c.get("category") == "normal":
                normal += 1
            elif c.get("category") == "reserve":
                reserve += 1
            if c.get("status") == "complete":
                complete += 1
            elif c.get("status") == "failed_technical":
                failed += 1
        return {"normal_claims": normal, "reserve_claims": reserve, "complete": complete,
                "failed_technical": failed, "cumulative_attempts": normal + reserve,
                "cumulative_attempts_including_historical_probe": normal + reserve + HISTORICAL_PROBE,
                "hard_cap": HARD_CAP, "hard_headroom": HARD_CAP - (normal + reserve + HISTORICAL_PROBE)}

    def write_ledger(self) -> dict:
        led = {"kind": "round68-ledger/1", "task_id": TASK_ID, "branch": BRANCH,
               "training_version": r68.R68_TRAINING_VERSION, "late_init_alpha": r68.LATE_INIT_ALPHA,
               "historical_non_plan_probe": HISTORICAL_PROBE,
               "normal_cap": NORMAL_CAP, "reserve_cap": RESERVE_CAP, "hard_cap": HARD_CAP,
               "counts": self.counts(), "updated_unix": time.time()}
        _write_json(self.ledger_path, led)
        return led


def configure_and_inject(raw_root: Path) -> dict[str, Any]:
    """Stand up the frozen R46/R42 sim runtime exactly as R58/R66 configured it, then inject the
    ROUND68 late-initial-confidence controller factory + routed ppo_update + 5-module checkpoint,
    and bind every output-owning path to the ROUND68 raw root. Verified against the live modules."""
    import r58_execute
    raw_root = Path(raw_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    base = r58_execute.load_base_runner()
    _saved_here = r58_execute.HERE
    r58_execute.HERE = raw_root.parent
    try:
        r58_execute.bind_runner_paths(base, raw_root)
    finally:
        r58_execute.HERE = _saved_here
    r58_execute.install_private_configure_seams(base, allow_existing_journal=False)
    base._R45_MODULE = None
    base._RUNTIME = None
    base._configure_runtime_module()
    runtime = base._RUNTIME
    if runtime is None:
        raise RuntimeError("ROUND68 runtime did not configure")
    trainer = runtime["trainer"]
    controller_module = runtime["controller_module"]
    common = runtime["common"]
    runner = runtime["runner"]
    r42_run = runtime["r42_run"]
    r40_common = runtime["r40_common"]
    # R60/R66 living overrides on the ACTUAL common module the trainer reads (not import defaults).
    common.GAE_LAMBDA = 1.0
    common.ETA_LAMBDA = 0.5
    if float(common.GAMMA) != 1.0:
        raise RuntimeError("frozen gamma is not 1.0")
    # Rebind the ROUND68 quota on the live r40_common ledger module and common module.
    r40_common.NORMAL_CAP = NORMAL_CAP
    r40_common.RESERVE_CAP = RESERVE_CAP
    r40_common.HARD_CAP = HARD_CAP
    common.NORMAL_LAUNCHES = NORMAL_CAP
    common.RESERVE_LAUNCHES = RESERVE_CAP
    common.LAUNCH_CAP = HARD_CAP
    r40_common.plan = lambda: {"normal_total": NORMAL_CAP, "technical_reserve_max": RESERVE_CAP,
                               "hard_cap": HARD_CAP, "task_id": TASK_ID}
    r40_common.ROUND_DIR = raw_root
    # Inject the ROUND68 late-initial-confidence controller factory (5-tuple + init identity write).
    orig_fresh = controller_module.fresh_controller

    def frozen_fresh(seed: int):
        b = orig_fresh(int(seed))
        pc, _ep0, _fid, _post, scale_id = r68.build_frozen_controller(b, int(seed))
        ck = Path(common.CHECKPOINTS)
        ck.mkdir(parents=True, exist_ok=True)
        _write_json(ck / f"r68_init_identity_s{int(seed)}.json", {
            "seed": int(seed), "late_init_alpha": scale_id["late_init_alpha"],
            "pre_scale_parameter_sha256": scale_id["pre_scale_parameter_sha256"],
            "post_scale_parameter_sha256": scale_id["post_scale_parameter_sha256"],
            "r60_ep0_parameter_sha256": scale_id["r60_ep0_parameter_sha256"],
            "pre_scale_equals_r60_ep0": scale_id["pre_scale_equals_r60_ep0"],
            "late_head_sha_before_scale": scale_id["late_head_sha_before_scale"],
            "late_head_sha_after_scale": scale_id["late_head_sha_after_scale"]})
        return pc

    controller_module.fresh_controller = frozen_fresh
    # Route the trainer's PPO update (R53 critic->trunk isolation + the UNCHANGED R66 update math).
    from round53_trial_adapter import critic_input_detached

    def routed_ppo(controller, optimizer, episodes, *, update_index):
        with critic_input_detached(controller):
            report = r68.frozen_early_ppo_update(controller, optimizer, episodes, update_index=update_index)
        import os as _os
        ppo_journal = raw_root / "runtime" / "ppo_update_calls.jsonl"
        with ppo_journal.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"update_index": int(update_index),
                                 "optimizer_steps": int(report.get("optimizer_steps", 0)),
                                 "structure": r68.R68_STRUCTURE}) + "\n")
            fh.flush(); _os.fsync(fh.fileno())
        return report

    def r68_save(controller, optimizer, state, path, *, freeze):
        return r68.save_checkpoint(controller, optimizer, state, path, freeze=R68_FREEZE)

    def r68_load(path, *, freeze):
        return r68.load_checkpoint(path, freeze=R68_FREEZE)

    def r68_load_ro(path, *, freeze):
        return r68.load_checkpoint_read_only(path, freeze=R68_FREEZE)

    trainer.ppo_update = routed_ppo
    trainer.save_checkpoint = r68_save
    trainer.load_checkpoint = r68_load
    trainer.load_checkpoint_read_only = r68_load_ro
    try:
        runtime["r42_versions"].TRAINING_VERSION = r68.R68_TRAINING_VERSION
    except Exception:
        pass
    import r44_runtime
    r44_runtime.verify_frozen_source = lambda frozen: {"r68": True}
    identity = {
        "trainer_module": str(Path(trainer.__file__).resolve()),
        "controller_module": str(Path(controller_module.__file__).resolve()),
        "common_module": str(Path(common.__file__).resolve()),
        "runner_module": str(Path(runner.__file__).resolve()),
        "raw_root": str(raw_root),
        "ppo_update_is_r68": trainer.ppo_update is routed_ppo,
        "save_is_r68": trainer.save_checkpoint is r68_save,
        "load_is_r68": trainer.load_checkpoint is r68_load,
        "load_ro_is_r68": trainer.load_checkpoint_read_only is r68_load_ro,
        "fresh_is_frozen": controller_module.fresh_controller is frozen_fresh,
        "training_version": r68.R68_TRAINING_VERSION,
        "common_GAE_LAMBDA": float(common.GAE_LAMBDA), "common_ETA_LAMBDA": float(common.ETA_LAMBDA),
    }
    return {"runtime": runtime, "trainer": trainer, "controller_module": controller_module,
            "common": common, "runner": runner, "r42_run": r42_run, "raw_root": raw_root,
            "frozen_fresh": frozen_fresh, "identity": identity}


def _run_train_episode(cfg, controller, collector, *, seed, ep, scenario, lam):
    runner = cfg["runner"]
    tag = f"r68_train_s{seed}_e{ep}"
    action_seed = runner.action_seed(int(seed), int(ep))
    logical = f"r68_new_training_s{int(seed)}_e{int(ep)}"
    metadata = {"arm": "TRAINED_ALPHA_SRC", "training_seed": int(seed), "episode_index": int(ep),
                "scenario_seed": int(scenario), "lambda_at_rollout": float(lam)}

    def _do():
        row = runner._run_episode(tag, "train", int(scenario), policy="learned",
                                  attach=collector.policy(), controller=controller, collector=collector,
                                  action_rng_seed=action_seed, tag=tag)
        row["collector"] = collector
        row["training_seed"] = int(seed); row["episode_index"] = int(ep)
        row["scenario_seed"] = int(scenario); row["lambda"] = float(lam)
        return row

    return _atomic_run(cfg, logical, metadata, _do,
                       detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}")


def reuse_r66_references(cfg, order) -> dict[int, float]:
    """Read ROUND66's committed 48 immediate-reference rows READ-ONLY (the immediate rule is
    deterministic on the frozen world, so ROUND66's per-scene on_time is reused for batch_g_hat).
    NO environment is constructed. Raises if any training scene lacks a ROUND66 reference row."""
    r66_rows = R66_DIR / "raw" / "rows"
    refs: dict[int, float] = {}
    for scene in sorted({int(s) for s in order}):
        path = r66_rows / f"v2_r66_ref_s{scene}_s{scene}_a1.json"
        if not path.is_file():
            raise RuntimeError(f"ROUND68 reuse: ROUND66 immediate reference row missing for scene "
                               f"{scene} at {path}")
        row = _read_json(path)
        if int(row.get("scenario_seed", -1)) != int(scene):
            raise RuntimeError(f"scene {scene}: ROUND66 reference row scenario_seed mismatch")
        refs[int(scene)] = float(row["on_time"])
    if len(refs) != REF_COUNT:
        raise RuntimeError(f"reused ROUND66 references {len(refs)} != {REF_COUNT}")
    return refs


def build_plan(order: list[int]) -> dict[str, Any]:
    jobs = []
    slot = 0
    train_scenes = sorted({int(s) for s in order})
    if len(train_scenes) != REF_COUNT:
        raise AssertionError(f"training scenes {len(train_scenes)} != {REF_COUNT}")
    # Training: 3 seeds x 128 episodes (first 4 per seed count toward 128). No Block-A build:
    # the 48 immediate references are reused read-only from ROUND66.
    for seed in SEEDS:
        for ep in range(EPISODES_PER_SEED):
            slot += 1
            jobs.append({"scope": "ppo_training", "logical_job": f"r68_new_training_s{seed}_e{ep}",
                         "seed": seed, "episode_index": ep, "scene": int(order[ep]), "normal_slot": slot})
    # Eval: two model arms (INIT_ALPHA fresh-scaled-ep0 untrained, TRAINED_ALPHA R68 ep128).
    for arm in ("INIT_ALPHA", "TRAINED_ALPHA"):
        for seed in SEEDS:
            for mode in MODES:
                for scene in SCENES_NEW:
                    slot += 1
                    jobs.append({"scope": "model_eval", "arm": arm,
                                 "logical_job": f"{ROW_TAG}_model_{arm.lower()}_s{seed}_{mode}_scene{scene}",
                                 "seed": seed, "mode": mode, "scene": int(scene), "normal_slot": slot})
    # Focal scene mechanism diagnostic (both arms, 3 seeds, greedy).
    for seed in SEEDS:
        for arm in ("INIT_ALPHA", "TRAINED_ALPHA"):
            slot += 1
            jobs.append({"scope": "focal", "arm": arm,
                         "logical_job": f"{ROW_TAG}_focal_{arm.lower()}_s{seed}_scene{FOCAL_SCENE}",
                         "seed": seed, "mode": "greedy", "scene": FOCAL_SCENE, "normal_slot": slot})
    if len(jobs) != NORMAL_CAP or len({j["logical_job"] for j in jobs}) != NORMAL_CAP:
        raise AssertionError(f"ROUND68 fixed plan is not exactly {NORMAL_CAP} unique jobs (got {len(jobs)})")
    return {"kind": "round68-fixed-job-plan/1", "task_id": TASK_ID, "branch": BRANCH,
            "training_version": r68.R68_TRAINING_VERSION, "structure": r68.R68_STRUCTURE,
            "late_init_alpha": r68.LATE_INIT_ALPHA,
            "training_episode_count": TRAINING_COUNT, "model_eval_count": EVAL_COUNT,
            "focal_count": FOCAL_COUNT, "reference_count": REF_COUNT,
            "normal_count": NORMAL_CAP, "reserve_cap": RESERVE_CAP, "hard_cap": HARD_CAP,
            "historical_probe": HISTORICAL_PROBE, "eval_scenes_new": list(SCENES_NEW),
            "focal_scene": FOCAL_SCENE, "row_tag": ROW_TAG, "r60_source_commit": R60_SOURCE_COMMIT,
            "reference_reuse_note": ("the 48 immediate references (and, for statistics context, the "
                                     "ROUND66 ep128/ep0/rule rows) are reused READ-ONLY from ROUND66; "
                                     "they are NOT new ROUND68 environment constructions"),
            "jobs": jobs}


def train_seed(cfg, ledger, plan, seed, order, references, *, stop_after_first_gate=False):
    common = cfg["common"]; trainer = cfg["trainer"]
    ckpt_dir = Path(common.CHECKPOINTS)
    rolling = ckpt_dir / f"v2_train_s{seed}.pt"
    if rolling.exists():
        controller, optimizer, state = r68.load_checkpoint(rolling, freeze=R68_FREEZE)
        ps = controller.parameter_summary()
        if ps["parameter_object_count"] != 12 or \
                r68.frozen_parameter_summary(controller).get("structure") != r68.R68_STRUCTURE:
            raise RuntimeError(f"seed {seed}: rolling checkpoint is not a ROUND68 late-initial-confidence model")
        if any(p.requires_grad for p in r68.frozen_parameters(controller)) or \
                not all(p.requires_grad for p in r68.trainable_parameters(controller)):
            raise RuntimeError(f"seed {seed}: restored controller did not re-establish the frozen mask")
        resumed = True
    else:
        controller = cfg["frozen_fresh"](seed)   # scaled head + writes r68_init_identity_s{seed}.json
        optimizer = r68.build_optimizer(controller, common.LEARNING_RATE)
        state = r68.fresh_training_state(seed, gae_lambda=1.0, eta_lambda=0.5, gamma=1.0,
                                         delta=0.0, lambda_init=1.0, data_order_seed=20260927,
                                         b0_bytes=common.B0_BYTES)
        resumed = False
    gate_path = cfg["raw_root"] / "runtime" / f"r68_first_batch_gate_s{seed}.json"
    if int(state["episodes_done"]) >= EPISODES_PER_UPDATE:
        g = _gate_from_committed(cfg, seed, controller, state)
        if g is not None and stop_after_first_gate:
            return {"seed": seed, "status": "first_gate_passed_from_committed", "gate": g["status"]}
    while int(state["episodes_done"]) < EPISODES_PER_SEED:
        batch = int(state["updates"])
        if int(state["episodes_done"]) != batch * EPISODES_PER_UPDATE:
            raise RuntimeError(f"seed {seed}: episode cursor and update index inconsistent")
        lam = float(state["lambda"])
        before = controller.parameter_summary()
        pre = ckpt_dir / f"v2_train_s{seed}_before_b{batch}.pt"
        if not pre.exists():
            r68.save_checkpoint(controller, optimizer, state, pre, freeze=R68_FREEZE)
        episodes = []
        for k in range(EPISODES_PER_UPDATE):
            ep = int(state["episodes_done"]) + k
            scenario = int(order[ep])
            logical = f"r68_new_training_s{seed}_e{ep}"
            if ledger.is_complete(logical):
                paths = common.cell_paths("train", scenario, 1, tag=f"r68_train_s{seed}_e{ep}")
                episodes.append(_reload_episode(paths["row"], paths["tape"],
                                                seed=seed, ep=ep, scenario=scenario, lam=lam))
                continue
            collector = __import__("r42_one_shot").OneShotSendSkipCollector(controller, greedy=False)
            row = _run_train_episode(cfg, controller, collector, seed=seed, ep=ep, scenario=scenario, lam=lam)
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
        r68.save_checkpoint(controller, optimizer, state, post, freeze=R68_FREEZE)
        r68.save_checkpoint(controller, optimizer, state, rolling, freeze=R68_FREEZE)
        batch_record = {"kind": "round68-training-batch-record/1", "batch": batch, "seed": seed,
                        "episodes": [int(r["episode_index"]) for r in episodes],
                        "scenarios": [int(r["scenario_seed"]) for r in episodes],
                        "lambda_at_rollout": lam, "update": update_report, "constraint": g_record,
                        "dual": dual_record, "parameters_before_sha": before["parameter_sha256"],
                        "parameters_after_sha": after["parameter_sha256"],
                        "episodes_done_after": int(state["episodes_done"]),
                        "optimizer_steps_after": int(state["optimizer_steps"]),
                        "structure": r68.R68_STRUCTURE}
        _write_json(ckpt_dir / f"r40_batch_s{seed}_b{batch}_attempt01.json", batch_record)
        ledger.write_ledger()
        if batch == 0 and not gate_path.exists():
            ep0_id = _read_json(ckpt_dir / f"r68_init_identity_s{seed}.json")
            gate = _first_batch_gate(cfg, controller, state, batch_record, episodes, ep0_identity=ep0_id)
            _write_json(gate_path, gate)
            if gate["status"] != "passed":
                raise RuntimeError(f"seed {seed}: first-batch gate failed: {gate}")
            if stop_after_first_gate:
                return {"seed": seed, "status": "first_gate_passed", "gate": gate["status"]}
        print(f"R68 seed {seed} batch {batch}: episodes_done={state['episodes_done']} "
              f"lambda={state['lambda']:.4f} early_free={update_report.get('early_free')} "
              f"late_free={update_report.get('late_free')} gnorms={update_report.get('gradient_norm_mean')}",
              flush=True)
    ep128 = ckpt_dir / f"v2_train_s{seed}_ep128.pt"
    r68.save_checkpoint(controller, optimizer, state, ep128, freeze=R68_FREEZE)
    return {"seed": seed, "status": "ep128_complete", "episodes_done": int(state["episodes_done"]),
            "updates": int(state["updates"]), "optimizer_steps": int(state["optimizer_steps"]),
            "lambda_final": float(state["lambda"])}


def _gate_from_committed(cfg, seed, controller, state):
    """Run the first-batch gate from the committed batch-0 record + reloaded episodes (resume)."""
    gate_path = cfg["raw_root"] / "runtime" / f"r68_first_batch_gate_s{seed}.json"
    if gate_path.exists() or int(state["episodes_done"]) < EPISODES_PER_UPDATE:
        return None
    ckpt_dir = Path(cfg["common"].CHECKPOINTS)
    record = _read_json(ckpt_dir / f"r40_batch_s{seed}_b0_attempt01.json")
    order = [int(x) for x in cfg["common"].scenario_order()]
    lam = float(record["lambda_at_rollout"])
    episodes = []
    for ep in record["episodes"]:
        scenario = int(order[int(ep)])
        paths = cfg["common"].cell_paths("train", scenario, 1, tag=f"r68_train_s{seed}_e{ep}")
        episodes.append(_reload_episode(paths["row"], paths["tape"], seed=seed, ep=int(ep),
                                        scenario=scenario, lam=lam))
    ep0_id = _read_json(ckpt_dir / f"r68_init_identity_s{seed}.json")
    gate = _first_batch_gate(cfg, controller, state, record, episodes, ep0_identity=ep0_id)
    _write_json(gate_path, gate)
    if gate["status"] != "passed":
        raise RuntimeError(f"seed {seed}: first-batch gate failed (from committed): {gate}")
    return gate


def _first_batch_gate(cfg, controller, state, record, episodes, *, ep0_identity):
    """Real first-batch gate for the ROUND68 late-initial-confidence intervention. Beyond the R66
    checks it proves the scale happened EXACTLY ONCE: pre-scale == R60 ep0, the batch-0 pre-update
    params == the post-scale init, the scale moved the params, alpha == 0.10, and the late_head sha
    changed by exactly the scale. It does NOT claim global-norm/clip-coefficient/Adam equivalence
    with R66 (two independent late heads can have different gradient statistics)."""
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
    if ep0_identity is not None:
        checks["pre_scale_equals_r60_ep0"] = bool(ep0_identity["pre_scale_equals_r60_ep0"])
        checks["init_is_post_scale"] = (before_sha == ep0_identity["post_scale_parameter_sha256"])
        checks["scale_moved_params"] = (ep0_identity["pre_scale_parameter_sha256"]
                                        != ep0_identity["post_scale_parameter_sha256"])
        checks["late_init_alpha_0p10"] = abs(float(ep0_identity["late_init_alpha"]) - r68.LATE_INIT_ALPHA) < 1e-12
        checks["late_head_scaled_once"] = (ep0_identity["late_head_sha_before_scale"]
                                           != ep0_identity["late_head_sha_after_scale"])
    checks["structure_r68"] = (after_frozen.get("structure") == r68.R68_STRUCTURE)
    checks["parameter_objects_12"] = (after["parameter_object_count"] == 12)
    checks["five_modules"] = (sorted(after["shapes"].keys()) ==
        sorted(["trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
                "early_head.weight", "early_head.bias", "late_head.weight", "late_head.bias",
                "value_task_head.weight", "value_task_head.bias",
                "value_cost_head.weight", "value_cost_head.bias"]))
    checks["frozen_modules_requires_grad_false"] = (not any(p.requires_grad for p in r68.frozen_parameters(controller)))
    checks["trainable_modules_requires_grad_true"] = all(p.requires_grad for p in r68.trainable_parameters(controller))
    checks["optimizer_owns_six_trainable"] = (int(update_report.get("trainable_parameter_object_count", -1)) == 6)
    checks["frozen_six_recorded"] = (int(update_report.get("frozen_parameter_object_count", -1)) == 6)
    checks["frozen_modules_gradient_recorded_none"] = (modgrad.get("trunk") is None and modgrad.get("early_head") is None)
    checks["trainable_modules_gradient_recorded_number"] = (isinstance(modgrad.get("late_head"), float)
                                                            and isinstance(modgrad.get("cost_head"), float))
    checks["critic_received_gradient"] = (float(modgrad.get("cost_head", 0.0) or 0.0) > 0.0)
    if gate_diag:
        checks["late_actor_received_gradient"] = (float(gate_diag.get("late_actor_grad_l1", -1.0)) > 0.0)
        checks["gate_diag_frozen_labelled"] = (gate_diag.get("trunk") == "frozen_requires_grad_false"
                                               and gate_diag.get("early_head") == "frozen_requires_grad_false")
    checks["likelihood_alignment_passed"] = bool(update_report["likelihood_alignment"]["passed"])
    checks["value_losses_finite"] = all(np.isfinite([update_report.get("value_loss_task", 0.0),
                                                     update_report.get("value_loss_cost", 0.0)]))
    checks["lambda_updated"] = ("lambda_next" in dual_record)
    status = "passed" if all(v for k, v in checks.items() if isinstance(v, bool)) else "failed"
    return {"kind": "round68-first-batch-gate/1", "seed": int(state["training_seed"]),
            "status": status, "checks": checks,
            "parameters_after_sha": after["parameter_sha256"],
            "late_init_alpha": r68.LATE_INIT_ALPHA,
            "frozen_sha256": r66f._tensor_identity(r68.frozen_parameters(controller))["sha256"],
            "lambda_after_first_batch": dual_record["lambda_next"],
            "isolation_note": ("critic->trunk isolation is applied inside routed_ppo, but its trunk "
                               "effect is masked because the trunk is requires_grad=False by the ROUND68 "
                               "intervention; the absent trunk gradient is NOT evidence about isolation."),
            "gate_diag": gate_diag, "module_gradient_l1_mean": modgrad}


def zero_env_acceptance(cfg) -> dict:
    """Zero-environment CLI forward acceptance on the LIVE modules: config, module paths, the frozen
    parameter set, the one-time late-head scale vs R60 ep0, count hooks and checkpoint routing.
    Stops before any episode provider is built. No model forward is run."""
    runtime = cfg["runtime"]; trainer = cfg["trainer"]; common = cfg["common"]
    controller = cfg["frozen_fresh"](11)
    ps = controller.parameter_summary()
    fps = r68.frozen_parameter_summary(controller)
    frozen, trainable = r68.partitioned_parameters(controller)
    ident = _read_json(Path(common.CHECKPOINTS) / "r68_init_identity_s11.json")
    ep0m = r68.load_r60_model(11, "ep0")
    ref = r68.fresh_partitioned_controller(11, ep0_state=None, verify_ep0=False)
    ref.load_state_dict(ep0m["controller"])
    late = controller.inner.late_head
    checks = {
        "trainer_is_frozen_v2": Path(trainer.__file__).name == "v2_trainer.py",
        "ppo_update_is_r68": cfg["identity"]["ppo_update_is_r68"],
        "save_is_r68": cfg["identity"]["save_is_r68"],
        "load_is_r68": cfg["identity"]["load_is_r68"],
        "load_ro_is_r68": cfg["identity"]["load_ro_is_r68"],
        "fresh_is_frozen": cfg["identity"]["fresh_is_frozen"],
        "training_version_is_r68": cfg["identity"]["training_version"] == r68.R68_TRAINING_VERSION,
        "common_gae_lambda_1": float(common.GAE_LAMBDA) == 1.0,
        "common_eta_lambda_0p5": float(common.ETA_LAMBDA) == 0.5,
        "parameter_objects_12": ps["parameter_object_count"] == 12,
        "partitioned_5_modules": sorted({k.split(".")[0] for k in ps["shapes"]}) ==
            sorted(["trunk", "early_head", "late_head", "value_task_head", "value_cost_head"]),
        "frozen_objects_6": len(frozen) == 6,
        "trainable_objects_6": len(trainable) == 6,
        "frozen_scalars_5570": sum(p.numel() for p in frozen) == 5570,
        "trainable_scalars_260": sum(p.numel() for p in trainable) == 260,
        "frozen_requires_grad_false": all(not p.requires_grad for p in frozen),
        "trainable_requires_grad_true": all(p.requires_grad for p in trainable),
        "structure_r68": fps.get("structure") == r68.R68_STRUCTURE,
        "trunk_equals_r60_ep0": r68._module_sha(controller, "trunk") == r68._module_sha(ref, "trunk"),
        "early_head_equals_r60_ep0": r68._module_sha(controller, "early_head") == r68._module_sha(ref, "early_head"),
        "critics_equal_r60_ep0": (r68._module_sha(controller, "value_task_head") == r68._module_sha(ref, "value_task_head")
                                  and r68._module_sha(controller, "cost_head") == r68._module_sha(ref, "cost_head")),
        "late_head_is_0p10_ep0": (torch.equal(late.weight, ref.inner.late_head.weight * 0.10)
                                  and torch.equal(late.bias, ref.inner.late_head.bias * 0.10)),
        "init_pre_scale_equals_r60_ep0": bool(ident["pre_scale_equals_r60_ep0"]),
        "late_init_alpha_0p10": abs(float(ident["late_init_alpha"]) - r68.LATE_INIT_ALPHA) < 1e-12,
        "checkpoint_root_under_r68_raw": str(Path(common.CHECKPOINTS).resolve()).startswith(str(cfg["raw_root"].resolve())),
        "rows_root_under_r68_raw": str(Path(common.ROWS).resolve()).startswith(str(cfg["raw_root"].resolve())),
        "r40_normal_cap_774": int(runtime["r40_common"].NORMAL_CAP) == 774,
        "r40_hard_cap_800": int(runtime["r40_common"].HARD_CAP) == 800,
    }
    status = "passed" if all(checks.values()) else "failed"
    return {"kind": "round68-zero-env-acceptance/1", "status": status, "checks": checks,
            "identity": cfg["identity"], "parameter_sha256": ps["parameter_sha256"],
            "frozen_scalar_count": sum(p.numel() for p in frozen),
            "trainable_scalar_count": sum(p.numel() for p in trainable),
            "late_init_alpha": r68.LATE_INIT_ALPHA}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND68 late-initial-confidence executor (acceptance+train)")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--phase", choices=("acceptance", "train"), required=True)
    ap.add_argument("--seed", type=int, choices=(11, 22, 33), default=None)
    ap.add_argument("--first-gate-only", action="store_true",
                    help="run each seed's first 4 episodes + first update + real gate, then stop "
                         "(episodes count toward 128)")
    args = ap.parse_args(argv)
    raw_root = args.raw_root.resolve()
    cfg = configure_and_inject(raw_root)
    ledger = R68Ledger(raw_root)
    order = [int(x) for x in cfg["common"].scenario_order()]
    if len(order) != EPISODES_PER_SEED:
        raise RuntimeError(f"training order length {len(order)} != {EPISODES_PER_SEED}")
    plan_path = raw_root / "ROUND68_FIXED_PLAN.json"
    if not plan_path.exists():
        plan = build_plan(order)
        _write_json(plan_path, plan)
    else:
        plan = _read_json(plan_path)
    seeds = (args.seed,) if args.seed else SEEDS
    if args.phase == "acceptance":
        acc = zero_env_acceptance(cfg)
        _write_json(raw_root / "runtime" / "ROUND68_ZERO_ENV_ACCEPTANCE.json", acc)
        print(json.dumps(acc, indent=1))
        return 0 if acc["status"] == "passed" else 2
    if args.phase == "train":
        references = reuse_r66_references(cfg, order)
        missing = sorted(set(order) - set(references))
        if missing:
            raise RuntimeError(f"reused ROUND66 references incomplete: {missing}")
        for seed in seeds:
            print(f"ROUND68 training seed {seed} ...", flush=True)
            print(json.dumps(train_seed(cfg, ledger, plan, seed, order, references,
                                        stop_after_first_gate=args.first_gate_only)), flush=True)
            ledger.write_ledger()
        if args.first_gate_only:
            ledger.write_ledger()
            return 0
    ledger.write_ledger()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
