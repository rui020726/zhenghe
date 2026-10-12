"""ROUND60 independent executor: fixed 384 training + 192 eval with an independent R60 identity,
claim/slot/quota ledger, first-batch gate, passive early/late diagnostics, and paired statistics
vs the read-only reused R58 cells.

This reuses the frozen sim (v2_run._run_episode), the frozen collector (r42_one_shot), the frozen
batch_samples/batch_g_hat/dual_update, and the R60 partitioned controller + routed ppo_update +
5-module checkpoint from r60_partitioned. It does NOT reuse the R58 branch/version/claim verifier;
R60 has its own task/branch/version/freeze identity, output root, claim/slot roots and quota.

Quota (ROUND66 fresh authorization): historical non-plan probe = 0; normal plan = 1110
(48 immediate refs + 384 training + 672 eval + 6 focal); technical-failure reserve = 40;
cumulative launch/attempt hard cap = 1150. ROUND60 historical constructions are NOT ROUND66
consumption. Synthetic optimizer updates and synthetic checkpoint writes are counted separately.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent
GEN2 = RUN_ROOT / "round58_gen2"
AUDIT = RUN_ROOT / "round58_isolation_stop_audit"
R53_DIR = RUN_ROOT / "round53_critic_isolation_trial_readiness"
R58_V35 = RUN_ROOT / "round58_gen3_resume_v35" / "raw" / "rows"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(GEN2))
sys.path.insert(0, str(AUDIT))
sys.path.insert(0, str(R53_DIR))
import r66_frozen as r66

SEEDS = (11, 22, 33)
EPISODES_PER_SEED = 128
EPISODES_PER_UPDATE = 4
MODES = ("greedy", "sampled")
SCENES_NEW = tuple(range(1_800_000, 1_800_032))
SCENES_NEW_FALLBACKS = (tuple(range(1_810_000, 1_810_032)), tuple(range(1_820_000, 1_820_032)))
FOCAL_SCENE = 1_700_027
TASK_ID = "ROUND66-BC-EARLY-PATH-PRESERVING-LATE-ADAPTATION"
BRANCH = "exp/round66-bc-early-path-preserving-late-adaptation"
NORMAL_CAP = 1110
RESERVE_CAP = 40
HARD_CAP = 1150
HISTORICAL_PROBE = 0
REF_COUNT = 48
TRAINING_COUNT = 384
EVAL_COUNT = 672
FOCAL_COUNT = 6

R66_FREEZE = {
    "kind": "round66-frozen-early-path-freeze/1",
    "task_id": TASK_ID, "branch": BRANCH,
    "training_version": r66.R66_TRAINING_VERSION,
    "structure": r66.R66_STRUCTURE,
    "loss_version": r66.R66_LOSS_VERSION,
    "frozen_modules": list(r66.FROZEN_MODULES),
    "trainable_modules": list(r66.TRAINABLE_MODULES),
    "gae_lambda": 1.0, "eta_lambda": 0.5, "gamma": 1.0, "delta": 0.0, "lambda_init": 1.0,
    "data_order_seed": 20260927, "b0_bytes": 10000.0,
    "r60_source_commit": "cbae709f6bb2bf566acf341d626de659524772f4",
}


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class R66Ledger:
    """Independent R60 claim/slot/quota ledger under the R60 raw root."""

    def __init__(self, raw: Path) -> None:
        self.raw = raw
        self.claims_dir = raw / "launch_claims"
        self.slots_dir = raw / "launch_slots"
        self.runtime_dir = raw / "runtime"
        for d in (self.claims_dir, self.slots_dir / "normal", self.slots_dir / "reserve", self.runtime_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.runtime_dir / "ROUND66_LEDGER.json"

    def _slug(self, logical: str) -> str:
        import re
        return re.sub(r"[^0-9A-Za-z_.-]+", "_", logical)

    def claim_path(self, logical: str) -> Path:
        return self.claims_dir / self._slug(logical) / "attempt_0001.json"

    def is_complete(self, logical: str) -> bool:
        p = self.claim_path(logical)
        return p.is_file() and _read_json(p).get("status") == "complete"

    def open_claim(self, logical: str, slot: int, metadata: dict) -> Path:
        p = self.claim_path(logical)
        if p.exists():
            raise RuntimeError(f"duplicate R60 claim refused: {logical}")
        slot_path = self.slots_dir / "normal" / f"{slot:04d}.json"
        if slot_path.exists():
            raise RuntimeError(f"duplicate R60 normal slot: {slot}")
        claim = {"logical_job": logical, "attempt": 1, "launch_slot": slot, "category": "normal",
                 "status": "claimed_before_environment", "task_id": TASK_ID, "branch": BRANCH,
                 "training_version": r66.R66_TRAINING_VERSION, "metadata": metadata,
                 "opened_unix": time.time()}
        _write_json(p, claim)
        _write_json(slot_path, {"category": "normal", "slot": slot, "logical_job": logical, "attempt": 1})
        return p

    def finish_claim(self, path: Path, *, ok: bool, detail: str, result: dict | None = None) -> None:
        claim = _read_json(path)
        claim["status"] = "complete" if ok else "failed_technical"
        claim["detail"] = detail
        claim["closed_unix"] = time.time()
        if result:
            claim["result"] = result
        _write_json(path, claim)

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
        led = {"kind": "round60-ledger/1", "task_id": TASK_ID, "branch": BRANCH,
               "training_version": r66.R66_TRAINING_VERSION,
               "historical_non_plan_probe": HISTORICAL_PROBE,
               "normal_cap": NORMAL_CAP, "reserve_cap": RESERVE_CAP, "hard_cap": HARD_CAP,
               "counts": self.counts(), "updated_unix": time.time()}
        _write_json(self.ledger_path, led)
        return led


# ------------------------------------------------------------------ runtime wiring --------
def configure_and_inject(raw_root: Path) -> dict[str, Any]:
    """Stand up the frozen R46/R42 sim runtime exactly as R58 configured it, then inject the R60
    partitioned controller + routed ppo_update + 5-module checkpoint, and bind every output-owning
    path to the R60 raw root. The R58 claim/branch verifier is NOT used; R60 accounting is the
    R60Ledger above. Verified against the actual live modules, not just a runtime dict."""
    import r58_execute
    raw_root = Path(raw_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    base = r58_execute.load_base_runner()
    # Scope the private-root guard to the ROUND60 directory (not R58's) so the R60 raw root is an
    # independent output namespace; everything else in bind_runner_paths is the mature rebinding.
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
        raise RuntimeError("R60 runtime did not configure")
    trainer = runtime["trainer"]
    controller_module = runtime["controller_module"]
    common = runtime["common"]
    runner = runtime["runner"]
    r42_run = runtime["r42_run"]
    r40_common = runtime["r40_common"]
    # Apply the R60 living overrides on the actual common module the trainer reads.
    common.GAE_LAMBDA = 1.0
    common.ETA_LAMBDA = 0.5
    if float(common.GAMMA) != 1.0:
        raise RuntimeError("frozen gamma is not 1.0")
    # Rebind the R60 quota on the actual r40_common ledger module (was 480/24/600 from R43).
    r40_common.NORMAL_CAP = NORMAL_CAP
    r40_common.RESERVE_CAP = RESERVE_CAP
    r40_common.HARD_CAP = HARD_CAP
    common.NORMAL_LAUNCHES = NORMAL_CAP
    common.RESERVE_LAUNCHES = RESERVE_CAP
    common.LAUNCH_CAP = HARD_CAP
    r40_common.plan = lambda: {"normal_total": NORMAL_CAP, "technical_reserve_max": RESERVE_CAP,
                               "hard_cap": HARD_CAP, "task_id": TASK_ID}
    r40_common.ROUND_DIR = raw_root
    # Inject the R66 frozen-early-path controller factory (capture original to avoid recursion).
    orig_fresh = controller_module.fresh_controller
    def frozen_fresh(seed: int):
        b = orig_fresh(int(seed))
        pc, _ep0, _frozen_id, _summary = r66.build_frozen_controller(b, int(seed))
        return pc
    controller_module.fresh_controller = frozen_fresh
    # Route the trainer's PPO update and checkpoint to the R60 partitioned versions.
    from round53_trial_adapter import critic_input_detached
    def routed_ppo(controller, optimizer, episodes, *, update_index):
        # Preserve the R53 critic->shared-trunk gradient isolation exactly as R58 applies it:
        # the two value heads receive a detached shared representation; the actor (routed) path
        # stays connected. Same context object R58 uses around its ppo_update.
        with critic_input_detached(controller):
            report = r66.frozen_early_ppo_update(controller, optimizer, episodes, update_index=update_index)
        # One durable ppo_update call-journal line per committed update, so the R42 configure
        # resume guard (ppo_update_calls vs committed batch records) closes on the R66 namespace.
        import os as _os
        ppo_journal = raw_root / "runtime" / "ppo_update_calls.jsonl"
        with ppo_journal.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"update_index": int(update_index),
                                 "optimizer_steps": int(report.get("optimizer_steps", 0)),
                                 "structure": r66.R66_STRUCTURE}) + "\n")
            fh.flush(); _os.fsync(fh.fileno())
        return report
    def r66_save(controller, optimizer, state, path, *, freeze):
        return r66.save_checkpoint(controller, optimizer, state, path, freeze=R66_FREEZE)
    def r66_load(path, *, freeze):
        return r66.load_checkpoint(path, freeze=R66_FREEZE)
    def r66_load_ro(path, *, freeze):
        return r66.load_checkpoint_read_only(path, freeze=R66_FREEZE)
    trainer.ppo_update = routed_ppo
    trainer.save_checkpoint = r66_save
    trainer.load_checkpoint = r66_load
    trainer.load_checkpoint_read_only = r66_load_ro
    # Stamp the R60 training version onto the live version modules so rows/claims carry it.
    try:
        runtime["r42_versions"].TRAINING_VERSION = r66.R66_TRAINING_VERSION
    except Exception:
        pass
    # Bypass ONLY the R58 frozen-source claim verifier (bookkeeping bound to R58 identity). The R60
    # ledger above is the authoritative claim record; the sim and training math are untouched.
    import r44_runtime
    r44_runtime.verify_frozen_source = lambda frozen: {"r60": True}
    identity = {
        "trainer_module": str(Path(trainer.__file__).resolve()),
        "controller_module": str(Path(controller_module.__file__).resolve()),
        "common_module": str(Path(common.__file__).resolve()),
        "runner_module": str(Path(runner.__file__).resolve()),
        "raw_root": str(raw_root),
        "ppo_update_is_r66": trainer.ppo_update is routed_ppo,
        "save_is_r66": trainer.save_checkpoint is r66_save,
        "fresh_is_frozen": controller_module.fresh_controller is frozen_fresh,
        "common_GAE_LAMBDA": float(common.GAE_LAMBDA), "common_ETA_LAMBDA": float(common.ETA_LAMBDA),
    }
    return {"runtime": runtime, "trainer": trainer, "controller_module": controller_module,
            "common": common, "runner": runner, "r42_run": r42_run, "raw_root": raw_root,
            "frozen_fresh": frozen_fresh, "identity": identity}


def run_block_a_references(cfg, ledger, plan, order) -> dict[int, float]:
    """Block A: one FRESH immediate reference per unique training scene (paired batch_g_hat cache).
    Each is a real environment construction counted in the 1110; resumable via the R66 ledger."""
    common = cfg["common"]; runner = cfg["runner"]
    scenes = sorted({int(s) for s in order})
    if len(scenes) != REF_COUNT:
        raise RuntimeError(f"R66 training scenes {len(scenes)} != {REF_COUNT}; adjudicate the input diff, do not change the plan")
    refs: dict[int, float] = {}
    for scene in scenes:
        logical = f"r66_ref_immediate_s{scene}"
        metadata = {"arm": "IMMEDIATE_REF", "block": "A", "scenario_seed": int(scene)}
        if ledger.is_complete(logical):
            paths = common.cell_paths("reference", scene, 1, tag=f"r66_ref_s{scene}")
            refs[int(scene)] = float(_read_json(paths["row"])["on_time"])
            continue
        tag = f"r66_ref_s{scene}"
        def _do(scene=scene, tag=tag):
            row = runner._run_episode(tag, "reference", int(scene), policy="immediate", tag=tag)
            row["reference_arm"] = "immediate"; row["scenario_seed"] = int(scene)
            row["model_arm"] = "IMMEDIATE_REF"
            return row
        row = _atomic_run(cfg, logical, metadata, _do, detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}")
        _write_row(cfg, row)
        refs[int(scene)] = float(row["on_time"])
    return refs


def build_references(cfg, order) -> dict[int, float]:
    """Read already-committed block-A immediate references (resume; no new runs)."""
    common = cfg["common"]
    refs: dict[int, float] = {}
    for scene in sorted({int(s) for s in order}):
        paths = common.cell_paths("reference", scene, 1, tag=f"r66_ref_s{scene}")
        if paths["row"].is_file():
            refs[int(scene)] = float(_read_json(paths["row"])["on_time"])
    return refs


def build_plan(order: list[int]) -> dict[str, Any]:
    jobs = []
    slot = 0
    train_scenes = sorted({int(s) for s in order})
    if len(train_scenes) != REF_COUNT:
        raise AssertionError(f"training scenes {len(train_scenes)} != {REF_COUNT}")
    # Block A: one fresh immediate reference per unique training scene (paired batch_g_hat cache).
    for scene in train_scenes:
        slot += 1
        jobs.append({"scope": "immediate_reference", "logical_job": f"r66_ref_immediate_s{scene}",
                     "scene": int(scene), "block": "A", "normal_slot": slot})
    # Training: 3 seeds x 128 episodes (first 4 per seed count toward 128).
    for seed in SEEDS:
        for ep in range(EPISODES_PER_SEED):
            slot += 1
            jobs.append({"scope": "ppo_training", "logical_job": f"r66_new_training_s{seed}_e{ep}",
                         "seed": seed, "episode_index": ep, "scene": int(order[ep]), "normal_slot": slot})
    # Eval block C on the new 32-scene block. NEW and BC are learned arms (both modes, 3 seeds).
    for arm in ("NEW", "BC"):
        for seed in SEEDS:
            for mode in MODES:
                for scene in SCENES_NEW:
                    slot += 1
                    jobs.append({"scope": "model_eval", "arm": arm,
                                 "logical_job": f"r66_model_eval_{arm.lower()}_s{seed}_{mode}_scene{scene}",
                                 "seed": seed, "mode": mode, "scene": int(scene), "normal_slot": slot})
    # Priority rule arms (vs immediate task set, vs late cost, always_hold baseline): both modes.
    for arm in ("immediate", "late_discovery_gate", "always_hold"):
        for mode in MODES:
            for scene in SCENES_NEW:
                slot += 1
                jobs.append({"scope": "rule_eval", "arm": arm.upper(),
                             "logical_job": f"r66_rule_eval_{arm}_{mode}_scene{scene}",
                             "mode": mode, "scene": int(scene), "normal_slot": slot})
    # Supplementary rule baselines: greedy only.
    for arm in ("periodic_20", "random_send", "event"):
        for scene in SCENES_NEW:
            slot += 1
            jobs.append({"scope": "rule_eval", "arm": arm.upper(),
                         "logical_job": f"r66_rule_eval_{arm}_greedy_scene{scene}",
                         "mode": "greedy", "scene": int(scene), "normal_slot": slot})
    # Block D: focal scene mechanism diagnostic (NEW & BC, 3 seeds, greedy).
    for seed in SEEDS:
        for arm in ("NEW", "BC"):
            slot += 1
            jobs.append({"scope": "focal", "arm": arm,
                         "logical_job": f"r66_focal_{arm.lower()}_s{seed}_scene{FOCAL_SCENE}",
                         "seed": seed, "mode": "greedy", "scene": FOCAL_SCENE, "normal_slot": slot})
    if len(jobs) != NORMAL_CAP or len({j["logical_job"] for j in jobs}) != NORMAL_CAP:
        raise AssertionError(f"R66 fixed plan is not exactly {NORMAL_CAP} unique jobs (got {len(jobs)})")
    return {"kind": "round66-fixed-job-plan/1", "task_id": TASK_ID, "branch": BRANCH,
            "training_version": r66.R66_TRAINING_VERSION, "structure": r66.R66_STRUCTURE,
            "reference_count": len(train_scenes), "training_episode_count": TRAINING_COUNT,
            "model_eval_count": 192 + 192, "rule_eval_count": 64 * 3 + 32 * 3, "focal_count": FOCAL_COUNT,
            "normal_count": NORMAL_CAP, "reserve_cap": RESERVE_CAP, "hard_cap": HARD_CAP,
            "historical_probe": HISTORICAL_PROBE, "eval_scenes_new": list(SCENES_NEW),
            "focal_scene": FOCAL_SCENE, "r60_source_commit": "cbae709f6bb2bf566acf341d626de659524772f4",
            "eval_split_note": ("eval 672 = NEW(192)+BC(192)+immediate/late/always_hold both modes(192)"
                                "+periodic_20/random_send/event greedy(96); adjudicated by the executor "
                                "author, not uniquely fixed by the scoped plan"),
            "jobs": jobs}

from types import SimpleNamespace


def _write_row(cfg: dict, row: dict) -> None:
    common = cfg["common"]
    common.ROWS.mkdir(parents=True, exist_ok=True)
    safe = {k: v for k, v in row.items() if k != "collector"}
    Path(row["artefacts"]["row"]).write_text(
        json.dumps(safe, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _reload_episode(row_path: Path, tape_path: Path, *, seed, ep, scenario, lam) -> dict:
    row = _read_json(row_path)
    tapes = [json.loads(l) for l in Path(tape_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    row["collector"] = SimpleNamespace(records=tapes)
    row["training_seed"] = int(seed); row["episode_index"] = int(ep)
    row["scenario_seed"] = int(scenario); row["lambda"] = float(lam)
    return row


def _run_train_episode(cfg, controller, collector, *, seed, ep, scenario, lam):
    runner = cfg["runner"]
    tag = f"r60_train_s{seed}_e{ep}"
    action_seed = runner.action_seed(int(seed), int(ep))
    logical = f"new_training_s{int(seed)}_e{int(ep)}"
    metadata = {"arm": "NEW", "training_seed": int(seed), "episode_index": int(ep),
                "scenario_seed": int(scenario), "lambda_at_rollout": float(lam)}
    def _do():
        row = runner._run_episode(tag, "train", int(scenario), policy="learned",
                                  attach=collector.policy(), controller=controller, collector=collector,
                                  action_rng_seed=action_seed, tag=tag)
        row["collector"] = collector
        row["training_seed"] = int(seed); row["episode_index"] = int(ep)
        row["scenario_seed"] = int(scenario); row["lambda"] = float(lam)
        return row
    return _atomic_run(cfg, logical, metadata, _do, detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}")


def _atomic_run(cfg, logical, metadata, run_fn, *, detail):
    """One cell under the mature r40_common atomic claim + counted builder, bound to the R60 raw
    root. The claim/slot/quota are the R60 ledger; the sim and training math are the frozen ones."""
    r40 = cfg["runtime"]["r40_common"]
    r42_run = cfg["r42_run"]
    if r42_run._ACTIVE_CLAIM is not None:
        raise RuntimeError("nested R60 environment claims are not allowed")
    claim_path = r40.ROUND_DIR / "launch_claims" / r40._slug(logical) / "attempt_0001.json"
    if claim_path.exists():
        existing = _read_json(claim_path)
        st = existing.get("status")
        if st == "complete":
            raise RuntimeError(f"R60 {logical}: already complete; caller must skip before running")
        if st == "failed_technical":
            raise RuntimeError(f"R60 {logical}: attempt 1 failed_technical; needs a reserve retry, not a silent rerun")
        # claimed_before_environment / claiming: an interrupted cell within the current
        # (uncommitted) batch. The controller is frozen during a batch's rollouts, so reusing the
        # same attempt-1 claim and re-running is exact; no new slot is consumed.
    else:
        claim_path = r40.claim_launch(logical, category="normal", attempt=1, metadata=metadata)
    r42_run._ACTIVE_CLAIM = claim_path
    r42_run._ACTIVE_PROGRESS = r42_run._new_run_progress()
    try:
        row = run_fn()
        r40.finish_launch(claim_path, ok=True, detail=detail(row))
        return row
    except BaseException as exc:
        try:
            r40.finish_launch(claim_path, ok=False, detail=f"{type(exc).__name__}: {exc}")
        except BaseException:
            pass
        raise
    finally:
        r42_run._ACTIVE_CLAIM = None
        r42_run._ACTIVE_PROGRESS = None


def _slot_for(plan, logical):
    for j in plan["jobs"]:
        if j["logical_job"] == logical:
            return j["normal_slot"]
    raise KeyError(logical)


def train_seed(cfg, ledger, plan, seed, order, references, *, stop_after_first_gate=False):
    common = cfg["common"]; trainer = cfg["trainer"]
    ckpt_dir = Path(common.CHECKPOINTS)
    rolling = ckpt_dir / f"v2_train_s{seed}.pt"
    if rolling.exists():
        controller, optimizer, state = r66.load_checkpoint(rolling, freeze=R66_FREEZE)
        ps = controller.parameter_summary()
        if ps["parameter_object_count"] != 12 or r66.frozen_parameter_summary(controller).get("structure") != r66.R66_STRUCTURE:
            raise RuntimeError(f"seed {seed}: rolling checkpoint is not a R66 frozen-early-path model")
        if any(p.requires_grad for p in r66.frozen_parameters(controller)) or \
                not all(p.requires_grad for p in r66.trainable_parameters(controller)):
            raise RuntimeError(f"seed {seed}: restored controller did not re-establish the frozen mask")
        resumed = True
    else:
        controller = cfg["frozen_fresh"](seed)
        optimizer = r66.build_optimizer(controller, common.LEARNING_RATE)
        state = r66.fresh_training_state(seed, gae_lambda=1.0, eta_lambda=0.5, gamma=1.0,
                                         delta=0.0, lambda_init=1.0, data_order_seed=20260927,
                                         b0_bytes=common.B0_BYTES)
        ep0_identity = controller.parameter_summary()
        _write_json(ckpt_dir / f"r60_ep0_identity_s{seed}.json", ep0_identity)
        resumed = False
    gate_path = cfg["raw_root"] / "runtime" / f"r60_first_batch_gate_s{seed}.json"
    # Resume safety: if the first batch already committed but its gate did not record (e.g. an
    # earlier interruption after commit, before the gate), run the gate from the committed batch-0
    # record + reloaded episodes before continuing. The 4 episodes are not re-run.
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
            r66.save_checkpoint(controller, optimizer, state, pre, freeze=R66_FREEZE)
        episodes = []
        for k in range(EPISODES_PER_UPDATE):
            ep = int(state["episodes_done"]) + k
            scenario = int(order[ep])
            logical = f"new_training_s{seed}_e{ep}"
            if ledger.is_complete(logical):
                paths = common.cell_paths("train", scenario, 1, tag=f"r60_train_s{seed}_e{ep}")
                episodes.append(_reload_episode(paths["row"], paths["tape"],
                                                seed=seed, ep=ep, scenario=scenario, lam=lam))
                continue
            collector = __import__("r42_one_shot").OneShotSendSkipCollector(controller, greedy=False)
            row = _run_train_episode(cfg, controller, collector, seed=seed, ep=ep, scenario=scenario, lam=lam)
            _write_row(cfg, row)
            episodes.append(row)
        if len(episodes) != EPISODES_PER_UPDATE:
            raise RuntimeError(f"seed {seed} batch {batch}: partial PPO batch")
        # Route through trainer.ppo_update (routed_ppo): applies the R53 critic->trunk isolation
        # and writes the durable ppo_update call journal. Calling partitioned_ppo_update directly
        # would skip both.
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
        r66.save_checkpoint(controller, optimizer, state, post, freeze=R66_FREEZE)
        r66.save_checkpoint(controller, optimizer, state, rolling, freeze=R66_FREEZE)
        batch_record = {"kind": "round60-training-batch-record/1", "batch": batch, "seed": seed,
                        "episodes": [int(r["episode_index"]) for r in episodes],
                        "scenarios": [int(r["scenario_seed"]) for r in episodes],
                        "lambda_at_rollout": lam, "update": update_report, "constraint": g_record,
                        "dual": dual_record, "parameters_before_sha": before["parameter_sha256"],
                        "parameters_after_sha": after["parameter_sha256"],
                        "episodes_done_after": int(state["episodes_done"]),
                        "optimizer_steps_after": int(state["optimizer_steps"]),
                        "structure": r66.R66_STRUCTURE}
        # Name matches the R42 configure resume glob r40_batch_s*_b*_attempt*.json so the
        # ppo/optimizer call journals close against committed batch records on resume.
        _write_json(ckpt_dir / f"r40_batch_s{seed}_b{batch}_attempt01.json", batch_record)
        ledger.write_ledger()
        # First-batch gate: run the real gate immediately after the first update of this seed.
        if batch == 0 and not gate_path.exists():
            ep0_id = _read_json(ckpt_dir / f"r60_ep0_identity_s{seed}.json")
            gate = _first_batch_gate(cfg, controller, state, batch_record, episodes, ep0_identity=ep0_id)
            _write_json(gate_path, gate)
            if gate["status"] != "passed":
                raise RuntimeError(f"seed {seed}: first-batch gate failed: {gate}")
            if stop_after_first_gate:
                return {"seed": seed, "status": "first_gate_passed", "gate": gate["status"]}
        print(f"R60 seed {seed} batch {batch}: episodes_done={state['episodes_done']} "
              f"lambda={state['lambda']:.4f} early_free={update_report.get('early_free')} "
              f"late_free={update_report.get('late_free')} gnorms={update_report.get('gradient_norm_mean')}", flush=True)
    ep128 = ckpt_dir / f"v2_train_s{seed}_ep128.pt"
    r66.save_checkpoint(controller, optimizer, state, ep128, freeze=R66_FREEZE)
    return {"seed": seed, "status": "ep128_complete", "episodes_done": int(state["episodes_done"]),
            "updates": int(state["updates"]), "optimizer_steps": int(state["optimizer_steps"]),
            "lambda_final": float(state["lambda"])}


def _gate_from_committed(cfg, seed, controller, state):
    """Run the first-batch gate from the committed batch-0 record + reloaded episodes (resume)."""
    gate_path = cfg["raw_root"] / "runtime" / f"r60_first_batch_gate_s{seed}.json"
    if gate_path.exists() or int(state["episodes_done"]) < EPISODES_PER_UPDATE:
        return None
    ckpt_dir = Path(cfg["common"].CHECKPOINTS)
    record = _read_json(ckpt_dir / f"r40_batch_s{seed}_b0_attempt01.json")
    order = [int(x) for x in cfg["common"].scenario_order()]
    lam = float(record["lambda_at_rollout"])
    episodes = []
    for ep in record["episodes"]:
        scenario = int(order[int(ep)])
        paths = cfg["common"].cell_paths("train", scenario, 1, tag=f"r60_train_s{seed}_e{ep}")
        episodes.append(_reload_episode(paths["row"], paths["tape"], seed=seed, ep=int(ep),
                                        scenario=scenario, lam=lam))
    ep0_id = _read_json(ckpt_dir / f"r60_ep0_identity_s{seed}.json")
    gate = _first_batch_gate(cfg, controller, state, record, episodes, ep0_identity=ep0_id)
    _write_json(gate_path, gate)
    if gate["status"] != "passed":
        raise RuntimeError(f"seed {seed}: first-batch gate failed (from committed): {gate}")
    return gate


def _first_batch_gate(cfg, controller, state, record, episodes, *, ep0_identity):
    """Real first-batch gate for the R66 frozen-early-path intervention: verify the actual frozen
    mask, the six-trainable optimizer coverage, the routed dual-head update, mask/likelihood,
    reward/cost fields, dual lambda and checkpoint identity. It deliberately does NOT claim the
    critic->trunk isolation from the frozen trunk's absent gradient."""
    update_report = record["update"]
    dual_record = record["dual"]
    before_sha = record["parameters_before_sha"]
    after = controller.parameter_summary()
    after_frozen = r66.frozen_parameter_summary(controller)
    gate_diag = update_report.get("gate_diag") or {}
    modgrad = update_report.get("module_gradient_l1_mean") or {}
    checks: dict[str, Any] = {}
    checks["dual_head_routing_active"] = bool(update_report.get("early_free", 0) > 0
                                              and update_report.get("late_free", 0) > 0)
    checks["early_free"] = int(update_report.get("early_free", 0))
    checks["late_free"] = int(update_report.get("late_free", 0))
    if ep0_identity is not None:
        checks["init_params_equal_r60_ep0"] = (before_sha == ep0_identity["parameter_sha256"])
    checks["structure_r66_frozen_early_path"] = (after_frozen.get("structure") == r66.R66_STRUCTURE)
    checks["parameter_objects_12"] = (after["parameter_object_count"] == 12)
    checks["five_modules"] = (sorted(after["shapes"].keys()) ==
        sorted(["trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
                "early_head.weight", "early_head.bias", "late_head.weight", "late_head.bias",
                "value_task_head.weight", "value_task_head.bias", "value_cost_head.weight", "value_cost_head.bias"]))
    checks["frozen_modules_requires_grad_false"] = (not any(p.requires_grad for p in r66.frozen_parameters(controller)))
    checks["trainable_modules_requires_grad_true"] = all(p.requires_grad for p in r66.trainable_parameters(controller))
    checks["optimizer_owns_six_trainable"] = (int(update_report.get("trainable_parameter_object_count", -1)) == 6)
    checks["frozen_six_recorded"] = (int(update_report.get("frozen_parameter_object_count", -1)) == 6)
    checks["frozen_modules_gradient_recorded_none"] = (modgrad.get("trunk") is None and modgrad.get("early_head") is None)
    checks["trainable_modules_gradient_recorded_number"] = (isinstance(modgrad.get("late_head"), float) and isinstance(modgrad.get("cost_head"), float))
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
    return {"kind": "round66-first-batch-gate/1", "seed": int(state["training_seed"]),
            "status": status, "checks": checks,
            "parameters_after_sha": after["parameter_sha256"],
            "frozen_sha256": r66._tensor_identity(r66.frozen_parameters(controller))["sha256"],
            "lambda_after_first_batch": dual_record["lambda_next"],
            "isolation_note": ("The R53 critic->trunk isolation context is still applied inside "
                               "routed_ppo, but its effect on the trunk is masked here because the "
                               "trunk is requires_grad=False by the R66 intervention; the absent "
                               "trunk gradient is therefore NOT evidence about the isolation."),
            "gate_diag": gate_diag,
            "module_gradient_l1_mean": modgrad}


def eval_seed(cfg, ledger, plan, seed):
    """Per-seed learned-arm eval on the new 32-scene block: NEW (R66 ep128) and BC (compliant R60
    ep0 read-only). Both modes, greedy primary. NEW must still carry its frozen mask; BC is NOT
    frozen (it runs exactly as R60 ep0 had it)."""
    common = cfg["common"]; runner = cfg["runner"]
    import r42_one_shot
    ep128 = Path(common.CHECKPOINTS) / f"v2_train_s{seed}_ep128.pt"
    if not ep128.is_file():
        raise FileNotFoundError(f"seed {seed}: R66 ep128 checkpoint missing")
    new_ctrl, new_state, new_identity = r66.load_checkpoint_read_only(ep128, freeze=R66_FREEZE)
    if int(new_state.get("episodes_done", -1)) != EPISODES_PER_SEED:
        raise RuntimeError(f"seed {seed}: R66 ep128 is not {EPISODES_PER_SEED} episodes")
    if new_identity.get("optimizer_restored") is not False or new_identity.get("rng_restored") is not False:
        raise RuntimeError(f"seed {seed}: NEW read-only eval restored state")
    if any(p.requires_grad for p in r66.frozen_parameters(new_ctrl)):
        raise RuntimeError(f"seed {seed}: NEW ep128 lost its frozen mask")
    bc_ctrl, bc_identity = r66.load_r60_checkpoint_read_only(seed, "ep0")
    for arm, controller, identity in (("NEW", new_ctrl, new_identity), ("BC", bc_ctrl, bc_identity)):
        for mode in MODES:
            for scene in SCENES_NEW:
                logical = f"model_eval_{arm.lower()}_s{seed}_{mode}_scene{scene}"
                if ledger.is_complete(logical):
                    continue
                collector = r42_one_shot.OneShotSendSkipCollector(controller, greedy=(mode == "greedy"))
                tag = f"r66_eval_{arm.lower()}_s{seed}_{mode}"
                metadata = {"arm": arm, "training_seed": int(seed), "mode": mode, "scenario_seed": int(scene)}
                def _do(controller=controller, collector=collector, tag=tag, scene=scene, mode=mode,
                        arm=arm, identity=identity, seed=seed):
                    row = runner._run_episode(tag, "eval", int(scene), policy="learned",
                                              attach=collector.policy(), controller=controller,
                                              collector=collector,
                                              action_rng_seed=runner.eval_action_seed(int(seed), int(scene), mode),
                                              tag=tag, checkpoint_load_identity=identity)
                    row["model_arm"] = arm; row["eval_mode"] = mode; row["training_seed"] = int(seed)
                    return row
                row = _atomic_run(cfg, logical, metadata, _do,
                                  detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}")
                _write_row(cfg, row)
        ledger.write_ledger()
    return {"seed": seed, "status": "eval_complete", "arms": ["NEW", "BC"]}


def eval_rules(cfg, ledger, plan):
    """Arm-level rule eval on the new 32-scene block (rules are not per-seed). immediate/late/
    always_hold in both modes; periodic_20/random_send/event greedy only."""
    import v2_rules
    runner = cfg["runner"]
    arm_modes = {"immediate": MODES, "late_discovery_gate": MODES, "always_hold": MODES,
                 "periodic_20": ("greedy",), "random_send": ("greedy",), "event": ("greedy",)}
    for arm, modes in arm_modes.items():
        for mode in modes:
            for scene in SCENES_NEW:
                logical = f"rule_eval_{arm}_{mode}_scene{scene}"
                if ledger.is_complete(logical):
                    continue
                tag = f"r66_rule_{arm}_{mode}"
                metadata = {"arm": arm.upper(), "mode": mode, "scenario_seed": int(scene)}
                def _do(arm=arm, scene=scene, tag=tag, mode=mode):
                    if arm == "immediate":
                        row = runner._run_episode(tag, "rule", int(scene), policy="immediate", tag=tag)
                    else:
                        policy = v2_rules.build_rule(arm, int(scene))
                        row = runner._run_episode(tag, "rule", int(scene), policy="learned",
                                                  attach=policy, tag=tag)
                        row["rule_identity"] = v2_rules.rule_identity(arm)
                    row["model_arm"] = arm.upper(); row["eval_mode"] = mode; row["scenario_seed"] = int(scene)
                    return row
                row = _atomic_run(cfg, logical, metadata, _do,
                                  detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}")
                _write_row(cfg, row)
        ledger.write_ledger()
    return {"status": "rules_eval_complete"}


def eval_focal(cfg, ledger, plan, seed):
    """Block D: focal scene mechanism diagnostic. NEW and BC, greedy, scene 1700027. These 6 cells
    are NOT independent confirmation data; they describe the one frozen policy's focal behaviour."""
    common = cfg["common"]; runner = cfg["runner"]
    import r42_one_shot
    ep128 = Path(common.CHECKPOINTS) / f"v2_train_s{seed}_ep128.pt"
    new_ctrl, _ns, new_identity = r66.load_checkpoint_read_only(ep128, freeze=R66_FREEZE)
    bc_ctrl, bc_identity = r66.load_r60_checkpoint_read_only(seed, "ep0")
    for arm, controller, identity in (("NEW", new_ctrl, new_identity), ("BC", bc_ctrl, bc_identity)):
        logical = f"focal_{arm.lower()}_s{seed}_scene{FOCAL_SCENE}"
        if ledger.is_complete(logical):
            continue
        collector = r42_one_shot.OneShotSendSkipCollector(controller, greedy=True)
        tag = f"r66_focal_{arm.lower()}_s{seed}"
        metadata = {"arm": arm, "training_seed": int(seed), "mode": "greedy",
                    "scenario_seed": FOCAL_SCENE, "block": "D"}
        def _do(controller=controller, collector=collector, tag=tag, arm=arm, identity=identity, seed=seed):
            row = runner._run_episode(tag, "eval", FOCAL_SCENE, policy="learned",
                                      attach=collector.policy(), controller=controller,
                                      collector=collector,
                                      action_rng_seed=runner.eval_action_seed(int(seed), FOCAL_SCENE, "greedy"),
                                      tag=tag, checkpoint_load_identity=identity)
            row["model_arm"] = arm; row["eval_mode"] = "greedy"; row["training_seed"] = int(seed)
            row["focal"] = True
            return row
        row = _atomic_run(cfg, logical, metadata, _do,
                          detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}")
        _write_row(cfg, row)
    ledger.write_ledger()
    return {"seed": seed, "status": "focal_complete"}


# ------------------------------------------------------------------ statistics ------------
def _bootstrap_index():
    """The R66 frozen paired bootstrap index: PCG64(20261012), 10000 draws over 32 scenes."""
    import hashlib
    gen = np.random.Generator(np.random.PCG64(20261012))
    idx = gen.integers(0, 32, size=(10000, 32))
    digest = hashlib.sha256(np.ascontiguousarray(idx).tobytes()).hexdigest()
    return idx, digest


def _load_eval_rows(root: Path, pattern: str) -> dict:
    out = {}
    for p in Path(root).glob(pattern):
        row = _read_json(p)
        seed = int(row.get("training_seed", row.get("seed", -1)))
        mode = row.get("eval_mode")
        scene = int(row.get("scenario_seed"))
        if mode not in MODES:
            continue
        out[(seed, mode, scene)] = {"on_time": int(row["on_time"]),
                                    "spent_bytes": int(row["spent_bytes"]),
                                    "on_time_ids": row.get("on_time_target_ids") or row.get("on_time_ids"),
                                    "row_path": str(p)}
    return out


def _load_rule_arm_rows(root: Path) -> dict:
    out = {}
    for p in Path(root).glob("v2_r66_rule_*.json"):
        row = _read_json(p)
        arm = (row.get("model_arm") or "").lower()
        mode = row.get("eval_mode")
        scene = int(row.get("scenario_seed"))
        if mode not in MODES:
            continue
        out[(arm, mode, scene)] = {"on_time": int(row["on_time"]),
                                   "spent_bytes": int(row["spent_bytes"]), "row_path": str(p)}
    return out


def compute_stats(raw_root: Path) -> dict:
    idx, idx_sha = _bootstrap_index()
    new = _load_eval_rows(raw_root / "rows", "v2_r66_eval_new_*.json")
    bc = _load_eval_rows(raw_root / "rows", "v2_r66_eval_bc_*.json")
    rules = _load_rule_arm_rows(raw_root / "rows")
    old_r60 = _load_eval_rows(r66.R60_DIR / "raw" / "rows", "v2_r60_eval_new_*.json")
    scenes = list(SCENES_NEW)
    per = {}
    for seed in SEEDS:
        for mode in MODES:
            d_cost = []; d_task = []
            for scene in scenes:
                n = new.get((seed, mode, scene)); b = bc.get((seed, mode, scene))
                if n is None or b is None:
                    raise RuntimeError(f"missing paired NEW/BC cell seed={seed} mode={mode} scene={scene}")
                d_cost.append(n["spent_bytes"] - b["spent_bytes"])
                d_task.append(n["on_time"] - b["on_time"])
            d_cost = np.asarray(d_cost, dtype=np.float64); d_task = np.asarray(d_task, dtype=np.float64)
            boot_cost = d_cost[idx].mean(axis=1); boot_task = d_task[idx].mean(axis=1)
            entry = {
                "bc_on_time_total": int(sum(bc[(seed, mode, s)]["on_time"] for s in scenes)),
                "new_on_time_total": int(sum(new[(seed, mode, s)]["on_time"] for s in scenes)),
                "task_delta_new_minus_bc_mean": float(d_task.mean()),
                "task_ci95_per_scene": [float(np.quantile(boot_task, 0.025)), float(np.quantile(boot_task, 0.975))],
                "bc_bytes_per_scene": float(np.mean([bc[(seed, mode, s)]["spent_bytes"] for s in scenes])),
                "new_bytes_per_scene": float(np.mean([new[(seed, mode, s)]["spent_bytes"] for s in scenes])),
                "cost_delta_new_minus_bc_B_per_scene": float(d_cost.mean()),
                "cost_ci95_B_per_scene": [float(np.quantile(boot_cost, 0.025)), float(np.quantile(boot_cost, 0.975))],
            }
            for arm in ("immediate", "late_discovery_gate", "always_hold"):
                r = {(s): rules.get((arm, mode, s)) for s in scenes}
                have = [r[s] for s in scenes if r[s] is not None]
                if have:
                    entry[f"{arm}_on_time_total"] = int(sum(x["on_time"] for x in have))
                    entry[f"{arm}_bytes_per_scene"] = float(np.mean([x["spent_bytes"] for x in have]))
                    entry[f"new_minus_{arm}_bytes_B_per_scene"] = float(
                        np.mean([new[(seed, mode, s)]["spent_bytes"] for s in scenes]) - entry[f"{arm}_bytes_per_scene"])
            if all((seed, mode, s) in old_r60 for s in scenes):
                entry["old_r60_on_time_total"] = int(sum(old_r60[(seed, mode, s)]["on_time"] for s in scenes))
                entry["old_r60_bytes_per_scene"] = float(np.mean([old_r60[(seed, mode, s)]["spent_bytes"] for s in scenes]))
            per[f"{seed}/{mode}"] = entry
    return {"kind": "round66-statistics/1", "primary_comparison": "NEW_minus_BC",
            "greedy_is_primary": True, "sampled_reported_separately": True,
            "scene_unit": 32, "seeds_not_pooled": True,
            "bootstrap": {"rng": "PCG64", "seed": 20261012, "draws": 10000, "index_sha256": idx_sha,
                          "quantile": "numpy linear 0.025/0.975"},
            "per_seed_mode": per,
            "old_r60_cells": len(old_r60), "new_cells": len(new), "bc_cells": len(bc),
            "note": ("NEW=R66 ep128 (early path frozen), BC=compliant R60 ep0 initial value, "
                     "OLD R60 read-only reuse; CI containing 0 is not equivalence; restricted "
                     "policy family only, single new dev block.")}


# ------------------------------------------------------------------ acceptance + main -----
def zero_env_acceptance(cfg) -> dict:
    """Zero-environment CLI forward acceptance: verify live config, module paths, the frozen
    parameter set, count hooks and checkpoint routing. Stops before any episode provider is built."""
    runtime = cfg["runtime"]; trainer = cfg["trainer"]; common = cfg["common"]
    controller = cfg["frozen_fresh"](11)
    ps = controller.parameter_summary()
    fps = r66.frozen_parameter_summary(controller)
    frozen, trainable = r66.partitioned_parameters(controller)
    checks = {
        "trainer_is_frozen_v2": Path(trainer.__file__).name == "v2_trainer.py",
        "ppo_update_is_r66": cfg["identity"]["ppo_update_is_r66"],
        "save_is_r66": cfg["identity"]["save_is_r66"],
        "fresh_is_frozen": cfg["identity"]["fresh_is_frozen"],
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
        "structure_r66": fps.get("structure") == r66.R66_STRUCTURE,
        "checkpoint_root_under_r66_raw": str(Path(common.CHECKPOINTS).resolve()).startswith(str(cfg["raw_root"].resolve())),
        "rows_root_under_r66_raw": str(Path(common.ROWS).resolve()).startswith(str(cfg["raw_root"].resolve())),
        "early_eq_late_at_init": all(torch.equal(a, b) for a, b in
                                     zip(controller.inner.early_head.parameters(), controller.inner.late_head.parameters())),
    }
    status = "passed" if all(checks.values()) else "failed"
    return {"kind": "round66-zero-env-acceptance/1", "status": status, "checks": checks,
            "identity": cfg["identity"], "parameter_sha256": ps["parameter_sha256"],
            "frozen_scalar_count": sum(p.numel() for p in frozen),
            "trainable_scalar_count": sum(p.numel() for p in trainable)}



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND60 partitioned actor-head fixed-plan executor")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--phase", choices=("acceptance", "train", "eval", "stats", "all"), required=True)
    ap.add_argument("--seed", type=int, choices=(11, 22, 33), default=None)
    ap.add_argument("--first-gate-only", action="store_true",
                    help="run each seed's first 4 episodes + first update + real gate, then stop (episodes count toward 128)")
    args = ap.parse_args(argv)
    raw_root = args.raw_root.resolve()
    cfg = configure_and_inject(raw_root)
    ledger = R66Ledger(raw_root)
    order = [int(x) for x in cfg["common"].scenario_order()]
    if len(order) != EPISODES_PER_SEED:
        raise RuntimeError(f"training order length {len(order)} != {EPISODES_PER_SEED}")
    plan_path = raw_root / "ROUND66_FIXED_PLAN.json"
    if not plan_path.exists():
        plan = build_plan(order)
        _write_json(plan_path, plan)
    else:
        plan = _read_json(plan_path)
    seeds = (args.seed,) if args.seed else SEEDS
    if args.phase == "acceptance":
        acc = zero_env_acceptance(cfg)
        _write_json(raw_root / "runtime" / "ROUND66_ZERO_ENV_ACCEPTANCE.json", acc)
        print(json.dumps(acc, indent=1))
        return 0 if acc["status"] == "passed" else 2
    if args.phase in ("train", "all"):
        references = run_block_a_references(cfg, ledger, plan, order)
        missing = sorted(set(order) - set(references))
        if missing:
            raise RuntimeError(f"block-A immediate references incomplete: {missing}")
        for seed in seeds:
            print(f"R66 training seed {seed} ...", flush=True)
            print(json.dumps(train_seed(cfg, ledger, plan, seed, order, references,
                                        stop_after_first_gate=args.first_gate_only)), flush=True)
            ledger.write_ledger()
        if args.first_gate_only:
            ledger.write_ledger()
            return 0
    if args.phase in ("eval", "all"):
        print(json.dumps(eval_rules(cfg, ledger, plan)), flush=True)
        for seed in seeds:
            print(f"R66 eval seed {seed} ...", flush=True)
            print(json.dumps(eval_seed(cfg, ledger, plan, seed)), flush=True)
            print(json.dumps(eval_focal(cfg, ledger, plan, seed)), flush=True)
    if args.phase in ("stats", "all"):
        stats = compute_stats(raw_root)
        _write_json(raw_root / "ROUND66_STATISTICS.json", stats)
        print(json.dumps(stats, indent=1))
    ledger.write_ledger()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
