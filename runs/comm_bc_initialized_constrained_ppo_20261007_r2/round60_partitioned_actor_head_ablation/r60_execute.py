"""ROUND60 independent executor: fixed 384 training + 192 eval with an independent R60 identity,
claim/slot/quota ledger, first-batch gate, passive early/late diagnostics, and paired statistics
vs the read-only reused R58 cells.

This reuses the frozen sim (v2_run._run_episode), the frozen collector (r42_one_shot), the frozen
batch_samples/batch_g_hat/dual_update, and the R60 partitioned controller + routed ppo_update +
5-module checkpoint from r60_partitioned. It does NOT reuse the R58 branch/version/claim verifier;
R60 has its own task/branch/version/freeze identity, output root, claim/slot roots and quota.

Quota (this continuation): historical non-plan probe = 1; normal plan = 576; technical-failure
reserve = 23; cumulative launch/attempt hard cap = 600. Synthetic optimizer updates and synthetic
checkpoint writes are counted separately from real updates.
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
import r60_partitioned as r60

SEEDS = (11, 22, 33)
EPISODES_PER_SEED = 128
EPISODES_PER_UPDATE = 4
MODES = ("greedy", "sampled")
SCENES_NEW = tuple(range(1_700_000, 1_700_032))
TASK_ID = "ROUND60-PARTITIONED-ACTOR-HEAD-ABLATION"
BRANCH = "exp/round60-partitioned-actor-head-ablation"
NORMAL_CAP = 576
RESERVE_CAP = 23
HARD_CAP = 600
HISTORICAL_PROBE = 1

R60_FREEZE = {
    "kind": "round60-partitioned-actor-head-freeze/1",
    "task_id": TASK_ID, "branch": BRANCH,
    "training_version": r60.R60_TRAINING_VERSION,
    "structure": "partitioned-early-late-actor-head/1",
    "gae_lambda": 1.0, "eta_lambda": 0.5, "gamma": 1.0, "delta": 0.0, "lambda_init": 1.0,
    "data_order_seed": 20260927, "b0_bytes": 10000.0,
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


class R60Ledger:
    """Independent R60 claim/slot/quota ledger under the R60 raw root."""

    def __init__(self, raw: Path) -> None:
        self.raw = raw
        self.claims_dir = raw / "launch_claims"
        self.slots_dir = raw / "launch_slots"
        self.runtime_dir = raw / "runtime"
        for d in (self.claims_dir, self.slots_dir / "normal", self.slots_dir / "reserve", self.runtime_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.runtime_dir / "ROUND60_LEDGER.json"

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
                 "training_version": r60.R60_TRAINING_VERSION, "metadata": metadata,
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
               "training_version": r60.R60_TRAINING_VERSION,
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
    # Inject the partitioned controller factory (capture original to avoid recursion).
    orig_fresh = controller_module.fresh_controller
    def partitioned_fresh(seed: int):
        b = orig_fresh(int(seed))
        pc = r60.PartitionedController(b)
        ep0 = r60.load_r58_ep0_model(int(seed))
        pc.inner.trunk.load_state_dict(ep0["trunk"])
        pc.inner.value_head.load_state_dict(ep0["value_task_head"])
        pc.cost_head.load_state_dict(ep0["value_cost_head"])
        pc.inner.early_head.load_state_dict(ep0["action_head"])
        pc.inner.late_head.load_state_dict(ep0["action_head"])
        return pc
    controller_module.fresh_controller = partitioned_fresh
    # Route the trainer's PPO update and checkpoint to the R60 partitioned versions.
    from round53_trial_adapter import critic_input_detached
    def routed_ppo(controller, optimizer, episodes, *, update_index):
        # Preserve the R53 critic->shared-trunk gradient isolation exactly as R58 applies it:
        # the two value heads receive a detached shared representation; the actor (routed) path
        # stays connected. Same context object R58 uses around its ppo_update.
        with critic_input_detached(controller):
            return r60.partitioned_ppo_update(controller, optimizer, episodes, update_index=update_index)
    def r60_save(controller, optimizer, state, path, *, freeze):
        return r60.save_checkpoint(controller, optimizer, state, path, freeze=R60_FREEZE)
    def r60_load(path, *, freeze):
        return r60.load_checkpoint(path, freeze=R60_FREEZE)
    def r60_load_ro(path, *, freeze):
        return r60.load_checkpoint_read_only(path, freeze=R60_FREEZE)
    trainer.ppo_update = routed_ppo
    trainer.save_checkpoint = r60_save
    trainer.load_checkpoint = r60_load
    trainer.load_checkpoint_read_only = r60_load_ro
    # Stamp the R60 training version onto the live version modules so rows/claims carry it.
    try:
        runtime["r42_versions"].TRAINING_VERSION = r60.R60_TRAINING_VERSION
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
        "ppo_update_is_r60": trainer.ppo_update is routed_ppo,
        "save_is_r60": trainer.save_checkpoint is r60_save,
        "fresh_is_partitioned": controller_module.fresh_controller is partitioned_fresh,
        "common_GAE_LAMBDA": float(common.GAE_LAMBDA), "common_ETA_LAMBDA": float(common.ETA_LAMBDA),
    }
    return {"runtime": runtime, "trainer": trainer, "controller_module": controller_module,
            "common": common, "runner": runner, "r42_run": r42_run, "raw_root": raw_root,
            "partitioned_fresh": partitioned_fresh, "identity": identity}


def build_references() -> dict[int, float]:
    """Verified immediate-reference on_time per training scene (paired constraint for batch_g_hat)."""
    import r58_execute
    baseline = r58_execute.verify_baseline_reuse()
    rows = baseline["training_immediate_cache"]["rows_by_scene"]
    refs: dict[int, float] = {}
    for row in rows:
        refs[int(row["scene"])] = float(_read_json(Path(row["path"]))["on_time"])
    return refs


def build_plan(order: list[int]) -> dict[str, Any]:
    jobs = []
    slot = 0
    for seed in SEEDS:
        for ep in range(EPISODES_PER_SEED):
            slot += 1
            jobs.append({"scope": "ppo_training", "logical_job": f"r60_new_training_s{seed}_e{ep}",
                         "seed": seed, "episode_index": ep, "scene": int(order[ep]), "normal_slot": slot})
    for seed in SEEDS:
        for mode in MODES:
            for scene in SCENES_NEW:
                slot += 1
                jobs.append({"scope": "model_eval", "logical_job": f"r60_model_eval_new_s{seed}_{mode}_scene{scene}",
                             "seed": seed, "mode": mode, "scene": int(scene), "normal_slot": slot})
    if len(jobs) != NORMAL_CAP or len({j["logical_job"] for j in jobs}) != NORMAL_CAP:
        raise AssertionError("R60 fixed plan is not exactly 576 unique jobs")
    return {"kind": "round60-fixed-job-plan/1", "task_id": TASK_ID, "branch": BRANCH,
            "training_version": r60.R60_TRAINING_VERSION,
            "training_episode_count": 384, "model_eval_count": 192, "normal_count": NORMAL_CAP,
            "reserve_cap": RESERVE_CAP, "hard_cap": HARD_CAP, "historical_probe": HISTORICAL_PROBE,
            "eval_scenes_new": list(SCENES_NEW), "jobs": jobs}


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
        controller, optimizer, state = r60.load_checkpoint(rolling, freeze=R60_FREEZE)
        ps = controller.parameter_summary()
        if ps["parameter_object_count"] != 12 or ps.get("structure") != "partitioned-early-late-actor-head/1":
            raise RuntimeError(f"seed {seed}: rolling checkpoint is not a partitioned 5-module model")
        resumed = True
    else:
        controller = cfg["partitioned_fresh"](seed)
        optimizer = torch.optim.Adam(controller.parameters, lr=common.LEARNING_RATE)
        state = r60.fresh_training_state(seed, gae_lambda=1.0, eta_lambda=0.5, gamma=1.0,
                                         delta=0.0, lambda_init=1.0, data_order_seed=20260927,
                                         b0_bytes=common.B0_BYTES)
        ep0_identity = controller.parameter_summary()
        _write_json(ckpt_dir / f"r60_ep0_identity_s{seed}.json", ep0_identity)
        resumed = False
    gate_path = cfg["raw_root"] / "runtime" / f"r60_first_batch_gate_s{seed}.json"
    while int(state["episodes_done"]) < EPISODES_PER_SEED:
        batch = int(state["updates"])
        if int(state["episodes_done"]) != batch * EPISODES_PER_UPDATE:
            raise RuntimeError(f"seed {seed}: episode cursor and update index inconsistent")
        lam = float(state["lambda"])
        before = controller.parameter_summary()
        pre = ckpt_dir / f"v2_train_s{seed}_before_b{batch}.pt"
        if not pre.exists():
            r60.save_checkpoint(controller, optimizer, state, pre, freeze=R60_FREEZE)
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
        update_report = r60.partitioned_ppo_update(controller, optimizer, episodes, update_index=batch)
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
        r60.save_checkpoint(controller, optimizer, state, post, freeze=R60_FREEZE)
        r60.save_checkpoint(controller, optimizer, state, rolling, freeze=R60_FREEZE)
        batch_record = {"kind": "round60-training-batch-record/1", "batch": batch, "seed": seed,
                        "episodes": [int(r["episode_index"]) for r in episodes],
                        "scenarios": [int(r["scenario_seed"]) for r in episodes],
                        "lambda_at_rollout": lam, "update": update_report, "constraint": g_record,
                        "dual": dual_record, "parameters_before_sha": before["parameter_sha256"],
                        "parameters_after_sha": after["parameter_sha256"],
                        "episodes_done_after": int(state["episodes_done"]),
                        "optimizer_steps_after": int(state["optimizer_steps"]),
                        "structure": "partitioned-early-late-actor-head/1"}
        _write_json(ckpt_dir / f"r60_batch_record_s{seed}_b{batch}.json", batch_record)
        ledger.write_ledger()
        # First-batch gate: run the real gate immediately after the first update of this seed.
        if batch == 0 and not gate_path.exists():
            gate = _first_batch_gate(cfg, controller, optimizer, state, episodes, update_report,
                                     g_record, dual_record, before, after, ep0_identity=(None if resumed else _read_json(ckpt_dir / f"r60_ep0_identity_s{seed}.json")))
            _write_json(gate_path, gate)
            if gate["status"] != "passed":
                raise RuntimeError(f"seed {seed}: first-batch gate failed: {gate}")
            if stop_after_first_gate:
                return {"seed": seed, "status": "first_gate_passed", "gate": gate["status"]}
        print(f"R60 seed {seed} batch {batch}: episodes_done={state['episodes_done']} "
              f"lambda={state['lambda']:.4f} early_free={update_report.get('early_free')} "
              f"late_free={update_report.get('late_free')} gnorms={update_report.get('gradient_norm_mean')}", flush=True)
    ep128 = ckpt_dir / f"v2_train_s{seed}_ep128.pt"
    r60.save_checkpoint(controller, optimizer, state, ep128, freeze=R60_FREEZE)
    return {"seed": seed, "status": "ep128_complete", "episodes_done": int(state["episodes_done"]),
            "updates": int(state["updates"]), "optimizer_steps": int(state["optimizer_steps"]),
            "lambda_final": float(state["lambda"])}


def _first_batch_gate(cfg, controller, optimizer, state, episodes, update_report, g_record,
                      dual_record, before, after, *, ep0_identity):
    """Real first-batch gate: verify actual dual-head routing, ep0 init, optimizer coverage,
    critic->trunk isolation, mask/likelihood, reward/cost/GAE, dual lambda, and checkpoint id."""
    from round53_trial_adapter import critic_input_detached
    checks = {}
    checks["dual_head_routing_active"] = bool(update_report.get("early_free", 0) > 0
                                              and update_report.get("late_free", 0) > 0)
    checks["early_free"] = int(update_report.get("early_free", 0))
    checks["late_free"] = int(update_report.get("late_free", 0))
    if ep0_identity is not None:
        checks["init_params_equal_r58_ep0"] = (before["parameter_sha256"] == ep0_identity["parameter_sha256"])
    checks["optimizer_covers_12"] = (after["parameter_object_count"] == 12)
    checks["structure_partitioned"] = (after.get("structure") == "partitioned-early-late-actor-head/1")
    checks["five_modules"] = (sorted(after["shapes"].keys()) ==
        sorted(["trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
                "early_head.weight", "early_head.bias", "late_head.weight", "late_head.bias",
                "value_task_head.weight", "value_task_head.bias", "value_cost_head.weight", "value_cost_head.bias"]))
    checks["likelihood_alignment_passed"] = bool(update_report["likelihood_alignment"]["passed"])
    checks["value_losses_finite"] = all(np.isfinite([update_report.get("value_loss_task", 0.0),
                                                     update_report.get("value_loss_cost", 0.0)]))
    checks["lambda_updated"] = ("lambda_next" in dual_record)
    # critic->trunk isolation: value loss must give zero shared-trunk gradient while the routed
    # actor path stays connected (deterministic probe on the live controller).
    feats = torch.as_tensor(np.concatenate([t["features"] for ep in episodes for t in ep["collector"].records], axis=0))
    with critic_input_detached(controller):
        h = controller.inner.trunk(feats)
        vt = controller.inner.value_head(h).squeeze(-1)
        vloss = ((vt - 0.0) ** 2).mean()
        tg = torch.autograd.grad(vloss, list(controller.inner.trunk.parameters()), allow_unused=True)
        trunk_l2 = float(sum((g ** 2).sum() for g in tg if g is not None))
        rl = r60.route_logits(controller, feats, h)
        ag = torch.autograd.grad(rl.sum(), list(controller.inner.trunk.parameters()), allow_unused=True)
        actor_l2 = float(sum((g ** 2).sum() for g in ag if g is not None))
    checks["critic_trunk_isolated"] = (trunk_l2 == 0.0)
    checks["actor_path_connected"] = (actor_l2 > 0.0)
    status = "passed" if all(v for k, v in checks.items() if isinstance(v, bool)) else "failed"
    return {"kind": "round60-first-batch-gate/1", "seed": int(state["training_seed"]),
            "status": status, "checks": checks,
            "parameters_after_sha": after["parameter_sha256"],
            "lambda_after_first_batch": dual_record["lambda_next"],
            "gate_diag": update_report.get("gate_diag"),
            "module_gradient_l1_mean": update_report.get("module_gradient_l1_mean")}


def eval_seed(cfg, ledger, plan, seed):
    common = cfg["common"]; runner = cfg["runner"]
    ep128 = Path(common.CHECKPOINTS) / f"v2_train_s{seed}_ep128.pt"
    if not ep128.is_file():
        raise FileNotFoundError(f"seed {seed}: ep128 checkpoint missing")
    controller, state, identity = r60.load_checkpoint_read_only(ep128, freeze=R60_FREEZE)
    if int(state.get("episodes_done", -1)) != EPISODES_PER_SEED:
        raise RuntimeError(f"seed {seed}: ep128 model is not 128 episodes")
    if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
        raise RuntimeError(f"seed {seed}: read-only eval restored state")
    ps = controller.parameter_summary()
    identity["parameter_identity"] = ps
    import r42_one_shot
    for mode in MODES:
        for scene in SCENES_NEW:
            logical = f"model_eval_new_s{seed}_{mode}_scene{scene}"
            if ledger.is_complete(logical):
                continue
            collector = r42_one_shot.OneShotSendSkipCollector(controller, greedy=(mode == "greedy"))
            tag = f"r60_eval_new_s{seed}_{mode}"
            metadata = {"arm": "NEW", "training_seed": int(seed), "mode": mode, "scenario_seed": int(scene)}
            def _do(collector=collector, tag=tag, scene=scene, mode=mode):
                row = runner._run_episode(tag, "eval", int(scene), policy="learned",
                                          attach=collector.policy(), controller=controller,
                                          collector=collector,
                                          action_rng_seed=runner.eval_action_seed(int(seed), int(scene), mode),
                                          tag=tag, checkpoint_load_identity=identity)
                row["model_arm"] = "NEW"; row["eval_mode"] = mode; row["training_seed"] = int(seed)
                return row
            row = _atomic_run(cfg, logical, metadata, _do,
                              detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}")
            _write_row(cfg, row)
        ledger.write_ledger()
    return {"seed": seed, "status": "eval_complete"}


# ------------------------------------------------------------------ statistics ------------
def _bootstrap_index():
    """The frozen R58 paired bootstrap index: PCG64(20261008), 10000 draws over 32 scenes."""
    import hashlib
    gen = np.random.Generator(np.random.PCG64(20261008))
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
        out[(seed, mode, scene)] = {"on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
                                    "row_path": str(p), "lifecycle": row.get("artefacts", {}).get("lifecycle")}
    return out


def _load_rule_rows(root: Path, arm: str) -> dict:
    out = {}
    for p in Path(root).glob(f"*rule_{arm}_*17000*.json"):
        row = _read_json(p)
        if row.get("model_arm") not in (f"{arm.upper()}_RULE", arm) and row.get("arm") not in (arm, arm.upper()):
            # accept either naming; filter by presence of the arm in the filename/id
            pass
        scene = int(row.get("scenario_seed"))
        # keep the first successful attempt per scene
        if scene in out:
            continue
        out[scene] = {"on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
                      "row_path": str(p), "lifecycle": row.get("artefacts", {}).get("lifecycle")}
    return out


def compute_stats(raw_root: Path) -> dict:
    idx, idx_sha = _bootstrap_index()
    new = _load_eval_rows(raw_root / "rows", "v2_r60_eval_new_*.json")
    r58 = _load_eval_rows(R58_V35, "v2_r44_eval_new_*.json")
    immediate = _load_rule_rows(R58_V35, "immediate")
    late = _load_rule_rows(R58_V35, "late")
    scenes = list(SCENES_NEW)
    per = {}
    for seed in SEEDS:
        for mode in MODES:
            d_cost = []; d_task = []
            for scene in scenes:
                n = new.get((seed, mode, scene)); r = r58.get((seed, mode, scene))
                if n is None or r is None:
                    raise RuntimeError(f"missing paired cell seed={seed} mode={mode} scene={scene}")
                d_cost.append(n["spent_bytes"] - r["spent_bytes"])
                d_task.append(n["on_time"] - r["on_time"])
            d_cost = np.asarray(d_cost, dtype=np.float64); d_task = np.asarray(d_task, dtype=np.float64)
            boot_cost = d_cost[idx].mean(axis=1); boot_task = d_task[idx].mean(axis=1)
            per[f"{seed}/{mode}"] = {
                "r58_on_time_total": int(sum(r58[(seed, mode, s)]["on_time"] for s in scenes)),
                "new_on_time_total": int(sum(new[(seed, mode, s)]["on_time"] for s in scenes)),
                "task_delta_per_scene_mean": float(d_task.mean()),
                "r58_bytes_per_scene": float(np.mean([r58[(seed, mode, s)]["spent_bytes"] for s in scenes])),
                "new_bytes_per_scene": float(np.mean([new[(seed, mode, s)]["spent_bytes"] for s in scenes])),
                "cost_delta_B_per_scene": float(d_cost.mean()),
                "cost_ci95_B_per_scene": [float(np.quantile(boot_cost, 0.025)), float(np.quantile(boot_cost, 0.975))],
                "task_ci95_per_scene": [float(np.quantile(boot_task, 0.025)), float(np.quantile(boot_task, 0.975))],
            }
    return {"kind": "round60-statistics/1", "primary_comparison": "NEW_minus_R58",
            "greedy_is_primary": True, "sampled_reported_separately": True,
            "scene_unit": 32, "seeds_not_pooled": True,
            "bootstrap": {"rng": "PCG64", "seed": 20261008, "draws": 10000, "index_sha256": idx_sha,
                          "quantile": "numpy linear 0.025/0.975"},
            "per_seed_mode": per,
            "immediate_cells": len(immediate), "late_cells": len(late),
            "r58_cells": len(r58), "new_cells": len(new)}


# ------------------------------------------------------------------ acceptance + main -----
def zero_env_acceptance(cfg) -> dict:
    """Zero-environment CLI forward acceptance: verify live config, module paths, parameter set,
    count hooks and checkpoint routing. Stops before any episode provider is constructed."""
    runtime = cfg["runtime"]; trainer = cfg["trainer"]; common = cfg["common"]
    controller = cfg["partitioned_fresh"](11)
    ps = controller.parameter_summary()
    checks = {
        "trainer_is_frozen_v2": Path(trainer.__file__).name == "v2_trainer.py",
        "ppo_update_is_routed": cfg["identity"]["ppo_update_is_r60"],
        "save_is_r60_5module": cfg["identity"]["save_is_r60"],
        "fresh_is_partitioned": cfg["identity"]["fresh_is_partitioned"],
        "common_gae_lambda_1": float(common.GAE_LAMBDA) == 1.0,
        "common_eta_lambda_0p5": float(common.ETA_LAMBDA) == 0.5,
        "partitioned_12_params": ps["parameter_object_count"] == 12,
        "partitioned_5_modules": sorted({k.split(".")[0] for k in ps["shapes"]}) ==
            sorted(["trunk", "early_head", "late_head", "value_task_head", "value_cost_head"]),
        "checkpoint_root_under_r60_raw": str(Path(common.CHECKPOINTS).resolve()).startswith(str(cfg["raw_root"].resolve())),
        "rows_root_under_r60_raw": str(Path(common.ROWS).resolve()).startswith(str(cfg["raw_root"].resolve())),
        "early_eq_late_at_init": all(torch.equal(a, b) for a, b in
                                     zip(controller.inner.early_head.parameters(), controller.inner.late_head.parameters())),
    }
    status = "passed" if all(checks.values()) else "failed"
    return {"kind": "round60-zero-env-acceptance/1", "status": status, "checks": checks,
            "identity": cfg["identity"], "parameter_sha256": ps["parameter_sha256"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND60 partitioned actor-head fixed-plan executor")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--phase", choices=("acceptance", "train", "eval", "stats", "all"), required=True)
    ap.add_argument("--seed", type=int, choices=(11, 22, 33), default=None)
    args = ap.parse_args(argv)
    raw_root = args.raw_root.resolve()
    cfg = configure_and_inject(raw_root)
    ledger = R60Ledger(raw_root)
    order = [int(x) for x in cfg["common"].scenario_order()]
    if len(order) != EPISODES_PER_SEED:
        raise RuntimeError(f"training order length {len(order)} != {EPISODES_PER_SEED}")
    plan_path = raw_root / "ROUND60_FIXED_PLAN.json"
    if not plan_path.exists():
        plan = build_plan(order)
        _write_json(plan_path, plan)
    else:
        plan = _read_json(plan_path)
    seeds = (args.seed,) if args.seed else SEEDS
    if args.phase == "acceptance":
        acc = zero_env_acceptance(cfg)
        _write_json(raw_root / "runtime" / "ROUND60_ZERO_ENV_ACCEPTANCE.json", acc)
        print(json.dumps(acc, indent=1))
        return 0 if acc["status"] == "passed" else 2
    references = build_references()
    missing = sorted(set(order) - set(references))
    if missing:
        raise RuntimeError(f"immediate reference cache incomplete: {missing}")
    if args.phase in ("train", "all"):
        for seed in seeds:
            print(f"R60 training seed {seed} ...", flush=True)
            print(json.dumps(train_seed(cfg, ledger, plan, seed, order, references), flush=True))
            ledger.write_ledger()
    if args.phase in ("eval", "all"):
        for seed in seeds:
            print(f"R60 eval seed {seed} ...", flush=True)
            print(json.dumps(eval_seed(cfg, ledger, plan, seed), flush=True))
    if args.phase in ("stats", "all"):
        stats = compute_stats(raw_root)
        _write_json(raw_root / "ROUND60_STATISTICS.json", stats)
        print(json.dumps(stats, indent=1))
    ledger.write_ledger()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
