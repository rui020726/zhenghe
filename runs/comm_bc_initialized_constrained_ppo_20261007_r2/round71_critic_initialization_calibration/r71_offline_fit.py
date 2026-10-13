#!/usr/bin/env python
"""ROUND71 offline critic calibration: teacher rebuild, guarded fit, head install, INIT save.

The ONLY scientific intervention of ROUND71 happens here: each seed's dual critic affine heads
are replaced by a per-seed dual-output ridge fitted on that seed's OWN ROUND68 128-episode
teacher data over the frozen R68 ep0 trunk's true mean-over-targets representation.

Fidelity rules encoded below:
  * the teacher targets are rebuilt with the REAL ``v2_trainer.batch_samples`` on the committed
    ROUND68 tapes (actions' pre-decision 19-D inputs + same-time-contract MC task/positive-cost
    returns); nothing is re-simulated and no environment is constructed;
  * every real (N,19) trunk input row is charged to the persisted ForwardLedger BEFORE the
    forward (cap 72,000 rows, batches of at most 1,024 rows, no refund on failure or repeat);
  * pooling is mean-over-targets AFTER the trunk, per joint decision row, exactly as
    ``r66_frozen.frozen_early_ppo_update`` pools critic values;
  * the ridge configuration is LOCKED (1e-3, all-row centre/scale, constant column scale 1,
    intercept unpenalised, one fit per seed, no CV, no seed selection);
  * installation touches ONLY the two critic parameters (``copy_``), never re-scales and never
    draws RNG; trunk / early_head / late_head are proven byte-identical afterwards;
  * a fresh optimizer, lambda=1 and the R68 registered initial RNG follow the install;
  * an existing INIT artefact is never silently overwritten (DuplicateOutputGuard).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN_ROOT = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
R68_DIR = RUN_ROOT / "round68_late_initial_confidence_ablation"
READINESS_DIR = REPO / "analysis" / "round71-critic-calibration-readiness"
for _d in (READINESS_DIR, str(R68_DIR)):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from r71_critic_calibration_seam import (  # noqa: E402
    BATCH_INPUT_ROW_CAP, OFFLINE_INPUT_ROW_CAP, DuplicateOutputGuard, ForwardLedger,
    LicenseGate, LicenseInvalid, LicenseMissing,
)
import r71_runtime_binding as readiness  # noqa: E402
from r71_runtime_binding import ModelReadBoundary  # noqa: E402

SEEDS = (11, 22, 33)
EPISODES_PER_SEED = 128
BATCHES_PER_SEED = 32
COUNTS_PATH = READINESS_DIR / "ROUND71_OFFLINE_INPUT_COUNT.json"


class _Records:
    """Minimal shim exposing ``.records`` so the REAL batch_samples can run on JSON tapes."""

    def __init__(self, records):
        self.records = records


def registered_counts() -> dict:
    """The read-only registered per-seed teacher-row counts (reused, never re-estimated)."""
    counts = json.loads(COUNTS_PATH.read_text(encoding="utf-8"))
    return counts


def teacher_episode_order(seed: int) -> list:
    """(episode, scene) order for one seed from the 32 committed ROUND68 batch records."""
    ck = R68_DIR / "raw" / "checkpoints"
    order = []
    for b in range(BATCHES_PER_SEED):
        path = ck / f"r40_batch_s{int(seed)}_b{b}_attempt01.json"
        if not path.is_file():
            raise RuntimeError(f"seed {seed}: committed ROUND68 batch record missing: {path}")
        rec = json.loads(path.read_text(encoding="utf-8"))
        for ep, scene in zip(rec["episodes"], rec["scenarios"]):
            order.append((int(ep), int(scene)))
    if len(order) != EPISODES_PER_SEED:
        raise RuntimeError(f"seed {seed}: teacher episode order has {len(order)} entries")
    return order


def row_path(seed: int, ep: int) -> Path:
    hits = sorted((R68_DIR / "raw" / "rows").glob(f"v2_r68_train_s{int(seed)}_e{int(ep)}_s*_a1.json"))
    if len(hits) != 1:
        raise RuntimeError(f"seed {seed} episode {ep}: {len(hits)} row matches")
    return hits[0]


def rebuild_episode_samples(seed: int, ep: int) -> tuple:
    """One episode's decision-step samples through the REAL v2_trainer.batch_samples.

    Returns (samples, fidelity) where fidelity re-proves the reward-stream bundling against the
    row's own committed totals (task_reward_total / airtime_tap_total), so a mis-built episode
    dict fails loudly instead of silently poisoning the teacher targets.
    """
    import v2_trainer as trainer
    row = json.loads(row_path(seed, ep).read_text(encoding="utf-8"))
    tape = Path(row["artefacts"]["tape"])
    records = [json.loads(line) for line in tape.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records:
        raise RuntimeError(f"seed {seed} episode {ep}: empty tape")
    episode = {"collector": _Records(records),
               "cost_stream": row["cost_stream"],
               "task_stream": row["task_stream"],
               "steps": int(row["steps"]),
               "lambda": float(row["lambda"]),
               "episode_index": int(row["episode_index"])}
    samples = trainer.batch_samples([episode])
    # fidelity: bundled rewards + unassigned prefix must reproduce the committed episode totals
    task_total = float(row.get("task_reward_total", 0.0))
    cost_total = float(row.get("airtime_tap_total", 0.0))
    fidelity = {"episode": int(ep), "scene": int(row["scenario_seed"]),
                "decision_steps": len(samples),
                "target_rows": int(sum(len(s["targets"]) for s in samples)),
                "lambda": float(row["lambda"])}
    # fidelity is asserted on the reward streams directly: every tick's post-action reward is
    # assigned to some decision (or the unassigned prefix), so the stream total must reproduce
    # the committed episode totals exactly.
    streams = _bundled_stream_totals(row, records)
    fidelity["bundled_task_plus_prefix"] = streams["task"]
    fidelity["committed_task_reward_total"] = task_total
    fidelity["bundled_cost_plus_prefix"] = streams["cost"]
    fidelity["committed_airtime_tap_total"] = cost_total
    for key, got, want in (("task", streams["task"], task_total),
                           ("cost", streams["cost"], cost_total)):
        if abs(got - want) > 1e-6 * max(1.0, abs(want)):
            raise RuntimeError(
                f"seed {seed} episode {ep}: bundled {key} stream {got} != committed total {want}")
    return samples, fidelity


def _bundled_stream_totals(row: dict, records: list) -> dict:
    """Recompute _bundle's assignment in pure python and return (task, cost) totals."""
    from bisect import bisect_right
    decision_steps = sorted(int(r["step"]) for r in records)
    horizon = int(row["steps"])
    cost_stream = {int(k): float(v["airtime_bytes"]) for k, v in row["cost_stream"].items()}
    task_stream = {int(k): float(v) for k, v in row["task_stream"].items()}
    out = {}
    for name, stream in (("task", task_stream), ("cost", cost_stream)):
        total = 0.0
        for step in range(0, horizon + 1):
            total += float(stream.get(step, 0.0))
        out[name] = total
    return out


def teacher_batches(seed: int, *, cap: int = BATCH_INPUT_ROW_CAP, limit_batches: int = 0):
    """Yield (matrix, counts, returns, meta) per <=cap-input-row batch, in the registered order.

    ``limit_batches>0`` stops after that many batches (the gate probe uses one batch).
    """
    samples = []
    fidelity = []
    for ep, scene in teacher_episode_order(seed):
        ep_samples, ep_fidelity = rebuild_episode_samples(seed, ep)
        samples.extend(ep_samples)
        fidelity.append(ep_fidelity)
    joint_rows = len(samples)
    target_rows = int(sum(len(s["targets"]) for s in samples))
    registered = registered_counts()["per_seed"][str(int(seed))]
    if joint_rows != int(registered["joint_decision_rows"]):
        raise RuntimeError(
            f"seed {seed}: rebuilt joint decision rows {joint_rows} != registered "
            f"{registered['joint_decision_rows']}")
    if target_rows != int(registered["target_level_19d_input_rows"]):
        raise RuntimeError(
            f"seed {seed}: rebuilt target-level input rows {target_rows} != registered "
            f"{registered['target_level_19d_input_rows']}")
    batch, total = [], 0
    emitted = 0
    for index, sample in enumerate(samples):
        k = len(sample["targets"])
        if batch and total + k > cap:
            emitted += 1
            yield _pack(batch, fidelity)
            if limit_batches and emitted >= limit_batches:
                return
            batch, total = [], 0
        batch.append(sample)
        total += k
    if batch:
        emitted += 1
        yield _pack(batch, fidelity)


def _pack(batch, fidelity):
    matrix = np.concatenate([np.asarray(s["features"], dtype=np.float32) for s in batch], axis=0)
    counts = [len(s["targets"]) for s in batch]
    returns = np.array([[float(s["return_task"]), float(s["return_cost"])] for s in batch],
                       dtype=np.float64)
    meta = {"joint_rows": len(batch), "target_rows": int(sum(counts)),
            "episodes": sorted({int(s["episode_index"]) for s in batch})}
    return matrix, counts, returns, meta


def r68_initial_controller(seed: int):
    """The verified ROUND68 initial controller as the same 5-tuple ``fresh_r68_controller``
    returned BEFORE the harness rebound ``controller_module.fresh_controller``.

    ``configure_and_inject`` (run by every ROUND71 entry point) rebinds
    ``v2_controller.fresh_controller`` to the frozen partitioned factory, so calling
    ``r68.fresh_r68_controller`` AFTER configure wraps an already-partitioned controller a
    second time and fails.  ROUND60's own ``fresh_partitioned_controller`` documents the same
    hazard and builds ``v2_controller.StageCController`` directly.  This is the identical
    construction (same class, same seed, same RNG consumption order) through the identical
    verified chain (R60 ep0 model-only load + frozen early path + one late-head scale), and it
    writes no identity file.
    """
    import r66_frozen as r66
    import r68_frozen as r68
    import v2_controller as vc
    r66.configure_source_imports()          # idempotent; re-verifies canonical byte identity
    base = vc.StageCController(int(seed))
    return r68.build_frozen_controller(base, int(seed))


def offline_fit_seed(*, seed: int, gate: LicenseGate, model_read_approval, run_identity,
                     ledger: ForwardLedger, init_path: Path, identity_path: Path,
                     overwrite: bool = False) -> dict:
    """The full per-seed calibration: R68 ep0 -> guarded extraction -> ridge -> install -> INIT.

    Gates first (no .pt is opened without a verified licence and an explicit model-read
    approval), then the R68 ep0 model-only controller, then the fit and the install, then the
    fresh optimizer / lambda / initial RNG, then the NEW INIT checkpoint and its identity.
    """
    if not isinstance(gate, LicenseGate):
        raise LicenseInvalid("a LicenseGate is required for the offline fit")
    gate.require("fit_real_data")
    gate.require("model_forward")
    gate.require("checkpoint_write")
    if not model_read_approval:
        raise ModelReadBoundary(
            "ROUND71 offline fit stops at the model read-in boundary: an explicit model-read "
            "approval is required before the ep0 model is read")
    for path in (init_path, identity_path):
        DuplicateOutputGuard(path).check(overwrite=overwrite)
    import r66_frozen as r66
    import r68_frozen as r68
    import v2_trainer as trainer

    # 1+2: the verified ROUND68 initial controller (model-only ROUND60 ep0 + late scale)
    controller, ep0, frozen_identity, post, scale_id = r68_initial_controller(int(seed))
    rng_after_construction = trainer.rng_state()
    init_payload = {
        "identity": {"seed": int(seed), "structure": readiness.SEAM_R68_STRUCTURE,
                     "training_version": readiness.R68_TRAINING_VERSION_EXPECTED,
                     "module_set": list(readiness.MODULES), "source_kind": "r68_ep0"},
        "file_sha256": ep0["file_sha256"],
        "parameter_sha256": ep0["parameter_sha256"],
        "post_scale_parameter_sha256": post["parameter_sha256"],
        "pre_scale_equals_r60_ep0": scale_id["pre_scale_equals_r60_ep0"],
        "episodes_done": ep0["episodes_done"], "updates": ep0["updates"],
        "load_mode": ep0["load_mode"],
        "optimizer_restored": bool(ep0["optimizer_restored"]),
        "lambda_restored": False, "rng_restored": bool(ep0["rng_restored"]),
        "from_trained_ep128": False}
    init_role = readiness.validate_calibration_source(init_payload,
                                                      role="r68_ep0_model_only_init")
    mask = r66.assert_frozen_mask(controller)
    if not scale_id["pre_scale_equals_r60_ep0"]:
        raise RuntimeError(f"seed {seed}: pre-scale parameters are not the compliant ROUND60 ep0")

    # 3-5: guarded extraction, fixed ridge, install into the two critic parameters only
    provider = readiness.CalibrationProvider(controller, ledger)
    fit = provider.extract_and_fit(teacher_batches(int(seed)))
    head_set = provider.calibrated_heads(fit)
    install = provider.install_critic_heads(head_set)
    if not install["real_backend"]:
        raise RuntimeError("the real torch copy_ install backend was not used")
    rng_after_install = trainer.rng_state()
    if rng_after_install != rng_after_construction:
        raise RuntimeError(f"seed {seed}: the calibration changed the RNG state")

    # 6: fresh optimizer and fresh training state with the ROUND71 contract
    optimizer = r66.build_optimizer(controller, 3e-4)
    state = r68.fresh_training_state(
        int(seed), gae_lambda=1.0, eta_lambda=0.5, gamma=1.0, delta=0.0, lambda_init=1.0,
        data_order_seed=20260927, b0_bytes=10000.0, alpha=r68.LATE_INIT_ALPHA)
    state["training_version"] = readiness.R71_TRAINING_VERSION
    state["loss_version"] = readiness.R71_LOSS_VERSION
    state["state_schema"] = readiness.R71_STATE_SCHEMA
    state["optimizer_schema"] = readiness.R71_OPTIMIZER_SCHEMA
    state["structure"] = readiness.R71_STRUCTURE
    state["calibration_route"] = readiness.R71_CALIBRATION_ROUTE
    state["run_identity"] = run_identity.as_dict() if run_identity is not None else None
    state["critic_heads_calibration"] = {
        "ridge": fit["ridge"], "rows": fit["rows"],
        "joint_rows": ledger.joint_rows, "input_rows_charged": ledger.rows,
        "changed_modules": install["changed_modules"],
        "digest_before": install["digest_before"], "digest_after": install["digest_after"]}

    # 7: the NEW INIT checkpoint + identity evidence
    summary = controller.parameter_summary()
    module_sha = {name: r68._module_sha(controller, name)
                  for name in ("trunk", "early_head", "late_head", "value_task_head", "cost_head")}
    identity = {
        "kind": "round71-init-identity/1", "seed": int(seed),
        "task_id": "ROUND71-APPROVED-CALIBRATION-FIXED-TRIAL",
        "source": {"r60_ep0_parameter_sha256": ep0["parameter_sha256"],
                   "r60_ep0_file_sha256": ep0["file_sha256"],
                   "pre_scale_equals_r60_ep0": scale_id["pre_scale_equals_r60_ep0"],
                   "r68_post_scale_parameter_sha256": post["parameter_sha256"],
                   "late_init_alpha": r68.LATE_INIT_ALPHA},
        "calibration": {
            "value_task_head_sha256": module_sha["value_task_head"],
            "value_cost_head_sha256": module_sha["cost_head"],
            "trunk_sha256": module_sha["trunk"],
            "early_head_sha256": module_sha["early_head"],
            "late_head_sha256": module_sha["late_head"],
            "fit": {"ridge": fit["ridge"], "rows": fit["rows"],
                    "center_scale_scope": fit["center_scale_scope"],
                    "constant_column_scale": fit["constant_column_scale"],
                    "intercept_penalized": fit["intercept_penalized"]},
            "install": install},
        "init_parameter_sha256": summary["parameter_sha256"],
        "init_parameter_object_count": summary["parameter_object_count"],
        "rng_state_sha256": _rng_digest(rng_after_install),
        "rng_unchanged_by_calibration": True,
        "optimizer_fresh": True, "lambda_init": 1.0,
        "offline_ledger": {"rows": ledger.rows, "calls": ledger.calls,
                           "joint_rows": ledger.joint_rows, "remaining": ledger.remaining(),
                           "cap": OFFLINE_INPUT_ROW_CAP},
        "frozen_mask": mask, "init_role": init_role}
    readiness.r71_save_checkpoint(controller, optimizer, state, init_path)
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
    return {"seed": int(seed), "init_path": str(init_path), "identity_path": str(identity_path),
            "fit": fit, "install": install, "identity": identity,
            "ledger": {"rows": ledger.rows, "calls": ledger.calls,
                       "joint_rows": ledger.joint_rows, "remaining": ledger.remaining()}}


def gate_fit_probe(*, seed: int, gate: LicenseGate, model_read_approval, ledger: ForwardLedger,
                   limit_batches: int = 1) -> dict:
    """A bounded REAL fit used only by the pre-launch gates (never saved as an INIT).

    Proves the transform/install/invariance machinery on real forwards and real parameters
    with a small, charged sample; its coefficients are discarded.
    """
    if not isinstance(gate, LicenseGate):
        raise LicenseInvalid("a LicenseGate is required for the gate probe")
    gate.require("fit_real_data")
    gate.require("model_forward")
    if not model_read_approval:
        raise ModelReadBoundary("the gate probe stops at the model read-in boundary")
    import r66_frozen as r66
    import r68_frozen as r68
    import v2_trainer as trainer
    controller, ep0, frozen_identity, post, scale_id = r68_initial_controller(int(seed))
    rng_after_construction = trainer.rng_state()
    before_digest = readiness.module_parameter_digest(controller)
    provider = readiness.CalibrationProvider(controller, ledger)
    fit = provider.extract_and_fit(teacher_batches(int(seed), limit_batches=limit_batches))
    head_set = provider.calibrated_heads(fit)
    install = provider.install_critic_heads(head_set)
    after_digest = readiness.module_parameter_digest(controller)
    rng_after_install = trainer.rng_state()
    return {"seed": int(seed), "fit": fit, "install": install,
            "digest_before": before_digest, "digest_after": after_digest,
            "rng_unchanged": rng_after_install == rng_after_construction,
            "controller": controller,
            "ep0": ep0, "post": post, "scale_id": scale_id,
            "frozen_mask": r66.assert_frozen_mask(controller),
            "ledger": {"rows": ledger.rows, "calls": ledger.calls,
                       "joint_rows": ledger.joint_rows, "remaining": ledger.remaining()}}


def _rng_digest(rng_state) -> str:
    import hashlib
    import numpy as _np
    blob = json.dumps(rng_state, sort_keys=True, default=lambda o: _np.asarray(o).tolist())
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()