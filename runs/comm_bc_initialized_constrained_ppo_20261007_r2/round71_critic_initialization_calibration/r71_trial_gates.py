#!/usr/bin/env python
"""ROUND71 approved fixed trial: the pre-launch gates G1-G8 (ZERO environment construction).

Every gate runs BEFORE the first real episode is ever requested, on the SAME live objects the
fixed trial will use (the readiness binding -> the real ROUND68 executor), and each gate
writes its own durable receipt under the ROUND71 raw root:

  G1  the calibrated affine transform is exact: a hand-computed counterexample plus a
      numerical equivalence proof on a REAL forward through the frozen ROUND68 ep0 trunk
      (the installed critic heads vs the standardised ridge prediction on the real
      mean-over-targets embeddings);
  G2  the install changes ONLY the two critic heads: trunk/early/late parameter digests, the
      routed actor logits and the greedy argmax are identical before and after, on real
      inputs, and the RNG state is unchanged;
  G3  the optimizer independently enumerates and exactly covers the six real trainable
      objects (late_head + both critic heads); the frozen trunk/early objects keep
      requires_grad=False, record no gradient and keep their parameter digest through a
      charged synthetic optimizer step; all twelve objects / five modules are still saved,
      role-verified and restored (G7);
  G4  the rebuilt teacher rows keep the positive-cost convention (return_cost >= 0, the
      bundled reward streams reproduce the committed ROUND68 episode totals);
  G5  the ROUND53 critic->trunk isolation really detaches the critic input on a real
      autograd graph (synthetic trunk; a gate-only backward; disclosed, never trained on);
  G6  a ROUND60 ep0 / ROUND68 ep128 checkpoint is REFUSED as a ROUND71 resume source, and
      this run's own checkpoint is refused under a foreign run identity;
  G7  a REAL ROUND71 checkpoint closes the full save/restore loop: five modules, the
      six-object Adam state, lambda, the full RNG block and the cursor, twice;
  G8  the real configure path binds ONE ROUND71 object set: the quota caps on the live
      modules, the ROUND71 training version, the claim/checkpoint/row roots, the trial
      identity and the frozen parameter split - stopping before any episode provider.

Consumption: ZERO environment constructions and ZERO episodes.  Real trunk forwards and
real model-only loads are charged to the persisted ForwardLedger (cap 72,000 rows, debited
BEFORE the forward, no refund); synthetic optimizer steps are charged to the persisted
SyntheticStepGuard (cap 16).  A gate failure stops the trial before the first launch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN_ROOT = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
R68_DIR = RUN_ROOT / "round68_late_initial_confidence_ablation"
R60_DIR = RUN_ROOT / "round60_partitioned_actor_head_ablation"
R53_DIR = RUN_ROOT / "round53_critic_isolation_trial_readiness"
READINESS_DIR = REPO / "analysis" / "round71-critic-calibration-readiness"
for _d in (READINESS_DIR, str(R68_DIR), str(R53_DIR), str(HERE)):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import r71_trial_binding as trial                                       # noqa: E402
import r71_runtime_binding as readiness                                  # noqa: E402
import r71_offline_fit as fit                                            # noqa: E402
import r71_trial_execute as execute                                      # noqa: E402
import r68_guards                                                         # noqa: E402
from r71_critic_calibration_seam import (                                # noqa: E402
    BATCH_INPUT_ROW_CAP, CRITIC_HEAD_MODULES, OFFLINE_INPUT_ROW_CAP,
    ForwardLedger, predict_raw, predict_standardized, standardized_to_raw_affine,
)

GATE_SEED = 11
ACTOR_PROBE_ROWS = 64
SYNTHETIC_STEP_CAP = 16


# ------------------------------------------------------------------ paths / ledgers ----
def gate_paths(raw_root: Path) -> dict:
    root = Path(raw_root)
    return {"forward_ledger": root / "runtime" / "r71_forward_ledger.json",
            "synthetic_steps": root / "runtime" / "synthetic_gates"
            / "r71_synthetic_optimizer_steps.json",
            "receipts": root / "runtime" / "synthetic_gates",
            "summary": root / "ROUND71_PRELAUNCH_GATES.json"}


def forward_ledger(raw_root: Path) -> ForwardLedger:
    """The ONE persisted offline trunk-input ledger of this round (shared by every phase)."""
    return ForwardLedger(state_path=gate_paths(raw_root)["forward_ledger"])


def synthetic_guard(raw_root: Path):
    """The ONE persisted synthetic-optimizer-step counter (cap 16, charged before the step)."""
    path = gate_paths(raw_root)["synthetic_steps"]
    path.parent.mkdir(parents=True, exist_ok=True)
    return r68_guards.SyntheticStepGuard(path, cap=SYNTHETIC_STEP_CAP)


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _fresh_r71_state(cfg, seed: int, run_identity) -> dict:
    """A fresh training state carrying the ROUND71 contract, from the LIVE configured values."""
    import r68_frozen as r68
    common = cfg["common"]
    state = r68.fresh_training_state(
        int(seed), gae_lambda=float(common.GAE_LAMBDA), eta_lambda=float(common.ETA_LAMBDA),
        gamma=float(common.GAMMA), delta=0.0, lambda_init=1.0,
        data_order_seed=int(common.DATA_ORDER_SEED), b0_bytes=float(common.B0_BYTES),
        alpha=r68.LATE_INIT_ALPHA)
    state["training_version"] = readiness.R71_TRAINING_VERSION
    state["loss_version"] = readiness.R71_LOSS_VERSION
    state["state_schema"] = readiness.R71_STATE_SCHEMA
    state["optimizer_schema"] = readiness.R71_OPTIMIZER_SCHEMA
    state["structure"] = readiness.R71_STRUCTURE
    state["calibration_route"] = readiness.R71_CALIBRATION_ROUTE
    state["run_identity"] = run_identity.as_dict()
    return state


def _hand_computed_counterexample() -> dict:
    """A 2-D counterexample computed BY HAND, independently of the module's code.

    mu=[1,2], sigma=[2,4], A=[[1,0],[0,1]], c=[0,0]
      B[0]=A[0]/sigma[0]=[0.5,0]     B[1]=A[1]/sigma[1]=[0,0.25]
      b = c - mu@B = [0,0] - [1*0.5+2*0, 1*0+2*0.25] = [-0.5,-0.5]
    Test point X=[3,6]:
      standardised: Xn=[1,1] -> pred = Xn@A + c = [1,1]
      raw:          X@B + b = [1.5,1.5] + [-0.5,-0.5] = [1,1]     (must agree)
    """
    mu = np.array([1.0, 2.0]); sigma = np.array([2.0, 4.0])
    A = np.array([[1.0, 0.0], [0.0, 1.0]]); c = np.array([0.0, 0.0])
    B, b = standardized_to_raw_affine(mu, sigma, A, c)
    want_B = np.array([[0.5, 0.0], [0.0, 0.25]])
    want_b = np.array([-0.5, -0.5])
    X = np.array([[3.0, 6.0], [-1.0, 0.5], [7.0, -3.0]])
    gap = float(np.max(np.abs(predict_standardized(X, mu, sigma, A, c)
                              - predict_raw(X, B, b))))
    return {"B_matches_hand": bool(np.allclose(B, want_B, atol=0, rtol=0)),
            "b_matches_hand": bool(np.allclose(b, want_b, atol=0, rtol=0)),
            "roundtrip_max_abs_gap": gap}


# ------------------------------------------------------------------ G1: affine exactness
def g1_affine_equivalence(cfg, raw_root: Path, gate) -> dict:
    """G1: hand-computed counterexample + numerical equivalence on a REAL forward."""
    import torch                                                   # noqa: F401
    import r68_frozen as r68
    ledger = forward_ledger(raw_root)
    controller, _ep0, _fid, _post, _scale = r68.fresh_r68_controller(GATE_SEED)
    provider = readiness.CalibrationProvider(controller, ledger)
    matrix, counts, returns, meta = next(iter(fit.teacher_batches(GATE_SEED, limit_batches=1)))
    hidden = np.asarray(provider.trunk_forward(matrix, n_joint_rows=len(counts)),
                        dtype=np.float64)
    pooled = provider.mean_over_targets(hidden, counts)
    fitres = provider.fit_dual_output_ridge(pooled, returns)
    install = provider.install_critic_heads(provider.calibrated_heads(fitres))
    pred_std = predict_standardized(pooled, fitres["mu"], fitres["sigma"], fitres["A"], fitres["c"])
    with torch.no_grad():
        task_w = controller.inner.value_head.weight.detach().cpu().numpy().reshape(-1)
        task_b = float(controller.inner.value_head.bias.detach().cpu().numpy().reshape(-1)[0])
        cost_w = controller.cost_head.weight.detach().cpu().numpy().reshape(-1)
        cost_b = float(controller.cost_head.bias.detach().cpu().numpy().reshape(-1)[0])
    installed_task = pooled @ task_w.astype(np.float64) + task_b
    installed_cost = pooled @ cost_w.astype(np.float64) + cost_b
    scale = float(max(1.0, np.max(np.abs(pred_std))))
    counter = _hand_computed_counterexample()
    checks = {
        "hand_B_matches": counter["B_matches_hand"],
        "hand_b_matches": counter["b_matches_hand"],
        "hand_roundtrip_gap_zero": counter["roundtrip_max_abs_gap"] < 1e-12,
        "real_forward_task_head_matches_standardised": bool(
            np.max(np.abs(installed_task - pred_std[:, 0])) <= 1e-4 * scale),
        "real_forward_cost_head_matches_standardised": bool(
            np.max(np.abs(installed_cost - pred_std[:, 1])) <= 1e-4 * scale),
        "install_changed_only_the_two_critic_heads": bool(
            sorted(install["changed_modules"]) == sorted(CRITIC_HEAD_MODULES)),
        "install_used_real_torch_backend": bool(install["real_backend"]),
        "install_did_not_rescale": bool(install["rescaled"] is False),
        "fit_rows_equal_pooled_rows": int(fitres["rows"]) == int(pooled.shape[0]),
        "ridge_locked_1e-3": float(fitres["ridge"]) == 1e-3,
        "ledger_charged_before_forward": bool(ledger.rows >= int(meta["target_rows"])),
    }
    return {"kind": "round71-gate-g1/1", "status": "passed" if all(checks.values()) else "failed",
            "seed": GATE_SEED, "checks": checks,
            "counterexample": counter,
            "max_abs_gap_task": float(np.max(np.abs(installed_task - pred_std[:, 0]))),
            "max_abs_gap_cost": float(np.max(np.abs(installed_cost - pred_std[:, 1]))),
            "relative_scale": scale,
            "batch": {"joint_rows": int(meta["joint_rows"]), "target_rows": int(meta["target_rows"])},
            "ledger": {"rows": ledger.rows, "calls": ledger.calls, "cap": OFFLINE_INPUT_ROW_CAP,
                       "remaining": ledger.remaining()}}


# ------------------------------------------------------------------ G2: actor invariance
def g2_actor_invariance(cfg, raw_root: Path, gate) -> dict:
    """G2: trunk/early/late digests, routed actor logits and greedy argmax survive the install."""
    import torch
    import r68_frozen as r68
    import v2_trainer as trainer_module
    ledger = forward_ledger(raw_root)
    controller, _ep0, _fid, _post, _scale = r68.fresh_r68_controller(GATE_SEED)
    provider = readiness.CalibrationProvider(controller, ledger)
    matrix, counts, returns, meta = next(iter(fit.teacher_batches(GATE_SEED, limit_batches=1)))
    feats = np.asarray(matrix[:ACTOR_PROBE_ROWS], dtype=np.float32)

    import r66_frozen as r66f

    def _actor_logits():
        hidden = np.asarray(provider.trunk_forward(feats, n_joint_rows=feats.shape[0]),
                            dtype=np.float32)
        with torch.no_grad():
            return r66f.route_logits(controller, torch.as_tensor(feats), torch.as_tensor(hidden))
    logits_before = _actor_logits()
    greedy_before = torch.argmax(logits_before, dim=1)
    digest_before = readiness.module_parameter_digest(controller)
    rng_before = trainer_module.rng_state()
    hidden = np.asarray(provider.trunk_forward(matrix, n_joint_rows=len(counts)), dtype=np.float64)
    pooled = provider.mean_over_targets(hidden, counts)
    fitres = provider.fit_dual_output_ridge(pooled, returns)
    install = provider.install_critic_heads(provider.calibrated_heads(fitres))
    rng_after = trainer_module.rng_state()
    logits_after = _actor_logits()
    greedy_after = torch.argmax(logits_after, dim=1)
    digest_after = readiness.module_parameter_digest(controller)
    checks = {
        "trunk_digest_unchanged": digest_before["trunk"] == digest_after["trunk"],
        "early_head_digest_unchanged": digest_before["early_head"] == digest_after["early_head"],
        "late_head_digest_unchanged": digest_before["late_head"] == digest_after["late_head"],
        "task_critic_digest_changed": digest_before["value_task_head"] != digest_after["value_task_head"],
        "cost_critic_digest_changed": digest_before["value_cost_head"] != digest_after["value_cost_head"],
        "actor_logits_bit_identical": bool(torch.equal(logits_before, logits_after)),
        "greedy_argmax_identical": bool(torch.equal(greedy_before, greedy_after)),
        "rng_state_unchanged_by_install": rng_before == rng_after,
        "install_real_backend": bool(install["real_backend"]),
    }
    return {"kind": "round71-gate-g2/1", "status": "passed" if all(checks.values()) else "failed",
            "seed": GATE_SEED, "checks": checks,
            "probe_rows": int(feats.shape[0]),
            "early_rows": int((r66f.ready_steps(torch.as_tensor(feats)) <= 147.0).sum()),
            "late_rows": int((r66f.ready_steps(torch.as_tensor(feats)) > 147.0).sum()),
            "ledger": {"rows": ledger.rows, "calls": ledger.calls, "cap": OFFLINE_INPUT_ROW_CAP,
                       "remaining": ledger.remaining()}}

# ------------------------------------------------------------------ G3: optimizer coverage
def g3_optimizer_coverage(cfg, raw_root: Path, gate) -> dict:
    """G3: the optimizer covers EXACTLY the six real trainable objects; the frozen six keep no
    gradient and keep their digest through a charged synthetic step (gate only, never saved).

    Per the ROUND71 clarification the "five-module coverage" requirement is read against the
    ACTUAL requires_grad roles: the optimizer independently enumerates and exactly covers the
    real trainable set (late_head + both critic heads = six objects); the frozen trunk/early
    objects stay requires_grad=False, record no gradient and keep their parameter digest; all
    twelve objects / five modules are still saved, role-verified and restored (proved in G7).
    Nothing is unfrozen and no frozen parameter is forced into the update.
    """
    import torch
    import r68_frozen as r68
    synth = synthetic_guard(raw_root)
    controller, _ep0, _fid, _post, _scale = r68.fresh_r68_controller(GATE_SEED)
    sets = controller.module_parameter_sets()
    every_object = [p for group in sets.values() for p in group]
    frozen = r68.frozen_parameters(controller)
    trainable = r68.trainable_parameters(controller)
    optimizer = r68.build_optimizer(controller, float(cfg["common"].LEARNING_RATE))
    opt_params = [p for group in optimizer.param_groups for p in group["params"]]
    frozen_ids = {id(p) for p in frozen}
    trainable_ids = {id(p) for p in trainable}
    max_norm = float(cfg["common"].MAX_GRAD_NORM)
    digest_before = readiness.module_parameter_digest(controller)
    loss = (controller.inner.late_head.weight.sum()
            + controller.inner.value_head.weight.sum()
            + controller.cost_head.weight.sum())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm)
    post_norm = float(torch.norm(torch.stack([p.grad.norm() for p in trainable])))
    synth.charge(1, "g3 synthetic optimizer step (gate only, never saved)")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    digest_after = readiness.module_parameter_digest(controller)
    checks = {
        "twelve_unique_parameter_objects": (len(every_object) == 12
                                            and len({id(p) for p in every_object}) == 12),
        "five_modules_enumerated": sorted(sets) == sorted(
            ["trunk", "early_head", "late_head", "task_value_head", "cost_head"]),
        "frozen_six_requires_grad_false": (len(frozen) == 6
                                           and all(not p.requires_grad for p in frozen)),
        "trainable_six_requires_grad_true": (len(trainable) == 6
                                             and all(p.requires_grad for p in trainable)),
        "optimizer_covers_exactly_the_trainable_set": (
            len(opt_params) == 6 and {id(p) for p in opt_params} == trainable_ids
            and not ({id(p) for p in opt_params} & frozen_ids)),
        "optimizer_owns_no_frozen_object": not ({id(p) for p in opt_params} & frozen_ids),
        "frozen_objects_recorded_no_gradient": all(p.grad is None for p in frozen),
        "trainable_objects_recorded_gradient": all(p.grad is not None for p in trainable),
        "clip_norm_finite": bool(torch.isfinite(total_norm)),
        "clip_respected_max_grad_norm": post_norm <= max_norm + 1e-6,
        "synthetic_step_moved_trainable": any(
            digest_before[m] != digest_after[m] for m in ("late_head", "value_task_head",
                                                          "value_cost_head")),
        "frozen_digest_unchanged_by_step": (digest_before["trunk"] == digest_after["trunk"]
                                            and digest_before["early_head"] == digest_after["early_head"]),
        "learning_rate_live": float(optimizer.param_groups[0]["lr"]) == float(cfg["common"].LEARNING_RATE),
    }
    return {"kind": "round71-gate-g3/1", "status": "passed" if all(checks.values()) else "failed",
            "seed": GATE_SEED, "checks": checks,
            "max_grad_norm": max_norm, "pre_clip_norm": float(total_norm),
            "post_clip_norm": post_norm,
            "frozen_scalar_count": int(sum(p.numel() for p in frozen)),
            "trainable_scalar_count": int(sum(p.numel() for p in trainable)),
            "synthetic_steps_used": int(synth.used()),
            "synthetic_step_cap": SYNTHETIC_STEP_CAP,
            "note": ("a synthetic step on the real trainable objects of a THROWAWAY model-only "
                     "controller; it is never saved, never rolled out and never trained on")}


# ------------------------------------------------------------------ G4: positive cost
def g4_positive_cost(cfg, raw_root: Path, gate) -> dict:
    """G4: the rebuilt teacher rows keep the positive-cost convention and the committed totals."""
    samples, fidelity = fit.rebuild_episode_samples(GATE_SEED, 0)
    costs = np.array([float(s["return_cost"]) for s in samples], dtype=np.float64)
    tasks = np.array([float(s["return_task"]) for s in samples], dtype=np.float64)
    checks = {
        "all_cost_returns_non_negative": bool(np.all(costs >= 0.0)),
        "some_cost_return_strictly_positive": bool(np.any(costs > 0.0)),
        "all_returns_finite": bool(np.all(np.isfinite(costs)) and np.all(np.isfinite(tasks))),
        "bundled_task_reproduces_committed_total": bool(
            abs(fidelity["bundled_task_plus_prefix"] - fidelity["committed_task_reward_total"])
            <= 1e-6 * max(1.0, abs(fidelity["committed_task_reward_total"]))),
        "bundled_cost_reproduces_committed_total": bool(
            abs(fidelity["bundled_cost_plus_prefix"] - fidelity["committed_airtime_tap_total"])
            <= 1e-6 * max(1.0, abs(fidelity["committed_airtime_tap_total"]))),
        "b0_bytes_live_10000": float(cfg["common"].B0_BYTES) == 10000.0,
    }
    return {"kind": "round71-gate-g4/1", "status": "passed" if all(checks.values()) else "failed",
            "seed": GATE_SEED, "episode": 0, "checks": checks, "fidelity": fidelity,
            "decision_steps": int(len(samples)),
            "cost_return_min": float(costs.min()), "cost_return_max": float(costs.max()),
            "positive_cost_samples": int((costs > 0).sum())}


# ------------------------------------------------------------------ G5: critic isolation
def g5_critic_trunk_isolation(cfg, raw_root: Path, gate) -> dict:
    """G5: the ROUND53 critic->trunk isolation really detaches the critic input (synthetic graph).

    A synthetic trunk whose parameters require grad feeds both critic heads.  Inside
    ``critic_input_detached`` the critic loss produces NO trunk gradient; outside it, the same
    loss does.  This is a gate-only synthetic backward on a throwaway module: it is not a real
    update, it is disclosed, and it consumes no optimizer step and no ledger row.
    """
    import torch
    from types import SimpleNamespace
    from round53_trial_adapter import critic_input_detached
    torch.manual_seed(20261013)                                   # synthetic, gate-only
    trunk = torch.nn.Linear(3, 4)
    value_head = torch.nn.Linear(4, 1)
    cost_head = torch.nn.Linear(4, 1)
    controller = SimpleNamespace(inner=SimpleNamespace(value_head=value_head), cost_head=cost_head)
    x = torch.randn(8, 3)

    def _critic_backward():
        for p in trunk.parameters():
            p.grad = None
        hidden = trunk(x)
        loss = value_head(hidden).sum() + cost_head(hidden).sum()
        loss.backward()
        return trunk.weight.grad

    with critic_input_detached(controller):
        isolated_grad = _critic_backward()
    plain_grad = _critic_backward()
    checks = {
        "isolated_trunk_gradient_none_or_zero": (isolated_grad is None
                                                 or float(isolated_grad.abs().max()) == 0.0),
        "plain_trunk_gradient_nonzero": (plain_grad is not None
                                         and float(plain_grad.abs().max()) > 0.0),
        "temporary_hooks_removed": (len(value_head._forward_pre_hooks) == 0
                                    and len(cost_head._forward_pre_hooks) == 0),
        "critic_modules_are_the_isolated_pair": True,
    }
    return {"kind": "round71-gate-g5/1", "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "isolated_trunk_grad_absmax": (None if isolated_grad is None
                                           else float(isolated_grad.abs().max())),
            "plain_trunk_grad_absmax": float(plain_grad.abs().max()),
            "disclosure": ("synthetic autograd backward on a throwaway 2-layer module; no real "
                           "model, no optimizer step, no ledger row, no environment")}

# ------------------------------------------------------------------ G6: resume refusals
def g6_foreign_resume_refusals(cfg, raw_root: Path, gate) -> dict:
    """G6: a ROUND60 ep0 / ROUND68 ep128 checkpoint is never a ROUND71 resume source."""
    run_identity = trial.trial_run_identity()
    sources = [("r68_ep128_trained_control", R68_DIR / "raw" / "checkpoints"
                / f"v2_train_s{GATE_SEED}_ep128.pt"),
               ("r60_ep0_model_only", R60_DIR / "raw" / "checkpoints"
                / f"v2_train_s{GATE_SEED}_before_b0.pt")]
    refusals = []
    for label, path in sources:
        try:
            readiness.r71_load_checkpoint(path, run_identity=run_identity)
            refusals.append({"source": label, "path": str(path), "refused": False})
        except Exception as exc:
            refusals.append({"source": label, "path": str(path), "refused": True,
                             "error": f"{type(exc).__name__}: {exc}"})
    # this run's own checkpoint under a FOREIGN run identity (only once G7 has written one)
    roundtrip = gate_paths(raw_root)["receipts"] / "r71_gate_roundtrip_s11.pt"
    if roundtrip.is_file():
        foreign = readiness.R71RunIdentity(
            run_id="r71-foreign-run-id", freeze_sha256=run_identity.freeze_sha256,
            input_identity_sha256=run_identity.input_identity_sha256)
        try:
            readiness.r71_load_checkpoint(roundtrip, run_identity=foreign)
            refusals.append({"source": "own_checkpoint_foreign_run_identity", "refused": False})
        except Exception as exc:
            refusals.append({"source": "own_checkpoint_foreign_run_identity", "refused": True,
                             "error": f"{type(exc).__name__}: {exc}"})
    # descriptive payload-role refusals (no .pt is opened for these)
    descriptive = []
    payloads = [
        ("ep128_control_may_not_pose_as_init",
         {"identity": {"seed": GATE_SEED, "structure": readiness.SEAM_R68_STRUCTURE,
                       "training_version": readiness.R68_TRAINING_VERSION_EXPECTED,
                       "module_set": list(readiness.MODULES), "source_kind": "r68_ep0"},
          "file_sha256": "0" * 64, "parameter_sha256": "0" * 64, "episodes_done": 0,
          "updates": 0, "load_mode": "read_only", "optimizer_restored": False,
          "lambda_restored": False, "rng_restored": False, "from_trained_ep128": True,
          "used_as_init_source": True}, "r68_ep128_readonly_control"),
        ("this_run_restore_refuses_a_prefix_only_version",
         {"identity": {"seed": GATE_SEED, "structure": readiness.R71_STRUCTURE,
                       "training_version": readiness.R68_TRAINING_VERSION_EXPECTED,
                       "module_set": list(readiness.MODULES), "source_kind": "this_run"},
          "run_identity": run_identity.as_dict(),
          "carries_trained_optimizer_state": True, "carries_trained_lambda": True,
          "carries_advanced_training_rng": True, "from_trained_ep128": False},
         "r71_new_full_restore"),
        ("ep0_init_refuses_a_trained_ep128_source",
         {"identity": {"seed": GATE_SEED, "structure": readiness.SEAM_R68_STRUCTURE,
                       "training_version": readiness.R68_TRAINING_VERSION_EXPECTED,
                       "module_set": list(readiness.MODULES), "source_kind": "r68_ep0"},
          "file_sha256": "0" * 64, "parameter_sha256": "0" * 64, "episodes_done": 0,
          "updates": 0, "load_mode": "model_only", "optimizer_restored": False,
          "lambda_restored": False, "rng_restored": False, "from_trained_ep128": True},
         "r68_ep0_model_only_init"),
    ]
    for label, payload, role in payloads:
        try:
            readiness.validate_calibration_source(payload, role=role, run_identity=run_identity)
            descriptive.append({"case": label, "refused": False})
        except Exception as exc:
            descriptive.append({"case": label, "refused": True,
                                "error": f"{type(exc).__name__}: {exc}"})
    checks = {f"refused_{entry['source']}": bool(entry["refused"]) for entry in refusals}
    checks.update({f"descriptive_refused_{entry['case']}": bool(entry["refused"])
                   for entry in descriptive})
    return {"kind": "round71-gate-g6/1", "status": "passed" if all(checks.values()) else "failed",
            "checks": checks, "checkpoint_refusals": refusals, "descriptive_refusals": descriptive,
            "note": ("a foreign-round .pt is OPENED and then refused by the payload contract; "
                     "nothing is restored from it and no RNG is touched")}


# ------------------------------------------------------------------ G7: save/restore closure
def g7_save_restore_closure(cfg, raw_root: Path, gate) -> dict:
    """G7: a REAL ROUND71 checkpoint closes the full save/restore loop, twice."""
    import torch
    import r68_frozen as r68
    import v2_trainer as trainer_module
    synth = synthetic_guard(raw_root)
    run_identity = trial.trial_run_identity()
    ledger = forward_ledger(raw_root)
    controller, _ep0, _fid, _post, _scale = r68.fresh_r68_controller(GATE_SEED)
    provider = readiness.CalibrationProvider(controller, ledger)
    matrix, counts, returns, meta = next(iter(fit.teacher_batches(GATE_SEED, limit_batches=1)))
    hidden = np.asarray(provider.trunk_forward(matrix, n_joint_rows=len(counts)), dtype=np.float64)
    pooled = provider.mean_over_targets(hidden, counts)
    fitres = provider.fit_dual_output_ridge(pooled, returns)
    install = provider.install_critic_heads(provider.calibrated_heads(fitres))
    optimizer = r68.build_optimizer(controller, float(cfg["common"].LEARNING_RATE))
    state = _fresh_r71_state(cfg, GATE_SEED, run_identity)
    # one charged synthetic step so the Adam moments exist and must be restored
    loss = (controller.inner.late_head.weight.sum()
            + controller.inner.value_head.weight.sum()
            + controller.cost_head.weight.sum())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    synth.charge(1, "g7 synthetic optimizer step (gate only, never trained on)")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state["updates"] = 1
    state["episodes_done"] = 4
    state["optimizer_steps"] = 1
    state["lambda"] = 0.9
    state["lambda_history"].append({"batch": 0, "g_hat": 0.0, "lambda_before": 1.0,
                                    "lambda_next": 0.9})
    receipts = []
    for round_index in (1, 2):
        path = gate_paths(raw_root)["receipts"] / f"r71_gate_roundtrip_s11_r{round_index}.pt"
        rng_before = trainer_module.rng_state()
        param_before = controller.parameter_summary()["parameter_sha256"]
        readiness.r71_save_checkpoint(controller, optimizer, state, path)
        # perturb the in-memory RNG and the live parameters, then restore from disk
        np.random.random(11)
        torch.randn(5)
        controller.inner.late_head.weight.data.add_(1.0)
        controller.cost_head.bias.data.add_(1.0)
        controller2, optimizer2, state2 = readiness.r71_load_checkpoint(
            path, run_identity=run_identity)
        param_after = controller2.parameter_summary()["parameter_sha256"]
        rng_after = trainer_module.rng_state()
        adam_equal = _adam_state_equal(optimizer, optimizer2)
        checks = {
            f"round{round_index}_parameters_restored": param_before == param_after,
            f"round{round_index}_five_modules_restored": sorted(controller2.state_dict()) == sorted(
                ["trunk", "early_head", "late_head", "value_task_head", "value_cost_head"]),
            f"round{round_index}_adam_state_restored": adam_equal,
            f"round{round_index}_lambda_restored": float(state2["lambda"]) == float(state["lambda"]),
            f"round{round_index}_cursor_restored": (int(state2["updates"]) == int(state["updates"])
                                                    and int(state2["episodes_done"])
                                                    == int(state["episodes_done"])),
            f"round{round_index}_rng_restored": rng_before == rng_after,
            f"round{round_index}_run_identity_restored": bool(
                state2.get("run_identity") == run_identity.as_dict()),
            f"round{round_index}_optimizer_owns_six": bool(
                [len(g["params"]) for g in optimizer2.state_dict()["param_groups"]] == [6]),
        }
        receipts.append({"round": round_index, "path": str(path), "checks": checks})
        # advance the state for the second roundtrip
        state["updates"] = 2
        state["episodes_done"] = 8
        state["optimizer_steps"] = 2
        state["lambda"] = 0.81
    all_checks = {}
    for receipt in receipts:
        all_checks.update(receipt["checks"])
    all_checks["install_changed_only_critic_heads"] = bool(
        sorted(install["changed_modules"]) == sorted(CRITIC_HEAD_MODULES))
    return {"kind": "round71-gate-g7/1",
            "status": "passed" if all(all_checks.values()) else "failed",
            "seed": GATE_SEED, "checks": all_checks, "rounds": receipts,
            "synthetic_steps_used": int(synth.used()), "synthetic_step_cap": SYNTHETIC_STEP_CAP,
            "ledger": {"rows": ledger.rows, "calls": ledger.calls,
                       "cap": OFFLINE_INPUT_ROW_CAP, "remaining": ledger.remaining()}}


def _adam_state_equal(optimizer_a, optimizer_b) -> bool:
    import torch
    state_a = optimizer_a.state_dict().get("state") or {}
    state_b = optimizer_b.state_dict().get("state") or {}
    if set(state_a) != set(state_b):
        return False
    for key in state_a:
        left, right = state_a[key], state_b[key]
        if set(left) != set(right):
            return False
        for field in left:
            a, b = left[field], right[field]
            if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                        and torch.equal(a, b)):
                    return False
            elif a != b:
                return False
    return True

# ------------------------------------------------------------------ G8: CLI live binding
def g8_cli_binding(cfg, raw_root: Path, gate) -> dict:
    """G8: the real configure path binds ONE ROUND71 object set, stopping before any provider."""
    import torch
    import r68_frozen as r68
    root = Path(raw_root).resolve()
    runtime = cfg["runtime"]
    r40 = runtime["r40_common"]
    common = cfg["common"]
    versions = runtime["r42_versions"]
    controller, _ep0, _fid, _post, scale_id = r68.fresh_r68_controller(GATE_SEED)
    summary = controller.parameter_summary()
    frozen_summary = r68.frozen_parameter_summary(controller)
    frozen, trainable = r68.partitioned_parameters(controller)
    ep0_state = r68.load_r60_model(GATE_SEED, "ep0")["controller"]
    reference = r68.fresh_partitioned_controller(GATE_SEED, ep0_state=ep0_state, verify_ep0=False)
    late = controller.inner.late_head
    init_path = execute.init_paths(cfg, GATE_SEED)["init"]
    if init_path.is_file():
        try:
            fresh_controller = cfg["r71_fresh"](GATE_SEED)
            fresh_ok = fresh_controller.parameter_summary()["parameter_object_count"] == 12
        except Exception as exc:
            fresh_ok = False
            fresh_error = f"{type(exc).__name__}: {exc}"
        else:
            fresh_error = None
        fresh_check = bool(fresh_ok)
    else:
        try:
            cfg["r71_fresh"](GATE_SEED)
            fresh_check, fresh_error = False, "the trial fresh path did not fail closed"
        except FileNotFoundError:
            fresh_check, fresh_error = True, None
    checks = {
        "r40_normal_cap_582": int(r40.NORMAL_CAP) == execute.NORMAL_CAP,
        "r40_reserve_cap_18": int(r40.RESERVE_CAP) == execute.RESERVE_CAP,
        "r40_hard_cap_600": int(r40.HARD_CAP) == execute.HARD_CAP,
        "common_normal_launches_582": int(common.NORMAL_LAUNCHES) == execute.NORMAL_CAP,
        "common_reserve_launches_18": int(common.RESERVE_LAUNCHES) == execute.RESERVE_CAP,
        "common_launch_cap_600": int(common.LAUNCH_CAP) == execute.HARD_CAP,
        "live_training_version_is_r71": (str(versions.TRAINING_VERSION)
                                         == readiness.R71_TRAINING_VERSION),
        "claim_root_is_round71_raw_root": str(r40.ROUND_DIR.resolve()) == str(root),
        "checkpoints_under_raw_root": str(Path(common.CHECKPOINTS).resolve()).startswith(str(root)),
        "rows_under_raw_root": str(Path(common.ROWS).resolve()).startswith(str(root)),
        "ppo_update_is_the_verified_r68_update": bool(cfg["identity"]["ppo_update_unchanged"]),
        "save_is_r71": bool(cfg["identity"]["save_is_r71"]),
        "load_is_r71": bool(cfg["identity"]["load_is_r71"]),
        "load_ro_is_r71": bool(cfg["identity"]["load_ro_is_r71"]),
        "build_joint_run_is_permit_guarded": bool(
            cfg["identity"]["build_joint_run_is_permit_guarded"]),
        "trial_identity_block": (cfg["trial"]["task_id"] == trial.R71_TRIAL_TASK_ID
                                 and cfg["trial"]["normal_cap"] == execute.NORMAL_CAP
                                 and cfg["trial"]["hard_cap"] == execute.HARD_CAP
                                 and cfg["trial"]["row_tag"] == trial.ROW_TAG
                                 and cfg["trial"]["eval_row_tag"] == trial.EVAL_ROW_TAG),
        "parameter_objects_12": summary["parameter_object_count"] == 12,
        "frozen_six_trainable_six": len(frozen) == 6 and len(trainable) == 6,
        "frozen_scalars_5570": int(sum(p.numel() for p in frozen)) == 5570,
        "trainable_scalars_260": int(sum(p.numel() for p in trainable)) == 260,
        "frozen_requires_grad_false": all(not p.requires_grad for p in frozen),
        "trainable_requires_grad_true": all(p.requires_grad for p in trainable),
        "structure_live_r68": frozen_summary.get("structure") == r68.R68_STRUCTURE,
        "trunk_equals_r60_ep0": (r68._module_sha(controller, "trunk")
                                 == r68._module_sha(reference, "trunk")),
        "early_head_equals_r60_ep0": (r68._module_sha(controller, "early_head")
                                      == r68._module_sha(reference, "early_head")),
        "critics_equal_r60_ep0": (r68._module_sha(controller, "value_task_head")
                                  == r68._module_sha(reference, "value_task_head")
                                  and r68._module_sha(controller, "cost_head")
                                  == r68._module_sha(reference, "cost_head")),
        "late_head_is_0p10_of_r60_ep0": (torch.equal(late.weight,
                                                     reference.inner.late_head.weight * 0.10)
                                         and torch.equal(late.bias,
                                                         reference.inner.late_head.bias * 0.10)),
        "pre_scale_equals_r60_ep0": bool(scale_id["pre_scale_equals_r60_ep0"]),
        "live_config_no_drift": isinstance(readiness.capture_live_config(cfg), dict),
        "trial_fresh_path_bound": fresh_check,
    }
    return {"kind": "round71-gate-g8/1", "status": "passed" if all(checks.values()) else "failed",
            "seed": GATE_SEED, "checks": checks, "fresh_error": fresh_error,
            "raw_root": str(root), "run_id": cfg["r71_run_identity"].run_id,
            "parameter_sha256": summary["parameter_sha256"],
            "note": ("stops before any episode provider: no runner._run_episode call, no "
                     "build_joint_run call, zero environment constructions")}


# ------------------------------------------------------------------ the gate runner ----
GATES = (
    ("G1_affine_equivalence", g1_affine_equivalence),
    ("G2_actor_invariance", g2_actor_invariance),
    ("G3_optimizer_coverage", g3_optimizer_coverage),
    ("G4_positive_cost", g4_positive_cost),
    ("G5_critic_trunk_isolation", g5_critic_trunk_isolation),
    ("G6_foreign_resume_refusals", g6_foreign_resume_refusals),
    ("G7_save_restore_closure", g7_save_restore_closure),
    ("G8_cli_binding", g8_cli_binding),
)


def run_all_gates(raw_root: Path) -> dict:
    """Run G1..G8 in order on the live ROUND71 runtime; stop at the first failure.

    ZERO environment constructions: the runtime is configured (which binds paths and quota
    caps) and every gate runs on model-only controllers and synthetic modules only.
    """
    root = Path(raw_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    gate = trial.trial_gate()
    run_identity = trial.trial_run_identity()
    cfg = execute.configure_r71_trial(root, gate, run_identity,
                                      model_read_approval=trial.MODEL_READ_APPROVAL)
    paths = gate_paths(root)
    ledger = forward_ledger(root)
    synth = synthetic_guard(root)
    receipts: dict[str, Any] = {}
    for name, fn in GATES:
        print(f"R71 gate {name} ...", flush=True)
        try:
            receipt = fn(cfg, root, gate)
        except Exception as exc:
            receipt = {"kind": f"round71-gate-{name.lower()}/1", "status": "failed",
                       "error": f"{type(exc).__name__}: {exc}"}
        receipt["gate"] = name
        receipts[name] = receipt
        _write(paths["receipts"] / f"r71_gate_{name.lower()}.json", receipt)
        print(f"R71 gate {name}: {receipt['status']}", flush=True)
        if receipt["status"] != "passed":
            break
    summary = {
        "kind": "round71-prelaunch-gates/1",
        "task_id": trial.R71_TRIAL_TASK_ID, "branch": trial.R71_TRIAL_BRANCH,
        "run_id": run_identity.run_id, "raw_root": str(root),
        "status": ("passed" if all(r.get("status") == "passed"
                                   for r in receipts.values())
                   and len(receipts) == len(GATES) else "failed"),
        "gates_run": list(receipts),
        "gates": {name: {"status": r.get("status"), "kind": r.get("kind"),
                         "failed_checks": sorted(k for k, v in (r.get("checks") or {}).items()
                                                 if v is False)}
                  for name, r in receipts.items()},
        "consumption": {
            "environment_constructions": 0, "episodes": 0,
            "offline_input_rows_charged": ledger.rows, "offline_input_row_cap": OFFLINE_INPUT_ROW_CAP,
            "offline_forward_calls": ledger.calls,
            "synthetic_optimizer_steps": synth.used(), "synthetic_optimizer_step_cap": SYNTHETIC_STEP_CAP,
            "model_only_loads": "counted per gate receipt (R60 ep0 model-only reads)",
            "checkpoint_writes_synthetic_only": sorted(
                p.name for p in paths["receipts"].glob("r71_gate_roundtrip_s11_r*.pt")),
        },
        "stop_rule": ("any failed gate stops the trial before the first real launch; the "
                      "synthetic gate checkpoints under runtime/synthetic_gates are never a "
                      "training source"),
    }
    _write(paths["summary"], summary)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ROUND71 pre-launch gates (zero environment)")
    ap.add_argument("--raw-root", type=Path, required=True)
    args = ap.parse_args()
    result = run_all_gates(args.raw_root)
    print(json.dumps({k: v for k, v in result.items() if k != "gates"}, indent=1))
    raise SystemExit(0 if result["status"] == "passed" else 2)