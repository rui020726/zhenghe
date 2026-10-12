"""ROUND68 frozen identity + late-initial-confidence intervention.

ROUND68 asks: ROUND66's late_head parameters demonstrably moved, yet the behaviour was identical to
BC; ROUND67 showed the visited late probabilities were highly saturated (p ~ 0.999). Does lowering
the late head's INITIAL confidence (a one-time 0.10 scaling of late_head.weight and late_head.bias on
the fresh model built from the compliant ROUND60 ep0) let the trainable late head produce any
additional learning effect? This is NOT a claim that saturation is the only root cause.

The ONLY scientific difference from ROUND66 is the one-time late-head init scale. Everything else is
inherited: trunk + early_head stay frozen (requires_grad=False); only late_head + both critics train;
the critic->trunk isolation, the routed forward, the sequential mask, the joint log-probability, the
combined advantage lambda*A_task - A_cost, GAE, ALL_ROWS, the global gradient clip, the ratio clip,
the beta=1.00 gate prior and the dual update are all ROUND66's, unchanged. The scale is applied ONCE
at fresh construction and is NEVER a per-forward temperature and NEVER re-applied on resume.

This module wraps r66_frozen (imported from the ROUND66 directory) and adds:
  * LATE_INIT_ALPHA = 0.10 and scale_late_head_init(controller, alpha);
  * build_frozen_controller(base, seed) = r66.build_frozen_controller(...) then one scale;
  * an R68 training/model version + checkpoint schema that carries alpha, the frozen mask and the
    ROUND60 ep0 source identity, and that REJECTS ROUND60 and ROUND66 checkpoints as resume sources;
  * R68 save/load/restore that never re-scale on resume.
"""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
_R2 = HERE.parent
_R66_DIR = _R2 / "round66_bc_early_path_preserving_late_adaptation"
if str(_R66_DIR) not in sys.path:
    sys.path.insert(0, str(_R66_DIR))
import r66_frozen as r66  # noqa: E402
import torch  # noqa: E402

# Re-export the R66 machinery unchanged (training update, optimizer, controller, R60 sources, ...).
from r66_frozen import (  # noqa: E402,F401
    PartitionedController, fresh_partitioned_controller, load_r60_model, r60_source_identity,
    partitioned_parameters, freeze_early_path, trainable_parameters, frozen_parameters,
    frozen_parameter_summary, build_optimizer, frozen_early_ppo_update, likelihood_alignment,
    FROZEN_MODULES, TRAINABLE_MODULES, configure_source_imports, load_r60_checkpoint_read_only,
)

# ------------------------------------------------------------------ R68 identity ------------
LATE_INIT_ALPHA = 0.10
R68_TRAINING_VERSION = r66.R66_TRAINING_VERSION + "+r68-late-initial-confidence-alpha0.10/1"
R68_LOSS_VERSION = r66.R66_LOSS_VERSION  # loss unchanged; the scale is init-only
R68_STATE_SCHEMA = r66.R66_STATE_SCHEMA + "+r68-late-init-alpha/1"
R68_OPTIMIZER_SCHEMA = r66.R66_OPTIMIZER_SCHEMA + "+r68-late-init-alpha/1"
R68_TRAINER_TAG = "constrained-cost-v2-frozen-early-path-late-init-alpha"
R68_STRUCTURE = r66.R66_STRUCTURE + "+late-initial-confidence-alpha0.10/1"
R68_FREEZE_KIND = "round68-late-initial-confidence-freeze/1"


def frozen_parameter_summary(controller: "PartitionedController") -> dict[str, Any]:
    """ROUND68 structure stamp: identical tensors/mask to ROUND66 but the ROUND68 checkpoint
    contract version, so resume/gate/acceptance structure checks are ROUND68-specific."""
    summary = dict(r66.frozen_parameter_summary(controller))
    summary["structure"] = R68_STRUCTURE
    return summary

def _module_sha(controller: "PartitionedController", module_name: str) -> str:
    module = {"trunk": controller.inner.trunk, "early_head": controller.inner.early_head,
              "late_head": controller.inner.late_head,
              "value_task_head": controller.inner.value_head,
              "cost_head": controller.cost_head}[module_name]
    digest = hashlib.sha256()
    for name, tensor in sorted(module.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def scale_late_head_init(controller: "PartitionedController",
                         alpha: float = LATE_INIT_ALPHA) -> dict[str, Any]:
    """One-time late_head.weight *= alpha and late_head.bias *= alpha on the fresh controller.

    A single positive factor scales the late logit margin (logit_SEND - logit_SKIP) by alpha without
    changing any greedy argmax (sign preserved). Only the two late_head tensors move; trunk,
    early_head and both critics are byte-identical before and after. Returns the before/after
    identity so a caller can prove the scale happened exactly once.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0,1); got {alpha}")
    before = {name: _module_sha(controller, name)
              for name in ("trunk", "early_head", "late_head", "value_task_head", "cost_head")}
    late = controller.inner.late_head
    if not hasattr(late, "weight") or not hasattr(late, "bias"):
        raise RuntimeError("late_head is not a Linear with weight/bias")
    with torch.no_grad():
        late.weight.mul_(float(alpha))
        late.bias.mul_(float(alpha))
    after = {name: _module_sha(controller, name)
             for name in ("trunk", "early_head", "late_head", "value_task_head", "cost_head")}
    unchanged = all(before[n] == after[n] for n in ("trunk", "early_head", "value_task_head", "cost_head"))
    if not unchanged or before["late_head"] == after["late_head"]:
        raise RuntimeError("scale_late_head_init moved modules other than late_head, or none at all")
    return {"late_init_alpha": float(alpha),
            "scaled_tensors": ["late_head.weight", "late_head.bias"],
            "late_head_sha_before_scale": before["late_head"],
            "late_head_sha_after_scale": after["late_head"],
            "unchanged_modules_verified": ["trunk", "early_head", "value_task_head", "cost_head"],
            "full_parameter_sha256_after_scale": controller.parameter_summary()["parameter_sha256"]}


def build_frozen_controller(base_controller: Any, seed: int,
                            alpha: float = LATE_INIT_ALPHA) -> tuple:
    """R66 frozen-early-path controller from the compliant ROUND60 ep0, THEN one late-head scale.

    r66.build_frozen_controller loads all five modules bit-exactly from ROUND60 ep0 and freezes the
    early path, verifying the parameters equal the ROUND60 ep0 identity. Only after that verified
    equality do we scale late_head once. So the pre-scale controller IS the ROUND60 ep0 value and the
    post-scale controller is the ROUND68 initial value. ROUND60 ep128, Adam, lambda and training RNG
    are never touched.
    """
    controller, ep0, frozen_identity, summary = r66.build_frozen_controller(base_controller, seed)
    pre_scale_sha = controller.parameter_summary()["parameter_sha256"]
    scale_identity = scale_late_head_init(controller, alpha)
    post = controller.parameter_summary()
    scale_identity["pre_scale_parameter_sha256"] = pre_scale_sha
    scale_identity["pre_scale_equals_r60_ep0"] = (pre_scale_sha == ep0["parameter_sha256"])
    scale_identity["post_scale_parameter_sha256"] = post["parameter_sha256"]
    scale_identity["r60_ep0_parameter_sha256"] = ep0["parameter_sha256"]
    if not scale_identity["pre_scale_equals_r60_ep0"]:
        raise RuntimeError(f"seed {seed}: pre-scale controller is not the compliant ROUND60 ep0 value")
    return controller, ep0, frozen_identity, post, scale_identity


def fresh_r68_controller(seed: int, alpha: float = LATE_INIT_ALPHA):
    """Convenience: build the R68 frozen, late-scaled controller from ROUND60 ep0 for seed."""
    import v2_controller  # noqa: F401  (ensures the canonical controller module is importable)
    base = r66.configure_source_imports()  # idempotent; guarantees canonical modules on sys.path
    import v2_controller as vc
    b = vc.fresh_controller(int(seed))
    return build_frozen_controller(b, int(seed), alpha)


def fresh_training_state(training_seed: int, *, gae_lambda: float = 1.0, eta_lambda: float = 0.5,
                         gamma: float = 1.0, delta: float = 0.0, lambda_init: float = 1.0,
                         data_order_seed: int = 20260927, b0_bytes: float = 10000.0,
                         alpha: float = LATE_INIT_ALPHA) -> dict[str, Any]:
    state = r66.fresh_training_state(int(training_seed), gae_lambda=gae_lambda, eta_lambda=eta_lambda,
                                     gamma=gamma, delta=delta, lambda_init=lambda_init,
                                     data_order_seed=data_order_seed, b0_bytes=b0_bytes)
    state["training_version"] = R68_TRAINING_VERSION
    state["loss_version"] = R68_LOSS_VERSION
    state["state_schema"] = R68_STATE_SCHEMA
    state["optimizer_schema"] = R68_OPTIMIZER_SCHEMA
    state["structure"] = R68_STRUCTURE
    state["late_init_alpha"] = float(alpha)
    return state


# ------------------------------------------------------------------ R68 checkpoint ----------
def save_checkpoint(controller: "PartitionedController", optimizer, state: dict[str, Any],
                    path: Path, *, freeze: dict[str, Any]) -> Path:
    """Save the R68 five modules, the six-parameter optimizer and the frozen mask + late-init alpha.

    The scaled late_head is stored as-is, so a resume loads the already-scaled head and never
    re-scales. The frozen mask records the late_init_alpha and the scaled late_head sha so a restore
    can prove the head it loads is the scaled one.
    """
    import v2_trainer as trainer
    import os
    frozen_identity = freeze_early_path(controller)
    late_sha = _module_sha(controller, "late_head")
    payload = {
        "schema": R68_STATE_SCHEMA, "trainer_tag": R68_TRAINER_TAG,
        "training_version": R68_TRAINING_VERSION,
        "training_version_source": trainer.TRAINING_VERSION,
        "reward_timing_version": trainer.REWARD_TIMING_VERSION,
        "loss_version": R68_LOSS_VERSION,
        "gae_schema": trainer.GAE_SCHEMA, "optimizer_schema": R68_OPTIMIZER_SCHEMA,
        "reward_timing_schema": trainer.REWARD_TIMING_SCHEMA,
        "structure": R68_STRUCTURE,
        "late_init_alpha": float(state.get("late_init_alpha", LATE_INIT_ALPHA)),
        "freeze": freeze,
        "controller": {key: value for key, value in controller.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "optimizer_param_groups": [len(g["params"]) for g in optimizer.state_dict()["param_groups"]],
        "frozen_mask": {"frozen_modules": list(FROZEN_MODULES),
                        "trainable_modules": list(TRAINABLE_MODULES),
                        "frozen_sha256": frozen_identity["sha256"],
                        "frozen_scalar_count": frozen_identity["scalar_count"],
                        "frozen_parameter_object_count": frozen_identity["tensor_count"],
                        "late_init_alpha": float(state.get("late_init_alpha", LATE_INIT_ALPHA)),
                        "late_head_scaled_sha256": late_sha,
                        "note": "requires_grad is not serialised; late_head is stored already scaled; resume never re-scales"},
        "training_state": state,
        "rng": trainer.rng_state(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def _reject_foreign_contract(path: Path, payload: dict[str, Any]) -> None:
    """ROUND60 and ROUND66 checkpoints are never a ROUND68 training-resume source."""
    name = Path(path).name
    if payload.get("schema") in (r66.R60_STATE_SCHEMA, r66.R66_STATE_SCHEMA) or \
            payload.get("trainer_tag") in (r66.R60_TRAINER_TAG, r66.R66_TRAINER_TAG):
        raise RuntimeError(
            f"{name}: this is a ROUND60/ROUND66 checkpoint; those are model-only sources and are "
            "REJECTED as a ROUND68 training-resume source")
    if payload.get("schema") != R68_STATE_SCHEMA or payload.get("trainer_tag") != R68_TRAINER_TAG:
        raise RuntimeError(f"{name}: not a ROUND68 late-initial-confidence checkpoint")


def _validate_base(path: Path, payload: dict[str, Any], *, freeze: dict[str, Any]) -> dict[str, Any]:
    _reject_foreign_contract(path, payload)
    if payload.get("loss_version") != R68_LOSS_VERSION:
        raise RuntimeError(f"{Path(path).name}: loss version mismatch")
    if payload.get("gae_schema") != "per-decision-next-value/1" or payload.get("optimizer_schema") != R68_OPTIMIZER_SCHEMA:
        raise RuntimeError(f"{Path(path).name}: trainer contract schema mismatch")
    if payload.get("structure") != R68_STRUCTURE:
        raise RuntimeError(f"{Path(path).name}: structure mismatch")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict) or (
            training_state.get("loss_version") != R68_LOSS_VERSION
            or training_state.get("state_schema") != R68_STATE_SCHEMA
            or training_state.get("optimizer_schema") != R68_OPTIMIZER_SCHEMA):
        raise RuntimeError(f"{Path(path).name}: training state version mismatch")
    if payload.get("freeze") != freeze:
        raise RuntimeError(f"{Path(path).name}: freeze identity mismatch; refusing to resume")
    return training_state


def _restore_controller(payload: dict[str, Any], training_state: dict[str, Any]) -> "PartitionedController":
    seed = int(training_state["training_seed"])
    controller = fresh_partitioned_controller(seed, ep0_state=None, verify_ep0=False)
    saved = payload["controller"]
    if set(saved) != {"trunk", "early_head", "late_head", "value_task_head", "value_cost_head"}:
        raise RuntimeError("checkpoint controller modules are not the partitioned five-module schema")
    controller.load_state_dict(saved)  # late_head loaded ALREADY scaled; no re-scale here
    identity = freeze_early_path(controller)
    mask = payload.get("frozen_mask") or {}
    if mask.get("frozen_sha256") != identity["sha256"]:
        raise RuntimeError("restored frozen mask does not match the restored frozen tensors")
    if list(mask.get("frozen_modules", [])) != list(FROZEN_MODULES):
        raise RuntimeError("restored frozen module list mismatch")
    if list(mask.get("trainable_modules", [])) != list(TRAINABLE_MODULES):
        raise RuntimeError("restored trainable module list mismatch")
    if "late_head_scaled_sha256" in mask and mask["late_head_scaled_sha256"] != _module_sha(controller, "late_head"):
        raise RuntimeError("restored late_head is not the scaled head recorded at save")
    return controller


def load_checkpoint(path: Path, *, freeze: dict[str, Any]):
    """Resume only a ROUND68 late-initial-confidence checkpoint from this exact version."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    training_state = _validate_base(path, payload, freeze=freeze)
    controller = _restore_controller(payload, training_state)
    optimizer = build_optimizer(controller, 3e-4)
    if [len(g["params"]) for g in payload["optimizer"]["param_groups"]] != [6]:
        raise RuntimeError("ROUND68 checkpoint optimizer is not the six-parameter trainable set")
    optimizer.load_state_dict(payload["optimizer"])
    import v2_trainer as trainer
    trainer.restore_rng(payload.get("rng"))
    return controller, optimizer, training_state


def load_checkpoint_read_only(path: Path, *, freeze: dict[str, Any]):
    """Read-only load of a ROUND68 checkpoint: weights and frozen mask only, never RNG or Adam."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    training_state = _validate_base(path, payload, freeze=freeze)
    controller = _restore_controller(payload, training_state)
    identity = {"load_mode": "read_only", "training_version": payload.get("training_version"),
                "reward_timing_version": payload.get("reward_timing_version"),
                "optimizer_restored": False, "rng_restored": False,
                "structure": R68_STRUCTURE, "late_init_alpha": payload.get("late_init_alpha"),
                "frozen_mask": payload.get("frozen_mask"),
                "parameter_identity": frozen_parameter_summary(controller)}
    return controller, training_state, identity


print("r68_frozen module loaded; LATE_INIT_ALPHA=", LATE_INIT_ALPHA)
