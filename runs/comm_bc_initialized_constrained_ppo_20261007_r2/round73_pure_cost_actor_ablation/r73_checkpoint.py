# -*- coding: utf-8 -*-
"""ROUND73 checkpoint contract: the five modules under the pure-cost actor objective.

The ROUND71 five-module checkpoint contract is extended by exactly one thing: the actor
objective identity (``pure-cost/1``).  Everything structural - the five modules, the six
trainable parameter objects, the frozen trunk/early_head mask, the optimizer layout, the
training-state schema - is inherited unchanged from the verified ROUND71 chain.

Consequences enforced here:

  * an ROUND73 training checkpoint carries the ROUND73 training version, which EXTENDS the
    ROUND71 one (never a truncation), so a ROUND71 training checkpoint (including an ep128)
    is refused as a ROUND73 resume source and vice versa;
  * the per-seed INIT is the ROUND71 calibrated INIT file itself, copied byte-exact and
    loaded model-only; it is validated against the ROUND71 contract AND against the
    registered ROUND73 init identity (file/parameter/module digests from the ROUND71
    checkpoint index), never against a name or a caller flag;
  * a full restore requires the optimizer, lambda and advanced training RNG to be present
    and the payload run-identity block to equal the live one field for field.

Importing this module opens no .pt and runs no forward.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN_ROOT = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
R66_DIR = RUN_ROOT / "round66_bc_early_path_preserving_late_adaptation"
R68_DIR = RUN_ROOT / "round68_late_initial_confidence_ablation"
READINESS_DIR = REPO / "analysis" / "round71-critic-calibration-readiness"
for _d in (READINESS_DIR, str(R68_DIR), str(R66_DIR)):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import r71_runtime_binding as readiness                       # noqa: E402
import r73_pure_cost_objective as r73                         # noqa: E402
from r71_critic_calibration_seam import FROZEN_MODULES, MODULES, TRAINABLE_MODULES  # noqa: E402

# ---------------------------------------------------------------------- identity ----
# The ROUND73 training version EXTENDS the verified ROUND71 one; it never truncates it.
R73_TRAINING_VERSION = readiness.R71_TRAINING_VERSION + "+r73-pure-cost-actor-objective/1"
assert r73.R73_TRAINING_VERSION == R73_TRAINING_VERSION, (
    "r73_pure_cost_objective.R73_TRAINING_VERSION drifted from the ROUND71-extended literal")

R73_LOSS_VERSION = readiness.R71_LOSS_VERSION + "+pure-cost-actor-objective/1"
R73_STATE_SCHEMA = (readiness.R71_STATE_SCHEMA + "+r73-pure-cost-actor-objective/1")
R73_OPTIMIZER_SCHEMA = (readiness.R71_OPTIMIZER_SCHEMA + "+r73-pure-cost-actor-objective/1")
R73_STRUCTURE = (readiness.R71_STRUCTURE + "+r73-pure-cost-actor-objective/1")
R73_TRAINER_TAG = readiness.R71_TRAINER_TAG + "-pure-cost-actor"
R73_FREEZE_KIND = "round73-pure-cost-actor-freeze/1"
R73_ACTOR_OBJECTIVE = r73.R73_ACTOR_OBJECTIVE
R73_OBJECTIVE_VERSION = r73.R73_OBJECTIVE_VERSION

R73_FREEZE = dict(
    readiness.R71_FREEZE,
    kind=R73_FREEZE_KIND,
    training_version=R73_TRAINING_VERSION,
    structure=R73_STRUCTURE,
    loss_version=R73_LOSS_VERSION,
    actor_objective=R73_ACTOR_OBJECTIVE,
    objective_version=R73_OBJECTIVE_VERSION,
    control_r71="A = lambda_dual*A_task - A_cost",
    new_r73="A = -A_cost (task term and lambda dropped from the actor surrogate only)",
    intervention_scope=("ONLY the combined actor advantage formed in "
                        "canonical_v2_trainer.batch_samples; dual critics, GAE, reward timing, "
                        "mask, entropy, global clip and the dual-lambda update are unchanged"),
    init_source=("the ROUND71 calibrated INIT checkpoint per seed, copied byte-exact and loaded "
                 "model-only (five modules; fresh optimizer / lambda / training RNG); no offline "
                 "fit is run and no ROUND71 trained weights, Adam state or lambda are inherited"),
    inherits_from=("round71-critic-initialization-calibration (runtime, loss, config, routed "
                   "update, critic->trunk isolation, five-module checkpoint); only the actor "
                   "advantage differs"),
)

_FOREIGN_TAGS = (readiness.R71_TRAINER_TAG,
                 "constrained-cost-v2-partitioned-early-late-actor-head",
                 "constrained-cost-v2-frozen-early-path",
                 "constrained-cost-v2-frozen-early-path-late-init-alpha")


# ---------------------------------------------------------------------- helpers ----
def _reject_foreign(path, payload: dict) -> None:
    """ROUND60/66/68/71 checkpoints are never a ROUND73 training-resume source."""
    name = Path(path).name
    if payload.get("trainer_tag") in _FOREIGN_TAGS:
        raise RuntimeError(
            f"{name}: this is a ROUND60/66/68/71 checkpoint; those are model-only or "
            f"foreign-round sources and are REJECTED as a ROUND73 training-resume source")
    if payload.get("schema") != R73_STATE_SCHEMA or payload.get("trainer_tag") != R73_TRAINER_TAG:
        raise RuntimeError(f"{name}: not a ROUND73 pure-cost-actor checkpoint")


def _validate_r73_base(path, payload: dict) -> dict:
    _reject_foreign(path, payload)
    if payload.get("loss_version") != R73_LOSS_VERSION:
        raise RuntimeError(f"{Path(path).name}: loss version mismatch")
    if payload.get("actor_objective") != R73_ACTOR_OBJECTIVE:
        raise RuntimeError(f"{Path(path).name}: actor objective is not {R73_ACTOR_OBJECTIVE!r}")
    if payload.get("gae_schema") != "per-decision-next-value/1" or \
            payload.get("optimizer_schema") != R73_OPTIMIZER_SCHEMA:
        raise RuntimeError(f"{Path(path).name}: trainer contract schema mismatch")
    if payload.get("structure") != R73_STRUCTURE:
        raise RuntimeError(f"{Path(path).name}: structure mismatch")
    if payload.get("freeze") != R73_FREEZE:
        raise RuntimeError(f"{Path(path).name}: freeze identity mismatch; refusing to resume")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict) or (
            training_state.get("loss_version") != R73_LOSS_VERSION
            or training_state.get("state_schema") != R73_STATE_SCHEMA
            or training_state.get("optimizer_schema") != R73_OPTIMIZER_SCHEMA):
        raise RuntimeError(f"{Path(path).name}: training state version mismatch")
    return training_state


def _restore_controller(payload: dict, training_state: dict):
    import r66_frozen as r66
    seed = int(training_state["training_seed"])
    controller = r66.fresh_partitioned_controller(seed, ep0_state=None, verify_ep0=False)
    saved = payload["controller"]
    if set(saved) != set(MODULES):
        raise RuntimeError("checkpoint controller modules are not the partitioned five-module schema")
    controller.load_state_dict(saved)
    identity = r66.freeze_early_path(controller)
    mask = payload.get("frozen_mask") or {}
    if mask.get("frozen_sha256") != identity["sha256"]:
        raise RuntimeError("restored controller does not match the frozen mask identity")
    return controller


def r73_save_checkpoint(controller, optimizer, state: dict, path) -> str:
    """Save the ROUND73 five modules + six-parameter optimizer + frozen mask + objective id."""
    import torch
    import v2_trainer as trainer
    import r66_frozen as r66
    frozen_identity = r66.freeze_early_path(controller)
    optimizer_state = optimizer.state_dict()
    rng_payload = trainer.rng_state()
    payload = {
        "schema": R73_STATE_SCHEMA, "trainer_tag": R73_TRAINER_TAG,
        "training_version": R73_TRAINING_VERSION,
        "training_version_source": trainer.TRAINING_VERSION,
        "reward_timing_version": trainer.REWARD_TIMING_VERSION,
        "loss_version": R73_LOSS_VERSION,
        "actor_objective": R73_ACTOR_OBJECTIVE,
        "objective_version": R73_OBJECTIVE_VERSION,
        "gae_schema": trainer.GAE_SCHEMA, "optimizer_schema": R73_OPTIMIZER_SCHEMA,
        "reward_timing_schema": trainer.REWARD_TIMING_SCHEMA,
        "structure": R73_STRUCTURE,
        "calibration_route": readiness.R71_CALIBRATION_ROUTE,
        "late_init_alpha": float(state.get("late_init_alpha", 0.10)),
        "freeze": R73_FREEZE,
        "controller": {key: value for key, value in controller.state_dict().items()},
        "optimizer": optimizer_state,
        "optimizer_param_groups": [len(g["params"]) for g in optimizer_state["param_groups"]],
        "frozen_mask": {"frozen_modules": list(FROZEN_MODULES),
                        "trainable_modules": list(TRAINABLE_MODULES),
                        "frozen_sha256": frozen_identity["sha256"],
                        "frozen_scalar_count": frozen_identity["scalar_count"],
                        "frozen_parameter_object_count": frozen_identity["tensor_count"],
                        "critic_heads_calibrated": True,
                        "note": "requires_grad is not serialised; the calibrated critic heads "
                                "are stored as-is; resume never re-fits or re-scales"},
        "training_state": state,
        "run_identity": state.get("run_identity"),
        "rng": rng_payload,
        # self-describing restore roles (a full ROUND73 training resume, never model-only).
        # Computed from the locals above: the dict literal below must never read `payload`,
        # which does not exist until the literal finishes evaluating.
        "carries_trained_optimizer_state": bool(optimizer_state["state"]),
        "carries_trained_lambda": "lambda" in state and "lambda_history" in state,
        "carries_advanced_training_rng": rng_payload is not None,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return str(path)


def r73_load_checkpoint(path, *, run_identity=None):
    """Resume only a ROUND73 checkpoint of THIS run (payload-bound role validation)."""
    import torch
    import r66_frozen as r66
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    training_state = _validate_r73_base(path, payload)
    if run_identity is None:
        raise RuntimeError("a live ROUND73 run identity is required to restore a checkpoint")
    if not run_identity.matches_block(payload.get("run_identity")):
        raise RuntimeError(f"{Path(path).name}: the payload run_identity block does not equal the "
                           f"live ROUND73 identity field for field")
    for flag in ("carries_trained_optimizer_state", "carries_trained_lambda",
                 "carries_advanced_training_rng"):
        if not payload.get(flag):
            raise RuntimeError(f"{Path(path).name}: this run's full restore must carry {flag}")
    if payload.get("from_trained_ep128"):
        raise RuntimeError(f"{Path(path).name}: a trained ep128 is never a restore source")
    controller = _restore_controller(payload, training_state)
    optimizer = r66.build_optimizer(controller, 3e-4)
    if [len(g["params"]) for g in payload["optimizer"]["param_groups"]] != [6]:
        raise RuntimeError("ROUND73 checkpoint optimizer is not the six-parameter trainable set")
    optimizer.load_state_dict(payload["optimizer"])
    import v2_trainer as trainer
    trainer.restore_rng(payload.get("rng"))
    return controller, optimizer, training_state


def r73_load_checkpoint_read_only(path):
    """Read-only load of a ROUND73 checkpoint: weights and frozen mask only, never RNG or Adam."""
    import torch
    import r66_frozen as r66
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    training_state = _validate_r73_base(path, payload)
    controller = _restore_controller(payload, training_state)
    identity = {"load_mode": "read_only", "training_version": payload.get("training_version"),
                "reward_timing_version": payload.get("reward_timing_version"),
                "actor_objective": payload.get("actor_objective"),
                "optimizer_restored": False, "rng_restored": False,
                "structure": R73_STRUCTURE,
                "frozen_mask": payload.get("frozen_mask"),
                "parameter_identity": r66.frozen_parameter_summary(controller)}
    return controller, training_state, identity


def r73_load_init_read_only(path, *, registered: dict):
    """Read-only load of the ROUND71 calibrated INIT (model-only) behind its registered identity.

    The INIT payload is a ROUND71 checkpoint by construction (it IS the ROUND71 calibrated
    init file, copied byte-exact); it is accepted only for the model-only INIT role and only
    while its file/parameter identity equals the registered ROUND73 init identity, which is
    taken from the ROUND71 checkpoint index.  No optimizer, lambda or training RNG is read.
    """
    import torch
    import r66_frozen as r66
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # the ROUND71 contract, field for field (this file was written by the ROUND71 chain)
    if payload.get("schema") != readiness.R71_STATE_SCHEMA or \
            payload.get("trainer_tag") != readiness.R71_TRAINER_TAG or \
            payload.get("training_version") != readiness.R71_TRAINING_VERSION or \
            payload.get("structure") != readiness.R71_STRUCTURE or \
            payload.get("loss_version") != readiness.R71_LOSS_VERSION or \
            payload.get("optimizer_schema") != readiness.R71_OPTIMIZER_SCHEMA or \
            payload.get("freeze") != readiness.R71_FREEZE:
        raise RuntimeError(f"{path.name}: not the registered ROUND71 calibrated INIT payload")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise RuntimeError(f"{path.name}: the INIT payload carries no training state")
    if int(training_state.get("episodes_done", -1)) != 0 or int(training_state.get("updates", -1)) != 0:
        raise RuntimeError(f"{path.name}: the INIT payload is not an untrained initialisation")
    if payload.get("training_version") == R73_TRAINING_VERSION:
        raise RuntimeError(f"{path.name}: an ROUND73 training checkpoint is never an INIT source")
    # registered identity: file sha + parameter sha + per-module digests
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != registered["file_sha256"]:
        raise RuntimeError(f"{path.name}: file sha256 != the registered ROUND73 init identity")
    controller = _restore_controller(payload, training_state)
    summary = controller.parameter_summary()
    if summary["parameter_sha256"] != registered["parameter_sha256"]:
        raise RuntimeError(f"{path.name}: parameters != the registered ROUND73 init identity")
    if summary["parameter_object_count"] != 12:
        raise RuntimeError(f"{path.name}: the INIT controller is not the five-module schema")
    cal = registered["calibration"]
    module_sha = _registered_module_sha(controller)
    for key, want in (("trunk", cal["trunk_sha256"]), ("early_head", cal["early_head_sha256"]),
                      ("late_head", cal["late_head_sha256"]),
                      ("value_task_head", cal["value_task_head_sha256"]),
                      ("value_cost_head", cal["value_cost_head_sha256"])):
        if module_sha[key] != want:
            raise RuntimeError(f"{path.name}: INIT module {key} != the registered identity")
    identity = {"load_mode": "model_only_init", "optimizer_restored": False, "rng_restored": False,
                "lambda_restored": False, "training_version": payload.get("training_version"),
                "structure": payload.get("structure"),
                "parameter_identity": summary,
                "module_sha256": module_sha,
                "registered_init_identity": registered}
    return controller, training_state, identity


def _registered_module_sha(controller) -> dict:
    """Per-module digests under the SAME helper that produced the registered identity.

    The registered per-module digests are copied verbatim from the ROUND71 checkpoint
    index, which was written by ``readiness.module_parameter_digest``.  ``r68._module_sha``
    serialises the same parameters in a different order and byte encoding, so it cannot
    verify them: measured on the seed-11 INIT it matches NONE of the five registered
    digests while ``module_parameter_digest`` matches all five.  The model, the ROUND71
    index and the registered identity are unchanged; only the verification helper is.
    """
    digest = readiness.module_parameter_digest(controller)
    return {name: digest.get(name) for name in MODULES}
