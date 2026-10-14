"""ROUND74 constrained early-head adaptation: the R74 real wiring (the ONE scientific change).

Relative to the verified ROUND71 critic-calibration controller the ONLY structural change is that
``early_head`` becomes trainable. The shared trunk stays frozen; ``late_head`` and the two critics
(task ``inner.value_head`` / cost ``cost_head``) keep their ROUND71 training role. Everything else
is inherited bit-for-bit from the R66/R68/R71 living configuration through the real modules:

  * controller construction and RNG order  -> r66.fresh_partitioned_controller (StageCController)
  * routed early/late forward, likelihood  -> r66.route_logits / r66.likelihood_alignment / onpolicy
  * the PPO Lagrangian (clipped surrogate, dual-critic value, entropy, beta=1.00 gate prior,
    lambda*A_task - A_cost advantage, ALL_ROWS pooling) is a faithful transcription of
    r66.frozen_early_ppo_update (itself a faithful transcription of v2_trainer.ppo_update); the
    surrogate is NEVER rewritten here.
  * INIT is the registered ROUND71 calibration INIT, loaded MODEL-ONLY (five modules); a fresh
    Adam / lambda=1 / RNG / cursor follows. R60/R66/R68/R71/R73 FULL resume is refused.

Trainable partition (R74): early_head + late_head + value_task_head + value_cost_head = 8 objects.
Frozen partition (R74): trunk = 4 objects. Total 12 objects.

PHASE B1 loads NO task model for environment science; it only qualifies the wiring on controlled
legal inputs and a handful of charged real optimizer steps through the R74 production resource
entry (r74_guard_prod.R74ResourcesProd.production()).
"""
from __future__ import annotations
import copy, hashlib, json, os, sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                       # .../zhenghe-round42
RUN_ROOT = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
R66_DIR = RUN_ROOT / "round66_bc_early_path_preserving_late_adaptation"
R68_DIR = RUN_ROOT / "round68_late_initial_confidence_ablation"
R71_DIR = RUN_ROOT / "round71_critic_initialization_calibration"
READINESS_DIR = REPO / "analysis" / "round71-critic-calibration-readiness"
R71_INIT_DIR = R71_DIR / "raw" / "checkpoints"
for _d in (READINESS_DIR, str(R68_DIR), str(R66_DIR)):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import torch                                     # noqa: E402
import numpy as np                               # noqa: E402

# ------------------------------------------------------------------ R74 identity contract
R74_TRAINER_TAG = "constrained-cost-v2-r74-early-head-trainable"
R74_TRAINING_VERSION = "round74-constrained-early-head-adaptation/1"
R74_STRUCTURE = "partitioned-early-late-actor-head/1+early-head-trainable-late-adaptation/1"
R74_LOSS_VERSION = "constrained-lagrangian/2+partitioned-actor-head/1+early-head-trainable/1"
R74_STATE_SCHEMA = ("constrained-cost-state/2+partitioned-actor-head/1+early-head-trainable/1")
R74_OPTIMIZER_SCHEMA = "stage-c-early-head-trainable-eight-module-parameter-set/1"
R74_CALIBRATION_ROUTE = ("critic-representation-stop-gradient/1+r71-dual-output-ridge-init/1"
                         "+r74-early-head-trainable/1")

FROZEN_MODULES = ("trunk",)                       # THE single change vs R66/R71 (early moves out)
TRAINABLE_MODULES = ("early_head", "late_head", "value_task_head", "value_cost_head")
MODULES = ("trunk", "early_head", "late_head", "value_task_head", "value_cost_head")
SEEDS = (11, 22, 33)

# R60/R66/R68/R71/R73 schemas+tags that are NEVER an R74 training-resume source.
_FOREIGN_SCHEMAS = (
    "constrained-cost-state/2",                                   # R60/R66 four-module base
    "constrained-cost-state/2+partitioned-actor-head/1",          # R60/R66 partitioned
    "constrained-cost-state/2+partitioned-actor-head/1+frozen-early-path/1",   # R66
)
_FOREIGN_TAGS = (
    "constrained-cost-v2",
    "constrained-cost-v2-partitioned-early-late-actor-head",
    "constrained-cost-v2-frozen-early-path",
    "constrained-cost-v2-frozen-early-path-late-init-alpha",
)


def _module_of(controller, name: str):
    return {"trunk": controller.inner.trunk,
            "early_head": controller.inner.early_head,
            "late_head": controller.inner.late_head,
            "value_task_head": controller.inner.value_head,
            "value_cost_head": controller.cost_head}[name]


# ------------------------------------------------------------------ configure / live config
def configure() -> dict:
    """Stand up the real runtime exactly as R66/R71 do and capture the LIVE frozen values."""
    import r66_frozen as r66
    cfg = r66.configure_source_imports()      # canonical modules on path + byte identity + living overrides
    import v2_common as common
    live = {
        "GAMMA": float(common.GAMMA), "GAE_LAMBDA": float(common.GAE_LAMBDA),
        "ETA_LAMBDA": float(common.ETA_LAMBDA), "B0_BYTES": float(common.B0_BYTES),
        "DATA_ORDER_SEED": int(common.DATA_ORDER_SEED), "EPOCHS_PER_BATCH": int(common.EPOCHS_PER_BATCH),
        "MINIBATCH_STEPS": int(common.MINIBATCH_STEPS), "CLIP": float(common.CLIP),
        "VALUE_COEFFICIENT": float(common.VALUE_COEFFICIENT), "ENTROPY_BONUS": float(common.ENTROPY_BONUS),
        "LEARNING_RATE": float(common.LEARNING_RATE), "MAX_GRAD_NORM": float(common.MAX_GRAD_NORM),
        "READY_CUT": float(r66.READY_CUT), "HORIZON": float(r66.HORIZON), "GATE_BETA": float(r66.GATE_BETA),
    }
    cfg["live"] = live
    return cfg


# ------------------------------------------------------------------ partition + mask
def partition(controller) -> tuple[list, list]:
    frozen: list = []
    for name in FROZEN_MODULES:
        frozen.extend(_module_of(controller, name).parameters())
    trainable: list = []
    for name in TRAINABLE_MODULES:
        trainable.extend(_module_of(controller, name).parameters())
    fids = [id(p) for p in frozen]
    tids = [id(p) for p in trainable]
    if len(fids) != len(set(fids)) or len(tids) != len(set(tids)):
        raise RuntimeError("R74 partition contains a duplicate parameter object")
    if set(fids) & set(tids):
        raise RuntimeError("R74 frozen and trainable parameter sets overlap")
    allids = [id(p) for p in controller.parameters]
    if len(frozen) + len(trainable) != len(allids) or len(allids) != len(set(allids)):
        raise RuntimeError("R74 frozen + trainable does not partition the parameter objects")
    return frozen, trainable


def set_mask(controller) -> dict:
    """Apply the R74 mask: trunk frozen; early/late/both critics trainable. Returns the facts."""
    for p in _module_of(controller, "trunk").parameters():
        p.requires_grad_(False)
    for name in TRAINABLE_MODULES:
        for p in _module_of(controller, name).parameters():
            p.requires_grad_(True)
    return assert_mask(controller)


def assert_mask(controller) -> dict:
    frozen, trainable = partition(controller)
    if any(p.requires_grad for p in frozen):
        raise RuntimeError("R74 trunk is NOT frozen")
    if any(not p.requires_grad for p in trainable):
        raise RuntimeError("R74 early/late/critic is frozen but must stay trainable")
    return {"frozen_modules": list(FROZEN_MODULES), "trainable_modules": list(TRAINABLE_MODULES),
            "frozen_parameter_object_count": len(frozen),
            "trainable_parameter_object_count": len(trainable),
            "frozen_scalar_count": int(sum(p.numel() for p in frozen)),
            "trainable_scalar_count": int(sum(p.numel() for p in trainable)),
            "total_parameter_object_count": len(frozen) + len(trainable),
            "total_scalar_count": int(sum(p.numel() for p in frozen + trainable))}


def trainable_parameters(controller) -> list:
    assert_mask(controller)
    _f, t = partition(controller)
    return t


def frozen_parameters(controller) -> list:
    assert_mask(controller)
    f, _t = partition(controller)
    return f


def build_optimizer(controller, lr: float):
    return torch.optim.Adam(trainable_parameters(controller), lr=float(lr))


def module_parameter_digest(controller) -> dict:
    out = {}
    for name in MODULES:
        h = hashlib.sha256()
        for p in _module_of(controller, name).parameters():
            h.update(p.detach().contiguous().cpu().numpy().tobytes())
        out[name] = h.hexdigest()
    return out


# ------------------------------------------------------------------ INIT (R71 model-only)
def r71_init_path(seed: int) -> Path:
    return R71_INIT_DIR / f"v2_train_s{int(seed)}_init.pt"


def load_init(seed: int, *, expected_param_sha256: str | None = None) -> dict:
    """Build the R74 INIT: R71 calibration INIT loaded MODEL-ONLY, then the R74 mask + fresh state.

    The controller is constructed through the identical R66 chain so RNG is consumed in the frozen
    pilot order; the five modules are then overwritten bit-exactly from the R71 INIT controller
    state. R71's optimizer / lambda / RNG / cursor are NOT inherited (fresh). A R60/R66/R68/R71/R73
    FULL-resume payload is refused as a training source; only its five-module controller state is a
    valid model-only INIT.
    """
    import r66_frozen as r66
    import v2_trainer as trainer
    path = r71_init_path(seed)
    if not path.is_file():
        raise FileNotFoundError(f"R71 INIT not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    controller_state = _validate_r71_model_only_source(path, payload)
    # identical construction (same RNG consumption order as r68_initial_controller)
    controller = r66.fresh_partitioned_controller(int(seed), ep0_state=None, verify_ep0=False)
    rng_after_construction = trainer.rng_state()
    controller.load_state_dict(controller_state)          # bit-exact five modules (no RNG)
    mask = set_mask(controller)                           # THE R74 change: early_head trainable
    summary = controller.parameter_summary()
    if expected_param_sha256 and summary["parameter_sha256"] != expected_param_sha256:
        raise RuntimeError("R74 INIT parameter identity != the registered R71 calibration INIT")
    identity = {
        "seed": int(seed), "source": "r71_calibration_init_model_only",
        "r71_init_path": str(path),
        "r71_init_file_sha256": _file_sha(path),
        "init_parameter_sha256": summary["parameter_sha256"],
        "parameter_object_count": summary["parameter_object_count"],
        "parameter_count": summary["parameter_count"],
        "structure": R74_STRUCTURE, "training_version": R74_TRAINING_VERSION,
        "frozen_mask": mask, "optimizer_fresh": True, "lambda_init": 1.0,
        "from_trained_ep128": False, "rng_after_construction_sha256": _rng_digest(rng_after_construction),
        "r71_init_stored_rng_sha256": _rng_digest(payload.get("rng")),
    }
    return {"controller": controller, "rng_after_construction": rng_after_construction,
            "identity": identity, "r71_payload_versions": {
                "schema": payload.get("schema"), "training_version": payload.get("training_version"),
                "structure": payload.get("structure")}}


def fresh_training_state(seed: int, live: dict) -> dict:
    import v2_trainer as trainer
    state = trainer.fresh_training_state(int(seed))
    state["gae_lambda"] = float(live["GAE_LAMBDA"])
    state["eta_lambda"] = float(live["ETA_LAMBDA"])
    state["gamma"] = float(live["GAMMA"])
    state["lambda_init"] = 1.0
    state["lambda"] = 1.0
    state["training_version"] = R74_TRAINING_VERSION
    state["loss_version"] = R74_LOSS_VERSION
    state["state_schema"] = R74_STATE_SCHEMA
    state["optimizer_schema"] = R74_OPTIMIZER_SCHEMA
    state["structure"] = R74_STRUCTURE
    state["calibration_route"] = R74_CALIBRATION_ROUTE
    state["training_seed"] = int(seed)
    return state


# ------------------------------------------------------------------ routed forward + loss
def ready_step_of(feature_row) -> int:
    f = np.asarray(feature_row, dtype=np.float64)
    return int(round(float(f[18]) * 1000.0) - round(float(f[0])))


def forward_logits_values(controller, features: torch.Tensor):
    """Real routed forward: hidden=trunk(features); logits=route_logits; task/cost values."""
    import r66_frozen as r66
    hidden = controller.inner.trunk(features)
    logits = r66.route_logits(controller, features, hidden)
    v_task = controller.inner.value_head(hidden).squeeze(-1)
    v_cost = controller.cost_head(hidden).squeeze(-1)
    return logits, v_task, v_cost, hidden


def minibatch_loss(controller, samples: list[dict], *, beta: float, common, want_grads: bool = False):
    """The REAL R74 PPO Lagrangian on ONE minibatch (faithful transcription of the R66 update loss).

    Returns (loss, info). info carries the per-module PRE-CLIP gradient L1 (early/late/task/cost)
    via autograd.grad on the eight trainable objects (allow_unused=True; an unused head reads 0.0,
    never a fabricated measurement). The surrogate / value / entropy / gate terms and the combined
    advantage lambda*A_task - A_cost are exactly the living ones; nothing is rewritten.
    """
    from onpolicy import action_entropy, joint_log_prob
    import r66_frozen as r66
    if not samples:
        raise ValueError("empty minibatch")
    stacked = torch.as_tensor(np.concatenate([np.asarray(s["features"], dtype=np.float32)
                                              for s in samples], axis=0))
    logits, v_task, v_cost, _hidden = forward_logits_values(controller, stacked)
    row = 0
    new_lp, ent, pooled_task, pooled_cost, gate_skip = [], [], [], [], []
    early_free = late_free = 0
    for s in samples:
        count = len(s["targets"])
        rows = torch.arange(row, row + count)
        row += count
        picked = torch.tensor([s["actions"][t] for t in s["targets"]], dtype=torch.long)
        usable = torch.tensor([1.0 if t in s["free"] else 0.0 for t in s["targets"]],
                              dtype=torch.float32)
        new_lp.append(joint_log_prob(logits[rows], picked, usable))
        ent.append(action_entropy(logits[rows], usable))
        pooled_task.append(v_task[rows].mean())
        pooled_cost.append(v_cost[rows].mean())
        for gi, gt in enumerate(s["targets"]):
            if gt not in s["free"]:
                continue
            feat = np.asarray(s["features"], dtype=np.float64)[gi]
            ready = int(round(float(feat[18]) * 1000.0) - round(float(feat[0])))
            if ready <= r66.READY_CUT:
                early_free += 1
                if float(feat[5]) == 1.0:
                    gate_skip.append(torch.log_softmax(logits[rows][gi], dim=-1)[r66.SKIP_INDEX])
            else:
                late_free += 1
    new_lp_t = torch.stack(new_lp)
    ent_t = torch.stack(ent)
    old_lp = torch.tensor([s["log_prob"] for s in samples], dtype=torch.float32)
    adv = torch.tensor([s["advantage"] for s in samples], dtype=torch.float32)
    t_task = torch.tensor([s["return_task"] for s in samples], dtype=torch.float32)
    t_cost = torch.tensor([s["return_cost"] for s in samples], dtype=torch.float32)
    ratio = torch.exp(new_lp_t - old_lp)
    surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1.0 - common["CLIP"], 1.0 + common["CLIP"]) * adv)
    value_task_loss = ((torch.stack(pooled_task) - t_task) ** 2).mean()
    value_cost_loss = ((torch.stack(pooled_cost) - t_cost) ** 2).mean()
    gate_loss = (-(torch.stack(gate_skip)).sum() / len(samples)) if gate_skip else torch.zeros(())
    loss = (-surrogate.mean() + common["VALUE_COEFFICIENT"] * (value_task_loss + value_cost_loss)
            - common["ENTROPY_BONUS"] * ent_t.mean() + beta * gate_loss)
    # pre-clip per-module gradient L1 on the eight trainable objects (no step)
    groups = {name: list(_module_of(controller, name).parameters()) for name in TRAINABLE_MODULES}
    trainable = trainable_parameters(controller)
    info = {"early_free": early_free, "late_free": late_free,
            "loss_requires_grad": bool(loss.requires_grad),
            "gate_terms": len(gate_skip), "samples": len(samples)}
    if want_grads and loss.requires_grad:
        g = torch.autograd.grad(loss, trainable, allow_unused=True)
        gmap = {id(p): (0.0 if gi is None else float(gi.detach().abs().sum()))
                for p, gi in zip(trainable, g)}
        for name, params in groups.items():
            info[f"grad_l1_{name}"] = float(sum(gmap[id(p)] for p in params))
        total = torch.sqrt(sum((gi ** 2).sum() for gi in g if gi is not None))
        info["preclip_total_norm"] = float(total)
    else:
        for name in groups:
            info[f"grad_l1_{name}"] = None
        info["preclip_total_norm"] = None
    return loss, info


# ------------------------------------------------------------------ full-state checkpoint (R74)
def save_checkpoint(controller, optimizer, state: dict, path, *, live: dict) -> str:
    frozen, trainable = partition(controller)
    fdig = hashlib.sha256()
    for i, p in enumerate(frozen):
        fdig.update(f"{i}".encode()); fdig.update(p.detach().contiguous().cpu().numpy().tobytes())
    import v2_trainer as trainer
    payload = {
        "schema": R74_STATE_SCHEMA, "trainer_tag": R74_TRAINER_TAG,
        "training_version": R74_TRAINING_VERSION,
        "training_version_source": trainer.TRAINING_VERSION,
        "reward_timing_version": trainer.REWARD_TIMING_VERSION,
        "loss_version": R74_LOSS_VERSION, "gae_schema": trainer.GAE_SCHEMA,
        "optimizer_schema": R74_OPTIMIZER_SCHEMA, "reward_timing_schema": trainer.REWARD_TIMING_SCHEMA,
        "structure": R74_STRUCTURE, "calibration_route": R74_CALIBRATION_ROUTE,
        "frozen_mask": {"frozen_modules": list(FROZEN_MODULES),
                        "trainable_modules": list(TRAINABLE_MODULES),
                        "frozen_sha256": fdig.hexdigest(),
                        "frozen_scalar_count": int(sum(p.numel() for p in frozen)),
                        "frozen_parameter_object_count": len(frozen),
                        "trainable_parameter_object_count": len(trainable)},
        "controller": {k: v for k, v in controller.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "optimizer_param_groups": [len(g["params"]) for g in optimizer.state_dict()["param_groups"]],
        "training_state": state, "rng": trainer.rng_state(),
    }
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, tmp); os.replace(tmp, path)
    return str(path)


def _reject_foreign_resume(path, payload: dict) -> None:
    name = Path(path).name
    if payload.get("schema") in _FOREIGN_SCHEMAS or payload.get("trainer_tag") in _FOREIGN_TAGS:
        raise RuntimeError(f"{name}: R60/R66/R68 partitioned checkpoint; REJECTED as an R74 full-resume source")
    if payload.get("trainer_tag") in ("constrained-cost-v2-critic-calibration",) or \
            (payload.get("schema") == "constrained-cost-state/2+partitioned-actor-head/1+frozen-early-path/1"):
        raise RuntimeError(f"{name}: R71/R73 checkpoint; REJECTED as an R74 full-resume source")
    # An R74 payload is the ONLY acceptable full-restore; anything else with an optimizer is refused.
    if "optimizer" in payload and (payload.get("schema") != R74_STATE_SCHEMA
                                   or payload.get("trainer_tag") != R74_TRAINER_TAG):
        raise RuntimeError(f"{name}: not an R74 checkpoint; refusing full resume")


def load_checkpoint(path, *, live: dict):
    """Resume ONLY an R74 checkpoint of this contract (full state: 5 modules, Adam, lambda, RNG, cursor)."""
    import v2_trainer as trainer
    import r66_frozen as r66
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != R74_STATE_SCHEMA or payload.get("trainer_tag") != R74_TRAINER_TAG:
        raise RuntimeError(f"{Path(path).name}: not an R74 checkpoint; refusing full resume")
    ts = payload.get("training_state")
    if not isinstance(ts, dict) or ts.get("state_schema") != R74_STATE_SCHEMA:
        raise RuntimeError(f"{Path(path).name}: training state schema mismatch")
    controller = r66.fresh_partitioned_controller(int(ts["training_seed"]), ep0_state=None, verify_ep0=False)
    controller.load_state_dict(payload["controller"])
    set_mask(controller)
    optimizer = build_optimizer(controller, live["LEARNING_RATE"])
    if [len(g["params"]) for g in payload["optimizer"]["param_groups"]] != [len(TRAINABLE_MODULES) * 2]:
        raise RuntimeError("R74 checkpoint optimizer is not the eight-parameter trainable set")
    optimizer.load_state_dict(payload["optimizer"])
    trainer.restore_rng(payload.get("rng"))
    return controller, optimizer, ts


# ------------------------------------------------------------------ helpers
def _validate_r71_model_only_source(path, payload: dict) -> dict:
    """Accept the ROUND71 calibration INIT as a MODEL-ONLY source and return its five modules.

    This is NOT a full resume: the R71 optimizer / lambda / RNG / cursor are never read. The source
    is validated by its R71 contract (schema, structure, calibration route) and the five-module
    controller schema; the exact parameter identity is checked by the caller against the registered
    R71 INIT sha. A R60/R66/R68 payload (wrong schema) is refused here as a model-only source too.
    """
    import r71_runtime_binding as readiness
    if payload.get("schema") != readiness.R71_STATE_SCHEMA:
        raise RuntimeError(f"{Path(path).name}: not a ROUND71 critic-calibration INIT; schema "
                           f"{payload.get('schema')!r}; refusing as an R74 model-only source")
    if payload.get("structure") != readiness.R71_STRUCTURE:
        raise RuntimeError(f"{Path(path).name}: structure {payload.get('structure')!r} is not the "
                           f"R71 partitioned frozen-early-path INIT")
    if "r71-dual-output-ridge-init" not in str(payload.get("calibration_route", "")):
        raise RuntimeError(f"{Path(path).name}: not the ridge-calibrated R71 INIT")
    controller_state = payload.get("controller")
    if not isinstance(controller_state, dict) or set(controller_state) != set(MODULES):
        raise RuntimeError(f"{Path(path).name}: controller is not the partitioned five-module schema")
    return controller_state


def _file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rng_digest(rng_state) -> str:
    return hashlib.sha256(json.dumps(rng_state, sort_keys=True, default=str).encode("utf-8")).hexdigest()