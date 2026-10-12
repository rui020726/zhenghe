"""ROUND66 BC early-path-preserving late adaptation (one frozen early computation path).

This module is the ONLY structural intervention of ROUND66. Everything it does not touch is
inherited field-by-field from the accepted ROUND60 living configuration, which itself inherited
the accepted ROUND58 living configuration verified from the R58 ep128 checkpoint
``training_state`` (see FREEZE notes below), not from module import defaults.

The ROUND60 architecture is kept verbatim: 19-dim observation, shared Tanh trunk, TWO
independent early/late SEND/SKIP heads routed by ``ready_step = round(f[18]*1000) -
round(f[0])`` (<=147 early, >147 late), and a task-value head plus a cost-value head on the same
trunk. Routing is NOT an action rule: early candidates may still SEND, late candidates may
still SKIP, no hard send/skip is added, no extra emission is introduced, and the sequential
budget mask is never bypassed. Sampling, greedy decoding, PPO likelihood recomputation and
read-only evaluation all call the SAME routed forward so they cannot drift apart.

ROUND66 adds exactly one thing on top of that ROUND60 structure: the WHOLE early policy
computation path is frozen. ``trunk`` and ``early_head`` get ``requires_grad=False``; only
``late_head``, ``value_task_head`` and ``value_cost_head`` stay trainable. Because the trunk is
shared, freezing it also freezes the representation both critics read. That is the point of the
restricted policy family being tested, and it is NOT a claim that the R53 critic->trunk gradient
isolation was repaired by it. The R53 ``critic_input_detached`` context is still applied around
every update exactly as R58/R60 applied it; the frozen modules simply receive no gradient at all
because they do not require one.

Initialisation is model-only from each seed's compliant ROUND60 ep0 checkpoint
(``v2_train_s{seed}_before_b0.pt``): all five modules are bit-equal to that ROUND60 ep0. ROUND60
ep128, and any ROUND60 Adam state, lambda or training RNG, are never inherited.

OLD (accepted ROUND58): one shared SEND/SKIP actor output head ``inner.action_head`` feeding a
shared trunk, with a task-value head and a cost-value head on the same trunk, and a
beta=1.00 / ready_cut=147 gate prior that softly pushes early first-emission candidates toward
SKIP.

NEW (ROUND60): the shared trunk, the task-value head and the cost-value head are unchanged (the
same objects, the same initial values). The single ``action_head`` is replaced by TWO independent
heads, ``early_head`` and ``late_head``, each an exact copy of the original ``action_head`` and
each its own parameter object (never shared). Routing uses only the legal 19-dim observation and
the frozen constant:

    ready_step = round(f[18] * 1000) - round(f[0])
    ready_step <= 147  -> early_head
    ready_step >  147  -> late_head

Routing is NOT an action rule. Early candidates may still SEND, late candidates may still SKIP,
no hard send/skip is added, no extra emission is introduced, and the sequential budget mask is
never bypassed. Sampling, greedy decoding, PPO likelihood recomputation and read-only evaluation
all call the SAME ``PartitionedController.forward`` so they cannot drift apart. No task outcome,
receiver knowledge, scenario id or target id ever enters routing.

The frozen canonical modules on disk (``v2_trainer.py``, ``v2_controller.py``, ``v2_common.py``,
``sb_controller.py``, ``train_pilot.py``, ``budget_masked.py``, ``r42_one_shot.py``) are imported
unchanged and are never edited. This file adds a parallel, versioned controller + trainer +
checkpoint contract for the partitioned structure only.

FREEZE (verified against R58 ep128 ``v2_train_s33_after_b31.pt`` training_state):
    gamma=1.0, gae_lambda=1.0, eta_lambda=0.5, delta=0.0, lambda_init=1.0,
    data_order_seed=20260927, b0_bytes=10000.0, lr=3e-4, clip=0.2, entropy_bonus=0.01,
    value_coefficient=0.5, max_grad_norm=0.5, epochs_per_batch=4, minibatch_steps=256,
    episodes_per_update=4, log_prob_tolerance=1e-6, gate_beta=1.00, ready_cut=147.
The module import defaults gae_lambda=0.95 / eta_lambda=0.05 are the OLD values and are NOT used;
this module asserts the live overrides before it will train.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ------------------------------------------------------------------ frozen identity ------
# Canonical source tree (sibling repo) that owns the frozen trainer/controller/pilot modules.
_HERE = Path(__file__).resolve().parent
_RUN_ROOT = _HERE.parent
_REPO = _RUN_ROOT.parents[1]
_WORKSPACE = _REPO.parent
CANON = _WORKSPACE / "zhenghe" / "runs" / "comm_cost_v2_rerun_20260928"
STAGE_B = _WORKSPACE / "zhenghe" / "runs" / "comm_cost_stage_b_20260927"
PILOT = _WORKSPACE / "zhenghe" / "runs" / "comm_budget_pilot_20260919"
R43_DIR = _REPO / "runs" / "comm_one_shot_full_trace_ablation_20261004"
R40_DIR = _WORKSPACE / "zhenghe-round40" / "runs" / "comm_first_report_training_ablation_20261003"

# The exact frozen trainer byte identity ROUND58 accepted (LIVE_MODULE_IDENTITY.json).
EXPECTED_TRAINER_SHA256 = "0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841"
EXPECTED_COMMON_SHA256 = "66d019a344d9cfef3c67bd8b722f8cc27144b9f71ddc8e5496451929d12f746a"
EXPECTED_CONTROLLER_SHA256 = "9b2362ee5dc436c04e5fd2d841646d637b5b3f371607bd13a759a60c4cb4cab1"

# ROUND60 contract identities, kept ONLY so this module can recognise an ROUND60 checkpoint for
# what it is: a read-only model-only source (initialisation or comparison arm). An ROUND60
# checkpoint is NEVER a valid ROUND66 training-resume source.
R60_TRAINING_VERSION = (
    "constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/dual-eta-lambda-0p5/1/"
    "rule-bc-late147-init/1/published-completion-event-at-relay-tick/2"
    "+r53-critic-representation-stop-gradient/1+r56-soft-early-report-prior-beta0p10/1"
    "+r58-prior-strength-beta1p00/1+r60-partitioned-early-late-actor-head/1"
)
R60_LOSS_VERSION = "constrained-lagrangian/2+partitioned-actor-head/1"
R60_STATE_SCHEMA = "constrained-cost-state/2+partitioned-actor-head/1"
R60_OPTIMIZER_SCHEMA = "stage-c-partitioned-five-module-parameter-set/1"
R60_TRAINER_TAG = "constrained-cost-v2-partitioned"

# NEW ROUND66 training / model structure version and checkpoint contract. The ROUND60
# partitioned five-module architecture is inherited unchanged; the only addition is the frozen
# early computation path, which is recorded in the optimizer schema because it changes which
# parameter objects the optimizer owns.
R66_TRAINING_VERSION = R60_TRAINING_VERSION + "+r66-frozen-early-path-late-adaptation/1"
R66_LOSS_VERSION = "constrained-lagrangian/2+partitioned-actor-head/1+frozen-early-path/1"
R66_STATE_SCHEMA = "constrained-cost-state/2+partitioned-actor-head/1+frozen-early-path/1"
R66_OPTIMIZER_SCHEMA = "stage-c-frozen-early-path-six-module-parameter-set/1"
R66_TRAINER_TAG = "constrained-cost-v2-frozen-early-path"
R66_STRUCTURE = "partitioned-early-late-actor-head/1+frozen-early-path-late-adaptation/1"

# The ROUND60 output root that owns the compliant ROUND60 ep0 / ep128 checkpoints ROUND66 reads
# model-only. Paths are resolved from this table, never globbed.
R60_DIR = _RUN_ROOT / "round60_partitioned_actor_head_ablation"
R60_MODEL_ROLES = {
    "ep0": {"filename": "v2_train_s{seed}_before_b0.pt", "role": "model_only_initialisation"},
    "ep128": {"filename": "v2_train_s{seed}_ep128.pt", "role": "model_only_comparison"},
}

# Expected identity of the COMPLIANT ROUND60 sources, taken from the ROUND60 ep0 identity files
# (``r60_ep0_identity_s{seed}.json``) and the ROUND60 reconciliation verdict. A schema check alone
# is not enough: the loader must fail closed if the file bytes or the parameter identity drift,
# and it must never accept a file that lives under ROUND60's quarantined non-compliant namespace.
R60_EXPECTED_IDENTITY = {
    11: {
        "ep0": {"file_sha256": "f267558d0cc8933987c09576b8d6d6ad512dbc3c73653d059104705c1faca437",
                "parameter_sha256": "fb82ad80724afa37ff8f3cbd5bd8bae958e221586eb799bdf61701af80878ec4",
                "episodes_done": 0, "updates": 0},
        "ep128": {"file_sha256": "1c765c27294c673aa82b1ab7cd8dd8f379699ede8f924cc3438bfcd73fa0c60e",
                  "parameter_sha256": "809f35f77ae2659526b70a49b543996c68f0f5942c633d8f69e3f00f3e4d06db",
                  "episodes_done": 128, "updates": 32},
    },
    22: {
        "ep0": {"file_sha256": "249f769770fd0bffcc8df345539088890f9323f1338a224e3c8195d35adc8555",
                "parameter_sha256": "a9e7776171e5581dd0c3c972dbbc9504337b2513d18b691a137d528d6221b46b",
                "episodes_done": 0, "updates": 0},
        "ep128": {"file_sha256": "79bd3769d6382e5e908b22db2794ebfbb22cab430162fb211d87bb196550e234",
                  "parameter_sha256": "286e2f84ab7d0abc65a3201495ff41f3ad8423775b456c5e379765fd43d75365",
                  "episodes_done": 128, "updates": 32},
    },
    33: {
        "ep0": {"file_sha256": "5f47f28b2678a0091e2370345b622aad7bdf1d8064b19d3ed57282bc8f005e93",
                "parameter_sha256": "10bce624e9fa8079144b1150b6bbc6ece2be17771233a6e7bf3720770bea5b26",
                "episodes_done": 0, "updates": 0},
        "ep128": {"file_sha256": "573e4c26a4fee6f01405d4386ca73b518eeb1adb9cd8409bc83661e4113f6c5a",
                  "parameter_sha256": "bce71a0e9810122c146950f12cf818a9a359406312d96dfc9c7a457ebb654e2a",
                  "episodes_done": 128, "updates": 32},
    },
}
# Namespaces that hold ROUND60 material which is NOT the compliant training chain.
R60_REJECTED_NAMESPACES = ("_quarantine", "rejected", "failures")

# R58 ep0 (BC_POST model-only) init sources and their verified identity (ROUND46_R2_V6_INPUT_IDENTITY).
R58_EP0_MODELS = {
    11: {"path": _RUN_ROOT / ".." / "comm_rule_warmstart_cost_learning_20261006_r2" / "checkpoints" / "BC_POST_MODEL_s11.pt",
         "checkpoint_sha256": "76ffc86c4aca26fb3b27380d5195a0cbd07822f7b85d1f1070a5550776dd458a",
         "parameter_sha256": "37303c494cd997894fba1441cf9d2cde86cf2135e71f443a4ed1bcb9ec7f0330",
         "action_head_sha256": "aad35fc6ba3fc43bfe174cb08a5fabb8180654765ba9ab39b6b71fdc42b8b102"},
    22: {"path": _RUN_ROOT / ".." / "comm_rule_warmstart_cost_learning_20261006_r2" / "checkpoints" / "BC_POST_MODEL_s22.pt",
         "checkpoint_sha256": "3cdb83785c51f9e40005d66a1d024d06a5ba6940616e22f3b01a9bf65b5e609d",
         "parameter_sha256": "e8531e0aff95ed207bc70026247d45869656d266aa974838a7ec2cfe352479b0",
         "action_head_sha256": "0d0d3581d130341225634b5543f0b926cd21112e0631de19089f6f156da05bd2"},
    33: {"path": _RUN_ROOT / ".." / "comm_rule_warmstart_cost_learning_20261006_r2" / "checkpoints" / "BC_POST_MODEL_s33.pt",
         "checkpoint_sha256": "9329ef048dfc5c1805af5756c7ea0962eedfc8d01aa26841e35af7001616803a",
         "parameter_sha256": "10bae22f3896adc507f05c5d868aad43b0441d19f754f2ef4dc373fc964e57a5",
         "action_head_sha256": "5fff65a080c1e6587ee7206b097f706c1e11093b9dc914f2076506bf66b01af3"},
}
SEEDS = (11, 22, 33)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_source_imports() -> dict[str, Any]:
    """Put the frozen canonical modules on sys.path exactly as R45/R58 configured them.

    Returns the imported modules. Verifies the frozen trainer/common/controller byte identity so
    a drifted canonical tree fails closed before any science runs.
    """
    import sys
    for path in (CANON, STAGE_B, PILOT, R43_DIR, R40_DIR):
        text = str(path.resolve())
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    import v2_trainer as trainer
    import v2_common as common
    import v2_controller as controller_module
    import cost_math  # noqa: F401
    if sha256_file(CANON / "v2_trainer.py") != EXPECTED_TRAINER_SHA256:
        raise RuntimeError("canonical v2_trainer.py byte identity changed; refusing ROUND60")
    if sha256_file(CANON / "v2_common.py") != EXPECTED_COMMON_SHA256:
        raise RuntimeError("canonical v2_common.py byte identity changed; refusing ROUND60")
    if sha256_file(CANON / "v2_controller.py") != EXPECTED_CONTROLLER_SHA256:
        raise RuntimeError("canonical v2_controller.py byte identity changed; refusing ROUND60")
    # ROUND60 uses the R58 living overrides, not the module defaults (0.95 / 0.05).
    common.GAE_LAMBDA = 1.0
    common.ETA_LAMBDA = 0.5
    if float(common.GAMMA) != 1.0:
        raise RuntimeError("frozen gamma is not 1.0")
    return {"trainer": trainer, "common": common, "controller_module": controller_module}


# ------------------------------------------------------------------ routing ---------------
READY_CUT = 147.0
HORIZON = 1000.0
SKIP_INDEX = 0
SEND_INDEX = 1


def ready_steps(features: torch.Tensor) -> torch.Tensor:
    """Per-row ready_step = round(f[18]*1000) - round(f[0]) from the legal 19-dim input only."""
    f = features.to(torch.float64)
    return torch.round(f[:, 18] * HORIZON) - torch.round(f[:, 0])


def route_logits(controller: "PartitionedController", features: torch.Tensor,
                 hidden: torch.Tensor) -> torch.Tensor:
    """Gather each row's SEND/SKIP logits from its routed head. early <=147, late >147."""
    early = controller.inner.early_head(hidden)
    late = controller.inner.late_head(hidden)
    is_early = (ready_steps(features) <= READY_CUT).unsqueeze(1)
    return torch.where(is_early, early, late)


# ------------------------------------------------------------------ controller ------------
class _PartitionedInner:
    """Holds the shared trunk, the two independent actor heads and the task-value head."""

    def __init__(self, base_inner: Any) -> None:
        # Reuse the frozen pilot Controller's submodules verbatim; only the action head splits.
        self.trunk = base_inner.trunk
        self.value_head = base_inner.value_head
        # Two INDEPENDENT head objects, each an exact copy of the original action head. deepcopy
        # draws no RNG, so constructing them cannot perturb the frozen training random start.
        self.early_head = copy.deepcopy(base_inner.action_head)
        self.late_head = copy.deepcopy(base_inner.action_head)


class PartitionedController:
    """Shared trunk + dual critic (unchanged) + two independent early/late SEND/SKIP heads."""

    def __init__(self, base_controller: Any) -> None:
        self.inner = _PartitionedInner(base_controller.inner)
        self.cost_head = base_controller.cost_head  # cost critic unchanged (same object)

    # sampler contract: (action logits, task values), exactly what the collectors expect.
    def forward(self, batch: torch.Tensor):
        hidden = self.inner.trunk(batch)
        logits = route_logits(self, batch, hidden)
        return logits, self.inner.value_head(hidden).squeeze(-1)

    def cost_values(self, batch: torch.Tensor) -> torch.Tensor:
        return self.cost_head(self.inner.trunk(batch)).squeeze(-1)

    @property
    def parameters(self) -> list:
        # Every unique parameter exactly once: trunk, early_head, late_head, task head, cost head.
        return [*self.inner.trunk.parameters(),
                *self.inner.early_head.parameters(),
                *self.inner.late_head.parameters(),
                *self.inner.value_head.parameters(),
                *self.cost_head.parameters()]

    def module_parameter_sets(self) -> dict[str, list]:
        return {
            "trunk": list(self.inner.trunk.parameters()),
            "early_head": list(self.inner.early_head.parameters()),
            "late_head": list(self.inner.late_head.parameters()),
            "task_value_head": list(self.inner.value_head.parameters()),
            "cost_head": list(self.cost_head.parameters()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "trunk": self.inner.trunk.state_dict(),
            "early_head": self.inner.early_head.state_dict(),
            "late_head": self.inner.late_head.state_dict(),
            "value_task_head": self.inner.value_head.state_dict(),
            "value_cost_head": self.cost_head.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        expected = {"trunk", "early_head", "late_head", "value_task_head", "value_cost_head"}
        if set(payload) != expected:
            raise ValueError(f"partitioned controller state keys {sorted(payload)} != {sorted(expected)}")
        self.inner.trunk.load_state_dict(payload["trunk"])
        self.inner.early_head.load_state_dict(payload["early_head"])
        self.inner.late_head.load_state_dict(payload["late_head"])
        self.inner.value_head.load_state_dict(payload["value_task_head"])
        self.cost_head.load_state_dict(payload["value_cost_head"])

    def parameter_summary(self) -> dict[str, Any]:
        tensors: dict[str, torch.Tensor] = {}
        for prefix, module in (("trunk", self.inner.trunk), ("early_head", self.inner.early_head),
                               ("late_head", self.inner.late_head),
                               ("value_task_head", self.inner.value_head),
                               ("value_cost_head", self.cost_head)):
            for name, tensor in module.named_parameters():
                tensors[f"{prefix}.{name}"] = tensor.detach().cpu()
        ids = [id(p) for p in self.parameters]
        if len(ids) != len(set(ids)):
            raise RuntimeError("partitioned controller shares a parameter object across modules")
        digest = hashlib.sha256()
        for name in sorted(tensors):
            digest.update(name.encode("utf-8"))
            digest.update(tensors[name].contiguous().numpy().tobytes())
        return {
            "parameter_count": int(sum(t.numel() for t in tensors.values())),
            "parameter_object_count": len(ids),
            "parameter_sha256": digest.hexdigest(),
            "shapes": {name: list(t.shape) for name, t in sorted(tensors.items())},
            "heads": ("early_head", "late_head", "value_task_head", "value_cost_head"),
            "structure": "partitioned-early-late-actor-head/1",
            "routing": "ready_step=round(f[18]*1000)-round(f[0]); early<=147, late>147",
        }


def fresh_partitioned_controller(seed: int, *, ep0_state: dict[str, Any] | None = None,
                                 verify_ep0: bool = True) -> PartitionedController:
    """Build the partitioned controller from the same-seed R58 ep0 (BC_POST) model params.

    ``fresh_controller(seed)`` is called first so RNG is consumed in the frozen pilot order; the
    two extra heads are deepcopies (no RNG draw). When ``ep0_state`` is given the four OLD modules
    are loaded from the R58 ep0 model and both new heads are set to the OLD action head, so trunk,
    dual critic and both heads equal the R58 ep0 values. ep128 and any OLD Adam/lambda are never
    loaded here.
    """
    import v2_controller
    # Use the StageCController class directly, not fresh_controller(), so a harness that patches
    # controller_module.fresh_controller to a partitioned factory cannot cause recursion here.
    base = v2_controller.StageCController(int(seed))
    controller = PartitionedController(base)
    if ep0_state is not None:
        expected = {"trunk", "action_head", "value_task_head", "value_cost_head"}
        if set(ep0_state) != expected:
            raise ValueError(f"R58 ep0 model keys {sorted(ep0_state)} != old four-module schema")
        controller.inner.trunk.load_state_dict(ep0_state["trunk"])
        controller.inner.value_head.load_state_dict(ep0_state["value_task_head"])
        controller.cost_head.load_state_dict(ep0_state["value_cost_head"])
        # Both independent heads start exactly equal to the OLD action head.
        controller.inner.early_head.load_state_dict(ep0_state["action_head"])
        controller.inner.late_head.load_state_dict(ep0_state["action_head"])
    if verify_ep0 and ep0_state is not None:
        role = R58_EP0_MODELS[int(seed)]
        path = Path(role["path"]).resolve()
        if sha256_file(path) != role["checkpoint_sha256"]:
            raise RuntimeError(f"R58 ep0 checkpoint byte identity changed for seed {seed}")
    return controller


def load_r58_ep0_model(seed: int) -> dict[str, Any]:
    """Read the four OLD modules from the verified R58 ep0 (BC_POST model-only) checkpoint."""
    role = R58_EP0_MODELS[int(seed)]
    path = Path(role["path"]).resolve()
    if sha256_file(path) != role["checkpoint_sha256"]:
        raise RuntimeError(f"R58 ep0 checkpoint byte identity changed for seed {seed}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("controller")
    if set(state) != {"trunk", "action_head", "value_task_head", "value_cost_head"}:
        raise ValueError("R58 ep0 controller modules differ from the old four-module schema")
    return state


# --------------------------------------------------------- ROUND60 model-only sources ------
def r60_model_path(seed: int, role: str) -> Path:
    """Resolve the exact ROUND60 checkpoint path for a seed and role. Never globbed."""
    spec = R60_MODEL_ROLES[role]
    path = (R60_DIR / "raw" / "checkpoints" / spec["filename"].format(seed=int(seed))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ROUND60 {role} checkpoint missing for seed {seed}: {path}")
    return path


def _require_compliant_r60_path(path: Path) -> None:
    """Refuse anything that is not inside ROUND60's compliant checkpoint namespace."""
    resolved = Path(path).resolve()
    root = (R60_DIR / "raw" / "checkpoints").resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"ROUND60 source {resolved} is outside the compliant checkpoint root")
    for bad in R60_REJECTED_NAMESPACES:
        if bad in resolved.parts:
            raise RuntimeError(
                f"ROUND60 source {resolved} lives under the non-compliant '{bad}' namespace; "
                "ROUND66 may only read the compliant training chain")


def r60_source_identity(seed: int, role: str) -> dict[str, Any]:
    """The expected identity of a compliant ROUND60 source, or a hard error."""
    expected = R60_EXPECTED_IDENTITY[int(seed)][role]
    return {"seed": int(seed), "role": role, "load_role": R60_MODEL_ROLES[role]["role"],
            "expected_file_sha256": expected["file_sha256"],
            "expected_parameter_sha256": expected["parameter_sha256"],
            "expected_episodes_done": expected["episodes_done"],
            "expected_updates": expected["updates"]}


def load_r60_model(seed: int, role: str) -> dict[str, Any]:
    """Read the five ROUND60 modules from a compliant ROUND60 checkpoint, model-only.

    This is the ONLY way ROUND66 touches ROUND60 weights. Three things must all hold or the load
    fails closed: the file must sit inside ROUND60's compliant checkpoint namespace (never the
    quarantined non-compliant one), its bytes must match the expected SHA256, and the parameter
    identity plus the training cursor must match the compliant ROUND60 chain. The returned payload
    is explicitly marked ``model_only``: it can initialise a ROUND66 controller or act as the
    comparison arm, and it can never be handed to a ROUND66 training resume.
    """
    path = r60_model_path(seed, role)
    _require_compliant_r60_path(path)
    expected = R60_EXPECTED_IDENTITY[int(seed)][role]
    file_sha = sha256_file(path)
    if file_sha != expected["file_sha256"]:
        raise RuntimeError(
            f"{path.name}: ROUND60 {role} file bytes drifted (got {file_sha}, "
            f"expected {expected['file_sha256']}); refusing ROUND66")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != R60_STATE_SCHEMA or payload.get("trainer_tag") != R60_TRAINER_TAG:
        raise RuntimeError(f"{path.name}: not a ROUND60 partitioned checkpoint")
    if payload.get("loss_version") != R60_LOSS_VERSION:
        raise RuntimeError(f"{path.name}: ROUND60 loss version mismatch")
    if payload.get("optimizer_schema") != R60_OPTIMIZER_SCHEMA:
        raise RuntimeError(f"{path.name}: ROUND60 optimizer schema mismatch")
    if payload.get("structure") != "partitioned-early-late-actor-head/1":
        raise RuntimeError(f"{path.name}: ROUND60 structure mismatch")
    controller = payload.get("controller")
    if set(controller) != {"trunk", "early_head", "late_head", "value_task_head", "value_cost_head"}:
        raise RuntimeError(f"{path.name}: ROUND60 controller modules are not the five-module schema")
    state = payload.get("training_state") or {}
    episodes_done = int(state.get("episodes_done", -1))
    updates = int(state.get("updates", -1))
    if episodes_done != expected["episodes_done"] or updates != expected["updates"]:
        raise RuntimeError(
            f"{path.name}: ROUND60 {role} training cursor is episodes_done={episodes_done} "
            f"updates={updates}, expected {expected['episodes_done']}/{expected['updates']}")
    parameter_sha = _state_dict_sha256(controller)
    if parameter_sha != expected["parameter_sha256"]:
        raise RuntimeError(
            f"{path.name}: ROUND60 {role} parameter identity drifted (got {parameter_sha}, "
            f"expected {expected['parameter_sha256']}); refusing ROUND66")
    return {"path": str(path), "file_sha256": file_sha, "role": role,
            "load_role": R60_MODEL_ROLES[role]["role"], "load_mode": "model_only",
            "optimizer_restored": False, "rng_restored": False,
            "training_version": payload.get("training_version"),
            "episodes_done": episodes_done, "updates": updates,
            "parameter_sha256": parameter_sha,
            "expected_identity": r60_source_identity(seed, role),
            "lambda_at_source": float(state.get("lambda", float("nan"))),
            "controller": controller}


#: The module-prefix naming PartitionedController.parameter_summary() uses, so a state dict can be
#: hashed without constructing a controller and still land on the same identity.
_PARAM_PREFIX = {"trunk": "trunk", "early_head": "early_head", "late_head": "late_head",
                 "value_task_head": "value_task_head", "value_cost_head": "value_cost_head"}


def _state_dict_sha256(state: dict[str, Any]) -> str:
    """Hash a five-module state dict exactly the way parameter_summary() hashes the parameters."""
    tensors: dict[str, Any] = {}
    for module, prefix in _PARAM_PREFIX.items():
        if module not in state:
            raise RuntimeError(f"state dict is missing module {module}")
        for key in sorted(state[module]):
            tensors[f"{prefix}.{key}"] = state[module][key]
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        digest.update(tensors[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# ------------------------------------------------- frozen early computation path -----------
FROZEN_MODULES = ("trunk", "early_head")
TRAINABLE_MODULES = ("late_head", "value_task_head", "value_cost_head")


def _module_of(controller: "PartitionedController", name: str) -> Any:
    return {"trunk": controller.inner.trunk,
            "early_head": controller.inner.early_head,
            "late_head": controller.inner.late_head,
            "value_task_head": controller.inner.value_head,
            "value_cost_head": controller.cost_head}[name]


def partitioned_parameters(controller: "PartitionedController") -> tuple[list, list]:
    """Independently enumerate the (frozen, trainable) parameter objects by module.

    Only object identity is checked here - duplicates, overlap and completeness. Whether the
    ``requires_grad`` flags actually match is a separate, explicit assertion
    (``assert_frozen_mask``) so a freshly restored or freshly loaded controller can be inspected
    before its mask is established.
    """
    frozen: list[Any] = []
    for name in FROZEN_MODULES:
        frozen.extend(_module_of(controller, name).parameters())
    trainable: list[Any] = []
    for name in TRAINABLE_MODULES:
        trainable.extend(_module_of(controller, name).parameters())
    frozen_ids = [id(p) for p in frozen]
    trainable_ids = [id(p) for p in trainable]
    if len(frozen_ids) != len(set(frozen_ids)):
        raise RuntimeError("R66 frozen enumeration contains a duplicate parameter object")
    if len(trainable_ids) != len(set(trainable_ids)):
        raise RuntimeError("R66 trainable enumeration contains a duplicate parameter object")
    if set(frozen_ids) & set(trainable_ids):
        raise RuntimeError("R66 frozen and trainable parameter sets overlap")
    all_ids = [id(p) for p in controller.parameters]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("R66 controller shares a parameter object across modules")
    if len(frozen) + len(trainable) != len(all_ids):
        raise RuntimeError("R66 frozen + trainable does not partition the parameter objects")
    return frozen, trainable


def assert_frozen_mask(controller: "PartitionedController") -> dict[str, Any]:
    """Assert the early path is frozen and the rest is trainable; return the mask facts."""
    frozen, trainable = partitioned_parameters(controller)
    if any(p.requires_grad for p in frozen):
        raise RuntimeError("R66 early path is NOT frozen: trunk/early_head still require grad")
    if any(not p.requires_grad for p in trainable):
        raise RuntimeError("R66 late_head or a critic is frozen but must stay trainable")
    return {"frozen_parameter_object_count": len(frozen),
            "trainable_parameter_object_count": len(trainable),
            "frozen_scalar_count": int(sum(p.numel() for p in frozen)),
            "trainable_scalar_count": int(sum(p.numel() for p in trainable)),
            "frozen_modules": list(FROZEN_MODULES),
            "trainable_modules": list(TRAINABLE_MODULES)}


def freeze_early_path(controller: "PartitionedController") -> dict[str, Any]:
    """Freeze the WHOLE early policy computation path: shared trunk + early_head.

    Returns the frozen identity (parameter object ids, scalar counts and the sha256 over the
    frozen tensors) so a caller can prove the frozen tensors never moved without ever passing a
    ``requires_grad=False`` parameter to ``autograd.grad``.
    """
    frozen, _trainable = partitioned_parameters(controller)
    for parameter in frozen:
        parameter.requires_grad_(False)
    mask = assert_frozen_mask(controller)
    identity = _tensor_identity(frozen)
    identity.update(mask)
    return identity


def trainable_parameters(controller: "PartitionedController") -> list:
    """The six trainable parameter objects: late_head plus both critics. Mask asserted."""
    assert_frozen_mask(controller)
    _frozen, trainable = partitioned_parameters(controller)
    return trainable


def frozen_parameters(controller: "PartitionedController") -> list:
    """The six frozen parameter objects: trunk + early_head. Mask asserted."""
    assert_frozen_mask(controller)
    frozen, _trainable = partitioned_parameters(controller)
    return frozen


def _tensor_identity(tensors: list) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    shapes: dict[str, list[int]] = {}
    for index, tensor in enumerate(tensors):
        digest.update(f"{index}".encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        count += int(tensor.numel())
        shapes[f"{index}"] = list(tensor.shape)
    return {"tensor_count": len(tensors), "scalar_count": count, "sha256": digest.hexdigest(),
            "shapes": shapes}


def frozen_parameter_summary(controller: "PartitionedController") -> dict[str, Any]:
    """The R66 parameter summary: the inherited R60 five-module identity plus the frozen split.

    Works on a controller whose mask may not be established yet (a freshly loaded ROUND60
    comparison arm); ``early_path_frozen`` then reports the actual flag state instead of raising.
    """
    base = controller.parameter_summary()
    frozen, trainable = partitioned_parameters(controller)
    frozen_identity = _tensor_identity(frozen)
    trainable_identity = _tensor_identity(trainable)
    actually_frozen = (not any(p.requires_grad for p in frozen)
                       and all(p.requires_grad for p in trainable))
    return {**base,
            "structure": R66_STRUCTURE,
            "routing": "ready_step=round(f[18]*1000)-round(f[0]); early<=147, late>147",
            "frozen_modules": list(FROZEN_MODULES),
            "trainable_modules": list(TRAINABLE_MODULES),
            "frozen_parameter_object_count": len(frozen),
            "trainable_parameter_object_count": len(trainable),
            "frozen_scalar_count": frozen_identity["scalar_count"],
            "trainable_scalar_count": trainable_identity["scalar_count"],
            "frozen_sha256": frozen_identity["sha256"],
            "trainable_sha256": trainable_identity["sha256"],
            "early_path_frozen": bool(actually_frozen),
            "freeze_note": ("trunk+early_head requires_grad=False; the frozen tensors receive no "
                            "gradient because they do not require one, which is not evidence "
                            "about the R53 critic->trunk isolation")}


def build_optimizer(controller: "PartitionedController", lr: float) -> torch.optim.Optimizer:
    """The R66 optimizer: Adam over the six trainable objects only."""
    return torch.optim.Adam(trainable_parameters(controller), lr=float(lr))


def build_frozen_controller(base_controller: Any, seed: int) -> tuple:
    """Wrap a freshly constructed base controller into the R66 frozen-early-path controller.

    ``base_controller`` must come from the same construction call ROUND60 used
    (``controller_module.fresh_controller(seed)``), so the frozen pilot RNG order is consumed
    identically and constructing the two extra heads (deepcopy, no RNG draw) cannot shift the
    training random start. All five modules are then overwritten bit-exactly from the compliant
    ROUND60 ep0 checkpoint and the whole early computation path is frozen. ROUND60 ep128, and any
    ROUND60 Adam state, lambda or training RNG, are never touched here.
    """
    ep0 = load_r60_model(seed, "ep0")
    controller = PartitionedController(base_controller)
    controller.load_state_dict(ep0["controller"])
    frozen_identity = freeze_early_path(controller)
    summary = frozen_parameter_summary(controller)
    if summary["parameter_sha256"] != ep0["parameter_sha256"]:
        raise RuntimeError(
            f"seed {seed}: R66 initial parameters do not match the compliant ROUND60 ep0 identity")
    return controller, ep0, frozen_identity, summary


# ------------------------------------------------------------------ likelihood -------------
def likelihood_alignment(controller: PartitionedController, samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute each sample's joint log-probability through the SAME routed forward as sampling."""
    from onpolicy import joint_log_prob
    import v2_common as common
    gaps: list[float] = []
    for sample in samples:
        with torch.no_grad():
            logits, _values = controller.forward(
                torch.as_tensor(sample["features"], dtype=torch.float32))
            recomputed = float(joint_log_prob(
                logits,
                torch.tensor([sample["actions"][t] for t in sample["targets"]], dtype=torch.long),
                torch.tensor([1.0 if t in sample["free"] else 0.0 for t in sample["targets"]],
                             dtype=torch.float32)))
        gaps.append(abs(recomputed - float(sample["log_prob"])))
    return {"samples": len(samples),
            "max_abs_gap": max(gaps) if gaps else 0.0,
            "tolerance": common.LOG_PROB_TOLERANCE,
            "passed": (not gaps) or max(gaps) <= common.LOG_PROB_TOLERANCE}


# ------------------------------------------------------------------ PPO update ------------
GATE_BETA = 1.00


def frozen_early_ppo_update(controller: PartitionedController, optimizer: torch.optim.Optimizer,
                            episodes: list[dict[str, Any]], *, update_index: int,
                            diag_updates: tuple[int, ...] = (0, 15, 31),
                            gate_beta: float = GATE_BETA) -> dict[str, Any]:
    """One PPO update over a 4-episode batch with routed early/late actor logits, early path frozen.

    Faithful transcription of the frozen ``v2_trainer.ppo_update`` and the R58 beta=1.00 gate
    prior. The routed early/late heads, the sequential mask, the joint log-probability, the
    entropy, the combined advantage, both critic targets and the ALL_ROWS reduction are exactly
    ROUND60's. ROUND66 changes ONLY which parameter objects the optimizer owns: ``trunk`` and
    ``early_head`` are ``requires_grad=False``, so ``optimizer.zero_grad`` / ``clip_grad_norm_`` /
    ``optimizer.step`` act on the six trainable objects (late_head + both critics) and the frozen
    modules receive no gradient at all because they do not require one.

    The beta=1.00 / ready_cut=147 gate prior is still computed over early first-emission free
    candidates exactly as before. Those candidates route to the frozen ``early_head``, so the
    prior cannot move the early path. That is the expected effect of this intervention and it is
    recorded as such; no loss term, weight or coefficient was added or changed to produce it.
    """
    import v2_trainer as trainer
    import v2_common as common
    from onpolicy import action_entropy, joint_log_prob

    # Fail closed if the caller handed us a controller whose early path is not actually frozen.
    frozen_parameters(controller)
    trainable_parameters(controller)
    samples = trainer.batch_samples(episodes)
    prefix_task, prefix_cost_bytes = trainer._unassigned_prefix_totals(episodes)
    alignment = likelihood_alignment(controller, samples)
    if not alignment["passed"]:
        raise RuntimeError(f"update {update_index}: logged probability does not describe the draw: {alignment}")
    if not samples:
        return {"samples": 0, "minibatches": 0, "optimizer_steps": 0,
                "unassigned_prefix_task_reward": prefix_task,
                "unassigned_prefix_cost_bytes": prefix_cost_bytes,
                "unassigned_prefix_cost_normalized": prefix_cost_bytes / common.B0_BYTES,
                "likelihood_alignment": alignment, "note": "no decision step in the batch"}
    shuffle = random.Random(int(hashlib.sha256(
        f"{common.DATA_ORDER_SEED}:{int(update_index)}".encode("utf-8")).hexdigest()[:16], 16))
    order = list(range(len(samples)))
    totals = {"surrogate": 0.0, "value_task": 0.0, "value_cost": 0.0, "entropy": 0.0,
              "kl": 0.0, "clip": 0.0, "grad": 0.0, "gate": 0.0, "gate_rows": 0,
              "early_free": 0, "late_free": 0}
    modules = (("trunk", controller.inner.trunk),
               ("early_head", controller.inner.early_head),
               ("late_head", controller.inner.late_head),
               ("task_value_head", controller.inner.value_head),
               ("cost_head", controller.cost_head))
    module_gradient_reads: list[dict[str, float]] = []
    gate_records: list[dict[str, Any]] = []
    gate_diag: dict[str, Any] = {"pending": update_index in diag_updates, "done": False, "record": None}
    minibatches = 0
    for _epoch in range(common.EPOCHS_PER_BATCH):
        shuffle.shuffle(order)
        for start in range(0, len(order), common.MINIBATCH_STEPS):
            chunk = [samples[index] for index in order[start:start + common.MINIBATCH_STEPS]]
            stacked = torch.as_tensor(np.concatenate([item["features"] for item in chunk], axis=0))
            hidden = controller.inner.trunk(stacked)
            logits = route_logits(controller, stacked, hidden)   # <-- the only structural change
            candidate_values_task = controller.inner.value_head(hidden).squeeze(-1)
            candidate_values_cost = controller.cost_head(hidden).squeeze(-1)
            row = 0
            new_log_prob: list[torch.Tensor] = []
            entropy: list[torch.Tensor] = []
            pooled_task: list[torch.Tensor] = []
            pooled_cost: list[torch.Tensor] = []
            gate_skip_log_probs: list[torch.Tensor] = []
            early_free = 0
            late_free = 0
            for item in chunk:
                count = len(item["targets"])
                rows = torch.arange(row, row + count)
                row += count
                picked = torch.tensor([item["actions"][t] for t in item["targets"]], dtype=torch.long)
                usable = torch.tensor([1.0 if t in item["free"] else 0.0
                                       for t in item["targets"]], dtype=torch.float32)
                new_log_prob.append(joint_log_prob(logits[rows], picked, usable))
                entropy.append(action_entropy(logits[rows], usable))
                pooled_task.append(candidate_values_task[rows].mean())
                pooled_cost.append(candidate_values_cost[rows].mean())
                # Gate prior (beta=1.00, ready_cut=147): early first-emission free candidates.
                for gate_index, gate_target in enumerate(item["targets"]):
                    if gate_target not in item["free"]:
                        continue
                    gate_feature = item["features"][gate_index]
                    ready = int(round(float(gate_feature[18]) * HORIZON) - round(float(gate_feature[0])))
                    if ready <= READY_CUT:
                        early_free += 1
                        if float(gate_feature[5]) == 1.0:
                            gate_skip_log_probs.append(
                                torch.log_softmax(logits[rows][gate_index], dim=-1)[SKIP_INDEX])
                    else:
                        late_free += 1
            new_log_prob_t = torch.stack(new_log_prob)
            entropy_t = torch.stack(entropy)
            old_log_prob = torch.tensor([item["log_prob"] for item in chunk], dtype=torch.float32)
            advantage = torch.tensor([item["advantage"] for item in chunk], dtype=torch.float32)
            target_task = torch.tensor([item["return_task"] for item in chunk], dtype=torch.float32)
            target_cost = torch.tensor([item["return_cost"] for item in chunk], dtype=torch.float32)
            ratio = torch.exp(new_log_prob_t - old_log_prob)
            surrogate = torch.min(ratio * advantage,
                                  torch.clamp(ratio, 1.0 - common.CLIP, 1.0 + common.CLIP) * advantage)
            value_task_loss = ((torch.stack(pooled_task) - target_task) ** 2).mean()
            value_cost_loss = ((torch.stack(pooled_cost) - target_cost) ** 2).mean()
            gate_loss = (-(torch.stack(gate_skip_log_probs)).sum() / len(chunk)
                         if gate_skip_log_probs else torch.zeros((), dtype=torch.float32))
            loss = (-surrogate.mean()
                    + common.VALUE_COEFFICIENT * (value_task_loss + value_cost_loss)
                    - common.ENTROPY_BONUS * entropy_t.mean()
                    + gate_beta * gate_loss)
            if gate_diag["pending"] and not gate_diag["done"]:
                # ONLY trainable parameters may be handed to autograd.grad. The frozen trunk and
                # early_head have requires_grad=False, so passing them would raise; and a zero
                # reading on them would say nothing about the R53 critic->trunk isolation.
                trainable = trainable_parameters(controller)
                late_params = list(controller.inner.late_head.parameters())
                critic_params = (list(controller.inner.value_head.parameters())
                                 + list(controller.cost_head.parameters()))
                base_only = loss - gate_beta * gate_loss
                prior_only = gate_beta * gate_loss
                # allow_unused=True only tolerates an UNUSED input; it does not make a
                # requires_grad=False output differentiable. torch.where can also keep a
                # zero-valued graph, so graph existence is checked, never assumed. A minibatch
                # with no eligible early first-emission free candidate has a structurally
                # constant gate term: that is recorded as absent, never as a measured zero.
                record = {"early_free_rows": early_free, "late_free_rows": late_free,
                          "base_output_requires_grad": bool(base_only.requires_grad),
                          "prior_output_requires_grad": bool(prior_only.requires_grad),
                          "early_head": "frozen_requires_grad_false",
                          "trunk": "frozen_requires_grad_false",
                          "trainable_parameter_object_count": len(trainable)}
                gb = gg = None
                if base_only.requires_grad:
                    gb = torch.autograd.grad(base_only, trainable, retain_graph=True, allow_unused=True)
                    record["base_trainable_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in gb if g is not None))
                    gr = torch.autograd.grad(base_only, late_params, retain_graph=True, allow_unused=True)
                    record["late_actor_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in gr if g is not None))
                    gc = torch.autograd.grad(base_only, critic_params, retain_graph=True, allow_unused=True)
                    record["critic_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in gc if g is not None))
                else:
                    record["base_trainable_grad_l1"] = None
                    record["late_actor_grad_l1"] = None
                    record["critic_grad_l1"] = None
                    record["base_gradient"] = "absent_output_has_no_graph"
                if prior_only.requires_grad:
                    gg = torch.autograd.grad(prior_only, trainable, retain_graph=True, allow_unused=True)
                    record["prior_trainable_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in gg if g is not None))
                    gr2 = torch.autograd.grad(prior_only, late_params, retain_graph=True, allow_unused=True)
                    record["prior_late_actor_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in gr2 if g is not None))
                    if gb is not None:
                        dot = float(sum(float((a * b).sum()) for a, b in zip(gb, gg)
                                        if a is not None and b is not None))
                        nb = math_sqrt(sum(float((g ** 2).sum()) for g in gb if g is not None))
                        ng = math_sqrt(sum(float((g ** 2).sum()) for g in gg if g is not None))
                        record["dot"] = dot
                        record["cosine"] = (dot / (nb * ng)) if nb > 0 and ng > 0 else None
                else:
                    record["prior_trainable_grad_l1"] = None
                    record["prior_late_actor_grad_l1"] = None
                    record["prior_gradient"] = "absent_no_eligible_gate_rows"
                gate_diag["record"] = record
                gate_diag["done"] = True
            optimizer.zero_grad()
            loss.backward()
            module_gradient_reads.append({
                module_name: (None if module_name in FROZEN_MODULES else
                              sum(float(parameter.grad.detach().abs().sum())
                                  for parameter in module.parameters() if parameter.grad is not None))
                for module_name, module in modules})
            # The R66 global clip norm is over the six trainable objects only. ROUND60 clipped
            # over all twelve, so the two norms are not comparable quantities.
            norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(controller),
                                                  common.MAX_GRAD_NORM)
            optimizer.step()
            if not (torch.isfinite(loss) and torch.isfinite(norm)):
                raise RuntimeError(f"update {update_index}: non-finite loss or gradient norm")
            totals["surrogate"] += float(surrogate.mean().item())
            totals["value_task"] += float(value_task_loss.item())
            totals["value_cost"] += float(value_cost_loss.item())
            totals["entropy"] += float(entropy_t.mean().item())
            totals["kl"] += float((old_log_prob - new_log_prob_t).mean().item())
            totals["clip"] += float(((ratio - 1.0).abs() > common.CLIP).float().mean().item())
            totals["grad"] += float(norm)
            totals["gate"] += float(gate_loss.item())
            totals["gate_rows"] += len(gate_skip_log_probs)
            totals["early_free"] += early_free
            totals["late_free"] += late_free
            minibatches += 1
            gate_records.append({"update_index": int(update_index), "minibatch": minibatches - 1,
                                 "n_rows": len(chunk), "gate_rows": len(gate_skip_log_probs),
                                 "early_free": early_free, "late_free": late_free,
                                 "gate_loss_raw": float(gate_loss.item()),
                                 "gate_loss_weighted": float(gate_beta * gate_loss.item())})
    diag_record = None
    if gate_diag["done"]:
        diag_record = {"update_index": update_index, **gate_diag["record"]}
    return {
        "samples": len(samples), "minibatches": minibatches, "optimizer_steps": minibatches,
        "epochs": common.EPOCHS_PER_BATCH,
        "unassigned_prefix_task_reward": prefix_task,
        "unassigned_prefix_cost_bytes": prefix_cost_bytes,
        "unassigned_prefix_cost_normalized": prefix_cost_bytes / common.B0_BYTES,
        "policy_surrogate": round(totals["surrogate"] / minibatches, 6),
        "value_loss_task": round(totals["value_task"] / minibatches, 6),
        "value_loss_cost": round(totals["value_cost"] / minibatches, 6),
        "entropy": round(totals["entropy"] / minibatches, 6),
        "approx_kl": round(totals["kl"] / minibatches, 6),
        "clip_fraction": round(totals["clip"] / minibatches, 4),
        "gradient_norm_mean": round(totals["grad"] / minibatches, 4),
        "gate_loss": round(totals["gate"] / minibatches, 6) if minibatches else 0.0,
        "gate_rows": totals["gate_rows"], "early_free": totals["early_free"], "late_free": totals["late_free"],
        "module_gradient_l1_mean": {
            module_name: (None if module_name in FROZEN_MODULES else
                          round(sum(read[module_name] for read in module_gradient_reads)
                                / len(module_gradient_reads), 9))
            for module_name, _module in modules} if module_gradient_reads else {},
        "module_gradient_l1_min": {
            module_name: (None if module_name in FROZEN_MODULES else
                          round(min(read[module_name] for read in module_gradient_reads), 9))
            for module_name, _module in modules} if module_gradient_reads else {},
        "module_gradient_scope": ("per-minibatch pre-clip L1 of every TRAINABLE module's "
                                  "parameters, read after backward and before clipping/step; "
                                  "frozen modules (trunk, early_head) are None because "
                                  "requires_grad=False, which is a structural fact and not a "
                                  "measured zero gradient"),
        "gate_diag": diag_record, "gate_records": gate_records,
        "likelihood_alignment": alignment,
        "structure": R66_STRUCTURE,
        "frozen_early_path": True,
        "trainable_parameter_object_count": len(trainable_parameters(controller)),
        "frozen_parameter_object_count": len(frozen_parameters(controller)),
    }


def math_sqrt(value: float) -> float:
    import math
    return math.sqrt(value)


# ------------------------------------------------------------------ checkpoint ------------
def save_checkpoint(controller: PartitionedController, optimizer: torch.optim.Optimizer,
                    state: dict[str, Any], path: Path, *, freeze: dict[str, Any]) -> Path:
    """Save all five modules, the six-parameter optimizer state and the frozen-mask identity.

    ``requires_grad`` is not part of a ``state_dict``, so the frozen mask is persisted explicitly
    and re-verified on every restore BEFORE the six-parameter Adam is built.
    """
    import v2_trainer as trainer
    frozen_identity = freeze_early_path(controller)
    payload = {
        "schema": R66_STATE_SCHEMA, "trainer_tag": R66_TRAINER_TAG,
        "training_version": R66_TRAINING_VERSION,
        "training_version_source": trainer.TRAINING_VERSION,
        "reward_timing_version": trainer.REWARD_TIMING_VERSION,
        "loss_version": R66_LOSS_VERSION,
        "gae_schema": trainer.GAE_SCHEMA, "optimizer_schema": R66_OPTIMIZER_SCHEMA,
        "reward_timing_schema": trainer.REWARD_TIMING_SCHEMA,
        "structure": R66_STRUCTURE,
        "freeze": freeze,
        "controller": {key: value for key, value in controller.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "optimizer_param_groups": [len(group["params"]) for group in optimizer.state_dict()["param_groups"]],
        "frozen_mask": {"frozen_modules": list(FROZEN_MODULES),
                        "trainable_modules": list(TRAINABLE_MODULES),
                        "frozen_sha256": frozen_identity["sha256"],
                        "frozen_scalar_count": frozen_identity["scalar_count"],
                        "frozen_parameter_object_count": frozen_identity["tensor_count"],
                        "note": "requires_grad is not serialised in a state_dict; this mask is the persisted contract"},
        "training_state": state,
        "rng": trainer.rng_state(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    import os
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic commit; a crash never leaves a half-written checkpoint
    return path


def _reject_foreign_contract(path: Path, payload: dict[str, Any]) -> None:
    """An R58 single-head or an ROUND60 partitioned checkpoint is never a ROUND66 resume source."""
    name = Path(path).name
    if payload.get("schema") == R60_STATE_SCHEMA or payload.get("trainer_tag") == R60_TRAINER_TAG:
        raise RuntimeError(
            f"{name}: this is a ROUND60 partitioned checkpoint. ROUND60 checkpoints are "
            "model-only sources (initialisation or comparison arm) and are REJECTED as a "
            "ROUND66 training-resume source")
    if payload.get("schema") != R66_STATE_SCHEMA or payload.get("trainer_tag") != R66_TRAINER_TAG:
        raise RuntimeError(f"{name}: not a ROUND66 frozen-early-path checkpoint")


def _validate_base(path: Path, payload: dict[str, Any], *, freeze: dict[str, Any]) -> dict[str, Any]:
    _reject_foreign_contract(path, payload)
    if payload.get("loss_version") != R66_LOSS_VERSION:
        raise RuntimeError(f"{Path(path).name}: loss version mismatch")
    if payload.get("gae_schema") != "per-decision-next-value/1" or payload.get("optimizer_schema") != R66_OPTIMIZER_SCHEMA:
        raise RuntimeError(f"{Path(path).name}: trainer contract schema mismatch")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict) or (
            training_state.get("loss_version") != R66_LOSS_VERSION
            or training_state.get("state_schema") != R66_STATE_SCHEMA
            or training_state.get("optimizer_schema") != R66_OPTIMIZER_SCHEMA):
        raise RuntimeError(f"{Path(path).name}: training state version mismatch")
    if payload.get("freeze") != freeze:
        raise RuntimeError(f"{Path(path).name}: freeze identity mismatch; refusing to resume")
    return training_state


def _restore_controller(payload: dict[str, Any], training_state: dict[str, Any]) -> PartitionedController:
    seed = int(training_state["training_seed"])
    controller = fresh_partitioned_controller(seed, ep0_state=None, verify_ep0=False)
    saved = payload["controller"]
    if set(saved) != {"trunk", "early_head", "late_head", "value_task_head", "value_cost_head"}:
        raise RuntimeError("checkpoint controller modules are not the partitioned five-module schema")
    controller.load_state_dict(saved)
    # Re-establish and verify the frozen mask BEFORE any optimizer is built: requires_grad is not
    # part of a state_dict, so a restored controller is trainable everywhere until this runs.
    identity = freeze_early_path(controller)
    mask = payload.get("frozen_mask") or {}
    if mask.get("frozen_sha256") != identity["sha256"]:
        raise RuntimeError("restored frozen mask does not match the restored frozen tensors")
    if int(mask.get("frozen_parameter_object_count", -1)) != identity["tensor_count"]:
        raise RuntimeError("restored frozen parameter object count mismatch")
    if list(mask.get("frozen_modules", [])) != list(FROZEN_MODULES):
        raise RuntimeError("restored frozen module list mismatch")
    if list(mask.get("trainable_modules", [])) != list(TRAINABLE_MODULES):
        raise RuntimeError("restored trainable module list mismatch")
    return controller


def load_checkpoint(path: Path, *, freeze: dict[str, Any]):
    """Resume only a ROUND66 frozen-early-path checkpoint from this exact version.

    The ROUND60 contract is explicitly rejected here. Restore order matters: the five modules and
    the frozen mask are restored and verified FIRST, then the six-parameter Adam is built over the
    trainable objects only and its state loaded, then the training RNG and cursor.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    training_state = _validate_base(path, payload, freeze=freeze)
    controller = _restore_controller(payload, training_state)
    optimizer = build_optimizer(controller, 3e-4)
    if [len(group["params"]) for group in payload["optimizer"]["param_groups"]] != [6]:
        raise RuntimeError("ROUND66 checkpoint optimizer is not the six-parameter trainable set")
    optimizer.load_state_dict(payload["optimizer"])
    import v2_trainer as trainer
    trainer.restore_rng(payload.get("rng"))
    return controller, optimizer, training_state


def load_checkpoint_read_only(path: Path, *, freeze: dict[str, Any]):
    """Read-only load of a ROUND66 checkpoint: weights and frozen mask only, never RNG or Adam."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    training_state = _validate_base(path, payload, freeze=freeze)
    controller = _restore_controller(payload, training_state)
    identity = {"load_mode": "read_only", "training_version": payload.get("training_version"),
                "reward_timing_version": payload.get("reward_timing_version"),
                "optimizer_restored": False, "rng_restored": False,
                "structure": R66_STRUCTURE,
                "frozen_mask": payload.get("frozen_mask"),
                "parameter_identity": frozen_parameter_summary(controller)}
    return controller, training_state, identity


def load_r60_checkpoint_read_only(seed: int, role: str):
    """Read-only load of a compliant ROUND60 checkpoint as the BC / OLD comparison arm.

    The path, bytes, parameter identity and training cursor are all bound to the expected
    compliant ROUND60 identity first. No ROUND60 optimizer state, lambda or training RNG is ever
    touched, and the returned controller is NOT frozen here: the caller decides whether this arm
    runs with its early path frozen (BC) or exactly as ROUND60 had it (OLD).
    """
    source = load_r60_model(seed, role)
    path = Path(source["path"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    training_state = payload.get("training_state") or {}
    controller = fresh_partitioned_controller(int(training_state.get("training_seed", int(seed))),
                                              ep0_state=None, verify_ep0=False)
    controller.load_state_dict(source["controller"])
    identity = {"load_mode": "model_only_read_only", "role": R60_MODEL_ROLES[role]["role"],
                "arm_role": role, "source_schema": R60_STATE_SCHEMA,
                "source_path": str(path), "source_file_sha256": source["file_sha256"],
                "source_parameter_sha256": source["parameter_sha256"],
                "training_version": payload.get("training_version"),
                "optimizer_restored": False, "rng_restored": False,
                "episodes_done": source["episodes_done"], "updates": source["updates"],
                "lambda_at_source": source["lambda_at_source"],
                "parameter_identity": frozen_parameter_summary(controller)}
    if identity["parameter_identity"]["parameter_sha256"] != source["parameter_sha256"]:
        raise RuntimeError(f"{path.name}: read-back parameter identity differs from the source")
    return controller, identity


def fresh_training_state(training_seed: int, *, gae_lambda: float = 1.0, eta_lambda: float = 0.5,
                         gamma: float = 1.0, delta: float = 0.0, lambda_init: float = 1.0,
                         data_order_seed: int = 20260927, b0_bytes: float = 10000.0) -> dict[str, Any]:
    import v2_trainer as trainer
    return {"training_version": R66_TRAINING_VERSION,
            "reward_timing_version": trainer.REWARD_TIMING_VERSION,
            "loss_version": R66_LOSS_VERSION,
            "state_schema": R66_STATE_SCHEMA, "gae_schema": trainer.GAE_SCHEMA,
            "optimizer_schema": R66_OPTIMIZER_SCHEMA,
            "reward_timing_schema": trainer.REWARD_TIMING_SCHEMA,
            "structure": R66_STRUCTURE,
            "frozen_modules": list(FROZEN_MODULES),
            "trainable_modules": list(TRAINABLE_MODULES),
            "training_seed": int(training_seed), "lambda": lambda_init,
            "lambda_history": [], "updates": 0, "episodes_done": 0, "optimizer_steps": 0,
            "b0_bytes": b0_bytes, "delta": delta, "eta_lambda": eta_lambda, "gamma": gamma,
            "gae_lambda": gae_lambda, "data_order_seed": data_order_seed}


print("r66_frozen module loaded")
