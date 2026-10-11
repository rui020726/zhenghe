"""ROUND60 partitioned actor-head ablation: two independent early/late SEND/SKIP heads.

This module is the ONLY structural intervention of ROUND60. Everything it does not touch is
inherited field-by-field from the accepted ROUND58 living configuration, verified from the
R58 ep128 checkpoint ``training_state`` (see FREEZE notes below), not from module import
defaults.

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

# NEW training / model structure version and checkpoint contract. Old single-head checkpoints
# (action_head) are NOT valid NEW training-resume sources; they are model-only init sources.
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
    base = v2_controller.fresh_controller(int(seed))
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


def partitioned_ppo_update(controller: PartitionedController, optimizer: torch.optim.Optimizer,
                           episodes: list[dict[str, Any]], *, update_index: int,
                           diag_updates: tuple[int, ...] = (0, 15, 31),
                           gate_beta: float = GATE_BETA) -> dict[str, Any]:
    """One PPO update over a 4-episode batch with routed early/late actor logits.

    Faithful transcription of the frozen ``v2_trainer.ppo_update`` and the R58 beta=1.00 gate
    prior, with exactly one structural change: the minibatch logits come from the routed
    early/late heads instead of the single shared ``action_head``. The gate prior still softly
    pushes early first-emission free candidates toward SKIP; because those candidates route to
    ``early_head``, the prior now acts through the early head. ``batch_samples`` (GAE, combined
    advantage, both critic targets, ALL_ROWS reduction) is the frozen function unchanged.
    """
    import v2_trainer as trainer
    import v2_common as common
    from onpolicy import action_entropy, joint_log_prob

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
                actor_params = (list(controller.inner.trunk.parameters())
                                + list(controller.inner.early_head.parameters())
                                + list(controller.inner.late_head.parameters()))
                base_only = loss - gate_beta * gate_loss
                gb = torch.autograd.grad(base_only, actor_params, retain_graph=True, allow_unused=True)
                gg = torch.autograd.grad(gate_beta * gate_loss, actor_params, retain_graph=True, allow_unused=True)
                l1b = float(sum(float(g.detach().abs().sum()) for g in gb if g is not None))
                l1g = float(sum(float(g.detach().abs().sum()) for g in gg if g is not None))
                dot = float(sum(float((a * b).sum()) for a, b in zip(gb, gg) if a is not None and b is not None))
                nb = math_sqrt(sum(float((g ** 2).sum()) for g in gb if g is not None))
                ng = math_sqrt(sum(float((g ** 2).sum()) for g in gg if g is not None))
                # Per-head base-actor L1 so the diagnostic can show the two heads diverge.
                head_params = {"early": list(controller.inner.early_head.parameters()),
                               "late": list(controller.inner.late_head.parameters())}
                per_head_l1 = {}
                for name, params in head_params.items():
                    gr = torch.autograd.grad(base_only, params, retain_graph=True, allow_unused=True)
                    per_head_l1[name] = float(sum(float(g.detach().abs().sum()) for g in gr if g is not None))
                gate_diag["record"] = {
                    "base_actor_grad_l1": l1b, "prior_grad_l1": l1g, "dot": dot,
                    "cosine": (dot / (nb * ng)) if nb > 0 and ng > 0 else None,
                    "base_actor_grad_l1_by_head": per_head_l1,
                    "early_free_rows": early_free, "late_free_rows": late_free}
                gate_diag["done"] = True
            optimizer.zero_grad()
            loss.backward()
            module_gradient_reads.append({
                module_name: sum(float(parameter.grad.detach().abs().sum())
                                 for parameter in module.parameters() if parameter.grad is not None)
                for module_name, module in modules})
            norm = torch.nn.utils.clip_grad_norm_(controller.parameters, common.MAX_GRAD_NORM)
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
            module_name: round(sum(read[module_name] for read in module_gradient_reads) / len(module_gradient_reads), 9)
            for module_name, _module in modules} if module_gradient_reads else {},
        "module_gradient_l1_min": {
            module_name: round(min(read[module_name] for read in module_gradient_reads), 9)
            for module_name, _module in modules} if module_gradient_reads else {},
        "module_gradient_scope": "per-minibatch pre-clip L1 of every module's parameters, read after backward and before clipping/step",
        "gate_diag": diag_record, "gate_records": gate_records,
        "likelihood_alignment": alignment,
        "structure": "partitioned-early-late-actor-head/1",
    }


def math_sqrt(value: float) -> float:
    import math
    return math.sqrt(value)


# ------------------------------------------------------------------ checkpoint ------------
def save_checkpoint(controller: PartitionedController, optimizer: torch.optim.Optimizer,
                    state: dict[str, Any], path: Path, *, freeze: dict[str, Any]) -> Path:
    import v2_trainer as trainer
    payload = {
        "schema": R60_STATE_SCHEMA, "trainer_tag": R60_TRAINER_TAG,
        "training_version": R60_TRAINING_VERSION,
        "training_version_source": trainer.TRAINING_VERSION,
        "reward_timing_version": trainer.REWARD_TIMING_VERSION,
        "loss_version": R60_LOSS_VERSION,
        "gae_schema": trainer.GAE_SCHEMA, "optimizer_schema": R60_OPTIMIZER_SCHEMA,
        "reward_timing_schema": trainer.REWARD_TIMING_SCHEMA,
        "structure": "partitioned-early-late-actor-head/1",
        "freeze": freeze,
        "controller": {key: value for key, value in controller.state_dict().items()},
        "optimizer": optimizer.state_dict(),
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


def _validate_base(path: Path, payload: dict[str, Any], *, freeze: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != R60_STATE_SCHEMA or payload.get("trainer_tag") != R60_TRAINER_TAG:
        raise RuntimeError(f"{Path(path).name}: not a ROUND60 partitioned checkpoint")
    if payload.get("loss_version") != R60_LOSS_VERSION:
        raise RuntimeError(f"{Path(path).name}: loss version mismatch")
    if payload.get("gae_schema") != "per-decision-next-value/1" or payload.get("optimizer_schema") != R60_OPTIMIZER_SCHEMA:
        raise RuntimeError(f"{Path(path).name}: trainer contract schema mismatch")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict) or (
            training_state.get("loss_version") != R60_LOSS_VERSION
            or training_state.get("state_schema") != R60_STATE_SCHEMA
            or training_state.get("optimizer_schema") != R60_OPTIMIZER_SCHEMA):
        raise RuntimeError(f"{Path(path).name}: training state version mismatch")
    if payload.get("freeze") != freeze:
        raise RuntimeError(f"{Path(path).name}: freeze identity mismatch; refusing to resume")
    return training_state


def _restore_controller(payload: dict[str, Any], training_state: dict[str, Any]) -> PartitionedController:
    seed = int(training_state["training_seed"])
    controller = fresh_partitioned_controller(seed, ep0_state=None, verify_ep0=False)
    saved = payload["controller"]
    if set(saved) != {"trunk", "early_head", "late_head", "value_task_head", "value_cost_head"}:
        raise RuntimeError(f"checkpoint controller modules are not the partitioned five-module schema")
    controller.load_state_dict(saved)
    return controller


def load_checkpoint(path: Path, *, freeze: dict[str, Any]):
    """Resume only a ROUND60 partitioned checkpoint from this exact version. Old single-head
    (action_head) R58 checkpoints are rejected here; they are model-only init sources only."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    training_state = _validate_base(path, payload, freeze=freeze)
    controller = _restore_controller(payload, training_state)
    optimizer = torch.optim.Adam(controller.parameters, lr=3e-4)
    optimizer.load_state_dict(payload["optimizer"])
    import v2_trainer as trainer
    trainer.restore_rng(payload.get("rng"))
    return controller, optimizer, training_state


def load_checkpoint_read_only(path: Path, *, freeze: dict[str, Any]):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    training_state = _validate_base(path, payload, freeze=freeze)
    controller = _restore_controller(payload, training_state)
    identity = {"load_mode": "read_only", "training_version": payload.get("training_version"),
                "reward_timing_version": payload.get("reward_timing_version"),
                "optimizer_restored": False, "rng_restored": False,
                "structure": "partitioned-early-late-actor-head/1"}
    return controller, training_state, identity


def fresh_training_state(training_seed: int, *, gae_lambda: float = 1.0, eta_lambda: float = 0.5,
                         gamma: float = 1.0, delta: float = 0.0, lambda_init: float = 1.0,
                         data_order_seed: int = 20260927, b0_bytes: float = 10000.0) -> dict[str, Any]:
    import v2_trainer as trainer
    return {"training_version": R60_TRAINING_VERSION,
            "reward_timing_version": trainer.REWARD_TIMING_VERSION,
            "loss_version": R60_LOSS_VERSION,
            "state_schema": R60_STATE_SCHEMA, "gae_schema": trainer.GAE_SCHEMA,
            "optimizer_schema": R60_OPTIMIZER_SCHEMA,
            "reward_timing_schema": trainer.REWARD_TIMING_SCHEMA,
            "structure": "partitioned-early-late-actor-head/1",
            "training_seed": int(training_seed), "lambda": lambda_init,
            "lambda_history": [], "updates": 0, "episodes_done": 0, "optimizer_steps": 0,
            "b0_bytes": b0_bytes, "delta": delta, "eta_lambda": eta_lambda, "gamma": gamma,
            "gae_lambda": gae_lambda, "data_order_seed": data_order_seed}


print("r60_partitioned module loaded")
print("r60_partitioned module loaded")
