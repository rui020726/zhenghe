"""ROUND75 native PPO qualification wiring (trunk frozen; early/late heads + dual critic trainable).

QUALIFICATION-ONLY module. It is the technical eligibility wiring for the R71 scientific goal
(early learnable under the task constraint), not effect research and not a completed communication
module. Nothing in this module is executed during S0 (no torch.load, no model construction, no
forward/backward, no optimizer, no PPO, no env).

The update function ``r75_native_ppo_update`` is a faithful full-function adaptation of the REAL
``r66_frozen.frozen_early_ppo_update`` (round66_bc_early_path_preserving_late_adaptation/r66_frozen.py,
sha256 A22425729C9327FFCB1D57F4820F3B38F2BB7E35A6D6D81890FE64A75A246B10). Preserved verbatim:
``trainer.batch_samples`` + GAE, ``likelihood_alignment``, the ``DATA_ORDER_SEED``-seeded shuffle,
``EPOCHS_PER_BATCH`` x ``MINIBATCH_STEPS`` loop, the beta=1.00 / ready_cut=147 gate prior, the
clipped surrogate + dual-critic + entropy + gate loss, the global clip, the optimizer protocol and
the return contract.

THE ONLY structural change vs R66: the freeze set. R66 froze ("trunk", "early_head"); R75 freezes
("trunk",) only, so the trainable / optimizer / clip parameter set grows from 6 to 8 objects
(early_head + late_head + value_task_head + value_cost_head = 390 scalars; trunk stays 4 objects /
5440 scalars; 12 objects / 5830 scalars total). No loss term, weight, coefficient, reduction,
advantage combination, mask or protocol was added, removed or re-derived.

R75 additionally hard-wires the resource seam: every real NN/Adam path REQUIRES the unique ROUND75
central production ledger (``resources``). Passing None, or any simulated/fixture ledger, is
refused by source. Each physical optimizer step is charged BEFORE ``optimizer.step``; each real
trunk forward over target rows is charged BEFORE the forward. Failures do not refund.

Provenance caveats carried with this module:
- R58 rollout data is NOT an R71 trajectory. It is used here only as a real recorded batch for
  qualification, replayed through a gate-only controller whose behaviour actor is the R58
  single-head model copied into the two heads. Whether the recorded log-probs are reproduced is
  decided by the paid native ``likelihood_alignment``; if it fails, the run STOPS (the data is
  never rewritten and the alignment is never disabled).
- The gate-only controller is NOT the R71 research INIT. The research INIT remains the compliant
  R71 calibration model-only five-module checkpoint; qualification parameters never enter it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent                                   # .../comm_bc_initialized_constrained_ppo_20261007_r2
REPO = RUN_ROOT.parents[1]                               # .../zhenghe-round42
R66_DIR = RUN_ROOT / "round66_bc_early_path_preserving_late_adaptation"
for _d in (str(R66_DIR),):
    if str(_d) not in sys.path:
        sys.path.insert(0, _d)

import r66_frozen as r66                                  # noqa: E402  real frozen-early-path module
import v2_trainer as trainer                              # noqa: E402  frozen canonical trainer
import v2_common as common                                # noqa: E402  frozen canonical constants
from onpolicy import action_entropy, joint_log_prob       # noqa: E402  frozen canonical math

# ------------------------------------------------------------------ frozen identity ------
# The exact frozen canonical module byte identity accepted by ROUND58 and inherited since.
EXPECTED_TRAINER_SHA256 = "0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841"
EXPECTED_COMMON_SHA256 = "66d019a344d9cfef3c67bd8b722f8cc27144b9f71ddc8e5496451929d12f746a"
EXPECTED_R66_SOURCE_SHA256 = "a22425729c9327ffcb1d57f4820f3b38f2bb7e35a6d6d81890fe64a75a246b10"

R75_TRAINER_TAG = "constrained-cost-v2-r75-native-ppo-qualification"
R75_TRAINING_VERSION = "round75-native-ppo-qualification/1"
R75_STRUCTURE = ("partitioned-early-late-actor-head/1+native-ppo-qualification-trunk-frozen/1")
R75_LOSS_VERSION = "constrained-lagrangian/2+partitioned-actor-head/1+native-ppo-qualification/1"
R75_STATE_SCHEMA = ("constrained-cost-state/2+partitioned-actor-head/1+native-ppo-qualification/1")
R75_OPTIMIZER_SCHEMA = "stage-c-native-qualification-eight-object-parameter-set/1"

# THE single structural change vs R66/R71: early_head moves out of the frozen set.
FROZEN_MODULES = ("trunk",)
TRAINABLE_MODULES = ("early_head", "late_head", "value_task_head", "value_cost_head")
MODULES = ("trunk", "early_head", "late_head", "value_task_head", "value_cost_head")

READY_CUT = 147.0
HORIZON = 1000.0
SKIP_INDEX = 0
GATE_BETA = 1.00


def _module_of(controller, name: str):
    return {"trunk": controller.inner.trunk,
            "early_head": controller.inner.early_head,
            "late_head": controller.inner.late_head,
            "value_task_head": controller.inner.value_head,
            "value_cost_head": controller.cost_head}[name]


def partition(controller) -> tuple[list, list]:
    """Independently enumerate (frozen, trainable) parameter objects; object identity only."""
    frozen: list[Any] = []
    for name in FROZEN_MODULES:
        frozen.extend(_module_of(controller, name).parameters())
    trainable: list[Any] = []
    for name in TRAINABLE_MODULES:
        trainable.extend(_module_of(controller, name).parameters())
    frozen_ids = [id(p) for p in frozen]
    trainable_ids = [id(p) for p in trainable]
    if len(frozen_ids) != len(set(frozen_ids)):
        raise RuntimeError("R75 frozen enumeration contains a duplicate parameter object")
    if len(trainable_ids) != len(set(trainable_ids)):
        raise RuntimeError("R75 trainable enumeration contains a duplicate parameter object")
    if set(frozen_ids) & set(trainable_ids):
        raise RuntimeError("R75 frozen and trainable parameter sets overlap")
    all_ids = [id(p) for p in controller.parameters]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("R75 controller shares a parameter object across modules")
    if len(frozen) + len(trainable) != len(all_ids):
        raise RuntimeError("R75 frozen + trainable does not partition the parameter objects")
    return frozen, trainable


def assert_mask(controller) -> dict[str, Any]:
    """Assert the trunk-only freeze; return the mask facts (8 trainable / 4 frozen objects)."""
    frozen, trainable = partition(controller)
    if any(p.requires_grad for p in frozen):
        raise RuntimeError("R75 trunk is NOT frozen: trunk still requires grad")
    if any(not p.requires_grad for p in trainable):
        raise RuntimeError("R75 early_head/late_head/a critic is frozen but must stay trainable")
    return {"frozen_parameter_object_count": len(frozen),
            "trainable_parameter_object_count": len(trainable),
            "frozen_scalar_count": int(sum(p.numel() for p in frozen)),
            "trainable_scalar_count": int(sum(p.numel() for p in trainable)),
            "frozen_modules": list(FROZEN_MODULES),
            "trainable_modules": list(TRAINABLE_MODULES)}


def freeze_trunk_only(controller) -> dict[str, Any]:
    """requires_grad=False on the trunk only; return the frozen-tensor identity."""
    frozen, _trainable = partition(controller)
    for parameter in frozen:
        parameter.requires_grad_(False)
    mask = assert_mask(controller)
    digest = hashlib.sha256()
    for index, tensor in enumerate(frozen):
        digest.update(f"{index}".encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return {**mask, "frozen_sha256": digest.hexdigest(),
            "frozen_scalar_count": int(sum(t.numel() for t in frozen))}


def trainable_parameters(controller) -> list:
    """The EIGHT trainable parameter objects: early+late heads and both critics. Mask asserted."""
    assert_mask(controller)
    _frozen, trainable = partition(controller)
    return trainable


def frozen_parameters(controller) -> list:
    """The FOUR frozen parameter objects: the shared trunk. Mask asserted."""
    assert_mask(controller)
    frozen, _trainable = partition(controller)
    return frozen


def build_optimizer(controller, lr: float):
    """The R75 optimizer: Adam over the eight trainable objects only."""
    return torch.optim.Adam(trainable_parameters(controller), lr=float(lr))


def _require_production_ledger(resources) -> None:
    """Fail closed: real NN/Adam may only attach to the ROUND75 central production ledger."""
    if resources is None:
        raise RuntimeError(
            "R75 native PPO requires the ROUND75 central production ledger; a None/simulated "
            "attachment is refused by source (fixture execution mode is cancelled in R75)")
    if not getattr(resources, "r75_central_production", False):
        raise RuntimeError(
            "R75 native PPO received a resource object that is not the ROUND75 central "
            "production ledger; simulated attachment refused by source")


# ------------------------------------------------------------------ gate-only init ------
def build_gate_only_controller(seed: int) -> tuple:
    """QUALIFICATION-ONLY controller. NOT CALLED IN S0 (torch.load is a paid, licensed step).

    Uses the evidenced R58 single-head -> two-head conversion already implemented and verified in
    ``r66_frozen``: the four OLD modules (trunk, action_head, value_task_head, value_cost_head)
    are read from the byte-verified R58 ep0 BC_POST checkpoint, and BOTH independent heads are set
    to the OLD action head. Because the two heads are then value-identical, ``route_logits``
    (early<=147 / late>147) selects the same logits the single head produced, so the recorded
    behaviour log-probs are reproducible by construction (the gate prior is a training-loss term
    only and never enters the forward, exactly as the R71 alignment gap 2.3e-13 attests).

    These parameters are for qualification ONLY: they never enter the R71 research INIT and never
    become a research training checkpoint.
    """
    state = r66.load_r58_ep0_model(int(seed))            # byte-verified, model-only
    controller = r66.fresh_partitioned_controller(int(seed), ep0_state=state)
    frozen_identity = freeze_trunk_only(controller)
    summary = controller.parameter_summary()
    return controller, state, frozen_identity, summary


# ------------------------------------------------------------------ native PPO update ---
def r75_native_ppo_update(controller, optimizer, episodes: list[dict[str, Any]], *,
                          update_index: int, resources,
                          diag_updates: tuple[int, ...] = (0, 15, 31),
                          gate_beta: float = GATE_BETA) -> dict[str, Any]:
    """One native PPO update over a real recorded batch; trunk frozen; 8 trainable objects.

    Faithful adaptation of ``r66_frozen.frozen_early_ppo_update``. Everything except the freeze
    set, the parameter ownership (6 -> 8 objects) and the R75 resource seam is the real function.
    """
    _require_production_ledger(resources)
    frozen_parameters(controller)
    trainable_parameters(controller)

    # Real recorded batch -> samples (real batch_samples + GAE + reward-event attribution).
    # batch_samples performs no trunk forward (recorded-value tensor math only): no charge here.
    samples = trainer.batch_samples(episodes)
    prefix_task, prefix_cost_bytes = trainer._unassigned_prefix_totals(episodes)

    # Real likelihood recomputation through the SAME routed forward as sampling.
    resources.charge_offline_rows(
        sum(len(item["features"]) for item in samples),
        what=f"r75_native_update_{update_index}_alignment")
    alignment = r66.likelihood_alignment(controller, samples)
    if not alignment["passed"]:
        raise RuntimeError(
            f"update {update_index}: logged probability does not describe the draw: {alignment}")
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
            resources.charge_offline_rows(int(stacked.shape[0]),
                                          what=f"r75_native_update_{update_index}_minibatch_{minibatches}_trunk")
            hidden = controller.inner.trunk(stacked)
            logits = r66.route_logits(controller, stacked, hidden)
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
                # Passive diagnostic: raw early/late/critic gradients read AFTER backward and
                # BEFORE clip/step, via autograd.grad on the TRAINABLE objects only. The frozen
                # trunk has requires_grad=False and is never handed to autograd.grad.
                trainable = trainable_parameters(controller)
                early_params = list(controller.inner.early_head.parameters())
                late_params = list(controller.inner.late_head.parameters())
                critic_params = (list(controller.inner.value_head.parameters())
                                 + list(controller.cost_head.parameters()))
                base_only = loss - gate_beta * gate_loss
                prior_only = gate_beta * gate_loss
                record = {"early_free_rows": early_free, "late_free_rows": late_free,
                          "base_output_requires_grad": bool(base_only.requires_grad),
                          "prior_output_requires_grad": bool(prior_only.requires_grad),
                          "trunk": "frozen_requires_grad_false",
                          "trainable_parameter_object_count": len(trainable)}
                gb = gg = None
                if base_only.requires_grad:
                    gb = torch.autograd.grad(base_only, trainable, retain_graph=True, allow_unused=True)
                    record["base_trainable_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in gb if g is not None))
                    ge = torch.autograd.grad(base_only, early_params, retain_graph=True, allow_unused=True)
                    record["early_actor_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in ge if g is not None))
                    gr = torch.autograd.grad(base_only, late_params, retain_graph=True, allow_unused=True)
                    record["late_actor_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in gr if g is not None))
                    gc = torch.autograd.grad(base_only, critic_params, retain_graph=True, allow_unused=True)
                    record["critic_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in gc if g is not None))
                else:
                    record["base_trainable_grad_l1"] = None
                    record["early_actor_grad_l1"] = None
                    record["late_actor_grad_l1"] = None
                    record["critic_grad_l1"] = None
                    record["base_gradient"] = "absent_output_has_no_graph"
                if prior_only.requires_grad:
                    gg = torch.autograd.grad(prior_only, trainable, retain_graph=True, allow_unused=True)
                    record["prior_trainable_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in gg if g is not None))
                    ge2 = torch.autograd.grad(prior_only, early_params, retain_graph=True, allow_unused=True)
                    record["prior_early_actor_grad_l1"] = float(
                        sum(float(g.detach().abs().sum()) for g in ge2 if g is not None))
                    if gb is not None:
                        dot = float(sum(float((a * b).sum()) for a, b in zip(gb, gg)
                                        if a is not None and b is not None))
                        nb = math_sqrt(sum(float((g ** 2).sum()) for g in gb if g is not None))
                        ng = math_sqrt(sum(float((g ** 2).sum()) for g in gg if g is not None))
                        record["dot"] = dot
                        record["cosine"] = (dot / (nb * ng)) if nb > 0 and ng > 0 else None
                else:
                    record["prior_trainable_grad_l1"] = None
                    record["prior_early_actor_grad_l1"] = None
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
            # The R75 global clip norm is over the EIGHT trainable objects only.
            norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(controller),
                                                  common.MAX_GRAD_NORM)
            # Pre-charge BEFORE the physical step; a step failure does not refund.
            resources.charge_physical_step(
                what=f"r75_native_update_{update_index}_minibatch_{minibatches}")
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
        "approx_kl": round(totals["approx_kl"] / minibatches, 6) if minibatches else 0.0,
        "clip_fraction": round(totals["clip"] / minibatches, 4),
        "gradient_norm_mean": round(totals["grad"] / minibatches, 4),
        "gate_loss": round(totals["gate"] / minibatches, 6) if minibatches else 0.0,
        "gate_rows": totals["gate_rows"], "early_free": totals["early_free"],
        "late_free": totals["late_free"],
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
                                  "the frozen trunk is None because requires_grad=False, which "
                                  "is a structural fact and not a measured zero gradient"),
        "gate_diag": diag_record, "gate_records": gate_records,
        "likelihood_alignment": alignment,
        "structure": R75_STRUCTURE,
        "frozen_modules": list(FROZEN_MODULES),
        "trainable_modules": list(TRAINABLE_MODULES),
        "trainable_parameter_object_count": len(trainable_parameters(controller)),
        "frozen_parameter_object_count": len(frozen_parameters(controller)),
    }


def math_sqrt(value: float) -> float:
    import math
    return math.sqrt(value)


def update_accounting(n_samples: int, trunk_rows_per_epoch: int) -> dict[str, int]:
    """Pure arithmetic (no torch): the conservative per-update resource accounting.

    steps = EPOCHS_PER_BATCH x ceil(n_samples / MINIBATCH_STEPS); each minibatch is one physical
    optimizer step. Offline trunk target rows = one alignment pass + one pass per epoch.
    """
    import math as _math
    minibatches_per_epoch = max(1, _math.ceil(n_samples / 256))
    steps = 4 * minibatches_per_epoch
    return {"samples": int(n_samples),
            "minibatches_per_epoch": int(minibatches_per_epoch),
            "physical_optimizer_steps": int(steps),
            "offline_trunk_target_rows": int(trunk_rows_per_epoch * (4 + 1)),
            "epochs": 4, "minibatch_steps": 256}