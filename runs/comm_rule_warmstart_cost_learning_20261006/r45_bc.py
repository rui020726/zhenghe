"""ROUND45 frozen offline rule-BC initialization on the real R44 controller."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[1]
R44_DIR = ROOT / "runs" / "comm_dual_response_ablation_20261004"
R43_DIR = ROOT / "runs" / "comm_one_shot_full_trace_ablation_20261004"
ROUND40_DIR = ROOT.parent / "zhenghe-round40" / "runs" / "comm_first_report_training_ablation_20261003"
CANONICAL_ROOT = ROOT.parent / "zhenghe"
DATA_PATH = RUN_DIR / "BC_DATASET.jsonl"
AUDIT_PATH = RUN_DIR / "BC_DATA_AUDIT_R2.json"
EP0_PATH = R44_DIR / "ROUND44_EP0_INITIALIZATION.json"
R44_FREEZE_PATH = R44_DIR / "ROUND44_FROZEN_CONFIG.json"

SEEDS = (11, 22, 33)
FEATURE_WIDTH = 19
FIT_SCENES = set(range(1_300_000, 1_300_032))
VALIDATION_SCENES = set(range(1_300_032, 1_300_048))
BC_UPDATES = 1000
BC_LR = 0.001
MAX_GRAD_NORM = 0.5
BC_VERSION = "offline-rule-bc-late-discovery-gate-147/1"
PPO_TRAINING_VERSION = (
    "constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/"
    "dual-eta-lambda-0p5/1/rule-bc-late147-init/1/"
    "published-completion-event-at-relay-tick/2"
)

EXPECTED_EP0_PARAMETER_SHA256 = {
    11: "7eff0f40203fdae63437bb07a16204ab0289bf938e9c263f781b3adfc40d5eff",
    22: "b74c10c85c148a4b46b4513a6b04660446c31a778a6537a8f13e22ffc78e8095",
    33: "0ddebe7dc04019ebb079e25b9b4191b654e59167b928c416112393fbc6d0d3aa",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def _append_fsync(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def configure_source_imports() -> Any:
    """Import the controller from the registered canonical R44 source tree."""
    import_paths = [
        R44_DIR,
        R43_DIR,
        ROUND40_DIR,
        CANONICAL_ROOT / "runs" / "comm_cost_v2_rerun_20260928",
        CANONICAL_ROOT / "runs" / "comm_cost_stage_b_20260927",
        CANONICAL_ROOT / "runs" / "comm_budget_pilot_20260919",
    ]
    for path in reversed(import_paths):
        text = str(path.resolve())
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    import v2_controller

    expected = (CANONICAL_ROOT / "runs" / "comm_cost_v2_rerun_20260928"
                / "v2_controller.py").resolve()
    actual = Path(v2_controller.__file__).resolve()
    if actual != expected:
        raise RuntimeError(f"controller import provenance mismatch: {actual} != {expected}")
    return v2_controller


def _module_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _rng_snapshot() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(snapshot: dict[str, Any]) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch"])
    if snapshot["cuda"] is not None:
        torch.cuda.set_rng_state_all(snapshot["cuda"])


def load_verified_dataset() -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    audit = _json(AUDIT_PATH)
    if audit.get("kind") != "round45-bc-data-audit/1" or audit.get("status") != "passed":
        raise RuntimeError("ROUND45 dataset audit is not a pass")
    dataset_hash = sha256_file(DATA_PATH)
    if (dataset_hash != audit.get("output_dataset_sha256")
            or DATA_PATH.stat().st_size != int(audit.get("output_dataset_bytes", -1))):
        raise RuntimeError("BC dataset bytes differ from the R2 audit")
    records = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines()
               if line]
    counts = {split: {0: 0, 1: 0} for split in ("fit", "validation")}
    feature_labels: dict[tuple[str, ...], int] = {}
    for record in records:
        split = str(record["split"])
        label = int(record["teacher_label"])
        scene = int(record["scene_id"])
        features = record["feature_float32"]
        if split not in counts or label not in (0, 1) or len(features) != FEATURE_WIDTH:
            raise RuntimeError("BC dataset row has an invalid split, class, or feature width")
        if (split == "fit" and scene not in FIT_SCENES
                or split == "validation" and scene not in VALIDATION_SCENES):
            raise RuntimeError(f"scene {scene} falls outside its preregistered split")
        expected_action = "SEND_NOW" if label == 1 else "SKIP_EPISODE"
        if record.get("teacher_action") != expected_action:
            raise RuntimeError("dataset teacher action and integer label disagree")
        counts[split][label] += 1
        key = tuple(float(value).hex() for value in features)
        previous = feature_labels.setdefault(key, label)
        if previous != label:
            raise RuntimeError("same float32 feature maps to conflicting teacher labels")
    expected = {
        "fit": {0: 1242, 1: 306},
        "validation": {0: 525, 1: 213},
    }
    if counts != expected:
        raise RuntimeError(f"BC row class counts differ from the frozen audit: {counts}")
    if audit["split"]["development_scenes_used_for_fit_or_validation"]:
        raise RuntimeError("development scenes leaked into the BC dataset")
    return records, audit, dataset_hash


def inverse_frequency_weights(records: list[dict[str, Any]]) -> dict[int, float]:
    fit = [row for row in records if row["split"] == "fit"]
    counts = {label: sum(int(row["teacher_label"]) == label for row in fit)
              for label in (0, 1)}
    if any(value <= 0 for value in counts.values()):
        raise RuntimeError("fit set is missing a teacher class")
    total = len(fit)
    return {label: total / (2.0 * counts[label]) for label in (0, 1)}


def bc_parameters(controller: Any) -> list[torch.nn.Parameter]:
    """The actual canonical controller's shared trunk and action head, each once."""
    return [*controller.inner.trunk.parameters(),
            *controller.inner.action_head.parameters()]


def module_parameter_sets(controller: Any) -> dict[str, list[torch.nn.Parameter]]:
    return {
        "trunk": list(controller.inner.trunk.parameters()),
        "action_head": list(controller.inner.action_head.parameters()),
        "task_head": list(controller.inner.value_head.parameters()),
        "cost_head": list(controller.cost_head.parameters()),
    }


def validate_optimizer_scope(controller: Any, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    groups = module_parameter_sets(controller)
    expected = groups["trunk"] + groups["action_head"]
    actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    expected_ids = [id(parameter) for parameter in expected]
    actual_ids = [id(parameter) for parameter in actual]
    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("BC module enumeration contains duplicate parameter objects")
    if actual_ids != expected_ids:
        raise RuntimeError("BC Adam parameters are not exactly trunk plus action head in order")
    all_ids = [id(parameter) for module in groups.values() for parameter in module]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("canonical controller modules unexpectedly share parameter objects")
    if set(expected_ids) & {id(p) for p in groups["task_head"] + groups["cost_head"]}:
        raise RuntimeError("BC optimizer includes a critic-head parameter")
    return {
        "module_parameter_counts": {name: sum(p.numel() for p in params)
                                    for name, params in groups.items()},
        "optimizer_parameter_count": sum(parameter.numel() for parameter in actual),
        "optimizer_parameter_object_count": len(actual),
        "optimizer_order": ["shared_trunk", "action_head"],
        "task_head_included": False,
        "cost_head_included": False,
        "optimizer_ids_match_independent_module_enumeration": True,
        "zero_grad_clip_scope_evidence": (
            "one_bc_update assigns this independently-validated trunk/action list once; "
            "optimizer.zero_grad() clears that optimizer and clip_grad_norm_ receives the "
            "same local list"
        ),
    }


def make_bc_optimizer(controller: Any, *, lr: float = BC_LR) -> torch.optim.Adam:
    optimizer = torch.optim.Adam(bc_parameters(controller), lr=lr)
    validate_optimizer_scope(controller, optimizer)
    return optimizer


def greedy_actions_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Use the frozen OneShotSendSkipCollector SEND-on-tie greedy convention."""
    probabilities = torch.softmax(logits, dim=-1)
    return (probabilities[:, 1] >= probabilities[:, 0]).to(dtype=torch.long)


def prepare_bc_gradients(controller: Any, optimizer: torch.optim.Optimizer,
                         features: torch.Tensor, labels: torch.Tensor,
                         class_weights: torch.Tensor) -> dict[str, Any]:
    """Run the production forward/loss/backward/clip path, without applying an update."""
    validate_optimizer_scope(controller, optimizer)
    parameters = bc_parameters(controller)
    optimizer.zero_grad(set_to_none=True)
    logits, _task_values = controller.forward(features)
    loss = F.cross_entropy(logits, labels, weight=class_weights, reduction="mean")
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("BC loss is non-finite")
    loss.backward()
    action_bias_gradient = controller.inner.action_head.bias.grad.detach().clone()
    if any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("a selected shared-trunk/action parameter has no BC gradient")
    gradients_finite = all(bool(torch.isfinite(parameter.grad).all())
                           for parameter in parameters if parameter.grad is not None)
    if not gradients_finite:
        raise RuntimeError("BC gradient is non-finite")
    norm_before = torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
    norm_after = torch.sqrt(sum(parameter.grad.detach().pow(2).sum()
                                for parameter in parameters)).item()
    if not bool(torch.isfinite(norm_before)) or not math.isfinite(norm_after):
        raise RuntimeError("BC gradient norm is non-finite")
    return {
        "loss": float(loss.detach().item()),
        "grad_norm_before_clip": float(norm_before.detach().item()),
        "grad_norm_after_clip": float(norm_after),
        "gradients_finite": gradients_finite,
        "action_head_bias_gradient_preclip": [float(v) for v in action_bias_gradient.cpu()],
    }


def one_bc_update(controller: Any, optimizer: torch.optim.Optimizer,
                  features: torch.Tensor, labels: torch.Tensor,
                  class_weights: torch.Tensor) -> dict[str, Any]:
    """One full supplied batch update through the canonical controller BC path."""
    report = prepare_bc_gradients(controller, optimizer, features, labels, class_weights)
    optimizer.step()
    return report


def _metrics(controller: Any, records: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in records if row["split"] == split]
    features = torch.tensor([row["feature_float32"] for row in selected],
                            dtype=torch.float32, device="cpu")
    labels = torch.tensor([int(row["teacher_label"]) for row in selected], dtype=torch.long)
    with torch.no_grad():
        logits, _task_values = controller.forward(features)
        probs = torch.softmax(logits, dim=-1)
        predicted = greedy_actions_from_logits(logits)
    matrix = [[0, 0], [0, 0]]
    for truth, guess in zip(labels.tolist(), predicted.tolist()):
        matrix[truth][guess] += 1
    send_probability = probs[:, 1]
    return {
        "samples": len(selected),
        "confusion_true_rows_predicted_columns": matrix,
        "accuracy": float((predicted == labels).float().mean().item()),
        "teacher_agreement": float((predicted == labels).float().mean().item()),
        "predicted_action_counts": {
            "SKIP_EPISODE": int((predicted == 0).sum().item()),
            "SEND_NOW": int((predicted == 1).sum().item()),
        },
        "mean_probability": {
            "SKIP_EPISODE": float(probs[:, 0].mean().item()),
            "SEND_NOW": float(probs[:, 1].mean().item()),
        },
        "mean_probability_by_teacher": {
            "SKIP_EPISODE": float(send_probability[labels == 0].mean().item()),
            "SEND_NOW": float(send_probability[labels == 1].mean().item()),
        },
    }


def _expected_ep0_identities() -> dict[int, dict[str, Any]]:
    obj = _json(EP0_PATH)
    if obj.get("kind") != "round44-ep0-initialization-gate/1" or obj.get("status") != "passed":
        raise RuntimeError("R44 registered ep0 initialization identity is not passed")
    return {int(seed): payload for seed, payload in obj["by_seed"].items()}


def _optimizer_defaults(optimizer: torch.optim.Adam) -> dict[str, Any]:
    defaults = dict(optimizer.defaults)
    return {key: (list(value) if isinstance(value, tuple) else value)
            for key, value in sorted(defaults.items())}


def train_seed(seed: int, records: list[dict[str, Any]], *, dataset_sha256: str,
               expected_ep0: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"unregistered BC seed {seed}")
    checkpoint_dir = RUN_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"BC_POST_MODEL_s{seed}.pt"
    metrics_path = RUN_DIR / f"BC_METRICS_s{seed}.json"
    journal_path = RUN_DIR / "bc_journals" / f"BC_OPTIMIZER_STEPS_s{seed}.jsonl"
    if checkpoint_path.exists() or metrics_path.exists() or journal_path.exists():
        raise RuntimeError(f"seed {seed}: BC outputs already exist; inspect instead of rerunning")

    fit_records = [row for row in records if row["split"] == "fit"]
    validation_records = [row for row in records if row["split"] == "validation"]
    weights = inverse_frequency_weights(records)
    features = torch.tensor([row["feature_float32"] for row in fit_records],
                            dtype=torch.float32, device="cpu")
    labels = torch.tensor([int(row["teacher_label"]) for row in fit_records], dtype=torch.long)
    weight_tensor = torch.tensor([weights[0], weights[1]], dtype=torch.float32, device="cpu")

    import v2_controller

    ambient_rng = _rng_snapshot()
    try:
        controller = v2_controller.fresh_controller(seed)
        initial_hash = controller.parameter_summary()["parameter_sha256"]
        expected = expected_ep0[seed]
        if (initial_hash != expected["registered_and_fresh_parameter_sha256"]
                or initial_hash != EXPECTED_EP0_PARAMETER_SHA256[seed]):
            raise RuntimeError(f"seed {seed}: fresh R44 model differs from registered ep0")
        if any(parameter.device.type != "cpu" for parameter in controller.parameters):
            raise RuntimeError(f"seed {seed}: BC controller is not on CPU")
        task_head_before = _module_hash(controller.inner.value_head)
        cost_head_before = _module_hash(controller.cost_head)
        optimizer = make_bc_optimizer(controller)
        parameter_scope = validate_optimizer_scope(controller, optimizer)
        optimizer_defaults = _optimizer_defaults(optimizer)
        if (float(optimizer_defaults["lr"]) != BC_LR
                or tuple(optimizer_defaults["betas"]) != (0.9, 0.999)
                or float(optimizer_defaults["eps"]) != 1e-8
                or float(optimizer_defaults["weight_decay"]) != 0.0
                or bool(optimizer_defaults["amsgrad"])):
            raise RuntimeError("installed Torch Adam defaults differ from the frozen plan")
        start_metrics = {"fit": _metrics(controller, records, "fit"),
                         "validation": _metrics(controller, records, "validation")}
        step_history: list[dict[str, Any]] = []
        original_step = optimizer.step
        step_index = 0

        def journaled_step(*args: Any, **kwargs: Any):
            nonlocal step_index
            next_index = step_index + 1
            _append_fsync(journal_path, {
                "seed": seed, "step": next_index, "event": "optimizer_step_invoked",
                "dataset_sha256": dataset_sha256,
                "parameter_sha256_before": controller.parameter_summary()["parameter_sha256"],
            })
            result = original_step(*args, **kwargs)
            step_index = next_index
            _append_fsync(journal_path, {
                "seed": seed, "step": next_index, "event": "optimizer_step_returned",
                "dataset_sha256": dataset_sha256,
            })
            return result

        optimizer.step = journaled_step
        for step in range(1, BC_UPDATES + 1):
            record = one_bc_update(controller, optimizer, features, labels, weight_tensor)
            if not record["gradients_finite"] or not math.isfinite(record["loss"]):
                raise RuntimeError(f"seed {seed} BC step {step}: non-finite loss/gradient")
            record["step"] = step
            step_history.append(record)
        if step_index != BC_UPDATES:
            raise RuntimeError(f"seed {seed}: observed {step_index}/{BC_UPDATES} Adam calls")
        expected_objects = len(bc_parameters(controller))
        optimizer_steps = []
        for parameter in bc_parameters(controller):
            state = optimizer.state.get(parameter)
            if not state or int(state["step"].item()) != BC_UPDATES:
                raise RuntimeError("BC Adam parameter step state did not reach exactly 1000")
            optimizer_steps.append(int(state["step"].item()))
        if len(optimizer.state) != expected_objects or set(optimizer_steps) != {BC_UPDATES}:
            raise RuntimeError("BC Adam state does not match the selected module parameters")
        task_head_after = _module_hash(controller.inner.value_head)
        cost_head_after = _module_hash(controller.cost_head)
        if task_head_before != task_head_after or cost_head_before != cost_head_after:
            raise RuntimeError("BC changed a critic head's own parameters")
        final_metrics = {"fit": _metrics(controller, records, "fit"),
                         "validation": _metrics(controller, records, "validation")}
        final_hash = controller.parameter_summary()["parameter_sha256"]
        if final_hash == initial_hash:
            raise RuntimeError(f"seed {seed}: BC did not change model parameters")
        result = {
            "kind": "round45-bc-seed-metrics/1", "seed": seed,
            "bc_version": BC_VERSION, "ppo_training_version": PPO_TRAINING_VERSION,
            "dataset_sha256": dataset_sha256, "dataset_split": "fit scenes 1300000..1300031; validation scenes 1300032..1300047",
            "training": {
                "updates": BC_UPDATES, "optimizer": "torch.optim.Adam", "optimizer_defaults": optimizer_defaults,
                "class_weights_fit_only": {"SKIP_EPISODE": weights[0], "SEND_NOW": weights[1]},
                "fit_samples": len(fit_records), "validation_samples": len(validation_records),
                "loss": "weighted cross_entropy(mean)", "grad_clip_global_norm": MAX_GRAD_NORM,
                "full_batch": True, "shuffle": False, "early_stopping": False,
                "device": "cpu", "synthetic_optimizer_steps": 0,
            },
            "parameter_scope": parameter_scope,
            "parameter_chain": {
                "r44_fresh_ep0_sha256": initial_hash,
                "post_bc_sha256": final_hash,
                "task_head_own_parameters_unchanged": True,
                "task_head_sha256": task_head_after,
                "cost_head_own_parameters_unchanged": True,
                "cost_head_sha256": cost_head_after,
            },
            "optimizer_state": {
                "parameter_objects_with_state": len(optimizer.state),
                "step_values": sorted(set(optimizer_steps)),
                "optimizer_state_sha256": _state_hash(optimizer.state_dict()),
                "carried_to_ppo": False,
            },
            "initial_metrics": start_metrics,
            "final_metrics": final_metrics,
            "step_history": step_history,
            "step_journal": {"path": str(journal_path), "sha256": sha256_file(journal_path),
                             "invoked": step_index, "returned": step_index},
        }
        _write_exclusive(metrics_path, result)
        payload = {
            "kind": "round45-bc-post-model/1", "seed": seed,
            "bc_version": BC_VERSION, "ppo_training_version": PPO_TRAINING_VERSION,
            "dataset_sha256": dataset_sha256,
            "r44_fresh_ep0_sha256": initial_hash,
            "post_bc_parameter_sha256": final_hash,
            "controller": {name: value.detach().cpu().clone()
                           for name, value in controller.state_dict().items()},
            "bc_optimizer": copy.deepcopy(optimizer.state_dict()),
            "bc_updates": BC_UPDATES,
            "task_head_sha256": task_head_after,
            "cost_head_sha256": cost_head_after,
            "metrics_sha256": sha256_file(metrics_path),
            "parameter_scope": parameter_scope,
        }
        with checkpoint_path.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        return {
            "seed": seed, "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "metrics_path": str(metrics_path), "metrics_sha256": sha256_file(metrics_path),
            "parameter_sha256": final_hash, "optimizer_steps": step_index,
        }
    finally:
        _restore_rng(ambient_rng)


def _state_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item, key=lambda part: repr(part)):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            digest.update(type(item).__name__.encode("utf-8"))
            digest.update(repr(item).encode("utf-8"))
            digest.update(b";")

    visit(value)
    return digest.hexdigest()


def main() -> int:
    records, audit, dataset_hash = load_verified_dataset()
    ep0 = _expected_ep0_identities()
    for seed in SEEDS:
        if EXPECTED_EP0_PARAMETER_SHA256[seed] != ep0[seed][
                "registered_and_fresh_parameter_sha256"]:
            raise RuntimeError(f"seed {seed}: source ep0 identity differs from R44 record")
    v2_controller = configure_source_imports()
    entries = []
    for seed in SEEDS:
        entries.append(train_seed(seed, records, dataset_sha256=dataset_hash,
                                  expected_ep0=ep0))
        print(json.dumps(entries[-1], ensure_ascii=False), flush=True)
    result = {
        "kind": "round45-bc-training-summary/1", "status": "complete",
        "bc_version": BC_VERSION, "ppo_training_version": PPO_TRAINING_VERSION,
        "dataset_sha256": dataset_hash, "audit_sha256": sha256_file(AUDIT_PATH),
        "controller_module": str(Path(v2_controller.__file__).resolve()),
        "per_seed": entries, "supervised_optimizer_steps": sum(
            int(entry["optimizer_steps"]) for entry in entries),
        "synthetic_optimizer_steps": 0,
    }
    _write_exclusive(RUN_DIR / "BC_TRAINING_SUMMARY.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
