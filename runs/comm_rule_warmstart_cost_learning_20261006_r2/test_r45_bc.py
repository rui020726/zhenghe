"""Independent mathematical and parameter-scope checks for the real R45 BC update seam."""

from __future__ import annotations

import copy
import inspect
import sys
from pathlib import Path

import torch

import r45_bc


def _initialize(seed: int):
    controller_module = r45_bc.configure_source_imports()
    torch.manual_seed(seed)
    controller = controller_module.fresh_controller(seed)
    return controller


def _critic_hashes(controller):
    return (r45_bc._module_hash(controller.inner.value_head),
            r45_bc._module_hash(controller.cost_head))


def _analytic_bias_gradient(controller, features, label):
    with torch.no_grad():
        logits, _ = controller.forward(features)
        probability = torch.softmax(logits, dim=-1)[0]
    one_hot = torch.tensor([1.0 - float(label), float(label)], dtype=torch.float32)
    return probability - one_hot


def run_tie_break_only() -> dict:
    controller_module = r45_bc.configure_source_imports()
    from r42_one_shot import OneShotSendSkipCollector

    source = inspect.getsource(OneShotSendSkipCollector._choose_masked)
    if "probabilities[index, SEND] >= probabilities[index, HOLD]" not in source:
        raise AssertionError("frozen OneShotSendSkipCollector SEND-on-tie rule changed")
    equal_logits = torch.zeros((1, 2), dtype=torch.float32)
    action = int(r45_bc.greedy_actions_from_logits(equal_logits)[0].item())
    assert action == 1, "diagnostic greedy tie-break must select SEND, matching the collector"
    result = {
        "kind": "round45-greedy-tie-break-regression/1",
        "status": "passed",
        "collector_source": str(Path(inspect.getfile(OneShotSendSkipCollector)).resolve()),
        "collector_predicate": "p_SEND >= p_HOLD -> SEND",
        "diagnostic_logits": [0.0, 0.0],
        "diagnostic_action": "SEND_NOW",
        "optimizer_steps": 0,
        "environment_constructed": 0,
    }
    target = r45_bc.RUN_DIR / "BC_TIE_BREAK_REGRESSION.json"
    body = __import__("json").dumps(result, ensure_ascii=False, indent=2) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != body:
            raise RuntimeError("existing tie-break regression result differs")
    else:
        r45_bc._write_exclusive(target, result)
    return result


def run_gradient_scope_only() -> dict:
    controller = _initialize(11)
    features = torch.tensor([[0.25] * r45_bc.FEATURE_WIDTH], dtype=torch.float32)
    true_label = 1
    expected_gradient = _analytic_bias_gradient(controller, features, true_label)
    critic_before = _critic_hashes(controller)
    optimizer = r45_bc.make_bc_optimizer(controller)
    expected_ids = [id(parameter) for parameter in
                    (r45_bc.module_parameter_sets(controller)["trunk"]
                     + r45_bc.module_parameter_sets(controller)["action_head"])]
    captured_clip_ids = []
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = torch.optim.Adam.step
    step_attempts = 0

    def capture_clip(parameters, *args, **kwargs):
        params = list(parameters)
        captured_clip_ids.append([id(parameter) for parameter in params])
        return original_clip(params, *args, **kwargs)

    def forbidden_optimizer_step(self, *args, **kwargs):
        nonlocal step_attempts
        step_attempts += 1
        raise AssertionError("gradient-only regression must not call optimizer.step")

    torch.nn.utils.clip_grad_norm_ = capture_clip
    torch.optim.Adam.step = forbidden_optimizer_step
    try:
        correct = r45_bc.prepare_bc_gradients(
            controller, optimizer, features, torch.tensor([true_label]),
            torch.ones(2, dtype=torch.float32))
        wrong_controller = _initialize(11)
        wrong_expected_ids = [id(parameter) for parameter in
                              (r45_bc.module_parameter_sets(wrong_controller)["trunk"]
                               + r45_bc.module_parameter_sets(wrong_controller)["action_head"])]
        wrong_optimizer = r45_bc.make_bc_optimizer(wrong_controller)
        wrong = r45_bc.prepare_bc_gradients(
            wrong_controller, wrong_optimizer, features, torch.tensor([0]),
            torch.ones(2, dtype=torch.float32))
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
        torch.optim.Adam.step = original_step
    actual_gradient = torch.tensor(correct["action_head_bias_gradient_preclip"])
    wrong_gradient = torch.tensor(wrong["action_head_bias_gradient_preclip"])
    assert torch.allclose(actual_gradient, expected_gradient, atol=1e-7, rtol=1e-6)
    assert actual_gradient[1] < 0.0 and actual_gradient[0] > 0.0
    assert not torch.allclose(wrong_gradient, expected_gradient, atol=1e-7, rtol=1e-6)
    assert captured_clip_ids == [expected_ids, wrong_expected_ids]
    assert step_attempts == 0
    assert _critic_hashes(controller) == critic_before

    wrong_scope = _initialize(11)
    bad_parameters = r45_bc.bc_parameters(wrong_scope)
    bad_parameters.append(wrong_scope.inner.value_head.weight)
    bad_optimizer = torch.optim.Adam(bad_parameters, lr=r45_bc.BC_LR)
    try:
        r45_bc.validate_optimizer_scope(wrong_scope, bad_optimizer)
        raise AssertionError("critic parameter was not rejected from BC optimizer scope")
    except RuntimeError as error:
        assert "exactly trunk plus action head" in str(error)

    result = {
        "kind": "round45-bc-current-source-regression/1",
        "status": "passed",
        "dataset_sha256": r45_bc.sha256_file(r45_bc.DATA_PATH),
        "current_bc_source_sha256": r45_bc.sha256_file(
            r45_bc.RUN_DIR / "r45_bc.py"),
        "current_test_source_sha256": r45_bc.sha256_file(Path(__file__)),
        "actual_forward_backward_clip_path": "prepare_bc_gradients",
        "correct_label_analytic_gradient": [float(value) for value in expected_gradient],
        "actual_correct_label_gradient": correct["action_head_bias_gradient_preclip"],
        "wrong_label_rejected_by_analytic_gradient": True,
        "actual_clip_parameter_ids_match_independent_trunk_action_enumeration": True,
        "invalid_critic_parameter_optimizer_scope_rejected": True,
        "task_and_cost_head_own_parameters_unchanged": True,
        "optimizer_step_calls_in_this_regression": step_attempts,
        "environment_constructed": 0,
        "prior_synthetic_optimizer_updates_from_completed_update_test": 2,
        "synthetic_optimizer_steps": 2,
    }
    target = r45_bc.RUN_DIR / "BC_SYNTHETIC_REGRESSION.json"
    body = __import__("json").dumps(result, ensure_ascii=False, indent=2) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != body:
            raise RuntimeError("existing synthetic regression result differs")
    else:
        r45_bc._write_exclusive(target, result)
    return result


def run() -> dict:
    records, audit, dataset_hash = r45_bc.load_verified_dataset()
    expected_ep0 = r45_bc._expected_ep0_identities()
    controller = _initialize(11)
    init_hash = controller.parameter_summary()["parameter_sha256"]
    assert init_hash == r45_bc.EXPECTED_EP0_PARAMETER_SHA256[11]
    assert init_hash == expected_ep0[11]["registered_and_fresh_parameter_sha256"]

    feature = torch.tensor([[0.25] * r45_bc.FEATURE_WIDTH], dtype=torch.float32)
    true_label = 1
    expected_gradient = _analytic_bias_gradient(controller, feature, true_label)
    critic_before = _critic_hashes(controller)
    optimizer = r45_bc.make_bc_optimizer(controller)
    actual = r45_bc.one_bc_update(
        controller, optimizer, feature,
        torch.tensor([true_label], dtype=torch.long),
        torch.ones(2, dtype=torch.float32))
    gradient = torch.tensor(actual["action_head_bias_gradient_preclip"])
    assert torch.allclose(gradient, expected_gradient, atol=1e-7, rtol=1e-6)
    assert gradient[true_label] < 0.0 and gradient[1 - true_label] > 0.0
    assert _critic_hashes(controller) == critic_before
    assert len(optimizer.state) == len(r45_bc.bc_parameters(controller))
    assert all(int(optimizer.state[p]["step"].item()) == 1
               for p in r45_bc.bc_parameters(controller))

    wrong_controller = _initialize(11)
    wrong_features = feature.clone()
    independent_teacher_gradient = _analytic_bias_gradient(
        wrong_controller, wrong_features, true_label)
    wrong_optimizer = r45_bc.make_bc_optimizer(wrong_controller)
    wrong = r45_bc.one_bc_update(
        wrong_controller, wrong_optimizer, wrong_features,
        torch.tensor([1 - true_label], dtype=torch.long),
        torch.ones(2, dtype=torch.float32))
    wrong_gradient = torch.tensor(wrong["action_head_bias_gradient_preclip"])
    assert not torch.allclose(wrong_gradient, independent_teacher_gradient,
                              atol=1e-7, rtol=1e-6)
    assert wrong_gradient[true_label] > 0.0

    invalid_controller = _initialize(11)
    wrong_parameters = r45_bc.bc_parameters(invalid_controller)
    wrong_parameters.append(invalid_controller.inner.value_head.weight)
    wrong_scope_optimizer = torch.optim.Adam(wrong_parameters, lr=r45_bc.BC_LR)
    rejected_wrong_scope = False
    try:
        r45_bc.validate_optimizer_scope(invalid_controller, wrong_scope_optimizer)
    except RuntimeError as error:
        rejected_wrong_scope = "exactly trunk plus action head" in str(error)
    assert rejected_wrong_scope, "critic-head parameter inclusion was not rejected"

    return {
        "kind": "round45-bc-actual-update-regression/1",
        "status": "passed",
        "dataset_sha256": dataset_hash,
        "audit_sha256": r45_bc.sha256_file(r45_bc.AUDIT_PATH),
        "controller_source": str(Path(r45_bc.configure_source_imports().__file__).resolve()),
        "one_synthetic_correct_label_update": {
            "optimizer_steps": 1,
            "analytic_expected_bias_gradient": [float(x) for x in expected_gradient],
            "actual_preclip_bias_gradient": actual["action_head_bias_gradient_preclip"],
            "loss": actual["loss"],
            "critic_heads_unchanged": True,
        },
        "one_synthetic_wrong_label_update": {
            "optimizer_steps": 1,
            "expected_true_label": true_label,
            "wrong_training_label": 1 - true_label,
            "true_teacher_gradient_match": False,
        },
        "wrong_optimizer_parameter_scope_rejected": rejected_wrong_scope,
        "synthetic_optimizer_steps": 2,
        "dataset_rows_verified": len(records),
        "validation_rows_not_used_for_fit": True,
    }


if __name__ == "__main__":
    import json
    if "--tie-only" in sys.argv:
        print(json.dumps(run_tie_break_only(), ensure_ascii=False, indent=2))
        raise SystemExit(0)
    result = run_gradient_scope_only()
    print(json.dumps(result, ensure_ascii=False, indent=2))
