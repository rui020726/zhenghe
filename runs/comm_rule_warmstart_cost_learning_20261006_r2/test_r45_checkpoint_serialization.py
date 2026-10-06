"""BytesIO-only regression for the production nested BC checkpoint seam."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import r45_bc


def _rng_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["python"] != right["python"]:
        return False
    left_np, right_np = left["numpy"], right["numpy"]
    if (left_np[0] != right_np[0] or left_np[2:] != right_np[2:]
            or not np.array_equal(left_np[1], right_np[1])):
        return False
    if not torch.equal(left["torch"], right["torch"]):
        return False
    if left["cuda"] is None or right["cuda"] is None:
        return left["cuda"] is None and right["cuda"] is None
    return (len(left["cuda"]) == len(right["cuda"])
            and all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"])))


def _module_hashes(controller: Any) -> dict[str, str]:
    return {
        "trunk": r45_bc._module_hash(controller.inner.trunk),
        "action_head": r45_bc._module_hash(controller.inner.action_head),
        "task_value_head": r45_bc._module_hash(controller.inner.value_head),
        "cost_value_head": r45_bc._module_hash(controller.cost_head),
    }


def _forward(controller: Any, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
    action_logits, task_values = controller.forward(features)
    cost_values = controller.cost_values(features)
    return action_logits.detach().clone(), task_values.detach().clone(), cost_values.detach().clone()


def run() -> dict[str, Any]:
    v2_controller = r45_bc.configure_source_imports()
    ambient = r45_bc._rng_snapshot()
    try:
        controller = v2_controller.fresh_controller(11)
        before_parameter_hash = controller.parameter_summary()["parameter_sha256"]
        before_module_hashes = _module_hashes(controller)
        live_state = controller.state_dict()

        old_bug_reproduced = False
        old_bug_message = None
        try:
            {name: value.detach().cpu().clone()
             for name, value in controller.state_dict().items()}
        except AttributeError as error:
            old_bug_reproduced = True
            old_bug_message = str(error)
        if not old_bug_reproduced:
            raise AssertionError("frozen serializer expression did not fail on actual controller state")

        # Build a complete Adam state-shaped fixture without invoking optimizer.step().
        optimizer = r45_bc.make_bc_optimizer(controller)
        parameters = r45_bc.bc_parameters(controller)
        for index, parameter in enumerate(parameters, start=1):
            optimizer.state[parameter] = {
                "step": torch.tensor(7.0, dtype=torch.float32),
                "exp_avg": torch.full_like(parameter, index / 100.0),
                "exp_avg_sq": torch.full_like(parameter, index / 50.0),
            }
        optimizer_fixture = copy.deepcopy(optimizer.state_dict())
        if len(optimizer_fixture["state"]) != len(parameters):
            raise AssertionError("Adam state fixture does not cover all BC optimizer parameters")
        if any(set(state) != {"step", "exp_avg", "exp_avg_sq"}
               for state in optimizer_fixture["state"].values()):
            raise AssertionError("Adam state fixture does not contain the complete non-AMSGrad slots")

        features = torch.arange(38, dtype=torch.float32).reshape(2, 19) / 20.0
        with torch.no_grad():
            expected_forward = _forward(controller, features)
        rng_before_serialization = r45_bc._rng_snapshot()

        metadata = {
            "kind": "round45-bc-post-model/1",
            "fixture_only": True,
            "artifact_role": "synthetic_schema_fixture_not_a_real_bc_checkpoint",
            "training_execution": "none",
        }
        payload = r45_bc.build_bc_checkpoint_payload(controller, optimizer, metadata)
        if (r45_bc._state_hash(payload["bc_optimizer"])
                != r45_bc._state_hash(optimizer_fixture)):
            raise AssertionError("production payload builder changed the nested Adam state")

        buffer = io.BytesIO()
        r45_bc.write_bc_checkpoint(buffer, payload)
        serialized_bytes = len(buffer.getbuffer())
        buffer.seek(0)
        loaded_payload = r45_bc.read_bc_checkpoint(buffer)
        if (loaded_payload.get("fixture_only") is not True
                or loaded_payload.get("artifact_role")
                != "synthetic_schema_fixture_not_a_real_bc_checkpoint"):
            raise AssertionError("round-trip fixture was not explicitly marked synthetic")

        restored_controller = v2_controller.fresh_controller(11)
        r45_bc._restore_rng(rng_before_serialization)
        r45_bc.load_bc_controller_state(restored_controller, loaded_payload)
        after_module_hashes = _module_hashes(restored_controller)
        if after_module_hashes != before_module_hashes:
            raise AssertionError("four-module parameter hashes changed in save/load round-trip")
        with torch.no_grad():
            actual_forward = _forward(restored_controller, features)
        if not all(torch.equal(expected, actual)
                   for expected, actual in zip(expected_forward, actual_forward)):
            raise AssertionError("save/load round-trip changed actor/task/cost forward outputs")

        loaded_optimizer = loaded_payload["bc_optimizer"]
        if r45_bc._state_hash(loaded_optimizer) != r45_bc._state_hash(optimizer_fixture):
            raise AssertionError("save/load round-trip flattened or changed Adam state")
        if (set(loaded_optimizer) != {"state", "param_groups"}
                or len(loaded_optimizer["state"]) != len(parameters)
                or len(loaded_optimizer["param_groups"]) != 1
                or len(loaded_optimizer["param_groups"][0]["params"]) != len(parameters)):
            raise AssertionError("round-trip Adam structure does not match frozen BC optimizer scope")

        rng_after_serialization = r45_bc._rng_snapshot()
        if not _rng_equal(rng_before_serialization, rng_after_serialization):
            raise AssertionError("production save/read/load functions changed RNG state")
        if controller.parameter_summary()["parameter_sha256"] != before_parameter_hash:
            raise AssertionError("production save/read functions changed source controller parameters")

        result = {
            "kind": "round45-bc-checkpoint-serialization-regression/2",
            "status": "passed",
            "repair_version": r45_bc.BC_CHECKPOINT_SERIALIZATION_REPAIR_VERSION,
            "production_functions": [
                "build_bc_checkpoint_payload", "write_bc_checkpoint",
                "read_bc_checkpoint", "load_bc_controller_state",
            ],
            "roundtrip_transport": "io.BytesIO; no checkpoint file created",
            "fixture_only_not_a_real_bc_checkpoint": True,
            "old_frozen_comprehension_fails_on_actual_controller": old_bug_reproduced,
            "old_failure": old_bug_message,
            "module_parameter_sha256_before": before_module_hashes,
            "module_parameter_sha256_after": after_module_hashes,
            "actor_task_cost_forward_equal": True,
            "nested_state_tensor_count": sum(len(value) for value in live_state.values()),
            "adam_state_entries": len(loaded_optimizer["state"]),
            "adam_slot_keys": ["step", "exp_avg", "exp_avg_sq"],
            "adam_state_hash_before": r45_bc._state_hash(optimizer_fixture),
            "adam_state_hash_after": r45_bc._state_hash(loaded_optimizer),
            "serialized_bytes": serialized_bytes,
            "rng_unchanged": True,
            "optimizer_step_calls": 0,
            "synthetic_optimizer_updates": 0,
            "environment_constructed": 0,
        }
        target = r45_bc.RUN_DIR / "ROUND45_CHECKPOINT_SERIALIZATION_REGRESSION_R2.json"
        r45_bc._write_exclusive(target, result)
        return result
    finally:
        r45_bc._restore_rng(ambient)


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
