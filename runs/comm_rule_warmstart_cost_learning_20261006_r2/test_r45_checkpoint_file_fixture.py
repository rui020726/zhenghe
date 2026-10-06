"""Zero-update real-path fixture for the production BC checkpoint seam."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

import r45_bc
from test_r45_checkpoint_serialization import _forward, _module_hashes, _rng_equal


def run() -> dict[str, Any]:
    v2_controller = r45_bc.configure_source_imports()
    ambient = r45_bc._rng_snapshot()
    fixture_dir = r45_bc.RUN_DIR / "serialization_fixtures"
    if fixture_dir.exists():
        raise RuntimeError("serializer fixture directory exists; refusing to overwrite")
    fixture_dir.mkdir(parents=True)
    fixture_path = fixture_dir / "SYNTHETIC_NOT_REAL_BC_CHECKPOINT.fixture.pt"
    try:
        controller = v2_controller.fresh_controller(11)
        parameter_sha_before = controller.parameter_summary()["parameter_sha256"]
        if parameter_sha_before != r45_bc.EXPECTED_EP0_PARAMETER_SHA256[11]:
            raise AssertionError("fixture controller differs from the registered fresh ep0")
        module_sha_before = _module_hashes(controller)
        live_state = controller.state_dict()

        old_failure_reproduced = False
        old_failure = None
        try:
            {name: value.detach().cpu().clone()
             for name, value in controller.state_dict().items()}
        except AttributeError as error:
            old_failure_reproduced = True
            old_failure = str(error)
        if not old_failure_reproduced:
            raise AssertionError("old frozen comprehension unexpectedly accepted nested module state")

        optimizer = r45_bc.make_bc_optimizer(controller)
        bc_parameters = r45_bc.bc_parameters(controller)
        for index, parameter in enumerate(bc_parameters, start=1):
            optimizer.state[parameter] = {
                "step": torch.tensor(7.0, dtype=torch.float32),
                "exp_avg": torch.full_like(parameter, index / 100.0),
                "exp_avg_sq": torch.full_like(parameter, index / 50.0),
            }
        adam_state_before = copy.deepcopy(optimizer.state_dict())
        if (len(adam_state_before["state"]) != len(bc_parameters)
                or any(set(state) != {"step", "exp_avg", "exp_avg_sq"}
                       for state in adam_state_before["state"].values())):
            raise AssertionError("synthetic Adam slots do not match complete BC optimizer state")

        features = torch.arange(38, dtype=torch.float32).reshape(2, 19) / 20.0
        with torch.no_grad():
            expected_forward = _forward(controller, features)
        rng_before_io = r45_bc._rng_snapshot()

        payload = r45_bc.build_bc_checkpoint_payload(controller, optimizer, {
            "kind": "round45-bc-post-model/1",
            "fixture_only": True,
            "artifact_role": "synthetic_schema_fixture_not_a_real_bc_checkpoint",
            "training_execution": "none",
        })
        r45_bc.write_bc_checkpoint(fixture_path, payload)
        loaded_payload = r45_bc.read_bc_checkpoint(fixture_path)
        if (loaded_payload.get("fixture_only") is not True
                or loaded_payload.get("artifact_role")
                != "synthetic_schema_fixture_not_a_real_bc_checkpoint"):
            raise AssertionError("on-disk fixture lacks its synthetic-only marker")

        restored = v2_controller.fresh_controller(11)
        r45_bc._restore_rng(rng_before_io)
        r45_bc.load_bc_controller_state(restored, loaded_payload)
        module_sha_after = _module_hashes(restored)
        if module_sha_after != module_sha_before:
            raise AssertionError("four module hashes differ after path save/read/load")
        with torch.no_grad():
            actual_forward = _forward(restored, features)
        if not all(torch.equal(a, b) for a, b in zip(expected_forward, actual_forward)):
            raise AssertionError("actor/task/cost forward differs after path round-trip")

        adam_state_after = loaded_payload["bc_optimizer"]
        before_adam_hash = r45_bc._state_hash(adam_state_before)
        after_adam_hash = r45_bc._state_hash(adam_state_after)
        if (before_adam_hash != after_adam_hash
                or set(adam_state_after) != {"state", "param_groups"}
                or len(adam_state_after["state"]) != len(bc_parameters)
                or len(adam_state_after["param_groups"]) != 1
                or len(adam_state_after["param_groups"][0]["params"]) != len(bc_parameters)):
            raise AssertionError("full BC Adam state did not survive path round-trip")

        rng_after_io = r45_bc._rng_snapshot()
        if not _rng_equal(rng_before_io, rng_after_io):
            raise AssertionError("save/read/load path changed RNG state")
        if controller.parameter_summary()["parameter_sha256"] != parameter_sha_before:
            raise AssertionError("save/read path changed source controller parameters")

        result = {
            "kind": "round45-r2-file-path-checkpoint-regression/1",
            "status": "passed",
            "repair_version": r45_bc.BC_CHECKPOINT_SERIALIZATION_REPAIR_VERSION,
            "fixture_only_not_a_real_bc_checkpoint": True,
            "old_frozen_comprehension_fails": old_failure_reproduced,
            "old_failure": old_failure,
            "production_functions": [
                "build_bc_checkpoint_payload", "write_bc_checkpoint(path)",
                "read_bc_checkpoint(path)", "load_bc_controller_state",
            ],
            "fixture_path": str(fixture_path.resolve()),
            "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "fixture_size_bytes": fixture_path.stat().st_size,
            "controller_state_tensor_count": sum(len(module) for module in live_state.values()),
            "parameter_sha256_before": parameter_sha_before,
            "parameter_sha256_after": restored.parameter_summary()["parameter_sha256"],
            "module_parameter_sha256_before": module_sha_before,
            "module_parameter_sha256_after": module_sha_after,
            "actor_task_cost_forward_equal": True,
            "adam_state_entries": len(adam_state_after["state"]),
            "adam_slot_keys": ["step", "exp_avg", "exp_avg_sq"],
            "adam_state_hash_before": before_adam_hash,
            "adam_state_hash_after": after_adam_hash,
            "rng_unchanged": True,
            "optimizer_step_calls": 0,
            "synthetic_optimizer_updates": 0,
            "environment_constructed": 0,
            "checkpoint_claimed_as_experiment_model": False,
        }
        result["test_source_sha256"] = r45_bc.sha256_file(Path(__file__))
        result["production_source_sha256"] = r45_bc.sha256_file(Path(r45_bc.__file__))
        r45_bc._write_exclusive(
            r45_bc.RUN_DIR / "ROUND45_R2_FILE_PATH_SERIALIZATION_REGRESSION.json", result)
        return result
    finally:
        r45_bc._restore_rng(ambient)


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
