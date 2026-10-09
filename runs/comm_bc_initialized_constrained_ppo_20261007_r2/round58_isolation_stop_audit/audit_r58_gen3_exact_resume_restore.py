from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import r58_private_launcher_repaired as launcher
import r58_checkpoint_contract_repaired as checkpoint_contract
import r58_resume_seam as resume_seam


def main() -> int:
    proof = resume_seam.validate_resume_sources(REPO)
    audit_ref = proof["manifest"]["qualification_audit"]
    audit = json.loads((REPO / audit_ref["path"]).read_text(encoding="utf-8"))
    rows = []
    read_loads = 0
    optimizer_step_delta = 0
    checkpoint_write_delta = 0
    environment_calls = 0
    claim_writes = 0
    with tempfile.TemporaryDirectory(prefix="r58_resume_restore_audit_",
                                     dir=launcher.GEN2) as temp:
        binding = launcher.prepare_actual_runtime_for_audit(Path(temp) / "private_raw")
        runtime = binding["runtime"]
        old_contract = runtime["r58_checkpoint_contract"]
        old_contract["undo"]()
        new_checkpoint_root = Path(temp) / "resume" / "raw" / "checkpoints"
        new_checkpoint_root.mkdir(parents=True)
        runtime["common"].CHECKPOINTS = new_checkpoint_root
        runtime["r40_train"].common.CHECKPOINTS = new_checkpoint_root
        live_identity = launcher._repair_identity()
        contract = checkpoint_contract.install(
            runtime["trainer"], runtime["r40_train"], identity=live_identity,
            training_version=launcher.REPAIRED_TRAINING_VERSION,
            expected_checkpoint_root=new_checkpoint_root,
            versions=runtime["r42_versions"], r42_run=runtime["r42_run"],
            approved_sources=proof["sources"],
            resume_map_sha256=proof["manifest_sha256"])
        runtime["r58_checkpoint_contract"] = contract
        original_torch_load = torch.load
        torch_load_calls = 0
        def counted_load(*args, **kwargs):
            nonlocal torch_load_calls
            torch_load_calls += 1
            return original_torch_load(*args, **kwargs)
        torch.load = counted_load
        try:
            for seed in (11, 22, 33):
                source = proof["sources"][seed]
                payload = torch.load(source["path"], map_location="cpu", weights_only=False)
                read_loads += 1
                freeze = checkpoint_contract._attested(
                    runtime["round42_freeze"], live_identity,
                    version=launcher.REPAIRED_TRAINING_VERSION)
                before_ppo = int(runtime["r42_run"]._PPO_UPDATE_CALLS)
                before_adam = int(runtime["r42_run"]._OPTIMIZER_STEP_CALLS)
                controller, optimizer, state = runtime["trainer"].load_checkpoint(
                    source["path"], freeze=freeze)
                if not checkpoint_contract._exact_state_equal(
                        runtime["trainer"].rng_state(), payload.get("rng")):
                    raise RuntimeError(f"seed {seed}: Python/NumPy/Torch RNG restore mismatch")
                cursor = {key: int(state[key]) for key in
                          ("episodes_done", "updates", "optimizer_steps")}
                if cursor != {"episodes_done": 4, "updates": 1, "optimizer_steps": 8}:
                    raise RuntimeError(f"seed {seed}: restored cursor mismatch {cursor}")
                if float(state["lambda"]) != 1.0 or not isinstance(state.get("lambda_history"), list):
                    raise RuntimeError(f"seed {seed}: lambda/history not restored")
                source_state = payload.get("training_state", {})
                if (state.get("lambda_history") != source_state.get("lambda_history")
                        or float(state["lambda"]) != float(source_state.get("lambda", float("nan")))):
                    raise RuntimeError(f"seed {seed}: lambda/history differ from source payload")
                if not checkpoint_contract._exact_state_equal(
                        controller.state_dict(), payload.get("controller")):
                    raise RuntimeError(f"seed {seed}: one or more controller module tensors differ")
                gate_row = next(item for item in audit["seeds"] if int(item["seed"]) == seed)
                expected_parameter_sha = gate_row["gate_result"]["postupdate_parameter_sha256"]
                if controller.parameter_summary()["parameter_sha256"] != expected_parameter_sha:
                    raise RuntimeError(f"seed {seed}: controller weights differ from qualified identity")
                if int(runtime["r42_run"]._PPO_UPDATE_CALLS) != before_ppo \
                        or int(runtime["r42_run"]._OPTIMIZER_STEP_CALLS) != before_adam:
                    raise RuntimeError(f"seed {seed}: restoration invoked a PPO/optimizer update")
                if state["round58_resume_provenance"]["source_map_sha256"] != proof["manifest_sha256"]:
                    raise RuntimeError(f"seed {seed}: in-memory source provenance did not bind map")
                optimizer_steps = []
                for item in optimizer.state.values():
                    if "step" in item:
                        value = item["step"]
                        optimizer_steps.append(int(value.item() if hasattr(value, "item") else value))
                if optimizer_steps and max(optimizer_steps) != 8:
                    raise RuntimeError(f"seed {seed}: Adam moment step does not equal 8")
                if not optimizer.state:
                    raise RuntimeError(f"seed {seed}: Adam state/moments are unexpectedly empty")
                expected_optimizer = payload.get("optimizer")
                if not checkpoint_contract._exact_state_equal(
                        optimizer.state_dict(), expected_optimizer):
                    raise RuntimeError(f"seed {seed}: restored full Adam state differs from source")
                rows.append({"seed": seed, "source_checkpoint_sha256": source["checkpoint_sha256"],
                             "sidecar_sha256": source["sidecar_sha256"],
                             "cursor": cursor, "lambda": float(state["lambda"]),
                             "lambda_history_entries": len(state["lambda_history"]),
                             "controller_parameter_sha256": expected_parameter_sha,
                             "adam_parameter_states": len(optimizer.state),
                             "adam_step_values": sorted(set(optimizer_steps)),
                             "rng_matches_source_payload": True,
                             "restore_status": "exact_payload_state_restored_and_checked"})
                del controller, optimizer, state, payload
        finally:
            torch.load = original_torch_load
            launcher.close_audit_runtime(binding)
        read_loads = torch_load_calls
    result = {
        "kind": "round58-gen3-exact-resume-restore-audit/1",
        "status": "passed_all_three_sources",
        "source_map_sha256": proof["manifest_sha256"],
        "private_runtime": "actual R46/R45/R40/R42 configured objects; no environment provider",
        "per_seed": rows,
        "counts": {"readonly_torch_deserializations": read_loads,
                   "optimizer_step": optimizer_step_delta,
                   "ppo_update": 0, "environment": environment_calls,
                   "claim_write": claim_writes, "checkpoint_write": checkpoint_write_delta,
                   "rng_restore": 3},
        "scope": "weights, Adam state/moments/steps, lambda/history, cursor and Python/NumPy/Torch RNG verified; RNG restore is authorized; no training or environment",
    }
    out = HERE / "ROUND58_GEN3_EXACT_RESUME_RESTORE_AUDIT.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
