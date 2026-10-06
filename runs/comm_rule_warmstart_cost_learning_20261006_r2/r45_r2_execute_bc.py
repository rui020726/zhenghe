"""ROUND45-R thin attempt wrapper around the unchanged frozen BC trainer."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

import r45_bc


RUN_DIR = r45_bc.RUN_DIR
ATTEMPT = "ROUND45-R/fresh-seed-attempt/2"
EXPECTED_BRANCH = "exp/comm-rule-warmstart-cost-learning-r2"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(r45_bc.ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_committed_freeze(path: Path, freeze: dict[str, Any]) -> None:
    relative = f"runs/{RUN_DIR.name}/ROUND45_PRE_BC_FREEZE.json"
    committed = subprocess.run(
        ["git", "-C", str(r45_bc.ROOT), "cat-file", "blob", f"HEAD:{relative}"],
        check=True, capture_output=True).stdout
    if hashlib.sha256(committed).hexdigest() != r45_bc.sha256_file(path):
        raise RuntimeError("R2 pre-BC freeze differs from the committed Git blob")
    source_head = freeze.get("committed_source_head")
    if not isinstance(source_head, str) or not source_head:
        raise RuntimeError("R2 freeze does not name its committed source head")
    ancestry = subprocess.run(
        ["git", "-C", str(r45_bc.ROOT), "merge-base", "--is-ancestor",
         source_head, "HEAD"], capture_output=True)
    if ancestry.returncode != 0:
        raise RuntimeError("R2 frozen source head is not an ancestor of the execution head")


def _verify_all_frozen_sources(freeze: dict[str, Any]) -> dict[str, str]:
    expected = freeze.get("source_files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("R2 freeze has no complete source-file identity map")
    required = {
        "EXECUTION_BASIS_FULL.md", "ROUND45_R_CONTINUATION_BASIS.md",
        "ROUND45_CONFIGURATION_DIFF.json", "ROUND45_R2_REUSE_MANIFEST.json",
        "ROUND45_R2_FILE_PATH_SERIALIZATION_REGRESSION.json",
        "ROUND45_R2_AST_REUSE_RECEIPT.json", "r45_data.py",
        "r45_bc.py", "r45_preflight.py", "r45_r2_execute_bc.py",
        "BC_DATASET.jsonl",
        "serialization_fixtures/SYNTHETIC_NOT_REAL_BC_CHECKPOINT.fixture.pt",
    }
    if not required.issubset(expected):
        raise RuntimeError(f"R2 freeze source map omits required files: {sorted(required - set(expected))}")
    root = RUN_DIR.resolve()
    verified = {}
    for relative, digest in sorted(expected.items()):
        if not isinstance(relative, str) or not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("R2 freeze contains a malformed source identity")
        path = (RUN_DIR / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"frozen source path escapes R2 output: {relative}") from error
        if not path.is_file() or r45_bc.sha256_file(path) != digest:
            raise RuntimeError(f"R2 frozen source changed: {relative}")
        verified[relative] = digest
    return verified


def _verify_runtime_dependencies(freeze: dict[str, Any]) -> dict[str, Any]:
    import r45_preflight

    if Path(r45_bc.__file__).resolve() != (RUN_DIR / "r45_bc.py").resolve():
        raise RuntimeError("runtime imported a different frozen R2 BC trainer")
    if Path(r45_preflight.__file__).resolve() != (RUN_DIR / "r45_preflight.py").resolve():
        raise RuntimeError("runtime imported a different R2 preflight module")
    v2_controller = r45_bc.configure_source_imports()
    import sb_controller
    import train_pilot

    actual_controller_sources = {
        Path(module.__file__).name: r45_bc.sha256_file(Path(module.__file__))
        for module in (v2_controller, sb_controller, train_pilot)
    }
    expected_controller_sources = freeze.get("bc", {}).get(
        "controller_source_files_sha256", {})
    if actual_controller_sources != expected_controller_sources:
        raise RuntimeError("runtime canonical controller source hashes differ from freeze")
    actual_r44_identity = r45_preflight.verify_r44_inputs()
    if actual_r44_identity != freeze.get("bc", {}).get("r44_bc_input_identity"):
        raise RuntimeError("runtime R44/R43/R40 dependencies or input matrix differ from freeze")
    return {
        "controller_source_files_sha256": actual_controller_sources,
        "r44_bc_input_identity_sha256": hashlib.sha256(json.dumps(
            actual_r44_identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "r44_bc_input_identity": actual_r44_identity,
    }


def _module_hashes(controller: Any) -> dict[str, str]:
    return {
        "trunk": r45_bc._module_hash(controller.inner.trunk),
        "action_head": r45_bc._module_hash(controller.inner.action_head),
        "task_value_head": r45_bc._module_hash(controller.inner.value_head),
        "cost_value_head": r45_bc._module_hash(controller.cost_head),
    }


def _forward_fingerprints(controller: Any) -> dict[str, str]:
    features = torch.arange(57, dtype=torch.float32).reshape(3, 19) / 32.0
    with torch.no_grad():
        action_logits, task_values = controller.forward(features)
        cost_values = controller.cost_values(features)
    values = {
        "action_logits": action_logits,
        "task_values": task_values,
        "cost_values": cost_values,
    }
    return {
        name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in values.items()
    }


def _verify_saved_seed(seed: int, *, dataset_sha256: str) -> dict[str, Any]:
    checkpoint = RUN_DIR / "checkpoints" / f"BC_POST_MODEL_s{seed}.pt"
    metrics_path = RUN_DIR / f"BC_METRICS_s{seed}.json"
    journal_path = RUN_DIR / "bc_journals" / f"BC_OPTIMIZER_STEPS_s{seed}.jsonl"
    receipt_path = RUN_DIR / f"BC_CHECKPOINT_RELOAD_VERIFICATION_s{seed}.json"
    metrics = _json(metrics_path)
    training = metrics.get("training", {})
    journal_metrics = metrics.get("step_journal", {})
    optimizer_metrics = metrics.get("optimizer_state", {})
    parameter_chain = metrics.get("parameter_chain", {})
    if (metrics.get("kind") != "round45-bc-seed-metrics/1"
            or metrics.get("seed") != seed
            or metrics.get("bc_version") != r45_bc.BC_VERSION
            or metrics.get("ppo_training_version") != r45_bc.PPO_TRAINING_VERSION
            or training.get("updates") != r45_bc.BC_UPDATES
            or training.get("synthetic_optimizer_steps") != 0
            or training.get("device") != "cpu"
            or training.get("loss") != "weighted cross_entropy(mean)"
            or training.get("full_batch") is not True
            or training.get("shuffle") is not False
            or journal_metrics.get("invoked") != r45_bc.BC_UPDATES
            or journal_metrics.get("returned") != r45_bc.BC_UPDATES
            or optimizer_metrics.get("carried_to_ppo") is not False
            or optimizer_metrics.get("step_values") != [r45_bc.BC_UPDATES]
            or parameter_chain.get("task_head_own_parameters_unchanged") is not True
            or parameter_chain.get("cost_head_own_parameters_unchanged") is not True):
        raise RuntimeError(f"seed {seed}: production BC metrics schema/value contract differs")
    saved = r45_bc.read_bc_checkpoint(checkpoint)
    file_sha = r45_bc.sha256_file(checkpoint)
    metrics_sha = r45_bc.sha256_file(metrics_path)
    journal_sha = r45_bc.sha256_file(journal_path)
    journal_records = [json.loads(line) for line in journal_path.read_text(
        encoding="utf-8").splitlines() if line]
    invoked = sum(item.get("event") == "optimizer_step_invoked" for item in journal_records)
    returned = sum(item.get("event") == "optimizer_step_returned" for item in journal_records)
    if (saved.get("seed") != seed
            or saved.get("bc_updates") != r45_bc.BC_UPDATES
            or saved.get("bc_version") != r45_bc.BC_VERSION
            or saved.get("ppo_training_version") != r45_bc.PPO_TRAINING_VERSION
            or saved.get("dataset_sha256") != dataset_sha256
            or saved.get("metrics_sha256") != metrics_sha
            or training.get("updates") != r45_bc.BC_UPDATES
            or metrics.get("dataset_sha256") != dataset_sha256
            or journal_metrics.get("sha256") != journal_sha
            or invoked != r45_bc.BC_UPDATES
            or returned != r45_bc.BC_UPDATES
            or len(journal_records) != 2 * r45_bc.BC_UPDATES):
        raise RuntimeError(f"seed {seed}: saved BC model/metrics/journal identity mismatch")
    adam_state = saved["bc_optimizer"]
    if (r45_bc._state_hash(adam_state)
            != optimizer_metrics.get("optimizer_state_sha256")
            or len(adam_state.get("state", {}))
            != optimizer_metrics.get("parameter_objects_with_state")
            or not adam_state.get("param_groups")):
        raise RuntimeError(f"seed {seed}: complete BC Adam state failed the checkpoint readback")
    for item in adam_state["state"].values():
        if set(item) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError(f"seed {seed}: BC Adam state slot schema changed")

    expected_parameter_sha = parameter_chain["post_bc_sha256"]
    if saved.get("post_bc_parameter_sha256") != expected_parameter_sha:
        raise RuntimeError(f"seed {seed}: checkpoint/metrics final parameter digest mismatch")
    ambient = r45_bc._rng_snapshot()
    v2_controller = r45_bc.configure_source_imports()
    try:
        first = v2_controller.fresh_controller(seed)
        r45_bc._restore_rng(ambient)
        r45_bc.load_bc_controller_state(first, saved)
        first_parameter_sha = first.parameter_summary()["parameter_sha256"]
        first_module_sha = _module_hashes(first)
        first_forward = _forward_fingerprints(first)

        second = v2_controller.fresh_controller(seed)
        r45_bc._restore_rng(ambient)
        r45_bc.load_bc_controller_state(second, saved)
        second_parameter_sha = second.parameter_summary()["parameter_sha256"]
        second_module_sha = _module_hashes(second)
        second_forward = _forward_fingerprints(second)
    finally:
        r45_bc._restore_rng(ambient)
    if (first_parameter_sha != expected_parameter_sha
            or second_parameter_sha != expected_parameter_sha
            or first_module_sha != second_module_sha
            or first_forward != second_forward):
        raise RuntimeError(f"seed {seed}: read-only model reload identity/forward mismatch")
    if (first_module_sha["task_value_head"]
            != parameter_chain["task_head_sha256"]
            or first_module_sha["cost_value_head"]
            != parameter_chain["cost_head_sha256"]):
        raise RuntimeError(f"seed {seed}: critic module hashes disagree with BC metrics")

    result = {
        "kind": "round45-r2-bc-checkpoint-reload-verification/1",
        "status": "passed", "attempt": ATTEMPT, "seed": seed,
        "dataset_sha256": dataset_sha256,
        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": file_sha,
        "metrics_sha256": metrics_sha, "journal_sha256": journal_sha,
        "bc_updates_invoked_and_returned": r45_bc.BC_UPDATES,
        "journal_record_count": len(journal_records),
        "post_bc_parameter_sha256": expected_parameter_sha,
        "module_parameter_sha256": first_module_sha,
        "independently_loaded_forward_fingerprints_equal": first_forward,
        "optimizer_state_sha256": r45_bc._state_hash(adam_state),
        "optimizer_state_entries": len(adam_state["state"]),
        "optimizer_slots": ["step", "exp_avg", "exp_avg_sq"],
        "bc_optimizer_restored": False,
        "rng_restored_from_checkpoint": False,
        "environment_constructed": 0,
    }
    if receipt_path.exists():
        if _json(receipt_path) != result:
            raise RuntimeError(f"seed {seed}: existing reload receipt differs; preserve and stop")
    else:
        r45_bc._write_exclusive(receipt_path, result)
    result["readback_receipt_path"] = str(receipt_path.resolve())
    result["readback_receipt_sha256"] = r45_bc.sha256_file(receipt_path)
    return result


def _record_ledger(value: dict[str, Any]) -> None:
    r45_bc._append_fsync(RUN_DIR / "ROUND45_R2_BC_LEDGER.jsonl", value)


def main() -> int:
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("R2 BC runner is on the wrong branch")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("R2 tracked source changed after pre-BC freeze")
    freeze_path = RUN_DIR / "ROUND45_PRE_BC_FREEZE.json"
    freeze = _json(freeze_path)
    if (freeze.get("status") != "passed"
            or freeze.get("branch") != EXPECTED_BRANCH):
        raise RuntimeError("R2 pre-BC source/freeze identity is missing or stale")
    _verify_committed_freeze(freeze_path, freeze)
    verified_source_hashes = _verify_all_frozen_sources(freeze)
    if (freeze.get("source_files_sha256", {}).get("r45_r2_execute_bc.py")
            != r45_bc.sha256_file(Path(__file__))):
        raise RuntimeError("R2 BC runner differs from its committed freeze")
    runtime_identity = _verify_runtime_dependencies(freeze)
    records, audit, dataset_sha = r45_bc.load_verified_dataset()
    if dataset_sha != freeze["bc"]["dataset_sha256"]:
        raise RuntimeError("R2 BC dataset differs from its committed freeze")
    expected_ep0 = r45_bc._expected_ep0_identities()
    v2_controller = r45_bc.configure_source_imports()

    ledger_path = RUN_DIR / "ROUND45_R2_BC_LEDGER.jsonl"
    if not ledger_path.exists():
        with ledger_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({
                "kind": "round45-r2-bc-ledger-header/1", "attempt": ATTEMPT,
                "freeze_sha256": r45_bc.sha256_file(freeze_path),
                "verified_frozen_source_count": len(verified_source_hashes),
                "r44_bc_input_identity_sha256": runtime_identity[
                    "r44_bc_input_identity_sha256"],
                "dataset_sha256": dataset_sha, "branch": EXPECTED_BRANCH,
            }, sort_keys=True) + "\n")
            handle.flush()
            import os
            os.fsync(handle.fileno())
    else:
        lines = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
                 if line]
        if (not lines or lines[0].get("kind") != "round45-r2-bc-ledger-header/1"
                or lines[0].get("attempt") != ATTEMPT
                or lines[0].get("freeze_sha256") != r45_bc.sha256_file(freeze_path)
                or lines[0].get("dataset_sha256") != dataset_sha):
            raise RuntimeError("R2 BC ledger identity is invalid; preserve and stop")

    completed = []
    for seed in r45_bc.SEEDS:
        checkpoint = RUN_DIR / "checkpoints" / f"BC_POST_MODEL_s{seed}.pt"
        metrics_path = RUN_DIR / f"BC_METRICS_s{seed}.json"
        journal_path = RUN_DIR / "bc_journals" / f"BC_OPTIMIZER_STEPS_s{seed}.jsonl"
        readback_path = RUN_DIR / f"BC_CHECKPOINT_RELOAD_VERIFICATION_s{seed}.json"
        outputs = [checkpoint.exists(), metrics_path.exists(), journal_path.exists(),
                   readback_path.exists()]
        if any(outputs):
            if not all(outputs):
                raise RuntimeError(f"seed {seed}: partial R2 BC outputs exist; refusing replay")
            readback = _verify_saved_seed(seed, dataset_sha256=dataset_sha)
            if (readback.get("status") != "passed" or readback.get("seed") != seed
                    or readback.get("checkpoint_sha256") != r45_bc.sha256_file(checkpoint)):
                raise RuntimeError(f"seed {seed}: existing complete BC attempt fails exact reuse gate")
            completed.append({"seed": seed, "optimizer_steps": r45_bc.BC_UPDATES,
                              "checkpoint_sha256": r45_bc.sha256_file(checkpoint),
                              "reused_complete_seed": True})
            continue

        _record_ledger({"kind": "round45-r2-bc-seed-start/1", "attempt": ATTEMPT,
                        "seed": seed, "dataset_sha256": dataset_sha,
                        "expected_updates": r45_bc.BC_UPDATES})
        result = r45_bc.train_seed(seed, records, dataset_sha256=dataset_sha,
                                   expected_ep0=expected_ep0)
        readback = _verify_saved_seed(seed, dataset_sha256=dataset_sha)
        completed.append({"seed": seed, "optimizer_steps": result["optimizer_steps"],
                          "checkpoint_sha256": result["checkpoint_sha256"],
                          "readback_receipt_sha256": readback["readback_receipt_sha256"],
                          "reused_complete_seed": False})
        _record_ledger({"kind": "round45-r2-bc-seed-complete/1", "attempt": ATTEMPT,
                        "seed": seed, "optimizer_steps": result["optimizer_steps"],
                        "checkpoint_sha256": result["checkpoint_sha256"],
                        "readback_receipt_sha256": readback["readback_receipt_sha256"]})
        print(json.dumps({"seed": seed, "stage": "BC complete and readback passed",
                          "optimizer_steps": result["optimizer_steps"],
                          "checkpoint_sha256": result["checkpoint_sha256"]}), flush=True)

    summary = {
        "kind": "round45-r2-bc-training-summary/1", "status": "complete",
        "attempt": ATTEMPT, "branch": EXPECTED_BRANCH,
        "source_head": _git("rev-parse", "HEAD"),
        "freeze_sha256": r45_bc.sha256_file(freeze_path),
        "verified_frozen_source_count": len(verified_source_hashes),
        "runtime_dependency_identity_sha256": runtime_identity[
            "r44_bc_input_identity_sha256"],
        "dataset_sha256": dataset_sha,
        "prior_failed_real_bc_updates_preserved": 1000,
        "this_attempt_real_bc_optimizer_updates": sum(
            int(row["optimizer_steps"]) for row in completed),
        "family_cumulative_real_bc_optimizer_updates": 1000 + sum(
            int(row["optimizer_steps"]) for row in completed),
        "synthetic_optimizer_updates_used_before_r2": 2,
        "environment_constructed": 0,
        "per_seed": completed,
    }
    r45_bc._write_exclusive(RUN_DIR / "ROUND45_R2_BC_TRAINING_SUMMARY.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
