"""Create and verify the committed ROUND42 prelaunch freeze; constructs no SAR environment."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
ROUND42_ROOT = RUN_DIR.parents[1]
WORKSPACE_ROOT = ROUND42_ROOT.parent
INPUT_ROOT = WORKSPACE_ROOT / "zhenghe"
ROUND40_ROOT = WORKSPACE_ROOT / "zhenghe-round40"
ROUND41_ROOT = WORKSPACE_ROOT / "zhenghe-round41"
ROUND40_OUTPUT = ROUND40_ROOT / "runs" / "comm_first_report_training_ablation_20261003"
ROUND41_OUTPUT = ROUND42_ROOT / "runs" / "comm_temporal_decoder_diagnostic_20261003"
PYTHON = Path(r"D:\project_env\python.exe")
ROUND41_HEAD = "cc28a28fc5fb13e31ab9ed0fc3456797aadcc115"
ROUND40_HEAD = "8f8245a665b15960e70194fc6e245221b70db90e"
ROUND42_INITIAL_EXECUTION_HEAD = "3c0b61cbdb702c3e4d8b1a4668030bda2db102ab"
ROUND42_OLD_BASIS_HEAD = "ffa87ceb1a9d48abe27fc124ef1a963470878604"
ROUND42_OLD_DELIVERY_HEAD = "70faab3f451b3c17690c189216c315cccaf4a6d6"
BRANCH = "fix/comm-one-shot-runner-counting"
SEMANTICS = "one-shot-send-skip/1"
RUNNER_VERSION = "counted-jointrun-proxy/1"
REWARD_TIMING = "published-completion-event-at-relay-tick/2"
TRAINING_VERSION = (
    "constrained-cost-v2/one-shot-send-skip/1/"
    "published-completion-event-at-relay-tick/2"
)
RULE_ARMS = ("late_discovery_gate", "always_hold")
R42_CODE_FILES = (
    "r42_one_shot.py", "r42_versions.py", "r42_run.py", "r42_execute.py",
    "r42_analysis.py", "r42_report.py", "r42_preflight.py", "test_r42_one_shot.py",
    "test_r42_runner_counting.py",
)
R41_CODE_FILES = (
    "r41_common.py", "r41_hazard_median.py", "r41_preflight.py", "r41_run.py",
    "test_r41_hazard_median.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _assert(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(detail)


def _dependency_path(root: Path, relative: str, group: str) -> Path:
    if group == "stage_c_harness" and not relative.startswith("runs/"):
        return root / "runs" / "comm_cost_v2_rerun_20260928" / relative
    return root / relative


def _source_paths() -> None:
    for directory in (ROUND40_OUTPUT, INPUT_ROOT):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    module_dirs = [
        INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928",
        INPUT_ROOT / "runs" / "comm_cost_stage_b_20260927",
        INPUT_ROOT / "runs" / "comm_budget_action_mask_fix_20260922",
        INPUT_ROOT / "runs" / "comm_budget_pilot_20260919",
        INPUT_ROOT / "runs" / "comm_cost_controller_stage_a_20260927",
        INPUT_ROOT / "runs" / "comm_tight_budget_20260919",
        INPUT_ROOT / "runs" / "search_trace_and_policy_20260925",
    ]
    for directory in reversed(module_dirs):
        if not directory.is_dir():
            raise FileNotFoundError(f"frozen ROUND40 dependency missing: {directory}")
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    if str(INPUT_ROOT) not in sys.path:
        sys.path.insert(0, str(INPUT_ROOT))


def _checkpoint_forward_gate(r40_frozen: dict[str, Any], reconciliation: dict[str, Any]
                             ) -> dict[str, Any]:
    _source_paths()
    import torch
    import v2_common as common
    import v2_controller
    import v2_run as runner
    import v2_trainer as trainer
    import r40_versions

    active_modules = {
        "v2_common": INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "v2_common.py",
        "v2_controller": INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "v2_controller.py",
        "v2_run": INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "v2_run.py",
        "v2_trainer": INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "v2_trainer.py",
        "v2_reward_timing": INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "v2_reward_timing.py",
        "budget_masked": INPUT_ROOT / "runs" / "comm_budget_action_mask_fix_20260922" / "budget_masked.py",
        "sb_collector": INPUT_ROOT / "runs" / "comm_cost_stage_b_20260927" / "sb_collector.py",
        "onpolicy": INPUT_ROOT / "runs" / "comm_tight_budget_20260919" / "onpolicy.py",
        "cost_math": INPUT_ROOT / "runs" / "comm_cost_controller_stage_a_20260927" / "cost_math.py",
        "r40_versions": ROUND40_OUTPUT / "r40_versions.py",
    }
    for name, expected_path in active_modules.items():
        module = __import__(name)
        _assert(Path(module.__file__).resolve() == expected_path.resolve(),
                f"runtime import provenance mismatch for {name}: {module.__file__}")
    for name in ("comm", "comm_budget", "comm_learning", "comm_policy", "joint_driver",
                 "task_registry"):
        module = __import__(f"integrated_sar.{name}", fromlist=[name])
        _assert(Path(module.__file__).resolve() == (INPUT_ROOT / "integrated_sar" / f"{name}.py").resolve(),
                f"runtime production import provenance mismatch for integrated_sar.{name}: "
                f"{module.__file__}")

    common.DEVICE = torch.device("cpu")
    torch.set_num_threads(1)
    r40_freeze = r40_frozen["round40"]["freeze_identity"]
    r40_versions.install(trainer, runner, freeze=r40_freeze)
    declared = reconciliation["checkpoints"]["NEW_training_outputs"]
    input_hashes: dict[str, Any] = {}
    for seed in (11, 22, 33):
        rows = declared[str(seed)]
        for milestone in ("ep0", "ep128"):
            entry = rows[milestone]
            path = Path(entry["path"])
            _assert(path.is_file(), f"ROUND40 checkpoint is missing: {path}")
            before_sha = _sha256(path)
            _assert(before_sha == entry["file_sha256"],
                    f"ROUND40 {milestone} file SHA mismatch for seed {seed}")
            controller, state, identity = trainer.load_checkpoint_read_only(
                path, freeze=r40_freeze if milestone == "ep0" else None)
            _assert(int(state.get("training_seed", -1)) == seed,
                    f"ROUND40 {milestone} training seed mismatch: {path}")
            _assert(int(state.get("episodes_done", -1)) == entry["episodes_done"],
                    f"ROUND40 {milestone} episode cursor mismatch: {path}")
            _assert(identity.get("load_mode") == "read_only"
                    and identity.get("optimizer_restored") is False
                    and identity.get("rng_restored") is False,
                    f"ROUND40 {milestone} was not loaded read-only: {path}")
            parameter = controller.parameter_summary()
            _assert(parameter["parameter_sha256"] == entry["parameter_sha256"],
                    f"ROUND40 {milestone} parameter identity mismatch for seed {seed}")
            if milestone == "ep0":
                fresh = v2_controller.fresh_controller(seed)
                _assert(fresh.parameter_summary()["parameter_sha256"]
                        == parameter["parameter_sha256"],
                        f"ROUND42 NEW same-seed initialization differs from ep0: seed {seed}")
            with torch.no_grad():
                features = torch.zeros((4, 19), dtype=torch.float32)
                logits, values = controller.forward(features)
                cost_values = controller.cost_values(features)
            _assert(tuple(logits.shape) == (4, 2) and tuple(values.shape) == (4,)
                    and tuple(cost_values.shape) == (4,),
                    f"native forward shape mismatch: seed {seed} {milestone}")
            _assert(bool(torch.isfinite(logits).all() and torch.isfinite(values).all()
                         and torch.isfinite(cost_values).all()),
                    f"non-finite native forward: seed {seed} {milestone}")
            after_sha = _sha256(path)
            _assert(after_sha == before_sha, f"read-only load changed checkpoint bytes: {path}")
            input_hashes[f"s{seed}_{milestone}"] = {
                "path": str(path), "size_bytes": path.stat().st_size,
                "file_sha256_before": before_sha, "file_sha256_after": after_sha,
                "parameter_sha256": parameter["parameter_sha256"],
                "episodes_done": int(state["episodes_done"]),
                "updates": int(state["updates"]),
                "optimizer_steps": int(state["optimizer_steps"]),
                "lambda": float(state["lambda"]),
                "training_version": state.get("training_version"),
                "comm_action_eligibility": state.get("comm_action_eligibility"),
                "native_cpu_forward": {
                    "logits_shape": list(logits.shape), "task_value_shape": list(values.shape),
                    "cost_value_shape": list(cost_values.shape), "finite": True,
                },
                "read_only": identity,
            }
    return input_hashes


def _test_gate() -> dict[str, Any]:
    original = subprocess.run([str(PYTHON), str(RUN_DIR / "test_r42_one_shot.py")],
                              capture_output=True, text=True)
    original_output = original.stdout + original.stderr
    original_marker = next((line for line in original_output.splitlines()
                            if line.startswith("ROUND42_TEST_GUARD ")), None)
    if original.returncode != 0 or original_marker is None:
        raise RuntimeError(
            f"ROUND42 original pure-compute gate failed (exit={original.returncode}):\n"
            f"{original_output}")
    original_guard = json.loads(original_marker.split(" ", 1)[1])
    _assert(original_guard.get("passed") is True and original_guard.get("test_count") == 12,
            f"unexpected original ROUND42 pure-compute result: {original_guard}")

    seam = subprocess.run([str(PYTHON), str(RUN_DIR / "test_r42_runner_counting.py")],
                          capture_output=True, text=True)
    seam_output = seam.stdout + seam.stderr
    seam_marker = next((line for line in seam_output.splitlines()
                        if line.startswith("ROUND42_RUNNER_COUNTING_GUARD ")), None)
    if seam.returncode != 0 or seam_marker is None:
        raise RuntimeError(
            f"ROUND42 runner seam regression failed (exit={seam.returncode}):\n{seam_output}")
    seam_guard = json.loads(seam_marker.split(" ", 1)[1])
    _assert(seam_guard.get("passed") is True and seam_guard.get("test_count") == 8,
            f"unexpected ROUND42 runner seam result: {seam_guard}")
    forbidden = {
        "joint_run_constructors": int(original_guard["joint_run_constructors"])
        + int(seam_guard["real_joint_run_constructors"]),
        "optimizer_step_calls": int(original_guard["optimizer_step_calls"])
        + int(seam_guard["optimizer_step_calls"]),
        "checkpoint_write_calls": int(original_guard["checkpoint_write_calls"])
        + int(seam_guard["checkpoint_write_calls"]),
    }
    _assert(all(value == 0 for value in forbidden.values()),
            f"pure-compute gate observed forbidden activity: {forbidden}")
    regression = {
        "kind": "round42r-runner-counting-regression/1",
        "status": "passed",
        "command": [str(PYTHON), str(RUN_DIR / "test_r42_runner_counting.py")],
        "return_code": seam.returncode,
        "guard": seam_guard,
        "stdout_stderr": seam_output,
        "old_assignment_reproduction": "slots-backed instance run assignment raises AttributeError",
        "production_seam": "r42_run._make_counted_builder, the factory installed by configure()",
    }
    (RUN_DIR / "RUNNER_COUNTING_REGRESSION.json").write_text(
        json.dumps(regression, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "kind": "round42r-pure-compute-gate/1",
        "status": "passed",
        "commands": [
            [str(PYTHON), str(RUN_DIR / "test_r42_one_shot.py")],
            [str(PYTHON), str(RUN_DIR / "test_r42_runner_counting.py")],
        ],
        "test_count": int(original_guard["test_count"]) + int(seam_guard["test_count"]),
        "passed": True,
        "joint_run_constructors": forbidden["joint_run_constructors"],
        "optimizer_step_calls": forbidden["optimizer_step_calls"],
        "checkpoint_write_calls": forbidden["checkpoint_write_calls"],
        "original_12_freshly_rerun": original_guard,
        "runner_seam_regression": seam_guard,
        "test_names": [
            "free_choice_latches_once_and_revision_cannot_reopen",
            "skip_is_a_real_likelihood_entropy_and_actor_action",
            "closed_skip_stays_in_features_but_never_sends_or_charges",
            "real_send_costs_even_when_lost_and_closes_refreshes",
            "masks_do_not_decide_and_new_episode_resets_latch",
            "multi_target_latches_are_independent_and_budget_is_sequential",
            "batch_likelihood_matches_independent_action_prefix_product",
            "all_closed_rows_keep_both_critics_and_zero_actor_gradient",
            "old_training_restore_refuses_before_state_restore_old_readonly_succeeds",
            "round40_ep0_matches_fresh_same_seed_initialization",
            "unexpected_real_emit_refusal_stops_before_next_tick",
            "hand_derived_expected_objective_directions",
        ],
        "stdout_stderr": original_output + "\n--- runner seam ---\n" + seam_output,
        "simulator_constructed": False,
        "joint_run_constructed": False,
        "real_environment_constructed": False,
        "synthetic_optimizer_step": 0,
    }


def _verified_source_identity() -> dict[str, Any]:
    _assert(_git(ROUND42_ROOT, "branch", "--show-current") == BRANCH,
            "ROUND42-R branch mismatch")
    _assert(not _git(ROUND42_ROOT, "status", "--porcelain"),
            "ROUND42-R preflight requires a clean committed source tree")
    head = _git(ROUND42_ROOT, "rev-parse", "HEAD")
    for ancestor in (ROUND42_INITIAL_EXECUTION_HEAD, ROUND42_OLD_BASIS_HEAD,
                     ROUND42_OLD_DELIVERY_HEAD):
        subprocess.run(["git", "-C", str(ROUND42_ROOT), "merge-base", "--is-ancestor",
                        ancestor, head], check=True, stdout=subprocess.DEVNULL)
    stop_report_commit = _git(ROUND42_ROOT, "log", "-1", "--format=%H", "--",
                              "runs/comm_one_shot_report_selection_20261003/ROUND42_STOP_REPORT.md")
    _assert(stop_report_commit == ROUND42_OLD_DELIVERY_HEAD,
            f"old stop report delivery HEAD mismatch: {stop_report_commit}")
    return {
        "repair_basis_head": head,
        "old_first_execution_head": ROUND42_INITIAL_EXECUTION_HEAD,
        "old_execution_basis_head": ROUND42_OLD_BASIS_HEAD,
        "old_stop_delivery_head": ROUND42_OLD_DELIVERY_HEAD,
        "old_stop_report_commit_verified": stop_report_commit,
    }


def _carryover_manifest(source_identity: dict[str, Any]) -> dict[str, Any]:
    old_dir = ROUND42_ROOT / "runs" / "comm_one_shot_report_selection_20261003"
    claim_path = old_dir / "launch_claims" / "immediate_reference_scene1300000" / "attempt_0001.json"
    slot_path = old_dir / "launch_slots" / "normal" / "0001.json"
    ledger_path = old_dir / "ROUND42_EXECUTION_LEDGER.json"
    stop_state_path = old_dir / "ROUND42_STOP_STATE.json"
    stop_report_path = old_dir / "ROUND42_STOP_REPORT.md"
    inventory_path = old_dir / "ROUND42_ARTIFACT_INVENTORY.json"
    runtime_ledger_path = old_dir / "runtime" / "ROUND42_EXECUTION_LEDGER.json"
    execution_stop_path = old_dir / "runtime" / "ROUND42_EXECUTION_STOP.json"
    first_exception_path = old_dir / "failures" / "ref_immediate_s1300000_r42.json"
    empty_log_path = old_dir / "logs" / "v2_reference_s1300000_a1_rp.jsonl"
    required = (claim_path, slot_path, ledger_path, stop_state_path, stop_report_path,
                inventory_path, runtime_ledger_path, execution_stop_path,
                first_exception_path, empty_log_path)
    for path in required:
        _assert(path.is_file(), f"ROUND42-R carryover evidence missing: {path}")
    claim = _read(claim_path)
    slot = _read(slot_path)
    ledger = _read(ledger_path)
    stop_state = _read(stop_state_path)
    inventory = _read(inventory_path)
    runtime_ledger = _read(runtime_ledger_path)
    execution_stop = _read(execution_stop_path)
    first_exception = _read(first_exception_path)
    _assert(claim.get("logical_job") == "immediate_reference_scene1300000"
            and claim.get("attempt") == 1 and claim.get("category") == "normal"
            and claim.get("status") == "failed_technical"
            and claim.get("environment_constructor_calls") == 1
            and claim.get("simulation_started") is False,
            f"old failure claim identity/status changed: {claim}")
    _assert(slot.get("category") == "normal" and slot.get("slot") == 1
            and slot.get("logical_job") == claim["logical_job"]
            and slot.get("attempt") == 1,
            f"old normal slot does not match claim: {slot}")
    _assert(ledger.get("normal_used") == 1 and ledger.get("reserve_used") == 0
            and ledger.get("total_used") == 1 and ledger.get("counts", {}).get(
                "environment_constructor_calls") == 1
            and ledger.get("counts", {}).get("simulation_started") == 0,
            f"old checked-in ledger does not match the failed claim: {ledger.get('counts')}")
    ledger_claim = next((item for item in ledger.get("claims", [])
                         if item.get("logical_job") == claim["logical_job"]
                         and item.get("attempt") == 1), None)
    stopped_claim = next((item for item in stop_state.get("failed_claims", [])
                          if item.get("logical_job") == claim["logical_job"]
                          and item.get("attempt") == 1), None)
    _assert(ledger_claim is not None and stopped_claim is not None
            and ledger_claim.get("status") == stopped_claim.get("status") == "failed_technical"
            and ledger_claim.get("environment_constructor_calls")
                == stopped_claim.get("environment_constructor_calls") == 1
            and ledger_claim.get("simulation_started")
                == stopped_claim.get("simulation_started") is False,
            "old claim, execution ledger and stop state do not agree")
    _assert(stop_state.get("exception", "").startswith("AttributeError:")
            and stop_state.get("normal_launch_claims") == 1
            and stop_state.get("reserve_launch_claims") == 0,
            "old stop state does not identify the read-only-run failure")
    _assert(runtime_ledger == ledger
            and execution_stop.get("exception", "").startswith("AttributeError:")
            and first_exception.get("exception", "").startswith("AttributeError:"),
            "old runtime ledger or exception artifacts disagree with the checked-in stop record")
    first_attempt_artifact = _sha256(old_dir / "ROUND42_FIRST_ATTEMPT_FAILURE.json")
    inventory_entry = next((item for item in inventory.get("tracked_audit_copies", [])
                            if item.get("repo_relative_path", "").endswith(
                                "/ROUND42_FIRST_ATTEMPT_FAILURE.json")), None)
    _assert(inventory_entry is not None
            and inventory_entry.get("sha256") == first_attempt_artifact
            and inventory_entry.get("matches_original") is True,
            "first-attempt exception record differs from the artifact inventory")
    stop_report_sha = _sha256(stop_report_path)
    stop_state_sha = _sha256(stop_state_path)
    stop_inventory = {item.get("repo_relative_path"): item
                      for item in inventory.get("execution_generated_stop_outputs", [])}
    _assert(stop_inventory.get(
        "runs/comm_one_shot_report_selection_20261003/ROUND42_STOP_REPORT.md", {}
    ).get("sha256") == stop_report_sha
            and stop_inventory.get(
                "runs/comm_one_shot_report_selection_20261003/ROUND42_STOP_STATE.json", {}
            ).get("sha256") == stop_state_sha,
            "old stop outputs do not match their inventory hashes")
    raw_inventory = {item.get("repo_relative_path"): item
                     for item in inventory.get("runtime_and_failure_evidence_outside_git", [])}
    _assert(raw_inventory.get(
        "runs/comm_one_shot_report_selection_20261003/runtime/ROUND42_EXECUTION_LEDGER.json", {}
    ).get("sha256") == _sha256(runtime_ledger_path)
            and raw_inventory.get(
                "runs/comm_one_shot_report_selection_20261003/runtime/ROUND42_EXECUTION_STOP.json", {}
            ).get("sha256") == _sha256(execution_stop_path)
            and raw_inventory.get(
                "runs/comm_one_shot_report_selection_20261003/failures/ref_immediate_s1300000_r42.json", {}
            ).get("sha256") == _sha256(first_exception_path),
            "old raw ledger, stop or exception artifacts do not match their inventory hashes")
    empty_log_entry = raw_inventory.get(
        "runs/comm_one_shot_report_selection_20261003/logs/v2_reference_s1300000_a1_rp.jsonl", {})
    _assert(empty_log_path.stat().st_size == 0
            and empty_log_entry.get("sha256") == _sha256(empty_log_path),
            "old empty log does not match the artifact inventory")
    paths = {
        "old_claim": claim_path,
        "old_normal_slot": slot_path,
        "old_execution_ledger": ledger_path,
        "old_stop_state": stop_state_path,
        "old_stop_report": stop_report_path,
        "old_artifact_inventory": inventory_path,
        "old_runtime_ledger": runtime_ledger_path,
        "old_execution_stop": execution_stop_path,
        "old_failure_output": first_exception_path,
        "old_empty_log": empty_log_path,
    }
    sources = {}
    for label, path in paths.items():
        sources[label] = {
            "repo_relative_path": path.relative_to(ROUND42_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest = {
        "kind": "round42r-carryover-ledger/1",
        "status": "verified_before_new_execution",
        "old_execution_identity": dict(source_identity),
        "sources": sources,
        "old_failure": {
            "logical_job": claim["logical_job"],
            "attempt": 1,
            "category": "normal",
            "status": "failed_technical",
            "builder_invocation": 1,
            "joint_run_constructed_calls": 1,
            "run_invocation_calls": 0,
            "completed_steps": 0,
            "basis": "The preserved traceback reaches run.run assignment after the original builder returned; the original claim, stop state and raw exception agree.",
            "claim_is_immutable_and_remains_failed": True,
        },
        "continuation": {
            "retry_attempt": 2,
            "retry_category": "reserve",
            "retry_reserve_slot": 1,
            "retry_of": sources["old_claim"]["repo_relative_path"],
            "remaining_new_normal_jobs": 671,
            "original_plan_normal_total_including_old_failed_claim": 672,
            "planned_successful_logical_jobs_if_all_pass": 672,
            "cumulative_constructor_calls_if_all_pass": 673,
            "technical_reserve_used_by_retry": 1,
            "technical_reserve_remaining": 27,
            "hard_cap": 700,
            "no_refund_of_old_failure": True,
        },
        "runtime_import_action": "copy old attempt 1 and normal slot 1 byte-for-byte into the isolated ROUND42-R ledger; keep its failed status; retry the same logical job as reserve attempt 2",
    }
    return manifest


def build_prelaunch_freeze() -> dict[str, Any]:
    source_identity = _verified_source_identity()
    repair_basis_head = source_identity["repair_basis_head"]
    test_gate = _test_gate()
    carryover = _carryover_manifest(source_identity)
    carryover_path = RUN_DIR / "ROUND42R_CARRYOVER.json"
    _assert(not carryover_path.exists(), f"ROUND42-R carryover manifest already exists: {carryover_path}")
    carryover_path.write_text(json.dumps(carryover, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    carryover_sha256 = _sha256(carryover_path)
    r40_config_path = ROUND40_OUTPUT / "ROUND40_FROZEN_CONFIG.json"
    r40_frozen = _read(r40_config_path)
    r41_frozen = _read(ROUND41_OUTPUT / "ROUND41_FROZEN_CONFIG.json")
    reconciliation = _read(ROUND40_OUTPUT / "CHECKPOINT_RECONCILIATION.json")
    prior_r42_freeze_path = (
        ROUND42_ROOT / "runs" / "comm_one_shot_report_selection_20261003"
        / "ROUND42_FROZEN_CONFIG.json")
    prior_r42_freeze_sha256 = _sha256(prior_r42_freeze_path)
    prior_r42_frozen = _read(prior_r42_freeze_path)
    r40 = r40_frozen["round40"]
    r40_identity = r40["freeze_identity"]
    prior_r42 = prior_r42_frozen["round42"]
    _assert(prior_r42.get("training_version") == TRAINING_VERSION
            and prior_r42.get("reward_timing_version") == REWARD_TIMING
            and prior_r42.get("comm_action_eligibility") == "first-report-only/1"
            and prior_r42.get("comm_action_semantics") == SEMANTICS,
            "ROUND42-R changed the original training or action-semantics version")
    _assert(prior_r42.get("objective") == r40["objective"]
            and prior_r42.get("ppo_hyperparameters") == r40["ppo_hyperparameters"]
            and prior_r42.get("steps_per_episode") == r40["steps_per_episode"]
            and prior_r42.get("row_filtering") == r40["row_filtering"],
            "ROUND42-R changed objective, PPO, episode length, or row filtering")
    _assert(prior_r42.get("training_seeds") == [11, 22, 33]
            and prior_r42.get("episodes_per_seed") == 128
            and prior_r42.get("episodes_per_update") == 4
            and prior_r42.get("batches_per_seed") == 32
            and prior_r42.get("training_scenes") == [1_300_000, 1_300_047, 48]
            and prior_r42.get("development_scenes") == list(range(1_301_000, 1_301_016))
            and prior_r42.get("actor_feature_width") == 19,
            "ROUND42-R changed training amount, scenes, seeds, or policy feature width")
    prior_plan = prior_r42["plan"]
    _assert(prior_plan["normal_total"] == 672
            and prior_plan["technical_reserve_max"] == 28
            and prior_plan["hard_cap"] == 700
            and prior_plan["immediate_reference"]["total"] == 64
            and prior_plan["new_training"]["total"] == 384
            and prior_plan["old_new_dev"]["total"] == 192
            and prior_plan["rules"]["total"] == 32,
            "ROUND42-R changed the original launch plan or hard cap")

    _assert(_git(ROUND41_ROOT, "branch", "--show-current")
            == "exp/comm-temporal-decoder-diagnostic", "ROUND41 branch mismatch")
    _assert(_git(ROUND41_ROOT, "rev-parse", "HEAD") == ROUND41_HEAD,
            "ROUND41 delivery HEAD mismatch")
    _assert(_git(ROUND40_ROOT, "branch", "--show-current")
            == "exp/comm-first-report-training-ablation", "ROUND40 branch mismatch")
    _assert(_git(ROUND40_ROOT, "rev-parse", "HEAD") == ROUND40_HEAD,
            "ROUND40 delivery HEAD mismatch")
    _assert(_git(ROUND42_ROOT, "branch", "--show-current") == BRANCH,
            "ROUND42 branch mismatch")
    canonical_head = _git(INPUT_ROOT, "rev-parse", "HEAD")
    expected_canonical_head = r41_frozen["execution_identity"][
        "round40_canonical_input_head"]
    _assert(canonical_head == expected_canonical_head,
            f"canonical read-only input HEAD mismatch: {canonical_head} != "
            f"{expected_canonical_head}")
    subprocess.run(["git", "-C", str(ROUND42_ROOT), "merge-base", "--is-ancestor",
                    ROUND41_HEAD, _git(ROUND42_ROOT, "rev-parse", "HEAD")],
                   check=True, stdout=subprocess.DEVNULL)

    # Verify the frozen ROUND40 scripts and imported production/training modules at both the
    # archived execution root and the canonical read-only input root.
    r40_code_hashes: dict[str, str] = {}
    for name, expected in r40_identity["round40_code_sha256"].items():
        path = ROUND40_OUTPUT / name
        actual = _sha256(path)
        _assert(actual == expected, f"ROUND40 versioned script hash mismatch: {name}")
        r40_code_hashes[name] = actual
    inherited_deps = r40_identity["round36_dependency_hashes"]
    dependency_hashes: dict[str, str] = {}
    round42_checkout_dependency_hashes: dict[str, dict[str, Any]] = {}
    for group in ("production", "imported_training_modules", "stage_c_harness"):
        for relative, record in inherited_deps[group].items():
            expected = record["sha256"]
            canonical = _dependency_path(INPUT_ROOT, relative, group)
            current = _dependency_path(ROUND42_ROOT, relative, group)
            canonical_hash = _sha256(canonical)
            current_hash = _sha256(current)
            _assert(canonical_hash == expected,
                    f"frozen ROUND40 canonical dependency changed: {relative}")
            dependency_hashes[relative] = expected
            round42_checkout_dependency_hashes[relative] = {
                "sha256": current_hash,
                "matches_round40_active_dependency": current_hash == expected,
                "active_for_round42_execution": False,
            }

    # The ROUND41 checkout contains later committed source snapshots for some production files.
    # ROUND40/41 both ran against the separately recorded canonical input HEAD; ROUND42 follows
    # that same rule and asserts imported module paths above. Record the checkout differences
    # rather than copying/rewriting production files to make hashes appear equal.

    r41_code_hashes: dict[str, str] = {}
    for name, expected in r41_frozen["round41_code_sha256"].items():
        path = ROUND41_OUTPUT / name
        actual = _sha256(path)
        _assert(actual == expected, f"ROUND41 source hash mismatch: {name}")
        r41_code_hashes[name] = actual

    round42_code_hashes: dict[str, str] = {}
    for name in R42_CODE_FILES:
        path = RUN_DIR / name
        _assert(path.is_file(), f"ROUND42 source file not yet present: {name}")
        round42_code_hashes[name] = _sha256(path)

    pure_gate_path = RUN_DIR / "ROUND42_PURE_COMPUTE_GATE.json"
    pure_gate_path.write_text(json.dumps(test_gate, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    runner_regression_sha256 = _sha256(RUN_DIR / "RUNNER_COUNTING_REGRESSION.json")
    pure_gate_sha256 = _sha256(pure_gate_path)

    checkpoint_content = _checkpoint_forward_gate(r40_frozen, reconciliation)
    scene_file = _read(ROUND40_OUTPUT / "SCENARIO_ORDER.json")
    training_order = list(scene_file["order"])
    order_sha = hashlib.sha256(json.dumps(training_order).encode("utf-8")).hexdigest()
    _assert(len(training_order) == 128 and order_sha == r40["training_order_sha256"],
            "ROUND40 training episode order does not match its frozen identity")
    _assert(prior_r42.get("training_order_sha256") == order_sha,
            "ROUND42-R changed the frozen training episode order")
    _assert(set(training_order) <= set(range(1_300_000, 1_300_048)),
            "ROUND40 order contains a scene outside its registered 48-scene block")

    # Frozen task/allocation weights are read-only inputs outside this repository snapshot.
    frozen_search_allocation: dict[str, Any] = {}
    for path in (
        INPUT_ROOT / "runs" / "cur_v2rep05_s2" / "best" / "shared_mappo_model_21_3100.pth",
        INPUT_ROOT / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth",
    ):
        _assert(path.is_file(), f"frozen search/allocation checkpoint missing: {path}")
        frozen_search_allocation[path.name] = {
            "path": str(path), "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    code_identity = {
        "round40_code_sha256": r40_code_hashes,
        "round41_code_sha256": r41_code_hashes,
        "round42_code_sha256": round42_code_hashes,
        "inherited_training_and_production_dependency_sha256": dependency_hashes,
        "round42_checkout_dependency_sha256": round42_checkout_dependency_hashes,
    }
    unchanged_fields = {
        "reward_timing_version": REWARD_TIMING,
        "eligibility_gate": "first-report-only/1",
        "objective": r40["objective"],
        "ppo_hyperparameters": r40["ppo_hyperparameters"],
        "steps_per_episode": r40["steps_per_episode"],
        "training_order_sha256": order_sha,
        "training_scenes": [1_300_000, 1_300_047, 48],
        "development_scenes": list(range(1_301_000, 1_301_016)),
        "actor_feature_width": 19,
        "row_filtering": r40["row_filtering"],
        "frozen_search_allocation_checkpoints": frozen_search_allocation,
        "production_and_training_dependency_sha256": dependency_hashes,
    }
    _assert(REWARD_TIMING == r40["reward_timing_version"],
            "ROUND42 changed the ROUND40 reward timing version")
    _assert(r40["comm_action_eligibility"] == "first-report-only/1",
            "ROUND42 changed first-report eligibility")
    _assert(r40["objective"] == unchanged_fields["objective"]
            and r40["ppo_hyperparameters"] == unchanged_fields["ppo_hyperparameters"]
            and int(r40["steps_per_episode"]) == unchanged_fields["steps_per_episode"]
            and r40["row_filtering"] == unchanged_fields["row_filtering"],
            "ROUND42 research configuration differs from the frozen ROUND40 values")
    configuration_diff = {
        "kind": "round42-configuration-diff/1",
        "base_round": 40,
        "round42r_prior_freeze_sha256": prior_r42_freeze_sha256,
        "round42r_science_configuration_equal_to_prior_freeze": {
            "fields_checked": [
                "training_version", "reward_timing_version", "comm_action_eligibility",
                "comm_action_semantics", "training_seeds", "episodes_per_seed",
                "episodes_per_update", "batches_per_seed", "training_order_sha256",
                "training_scenes", "development_scenes", "objective",
                "ppo_hyperparameters", "steps_per_episode", "row_filtering",
                "old_models", "new_initial_models", "frozen_search_allocation_checkpoints",
            ],
            "status": "verified_equal_except_runner_identity_and_carryover_ledger",
        },
        "action_semantics_change": {
            "old": "first-report-only HOLD can be reconsidered on later candidate rows",
            "new": SEMANTICS,
            "new_contract": "one free SEND_NOW or SKIP_EPISODE choice per target per episode; later candidates fixed HOLD",
        },
        "training_version_change": {
            "old": r40["training_version"],
            "new": TRAINING_VERSION,
        },
        "runner_execution_identity_change": {
            "old_runner": "ROUND42 counted_build instance-method assignment (failed before run)",
            "new_runner": RUNNER_VERSION,
            "training_version_changed_by_runner_fix": False,
        },
        "identity_and_output_changes": [
            "new action-semantics marker in freeze/tape/checkpoint/training_state/rows",
            "new checkpoint identity; ROUND40 checkpoint resume is rejected",
            "ROUND40 ep0 remains read-only initialization identity and OLD ep128 remains read-only evaluation input",
            "ROUND42-R isolated output namespace and 672+28 process-safe launch ledger with one old failed normal claim carried forward",
        ],
        "runtime_source_provenance": {
            "active_source": "canonical read-only input checkout pinned to ROUND40's recorded base HEAD",
            "round42_checkout_dependency_snapshot": round42_checkout_dependency_hashes,
            "note": "Some tracked production files in the ROUND41-derived worktree differ from the pinned canonical experiment inputs; the runner imports and verifies the canonical paths. No production files are copied or modified to hide this distinction.",
        },
        "held_constant": unchanged_fields,
        "unexpected_research_configuration_differences": [],
    }
    configuration_diff_sha = _canonical_sha256(configuration_diff)
    freeze_identity: dict[str, Any] = {
        "kind": "round42-one-shot-send-skip-training-freeze/1",
        "base_head": ROUND41_HEAD,
        "round40_head": ROUND40_HEAD,
        "round42_basis_commit": repair_basis_head,
        "round42_previous_freeze_sha256": prior_r42_freeze_sha256,
        "runner_version": RUNNER_VERSION,
        "carryover_manifest_sha256": carryover_sha256,
        "runner_counting_regression_sha256": runner_regression_sha256,
        "pure_compute_gate_sha256": pure_gate_sha256,
        "training_version": TRAINING_VERSION,
        "reward_timing_version": REWARD_TIMING,
        "comm_action_eligibility": "first-report-only/1",
        "comm_action_semantics": SEMANTICS,
        "round40_freeze_sha256": _canonical_sha256(r40_identity),
        "round41_freeze_sha256": _canonical_sha256(r41_frozen),
        "scenario_order_sha256": order_sha,
        "old_checkpoint_parameter_identities": r40_identity[
            "old_checkpoint_parameter_identities"],
        "round40_initial_parameter_identities": r40_identity[
            "initial_parameter_identities"],
        "round40_ep128_parameter_identities": {
            seed: reconciliation["checkpoints"]["NEW_training_outputs"][seed]["ep128"][
                "parameter_sha256"] for seed in ("11", "22", "33")
        },
        "round40_source_checkpoint_hashes": {
            f"s{seed}_{milestone}": checkpoint_content[f"s{seed}_{milestone}"][
                "file_sha256_before"]
            for seed in (11, 22, 33) for milestone in ("ep0", "ep128")
        },
        "round42_code_sha256": round42_code_hashes,
        "active_dependency_sha256": dependency_hashes,
        "round42_checkout_dependency_sha256": round42_checkout_dependency_hashes,
        "configuration_diff_sha256": configuration_diff_sha,
        "output_directory_repo_relative": "runs/comm_one_shot_report_selection_20261003_r",
    }
    freeze_identity["freeze_sha256"] = _canonical_sha256(freeze_identity)

    old_models = {
        str(seed): {
            **checkpoint_content[f"s{seed}_ep128"],
            "arm": "OLD",
            "protocol_adaptation": "ROUND40 first-report-only weights read-only under ROUND42 one-shot SEND/SKIP",
        }
        for seed in (11, 22, 33)
    }
    initial_models = {
        str(seed): checkpoint_content[f"s{seed}_ep0"] for seed in (11, 22, 33)
    }
    _assert(prior_r42.get("old_models") == old_models
            and prior_r42.get("new_initial_models") == initial_models
            and prior_r42.get("frozen_search_allocation_checkpoints") == frozen_search_allocation,
            "ROUND42-R changed OLD/NEW checkpoint or search/allocation input identities")

    frozen = {
        "kind": "round42-one-shot-send-skip-prelaunch-freeze/1",
        "status": "passed_before_environment_start",
        "execution_identity": {
            "branch": BRANCH,
            "round42_worktree": str(ROUND42_ROOT),
            "round42_basis_commit": repair_basis_head,
        "round42_previous_freeze_sha256": prior_r42_freeze_sha256,
            "round42_prior_stop_delivery_head": ROUND42_OLD_DELIVERY_HEAD,
            "round42_initial_execution_head": ROUND42_INITIAL_EXECUTION_HEAD,
            "runner_version": RUNNER_VERSION,
            "carryover_manifest_sha256": carryover_sha256,
            "runner_counting_regression_sha256": runner_regression_sha256,
            "pure_compute_gate_sha256": pure_gate_sha256,
            "round41_branch": "exp/comm-temporal-decoder-diagnostic",
            "round41_delivery_head": ROUND41_HEAD,
            "round41_worktree": str(ROUND41_ROOT),
            "round40_branch": "exp/comm-first-report-training-ablation",
            "round40_delivery_head": ROUND40_HEAD,
            "round40_worktree": str(ROUND40_ROOT),
            "canonical_read_only_input_root": str(INPUT_ROOT),
            "canonical_read_only_input_head": canonical_head,
            "python_executable": str(PYTHON),
            "round40_code_sha256": r40_code_hashes,
            "round41_code_sha256": r41_code_hashes,
            "round42_code_sha256": round42_code_hashes,
            "frozen_dependency_sha256": dependency_hashes,
            "round42_checkout_dependency_sha256": round42_checkout_dependency_hashes,
        },
        "round42": {
            "training_version": TRAINING_VERSION,
            "runner_version": RUNNER_VERSION,
            "reward_timing_version": REWARD_TIMING,
            "comm_action_eligibility": "first-report-only/1",
            "comm_action_semantics": SEMANTICS,
            "old_checkpoint_training_restore": "rejected_before_controller_optimizer_or_rng_restore",
            "old_checkpoint_read_only_evaluation": "allowed_with_explicit_protocol_adaptation_identity",
            "round40_freeze_sha256": _canonical_sha256(r40_identity),
            "freeze_identity": freeze_identity,
            "allowed_differences_vs_round40": [
                "target action semantics: one-shot SEND_NOW/SKIP_EPISODE with per-target episode latch",
                "training version, checkpoint/freeze identity and action-semantics fields",
                "ROUND42-R runner counting repair, isolated output namespace and carried-over launch ledger",
                "read-only OLD/NEW comparisons use the ROUND42 one-shot evaluation protocol",
            ],
            "unexpected_research_configuration_differences": [],
            "training_seeds": [11, 22, 33],
            "episodes_per_seed": 128,
            "episodes_per_update": 4,
            "batches_per_seed": 32,
            "training_order": training_order,
            "training_order_sha256": order_sha,
            "training_scenes": [1_300_000, 1_300_047, 48],
            "development_scenes": list(range(1_301_000, 1_301_016)),
            "objective": r40["objective"],
            "ppo_hyperparameters": r40["ppo_hyperparameters"],
            "steps_per_episode": r40["steps_per_episode"],
            "world_config_source": "ROUND40_FROZEN_CONFIG.json; no fields changed",
            "candidate_order": "inherited relay observation order",
            "actor_feature_width": 19,
            "row_filtering": r40["row_filtering"],
            "old_models": old_models,
            "new_initial_models": initial_models,
            "frozen_search_allocation_checkpoints": frozen_search_allocation,
            "plan": {
                "immediate_reference": {"train_scenes": 48, "dev_scenes": 16, "total": 64},
                "new_training": {"seeds": 3, "episodes_each": 128, "total": 384},
                "old_new_dev": {"arms": ["OLD", "NEW"], "seeds": 3,
                                "modes": ["greedy", "sampled"], "scenes": 16, "total": 192},
                "rules": {"arms": list(RULE_ARMS), "scenes": 16, "total": 32},
                "normal_total": 672,
                "technical_reserve_max": 28,
                "hard_cap": 700,
                "continuation": {
                    "new_normal_jobs": 671,
                    "carryover_old_failed_normal_claims": 1,
                    "retry_uses_reserve": 1,
                    "logical_job_success_target": 672,
                    "cumulative_constructors_if_all_pass": 673,
                    "carryover_manifest_sha256": carryover_sha256,
                },
                "reserve_policy": "attempts after first logical cell attempt consume reserve and do not refund prior claims",
            },
            "prelaunch_gate": {
                "pure_compute": test_gate,
                "checkpoint_content_and_forward": checkpoint_content,
                "environment_constructed": 0,
                "joint_run_constructed": 0,
                "optimizer_step_calls": 0,
                "checkpoint_write_calls": 0,
            },
        },
    }

    version_manifest = {
        "kind": "round42-training-version-manifest/1",
        "training_version": TRAINING_VERSION,
        "runner_version": RUNNER_VERSION,
        "reward_timing_version": REWARD_TIMING,
        "comm_action_eligibility": "first-report-only/1",
        "comm_action_semantics": SEMANTICS,
        "old_round40_training_version": r40["training_version"],
        "old_checkpoint_training_restore": "rejected",
        "round40_ep0_initialization": "read-only source; NEW optimizer and state start fresh",
        "old_round40_ep128_evaluation": "read-only under explicit ROUND42 protocol adaptation",
        "round42_checkpoint_identity_fields": [
            "training_version", "reward_timing_version", "comm_action_eligibility",
            "comm_action_semantics", "freeze", "training_state",
        ],
        "strict_resume_entry": "r42_versions.install -> trainer.load_checkpoint",
        "round42_freeze_sha256": freeze_identity["freeze_sha256"],
        "old_checkpoint_param_identities": freeze_identity["round40_ep128_parameter_identities"],
        "initial_parameter_identities": r40_identity["initial_parameter_identities"],
        "round42_code_sha256": round42_code_hashes,
        "round42_previous_freeze_sha256": prior_r42_freeze_sha256,
        "active_dependency_sha256": dependency_hashes,
        "round42_checkout_dependency_sha256": round42_checkout_dependency_hashes,
        "configuration_diff_sha256": configuration_diff_sha,
        "optimizer_step_allowed_only_in_real_round42_training": True,
    }

    (RUN_DIR / "ROUND42_FROZEN_CONFIG.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RUN_DIR / "TRAINING_VERSION_MANIFEST.json").write_text(
        json.dumps(version_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RUN_DIR / "ROUND42_CONFIGURATION_DIFF.json").write_text(
        json.dumps(configuration_diff, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    preflight = {
        "kind": "round42-prelaunch-preflight/1",
        "status": "passed",
        "execution_identity": frozen["execution_identity"],
        "round40_delivery_head_verified": ROUND40_HEAD,
        "round41_delivery_head_verified": ROUND41_HEAD,
        "round42_basis_head": repair_basis_head,
        "round42_previous_stop_delivery_head": ROUND42_OLD_DELIVERY_HEAD,
        "runner_version": RUNNER_VERSION,
        "carryover_manifest_sha256": carryover_sha256,
        "runner_counting_regression_sha256": runner_regression_sha256,
        "pure_compute_gate_sha256": pure_gate_sha256,
        "round42_code_sha256": round42_code_hashes,
        "round42_previous_freeze_sha256": prior_r42_freeze_sha256,
        "round40_training_order_sha256": order_sha,
        "round40_initial_and_ep128_checkpoints": checkpoint_content,
        "frozen_search_allocation_checkpoints": frozen_search_allocation,
        "pure_compute_gate": {
            "test_count": test_gate["test_count"],
            "passed": test_gate["passed"],
            "joint_run_constructed": test_gate["joint_run_constructors"],
            "optimizer_step_calls": test_gate["optimizer_step_calls"],
            "checkpoint_write_calls": test_gate["checkpoint_write_calls"],
        },
        "plan": frozen["round42"]["plan"],
        "first_environment_start_authorized_after_local_commit": True,
    }
    (RUN_DIR / "ROUND42_PREFLIGHT.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return frozen


def main() -> int:
    if not PYTHON.is_file():
        raise FileNotFoundError(f"registered Python runtime unavailable: {PYTHON}")
    import torch
    frozen = build_prelaunch_freeze()
    print(json.dumps({
        "status": frozen["status"],
        "round41_head": frozen["execution_identity"]["round41_delivery_head"],
        "round40_head": frozen["execution_identity"]["round40_delivery_head"],
        "training_version": frozen["round42"]["training_version"],
        "test_count": frozen["round42"]["prelaunch_gate"]["pure_compute"]["test_count"],
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cpu_only": True,
        "round42_code_hashes": frozen["execution_identity"]["round42_code_sha256"],
        "normal_launches": 672,
        "technical_reserve": 28,
        "runner_version": frozen["execution_identity"]["runner_version"],
        "carryover_manifest_sha256": frozen["execution_identity"]["carryover_manifest_sha256"],
        "environment_constructed": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
