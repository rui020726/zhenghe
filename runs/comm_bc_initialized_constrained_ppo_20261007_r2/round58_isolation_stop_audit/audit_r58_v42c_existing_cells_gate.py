from __future__ import annotations

import builtins
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import runpy
import sys
import traceback
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN_ROOT = REPO / "runs/comm_bc_initialized_constrained_ppo_20261007_r2"
RAW = RUN_ROOT / "round58_gen3/raw"
RESUME_RAW = RUN_ROOT / "round58_gen3_resume_v35/raw"
ALIAS_RAW = RUN_ROOT / "round58_gen3_resume_v18/raw"
LAUNCHER = HERE / "r58_private_launcher_repaired.py"
AUTH = HERE / "ROUND58_RESUME_EXECUTION_AUTHORIZATION_V42.json"
FREEZE = HERE / "ROUND58_RESUME_FREEZE_V42.json"
QUOTA = HERE / "ROUND58_GEN3_CUMULATIVE_QUOTA.json"
SCENES = HERE / "ROUND58_GEN3_RESUME_SCENE_OCCUPANCY_RECHECK.json"
PROOF = HERE / "ROUND58_GEN3_PLAN_IDENTITY_MIGRATION_PROOF_V42.json"
OUTPUT = HERE / "ROUND58_V42C_ACTUAL_EXISTING_CELLS_GATE.json"
PROTECTED = (RUN_ROOT / "ROUND46_CURRENT_TASK.json",
             RUN_ROOT / "ROUND46_R2_V6_STOP_STATE.json")


class RestoreGateStop(BaseException):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) \
            and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) \
            and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) \
            and left.keys() == right.keys() \
            and all(same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) \
            and len(left) == len(right) \
            and all(same(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) \
            and left.dtype == right.dtype and left.shape == right.shape \
            and np.array_equal(left, right)
    try:
        return bool(left == right)
    except Exception:
        return False


def rng_fingerprint() -> dict[str, str]:
    return {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
    }


ALIAS_HASHES = {
    "r40_train_resume_adapter.py": "688dee0c60a5ab7893991d10948a7c5cff12ec6ba6f91fce07c3439a0ee7f2ca",
    "r42_one_shot.py": "3c342108148de42f395faeec25544976eed7efdfdd32a99685d73bd59101ab0d",
    "r42_versions.py": "26b9e1a3d4f3b9a5d39ca020a5ebd3653615c09a76a77ae4138f4ab98420a3d8",
    "r43_call_counter_hooks.py": "02f80072a14d7305b619edd0db9617a7b0a0f31315b3265aebf1e694ac7398f7",
}


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite restore-gate evidence: {OUTPUT}")
    for path in (LAUNCHER, AUTH, FREEZE, QUOTA, SCENES, PROOF):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not ALIAS_RAW.is_dir():
        raise RuntimeError("V18 resume runtime-alias root is not provisioned")
    existing = {p.name for p in ALIAS_RAW.iterdir() if p.is_file()}
    if existing != set(ALIAS_HASHES) | {"ROUND45_CURRENT_TASK.json"}:
        raise RuntimeError(f"V16 resume root contains unexpected/missing files: {sorted(existing)}")
    for name, expected in ALIAS_HASHES.items():
        if sha(ALIAS_RAW / name) != expected:
            raise RuntimeError(f"V12 resume runtime alias hash mismatch: {name}")

    protected_before = {str(path): sha(path) for path in PROTECTED}
    history_before = {str(path): sha(path) for path in (
        RAW / "ROUND46_FIRST_FOUR_s11.json", RAW / "ROUND46_FIRST_FOUR_s22.json",
        RAW / "ROUND46_FIRST_FOUR_s33.json",
        RUN_ROOT / "round58_gen3_resume/raw/runtime/checkpoint_writes.jsonl",
        RUN_ROOT / "round58_gen3_resume/raw/runtime/ROUND53_EXECUTION_LEDGER.json")}
    rng_before = rng_fingerprint()
    input_hashes = {p.name: sha(p) for p in (AUTH, FREEZE, QUOTA, SCENES, PROOF)}
    events: dict[str, Any] = {
        "checkpoint_loads": [], "rng_restore_calls": [], "write_attempts": [],
        "empty_checkpoint_journal_spy": [], "derived_ledger_spy": [],
        "suppressed_directories": [], "optimizer_step_calls": 0,
        "torch_save_calls": 0, "environment_provider_denials": 0,
        "claim_denials": 0, "torch_load_calls": 0,
    }

    def is_inside_resume(path: Any) -> bool:
        try:
            Path(path).resolve().relative_to(RESUME_RAW.resolve())
            return True
        except (ValueError, OSError, TypeError):
            return False

    def deny(kind: str, target: Any):
        events["write_attempts"].append({"kind": kind, "target": str(target)})
        raise AssertionError(f"restore gate forbidden side effect {kind}: {target}")

    original_open = builtins.open
    original_path_open = Path.open
    original_write_text = Path.write_text
    original_write_bytes = Path.write_bytes
    original_touch = Path.touch
    original_mkdir = Path.mkdir
    original_os_open = os.open
    original_os_makedirs = os.makedirs
    original_replace, original_rename = os.replace, Path.rename
    original_path_replace = Path.replace
    original_torch_save, original_torch_load = torch.save, torch.load
    original_opt_step, original_adam_step = torch.optim.Optimizer.step, torch.optim.Adam.step
    original_py_setstate = random.setstate
    original_np_setstate = np.random.set_state
    original_torch_setstate = torch.set_rng_state
    original_torch_random_setstate = torch.random.set_rng_state
    original_cuda_setstate = torch.cuda.set_rng_state
    original_cuda_setstate_all = torch.cuda.set_rng_state_all
    original_argv, original_bytecode = sys.argv[:], sys.dont_write_bytecode
    original_profile, original_trace = sys.getprofile(), sys.gettrace()
    journal = RESUME_RAW / "runtime/checkpoint_writes.jsonl"
    ledger = RESUME_RAW / "runtime/ROUND53_EXECUTION_LEDGER.json"

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
            deny("builtins.open", file)
        return original_open(file, mode, *args, **kwargs)

    def guarded_path_open(self, mode="r", *args, **kwargs):
        if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
            deny("Path.open", self)
        return original_path_open(self, mode, *args, **kwargs)

    def guarded_write_bytes(self, data):
        target = Path(self).resolve()
        if target != journal.resolve() or data != b"":
            deny("Path.write_bytes", target)
        events["empty_checkpoint_journal_spy"].append({
            "path": str(target), "mode": "wb", "bytes": 0,
            "action": "memory_only_no_file_created"})
        if len(events["empty_checkpoint_journal_spy"]) != 1:
            deny("duplicate empty checkpoint journal", target)
        return 0

    def guarded_write_text(self, data, *args, **kwargs):
        target = Path(self).resolve()
        if target != ledger.resolve() or not isinstance(data, str):
            deny("Path.write_text", target)
        parsed = json.loads(data)
        if int(parsed.get("environment_consumed", -1)) < 0:
            deny("derived ledger has no historical consumption", target)
        events["derived_ledger_spy"].append({
            "path": str(target), "bytes": len(data.encode(kwargs.get("encoding") or "utf-8")),
            "reported_environment_consumed": int(parsed["environment_consumed"]),
            "action": "memory_only_no_file_created"})
        return len(data)

    def guarded_touch(self, *args, **kwargs):
        deny("Path.touch", self)

    def guarded_mkdir(self, *args, **kwargs):
        target = Path(self).resolve()
        if is_inside_resume(target):
            events["suppressed_directories"].append(str(target))
            return None
        if target.exists() and target.is_dir():
            return None
        deny("Path.mkdir", target)

    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            deny("os.open", path)
        return original_os_open(path, flags, *args, **kwargs)

    def guarded_makedirs(name, *args, **kwargs):
        if is_inside_resume(name):
            events["suppressed_directories"].append(str(Path(name).resolve()))
            return None
        deny("os.makedirs", name)

    def guarded_torch_save(*args, **kwargs):
        events["torch_save_calls"] += 1
        deny("torch.save", kwargs.get("f", args[1] if len(args) > 1 else "unknown"))

    def guarded_step(*_args, **_kwargs):
        events["optimizer_step_calls"] += 1
        raise AssertionError("optimizer.step forbidden during restore gate")

    def counted_load(*args, **kwargs):
        events["torch_load_calls"] += 1
        return original_torch_load(*args, **kwargs)

    def count_rng_restore(name, original):
        def wrapped(*args, **kwargs):
            events["rng_restore_calls"].append(name)
            return original(*args, **kwargs)
        return wrapped

    def forbidden_runtime_call(name):
        def wrapped(*_args, **_kwargs):
            if name == "episode_provider":
                events["environment_provider_denials"] += 1
            elif name == "claim":
                events["claim_denials"] += 1
            raise AssertionError(f"restore gate crossed {name} boundary")
        return wrapped

    state_holder: dict[str, Any] = {}

    def eval_handoff_gate(payload: dict[str, Any]) -> None:
        runtime, legacy, plan = payload["runtime"], payload["legacy"], payload["plan"]
        scenes = tuple(range(1_700_000, 1_700_032))
        if (tuple(legacy.SCENES_NEW) != scenes or tuple(plan["eval_scenes_new"]) != scenes
                or legacy.v2_controller is not runtime["controller_module"]
                or sys.modules.get("v2_controller") is not legacy.v2_controller):
            raise AssertionError("configured evaluator scenes/controller differ from frozen plan")
        from r53_contracts import TRAINING_VERSION as v53
        v56 = runpy.run_path(str(RUN_ROOT / "round56_soft_early_report_prior/r56_contracts.py"), run_name="r56_contracts_eval_gate")["TRAINING_VERSION"]
        if legacy.TV53 != v53 or legacy.TV56 != v56:
            raise AssertionError("comparator version identity differs from archived source contracts")
        matrix = [job for job in plan["jobs"] if job["scope"].startswith("model_eval_")
                  or job["scope"].startswith("rule_eval_")]
        if (len(matrix) != 640 or len({job["logical_job"] for job in matrix}) != 640
                or len([job for job in matrix if job["scope"].startswith("model_eval_")]) != 576
                or len([job for job in matrix if job["scope"].startswith("rule_eval_")]) != 64):
            raise AssertionError("evaluation matrix is not the frozen 576+64 unique keys")
        bridge = runtime.get("r58_eval_booking_bridge")
        run_module = runtime["r42_run"]
        if (not isinstance(bridge, dict) or bridge.get("status") != "installed"
                or bridge.get("plan_eval_keys") != 640
                or bridge.get("registered_failed_retry", {}).get("retry_status") != "complete"
                or bridge.get("registered_failed_rule_retry", {}).get("retry_status") != "complete"
                or getattr(runtime["runner"]._booked, "__name__", "") != "round58_v35_eval_booked"
                or getattr(run_module._execute_cell, "__name__", "") != "round58_v35_eval_execute"):
            raise AssertionError("actual eval booking/retry bridge is not active")
        r45 = sys.modules.get("round46_r45_runtime_adapter")
        if r45 is None or Path(r45.RUN_DIR).resolve() != RESUME_RAW.resolve():
            raise AssertionError("R45 receipt root is not bound to the V35 artifact root")
        old_claim = RAW / "launch_claims/model_eval_new_s11_greedy_scene1700000/attempt_0001.json"
        claim = json.loads(old_claim.read_text(encoding="utf-8"))
        old_row = RUN_ROOT / "round58_gen3_resume_v33/raw/rows/v2_r44_eval_new_s11_greedy_s1700000_a1.json"
        row = json.loads(old_row.read_text(encoding="utf-8"))
        original_active, original_receipt = run_module._ACTIVE_CLAIM, r45._record_artifact_receipt
        captured = []
        run_module._ACTIVE_CLAIM = old_claim
        r45._record_artifact_receipt = lambda *args: captured.append(args)
        try:
            runtime["runner"]._booked(row["label"], "eval", row, object())
            wrong = dict(row)
            wrong["scenario_seed"] = 1700001
            try:
                runtime["runner"]._booked(row["label"], "eval", wrong, object())
            except RuntimeError:
                pass
            else:
                raise AssertionError("V35 booking accepted a row outside its claim key")
        finally:
            r45._record_artifact_receipt = original_receipt
            run_module._ACTIVE_CLAIM = original_active
        if len(captured) != 1 or captured[0][2] != old_claim:
            raise AssertionError("V35 booker did not delegate exactly once to real R45 receipt function")
        previous_r45_root = r45.RUN_DIR
        r45.RUN_DIR = old_row.parent.parent
        try:
            receipt = r45._create_round45_artifact_receipt(row, claim)
        finally:
            r45.RUN_DIR = previous_r45_root
        if receipt.get("kind") != "round45-r2-artifact-receipt/2" or len(receipt.get("artifacts", {})) != 4:
            raise AssertionError("actual R45 receipt cannot close persisted row companions")
        events["booking_probe"] = {"old_failed_row_sha256": sha(old_row),
                                   "actual_R45_receipt_kind": receipt["kind"],
                                   "spied_booking_delegations": len(captured),
                                   "wrong_scene_rejected": True,
                                   "no_old_claim_mutation": True}
        import r58_v35_eval_booking as native_booking
        rule_claim_path = RAW / "launch_claims/rule_immediate_scene1700000/attempt_0001.json"
        rule_claim = json.loads(rule_claim_path.read_text(encoding="utf-8"))
        rule_row_path = RESUME_RAW / "rows/v2_r58_rule_immediate_s1700000_s1700000_a1.json"
        rule_row = json.loads(rule_row_path.read_text(encoding="utf-8"))
        native_receipt = native_booking._rule_receipt(r45, rule_row, rule_claim)
        if (set(native_receipt["artifacts"]) != {"row", "lifecycle", "transcript"}
                or Path(rule_row["artefacts"]["tape"]).exists()):
            raise AssertionError("native-rule actual artifacts were not three pieces")
        wrong_rule = dict(rule_row)
        wrong_rule["scenario_seed"] = 1700001
        try:
            native_booking._rule_receipt(r45, wrong_rule, rule_claim)
        except RuntimeError:
            pass
        else:
            raise AssertionError("native-rule receipt accepted tampered scene")
        events["native_rule_probe"] = {"failed_claim_sha256": sha(rule_claim_path),
                                        "failed_row_sha256": sha(rule_row_path),
                                        "actual_artifacts": sorted(native_receipt["artifacts"]),
                                        "tampered_scene_rejected": True}
        model_rows = []
        for seed in (11, 22, 33):
            source = payload["completed_training_sources"][seed]
            model, identity = payload["evaluation_loader"]("NEW", seed)
            events["debug_identity"] = {"seed": seed, "actual": identity, "expected_sha": source["checkpoint_sha256"]}
            if (identity.get("optimizer_restored") is not False
                    or identity.get("rng_restored") is not False
                    or identity.get("file_sha256") != source["checkpoint_sha256"]
                    or identity.get("r58_reused_completed_training_source", {}).get("model_only") is not True):
                raise AssertionError(f"seed {seed}: evaluator did not load exact model-only endpoint")
            model_rows.append({"seed": seed, "checkpoint_sha256": source["checkpoint_sha256"],
                               "summary_sha256": source["summary_sha256"]})
        for seed in (11, 22, 33):
            if (RESUME_RAW / f"ROUND46_TRAINING_s{seed}.json").exists():
                raise AssertionError("eval-only identity rewrote an immutable training summary")
        # Actual archived existing-cell route: no new provider/claim is permitted.
        model11, identity11 = payload["evaluation_loader"]("NEW", 11)
        original_completed = r45._completed_row
        def capture_completed(context, prior, **kwargs):
            c = json.loads(old_claim.read_text(encoding="utf-8"))
            c2 = json.loads((RAW / "launch_claims/model_eval_new_s11_greedy_scene1700000/attempt_0002.json").read_text(encoding="utf-8"))
            events["completed_debug"] = {
                "actual_context": context,
                "serialized_context": r45._serializable_key(context),
                "claim_key": c2.get("round45_key"),
                "claim_round": c2.get("round"),
                "claim_training_version": c2.get("training_version"),
                "expected_training_version": r45.RUNTIME["r42_versions"].TRAINING_VERSION,
                "claim_runner_version": c2.get("runner_version"),
                "expected_runner_version": r45.RUNNER_VERSION,
                "claim_semantics": c2.get("comm_action_semantics")}
            return original_completed(context, prior, **kwargs)
        r45._completed_row = capture_completed
        # Let the *actual archived evaluator* traverse all three model arms
        # and the 32 already-completed immediate rule cells. Stop after the
        # frozen late constructor but before its first provider/claim.
        reused = []
        saved_execute = run_module._execute_cell
        model_keys = {job["logical_job"] for job in matrix if job["scope"].startswith("model_eval_")}
        existing_keys = model_keys | {job["logical_job"] for job in matrix
                                      if job["scope"] == "rule_eval_immediate"}

        def observe_existing_only(label, logical, kind, scenario, metadata, **kwargs):
            if logical == "rule_late_scene1700000":
                if len(reused) != 608 or set(reused) != existing_keys:
                    raise AssertionError(f"actual CLI reused only {len(reused)}/608 prior cells")
                loaded_rule = sys.modules.get("v2_rules")
                if loaded_rule is None or sha(Path(loaded_rule.__file__)) != "a7ce36de2d465cadaba90bd70b36539b5e865a73be8c4e075b9faf5a86c53a57":
                    raise AssertionError("actual late-rule source is not the frozen module")
                policy = kwargs.get("runner_kwargs", {}).get("attach")
                if (type(policy) is not loaded_rule.LateDiscoveryGatePolicy
                        or policy.parameters() != {"cut_steps": 147.0}
                        or type(policy).decide is not type(policy).__mro__[1].decide):
                    raise AssertionError("actual late-rule constructor changed the frozen decision method")
                events["late_rule_constructor"] = {
                    "source": str(Path(loaded_rule.__file__).resolve()),
                    "sha256": sha(Path(loaded_rule.__file__)),
                    "parameters": policy.parameters(),
                    "decide_inherited_unchanged": True}
                events["existing_cell_reuse"] = {
                    "count": len(reused), "keys_sha256": hashlib.sha256(
                        "\n".join(sorted(reused)).encode()).hexdigest(),
                    "no_provider_or_claim": True}
                state_holder["event"] = {
                    "status": "passed_before_first_late_rule_provider",
                    "model_rows": model_rows, "matrix_rows": len(matrix),
                    "scenes": list(scenes), "training_replayed": False,
                    "configured_controller_source": str(Path(legacy.v2_controller.__file__).resolve())}
                raise RestoreGateStop("actual CLI reused 608 cells and bound frozen late rule; stopped before provider")
            if logical not in existing_keys or kind != "eval":
                raise AssertionError(f"unexpected logical cell before first rule: {logical}")
            folder = RAW / "launch_claims" / logical
            complete = [path for path in folder.glob("attempt_*.json")
                        if json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"]
            if len(complete) != 1:
                raise AssertionError(f"missing or duplicate complete claim {logical}")
            row = saved_execute(label, logical, kind, scenario, metadata, **kwargs)
            if (int(row.get("scenario_seed", -1)) != int(scenario)
                    or int(row.get("attempt", -1)) != int(json.loads(
                        complete[0].read_text(encoding="utf-8"))["attempt"])):
                raise AssertionError(f"reused model row identity mismatch {logical}")
            reused.append(logical)
            return row

        run_module._execute_cell = observe_existing_only
        return "round58-readonly-existing-matrix/1"

    eval_handoff_gate._r58_readonly_matrix_gate = True

    sys.dont_write_bytecode = True
    builtins.open = guarded_open
    Path.open = guarded_path_open
    Path.write_bytes = guarded_write_bytes
    Path.write_text = guarded_write_text
    Path.touch = guarded_touch
    Path.mkdir = guarded_mkdir
    os.open = guarded_os_open
    os.makedirs = guarded_makedirs
    os.replace = lambda src, dst: deny("os.replace", f"{src} -> {dst}")
    Path.replace = lambda self, target: deny("Path.replace", f"{self} -> {target}")
    Path.rename = lambda self, target: deny("Path.rename", f"{self} -> {target}")
    torch.save = guarded_torch_save
    torch.load = counted_load
    torch.optim.Optimizer.step = guarded_step
    torch.optim.Adam.step = guarded_step
    random.setstate = count_rng_restore("python", original_py_setstate)
    np.random.set_state = count_rng_restore("numpy", original_np_setstate)
    torch.set_rng_state = count_rng_restore("torch_cpu", original_torch_setstate)
    torch.random.set_rng_state = count_rng_restore("torch_random", original_torch_random_setstate)
    torch.cuda.set_rng_state = count_rng_restore("torch_cuda", original_cuda_setstate)
    torch.cuda.set_rng_state_all = count_rng_restore("torch_cuda_all", original_cuda_setstate_all)
    sys.argv = [str(LAUNCHER), "--authorization", str(AUTH), "--freeze", str(FREEZE),
                "--cumulative-quota", str(QUOTA), "--scene-audit", str(SCENES),
                "--output-root", str(RAW), "--mode", "recover_first_batch_gate"]
    caught, error = None, None
    try:
        runpy.run_path(str(LAUNCHER), run_name="__main__",
                       init_globals={"_ROUND58_V33_EVAL_HANDOFF_HOOK": eval_handoff_gate})
    except RestoreGateStop as exc:
        caught = str(exc)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        events["traceback"] = traceback.format_exc()
    finally:
        sys.setprofile(original_profile)
        sys.settrace(original_trace)
        sys.argv, sys.dont_write_bytecode = original_argv, original_bytecode
        builtins.open, Path.open = original_open, original_path_open
        Path.write_text, Path.write_bytes = original_write_text, original_write_bytes
        Path.touch, Path.mkdir = original_touch, original_mkdir
        os.open, os.makedirs = original_os_open, original_os_makedirs
        os.replace, Path.replace, Path.rename = original_replace, original_path_replace, original_rename
        torch.save, torch.load = original_torch_save, original_torch_load
        torch.optim.Optimizer.step, torch.optim.Adam.step = original_opt_step, original_adam_step
        random.setstate, np.random.set_state = original_py_setstate, original_np_setstate
        torch.set_rng_state, torch.random.set_rng_state = original_torch_setstate, original_torch_random_setstate
        torch.cuda.set_rng_state, torch.cuda.set_rng_state_all = original_cuda_setstate, original_cuda_setstate_all

    protected_after = {str(path): sha(path) for path in PROTECTED}
    history_after = {str(path): sha(path) for path in (
        RAW / "ROUND46_FIRST_FOUR_s11.json", RAW / "ROUND46_FIRST_FOUR_s22.json",
        RAW / "ROUND46_FIRST_FOUR_s33.json",
        RUN_ROOT / "round58_gen3_resume/raw/runtime/checkpoint_writes.jsonl",
        RUN_ROOT / "round58_gen3_resume/raw/runtime/ROUND53_EXECUTION_LEDGER.json")}
    event = state_holder.get("event")
    success = bool(caught and event and not error and protected_before == protected_after
                   and history_before == history_after and not events["write_attempts"]
                   and events["optimizer_step_calls"] == 0 and events["torch_save_calls"] == 0
                   and events["environment_provider_denials"] == 0 and events["claim_denials"] == 0
                   and len(event.get("model_rows", [])) == 3 and event.get("matrix_rows") == 640)
    result = {"kind": "round58-v36-actual-cli-booking-gate/1",
              "status": "passed" if success else "failed", "event": event,
              "intentional_stop": caught, "unexpected_error": error,
              "traceback": events.get("traceback"), "booking_probe": events.get("booking_probe"),
              "native_rule_probe": events.get("native_rule_probe"),
              "late_rule_constructor": events.get("late_rule_constructor"),
              "existing_cell_reuse": events.get("existing_cell_reuse"), "completed_debug": events.get("completed_debug"), "debug_identity": events.get("debug_identity"),
              "side_effects": {"environment": 0, "claims": 0, "optimizer_steps": events["optimizer_step_calls"],
                               "checkpoint_payload_writes": events["torch_save_calls"],
                               "RNG_restore_calls": events["rng_restore_calls"],
                               "torch_load_calls": events["torch_load_calls"],
                               "empty_journal_memory_spy": events["empty_checkpoint_journal_spy"],
                               "derived_ledger_memory_spy": events["derived_ledger_spy"],
                               "write_attempts": events["write_attempts"]},
              "protected_sha256_before": protected_before,
              "protected_sha256_after": protected_after,
              "source_files_unchanged": history_before == history_after,
              "input_sha256": input_hashes}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not success:
        raise SystemExit(f"V36 actual CLI booking gate failed: {OUTPUT}")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
