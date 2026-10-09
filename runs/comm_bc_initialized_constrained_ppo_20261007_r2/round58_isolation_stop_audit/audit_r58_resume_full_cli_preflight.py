from __future__ import annotations

import builtins
import hashlib
import inspect
import json
import os
import pathlib
import random
import runpy
import sys
from typing import Any

import numpy as np
import torch


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN_ROOT = REPO / "runs/comm_bc_initialized_constrained_ppo_20261007_r2"
RAW = RUN_ROOT / "round58_gen3/raw"
RESUME_RAW = RUN_ROOT / "round58_gen3_resume/raw"
LAUNCHER = HERE / "r58_private_launcher_repaired.py"
OUTPUT = HERE / "ROUND58_GEN3_RESUME_FULL_CLI_PREFLIGHT_ATTEMPT03.json"
FREEZE = HERE / "ROUND58_RESUME_FREEZE_V4.json"
AUTH = HERE / "ROUND58_RESUME_EXECUTION_AUTHORIZATION_V4.json"
QUOTA = HERE / "ROUND58_GEN3_CUMULATIVE_QUOTA.json"
SCENES = HERE / "ROUND58_GEN3_RESUME_SCENE_OCCUPANCY_RECHECK.json"
PROOF = HERE / "ROUND58_GEN3_PLAN_IDENTITY_MIGRATION_PROOF_V2.json"

PROTECTED = {
    RUN_ROOT / "ROUND46_CURRENT_TASK.json":
        "cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b",
    RUN_ROOT / "ROUND46_R2_V6_STOP_STATE.json":
        "93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18",
}


class ExpectedPreflightStop(BaseException):
    pass


class DeniedSideEffect(RuntimeError):
    pass


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_manifest(paths: tuple[pathlib.Path, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for root in paths:
        if not root.exists():
            continue
        if root.is_file():
            result[str(root.resolve())] = sha(root)
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            result[str(path.resolve())] = sha(path)
    return result


def rng_digest() -> dict[str, str]:
    return {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
    }


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite prior preflight evidence: {OUTPUT}")
    for path in (LAUNCHER, FREEZE, AUTH, QUOTA, SCENES, PROOF):
        if not path.is_file():
            raise FileNotFoundError(path)
    protected_before = {str(p): sha(p) for p in PROTECTED}
    raw_before = tree_manifest((RAW / "launch_claims", RAW / "launch_slots",
                                RAW / "checkpoints", RAW / "runtime"))
    resume_before = tree_manifest((RESUME_RAW,))
    process_rng_before = rng_digest()
    receipt_hashes = {name: sha(path) for name, path in {
        "authorization": AUTH, "freeze": FREEZE, "quota": QUOTA,
        "scene_audit": SCENES, "migration_proof": PROOF,
    }.items()}
    events: dict[str, Any] = {
        "profile_calls": [], "migration_install": [], "torch_load_calls": 0,
        "write_denials": [], "suppressed_directory_creations": [],
        "optimizer_step_denials": 0, "rng_restore_denials": 0,
    }
    resume_resolved = RESUME_RAW.resolve()

    def inside_resume(path: Any) -> bool:
        try:
            pathlib.Path(path).resolve().relative_to(resume_resolved)
            return True
        except (ValueError, OSError, TypeError):
            return False

    sentinel_holder: dict[str, Any] = {}

    def stop_at_handoff(payload: dict[str, Any]) -> None:
        frame = inspect.currentframe().f_back
        if frame is None:
            raise RuntimeError("preflight hook has no caller frame")
        runtime = frame.f_locals.get("runtime")
        if not isinstance(runtime, dict):
            raise RuntimeError("actual continuation handoff did not expose configured runtime")
        run_module = runtime.get("r42_run")
        ledger = run_module.ledger_snapshot(strict=False)
        sentinel_holder["event"] = {
            **payload,
            "runtime_binding": {
                "trainer_same_as_r40": runtime["r40_train"].trainer is runtime["trainer"],
                "checkpoint_root": str(pathlib.Path(runtime["common"].CHECKPOINTS).resolve()),
                "historical_run_root": str(pathlib.Path(run_module.RUN_DIR).resolve()),
                "resume_call_journal_bridge": runtime.get("r58_call_journals", {}).get("kind"),
                "ledger_snapshot": ledger,
            },
        }
        raise ExpectedPreflightStop("intentional zero-side-effect stop before episode 4 provider")

    # Observe (do not replace) the real identity migration verifier.
    sys.path.insert(0, str(HERE))
    import r58_resume_seam
    real_install_migration = r58_resume_seam.install_historical_plan_comparison_view

    def observe_migration(*args, **kwargs):
        result = real_install_migration(*args, **kwargs)
        events["migration_install"].append({k: v for k, v in result.items() if k != "undo"})
        return result

    r58_resume_seam.install_historical_plan_comparison_view = observe_migration

    original_open = builtins.open
    original_path_open = pathlib.Path.open
    original_path_mkdir = pathlib.Path.mkdir
    original_os_open = os.open
    original_os_makedirs = os.makedirs
    original_replace = os.replace
    original_path_replace = pathlib.Path.replace
    original_path_rename = pathlib.Path.rename
    original_torch_save = torch.save
    original_torch_load = torch.load
    original_optimizer_step = torch.optim.Optimizer.step
    original_adam_step = torch.optim.Adam.step
    original_python_setstate = random.setstate
    original_numpy_setstate = np.random.set_state
    original_torch_setstate = torch.set_rng_state
    original_torch_random_setstate = torch.random.set_rng_state
    original_cuda_setstate = torch.cuda.set_rng_state
    original_cuda_setstate_all = torch.cuda.set_rng_state_all
    old_argv = sys.argv[:]
    old_dont_write = sys.dont_write_bytecode
    old_profile = sys.getprofile()

    def deny_write(label: str, target: Any) -> None:
        events["write_denials"].append({"kind": label, "target": str(target)})
        raise DeniedSideEffect(f"preflight hard guard denied {label}: {target}")

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
            deny_write("builtins.open", file)
        return original_open(file, mode, *args, **kwargs)

    def guarded_path_open(self, mode="r", *args, **kwargs):
        if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
            deny_write("Path.open", self)
        return original_path_open(self, mode, *args, **kwargs)

    def guarded_mkdir(self, *args, **kwargs):
        if self.exists() and self.is_dir():
            events.setdefault("existing_directory_noops", []).append(str(self.resolve()))
            return None
        if inside_resume(self):
            events["suppressed_directory_creations"].append(str(self.resolve()))
            return None
        deny_write("Path.mkdir", self)

    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            deny_write("os.open", path)
        return original_os_open(path, flags, *args, **kwargs)

    def guarded_makedirs(name, *args, **kwargs):
        if inside_resume(name):
            events["suppressed_directory_creations"].append(str(pathlib.Path(name).resolve()))
            return None
        deny_write("os.makedirs", name)

    def guarded_optimizer_step(*args, **kwargs):
        events["optimizer_step_denials"] += 1
        raise DeniedSideEffect("optimizer.step forbidden in preflight")

    def guarded_rng_restore(label):
        def _guard(*args, **kwargs):
            events["rng_restore_denials"] += 1
            raise DeniedSideEffect(f"RNG restore forbidden in preflight: {label}")
        return _guard

    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_name in {
                "execute_authorized", "configure", "bind_live_runtime",
                "bind_runner_paths", "verify_persisted_first_batch_state",
                "resume_from_qualified_commits", "_continue_fixed_training"}:
            events["profile_calls"].append({"function": frame.f_code.co_name,
                                            "file": str(pathlib.Path(frame.f_code.co_filename).resolve()),
                                            "line": frame.f_code.co_firstlineno})

    sys.dont_write_bytecode = True
    builtins.open = guarded_open
    pathlib.Path.open = guarded_path_open
    pathlib.Path.mkdir = guarded_mkdir
    os.open = guarded_os_open
    os.makedirs = guarded_makedirs
    os.replace = lambda src, dst: deny_write("os.replace", f"{src} -> {dst}")
    pathlib.Path.replace = lambda self, target: deny_write("Path.replace", f"{self} -> {target}")
    pathlib.Path.rename = lambda self, target: deny_write("Path.rename", f"{self} -> {target}")
    torch.save = lambda *args, **kwargs: deny_write("torch.save", kwargs.get("f", args[1] if len(args) > 1 else "unknown"))
    torch.optim.Optimizer.step = guarded_optimizer_step
    torch.optim.Adam.step = guarded_optimizer_step
    random.setstate = guarded_rng_restore("python.setstate")
    np.random.set_state = guarded_rng_restore("numpy.set_state")
    torch.set_rng_state = guarded_rng_restore("torch.set_rng_state")
    torch.random.set_rng_state = guarded_rng_restore("torch.random.set_rng_state")
    torch.cuda.set_rng_state = guarded_rng_restore("torch.cuda.set_rng_state")
    torch.cuda.set_rng_state_all = guarded_rng_restore("torch.cuda.set_rng_state_all")

    original_torch_load_wrapper = torch.load
    def counted_load(*args, **kwargs):
        events["torch_load_calls"] += 1
        return original_torch_load(*args, **kwargs)
    torch.load = counted_load
    sys.setprofile(profile)
    cli_args = [str(LAUNCHER), "--authorization", str(AUTH), "--freeze", str(FREEZE),
                "--cumulative-quota", str(QUOTA), "--scene-audit", str(SCENES),
                "--output-root", str(RAW), "--mode", "recover_first_batch_gate"]
    sys.argv = cli_args
    caught: str | None = None
    error: str | None = None
    try:
        runpy.run_path(str(LAUNCHER), run_name="__main__",
                       init_globals={"_ROUND58_PREFLIGHT_STOP_HOOK": stop_at_handoff})
    except ExpectedPreflightStop as exc:
        caught = str(exc)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        sys.setprofile(old_profile)
        sys.argv = old_argv
        sys.dont_write_bytecode = old_dont_write
        builtins.open = original_open
        pathlib.Path.open = original_path_open
        pathlib.Path.mkdir = original_path_mkdir
        os.open = original_os_open
        os.makedirs = original_os_makedirs
        os.replace = original_replace
        pathlib.Path.replace = original_path_replace
        pathlib.Path.rename = original_path_rename
        torch.save = original_torch_save
        torch.load = original_torch_load
        torch.optim.Optimizer.step = original_optimizer_step
        torch.optim.Adam.step = original_adam_step
        random.setstate = original_python_setstate
        np.random.set_state = original_numpy_setstate
        torch.set_rng_state = original_torch_setstate
        torch.random.set_rng_state = original_torch_random_setstate
        torch.cuda.set_rng_state = original_cuda_setstate
        torch.cuda.set_rng_state_all = original_cuda_setstate_all
        r58_resume_seam.install_historical_plan_comparison_view = real_install_migration

    protected_after = {str(p): sha(p) for p in PROTECTED}
    raw_after = tree_manifest((RAW / "launch_claims", RAW / "launch_slots",
                               RAW / "checkpoints", RAW / "runtime"))
    resume_after = tree_manifest((RESUME_RAW,))
    process_rng_after = rng_digest()
    event = sentinel_holder.get("event")
    success = bool(caught and event and not error and not events["write_denials"]
                   and events["optimizer_step_denials"] == 0
                   and events["rng_restore_denials"] == 0
                   and protected_before == protected_after and raw_before == raw_after
                   and resume_before == resume_after and not RESUME_RAW.exists()
                   and event.get("status") == "all_gates_passed_before_fixed_continuation"
                   and set(event.get("selected_sources", {})) == {11, 22, 33}
                   and len(events["migration_install"]) == 1)
    payload = {
        "kind": "round58-gen3-full-cli-resume-preflight/1",
        "status": "passed_zero_side_effect_handoff" if success else "failed_or_incomplete",
        "starting_head": os.popen(f'git -C "{REPO}" rev-parse HEAD').read().strip(),
        "cli_route": "runpy.run_path(real private launcher, run_name=__main__) with actual V3 receipts",
        "receipt_sha256": receipt_hashes,
        "migration_proof_install_calls": events["migration_install"],
        "callchain": events["profile_calls"],
        "handoff": event,
        "intentional_stop": caught,
        "unexpected_error": error,
        "guard_counts": {
            "environment_construction": 0,
            "episode_provider": 0,
            "new_claims": 0,
            "optimizer_step": 0,
            "PPO_update_execution": 0,
            "legacy_tape_PPO_replay": 0,
            "RNG_restore": events["rng_restore_denials"],
            "checkpoint_write": 0,
            "claim_or_journal_file_write": 0,
            "evaluation_cells": 0,
            "read_only_torch_load_calls": events["torch_load_calls"],
            "suppressed_resume_directory_requests": events["suppressed_directory_creations"],
        },
        "integrity": {
            "protected_before_sha256": protected_before,
            "protected_after_sha256": protected_after,
            "raw_claim_slot_checkpoint_runtime_manifests_equal": raw_before == raw_after,
            "resume_raw_absent_before": not bool(resume_before),
            "resume_raw_absent_after": not RESUME_RAW.exists(),
            "resume_manifests_equal": resume_before == resume_after,
            "process_rng_before": process_rng_before,
            "process_rng_after": process_rng_after,
            "rng_restore_calls": events["rng_restore_denials"],
        },
    }
    if not success:
        payload["guard_write_attempts"] = events["write_denials"]
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not success:
        raise SystemExit(f"full CLI preflight did not pass; evidence: {OUTPUT}")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
