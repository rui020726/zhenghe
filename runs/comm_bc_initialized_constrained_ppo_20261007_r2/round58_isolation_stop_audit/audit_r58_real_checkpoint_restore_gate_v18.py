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
RESUME_RAW = RUN_ROOT / "round58_gen3_resume_v18/raw"
LAUNCHER = HERE / "r58_private_launcher_repaired.py"
AUTH = HERE / "ROUND58_RESUME_EXECUTION_AUTHORIZATION_V18.json"
FREEZE = HERE / "ROUND58_RESUME_FREEZE_V18.json"
QUOTA = HERE / "ROUND58_GEN3_CUMULATIVE_QUOTA.json"
SCENES = HERE / "ROUND58_GEN3_RESUME_SCENE_OCCUPANCY_RECHECK.json"
PROOF = HERE / "ROUND58_GEN3_PLAN_IDENTITY_MIGRATION_PROOF_V18.json"
OUTPUT = HERE / "ROUND58_GEN3_REAL_RESTORE_GATE_V18.json"
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
    if not RESUME_RAW.is_dir():
        raise RuntimeError("V18 resume runtime-alias root is not provisioned")
    existing = {p.name for p in RESUME_RAW.iterdir() if p.is_file()}
    if existing != set(ALIAS_HASHES):
        raise RuntimeError(f"V16 resume root contains unexpected/missing files: {sorted(existing)}")
    for name, expected in ALIAS_HASHES.items():
        if sha(RESUME_RAW / name) != expected:
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
        if int(parsed.get("environment_consumed", -1)) != 28:
            deny("derived ledger historical consumption mismatch", target)
        events["derived_ledger_spy"].append({
            "path": str(target), "bytes": len(data.encode(kwargs.get("encoding") or "utf-8")),
            "historical_consumed": 28, "action": "memory_only_no_file_created"})
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

    def real_restore_gate(payload: dict[str, Any]) -> None:
        runtime = payload["runtime"]
        trainer = runtime["trainer"]
        if runtime["r40_train"].trainer is not trainer:
            raise AssertionError("configured R40 trainer object differs from actual loader trainer")
        if runtime["r58_checkpoint_contract"].get("wrapped_load") is not True:
            raise AssertionError("actual configured trainer loader is not the private checkpoint contract")
        if runtime["r58_repaired_adapter"].get("kind") != "r58-gate-prior-repaired/1":
            raise AssertionError("critic/prior adapter is not the configured live adapter")
        runner = runtime.get("runner")
        if runner is not None and hasattr(runner, "run_training_episode"):
            runner.run_training_episode = forbidden_runtime_call("episode_provider")
        run_module = runtime["r42_run"]
        for name in ("_claim", "claim_episode", "run_episode", "run_training_episode"):
            if hasattr(run_module, name):
                setattr(run_module, name, forbidden_runtime_call("claim" if "claim" in name else "episode_provider"))

        source_proof = payload["proof"]
        frozen = payload["freeze"]
        # Exercise the exact lazy Stage-B import through the configured import bridge.
        # This reads only the frozen archived configuration; it does not construct an environment.
        common_module = runtime["common"]
        stage_b_fields, stage_b_substitutions = common_module.stage_b_inherited_config(1300003)
        sp_common = sys.modules.get("sp_common")
        approved_source = json.loads(FREEZE.read_text(encoding="utf-8"))["repair_identity"]["stage_b_sp_common_source"]
        if (sp_common is None
                or Path(getattr(sp_common, "__file__", "")).resolve() != Path(approved_source["path"]).resolve()
                or sha(Path(sp_common.__file__)) != approved_source["sha256"]
                or stage_b_fields.get("seed") != 1300003
                or stage_b_fields.get("steps") != 1000):
            raise AssertionError("live Stage-B lazy import/config identity did not match V18 freeze")
        stage_b_binding = {"path": str(Path(sp_common.__file__).resolve()),
                           "sha256": sha(Path(sp_common.__file__)),
                           "binding_version": approved_source["binding_version"],
                           "seed": stage_b_fields["seed"], "steps": stage_b_fields["steps"],
                           "field_count": len(stage_b_fields),
                           "substitutions": stage_b_substitutions}
        summary = []
        for seed in (11, 22, 33):
            source = source_proof["sources"][seed]
            selected = Path(payload["selected_sources"][seed]).resolve()
            if selected != Path(source["path"]).resolve():
                raise AssertionError(f"seed {seed}: selected path differs from hash-closed resume map")
            raw = counted_load(selected, map_location="cpu", weights_only=False)
            controller, optimizer, state = trainer.load_checkpoint(selected, freeze=frozen)
            if not same(controller.state_dict(), raw.get("controller")):
                raise AssertionError(f"seed {seed}: restored controller tensors differ from source payload")
            if not same(optimizer.state_dict(), raw.get("optimizer")):
                raise AssertionError(f"seed {seed}: restored Adam state/moments differ from source payload")
            expected_state = raw.get("training_state")
            state_core = dict(state)
            state_core.pop("round58_resume_provenance", None)
            source_state = dict(expected_state)
            historical_identity = source.get("original_identity")
            current_identity = json.loads(FREEZE.read_text(encoding="utf-8")).get("repair_identity")
            if not isinstance(historical_identity, dict) or not isinstance(current_identity, dict):
                raise AssertionError(f"seed {seed}: missing historical/current repair identity evidence")
            if not same(source_state.get("round58_repair_attestation"), historical_identity):
                raise AssertionError(f"seed {seed}: source training-state attestation differs from authenticated historical sidecar")
            if not same(state_core.get("round58_repair_attestation"), current_identity):
                raise AssertionError(f"seed {seed}: restored state is not bound to current approved repair identity")
            source_state.pop("round58_repair_attestation", None)
            state_core.pop("round58_repair_attestation", None)
            state_mismatches = [key for key in source_state
                                if key not in state_core or not same(state_core[key], source_state[key])]
            if state_mismatches:
                raise AssertionError(f"seed {seed}: restored training-state source fields differ: {state_mismatches}")
            if (int(state.get("episodes_done", -1)), int(state.get("updates", -1)),
                    int(state.get("optimizer_steps", -1))) != (4, 1, 8):
                raise AssertionError(f"seed {seed}: restored cursor is not episode4/update1/step8")
            if float(state.get("lambda", state.get("lambda_cost", -1))) != 1.0:
                raise AssertionError(f"seed {seed}: restored λ is not the qualified value 1.0")
            groups = optimizer.state_dict().get("param_groups", [])
            if not groups or any(int(item.get("step", -1)) != 8
                                 for item in optimizer.state_dict().get("state", {}).values()):
                raise AssertionError(f"seed {seed}: Adam step/moments do not close at eight calls")
            binding = getattr(trainer, "_round58_train_pilot_binding", {})
            if (binding.get("path") != source_proof["manifest"].get("canonical_train_pilot_path")
                    and Path(binding.get("path", "")).resolve() !=
                    (REPO.parent / "zhenghe/runs/comm_budget_pilot_20260919/train_pilot.py").resolve()):
                raise AssertionError(f"seed {seed}: restored controller module path is not frozen canonical source")
            if binding.get("sha256") != "87dcf2e5251c99fc533ba52480a43da89902a3a0c371f5b9cfec6240d202ae81":
                raise AssertionError(f"seed {seed}: restored controller module hash is not frozen")
            if not same(trainer.rng_state(), raw.get("rng")):
                raise AssertionError(f"seed {seed}: restored Python/NumPy/Torch RNG differs from checkpoint")
            summary.append({
                "seed": seed, "source_checkpoint_sha256": source["checkpoint_sha256"],
                "selected_path": str(selected), "controller_exact": True,
                "optimizer_state_exact": True, "param_group_count": len(groups),
                "adam_state_entries": len(optimizer.state_dict().get("state", {})),
                "adam_steps": sorted({int(x.get("step", -1)) for x in
                                      optimizer.state_dict().get("state", {}).values()}),
                "lambda": float(state.get("lambda", state.get("lambda_cost", -1))),
                "history_and_cursor_exact": True,
                "historical_attestation_matches_source_sidecar": True,
                "restored_attestation_matches_current_freeze": True,
                "training_state_source_keys_exact": sorted(source_state),
                "training_state_additional_keys": sorted(set(state_core) - set(source_state)),
                "cursor": {key: state.get(key) for key in
                           ("episodes_done", "updates", "optimizer_steps")},
                "rng_exact": True, "train_pilot_binding": binding,
            })
        state_holder["event"] = {
            "kind": "round58-real-checkpoint-restore-gate/2",
            "status": "three_seed_exact_restore_passed_stopped_before_episode_provider",
            "seeds": summary,
            "selected_sources": payload["selected_sources"],
            "configured_trainer_is_r40_trainer": True,
            "critic_prior_adapter_live": True,
            "live_stage_b_sp_common_binding": stage_b_binding,
        }
        raise RestoreGateStop("all three actual checkpoints restored; stop before episode provider")

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
                       init_globals={"_ROUND58_REAL_RESTORE_GATE_HOOK": real_restore_gate})
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
                   and len(event.get("seeds", [])) == 3
                   and {x["seed"] for x in event["seeds"]} == {11, 22, 33})
    result = {
        "kind": "round58-gen3-real-restore-gate-report/2",
        "status": "passed" if success else "failed",
        "head": os.popen(f'git -C "{REPO}" rev-parse HEAD').read().strip(),
        "cli_route": "clean Python process, runpy actual private launcher __main__, V18 receipts and V18 migration proof; exact frozen sp_common binding is exercised by the live Stage-B configuration import; exact source aliases hash-provisioned in V16 root",
        "restore_gate": event,
        "intentional_stop": caught,
        "unexpected_error": error,
        "input_sha256": input_hashes,
        "side_effects": {
            "environment_constructions": 0, "episode_provider_calls": 0,
            "claims": 0, "optimizer_steps": 0, "PPO_updates": 0,
            "legacy_tape_replay": 0, "checkpoint_payload_writes": 0,
            "evaluation_cells": 0, "RNG_restore_calls": events["rng_restore_calls"],
            "torch_load_calls": events["torch_load_calls"],
            "torch_save_calls": events["torch_save_calls"],
            "empty_journal_memory_spy": events["empty_checkpoint_journal_spy"],
            "derived_ledger_memory_spy": events["derived_ledger_spy"],
            "suppressed_directory_requests": events["suppressed_directories"],
            "preprovisioned_runtime_aliases": {name: sha(RESUME_RAW / name) for name in sorted(ALIAS_HASHES)},
        },
        "protected_sha256_before": protected_before,
        "protected_sha256_after": protected_after,
        "source_and_v10_failure_files_unchanged": history_before == history_after,
        "process_rng_before": rng_before,
        "process_rng_after": rng_fingerprint(),
        "write_attempts": events["write_attempts"],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not success:
        raise SystemExit(f"actual R58 restore gate did not pass: {OUTPUT}")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
