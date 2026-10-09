"""Run the real R42 first-batch gate read-only against the three gen3 batches."""
from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import PureWindowsPath
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN_ROOT = HERE.parents[0]
GEN3_RAW = RUN_ROOT / "round58_gen3" / "raw"
AUDIT_FILE = HERE / "ROUND58_GEN3_FREEZE.json"
SOURCE_HEAD = "3f1a721e6414898c7ccdd0e786b0bcfef97dd5f8"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_historical_source_attestation(freeze: dict[str, Any]) -> dict[str, Any]:
    identity = freeze["repair_identity"]
    source_map = {
        "adapter_sha256": HERE / "r58_gate_prior_repaired.py",
        "checkpoint_contract_sha256": HERE / "r58_checkpoint_contract_repaired.py",
        "candidate_config_sha256": HERE / "ROUND58_PRELAUNCH_CANDIDATE.json",
        "candidate_input_identity_sha256": HERE / "ROUND58_CANDIDATE_INPUT_IDENTITY.json",
        "execution_basis_sha256": HERE / "ROUND58_GEN3_EXECUTION_BASIS.json",
    }
    actual = {}
    for key, path in source_map.items():
        value = sha256(path)
        actual[key] = {"path": str(path), "sha256": value,
                       "matches_attestation": value == identity[key]}
    # The private launcher was amended in this audit to bridge the PPO counter.
    # Verify the old, actually attested bytes from the pre-audit source commit.
    launcher_rel = HERE.relative_to(REPO) / "r58_private_launcher_repaired.py"
    old = subprocess.run(["git", "show", f"{SOURCE_HEAD}:{launcher_rel.as_posix()}"],
                         cwd=REPO, capture_output=True, check=True).stdout
    old_hash = hashlib.sha256(old).hexdigest()
    actual["private_launcher_sha256"] = {
        "path": f"git:{SOURCE_HEAD}:{launcher_rel.as_posix()}",
        "sha256": old_hash, "matches_attestation": old_hash == identity["private_launcher_sha256"]}
    current_launcher_hash = sha256(HERE / "r58_private_launcher_repaired.py")
    actual["private_launcher_current"] = {
        "path": str(HERE / "r58_private_launcher_repaired.py"),
        "sha256": current_launcher_hash,
        "matches_historical_attestation": current_launcher_hash == identity["private_launcher_sha256"]}
    canon = REPO.parent / "zhenghe" / "runs" / "comm_cost_v2_rerun_20260928"
    for key, path in (("trainer_sha256", canon / "v2_trainer.py"),
                      ("common_sha256", canon / "v2_common.py")):
        value = sha256(path)
        actual[key] = {"path": str(path), "sha256": value,
                       "matches_attestation": value == identity[key]}
    seed11_path = GEN3_RAW / "checkpoints" / "v2_train_s11_before_b0.pt"
    payload_identity = torch.load(seed11_path, map_location="cpu", weights_only=False)[
        "round58_repair_attestation"]
    differing = [key for key in set(identity) | set(payload_identity)
                 if identity.get(key) != payload_identity.get(key)]
    canonical = copy.deepcopy(identity)
    path_exception = None
    if differing == ["adapter_path"] or set(differing) == {"adapter_path"}:
        expected_name = PureWindowsPath(identity["adapter_path"]).name
        actual_path = Path(payload_identity["adapter_path"]).resolve()
        actual_digest = sha256(actual_path)
        if (expected_name != "r58_gate_prior_repaired.py"
                or actual_path != (HERE / "r58_gate_prior_repaired.py").resolve()
                or actual_digest != identity["adapter_sha256"]):
            raise RuntimeError("repair attestation adapter_path exception failed basename/path/hash binding")
        # Preserve the exact serialized path spelling after its location and
        # bytes have been proven; resolving Unicode paths through the Windows
        # console code page can itself produce a different Python string.
        canonical["adapter_path"] = payload_identity["adapter_path"]
        path_exception = {"field": "adapter_path", "freeze_value_ascii": ascii(identity["adapter_path"]),
            "checkpoint_value": str(actual_path), "reason": "freeze receipt contains a path encoding mismatch",
            "accepted_only_after": ["exact basename", "exact registered adapter SHA256",
                                    "resolved checkpoint path equals repository adapter"]}
    elif differing:
        raise RuntimeError(f"repair identity payload differs from freeze beyond registered path normalization: {differing}")
    return {"identity": identity, "canonical_identity": canonical,
            "payload_identity_differing_fields": differing,
            "path_metadata_exception": path_exception,
            "sources": actual,
            "all_historical_sources_match": all(
                row["matches_attestation"] for key, row in actual.items()
                if key != "private_launcher_current")}


def _rng_snapshot() -> dict[str, Any]:
    py = random.getstate()
    np_state = np.random.get_state()
    return {"python": py, "numpy_name": np_state[0], "numpy_keys": np_state[1].copy(),
            "numpy_pos": np_state[2:], "torch_cpu": torch.get_rng_state().clone()}


def _same_rng(a: dict[str, Any], b: dict[str, Any]) -> dict[str, bool]:
    return {"python_unchanged": a["python"] == b["python"],
            "numpy_unchanged": (a["numpy_name"] == b["numpy_name"]
                                and np.array_equal(a["numpy_keys"], b["numpy_keys"])
                                and a["numpy_pos"] == b["numpy_pos"]),
            "torch_cpu_unchanged": torch.equal(a["torch_cpu"], b["torch_cpu"])}


def _install_zero_side_effect_guards(runtime: dict[str, Any], counters: dict[str, int]):
    from unittest import mock
    stack = []

    def deny(name):
        def blocked(*_args, **_kwargs):
            counters[name] += 1
            raise RuntimeError(f"zero-side-effect guard rejected {name}")
        return blocked

    stack.append(mock.patch.object(torch.optim.Optimizer, "step", deny("optimizer_step")))
    if hasattr(torch.optim.Adam, "step"):
        stack.append(mock.patch.object(torch.optim.Adam, "step", deny("optimizer_step")))
    stack.append(mock.patch.object(torch, "save", deny("checkpoint_write")))
    stack.append(mock.patch.object(random, "setstate", deny("rng_restore")))
    stack.append(mock.patch.object(np.random, "set_state", deny("rng_restore")))
    stack.append(mock.patch.object(torch, "set_rng_state", deny("rng_restore")))
    if torch.cuda.is_available():
        stack.append(mock.patch.object(torch.cuda, "set_rng_state", deny("rng_restore")))
        stack.append(mock.patch.object(torch.cuda, "set_rng_state_all", deny("rng_restore")))

    root = GEN3_RAW.resolve()
    original_open = Path.open
    original_write_text, original_write_bytes, original_touch = (
        Path.write_text, Path.write_bytes, Path.touch)
    original_builtin_open = builtins.open
    path_type = type(Path("."))
    original_resolve = path_type.resolve

    def under_gen3(path) -> bool:
        try:
            Path(path).resolve().relative_to(root)
            return True
        except (ValueError, OSError, TypeError):
            return False

    def guarded_path_open(path, mode="r", *args, **kwargs):
        if under_gen3(path) and any(flag in mode for flag in ("w", "a", "x", "+")):
            return deny("claim_journal_checkpoint_write")(path, mode, *args, **kwargs)
        return original_open(path, mode, *args, **kwargs)

    def guarded_builtin_open(file, mode="r", *args, **kwargs):
        if under_gen3(file) and any(flag in mode for flag in ("w", "a", "x", "+")):
            return deny("claim_journal_checkpoint_write")(file, mode, *args, **kwargs)
        return original_builtin_open(file, mode, *args, **kwargs)

    def guarded_write(method, method_name):
        def call(path, *args, **kwargs):
            if under_gen3(path):
                return deny("claim_journal_checkpoint_write")(path, *args, **kwargs)
            return method(path, *args, **kwargs)
        return call

    stack += [mock.patch.object(Path, "open", guarded_path_open),
              mock.patch.object(Path, "write_text", guarded_write(original_write_text, "write_text")),
              mock.patch.object(Path, "write_bytes", guarded_write(original_write_bytes, "write_bytes")),
              mock.patch.object(Path, "touch", guarded_write(original_touch, "touch")),
              mock.patch.object(builtins, "open", guarded_builtin_open),
              mock.patch.object(os, "replace", deny("claim_journal_checkpoint_write"))]
    # CPython's Windows _getfinalpathname raises WinError 5 under this managed
    # workspace for Unicode paths. For exact, hash-verified read-only frozen
    # inputs, a lexical absolute path preserves the root/hash predicate; symlinks
    # are explicitly rejected rather than dereferenced.
    frozen_files = {Path(row["path"]).absolute()
                    for row in runtime.get("_r58_audit_frozen_path_records", [])}
    frozen_root = Path(Path.cwd().parent / "zhenghe").absolute()
    def lexical_resolve(path, strict=False):
        try:
            return original_resolve(path, strict=strict)
        except PermissionError:
            lexical = Path(os.path.abspath(str(path)))
            if lexical == frozen_root or lexical in frozen_files:
                if strict and not lexical.exists():
                    raise FileNotFoundError(str(lexical))
                cursor = lexical
                while cursor != cursor.parent:
                    if cursor.is_symlink():
                        raise RuntimeError(f"refusing symlink in frozen read-only path: {cursor}")
                    cursor = cursor.parent
                return lexical
            raise
    stack.append(mock.patch.object(path_type, "resolve", lexical_resolve))
    # Environment construction has no role in this gate. Guard the actual R42
    # constructor object when present, and the R40 training entry unconditionally.
    r42_run = runtime["r42_run"]
    if hasattr(r42_run, "JointRun"):
        stack.append(mock.patch.object(r42_run, "JointRun", deny("environment_construct")))
    r40 = runtime["r40_train"]
    stack.append(mock.patch.object(r40, "train_seed", deny("episode_provider")))
    for patcher in stack:
        patcher.start()
    return stack


def _stop_guards(stack):
    for patcher in reversed(stack):
        patcher.stop()


def _validate_serialized_identity(runtime, frozen_gen3, seed, identity_module, base_freeze):
    ckpt_dir = GEN3_RAW / "checkpoints"
    record_files = sorted(ckpt_dir.glob(f"r40_batch_s{seed}_b0_attempt*.json"))
    if len(record_files) != 1:
        raise RuntimeError(f"seed {seed}: expected exactly one batch0 record, found {len(record_files)}")
    record_path = record_files[0]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    commit_path = ckpt_dir / f"r40_batch_commit_s{seed}_b0.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if commit.get("batch_record") != record_path.name or commit.get("batch_record_sha256") != sha256(record_path):
        raise RuntimeError(f"seed {seed}: commit does not bind exact batch record")
    if record.get("episodes") != [0, 1, 2, 3] or record.get("training_seed") != seed:
        raise RuntimeError(f"seed {seed}: batch0 identity/episode range mismatch")
    if record.get("training_version") != frozen_gen3["training_version"]:
        raise RuntimeError(f"seed {seed}: batch record training version mismatch")
    base = base_freeze
    results = []
    for role, name in (("pre", record["preupdate_checkpoint"]),
                       ("post", record["postupdate_checkpoint"])):
        path = ckpt_dir / name
        expected_hash = (record["preupdate_checkpoint_sha256"] if role == "pre"
                         else commit["postupdate_checkpoint_sha256"])
        actual_hash = sha256(path)
        if expected_hash != actual_hash:
            raise RuntimeError(f"seed {seed}: {role} checkpoint hash mismatch")
        sidecar_path = path.with_name(path.name + ".r58.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (sidecar.get("kind") != "round58-checkpoint-sidecar/1"
                or sidecar.get("checkpoint_name") != path.name
                or sidecar.get("checkpoint_sha256") != actual_hash
                or sidecar.get("training_version") != frozen_gen3["training_version"]
                or sidecar.get("round58_repair_attestation") != frozen_gen3["repair_identity"]):
            raise RuntimeError(f"seed {seed}: {role} sidecar identity/hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = {"training_version": frozen_gen3["training_version"],
                    "action_semantics": "one-shot-send-skip/1",
                    "base_freeze": base,
                    "repair_identity": frozen_gen3["repair_identity"]}
        differences = identity_module.identity_differences(payload, **expected)
        if differences:
            raise RuntimeError(f"seed {seed}: {role} identity mismatch: {differences}")
        if payload.get("comm_action_eligibility") != "first-report-only/1":
            raise RuntimeError(f"seed {seed}: {role} action eligibility mismatch")
        results.append({"role": role, "path": str(path), "sha256": actual_hash,
                        "sidecar": str(sidecar_path), "sidecar_sha256": sha256(sidecar_path),
                        "payload_training_version": payload.get("training_version"),
                        "action_semantics": payload.get("comm_action_semantics"),
                        "attestation_match": True,
                        "base_freeze_exact_after_metadata_normalization": True,
                        "optimizer_state_present": "optimizer" in payload,
                        "training_state_cursor": {k: payload["training_state"].get(k)
                            for k in ("episodes_done", "updates", "optimizer_steps", "training_seed")}})
    return {"batch_record": str(record_path), "batch_record_sha256": sha256(record_path),
            "commit_receipt": str(commit_path), "commit_sha256": sha256(commit_path),
            "record_commit_binding": True, "episodes": record["episodes"],
            "recorded_optimizer_steps": record.get("update", {}).get("optimizer_steps"),
            "checkpoint_identity": results}


def _verify_first12_claim_slot_records() -> dict[str, Any]:
    claims = []
    for path in sorted((GEN3_RAW / "launch_claims").glob("new_training_s*/attempt_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        claims.append((path, row))
    slots = []
    for path in sorted((GEN3_RAW / "launch_slots" / "normal").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if str(row.get("logical_job", "")).startswith("new_training_s"):
            slots.append((path, row))
    expected = {f"new_training_s{seed}_e{episode}"
                for seed in (11, 22, 33) for episode in range(4)}
    claim_keys = {(row.get("logical_job"), int(row.get("attempt", 0)))
                  for _path, row in claims}
    slot_map = {(row.get("logical_job"), int(row.get("attempt", 0))): row
                for _path, row in slots}
    if (len(claims) != 12 or len(slots) != 12
            or {key[0] for key in claim_keys} != expected or len(claim_keys) != 12
            or set(slot_map) != claim_keys):
        raise RuntimeError("first-12 launch claims/normal slots are not one-to-one")
    invalid = []
    for path, claim in claims:
        key = (claim.get("logical_job"), int(claim.get("attempt", 0)))
        slot = slot_map[key]
        if (claim.get("status") != "complete" or claim.get("category") != "normal"
                or claim.get("round45_attempt_kind") != "normal"
                or int(claim.get("environment_constructor_calls", 0)) != 1
                or int(claim.get("run_invocation_calls", 0)) != 1
                or claim.get("simulation_started") is not True
                or int(claim.get("launch_slot", -1)) != int(slot.get("slot", -2))
                or slot.get("category") != "normal"):
            invalid.append(str(path))
    if invalid:
        raise RuntimeError(f"historical first-12 claim/slot validation failed: {invalid}")
    return {"claim_count": len(claims), "slot_count": len(slots),
            "one_to_one": True, "episodes_by_seed": {str(seed): [0, 1, 2, 3]
                for seed in (11, 22, 33)},
            "claim_files": [{"path": str(path), "sha256": sha256(path),
                "logical_job": row["logical_job"], "slot": row["launch_slot"],
                "category": row["category"], "status": row["status"]}
                for path, row in claims],
            "slot_files": [{"path": str(path), "sha256": sha256(path),
                "logical_job": row["logical_job"], "slot": row["slot"],
                "category": row["category"]} for path, row in slots]}


def _install_no_rng_restore_readonly_loader(runtime):
    """Split read/validation from the legacy ambient-RNG-preserving wrapper.

    The actual gate calls trainer.load_checkpoint_read_only. Its configured
    R42 wrapper restores ambient RNG in a finally block, which this task forbids.
    We preserve the canonical base read-only loader and all its freeze/model
    validation, while omitting only that ambient-state restore operation.
    Sidecar/hash/attestation validation has already run independently.
    """
    trainer = runtime["trainer"]
    current = trainer.load_checkpoint_read_only
    def captures(fn):
        return dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
    contract_cells = captures(current)
    r42_loader = contract_cells.get("load_ro0")
    if r42_loader is None:
        raise RuntimeError("live checkpoint contract lacks its R42 read-only delegate")
    r42_cells = captures(r42_loader.cell_contents)
    base_cell = r42_cells.get("base_read_only")
    if base_cell is None or not callable(base_cell.cell_contents):
        raise RuntimeError("R42 read-only loader lacks the canonical base read-only function")
    base_read_only = base_cell.cell_contents
    COMM_ACTION_SEMANTICS = runtime["r42_versions"].COMM_ACTION_SEMANTICS

    def no_rng_restore_read_only(path, *, freeze=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("training_state")
        semantics = payload.get("comm_action_semantics")
        state_semantics = state.get("comm_action_semantics") if isinstance(state, dict) else None
        if semantics is not None and semantics != COMM_ACTION_SEMANTICS:
            raise RuntimeError(f"{Path(path).name}: unexpected action semantics in read-only load")
        if state_semantics is not None and state_semantics != COMM_ACTION_SEMANTICS:
            raise RuntimeError(f"{Path(path).name}: training_state action semantics mismatch")
        selected = freeze
        if selected is None:
            selected = payload.get("freeze")
            if not isinstance(selected, dict):
                raise RuntimeError(f"{Path(path).name}: checkpoint has no embedded freeze identity")
        controller, loaded_state, identity = base_read_only(path, freeze=selected)
        identity.update({
            "comm_action_semantics": semantics,
            "action_semantics_interpretation": COMM_ACTION_SEMANTICS,
            "read_only_for_round40": True, "read_only_for_round42": True,
            "read_only_for_round44": True, "optimizer_restored": False,
            "rng_restored": False, "ambient_rng_preserved": False,
            "audit_rng_restore_suppressed": True,
        })
        return controller, loaded_state, identity
    trainer.load_checkpoint_read_only = no_rng_restore_read_only
    return {"original_loader": current, "base_read_only": base_read_only,
            "replacement": no_rng_restore_read_only,
            "contract_delegate_identity": id(r42_loader.cell_contents),
            "base_loader_identity": id(base_read_only)}


def _normalize_frozen_input_paths_for_gate(frozen: dict[str, Any]):
    """Repair only broken path spellings, retaining the actual frozen SHA/size checks."""
    result = copy.deepcopy(frozen)
    # cwd is supplied by the shell as a Unicode-correct repository path; __file__
    # can inherit a console-code-page spelling on Windows.
    trusted_root = Path.cwd().parent / "zhenghe"
    if not trusted_root.is_dir():
        raise RuntimeError("canonical frozen input root is not present at the trusted sibling path")
    root_text = str(result["execution_identity"]["canonical_read_only_input_root"])
    if PureWindowsPath(root_text).name != "zhenghe":
        raise RuntimeError("unexpected frozen canonical root identity")
    result["execution_identity"]["canonical_read_only_input_root"] = str(trusted_root)
    path_results = []
    for name, record in result["round42"]["frozen_search_allocation_checkpoints"].items():
        parts = PureWindowsPath(record["path"]).parts
        try:
            index = parts.index("runs")
        except ValueError as exc:
            raise RuntimeError(f"frozen checkpoint path lacks the registered runs suffix: {name}") from exc
        candidate = trusted_root.joinpath(*parts[index:]).absolute()
        if not candidate.is_file():
            raise RuntimeError(f"frozen checkpoint is not readable at its trusted lexical path: {name}")
        got_sha, got_size = sha256(candidate), candidate.stat().st_size
        if got_sha != record["sha256"] or got_size != int(record["size_bytes"]):
            raise RuntimeError(f"frozen checkpoint bytes/size differ for {name}")
        path_results.append({"name": name, "path": str(candidate), "sha256": got_sha,
                             "size_bytes": got_size, "matches_frozen": True,
                             "normalization": "path only; exact original SHA256 and size retained"})
        record["path"] = str(candidate)
    return result, path_results


def run() -> dict[str, Any]:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HERE))
    launcher = load_module(HERE / "r58_private_launcher_repaired.py", "r58_private_launcher_gate_audit")
    identity_module = load_module(HERE / "r58_identity_ledger_close_v2.py", "r58_identity_ledger_gate_audit")
    frozen_gen3 = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    source_audit = verify_historical_source_attestation(frozen_gen3)
    if not source_audit["all_historical_sources_match"]:
        raise RuntimeError("historical checkpoint attestation source hashes do not all match")
    frozen_gen3["repair_identity"] = source_audit["canonical_identity"]
    results = {"kind": "round58-gen3-real-first-batch-gate-close/1",
               "status": "running", "starting_head": SOURCE_HEAD,
               "gate_source": {}, "source_attestation": source_audit,
               "source_attestation_checkpoint_deserializations": 1,
               "pre_gate_checkpoint_deserializations": 0,
               "runtime_setup_checkpoint_deserializations": 0,
               "seeds": [], "side_effect_totals": {
                   "environment_construct": 0, "episode_provider": 0,
                   "claim_journal_checkpoint_write": 0, "optimizer_step": 0,
                   "PPO_update_execution": 0, "PPO_noop_spy_invocations": 0,
                   "RNG_restore": 0, "checkpoint_write": 0,
                   "evaluation_cells": 0, "new_claims": 0,
                   "RNG_state_restores": 0, "RNG_state_changes": 0,
                   "episode_tapes_replayed": 0}}
    results["historical_first12_claim_slot_recheck"] = _verify_first12_claim_slot_records()
    with tempfile.TemporaryDirectory(dir=launcher.GEN2) as temp:
        raw_temp = Path(temp) / "audit_raw"
        setup_loads = {"count": 0}
        memory_writer_calls = []
        def memory_writer(path, rows):
            memory_writer_calls.append({"path": str(path), "row_count": len(rows)})
        original_setup_torch_load = torch.load
        def count_setup_load(*args, **kwargs):
            setup_loads["count"] += 1
            return original_setup_torch_load(*args, **kwargs)
        torch.load = count_setup_load
        try:
            binding = launcher.prepare_actual_runtime_for_audit(raw_temp, writer=memory_writer)
        finally:
            torch.load = original_setup_torch_load
        results["runtime_setup_checkpoint_deserializations"] = setup_loads["count"]
        runtime = binding["runtime"]
        analysis = runtime["r42_analysis"]
        results["gate_source"] = {"module": analysis.__name__,
            "path": str(Path(analysis.__file__).resolve()), "sha256": sha256(Path(analysis.__file__)),
            "function": "first_batch_gate", "callable_identity": id(analysis.first_batch_gate)}
        # The configured live freeze is the exact approved R46 science freeze.
        # The gen3 serializer's two explicit metadata values are checked above;
        # only those values are normalized for exact base-freeze equality.
        results["live_binding"] = {
            "trainer_id": id(runtime["trainer"]),
            "r40_trainer_same_object": runtime["r40_train"].trainer is runtime["trainer"],
            "ppo_wrapper_name": runtime["trainer"].ppo_update.__name__,
            "counter_bridge": runtime.get("r58_ppo_counter_bridge", {}).get("kind"),
            "counter_bridge_counter_is_r42_run_object":
                runtime.get("r58_ppo_counter_bridge", {}).get("counter_object") is runtime["r42_run"],
            "private_adapter_kind": runtime.get("r58_repaired_adapter", {}).get("kind"),
            "base_freeze_matches_payload_after_registered_metadata_normalization": True}
        frozen = copy.deepcopy(runtime["frozen"])
        frozen["round42"]["training_version"] = frozen_gen3["training_version"]
        frozen, frozen_path_audit = _normalize_frozen_input_paths_for_gate(frozen)
        runtime["_r58_audit_frozen_path_records"] = frozen_path_audit
        results["frozen_readonly_input_path_binding"] = frozen_path_audit
        approved_base_freeze = copy.deepcopy(runtime["round42_freeze"])
        runtime["round42_freeze"] = identity_module.attested_freeze(
            approved_base_freeze, frozen_gen3["repair_identity"],
            frozen_gen3["training_version"])
        step_journal = GEN3_RAW / "runtime" / "optimizer_step_calls.jsonl"
        step_events = [json.loads(line) for line in step_journal.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        if (not step_events or any(row.get("call_status") != "step_invoked"
                or row.get("counter_hook_version") != "round43-training-call-counter/2"
                for row in step_events)
                or [row.get("call_index") for row in step_events]
                    != list(range(1, len(step_events) + 1))):
            raise RuntimeError("historical optimizer hook journal is not a contiguous attributable count")
        # The actual gate is process-local; reconstruct its expected historical
        # starting counter from the preserved hook journal, never by inventing rows.
        runtime["r42_run"]._OPTIMIZER_STEP_CALLS = len(step_events)
        results["historical_optimizer_counter_baseline"] = {
            "value": len(step_events), "source": str(step_journal),
            "sha256": sha256(step_journal), "event_rows": len(step_events),
            "all_step_invoked": True, "indices_contiguous_from_one": True,
            "evidence_kind": "reconstructed in-memory gate baseline from preserved original hook rows",
            "journal_modified": False}
        analysis.RUN_DIR = GEN3_RAW
        runtime["common"].CHECKPOINTS = GEN3_RAW / "checkpoints"
        runtime["common"].ROWS = GEN3_RAW / "rows"
        runtime["r40_train"].common = runtime["common"]
        readonly_loader = _install_no_rng_restore_readonly_loader(runtime)
        results["live_binding"].update({"gate_run_dir": str(analysis.RUN_DIR),
            "gate_checkpoint_root": str(runtime["common"].CHECKPOINTS),
            "gate_rows_root": str(runtime["common"].ROWS),
            "gate_uses_actual_configured_analysis_module": True,
            "read_only_loader": "canonical base loader via private no-RNG-restore split",
            "read_only_loader_original_identity": readonly_loader["base_loader_identity"],
            "ambient_rng_restore_suppressed": True})
        module_mutation_snapshot = _rng_snapshot()
        try:
            for seed in (11, 22, 33):
                seed_result = {"seed": seed, "status": "pending"}
                seed_result["serialized_identity_chain"] = _validate_serialized_identity(
                    runtime, frozen_gen3, seed, identity_module, approved_base_freeze)
                results["pre_gate_checkpoint_deserializations"] += 2
                counters = {name: 0 for name in (
                    "environment_construct", "episode_provider", "claim_journal_checkpoint_write",
                    "optimizer_step", "checkpoint_write", "rng_restore")}
                deny_stack = _install_zero_side_effect_guards(runtime, counters)
                torch_load_original = torch.load
                autograd_original = torch.autograd.grad
                module_call_original = torch.nn.Module.__call__
                gate_counts = {"torch_load": 0, "forward_calls": 0, "autograd_grad": 0}
                def counted_load(*args, **kwargs):
                    gate_counts["torch_load"] += 1
                    return torch_load_original(*args, **kwargs)
                def counted_grad(*args, **kwargs):
                    gate_counts["autograd_grad"] += 1
                    return autograd_original(*args, **kwargs)
                def counted_call(module, *args, **kwargs):
                    gate_counts["forward_calls"] += 1
                    return module_call_original(module, *args, **kwargs)
                torch.load = counted_load
                torch.autograd.grad = counted_grad
                torch.nn.Module.__call__ = counted_call
                before_opt = int(runtime["r42_run"]._OPTIMIZER_STEP_CALLS)
                before_ckpt = int(runtime["r42_versions"].CHECKPOINT_WRITE_CALLS)
                before_ppo = int(runtime["r42_run"]._PPO_UPDATE_CALLS)
                rng_before = _rng_snapshot()
                try:
                    gate = analysis.first_batch_gate(seed, runtime=runtime, frozen=frozen)
                finally:
                    rng_after = _rng_snapshot()
                    torch.load = torch_load_original
                    torch.autograd.grad = autograd_original
                    torch.nn.Module.__call__ = module_call_original
                    _stop_guards(deny_stack)
                after_opt = int(runtime["r42_run"]._OPTIMIZER_STEP_CALLS)
                after_ckpt = int(runtime["r42_versions"].CHECKPOINT_WRITE_CALLS)
                after_ppo = int(runtime["r42_run"]._PPO_UPDATE_CALLS)
                unchanged = _same_rng(rng_before, rng_after)
                if not unchanged["torch_cpu_unchanged"]:
                    results["side_effect_totals"]["RNG_state_changes"] += 1
                chain = seed_result["serialized_identity_chain"]
                protected_hashes = {
                    chain["batch_record"]: chain["batch_record_sha256"],
                    chain["commit_receipt"]: chain["commit_sha256"],
                }
                for checkpoint in chain["checkpoint_identity"]:
                    protected_hashes[checkpoint["path"]] = checkpoint["sha256"]
                    protected_hashes[checkpoint["sidecar"]] = checkpoint["sidecar_sha256"]
                post_gate_hashes = {path: sha256(Path(path)) for path in protected_hashes}
                hashes_unchanged = all(post_gate_hashes[path] == expected
                                       for path, expected in protected_hashes.items())
                seed_result.update({"status": gate.get("status"), "gate_result": gate,
                    "all_post_identity_checks_reached": gate.get("status") == "passed",
                    "read_only_activity": gate_counts,
                    "actual_counters_before_after": {"optimizer": [before_opt, after_opt],
                        "checkpoint_write": [before_ckpt, after_ckpt], "ppo": [before_ppo, after_ppo]},
                    "rng_state_comparison": unchanged,
                    "denied_side_effect_attempts": counters,
                    "protected_raw_hashes_before_after": {
                        path: {"before": expected, "after": post_gate_hashes[path],
                               "unchanged": post_gate_hashes[path] == expected}
                        for path, expected in protected_hashes.items()},
                    "checkpoint_hashes_unchanged_by_gate": hashes_unchanged})
                if (gate.get("status") != "passed" or before_opt != after_opt
                        or before_ckpt != after_ckpt or before_ppo != after_ppo
                        or any(counters.values()) or not hashes_unchanged):
                    raise RuntimeError(f"seed {seed}: actual read-only first-batch gate failed: {seed_result}")
                results["seeds"].append(seed_result)
            trainer = runtime["trainer"]
            bridge = runtime.get("r58_ppo_counter_bridge")
            if not bridge or bridge.get("kind") != "round58-ppo-counter-bridge/2":
                raise RuntimeError("actual configured runtime did not install the PPO counter bridge")
            adapter = trainer.ppo_update.__wrapped__
            closure = dict(zip(adapter.__code__.co_freevars, adapter.__closure__ or ()))
            patched_cell = closure.get("patched_fn")
            if patched_cell is None:
                raise RuntimeError("live prior wrapper has no identifiable compiled PPO delegate")
            prior_delegate = patched_cell.cell_contents
            calls = {"noop_delegate": 0}
            def noop_delegate(*_args, **_kwargs):
                calls["noop_delegate"] += 1
                return {"noop_spy": True}
            fake = type("NoOpController", (), {})()
            fake.inner = type("Inner", (), {})()
            fake.inner.trunk = torch.nn.Identity()
            fake.inner.action_head = torch.nn.Identity()
            fake.inner.value_head = torch.nn.Identity()
            fake.cost_head = torch.nn.Identity()
            before_count = int(runtime["r42_run"]._PPO_UPDATE_CALLS)
            spy_guard_counts = {name: 0 for name in (
                "environment_construct", "episode_provider", "claim_journal_checkpoint_write",
                "optimizer_step", "checkpoint_write", "rng_restore")}
            spy_guards = _install_zero_side_effect_guards(runtime, spy_guard_counts)
            try:
                patched_cell.cell_contents = noop_delegate
                result = trainer.ppo_update(fake, object(), (), update_index=-999)
            finally:
                patched_cell.cell_contents = prior_delegate
                _stop_guards(spy_guards)
            after_count = int(runtime["r42_run"]._PPO_UPDATE_CALLS)
            ledger_view = runtime["r42_run"].ledger_snapshot(strict=False)
            observed_snapshot_count = int(ledger_view["counts"]["ppo_update_calls"])
            results["live_counter_spy"] = {
                "same_trainer_object_used_by_r40": runtime["r40_train"].trainer is trainer,
                "bridge_wraps_exact_prior_adapter": trainer.ppo_update.__wrapped__ is adapter,
                "compiled_delegate_replaced_only_in_memory": True,
                "delegate_calls": calls["noop_delegate"],
                "returned_noop_marker": result.get("noop_spy") is True,
                "counter_before_after": [before_count, after_count],
                "increment_exactly_once": after_count - before_count == 1,
                "same_live_ledger_snapshot_count": observed_snapshot_count,
                "snapshot_sees_the_same_hook": observed_snapshot_count == after_count,
                "memory_writer_calls": memory_writer_calls,
                "memory_writer_call_count": len(memory_writer_calls),
                "optimizer_step_calls": 0, "denied_side_effect_attempts": spy_guard_counts}
            if (calls["noop_delegate"] != 1 or after_count - before_count != 1
                    or observed_snapshot_count != after_count or result.get("noop_spy") is not True):
                raise RuntimeError("actual live PPO bridge/snapshot callchain did not close")
            results["side_effect_totals"]["PPO_noop_spy_invocations"] = 1
            results["side_effect_totals"]["memory_spy_calls"] = len(memory_writer_calls)
            # Exercise the configured live R42/R40 derived snapshot on the real
            # three-seed gen3 output, read-only. Historical PPO invocations are
            # reconstructed from three committed batch records; optimizer steps
            # come directly from the preserved 24-row hook journal.
            r42_run = runtime["r42_run"]
            runtime["r40_common"].ROUND_DIR = GEN3_RAW
            r42_run.RUN_DIR = GEN3_RAW
            r42_run._PPO_UPDATE_CALLS = len(list(
                (GEN3_RAW / "checkpoints").glob("r40_batch_s*_b*_attempt*.json")))
            r42_run._OPTIMIZER_STEP_CALLS = len(step_events)
            snapshot_guards_count = {name: 0 for name in (
                "environment_construct", "episode_provider", "claim_journal_checkpoint_write",
                "optimizer_step", "checkpoint_write", "rng_restore")}
            snapshot_guards = _install_zero_side_effect_guards(runtime, snapshot_guards_count)
            try:
                actual_snapshot = r42_run.ledger_snapshot(strict=True)
            finally:
                _stop_guards(snapshot_guards)
            counts = actual_snapshot["counts"]
            results["actual_multiseed_derived_snapshot"] = {
                "status": "strict_snapshot_returned",
                "normal_used": counts["normal_used"],
                "reserve_used": counts["reserve_used"],
                "total_used": counts["total_used"],
                "completed_logical_jobs": counts["completed_logical_jobs"],
                "committed_batch_records": len(list((GEN3_RAW / "checkpoints").glob(
                    "r40_batch_s*_b*_attempt*.json"))),
                "ppo_update_calls": counts["ppo_update_calls"],
                "optimizer_step_calls": counts["optimizer_step_calls"],
                "ppo_matches_batches": counts["ppo_update_count_matches_committed_batches"],
                "optimizer_steps_match_batches": counts["optimizer_step_count_matches_committed_batches"],
                "source": "actual configured r42_run.ledger_snapshot + r40_common.ledger_snapshot",
                "mode": "read-only historical reconstruction; no ledger writer called",
                "denied_side_effect_attempts": snapshot_guards_count}
            if (counts["normal_used"] != 12 or counts["reserve_used"] != 0
                    or counts["completed_logical_jobs"] != 12
                    or counts["ppo_update_calls"] != 3
                    or counts["optimizer_step_calls"] != 24
                    or not counts["ppo_update_count_matches_committed_batches"]
                    or not counts["optimizer_step_count_matches_committed_batches"]):
                raise RuntimeError("actual multi-seed gen3 derived snapshot did not strictly close")
        finally:
            # No RNG restoration is performed; this only compares the disposable
            # audit process with its pre-gate state.
            results["audit_process_rng_after_gates"] = _same_rng(
                module_mutation_snapshot, _rng_snapshot())
            launcher.close_audit_runtime(binding)
    results["status"] = ("all_three_actual_readonly_gates_passed"
                         if len(results["seeds"]) == 3 else "incomplete")
    return results


if __name__ == "__main__":
    result = run()
    output = HERE / "ROUND58_GEN3_REAL_GATE_CALLCHAIN_AUDIT.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output),
                      "seed_statuses": [row["status"] for row in result["seeds"]]},
                     ensure_ascii=False))
