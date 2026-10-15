# -*- coding: utf-8 -*-
"""ROUND75 S1E non-neural fixture/receipt repair probe.

Top-level imports are stdlib only.  The single authorized B process imports NumPy
and Torch locally, compares RNG states read-only, and then exercises the unchanged
scientific ``_save_stage_snapshot`` default ABI with one self-created fixture.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import random
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "raw" / "s1e_probe"
RESULT_PATH = OUTPUT_ROOT / "phase_b_result.json"
CASES_PATH = OUTPUT_ROOT / "phase_b_cases.jsonl"
CONSUMED_PATH = OUTPUT_ROOT / "phase_b_consumed.json"
REQUEST_ID = "R75_S1E_FIXTURE_RECEIPT_REPAIR_20261015"


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def consume_once(path: Path, payload: dict) -> bool:
    """Create the separate child O_EXCL consumption record exactly once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _load_runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("r75_s1_runner_s1e", HERE / "r75_s1_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CaseRecorder:
    """Append, flush and fsync one durable line immediately after each case."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("a", encoding="utf-8", newline="\n")
        self.lines = 0

    def append(self, record: dict) -> dict:
        self.handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.lines += 1
        return record

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def _assert_supported_payload(value, where: str = "payload",
                              allowed_tensor_types: tuple = ()) -> None:
    """Accept only exact native types and an explicitly provided Tensor type."""
    module = type(value).__module__
    if module == "numpy" or module.startswith("numpy."):
        raise TypeError(f"{where} contains forbidden {module}.{type(value).__name__}")
    if type(value) in (type(None), bool, int, float, complex, str, bytes, bytearray):
        return
    if type(value) is list or type(value) is tuple:
        for index, item in enumerate(value):
            _assert_supported_payload(item, f"{where}[{index}]", allowed_tensor_types)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{where} has a non-string key {type(key).__name__}")
            _assert_supported_payload(item, f"{where}.{key}", allowed_tensor_types)
        return
    if type(value) in allowed_tensor_types:
        if not hasattr(value, "dtype") or not hasattr(value, "shape"):
            raise TypeError(f"{where} is a permitted tensor type without pinned dtype/shape")
        device = getattr(value, "device", None)
        if device is not None and not (str(device) == "cpu" or
                                       getattr(device, "type", None) == "cpu"):
            raise TypeError(f"{where} tensor is not a CPU tensor")
        return
    raise TypeError(f"{where} contains unsupported {type(value).__module__}.{type(value).__name__}")


def encode_numpy_mt19937_state(state: tuple) -> dict:
    """Encode a legacy NumPy MT19937 state into pure Python fields, without drawing."""
    if not isinstance(state, tuple) or len(state) != 5:
        raise TypeError("legacy NumPy state must be the 5-item MT19937 tuple")
    keys = state[1]
    shape = tuple(getattr(keys, "shape", ()) or (len(keys),))
    return {
        "codec": "r75_numpy_mt19937_state/1",
        "algorithm": str(state[0]),
        "keys_dtype": str(getattr(keys, "dtype", None)),
        "keys_shape": [int(item) for item in shape],
        "keys": [int(item) for item in keys],
        "position": int(state[2]),
        "position_type": type(state[2]).__name__,
        "has_gauss": int(state[3]),
        "has_gauss_type": type(state[3]).__name__,
        "cached_gaussian": float(state[4]),
        "cached_gaussian_type": type(state[4]).__name__,
    }


def decode_numpy_mt19937_state(encoded: dict, np_mod):
    """Rebuild the comparison tuple without calling set_state or seed."""
    if encoded.get("codec") != "r75_numpy_mt19937_state/1":
        raise TypeError("unknown NumPy fixture codec")
    keys = np_mod.asarray(encoded["keys"], dtype=encoded["keys_dtype"])
    keys = keys.reshape(tuple(encoded["keys_shape"]))
    return (str(encoded["algorithm"]), keys, int(encoded["position"]),
            int(encoded["has_gauss"]), float(encoded["cached_gaussian"]))


def build_stage_full_state(np_mod, torch_mod) -> dict:
    torch_state = torch_mod.get_rng_state()
    if hasattr(torch_state, "cpu"):
        torch_state = torch_state.cpu()
    else:
        raise TypeError("Torch RNG state is not CPU-tensor-like")
    full_state = {
        "kind": "non_neural_rng_fixture",
        "torch_rng_state": torch_state,
        "numpy_rng_state": encode_numpy_mt19937_state(np_mod.random.get_state()),
        "python_random_state_repr": repr(random.getstate()),
        "data_identity": {"fixture": "self-created non-neural only"},
    }
    _assert_supported_payload(full_state, "fixture.full_state", (type(torch_state),))
    return full_state


class ProbeState:
    record = {"stage": "S1E_non_neural_fixture"}
    unread: list[str] = []

    def counter_report(self):
        return {"non_neural_fixture": {"counted": 1}}


def _parent_frozen_identity() -> dict:
    path = OUTPUT_ROOT / "phase_b_parent_attempt.json"
    if not path.is_file():
        return {"status": "missing", "head": "missing", "parent_head": "missing",
                "branch": "missing", "error": "parent attempt marker is missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exception:
        return {"status": "missing", "head": "missing", "parent_head": "missing",
                "branch": "missing", "error": _exception_record(exception)}
    if not isinstance(payload, dict) or payload.get("kind") != "round75-s1e-parent-attempt/1":
        return {"status": "invalid", "head": "missing", "parent_head": "missing",
                "branch": "missing", "error": "invalid parent attempt marker kind"}
    return {"status": "present", "head": payload.get("head"),
            "parent_head": payload.get("parent_head"), "branch": payload.get("branch")}


def _stage_case(runner, np_mod, torch_mod, counters):
    directory = OUTPUT_ROOT / "non_neural_fixture"
    path = directory / "rng_fixture_stage.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return {"case_id": 15, "name": "default_torch_save_load_fixture",
                "kind": "stage_save", "passed": False,
                "error": "fixture already exists; refusing reuse"}
    full_state = build_stage_full_state(np_mod, torch_mod)
    allowed_tensor_types = (type(full_state["torch_rng_state"]),)
    original_torch_save = _guarded_default_torch_save(torch_mod, allowed_tensor_types)
    counters["fixture_save_attempt"] += 1
    try:
        identity = runner._save_stage_snapshot(
            path, stage="S1E_non_neural_fixture", label="weights_only_safe_fixture",
            state=ProbeState(), full_state=full_state)
    finally:
        torch_mod.save = original_torch_save
    counters["fixture_save_completed"] += 1
    size = path.stat().st_size
    if size > 1024 * 1024:
        return {"case_id": 15, "name": "default_torch_save_load_fixture",
                "kind": "stage_save", "passed": False,
                "error": "fixture exceeds 1 MiB", "bytes": size}
    counters["fixture_load_attempt"] += 1
    counters["fixture_read_attempt"] += 1
    loaded = torch_mod.load(path, map_location="cpu", weights_only=True)
    counters["fixture_load_completed"] += 1
    counters["fixture_read_completed"] += 1
    _assert_supported_payload(loaded, "fixture.loaded_payload", allowed_tensor_types)
    loaded_state = loaded["full_state"]
    rebuilt_numpy = decode_numpy_mt19937_state(loaded_state["numpy_rng_state"], np_mod)
    same_torch = runner._raw_equal(full_state["torch_rng_state"],
                                   loaded_state["torch_rng_state"])
    same_numpy = runner._raw_equal(np_mod.random.get_state(), rebuilt_numpy)
    same_python = full_state["python_random_state_repr"] == loaded_state["python_random_state_repr"]
    runtime_identity = runner._execution_identity()
    parent_identity = _parent_frozen_identity()
    head_consistent = bool(
        parent_identity["status"] == "present" and
        identity["execution_head"] == runtime_identity.get("head") and
        identity["execution_head"] == parent_identity.get("head") and
        identity["execution_head"] not in (None, "", "unknown", "missing") and
        parent_identity.get("parent_head") == runtime_identity.get("parent_head") and
        parent_identity.get("branch") == runtime_identity.get("branch")
    )
    passed = bool(
        size == identity["bytes"] and
        loaded["stage"] == "S1E_non_neural_fixture" and
        loaded["request_id"] == REQUEST_ID and
        loaded["plan_frozen_head_is_not_execution_head"] is True and
        loaded["execution_identity"]["head"] == identity["execution_head"] and
        head_consistent and
        loaded["data_identity"]["fixture"] == "self-created non-neural only" and
        isinstance(same_torch, bool) and same_torch and
        isinstance(same_numpy, bool) and same_numpy and
        isinstance(same_python, bool) and same_python
    )
    return {"case_id": 15, "name": "default_torch_save_load_fixture",
            "kind": "stage_save", "got": bool(passed), "got_type": type(passed).__name__,
            "wanted": True, "passed": passed, "path": str(path), "bytes": size,
            "sha256": identity["sha256"], "torch_save": 1, "torch_load": 1,
            "weights_only": True, "checkpoint_loads": 0, "rng_fixture_reads": 1,
            "execution_head": identity["execution_head"],
            "runtime_head": runtime_identity.get("head"),
            "parent_frozen_head": parent_identity.get("head"),
            "parent_frozen_parent_head": parent_identity.get("parent_head"),
            "parent_frozen_branch": parent_identity.get("branch"),
            "head_identity_consistent": head_consistent}


def _guarded_default_torch_save(torch_mod, allowed_tensor_types: tuple):
    """Install a one-shot whole-object guard over the real default backend."""
    original_torch_save = torch_mod.save
    def guarded_torch_save(obj, file_obj):
        _assert_supported_payload(obj, "default_torch_save.object", allowed_tensor_types)
        return original_torch_save(obj, file_obj)
    torch_mod.save = guarded_torch_save
    return original_torch_save


def _build_case_specs(np_mod, torch_mod, runner):
    """Build 15 lazy case closures; no comparator runs at construction time."""
    torch_state = torch_mod.get_rng_state()
    numpy_state = np_mod.random.get_state()
    python_state = random.getstate()
    torch_same = torch_state.clone()
    torch_diff = torch_state.clone()
    torch_diff[0] = (int(torch_diff[0]) + 1) % 256
    torch_dtype = torch_state.to(torch_mod.int16)
    torch_shape = torch_state[:1].clone()
    numpy_same = copy.deepcopy(numpy_state)
    numpy_diff = copy.deepcopy(numpy_state)
    numpy_diff[1][0] = int(numpy_diff[1][0]) + 1
    numpy_dtype = list(numpy_state)
    numpy_dtype[1] = numpy_state[1].astype(np_mod.int32)
    numpy_dtype = tuple(numpy_dtype)
    numpy_shape = list(numpy_state)
    numpy_shape[1] = numpy_state[1][:1]
    numpy_shape = tuple(numpy_shape)
    python_same = copy.deepcopy(python_state)
    python_type = list(python_state)
    python_type[1] = list(python_state[1])
    python_type = tuple(python_type)
    mixed_same = {"torch": torch_state.clone(), "numpy": copy.deepcopy(numpy_state),
                  "python": copy.deepcopy(python_state)}
    mixed_diff = {"torch": torch_diff.clone(), "numpy": copy.deepcopy(numpy_state),
                  "python": copy.deepcopy(python_state)}
    direct = [
        ("torch_same_clone", torch_state, torch_same, True),
        ("torch_one_byte_different", torch_state, torch_diff, False),
        ("torch_dtype_different", torch_state, torch_dtype, False),
        ("torch_shape_different", torch_state, torch_shape, False),
        ("numpy_same_copy", numpy_state, numpy_same, True),
        ("numpy_one_value_different", numpy_state, numpy_diff, False),
        ("numpy_dtype_different", numpy_state, numpy_dtype, False),
        ("numpy_shape_different", numpy_state, numpy_shape, False),
        ("python_nested_same", python_state, python_same, True),
        ("python_type_different", python_state, python_type, False),
        ("mixed_payload_same", mixed_same, copy.deepcopy(mixed_same), True),
        ("mixed_payload_different", mixed_same, mixed_diff, False),
    ]
    saved_same = {"_numpy_state": copy.deepcopy(numpy_state),
                  "_torch_state": torch_state.clone(),
                  "_python_state": copy.deepcopy(python_state)}
    saved_torch_diff = {"_numpy_state": copy.deepcopy(numpy_state),
                        "_torch_state": torch_diff.clone(),
                        "_python_state": copy.deepcopy(python_state)}
    cases = []
    for case_id, (name, live, expected, wanted) in enumerate(direct, start=1):
        def factory(live=live, expected=expected, wanted=wanted):
            got = runner._raw_equal(live, expected)
            return {"got": bool(got), "got_type": type(got).__name__,
                    "wanted": bool(wanted),
                    "passed": isinstance(got, bool) and bool(got) == bool(wanted)}
        cases.append({"case_id": case_id, "name": name, "kind": "direct",
                      "factory": factory})

    def same_factory():
        got = runner._rng_payload_differences(saved_same, np_mod, torch_mod)
        return {"got": got, "got_type": type(got).__name__, "wanted": [],
                "passed": isinstance(got, list) and got == []}
    cases.append({"case_id": 13, "name": "rng_differences_all_same",
                  "kind": "integration", "factory": same_factory})

    def torch_diff_factory():
        got = runner._rng_payload_differences(saved_torch_diff, np_mod, torch_mod)
        passed = (isinstance(got, list) and len(got) == 1 and
                  got[0].get("where") == "rng_payload:torch_rng_state")
        return {"got": got, "got_type": type(got).__name__,
                "wanted": [{"where": "rng_payload:torch_rng_state"}], "passed": passed}
    cases.append({"case_id": 14, "name": "rng_differences_torch_one_byte",
                  "kind": "integration", "factory": torch_diff_factory})
    return cases


def _stage_case_spec(runner, np_mod, torch_mod, counters):
    return {"case_id": 15, "name": "default_torch_save_load_fixture",
            "kind": "stage_save",
            "factory": lambda: _stage_case(runner, np_mod, torch_mod, counters)}


def _exception_record(exception: BaseException) -> dict:
    return {"type": type(exception).__name__, "message": str(exception),
            "traceback": traceback.format_exc(limit=12)}


def _safe_rng_after(runner, np_mod, torch_mod):
    try:
        return {"status": "present", "identity": runner._rng_identity_full(np_mod, torch_mod)}
    except BaseException as exception:
        return {"status": "missing", "error": _exception_record(exception)}


def run_case(case_id: int, name: str, kind: str, function, recorder: CaseRecorder,
             counters: dict, runner, np_mod, torch_mod):
    counters["case_attempt"] += 1
    try:
        record = function()
        record.update({"case_id": case_id, "name": name, "kind": kind})
        record["rng_after"] = _safe_rng_after(runner, np_mod, torch_mod)
        if not isinstance(record.get("passed"), bool):
            record["passed"] = False
            record.setdefault("error", "case did not return a boolean passed field")
    except BaseException as exception:
        counters["case_failed"] += 1
        record = {"case_id": case_id, "name": name, "kind": kind, "passed": False,
                  "exception": _exception_record(exception),
                  "rng_after": _safe_rng_after(runner, np_mod, torch_mod)}
    else:
        counters["case_completed"] += 1
    return recorder.append(record)


def _execute_case_specs(case_specs, recorder: CaseRecorder, counters: dict,
                        runner, np_mod, torch_mod):
    """Run the fixed cases one by one through the protected receipt path."""
    results = []
    for spec in case_specs:
        result = run_case(spec["case_id"], spec["name"], spec["kind"],
                          spec["factory"], recorder, counters, runner, np_mod, torch_mod)
        results.append(result)
        if not result["passed"]:
            break
    return results


def _write_result(payload: dict) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_PATH.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, RESULT_PATH)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ROUND75 S1E fixture/receipt repair B probe")
    parser.add_argument("--run-b", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_b:
        print("B is not authorized here; launch only through the one S1E parent license",
              file=sys.stderr)
        return 2
    if not consume_once(CONSUMED_PATH, {"kind": "round75-s1e-phase-b-consumption/1",
                                        "request_id": REQUEST_ID,
                                        "note": "separate child O_EXCL record; parent marker is not rewritten"}):
        print("S1E B already consumed", file=sys.stderr)
        return 3
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    counters = {key: 0 for key in (
        "case_attempt", "case_completed", "case_failed", "numpy_import_attempt",
        "numpy_import_completed", "torch_import_attempt", "torch_import_completed",
        "fixture_save_attempt", "fixture_save_completed", "fixture_load_attempt",
        "fixture_load_completed", "fixture_read_attempt", "fixture_read_completed")}
    recorder = CaseRecorder(CASES_PATH)
    before = {"status": "missing", "error": {"type": "NotAttempted", "message": "", "traceback": ""}}
    after = dict(before)
    cases = []
    runner = None
    np_mod = None
    torch_mod = None
    try:
        runner = _load_runner()
        runner.EXECUTION_REQUEST_ID = REQUEST_ID
        counters["numpy_import_attempt"] += 1
        np_mod = importlib.import_module("numpy")
        counters["numpy_import_completed"] += 1
        counters["torch_import_attempt"] += 1
        torch_mod = importlib.import_module("torch")
        counters["torch_import_completed"] += 1
        before = {"status": "present", "identity": runner._rng_identity_full(np_mod, torch_mod)}
        case_specs = _build_case_specs(np_mod, torch_mod, runner)
        case_specs.append(_stage_case_spec(runner, np_mod, torch_mod, counters))
        cases = _execute_case_specs(case_specs, recorder, counters, runner,
                                    np_mod, torch_mod)
    except BaseException as exception:
        cases.append(recorder.append({"case_id": max([len(cases) + 1, 1]), "name": "process_setup",
                                      "kind": "setup", "passed": False,
                                      "exception": _exception_record(exception)}))
    finally:
        if np_mod is not None and torch_mod is not None and runner is not None:
            after = _safe_rng_after(runner, np_mod, torch_mod)
        recorder.close()
        rng_unchanged = (before.get("status") == "present" and
                         after.get("status") == "present" and
                         before["identity"] == after["identity"])
        payload = {
            "kind": "round75-s1e-phase-b-result/1", "request_id": REQUEST_ID,
            "python": sys.version,
            "numpy_version": getattr(np_mod, "__version__", "missing"),
            "torch_version": getattr(torch_mod, "__version__", "missing"),
            "authorized_imports": ["numpy", "torch", "r75_s1_runner"],
            "case_count": len(cases), "cases": cases, "cases_durable_path": str(CASES_PATH),
            "cases_durable_lines": recorder.lines,
            "rng_before": before, "rng_after": after, "rng_unchanged": rng_unchanged,
            "operations": counters,
            "scientific_checkpoint_loads": 0,
            "passed": bool(len(cases) == 15 and rng_unchanged and
                           all(item["passed"] for item in cases) and
                           counters["fixture_save_completed"] == 1 and
                           counters["fixture_load_completed"] == 1 and
                           counters["fixture_read_completed"] == 1),
        }
        _write_result(payload)
        print(json.dumps({"kind": "round75-s1e-phase-b-summary/1", "case_count": payload["case_count"],
                          "passed": payload["passed"], "rng_unchanged": rng_unchanged,
                          "operations": counters}, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
