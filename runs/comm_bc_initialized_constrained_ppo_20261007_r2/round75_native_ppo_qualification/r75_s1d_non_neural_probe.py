# -*- coding: utf-8 -*-
"""S1D non-neural comparator/stage-save probe.

Top-level imports are stdlib only so the A suite can inspect capture/consumption helpers. The
B main imports NumPy and Torch locally and performs no model, optimizer, PPO or environment work.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE
RUNNER_PATH = HERE / "r75_s1_runner.py"
RESULT_PATH = HERE / "raw" / "s1d_probe" / "phase_b_result.json"
CONSUMED_PATH = HERE / "raw" / "s1d_probe" / "phase_b_consumed.json"


def _sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def capture_process(argv, runner=subprocess.run):
    """Run one command with PIPE captures and retain actual stdout/stderr bytes."""
    completed = runner(list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = completed.stdout or b""
    stderr = completed.stderr or b""
    return {"argv": list(argv), "returncode": completed.returncode,
            "stdout_bytes": stdout, "stderr_bytes": stderr,
            "stdout_bytes_len": len(stdout), "stderr_bytes_len": len(stderr),
            "stdout_sha256": _sha_bytes(stdout), "stderr_sha256": _sha_bytes(stderr)}


def consume_once(path, payload):
    """Create one O_EXCL actual-consumption record; a pre-run manifest is never rewritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _load_runner():
    spec = importlib.util.spec_from_file_location("r75_s1_runner_s1d", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ProbeState:
    record = {}
    unread = []
    def counter_report(self):
        return {"non_neural_fixture": {"counted": 1}}


def _build_cases(np, torch, runner):
    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    torch_same = torch_state.clone()
    torch_diff = torch_state.clone()
    torch_diff[0] = (int(torch_diff[0]) + 1) % 256
    torch_dtype = torch_state.to(torch.int16)
    torch_shape = torch_state[:1].clone()
    numpy_same = copy.deepcopy(numpy_state)
    numpy_diff = copy.deepcopy(numpy_state)
    numpy_diff[1][0] = int(numpy_diff[1][0]) + 1
    numpy_dtype = list(numpy_state)
    numpy_dtype[1] = numpy_state[1].astype(np.int32)
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
    cases = []
    for name, live, expected, wanted in direct:
        got = runner._raw_equal(live, expected)
        cases.append({"name": name, "kind": "direct", "got": bool(got),
                      "got_type": type(got).__name__, "wanted": bool(wanted),
                      "passed": isinstance(got, bool) and bool(got) == bool(wanted)})
    saved_same = {"_numpy_state": copy.deepcopy(numpy_state),
                  "_torch_state": torch_state.clone(),
                  "_python_state": copy.deepcopy(python_state)}
    problems_same = runner._rng_payload_differences(saved_same, np, torch)
    cases.append({"name": "rng_differences_all_same", "kind": "integration",
                  "got": problems_same, "got_type": type(problems_same).__name__,
                  "wanted": [], "passed": isinstance(problems_same, list) and problems_same == []})
    saved_torch_diff = {"_numpy_state": copy.deepcopy(numpy_state),
                        "_torch_state": torch_diff.clone(),
                        "_python_state": copy.deepcopy(python_state)}
    problems_torch = runner._rng_payload_differences(saved_torch_diff, np, torch)
    passed_torch = (isinstance(problems_torch, list) and len(problems_torch) == 1 and
                    problems_torch[0].get("where") == "rng_payload:torch_rng_state")
    cases.append({"name": "rng_differences_torch_one_byte", "kind": "integration",
                  "got": problems_torch, "got_type": type(problems_torch).__name__,
                  "wanted": [{"where": "rng_payload:torch_rng_state"}], "passed": passed_torch})
    return cases


def _stage_case(runner, np, torch):
    directory = HERE / "raw" / "s1d_probe" / "non_neural_fixture"
    path = directory / "rng_fixture_stage.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return {"name": "default_torch_save_load_fixture", "kind": "stage_save",
                "passed": False, "error": "fixture already exists; refusing reuse"}
    full_state = {"kind": "non_neural_rng_fixture",
                  "torch_rng_state": torch.get_rng_state().clone(),
                  "numpy_key": np.random.get_state()[1].copy(),
                  "python_repr": repr(random.getstate()),
                  "data_identity": {"fixture": "self-created non-neural only"}}
    identity = runner._save_stage_snapshot(path, stage="S1D_non_neural_fixture", label="torch_save_abi",
                                           state=_ProbeState(), full_state=full_state)
    size = path.stat().st_size
    if size > 1024 * 1024:
        return {"name": "default_torch_save_load_fixture", "kind": "stage_save",
                "passed": False, "error": "fixture exceeds 1 MiB", "bytes": size}
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    same_torch = runner._raw_equal(full_state["torch_rng_state"], loaded["full_state"]["torch_rng_state"])
    same_numpy = runner._raw_equal(full_state["numpy_key"], loaded["full_state"]["numpy_key"])
    passed = (size == identity["bytes"] and
              loaded["stage"] == "S1D_non_neural_fixture" and
              loaded["execution_identity"]["head"] == identity["execution_head"] and
              loaded["plan_frozen_head_is_not_execution_head"] is True and
              loaded["data_identity"]["fixture"] == "self-created non-neural only" and
              isinstance(same_torch, bool) and same_torch and
              isinstance(same_numpy, bool) and same_numpy)
    return {"name": "default_torch_save_load_fixture", "kind": "stage_save",
            "passed": bool(passed), "path": str(path), "bytes": size,
            "sha256": identity["sha256"], "torch_save": 1, "torch_load": 1,
            "weights_only": True, "checkpoint_loads": 0, "rng_fixture_reads": 1,
            "execution_head": identity["execution_head"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="ROUND75 S1D non-neural B probe")
    parser.add_argument("--run-b", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_b:
        print("B is not authorized; use only with the S1D parent-launched --run-b license")
        return 2
    if not consume_once(CONSUMED_PATH, {"kind": "round75-s1d-phase-b-consumption/1",
                                        "request_id": "R75_S1D_NON_NEURAL_REPAIR_20261015",
                                        "note": "separate O_EXCL record; manifest is not rewritten"}):
        print("S1D B already consumed", file=sys.stderr)
        return 3
    runner = _load_runner()
    import numpy as np
    import torch
    before = runner._rng_identity_full(np, torch)
    cases = _build_cases(np, torch, runner)
    cases.append(_stage_case(runner, np, torch))
    after = runner._rng_identity_full(np, torch)
    payload = {"kind": "round75-s1d-phase-b-result/1",
               "request_id": "R75_S1D_NON_NEURAL_REPAIR_20261015",
               "python": sys.version, "numpy_version": np.__version__,
               "torch_version": torch.__version__,
               "authorized_imports": ["numpy", "torch", "r75_s1_runner"],
               "case_count": len(cases), "cases": cases,
               "rng_before": before, "rng_after": after,
               "rng_unchanged": before == after,
               "passed": (all(item["passed"] for item in cases) and before == after)}
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
                           encoding="utf-8")
    print(json.dumps({"kind": "round75-s1d-phase-b-summary/1", "case_count": payload["case_count"],
                      "passed": payload["passed"], "rng_unchanged": payload["rng_unchanged"]},
                     indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
