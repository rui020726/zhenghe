# ROUND75 S1E fixture/receipt repair — Phase A

Request ID: `R75_S1E_FIXTURE_RECEIPT_REPAIR_20261015`  
Branch: `exp/round75-native-ppo-qualification`  
Start HEAD: `50feb386dd15af593b6e016c6a300d4af413b35c`  
Parent: `bffb6b15e8525bc7fa8711aa78cc31e0ab23799c`  
Output root: `runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/raw/s1e_probe/`

## Result

Phase A is implemented and frozen. It is **not** a B execution. The real CPU/NumPy/Torch child count is 0; the scientific fixture save/load count is 0. No scientific checkpoint/model load, controller/optimizer, forward/backward, optimizer/PPO, environment, CUDA, native/canonical/R66/guard import, S1 or training occurred.

Seven affected pure-stdlib unittest attempts were run and individually retained. The final attempt ran 23 cases and passed with exit 0. Earlier attempts were caused by wrong unittest discovery from the repository root and fake-object ABI defects; their original logs remain.

## Source frozen for decision review

| file | bytes | SHA256 |
| --- | ---: | --- |
| `r75_s1e_fixture_receipt_repair.py` | 18688 | `40e3dd2bf3fb2da9d5b5576ef12c62e7a91d4427eff68d0f4e2d378b462b8b98` |
| `r75_s1e_parent_capture.ps1` | 4280 | `eb8964079eb63e7a2eb562f3886283a63c2b68b68d93a53ccfae9ea4a3a4d6b1` |
| `test_r75_s1e_fixture_receipt_repair.py` | 16755 | `59abcb315fc539a632b3e3043c2f0e39995f1bbc7fa67858ccdef079f675b876` |

Unchanged scientific identities:

- `r75_s1_runner.py`: `872f8bcd6c6c7d92c699e7887cf81d39fbf1f6218b31bf713160d907a148eae7`
- `r75_frozen.py`: `83813d6e06971a1e1fc1e25d9e9f94b0fa72310e300e037c0fa73d539c540b80`
- `r75_guard.py`: `4d730f71fb34ad411043c65141b6bcbb3f7d2f3c4668a6b53054e6400e35d55f`
- central ledger `raw/runtime/r75_total_ledger.jsonl`: `fff8dc1f3d9222ed280a5dc9139f3f82de4c5537ccbc69894d340b44e605475d`

## Fixture contract

The NumPy MT19937 state is encoded before serialization. The codec keeps algorithm, key dtype, key shape, all key values, position, position type, `has_gauss`, cached Gaussian and their type names as Python primitives:

```python
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
```

The Torch RNG remains a CPU Tensor. It is read once with `get_rng_state()` and converted only with `.cpu()`; there is no seed, set_state, draw or checkpoint load. `build_stage_full_state` then runs `_assert_supported_payload`.

The payload guard rejects every NumPy module object before the Python built-in branches (including an `np.float64` scalar inheriting `float`). It recursively accepts only Python primitive/containers plus one tensor-like object with pinned `dtype` and `shape`:

```python
module = type(value).__module__
if module == "numpy" or module.startswith("numpy."):
    raise TypeError(f"{where} contains forbidden {module}.{type(value).__name__}")
```

Case 15 still calls the unchanged scientific helper:

```python
identity = runner._save_stage_snapshot(
    path, stage="S1E_non_neural_fixture", label="weights_only_safe_fixture",
    state=ProbeState(), full_state=full_state)
```

No `save_fn` is supplied, so the runner takes `torch.save` from `sys.modules["torch"]`. It writes through the real temporary-file and `os.replace` path. The same fixture is then read exactly once with:

```python
loaded = torch_mod.load(path, map_location="cpu", weights_only=True)
```

There is no `weights_only=False`, no allowlist, no safe-globals registration, and no scientific state-serialization change. The fixture path is confined to `raw/s1e_probe/non_neural_fixture/rng_fixture_stage.pt`; S1D remains untouched.

## Durable case and RNG receipts

`CaseRecorder.append` writes each case as one JSON line, flushes, and `fsync`s before returning. `run_case` increments case attempt, records a boolean or exception (type, message, traceback), and then obtains a read-only RNG-after fingerprint. If that fingerprint itself fails, the receipt records `rng_after.status = "missing"` plus the error; it never substitutes `rng_before`.

The B process computes the complete pre-import/import-time RNG identity once. After cases (or on setup failure), the finally block obtains after identity and marks it missing if unavailable. Final pass requires all 15 cases, complete unchanged before/after identity, one save, one load, and one fixture read. Any failure stops remaining cases; no retry or repair occurs.

## Parent capture contract

`r75_s1e_parent_capture.ps1` uses PowerShell `Start-Process -PassThru -WindowStyle Hidden` only. It creates a CreateNew attempt marker and a CreateNew parent-capture record; stale stdout/stderr/capture files refuse launch before a second attempt can start. It records child argv, PID, process start/end, exit code and binary artifacts:

```powershell
$Process = Start-Process -FilePath $Python `
    -ArgumentList $ChildArgv `
    -WorkingDirectory $Here `
    -RedirectStandardOutput $StdoutPath `
    -RedirectStandardError $StderrPath `
    -WindowStyle Hidden `
    -PassThru
```

stdout/stderr are redirected to files and later summarized by file existence, length and `Get-FileHash -Algorithm SHA256`. There is no Node `spawnSync`, no text reader re-encoding, and no result-copy substitution. A parent failure and a started Python child remain separately recorded. The child also creates its own O_EXCL consumption receipt.

## A operations and raw evidence

Raw test logs, with no raw fixture produced:

| log | bytes | SHA256 |
| --- | ---: | --- |
| `a_test_run.log` | 726 | `1ac3163f2053faeb6f5873080aa5106425c13bbc99ef014c20d822a37a66b8d4` |
| `a_test_run_attempt2.log` | 17483 | `6dc9840bcd88aabf7ea619be71d569549661fe97fa0ddeeee4adcbc2d3386104` |
| `a_test_run_attempt3.log` | 6686 | `5e9f415d2683099af50a963e11b1465f76a887322dfe8abdd43679a7cebbb343` |
| `a_test_run_attempt4.log` | 5060 | `197a490096129c05bd591b4c88b5147ad9d3234286e6e8c37de820a75f31afda` |
| `a_test_run_attempt5.log` | 3227 | `df0e2a4b591b74713bf5e3f56df9a2afd618a1a00497161ab3f768f5e58cb444` |
| `a_test_run_attempt6.log` | 2593 | `b54a1bdebe87bbc676f2cfffec55cd02a74284f9fdad605ba214d71f571e1a38` |
| `a_test_run_attempt7.log` | 2715 | `7df6900010b20cf91ef5ee47e8ab2dc9b1afd219a1048471a17a99e8e85f9054` |

Operations: 7 pure test processes; 0 real S1E B process; 0 NumPy/Torch import; 0 fixture save; 0 fixture load/read; 0 scientific checkpoint/model load; 0 optimizer/PPO/environment operation; 0 ledger charge. The first test attempt failed only on module discovery; the final exit was 0 with `Ran 23 tests ... OK`.

## Boundary

The final tests use fakes for NumPy/Torch serialization. They prove wiring, codec and receipt behavior but do not prove the real Torch weights-only backend accepts this exact fixture. That is precisely the pending one-process B check. A passing A does not qualify S1, training, evaluation or any scientific conclusion. B remains behind the explicit decision-source review and its one-license execution path.

## Phase-A review revision 2

Decision review did not release B. The following narrow fixes were made without changing the scientific runner, wrapper, guard, native path, data or old originals.

1. **Lazy cases.** `_build_case_specs` now returns 15 closures. The 14 comparator cases do not call `_raw_equal` or `_rng_payload_differences` during construction. Main calls `_execute_case_specs`; every computation is inside `run_case` and is appended/flushed/fsynced before the next case. A pure main-path counterexample proves case 1 is durable, case 2 raises, and case 3 is never called.
2. **Exact payload guard.** The guard accepts only exact `NoneType`, `bool`, `int`, `float`, `complex`, `str`, `bytes`, `bytearray`, `list`, `tuple` and `dict`; dict keys must be exact `str`. It accepts only an explicitly passed Tensor type, requiring pinned dtype/shape and CPU. Custom dict/list/float/str subclasses, duck tensors, NumPy arrays/scalars and Path are rejected.
3. **Whole-object save guard.** Before the single default save, the probe temporarily wraps `torch_mod.save`; it strictly checks the complete object that `_save_stage_snapshot` hands to `torch.save`, then forwards once to the original save and restores it in `finally`. No `save_fn` is passed to the runner and no second save/load occurs.
4. **Case-15 identity.** The stage result compares loaded execution head, snapshot execution head, live git HEAD/parent/branch, and the HEAD/parent/branch in the CreateNew parent attempt marker. Missing parent identity fails; it is not filled with zero.
5. **Parent metadata.** A successful `Start-Process` immediately marks `actual_python_child=true` and reads PID before optional metadata reads. StartTime is separately faulted as `child_metadata_error`; WaitForExit is not skipped. `ChildExitTime` becomes `missing` if unavailable and observer `ended_at` remains separate. `child_argv` now records Python executable, script and `--run-b`; real cwd is recorded.
6. **Binary capture evidence.** The parent records actual PowerShell version, OS, Windows platform, and the Windows file-handle basis for redirected stdio. A selftest exercises the actual `BinaryArtifact` component on NUL/non-UTF-8 bytes without starting a child. The current environment lacks `Get-FileHash`, so the parent uses `.NET SHA256.ComputeHash` on a binary file stream.

Frozen revision-2 identities:

| file | bytes | SHA256 |
| --- | ---: | --- |
| `r75_s1e_fixture_receipt_repair.py` | 22297 | `c7b651fe468a59869da6c5d45e5ecf491efa8d821d95975892fcc2e753fc7d7b` |
| `r75_s1e_parent_capture.ps1` | 8100 | `978334789c76e7b5b5b79417af892eb652049b139a1499e5ca37481f0e09f44c` |
| `test_r75_s1e_fixture_receipt_repair.py` | 20243 | `83b2069febaa0334d015d7a1ac1b6d186cfdd1333abede962ca4f9ddcfcc9bf0` |

Revision-2 test attempts, all retained with explicit Python exit receipts:

| attempt | outcome | log SHA256 |
| --- | --- | --- |
| 8 | 25 cases, 2 failures; also exposed missing `Get-FileHash` | `94a2491073e9507d743452561a852dda7566306a462b08a9a2dcaa122ef63a81` |
| 9 | 24 cases, 1 stale binary-artifact assertion failure | `bb81acc752e6b6166a7c5f41d3932463ed23ad5b638bcdbda3246525b415979e` |
| 10 | 24 cases, 1 selftest CreateNew collision from the attempt-8 residue | `1c7ec4440bf96cd6595f67b1a07888fa2b906354c19619d227f12531104d03e2` |
| 11 | 24 cases, all passed; includes PowerShell binary selftest | `9d7d4ea3bd090a2b7cddd36d1ff4cdda97009e1d738c80940eebfc818dcf3471` |

Revision-2 operations add 4 pure Python test processes and one PowerShell selftest process started by the final Python test; it started no child. There remain 0 real S1E B processes, 0 NumPy/Torch imports, 0 fixture saves/loads/reads, 0 scientific checkpoint/model loads, 0 optimizer/PPO/environment operations and 0 ledger charges. The failed attempt-8 selftest residue remains in `raw/s1e_probe/`; later selftests use unique paths and remove their own artifacts.

## Phase-A review revision 3

Decision review again did not release B. The same-task A scope allowed only the three named implementation fixes, the three affected existing pure tests, and this close-out. The top-level test-method count remains 24; no new top-level case was added.

1. **Single stage save.** `_stage_case` now calls `runner._save_stage_snapshot` exactly once at `r75_s1e_fixture_receipt_repair.py:194`. The earlier real double save is gone; the load still occurs after the single save.
2. **Whole default backend.** `_guarded_default_torch_save` wraps the complete default `torch.save` object, validates the whole payload before forwarding once to the original backend, and restores it in `finally`. The independent rejection exercise uses complete metadata plus an unknown object and then proves `fake_torch.save_count == 1`.
3. **Parent identity and ordering.** The branch is now read with `git -C $Here rev-parse --abbrev-ref HEAD`. `Start-Process` success sets `ActualChild = $true` before fault-tolerant PID/StartTime reads. The one child is always awaited; there is no retry and no kill.

Frozen revision-3 identities:

| file | bytes | SHA256 |
| --- | ---: | --- |
| `r75_s1e_fixture_receipt_repair.py` | 22393 | `21f0105295fe5880565316e625f06d58e2757a38131aa7f5517fdc9c7159af40` |
| `r75_s1e_parent_capture.ps1` | 8176 | `50f555ef717e82f0b00ab0fb4dad0ea72771546724b5881835b5aa75bec49471` |
| `test_r75_s1e_fixture_receipt_repair.py` | 21375 | `75cf51cc160739c65af0d79595f495d4afc3355380080492b972515f129dbc67` |

Attempt 12 ran the three affected existing cases and passed:

```text
test_fake_weights_only_stage_case_saves_and_loads_once (test_r75_s1e_fixture_receipt_repair.SaveAndStageTests) ... ok
test_parent_uses_powershell_start_process_hidden_once (test_r75_s1e_fixture_receipt_repair.ParentAndConsumptionTests) ... ok
test_parent_captures_stdio_as_binary_file_hashes (test_r75_s1e_fixture_receipt_repair.ParentAndConsumptionTests) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.356s

OK
```

Its exit receipt is `0`.

## Revision-2 process errata

The earlier claim of "one PowerShell selftest" was wrong. Actual PowerShell binary selftest processes occurred in attempts 8, 10, 11 and 12; each is registered separately. Attempt 8 actually ran 25 distinct top-level cases and had 2 failures. That attempt and its residue are not erased by later test rewrites or deduplication. Attempts 1-7 Python exits were not durably captured and are registered as unknown/uncaught, not zero. Attempt 12 also started one PowerShell selftest; it started no B child.

Cumulative A process ledger: 12 pure-Python test attempt processes (attempts 1-12), 4 PowerShell selftest processes, 0 real S1E B processes, 0 parent B launch attempts. NumPy/Torch imports, fixture saves/loads/reads, scientific checkpoint/model loads, optimizer/PPO operations, environments and ledger charges remain 0. JSON-validation Python invocations are bookkeeping, not test attempts. The revision-2 field immediately above originally understated PowerShell selftests; the authoritative count is this cumulative ledger.

## Revision-3 full evidence appendix

The following disk text is included without filtering. Probe source is split into chunks of at most 160 lines. The parent script is included in full. The key backend-count/metadata-rejection evidence is represented both by the existing test at `test_fake_weights_only_stage_case_saves_and_loads_once` and by the attempt-12 disk log above.

### Probe source 1/3 (1..160, 22393 bytes, sha256:21f0105295fe5880565316e625f06d58e2757a38131aa7f5517fdc9c7159af40)
```text
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
```

### Probe source 2/3 (161..320, 22393 bytes, sha256:21f0105295fe5880565316e625f06d58e2757a38131aa7f5517fdc9c7159af40)
```text
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
```

### Probe source 3/3 (321..475, 22393 bytes, sha256:21f0105295fe5880565316e625f06d58e2757a38131aa7f5517fdc9c7159af40)
```text
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
```

### Parent script 1/2 (1..160, 8176 bytes, sha256:50f555ef717e82f0b00ab0fb4dad0ea72771546724b5881835b5aa75bec49471)
```text
param([switch]$SelfTest = $false)

# ROUND75 S1E parent capture. PowerShell only: it launches the one authorized Python child.
$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Join-Path $Here "raw\s1e_probe"
New-Item -ItemType Directory -Force -Path $Root | Out-Null

$RequestId = "R75_S1E_FIXTURE_RECEIPT_REPAIR_20261015"
$Python = "D:\project_env\python.exe"
$ChildScript = Join-Path $Here "r75_s1e_fixture_receipt_repair.py"
$ChildArgv = @($Python, $ChildScript, "--run-b")
$ChildArgumentList = @($ChildScript, "--run-b")
$StdoutPath = Join-Path $Root "phase_b_stdout.bin"
$StderrPath = Join-Path $Root "phase_b_stderr.bin"
$AttemptPath = Join-Path $Root "phase_b_parent_attempt.json"
$CapturePath = Join-Path $Root "phase_b_parent_capture.json"

foreach ($PathToCheck in @($StdoutPath, $StderrPath, $AttemptPath, $CapturePath)) {
    if (Test-Path -LiteralPath $PathToCheck) {
        throw "S1E parent refuses a second launch: $PathToCheck already exists"
    }
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "S1E parent did not find the authorized Python executable: $Python"
}

function BinaryArtifact {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return [ordered]@{path=$Path; exists=$false; bytes="missing"; sha256="missing"}
    }
    $Item = Get-Item -LiteralPath $Path
    $Sha256 = [System.Security.Cryptography.SHA256]::Create()
    $FileStream = [System.IO.File]::Open($Path,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::ReadWrite)
    try {
        $HashBytes = $Sha256.ComputeHash($FileStream)
    } finally {
        $FileStream.Dispose()
        $Sha256.Dispose()
    }
    $HashHex = [System.BitConverter]::ToString($HashBytes).Replace("-", "").ToLowerInvariant()
    return [ordered]@{path=$Path; exists=$true; bytes=$Item.Length;
                      sha256=$HashHex}
}

function Write-JsonCreateNew {
    param([string]$Path, [object]$Object)
    $Utf8 = New-Object System.Text.UTF8Encoding($false)
    $Bytes = $Utf8.GetBytes((ConvertTo-Json -InputObject $Object -Depth 12))
    $Stream = [System.IO.File]::Open($Path,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None)
    try {
        $Stream.Write($Bytes, 0, $Bytes.Length)
        $Stream.Flush($true)
    } finally {
        $Stream.Dispose()
    }
}

$PowerShellVersion = $PSVersionTable.PSVersion.ToString()
$OsDescription = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
$IsWindowsPlatform = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
    [System.Runtime.InteropServices.OSPlatform]::Windows)
$StdioImplementationBasis = if ($IsWindowsPlatform) {
    "Windows CreateProcess Start-Process branch supplies redirected files as process stdio file handles; no line reader is used."
} else {
    "Non-Windows Start-Process uses a line-oriented implementation; this parent must not be used for the S1E B capture."
}

if ($SelfTest) {
    $BinaryPath = Join-Path $Root ("parent_binary_selftest_" + [Guid]::NewGuid().ToString("N") + ".bin")
    $BinaryBytes = [byte[]]@(0x00, 0xFF, 0x00, 0xD8, 0x80, 0x00, 0x41, 0x00)
    $ExpectedSha = [System.BitConverter]::ToString(
        [System.Security.Cryptography.SHA256]::Create().ComputeHash($BinaryBytes)
    ).Replace("-", "").ToLowerInvariant()
    $BinaryStream = [System.IO.File]::Open($BinaryPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None)
    try { $BinaryStream.Write($BinaryBytes, 0, $BinaryBytes.Length) } finally { $BinaryStream.Dispose() }
    $Observed = BinaryArtifact -Path $BinaryPath
    $Passed = ($Observed.bytes -eq $BinaryBytes.Length -and
               $Observed.sha256 -eq $ExpectedSha)
    $SelfTestResult = [ordered]@{
        kind = "round75-s1e-parent-binary-selftest/1"
        request_id = $RequestId
        powershell_version = $PowerShellVersion
        os_description = $OsDescription
        is_windows = $IsWindowsPlatform
        stdio_implementation_basis = $StdioImplementationBasis
        binary_path = $BinaryPath
        expected_bytes = $BinaryBytes.Length
        expected_sha256 = $ExpectedSha
        observed_artifact = $Observed
        passed = [bool]$Passed
        child_started = $false
    }
    Remove-Item -LiteralPath $BinaryPath -Force
    ConvertTo-Json -InputObject $SelfTestResult -Depth 8
    if ($Passed) { exit 0 } else { exit 1 }
}

$Head = (& git -C $Here rev-parse HEAD)
if ($LASTEXITCODE -ne 0) { throw "S1E parent could not read HEAD" }
$ParentHead = (& git -C $Here rev-parse HEAD^)
if ($LASTEXITCODE -ne 0) { throw "S1E parent could not read HEAD^" }
$Branch = (& git -C $Here rev-parse --abbrev-ref HEAD)
if ($LASTEXITCODE -ne 0) { throw "S1E parent could not read branch" }

$Attempt = [ordered]@{
    kind = "round75-s1e-parent-attempt/1"
    request_id = $RequestId
    created_at = [DateTimeOffset]::UtcNow.ToString("o")
    parent_pid = $PID
    head = $Head
    parent_head = $ParentHead
    branch = $Branch
    child_argv = $ChildArgv
    child_script = $ChildScript
    working_directory = $Here
    stdout_path = $StdoutPath
    stderr_path = $StderrPath
    launch_state = "issued"
}
Write-JsonCreateNew -Path $AttemptPath -Object $Attempt

$StartedAt = [DateTimeOffset]::UtcNow.ToString("o")
$Process = $null
$ChildPid = $null
$ChildStartTime = "missing"
$ChildExitTime = "missing"
$ExitCode = "missing"
$ActualChild = $false
$ParentError = $null
$ChildMetadataError = $null
$WaitError = $null
$ExitReadError = $null

try {
    $Process = Start-Process -FilePath $Python `
        -ArgumentList $ChildArgumentList `
        -WorkingDirectory $Here `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -WindowStyle Hidden `
        -PassThru
    $ActualChild = $true
} catch {
    $ParentError = $_.ToString()
}

if ($ActualChild) {
    try { $ChildPid = $Process.Id } catch {
```

### Parent script 2/2 (161..223, 8176 bytes, sha256:50f555ef717e82f0b00ab0fb4dad0ea72771546724b5881835b5aa75bec49471)
```text
        $ChildMetadataError = $_.ToString()
    }
    try { $ChildStartTime = $Process.StartTime.ToString("o") } catch {
        $ChildMetadataError = $_.ToString()
    }
    try { $null = $Process.WaitForExit() } catch {
        $WaitError = $_.ToString()
        $ChildExitTime = "missing"
        $ExitCode = "missing"
    }
    if ($null -eq $WaitError) {
        try { $ChildExitTime = $Process.ExitTime.ToString("o") } catch {
            $ChildExitTime = "missing"
            $ExitReadError = $_.ToString()
        }
        try { $ExitCode = $Process.ExitCode } catch {
            $ExitCode = "missing"
            $ExitReadError = $_.ToString()
        }
    }
}
$EndedAt = [DateTimeOffset]::UtcNow.ToString("o")

$Capture = [ordered]@{
    kind = "round75-s1e-parent-capture/1"
    request_id = $RequestId
    parent_pid = $PID
    head = $Head
    parent_head = $ParentHead
    branch = $Branch
    working_directory = $Here
    attempt_marker = $AttemptPath
    parent_launch_attempted = $true
    parent_launch_failed = [bool]$ParentError
    parent_error = $ParentError
    actual_python_child = $ActualChild
    python_started = $ActualChild
    child_pid = $ChildPid
    child_argv = $ChildArgv
    child_metadata_error = $ChildMetadataError
    wait_error = $WaitError
    exit_read_error = $ExitReadError
    stdout_path = $StdoutPath
    stderr_path = $StderrPath
    stdout_artifact = BinaryArtifact -Path $StdoutPath
    stderr_artifact = BinaryArtifact -Path $StderrPath
    started_at = $StartedAt
    child_start_time = $ChildStartTime
    child_exit_time = $ChildExitTime
    ended_at = $EndedAt
    exit_code = $ExitCode
    powershell_version = $PowerShellVersion
    os_description = $OsDescription
    is_windows = $IsWindowsPlatform
    stdio_implementation_basis = $StdioImplementationBasis
    retry_count = 0
    note = "One PowerShell Start-Process launch; stdout/stderr are read only from the redirected files as binary hashes."
}
Write-JsonCreateNew -Path $CapturePath -Object $Capture

if ($ParentError) { exit 2 }
if (-not $ActualChild) { exit 2 }
exit [int]$ExitCode
```

### Key test: single save, metadata rejection, backend count (test lines 277..327)
```python
    def test_fake_weights_only_stage_case_saves_and_loads_once(self):
        self.patch_output_root()
        self.patch_stage_identity()
        self.write_parent_attempt()
        fake_torch = FakeTorchModule()
        original = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        runner.EXECUTION_REQUEST_ID = probe.REQUEST_ID
        counters = {key: 0 for key in ("fixture_save_attempt", "fixture_save_completed",
                                       "fixture_load_attempt", "fixture_load_completed",
                                       "fixture_read_attempt", "fixture_read_completed")}
        try:
            result = probe._stage_case(runner, FakeNumpyModule(), fake_torch, counters)
        finally:
            if original is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original
        runner.EXECUTION_REQUEST_ID = None
        self.assertTrue(result["passed"])
        self.assertEqual(result["got_type"], "bool")
        self.assertEqual(result["weights_only"], True)
        self.assertEqual(result["head_identity_consistent"], True)
        self.assertIsNotNone(fake_torch.last_saved)
        saved = fake_torch.last_saved
        self.assertEqual(set(saved), {"kind", "stage", "label", "request_id", "plan_frozen_head",
                                      "plan_frozen_head_is_not_execution_head", "execution_identity",
                                      "code_identity", "data_identity", "counters", "unread",
                                      "record", "full_state"})
        self.assertIn("full_state", saved)
        self.assertEqual(counters["fixture_save_attempt"], 1)
        self.assertEqual(counters["fixture_save_completed"], 1)
        self.assertEqual(fake_torch.save_count, 1)
        self.assertEqual(counters["fixture_load_completed"], 1)
        self.assertEqual(counters["fixture_read_completed"], 1)
        complete_like = {
            "kind": "round75-s1-stage-snapshot/1", "stage": "test", "label": "whole-object",
            "request_id": probe.REQUEST_ID, "plan_frozen_head": "plan",
            "plan_frozen_head_is_not_execution_head": True,
            "execution_identity": {"head": "test-head"}, "code_identity": {},
            "data_identity": {}, "counters": {}, "unread": [], "record": {},
            "full_state": {}, "unknown_metadata": object(),
        }
        original_backend = probe._guarded_default_torch_save(fake_torch, (FakeTensor,))
        try:
            with self.assertRaises(TypeError):
                fake_torch.save(complete_like, self.tmp / "must-not-write")
        finally:
            fake_torch.save = original_backend
        self.assertEqual(fake_torch.save_count, 1)

```
