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
