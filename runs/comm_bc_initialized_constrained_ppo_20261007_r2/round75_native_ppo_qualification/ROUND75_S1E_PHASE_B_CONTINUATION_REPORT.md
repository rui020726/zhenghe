# ROUND75 S1E Phase B continuation

Request ID: `R75_S1E_PHASE_B_CONTINUATION_20261015`
Execution HEAD: `71737f4ece9eed957d1f4e73f8a9169bc6582bd9`
Execution parent: `bafc9191823af87fe31663bb86cd2564ef04026b`
Branch: `exp/round75-native-ppo-qualification`
Output root: `runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/raw/s1e_probe/`

## Precheck

Read-only PowerShell confirmed the branch, execution HEAD/parent, and all nine frozen identities. There were no tracked worktree changes. PowerShell was Core 7.6.5 on Windows 10.0.26200; the parent used the same `pwsh.exe`.

All eight planned B artifacts were absent before launch: consumed, parent attempt, parent capture, result, cases JSONL, fixture, stdout and stderr. The two ROUND46 protected files retained their required hashes.

## Launch

Exactly one normal parent entry was run through `r75_s1e_parent_capture.ps1` with hidden `pwsh.exe`; `-SelfTest` was not used. There was no retry, fallback, Node spawn, second Python probe or second parent launch.

- Parent PID: `72568`
- Parent start: `2026-10-15T11:09:40.6278269+08:00`
- Parent exit: `2026-10-15T11:09:46.0955087+08:00`
- Parent exit code: `1`
- Child PID: `57336`
- Child start: `2026-10-15T11:09:41.2516217+08:00`
- Child exit: `2026-10-15T11:09:45.9962751+08:00`
- Child exit code: `1`
- Parent error / metadata error / wait error / exit-read error: none

The parent recorded the real child argv as `D:\project_env\python.exe`, the current absolute probe path, and `--run-b`; working directory was the ROUND75 output directory.

## Result

The result is a real, durable failure, not a qualification. It reached all 15 fixed cases. The independent case-records count is **14 passed, 1 failed, 0 not reached**. `operations.case_failed` is only the exception counter and was 0 because case 15 returned a normal boolean `False`; it must not be reported as no failures.

Top-level result:

```json
{
  "kind": "round75-s1e-phase-b-result/1",
  "request_id": "R75_S1E_FIXTURE_RECEIPT_REPAIR_20261015",
  "python": "3.10.19 | packaged by Anaconda, Inc. | (main, Oct 21, 2025, 16:41:31) [MSC v.1929 64 bit (AMD64)]",
  "numpy_version": "2.2.6",
  "torch_version": "2.9.0+cu128",
  "case_count": 15,
  "passed": false,
  "rng_unchanged": true,
  "scientific_checkpoint_loads": 0
}
```

Real stdout summary:

```text
{"case_count": 15, "kind": "round75-s1e-phase-b-summary/1", "operations": {"case_attempt": 15, "case_completed": 15, "case_failed": 0, "fixture_load_attempt": 1, "fixture_load_completed": 1, "fixture_read_attempt": 1, "fixture_read_completed": 1, "fixture_save_attempt": 1, "fixture_save_completed": 1, "numpy_import_attempt": 1, "numpy_import_completed": 1, "torch_import_attempt": 1, "torch_import_completed": 1}, "passed": false, "rng_unchanged": true}
```

`phase_b_stderr.bin` is 0 bytes.

| case | name | passed |
| ---: | --- | --- |
| 1 | torch_same_clone | true |
| 2 | torch_one_byte_different | true |
| 3 | torch_dtype_different | true |
| 4 | torch_shape_different | true |
| 5 | numpy_same_copy | true |
| 6 | numpy_one_value_different | true |
| 7 | numpy_dtype_different | true |
| 8 | numpy_shape_different | true |
| 9 | python_nested_same | true |
| 10 | python_type_different | true |
| 11 | mixed_payload_same | true |
| 12 | mixed_payload_different | true |
| 13 | rng_differences_all_same | true |
| 14 | rng_differences_torch_one_byte | true |
| 15 | default_torch_save_load_fixture | false |

Case 15's durable record reports one save, one weights-only load, one fixture read, `checkpoint_loads=0`, fixture bytes 19881, fixture SHA256 `d78209a2916f9b5324a47bc004ad0b90cd1cbcdf220c672fe16e9f13683c02de`, and `head_identity_consistent=false`. Its saved execution head, runtime head, and frozen parent-attempt head all read `71737f4ece9eed957d1f4e73f8a9169bc6582bd9`; the frozen parent-attempt parent head and current git parent head read `bafc9191823af87fe31663bb86cd2564ef04026b`. The composite identity gate nevertheless evaluated false. This report records the observed durable outcome without inventing a hidden field or rerunning the probe.

## Actual B consumption

- Real parent launch attempts: 1
- Real Python B child processes: 1
- NumPy imports: 1
- Torch imports: 1
- Fixture save attempts/completed: 1/1
- Fixture load attempts/completed: 1/1
- Fixture read attempts/completed: 1/1
- Scientific checkpoint/model loads: 0
- Forward/backward: 0
- Optimizer/PPO operations: 0
- Environment constructions: 0
- CUDA operations: 0
- Ledger charges: 0

Post-run read-only file-hash reads included the raw B originals, old A evidence, central identities and protected files. They are not additional fixture `torch.load` reads and did not alter files.

## Evidence identities

| evidence | bytes | SHA256 |
| --- | ---: | --- |
| `phase_b_consumed.json` | 183 | `8e3ccd7ae88fda2d6eedf11c4095befafbbc1a64361198ad1ab6a0bb0f84144a` |
| `phase_b_parent_attempt.json` | 1315 | `4ea4b23942c7cff96d38ad9459eebf4767c2c570908d7f74489c9a847a8558b8` |
| `phase_b_parent_capture.json` | 2815 | `a40a9dfd1d0922b217ab0c97442c013e74f620a241be6b5f13065b7ce0435515` |
| `phase_b_result.json` | 23605 | `b6d73503ebf0a813fe53656f0c5f7acd13ea764b8477d833a21068c1bc975e21` |
| `phase_b_cases.jsonl` | 12187 | `de8fbcad98a6dff7caccaf6298fcc436d67a854c1a5f402d7aaac1f52462ad8f` |
| `non_neural_fixture/rng_fixture_stage.pt` | 19881 | `d78209a2916f9b5324a47bc004ad0b90cd1cbcdf220c672fe16e9f13683c02de` |
| `phase_b_stdout.bin` | 457 | `000a8b4b488018acbdea366b71a55274300b5a2c5be49ceb6e94ea7deb89c601` |
| `phase_b_stderr.bin` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The nine central identities were byte-identical before and after B: probe, parent, test, scientific runner, wrapper, frozen native, guard, replay adapter and central ledger. The old fixture did not exist before B; therefore there is no old fixture SHA to compare. The SHA above is the one new fixture identity.

Old A evidence remains in place. Committed logs for attempts 1 and 7-11 have their previously registered SHA values; post-run hashes matched. Attempt exit receipts other than attempt 11 were not all individually committed before B, so their pre-B SHAs are registered as unknown where absent from the manifest. Their post-run identities include attempt 8 `f1b2f662800122bed0ff255693df89c4487fbdcf453d3524a42d4ec20c3d9c04`, attempt 9 the same, attempt 10 `f1b2f662800122bed0ff255693df89c4487fbdcf453d3524a42d4ec20c3d9c04`, and attempt 12 `13bf7b3039c63bf5a50491fa3cfd8eb4e699d1ba1436315aef9cbe5711530354`. Attempt 8 remains a real 25-case overrun and is not retroactively capped.

## Boundary

Because case 15 failed, this run does **not** qualify the real Torch weights-only fixture flow. It does not imply a general Torch backend defect, and it says nothing about S1, PPO, models, training, environments or scientific qualification. No optimizer/trunk/environment quota was used; the historical S1B counts remain unchanged.

## Post-close erratum

Read-only PowerShell reconciliation of `phase_b_cases.jsonl` with `phase_b_result.json` compared all 15 durable records by `case_id`, `name`, `passed` and `got_type`. All 15 matched. Case 15 remains the single normal `False`; `case_failed=0` remains only the exception counter.

The unexplained phrase in the earlier section is now bounded by source and Git-interface evidence. Frozen `r75_s1_runner.py` at line 1064 calls `git_value("--abbrev-ref", "HEAD")`; because `git_value` inserts the repository option before the arguments, this becomes the invalid command `git -C <repo> --abbrev-ref HEAD`. A direct read-only check returned exit 129 with `unknown option: --abbrev-ref`. The correct command, `git -C <repo> rev-parse --abbrev-ref HEAD`, returned exit 0 and `exp/round75-native-ppo-qualification`. Since runner `git_value` returns `"unknown"` on a nonzero Git result, the B's runtime branch would have been `unknown` while the parent marker had the real branch. That explains the observed `head_identity_consistent=false` without asserting that the runtime branch was separately saved: it was not in case 15. This is evidence-based diagnosis, not a reclassification to pass.

Scope erratum: the unchanged central set verified before and after B was exactly three S1E source files, five scientific source files, and the central ledger. This report does not claim a complete before/after inventory for the broader old `raw/runtime` set or every old S1/S1B/S1C/S1D original; no such complete before list was taken for B. Historical evidence is retained, not reconstructed.

Evidence erratum: the verdict's replay adapter SHA was initially truncated and is corrected to `0034fa40fbb55362646c84e8ef987e1aa01d72c619bd1c82a244dc88a0224dc2`. No original B output was changed, and no retry or test was run.
