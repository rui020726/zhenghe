# ROUND75 S1F Git Identity Repair - Phase B

## Status

`FUNCTIONAL_PASS_WITH_CAPTURE_COMPLETENESS_NOT_CLOSED`. This is the single
authorized S1F Phase B execution. It is not a retry of S1E, a scientific S1
pass, or authorization for further CPU work.

## Frozen Execution Identity

- Repository: `D:\code\整合\zhenghe-round42`
- Branch: `exp/round75-native-ppo-qualification`
- Execution HEAD: `d2a634facbc25bea1acd7be3e01474ddca0a5a08`
- Execution parent: `70339f7efd57be07946cf8665815534675787517`
- Frozen source SHAs were re-verified before launch and again after B; all
  remained unchanged.

## Single Launch

One hidden parent launch used frozen `r75_s1f_parent_capture.ps1` from the
round75 directory. The parent PID was `49616`. It started one real CPU Python
child, PID `56120`, with frozen
`D:\project_env\python.exe r75_s1f_fixture_receipt_repair.py --run-b`.

The observation shell was PowerShell 7.6.5. The parent capture record reports
Windows PowerShell `5.1.26100.9444` and OS build `10.0.26200`. These are kept
separate because they are different processes.

## Capture Completeness Boundary

The parent observation exited `0`, but this must not be promoted to a Python
child exit. `phase_b_parent_capture.json` records `child_exit_time` as
`missing`, `exit_code` as null, and an `exit_read_error`. The actual child OS
exit code and exit time therefore remain unknown. No retry, repair, rerun, or
inferred value was used.

The functional evidence below is valid independently, but end-to-end process
capture completeness is not closed.

## Functional Results

`phase_b_result.json`, all 15 durable JSONL rows, and stdout all agree:

- cases attempted: 15
- cases completed: 15
- durable `passed=true`: 15
- durable `passed=false`: 0
- exception-counter `case_failed`: 0
- distinct case names: 15

Case names in order: `torch_same_clone`, `torch_one_byte_different`,
`torch_dtype_different`, `torch_shape_different`, `numpy_same_copy`,
`numpy_one_value_different`, `numpy_dtype_different`,
`numpy_shape_different`, `python_nested_same`, `python_type_different`,
`mixed_payload_same`, `mixed_payload_different`, `rng_differences_all_same`,
`rng_differences_torch_one_byte`, `default_torch_save_load_fixture`.

| Operation | Attempt | Completed |
|---|---:|---:|
| NumPy import | 1 | 1 |
| Torch import | 1 | 1 |
| Fixture save | 1 | 1 |
| Fixture load | 1 | 1 |
| Fixture read | 1 | 1 |
| Case exception | 0 | 15 |

Case 15 used the default runner snapshot path and then loaded only its own new
fixture with `weights_only=true`. The fixture is 19,497 bytes with SHA-256
`69d6472683ccd7067dad15db35467d35584b05ea3815e7e2f53e56843a438113`. Scientific
checkpoint loads were 0.

## RNG Identity

Before and after NumPy/Torch import and all cases, the complete RNG identity
records in `phase_b_result.json` are equal:

- Python tuple repr SHA-256:
  `165c6cd377a442d963eb4df20cfbffc55459c9c37f0c1778b693583a61003043`
- NumPy MT19937 keys SHA-256:
  `c034fe2b67ed775bfe451737ada67643b85b20912b4c5e5974f425e9d5f3b3ef`,
  position 623, cached Gaussian 0
- Torch uint8 state SHA-256:
  `698521d01a06e64f5a3dd2aaca18aa6de7e917aa785dfd11acd2069eb1fc77d2`

The result explicitly reports `rng_unchanged=true`. No seed, random draw, RNG
set-state, CUDA, model construction, optimizer, PPO, environment, training, or
evaluation was performed.

## Inventory

Before B, `B_before_inventory.json` had SHA-256
`f52c88f149f37f6033b3521a2d7d50f64444e423cfdd2bf34303bdb70544165d` and was not
rewritten after launch. It contains 108 file entries. Its scope label says
`raw_other_complete_except_s1f_probe`, but the actual entries include 6 existing
Phase A evidence files under `raw/s1f_probe`. That scope field is therefore
treated as a historical label, not as a scope proof. It also duplicates the 9
`raw/runtime` entries already covered by its 78-entry raw scan.

After B, `B_after_inventory.json` has SHA-256
`d6ef2b3b17fe17ce6bf68c466ee14cb0376740b7053d523013c3f0286ad049b2` and contains
108 entries. It scans the full raw tree excluding itself, plus 21 registered
top-level evidence files. Under normalized roots, all 99 unique before entries
remain unchanged. The 9 newly added raw artifacts are `B_before_inventory.json`,
the 7 `phase_b_*` durable records, and the fixture. There are no changed or
missing prior entries. mtimes are not used as authenticity evidence.

The before inventory itself was initially written outside the repo because the
writer used a relative path. The 55,186-byte original with SHA-256
`a956904d44cd98629f17e6d823549c16cdc20d5724aec67e50dd61fbb9a440d9` was moved
into the intended path and its repo field corrected before launch. No Python,
Torch, environment, or scientific execution was involved in that bookkeeping
fix. It was not O_EXCL-created in place.

## Unchanged Material

Protected files retained these SHA-256 values:

- `ROUND46_CURRENT_TASK.json`:
  `cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b`
- `ROUND46_R2_V6_STOP_STATE.json`:
  `93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18`
- `r59_analysis_run.log`:
  `ccc35a2420d6cc1ac81d0b745a977b9f7fc93e2e0ac7997054ec9aac4b130271`

The frozen runner, F probe, F parent, F test, wrapper, native module, guard,
adapter, and central ledger all retained their frozen SHAs.

## Consumption

Actual S1F B consumption: one parent launch attempt, one real CPU Python
process, one NumPy import, one Torch import, one default-stage save, one
weights-only fixture load, one fixture read, and 15 case attempts. There were 0
real case failures, 0 scientific checkpoint/model loads, 0 optimizer/PPO
operations, 0 environments, 0 CUDA operations, and 0 scientific ledger charges.

Raw artifacts and the fixture are intentionally not committed.