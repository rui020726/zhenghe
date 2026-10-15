# ROUND75 S1G A-R1 Failure Closeout

Request: `R75_S1G_AR1_PRECHECK_BASELINE_REPAIR_20261015`  
Branch: `exp/round75-native-ppo-qualification`  
Base HEAD: `cf7a44033a0dd9e83763e5d6b995bda70a4f74ec`  
Base parent: `ff5a1409812633dfcce00288858576960b77ec39`

This is a failure closeout only. The two AR1 Python processes were consumed, the fixed suite did
not pass, and no third process, source repair, Torch/NN/optimizer/PPO/environment work, or Phase B
run was started.

## Execution identity

- Porcelain before commit: four modified in-scope sources plus the untracked historical R59 log.
- The complete relative diff against `cf7a44033a0dd9e83763e5d6b995bda70a4f74ec`, including the new
  AR1 plan, is frozen as `ROUND75_S1G_AR1_FAILURE_DIFF.patch`.
- No test logs, receipts, checkpoints, runtime originals, or protected files were staged.

## Process receipts

1. `test_r75_s1g_phase_a_syntax_check_exit.json`, PID 39064, exit 1. The wrapper produced an
   invalid inline `-c` program (`SyntaxError: invalid syntax` at `import`). It was a 0-case syntax
   wrapper failure, not a test result.
2. `test_r75_s1g_phase_a_ar1_exit.json`, PID 75188, exit 1. It ran exactly 6 cases with
   1 failure and 2 errors. It must not be described as 12 cases or as passing.

The four stdout/stderr originals and their SHAs are registered in
`ROUND75_S1G_AR1_FAILURE_VERDICT.json` and were not staged.

## Real runtime baseline

The nine runtime originals remain byte-for-byte:

| file | bytes | SHA256 |
|---|---:|---|
| `r75_environment_construction_guard.json` | 223 | `be58a93de378a1fe9e05bc3f2d2cb7059f705b161ef59191c6aab09701c3981b` |
| `r75_environment_construction.json` | 288 | `abd847cfce1c782c7d6a3832557d865083fcf0695e24aab002e54754a39182f6` |
| `r75_joint_record_guard.json` | 398 | `eb004d98cd0689e16f678856be7bae95df99b8b168a4ce1dec9d82aec39bc567` |
| `r75_joint_record.json` | 273 | `08544238cf7e06c57ee22c0d87589d27bff9c259a09eac185375e29e3e0cb1c3` |
| `r75_offline_trunk_target_row_guard.json` | 11329 | `6ac36b9182a0d14f82b7fe645c67fb8b7389282326f5b3cae353d6da5edea8e9` |
| `r75_offline_trunk_target_row.json` | 297 | `5531991a84b76129693a9993eb511354ccfcde0d019b6f35a04b1ffb36244527` |
| `r75_physical_optimizer_step_guard.json` | 1087 | `5d4c65c6de6b4d2c753854fbd4117abd0ab4032220a2f91d56ddb2eb76e9c168` |
| `r75_physical_optimizer_step.json` | 288 | `23a4022bb542fc9e5cef6d4f23cdef66f2795128c689beab904e5dbf054328aa` |
| `r75_total_ledger.jsonl` | 19980 | `fff8dc1f3d9222ed280a5dc9139f3f82de4c5537ccbc69894d340b44e605475d` |

The ledger has exactly 88 events. Charged events are: physical optimizer step 5 x 1 normal;
offline trunk rows 70 events / 840 normal rows (n=1 x19, n=2 x20, n=3 x4, n=4 x18, n=5 x5,
n=168 x4); joint record 1 x 1 normal. Environment construction is 0. The other 12 ledger events
are 4 central opens, 4 legacy zero supersessions, and 4 guard opens.

## Determined failure causes

- The current resources API is `R75Resources.charge_offline_rows(n, what, *, category='normal')`
  at `r75_guard.py:315`. There is no `R75Resources.charge_trunk_forward`. The row guard does have
  a separate `charge_trunk_forward(matrix, what, *, category='normal')` seam at `r75_guard.py:139`.
  The test called the nonexistent resources method at `test_r75_s1g_phase_a.py:238` and `:264`.
- The fake optimizer was created as `type("Optimizer", (), {"step": lambda self: None})()` at
  `test_r75_s1g_phase_a.py:241`; it has neither a `calls` counter nor an incrementing `step`.
- The fake Popen assigned `communicate(_self, timeout=None, _timed_out=...)` directly to the
  instance. The outer call `proc.communicate(timeout=...)` therefore misses `_self`. The timed-out
  branch also lacks `kill()` and a second `communicate()`.
- `check_active_production_baseline` checks files, event count, guard/ledger reconciliation,
  per-unit fields, event sums, and residual locks. It does not check the declared
  `active_production_baseline.total_ledger_sha256`; that is why the hash-tamper loop was the
  recorded failure.

## Scope

`r75_frozen.py`, `r75_guard.py`, and `r75_replay_adapter.py` were not edited. Scientific behavior,
caps, and the real runtime baseline were not changed. The commit freezes the failed source and the
failure-only derived documents; it does not qualify the code or authorize Phase B.
