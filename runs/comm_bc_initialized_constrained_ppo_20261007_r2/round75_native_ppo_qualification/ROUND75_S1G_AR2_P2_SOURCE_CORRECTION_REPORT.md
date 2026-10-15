# ROUND75 S1G A-R2 P2 source correction

Status: source correction complete, awaiting source review. No Python process was run. T2 and B9 remain unreleased. T1 remains a real failure and consumption of 1/2 with no refund.

## Minimal source correction

`r75_s1_precheck.py` now computes the actual total-ledger SHA256 directly from the real active baseline file using the module's existing `sha256_file` helper:

```python
sha256_file(root / "r75_total_ledger.jsonl") == declared_total_sha
```

No validator was added, no snapshot shape was changed, and no outer logic was changed. The corresponding precheck pins were updated only in `r75_s1g_outer_exec.py` and `test_r75_s1g_phase_a.py`.

## Work identity

- `r75_s1_precheck.py`: `88b88c3854f25f81a1398d37066b5d62f4941a7529b349fe5a9d20361733625b`
- `r75_s1g_outer_exec.py`: `24693d8818ea5e99d7f0f80941f8258fe1ab0b5d07b05d13ec14908c0c619223`
- `test_r75_s1g_phase_a.py`: `ff81446f3c12558870cea589cdf4b8671c4fb3ae53a2b63f745d8e6aef358474`

The five frozen identities remain unchanged:

- `r75_s1_runner.py`: `4f84184a25d9058df97abf04fb00febe1d6b8c123944fcc9814e78651d3835d4`
- `r75_frozen.py`: `bb4bae98518cc482f4b733794ee1c3f93097c56c5b3e85d6fed2c65f2a72198e`
- `r75_guard.py`: `4d730f71fb34ad411043c65141b6bcbb3f7d2f3c4668a6b53054e6400e35d55f`
- `r75_replay_adapter.py`: `0034fa40fbb55362646c84e8ef987e1aa01d72c619bd1c82a244dc88a0224dc2`
- `ROUND75_S1G_AR1_RUN_PLAN.json`: `6303c093d0e19d71081e5c1c5d6393f21733d8f7c9c9fb87ec6fd21eecd7ebab`

## T1 evidence-label correction

The original `ROUND75_S1G_AR2_ATTEMPT1_RECEIPT.json` labels `production_runtime_before_identity` and `production_runtime_after_identity`, but their listed contents are eight source/plan files plus `ROUND75_S1G_AR2_TASK_STATE.json`, not the nine real `raw/runtime` production files. The original receipt and logs remain unchanged. This correction is the authority for that labeling error. T1 did not verify nine-file `raw/runtime` before/after identity.

The actual nine `raw/runtime` files were read separately and match the active `88`-event baseline:

| File | Bytes | SHA256 |
|---|---:|---|
| `r75_environment_construction.json` | 288 | `abd847cfce1c782c7d6a3832557d865083fcf0695e24aab002e54754a39182f6` |
| `r75_environment_construction_guard.json` | 223 | `be58a93de378a1fe9e05bc3f2d2cb7059f705b161ef59191c6aab09701c3981b` |
| `r75_joint_record.json` | 273 | `08544238cf7e06c57ee22c0d87589d27bff9c259a09eac185375e29e3e0cb1c3` |
| `r75_joint_record_guard.json` | 398 | `eb004d98cd0689e16f678856be7bae95df99b8b168a4ce1dec9d82aec39bc567` |
| `r75_offline_trunk_target_row.json` | 297 | `5531991a84b76129693a9993eb511354ccfcde0d019b6f35a04b1ffb36244527` |
| `r75_offline_trunk_target_row_guard.json` | 11329 | `6ac36b9182a0d14f82b7fe645c67fb8b7389282326f5b3cae353d6da5edea8e9` |
| `r75_physical_optimizer_step.json` | 288 | `23a4022bb542fc9e5cef6d4f23cdef66f2795128c689beab904e5dbf054328aa` |
| `r75_physical_optimizer_step_guard.json` | 1087 | `5d4c65c6de6b4d2c753854fbd4117abd0ab4032220a2f91d56ddb2eb76e9c168` |
| `r75_total_ledger.jsonl` | 19980 | `fff8dc1f3d9222ed280a5dc9139f3f82de4c5537ccbc69894d340b44e605475d` |

## Consumption

- A-R2 T1: `1 / 2`, failed with `PID 73416`, verified OS exit `1`.
- A-R2 T2: not released, not run.
- B9: not released, not run.
- P2 Python runs: `0`.
