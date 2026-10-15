# ROUND75 S1G Phase A evidence close-out

This is a read-only evidence close-out. It freezes the failed Phase A source and the four test
attempts; it does not repair source, rerun tests, or authorize Phase B.

## Git state

- Branch: `exp/round75-native-ppo-qualification`
- Base HEAD: `ff5a1409812633dfcce00288858576960b77ec39`
- Base parent: `d2a634facbc25bea1acd7be3e01474ddca0a5a08`
- Tracked modifications at manifest time: `r75_frozen.py`, `r75_s1_runner.py`
- Named ignored-to-be-force-added files: `r75_s1g_outer_exec.py`, `ROUND75_S1G_RUN_PLAN.json`,
  `test_r75_s1g_phase_a.py`, and the S1G report/verdict/stop/task/evidence docs.
- Excluded from the commit: raw output, receipts, logs, checkpoints, protected ROUND46 files,
  and the unrelated R59 analysis log.

## Four real exit receipts

The four exit codes are 1, 1, 0, 1. The first three attempts executed six cases each, for 18
case executions. Attempt 4 failed at import before executing any case.

```json
{
  "kind": "round75-s1g-phase-a-test-exit/1",
  "request_id": "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015",
  "python": "D:\\project_env\\python.exe",
  "test_sha256": "ad4597717e703179eb69746fa64b1355cc5efd0cc1d3e14ff093feb1313e26ca",
  "log_sha256": "9ea37f29098058b271f1fd0547de201d22dba8cd3b257777930037033504905c",
  "exit_code": 1,
  "passed": false,
  "recorded_at": "2026-10-15T07:48:13.5054325Z"
}
```

```json
{
  "kind": "round75-s1g-phase-a-test-exit/1",
  "attempt": 2,
  "request_id": "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015",
  "python": "D:\\project_env\\python.exe",
  "test_sha256": "aa26bf25050a5a87b78aad26bf1083c1f85e442b7978cefa7bd12875d78ff785",
  "log_sha256": "7f1e55e285f89a9764d2118edc2ed605c8e7981412921f1e2d2d24960daae53d",
  "exit_code": 1,
  "passed": false,
  "recorded_at": "2026-10-15T07:49:56.4619592Z"
}
```

```json
{
  "kind": "round75-s1g-phase-a-test-exit/1",
  "attempt": 3,
  "request_id": "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015",
  "python": "D:\\project_env\\python.exe",
  "test_sha256": "5f5e87504b59aa324e5ded84b0864d09d8131b1489e1a09a55a43fd7832770ff",
  "log_sha256": "e987e7caaa7745c77aa353fe6ade2dc4551c208c67306ae7cd6a93bca3c9c06d",
  "exit_code": 0,
  "passed": true,
  "recorded_at": "2026-10-15T07:51:02.0599739Z"
}
```

```json
{
  "kind": "round75-s1g-phase-a-test-exit/1",
  "attempt": 4,
  "request_id": "R75_S1G_NATIVE_QUALIFICATION_RETRY_REVIEW_20261015",
  "python": "D:\\project_env\\python.exe",
  "test_sha256": "01c62fb000c9d61db22e324743f76bceeeb42c5528beebc6164bb3a990c374d4",
  "log_sha256": "98073e8998db562bc235c514814014aef3afe38f2be3c477064d5c3b7f22d41e",
  "exit_code": 1,
  "passed": false,
  "recorded_at": "2026-10-15T08:02:25.0131532Z"
}
```

## Production runtime identity

| File | SHA256 | Bytes |
|---|---|---:|
| `r75_environment_construction_guard.json` | `be58a93de378a1fe9e05bc3f2d2cb7059f705b161ef59191c6aab09701c3981b` | 223 |
| `r75_environment_construction.json` | `abd847cfce1c782c7d6a3832557d865083fcf0695e24aab002e54754a39182f6` | 288 |
| `r75_joint_record_guard.json` | `eb004d98cd0689e16f678856be7bae95df99b8b168a4ce1dec9d82aec39bc567` | 398 |
| `r75_joint_record.json` | `08544238cf7e06c57ee22c0d87589d27bff9c259a09eac185375e29e3e0cb1c3` | 273 |
| `r75_offline_trunk_target_row_guard.json` | `6ac36b9182a0d14f82b7fe645c67fb8b7389282326f5b3cae353d6da5edea8e9` | 11,329 |
| `r75_offline_trunk_target_row.json` | `5531991a84b76129693a9993eb511354ccfcde0d019b6f35a04b1ffb36244527` | 297 |
| `r75_physical_optimizer_step_guard.json` | `5d4c65c6de6b4d2c753854fbd4117abd0ab4032220a2f91d56ddb2eb76e9c168` | 1,087 |
| `r75_physical_optimizer_step.json` | `23a4022bb542fc9e5cef6d4f23cdef66f2795128c689beab904e5dbf054328aa` | 288 |
| `r75_total_ledger.jsonl` | `fff8dc1f3d9222ed280a5dc9139f3f82de4c5537ccbc69894d340b44e605475d` | 19,980 |

## Ledger facts

The total ledger has 88 events. Actual charged summaries are:

- physical optimizer step: 5 charged events, 5 normal, 0 technical;
- offline trunk target row: 70 charged calls, 840 normal rows, 0 technical rows;
- joint record: 1 charged batch read, 1 normal;
- environment construction: 0.

This exactly matches the documented old baseline. It is not a Phase B post-run state and must
not be described as an all-zero system.

## Precheck mismatch

The current S1G runner path calls `r75_s1_precheck.main()`, but `r75_s1_precheck.py` hardcodes:

```python
PLAN_PATH = HERE / "ROUND75_S1_RUN_PLAN.json"
```

The runner does not switch that module-level path when `--run-s1g` is used. In addition, both
plans pin `r75_frozen.py` to
`83813d6e06971a1e1fc1e25d9e9f94b0fa72310e300e037c0fa73d539c540b80`, while the final working
`r75_frozen.py` is
`bb4bae98518cc482f4b733794ee1c3f93097c56c5b3e85d6fed2c65f2a72198e`. The precheck also asserts
a zero ledger, while S1G explicitly starts from the non-zero old production baseline. Therefore
the current S1G launch path is not eligible for Phase B.

## Key source evidence

The complete six-case source remains `test_r75_s1g_phase_a.py`. Its current content is included
in the committed full cached diff and is not re-run here. The current runner license check is
at `r75_s1_runner.py:1607`; it contains the final unverified `IndentationError` at line 1613.
The A1/A2/A3, stage-save, comparison and fee-route call points are cited in
`ROUND75_S1G_PHASE_A_EVIDENCE_MANIFEST.json`. The outer missing/invalid exit fail-closed path is
at `r75_s1g_outer_exec.py:452` and `:491`.
