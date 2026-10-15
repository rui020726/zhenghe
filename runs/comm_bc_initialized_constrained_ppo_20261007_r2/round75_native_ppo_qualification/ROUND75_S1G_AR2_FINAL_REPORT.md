# ROUND75 S1G A-R2 final closeout

Status: technical source-preparation suite passed on the final authorized pure-stdlib Python process. This is not a communication-effect result. T2 and B9 remain separate decisions; B9 is still unreleased.

## Result

- Attempt 1: real failure. Python PID `73416`, verified OS exit `1`, `6` top-level tests with `errors=1`. The visible failure was the erroneous `snapshot["total_ledger_sha256"]` read in the original guidance. Consumption is `1 / 2`, with no refund.
- Attempt 2: real pass. Python PID `23816`, verified OS exit `0`, `6` top-level tests, `OK`. Consumption is now exhausted at `2 / 2`.
- No third Python process was run, and no source was changed after attempt 2.

## Process identity

Attempt 2 ran:

```text
D:\project_env\python.exe D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2\round75_native_ppo_qualification\test_r75_s1g_phase_a.py
```

Working directory:

```text
D:\code\整合\zhenghe-round42
```

The two attempts and their raw logs/receipts are separate original evidence. The T1 receipt labeling correction remains: T1 did not verify the real nine-file `raw/runtime` before/after identity.

## Runtime budget summary

The real `raw/runtime` nine-file set was byte-identical before and after attempt 2. The central ledger remains 19,980 bytes with SHA256 `fff8dc1f3d9222ed280a5dc9139f3f82de4c5537ccbc69894d340b44e605475d`. The declared active baseline remains 88 events, with:

| Counter | Used normal | Used reserve | Charged events |
|---|---:|---:|---:|
| `physical_optimizer_step` | 5 | 0 | 5 |
| `offline_trunk_target_row` | 840 | 0 | 70 |
| `joint_record` | 1 | 0 | 1 |
| `environment_construction` | 0 | 0 | 0 |

No canonical charge, technical registration, environment construction, real model work, Adam/optimizer/PPO work, RNG restore, or B parent/child process was performed by this A-R2 closeout.

## Source identity

Final source SHA256 values:

| File | SHA256 |
|---|---|
| `r75_s1_precheck.py` | `88b88c3854f25f81a1398d37066b5d62f4941a7529b349fe5a9d20361733625b` |
| `r75_s1g_outer_exec.py` | `24693d8818ea5e99d7f0f80941f8258fe1ab0b5d07b05d13ec14908c0c619223` |
| `test_r75_s1g_phase_a.py` | `ff81446f3c12558870cea589cdf4b8671c4fb3ae53a2b63f745d8e6aef358474` |

The frozen runner/native/guard/adapter/AR1-plan identities are unchanged.
