# ROUND76 R1 Failure Audit Appendix

This appendix records an independent read-only audit after the R1-B failure.  It does not
authorize any new parent, child, Python process, test, model, NN, optimizer, environment,
or R2 experiment.  Existing raw evidence, contract, permit, report, and old task state were
not modified by this audit.

## Failure Classification

1. One outer tool command failed before reaching the target parent because
   `C:/Program Files/PowerShell/7/pwsh.exe` did not exist on this host.  This is a failed
   startup command, not a parent invocation, child, or qualification attempt.
2. The first actual invocation of `r76_b_parent_r1.ps1` reached the parent gate and failed
   because `current_stage` was `R1_B_CPU_PARENT_CAPTURE_RELEASED_NOT_STARTED` rather than
   the exact expected `R1_B_CPU_PARENT_CAPTURE_RELEASED`.  It produced no child, claim,
   receipt, or model.  Under the release rule, parent invocation itself consumed the unique
   R1 parent allowance.
3. A text-only correction of `ROUND76_R1_TASK_STATE.json` then set the exact stage string.
4. The second actual invocation of `r76_b_parent_r1.ps1` launched child PID 11364 and failed
   in `configure_qualification_runtime` on `import r68_execute`.  It did not reach model,
   guard construction, optimizer, checkpoint, or environment.

## Actual Consumption

| Scope | Actual | Contract / authorized cap |
|---|---:|---:|
| R1 parent script invocations | 2 | 1 |
| R1 qualification child processes | 1 | 1 |
| A1 pure stdlib Python processes | 1 | 1 |
| A1 distinct cases / executions | 4 / 4 | 4 / 4 |
| Old B1 Python processes / cases | 2 / 6 | 2 / 6 |
| Old B2 parent / child | 1 / 1 | 1 / 1 |
| All R76 parent script invocations | 3 | 2 |

The R1 parent allowance was exceeded by one invocation.  The all-R76 parent allowance was
exceeded by one invocation.  Therefore the R1 technical qualification is invalid and no
scientific claim may be built on it.  A1's four passing pure cases do not repair that
qualification result.

## Source And Protection Recheck

At audit time, the working tree still matched the 14-file source manifest byte-for-byte.
The complete HEAD is `39b6bf966f0a1644fc0b059043d32cccabd8f935`, and the frozen R1 source
head remains `c0b7bce381a7fe516cf92cd856805ba81f8e3c3c`.  The old R75 ledger and both
protected ROUND46 files matched their pinned SHA256 values.

## Failure-Path Diagnostic

`r76_b_driver.run_b2` first validates the release gate, bundle, and input counts, then calls
`configure_qualification_runtime`.  That function inserts `r76_execute.R66_DIR` on
`sys.path`, imports `r66_execute`, and then fails at `import r68_execute`.  The real file
exists at
`runs/comm_bc_initialized_constrained_ppo_20261007_r2/round68_late_initial_confidence_ablation/r68_execute.py`
with SHA256
`c123dfe753effc5b6e35b8ce680e6bb1092a80a0d0deb40a0971df09e1c2cd3e`.  The observed symptom is
therefore a live `sys.path` wiring failure, not a missing file.  Because the traceback stops
inside that import chain, no model, guard construction, optimizer, checkpoint, or environment
operation is evidenced.  However, importing the surrounding runtime chain may have imported
NumPy/Torch as a module-load side effect; that is not evidence of model construction.

No repair, retry, or additional execution is authorized by this appendix.
