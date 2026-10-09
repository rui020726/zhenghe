# ROUND58 gen3 resume prelaunch evidence

Status: exact historical restore and actual resume-entry verification passed for all three seeds; fixed execution is not started until the next versioned freeze/receipt is committed.

## Approval basis

The owner approval relayed in the current user instruction is recorded in `ROUND58_GEN3_RESUME_APPROVAL_SOURCE.json`, sourced from thread `01a0a044-592f-73e3-838c-19c1d79465d3`, quote: “核验通过后批准完成剩余固定计划”. Scope is episode 4 onward, 372 training episodes plus 640 fixed evaluation cells, with no replay of the first 12 episodes or original three batch updates and no parameter/quota expansion.

## Exact restore

`ROUND58_GEN3_APPROVED_RESUME_SOURCES.json` maps exactly one qualified immutable source per seed. `audit_r58_gen3_exact_resume_restore.py` completed with exit code 0. For seeds 11/22/33 it checked checkpoint+sidecar hashes, model parameters, all 10 Adam states and step=8, lambda/history, cursor `(episodes_done=4, updates=1, optimizer_steps=8)`, and Python/NumPy/Torch RNG equality to the approved payload. No optimizer step, PPO update, environment, claim write, checkpoint write, or batch replay occurred. Three authorized in-memory RNG restores were performed by the audit.

## Actual entry seam

`audit_r58_gen3_resume_entry.py` completed with exit code 0 through the actual configured R46/R45/R40/R42 objects and actual `train_seed` loader path. All three seeds selected only their exact mapped gen3 checkpoint and reached episode 4 / scenario 1300003 immediately before the guarded episode provider. The audit intercepted six requested checkpoint saves in memory and made no environment, optimizer, PPO, checkpoint, claim, or journal writes. Historical accounting independently closed at 12 normal claims, 0 reserve claims, 24 optimizer calls, 3 PPO calls derived from unique committed batches, and 12 direct checkpoint journal rows (4 per seed). A temporary resume output root was used only by the audit; the persistent `round58_gen3_resume` directory remains absent.

## Narrow regression

`run_r58_resume_narrow_tests.py` recursively flattens nested unittest suites, fixing the earlier harness-only aggregation failure. It discovered 28 tests, excluded one historical-source test because Git object `b38b868` is unavailable locally, and executed 27: 27 passed, 0 failures, 0 errors, exit code 0. The current source and resume seam are also `py_compile` clean.

## Next step boundary

Before any real continuation, commit this evidence and the current private source changes; then create a new recovery-mode freeze and authorization receipt bound to that committed code, source map, existing cumulative quota, scene audit, output/checkpoint roots, and approval source. The actual entry is not a training authorization by itself. No execution has begun in this prelaunch evidence stage.
