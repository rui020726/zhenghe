# ROUND58 Gen3 resume startup stop

Status: **blocked at frozen recovery-plan identity check; no fixed-plan continuation began.** The approved CLI was invoked once, exited 1, and was not retried or hot-fixed.

## Passed before the attempt

- Exact restore audit: all three mapped sources passed. Model weights, complete Adam state and step 8, λ/history, cursor `(4 episodes, 1 update, 8 optimizer steps)`, and RNG matched the source payload. No optimizer/PPO/environment/checkpoint write occurred; the read-only restore audit performed its explicitly authorized in-memory RNG restore for each seed.
- Actual resume entry audit: all three live `train_seed` paths reached episode 4 / scene 1300003 immediately before the guarded episode provider. No environment, optimizer, PPO, disk checkpoint, claim, or journal write occurred.
- Narrow regression: `run_r58_resume_narrow_tests.py` discovered 31, excluded one old-source test because Git object `b38b868` is absent locally, executed 30, passed 30, failed 0, errored 0, exit 0. The nested `TestSuite` aggregation issue is fixed.
- Current scene block recheck was result-blind: 3,372 claim JSON files scanned; no matching claim scene, registered scene identity, third-party reservation, or artifact-path scene ID. The range remains ROUND58's own reservation.
- The recovery authorization and freeze passed `_validate_external_grant` read-only validation and bind the owner approval source, source map, code identity, quota receipt, scene audit, and distinct resume checkpoint root.

## Actual startup failure

The actual CLI was invoked in `recover_first_batch_gate` mode from committed source/freeze. It reached the preserved `round58_gen2/r58_execute.py:1163` recovery-plan equality check and stopped with:

`RuntimeError: recovery plan identity does not match the authorized fixed plan`

This check runs before base-runner loading/configuration, claim allocation, RNG restoration, environment construction, optimizer/PPO calls, or checkpoint writes. Read-only comparison found a single top-level mismatch: `r58_repair_identity`. The current approved resume identity contains the new counter/ledger/resume-source/journal bindings and current private launcher/checkpoint-contract hashes; the existing gen3 persisted plan contains the earlier identity. Specifically, current vs persisted `private_launcher_sha256` is `ab054d410c9cc86869110e76a6fcf3922f86482f9b7cf22a2e971b18792b9d70` vs `56e0b12819b5193437d0ef9b19ca2c6df2040f088f716df5dc1a3874065340c4`, and `checkpoint_contract_sha256` is `d7eac8b9d568890013d2af0167970ef2fee7d3b055f2f4015d5937e3989e8aeb` vs `c23dc0c89d8a6d423591cc684e47cf5a25130a279710243fc60dedf7c39ec95b`; the new resume metadata fields are absent from that old plan. The old plan and all historical receipts were left untouched.

The complete traceback is preserved in `ROUND58_GEN3_RESUME_EXECUTION.log` (SHA256 `8107b64def017d889b83dc5f5823a17d28e35bc2359eb02e0d4c3671231e9033`). The machine-readable field comparison and side-effect ledger are in `ROUND58_GEN3_RESUME_STARTUP_IDENTITY_STOP.json`.

## Accounting at stop

- Historical environments: 16.
- Existing Gen3 normal claims: 12; reserve claims: 0.
- Cumulative actual consumption: 28 / 1064; remaining planned normal jobs: 1012; reserve availability: 24; cumulative capacity: 1036.
- This CLI attempt added 0 claims, 0 environments/simulations, 0 optimizer steps, 0 PPO calls, 0 checkpoint writes, 0 RNG restores, and 0 evaluation cells. The new `round58_gen3_resume/raw` root was not created; the Gen3 claim directory still has 12 files. No Python process remained active.
- No source/freeze/approval/old plan was changed after the failure. No further run, retry, or evaluation was attempted.

## Stop disposition

Following the frozen stop discipline, the identity failure is retained as a blocker. Do not rewrite the persisted plan, weaken equality, or patch-and-continue under this execution identity. A new explicitly approved/versioned recovery identity must specify how the immutable historical plan is reconciled with the resume-only adapter identity; then it must be frozen before any retry. The 372 training episodes and 640 evaluation cells remain unexecuted.

Protected file hashes remain the originally specified values; both files were excluded from all commits.
