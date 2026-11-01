# Recovery 10 run handoff (2026-11-01)

Status: **G3 PASS, WORLD AUTHORIZATION WRITTEN, TRAINING RUNNING** (2 workers, 2 active runs).

This file lives outside the frozen `revision_05` source closure on purpose.

## What this stage is

Continues only **T1/91002** and **T2/91002** from recovery_09b's completed 5,000-episode
checkpoints to **10,000 completed episodes** each (5,000 + 5,000 = 10,000 new Worlds).
The other four runs are retired at their R9b terminal state and are not continued.

Seed choice: 2026-10-31 DEV_REUSED evaluation (T1/91002 82/192 = 42.71%, T2/91002 104/192 =
54.17%), an exploratory 8-scene reading with no confidence interval.

## Frozen artefacts (revision_05)

- Context `protocol/RECOVERY_RUNTIME_CONTEXT.json` sha256
  `7a3840f396e722a70fe74141ee1600d866e1d8948a932022cb7da8172c83a9d6`
- G3 report `execution/recovery_runtime/G3_REPORT.json` sha256
  `7ee8ada79a8c09af9ac440c3599a3efef658d28d9117c2c62d253e94cb9c0a36`
  (status PASS_RESTORED_OBSERVED, returncode 0)
- World authorization `execution/recovery_world_authorization_01/WORLD_AUTHORIZATION.json`
  sha256 `63d3b75d830e9fd49a6eae54fd74c2942b95878499c30230aaac923de88a1b09`,
  max_worlds 10,000, scope `LONG_RECOVERY_STAGE_TO_COMPLETED_EPISODE_TARGET_ONE_ROOT_ONLY`,
  decision_ref `USER_20261101_ROUTE2_CONTINUE_BEST_SEEDS_TO_10K`
- Pure tests: 20 tests (`test_recovery07_terminal_adapter`, `test_recovery_09_long_stage`,
  `test_recovery_10_retirement`), OK

## Disclosed protocol-adjacent changes (all pinned in this revision's source manifest)

1. `--retire-run ARM:SEED` retires a run at its parent completed target; retired entries get
   `new_world_cap: 0`, `active: false` and `revision_retired: true` (only retired entries carry
   that key, so ancestor run entries re-derive byte-identically).
2. Five reconstruction bounds are scoped to long-stage parents instead of being removed:
   per-run and world bounds in `validate_reconstructed_parent_provenance`, the
   `validate_closed_run` segment bound, the `completed_parent` registered-cap check, and the
   task-certificate disposition in `validate_task_termination`. Ordinary 120-episode /
   720-world stages keep the original rules.
3. `_require_clean_recovery_training` skips inactive runs, and `run_train` no longer counts a
   retired-by-choice run as converged.
4. Only `protocol/RECOVERY_DESCENDANT_STAGE.json` may exceed the 10 MB protocol-artifact cap
   (it carries one closed-row pin per parent World, ~18.9 MB here; allowance 64 MB). Every
   other artifact keeps the frozen cap.

## Parent admission

R9b was launched without a contemporaneous monitor wrapper, so it is admitted through the
observerless post-hoc provenance index at
`execution/recovery_parent_evidence/revision_01/`, with its two poll records transcribed into
`recovery_09b/revision_01/execution/recovery_train_observer_20261101/`. Explicit unknowns are
kept (`observer_preparation: null`, `child_exit_code` unknown, `source_context_auth_drift`
unknown, `training_interpreter_history` unknown); `training_session_id` is the recorded
training PID 17212.

## Failed attempts preserved (not repaired, not reused)

- `revision_01`: partial write; descriptor (18.9 MB) hit the 10 MB cap. Four protocol
  artifacts remain in place.
- `revision_02`: written completely, but its descriptor did not record the retirement list,
  so `_base` could not re-derive its runs.
- `revision_03`: recorded the retirement list but parsed the seed as a string.
- `revision_04`: correct build, but startup refused on zero-length retired intervals
  (`recovery run cursor/cap is not a finite four-episode interval`); 0 claims consumed.

## Training

- Launcher: `execution/recovery_train_run_01/launcher.json` (detached, PID 46936)
- Gate: `execution/ledger/GATE_FIRST_UPDATE.json` written; both runs active
- Measured at 2026-11-01T01:53Z: 5.0 claims/min (~24 s/episode/worker), 50 claims,
  T1/91002 27 / T2/91002 23
- Expected: 10,000 Worlds / 5.0 per min ≈ 33 h, finishing 2026-11-02 early-to-mid evening local

## Restart boundary

A restart may be planned only from a complete 120-episode checkpoint and requires an
independent audit. Do not replay unresolved claims, rerun consumed Worlds, reset the
optimizer, reseed the RNG streams, or merge post-checkpoint claims into the lineage.
