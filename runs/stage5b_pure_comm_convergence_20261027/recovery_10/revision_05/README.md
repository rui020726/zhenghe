# Recovery 10: continue the two best seeds to 10,000 completed episodes

Status: **PREPARATION ONLY**. No context, G3 report, world authorization, or training World
has been generated in this root yet.

R10 continues only **T1/91002** and **T2/91002** from recovery_09b's completed 5,000-episode
checkpoints to **10,000 completed episodes** each (2 x 5,000 = 10,000 new Worlds). The other
four runs stay at their R9b terminal state and are not continued here. The seed choice came
from the 2026-10-31 DEV_REUSED evaluation (T1/91002 82/192 = 42.71%, T2/91002 104/192 =
54.17%), which is an exploratory reading on eight used DEV scenes with no confidence interval.

## Disclosed changes to this revision's source (pinned by its own source manifest)

1. `recovery_descendant.next_runs`/`build_next_stage` accept an explicit `retired_runs` set,
   exposed as `--retire-run ARM:SEED`. A retired run keeps its parent cursor and is recorded
   with `new_world_cap: 0` and `active: false` (no new run-entry key is added, because later
   stages re-derive and byte-compare ancestor run entries), and the builder refuses to retire
   a run that is not already sitting at its parent completed target. Retirement is only
   honoured in long-target mode; ordinary 120-episode stages are unaffected.
2. Long-stage parent admission. R9b is an 18,840-world long stage, and the reconstruction
   guards only ever admitted 720-world segments, so five bounds are now **scoped to the
   parent being admitted** instead of being removed: a parent whose context carries
   `target_completed_episodes` is bounded by its own frozen caps, while ordinary bounded
   stages keep the original 120-episode / 720-world rules unchanged.
   * `validate_reconstructed_parent_provenance`: per-run bound = that run's own cap; world
     bound = the sum of the parent's active caps.
   * `validate_closed_run(..., segment_bound=...)`: 120 for ordinary parents, the run's own
     cap for a long-stage parent.
   * `completed_parent` cap check: `parent exceeds its finite registered cap` now compares
     against the parent's frozen `slot_plan.world_cap` for long stages.
   * `validate_task_termination(..., long_stage=...)`: a certificate raised inside a long
     stage must stay immutable while the run closes its cap (`stopped_on_task_certificate`
     false, continuation marker matching), instead of being required to stop at the check.
   * `_long_stage_target_for_runs`: accepts a run that this revision deliberately retires at
     its parent completed count (see 1).
3. Retirement is now visible to the runtime without touching ancestor run entries: only a
   retired run carries `revision_retired: true`, `_require_clean_recovery_training` skips
   inactive runs when checking for pre-existing checkpoints, and `run_train` no longer counts a
   retired-by-choice run as converged (so `certified_run_count`/`convergence_complete` stay
   honest for this two-run stage).
4. `tests/test_recovery_10_retirement.py` covers retirement, the marker and the ordinary-mode
   no-op.
5. `compact.write_bytes_exclusive` takes an optional per-call cap, and `write_bundle` uses a
   64 MB allowance **only** for `protocol/RECOVERY_DESCENDANT_STAGE.json`, because a long-stage
   parent contributes one closed-row pin per consumed World to that descriptor (R9b's closure
   is ~19 MB). Every other protocol artifact keeps the frozen 10 MB cap.
   Two earlier attempts are left untouched as records, not repaired or reused:
   `../revision_01` failed mid-write on the descriptor size cap (four partial artifacts), and
   `../revision_02` was written completely but its context cannot be re-derived, because the
   retirement list was not recorded in its descriptor; `_base` now re-derives retired runs from
   `descriptor["retired_runs"]`. `../revision_05` recorded the retirement list in its descriptor but parsed the seed as a string when re-deriving it, so its context could not be loaded. `../revision_04` was written and re-derived correctly but refused at startup: `_require_clean_recovery_training` called `run_spec` on the zero-length retired intervals (`recovery run cursor/cap is not a finite four-episode interval`). This root is `revision_05`.

Science settings, native physics/allocation, model and PPO hyperparameters, the 58,600-byte
budget, B0=10000, 19 observations, 24 tasks, 1,000 steps, `EPISODES_PER_UPDATE=4`, the
120-episode checkpoint interval, `WORKERS_MAX=3` and `threads_per_worker=1` are inherited
unchanged.

## Parent evidence (recovery_09b, observerless post-hoc)

R9b was launched without a contemporaneous monitor wrapper, so it is admitted through the
generic post-hoc provenance index:

    execution/recovery_parent_evidence/revision_01/{parent_provenance,terminal_summary,pins_before,pins_after}.json

The two poll records live in the parent root at
`recovery_09b/revision_01/execution/recovery_train_observer_20261101/` and are transcriptions
of real supervising checks, written after the fact. Explicit unknowns are preserved rather
than filled in: `observer_preparation: null`, `child_exit_code:
UNKNOWN_NOT_SEPARATELY_RECORDED`, `source_context_auth_drift:
UNKNOWN_NOT_COMPUTABLE_FROM_POSTHOC_PINS`, `training_interpreter_history:
UNKNOWN_NOT_RECORDED`. `training_session_id` is the recorded training PID (17212), not a tool
session identifier; see `observer_note.json` beside the polls. Build them with:

    & 'D:\project_env\python.exe' -B ..\build_parent_evidence.py

## Build

    $R9B = 'D:\code\整合\zhenghe-round42\runs\stage5b_pure_comm_convergence_20261027\recovery_09b\revision_01'
    $R10 = 'D:\code\整合\zhenghe-round42\runs\stage5b_pure_comm_convergence_20261027\recovery_10\revision_05'
    & 'D:\project_env\python.exe' -B -m stage5b_comm.recovery_descendant --parent $R9B --parent-provenance "$R10\execution\recovery_parent_evidence\revision_01\parent_provenance.json" --newroot $R10 --target-completed-episodes 10000 --checkpoint-interval-episodes 120 --retire-run T1:91001 --retire-run T1:91003 --retire-run T2:91001 --retire-run T2:91003

That command is a no-write preview. After checking the reported caps, repeat with `--write`.

## Later phases

    & 'D:\project_env\python.exe' -B stage5b.py recovery-g3 --root $R10 --workers 2 --recovery-context "$R10\protocol\RECOVERY_RUNTIME_CONTEXT.json"

    & 'D:\project_env\python.exe' -B stage5b.py recovery-train --root $R10 --workers 2 --recovery-context "$R10\protocol\RECOVERY_RUNTIME_CONTEXT.json" --recovery-g3-report "$R10\execution\recovery_runtime\G3_REPORT.json" --recovery-world-authorization "$R10\execution\recovery_world_authorization_01\WORLD_AUTHORIZATION.json"

The training authorization must be a separate explicit decision bound to this context, its
G3 report, parent, candidates and target, with scope
`LONG_RECOVERY_STAGE_TO_COMPLETED_EPISODE_TARGET_ONE_ROOT_ONLY` and `max_worlds` equal to
this root's active-run cap.

## Pure tests

    & 'D:\project_env\python.exe' -B -m unittest tests.test_recovery07_terminal_adapter tests.test_recovery_09_long_stage tests.test_recovery_10_retirement -v

## Restart boundary

A restart plan may be calculated only from a complete 120-episode checkpoint. Do not replay
unresolved claims, rerun consumed Worlds, reset the optimizer, reseed the RNG streams, or
merge post-checkpoint claims into the lineage. If this root is interrupted after claims
begin, preserve it and require an independent audit before choosing a new restart stage.
