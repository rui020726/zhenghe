# ROUND34 execution basis and preflight

Date: 2026-09-30 (workspace date)

## Scope and limits

This round permits a minimal experiment-runner reward-capture correction and pure-compute tests. It forbids starting a SAR environment, opening a new stream, training, optimizer updates, changing objective or policy semantics, editing archived outputs/checkpoints, networking, and touching sealed blocks. Work is limited to the ROUND32/33 runner lineage and new ROUND34 artifacts.

## Repository and branch

- Repository: `D:\code\整合\zhenghe`.
- Verified ROUND33 delivery commit: `c7d47d25e638cbe3728cf36e3a22b9cfc563a353` (`audit: record round 33 temporal credit findings`).
- ROUND32 delivery head `01e5a677cd18a7abfdb97585315ba05251d4e0de` is an ancestor of the ROUND33 head.
- Created `fix/comm-task-reward-event-timing` at the verified ROUND33 head. It existed neither as a local branch nor as an alternate starting point at creation; no reset or overwrite was used.
- Initial tracked worktree was clean. Existing unrelated untracked files were left untouched and are excluded from all ROUND34 commits.

## Source order and clock axes

| Order | Code action | Relevant time / state |
|---:|---|---|
| 1 | The policy is sampled inside `CommRelay.tick`; the cost tap records the relay counter delta after that call. | The relay advances to tick `t`; the corresponding tape row uses that relay tick when a decision row is emitted. |
| 2 | `Task03Adapter.step` calls `env.step(dt)`. | The physical environment advances to its new time for tick `t`. |
| 3 | `Task03Adapter.step` reads task progress and constructs `ProgressEvent(observed_at=env.current_time)`. | The event time is the post-step environment time, e.g. `t`; event creation happens after the wrapped raw `env.step` has returned. |
| 4 | `JointSAROrchestrator.step` publishes returned progress events synchronously. The registry subscriber was installed in the orchestrator constructor. | Registry state is committed during publish. A reward observer installed later on the same bus runs after the registry subscriber. |
| 5 | The orchestrator advances `clock.now` to the decision time. | During the synchronous event publication, `clock.now` is still the pre-step time; immediately afterwards it becomes the event's post-step time. |
| 6 | A later strategy sample begins. | The previous completion is already in registry state before the next policy sample. |

The old `v2_run._install_taps` wraps the raw `environment.step`, then scans registry snapshots as soon as the raw step returns. At this point the adapter has not constructed or returned the new `ProgressEvent`; the orchestrator has not published it; and the registry still holds the previous state. On the next tick the scan sees the committed completion but labels it with that later `relay._tick`. This is a capture-order problem; the bundler's “latest tape decision at or before reward tick” rule is separately correct.

## Raw-record instance checked

The archived `ACTIVE_ROWS`, seed 11, episode 0, scenario seed 1300027 data are present. For `piloc-e1-target-3`:

- the lifecycle record is a `completed` progress event at timestamp `8.0`, with `expired=false` and progress event id `task03-progress:e0:piloc-e1-target-3:8`;
- a tape row exists at step 8, followed by a tape row at step 9;
- the archived episode row has task reward 1 for this target at `task_stream[9]` (and total `on_time=6`, `task_reward_total=6`).

This one event is independently explained by the source order above: its physical completion is at relay/tape tick 8, while the old tap first sees the updated registry inside the next tick's raw `env.step` and writes the unit at tick 9. This confirms the ROUND33 failure mode for a concrete target. ROUND33's aggregate counts remain a prior audit result; they are reusable only where the audited source and data identity match.

## Runtime and checkpoint preflight

- Explicit runtime available: `D:\anaconda\python.exe`, Python 3.13.5, NumPy 2.1.3.
- PyTorch import fails with `ModuleNotFoundError: No module named 'torch'`; there is no available project virtual environment. No package installation or download was attempted.
- The 18 ROUND32 pre-update `.pt` files and their JSON sidecars/index are present. Their bytes can be hashed locally, but without PyTorch they cannot be safely deserialized by the project loader, so tensor state, optimizer state, parameter summaries, module hashes, forward, and likelihood are not verifiable on this machine.
- ROUND33 reports 18/18 byte hashes equal the index and 6/18 metadata matches. Those statements are not upgraded to checkpoint-content verification here. ROUND34 will record each validation layer independently and mark every unavailable layer explicitly.

## Execution counters at preflight

| Activity | Count |
|---|---:|
| New SAR environment starts | 0 |
| New input streams | 0 |
| Real training updates | 0 |
| Synthetic optimizer updates | 0 |
| Existing checkpoint writes | 0 |
| Network attempts | 0 |

