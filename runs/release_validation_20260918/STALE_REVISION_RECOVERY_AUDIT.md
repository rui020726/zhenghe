# Stale-command rejection and recovery audit — 400-seed independent block

Branch `release/coalition-controller-validation`, brief section 3. Evidence:
`stale_revision_audit.py` over all 800 lifecycle logs of the committed 400-seed block →
`STALE_REVISION_AUDIT.json`. No protocol, label or statistical definition of that block was
touched; this reads its artefacts.

**Bottom line.** The refusals are correct, they are the *same* behaviour in both arms, and
they are not a recovery problem: every refused command ran nothing, every affected robot
held its previous work, and every one received a command it was allowed to carry out one
tick later. No repeated stale attempt, no permanently occupied robot, no invalid command
executed. Evidence and one new targeted test were added; no production change was needed.

## 1. Where a `stale_task_revision` refusal comes from

Tick order in `orchestrator.py`: `sync_tasks(registry.snapshots())` at `:92` writes the
adapter's revision cache **before** `executor.step()` at `:94` runs the environment tick that
can complete a task; only then are `active_snapshots()` read (`:102`), the allocator builds
commands (`:141`) and the gate inspects them (`:143`). For a task still active, the stamp is
that tick's live revision (`task03.py:672-699`), so the gate's comparison is equal by
construction. For a task that **completed during this tick**, the registry drops it from
`active_snapshots()`, so `_command_revision` falls back to the cache — which is exactly one
revision behind, because the completion bumped the counter after `sync_tasks` ran.
`assignment_gate.py:113-119` then refuses it as `stale_task_revision`.

So the shape is a same-tick completion race, and it predicts that **every** such refusal
should name a task whose status is already `completed`. That prediction is what the data
tests, and it holds for all 24 events.

## 2. The two candidate-only events, in full

| | seed 30215 / target-2 | seed 30388 / target-0 |
|---|---|---|
| refused at | t = 572.0 | t = 23.0 |
| command | `assignment:757`, issued 572.0, valid until 582.0 | `assignment:18`, issued 23.0, valid until 33.0 |
| revision carried by the command | 24 | 7 |
| registry revision at inspection | 25 | 8 |
| revisions seen for that target in that tick | 24 (pre-step), 25 (completion) | 7, 8 |
| when the task updated | same tick, at the step immediately before the decision | same tick |
| task state at refusal | `completed`, stage 1, progress 1.0, remaining all zero | `completed`, stage 0, progress 1.0, remaining all zero |
| belief the decision used | the robot's own, for a task it still believed open | same |
| assignment/reservation before | executing target-5 since 553.0 | executing target-1 since 14.0 |
| assignment/reservation after | refused command applied nothing; re-proposed and accepted at 573.0 | refused command applied nothing; accepted at 24.0 for live target-5 |
| did work state change | no — the previous assignment was untouched | no |
| next valid update and recovery | one tick (cache refreshed by `sync_tasks` at 573) | one tick |
| same stale command re-sent | no | no |
| harm to the current or another task | none; the target it went to had already finished, and the robot was then sent there to settle its belief on scene | none |

30388 is the plain case: refused at 23, sent to a live task at 24. 30215 is the interesting
one — the robot was re-commanded **to the completed task** one tick later and travelled to
it, which is exactly what the cached stamp exists for (`test_a_closed_task_is_settled_on_
scene_and_the_live_task_runs`): the registry has closed the task but invariant 9 forbids
telling the robot so, so on-scene verification is what updates its belief.

## 3. The four metrics, kept separate

| Metric | baseline | candidate |
|---|---|---|
| rejected commands, by reason | 11, all `stale_task_revision` | 13, all `stale_task_revision` |
| other reasons (`expired_command`, `unknown_task`, `missing_task_revision`) | 0 | 0 |
| invalid commands actually executed | **0** of 489 444 | **0** of 499 875 |
| commands refused before reaching the executor | 11 | 13 |
| recovery latency to the next command the gate let through | 1 tick in 11/11 | 1 tick in 13/13 |
| repeated attempts of the same stale command | 0 | 0 |
| robots left with no later command in the episode | 0 | 0 |
| refusals whose target was already `completed` | 11/11 | 13/13 |
| episodes with ≥1 refusal | 6 | 8 |

The candidate's two extra refusals are two more robots exposed to the same race, not a new
kind of event: the mechanism re-decides more robots, so more commands are built, and two of
them happened to be built in the tick a task finished.

One correction worth recording: an earlier pass of this check reported 11 and 13 "executed
with a stale revision". Those were the refused commands themselves — the logger writes the
`assignment` row when the allocator **builds** the command, before the gate inspects it, and
the paired `assignment_rejected` row names the same `command_id`. Counting only unpaired
rows, the real number is zero in both arms. `executed_invalid_check` now pairs by command id
and reports `commands_rejected_before_execution` separately from `assignments_executed`.

## 4. Acceptance criteria

* **A refusal produces no execution side effect.** The gate appends only to `accepted` and
  calls the executor when that list is non-empty (`assignment_gate.py:56-77`), so
  `env.apply_action` never runs; measured directly in 2.
* **No previously valid work is lost.** In all 24 events the robot was under an
  already-executed command and kept it.
* **No infinite retry and no permanent occupation.** 0 repeated attempts; `pending_replan`
  is consumed once when merged into the next advice (`ReplanCoordinator.advise`,
  `assignment_gate.py:136-142`), so the request is one-shot, and the robot's own decision on
  the next tick either passes or is refused again for a *new* reason, which would show as a
  second refusal — it did not.
* **Recoverable once legal information arrives.** Recovery is exactly the next
  `sync_tasks`, one tick, in 24/24 cases.
* **Gate and hard mask unchanged in strength.** Same rule, same thresholds, no edit; the
  baseline arm shows the identical reason distribution, so the candidate did not weaken or
  bypass it. Invariant 6 (mask last) is untouched by this round's change and its tests pass.

## 5. Tests

Already pinned before this round: `AssignmentGateTests` (expired and stale refusals execute
nothing, current revision executes, no-task command never stale) and
`ReplanSchedulingTests.test_rejection_replans_once_on_the_next_tick` (exactly one replan,
not a loop). Added here: `StaleTerminalBeliefTests.test_a_task_completing_mid_tick_is_
refused_once_and_then_travelled`, which reproduces the observed tick ordering — cache
written, task completed, command built one revision behind — and asserts a single refusal
whose revision is exactly one behind, no change to the robot's execution state, a accepted
command on the following tick, and the belief settled on scene.
