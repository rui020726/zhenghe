# Release blockers — coalition redundant-member release

Head of this round: `d194c31` on `fix/task-membership-consistency`, on top of
`release/coalition-controller-validation@bed60f9`. Both switches remain default-off on every
branch. Nothing here is merged; the earlier delivery's "all correctness criteria met" and
"recommend default enablement" statements are withdrawn.

**Headline blocker, as stated:** 阶段信息陈旧导致错误释放，且环境旧任务成员残留可能掩盖覆盖损失。
Both halves are now confirmed, and both have a fix — which changes the blocker's status, not
its history.

## Blocker register

| # | Blocker | Status | What closed it | What remains |
|---|---|---|---|---|
| B1 | The release test used a requirement vector from which the candidate's own contribution had already been subtracted, so every sole provider read as redundant | **closed** | `fb1c85b` — compare believed coverage with and without the robot, on the stage's gross requirement | — |
| B2 | 19 of 1 498 acted-on releases removed a capability the stage really needed | **closed on the measured episodes** | `d194c31` — spareness is judged against the believed next stage too | 0 of 118 harmful on the identical episodes; not yet re-measured on unseen scenarios |
| B3 | The environment keeps a re-tasked robot on the task it left, so that task stays covered, keeps advancing its work clock, can finish with nobody present, and can steal the robot back at completion | **fixed, default off** | `98a34b3` — `--task-membership-consistency` | Not merged; the 400-seed and stress results were produced **without** it |
| B4 | No execution-side check existed for the release precondition; only the destination was validated | **closed as designed** | `7bbc3ac` + `d194c31` — `ReleaseProvenance` on the advice, stamped on the command, checked at the gate | Fires rarely (1 refusal on 15 development episodes); the planner-side guard does most of the work |
| B5 | The earlier claim that the mechanism causes no coverage breaks was unfalsifiable while B3 held | **resolved** | cells A/C identical, cell D zero ghost ticks | Any statement about coverage safety now has to cite the environment variant |
| B6 | Stale-command refusals (24 across 800 held-out logs) | **never a blocker** | audited: same-tick completion race, 0 executed, 0 repeats, 1-tick recovery | — |
| B7 | One new candidate-only target failure under load (33011/target-7) | **open, retained against the candidate** | traced: two environment working timeouts, then a coverage reset that the old bookkeeping hid | It does not occur in cell D; that is a reason to re-measure, not a reason to drop it |

## The evidence status of every earlier claim

| Claim as made | Status now |
|---|---|
| "400 unseen seeds, +0.0083 paired, CI [+0.0042, +0.0138], significant" | **Old candidate version evidence.** It measured B1's defective test, on an environment where B3 masks coverage loss. Its standing under B3 has not been re-assessed, and it is relabelled as such in `RELEASE_CANDIDATE.md` |
| "zero coverage resets, zero lost work, zero per-target regressions" | **Confounded, not refuted.** With B3 removed the same measurement can now see loss; on the 17 episodes that contained the 19, cell B still shows 131/131, so no loss appears there — but that is a fresh result, not a validation of the old one |
| "the 1.3 % residue is inherent to online rules and is disclosed" | **Withdrawn.** It was an open correctness failure, and it has since been closed on the measured episodes |
| "recommend default enablement" | **Withdrawn.** No default recommendation is made in either round |
| "+2.3 % travel under load, ~1.5× decision tail" | **Still measured**, on the unfixed environment; needs re-measurement on the fixed one |

## Configuration state

| Switch | Default | Meaning |
|---|---|---|
| `release_redundant_members` | **False** | the planner may re-decide a working robot |
| `suppress_recommand_own_task` | **False** | no command when a woken robot re-picks its own task |
| `check_release_source_stage` | **False** | gate refuses a release whose source stage has moved |
| `task_membership_consistency` | **False** | environment detaches a re-tasked robot |
| `coordinator_policy` | `heuristic` | the coalition planner is not the default either |

Rollback for any of them is leaving it off; for the whole line, `git checkout
fix/coalition-commitment-continuity` (`868d68a`).

## What is worth doing next, and why

1. **Re-measure the four-cell comparison on fresh seeds with the corrected environment as the
   baseline of record.** The current numbers are 15 + 17 placement episodes on
   already-read seeds. C-versus-D on unseen scenarios is the only thing that can support a
   release recommendation again.
2. **Decide the environment repair separately from the controller.** B3 is a real defect that
   affects *any* policy that re-tasks a working robot, including paths unrelated to this
   candidate, and it changes what "completed" means for tasks that finished with nobody on
   site. It should not ride along quietly with a controller merge.
3. **Re-examine the checkpoint question only as a measurement, not a training task.** The
   frozen CTDE actor was trained under the masking behaviour. This round changed no weights
   and started no training; whether that checkpoint still behaves the same on the corrected
   environment is a replay question and is currently unanswered.
4. **The residual a belief cannot see.** A release judged inside a correctly identified stage,
   from a requirement the belief states correctly, can still turn out wrong. No online rule
   can rule that out; the roster fix makes it visible instead of absorbed, and that is the
   most this round can offer.

## Answers to the closing questions

1. **Ghost capability contribution:** yes, confirmed and reproduced minimally — 20 ticks in
   the two-task case, 185 of 232 watched releases producing at least one, 87 with a robot on
   two rosters at once, 177 source tasks finishing while still crediting an absent robot.
2. **What the repair changes in environment behaviour:** an accepted cross-task or
   depot-bound reassignment now detaches the robot from every roster listing it and
   recomputes those stages; re-selecting the current task is a true no-op. Deadlines,
   durations, the random working timeout, the coverage-interrupt reset and the arrival
   criterion are untouched. The baseline arm is unaffected to the tick (A ≡ C).
3. **Are all 19 blocked:** on the identical episodes, yes — 0 of 118 acted-on releases lowered
   real coverage, against 19 of 1 498 before. The guard that did it is the next-stage test,
   not the gate.
4. **Is the corrected candidate still better than the same-environment baseline:** yes on
   these episodes — C 81/90 with 8 expired versus D 90/90 with 0, on the fixed environment,
   with zero ghost ticks and zero coverage resets. This is a placement result, not an
   estimate.
5. **Is a fresh independent test worth arranging for a frozen new version:** yes, and it is
   the only thing that can restore a release recommendation. It should freeze the controller
   and the environment variant together, since the numbers are not comparable across them.
6. **Existing checkpoints on the corrected environment:** unmeasured. No retraining is
   triggered by the environment fix, and none was started; the question is answerable by
   replay alone.
