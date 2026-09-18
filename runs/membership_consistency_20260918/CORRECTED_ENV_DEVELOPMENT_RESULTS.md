# Corrected-environment development results — environment repair separated from controller effect

Branch `fix/task-membership-consistency`. Cells and episodes, per-cell code state:
environment repair at `98a34b3`, controller guard at `d194c31`, both switches default off.
Evidence: `membership_measurement.py` → `CELL_SUMMARY_dev.json`, `CELL_SUMMARY_stress.json`,
`single_33011/`, `divergence_33011.py`.

This is a placement experiment on already-read seeds. It carries no generalization claim;
the only purpose is to say which part of the earlier result came from the environment and
which from the controller.

## 1. The four cells

| Cell | Environment | Controller |
|---|---|---|
| A | roster as it has always behaved | binding baseline (`coalition_planner` + sticky, both switches off) |
| B | roster as it has always behaved | release + suppression, coverage-preserving test, next-stage guard |
| C | roster consistency enforced | binding baseline |
| D | roster consistency enforced | release + suppression + next-stage guard + source-stage gate |

Episodes: the 15-seed development block (5 failure seeds, 10 fixed normal controls) and the
17 stress episodes that contained the 19 harmful releases. Switches-off identity under the
final code was re-checked against the committed binding arm: 5 645 decision rows across 5
seeds, **IDENTICAL** on every field.

## 2. Development block, 15 episodes

| | on-time / detected | expired | coverage resets | lost work | ghost ticks | dual-roster robot-ticks | release refusals |
|---|---|---|---|---|---|---|---|
| A | 81 / 90 | 8 | 1 | 1.20 | 1 | 0 | 0 |
| B | 90 / 90 | 0 | 0 | 0.00 | **156** | **597** | 0 |
| C | 81 / 90 | 8 | 1 | 1.20 | 1 | 0 | 0 |
| D | **90 / 90** | **0** | **0** | **0.00** | **0** | **0** | 1 |

## 3. The 17 stress episodes that contained the 19

| | on-time / detected | expired | resets | lost work | ghost ticks | dual-roster | refusals |
|---|---|---|---|---|---|---|---|
| A | 131 / 131 | 0 | 0 | 0.00 | 0 | 0 | 0 |
| B | 131 / 131 | 0 | 0 | 0.00 | **195** | **538** | 0 |
| C | 131 / 131 | 0 | 0 | 0.00 | 0 | 0 | 0 |
| D | 131 / 131 | 0 | 0 | 0.00 | 0 | 0 | 0 |

## 4. The four questions, answered separately

**What does the environment repair do to the baseline?** Nothing measurable. A and C are
identical on every column, on both episode sets: same 81/90, same 8 expired, same single
reset, same 1.20 lost work. That is the expected shape — with no releases there is no
re-tasked-but-still-listed robot to detach — and it is why the repair can be evaluated
without re-baselining anything.

**Did the old candidate's gain depend on ghost capability?** It could not have been
distinguished from it, and that was the real problem with the earlier claim. Under the old
bookkeeping the candidate's releases left a robot credited to a task it had abandoned: 156
ghost ticks and 597 dual-roster robot-ticks on the development block, 195 and 538 on the
stress episodes. Any of those could have been carrying completions, and no measurement in
cell A or B could tell. In cell D the same 90/90 on-time appears with **zero** ghost ticks,
so the gain is now demonstrably not resting on the roster artefact.

**Does the candidate still help under correct member semantics?** Yes, on this block: 81 →
90 on-time, 8 → 0 expired, with no coverage reset and no lost work. The gate refuses one
release on the development block and none on the stress episodes, so the effect is delivered
by the planner, not by forcing commands through.

**Does the honest environment expose real coverage breaks?** It exposes them where they
exist, and one episode shows one that the old environment hid — see §5. It also shows that
cells B and D do not lose targets either way, so nothing previously reported as "no
regression" turns out to have been a regression once the mask is removed. The single
exception is 33011, which the old environment recorded as a 2.531-step loss and 1 reset and
the corrected pair resolves.

## 5. Seed 33011, all four cells

Run separately and kept out of the pre-registered aggregates above.

| Cell | on-time | resets | lost work | ghost | dual-roster |
|---|---|---|---|---|---|
| A | 9 / 9 | 0 | 0 | 0 | 0 |
| B | **8 / 9** | **1** | **2.531** | 11 | 59 |
| C | 9 / 9 | 0 | 0 | 0 | 0 |
| D | **9 / 9** | 0 | 0 | 0 | 0 |

The stress block's single new failure survives under the current controller on the unfixed
environment (B) and is gone on the fixed one (D), with the loss now visible as a real reset
in B rather than absorbed silently.

## 6. The targeted gates

| Scenario | Result | Pinned by |
|---|---|---|
| roster residue minimal case | 20 ghost ticks → 0 | `membership_min_repro.py`, `test_an_accepted_reassignment_stops_the_old_task_being_covered` |
| two same-ability members released together | exactly one leaves | `test_two_mutually_redundant_members_do_not_both_leave`, `test_the_second_of_two_over_provisioned_members_is_still_released` |
| stage moves between planning and execution | refused, no side effect, robot keeps working | `test_a_release_whose_source_stage_moved_is_refused_and_runs_nothing` |
| two commands compete for one remaining unit | no coverage reset, no lost work in D | cell D columns, `test_one_robot_never_supplies_two_tasks_at_once` |
| destination already closed at the gate | refused once, recovered next tick | `test_a_task_completing_mid_tick_is_refused_once_and_then_travelled` |
| robot re-picks its current task | true no-op, clock untouched | `test_re_picking_the_current_task_is_a_true_no_op` |
| the 19 episodes | 0 harmful acted-on releases of 118 | `HARM_SCAN.json` |

Gate requirements: **zero** ghost advancement and **zero** cross-task double contribution in
D on both episode sets; **zero** executed releases that lowered real coverage; refusals that
change nothing; and no stall from repeated refusals — 1 refusal across 15 development
episodes, 0 across 17 stress episodes, with the affected seeds still completing everything
they completed in A.

## 7. What these cells do not license

A, B, C and D all sit on seeds that have been read for other purposes, at 15 and 17
episodes. They establish that the mechanism works in an environment whose bookkeeping no
longer hides the failure mode; they do not estimate its effect size on unseen scenarios, and
the difference D−A must not be read as a policy improvement percentage. The only column pair
that judges the controller under correct semantics is **C against D**, and even that is
placement-scale.
