# Redundant-release safety review: five defects found, fixed, and re-measured

Review of the mechanism shipped in `6915d63`, on `validation/coalition-redundant-release`,
against the six checks the brief prescribes. Every finding below is a real defect in code
that had already passed a closed-loop gate — the gate could not see them because each
needs a configuration the development seeds did not produce, or a metric the gate did not
compute. All are fixed before any held-out number exists.

## 1. Joint release could strip a task's coverage — fixed

`release_redundant_members` judged each working robot on its own, treating every other
"under orders" robot as coverage. Two robots covering the same single unit each look
redundant *because of the other*; if both have somewhere else to go, both are planned
away and the task is left uncovered. Independent verdicts merged without recomputing the
resulting coverage.

`_joint_redundant` now decides in one deterministic pass: candidates are walked in sorted
order and each accepted release **leaves the pool before the next robot is judged**, so
after the whole release set is applied the task still covers its believed requirement.
If the stage was fully covered before, it is fully covered after — by construction, not
by hope. Pinned by `test_two_mutually_redundant_members_do_not_both_leave` (two dim-4
robots, two open dim-4 alternatives: exactly one may leave).

## 2. A robot in transit is not coverage — fixed

The filler pool was `committed` — every robot whose snapshot says the task, **including
one still `moving`**. The environment counts only arrived members, so a traveller was
being used as the reason to release a robot that was actually on site doing the work.
Now the pool and the candidate set are both restricted to `ONSITE_STATUSES`
(`"working"`), which is the environment's arrived-and-contributing state. Two tests pin
both directions: a travelling teammate may not make an on-site robot redundant, and a
travelling candidate is never released (it is already committed to arriving).

Belief `members` are deliberately **not** used as the coverage source: their freshness is
bounded by message delivery, and the coordinator's snapshot view of each robot's own
`current_task`/`status` is the same source `_renew` has always used — so no new online
information is introduced, and a message-delay or stale-revision scenario cannot invent
coverage that isn't there.

## 3. The switches were unreachable for the planner arm — fixed

`_build_coordinator` returned the plain planner before ever testing
`release_redundant_members`, so an arm asking for the mechanism through the field got the
baseline. The four-arm ablation produced byte-identical results before the fix — that
sameness *was* the bug's signature. Both halves are now independent
`RunConfiguration` fields wired to their own component, with the policy name as the
shorthand for both (`2cc60dc`).

## 4. Arms could share a label — fixed

`arm_name` encoded neither new flag, so `coalplan0.25sticky` covered baseline, suppression
only, release only and both: pooled files would have mixed policies silently, the exact
failure the label exists to prevent. Labels now carry `+redrel` / `+noself`
(`2cc60dc`), and `trace_meta` records both fields beside the arm claim.

## 5. One suppression flag meant two different things — fixed, schema /3

Extending the local `suppressed` for the own-task rule also flipped
`commitment_suppressed_command`, so a declined reassignment was indistinguishable from a
held commitment. The decision record now separates them: `commitment_suppressed_command`
means the commitment held the command, `own_task_suppressed` means the robot re-picked its
own task. Schema `allocation-decision-trace/3` (`a167b3d`), with a test asserting both
fields and the `"replan"` release reason on the same row.

## 6. What the review checked and found already correct

* **One destination per robot per decision**: `_plan`'s `picked` set already guarantees
  it; now also asserted (`test_one_robot_ends_up_at_exactly_one_task`) — two open tasks
  competing for a released robot yield exactly one reservation.
* **A needed member is never released**: the release predicate is `_claim_slot(...) is
  None`, the same test `_renew` uses to stop renewing; a robot that fills an open unit is
  not redundant and keeps its commitment.
* **A declined reassignment reports no success**: `command_id` None,
  `reservation_executed` False, `commitment_overridden` False, `forced_choice` None.
* **No work-clock reset**: with the suppression on, an own-task re-pick issues no command,
  so `working_start_time` and the armed timeout survive; measured as 120 → 0
  re-commands in the ablation.
* **Belief-only inputs**: requirements from the robot's own belief, teammates' abilities
  from snapshots of their own state; the task registry is never read on this path
  (structurally, plus `test_no_registry_truth_is_needed_to_decide_a_release`).
* **Hard mask, terminal beliefs, TTL, reset**: unchanged, and the pre-existing
  `commitment_validity` ladder still stops at its first failed condition.
* **Repeated messages do not duplicate assignments**: replans are one-shot per
  allocation cycle in the orchestrator, and a redirect that lands stops being a release
  candidate on the next cycle (its own task changed), verified in
  `test_after_the_redirect_the_new_commitment_renews_without_a_new_replan`.

## 7. One candidate-only rejection, kept not hidden

Seed 20030: one `assignment_rejected` with reason `stale_task_revision` — a released
robot was sent at a task the registry had already closed while its own belief still
showed it open. The gate refused the command; the release does not bypass legality. It
appears in the release arms only (C and D, the same event) and is carried into the
held-out correctness checklist as a counted, attributed event rather than averaged away.

## 8. Conservative behaviour and where it applies

When the information cannot decide, the mechanism abstains: no belief for the current task
→ not a candidate (existing replan path unchanged); believed deadline passed → no release;
candidate not credibly on site → no release; the plan has no different task → no replan
and no command. Each abstention leaves the trajectory exactly as the baseline would.

## Verification

* Suite 500 passed, 2 skipped by design; smoke passes.
* Dev-seed closed loop re-run after the joint/on-site fixes: target-level outcomes
  identical to the previously committed gate (6/6 rescued, 21→30 completed, 8→0 expired,
  zero churn), so the tightening costs nothing on the cases the mechanism was built for.
* Four-arm ablation re-run under schema /3 with the pre-registered 15-seed set:
  see `MECHANISM_ABLATION.md`.
* Held-out: identity precondition (all switches off) reproduces the committed 868d68a-era
  trajectory on all five development seeds, 5645 decision rows, field by field,
  before any held-out stream was recorded.
