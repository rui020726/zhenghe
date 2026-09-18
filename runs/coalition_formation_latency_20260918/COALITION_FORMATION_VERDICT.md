# What actually killed the six: assembly and handoff, not belief, capacity or coverage

**Verdict.** The binding planner's six remaining `candidate_ordering_error` expiries are
**not** ordering failures, and the label was wrong. Five of the six delivered **all** of
their work — whole-task progress 1.0 — and are reported expired only because the last
tick of it landed past the deadline (by 1.3, 2.4, 7.1, 10.4 and 85.9 steps; four are
near-misses, one — 20003/target-1 — assembled 157 steps after detection and was never
close). The sixth was never covered at all. Every one of the six had the detection letter
in **every** robot's hands on the detection
tick (`belief_spread = 0`), zero capable robots busy, zero coverage resets, and zero work
thrown away. What separates them from the successes is **when the covering coalition
finished assembling** — and, for two of them, how long the *next stage* waited after the
previous one completed.

Decision gate: the feasibility estimates A (earliest-belief dispatch) and B
(synchronised arrival) are 6/6 potentially feasible; C (stage lease) and E (final-tick
credit) are 0/6. The ordered gate therefore selects **Phase 10B, the assembly scheduler**.
A and B are *analytical feasibility estimates*, not measured policies -- they bound what
the information allowed, leave the effect on other targets unmeasured, and that caveat is
load-bearing and is carried into the design below.

## 1. The environment's stage clock, pinned

`STAGE_PROGRESS_SEMANTICS.md` traces one `step` end to end; 18 tests in
`test_deadline_tick_semantics.py` pin the ten boundary cases. The answers that matter:

| question | answer |
|---|---|
| `time_left` reset when coverage incomplete? | **yes, to the full stage duration** (`old_task_env.py:1152-1154`) |
| assigned-but-not-arrived counts as coverage? | **no** — `members` is appended only on arrival (`:1064`) |
| extra agents change work speed? | **no** — flat `time_left -= self.dt` (`:1121`) |
| stage duration fixed? | **yes**, equal across a task's stages (`:172`) |
| deadline uses `>` or `>=`? | **strict `>`** (`:1102`) |
| `dt` always `self.dt`? | yes in production; `step(dt)` and `task_update` use different variables |

One genuine defect, **reported and deliberately not fixed**: `task_update` runs the expiry
loop before the work loop, so a stage that finishes on the very tick its deadline is
crossed ends with both `finished` and `expired` set, and `Task03Adapter._read_progress`
(`task03.py:1258`) reads `expired` first. Minimal repro, bounded impact and a suggested
direction are in `DEADLINE_TICK_TESTS.md`. It cannot explain these six (none of them
finished), but it can misreport a genuine completion as an expiry.

## 2. The six, measured

Per-target timelines in `stage_timelines.json`, event tables and causes in
`case_reports.json`.

| target | done? | budget | belief spread | arrival spread | first cover | resets | lost work | busy/total |
|---|---|---|---|---|---|---|---|---|
| 20003/target-1 | **yes** | 70.9 | 0 | 157 | 157 | 0 | 0 | 0/6 |
| 20003/target-0 | no | 64.0 | 0 | 14 | never | 0 | 0 | 0/4 |
| 20008/target-2 | **yes** | 77.3 | 0 | 4 | 4 | 0 | 0 | 0/6 |
| 20052/target-3 | **yes** | 40.8 | 0 | 42 | 42 | 0 | 0 | 0/4 |
| 20070/target-2 | **yes** | 72.1 | 0 | 81 | 9 | 0 | 0 | 0/4 |
| 20083/target-4 | **yes** | 58.6 | 0 | 66 | 66 | 0 | 0 | 0/6 |

Three facts close off most of the hypothesis space:

1. **`belief_spread = 0` on all six.** Every robot that could contribute had the letter on
   the detection tick. Communication was not late; `communication_or_belief_late` cannot
   fire.
2. **`capable_agents_busy_at_detection = 0` on all six.** No capable robot was committed
   elsewhere when the target appeared, so neither `capable_agent_unavailable` nor
   `fleet_capacity` can fire.
3. **`coverage_reset_count = 0` and `coverage_work_lost = 0` on all six.** Coverage, once
   formed, was never broken inside a stage, so `complete_coverage_interrupted` cannot fire
   and a stage-scoped lease has nothing to protect.

What is left is when the set completed: 4, 9, 42, 66 and 157 steps after detection, against
budgets of 40.8-77.3. The one target that never completed coverage (20003/target-0) had its
single dispatched robot arrive at t=19 and leave at t=124 without the requirement ever being
met — the coalition was never assembled.

### Primary cause per target

| target | primary | secondary |
|---|---|---|
| 20003/target-1 | `arrival_desynchronised` | `planner_dispatch_late` |
| 20003/target-0 | `planner_dispatch_late` | `arrival_desynchronised` |
| 20008/target-2 | `stage_transition_handoff` | — |
| 20052/target-3 | `arrival_desynchronised` | `planner_dispatch_late` |
| 20070/target-2 | `stage_transition_handoff` | `arrival_desynchronised`, `planner_dispatch_late` |
| 20083/target-4 | `arrival_desynchronised` | `planner_dispatch_late` |

Two of the six are handoff failures, not first-cover failures: `20008/target-2` covered
stage 0 at t=10 and stage 1 at t=18, then waited **85 steps** for stage 2 (first cover
t=111); `20070/target-2` covered stage 0 at t=13 and waited **66 steps** for stage 1
(t=85). At a stage boundary `task_update` releases the members whose abilities no longer
contribute (`:1145-1151`), so the next stage starts uncovered by construction — and nothing
in the current planner plans for it.

## 3. Matched controls

Each failure is compared against the five successful targets nearest in normalised
covariate space over delivery budget, stage count, mean duration, requirement shape,
detection time, active load and capable-robot count (`case_reports.py`,
`matched_controls`). Not hand-picked easy wins: the nearest control to
`20003/target-1` is `20008/target-3` (budget 91.0 vs 70.9), and to `20052/target-3` is
`20083/target-3` (budget 51.8 vs 40.8).

The comparison is unambiguous. Across all thirty controls the median
`coverage_reset_count` is 0 and the median `coverage_work_lost` is 0 — identical to the
failures. The failures differ in exactly one place: **arrival spread**. Controls assemble
in 1-71 steps; four of the six failures take 42-157. The failures are not worse at
holding a coalition; they are worse at *forming* one.

## 4. The five counterfactuals (analytical feasibility estimates)

`counterfactuals.json`. Every one is computed from the recorded timelines plus the
straight-line travel bound the offline attribution already uses; no simulator re-run.
A, B and D never re-run the episode, so their effect on the other targets is
**unmeasured** — the table reports feasibility only, and reports nothing as "net".

| counterfactual | potentially feasible | global effect | gate | online-invisible | kind |
|---|---|---|---|---|---|
| A earliest-belief dispatch | 6/6 | unmeasured | **pass** | no | analytical feasibility estimate |
| B synchronised arrival | 6/6 | unmeasured | **pass** | no | analytical feasibility estimate |
| C stage-scoped lease | 0/6 | — | fail | no | measured (no coverage to protect) |
| D instant communication | 6/6 | unmeasured | pass | **yes** | diagnostic, delta-vs-A |
| E deadline final-tick credit | 0/6 | — | fail | no | measured (ordering only) |

The outcome split the feasibility numbers have to be read against:

| terminal state | targets |
|---|---|
| completed on time | none |
| finished after deadline | 20003/target-1 (+85.9), 20008/target-2 (+10.4), 20052/target-3 (+1.3), 20070/target-2 (+2.4), 20083/target-4 (+7.1) |
| never finished | 20003/target-0 (progress 0.0, shortfall 12.0 steps) |

* **A and B are feasibility estimates, not measurements.** Both assume a perfect
  dispatcher sends the minimal covering set at the earliest legal moment and that travel
  is the shortest straight line. They bound what the *information* allowed. A measured
  policy must still clear the closed-loop gate on its own, and the fact that feasibility
  is 6/6 does not mean an implementable scheduler reaches 6/6 — the prototype's own gate
  (section 6) is the demonstration of exactly that gap.
* **B collapses into A here**, because the six never broke coverage: synchronising arrival
  can only help by preventing a break, and there was none.
* **C is vacuous by measurement, not by assumption**: zero resets and zero lost work means
  a lease has nothing to protect. This is the strongest single result against the
  stage-lease hypothesis.
* **D ≡ A here, and its communication-only delta is exactly 0.0 on all six.** A
  dispatches the tick the last needed robot has the letter; with `belief_spread = 0` that
  tick is already the detection tick, so "instant communication" changes nothing that
  "earliest-belief dispatch" did not already assume. D's 6/6 is **inherited from A's
  dispatch-timing assumption and must not be read as a communication gain**
  (`communication_only_delta_steps: 0.0` per target in `counterfactuals.json`). It is
  retained as a labelled diagnostic and must never enter an online actor.
* **E rescues nothing**: the overruns are 1.3, 2.4, 7.1, 10.4 and 85.9 steps for the five
  that finished late — none inside the one-tick boundary window — and the sixth never
  finished, so reordering the deadline test cannot help it either.

## 5. Decision gate

Applied in the order the brief specifies:

- **A** (final-tick credit ≥ 3/6): **fails**, 0/6. → no `ENV_TIME_SEMANTICS_REVIEW.md`
  escalation is triggered by this gate; the off-by-one is reported as a separate finding.
- **B** (stage lease ≥ 3/6): **fails**, 0/6.
- **C** (earliest-belief dispatch or synchronised arrival ≥ 3/6): **passes**, 6/6 each.
- **D** (only instant communication): not reached.
- **E** (capable agents objectively unavailable): **does not apply** — 0 busy on all six.
- **F** (nothing passes): not reached.

**Selected direction: Phase 10B, the assembly scheduler.** The preference order
(stage lease > earliest dispatch > synchronised arrival > communication) is respected: the
higher-preference stage lease failed its own gate, so dispatch/arrival is the smallest-blast
option that passed.

## 6. Prototype

Phase 10B is implemented as `integrated_sar/assembly.py` plus a default-off option on the
coalition planner, `coalition_assembly_scheduler`. See `ASSEMBLY_SCHEDULER_DESIGN.md` for
the contract, the default-off identity argument and the test matrix.

**Its small-scale gate fails, and the result is in `ASSEMBLY_GATE.md`.** On the five
failure seeds the baseline and the prototype are identical: completed 21 both arms,
expired 8 both arms, one coverage reset and 1.2 lost work-steps in each, zero unjustified
overrides, zero rejected commands, zero deadlocks. The synchronisation term *is* active --
13 of 1085 decision rows in seed 20003 carry a bias raised by exactly `0.15 * 0.25 =
0.0375` -- but it flips **no** choice, because the allocator's own logits separate the
candidates by far more than that. Per the brief's Phase 12 step 7 the batch evaluation is
therefore not started.

So the gate selects the *direction* on the strength of the analytical feasibility
estimates, and the prototype built for that direction does not clear its own gate. That is
recorded as the outcome rather than as a partial success: the prototype is safe, inert by
default and its planner is a usable component, but it does not rescue any of the six.

## 7. What this does not establish

* **A and B are analytical feasibility estimates.** The gate is satisfied by a bound on
  what the information allowed, not by a measured policy, and the effect on the other
  targets in the episode is unmeasured. The prototype's own small-scale gate is the real
  test, and it is reported separately.
* **The six are a residual class.** `candidate_ordering_error` fires when nothing else did;
  this analysis replaces that label with measured causes, but the *class totals* over the
  whole split are unchanged.
* **Two of six are handoff failures**, which the assembly scheduler addresses only
  indirectly (it plans the first cover; a stage boundary re-plans from scratch). Whether
  the prototype fixes them is measured, not assumed.
* **`first_cover_at` comes from lifecycle rows**, written on state change; a gap is not by
  itself evidence of an interruption. Checked directly: within-stage fraction never moved
  backwards and whole-task progress never regressed on these six, so the `:1154` refill
  never fired on a stage they were working.
* **The believed deadline is one value per target across its sightings**, so stale-deadline
  drift is not what produced these numbers.

## Reproduce

```
D:/project_env/python.exe runs/coalition_formation_latency_20260918/stage_timeline.py
D:/project_env/python.exe runs/coalition_formation_latency_20260918/case_reports.py
D:/project_env/python.exe runs/coalition_formation_latency_20260918/counterfactuals.py
D:/project_env/python.exe -m unittest integrated_sar.tests.test_deadline_tick_semantics -v
```

Artefacts: `STAGE_PROGRESS_SEMANTICS.md`, `DEADLINE_TICK_TESTS.md`, `stage_timelines.json`,
`case_reports.json`, `counterfactuals.json`, `stage_timeline.py`, `case_reports.py`,
`counterfactuals.py`. Nothing here is imported by `integrated_sar`.
