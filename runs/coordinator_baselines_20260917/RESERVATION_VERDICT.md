# A coalition reservation interface, and whether a rule planner can use it

**Verdict.** The interface is the lever, and a deterministic planner that uses it clears
the gate. Adding an agent-specific *reservation* -- this robot covers this named
capability slot of this task, until this time, at this believed revision -- to
`CoordinationAdvice`, and planning the coalition with a belief-only greedy set-cover,
takes held-out expiries from **53 to 16 (−70 %)** and completion from **0.910 to 0.973
(+6.3 pp)** against `rs0.25` on 100 fresh seeds, with paired differences whose 95 % CIs
exclude zero (`completion +0.0623 ± 0.0310`, `expired −0.370 ± 0.184`). Group idling
falls with it: `short_group_idle_steps` summed over all targets −47 %, and
`members_waiting_steps` −64 % (median 54 → 13 steps). No illegal action reached the
executor in any run, no registry fact reached a decision, and the pre-existing arms
reproduce their earlier numbers exactly, so nothing here is a regression.

The phase's premise was that the *interface*, not the reward scale, was the binding
constraint. That is now measured, not argued: the earlier coalition-aware policy changed
20 of 35 838 development decisions because a candidate's own belief rarely contains a gap
it could fill, while this one changes the *dispatch*, and 37 of 100 seeds improve.

## What was built

| Stage | Artefact | Default behaviour |
|---|---|---|
| A | `CoalitionReservation` on `CoordinationAdvice.reservations`, keyed `(agent, task)`: capability slot, slot dimension, believed task revision, issue/expiry, score bounded to `[-1, 1]`. `Task03Adapter` folds a held commitment into the *same* bounded residual channel as `pair_bias`, still before the environment hard mask | field defaults empty; with no reservations the residual arithmetic is unchanged, so old behaviour, old checkpoints and every existing test are bit-identical |
| B | `CoalitionPlannerCoordinator` (`--coordinator coalition_planner`): deterministic, belief-only greedy set-cover over the free robots and each task's open unit slots | `--coordinator heuristic`, the untouched `HeuristicCoordinator` |

PILOC, the CTDE allocator and every checkpoint are frozen. No weights were written, no
training was started, and the planner is a rule policy: it exists to test the interface,
not to be the answer.

## The interface, and what makes it safe

A per-pair bias scores a robot against a task in isolation, so two robots that would fill
the same capability slot are scored identically and nothing stops the allocator from
sending both. A reservation names the slot. Occupancy is exclusive by construction, so
the robots a plan sends are the robots the plan chose.

`remaining_requirements` is read as a flat list of unit slots: `(3, 0, 0, 0, 0)` has
slots 0, 1 and 2, all in dimension 0, and a robot with one unit of that ability covers
exactly one. Claiming a slot adds the robot's abilities to the task's committed set,
which hands the next robot in that dimension the next unit index -- that increment is
the exclusivity.

The four safety properties the rest of the architecture rests on are all preserved:

* **invariant 5** -- the planner only ever *advises*. Its reservations ride the ordinary
  `residual_scale` channel, so `final = ctde + offset + bias` still holds exactly, and
  the traced `bias` vector still re-derives `final`.
* **invariant 6** -- the environment hard mask is still the last thing applied. A
  commitment on a masked action cannot select it.
* **invariant 9** -- the planner never reads `CoordinatorInput.tasks`. Every number on a
  robot's row comes from that robot's own `AgentTaskBelief`; the only other input is the
  team's own assignment plan (`AgentSnapshot.current_task`), which is the allocator's own
  issued order, not a belief and not a registry fact. A robot holding a stale belief is
  planned against its stale view. Nothing is ever published on the pool-level `task_bias`,
  which would broadcast one robot's knowledge to robots the message never reached.
* **lifecycle** -- the advice's `valid_until` is capped by the earliest reservation
  expiry, so a cached advice cannot apply a stale commitment. Each cycle reports why a
  previously held commitment is gone: *expired* by the clock, *superseded* by a revision
  bump or knowledge that left the robot's view, or *cancelled* because the new plan
  declined to re-issue it. `reset()` clears the store and the numbering.

Two further decisions worth recording, because both are cases where the heuristic's
behaviour was deliberately not copied:

* a robot that is not free is left out of the pair channel **entirely**, which is a
  stronger guarantee than the busy penalty it replaces -- a large enough positive term
  could still outweigh a penalty, and cannot outweigh absence;
* a task the robot's own belief has put past its deadline leaves **both** channels rather
  than being biased downwards. Urgency saturates at 1 for a deadline that has already
  passed, so a priority term for it would be the strongest possible pull towards dead work.

## Data protocol

* **Development** -- 30 seeds (`10000-10029`) from `runs/joint_eval_20260916/streams`,
  the same block the two earlier baselines were designed on. Read repeatedly; selects
  arms and nothing else.
* **Held-out** -- 100 seeds (`20000-20099`) recorded by `heldout.py record` in the
  previous phase, before this arm existed. One-shot: no threshold and no weight was
  chosen on them.
* **Replay audit** -- 700 `(run, seed)` pairs verified event by event against the
  recorded stream, **0 mismatches, 588 detections per arm**. No paired difference below
  can come from a different search history.
* **Regression check** -- `rs0.25` and `coal0.25` were re-run end to end after the contract
  and trace changes. Every one of the recorded lifecycle fields for all 100 seeds each is
  identical to the summary the previous phase left behind, field for field. (The log is
  byte-reproducible too, but it records the argv it was given, so two invocations that
  spell the same path differently differ in that one record; re-running one seed with the
  identical argv reproduces its checksum exactly.)
* Differences are **paired per seed** with a 95 % CI; per-arm means are shown for
  orientation only.

The planner arm is a *later* arm in the sense that matters: the development set selected
it. The held-out number is therefore a confirmation of a development decision, not an
independent selection -- the test set can reject the dev choice and cannot be used to pick
another one.

## Held-out result (100 seeds, 588 targets)

| arm | completed | expired | never assigned | completion | mean time | util. |
|---|---|---|---|---|---|---|
| no coordinator | 469 | 119 | 50 | 0.798 | 41.1 | 0.084 |
| `cal0.25` | 524 | 64 | 0 | 0.891 | 36.8 | 0.095 |
| `rs0.25` | 535 | 53 | 0 | 0.910 | 37.4 | 0.089 |
| `coal0.25` | 538 | 50 | 0 | 0.915 | 37.1 | 0.087 |
| **`coalplan0.25`** | **572** | **16** | **0** | **0.973** | **29.0** | **0.066** |

The last column is the fleet's own reported utilisation. The planner's is the *lowest* of
any arm while completing the most targets: the fleet spends less of its time travelling
without producing work, which is the same story the completion time tells from the other
side (29.0 against 37.4 steps).

Paired against `rs0.25` (`mean ± 1.96 s.e.` over 100 seeds):

| comparison | completion | expired | expired targets |
|---|---|---|---|
| `nocoord − rs0.25` | −0.1131 ± 0.0409 | +0.660 ± 0.240 | +66 |
| `cal0.25 − rs0.25` | −0.0185 ± 0.0273 | +0.110 ± 0.162 | +11 |
| `coal0.25 − rs0.25` | +0.0051 ± 0.0073 | −0.030 ± 0.044 | −3 |
| **`coalplan0.25 − rs0.25`** | **+0.0623 ± 0.0310** | **−0.370 ± 0.184** | **−37** |

The last two rows are the consistency check as much as the result. `coal0.25 − rs0.25`
reproduces the previous phase's `+0.0051 ± 0.0073` and `cal0.25 − rs0.25` its
`−0.019 ± 0.027`, digit for digit: the contract change and the metrics work perturbed no
existing arm's lifecycle.

Where the 16 remaining expiries come from -- the previous phase's, for comparison:

| arm | expired | finished after deadline | never assigned | capability blocked | stopped mid work |
|---|---|---|---|---|---|
| `rs0.25` | 53 | 53 | 0 | 0 | 0 |
| `coal0.25` | 50 | 50 | 0 | 0 | 0 |
| `coalplan0.25` | 16 | 14 | 1 | 1 | 0 |

The failure mode has changed shape. Under `rs0.25` *every* expiry is a group that arrived
too late; the planner leaves 14 of those and adds one target that was never assigned and
one blocked on a capability. It did not remove late arrival, it removed most of it -- the
remaining lever is *when* the group is convened, not *whether* the right group is.

## The extra instrumentation

Recorded per run, all of it from artefacts the pipeline already wrote, none of it read
back by a decision:

| what | where it comes from |
|---|---|
| detection → reservation, → group on site, → first step of work | allocation trace, per target |
| `short_group_idle_steps`, `members_waiting_steps` | `expired_attribution` (pre-existing), now reported per arm |
| reservations planned / honoured / broken | allocation trace, per decision under a commitment |
| reservations expired / superseded / cancelled, redundant candidates declined | the planner's own per-cycle counters, summed once per advice revision |
| over-dispatch | abilities on site plus en route in excess of the stage requirement, measured at the target |
| missing-capability vector | the requirement still open at the last progress report at or before the deadline, over expired targets only, by ability dimension |

The missing-capability vector is deliberately **not** read from the decision trace. The
trace only carries a ground-truth block for the target a decision is about, so its "last
look" at a target is normally the pre-arrival one: on the held-out set it reports ~32 % of
*all* dispatched targets as still open on every dimension on every arm, which measures when
the fleet stopped looking rather than what it failed to deliver. Read from the lifecycle
side, at the deadline, it answers the question that was asked.

Held-out, 100 seeds:

| metric | `rs0.25` | `coal0.25` | `coalplan0.25` |
|---|---|---|---|
| dispatched targets ever reserved | 0 / 588 | 0 / 588 | 583 / 587 (never 4) |
| reservations planned (sum over cycles) | 0 | 0 | 1 179 |
| decisions made under a commitment | 0 | 0 | 1 179 |
| … that took the committed task | — | — | 962 (81.6 %) |
| … that went elsewhere | — | — | 217 |
| ended by the plan: superseded / cancelled | 0 / 0 | 0 / 0 | 1 / 356 |
| ended by the clock: expired | 0 | 0 | **0** |
| redundant candidates declined | 0 | 0 | 1 500 |
| detection → group on site, median steps | 31.0 | 31.0 | **15.0** |
| detection → first step of work, median steps | 46.0 | 43.0 | **19.0** |
| `short_group_idle_steps`, sum over all targets | 7 698 | 7 200 | **4 052** |
| `members_waiting_steps`, sum (targets > 0) | 3 374 (65) | 3 176 (62) | **1 219 (38)** |
| stall of the *expired* targets, median (p90) | 67 (135) | 65 (118) | 65.5 (117) |
| missing capability at the deadline, per dimension | [12, 6, 11, 18, 9] of 53 | [12, 5, 11, 16, 9] of 50 | [2, 1, 5, 4, 2] of 14 |
| over-dispatch: targets ever over-supplied | 44.7 % | 44.7 % | 41.9 % |

Five things this says that the headline numbers do not:

1. **The advice window does its job.** `expired` reservations is exactly zero on every
   arm: the plan is re-issued before it can lapse, and `superseded` is 1 across 100 seeds,
   so the beliefs barely move once a task is known. The lifecycle is not leaking stale
   commitments.
2. **The allocator mostly takes the plan, and it is the allocator's choice.** 81.6 % of
   the decisions made under a commitment took the committed task. The remaining 217 are
   the interface working as designed -- the reservation is advice, the allocator decides,
   and the hard mask is still last.
3. **Redundancy is prevented, not penalised.** 1 500 candidate (robot, task) pairs were
   declined across the held-out set because the unit they would supply was already on the
   way. That refusal is the mechanism behind the 47 % drop in group idling, and it is why
   over-dispatch falls as well.
4. **The gain is time, not composition.** At the deadline the expired targets were still
   short in every ability dimension under both arms; the planner's remaining 14 expiries
   are the same *kind* of failure, just fewer of them. Detection → first work halves
   (46 → 19 steps). So what the planner bought is arriving sooner with a group that can
   work, which is exactly the `finished_after_deadline` bucket it emptied.
5. **The surviving expiries are the hard ones.** The stall of the *expired* targets is
   unchanged (median 67 → 65.5): conditioned on being late, the planner is not better.
   The improvement lives entirely in how many targets reach that state.

## Decision gate

| gate | threshold | measured | |
|---|---|---|---|
| expired vs `rs0.25` | ≥ 20 % lower | **−70 %** (53 → 16) | pass |
| short-group idling | clearly lower | `short_group_idle` sum −47 %, `members_waiting` sum −64 % | pass |
| completion | not worse | **+6.3 pp** (0.910 → 0.973) | pass |
| illegal actions | zero | 0 rejected assignments in all 100 planner runs | pass |
| leaks | zero | planner never reads `CoordinatorInput.tasks`; committed by unit test | pass |
| regression | zero | 369 tests pass; `coal0.25`/`cal0.25` reproduce their held-out numbers exactly | pass |

The rule planner passes, so no failure attribution is owed. What follows is the design of
the learned stage, not a second rule experiment.

## Next stage: `LearnedCoalitionCoordinator`

The rule planner is a floor, not a target: it decides the *group* well and the *timing*
only as well as gap-fill plus urgency can. The learned stage should change the ordering
inside the same interface, and nothing else.

**Keep the contract frozen.** `CoordinationAdvice.reservations` does not change. That is
the point of having validated it: the learned module emits the same `(agent, task, slot,
revision, expiry, bounded score)` structure, and every safety property above is inherited
rather than re-argued. In particular the learned output must remain a **slot preference**
-- a ranking over `(robot, task, slot)` triples -- and must not be allowed to express
itself as a pair bias, because a pair bias is exactly the unsaturated form the counterfactual
measured as inert.

**Learn the ordering, not the feasibility.** The planner's two hard parts -- which slots
exist, and who may occupy one -- stay rule-based and non-negotiable. What is learned is
the score that orders the greedy rounds, i.e. a bounded residual on top of
`_row_value`'s gap-fill and urgency terms. Initialise it at zero, so at step 0 the learned
policy *is* the rule planner and training can only improve it.

**Training signal.** The three things the interface now measures are the reward:
`honoured − broken` (did the allocator take the plan), the reservation's expiry, and the
step delay from detection to first work, attributed back through `source_event_id`. This
is exactly the delayed-credit association the architecture already provides for, and it
avoids the trap the reward work hit: nothing here needs a terminal-reward term, because
the outcome is a per-target time, not an episode return.

**Safety and fallback.** Outputs must be bounded and masked the same way the rule planner
is: only free robots, only slots that exist, and a NaN or missing score falls back to the
rule plan. Deterministic (argmax) evaluation must be a first-class mode from the start --
the gate is measured on it.

**Evaluation protocol.** Reuse this phase's protocol unchanged: frozen detection replays,
the audit as a precondition, paired per-seed differences with a 95 % CI on fresh seeds,
and the same gate table. A learned coordinator that cannot beat `coalplan0.25` on that
table has not earned its parameters, and should be reported as such rather than tuned
into a win.

**Known risks.** (1) The remaining 14 late expiries are a *timing* problem across the
search/allocation boundary. The plan already reaches almost every target (only 4 of 587
dispatched targets were never reserved), so the ceiling is not the plan's coverage: it is
that no free robot exists early enough. Measure the robot-busy share and the
detection → reservation p90 (14 steps here) before attributing the rest to the learner.
(2) The whole result rests on the detection streams being frozen while the search policy
is unchanged: a different PILOC checkpoint invalidates every number here and requires
recording again. (3) `residual_scale` was fixed at `0.25` for all three policies; the
commitment weight (`0.35`) was never swept, so the planner's edge is measured at one point
of a two-dimensional budget rather than at its optimum. (4) The gain is in *when* the
group arrives, not in *what* it can do: 81.6 % of decisions under a commitment took it, so
one decision in five still goes elsewhere, and the learner should be judged on that ratio
as much as on completion.

## Reproduce

```bash
# the planner arm, held-out
python runs/coordinator_baselines_20260917/heldout.py run --only coalplan0.25
python runs/coordinator_baselines_20260917/heldout.py audit      # 700 pairs, 0 mismatches
python runs/coordinator_baselines_20260917/heldout.py report

# the development set, including the new arm
python runs/coordinator_baselines_20260917/dev_sweep.py run --mode full --cal 0.25 --coalition --planner
python runs/coordinator_baselines_20260917/dev_sweep.py report --mode full

# the small machine-readable summary this document quotes
python runs/coordinator_baselines_20260917/reservation_verdict.py
```

`reservation_verdict.json` holds the numbers above as parsed aggregates and is versioned;
`replay_audit.json` holds the stream audit. Both are small. The raw lifecycle logs,
decision traces, streams and `heldout_summary.json` are runtime artefacts and are not
versioned.
