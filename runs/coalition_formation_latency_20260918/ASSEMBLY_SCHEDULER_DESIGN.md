# Assembly scheduler: design, and why it is the one mechanism the evidence supports

Phase 10B. Selected by the ordered decision gate in `COALITION_FORMATION_VERDICT.md`:
counterfactuals A (earliest-belief dispatch) and B (synchronised arrival) each rescue 6/6,
while C (stage-scoped lease) and E (final-tick credit) rescue 0/6. The preference order
puts stage lease first, and it failed its own gate, so dispatch/arrival is the smallest-blast
option that passed.

## What the measurements force

Three facts from `case_reports.json` close off most of the design space before any code:

| fact | consequence |
|---|---|
| `belief_spread = 0` on all six | the letters arrive at detection; nothing to fix in communication |
| `capable_agents_busy_at_detection = 0` on all six | no capable robot was taken; not a capacity problem |
| `coverage_reset_count = 0`, `coverage_work_lost = 0` | coverage never broke; nothing for a lease to protect |

What is left is *when the covering set completed*: 4, 9, 42, 66 and 157 steps after
detection against budgets of 40.8-77.3, plus one target never covered at all. And two of
the six are handoff failures: `20008/target-2` waited **85 steps** after stage 1 completed,
`20070/target-2` waited **66 steps** after stage 0. So the mechanism has to do two things:
form the set sooner, and form it *together*.

## The contract

`integrated_sar/assembly.py`, a pure function over what each robot was told:

* **Candidates** are free robots whose **own** belief contains the task. A robot that has
  not heard about the target is never planned for, and its position is never assumed
  (invariant 9). A robot already under orders is not a candidate either.
* **Travel** is `_travel_steps(agent.position, belief.position, ROBOT_VELOCITY)` -- the
  straight line from *believed* positions at the speed `old_task_env` actually moves
  robots by. No map, no path, no future position.
* **The covering set is minimal by construction**: greedy over the live gap, re-scoring
  each round, so a robot whose only contribution an earlier pick already supplied scores
  zero and is passed over. Redundancy is prevented, not penalised.
* **Synchronisation** is the new term: among robots that fill the same live gap, prefer the
  one whose ETA is closest to the set's current latest, so the set completes on one tick
  instead of trickling in.
* **Feasibility** uses the robot's own believed deadline, never a stage duration.
* **Determinism**: ties break on `agent_id` ascending, so the same input always yields the
  same plan.

`order_tasks` ranks plans by feasibility, then deadline slack, then active load -- and
**has no stage-duration input at all**. That is deliberate: the duration-relevance gate
already showed that ordering by work time rescues none of the six, so a duration term
would be dead weight with a plausible-looking shape.

## Default-off identity

The prototype is a new policy, `coalition_assembly_scheduler`, and a new planner flag,
`assembly_scheduling`, defaulting to `False`:

* with the flag off, `advise` does not call `_synchronise`, so **no code path reads a
  position or an ETA** and the binding arm's pair biases are exactly what the planner
  produced;
* with it on, the term is **additive and never negative**, bounded by `assembly_weight`
  (0.15) which sits well inside the `[-1, 1]` band the other policy terms respect. It can
  break a tie between two robots for one task; it cannot overturn a large preference, make
  an illegal action legal (invariant 6), or pull a robot towards a task it has not heard
  about (invariant 9).

`test_assembly_scheduler.DefaultOffIdentityTests` pins the mechanism rather than diffing
two runs: the flag defaults off, the off arm reports zero synchronised dispatches, and both
settings stay inside the band.

## Test matrix

27 tests in `integrated_sar/tests/test_assembly_scheduler.py`, covering the brief's list:
single-stage complementary capability, multi-stage capability change, stage transition,
a ragged arrival, a tight arrival, a slot already covered by another member, a robot
without the belief, a busy robot, a finished belief, an expired belief, an empty fleet,
determinism, id tie-break, task ordering (including that it never reads a duration),
serialisation, and the default-off identity.

## What this design does not claim

* A and B are **analytical feasibility estimates**, not measurements of this prototype.
  The prototype must clear the small-scale gate on its own; `ASSEMBLY_GATE.md` records
  that result.
* The two handoff failures are addressed only **indirectly**: the scheduler plans the first
  cover, and at a stage boundary the requirement changes and `task_update` releases the
  members whose abilities no longer contribute, so the next stage is re-planned from
  scratch. Whether that is enough is measured, not assumed.
* No network, tensor, observation shape or checkpoint is touched. The policy is
  interpretable and deterministic; nothing is trained.
