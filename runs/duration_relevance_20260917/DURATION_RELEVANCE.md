# Stage-duration relevance gate: the dropped duration has nothing to say about these six

**Verdict.** The gate **fails**. Of the six remaining `candidate_ordering_error` expiries in
`analysis_legacy` under binding, duration-aware ordering rescued **0**, against a required
**3**. At the stronger test — replanning whole episodes with a coalition planner that is told
the work times its belief discards — it rescued **0** and *lost* one target that previously
completed. `AgentTaskBelief` is unchanged, `feat/belief-stage-duration-slack` was not created,
and no Phase B arm exists. This directory is the whole result.

| target | legal sightings | still inside its believed budget | chosen in those | best margin (steps) | promoted by duration | covered work owed when last seen |
|---|---|---|---|---|---|---|
| `20003/piloc-e1-target-1` | 17 | 2 | 1 | 56.9 | 0 | 0.239 of a 10.24 step stage |
| `20003/piloc-e1-target-0` | 734 | 2 | 1 | 50.8 | 0 | never covered at all |
| `20008/piloc-e1-target-2` | 8 | 6 | 6 | 97.3 | 0 | 0.710 of an 11.71 step stage |
| `20052/piloc-e1-target-3` | 11 | 10 | 2 | 44.3 | 0 | 0.266 of a 3.27 step stage |
| `20070/piloc-e1-target-2` | 8 | 6 | 2 | 71.2 | 0 | 0.551 of a 7.55 step stage |
| `20083/piloc-e1-target-4` | 6 | 5 | 2 | 47.9 | 0 | 0.285 of a 9.28 step stage |
| **total** | **784** | **31** | **14** | | **0** | |

Three numbers carry the verdict.

**1. The duration never once says "impossible".** Subtracting arrival time and the covered
work still owed from the believed budget leaves **44 to 97 steps of margin** at the best
decision each target ever saw. A duration term can only do one thing to an ordering: push a
candidate that cannot fit its own work behind one that can. With nothing infeasible, there is
nothing to move — across all 784 re-scored decisions, one of the six was promoted **0 times**.

**2. When these targets do appear, they are already lost.** 31 of 784 legal sightings
(4 %) happened while the target was still inside its believed deadline. At the other 753 a
duration-aware allocator's honest opinion is "give up sooner", not "pick this one" — and that
is the direction it actually moved things (the one non-zero demotion, on
`20008/piloc-e1-target-2`). Of the 31 in-budget decisions the run already chose the target in
**14**, including **all 6** where `20008/piloc-e1-target-2` was feasible and top of a menu of
one. Ordering was not where the loss was hiding.

**3. Five of six died by less than one tick.** They reached 96.4-98.0 % whole-task progress —
91.9-97.7 % of the stage they stood on — and were left owing **0.24-0.71** steps of covered
work. The sixth was never covered at all. Two of the five lost their budget to assembly
latency: `20003/piloc-e1-target-1` had a believed deadline of 85.1 and its first covered tick at
**161.0**, and `20083/piloc-e1-target-4` had deadline 70.9 against first cover at **69.0** — 1.9
steps of budget in front of a 9.28 step stage. The other three had cover within 4-42 steps of
detection and simply needed two-thirds of a tick more than the deadline allowed. Deadline expiry
here is soft (`task_allocation/old_task_env.py:1098-1108`) — it does not release the coalition —
so the work kept flowing after the deadline and the log shows it nearly finishing. Neither shape
is a menu-ordering problem: no reordering recovers 157 steps of assembly latency, and none
manufactures two-thirds of a tick.

## Why the term is structurally near-vacuous here

The scenario generator builds the deadline *out of* the work time
(`task_allocation/old_task_env.py:210-216`):

```
estimated = travel_time + total_work + 30.0 + 0.3 * total_work + U(0, 50)
deadline  = base_time + max(estimated * tightness, estimated * 0.95)
```

So "drive there and do the work fits inside the deadline" is a property of how the scenario was
made, not a fact an allocator has to discover. On these six the stage durations are 3.27-11.97
steps, whole-task work 9.28-28.31 steps, the largest travel ETA any of them ever faced is
**1.19** steps, and their `deadline_offset` is 62.7-105.6. This is the same wall the
`deadline_aware` oracle hit in the learnability gate (0 of 16 expiries fixed): the feasibility
arithmetic was never the missing information.

## How the gate was run

Both levels, on the five seeds carrying the six failures (20003, 20008, 20052, 20070, 20083).

**Precondition — the analysis runs on the recorded run.** Each seed was replayed from its
committed `run_config` with the arm's own configuration, and every one of the **5 645** decision
rows was compared field by field against the committed `coalplan0.25sticky` trace: **0
differences**, with the recording hooks attached. `arm_configuration` refuses to continue if the
rebuilt configuration resolves to a different backend than the recorded run.

**Information legality.** Every number comes from what the deciding robot had been told:
the candidate set, the mask and the ranking from the decision record; position, deadline,
stage index and whole-task progress from that robot's own `AgentTaskBelief` snapshot; and the
stage durations from the `DetectionMessage` deliveries *that robot accepted*, gated on
`first_received <= now`. `CoordinatorInput.tasks` (the registry truth) is never read — the
recording hook deliberately captures only `typed_input.agents`. No robot's durations are ever
shared with another.

**The duration is the same number the simulator counts down.** `CommRelay._build_message`
copies `snapshot.stages` whole, `Task03Adapter._inject_task` writes `"time": float(spec.duration)`,
and `Task03Adapter.deliver_detection` then keeps `requirements` and drops `duration`. So this is
not an estimate of the environment's clock — it is that clock, in the message, discarded on the
way into the belief. Nothing needed to be estimated, and no second duration semantics was
created: the reusable primitives are the environment's own stage spec plus
`coordinator._travel_steps` at `ROBOT_VELOCITY`.

**Decision level.** Every allocation decision re-scored: travel ETA, current-stage remaining
work, future-stage work, deadline slack, `margin = slack - eta - remaining`, the order the run
used and the order the same candidates get once the work time is subtracted.

**Plan level — the decisive one.** `DurationAwarePlanner` subclasses the binding arm's own
planner and changes only its read on work time; five full episodes per mode:

- `prune` — drop a (robot, task) pair whose own belief says it cannot arrive and finish. 44
  pairs pruned across the five seeds. **Every outcome in every episode identical.**
- `rank` — same margin as one bounded term in the planner's row value. **0 of the six rescued;
  `20083/piloc-e1-target-3` went `completed -> expired`.** Net −1.

## What this does not rule out

- The six are the **residual** attribution class — `candidate_ordering_error` fires when nothing
  else did. This gate says the *work duration* is not the missing discriminator for them. It
  does not prove no reordering of any kind helps, and it is one formulation per mode
  (`horizon 50`, `weight 0.35`), not a sweep.
- The feasibility term is per-candidate and cannot see contention. The plan-level run can, and
  still found nothing.
- `first_cover_at` and the progress figures come from lifecycle rows, which are written on state
  change; a gap in them is not evidence of an interruption. Checked directly: for all five
  covered targets the within-stage fraction never moved backwards and whole-task progress never
  regressed, so the refill at `old_task_env.py:1154` never fired on a stage they were working.
  Their loss is late start and slow finish, not thrown-away work.
- The believed deadline is a single value per target across all its sightings, so stale-belief
  deadline drift is not what produced the 4 %-in-budget figure.
- The evidence also shows a third shape, outside this gate's scope: because expiry is soft, the
  coalition kept working and `20003/piloc-e1-target-1` reached 97.7 % of a stage starting at
  t = 161 against a deadline of 85.1. A *hard* duration-based filter would have thrown that work
  away sooner. Making the near-finish land inside the deadline is a scheduling question, not an
  ordering one, and this gate says nothing about it either way.

## Consequence and next lever

Phase B is cancelled by its own gate: no `AgentTaskBelief` field, no `DetectionMessage`
propagation, no `coalition_binding_duration` arm, no legacy-100 evaluation, no pre-registration.
The cost of this result is one analysis directory.

If the six are to be fixed, the measurement that separates them from the successes is not
*can this robot finish this task* — it is **how long until a covering coalition actually forms**.
`first_cover_at - detected_at` on these six is 4, 9, 42, 66 and 157 steps, plus one target that
was never covered at all, against a total budget of 62.7-105.6 steps. That is a coordination
latency and execution continuity question, which is where the learnability gate already pointed,
and a duration term in the belief does not answer it.

## Reproduce

```
D:/project_env/python.exe runs/duration_relevance_20260917/duration_relevance.py record
D:/project_env/python.exe runs/duration_relevance_20260917/duration_relevance.py analyse
D:/project_env/python.exe runs/duration_relevance_20260917/duration_relevance.py counterfactual prune
D:/project_env/python.exe runs/duration_relevance_20260917/duration_relevance.py counterfactual rank
D:/project_env/python.exe runs/duration_relevance_20260917/duration_relevance.py gate
```

Artefacts: `DURATION_GATE.json` (every column of the table above, per target),
`duration_facts.json` (the per-decision re-scoring), `counterfactual_{prune,rank}.json` (the
episode-level outcomes), `deliveries_s<seed>.json` (what each robot was told, and when). `record`
and `counterfactual` rewrite the replay traces into this directory; the travel ETAs and the
progress-monotonicity checks quoted above are a direct pass over those traces and the committed
lifecycle logs. Nothing here is imported by `integrated_sar`.
