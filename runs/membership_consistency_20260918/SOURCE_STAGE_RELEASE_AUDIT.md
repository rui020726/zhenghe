# Source-stage release audit — the 19, and the gate that closes the race

Branch `fix/task-membership-consistency`, commits `7bbc3ac` (contract and gate) and
`d194c31` (the controller-side guard and the gate restriction). Evidence:
`release_harm_scan.py` → `HARM_SCAN.json`, `integrated_sar/tests/test_release_source_gate.py`,
`membership_measurement.py` cells.

**Answer to "were all 19 blocked": yes on the episodes that produced them — 0 harmful
executed releases out of 118, on the identical 17 episodes where the count was 19 out of
1498.** But the mechanism that achieved it is not the gate. Read §4 before quoting any
number here.

## 1. What the 19 have in common

All 19 were releases the coordinator judged, acted on, and where taking the robot off its
task would have lowered the coverage the stage really had. Every one of them sits at
**truth stage index ≥ 1**: eighteen at stage 1, one at stage 2. None is at stage 0.

| Episodes and stages of the 19 | |
|---|---|
| A_tighter_deadline | 32022 s1, 32030 s1, 32036 s1, 32038 s1, 32044 s1 |
| B_higher_load | 33008 s1, 33037 s1, 33041 s1 ×2, 33042 s1, 33052 s1, 33055 s1, 33055 s1 |
| C_tight_and_heavy | 34004 s1, 34018 s1, 34034 s1, 34036 s1, 34041 s1, 34050 s2 |

That distribution is the diagnosis: the judgement was made about a stage that had already
been replaced. The belief a robot carries freezes at the moment it was told something, and a
robot that stays on site never re-verifies, so across a stage boundary the belief describes
the stage that finished, not the one now running.

Two of the three earlier signals agree: on the development block the release-candidate
examinations showed 118 of 3 556 with a believed stage differing from the truth (3.3 %), and
13 of those became executed releases.

What is **inferred** rather than directly recorded for these 19: the believed stage index at
planning time. The recorded quantity is the truth stage at the moment of the release. The
inference is the only one consistent with "every case is at a stage boundary" and with the
mechanism in `MEMBERSHIP_CONSISTENCY_AUDIT.md` §3.

## 2. What the gate checks, and where

The advice now publishes, per released robot, the source task with the stage and revision
that robot's own belief carried at the moment of deciding
(`contracts.ReleaseProvenance`, `CoordinationAdvice.releases`). The allocator stamps it on
the command it builds for that robot, and only when the destination is a different task.
`AssignmentGate._inspect` compares the source stage against the registry when the command
arrives, after the existing TTL and destination checks:

| Condition | Result |
|---|---|
| source stage advanced since the judgement | refused, `stale_release_stage` |
| source task gone from the registry | not screened (nothing to compare); the release proceeds |
| source unchanged | executed |
| destination expired or revision-mismatched | refused exactly as before, in the same order |

A refusal runs nothing: the command is dropped before the executor, the robot keeps the work
it had, and the robot is queued for one replan. Nothing is upgraded — the stale command is
never silently re-stamped to the newer revision.

## 3. A false start worth recording

The first version also refused when the source *revision* differed. On the development block
that rejected **979** of the releases and refused 568 on the stress episodes, because the
registry bumps a task's revision on every accepted event — a belief is almost always older
than the live value by construction. A revision comparison is not a stage-consistency test;
it is a "did anything at all happen" test, and here everything happens every tick.

That is the brief's warning made concrete: freshness of the message does not establish
stage identity. The revision comparison is removed, and
`test_a_revision_move_alone_is_not_a_refusal` pins it so it cannot come back looking
principled.

## 4. What actually removed the 19

With the gate restricted to the stage, the refusals are: **1** on the 15 development seeds,
**0** on the 17 stress episodes. So the gate is not what fixed the count.

The controller-side guard is. `d194c31` extends the joint redundancy test to the believed
**next** stage: a robot whose ability is needed by the stage the belief says is coming is not
spare, even when the stage it stands on is covered twice over. That is the same delivered
information, used one field further — `AgentTaskBelief.future_requirements`, which every
message already carries and which nothing in the coordinator had ever read.

| Measurement, identical 17 episodes | before | after |
|---|---|---|
| releases judged | 1 546 | 1 520 |
| releases acted on | 85 | 118 |
| acted-on releases that lowered real coverage | **19 of 1 498 across the full block** | **0 of 118** |
| ghost ticks (stage clock advancing on an absent body) | 195 | **0** (with the roster fix) |
| releases credited to two tasks at once | 538 robot-ticks | **0** |

The gate stays, because the race it closes is real — the plan and the execution are separate
steps, and one refused release on the development block is one release that would otherwise
have gone out against a moved stage. It is just not the load-bearing part.

## 5. Boundaries respected

* The planner and the actor still read only beliefs. The source stage is *this robot's own*
  believed value; the comparison happens at the execution boundary, which has always read the
  registry to validate commands (invariant 9 constrains planning inputs, and the gate is not
  a planner).
* **No new information is fed back into planning.** A refusal produces a replan request,
  which is the pre-existing channel, and the next judgement is made from whatever the robot's
  belief has since become. Making the refusal informative — telling the planner the stage
  moved — would be an interface change, and it has not been made.
* Multi-robot batches: the gate compares each command against the same registry state, so it
  cannot detect two releases that are individually stage-current but jointly strip a stage.
  That protection stays where it was, in the planner's joint accounting, which is why
  `test_two_releases_from_one_source_in_the_same_batch_are_both_refused` pins the narrower
  guarantee it does give: once the source has moved, neither half slips through.
* The residual a belief cannot see: a release judged inside a stage that is correctly
  identified, where the believed requirement is right and the robot nevertheless turns out to
  be needed, is out of reach of any online rule. The roster fix is what makes such a case
  *visible* rather than silently absorbed.
