# Learnability verdict: the residual failures are not ordering failures

**Verdict.** The gate **fails**. A same-information oracle that may only reorder the rule
planner's legal candidates is **bit-for-bit identical** to the rule planner -- 11 615
compared decision rows, 0 differences, and 0 paired differences in expired or completed
targets across 160 seeds on two independent blocks. It fixes **0 %** of the remaining
expiries against a target of 30 %. No neural network is trained on this evidence.

Read the 0 % for exactly what it is: **this** oracle variant found no improvement. It is a
*feasibility-filtered* planner -- it starts from the rule plan and returns commitments whose
reserved group cannot arrive inside the believed budget -- and it is one candidate
reordering, not an upper bound over every policy that sees the same information. A strictly
better ordering may still exist. What this result rules out is that the first obvious one
reaches it, and it moves the burden of proof to whoever proposes another.

What the attribution says instead is where the remaining loss actually lives, and it is not
in the coordinator:

| failure class (of the planner's remaining expired) | `analysis_legacy` (16) | `analysis_fresh` (1) |
|---|---|---|
| reservation / execution break | **12** | 1 |
| missing or late belief | 2 | 0 |
| candidate ordering error | 0 | 0 |
| task contention / opportunity cost | 0 | 0 |
| capacity conflict | 0 | 0 |
| travel or deadline impossible | 0 | 0 |

Every single expiry is physically avoidable: `infeasible_with_travel` is **0** on both
blocks, and a dispatcher that got a robot on site within 5-40 steps of detection and let it
work uninterrupted would have saved **100 %** of them. The binding constraints are
*execution continuity* and *information*, in that order.

## How the gate was run

**The oracle.** `DeadlineAwareCoalitionCoordinator` (`--coordinator deadline_aware`) sees
exactly what the rule planner sees: the same free robots, the same believed tasks, the same
unit slots, the same exclusivity, and never `CoordinatorInput.tasks`. It may only reorder
the planner's own output, so whatever it fixes is fixed by ordering. It is a
feasibility-filtered planner, not a strict upper bound over all same-information policies:
it searches exactly one neighbourhood of the rule plan, so its 0 % is evidence about that
neighbourhood and about nothing larger.

Its first version was not that. It built coalitions from scratch and demanded a complete
cover from the robots free *right now*, which is strictly more brittle than the planner's
incremental filling -- a task needing three robots with two free got no reservation at all
instead of two -- and on the legacy block it made expiry *worse* (16 to 30). That failure is
recorded rather than discarded, because it is evidence: the "obvious" stronger matcher is a
regression, and the reason is that withholding reservations costs more than imperfect
composition. The corrected oracle starts from the rule plan and only returns commitments
whose reserved group cannot arrive inside the budget its own members believe in, then
re-plans the released robots. On the legacy block that version returns **nothing**.

Two limits it cannot escape, both because of what the belief carries:

* `AgentTaskBelief` keeps the requirement and the deadline but **drops the stage work
  duration** that `DetectionMessage` delivered. Only travel can be subtracted from a
  budget, so neither a rule planner nor a network can reason about "travel + work <=
  deadline".
* It cannot see a detection that has not reached a robot yet.

**The attribution.** Six classes, assigned by an ordered deterministic procedure over the
artefacts each run already wrote: physics first, then a broken commitment, then belief
latency, then capacity versus choice (decided by how saturated the fleet was at the dispatch
instant), and only then an ordering error. A target that was never assigned produces no
progress record and hence no timeline, so it is recovered from the lifecycle log -- dropping
it would flatter the planner by hiding exactly the communication-chain failure.

**The hindsight ceiling.** `expired_attribution` is diagnostic-only; no online module imports
it and a test pins that. It reads ground truth, which is why it estimates a ceiling rather
than a policy.

**The data.** `analysis_legacy` (seeds 20000-20099) was reported as a held-out test set in
the previous phase and has been read while designing this gate, so it is demoted -- calling
it a test set now would be the leak it is. `analysis_fresh` (21000-21059) was recorded after
the gate was specified, with the same frozen PILOC and CTDE checkpoints and the same
protocol. Splits are disjoint; the registry, the digests and the protocol are in
`manifest.json`.

## Why the two blocks disagree so much, and why it does not change the verdict

The blocks are the same size per seed (~5.9 detections) and the same protocol, yet the
expiry rate differs by an order of magnitude: 16 / 588 (2.7 %) on the legacy block against
1 / 355 (0.28 %) on the fresh one. Under a common rate that difference is not plausible
chance, so the two blocks are **not exchangeable** and must not be pooled into one number.
The likely cause is that the streams were recorded in two different sessions with two
different code states behind the search pass, which makes block difficulty a property of
the recording and not of the seeds.

This does not soften the verdict, because the gate's decisive measurement is the
*within-block, paired* comparison, and on both blocks it is exactly zero. What it does
change is precision: with one event on the fresh block, the fixable share there is estimated
from almost nothing. The legacy block carries the evidence, and it is the block where the
most expiries live.

## Recommendation: stop, and fix these three things first

Ordered by the size of the loss they address, with the evidence that points at each.

1. **Make a held commitment binding on the executor (scheduling).** 12 of 14 attributed
   expiries are execution breaks: the plan held a commitment for the target and the
   allocator took another action (11), or work already done was thrown away (2). The
   reservation binds at plan time but the robot is re-decided on every allocation cycle,
   and nothing stops the residual from flipping it off its task. Concretely:
   `Task03Adapter.allocate` re-decides any agent with `needs_decision`; an agent holding a
   live commitment for a task still in its belief should not be re-decided unless it is in
   `advice.replan_agents` or its target has become hard-masked. That is a small, testable
   change, and it attacks the dominant class directly rather than the class with zero
   members.
2. **Carry the stage work duration into the belief (observation / contract).**
   `DetectionMessage` delivers `stages` with their durations; `AgentTaskBelief` drops them.
   Until the belief carries "how long the remaining work takes", no planner -- rule or
   learned -- can check "travel + work <= deadline", and every budget check is optimistic by
   exactly the work time. This is the one missing feature for time-aware coordination, and
   it is information that is already on the wire.
3. **Treat capacity as a fleet question, not a coordination one.** The one
   never-assigned target on the legacy block was known (delivery probability is 1.0, delay
   0) and still not served: no robot was free and able. If the execution-break class
   shrinks after (1) and a residual remains, the lever is fleet composition or staged
   dispatch, not a better ordering of the same six robots.

**Re-running the gate.** The instrument is reusable as-is. After (1) and (2), record a new
fresh block, re-run `coalplan0.25` against `daw0.25`, and read the same table. If a real
ordering gap appears, `learnability.py verdict` will say so with the same definition of
"fixable" -- and only then is there something for a `LearnedCoalitionCoordinator` to learn.
If the ordering classes stay at zero, there is still nothing.

## What a learned coordinator would have been trained on

For completeness, since the phase was designed to reach this decision: the candidate set is
already ordered as well as the visible features allow. A network predicting a bounded
residual over `(robot, task, slot)` triples, initialised at zero, would be trained to
reproduce the rule planner exactly and would then be graded on differences that the oracle
shows are worth nothing. The gate exists to stop that spend, and it did.

## Reproduce

```bash
python runs/learnability_20260917/learnability.py manifest
python runs/learnability_20260917/learnability.py record --split analysis_fresh
python runs/learnability_20260917/learnability.py run --split analysis_fresh
python runs/learnability_20260917/learnability.py run --split analysis_legacy
python runs/learnability_20260917/learnability.py verdict
```

`learnability_verdict.json` (small) holds the numbers above; `manifest.json` holds the
splits, digests and protocol. Raw streams, lifecycle logs and decision traces are runtime
artefacts and are not versioned.
