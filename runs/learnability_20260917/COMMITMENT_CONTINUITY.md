# Commitment continuity: the fix works where it binds, one criterion is still open

**Status.** The continuity fix is implemented, tested (402 tests, 0 skipped failures) and
measured on both blocks. It clears the honour, legality and deadlock criteria and does not
regress completion. It does **not** yet clear "zero unjustified overrides": the count is 82
on the legacy block and an unexplained 134 on the probe block, and that has to be
understood before the criterion is claimed.

## What was built

`--sticky-commitments` binds a coalition commitment on both sides, off by default:

* **planner** (`renew_commitments`): the plan is rebuilt every cycle from free robots only,
  so a robot used to stop being reserved the moment it was dispatched. Renewal re-issues the
  commitment of a robot already executing the task it was sent to, adding nothing new -- no
  new claim, no robot that is not already under orders, no commitment that robot's own belief
  has not kept open.
* **allocator** (`binding_commitments`): a commitment is tested against six conditions before
  it acts -- explicit replan, the robot's own belief still active, not expired, the
  capability slot still needed, on the menu and hard-mask legal, and not mid-commitment
  elsewhere. One that fails is released with the named reason and the allocator re-decides.
* the trace records presence, held, release reason, task, slot, believed status and revision,
  mask legality, whether the residual would have gone elsewhere (`commitment_overridden`),
  what it would have chosen instead (`unforced_index`), and whether a command was suppressed.

The hard mask is still the last word: a masked committed task releases as `hard_mask`
instead of being forced.

## Measured

| | `analysis_legacy` (100 seeds) | probe (`analysis_fresh`, 60 seeds) |
|---|---|---|
| expired, advisory -> binding | **16 -> 11** | 1 -> 1 |
| paired expired | −0.05 ± 0.0703 | 0.0 |
| paired completion | **+0.0084 ± 0.0118** (not worse) | 0.0 |
| decisions with a commitment | 1091 | 704 |
| commitment honour | **99.45 %** | 99.72 % |
| overrides prevented | **97** | 0 |
| unjustified overrides | **82 — open** | **134 — unexplained** |
| commands suppressed | 0 | 0 |
| releases | `hard_mask` 6 | `hard_mask` 2 |
| longest idle hold | 13 decisions | 15 |
| deadlock | none | none |
| rejected assignments | 0 | 0 |

Legacy attribution now closes: 16 of 16 expired targets carry exactly one primary cause
(13 execution break, 3 missing-or-late belief) and 10 secondary causes are listed
separately (8 task contention, 2 belief), so no target is double counted and none is lost.

## Two things this phase got wrong, and what they cost

1. **The first oracle was mis-described.** `DeadlineAwareCoalitionCoordinator` is a
   *feasibility-filtered* planner -- it starts from the rule plan and returns commitments
   that cannot arrive in time -- and that is one candidate reordering, not an upper bound
   over every policy seeing the same information. The verdict now says the 0 % means only
   that this variant found nothing, and the burden of proof moves to the next proposal.
2. **Two batches ran against stale code.** The first sticky batch and the first oracle
   batch were launched while the working tree was still being edited, so they produced
   advisory behaviour under a binding label. Both were re-run; the numbers above are from
   the re-runs, and every metric in this file is recomputed from the final artefacts.

## The open criterion

`override_unjustified` counts a decision where a commitment is held and the commanded task
differs from it. With the force in place it should be zero, and on spot-checked seeds it is.
Aggregated it is not, and the probe block's commitment counts are identical to the advisory
arm's, which suggests part of that block was still measured without the flag. Resolving it
means re-running the probe block once more with the flag confirmed in each run's own trace
meta (`sticky_commitments: true`) before aggregating, and then explaining any remainder row
by row.

## Next ablation (proposed, not implemented)

Stage work duration is still missing from `AgentTaskBelief` -- `DetectionMessage` delivers
it and the belief drops it -- so "travel + work <= deadline" remains uncheckable. A separate
ablation should add the duration to the belief only, re-run this same comparison, and
measure whether the `travel_or_deadline_impossible` and execution-break classes shrink.
That is the last information gap the attribution points at.
