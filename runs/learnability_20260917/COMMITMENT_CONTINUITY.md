# Commitment continuity: advisory plan versus binding plan

Arms share the same recorded detection stream per seed; the only difference is whether a standing coalition commitment binds the allocator. Every artefact carries a run signature, resume is fail-closed, and the preflight over the whole batch had to pass before this page was written.

## analysis_legacy

- seeds compared: 100
- expired: advisory 16 -> binding 9; paired -0.0700 (95% CI 0.0750)
- completion: paired 0.0118 (95% CI 0.0126)
- rejected assignments 0, discarded work mean on completed targets 0.0021 (on expired 0), first-work latency median 2.0000, deadlock False
- decisions under a commitment 1024, held 1023 (share 0.9990)
- counterfactual overrides blocked 151; executed overrides without a release reason **0**
- release reasons: `{'hard_mask': 1}`
- longest idle hold 13 decision cycles (deadlock threshold 20)

### Expiry attribution

- advisory arm expired 16: `{'candidate_ordering_error': 0, 'task_contention_error': 0, 'missing_or_late_belief': 3, 'travel_or_deadline_impossible': 0, 'reservation_or_execution_break': 13, 'capacity_conflict': 0}`
- binding arm expired 9: `{'candidate_ordering_error': 6, 'task_contention_error': 0, 'missing_or_late_belief': 3, 'travel_or_deadline_impossible': 0, 'reservation_or_execution_break': 0, 'capacity_conflict': 0}`
- survivors (7): expired under the advisory arm, no longer expired under the binding arm
  - seed 20002 piloc-e1-target-1: primary `reservation_or_execution_break`
  - seed 20002 piloc-e1-target-5: primary `reservation_or_execution_break`, secondary `['missing_or_late_belief']`
  - seed 20019 piloc-e1-target-0: primary `reservation_or_execution_break`, secondary `['task_contention_error']`
  - seed 20019 piloc-e1-target-3: primary `reservation_or_execution_break`
  - seed 20019 piloc-e1-target-5: primary `reservation_or_execution_break`, secondary `['task_contention_error']`
  - seed 20070 piloc-e1-target-4: primary `reservation_or_execution_break`
  - seed 20074 piloc-e1-target-2: primary `reservation_or_execution_break`

## analysis_fresh

- seeds compared: 60
- expired: advisory 1 -> binding 0; paired -0.0167 (95% CI 0.0327)
- completion: paired 0.0028 (95% CI 0.0054)
- rejected assignments 0, discarded work mean on completed targets 0 (on expired -), first-work latency median 2.0000, deadlock False
- decisions under a commitment 606, held 604 (share 0.9967)
- counterfactual overrides blocked 98; executed overrides without a release reason **0**
- release reasons: `{'hard_mask': 2}`
- longest idle hold 9 decision cycles (deadlock threshold 20)

### Expiry attribution

- advisory arm expired 1: `{'candidate_ordering_error': 0, 'task_contention_error': 0, 'missing_or_late_belief': 0, 'travel_or_deadline_impossible': 0, 'reservation_or_execution_break': 1, 'capacity_conflict': 0}`
- binding arm expired 0: `{'candidate_ordering_error': 0, 'task_contention_error': 0, 'missing_or_late_belief': 0, 'travel_or_deadline_impossible': 0, 'reservation_or_execution_break': 0, 'capacity_conflict': 0}`
- survivors (1): expired under the advisory arm, no longer expired under the binding arm
  - seed 21021 piloc-e1-target-4: primary `reservation_or_execution_break`

## Split diagnostics

### analysis_legacy

- purpose: analysis: previously reported as held-out, now demoted because the learnability gate was designed while reading it
- code commit: `04cc187ac8a987e059a5eef7548b315d2375577b`
- piloc sha256: `d57aed25b3d46dffb2bd269ea87140953828ba7f51a960bb01995ec054dfca33`
- ctde sha256: `c7c256da0fc20451c1104996ce494148f567e4f83acf265d9baae8f0f053500d`
- expired attributed 16 over 100 seeds; classes `{'candidate_ordering_error': 0, 'task_contention_error': 0, 'missing_or_late_belief': 3, 'travel_or_deadline_impossible': 0, 'reservation_or_execution_break': 13, 'capacity_conflict': 0}`
- secondary classes: `{'missing_or_late_belief': 2, 'task_contention_error': 8}`
- delivery budget median 72.1200, assignment latency median 26.0000, detected-at median 4.5000
- fleet: idle at detection median 6.0000, concurrent active tasks 0.0000, detection to first needs_decision 6.0000, earliest agent availability 0.0000, cycle offset 1.0000, wait to next cycle 3.0000
- communication: `{'comm_delay_ticks': 0, 'comm_delivery_probability': 1.0, 'allocation_interval': 5}`

### analysis_fresh

- purpose: analysis: fresh seeds recorded after the gate was specified, used to confirm the attribution and the oracle away from the legacy block
- code commit: `04cc187ac8a987e059a5eef7548b315d2375577b`
- piloc sha256: `d57aed25b3d46dffb2bd269ea87140953828ba7f51a960bb01995ec054dfca33`
- ctde sha256: `c7c256da0fc20451c1104996ce494148f567e4f83acf265d9baae8f0f053500d`
- expired attributed 1 over 60 seeds; classes `{'candidate_ordering_error': 0, 'task_contention_error': 0, 'missing_or_late_belief': 0, 'travel_or_deadline_impossible': 0, 'reservation_or_execution_break': 1, 'capacity_conflict': 0}`
- secondary classes: `{}`
- delivery budget median 60.3200, assignment latency median 4.0000, detected-at median 2.0000
- fleet: idle at detection median 6.0000, concurrent active tasks 0.0000, detection to first needs_decision 5.0000, earliest agent availability 0.0000, cycle offset 1.0000, wait to next cycle 3.0000
- communication: `{'comm_delay_ticks': 0, 'comm_delivery_probability': 1.0, 'allocation_interval': 5}`

Stage duration is deliberately not implemented here: the belief carries no work half of the budget, so that ablation stays a separate, independently seeded step.
