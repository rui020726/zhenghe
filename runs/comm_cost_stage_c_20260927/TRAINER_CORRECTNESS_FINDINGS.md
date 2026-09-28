# ROUND23 Stage C trainer correctness findings

## Scope and status

Repair branch `fix/comm-cost-stage-c-training-contracts` starts from the verified Stage C HEAD `cac8c60051f7cb89379104941113060758c36ab1`. Its first commit is the imported specification `8970ce964ca820e85bce19537f35b98dff08a682`.

The prior 677 launches, rows, tapes, verdict and passed-gate files remain in place. This repair started zero environments, performed zero real training steps, and did not load an old checkpoint into a training update. Synthetic tensors exercised `optimizer.step`; historical tape/checkpoint analysis was read-only. No production-tree source under `integrated_sar`, `task_allocation` or `piloc` changed.

## Findings and repairs

| Finding | Prior behavior | Repair and evidence |
|---|---|---|
| F1: cost sign | Raw airtime is nonnegative, but the trainer negated normalized cost before using `lambda*A_task - A_cost`, rewarding higher cost. | Keep measured cost positive through `batch_samples`, GAE and the cost critic. `test_sc_cost_objective.py` sends SEND `(Y=1,C/B0=.2)` and HOLD `(Y=0,C=0)` through actual PPO. An independent action enumeration and finite difference give `dJ/dtheta=(lambda-.2)*p*(1-p)`: at p=.5, `[-.05,-.025,.20,2.45]` for lambda `[0,.1,1,10]`; captured PPO loss gradients have the opposite sign. |
| F2: cost critic lifecycle | `cost_head` was outside `controller.parameters`, so Adam, zeroing, clipping and optimizer moments omitted it. | Include trunk, action head, task value head and cost head once each. `test_sc_cost_head_lifecycle.py` verifies module object-ID sets, deliberately rejects a missing cost head / duplicate trunk / outsider, gives each module a nonzero synthetic loss, then passes through real `batch_samples`/PPO/optimizer, zeroing, clipping, save and restore. Legacy v1 resume is rejected. |
| F3: GAE next-value | Every bootstrap row was zero, including nonterminal decision transitions. | The trainer shifts frozen next-row task/cost values into nonterminal rows and leaves the true terminal row at zero. The closed-form GAE example and `batch_samples` counterexample are in `test_sc_gae_next_value.py`; old behavior misses 0.3 in the first task advantage. |
| F4: reward timing | Tick rewards were sent to the strictly later decision. | Actions are sampled in `CommRelay.tick` before `env.step`; same-tick and inter-decision events belong to the latest action already sampled. Pre-first-action events are unassigned prefix totals; the tail belongs to the final action. The independent test covers same-tick, between, prefix and tail events and checks the checker’s interval-based recomputation. |

## Historical read-only results

`OPTIMIZER_PARAMETER_AUDIT.json` records the expected 10 tensors / 5700 values. Each of the three existing final checkpoints contains an 8-parameter optimizer with 8 state entries; its 2-tensor cost head is byte-equal to a reconstruction from that seed’s initialization. No initial or mid-checkpoint files were present. Old checkpoint files were not modified.

`CORRECTED_TAPE_RECOMPUTE.json` applies one deterministic start/middle/end tape selection (episodes 0, 64 and 127 for seeds 11, 22 and 33). It preserves source row/tape hashes. Old per-tape advantages were not logged; any old values shown are explicitly code-semantic reconstructions, not claimed recorded metrics. The repaired calculation is not a counterfactual policy result.

The former `STAGE_C_VERDICT.md`, `FIRST4_GATE_s11.json`, rows, tapes and checkpoints remain unchanged. The separate `STAGE_C_VERDICT_ERRATUM.json` records that execution completed, while valid learning of the registered N4 objective is **not established because of implementation defects**. Existing joint task/byte comparisons remain historical observations and are not relabeled as task-only improvements.

## Version boundary

`TRAINING_VERSION_MANIFEST.json` identifies trainer/loss/state, GAE, optimizer and reward-timing schemas. Correctness-v2 checkpoints use a separate output directory and require every schema field on restore. A v1 checkpoint cannot silently resume; any future authorized training must start from seed initialization. The repair itself does not authorize or start that training.
