# Assembly scheduler small-scale gate: the prototype is safe, and it changes nothing

**Verdict: the small-scale gate FAILS.** The prototype runs, its synchronisation term is
demonstrably active, and it rescues **0 of the 6** targets — the gate requires 3. Per the
brief's Phase 12 step 7 ("未通过则停止并归因"), the batch evaluation is not started and the
result is recorded here instead.

This is a negative result about a *specific bounded mechanism*, not about the diagnosis.
The counterfactual ceilings in `COALITION_FORMATION_VERDICT.md` are unaffected: they say a
perfect dispatcher could have saved 6/6, and this prototype is not a perfect dispatcher.

## The gate

Baseline `coalition_planner` (binding, sticky commitments) vs candidate
`coalition_assembly_scheduler`, same seeds, same recorded detection streams, 1000 steps.

| seed | arm | completed | expired | resets | lost work | override | rejected |
|---|---|---|---|---|---|---|---|
| 20003 | baseline | 1 | 4 | 0 | 0.0 | 0 | 0 |
| 20003 | candidate | 1 | 4 | 0 | 0.0 | 0 | 0 |
| 20008 | baseline | 5 | 1 | 1 | 1.2 | 0 | 0 |
| 20008 | candidate | 5 | 1 | 1 | 1.2 | 0 | 0 |
| 20052 | baseline | 5 | 1 | 0 | 0.0 | 0 | 0 |
| 20052 | candidate | 5 | 1 | 0 | 0.0 | 0 | 0 |
| 20070 | baseline | 5 | 1 | 0 | 0.0 | 0 | 0 |
| 20070 | candidate | 5 | 1 | 0 | 0.0 | 0 | 0 |
| 20083 | baseline | 5 | 1 | 0 | 0.0 | 0 | 0 |
| 20083 | candidate | 5 | 1 | 0 | 0.0 | 0 | 0 |
| **total** | baseline | **21** | **8** | 1 | 1.2 | 0 | 0 |
| **total** | candidate | **21** | **8** | 1 | 1.2 | 0 | 0 |

| gate | result |
|---|---|
| rescued at least 3 | **FAIL** — net 0 |
| no new execution break | pass — `executed_override_unjustified` 0 both arms |
| zero invalid commands | pass — 0 rejected assignments |
| zero deadlocks | pass — longest idle streak 0.0 (threshold 20) |
| no legacy regression | pass — completed 21 both arms |

`ASSEMBLY_GATE.json` carries the same numbers plus every per-target final status.

## Attribution: the term fires, and never matters

The prototype is wired correctly and its term is **active**. Comparing the decision traces
field by field for seed 20003:

* **13 of 1085 decision rows have a different pair bias**, each raised by exactly
  `assembly_weight * credit = 0.15 * 0.25 = 0.0375` (observed deltas: 0.0990→0.1365,
  0.1255→0.1630, 0.1534→0.1906, ...).
* **0 rows changed the chosen action** (`post_mask_choice` and `chosen_task` identical
  throughout).

So the mechanism does what it was designed to do -- it promotes a robot that can reach the
task at the same time as its peers -- and the promotion never crosses the decision
threshold. The residual `final_logits = ctde_logits + alpha * coordinator_bias` already
separates the candidates by far more than 0.0375, so a term of that size cannot reorder
them. Raising `assembly_weight` enough to flip a choice would no longer be a bounded tie
break and would be a different mechanism, which is exactly what the design deliberately
avoided.

This is the same wall the duration-relevance gate hit, one level up. There, a duration
term could not reorder the six because the feasibility arithmetic was already decided by
the scenario generator. Here, a synchronisation term cannot reorder them because the
learned allocator's own logits dominate. Both are instances of the same finding: **on these
six the ordering is not where the loss is**. The measured loss is the 4-157 step gap
between detection and the coalition completing, and closing that gap needs a mechanism
that changes *when robots are sent*, not one that re-ranks robots already on the menu.

## What the gate does establish

Four things, none of them about rescue:

1. The prototype is **safe**: zero unjustified overrides, zero rejected commands, zero
   deadlocks, no regression on any seed. The bounded-term design holds.
2. The prototype is **inert by default**: `assembly_scheduling` defaults off and the
   binding trajectory is untouched (`test_assembly_scheduler.DefaultOffIdentityTests`).
3. `integrated_sar/assembly.py` is a **usable planner in its own right**: deterministic,
   belief-only, minimal covering sets, arrival-synchronised. It is exercised by 27 tests
   and available for a mechanism that does act on dispatch timing.
4. The **decision traces are instrumented** well enough to attribute a null result: the
   bias deltas above are read straight off the committed trace.

## Reproduce

```
D:/project_env/python.exe runs/coalition_formation_latency_20260918/assembly_gate.py
```

Writes `ASSEMBLY_GATE.json` and the per-run lifecycle/decision traces under `gate/`.
Those traces are large and deliberately not committed.
