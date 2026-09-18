# Dispatch-trigger audit: the first blocking point, and the release that clears it

**Verdict.** All six remaining expiries share one first blocking condition, measured tick
by tick on instrumentation-only replays: **every able, believing robot was `working` with
`need_decision=False`, and no coordination-layer mechanism could re-decide a working
robot** — the planner plans free robots only (`coordinator.py::_is_free`), the allocator
skips robots outside `needs_decision | replan_agents` (`task03.py:523`), and the only
release from a stage nobody present can complete is the environment's random 50-120-step
working timeout (`old_task_env.py:143-150,822-839`). Belief, communication, capacity,
coverage and commitments are all cleared by the same measurement. A default-off,
belief-only planner fix (`coalition_redundant_release`) that releases *provably redundant*
members rescues **6/6 in closed-loop replay** with no new failures, no overrides, no
rejected commands and no churn.

## 1. Sources

| layer | artefact | status |
|---|---|---|
| source facts | `old_task_env.py`, `coordinator.py`, `task03.py`, `orchestrator.py` | cited inline |
| log observations | committed `rp/tr` traces of `coalplan0.25sticky` | read-only |
| instrumented replays | `probe_s*.jsonl` (this directory) | **identity-verified** |
| analytical estimates | `../coalition_formation_latency_20260918/counterfactuals.json` | labelled, not evidence of effect |
| closed-loop experiment | `CLOSED_LOOP_GATE.json` | measured |

Instrumentation-only is not assumed, it is checked: the probe wraps `advise`, `allocate`
and `step` read-only, and the replay reproduces the committed run **exactly** — identical
decision rows (1085/1085, 1082/1082, 1127/1127, 1166/1166, 1185/1185 on the five seeds)
and identical per-target finals.

## 2. The call chain, and where it first blocks

```
event → belief update → decision eligibility → planner invocation
      → candidate selection → reservation → allocator → accepted command → arrival
```

* **event → belief update: clear.** `belief_spread = 0` on all six (letter in hand at
  detection); stage/revision beliefs track the environment within the comm cadence
  (`ledger.json: believed_stage_first`).
* **planner invocation: clear.** `advise` runs on state change or every 5th step
  (285/1000 ticks in seed 20003); it saw every belief it needed.
* **decision eligibility: THE WALL.** `need_decision` is set False the moment a robot
  starts moving/working (`old_task_env.py:1021,1065`). Working robots appear in no
  allocation cycle: the planner's candidate line is `if usable and self._is_free(agent)`
  (`coordinator.py:629`, `IDLE_STATUSES = {"idle","available"}`), and the allocator's
  loop is `if not agent.needs_decision and agent_id not in advice.replan_agents: continue`
  (`task03.py:523`). The only exits from `working` are: stage-boundary keep/release
  (`:1145-1151`), arrival-verify failure (`:1058-1061`), task finished (`:1132`), or the
  **sampled working timeout U(50,120)** (`:143-150, :822-839`).
* **planner → reservation → allocator → command → arrival: clear once eligibility
  exists.** In the committed baseline, seed 20008 agent 3 became idle at ~t=104 (timeout)
  and in the *same tick* the planner minted reservation `rp0008` and the allocator
  commanded `piloc-e1-target-2` (`raw_choice` masked → `post_mask_choice` = target-2) —
  arrival t=111. Nothing upstream of eligibility was ever the bottleneck.

## 3. The wall, per target (pre-deadline open-gap ticks, from `ledger.json`)

| target | budget | open-gap ticks before its deadline | blocked by capable_all_locked | other | first cover base → candidate |
|---|---|---|---|---|---|
| 20003/target-1 | 70.9 | ~88 | **81** | 2 in transit | 161.0 → 8.0 |
| 20003/target-0 | 64.0 | 76 | **74** | 2 in transit | never → 20.0 |
| 20008/target-2 (stage 2) | 77.3 | 82 | **81** | 1 locked | 10.0* → 8.0 |
| 20052/target-3 | 40.8 | 41 | **40** | 1 no_plan, 1 no_command | 52.0 → 18.0 |
| 20070/target-2 (stage 1) | 72.1 | 66 | **70** | 1 no_plan | 13.0* → 5.0 |
| 20083/target-4 | 58.6 | 65 | **62** | 1 no_plan | 69.0 → 5.0 |

\* stage-0 first cover; the fatal wait was the *next* stage. `post_deadline` ticks are
counted separately and never as "blocked": past the deadline no dispatch can help.

The 14 matched controls show the **same** `capable_all_locked` profile (757 ticks) — the
lock is the system's normal operating mode, and the failures are simply the episodes
whose deadlines did not forgive it. What separates recoverable from fatal is whether the
locked robots were *doing anything*: across the five seeds, robots spent **2150
robot-ticks working with zero marginal contribution** — the requirement minus the other
members' abilities left no unit they could fill (`ledger.json: locked_spans`).

## 4. The two handoff cases, pinned (seed 20008, `piloc-e1-target-2`)

* t=17, stage 0→1: agents 2/3/4/5 released and immediately re-dispatched to target-0.
* t=26, stage 1→2: requirement `10011` (dims 0,3,4). Members [0,1] (abilities `00001`
  each) are **kept** — the keep rule needs only *any* overlap (`:1145-1151`), so both
  dim-4 robots stay although one suffices. The stage gap opens `[1,0,0,1,0]`.
* t=26→105: the gap needs dims 0 and 3, held by agents 4/5 and 2/3 — all four `working`
  target-0, whose own gap (dim 4) is held by the robots parked on target-2. **A
  cross-needs standstill**, invisible to the planner and the allocator, broken only by
  random timeouts at t≈104 (agent 3) and t≈109 (agent 5) — arrivals t=111, deadline
  111.6, work 11.71 → finished 10.4 steps late.
* 20070/target-2 is the same shape (52/54 locked ticks on stage 1, first cover 85).

## 5. The fix: redundant-member release (default off)

One mechanism, in the planner (`coordinator.py`), selected because the evidence places
the wall there — not a re-ranking term (the assembly-scheduler prototype already proved
those cannot cross the logits gap):

* a **working** robot whose own belief says the stage's remaining requirement minus the
  other ordered robots' abilities leaves it **no unit to fill** is treated as plannable
  (`_is_redundant` → `_claim_slot(...) is None` — the exact predicate `_renew` already
  uses to stop renewing such commitments);
* it is added to `_plan`'s candidates; **only if the plan finds a different task for it**
  does it enter `replan_agents`, which makes the existing allocator path re-decide it
  (`task03.py:523`). No alternative found → no replan, no command, nothing moves;
* the allocator suppresses the command when a replan-woke robot re-picks the task it is
  already executing (`suppress_recommand_own_task`, default off): re-issuing one would
  restart its work clock and disarm the working timeout — the churn the first gate run
  measured at 98 re-commands, zero after the suppression.

Constraints: no env/deadline/work change, no fleet/communication change, no per-tick
forced replanning (only redundant robots, only with an alternative), no revocation of a
commitment anyone still needs (the needed member never releases), no preemption of useful
work (zero marginal contribution is the release condition), hard mask untouched, no new
online information (belief requirements × snapshot abilities — both already visible), no
residual-weight change. A believed-dead task does not release; a believed-finished task
keeps the pre-existing replan path.

## 6. Verification

* 15 coordinator/allocator tests (`test_redundant_release.py`): default-off identity,
  release only with an alternative, needed member never released, dead/finished beliefs,
  redirect stability (commitment identity survives), determinism, bias band, no-registry
  participation, and both re-command suppression directions. Suite: **495 passed (2
  skipped by design)**; smoke `python -m integrated_sar` passes.
* Closed loop (`CLOSED_LOOP_GATE.json`, 5 seeds × 1000 steps, identical recorded
  streams, run-signature/preflight): **every gate passes** —

| metric | baseline | candidate |
|---|---|---|
| completed | 21 | **30** |
| expired | 8 | **0** |
| coverage resets | 1 | **0** |
| lost work-steps | 1.2 | **0** |
| executed_override_unjustified | 0 | 0 |
| rejected assignments | 0 | 0 |
| re-commands (own-task churn) | 0 | **0** |
| longest idle streak | 0.0 | 0.0 |

  Per-target rescue, closed loop: 20003/target-1, 20003/target-0 (never covered →
  completed), 20008/target-2, 20052/target-3, 20070/target-2, 20083/target-4 — six of
  six, none of them a progress-1.0-in-the-baseline case except target-0; five moved from
  *finished after deadline* to *finished on time*, target-0 from *never covered*.
  Baseline reproduces the committed 868d68a arm exactly (21/8, same per-target finals;
  probe-run decision rows byte-identical).

## 7. What this does and does not establish

* The closed-loop result is measured on the **five development seeds**. The paired
  expansion over `analysis_legacy` (100) and `analysis_fresh` (60) is reported
  separately; both blocks are development-read and neither is a held-out test set.
* The global effect claimed is limited to these replays: no claim is made about
  scenarios whose locks are *not* redundant (there the fix is a no-op by construction).
* The working-timeout keep/release and the any-overlap keep rule remain environment
  semantics; they are documented, not modified.

## Reproduce

```
D:/project_env/python.exe runs/dispatch_trigger_audit_20260918/probe.py            # replays + verifies identity
D:/project_env/python.exe runs/dispatch_trigger_audit_20260918/ledger.py           # builds ledger.json
D:/project_env/python.exe runs/dispatch_trigger_audit_20260918/closed_loop_gate.py # baseline vs candidate
D:/project_env/python.exe -m unittest discover -s integrated_sar/tests
```

`probe_*.jsonl`, `closedloop/*.jsonl` are raw and deliberately uncommitted; the
committed artefacts are `probe.py`, `ledger.py`, `closed_loop_gate.py`, `ledger.json`,
`CLOSED_LOOP_GATE.json`, and this file.
