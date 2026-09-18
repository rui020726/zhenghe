# Mechanism ablation: what each of the two switches actually contributes

Development-analysis set only: the five failure seeds plus ten deterministically drawn
normal seeds from the already-read legacy block, fixed in `ablation_seeds.json` before
the runs (`seed % 10 == 0`, then `== 1`, failures excluded). 15 seeds x 4 arms = 60
closed-loop replays on the same recorded streams, one frozen source, trace schema /3.
**No performance claim is made from this table** — it exists to attribute the mechanism.

| arm | completed | expired | active | resets | lost work | redundant working ticks | release cycles | replan attempts | accepted | declined | own-task no-ops | re-commands | accept latency | rejected | overrides |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A binding baseline | 81 | 8 | 1 | 1 | 1.2 | 5233 | 0 | 0 | 0 | 0 | 0 | 0 | — | 0 | 0 |
| B suppression only | 81 | 8 | 1 | 1 | 1.2 | 5233 | 0 | 0 | 0 | 0 | 0 | 0 | — | 0 | 0 |
| C release only | **90** | **0** | 0 | 0 | 0 | **3213** | 205 | 206 | 86 | 0 | 0 | **120** | 5.43 | 1 | 0 |
| D release + suppression | **90** | **0** | 0 | 0 | 0 | 3362 | 205 | 206 | 86 | 120 | **120** | **0** | 5.43 | 1 | 0 |

Per-target outcomes against A: C rescues **9** targets, D rescues **9**, and **neither
regresses a single target** that A finished on time.

## What the table says

* **All of the outcome gain belongs to C** — the planner's redundant-member release. B is
  byte-identical to A on every metric, including the decision counts: with nothing ever
  re-decided, the suppression has no branch to take, which is the default-off identity
  argument measured end to end rather than asserted.
* **All of the churn control belongs to the suppression.** C's 206 replan attempts split
  into 86 accepted reassignments and 120 cases where the woken robot re-picked the task it
  was already doing — and C then *issued a command for it*, restarting that robot's work
  clock and disarming its working timeout. D converts exactly those 120 into no-ops with
  zero commands, leaving the outcomes (90/0, nine rescues) untouched. The 120 is not a
  rounding error: 58 % of release attempts decline in C, and each decline was a state
  reset.
* **Zero-marginal-contribution working time falls ~39 %** (5233 → 3213 ticks) — the
  mechanism removes most, not all, of the measured waste; the remainder is redundancy the
  plan has nowhere to send (no alternative task needing that robot's abilities at that
  moment), which is the honest boundary of a replan-only fix.
* **Reassignment latency ≈ 5.4 ticks** from replan to acceptance — one coordination
  interval plus travel resolution, i.e. the robot moves on the next plan cycle that wants
  it.
* C and D differ slightly in redundant ticks (3213 vs 3362) because C's restarted work
  clocks change who is where later; both end at the same completions on this set.

## The one rejection

Seed 20030, one event, reason `stale_task_revision`: a released robot was sent at a task
the registry had already closed (its belief still showed it open, invariant 9 forbids
correcting that from outside), and the assignment gate refused the command. It appears in
C and D identically (the same event) and in neither A nor B. This is the gate working —
the release does not bypass legality — but it is a **candidate-only** event and is
carried into the held-out correctness checklist as something to count, not average away.

Reproduce: `D:/project_env/python.exe runs/coalition_validation_20260918/mechanism_ablation.py`
(resume-aware; aggregates into `ABLATION.json`; raw per-run logs uncommitted).
