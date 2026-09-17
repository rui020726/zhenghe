# Two interpretable baselines before any learned coordinator

**Verdict.** The depot preference is a calibration problem: a single scalar
(`--ctde-task-logit-offset`, default `0.0`, no retraining, no checkpoint touched) lifts
held-out completion from 0.798 to 0.892 and leaves the unset behaviour bit-for-bit
identical. It does *not* reach the heuristic residual (0.911, paired
`cal0.25 - rs0.25 = -0.019 +- 0.027`), so the residual earns its keep, but only as a
shaping term. The coalition-aware coordinator adds nothing measurable
(`coal0.25 - rs0.25 = +0.0051 +- 0.0073`, three targets of 588) and is inert on the
development set; the trace counterfactual says why: a candidate's own belief almost never
contains a capability gap it could fill, so a per-pair bias cannot convene the group that
the remaining 50 expiries are waiting for.

## What was built

| Stage | Artefact | Default behaviour |
|---|---|---|
| A | `--ctde-task-logit-offset`: a constant added to *legal task* logits only, before the residual; written to `RunConfiguration`, the run summary and the decision trace | `0.0` reproduces the old path exactly (checked in `test_ctde_task_logit_offset.py`, including a strict `load_state_dict`) |
| B | `CoalitionAwareCoordinator`: belief-only, `0.65 * fill_ratio + 0.35 * distance + 0.40 * busy`, per-pair, bounded to `[-1, 1]`, delivered through `CoordinationAdvice.pair_bias` so `residual_scale` and the hard mask still bound the decision | `--coordinator heuristic` (the existing `HeuristicCoordinator`, untouched) |

## Data protocol

* **Development** — 30 seeds (`10000-10029`) from `runs/joint_eval_20260916/streams`.
  These were read repeatedly while the offset and the coalition term were designed, so
  they select arms and nothing else.
* **Held-out** — 100 fresh seeds (`20000-20099`) recorded by `heldout.py record` *after*
  the arms were frozen. The recording pass writes no arm log, so all six arms are produced
  by the identical replay code path.
* **Replay audit** — `heldout.py audit` reduces every run to its ordered
  `(detected_at, target_id, row, col)` fingerprint and compares it with the stream:
  **600 (run, seed) pairs verified event-by-event, 0 mismatches, 588 detections per arm**.
  No paired difference below can come from a different search history.
* Reported differences are **paired per seed** with a 95 % CI; the tables' per-arm means
  are shown for orientation only.

## Stage A — scalar calibration

Development, 30 seeds, 171 targets:

| arm | completed | expired | never assigned | completion | util. | depot with legal task | group stall (median) |
|---|---|---|---|---|---|---|---|
| no coordinator | 136 | 35 | 18 | 0.796 | 0.075 | 0.949 | 0 |
| `cal0.05` | 155 | 16 | 6 | 0.909 | 0.079 | 0.879 | 1 |
| `cal0.10` | 156 | 15 | 1 | 0.914 | 0.084 | 0.625 | 6 |
| **`cal0.25`** | **157** | **14** | **0** | **0.920** | 0.085 | **0.000** | 25 |
| `rs0.25` | 152 | 19 | 0 | 0.891 | 0.094 | 0.000 | 67 |
| `coal0.25`, `coal+cal0.10`, `coal+cal0.25` | 152 | 19 | 0 | 0.891 | 0.094 | 0.000 | 67 |

Held-out, 100 seeds, 588 targets:

| arm | completed | expired | never assigned | completion | util. | depot with legal task | group stall (median) |
|---|---|---|---|---|---|---|---|
| no coordinator | 469 | 119 | 50 | 0.798 | 0.084 | 0.930 | 0 |
| `cal0.10` | 520 | 68 | 5 | 0.884 | 0.090 | 0.676 | 61.5 |
| `cal0.25` | 524 | 64 | 0 | 0.892 | 0.095 | 0.000 | 74 |
| `rs0.25` | 535 | 53 | 0 | 0.911 | 0.088 | 0.084 | 67 |
| **`coal0.25`** | **538** | **50** | **0** | **0.916** | 0.087 | 0.083 | 65 |
| `coal+cal0.10` | 538 | 50 | 0 | 0.916 | 0.088 | 0.000 | 65 |

Paired, held-out (`mean +- 1.96 s.e.` over 100 seeds; completion / mean completion time):

| comparison | completion | mean time | expired |
|---|---|---|---|
| `cal0.25 - rs0.25` | **-0.019 +- 0.027** | -0.5 +- 2.5 | +0.11 +- 0.16 |
| `coal0.25 - rs0.25` | **+0.005 +- 0.007** | -0.3 +- 0.9 | -0.03 +- 0.04 |
| `coal+cal0.10 - coal0.25` | 0.000 +- 0.000 | -0.2 +- 0.8 | 0.00 +- 0.00 |
| `cal0.25 - nocoord` | +0.095 +- 0.032 | -4.3 +- 2.7 | -0.55 +- 0.19 |
| `rs0.25 - nocoord` | +0.113 +- 0.041 | -3.8 +- 3.4 | -0.66 +- 0.24 |
| `coal0.25 - nocoord` | +0.118 +- 0.041 | -4.1 +- 3.3 | -0.69 +- 0.24 |

Reading: the offset removes the failure mode completely (0 depot decisions with a legal
task at `0.25`, no unassigned expiry), and the ranking it produces is inside the noise of
the heuristic residual — the opposite sign of the development set, which is what "inside
the noise" looks like. `0.10` is *not* enough: it still sends 67.6 % of legal-task
decisions home and leaves 5 targets never assigned, so the held-out set confirms the
30-seed development reading over the 5-seed one. No CTDE retraining was needed, and the
last-layer-only path the task allowed for was not entered.

## Stage B — coalition-aware coordinator

Same streams, same seeds, lifecycle comparison above; below is the trace-level
counterfactual (`dev_sweep.py compare`, re-derived from `ctde + offset + bias` under the
hard mask, with `m = 1` returning the recorded action as a self-check).

| | dev (30 seeds) | held-out (100 seeds) |
|---|---|---|
| paired decisions | 35 838 (unpaired 0/0) | 119 903 (unpaired 223/348) |
| different action chosen | 0 | 18 (all with a legal task) |
| legal-task decisions | 831 / 831 | 3 197 / 3 191 |
| residual vectors differ | 20 rows (legal 20, idle 20, busy 0) | 57 rows (legal 50, idle 50, busy 0) |
| `abs(delta)` on legal actions | median 0.081, max 0.163 | median 0.081, max 0.247 |
| amplify only the coalition delta: first flip | m=2 (3 decisions) | m=2 (5 decisions) |
| … at m=4 / saturating | 8 / **10** of 35 838 | 10 / **17** of 119 903 |

Reading: the two policies are numerically identical on 99.94 % of decisions because the
fill ratio is 1.0 nearly everywhere — in a candidate's own belief the remaining
requirement is already covered by its own ability, so the "reward the missing capability"
term has nothing to say. The 20/57 rows where the residuals do differ are idle robots
whose winner is already decided by a wide margin; amplifying the coalition-specific part
by 2x flips 3-5 decisions, by 16x it saturates at 10-17. The interface moves a single
robot's preference between actions it can already see; it cannot make a robot see a
partner it needs.

Group formation, per the truth fields the trace samples at decision instants (identical
across the strong arms, so no differential effect either way): a target is observed with
its current stage requirement covered for 8 % of dispatched targets, median 12-17 steps
after first dispatch (never observed for 541 of 588); adding en-route robots raises that
to ~20 % with a median gap of 9-10 steps. `never_covered` is dominated by the sampling
grid (the flag is only evaluated on decision rows), so it is usable as a between-arm
comparison, not as an absolute time-to-assemble.

## Decision gate

1. *Does scalar calibration match the heuristic coordinator?* Statistically the two are
   indistinguishable (`-0.019 +- 0.027`), but calibration is not ahead either, and the
   residual's state-dependence still buys ~4 targets per 100 seeds. So the depot problem
   was calibration, and the coordinator was never needed for it.
2. *Does the coalition baseline reduce the 19 expiries / 67-step stall?* No: on the
   development set it is bit-for-bit `rs0.25`; held-out it is +3 targets of 588 and 2
   steps of stall. It is not the strong baseline the gate was looking for.
3. *Proceed to a learned residual coordinator?* Avoidable expiry remains (held-out: 50
   expired under `coal0.25`, none unassigned, none infeasible at discovery, 65-step median
   group stall). But this experiment says the *per-pair bias* is not what decides those
   dispatches, so a learned coordinator on the same interface would be fighting the same
   saturation the counterfactual measures — the lever for group composition has to come
   from somewhere the interface can reach (who arrives with whom), not from a larger
   scalar on the existing pairs.

## Reproduce

```bash
python runs/coordinator_baselines_20260917/dev_sweep.py run --mode full --cal 0.05 0.1 0.25 --coalition
python runs/coordinator_baselines_20260917/dev_sweep.py report --mode full
python runs/coordinator_baselines_20260917/dev_sweep.py compare                      # dev traces
python runs/coordinator_baselines_20260917/heldout.py record --shard 0/3             # fresh streams
python runs/coordinator_baselines_20260917/heldout.py run --shard 0/3
python runs/coordinator_baselines_20260917/heldout.py run --only nocoord+cal0.25
python runs/coordinator_baselines_20260917/heldout.py audit
python runs/coordinator_baselines_20260917/heldout.py report
python runs/coordinator_baselines_20260917/dev_sweep.py compare \
    --dir runs/coordinator_baselines_20260917/heldout
```

`verdict.json` holds these numbers as parsed aggregates (it is what this file quotes).
`dev_coalition_compare.json` is the development-set trace evidence; the held-out
trace compare is condensed into `verdict.json` because the traces themselves are
runtime artefacts and are not versioned.

## Open risks

* The coalition arm's 3-target held-out edge over `rs0.25` is *detectable* only because
  the two agree on most seeds; it is not a better policy and must not be read as one.
* 223/348 held-out decisions could not be paired between `rs0.25` and `coal0.25`, since a
  different assignment changes which robot decides next; the trace evidence covers
  119 903 of ~120 250 decisions.
* Group-formation metrics sample at decision instants (`never_covered` above), so
  "time to a complete-capability group" is a lower-bound comparison, not a measurement.
* Both stage-B arms were measured at `residual_scale = 0.25`. The counterfactual sweep
  says what a larger scale does to the *winner* of an already-made decision; it does not
  say what a larger scale would do to the search trajectory that produced the menus.
* The 100 held-out streams were recorded with the same PILOC checkpoint as the
  development ones, so "fresh" means new scenes and new search paths, not a new activation
  policy; a change in the search policy would require recording again.