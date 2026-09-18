# Stress protocol — pre-registered before any stress episode is replayed

Frozen candidate: `release/coalition-controller-validation` at `fb1c85b` plus the
scenario-band plumbing committed with this document. This file, `stress_seeds.py` and
`STRESS_SEEDS.json` are committed before `stress_record.py` or `stress_replay.py` is run
against any of these 180 seeds.

## 1. What is being stressed, and with which existing parameter

The task environment already samples every deadline as
`estimated × uniform(low, high)`, floored at `0.95 × estimated`
(`task_allocation/old_task_env.py:201-216`), and already exposes the band as its
`deadline_tightness` argument. The integration layer never passed it, so this round plumbs
the existing argument through `RunConfiguration` / `--deadline-tightness` /
`build_task_env`. **No generation logic, sampler, floor or formula was changed**, and the
default remains the environment's own `(0.5, 2.0)`, which a test pins.
Task count is already a first-class configuration field (`num_goals`, `--num-goals`).

| Condition | `deadline_tightness` | `num_goals` | Why this value |
|---|---|---|---|
| A tighter deadline | `(0.4, 1.2)` | 6 | The band's upper bound 2.0 → 1.2 removes most slack; the environment's own 0.95 floor keeps every task inside the reachable range, so the condition is tight, not infeasible. The lower bound moves 0.5 → 0.4 only because the floor already dominates below 0.95. |
| B higher load | `(0.5, 2.0)` | 9 | 50 % more tasks for the same 6-robot fleet. The fleet is a fixed production parameter and does not move. |
| C both | `(0.4, 1.2)` | 9 | The combination, at the same two settings. |

Fixed for every arm of every condition, as in the earlier blocks: `steps=1000`, `dt=1.0`,
`rescue_species=3`, `rescue_per_species=2` (6 robots), `grid_size=60`,
`obstacle_density=0.15`, `detection_radius=5`, `search_agents=2`, PILOC
`runs/cur_v2rep05_s2/best/shared_mappo_model_21_3100.pth`, CTDE
`runs/joint_eval_20260916/ctde_final_7_1.pth`, `residual_scale=0.25`,
`coordination_interval=allocation_interval=5`, `command_ttl=10`,
`comm_delivery_probability=1.0`, `comm_delay_ticks=0`, `actor_kv_mode=tokens`.

Arms: **baseline** `coalition_planner` + `sticky_commitments`, both switches off (the
868d68a binding behaviour); **candidate** `coalition_redundant_release` (release +
own-command suppression on). The candidate is the coverage-preserving verdict: these seeds
test v2, not the verdict that produced the 400-seed block, and the two are never pooled.

## 2. Seeds

`STRESS_SEEDS.json`: 60 per condition from three disjoint bands (32000, 33000, 34000),
ascending, excluding every seed in the three committed data splits, the 20260916 test
manifest, the learnability manifest, the ablation set, the 400-seed block and the 60-seed
live-system block (exclusion union 950 at draw time). Verified: no overlap with spent
seeds, no seed shared between conditions.

## 3. Streams

One live PILOC pass per scenario seed under that condition's own configuration with the
coordinator off, writing the ordered `DetectionEvent` stream; both arms then replay the same
file for the same seed. Stream digests are recorded per condition and checked in preflight.

## 4. Statistics, declared now

* **Primary, per condition**: paired difference in per-seed on-time completion rate
  (targets completed by their deadline ÷ targets detected in that episode), seed-level
  paired bootstrap, 10 000 resamples, percentile intervals, RNG fixed at 20260918.
* **Multiplicity**: three conditions are three tests. Each is reported with its nominal
  95 % interval *and* a Bonferroni-adjusted 98.33 % interval. A claim of improvement needs
  the **adjusted** lower bound above zero; otherwise the wording is "insufficient evidence
  to judge the effect under this condition".
* **Secondary, per condition**: expired count, completion latency from detection to
  finish, zero-marginal-contribution working agent-ticks, coverage resets, discarded work,
  reassignment attempts/acceptances/latency, duplicate own-task commands, rejected commands
  by reason, longest no-progress streak, travel distance, and the coordinator decision
  time distribution.
* **Zero-detection rule** (as before): a seed where neither arm detects any target is
  excluded from the rate and listed by name; run-count closure is reported so the excluded
  and retained seeds add back to 60.
* **Ceiling effect**: if both arms sit near 100 % in a condition, that is reported as a
  ceiling and no seeds are added to make the condition harder.

## 5. Correctness acceptance, judged separately from performance

Per condition, the candidate must show: no coverage reset or discarded work it does not
also show in baseline; zero duplicate own-task commands; zero commands executed against a
stale revision or an expired command; zero unjustified commitment overrides; and no seed
with a continuous no-progress streak at or above 20 ticks. Any new candidate-only target
failure is listed by name even if the condition's aggregate improves.

## 6. Stop rules

All 60 seeds per condition run. No early stopping, no seed substitution, no re-draw, and
no difficulty adjustment after seeing a candidate result. A technical failure (worker crash,
unreadable artefact) may be re-run under the same signature; a policy failure is kept.
Source is frozen while a batch runs: any edit ends the affected batch and starts a new
signature directory.

## 7. What this block is and is not

A robustness analysis under three named stressors. Its 180 seeds are new, but they were
drawn knowing the candidate's mechanism, and the conditions were chosen by this round, so
the block is reported as **stress/robustness evidence, not a second independent
confirmation of the original claim**, and it is never averaged with the 400-seed block.
