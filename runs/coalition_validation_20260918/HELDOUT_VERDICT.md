# Held-out verdict: 400 unseen seeds, both arms, statistics fixed beforehand

Source freeze `1f4a598` (`integrated_sar` tree byte-identical to the earlier gate
freeze `304fc60`/`e96f34c`; verified with `git diff <freeze> HEAD -- integrated_sar`,
empty). 400 seeds drawn from 30000 ascending, disjoint from every historical manifest
(asserted in `HELDOUT_SEEDS.json`), streams recorded once per seed with live PILOC and
replayed identically by both arms. Preflight clean on 800 runs: one commit, correct
policy per arm, stream digests matching the recorded index.

**Identity precondition, measured before any held-out run**: with both switches off, the
configuration reproduces the committed 868d68a-era trajectory on the five development
seeds — 5645 decision rows, field by field, identical.

## Primary metric (pre-registered)

Paired per-seed on-time completion rate, candidate minus baseline:

| | value |
|---|---|
| seeds in test | 400 |
| mean difference | **+0.008333** |
| bootstrap 95 % CI (B=10 000, RNG 20260918) | **[+0.004167, +0.013750]** |
| t-interval (reference only) | [+0.003496, +0.013171] |
| claim | **significant improvement** (CI lower bound > 0) |

Zero-detection seeds: **0** — the pre-declared exclusion rule was never exercised, and no
seed was dropped. Denominators identical across arms in all 400 episodes (same stream).

## Count closure

| arm | detections | on time | expired | active/unresolved | episodes |
|---|---|---|---|---|---|
| baseline | 2355 | 2335 | 20 | 0 | 400 |
| candidate | 2355 | **2355** | **0** | 0 | 400 |

Every one of the 20 baseline expiries became an on-time completion; nothing was lost.
Rescued targets span 13+ distinct seeds; **candidate new failures: 0** (per-target list
is empty, not just net-neutral).

## Ceiling effect, stated plainly

The baseline already completes **99.15 %** of detected targets on time on this block, so
the entire available headroom is 20 targets. The percentage-point effect (+0.83 pp) is
therefore small by construction, and the honest reading is: *in regimes where the binding
planner still expires something, this mechanism removes it; in regimes where it does not,
the mechanism is inert.* No extra seeds were drawn, per the pre-fixed rule.

## Secondary metrics (paired, same seeds)

| metric | baseline | candidate |
|---|---|---|
| expired targets | 20 | 0 (paired mean −0.05, CI [−0.0825, −0.025]) |
| completion latency, detection → finish (mean / median) | 24.85 / 22.0 | **17.61 / 17.0** |
| zero-marginal-contribution working ticks | 105 055 | **89 553** (−14.8 %) |
| total working ticks | 135 501 | 108 209 (−20.2 %) |
| coverage resets / discarded work | 2 / 6.50 | **1 / 0.93** |
| release cycles / replan attempts | 0 / 11 | 5106 / 5119 |
| accepted reassignments / declines | 6 / 0 | 2056 / 3058 |
| own-task re-commands (work-clock resets) | 0 | **0** |
| unjustified overrides | 0 | 0 |
| rejected assignments | 11 (6 seeds) | 13 (8 seeds) |
| longest no-progress idle streak | 0.0 | 0.0 |

## Correctness acceptance

| check | result |
|---|---|
| no coverage loss from a joint release | **pass** — 0 regressed targets in 400 episodes |
| no illegal/rejected command beyond baseline | **2 extra refusals, attributed** (below) |
| no work-clock reset from duplicate commands | **pass** — 0 own-task re-commands |
| no truth leakage | **pass** — policies read beliefs and snapshots only; registry never consulted on the release path |
| no demonstrated scheduling deadlock | **pass** — 0 idle streaks; declines resolve through the armed working timeout, not silently |

**The two candidate-only rejections** (seeds 30215 at t=572, 30388 at t=23; both
`stale_task_revision`): a released robot was sent at a task whose registry revision had
already moved on while that robot's own belief still showed it open. The assignment gate
refused the command — the release cannot make a stale command executable, which is
invariant 6 holding rather than failing. The baseline shows the same *class* of event
(11 refusals on 6 seeds) without the release, so the mechanism adds two instances of an
existing, correctly-refused class and no new class of harm: no target was lost and no
illegal command executed.

## What would make this stronger

A block that no decision in this line has touched is what this is; a second one would
separate "validated on 400 unseen seeds" from "validated on 400 seeds drawn after the
mechanism existed". The ceiling here also means a deadline-tight enrichment (higher
saturation or tighter `tightness`) would measure the effect where it lives, instead of at
the edge of a saturated block.

Reproduce: `heldout_seeds.py` → `heldout_record.py` → `heldout_replay.py --identity` →
`heldout_replay.py` → `heldout_stats.py`; numbers in `HELDOUT_RESULTS.json`.
