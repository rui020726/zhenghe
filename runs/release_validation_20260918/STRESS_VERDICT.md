# Stress verdict — three pre-registered conditions, 180 unseen seeds

Protocol, parameters, seed manifest and stop rules were committed in `fd0dd76` before any
stress episode ran; the batch executed under `3fcef63` (production tree identical to
`fd0dd76`: `git diff fd0dd76 3fcef63 -- integrated_sar` is empty). Evidence:
`STRESS_RESULTS.json`, `STRESS_SAFETY.json`, `stress/B_higher_load/*_s33011.*`.

Candidate: `--coalition_planner + sticky + release + own-command suppression`, with the
coverage-preserving verdict. Baseline: the same planner, both switches off.

**Bottom line.** Under 50 % more tasks the candidate is significantly better even after
multiplicity adjustment. **It is not correct to say there is no correctness breach**: 19 of
1 498 acted-on releases removed a capability the stage really needed, and that stands as an
open coverage-protection failure. The reset and lost-work metrics that appear to exonerate it
are confounded, because the environment leaves a re-tasked robot on the old task's roster.
Under the tightened deadline band both arms are near ceiling, so that block judges nothing
either way. One target in one seed is a new candidate-only failure.

## 1. Run integrity

| | |
|---|---|
| Episodes | 360 of 360 (180 paired scenarios × 2 arms), no missing artefact |
| Preflight | 0 problems: every artefact names commit `3fcef63`, its own seed, the condition's goal count and deadline band, and the digest of the stream it replayed |
| Switches-off identity | the 180 instrumented candidate episodes reproduce the stored outcome exactly (completed, expired, per-target progress) |
| Zero-detection exclusions | none; all 180 scenarios detected at least one target in both arms |
| Count closure | per condition and arm, `on_time + finished_after_deadline + never_finished = detections` exactly |

## 2. Primary result per condition

Paired per-seed on-time completion rate; seed-level bootstrap, 10 000 resamples, RNG
20260918; Bonferroni interval at 1 − 0.05/3 = 98.33 % because three conditions are three
tests.

| Condition | Scenario | Baseline on-time | Candidate on-time | Mean paired Δ | 95 % CI | Bonferroni CI | Seeds better / worse | New failures |
|---|---|---|---|---|---|---|---|---|
| A tighter deadline | band (0.4, 1.2), 6 goals | 341/343 (99.4 %) | 343/343 | +0.0056 | [0.0000, 0.0167] | [0.0000, 0.0222] | 1 / 0 | 0 |
| B higher load | band (0.5, 2.0), 9 goals | 513/526 (97.5 %) | 525/526 (99.8 %) | **+0.0243** | [+0.0039, +0.0502] | **[+0.0019, +0.0579]** | 7 / 1 | **1** |
| C both | band (0.4, 1.2), 9 goals | 519/526 (98.7 %) | 521/526 (99.0 %) | +0.0037 | [0.0000, 0.0111] | [0.0000, 0.0148] | 1 / 0 | 0 |

* **B is the one condition with a claim**: the adjusted lower bound is above zero. Twelve
  late completions and one never-finished target became on-time; 13 rescued targets across
  7 seeds, against 1 new failure. Expired went 12 → 0.
* **A and C are not evidence of no effect, and not evidence of an effect either.** Both
  intervals touch zero, and both arms were already at 98.7–99.4 %: the ceiling, not the
  mechanism, is binding. The tightening did not produce the intended pressure — the
  environment floors every deadline at 0.95 × its own travel-and-work estimate, so the band
  mostly removed slack that the fleet was not using. No seeds were added to compensate.

## 3. Correctness, judged apart from performance

| Check | A | B | C |
|---|---|---|---|
| Coverage resets, mean per seed (baseline → candidate) | 0 → 0 | 0.017 → 0.017 | 0.017 → **0** |
| Discarded work steps, mean per seed | 0 → 0 | 0.039 → 0.042 | 0.136 → **0** |
| Duplicate own-task commands (work-clock resets) | 0 → 0 | 0 → 0 | 0 → 0 |
| Unjustified commitment overrides | 0 → 0 | 0 → 0 | 0 → 0 |
| Commands refused after execution | 2 → 4 | 2 → 4 | 0 → 2 |
| Invalid commands actually executed | 0 | 0 | 0 |
| Longest continuous no-progress streak | 0 | 0 | 0 |
| Reassignment attempts / acceptances per seed | 6.5 / 2.7 | 9.2 / 3.1 | 9.4 / 2.8 |
| Release judgements the plan then declined to act on | 3.8 | 6.1 | 6.6 |

Refusals are the same same-tick completion race audited in
`STALE_REVISION_RECOVERY_AUDIT.md`, still one per affected robot and still recovered one
tick later; none executed.

### Did release ever take a capability the stage really needed?

Instrumented replay of all 180 candidate episodes, recorder returning the verdict
unchanged:

| | A | B | C | total |
|---|---|---|---|---|
| Release judgements | 7 183 | 8 087 | 8 041 | 23 311 |
| …of which the robot was actually re-decided | 389 | 548 | 561 | **1 498** |
| …where removing it would have lowered real stage coverage | 5 | 8 | 6 | **19** |

19 of 1 498 acted-on releases (1.3 %) were belief-lag cases: the robot's believed gross
stage requirement was older than the stage the environment had moved to. **These are counted
as failures of the coverage protection, not as an accepted residue.** The observation that
none of the 18 affected seeds shows a coverage reset or discarded work is *not* an
exoneration: if the environment keeps crediting a robot that has been re-tasked away, then a
release that removes needed capability produces no reset to measure, so the same non-
intersection is what a confounded metric would look like either way. Settling it requires the
member-semantics audit (`MEMBERSHIP_CONSISTENCY_AUDIT.md`), not another read of these plots.
For comparison, the same measurement on the 15 development seeds under the old verdict put
the rate at 33 of 205 (16 %), and that one was systematic: the old test called every sole
provider redundant.

### The one new failure: seed 33011, target-7, condition B

Baseline finished it on time; the candidate left it at progress 0.78 when its deadline
passed. The candidate's stage 2 needs one dim-0 and one dim-4 unit. Timeline: robots 0 and 1
held dim 4 from 217; robot 0 was released by the environment's random working timeout at
271; the coalition then sent robot 4 in at 272 (accepted, with a reservation) and the stage
read fully covered at 274; robot 1 timed out at 276 and the dim-4 unit went uncovered again;
by 369 nobody was left. Across the whole episode, **no decision that woke a robot off
target-7 was a release decision** (`replanned ∧ current_task = target-7` returns zero rows),
and the reset landed on this seed while condition B's only baseline reset landed on a
different seed (33016). What changed hands is the fleet's distribution under nine tasks, and
the timeouts that follow from it — the failure is real and is counted against the
candidate, but it is not the joint release removing a needed capability.

## 4. Cost

| | A | B | C |
|---|---|---|---|
| Coordinator decision time p95, ms (baseline → candidate) | 0.23 → 0.41 | 0.33 → 0.57 | 0.38 → 0.65 |
| p99, ms | 0.38 → 0.60 | 0.73 → 1.09 | 1.11 → 1.42 |
| Rescue travel distance, mean per seed | 5.83 → 5.80 | 7.17 → **7.33** | 7.01 → 7.10 |

The decision path stays sub-millisecond at the 99th percentile per call; the release adds
roughly 30–40 % to the tail because a woken robot is planned against more tasks. Travel
rises 2.3 % in the condition that bought the significant gain, and is flat elsewhere.

## 5. What this block does and does not license

* It is robustness evidence on 180 seeds no earlier block touched. It is not a second
  independent confirmation of the original claim, and its numbers are never pooled with
  the 400-seed block (which measured the earlier, residual-based verdict).
* The load condition supports "clears late completions without breaking anything" at the
  adjusted level. The two deadline-band conditions are inconclusive at the ceiling and are
  reported as such; they are not a failure of the mechanism and no seeds were swapped in to
  make them look better.
* The 1.3 % belief-lag residue is inherent to deciding from a lagging belief and is
  disclosed, not engineered away: no online rule can see a stage change it has not been
  told about. What is no longer true is the systematic version of the problem — a sole
  provider is now never released on the strength of a zero it created itself.
