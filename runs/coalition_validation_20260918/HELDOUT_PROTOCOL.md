# Held-out protocol: frozen candidate, 400 new seeds, statistics fixed before running

Written and committed **before** any held-out stream was recorded or replayed. If the
candidate source changes after any held-out result is seen, this block becomes a
development block and cannot be re-labelled as independent for that version.

## 1. Frozen candidate

| item | value |
|---|---|
| branch | `validation/coalition-redundant-release` |
| candidate commit | `1f4a5988cc05becab717eecba551c3ea2d196727` — stamped into every run signature by the preflight; the `integrated_sar` tree is byte-identical to the two earlier freezes `304fc60` and `e96f34c` (`git diff` empty), so the identity precondition and the held-out runs share one source |
| stable baseline | `fix/coalition-commitment-continuity@868d68a` (untouched) |
| switches, baseline arm | `coordinator_policy=coalition_planner`, `sticky_commitments=True`, `release_redundant_members=False`, `suppress_recommand_own_task=False` |
| switches, candidate arm | same planner + `release_redundant_members=True` + `suppress_recommand_own_task=True` |
| residual scale / intervals / ttl / dt | 0.25 / 5 / 5 / 10.0 / 1.0 (identical to every prior arm of this line) |
| fleet, comm, env | 3 species x 2 = 6 robots, delivery p=1.0 delay 0, `TaskEnv` unchanged (no env/comm/deadline edit in this round) |
| checkpoints | PILOC `runs/cur_v2rep05_s2/best/shared_mappo_model_21_3100.pth`, CTDE `runs/joint_eval_20260916/ctde_final_7_1.pth` (digests written into `stream_index.json` and every run signature) |
| trace schema | `allocation-decision-trace/3` |
| episode | 1000 steps, 6 hidden goals, 60x60 grid, obstacle density 0.15, detection radius 5 |

Everything else is unchanged from the binding baseline: the two switches are the only
difference between the arms.

## 2. Seeds

`heldout_seeds.py` draws, in ascending order from 30000, the first 400 integers absent
from **every** historical manifest: the three `data_splits` blocks (dev 10000-10029,
analysis_legacy 20000-20099, analysis_fresh 21000-21059), `final_eval_20260916/test_seeds.json`
(2000000000+) and this round's `ablation_seeds.json`, each source hashed into the
manifest. Result: seeds 30000-30399, verified disjoint (assertion in the script). No
re-draw, no difficulty filtering, no post-hoc substitution.

## 3. Measurement design

* Streams are recorded once per seed with live PILOC and `--no-coordinator` (the same
  recording protocol as the 20260917 held-out block), then **both arms replay the
  identical file**. What is therefore measured is *allocation given the same detections*;
  search-side effects are out of scope here and are covered separately by the system
  closed loop (brief section six).
* Identity precondition: with all new switches off, the arm must reproduce the committed
  868d68a-era trajectory on the five development seeds, field by field
  (`heldout_replay.py --identity`), before any held-out number is read.
* Preflight per run: `run_signature` (code commit, arm, seed, checkpoint digests, config
  hash, stream digest, schema) + stream sha256 against `stream_index.json`. Technical
  failures (crash, IO) are rerun under the same signature; policy failures are kept.
* Single main process owns the shard manifest, preflight and aggregation; workers write
  only their own files, atomically.

## 4. Statistics, fixed now

* **Unit**: the seed. **Primary metric**: per-seed on-time completion rate = targets
  finished on time / targets detected. The denominator is identical in both arms because
  the stream is shared.
* **Zero-detection rule (pre-declared)**: a seed whose stream contains no detection
  contributes no rate; such seeds are excluded from the primary mean **by rule, before
  any result is seen**, and reported by count in the closure table. They are never
  dropped after the fact.
* **Estimator**: mean paired difference (candidate minus baseline) across seeds; 95% CI
  from a seed-level nonparametric bootstrap, B = 10 000 resamples, RNG seed 20260918,
  percentile interval. A t-based interval is reported alongside as reference only.
* **Claim rules**: "significant improvement" requires CI lower bound > 0. If the CI
  crosses 0, the result is reported as a trend or as insufficient evidence — never as a
  demonstrated gain, and never as absence of harm unless the correctness checks hold.
* **Secondary metrics**, same pairing: expired-count difference; completion latency for
  on-time targets (`time_finish - detected_at`); zero-marginal-contribution working
  ticks; coverage interrupts and discarded work; reassignment attempts / acceptances /
  declines / replan-to-accept latency; duplicate own-task commands; rejected and illegal
  commands; unjustified overrides; longest no-progress idle streak.
* **Candidate-only regressions are listed per target**, even when the aggregate improves.
* **Ceiling rule**: if both arms complete ≥ 99 % of detected targets, the ceiling effect
  is reported as such and no extra seeds are drawn.
* **Closure table**: detections, on-time completions, expired, active/unresolved,
  zero-detection episodes, runs per arm — counts must add up on both arms.

## 5. Correctness acceptance (independent of the performance claim)

1. No coverage loss attributable to a joint release: every target on time in baseline and
   not on time in candidate is individually attributed.
2. No illegal or rejected command that the baseline does not also exhibit; any
   candidate-only rejection is attributed to a mechanism and reported.
3. No work-clock reset from a duplicate command: candidate own-task re-commands = 0.
4. No truth leakage: policies read only beliefs and snapshots (structurally enforced and
   covered by invariant tests; the trace's truth block is written, never read back).
5. No demonstrated scheduling deadlock: zero-progress spans where able, believing robots
   were all locked are counted with their release cause; a span that no escape rule
   resolves is a failure, not a number to average away.

Deadlock and "no progress" are judged from the continuous logs (progress, coverage,
release reasons, waits), not from the final snapshot alone.

## 6. What may not change after results are seen

Switch values, parameters, checkpoints, schema, seed set, statistics, and the pass/fail
thresholds above. Changing any of them after seeing held-out results converts this block
into a development block for the new version.
