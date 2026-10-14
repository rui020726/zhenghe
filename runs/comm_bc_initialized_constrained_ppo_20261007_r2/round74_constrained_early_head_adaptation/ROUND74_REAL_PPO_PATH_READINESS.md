# ROUND74 — Real PPO Path Readiness (read-only)

**Status: read-only readiness prep under the decision-end pause. Zero new model load / forward /
backward / optimizer / PPO / gate / env execution. HEAD at writing:
`9f38c87035fab857c80abe84b81435e9b7dfe06e`. The candidate plan in section 6 is NOT an authorization;
the decision-end must fix source/code/plan and issue a separate continuation before any consumption.**

## 1. Purpose

Locate, with source identity, the real PPO call chain that PHASE B1 never exercised; record the
minimal 8-object wiring fix; inventory what real recorded-batch material exists; state conservative
step upper bounds; and give a feasibility verdict against the R74 quota.

## 2. Real PPO call chain (verified read-only this session)

| layer | file / identity | role |
|---|---|---|
| B1 runner | `r74_phase_b1.py` sha256 `95e93875fec9dc54205194b14993921fc96a4c9b8971399dd912f33e9da65b20` | drives ONLY `r74_model.minibatch_loss` (a standalone one-minibatch transcription) + manual `Adam.step`; never calls `ppo_update` / `batch_samples` / GAE / collector / env. B1 receipts: real `ppo_update` calls = 0, real env = 0. |
| frozen update | `round66_bc_early_path_preserving_late_adaptation/r66_frozen.py` sha256 `A22425729C9327FFCB1D57F4820F3B38F2BB7E35A6D6D81890FE64A75A246B10`, `frozen_early_ppo_update` lines 664-902 | the intended real update: `EPOCHS_PER_BATCH(4) x MINIBATCH_STEPS(256)` shuffle seeded by `DATA_ORDER_SEED:update_index`; loss = `-surrogate.mean() + VCOEF*(vt+vc) - ENT*ent + beta*gate`; `clip_grad_norm_(trainable_parameters(controller), MAX_GRAD_NORM)`; `optimizer.step()`; returns `optimizer_steps=minibatches`. |
| trainer | `D:\code\整合\zhenghe\runs\comm_cost_v2_rerun_20260928\v2_trainer.py` sha256 `0550B99EB1F10A45D57757CAEDB4CE365FE858D72E171537E38DC450FB49A841` | `ppo_update` line 179; `batch_samples` line 47; `likelihood_alignment` line 155; GAE via `cm.gae`. |
| constants | same tree `v2_common.py` | `DATA_ORDER_SEED=20260927`, `GAMMA=1.0`, `MAX_GRAD_NORM=0.5`, `EPOCHS_PER_BATCH=4`, `MINIBATCH_STEPS=256`, `B0_BYTES=10000.0`. **Trap: `GAE_LAMBDA` import default is 0.95, but the frozen live runs used `full-trace-gae-lambda-1` (recorded in the rollout `training_version` string). Live values must be bound from the frozen record, never from the import default.** |

`r66_frozen.py` hardcodes (read-only line refs): `FROZEN_MODULES = ("trunk", "early_head")` (line 483);
`trainable_parameters(controller)` (556) = 6 objects under the R71 mask; `Adam(trainable_parameters(...))`
(614); clip over the same 6 objects (840). These are the points the R74 wiring must diverge from.

## 3. Minimal 8-object fix (located; NOT implemented, NOT tested)

`r74_model.py` sha256 `dfd1c4744053a83d43727a9107e2fb3acdab86b5dcbe51484464cbff39b54260` already carries
the R74 partition: `FROZEN_MODULES = ("trunk",)`, `TRAINABLE_MODULES = ("early_head", "late_head",
"value_task_head", "value_cost_head")` -> 8 parameter objects, with `partition` /
`trainable_parameters` / `build_optimizer` / `set_mask` implemented.

Fix = one new function `r74_frozen_early_ppo_update`, a copy of `frozen_early_ppo_update` with
`FROZEN_MODULES = ("trunk",)` only, clip + optimizer over the 8 trainable objects, everything else
identical (same shuffle seed, same loss terms, same clip algorithm, same return contract). Do NOT edit
the clip algorithm to manufacture equivalence; do NOT route R74 through the R66 6-object path.

## 4. Batch evidence (read-only inventory this session)

### 4.1 R71 batch records are aggregate-only (NOT replayable as-is)

96 files `r40_batch_s{11,22,33}_b{0..31}_attempt01.json` (round71 `raw/checkpoints`). Present: kind,
seed, batch, episodes, scenarios, `lambda_at_rollout`, `update` diagnostics (`samples` as an integer
count, `minibatches`, `optimizer_steps`, `epochs`, `early_free`, `late_free`, `gate_rows`,
`trainable_parameter_object_count`=6, ...), constraint pairs, dual (`eta`=0.5),
`parameters_before_sha`/`parameters_after_sha`, structure string, `resumed_batch`. Absent: per-sample
features / actions / sequential mask / behavior log_prob / advantages / values (zero occurrences of
those keys). `batch_samples` cannot consume them. The identity chain checks out: b0
`parameters_before_sha` = `6f6581cf...` (s11), `29dbecdb...` (s22), `3685f4aa...` (s33), matching the
R71 INIT model-only identities.

### 4.2 Recorded per-batch shape (96/96 parsed)

| seed | batches | samples | optimizer steps | final `optimizer_steps_after` | final `episodes_done_after` |
|---|---|---|---|---|---|
| s11 | 32 | 11009 | 236 | 236 | 128 |
| s22 | 32 | 10565 | 220 | 220 | 128 |
| s33 | 32 | 11638 | 240 | 240 | 128 |

Per-batch steps = `4 x ceil(samples/256)` in {4, 8, 12, 20}; observed samples 194-1071. Reproducing
the full R71 training would cost 696 optimizer steps - impossible under the 16-step hard cap; any
probe must use a small batch.

### 4.3 R58-lineage rollout material exists in full (git-external, row + tape pairs)

386 paired episodes: s11/s22/s33 x e0..e127 (e4, e43 duplicated). Row (`raw/rows/*.json`) provides the
episode wrapper `batch_samples` needs: `steps`=1000, `cost_stream` (1000 ticks, `airtime_bytes`),
`task_stream`, `lambda_at_rollout`, `episode_index`, `training_seed`, `behaviour_sha256`,
`controller_parameters_after`, `training_version` (`... + r58-prior-strength-beta1p00/1`), and
`substituted_fields` (12 archived-null substitutions an adapter must surface, not silently accept).
Tape (`raw/logs/*_tape.jsonl`) provides per-decision-step records: `features` = list of per-candidate
19-d vectors, `targets`, `actions`, `actionable`/`free`/`forced`/`masked`, `log_prob` (original
recorded, not backfilled), `value`, `cost_value`, `step`; schema `round42-one-shot-send-skip-tape/1`,
`tape_schema` `onpolicy-tape/2`. Together they satisfy the `batch_samples` episode contract
(`collector.records`, `steps`, `cost_stream`, `task_stream`, `lambda`, `episode_index`).

First-batch material: `round58_gen2` 12 episodes (3 seeds x e0..e3; scenarios
1300027/1300014/1300028/1300044 - the same scenario set as R71 b0); decision steps 66+71+52+76 = 265
(s11/s33) and 266 (s22), matching R71 b0 `samples` 265/266/265.

**Provenance caveat:** these are R58-lineage rollouts (sampled decoder), NOT R71's own rollouts; R71
kept only aggregates. Replay is real recorded data through the real path; it does not reproduce R71's
batch data or behavior log_probs under the R71 model, and it is not evidence about R71's training.

### 4.4 Gaps (honest list)

- No R71-own rollout tape; no per-sample R71 batch payload anywhere in the tree.
- `likelihood_alignment` behavior on cross-lineage replay is untested.
- The real collector / environment sequential budget mask on a real candidate stream remains
  unexercised (B1 used synthetic free-sets).
- `GAE_LAMBDA` live-vs-default trap (section 2) must be bound from the frozen record before any run.

## 5. Conservative step upper bounds

Steps per real update = `4 x ceil(n_samples/256)`: 1 episode (~66-76 samples) -> 4; 2 (~120-150) -> 4;
3 (~190-215) -> 4; 4 recorded episodes (265) -> 8; full R71 batches (up to 1071) -> 20.

Candidate probe: establish non-empty Adam = 1; source next-update = 4 ONLY if the batch is <= 256
samples; full-restore same next-update = 4. Candidate total <= 9 steps. The 4-episode first batch
(265 samples) does NOT fit the <= 4-steps-per-update premise; a replay probe must use <= 3 recorded
episodes per seed (or any real batch with <= 256 samples).

## 6. Candidate plan (NOT authorization)

1. Implement `r74_frozen_early_ppo_update` (8-object) plus a read-only replay adapter
   (row + tape -> episode dicts), OR a genuine env rollout through the real collector (option (a)
   spends real env slots; option (b) is offline replay of recorded rollouts). Decision-end choice.
2. Wrap that real update in the source-vs-full-restore determinism gate (full-state save -> restore ->
   same next-update -> compare).
3. Accounting: every real `Adam.step` - dev, fixture-named, or production alike - is charged into the
   single central ledger BEFORE the step; the real runner removes the fixture execution mode entirely
   (the B1 fixture/production split is what caused the under-count).
4. Quota framing: synthetic steps normal 12 + technical 4 / hard 16 (>= 16 already physically consumed
   per the errata -> 0 consumable until re-baselined); offline rows hard 6000 / per-batch 1024 (28
   canonically charged); env normal 579 / reserve 21 / hard 600 (0 charged).
5. No retroactive allocation, no reclassification; technical failures do not refund.

## 7. Feasibility verdict

The real call chain is fully locatable with source identity; the 8-object fix is small and mechanical;
complete real recorded-batch material exists in git-external storage. The binding constraint is quota:
with the physical-step lower bound at the hard cap 16 and unrecovered dev residue, R74 currently has
0 consumable quota. The candidate plan (<= 9 steps) fits inside the original normal cap 12 ONLY if run
exactly once with a <= 256-sample real batch and zero extra probes, and ONLY after the decision-end
re-baselines quota and issues a separate continuation. Until then: no execution, no training start.

## 8. This session's zero-side-effect record

Operations were file reads, SHA-256 hashes, JSON manifest parsing, and ripgrep searches. New model
load / forward / backward / optimizer step / PPO call / env construction / checkpoint write: all 0.
Two pure-JSON probe scripts were written under `_readonly_probe/` (they generate the section-4
inventory numbers; zero NN/env). Existing audit files untouched; ROUND46 protected files verified
intact; this document and `ROUND74_PHASE_B1_ERRATA.json` only append.