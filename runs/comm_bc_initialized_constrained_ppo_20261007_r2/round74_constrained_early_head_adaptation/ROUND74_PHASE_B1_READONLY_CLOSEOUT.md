# ROUND74 PHASE B1 — Read-Only Close-Out (decision-end pause)

**Status: paused at read-only close-out per the decision-end instruction. Zero new model load /
forward / backward / optimizer / PPO / gate execution. PHASE B not entered. No other task/process
touched. R74 paid operations are paused; the overall research goal is not closed.**

## The core correction (withdrawn claim)

I previously framed the two `--mode fixture` runs as "simulated, 0 production charge." That is
**wrong**. `R74ResourcesProd.fixture(root)` separates only the durable ledger/counters. The torch
controller, `forward_logits_values`, `minibatch_loss` backward, clip, and `Adam.step` are the same
real objects in both modes (`r74_phase_b1.py`: line 77 `production()` vs line 80 `fixture()`; the rest
is shared). Every B1 run did **real** forward/backward/Adam.step; fixture only redirected where the
charge was recorded. So the canonical counter (toy 2 + production 3 = 5 steps; production-only 28
rows) under-counts the real work by the two fixture runs (6 steps, 56 rows).

## Actual consumption (per attempt)

| attempt | mode | exit | real Adam.step | 19-d rows | metered where | originals |
|---|---|---|---|---|---|---|
| A0 toy (prior/PhaseA) | toy | - | 2 | 0 | canonical | intact |
| A-dev (prior, raw scripts) | raw | - | >=4 UNKNOWN | >=10 UNKNOWN | none | **deleted this session** |
| B1 run1 | fixture | 1 | 3 | 28 | throwaway (deleted) | **deleted** |
| (delete _b1_fixture) | - | - | - | - | - | run1 receipt destroyed |
| B1 run2 | fixture | 0 | 3 | 28 | throwaway (rebuilt: 3/28) | intact |
| B1 run3 | production | 0 | 3 | 28 | canonical (2->5 / 0->28) | intact |

Firmly evidenced real Adam.step = **11** (toy 2 + 3x3). Plus prior dev scripts >=4 (exact UNKNOWN,
scripts deleted) -> total lower bound >=11, plausibly >=15. Firmly evidenced 19-d target rows = **84**
(3x28); canonical shows 28. Real `ppo_update` / `frozen_early_ppo_update` calls = **0**. Real Env = **0**.
Synthetic gate checkpoints written = 3 (+1 dev, deleted). Controller constructions = 15; torch.load = 18.
The rebuilt `_b1_fixture` is run2's artifact and does NOT stand in for the deleted run1 receipt; run1's
exit code and consumption are preserved only via captured stdout.

## Coverage: component checks only

Covered (actually exercised): controller construction + model-only R71 INIT (5 modules) + R74 mask;
routed forward boundary; autograd gradient routing + critic isolation; the `minibatch_loss`
transcription forward/backward; real Adam.step on the 8 trainable objects with trunk frozen; full-state
save/restore + determinism + foreign-resume refusal; the R74 charge-before-step + trunk-entry meter seams.

Missing (NOT exercised): the real multi-epoch `v2_trainer.ppo_update` / `r66.frozen_early_ppo_update`
loop (`EPOCHS_PER_BATCH`x`MINIBATCH_STEPS` + `DATA_ORDER_SEED` shuffle); `batch_samples` from real
rollouts; GAE; `likelihood_alignment`; the real collector / environment sequential budget mask;
end-to-end ALL_ROWS pooling; the real training CLI entrypoint. So G1-G5 passing qualifies the wiring
**components**, not the real PPO loss/backward/subsequent-update, and cannot justify starting training.

`r74_model.minibatch_loss` is a standalone one-minibatch transcription of `frozen_early_ppo_update`
(itself a transcription of `v2_trainer.ppo_update`); it is never called BY `ppo_update`, and the runner
never calls either. `ppo_update` (v2_trainer:179) drives `batch_samples` + `likelihood_alignment` + the
epoch/minibatch shuffle loop that the runner bypasses with hand-built synthetic samples.

## Minimal real-wiring fix (located read-only; NOT implemented/tested)

Replace the G4/G5 hand-built `minibatch_loss`+manual step with a call to `r66.frozen_early_ppo_update`
on a REAL batch produced by (a) a genuine env rollout -> real collector -> `batch_samples` + GAE, or
(b) offline replay of a frozen recorded R71 episode batch through `batch_samples`+GAE. Then wrap that
real update in the source-vs-full-restore determinism gate. (a) spends real Env slots; (b) is offline
replay. This is a decision-end choice.

## Quota

Canonical ledger preserved (5 steps / 28 rows / 0 env), not rewritten. Caps unchanged
(synthetic 12/4/16, offline 3000/3000/6000, env 579/21/600). No retroactive exemption, no
reclassification. Because the deleted dev scripts make the real total unprovable, **consumable
remaining is treated as 0** until re-baselined; this audit spends nothing.

## Identity / protection

HEAD `812b435b7350e20dbf32967e63eb68f69279adb1`. Source SHAs: `r74_phase_b1.py`
`95e93875fec9dc54205194b14993921fc96a4c9b8971399dd912f33e9da65b20`, `r74_model.py`
`dfd1c4744053a83d43727a9107e2fb3acdab86b5dcbe51484464cbff39b54260`, `r74_guard_prod.py`
`623b9ec6cd243d27321e8c745a0247c97126c9028141cb0ed4760dd90c0796db`. Production receipt
`raw/runtime/round74_phase_b1/ROUND74_PHASE_B1_VERDICT.json` (root copy sha
`504eb4549fef00b59dbec7a915c55557928547bca42a8cdee4024ee95c1c0c49`). ROUND46 files + R59 log
untouched. This close-out only APPENDS. Local commit only; no push/merge/deploy; no auto next round.