# ROUND73 Execution Basis (archived authorization)

Task: `ROUND73-PURE-COST-ACTOR-ABLATION` (S3 development diagnostic on the verified ROUND71 runtime).
Branch: `exp/round73-pure-cost-actor-ablation`. Run id: `r73-pure-cost-actor-20261014`.

## The single structural intervention

CONTROL (ROUND71, verified): the PPO actor advantage is `A = lambda_dual * A_task - A_cost`
(`cost_math.actor_advantage`, formed in `canonical_v2_trainer.batch_samples`).

NEW (ROUND73): `A = -A_cost`. The task term and lambda are dropped from the ACTOR surrogate only.

Everything else is inherited field-for-field from the accepted ROUND71 living configuration:
dual critic targets/losses, GAE (gamma=1, gae_lambda=1.0 live, not the 0.95 module default),
reward timing / event attribution, ALL_ROWS reduction, sequential budget mask, entropy bonus,
global gradient clip, dual-lambda update (eta=0.5, lambda_init=1.0), frozen trunk + early_head,
trainable late_head + both critic heads, five-module checkpoint structure, 19-dim observation,
one-shot SEND/SKIP, candidate order, 6 goals, 29300B budget, ACK off, 1000-step horizon,
search/assignment checkpoints, world fields, seeds 11/22/33 and their 128-episode order
(inherited from R71), greedy tie rule and sampled eval RNG scheme. The task critic still trains
and the real task reward is still collected/evaluated; lambda is still updated and reported; it
simply no longer enters the actor advantage.

The INIT of every seed is the ROUND71 calibrated INIT checkpoint, loaded model-only (five
modules, no optimizer, no lambda, no training RNG, no training cursor). ROUND73 performs no
offline fit and re-uses no ROUND71 trained weights, Adam state or lambda as a training source;
a ROUND71 training checkpoint is refused as a ROUND73 resume source.

## Quota (fresh and independent; nothing borrowed from R60/R61/R71)

- Normal plan: 579 environment constructions = 384 training (3 seeds x 128 episodes)
  + 192 model eval (NEW ep128 x 3 seeds x {greedy, sampled} x 32 scenes 1800000..1800031)
  + 3 focal (NEW ep128 x 3 seeds x greedy x scene 1700027).
- Technical reserve: at most 21.
- Cumulative hard cap: 600 environment constructions. Failures do not refund.
- Offline trunk-forward input rows: cap 6000 (normal <= 3000, reserve <= 3000), debited before
  the forward. ROUND73 runs no offline fit, so the planned use is 0.
- Synthetic optimizer steps: cap 8 (gates only).
- Read-only reuse (NOT new constructions): the ROUND71 same-seed/same-mode ep128 192-cell
  control rows, and the immediate/late/always_hold rule arms from the frozen ROUND68 rule
  context. The evaluation block 1800000..1800031 was already exposed for analysis and method
  selection; it is a development block, not an unseen confirmation set.

## Discipline

- Every launch goes through the durable claim/slot/permit regime; an attempt that already
  committed to an environment construction is charged again, never reused for free.
- Per seed: the first 4 episodes count toward the 128; after the first PPO update a REAL
  first-batch gate must pass before that seed continues; only the fixed ep128 is evaluated.
- No checkpoint selection, no seed selection, no re-running for performance. Performance
  degradation is not a technical failure.
- Statistics: scenario is the paired unit, seeds are not pooled, greedy is the primary reading
  and sampled is reported separately; the shared bootstrap index and interval convention of the
  ROUND68/R71 chain are inherited unchanged.
- No push, no merge, no deploy. Raw artefacts and weights stay git-external.