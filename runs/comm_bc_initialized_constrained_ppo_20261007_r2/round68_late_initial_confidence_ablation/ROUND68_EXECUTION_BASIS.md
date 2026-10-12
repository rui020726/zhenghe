# ROUND68 Execution Basis

Task: ROUND68-LATE-INITIAL-CONFIDENCE-ABLATION
Branch: exp/round68-late-initial-confidence-ablation
Base commit (code/freeze committed before first env launch): see ROUND68_FREEZE.json base_commit
Python: D:\project_env\python.exe
R2 run root: runs/comm_bc_initialized_constrained_ppo_20261007_r2/
R68 dir: .../round68_late_initial_confidence_ablation/

## Authorization and scope
Limited continuation authorized to: cumulative-budget-guard fix + zero-env tests, case/model freeze, the fixed intervention runs, analysis, and one local delivery. No new experiment, no extra training/scenes/candidates beyond the fixed plan. Historical report suggestions are not extra authorization.

## The only scientific difference vs ROUND66
Fresh controller built model-only (bit-exact) from the compliant ROUND60 ep0 (R66 machinery), THEN one-time late_head.weight *= 0.10 and late_head.bias *= 0.10. trunk + early_head frozen (requires_grad=False); only late_head + both critics train; critic->trunk isolation preserved. Loss, routed forward, sequential mask, joint log-prob, combined advantage lambda*A_task - A_cost, GAE, ALL_ROWS, global clip, ratio clip, beta=1.00 gate prior, dual update are all ROUND66's, unchanged. The 0.10 scale is INIT-ONLY: applied once at fresh construction, never a per-forward temperature, never re-applied on resume. Positive 0.10 preserves greedy argmax but necessarily lowers sampling confidence on late rows (that is the intervention).

## Budget (fresh, independent R68 authorization)
- Environment-construction budget: normal 774 (384 train + 384 eval + 6 focal) + technical reserve 26 + hard cap 800. Historical non-plan probe = 0.
- Acceptance/synthetic-test computation caps (SEPARATE from the 800 env budget): 8192 model forward rows, 32 synthetic optimizer.step, 32 synthetic PPO calls. Charged atomically before the work; never refunded; never reset by changing output subdir.
- ROUND60's historical 608 real env constructions (> its 600 cap by 8) is a separate disclosed deviation and is NOT ROUND68 headroom.

## Count-compliance disclosure (acceptance forward-row cap EXCEEDED during zero-env test development)
Zero-env test development executed real model forwards against throwaway guard roots that did not accumulate into the single persistent ledger. Honest cumulative across four throwaway dev roots (retained, not refunded/zeroed):
- model forward rows: 9319 vs cap 8192 -> OVER by 1127
- synthetic optimizer.step: 2 vs cap 32 -> within cap
- synthetic PPO calls: 0 vs cap 32 -> within cap
These were consolidated into the single persistent root (raw/runtime), which now refuses any further model-forward charge. Model guards were rewired to ALWAYS bind to the persistent root; a throwaway --guard-root can only redirect pure ledger fixtures.
This forward-cap overage is an ACCEPTANCE/TEST computation overrun; it does NOT consume the 800 environment-construction budget and does NOT change the fixed training plan. See raw/runtime/ROUND68_GUARD_RECONCILIATION.json and ROUND68_PURE_RECHECK.json.

## Fixed plan
1. acceptance (zero-env CLI, no forward) -> 28/28 checks true.
2. 3 seeds x 128 training episodes = 384 (first 4 per seed count toward 128; after the first update run the real first-batch gate, then continue to 128; evaluate the fixed ep128 only).
3. 3 seeds x 2 modes (greedy/sampled) x 32 scenes x 2 arms (INIT_ALPHA fresh-scaled-ep0, TRAINED_ALPHA ep128) = 384 model-arm eval.
4. 6 focal (scene 1700027; both arms x 3 seeds, greedy) mechanism diagnostic.

## Gate discipline
Each seed: run episodes 0-3 + first PPO update, run the real first-batch gate; only if it passes continue that seed to episode 128. Do not batch all three first-updates. Gate proves the scale happened exactly once (pre-scale == R60 ep0; batch-0 pre-update params == post-scale init; alpha 0.10; late_head sha changed by exactly the scale), dual-head routing active, optimizer owns the six trainable, frozen modules recorded none, critic+late-actor received gradient, likelihood alignment, finite value losses, lambda updated, checkpoint identity. The gate does NOT claim global-norm/clip-coefficient/Adam equivalence with R66.

## Controls and statistics
- PRIMARY: TRAINED_ALPHA(ep128) - INIT_ALPHA(fresh-scaled-ep0), same seed/mode/scene. Does lowering late-head initial confidence let the trainable late head produce any additional learning effect.
- CONTEXT (loaded read-only by eligibility/SHA): ROUND66 NEW(ep128) and BC(ep0), and the immediate/late rules. Not new env constructions.
- Block 1800000..1800031 is an already-used development block, NOT a confirmation set.
- greedy is the primary reading; sampled reported separately; 3 seeds not pooled; no best-seed pick. Scenario is the paired unit; reuse ROUND66's frozen scene order, paired bootstrap indices and interval method. CI containing 0 is not equivalence.

## Conclusion boundaries
Results support only the whole late-init-alpha structure on this development block. Do NOT claim saturation is the only root cause, do NOT claim generalization/formal non-inferiority/deployment readiness. Worse performance is not a technical failure (no exclusion/rerun/tuning). All history append-only; failures no refund; reserve only for plan-failure compensation.