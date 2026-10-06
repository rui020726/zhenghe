# ROUND45-R execution continuation basis

This is one explicitly authorized new technical attempt after the original ROUND45 seed 11 checkpoint serialization failure. It does not restore or rewrite that failed attempt.

## Parent and old attempt

- Parent delivery HEAD: `c80ffa843e0307bfae20ba85bb36458fe21144c0`.
- The prior attempt remains in `runs/comm_rule_warmstart_cost_learning_20261006/` and is read-only.
- Its seed 11 1,000 returned BC updates, metrics, journal, missing post-BC checkpoint, stop state, serializer fix and regression remain unchanged.
- The R2 branch/output are `exp/comm-rule-warmstart-cost-learning-r2` and `runs/comm_rule_warmstart_cost_learning_20261006_r2/`.

## Authorized count changes

- This attempt starts all three seeds 11/22/33 fresh from the registered R44 ep0 parameters and may run exactly 1,000 fixed BC updates per seed, at most 3,000 new real BC updates.
- Family cumulative BC supervision hard cap is 4,000: the old failed 1,000 plus at most 3,000 in R2. The old 1,000 is not treated as a resume and is never replayed or discounted.
- Previously used synthetic optimizer updates remain 2; cumulative cap remains 64. Reuse unchanged synthetic/teacher/data audits; do not rerun old synthetic optimizer updates.
- Environment authorization remains 528 normal + 24 technical reserve = 552 cumulative maximum: 48 BC-only controls, 384 PPO episodes, 96 final evaluations. Old ROUND45 constructed zero environments. No new environment quota, stream, smoke, test suite, or reference rollout is added.

## Required R2 sequence

1. Use a new branch from the parent delivery and a new output directory. Preserve all old attempt files. Freeze R2 sources and inputs locally before real BC.
2. Reuse the exact prior dataset and R2 data audit by byte hash; do not regenerate labels or rescan the old tapes. Keep the fixed canonical teacher: first emission and `ready_step > 147`, not current tick greater than 147.
3. Bind to the repaired nested checkpoint serializer. Before training, run a zero-update real filesystem save/read round-trip in a dedicated fixture path, through the production payload builder/writer/reader/controller loader. Verify tensors, all four module hashes, actor/task/cost forward equality, complete Adam slots and RNG preservation. Mark the fixture as synthetic, never as an experiment checkpoint.
4. Each fresh seed runs exactly 1,000 full-batch CPU Adam updates with the frozen fit-only inverse-frequency weighted CE, learning rate 0.001, global clip 0.5, zero weight decay, and only trunk plus action head parameters. Task/cost heads' own parameters remain unchanged. Save each resulting model and immediately reload it with the production loader; if any complete model state is missing or invalid, stop with no third attempt.
5. Freeze all three post-BC models and the original 528 logical environment keys. Use BC weights with a fresh full PPO Adam over the actual ten parameter objects, empty state/step zero, no carried BC moments; lambda 1, episode/update zero, registered R44 ep0 rollout RNG.
6. Run all 48 BC-only greedy controls. Continue automatically only if each seed preserves its immediate reference target identity set in every scene and its mean bytes are no greater than the fixed late-discovery rule.
7. If that gate passes, run the frozen 384 PPO episodes with per-seed first-four correctness gates, then the fixed 96 ep128 greedy/sampled matrix. Report comparisons separately against R44 NEW same seed/mode, fixed rules, and each seed's own BC-only greedy controls. Do not claim BC-only sampled results.

## Frozen science and stop conditions

All scientific math, observations, masks, teacher, data, hyperparameters, world/channel/budget, scene order, and gates continue unchanged from `EXECUTION_BASIS_FULL.md`. The R2 version records only the new technical attempt, repaired checkpoint serialization, and output identity. Old immediate references, fixed-rule rows, and the true R44 NEW ep128 96-cell matrix are reused after hash verification; R43 evaluation rows remain historical only.

Do not touch sealed blocks `800000..800399` or `900000..900199`. Do not push, merge, deploy, open ROUND46, overwrite old artifacts, rerun completed environment keys, or replay committed PPO updates. If another missing/incomplete BC model, scientific conflict, invalid gate, or unrecoverable model/optimizer/dual/cursor/RNG state occurs, preserve evidence and stop. No third BC attempt is authorized.
