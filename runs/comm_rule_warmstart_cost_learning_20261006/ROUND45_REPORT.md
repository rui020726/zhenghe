# ROUND45 execution report

## Status

**Stopped during seed 11 BC checkpoint serialization.** This is a technical execution failure, not a policy result. The fixed three-seed plan did not reach initialization evaluation, PPO training, or final evaluation. No research verdict comparing the BC initialized policy with R44 is available.

- Repository: `D:\code\整合\zhenghe-round42`
- Branch: `exp/comm-rule-warmstart-cost-learning`
- Frozen source commit: `6367afbb9fb19bde1a61f19634b70955dce77d70`
- Pre-BC freeze commit: `c6aa274c7d7719a75824a5088bc4af2e3142a6ea`
- Post-failure serializer repair commit: `bfa0a952f9c558331c4d4033584d0231e93742be`
- Frozen BC source SHA256: `04ff2b2cffaefc5fb15928d2e785079aa23539fc7d60194fba7c639792d3c398`
- Dataset SHA256: `93ffba45bfa484aa0f0aa5cbb334bd76c1284d0b21e7ca704b459ae6126f378d`

## Verified inputs and gates

- The fixed rule-label dataset has 2,286 rows: 1,548 fit and 738 validation. The input hash remained equal to its committed audit.
- Teacher timing and action labels were verified against the canonical `LateDiscoveryGatePolicy(cut_steps=147.0)` path. The float32 softmax tie diagnostic matches the frozen collector: `p_SEND >= p_HOLD` selects SEND.
- Pre-BC freeze passed using the registered Python/Torch environment. Fresh controller parameter identities matched the R44 ep0 identities for all three seeds. Adam defaults, dataset split, optimizer scope, and unchanged R44 training parameters were recorded in `ROUND45_PRE_BC_FREEZE.json`.
- ROUND44 NEW ep128 is the primary OLD comparison: its three checkpoints were read-only deserialized and their file hashes, training metadata, full parameter hashes, and four module hashes matched the R44 checkpoint index and parameter chains. The actual R44 strict-index auditor reconstructed all 96 evaluation rows and matched the delivered matrix; six groups have 16 rows each. The `(11, greedy, 1301000)` entry selects valid attempt 2.
- ROUND43's 96 rows remain a separate historical archive and are not the R45 primary baseline.

## Execution and exact counts

Seed 11 returned all 1,000 planned offline BC Adam updates. Its 2,000-line journal contains 1,000 invoked and 1,000 returned events. Metrics were written, then checkpoint payload creation raised `AttributeError` because the frozen serializer treated nested `OrderedDict` module states as tensors. The post-BC checkpoint was never created. Full exception and artifact digests are in `ROUND45_BC_FAILURE_SEED11.md`.

| Activity | Count | Status |
|---|---:|---|
| Real BC optimizer calls returned | 1,000 | Seed 11 only |
| Synthetic optimizer steps | 2 | Previously completed; cap 64 |
| BC seeds 22 and 33 | 0 | Not started |
| Durable post-BC model checkpoints | 0 | Seed 11 state unavailable |
| PPO updates / PPO optimizer calls | 0 | Not started |
| Normal / reserve environment claims | 0 / 0 | No claims created |
| Environment constructions / run calls / completed episodes | 0 / 0 / 0 | No simulation started |
| BC-only controls / final evaluations | 0 / 0 | Blocked by missing seed 11 model |

The saved seed 11 final parameter digest is not a recoverable model. The execution basis forbids replaying committed updates when complete model and optimizer state cannot be proven recoverable. No BC update was replayed, and the remaining seeds were not started after the first checkpoint seam failed.

## Post-failure repair check

After preserving the frozen execution and its failure evidence, the nested state-dict checkpoint serializer was minimally repaired as `round45-bc-nested-controller-state/2`. The targeted regression runs the production payload builder, `torch.save` writer, `torch.load` reader, and controller loader using `io.BytesIO`. It reproduces the old exception on a real canonical controller, checks all 10 tensors and all four module hashes, verifies actor/task/cost forward outputs are equal, and round-trips all six BC Adam state entries and their `step`/`exp_avg`/`exp_avg_sq` slots without an optimizer call. The fixture is marked synthetic; no checkpoint file was created.

This repair was made after the frozen run and was not used to resume training. It cannot restore seed 11's missing checkpoint. The run remains stopped; the repair is only a verified code correction for any separately authorized execution.

## Scientific interpretation

The completed seed 11 optimizer trajectory is not a usable policy result because its model state was not saved. No BC feasibility gate, no R45 PPO, and no R45 final model comparison can be reported. The BC parameter digest and metrics are preserved as execution evidence only; they do not support inference about action behavior or task/cost outcomes.
