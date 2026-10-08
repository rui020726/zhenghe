# ROUND58 repaired-entry integration (zero execution)

## Result boundary

The previous audit-only launcher is retained as historical evidence. This addition supplies a real, receipt-gated route into the preserved ROUND58 fixed-plan executor; it does **not** issue any receipt or run the scientific plan. No Path C replacement, environment, claim, PPO update, RNG restore, checkpoint write, or evaluation occurred. Current R58 effective training batches/updates/evaluation cells remain 0/0/0.

## Actual code path

The executable is `r58_private_launcher_repaired.py` (`main` and the module `__main__` entry are after all definitions; all five receipt/output arguments are required). It performs read-only checks before loading/configuring the live runtime:

1. Require absolute, existing human-approval, freeze, cumulative-quota, and scene-audit JSON files. They may be stored in the repository; test-marked receipts are rejected. Recheck file hashes immediately before dispatch to avoid receipt replacement between preflight and execution. No repository instruction requiring external-only storage was found, so that extra restriction was removed.
2. Bind the exact frozen plan hash, repaired source identity, BC checkpoint/parameter identities, output root `round58_gen3/raw`, prior-generation count manifest, and result-blind scene block `1700000..1700031` in the receipts. The history manifest is reopened and cross-checked against the count fields; unknown or contradictory categories fail closed. The full 1024-normal/1064-hard-cap arithmetic must close before runtime configuration.
3. Dispatch into the **preserved** `round58_gen2/r58_execute.py::execute_authorized`, not a parallel trainer. For the actual `bind_runner_paths` call only, the private adapter temporarily sets the preserved module's containment root to the single frozen `round58_gen3` generation root, then restores its `HERE` to gen2 in `finally`. The original containment and historical-root overlap checks remain active: gen3/raw is accepted, while gen2/raw and an outside escape are rejected. The adapter installs the repaired beta-1 prior/critic-isolation route on the actual trainer used by `r40_train`, then the actual R40 save/load/candidate-selection contract.
4. Append `+r58-repaired-private-generation3/1` to the runtime training version. The real R42/R40 `save_checkpoint` call retains its original serializer and gets the repair identity at its `torch.save` boundary, with the same identity in checkpoint `training_state` and `freeze`. The actual R40 `train_seed` wrapper attests fresh state and freeze; the actual rolling/after-batch selector checks every candidate’s version and attestation **before** its controller restore. Wrapped load/read-only-load check before the canonical loader can restore model/Adam/RNG. Old gen1/gen2 checkpoint payloads cannot satisfy that identity. The old launcher, canonical R40/R42/R45/R53/R56 files and historical state are untouched.

The route test calls the actual `main(argv)` parser and validator with five virtual files (approval, freeze, quota, scene audit, history manifest) in an in-memory filesystem; it does not construct `_ValidatedGrant` or bypass receipt checks. Its hypothetical approval/history are explicitly synthetic, assume zero historical consumption, and exist only in memory; they neither match the actual 16 historical environment calls nor authorize a real run. The validated flow reaches the preserved executor, actual configure/bind graph and fixed first-four handoff, where a sentinel stops **before** the episode provider. Separate `__main__` regression evidence: executing the prior source at `b38b868` exits 1 with the late-definition `NameError`; the repaired module CLI `--help` exits 0, and the full positive `main(argv)` route test exits 0. Repository-local real receipts are allowed when all identity, human-reference, quota and freeze checks pass; `fixture:true`/`environment:test` remains rejected.

The checkpoint guard also uses a hash-bound `.r58.json` sidecar. A private-generation checkpoint is rejected before `torch.load` unless its sidecar binds its exact bytes, generation version, and repair identity. The live R40 snapshot/milestone rename seams finalize the sidecar for the final filename; old checkpoint paths are never rewritten. A crash or sidecar/rename failure fails closed before a successful batch receipt. This protects the deserialization/selection boundary; tests did not simulate physical power loss or filesystem durability guarantees.

## Reversible binding and tests

`prepare_actual_runtime_for_audit` now snapshots the relevant module namespaces and `sys.path`; `close_audit_runtime` undoes the checkpoint and PPO adapters and restores those bindings in a `finally` path. The actual dispatcher similarly scopes its monkeypatches and restores the process bindings on normal return or exception; the production module entry is a dedicated Python process.

Narrow evidence: **6 private-launcher tests, exit 0** for the final entry/path changes; prior unchanged evidence is 16 adapter/entry tests, exit 0, plus 3 actual-R40 commit-seam tests, exit 0 (25 combined checks, all exit 0). The route passed through `main`, all real receipt validation, the preserved fixed executor, actual R46 configure/live R40 trainer, gen3 containment binding, and the fixed first-four handoff, then stopped before any environment constructor. In-memory serializer spying on the actual R40 save path verified the repaired version and matching top-level/state/freeze attestations. The actual R40 resume selector/load seam rejected an old payload before `_restore_controller`.

The transaction suite made 5 PPO function calls over stub episodes. Its Adam `.step` methods were replaced with no-op spies (20 callbacks total); no weights or optimizer state were updated. Existing prior tests add 104 gradient-capture-only fake callbacks. Therefore this validation has 124 fake callbacks in aggregate, **0 real `optimizer.step` calls**. Transaction and configure I/O are intercepted in memory; the live binder copied source `.py` modules into a temporary test root, which was removed at test exit. No checkpoint, batch receipt, claim, journal or training report was written. The test did not claim to validate physical disk power-loss atomicity.

## Still blocked; no authorization inferred

There is no real human execution approval, new scientific freeze, current scene-occupancy receipt, or reconciled cumulative quota receipt in this workspace. The preserved history reconciliation still reports 16 constructor/run calls, 48 evidence-supported historical optimizer calls, 0 effective R58 batches, and unresolved normal/reserve classification (gen1 four normal claims; gen2 twelve normal claims conflict with its registration’s reserve statement). The current full fixed plan cannot pass the remaining-normal-cap check while these records remain as classified. The original four rows/tapes remain hash-closed candidate data, not certified reproducible PPO input. This implementation does not relabel claims or override those conditions. The synthetic positive route fixture is deliberately a different, internally consistent hypothetical future ledger; it is not evidence that the live quota gate currently passes.

Thus no real authorization was created or used. The future execution route is integrated and exercised through the first-four handoff under an in-memory synthetic fixture and controlled stop, but it has not been authorized, frozen for a real run, or shown to complete training/evaluation. The research result remains **not executed**, not “experiment complete.”

## Counts and protected files

This task performed 0 environment constructions/starts/simulations, 0 claims, 0 real optimizer updates, 0 RNG restores, 0 checkpoint writes, 0 evaluation cells, 0 tape replays, and 0 pushes/merges/deployments. Prior 16/48 historical totals above are referenced from the existing reconciliation, not rebuilt or changed.

Protected files remained unchanged:

- `ROUND46_CURRENT_TASK.json`: SHA-256 `cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b`
- `ROUND46_R2_V6_STOP_STATE.json`: SHA-256 `93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18`

Machine-readable checks, source identities and test counts: `REPAIRED_ENTRY_INTEGRATION.json`.
