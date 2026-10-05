# ROUND44 comparator and exact-resume bridge

## Recorded failure

The original ROUND44 execution source is preserved at `2353c4616746cf0a7c9fcb3a519a7042a28ba176`. It completed seed11 episodes 0–3, committed batch0, and then stopped in `_compare_first_batch` with `NameError: run_module is not defined` while loading the old post-update optimizer. The real read-only reproduction is recorded in `runtime/ROUND44_OLD_COMPARATOR_REPRODUCTION.json`. It left the four claims, rows, tapes, checkpoint files, journals, counters, and RNG unchanged.

## Minimal code correction

`r44_execute._compare_first_batch` now loads both post-update optimizer states through the already configured `runtime["r42_run"].torch_load_cpu` dependency. It retains the existing optimizer-state digest comparison and all other batch0 checks. The old frozen source and its prelaunch identity remain intact.

The runner uses a separate `round44-exact-resume-state-bridge/1` identity for the mechanical continuation. Its whitelist binds only seed11's already committed batch0 state: the original scientific execution head, exact batch record/commit, four claims and launch slots, four row/tape pairs, and the saved ep4/update1/Adam12 checkpoint. No alternate seed, R43-trained model, or fresh seed11 state is accepted as this continuation source.

The original stop report, status, progress, ledger, journals, and checkpoints are preserved. Mutable files have byte snapshots in `runtime/ROUND44_PRE_RESUME_R1_SNAPSHOT/`; the snapshot manifest is `runtime/ROUND44_PRE_RESUME_R1_SNAPSHOT_MANIFEST.json`. A separate resume progress path and completion status path prevent the continuation from overwriting the original stop record.

## Gates before the next claim

The versioned gate script calls the actual saved-batch `first_batch_gate`, the corrected `_compare_first_batch`, and the real checkpoint restore path. It checks parameter, dual, and optimizer tampering as read-only negative cases. It also verifies full controller and Adam state, cursor/dual history, Python/NumPy/Torch RNG state and next values, CPU placement, frozen 128-scene order, and the episode4 scene. The trained R43 checkpoint must fail the actual ROUND44 training restore gate.

The two pure helper tests and source compilation passed before the actual-data gate. No new environment, claim, optimizer step, or checkpoint write has occurred during this repair. The next fixed tasks remain gated on the actual comparison and restore receipts.

## Scientific identity

The scientific training version and ROUND44 freeze are unchanged. `eta_lambda=0.5` remains the only scientific intervention; `gamma`, shared GAE, dual initialization, reward, observation, mask, PPO, data order, models, scenarios, and evaluation plan stay at their frozen values. The resume bridge changes only mechanical source and execution identity.
