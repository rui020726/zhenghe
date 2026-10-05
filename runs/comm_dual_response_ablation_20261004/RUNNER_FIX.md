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

## R2 row-comparison reconciliation

The R1 read-only gate stopped at episode 0 because its row projection treated the two round-specific `config_hash` values and wall-clock measurements as equal-required fields. R1's identity and the original stop/failure files remain untouched. Before changing the projection, `runtime/ROUND44_OLD_PROJECTION_FOUR_EPISODES_R2.json` captured every differing row field for all four saved seed11 batch-zero episodes; the tapes are byte-identical.

`runtime/ROUND44_SOURCE_CONFIG_RECONCILIATION_R2.json` records the actual runner/collector module paths and frozen source hashes. The loaded `integrated_sar.run_signature.config_hash` implementation independently recomputes each original R43 and R44 hash from that run's full saved configuration. For every episode, those hashes match both the row and lifecycle signature. The only configuration fields that differ are `output` and `comm_transcript`, and each points to its own verified row artifact. Replacing those two verified paths with fixed markers makes each paired configuration equal. The scenario and all other configuration/signature fields match. ROUND44's registered science diff remains only `eta_lambda: 0.05 -> 0.5`.

The R2 projection omits only top-level `seconds` and `collector_summary.timing.decide_milliseconds_per_call`. The frozen `v2_run.py` constructs `seconds` from `time.time()` elapsed time; the frozen `BudgetMaskedCollector.decide` uses `time.perf_counter()` for the per-call latency aggregate. Every other task/event/cost row field and every collector timing/count field remains compared. The action, feature, probability, reward, and likelihood tape must still match exactly. Each paired transcript must also match by file SHA256; an in-memory transcript-hash mismatch is a required negative case.

The R2 pure regression calls the actual `_compare_first_batch` entry with in-memory modifications. A wall-clock-only change passes, and the real verified R43/R44 labels and output paths reconcile. Parameter, dual, optimizer, unregistered config changes, invalid artifact paths, transcript content, task outcomes/identities, costs, task reward, budget counters, candidate counters, decision counters, actions, probabilities, and likelihood changes must fail. These checks write no environment output, optimizer update, or checkpoint.

R2 uses distinct identity, gate, regression, ready, and resume-progress paths. The resume-progress receipt identifies the R2 mechanical version while retaining the unchanged state-bridge version. It references the original R1 identity and failure hashes and still accepts only the R1 whitelist's exact seed11 ep4 checkpoint (SHA256 `42a5160ead497ef3ce5dcb2e9783fe094b30f6a1f245630149bdde2c75946d25`). The actual first-batch gate and exact four-module/Adam/dual/history/RNG/cursor restore checks remain mandatory before any new claim.

## R2 gate attempt and R3 assertion repair

The sealed R2 read-only gate passed the real saved-batch gate, the R2 comparator, all comparator negative cases, and the R2 comparison receipt. It stopped while checking that the trained R43 checkpoint is rejected: the actual loader correctly raised `training_version mismatch`, while the assertion accepted only `training version mismatch` or `freeze identity mismatch`. The attempt and its hashes are preserved in `runtime/ROUND44_R2_GATE_ATTEMPT_FAILURE.json`; R2's exact-restore and ready receipts remain absent. The ep4 live-state whitelist still passes, and the original ledger, progress, journals, checkpoint, stop report, and R1 identity retain their pre-resume hashes.

R3 changes only the expected-error matcher to recognize the loader's exact field-name spelling, while still rejecting unrelated errors. It has a new execution identity and separate exact-resume, comparator, ready, and progress paths. The R2 code identity, comparator receipt, comparison receipt, and failed attempt are verified as immutable inputs. The exact ep4 restore gate must pass under R3 before the fixed continuation can claim another environment.

## R3 gate attempt and R4 frozen-order assertion repair

The R3 gate passed the saved-batch/comparator checks and restored the approved ep4 checkpoint in memory through module, full Adam state, cursor/dual history, RNG state/next values, and complete-order validation. It then hit a stale literal requiring scene `1300014` at episode 4. The frozen 128-scene order assigns `1300014` to episode 1 and `1300003` to episode 4; the actual runtime order had already matched the full frozen list. The failed attempt is preserved at `runtime/ROUND44_R3_GATE_ATTEMPT_FAILURE.json`, and R3's exact-resume/ready receipts were not written.

R4 removes that stale literal and derives the next scene from `ROUND44_FROZEN_CONFIG.json` after verifying the whole runtime order equals the frozen 128-scene list. A pure regression checks the frozen-order helper. R4 has separate identity/gate/ready/progress paths and binds the R3 identity plus failure receipt. The source checkpoint and scientific training identity remain unchanged.
