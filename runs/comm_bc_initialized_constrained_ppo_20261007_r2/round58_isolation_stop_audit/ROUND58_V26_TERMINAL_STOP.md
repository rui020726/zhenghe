# ROUND58 V26 terminal stop (2026-10-10)

**Status: stopped on a deterministic private recovery-contract error. The fixed plan is not complete. Do not relaunch V26.** Frozen execution HEAD: `864bad654ce1b9cc11fe04079887f00540bf0245`. The controlled supervisor launched one real CLI child (PID 72520), which exited 1; supervisor PID 65324 also exited. Original stdout/stderr, state, checkpoints, sidecars and journals remain in place.

## Exact failure and call order

The real R40 `train_seed` selected the approved V25 b14 checkpoint (`4c4a8c7115a0bcc08cf58d5912d97bd0605d4bf38a3654b0768c57482e82694f`), loaded it, and copied it to the V26 rolling path (`r40_train_resume_adapter.py:352–363`). It reported seed11 cursor episode60/update15/lambda1.0. Before running an episode it saved `before_b15` (`:407–423`). The V26 prefix wrapper then **called `_latest_resume_source` again** (`r58_private_launcher_repaired.py:749–762`), after the V26 rolling copy existed. That selector chose the local V26 rolling checkpoint, for which no *local* exact b14 commit exists; `r58_checkpoint_contract_repaired.py:403` rejected it: `R58 selected current checkpoint lacks its exact batch commit`. Thus this is a source-selection timing defect, not evidence that the approved V25 b14 source lost its commit. No prefix was delivered and no episode provider was called in this V26 attempt.

The V26 copied checkpoint and before-b15 checkpoint both hash `75fdee2865347c05835d6184dcdc0e33421655e69d0ed2e3bfcec58e1464c9b4`; their sidecars and the two authentic checkpoint-write journal events are preserved. These V26 files are **not an eligible committed continuation source**. Do not infer that the V26 rolling copy itself has a valid b14 commit, and do not patch or retry this frozen identity.

## Actual ledger at stop

The independent before/after raw-claim audit in `ROUND58_V26_CONTROLLED_SUPERVISOR_STATE.json` is unchanged: gen3 73 claims/slots, 70 complete logical jobs, three failed technical attempts, 70 normal slots and three reserve slots; historic consumption 16; cumulative attempt slots 89/1064; cumulative actual constructor/run calls 87; 954 uncompleted logical jobs (314 training, 640 evaluation). This V26 attempt added **0** claim, environment construction, run, optimizer update, PPO call, batch commit or evaluation cell; it added **2 checkpoint payload writes** and two write-journal entries, followed by exit 1. Do not count the two uncommitted e60/e61 as a committed PPO batch. Last eligible committed seed11 source remains V25 b14 at episode60/update15/Adam88; seed22/33 retain their qualified batch0 source at episode4/update1/Adam8. No V26 model is eligible for results.

The pre-provider V26 gate passed, but its derived-ledger write spy was not invoked, as previously disclosed in `ROUND58_V26_GATE_LEDGER_CORRECTION.md`; it did not cover this actual call-order failure. The 18 narrow tests passed before launch; they are not evidence that the failed real continuation completed.

## Disposition

Freeze V26 as failed. A future separately authorized execution identity would have to bind the already validated external source *before* R40 copies it into the new output root, then pass that exact source to the prefix validator rather than reselecting from the mutated root. This is a proposed repair location, **not** an implemented or approved V27. The scientific configuration is unchanged; no effect estimate or final verdict exists. The user-mandated stop-on-core-error rule takes precedence over the prior instruction to continue on a valid gate.

Protected ROUND46 file SHA256 values were rechecked unchanged: `cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b` and `93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18`. No protected file was staged. No push, merge or deployment.
