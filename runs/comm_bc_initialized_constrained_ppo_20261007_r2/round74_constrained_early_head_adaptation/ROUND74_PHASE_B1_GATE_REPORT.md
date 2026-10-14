# ROUND74 PHASE B1 — Qualification Gate Report

**Scope:** real model / controller / batch / PPO qualification gates for the ONE R74 structural
change (early_head becomes trainable; shared trunk + late_head + dual critic keep the ROUND71 role).
**Zero environment constructions.** PHASE B (384 training / 192 eval / 3 focal) is NOT started; it
waits for a separate decision-end continuation.

**Control:** ROUND71 `fe53d476efb56baaae194d68d7f749a546b14389` (read-only, never a resume source).
**INIT:** per-seed R71 calibration INIT, model-only (5 modules bit-identical), fresh Adam/lambda=1/RNG.

## Result

Production run exit code **0**, all five gates **PASS**. Actual consumption this run:
- synthetic physical optimizer steps: **3** (counter 2 -> 5; normal cap 12 / reserve 4 / hard 16; remaining normal 7, hard 11; refusals 0)
- offline trunk target rows: **28** (counter 0 -> 28; per-batch cap 1024; hard 6000)
- environment constructions: **0** (env slot counter unchanged at 0)
- R43 training journal: untouched (Adam.step unchanged since import; r42_run / r46_execute_r2 never loaded)

A fixture-mode self-test (throwaway root + throwaway counters, simulated, 0 production charge) ran the
same five gates first and caught two wrong gate ASSERTIONS (not wiring faults), both fixed and
re-validated green before production:
1. the R74 optimizer is ONE Adam group over all 8 trainable objects -> `optimizer_param_groups == [8]`, not `[2,2,2,2]`;
2. torch sets `__wrapped__` on Adam.step natively, so `__wrapped__` is not an r43-hook signal -> replaced by `adam_step_unchanged_since_import` + `r43_training_driver_not_loaded`.

## Gates (all passed)

- **B1-G1 model_identity_and_mask** (seeds 11/22/33): INIT param sha == registered R71 INIT; file sha == registered; mask 8 trainable / 4 frozen / 12 total / 5830 scalars (390 + 5440); early_head trainable, trunk frozen; R74 structure string. 0 steps / 0 rows.
- **B1-G2 routed_forward_boundary**: ready<=147 -> early head, ready>147 -> late head, bit-exact on a mixed out-of-order 8-row batch; the ready=148 rows pin the boundary; offline meter fired and charged 8 rows. 8 rows.
- **B1-G3 gradient_routing_and_critic_isolation**: `critic_input_detached` removes the critic->trunk gradient on a throwaway module (0 metered rows); EARLY sample -> early grad>0 & late grad==0 & both critics>0; LATE sample -> late grad>0 & early grad==0 & both critics>0; actor advantage == lambda*A_task - A_cost. 8 rows.
- **B1-G4 controlled_real_adam_step**: one controlled legal real Adam step (charged) establishes non-empty Adam; trunk digest unchanged; early/late/task/cost digests moved; trunk params grad None (frozen), 8 trainable grads present; optimizer owns exactly 8 trainable objects. 1 step / 4 rows.
- **B1-G5 full_restore_and_resume_contract**: save payload is R74 schema/tag/structure, optimizer single group covers 8, frozen mask 5440; full restore param/Adam/RNG per-value match, lambda==1, state_schema R74; source vs full-restore of the SAME subsequent update -> identical param + Adam; foreign R71 INIT full resume REFUSED (RuntimeError); self R74 accepted. 2 steps / 8 rows.

## Living config captured (not import defaults)

GAMMA=1.0, GAE_LAMBDA=1.0, ETA_LAMBDA=0.5, CLIP=0.2, VALUE_COEFFICIENT=0.5, ENTROPY_BONUS=0.01,
LEARNING_RATE=3e-4, MAX_GRAD_NORM=0.5, EPOCHS_PER_BATCH=4, MINIBATCH_STEPS=256, READY_CUT=147, GATE_BETA=1.0.
(GAE_LAMBDA=1.0 and ETA_LAMBDA=0.5 are the R58/R71 living overrides, NOT the 0.95/0.05 module defaults.)

## Boundary of this result

Gate pass = technical qualification of the wiring only. It does NOT establish policy validity, does
NOT run the full multi-epoch ppo_update (that is PHASE B), and does NOT authorize 384/192/focal.