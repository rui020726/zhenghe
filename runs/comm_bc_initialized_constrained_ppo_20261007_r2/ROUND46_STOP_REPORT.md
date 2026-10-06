# ROUND46-R2/3 formal-entry stop report

## Outcome

The frozen ``--execute`` entry was invoked once from ``D:\code\整合\zhenghe-round42`` on branch ``fix/comm-bc-initialization-readback`` at HEAD ``730679cf86330e64fff92c0638335aeb4d222156``. It exited 1 before the first atomic logical-task claim. The inner call path reached R44 ``claim`` and the R45 ``verify_prelaunch_identity`` at ``runs/comm_rule_warmstart_cost_learning_20261006_r2/r45_r2_execute_r3.py:268``, which raised ``RuntimeError: R2 prelaunch/freeze status is not passed``. The outer ROUND46 prelaunch and status-guard receipts had passed; they did not establish that this inner verifier was correctly bound for the ROUND46 execution.

No runner or scientific code was changed after this failure. The entry was not retried. The existing ROUND45 cost-gate failure and source records remain unchanged.

## Counts and saved state

- Normal/reserve claims: 0 / 0; environment constructors: 0; ``run`` calls: 0; completed steps: 0.
- PPO updates: 0; optimizer steps: 0.
- Checkpoint-write journal: cumulative count 18, up from 16 before this entry. The two additional save calls occurred before claim and have ``episodes_done=0`` / ``updates=0``.
- Three seed11 files are present: ``v2_train_s11_ep0.pt``, ``v2_train_s11_before_b0.pt``, and ``v2_train_s11.pt``. All are byte-identical, SHA256 ``330a80eb646bf422fbe31024951cc88c15740cc97579c903b0686a78107f97ed``. They contain no committed training update and are preserved as failed-entry artifacts; do not use them as a trained continuation.
- The only per-seed progress record is seed11 episode 0 / update 0 / optimizer steps 0. Seed22 and seed33 progress files are absent; their individual progress is not inferred from that missing data. The strict execution ledger records no claim or update.

## Identity and evidence

- Runner ``r46_execute_r2.py`` SHA256: ``035d49799ebc3ce21eb6230fd36b6163af8bed7223b403e41b67c1d6712561f2``.
- Frozen config SHA256: ``eed24787261055d1cb9727a8f8d86924ae222050be458331009013f9308498ae``.
- Prelaunch receipt SHA256: ``ae5cc182976c810cf6d30cf1a387d3f064e83bd6f4e19820c08113d04d212ca5``.
- Status-guard regression receipt SHA256: ``689b6352bab0fe78cb69a06ce5544ba77fe753f03bd69c97b4b552bd59f6c64a``.
- Stop state: ``ROUND46_STOP_STATE.json``; execution failure: ``ROUND46_EXECUTION_FAILURE.json``; traceback transcription: ``ROUND46_EXECUTION_FAILURE_TRACEBACK.txt``.
- Raw checkpoint files and the runtime checkpoint journal remain local and are not added to Git. The failure JSON records their paths and hashes.

## Stop boundary

This is a preclaim wiring failure. No environment quota was claimed, and no training update was submitted. The fixed plan remains incomplete. Continue only after the specifically authorized R46 adapter binding correction, a new /4 execution identity and zero-environment seam regression; do not re-enter this /3 freeze.
