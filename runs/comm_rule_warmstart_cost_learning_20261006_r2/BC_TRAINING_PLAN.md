# ROUND45 BC training freeze

This file records the fixed offline initialization method before any real BC update.

- Task: ROUND45 rule-labeled offline BC initialization followed by the inherited R44 constrained PPO protocol.
- Teacher: registered canonical `late_discovery_gate_147`; it is used only to produce offline labels.
- Dataset: the hash-verified `BC_DATASET.jsonl` and `BC_DATA_AUDIT_R2.json`; fit scenes are 1300000–1300031 and validation scenes are 1300032–1300047. Development scenes are excluded from BC.
- Per seed: construct the canonical R44 fresh controller for seed 11, 22, or 33 and require exact equality with its R44 ep0 parameter digest. Train each controller on the complete shared 1,548-row fit split for exactly 1,000 full-batch updates.
- Objective: weighted two-class cross entropy, with inverse-frequency class weights calculated once from the fit split; Adam learning rate 0.001; global gradient clip 0.5; no weight decay, entropy term, shuffle, early stop, model selection, or validation-driven tuning.
- Optimizer scope: actual shared trunk plus action head only, each parameter object once. Task and cost critic head parameters are excluded and must remain byte-identical. Shared trunk updates can change critic outputs.
- Fit/validation agreement diagnostics use the frozen `OneShotSendSkipCollector` greedy decode, including its exact float32 softmax and `p_SEND >= p_HOLD` SEND-on-tie rule.
- Device: CPU. Installed experimental runtime is the R44 registered `D:\project_env\python.exe`; Torch version and optimizer defaults are frozen in `ROUND45_PRE_BC_FREEZE.json` before training.
- Randomness: no minibatch sampling or augmentation; the same fit row order is used in every update. Model construction follows the canonical `fresh_controller(seed)`. Ambient Python/NumPy/Torch RNG states are restored after offline BC.
- Counting: 3,000 real supervised Adam steps total. Synthetic update tests are counted separately and must remain at or below 64. No environment is constructed in BC training or its synthetic tests.
- Recovery: a seed with any existing BC step journal, metrics, or model checkpoint is not trained again automatically. An incomplete seed is preserved and treated as an execution interruption requiring reconciliation.
- ROUND45-R attempt 2 starts all three seeds fresh; the previous seed 11 attempt's 1,000 updates are permanently counted and never resumed or replayed. Before any R2 update, a dedicated synthetic fixture must exercise the repaired production save/read path on a real file path and verify all four modules, actor/task/cost forwards, the full six-parameter BC Adam state structure, and RNG preservation. The fixture is not an experiment checkpoint and uses zero optimizer steps.

The final BC initialization is the parameter state after update 1,000 only. Validation agreement is diagnostic and never selects a checkpoint or changes hyperparameters.
