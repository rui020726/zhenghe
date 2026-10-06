# ROUND45 post-failure checkpoint serialization repair

Repair version: `round45-bc-nested-controller-state/2`

The frozen seed 11 run failed because `StageCController.state_dict()` returns a four-module mapping whose values are nested module `OrderedDict`s. The old comprehension incorrectly called `.detach()` on each outer value.

The minimal repair adds production functions for building, writing, reading, and loading the existing BC checkpoint shape. Controller state stays nested under the same four keys (`trunk`, `action_head`, `value_task_head`, `value_cost_head`); Adam state stays under `bc_optimizer` with its original `state` and `param_groups` structure. The normal BC path now calls these same production functions. No training math, teacher, dataset, optimizer settings, or training version changed.

## Regression

`test_r45_checkpoint_serialization.py` uses a fresh canonical `StageCController` and an in-memory `BytesIO` stream. It reproduces the old comprehension failure, then exercises the actual payload builder, writer, reader, and controller loader. It verifies all 10 parameter tensors, the four module hashes, actor/task/cost forward equality, RNG preservation, and the complete six-parameter BC Adam state fixture including each `step`, `exp_avg`, and `exp_avg_sq` slot. The Adam fixture is populated directly; `optimizer.step()` is never called.

Receipt: `ROUND45_CHECKPOINT_SERIALIZATION_REGRESSION_R2.json` (`passed`). The fixture is explicitly marked synthetic, is never written to disk, and is not a real seed 11 checkpoint. Counts for this regression are zero optimizer steps and zero environments.

The fix was added after the frozen execution failed. It was not used to resume the run and cannot restore the missing real seed 11 weights. The ROUND45 execution remains stopped under the unrecoverable-state rule.
