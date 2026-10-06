# ROUND45 seed 11 checkpoint serialization failure

## Frozen execution identity

- Branch: `exp/comm-rule-warmstart-cost-learning`
- Frozen training source commit: `6367afbb9fb19bde1a61f19634b70955dce77d70`
- Pre-BC freeze commit: `c6aa274c7d7719a75824a5088bc4af2e3142a6ea`
- Frozen `r45_bc.py` SHA256: `04ff2b2cffaefc5fb15928d2e785079aa23539fc7d60194fba7c639792d3c398`
- Dataset SHA256: `93ffba45bfa484aa0f0aa5cbb334bd76c1284d0b21e7ca704b459ae6126f378d`

## Failure

Seed 11 completed all 1,000 full-batch BC optimizer calls. The step journal contains 1,000 `optimizer_step_invoked` and 1,000 `optimizer_step_returned` records, ending at step 1,000. The metrics file was flushed. The process then failed while preparing the post-BC checkpoint payload:

```text
Traceback (most recent call last):
  File "D:\project_env\lib\runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "D:\project_env\lib\runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "...\r45_bc.py", line 541, in <module>
    raise SystemExit(main())
  File "...\r45_bc.py", line 524, in main
    entries.append(train_seed(seed, records, dataset_sha256=dataset_hash,
  File "...\r45_bc.py", line 462, in train_seed
    "controller": {name: value.detach().cpu().clone()
  File "...\r45_bc.py", line 462, in <dictcomp>
    "controller": {name: value.detach().cpu().clone()
AttributeError: 'collections.OrderedDict' object has no attribute 'detach'
```

`StageCController.state_dict()` returns a mapping of module names to nested `OrderedDict` state dictionaries. The frozen serializer treated each module state dictionary as a tensor. The exception occurred after metrics were written but before `BC_POST_MODEL_s11.pt` was opened. There is no post-BC checkpoint or recoverable controller/optimizer state.

Preserved artifacts:

- `BC_METRICS_s11.json`, SHA256 `8ba01d9719c7178d032e2261535796d8797cc07acfd2c2c92d2d2165e6d74894`
- `bc_journals/BC_OPTIMIZER_STEPS_s11.jsonl`, SHA256 `754f3828e768ac4779904415d726ec7bde08be579f63d3dad5206ffffef27341`
- `ROUND45_PRE_BC_FREEZE.json`, SHA256 `4736b9ef893bcb5eed6e97946950266b8731c10aa039150f14503251a33f36b1`

The metrics preserve the initial parameter digest `7eff0f40203fdae63437bb07a16204ab0289bf938e9c263f781b3adfc40d5eff` and final digest `37303c494cd997894fba1441cf9d2cde86cf2135e71f443a4ed1bcb9ec7f0330`. A digest is not model state and cannot restore the missing checkpoint. Task and cost critic head parameter digests remained unchanged during the completed BC updates.

## Accounting and recovery boundary

- Real BC optimizer calls returned: 1,000 for seed 11.
- Synthetic optimizer calls previously used: 2 (within the 64-call cap).
- Seeds 22 and 33: no BC steps started.
- Post-BC checkpoint writes: 0.
- PPO updates/optimizer calls: 0.
- Environment constructions, run calls, completed episodes, and claims: 0.

The execution basis requires stopping when complete model/optimizer state cannot be proven recoverable, instead of repeating committed updates. No update was replayed. The missing seed 11 checkpoint blocks all 48 BC-only controls, all PPO training, and final evaluation.

## Post-failure serializer repair regression

After preserving the frozen source and failure artifacts, a zero-environment/zero-optimizer repair was added as `round45-bc-nested-controller-state/2`. The production payload builder, writer, reader, and controller loader now preserve the unchanged nested schema. A BytesIO regression with a fresh canonical four-module controller reproduces the old `AttributeError`, performs a `torch.save/load` round-trip, checks all 10 tensors and all four module hashes, and verifies actor/task/cost forward outputs are identical. A complete six-parameter BC Adam state fixture retains `state`, `param_groups`, `step`, `exp_avg`, and `exp_avg_sq`; it is manually populated without calling `optimizer.step()`.

Regression receipt: `ROUND45_CHECKPOINT_SERIALIZATION_REGRESSION_R2.json`, status `passed`, optimizer calls `0`, environments `0`; transport was `io.BytesIO` and no checkpoint file was created. The fixture is labeled synthetic and is not a real BC checkpoint. The earlier, narrower `ROUND45_CHECKPOINT_SERIALIZATION_REGRESSION.json` is preserved. This repair was not part of the frozen run and was not used to resume or rerun BC.
