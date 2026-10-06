# ROUND46-R2 V5 execution stopped before the first claim

## Outcome

The frozen `--execute` command stopped while creating seed 11's initial ep0
checkpoint, before the R42 atomic claim boundary. The V5 zero-environment preflight
and the actual R42 → R45 → R44 claim-binding regression had passed. The three real
first-batch gates did not run.

The error was:

```text
RuntimeError: seed 11: prior ep0 recovery receipt conflicts
```

The call path was `r46_execute_r2.py:execute` → `_run_first_four_and_all_gates` →
R45 `_call_training_seed` → R42 `train_seed` → R40 `_save_milestone(episodes_done=0)` →
R45 `save_milestone` → `_promote_ep0_to_rolling`.

## Cause observed

The R2 runtime already contained
`runtime/ROUND45_R2_EP0_RECOVERY_s11.json`, whose preserved receipt identifies an
ep0 checkpoint SHA256 of
`330a80eb646bf422fbe31024951cc88c15740cc97579c903b0686a78107f97ed`. The new V5
attempt wrote an untrained ep0 checkpoint with SHA256
`2ca8f764315795f5aadc89117901f23442c43c15e4d4802f5ae268765467950a`. The R45
promotion helper found the existing receipt at the same output path and rejected the
different checkpoint identity. The old receipt was not changed.

The new `v2_train_s11_ep0.pt` and `v2_train_s11.pt` are byte-identical, each 57,605
bytes with SHA256 `2ca8f764315795f5aadc89117901f23442c43c15e4d4802f5ae268765467950a`.
They record zero completed episodes and zero updates; their optimizer state is empty.
They are preserved as failure evidence and are **not authorized for resume**.

## Counts

| Measure | Before execute | After stop | Delta |
|---|---:|---:|---:|
| Normal claims | 0 | 0 | 0 |
| Reserve claims | 0 | 0 | 0 |
| Environment constructors | 0 | 0 | 0 |
| `run()` calls | 0 | 0 | 0 |
| Completed environment steps | 0 | 0 | 0 |
| PPO updates | 0 | 0 | 0 |
| Adam steps | 0 | 0 | 0 |
| Checkpoint write calls | 18 | 19 | +1 |
| Checkpoint journal entries | 18 | 19 | +1 |

The one additional checkpoint write is the failed attempt's untrained ep0
milestone. Historical checkpoint writes remain 18; the current cumulative ledger is
19. The 480 normal claims, reserve 24, and hard cap 504 remain unused.

## Preserved evidence

- `ROUND46_R2_V5_STOP_STATE.json` records the stop and actual ledger snapshot.
- `runtime/checkpoint_writes.jsonl` retains the nineteenth entry; it is not rewritten.
- The prior ep0 recovery receipt and the failed attempt's two untrained checkpoint
  files remain untouched after capture.
- ROUND45's failed control gate and the V3/V4 stop artifacts remain unchanged.
- Machine-readable paths and SHA256 values are in
  `ROUND46_R2_V5_EXECUTION_FAILURE.json`.

No environment, training update, or first-batch gate ran. No retry, checkpoint
restore, source hotfix, or alternate execution was attempted. The fixed plan is
stopped pending a separately authorized response to this frozen execution-path
failure.
