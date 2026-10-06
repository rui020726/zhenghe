# ROUND46-R2 runner and output-routing correction

## Scope

This correction is confined to the ROUND46 adapter and its zero-environment gates. The training semantic version, BC sources, model dimensions, optimizer, RNG source, scenario order, reward/cost math, quota, and fixed key plan remain unchanged. Runner identity remains `/3` until the corrected source is committed and frozen.

The earlier uncommitted `/3` prepare snapshot (before the journal-route correction) is preserved under `attempts/runner_v3_prejournal_refreeze/`; its archive manifest SHA256 is `662e4a9fd531a7b8559063b4f34a4e0830fe618cf5e4f497bb0225dc0d1ef075`. It was prepared at HEAD `1566e09f067a22e8e2822a063aa1c9c4971140c9`, before formal preflight, and had zero environments/claims/updates. The final `/3` freeze will be generated again from the committed corrected source.

## Corrected seams

### Git porcelain status

`_run_git()` previously applied `.strip()` to `git status --porcelain`, removing the first XY status column's leading space on the first output row. `_unexpected_source_status_lines()` therefore could misclassify the first allowed unstaged task-state path. The adapter now strips only trailing CR/LF for porcelain status and retains its prior trimming behavior for branch, HEAD, and other Git queries.

The allowlist remains exactly the two unstaged paths `CURRENT_TASK.json` and `ROUND46_CURRENT_TASK.json`. The status regression calls the real task-state updater, captures `repr()` of actual `git status --porcelain --untracked-files=no`, runs the real `verify_prelaunch_identity` hook, then changes the runner source and frozen config on disk one at a time and requires each to be rejected by the actual Git status check. All temporary bytes are restored and hashed/count-checked.

### Checkpoint journal output

ROUND45 loads its read-only source artifacts while configuring R42, and the actual R42 `journaled_save_checkpoint` wrapper captured ROUND45's `runtime/checkpoint_writes.jsonl` path in a closure. Changing only the adapter's `RUN_DIR` left that writer and `r42_run.RUN_DIR` pointed at ROUND45. ROUND46 now rebinds the real wrapper's `checkpoint_journal` closure cell and the R42 ledger output root to the isolated R2 directory before a new checkpoint save can occur. It does not edit the frozen R42/R45 source.

The accidentally misrouted ROUND46 JSONL is preserved at its original path, SHA256 `16d8f73c6606d44a4d5ebaf4b2c43d5976c8dc7f76ec4e0859b0974d14d0d8bd`. The exact bytes are also copied to the R2 runtime directory; the audit records each entry and matches destination paths and payload digests to the archived attempt or its fixture manifest. ROUND45's stop ledger records zero checkpoint writes. The 15 historical writes are:

| Entries | Runner identity | Classification | Evidence |
|---:|---|---|---|
| 6 | ROUND46-R2 runner `/2` | Synthetic readback fixtures | v2 fixture manifest paths and payload SHA256 values |
| 2 | ROUND46-R2 runner `/2` | Untrained ep0/before-batch0 saves before the v2 pre-claim stop | Archived stop and checkpoint payload SHA256 |
| 1 | ROUND46-R2 runner `/3` | Partial synthetic seed-11 fixture from the failed v3 development probe | Journal destination and preserved fixture payload SHA256 |
| 6 | ROUND46-R2 runner `/3` | Synthetic readback fixtures | runner_3 manifest paths and payload SHA256 values |

All 15 entries are ep0, have zero optimizer updates, and are not BC updates. None is removed or subtracted from the ledger. A new zero-environment route regression uses the actual `trainer.save_checkpoint` call once, appends one separately classified synthetic write to the R2 journal, and proves the ROUND45 journal hash remains unchanged. It performs no claim, environment run, PPO update, or Adam step.

## Verification state

Using `D:\project_env\python.exe` (Python 3.10.19; PyTorch 2.9.0+cu128; runtime device forced to CPU), the current source compiled and a read-only runtime configuration probe confirmed:

- R42 ledger `RUN_DIR` resolves to the R2 output;
- the actual save-wrapper closure resolves to `R2/runtime/checkpoint_writes.jsonl`;
- historical checkpoint calls and journal entries both equal 15;
- claims, environment constructors, run invocations, PPO updates, and optimizer steps remain zero;
- the old misrouted journal and R2 copy did not change during runtime configuration;
- the actual porcelain output retains its leading space.

The actual synthetic save route regression, final ROUND46 prepare/freeze, formal checkpoint/readback preflight, and real status-update-to-preclaim regression are still required before the fixed plan may enter its three seed first-four gates. No scientific execution has started.
