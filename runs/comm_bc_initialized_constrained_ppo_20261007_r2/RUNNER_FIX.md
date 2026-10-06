# ROUND46-R2 runner and output-routing correction

## Scope

This correction is confined to the ROUND46 adapter and its zero-environment gates. The training semantic version, BC sources, model dimensions, optimizer, RNG source, scenario order, reward/cost math, quota, and fixed key plan remain unchanged. The corrected runner is `/3`; the prepared freeze records source SHA256 `035d49799ebc3ce21eb6230fd36b6163af8bed7223b403e41b67c1d6712561f2` at prepare HEAD `7e149742a7805db75cf588716cbfbf1ae2202772`. The freeze commit is the next local commit.

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

The final prepared `ROUND46_CONFIGURATION_DIFF.json` matches the preserved pre-journal prepare on `training_version`, `scientific_changes`, `unchanged`, and `counts`; only execution identities/input audit hashes were refreshed. It still describes the sole scientific change as the same-seed ROUND45 four-module BC initialization. Prepare reported 0 environment constructions, 0 PPO updates, and 0 optimizer steps.

## Final freeze and preflight receipts

The corrected `/3` freeze was committed at `8040af95ff0fe0164bf6a334d4e68780b5b748d7`. Formal preflight passed from that commit. It exercised the actual R42 `trainer.save_checkpoint` wrapper once with a synthetic ep0 controller: checkpoint SHA256 `b7365848a2e8dfac2912e2966cf7a2ed68262befa94b76799688dc073ff622f7`, one new R2 journal entry (15 → 16), with the prior ROUND45 journal still at SHA256 `16d8f73c6606d44a4d5ebaf4b2c43d5976c8dc7f76ec4e0859b0974d14d0d8bd`. The route receipt SHA256 is `91de357f855a1ea04f9cfb5383433629aef20c8b363f8b1ca32bc23e12488378`.

The actual checkpoint readback regression also passed: three legacy fresh-load failures and three legacy resume failures were reproduced; fixed fresh and resume paths passed for seeds 11, 22, and 33. Its formal-preflight fixture save delta was zero because the version-3 fixtures were already verified. The cumulative 16 writes close as v2 fixtures 6 + v2 pre-claim saves 2 + partial v3 probe 1 + v3 fixtures 6 + route probe 1; unclassified 0. Preflight ended with claims 0, environment constructors 0, run calls 0, completed steps 0, PPO 0, Adam 0.

The task-state-update → `verify_prelaunch_identity` status regression passed and was committed before any claim. The first frozen `--execute` invocation then stopped at the real inner R44 claim path: `verify_frozen_source` called the R45 `verify_prelaunch_identity` at `r45_r2_execute_r3.py:268`, which raised `R2 prelaunch/freeze status is not passed`. The entry stopped before the atomic claim. Normal/reserve claims, environment constructors, run calls, steps, PPO, and Adam remain zero. The checkpoint-write journal rose from 16 to 18 because seed11 ep0 and before-b0 state were saved before claim; the three resulting ep0/before-b0/rolling files have identical bytes and remain untrained. See `ROUND46_STOP_REPORT.md` and the failure/traceback artifacts. Do not retry the /3 entry or hot-edit its freeze.

## ROUND46-R2 `/4` binding repair

The `/3` failure was confirmed as a stale module-global verifier pointer. The R44 `claim()` resolves `verify_frozen_source` from `r44_runtime.__dict__`; R45 initially assigns its verifier there, while R46 `/3` replaced only the outer R45 module attribute after runtime loading. The `/4` adapter binds its strict verifier to the actual R44 module global and checks that the wrapped R42 claim still reaches that R44 function. R44/R45 production source and claim behavior are unchanged.

The `/4` runner has a separate runner identity and versioned freeze, prelaunch, and stop paths. Its preflight reuses the committed `/3` model-readback, checkpoint-journal-route, and task-state status receipts only after source-function and artifact hashes match; it does not rerun their fixtures or write synthetic checkpoints. A new zero-environment seam regression calls the actual R42 → R45 → R44 claim wrapper and stops at a patched atomic-claim boundary. It must reproduce the old R45-binding rejection, reach the boundary once with the strict R46 binding, and reject source, config, key-plan, and model-identity mismatches before that boundary. The training semantic version remains unchanged.

At this documentation revision, the `/4` seam regression and frozen prelaunch gate are pending. No `/4` environment claim or training update has been made.

The first `/4 --prepare` call stopped while trying to recreate the checkpoint-journal copy: the preserved local journal has 18 entries because `/3` appended its two untrained ep0 saves to the 16-entry historical prefix. No V4 freeze was written and no execution counters changed. The adapter now opens the existing route audit read-only, verifies the 16-entry legacy prefix and the archived 18-entry stop snapshot, and leaves both journal files untouched.
