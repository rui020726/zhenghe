# Recovery 09b run handoff (2026-10-30)

Status: **G3 PASS, WORLD AUTHORIZATION WRITTEN, TRAINING RUNNING** (3 workers).

This file lives outside the frozen `revision_01` source closure on purpose; the README
inside the revision was pinned before these phases ran.

## What this root is

Sibling rebuild of recovery_09 from the same completed parent (R8), same 18,840-World
plan, same byte-identical seed plan (sha256
`0e3f61aa684aa9ff0faf72581d93ca23b7f78c9d8c2a07e194dc7f6b3c0ed2ef`). R9 was stopped
gracefully and is preserved read-only; see
`../recovery_09/ABANDONED_FOR_R9B.md`.

Single source change: `stage5b_comm/runner.py::_run_train_pool` now counts the gate future
against the real width (`limit = int(workers)`); previously `workers - 1` meant only two of
three workers ran. `WORKERS_MAX` remains 3 and `threads_per_worker` remains 1.

## Frozen artefacts

- Context: `protocol/RECOVERY_RUNTIME_CONTEXT.json` sha256
  `9b903b038378e93622c459709906b8679544b8d7380252bac9f5a2c64c610701`
- G3 report: `execution/recovery_runtime/G3_REPORT.json` sha256
  `8aa9449c7ea76afcf6c7483f808d5d689d93fed829c59c1c3750963cb91acbeb` (status
  PASS_RESTORED_OBSERVED, six runs)
- World authorization: `execution/recovery_world_authorization_01/WORLD_AUTHORIZATION.json`
  sha256 `912d9dd5b9b66ddf6d761daceb000411f9883ba7866c4b2064e0d8c7694fc595`,
  max_worlds 18840, scope `LONG_RECOVERY_STAGE_TO_COMPLETED_EPISODE_TARGET_ONE_ROOT_ONLY`,
  decision_ref `USER_20261030_CHOOSE_PLAN_B_FULL_WIDTH_REBUILD`
- Pure tests: `execution/recovery09b_puretests_run_01` (17 tests, exit 0)
- Context reload: `execution/recovery09b_context_reload_01` (exit 0, zero checkpoint or
  model loads, zero Worlds)

## Training

- Launcher: `execution/recovery_train_run_01/launcher.json` (detached, PID recorded there)
- Gate signal: `execution/ledger/GATE_FIRST_UPDATE.json`
- Measured throughput at 2026-10-30T01:52Z: 8.9 claims/min (three workers), rows written
  for T1/91001, T1/91002 and T1/91003
- Expected: ~35 h of pure throughput plus a solo tail for T2/91003, so roughly 37-38 h,
  finishing 2026-10-31 late evening local time

## Restart boundary

A restart may be planned only from a complete 120-episode checkpoint and requires an
independent audit. Do not replay unresolved claims, rerun consumed Worlds, reset the
optimizer, reseed the RNG streams, or merge post-checkpoint claims into the lineage.
