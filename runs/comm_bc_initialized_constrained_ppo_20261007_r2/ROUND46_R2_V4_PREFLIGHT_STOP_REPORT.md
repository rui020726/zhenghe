# ROUND46-R2 `/4` preflight stop

Status: **stopped before the V4 prelaunch gate and before any claim**.

## Failure

The committed `/4` reuse-only preflight exited 1 in `_require_committed_file` while checking `ROUND46_R2_V3_STOP_ARCHIVE.json`. The helper compared the raw Windows worktree SHA256 (`c66fb42f…`) directly with the Git blob SHA256 (`80438b58…`). The CRLF-normalized worktree hash is `80438b58…`, exactly equal to the committed blob. The archive content is unchanged; the verifier compared two different newline representations as if they were the same hash namespace.

This is a V4 preflight implementation error. The actual R42→R45→R44 claim binding regression did not run, and no V4 prelaunch gate or status-reuse receipt was produced. The `/4` freeze at `774c821d1b7e9786d4450ae36c3aa2c2c4811302` remains bound to runner SHA256 `822e5d1d…`. After capturing the failure, the worktree runner was restored to that frozen source byte-for-byte. It was not patched or rerun under `/4`.

## Counts and preserved stop evidence

- `/4` normal/reserve claims: 0/0.
- `/4` environment constructors, `run()` calls, completed steps, PPO updates, and Adam steps: 0.
- Cumulative checkpoint writes/journal entries remain 18/18 from the earlier `/3` attempt; V4 preflight delta is 0.
- Active checkpoints and progress records: 0. The ledger SHA256 remains `b1f627db…`; the checkpoint journal SHA256 remains `8f68b0ea…`, matching the archived `/3` stop snapshots.
- The ROUND45 failed control gate remains unchanged. No network or push occurred.

The earlier `/4 --prepare` call also stopped before writing a freeze because it tried to recreate the 15-entry historical journal copy after the `/3` stop had appended two untrained ep0 saves. Commit `351eed33e640c3c2edcc5d2345df0958587baa4f` changed prepare to read the existing audit and preserve the 18-entry stop snapshot; the next prepare succeeded with zero execution counts.

## Required handoff

The V4 gate failed, so the authorized automatic continuation conditions were not met. Preserve the V4 freeze and this stop report. Do not patch its runner or invoke `/4 --preflight`, `/4` binding regression, or `--execute` again. Correcting the hash-namespace check requires a separately authorized runner identity and freeze; training remains unstarted.

Full machine-readable counts and hashes are in `ROUND46_R2_V4_PREFLIGHT_FAILURE.json`; the saved stack is in `ROUND46_R2_V4_PREFLIGHT_FAILURE_TRACEBACK.txt`.
