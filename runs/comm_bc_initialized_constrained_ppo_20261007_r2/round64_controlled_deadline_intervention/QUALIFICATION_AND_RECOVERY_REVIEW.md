# ROUND64 Qualification and Recovery Review

## Scope and Decision

This is a read-only reconciliation of the existing ROUND64 execution. No episode, environment, claim, training update, optimizer call, RNG restore, or checkpoint write was performed. Existing reports, gates, ledger rows, and raw artifacts are unchanged.

The `/6` executor versus `/4` cumulative-guard label mismatch is real, but source-history review shows it is a bookkeeping/version-label deviation, not a change to the experiment's policy, world, model, or intervention path. The four completed matrix cells at deadline deltas -30 and 0 are therefore technically qualified for descriptive paired interpretation. The original strict L/L0 gate remains failed; its separate exact-path reconciliation remains separate and unchanged.

This qualification does not complete the fixed matrix. The d+30 FORCE_SEND attempt is nonterminal with unknown total steps; d+30 FORCE_SKIP and both d+60 cells were not launched. The original plan cannot be completed within existing headroom: three normal slots remain, the two reserve slots are spent, and four logical cells need complete runs. No further run is authorized by this review.

## Source Identity

All four completed matrix cells were launched from prelaunch commit `16cf1f92c308082112d0b9e39035e18450566443` (executor/model identity `/6`). That commit's relevant source hashes are:

| Source | SHA256 |
|---|---|
| `r64_execute.py` | `1545ba7ca7a9159c76d904abbc4e06ffe8c068a42027b97f26f303c0749ec519` |
| `r64_guard.py` | `c50d0facf95e0fa745033430283c4bad653e92e5bf5743866eb090b1103b8acd` |
| `r64_runtime.py` | `3ae654d240ed97ddf5296299f969849c9e0927ad08d2b41b4fcb36ce915bc099` |
| `r64_intervention.py` | `eb77f35d9bea539a183bbd041c20e2a5d4156d32ad649650a2b4742d7a0d8c46` |

Inherited strategy/execution files at that commit were also unchanged: R61 executor `3258e0f9d5d5aa7734d857c5b992210b2591be3fdf232df262a89503a4e2f74b`, R60 executor `470f8fc6322945b0b7ce3a475f98e66fdcdcfd0a5e30f6fbda3f612929affb28`, R60 partitioned model integration `faf4a9402b5c98a1bfca1bbe67fca1c387c7255f827f8ed5e0b1a6911f25b137`, and R58 executor `03bdc3ad047360fe232380b436d1711eee8fbb0aacdfbf384367d57e50664d12`.

Comparing the preceding `/4` launch commit `581c7c84652cb587a678ca7758c88bf64b1d44dd` to `/6` shows the scientific run/intervention path did not change. `r64_execute.py` changed its version string, added explicit normalization of the final embedded transcript output path, and added a separate gate-reconciliation function; the matrix precheck now reads that reconciliation. `r64_guard.py` only expanded its accepted-version set, while its active `EXECUTION_VERSION` remained `/4`, causing the ledger label mismatch. `r64_runtime.py`, `r64_intervention.py`, and the inherited R61/R60/R58 execution files were byte-identical. The changed code does not alter the frozen model, action override location/semantics, send cost, sequential budget mask, world, RNG path, or run call.

Each of the four cell records independently binds the same read-only model file SHA256 `573e4c26a4fee6f01405d4386ca73b518eeb1adb9cd8409bc83661e4113f6c5a` and parameter SHA256 `bce71a0e9810122c146950f12cf818a9a359406312d96dfc9c7a457ebb654e2a`; optimizer, training RNG, and training cursor were not restored. Each has matching post-run parameter SHA, 1000 returned steps, one environment construction and run return, and the intervention at target-5's first legal/affordable decision at step 134. The baseline pre-action prefixes within both pairs match.

## Qualified Paired Observations

At both completed deadlines, the frozen model's original greedy decision was SKIP (`P(SEND)=0.1937039047`, `P(SKIP)=0.8062961102`), with a legal 2,688 B quote and sufficient sequential budget. The same 19D float32 vector was recorded across all four cells.

| Deadline offset | FORCE_SKIP | FORCE_SEND | SEND minus SKIP |
|---|---|---|---|
| -30 | 5 on-time: targets 0-4; 4,812 B | 6 on-time: targets 0-5; 7,500 B | target-5 gained; no target lost; +1 task; +2,688 B |
| 0 | 5 on-time: targets 0-4; 4,812 B | 6 on-time: targets 0-5; 7,500 B | target-5 gained; no target lost; +1 task; +2,688 B |

In both SEND runs, target-5 was emitted once at step 134 for 2,688 B and was on time; the other emissions were unchanged in the recorded sequence. In both SKIP runs target-5 expired and was not emitted. These are two paired outcomes from one frozen model, one scene seed, and one implementation of the stochastic world. They are not independent replications, do not prove general necessity or safe skipping, and do not establish 19D feature sufficiency or deployment value.

## Consumption and Recovery

The preserved cumulative ledger contains 9 distinct consumed slots: 7 normal and 2 reserve. Builder calls: 9; environment constructions: 9; run entries: 9; returned runs: 8, each 1,000 steps (8,000 known steps). The ninth run entry is normal-07 d+30 FORCE_SEND. Its lifecycle log reaches timestamp/step 916, but there is no row, completed tape/transcript set, `run_returned`, or terminal ledger event. Its actual final step count and result remain unknown; 916 is only the last observed lifecycle timestamp, not a completed-step count.

Two earlier attempts consumed slots and returned 1,000 steps but were stopped at result validation: normal-01 (load-mode metadata label) and reserve-01 (missing persisted row). They remain technical failures and were not deduplicated or refunded. Six cells are complete and qualified: baseline L/L0 and the four matrix cells above. One started cell is nonterminal; three planned cells were never started. Attempt headroom is 3/12, all in normal slots; reserve headroom is 0. Completing the original four remaining cells would require four new attempts, so the fixed matrix needs a new explicit authorization/ceiling before any continuation.

Training side effects are all zero: optimizer updates 0, PPO replays 0, training RNG restores 0, checkpoint writes 0. This review used zero environment constructions, claims, runs, model loads/forwards, and writes to model/checkpoint artifacts. The read-only checks and narrow tests are recorded in the machine result.

The two protected ROUND46 files still match their required SHA256 values. The pre-existing untracked R59 log was left untouched. No historical output was rewritten.
