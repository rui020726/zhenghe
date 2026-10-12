# ROUND64 Approved Cap-13 Completion

## Decision and Run Identity

The explicit approval from decision thread `01a0a044-592f-73e3-838c-19c1d79465d3` (question item `call_LriqpEPUNvxCAaSbOwt2djZb`, index 0) authorized a cumulative ceiling of 13 attempts to finish only the original four unresolved cells. The original 12-attempt freeze remains unchanged. The appended approval record and its SHA are in `ROUND64_CAP13_APPROVAL.json`.

The `/8` executor, guard, model identity, approval, receipt, and 10-normal/3-reserve cap were bound before the next builder and committed at source HEAD `50fee3236f8fd093598cd9862b52d364dc600645`. The read-only prelaunch binding used the one frozen R64 raw root and the R60 verified seed-33 ep128 greedy checkpoint (file SHA256 `573e4c26a4fee6f01405d4386ca73b518eeb1adb9cd8409bc83661e4113f6c5a`; parameter SHA256 `bce71a0e9810122c146950f12cf818a9a359406312d96dfc9c7a457ebb654e2a`). Optimizer, training RNG, and cursor were not restored. The old strict gate and old reconciliation remain unchanged; `/8` writes to its own `ROUND64_GATE_RECONCILIATION_v8.json`.

The prelaunch plan check found 9 attempts consumed, 3 normal slots and 1 reserve slot available, and exactly 4 fixed cells required. A SHA-bound post-hoc receipt authorizes only `d+30_FORCE_SEND` on `reserve-03`; it preserves normal-07's original nonterminal ledger prefix and does not invent a historical return event. Prelaunch zero-environment acceptance is recorded in `ROUND64_CAP13_PREFLIGHT.json`.

## Fixed Matrix

The model's first legal, payable decision remained at step 134. Its original greedy action was SKIP (`P(SEND)=0.1937039047`, `P(SKIP)=0.8062961102`). SEND and SKIP arms had matching pre-action prefixes within each deadline pair. The same 19D float32 vector was observed in all eight matrix runs:

`[0,0,0,1,1,1,0,0,0,0,1,1,0,1,1,1,1,6,0.134]`

| Deadline delta | Deadline slack at decision | SEND result | SKIP result | SEND minus SKIP |
|---:|---:|---|---|---|
| -30 | 68.85 steps | 6 on time; targets 0-5; 7,500 B; target-5 completed at step 156 | 5 on time; targets 0-4; 4,812 B; target-5 expired at step 202 | +target-5, +1 task, +2,688 B |
| 0 | 98.85 steps | 6 on time; targets 0-5; 7,500 B; target-5 completed at step 156 | 5 on time; targets 0-4; 4,812 B; target-5 expired at step 232 | +target-5, +1 task, +2,688 B |
| +30 | 128.85 steps | 6 on time; targets 0-5; 7,482 B; target-5 completed at step 156 | 5 on time; targets 0-4; 4,812 B; target-5 expired at step 262 | +target-5, +1 task, +2,670 B |
| +60 | 158.85 steps | 6 on time; targets 0-5; 7,482 B; target-5 completed at step 156 | 5 on time; targets 0-4; 4,812 B; target-5 expired at step 292 | +target-5, +1 task, +2,670 B |

Across each pair, the full on-time ID-set difference was exactly `{piloc-e1-target-5}` gained under SEND, with no other target lost. SEND emitted target-5 once at step 134; its quote was 2,688 B at -30/0 and 2,670 B at +30/+60. The recorded later emissions for targets 1 and 3 were unchanged. SEND transcript entries include accepted deliveries, and the task chain completed; SKIP left the target detected until deadline expiry. Acceptance alone is not treated as task benefit. The separate paired JSON retains complete per-arm emissions, transcript and lifecycle chains.

The actor vector was identical despite the 90-step span in actual slack; no conflicting SEND/SKIP effect was observed among these four deadline conditions. This is not evidence that 19D is sufficient or that one can generally infer safety from this state. The fixed-world deadline change also changes feasibility and scheduling. One stochastic implementation per arm was run; downstream packet draws are not asserted to be identical after actions diverge.

## Cumulative Accounting

All 13 authorized attempts were consumed: 10 normal and 3 reserve; no headroom remains. There were 13 builder calls, 13 environment constructions, and 13 JointRun entries. Twelve runs returned 1,000 steps (12,000 known returned steps); one historical run remains of unknown length. Ten logical cells are qualified: L, L0, and all eight fixed matrix cells.

The two earlier validation failures remain counted: normal-01 returned 1,000 steps but failed its model-load metadata check; reserve-01 returned 1,000 steps but its row was not persisted. Normal-07 remains the historical interrupted attempt: its last observed lifecycle step is 916, but no terminal result or completed-step count exists. Its qualified reserve-03 retry returned 1,000 steps. No additional technical failure occurred. No attempt was deduplicated or refunded, and no fourteenth attempt was made.

Training side effects were zero: no optimizer/PPO updates, training RNG or cursor restore, training resume, or checkpoint writes. The frozen model file and parameters remained unchanged. Four read-only model-only loads were used across preflight and CLI setup; normal policy inference during the four authorized episodes is part of those runs.

## Interpretation and Artifacts

Within this one frozen model, scene and stochastic implementation, forcing SEND at target-5's legal free decision changed its outcome from deadline expiry to on-time completion at every tested deadline, at the listed measured byte costs. This does not establish general message necessity, safe skipping, population-level effect, noninferiority, or deployment value. The results are descriptive paired interventions, not significance tests.

`ROUND64_PAIRED_ANALYSIS_CAP13.json` contains all four pair records and the exact input/slack audit. `ROUND64_CAP13_FINAL_VERDICT.json` contains the machine result and cumulative counts. `ROUND64_CAP13_ORIGINALS_MANIFEST.json` lists Git-external raw originals and hashes. The earlier stopped verdict, partial analysis, original 12-cap freeze, old gates, and all previous raw files remain preserved.
