# recovery_10 mid-training evaluation — 2026-11-01 19:25 local

Status: **MID-TRAIN, NOT CONVERGED, NOT FORMAL VALIDATION.** Training is still running.

## What was evaluated

Two live checkpoints from `recovery_10/revision_05` (the stage that continues only the two
best seeds from R9b toward 10,000 completed episodes):

| Model | Checkpoint | File SHA-256 (prefix) | Progress |
|---|---|---|---|
| T1/91002 | `milestone_6680.pth` (cursor 6,680) | `9d5990924b3b079c…` | 1,776 / 5,000 new episodes |
| T2/91002 | `milestone_6800.pth` (cursor 6,800) | `a3778bae3e2bb760…` | 1,832 / 5,000 new episodes |

Harness: `midtrain_eval_after_recovery_10_20261101/run_midtrain_eval.py`, adapted from the
2026-10-31 final harness (checkpoint pins, R10 identity `fc31f4a4…`, labels,
`requested_worlds=16`). 16 Worlds = 2 models × 8 DEV scenes (2301901–2301908), eval world
seed 20261026, greedy actor, 1,000 steps, 24 tasks, 58,600-byte budget. 16/16 completed,
0 failures, 16 model loads. Decision ref `USER_20261101_SIDECHAT_MIDTRAIN_EVAL`.

Artifacts: `execution/{EVAL_PLAN,progress,SUMMARY}.json`, `execution/rows/`,
`execution/ledger/`, `execution/world_logs/`, `eval_stdout.log`.
`SUMMARY.json` sha256 `a71dcf179399b200f3f9ceeb463d779ee137f29c36e9166450826cb0d583492c`;
`EVAL_PLAN.json` sha256 `cb0b80df7d9e732df80cf0f590b64bbeb195a2519f4e993b04cf23f68e2c5db3`.

## Result

| Model | On-time | Tasks | Success rate | Same 8 scenes at 5,000 | Delta |
|---|---:|---:|---:|---:|---:|
| T1/91002 @6,680 | 96 | 192 | **50.00%** | 82/192 = 42.71% | **+14 tasks** |
| T2/91002 @6,800 | 55 | 192 | **28.65%** | 104/192 = 54.17% | **−49 tasks** |
| **All** | **151** | **384** | **39.32%** | — | — |

Per scene (on-time / 24 tasks), 5,000 → mid-train:

- T1/91002: +5, +1, 0, +3, +1, +3, +1, 0 — broad, small gains on every scene.
- T2/91002: −7, −8, −7, −6, −5, −4, −7, −5 — a uniform, large regression across all scenes.

The T2 drop is uniform across scenes, which is what a policy-level shift looks like rather than
a scene-specific effect. The T1 gain is small and also broad. Both are single-seed readings on
8 scenes (192 tasks per model); no confidence interval was computed and no significance is
claimed. The two arms moved in opposite directions, so the pooled 39.32% is not a meaningful
summary of either arm.

## Registered convergence checks so far (training-scene rows, not eval)

| Run | 5,400 | 5,880 | 6,360 | Last failure cause |
|---|---|---|---|---|
| T1/91002 | F (range 0.0356) | F (0.0199) | F (0.0196) | pair [0,2] CI [−0.0326,−0.0068] outside ±0.025 |
| T2/91002 | **P** (0.0087) | F (0.0141) | F (0.0166) | pair [0,2] CI [−0.0274,−0.0060] outside ±0.025 |

Training-scene window means sit around 0.72–0.74 for both runs, i.e. the checks fail on
window-to-window drift (paired CI), not on level. Two consecutive passes are required, so
neither run is certified yet; a single pass at 5,400 is not convergence.

## Where the training stands

At 2026-11-01 19:25 local (`11:25Z`): 3,613 claims / 3,604 rows of the 10,000-World stage,
31 checkpoints, no `GLOBAL_STOP`, parent PID 46936 plus two workers. Throughput 5.43
claims/min; remaining 6,387 ≈ **19.6 h**, i.e. about **2026-11-02 15:00 local** (`07:00Z`).

## Scope caveats (carry these into any review)

- `MIDTRAIN_DEV_REUSED`: eight previously used DEV scenes, not the sealed holdout; not formal
  validation; `goal_complete=false`; the original NO_GO stands.
- Checkpoints are mid-run, not at the 10,000 target; a later evaluation at the target is the
  only one that can be compared against the 5,000-point result.
- The T2 regression is unexplained. Candidates worth checking later: entropy/behaviour drift
  after 5,000, cost-objective interaction, or evaluation noise. None has been tested.
