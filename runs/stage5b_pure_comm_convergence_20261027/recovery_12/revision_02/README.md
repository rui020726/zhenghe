# revision_02 — unbounded budget, success-constrained byte cost, over (G, R, N)

This revision answers one question: **can a sender be taught to spend fewer bytes while
keeping at least 95 % of the strong rule's success rate, and can one policy do it across
different task counts (G), rescue rosters (R) and subscriber fan-outs (N)?**

Nothing here is a copy of an earlier revision's numbers. A cell is a new configuration, a new
scenario and a new world, so every cell gets its own baselines.

## What is different from revision_01

| | revision_01 | revision_02 |
|---|---|---|
| Budget | 58,600 B, binding | **1,000,000,000 B**, a ceiling only |
| Cost scale | `bytes / 10,000` | **`bytes / (N * G)`** |
| Cost coefficient | fixed 0.1 (T2) or paced | **lambda, a dual variable** |
| Scales | one fixed world | **(G, R, N) sampled per episode** |
| Feature dims | #14/G, #17/6 | **#2/G, #14/G, #17/6** |
| Strong rule | immediate, always_hold | **plus `dedup_immediate`** |
| Episodes | fixed 300 | **no target; breakers stop the loop** |

## Files

| File | Role |
|---|---|
| `stage5b_comm/scale_program.py` | the cell grid, `N <= R`, per-episode cell and seed |
| `stage5b_comm/unlimited_budget.py` | the 1e9 ceiling, `bytes/(N*G)`, the lambda rule, the verdict |
| `stage5b_comm/objective.py` | `A = A_task - lambda * A_cost` |
| `stage5b_comm/scale_features.py` | #2/G, #14/G, #17/6 on the way into the trunk |
| `stage5b_comm/dedup_policy.py` | the low-cost strong rule |
| `stage5b_comm/trainer.py` | the dual step at the update boundary; scaled cost reward |
| `stage5b_comm/world.py` | cell-driven `configuration_fields`, 1e9 authority binding |
| `tests/test_revision_02_gates.py` | 56 offline gate checks |
| `run_baselines.py` | three rules on every cell x scene |
| `train.py` | the training loop with circuit breakers |
| `evaluate.py` | greedy per-cell evaluation and the registered verdict |
| `analyse_checkpoints.py` | recomputes the per-cell tables, the send-set comparison against `normal_immediate`, and the same-target-set recheck straight from the archived transcripts; emits the markdown used in `REPORT_rev02.md` |

## The lambda rule

```
lambda <- clip(lambda + 2 * (q - 0.95 * q_ref), 0.25, 8)      lambda starts at 2.0
```

`q` is the batch's `sum(on_time) / sum(generated_goals)`. `q_ref` is `normal_immediate`'s rate
in the **same cell**. Above target raises the price of a byte, below target lowers it. The step
happens once per PPO update and never inside an episode: an advantage must not depend on a
price set after the step it describes.

A batch usually spans several cells, so `q_ref` is the on-time-weighted mean over the cells
actually present in the batch, and the cells without a reference are named rather than
substituted.

## Failure ledger (every failed attempt is recorded, per the task's "失败也记账")

| # | Failure | Cause | Fix |
|---|---|---|---|
| 1 | Output written to `revision_02/recovery_12/execution/...` | `STAGE = BASE.parent` then `STAGE / "recovery_12" / "revision_02"` resolved one level too deep | `ROOT = BASE`; the mis-resolved tree is kept at `recovery_12/stray_misresolved_path_probe_one/`, not deleted |
| 2 | `FileExistsError` on the second cell of a scene | rule-arm artifact namespace was `phase:arm:none:scene`, which cannot distinguish two cells | cell label folded into the arm slot: `eval:immediate_G24R6N6:none:2301901` |
| 3 | `dedup_immediate` sent 372 times over 24 targets | the view signature included `progress_delta`, which changes almost every tick | stage-structural rule; see #4 |
| 4 | `dedup_immediate` still sent 138 times | `members_changed` fires on every reassignment | default is now `stage_changed` only (`stage_only=True`); measured 53 sends |
| 5 | `test_cost_stream_and_ledger_agree_by_construction` failed | the test divided by `N*G` while the trainer multiplies by `1/(N*G)` | test corrected to the trainer's arithmetic |
| 6 | `test_material_change_resends` failed | the test assumed the broad rule was the default | split into default (stage) and broad (`stage_only=False`) cases |
| 7 | **`grid_v1`: 240 of 360 baseline worlds failed** with `resolved num_goals 12 != N24` | `verify_resolved_configuration` still compared `num_goals` against the frozen literal `C.NUM_GOALS = 24`, so every G=12 and G=48 cell was refused | the check now compares against **the cell's own** `expected["num_goals"]` and additionally asserts that value is in `C.SCALE_GOALS`. The G=24 cells that passed in `grid_v1` are unaffected; the failed cells were re-run as `grid_v2` |
| 8 | **Cost magnitude: `train_v1` sent 0 bytes for 342 of 366 updates** | the registered `c = bytes/(N*G)` makes one send ≈ 33 scale units at `N=6, G=12` against a task advantage of ≈ 0.41, so "never send" was optimal for every λ in `[0.25, 8]` | re-scaled to the reference cell so `(N, G) = (6, 24)` reproduces `bytes/10000` exactly and other cells keep the `N*G` ratio; `probe40` then sent in 40/40 episodes. `train_v1` is kept as the record |
| 9 | **Importance ratio 1.589 at update 0** (also the 11–17 `approx_kl` of `train_v2`) | the trunk wrap is **episode-scoped**: `world` restores it before the PPO update, so the update recomputed log-probs from raw `tape["features"]` while the policy had sampled rewritten rows | the transform became a pure function `scale_features.apply_rewrite`, and `trainer._apply_transform` reproduces each sample's row from its `cell_transform`. Measured `max|dlogp|` 1.091e-01 → **0.000e+00**; trainer `approx_kl` 0.186 → 0.00041. `train_v3` is the seam-fixed run |
| 10 | **Every learned/baseline ratio ≈ 3/8 in the first `eval_100`** | the learned side summed 3 scenes while the baseline side summed all 8, so the comparison was between different denominators | baselines are summed over exactly the scenes the learned worlds ran; `evaluate.py --recompute-from` re-aggregates an archived `EVAL.json` and inherits its scene set. Both outputs are kept (`eval_100_seamfix`, `eval_100_seamfix_recomputed`) |
| 11 | **In-loop evaluation at episode 200 returned 0/15 cells** | the eval artifact namespace had no checkpoint index, so every cell collided with the episode-100 leaf and raised `FileExistsError` | `episode_index` is now part of the namespace (`@<ep>`) and `evaluate_all_cells` **raises** instead of returning a null point, so an unmeasured point cannot pass as data |
| 12 | `_apply_transform` raised `IndexError: 900 out of bounds for size 517` | it used global `_row_start/_row_stop` offsets while the update works per minibatch chunk | rows are computed chunk-relative from `len(item["targets"])`, with a row-count assertion; a stray `torch` reference and an invalid `import numpy as np as _np` were fixed in the same edit |
| 13 | **18 of 1146 updates print `kl=nan`** | `train.py:347` formats with `update.get("approx_kl") or float("nan")`, and `trainer.py:348` returns `round(kl, 6)`, so a genuine `0.0` is falsy and prints as `nan` | **not a non-finite update**: the breaker at `train.py:348-354` reads the raw dict value (`0.0`, finite) and correctly did not fire. Source is left untouched while the run is live; `lambda_trace.json` records these as `approx_kl: null` with `approx_kl_printed_nan: true` instead of inventing a number |
| 14 | **23 checkpoint evaluations exited immediately** with `ValueError: invalid literal for int(): '2301901 2301902 2301903'` | the batch script passed `--scenes` space-separated; `evaluate.py:193` splits on `,` | relaunched with `2301901,2301902,2301903`. The failed attempt wrote no `EVAL.json`; its per-point logs are kept as `execution/eval_v3_<ep>.log` and overwritten by the retry |
| 15 | **Report header mis-stated milestone_1100** as "全部真子集（发送 890/1245）" and milestone_900's 6 differing worlds as all "同一步换一个目标" | the numbers were carried over from milestone_800 (893) instead of recomputed; the 900 breakdown was never split by category | `analyse_checkpoints.py` recomputes both from the transcripts: 1100 is **629/1245**, 31 strict subsets + 14 other (6 new send steps, 8 both); 900 is 4 target swaps + 2 new send steps. The analyser was validated by reproducing milestone_800's recorded 10/35/893 and 2 solid cells exactly before being used on the new points |

The `grid_v1` file is kept: `execution/baselines/grid_v1/BASELINES.json` records 120 successful
worlds and 240 failures. `grid_v2` is the complete grid; the analysis reads `grid_v2` and the
`grid_v1` failures stay on the record rather than being overwritten.

None of these was a pin change: every edit is in `recovery_12/revision_02/`, which is outside
the 98-file manifest.

## Gates before a world is opened

`tests/test_revision_02_gates.py` covers the six the task names, one class each:

* **BudgetGateTests** — 1e9 is finite, all three authorities agree inside the context, restore
  is exact, restore still happens when the body raises, a pre-moved authority is refused.
* **CostReconcileTests** — `bytes/(N*G)` divides once, zero scale is refused, `bytes/on_time`
  is `None` rather than infinity, bundled cost sums to the charged bytes.
* **CreditConservationTests** — the causal re-keying preserves the total; unmatched credit is
  reported, not dropped.
* **LambdaTests** — direction both ways, hold at target, both clamps, non-finite refused,
  missing `q_ref` refused, initial value is 2.0.
* **CellTests** — `N <= R` everywhere, 15 cells, unregistered values refused, seeds depend on
  `(scene, G, R, N, episode)` and reproduce, `rescue_per_species` matches the roster,
  expectation overrides name only what moved.
* **FeaturePathTests** — exactly #2/#14/#17 rewritten and #11 untouched, `off` is the identity,
  the tensor type survives, **all three read paths (sampling, cost critic, PPO update) reach the
  trunk rewritten**, the importance ratio at update 0 is < 1e-6, restore works for both a plain
  class and an `nn.Module`, idempotently.

Plus `DedupPolicyTests` and `VerdictTests` for the strong rule and the literal improvement test.

## Circuit breakers

1. non-finite loss / gradient / advantage;
2. cost not conserved (ledger vs tap);
3. credit not conserved (task credit vs on_time);
4. a credit mask under an unlimited budget (should be impossible, so it is a bug);
5. the same seam error three times running;
6. two consecutive evaluation points with success `< 0.5 * q_ref` and bytes `> 2x` immediate.

On any breaker the loop stops, `TRAIN_SUMMARY.json` is written with the breaker named and the
state kept.

## What a result may and may not claim

May: per cell, "success >= 95 % of `normal_immediate` and fewer bytes" — or not.
May not: a cross-cell ranking, a Pareto-frontier claim, or any sentence that attributes an R
difference to N.
