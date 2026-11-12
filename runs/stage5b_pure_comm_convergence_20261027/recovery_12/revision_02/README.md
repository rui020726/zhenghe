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
| 16 | **The header's 字节倍数 column held the send ratio on 10 of 16 rows** (e.g. 1100 written as 0.505, actually 0.422; 3500 written as 0.414, actually 0.502) | `selective_eval_summary.json` has a field named `sends` that is `learned_emissions / immediate_emissions`; it was copied into the bytes column for the rows that had no freshly computed stats | every ratio in the header is now aggregated from `spent_bytes` and `on_time` in each `EVAL.json` over the same 45 worlds, never copied from a summary field. Sends and bytes are different quantities: a dropped send removes its bytes, but the surviving sends keep their own fan-out cost |
| 17 | **milestone_100 aggregated against an 8-scene baseline** (3345 immediate sends / 2455 on_time, giving a spurious 0.369 success ratio) | the tag map picked `eval_100_seamfix`, whose `EVAL.json` still carries the denominator defect of #10; the corrected re-aggregation is `eval_100_seamfix_recomputed` (1245 / 892, ratio 1.016 / 1.000) | the map now points at `_recomputed`. That tag has no transcripts of its own (`--recompute-from` re-aggregates without copying), and the original tag's worlds use the pre-#2 namespace — one directory per scene with 15 cell leaves inside — so `world_dir` accepts both layouts. Both tags are kept |
| 18 | 23 evaluation directories are named `eval_<ep>_eval_<ep>_e<ep>` | `evaluate.py:205` already prefixes `eval_<episode>_`, and the batch script passed `--out-tag eval_<ep>_e<ep>` | cosmetic only: every `EVAL.json` is complete (45/45 worlds). The directories are **not** renamed, because renaming archived artifacts breaks the paths recorded in earlier reports; the tag map stores the actual names. Later points use `--out-tag e<ep>` |
| 19 | **the in-process evaluation never worked: every point raised `ValueError: invalid logical-slot artifact namespace`** | #11's fix stamped the namespace with `@<episode>`, but `world.artifact_directory` only accepts `[A-Za-z0-9_]+` in the arm slot, so every one of the 15 cells was rejected and `evaluate_all_cells` (correctly) refused to report a null point | the stamp is now `_c<episode>` (`eval:T2_c24:91002:2301901`). Verified: `smoke_v4b` evaluated **15/15 cells** in-process. All earlier numbers came from external `evaluate.py` runs, so no published result changes |
| 20 | 18 of 1146 updates print `kl=nan` (restated from #13) | `or float("nan")` turns a genuine `0.0` into `nan` | `train.py` now prints `n/a` only when the key is absent (`None`); a zero KL prints as `0.000000`. Verified in `train_v4_cell_lambda` |
| 21 | **`train_v3`'s single lambda sat on a clamp 191 times** (122 at 8.0) and G=48 could only stop sending at the ceiling | one price for a batch that mixes cells, and `N*G` pricing made a broadcast cost `6.0/G`: break-even `λ* = G/6` = 2/4/**8**, i.e. exactly the ceiling at G=48 | `train_v4_cell_lambda`: cost is charged bytes **per subscriber** (`λ* ≈ 3.99` at every G), and each of the 15 cells keeps its own lambda, moved only by that cell's EMA against its own `0.95 * q_ref` with step 0.25 |
| 22 | **a `--start-index` continuation numbers its checkpoints from 1**, not from the world index | `state["episode"] = len(episodes_log)`, so `train_v4e_cont`'s first milestone is `milestone_100.pth` while it holds the model after world 3699 and started from a checkpoint labelled 300 | recorded, not renamed: the run's `TRAIN_SUMMARY.json` carries `start_index` and `start_episode`, and `evaluate.py --episode N` matches the file name. Consequence to know when reading: `v4e/eval_100` is **exactly** `v4d/eval_400` — verified field by field — because v4e's first 100 episodes replay v4d's last 100 over the same seeds and world stream. That equality is the determinism check, not a missing training step |
| 23 | **the dual integrated the sampling success rate while the verdict measures the greedy one** (v4d: R=6 cells report 0.58 sampled, deploy 0.25) | `q_ema` was EMA'd from each training batch's `on_time / generated`, and a batch is drawn from the stochastic policy; the registered verdict is read off the argmax policy | `--dual-anchor` (v4f `single`, v4g `scene3`). `scene3` runs the in-process evaluation on the same three DEV scenes `evaluate.py` uses, assigns that aggregate to `q_ema` in **both** directions (v4f was one-way, so no R=12 cell was ever told it had margin), and freezes `q_ema` between scans (`q_ema_frozen=True`) so a batch cannot pull it back. Water line is `Q_TARGET_FRACTION_DUAL = 0.965`; the *verdict* stays at 0.95 and a test pins both numbers |
| 24 | v4f@200's 15-cell in-process point read one scene per cell, so the same R=6 cell swung 0.25–0.58 between scenes | `evaluate_all_cells` rotated through one scene per checkpoint | `--eval-scenes 3` (implied by `--dual-anchor scene3`): 45 worlds per point, per-cell aggregate `sum(on_time)/sum(generated)`, which is the external report's arithmetic. Cost measured: 964 s for one 4-episode smoke including 45 eval worlds |
| 26 | **the scalar price cannot place 15 cells in a 1-5 page window**, so the pass count stuck at 3/15 through ten arms | measured: at the same lambda≈4 one cell cut 1 page and another cut 97 (train_v4n, lambda sd 0.845); the send/hold logit margin is 12-14 with `|margin|<1` at 0 % | `--send-cap per_cell`: the dual moves an **integer page budget** per cell (one page = one unit of the verdict), starting at `normal_immediate`'s pages per episode minus `--send-cap-lead`; the policy still ranks its own targets and the budget only truncates the ordered list. `train_v4r_cap_lead1` reaches **9/15** (merged on-time 1.012, bytes 0.957), cross-checked by `evaluate.py` 15/15 identical |
| 27 | the first cap arm took its ceiling from the **current** policy's page count | `train_v4p` measured 17 pages/episode on G24_R6_N3 after the actor drifted, set a cap of 6 and locked the over-cut in (0/15) | the ceiling is now `normal_immediate.emissions / worlds` from the baseline, immune to drift |
| 25 | **the anchored dual stepped only for the cells a 4-episode batch happened to draw** | `dual_step` iterated `sorted(per_cell)`; with 15 cells and 4 episodes per update, a given cell moves in 1–2 of the 25 updates between scans instead of 25, so the integrator ran ~15x slower than the registered rule and depended on sampling luck. Measured on `train_v4g`: 300 episodes moved the R=12 prices by +0.10…+0.40 against a predicted +0.53 **per scan**, so λ never reached the break-even 0.251-cost ⇒ λ\* ≈ 4 | under `q_ema_frozen` the anchored rate is valid for every cell, so every anchored cell steps every update; without the freeze the batch's own rate is still the only rate and the original rule is unchanged. `test_frozen_anchor_steps_every_cell_not_only_the_batch` pins it. `train_v4g`/`train_v4h` keep their artifacts; the fix is exercised by `train_v4i_anchor_allcells` |

The `grid_v1` file is kept: `execution/baselines/grid_v1/BASELINES.json` records 120 successful
worlds and 240 failures. `grid_v2` is the complete grid; the analysis reads `grid_v2` and the
`grid_v1` failures stay on the record rather than being overwritten.

None of these was a pin change: every edit is in `recovery_12/revision_02/`, which is outside
the 98-file manifest.

### Round 43 (communication-efficiency columns and two traditional baselines)

| # | Failure | Cause | Fix |
|---|---|---|---|
| 28 | **the first efficiency extraction found 1 078 of 1 440 worlds "disagreeing with the archive"** | the shared baseline roots (`grid_v2`, `paper_throttles_v1`) hold every arm's worlds side by side, and the walk took every leaf under `world_logs/` for whichever arm was being measured — `normal_immediate`'s rows were compared against 14 other arms' transcripts | a shared root now selects only `eval__<arm>_*` directories; a dedicated root (one checkpoint per run) still takes every leaf. 1 440 worlds, 0 mismatches |
| 29 | `KeyError: 'spent_bytes'` after the extraction had already read every log twice | the transcript records what went on the air, not what it cost; bytes live in the archived row (`spent_bytes`), which the worker had not attached | the archived byte total is attached in the worker, so the efficiency readings and the table's byte column come from the same number rather than two definitions |
| 30 | the per-task ratios would have printed `ZeroDivisionError` / `inf` for `always_hold` | an arm that never sends has no on-time task to divide by | `n/a` is printed for `None`, and `always_hold` keeps its 0 channel uses with blank ratios. A reader must not see `0.00 B/task` for an arm that delivered nothing |
| 31 | the two traditional arms could not be addressed by their catalogue names | the catalogue registers `periodic` and `event_maxwait`; the paper needed `periodic_5` (the pre-registered Stage C arm) and wanted the two labels to say what triggers them, so `build_rule_policy(arm)` raised *unknown rule arm* on both. Note `_assert_transport` was never the problem: it compares the relay's policy name against `getattr(attach, "name")`, i.e. the same object it built | `TRADITIONAL_ARMS = ("periodic_k5", "event_maxwait")` plus two explicit branches that call `build_comm_policy("periodic", {"interval": 5})` and `build_comm_policy("event_maxwait")` with no parameters, and `adapter_rule` maps each label to itself. `TraditionalArmTests` pins the two parameter sets so a later edit cannot quietly retune them |
| 32 | **the paired `byte_ratio` difference compared two different ratios** (reported v4t − random = +0.164, i.e. "v4t costs more") | `reading()` was called with the *comparator's* row as the denominator, so each side of the difference was divided by a different world's bytes; the correct per-world difference is −0.022 | both sides now divide by `normal_immediate` in the same world. Re-ran: 72 tests, and the sign flipped on every arm-vs-arm byte row. The first pass's numbers were never published outside this run; `PAIRED_TESTS.json` is the corrected file and carries its own sha256 |
| 33 | **`slack_heuristic_cap` at the seven new goal counts sent 1.0 message per world** (94–98 % "saved", on-time collapsed to 5.6 %) | `build_tasks` resolved `immediate_pages` only for the capped and matched arms, so the throttle branch fell back to `pages.get(label, 1)` — a k=1 page budget of `max(1, 1-1) = 1` page for the whole episode | the page-count lookup now covers every arm that needs it (`v4t`, `v4s`, `slack_heuristic_cap`, `slack_matched_rate`). The first pass is **quarantined, not deleted**, at `execution/g_scaling/_invalid_slack_cap_v1/`; the re-run writes the same `execution/g_scaling/slack_heuristic_cap/` tag and its 175 worlds are all fresh |
| 34 | `slack_matched_rate` refused to start: "needs v4t pages for all 50 cells; have 35" | the guard demanded 50 cells, but this sweep only runs the seven new goal counts (35 cells); the archived three keep their sealed matched arm | the guard now asks for `len(NEW_GOALS) * len(PAIRS)` = 35, and the report still takes the archived three from `paper_slack_matched` |
| 35 | **the first IEEE-style render printed dummy boxes for every Chinese glyph inside a `$...$`-bearing label** | two separate causes: (i) `font.family="serif"` + a `font.serif` list does not enable per-glyph fallback — the family must be the explicit list `["Times New Roman", "SimSun", "STIXGeneral"]`; (ii) even with the list, matplotlib 3.10's mathtext renderer does not fall back for the *non-math* segments of a mixed string, so `任务规模 $G$` loses its Chinese | mixed labels use Unicode math italics (𝐺/𝑅/𝑁, served by STIXGeneral) with no `$...$`; pure-math strings like `$G^*=36$` keep mathtext. The render now fails loudly on any missing-glyph warning, and the PDFs embed all three fonts as TrueType subsets (`/FontFile2`, `pdf.fonttype=42`). The old dashboard-style figures are quarantined at `execution/figures/_dashboard_v1/` |
| 36 | the G-sweep report's table-5 headers claimed "13 世界/点" (R=6) and "12 世界/点" (R=12) | hardcoded strings in `g_scaling_report.py`; the actual scene rule gives R=6 = 3+8 = **11** worlds and R=12 = 3+8+3 = **14** (the 25 total was right, the split was wrong). Found while answering "why can the on-time ratio exceed 1" — the figure script computes the count from the cells and printed 11 | the generator now reads `worlds` from the aggregate instead of hardcoding; report regenerated (headers 11/14), `G_SCALING.json` byte-identical (sha `f746bf98…`), figure manifest re-appended. No number in any table changed — only the two header counts |
| 37 | **`slack_matched_rate` on the S axis could not find its matched depth** (48/48 worlds failed with *needs v4t's pages per episode*) | the matched table was keyed by the S-carrying label (`G24_R6_N6_S1`), but `run_rule_world` looks the depth up by the *base* cell label, and the cell tuple deliberately does not carry S | the task now ships a one-entry table keyed by the base label and holding **that S's own** v4t page count, so the matched arm is matched to the same world it is compared against |
| 38 | **13 of those worlds then failed with `FileExistsError`** on the retry | the first failed pass had already created 13 artifact leaves before the lookup raised, and `reserve_artifact_directory` refuses to reopen a leaf — correct behaviour, wrong state to retry from | the arm's directory was removed and the stage re-run from empty; the 48 final worlds are all fresh. A retry of a *partially written* arm needs an explicit clean, and that is now the recorded procedure |
| 39 | **the S-axis waste table was empty for every learned arm** and every R-axis world read as if it ran at S=1 | two separate key defects in `rs_waste.py`: the leaf pattern required `_S<k>` while learned arms write `_c100_s<k>` (so all learned worlds fell into the default bucket), and the default bucket itself was `1` instead of the frozen `2` | the pattern is now `_?[sS](\d+)$` (matches both spellings) and the R-axis default is 2. Re-extracted: 594 R worlds at S=2, 360 S worlds spread over all five counts |
| 40 | **the crossed-grid `normal_immediate` stage lost 48 worlds and reported no `RESULTS.json`** | the stage process exited between writing worlds and calling `merge`; 8 leaves also held a `lifecycle.jsonl` without its `.gz`, i.e. died mid-archive. `merge` is only reached when every future returns, and a silent process exit returns nothing | the 8 half-written leaves are quarantined under `execution/g_rs_grid/_incomplete_normal_immediate/` (not deleted); the 108 intact worlds were rebuilt into `RESULTS.json` by `analysis/g_rs_recover.py`, whose arithmetic was validated world-for-world against a completed arm (0 mismatches on `on_time`/`emissions`/`spent_bytes`/`distinct`) before it was allowed to write. The 48 missing worlds were then run for real |
| 41 | **every `slack_heuristic_cap` and `slack_matched_rate` world in the crossed grid ran with a 1-page budget** (94–99 % "saved", on-time collapsed) | both arms derived their page count from `immediate_pages()`, which reads this grid's own `RESULTS.json` — absent for the interrupted stage — and `build_tasks` fell back to `pages.get(label, 1)`. Same defect class as ledger #33, one pipeline later | the fallback is gone: a missing page count is now a hard `SystemExit`, and `v4t_pages()` additionally learned to merge the archived rows (it had been returning `None` for the 12 archived cells, which would have given the matched arm no cap at all). Both arms were quarantined under `_invalid_budget_v1/` and re-run from empty: 156 worlds each, 0 failures, G≥84 sends/world now 47 rather than 1 |
| 42 | the first crossed-grid report aggregated **331** worlds per arm instead of 162 | `load_arm` merged the whole archived G-sweep file, which also holds the N=3 rows and the five non-anchor scenes, so a G=72 aggregate mixed fan-outs that the grid is meant to hold fixed | the loader now accepts only cells inside this grid (G in the six counts, R in the three rosters, N=6) and only the three anchor scenes. Re-run: 162 worlds per arm, 54 (G,R,S) points × 3 |

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
