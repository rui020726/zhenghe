"""ROUND32 任务 G 之一：六条运行 × 32 批的轨迹重建（0 环境启动，只读落盘件）。

每批既取 ROUND32 的登记值，也从独立来源重算：reference/policy 的按时任务数来自该场景的
immediate reference 行与该集的训练行；`g_hat`、`λ_next` 按登记公式独立重算；actor 样本数与
masked 数从 tape 逐行数；N/M 与批记录里的逐 minibatch 读数核对。

时序按冻结契约：批内先 4 集 rollout、再 `ppo_update`、最后 `dual_update`，所以每批 actor 用的是
**更新前**的 λ；`lambda_at_rollout == lambda_before` 逐批核对。

sampled 训练轨迹不等于 greedy 约束轨迹：这里重建的是采样策略的约束统计，greedy 只在最终评估
里被观测，两套数字不互相代替。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import r32_common as common
from r32_train import batch_record_path


def mean(values):
    """Local descriptive mean (post-run analysis only; no training constant lives here)."""
    return sum(float(v) for v in values) / len(values)


def summary(values):
    """Local descriptive summary; the quantile rule is linear interpolation on sorted values."""
    floats = [float(v) for v in values]
    ordered = sorted(floats)
    n = len(ordered)
    average = mean(floats)

    def quantile(fraction):
        if n == 1:
            return ordered[0]
        position = (n - 1) * fraction
        low = int(position // 1)
        high = min(low + 1, n - 1)
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {"n": n, "mean": round(average, 9), "min": ordered[0], "max": ordered[-1],
            "q25": round(quantile(0.25), 9), "median": round(quantile(0.5), 9),
            "q75": round(quantile(0.75), 9),
            "std_pop": round((sum((v - average) ** 2 for v in floats) / n) ** 0.5, 9),
            "abs_mean": round(mean([abs(v) for v in floats]), 9)}


def reference_on_time() -> dict[int, float]:
    out: dict[int, float] = {}
    for path in sorted(common.ROWS.glob("r32_reference_s*_a1.json")):
        row = common.read_json(path)
        scenario = int(row["scenario_seed"])
        if scenario in out and out[scenario] != float(row["on_time"]):
            raise RuntimeError(f"scenario {scenario} has two different cached references")
        out[scenario] = float(row["on_time"])
    return out


def train_rows(reduction: str, seed: int) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in sorted(common.ROWS.glob(f"r32_train_{reduction}_s{seed}_e*_a*.json")):
        row = common.read_json(path)
        if row.get("reduction") != reduction or int(row["training_seed"]) != seed:
            raise RuntimeError(f"{path.name}: row identity mismatch for {reduction} s{seed}")
        out[int(row["episode_index"])] = row
    return out


def recount(tapes: list[dict[str, Any]]) -> dict[str, int]:
    n_rows = len(tapes)
    m_rows = sum(1 for tape in tapes if tape["actionable"])
    return {
        "n_rows": n_rows,
        "m_active_rows": m_rows,
        "masked_rows": n_rows - m_rows,
        "candidates": sum(len(tape["targets"]) for tape in tapes),
        "actionable_candidates": sum(len(tape["actionable"]) for tape in tapes),
        "masked_candidates": sum(len(tape["masked"]) for tape in tapes),
        "forced_candidates": sum(len(tape["forced"]) for tape in tapes),
    }


def constraint_statistic(y_reference: list[float], y_policy: list[float]) -> float:
    return (sum(y_reference) / len(y_reference)) - (sum(y_policy) / len(y_policy)) - common.DELTA


def dual_step(lam: float, g_hat: float) -> float:
    return max(0.0, lam + common.ETA_LAMBDA * g_hat)


def scan_non_finite(obj: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(scan_non_finite(value, f"{path}.{key}"))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(scan_non_finite(value, f"{path}[{index}]"))
    elif isinstance(obj, float) and not math.isfinite(obj):
        found.append(path)
    return found


def run(reduction: str, seed: int, references: dict[int, float],
        rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    order = common.scenario_order()
    batches: list[dict[str, Any]] = []
    problems: list[str] = []
    previous_next = None
    for index in range(32):
        record = common.read_json(batch_record_path(reduction, seed, index))
        episodes = [int(value) for value in record["episodes"]]
        scenarios = [int(value) for value in record["scenarios"]]
        if scenarios != [order[episode] for episode in episodes]:
            problems.append(f"batch {index}: scenarios differ from the frozen order")
        y_reference = [float(references[scenario]) for scenario in scenarios]
        y_policy = []
        counts = {key: 0 for key in ("n_rows", "m_active_rows", "masked_rows", "candidates",
                                     "actionable_candidates", "masked_candidates",
                                     "forced_candidates")}
        spent = []
        for episode in episodes:
            row = rows.get(episode)
            if row is None:
                problems.append(f"batch {index}: training row e{episode} missing")
                continue
            if int(row["scenario_seed"]) != scenarios[episodes.index(episode)]:
                problems.append(f"batch {index}: row e{episode} scenario differs from record")
            y_policy.append(float(row["on_time"]))
            spent.append(float(row["spent_bytes"]))
            for tape_key, value in recount(
                    list(common.iter_jsonl(Path(row["artefacts"]["tape"])))).items():
                counts[tape_key] += value
            if abs(float(row["lambda"]) - float(record["lambda_at_rollout"])) > 1e-12:
                problems.append(f"batch {index} e{episode}: row lambda != batch rollout lambda")
        g_recomputed = constraint_statistic(y_reference, y_policy)
        g_recorded = float(record["constraint"]["g_hat"])
        lam_before = float(record["dual"]["lambda_before"])
        lam_next = float(record["dual"]["lambda_next"])
        lam_recomputed = dual_step(lam_before, g_recorded)
        if abs(g_recomputed - g_recorded) > 1e-12:
            problems.append(f"batch {index}: g_hat recorded {g_recorded} != recomputed "
                            f"{g_recomputed}")
        if abs(lam_next - lam_recomputed) > 1e-12:
            problems.append(f"batch {index}: lambda_next != max(0, lambda_before+eta*g_hat)")
        if abs(float(record["lambda_at_rollout"]) - lam_before) > 1e-12:
            problems.append(f"batch {index}: lambda_at_rollout != lambda_before")
        if previous_next is not None and abs(lam_before - previous_next) > 1e-12:
            problems.append(f"batch {index}: lambda chain broken")
        if index == 0 and abs(lam_before - common.LAMBDA_INIT) > 1e-12:
            problems.append("batch 0: lambda_before != registered lambda_initial")
        previous_next = lam_next
        update = record.get("update") or {}
        reads = update.get("minibatch_reads") or []
        # reads are per minibatch and the batch runs EPOCHS_PER_BATCH epochs over the same
        # sample list, so the batch sums must be epochs x rows (gate arithmetic, fixed once and
        # applied consistently here too)
        epochs = int(common.v2c.EPOCHS_PER_BATCH)
        per_epoch: dict[int, list[int]] = {}
        for read in reads:
            slot = per_epoch.setdefault(int(read["epoch"]), [0, 0])
            slot[0] += int(read["n_rows"])
            slot[1] += int(read["m_active_rows"])
        read_n = sum(int(read["n_rows"]) for read in reads)
        read_m = sum(int(read["m_active_rows"]) for read in reads)
        epochs_ok = (len(per_epoch) == epochs
                     and all(pair[0] == counts["n_rows"] and pair[1] == counts["m_active_rows"]
                             for pair in per_epoch.values()))
        if read_n != epochs * counts["n_rows"]:
            problems.append(f"batch {index}: recorded N {read_n} != epochs x tape rows "
                            f"{epochs * counts['n_rows']}")
        if read_m != epochs * counts["m_active_rows"]:
            problems.append(f"batch {index}: recorded M {read_m} != epochs x tape active rows "
                            f"{epochs * counts['m_active_rows']}")
        if not epochs_ok:
            problems.append(f"batch {index}: per-epoch N/M sums do not each equal the tape "
                            f"counts: {sorted(per_epoch.items())}")
        if int(update.get("samples", -1)) != counts["n_rows"]:
            problems.append(f"batch {index}: update.samples != tape decision rows")
        if index + 1 < 32:
            following = common.read_json(batch_record_path(reduction, seed, index + 1))
            if (record["controller_parameters_after"]["parameter_sha256"]
                    != following["controller_parameters_before"]["parameter_sha256"]):
                problems.append(f"batch {index}: weight hash chain broken")
        non_finite = scan_non_finite({"dual": record["dual"], "constraint": record["constraint"],
                                      "update": {key: value for key, value in update.items()
                                                 if key != "minibatch_reads"},
                                      "reads": reads})
        if non_finite:
            problems.append(f"batch {index}: non-finite at {non_finite[:4]}")
        read_summary = record.get("update_read_summary") or {}
        batches.append({
            "batch": index, "episodes": episodes, "scenarios": scenarios,
            "y_reference": y_reference, "y_policy": y_policy,
            "paired_task_loss_by_scene": [round(r - p, 9) for r, p in zip(y_reference, y_policy)],
            "g_hat_recorded": g_recorded, "g_hat_recomputed": round(g_recomputed, 12),
            "lambda_before": lam_before, "lambda_after_recorded": lam_next,
            "lambda_after_recomputed": round(lam_recomputed, 12),
            "lambda_at_rollout": float(record["lambda_at_rollout"]),
            "actor_used_pre_update_lambda": bool(
                abs(float(record["lambda_at_rollout"]) - lam_before) < 1e-12),
            "spent_bytes_by_episode": spent,
            "cost_c_by_episode": [round(value / common.B0_BYTES, 9) for value in spent],
            "cost_c_batch_mean": round(mean([value / common.B0_BYTES for value in spent]),
                                       9),
            "tape_counts": counts,
            "recorded_N": read_n, "recorded_M": read_m,
            "active_row_share": round(counts["m_active_rows"] / counts["n_rows"], 6)
                                if counts["n_rows"] else None,
            "minibatches": len(reads),
            "optimizer_steps": int(update.get("optimizer_steps", 0)),
            "clipping_minibatches": int(read_summary.get("clipping_minibatches", 0)),
            "pre_clip_total_norm": read_summary.get("pre_clip_total_norm"),
            "post_clip_total_norm": read_summary.get("post_clip_total_norm"),
            "clip_scale": read_summary.get("clip_scale"),
            "actor_surrogate_mean": read_summary.get("actor_surrogate_mean"),
            "actor_surrogate_mean_all_rows_reference": read_summary.get(
                "actor_surrogate_mean_all_rows_reference"),
            "entropy_term": read_summary.get("entropy_term"),
            "value_loss_task": read_summary.get("value_loss_task"),
            "value_loss_cost": read_summary.get("value_loss_cost"),
            "approx_kl": read_summary.get("approx_kl"),
            "clip_fraction": read_summary.get("clip_fraction"),
            "module_pre_clip_l1_mean": read_summary.get("module_pre_clip_l1_mean"),
            "module_pre_clip_l2_mean": read_summary.get("module_pre_clip_l2_mean"),
            "non_finite_minibatches": int(read_summary.get("non_finite_minibatches", -1)),
            "reduction": record["reduction"], "encoder_arm": record["encoder_arm"],
            "pre_update_checkpoint": record.get("pre_update_checkpoint"),
            "parameters_before_sha256": record["controller_parameters_before"]["parameter_sha256"],
            "parameters_after_sha256": record["controller_parameters_after"]["parameter_sha256"],
        })
    g_values = [batch["g_hat_recomputed"] for batch in batches]
    lam_values = [batch["lambda_after_recorded"] for batch in batches]
    losses_all = [float(references[int(row["scenario_seed"])]) - float(row["on_time"])
                  for row in rows.values()]
    costs_all = [float(row["spent_bytes"]) / common.B0_BYTES for row in rows.values()]
    first, last = batches[:8], batches[-8:]
    summary = {
        "reduction": reduction, "training_seed": seed, "batches": len(batches),
        "episodes": len(rows),
        "lambda_initial": batches[0]["lambda_before"],
        "lambda_final": batches[-1]["lambda_after_recorded"],
        "lambda_min": min(lam_values), "lambda_max": max(lam_values),
        "lambda_range": round(max(lam_values) - min(lam_values), 12),
        "lambda_monotone_non_decreasing": bool(
            all(batches[i]["lambda_after_recorded"] >= batches[i]["lambda_before"] - 1e-12
                for i in range(len(batches)))),
        "batches_with_positive_g_hat": sum(1 for value in g_values if value > 0),
        "batches_with_zero_g_hat": sum(1 for value in g_values if value == 0),
        "batches_with_negative_g_hat": sum(1 for value in g_values if value < 0),
        "mean_paired_task_loss_all_episodes": round(mean(losses_all), 9),
        "mean_cost_c_all_episodes": round(mean(costs_all), 9),
        "first8_mean_paired_task_loss": round(mean(
            [value for batch in first for value in batch["paired_task_loss_by_scene"]]), 9),
        "last8_mean_paired_task_loss": round(mean(
            [value for batch in last for value in batch["paired_task_loss_by_scene"]]), 9),
        "first8_mean_cost_c": round(mean(
            [value for batch in first for value in batch["cost_c_by_episode"]]), 9),
        "last8_mean_cost_c": round(mean(
            [value for batch in last for value in batch["cost_c_by_episode"]]), 9),
        "active_row_share_run": round(sum(batch["tape_counts"]["m_active_rows"]
                                          for batch in batches)
                                      / max(1, sum(batch["tape_counts"]["n_rows"]
                                                   for batch in batches)), 9),
        "candidate_actionable_share_run": round(
            sum(batch["tape_counts"]["actionable_candidates"] for batch in batches)
            / max(1, sum(batch["tape_counts"]["candidates"] for batch in batches)), 9),
        "zero_actor_gradient_row_share_run": round(
            sum(batch["tape_counts"]["masked_rows"] for batch in batches)
            / max(1, sum(batch["tape_counts"]["n_rows"] for batch in batches)), 9),
        "optimizer_steps_total": sum(batch["optimizer_steps"] for batch in batches),
        "minibatches_total": sum(batch["minibatches"] for batch in batches),
        "clipping_minibatches_total": sum(batch["clipping_minibatches"] for batch in batches),
        "clipping_share_of_minibatches": round(
            sum(batch["clipping_minibatches"] for batch in batches)
            / max(1, sum(batch["minibatches"] for batch in batches)), 6),
        "non_finite_minibatches_total": sum(max(0, batch["non_finite_minibatches"])
                                            for batch in batches),
        "pre_update_checkpoints_present": [batch["batch"] for batch in batches
                                           if batch["pre_update_checkpoint"]],
        "lambda_chain_closed": not any("lambda chain" in problem or "lambda_before" in problem
                                        for problem in problems),
        "weight_chain_closed": not any("weight hash" in problem for problem in problems),
        "problems": problems,
    }
    return {"batches": batches, "summary": summary}


def main() -> int:
    references = reference_on_time()
    payload: dict[str, Any] = {
        "round": "32",
        "section": "training trajectories rebuilt from batch records, rows, references and tapes "
                   "(0 launches, 0 optimizer.step)",
        "contract": {
            "objective": "c = C/B0; g = E[Y_ref] - E[Y_policy] - delta; L_max = -E[c] - lambda*g; "
                         "A_actor = lambda*A_task - A_cost",
            "dual": "lambda_next = max(0, lambda_before + eta_lambda * g_hat)",
            "parameters": {"B0_bytes": common.B0_BYTES, "delta": common.DELTA,
                           "lambda_initial": common.LAMBDA_INIT, "eta_lambda": common.ETA_LAMBDA,
                           "episodes_per_batch": common.EPISODES_PER_UPDATE},
            "lambda_timing": "r32_train freezes lam for the batch, runs the four rollouts, calls "
                             "ppo_update and only then dual_update, so every batch's actor is "
                             "trained with the PRE-update lambda; verified field by field",
            "reduction_definition": {name: spec["actor_reduction"]
                                     for name, spec in common.REDUCTIONS.items()},
        },
        "reading_notes": [
            "this is the SAMPLED policy's constraint trajectory; the greedy decode has its own "
            "numbers only in PAIRED_COMPARISON.json",
            "g_hat is a 4-episode paired point estimate per batch; a batch mean of 0 can still "
            "contain scenes with loss and scenes with gain",
            "a curve still moving at batch 31 only shows the run had not stabilised; it does not "
            "establish that more training would help",
        ],
        "runs": {}, "cross_run": {}}
    for reduction in common.ARM_ORDER:
        for seed in common.TRAIN_SEEDS:
            key = f"{reduction}_s{seed}"
            payload["runs"][key] = run(reduction, int(seed), references,
                                       train_rows(reduction, int(seed)))
    for seed in common.TRAIN_SEEDS:
        base_run = payload["runs"][f"{common.ARM_ORDER[0]}_s{seed}"]
        test_run = payload["runs"][f"{common.ARM_ORDER[1]}_s{seed}"]
        payload["cross_run"][f"s{seed}"] = {
            "identical_scenario_sequence": [b["scenarios"] for b in base_run["batches"]]
                                           == [b["scenarios"] for b in test_run["batches"]],
            "identical_y_reference": [b["y_reference"] for b in base_run["batches"]]
                                     == [b["y_reference"] for b in test_run["batches"]],
            "identical_y_policy": [b["y_policy"] for b in base_run["batches"]]
                                  == [b["y_policy"] for b in test_run["batches"]],
            "identical_g_hat_sequence": [b["g_hat_recomputed"] for b in base_run["batches"]]
                                        == [b["g_hat_recomputed"] for b in test_run["batches"]],
            "identical_lambda_sequence": [b["lambda_after_recorded"] for b in base_run["batches"]]
                                         == [b["lambda_after_recorded"]
                                             for b in test_run["batches"]],
            "first_batch_where_lambda_differs": next(
                (index for index in range(32)
                 if abs(base_run["batches"][index]["lambda_after_recorded"]
                        - test_run["batches"][index]["lambda_after_recorded"]) > 1e-12), None),
            "first_batch_where_scenarios_or_actions_diverge": next(
                (index for index in range(32)
                 if base_run["batches"][index]["y_policy"] != test_run["batches"][index]["y_policy"]
                 or base_run["batches"][index]["tape_counts"]
                    != test_run["batches"][index]["tape_counts"]), None),
            "lambda_final_base": base_run["summary"]["lambda_final"],
            "lambda_final_test": test_run["summary"]["lambda_final"],
            "active_row_share_base": base_run["summary"]["active_row_share_run"],
            "active_row_share_test": test_run["summary"]["active_row_share_run"],
        }
    payload["totals"] = {
        "batch_records_read": sum(len(run_block["batches"])
                                  for run_block in payload["runs"].values()),
        "problems": {key: len(run_block["summary"]["problems"])
                     for key, run_block in payload["runs"].items()},
        "non_finite_total": sum(run_block["summary"]["non_finite_minibatches_total"]
                               for run_block in payload["runs"].values()),
    }
    out = common.OUTPUT_DIR / "TRAINING_CURVES.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({key: {
        "lambda_final": value["summary"]["lambda_final"],
        "g_pos_zero_neg": [value["summary"]["batches_with_positive_g_hat"],
                           value["summary"]["batches_with_zero_g_hat"],
                           value["summary"]["batches_with_negative_g_hat"]],
        "loss_all": value["summary"]["mean_paired_task_loss_all_episodes"],
        "loss_first8_last8": [value["summary"]["first8_mean_paired_task_loss"],
                              value["summary"]["last8_mean_paired_task_loss"]],
        "cost_c": value["summary"]["mean_cost_c_all_episodes"],
        "active_share": value["summary"]["active_row_share_run"],
        "clip_share": value["summary"]["clipping_share_of_minibatches"],
        "problems": len(value["summary"]["problems"])}
        for key, value in sorted(payload["runs"].items())}, ensure_ascii=False, indent=1))
    print("cross:", json.dumps(payload["cross_run"], ensure_ascii=False, indent=1))
    print("totals:", json.dumps(payload["totals"], ensure_ascii=False))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
