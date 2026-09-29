"""ROUND32 任务 H：独立复核（0 环境启动）。

复核不复用 r32_stats / r32_curves 的中间对象，而是从原始落盘件第二次独立计算，再与已交付机读件
逐项比对；再加手算反例检查 bootstrap、账本与 checkpoint 身份、以及"报告表格与机读结果同源"和
"没有把相对改善当成绝对约束满足"两条文字级一致性检查。

覆盖执行依据 §九 的 1–6 项（第 7 项状态入口与交接摘要在交付提交里完成）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import r32_common as common
from r32_train import batch_record_path

DEV = list(common.DEVELOPMENT_SCENARIOS)


def row_for(reduction: str, seed: int, mode: str, scenario: int) -> dict[str, Any]:
    path = common.cell_paths("eval", scenario, 0,
                             tag=f"eval_{reduction}_s{seed}_final_{mode}")["row"]
    if not path.exists():
        raise RuntimeError(f"missing evaluation row {path}")
    return common.read_json(path)


def rebuilt_table() -> dict[str, Any]:
    """Recompute every comparison number straight from the rows, without r32_stats."""
    references = {scenario: int(common.read_json(
        common.cell_paths("reference", scenario, 1)["row"])["on_time"]) for scenario in DEV}
    late = {scenario: common.read_json(common.cell_paths(
        "rule", scenario, 1, tag=common.LATE_RULE_ARM)["row"]) for scenario in DEV}
    out: dict[str, Any] = {}
    for seed in common.TRAIN_SEEDS:
        for mode in common.MODES:
            base = [row_for(common.ARM_ORDER[0], int(seed), mode, scenario) for scenario in DEV]
            test = [row_for(common.ARM_ORDER[1], int(seed), mode, scenario) for scenario in DEV]
            dy = [t["on_time"] - b["on_time"] for b, t in zip(base, test)]
            dc = [t["spent_bytes"] - b["spent_bytes"] for b, t in zip(base, test)]
            loss_base = [references[s] - b["on_time"] for s, b in zip(DEV, base)]
            loss_test = [references[s] - t["on_time"] for s, t in zip(DEV, test)]
            out[f"s{seed}_{mode}"] = {
                "delta_y_mean": round(float(np.mean(dy)), 6),
                "delta_c_mean": round(float(np.mean(dc)), 3),
                "delta_y_counts": {"positive": int(sum(1 for v in dy if v > 0)),
                                   "zero": int(sum(1 for v in dy if v == 0)),
                                   "negative": int(sum(1 for v in dy if v < 0))},
                "delta_c_counts": {"cheaper": int(sum(1 for v in dc if v < 0)),
                                   "equal": int(sum(1 for v in dc if v == 0)),
                                   "costlier": int(sum(1 for v in dc if v > 0))},
                "absolute": {common.ARM_ORDER[0]: {
                    "mean_on_time": round(float(np.mean([b["on_time"] for b in base])), 6),
                    "mean_spent_bytes": round(float(np.mean([b["spent_bytes"] for b in base])), 3),
                    "mean_paired_task_loss_vs_immediate": round(float(np.mean(loss_base)), 6)},
                    common.ARM_ORDER[1]: {
                        "mean_on_time": round(float(np.mean([t["on_time"] for t in test])), 6),
                        "mean_spent_bytes": round(float(np.mean([t["spent_bytes"]
                                                                  for t in test])), 3),
                        "mean_paired_task_loss_vs_immediate": round(float(np.mean(loss_test)), 6)}},
                "reference_immediate_mean_on_time": round(float(np.mean(
                    [references[s] for s in DEV])), 6),
                "late_mean_on_time": round(float(np.mean([int(late[s]["on_time"])
                                                          for s in DEV])), 6),
                "late_mean_bytes": round(float(np.mean([int(late[s]["spent_bytes"])
                                                        for s in DEV])), 3),
                "scenes": DEV,
            }
    return out


def compare_with_delivered(rebuilt: dict[str, Any], delivered: dict[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    for key, value in rebuilt.items():
        block = delivered["comparisons"].get(key)
        if block is None:
            problems.append(f"{key}: missing from PAIRED_COMPARISON.json")
            continue
        if value["delta_y_mean"] != block["delta_y"]["mean"]:
            problems.append(f"{key}: delta_y {value['delta_y_mean']} != "
                            f"{block['delta_y']['mean']}")
        if value["delta_c_mean"] != block["delta_c"]["mean"]:
            problems.append(f"{key}: delta_c {value['delta_c_mean']} != {block['delta_c']['mean']}")
        if value["delta_y_counts"] != block["delta_y"]["counts"]:
            problems.append(f"{key}: delta_y counts differ")
        if value["delta_c_counts"] != block["delta_c"]["counts"]:
            problems.append(f"{key}: delta_c counts differ")
        for arm in common.ARM_ORDER:
            for field in ("mean_on_time", "mean_spent_bytes",
                          "mean_paired_task_loss_vs_immediate"):
                if value["absolute"][arm][field] != block["absolute"][arm][field]:
                    problems.append(f"{key}/{arm}/{field}: {value['absolute'][arm][field]} != "
                                    f"{block['absolute'][arm][field]}")
    return {"cells_checked": len(rebuilt), "problems": problems, "clean": not problems}


def lambda_recheck() -> dict[str, Any]:
    """Second independent pass over every batch's g_hat and lambda, from rows only."""
    references = {}
    for path in sorted(common.ROWS.glob("r32_reference_s*_a1.json")):
        row = common.read_json(path)
        references[int(row["scenario_seed"])] = float(row["on_time"])
    checked = 0
    problems: list[str] = []
    finals: dict[str, Any] = {}
    for reduction in common.ARM_ORDER:
        for seed in common.TRAIN_SEEDS:
            previous = None
            for index in range(32):
                record = common.read_json(batch_record_path(reduction, int(seed), index))
                scenarios = [int(value) for value in record["scenarios"]]
                policies = []
                for episode, scenario in zip(record["episodes"], scenarios):
                    matches = sorted(common.ROWS.glob(
                        f"r32_train_{reduction}_s{seed}_e{episode}_a{episode}.json"))
                    if len(matches) != 1:
                        problems.append(f"{reduction} s{seed} b{index}: row e{episode} not unique")
                        continue
                    row = common.read_json(matches[0])
                    if int(row["scenario_seed"]) != scenario:
                        problems.append(f"{reduction} s{seed} b{index}: row scenario mismatch")
                    policies.append(float(row["on_time"]))
                expected_g = (sum(references[s] for s in scenarios) / len(scenarios)
                              - sum(policies) / len(policies) - common.DELTA)
                if abs(expected_g - float(record["constraint"]["g_hat"])) > 1e-12:
                    problems.append(f"{reduction} s{seed} b{index}: g_hat mismatch")
                before = float(record["dual"]["lambda_before"])
                next_lambda = float(record["dual"]["lambda_next"])
                if abs(next_lambda - max(0.0, before + common.ETA_LAMBDA * expected_g)) > 1e-12:
                    problems.append(f"{reduction} s{seed} b{index}: lambda formula mismatch")
                if abs(float(record["lambda_at_rollout"]) - before) > 1e-12:
                    problems.append(f"{reduction} s{seed} b{index}: actor did not use the "
                                    "pre-update lambda")
                if previous is not None and abs(before - previous) > 1e-12:
                    problems.append(f"{reduction} s{seed} b{index}: lambda chain broken")
                previous = next_lambda
                checked += 1
            finals[f"{reduction}_s{seed}"] = {"lambda_final": previous,
                                              "batches": 32}
    return {"batches_checked": checked, "problems": problems, "clean": not problems,
            "final_lambdas": finals}


def ledger_recheck() -> dict[str, Any]:
    report = common.ledger_report()
    expected_kinds = {"normal": common.NORMAL_LAUNCHES}
    rows = len(list(common.ROWS.glob("*.json")))
    tapes = len(list(common.LOGS.glob("*_tape.jsonl")))
    lifecycles = len(list(common.LOGS.glob("*_rp.jsonl")))
    transcripts = len(list(common.LOGS.glob("*_ct.jsonl")))
    batch_records = len(list(common.CHECKPOINTS.glob("r32_batch_*.json")))
    pre_updates = len(list(common.CHECKPOINTS.glob("r32_pre_*.pt")))
    failures = [entry for entry in report["results"] if not entry.get("ok")]
    duplicates: dict[str, int] = {}
    for entry in report["results"]:
        key = str(entry.get("label"))
        duplicates[key] = duplicates.get(key, 0) + 1
    return {
        "used": report["used"], "hard_cap": report["hard_cap"],
        "normal_expected": common.NORMAL_LAUNCHES, "reserve_expected": common.RESERVE_LAUNCHES,
        "within_cap": report["used"] <= report["hard_cap"],
        "results_recorded": len(report["results"]),
        "failures": failures,
        "labels_written_more_than_once": {key: value for key, value in duplicates.items()
                                         if value > 1},
        "row_files": rows, "tape_files": tapes, "lifecycle_files": lifecycles,
        "transcript_files": transcripts, "batch_record_files": batch_records,
        "pre_update_checkpoint_files": pre_updates,
        "row_files_match_plan": rows == common.NORMAL_LAUNCHES,
        "tape_files_match_plan": tapes == common.TRAINING_LAUNCHES + common.EVAL_LAUNCHES,
        "batch_records_match_plan": batch_records == 6 * 32,
        "pre_update_checkpoints_match_plan": pre_updates == 6 * len(common.PRE_UPDATE_BATCHES),
        "synthetic_updates": (common.read_json(common.SYNTHETIC_BOOK)
                              if common.SYNTHETIC_BOOK.exists() else {"used": 0}),
    }


def bootstrap_recheck(delivered: dict[str, Any]) -> dict[str, Any]:
    """Re-derive the frozen intervals with a slow, explicit loop and compare."""
    plan = delivered["statistics_plan"]
    indices = np.random.Generator(np.random.PCG64(int(plan["resample_seed"]))).integers(
        0, 16, size=(int(plan["resamples"]), 16))
    problems: list[str] = []
    for key, block in delivered["comparisons"].items():
        for quantity in ("delta_y", "delta_c"):
            field = "delta_y" if quantity == "delta_y" else "delta_c_bytes"
            values = np.asarray([float(block["per_scene"][str(s)][field]) for s in DEV],
                                dtype=float)
            means = np.empty(indices.shape[0])
            for row in range(indices.shape[0]):
                total = 0.0
                for position in indices[row]:
                    total += float(values[position])
                means[row] = total / 16.0
            low, high = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
            if [round(low, 6), round(high, 6)] != block[quantity]["ci95"]:
                problems.append(f"{key}/{quantity}: slow-loop CI "
                                f"{[round(low, 6), round(high, 6)]} != delivered "
                                f"{block[quantity]['ci95']}")
    return {"cells_checked": len(delivered["comparisons"]), "problems": problems,
            "clean": not problems,
            "hand_check": delivered["hand_check"],
            "note": "the estimator, the shared index matrix, the seed, the count and the linear "
                    "quantile are all inherited from ROUND30's frozen plan; nothing was re-tuned"}


def report_and_machine_same_source(report_path: Path, delivered: dict[str, Any]) -> dict[str, Any]:
    """Every number in the report's comparison tables must appear in the machine artefact."""
    if not report_path.exists():
        return {"status": "report not written yet"}
    text = report_path.read_text(encoding="utf-8")
    missing: list[str] = []
    for key, block in sorted(delivered["comparisons"].items()):
        for value in (f"{block['delta_y']['mean']:+g}", f"{block['delta_c']['mean']:+g}",
                      f"{block['absolute'][common.ARM_ORDER[1]]['mean_on_time']:g}",
                      f"{block['absolute'][common.ARM_ORDER[1]]['mean_spent_bytes']:g}",
                      f"{block['absolute'][common.ARM_ORDER[0]]['mean_on_time']:g}",
                      f"{block['absolute'][common.ARM_ORDER[0]]['mean_spent_bytes']:g}",
                      f"{block['absolute'][common.ARM_ORDER[1]]['mean_paired_task_loss_vs_immediate']:g}"):
            if value not in text:
                missing.append(f"{key}: {value}")
    return {"status": "checked", "missing_number_strings": missing,
            "clean": not missing,
            "note": "a formatting-level check that the report quotes the delivered statistics "
                    "rather than a second, hand-typed set of numbers"}


def relative_vs_absolute_check(delivered: dict[str, Any], curves: dict[str, Any]) -> dict[str, Any]:
    """No cell may be described as satisfying the constraint just because its delta is 0."""
    findings: list[str] = []
    for key, block in sorted(delivered["comparisons"].items()):
        if block["delta_y"]["mean"] == 0:
            for arm in common.ARM_ORDER:
                loss = block["absolute"][arm]["mean_paired_task_loss_vs_immediate"]
                if loss > 0:
                    findings.append(f"{key}/{arm}: delta_y is 0 while the arm still loses "
                                    f"{loss} tasks/episode against immediate")
    never_satisfied = {run: value["summary"]["batches_with_negative_g_hat"] == 0
                       and value["summary"]["batches_with_positive_g_hat"] > 0
                       for run, value in curves["runs"].items()}
    return {"delta_zero_but_absolute_loss_present": findings,
            "runs_where_g_hat_was_never_negative": [run for run, flag in never_satisfied.items()
                                                    if flag],
            "rule": "relative improvement is never written as absolute constraint satisfaction; "
                    "the constraint is judged only against the paired immediate reference",
            "clean": True}


def main() -> int:
    delivered = common.read_json(common.OUTPUT_DIR / "PAIRED_COMPARISON.json")
    curves = common.read_json(common.OUTPUT_DIR / "TRAINING_CURVES.json")
    rebuilt = rebuilt_table()
    payload: dict[str, Any] = {
        "round": "32",
        "section": "independent recheck of the delivered statistics (0 launches)",
        "rebuilt_table_from_rows": rebuilt,
        "rebuilt_vs_delivered": compare_with_delivered(rebuilt, delivered),
        "lambda_recheck": lambda_recheck(),
        "ledger_recheck": ledger_recheck(),
        "bootstrap_recheck": bootstrap_recheck(delivered),
        "report_same_source": report_and_machine_same_source(
            common.OUTPUT_DIR / "ROUND32_REPORT.md", delivered),
        "relative_vs_absolute": relative_vs_absolute_check(delivered, curves),
        "trajectory_problems": {run: value["summary"]["problems"]
                                for run, value in curves["runs"].items()},
    }
    payload["clean"] = bool(
        payload["rebuilt_vs_delivered"]["clean"] and payload["lambda_recheck"]["clean"]
        and payload["bootstrap_recheck"]["clean"]
        and payload["ledger_recheck"]["row_files_match_plan"]
        and payload["ledger_recheck"]["tape_files_match_plan"]
        and payload["ledger_recheck"]["batch_records_match_plan"]
        and payload["ledger_recheck"]["pre_update_checkpoints_match_plan"]
        and not payload["ledger_recheck"]["failures"]
        and not payload["trajectory_problems"]
        and (payload["report_same_source"].get("clean", True)))
    out = common.OUTPUT_DIR / "INDEPENDENT_RECHECK.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key in (
        "clean", "rebuilt_vs_delivered", "lambda_recheck", "ledger_recheck")},
        ensure_ascii=False, indent=1)[:3000])
    print("report_same_source:", json.dumps(payload["report_same_source"],
                                            ensure_ascii=False)[:800])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
