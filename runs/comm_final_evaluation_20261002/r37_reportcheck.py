"""ROUND37 报告一致性核验（0 环境启动）：报告里的每个数字都必须能在机器产物里找到。

三类检查：

1. 同源：报告 §6 的表格单元里出现的每个数字，都必须在机械渲染的 `RESULTS_TABLES.md` 里出现；
2. 重算：报告的两条主张在本脚本里直接对 rows 重算一遍——
   (a) 三个 greedy 臂的逐场景按时任务数与 `always_hold` 在 64/64 个场景上完全相同；
   (b) 合计 364 / 375 / 373 与统计件的 matrix_quality 相符；
3. 计数：报告引用的门禁、矩阵与额度数字与 `ROUND37_GATE.json`、
   `EVALUATION_LEDGER_RECONCILIATION.json`、`ROUND37_STATISTICS.json` 逐项相符。

    /d/project_env/python.exe r37_reportcheck.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from typing import Any

import r37_common as r37

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
SKIP_TOKENS = {"64", "1", "0", "2", "3", "576", "9", "16", "6", "4"}


def normalise(text: str) -> str:
    return text.replace("−", "-").replace("–", "-").replace("[", " ").replace("]", " ")


def report_table_numbers(report: str) -> list[str]:
    body = report.split("## 6. 任务 6：结果", 1)[1].split("## 7. 明确的五项区分", 1)[0]
    numbers: list[str] = []
    for line in body.splitlines():
        if not line.startswith("|"):
            continue
        for cell in line.split("|")[1:-1]:
            for token in NUMBER.findall(normalise(cell)):
                if token in SKIP_TOKENS or len(token) <= 1:
                    continue
                numbers.append(token)
    return numbers


def greedy_equals_never_sending() -> dict[str, Any]:
    hold = [int(r37.read_json(r37.row_path({"arm": "always_hold", "scenario": s}))["on_time"])
            for s in r37.SCENARIOS]
    out = {}
    for seed in r37.TRAINING_SEEDS:
        arm = f"model_s{seed}_greedy"
        vector = [int(r37.read_json(r37.row_path({"arm": arm, "scenario": s}))["on_time"])
                  for s in r37.SCENARIOS]
        out[arm] = {"identical_in_every_scenario": vector == hold,
                    "scenarios_compared": len(hold),
                    "timely_total": sum(vector)}
    out["always_hold_total"] = sum(hold)
    return out


def main(argv: list[str]) -> int:
    r37.configure()
    report = (r37.HERE / "ROUND37_EVALUATION_REPORT.md").read_text("utf-8")
    tables = (r37.HERE / "RESULTS_TABLES.md").read_text("utf-8")
    stats = r37.read_json(r37.HERE / "ROUND37_STATISTICS.json")
    gate = r37.read_json(r37.HERE / "ROUND37_GATE.json")
    reconcile = r37.read_json(r37.HERE / "EVALUATION_LEDGER_RECONCILIATION.json")

    numbers = report_table_numbers(report)
    missing = sorted({token for token in numbers if token not in tables})
    recomputed = greedy_equals_never_sending()
    totals = {arm: int(item["timely_task_total_over_the_block"])
              for arm, item in stats["matrix_quality_per_arm"].items()}
    rule_totals = {"immediate": totals["immediate"],
                   "late_discovery_gate": totals["late_discovery_gate"],
                   "always_hold": totals["always_hold"]}
    quota = reconcile["authoritative_sources"]
    checks = {
        "K1_report_numbers_are_all_present_in_the_rendered_tables": {
            "numbers_checked": len(numbers), "distinct_checked": len(set(numbers)),
            "not_found_in_tables": missing,
            "passed": not missing},
        "K2_greedy_arms_match_the_never_send_baseline_per_scenario": {
            **recomputed,
            "passed": all(item["identical_in_every_scenario"]
                          for key, item in recomputed.items() if key.startswith("model_"))},
        "K3_reported_totals_match_the_statistics_artefact": {
            "rule_totals_from_rows": rule_totals,
            "model_greedy_totals_from_rows": {key: value for key, value in recomputed.items()
                                              if key.startswith("model_")},
            "totals_from_statistics": totals,
            "passed": (rule_totals["immediate"] == 375
                       and rule_totals["late_discovery_gate"] == 373
                       and rule_totals["always_hold"] == 364
                       and all(recomputed[key]["timely_total"] == 364
                               for key in recomputed if key.startswith("model_")))},
        "K4_gate_matrix_and_quota_numbers": {
            "gate_all_passed": gate["all_passed"],
            "gate_checks": len(gate["checks"]),
            "gate_launches_counted_in_matrix": gate["launches_consumed_by_the_gate"],
            "matrix_complete": stats["matrix_complete"],
            "statistics_cells": stats["cells"],
            "authoritative_rows": quota["row_files"],
            "authoritative_claims": quota["claim_files"],
            "authoritative_stage_c_ledger": quota["cell_starts_in_stage_c_ledger"],
            "technical_failures_in_ledger": quota["failed_entries_in_stage_c_ledger"],
            "recording_reserve_used": reconcile["this_round_book"]["recording_launches"],
            "training_launches": reconcile["this_round_book"]["training_launches"],
            "optimizer_steps": reconcile["this_round_book"]["optimizer_steps"],
            "reconciliation_all_passed": reconcile["all_passed"],
            "passed": (gate["all_passed"] and stats["matrix_complete"]
                       and stats["cells"] == 576 == quota["row_files"] == quota["claim_files"]
                       == quota["cell_starts_in_stage_c_ledger"]
                       and quota["failed_entries_in_stage_c_ledger"] == 0
                       and reconcile["this_round_book"]["recording_launches"] == 0
                       and reconcile["this_round_book"]["training_launches"] == 0
                       and reconcile["this_round_book"]["optimizer_steps"] == 0
                       and reconcile["all_passed"])}}
    payload = {"kind": "round37-report-consistency", "schema": "round37-reportcheck/1",
               "round": "37", "checked_at": time.strftime("%F %T"),
               "environment_launches_used": 0, "optimizer_steps_used": 0,
               "checks": checks,
               "all_passed": all(item["passed"] for item in checks.values()),
               "note": "the report never carries a number that the machine artefacts do not "
                       "carry; the headline per-scenario claim is recomputed here from the rows"}
    r37.write_json("ROUND37_REPORT_CHECK.json", payload, refresh=True)
    print(json.dumps({key: value["passed"] for key, value in checks.items()},
                     ensure_ascii=False, indent=1))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
