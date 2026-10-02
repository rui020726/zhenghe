"""ROUND37 交付对账（0 环境启动）：claim、行、本轮账本与真实 ledger 四方闭合。

数字全部从落盘件重算：`claims/*.claim`、`rows/*.json`、`EVALUATION_LAUNCH_BOOK.json`、
`logs/START_LEDGER_p*.json`、`shards/*.json`。报告不许手打任何计数。

    /d/project_env/python.exe r37_reconcile.py
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import r37_common as r37

ARM_ORDER = r37.ALL_ARMS


def rows_by_arm() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {arm: [] for arm in ARM_ORDER}
    extras: list[str] = []
    for path in sorted((r37.HERE / "rows").glob("*.json")):
        row = r37.read_json(path)
        scenario = int(row["scenario_seed"])
        name = path.name
        matched = None
        for arm in ARM_ORDER:
            if r37.row_path({"arm": arm, "scenario": scenario}).name == name:
                matched = arm
                break
        if matched is None:
            extras.append(name)
        else:
            out[matched].append(scenario)
    return {"per_arm": out, "unmatched_row_files": extras}


def main(argv: list[str]) -> int:
    r37.configure()
    book = r37.book()
    claims = sorted(path.name for path in (r37.HERE / "claims").glob("job_*.claim"))
    claim_indices = sorted(int(name.split("_")[1].split(".")[0]) for name in claims)
    rows = rows_by_arm()
    ledger_files = sorted((r37.HERE / "logs").glob("START_LEDGER_p*.json"))
    per_process: dict[str, int] = {}
    failures_in_real_ledger: list[dict[str, Any]] = []
    for path in ledger_files:
        payload = r37.read_json(path)
        per_process[path.name] = int(payload.get("starts") or 0)
        failures_in_real_ledger.extend({"pid_file": path.name, **item}
                                       for item in (payload.get("results") or [])
                                       if not item.get("ok"))
    shards = [r37.read_json(path) for path in sorted((r37.HERE / "shards").glob("*.json"))]
    row_count = sum(len(value) for value in rows["per_arm"].values())
    duplicates = {arm: len(value) - len(set(value)) for arm, value in rows["per_arm"].items()}
    missing = {arm: sorted(set(r37.SCENARIOS) - set(value))
               for arm, value in rows["per_arm"].items()}

    authoritative = {
        "cell_starts_in_stage_c_ledger": r37.real_ledger_total(),
        "claim_files": len(claims),
        "row_files": row_count,
        "failed_entries_in_stage_c_ledger": len(failures_in_real_ledger),
        "expected": r37.NORMAL_EVALUATION_LAUNCHES}
    divergence = {
        "stage_c_ledger_minus_this_round_book": r37.real_ledger_total() - book["starts"],
        "book_events": len(book["events"]),
        "explanation": "the round-level book is a shared JSON file that four parallel shards "
                       "update read-modify-write; before the lock was added, one increment was "
                       "lost. The Stage C per-PID ledger (written before each driver is "
                       "constructed), the atomic claim files and the row files are independent "
                       "of that race and all agree",
        "consequence": "no cell was started twice, skipped, or lost: only the convenience "
                       "counter read low by one",
        "registered_in": "ROUND37_HARNESS_DEFECTS.json"}
    payload = {
        "kind": "round37-ledger-reconciliation", "schema": "round37-reconciliation/1",
        "round": "37", "reconciled_at": time.strftime("%F %T"),
        "authoritative_sources": authoritative,
        "book_divergence_and_explanation": divergence,
        "authorised": {"normal_evaluation_launches": r37.NORMAL_EVALUATION_LAUNCHES,
                       "recording_reserve": r37.RECORDING_RESERVE,
                       "failure_reserve": r37.FAILURE_RESERVE,
                       "hard_cap": r37.HARD_CAP,
                       "max_parallel_workers": r37.MAX_PARALLEL_WORKERS,
                       "training_launches": 0, "optimizer_steps": 0},
        "claims": {"held": len(claims), "distinct_indices": len(set(claim_indices)),
                   "expected_if_complete": r37.NORMAL_EVALUATION_LAUNCHES,
                   "duplicate_claim_files": len(claims) - len(set(claim_indices))},
        "rows": {"written": row_count, "per_arm_counts": {arm: len(value)
                                                         for arm, value in rows["per_arm"].items()},
                 "missing_scenarios_per_arm": {arm: value for arm, value in missing.items()
                                              if value},
                 "duplicate_scenarios_within_an_arm": {arm: value for arm, value in
                                                      duplicates.items() if value},
                 "unmatched_row_files": rows["unmatched_row_files"]},
        "this_round_book": {key: book[key] for key in ("starts", "completed",
                                                       "technical_failures", "reused_rows",
                                                       "recording_launches", "training_launches",
                                                       "optimizer_steps")},
        "failed_book_events": [item for item in book["events"] if not item["ok"]],
        "real_stage_c_ledger": {"per_process_starts": per_process,
                                "total": r37.real_ledger_total(),
                                "failed_entries": failures_in_real_ledger},
        "shard_summaries": shards,
        "parallel_processes_used": len(shards),
        "quota_accounting": {
            "normal_used": book["completed"],
            "recording_reserve_used": book["recording_launches"],
            "failure_reserve_used": book["technical_failures"],
            "hard_cap_headroom": r37.HARD_CAP - book["starts"],
            "note": "本轮判定不需要独立流录制（immediate 是九臂之一，其行即配对参照），"
                    "因此 64 的录制额度保持未动用，不得挪作他用"},
        "four_way_closure": {
            "claims_equal_rows": len(claims) == row_count,
            "claims_equal_stage_c_ledger": len(claims) == r37.real_ledger_total(),
            "rows_equal_stage_c_ledger": row_count == r37.real_ledger_total(),
            "matrix_complete": row_count == r37.NORMAL_EVALUATION_LAUNCHES,
            "every_arm_has_all_64_scenarios": all(
                len(value) == r37.SCENARIO_COUNT and sorted(set(value)) == sorted(r37.SCENARIOS)
                for value in rows["per_arm"].values()),
            "this_round_book_is_one_short_by_a_lost_write":
                r37.real_ledger_total() - book["starts"] == 1},
        "boundaries": {"training_launches": 0, "optimizer_steps": 0,
                       "parameters_or_data_moved": 0, "network_attempts": 0,
                       "old_round_artefacts_rewritten": 0,
                       "pushed": False},
        "all_passed": (len(claims) == row_count == r37.real_ledger_total()
                       == r37.NORMAL_EVALUATION_LAUNCHES
                       and not failures_in_real_ledger
                       and not rows["unmatched_row_files"]
                       and not any(missing.values()) and not any(duplicates.values())
                       and book["technical_failures"] == 0
                       and len(set(claim_indices)) == len(claims)
                       and all(len(value) == r37.SCENARIO_COUNT
                               and sorted(set(value)) == sorted(r37.SCENARIOS)
                               for value in rows["per_arm"].values()))}
    r37.write_json("EVALUATION_LEDGER_RECONCILIATION.json", payload, refresh=True)
    print(json.dumps({"authoritative": payload["authoritative_sources"],
                      "book": payload["this_round_book"],
                      "closure": payload["four_way_closure"],
                      "all_passed": payload["all_passed"]}, ensure_ascii=False, indent=1))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
