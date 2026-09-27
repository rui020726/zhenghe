"""Stage B 汇总：逐 seed 等价件、闭合摘要、重复性件与启动台账刷新。

只读 rows/ 与 GATE_RECHECK.json，不启动任何环境。台账刷新遵循第十八轮纪律：计划块冻结、
实时数只进不退。
"""

from __future__ import annotations

import json
import sys
from typing import Any

import sb_common as common


def _row(path_name: str, seed: int) -> dict[str, Any]:
    return common.read_json(common.cell_paths(path_name, seed)["row"])


def _gates() -> dict[str, Any]:
    return common.read_json(common.HERE / "GATE_RECHECK.json")


def _cell_facts(path_name: str, seed: int) -> dict[str, Any]:
    row = _row(path_name, seed)
    counters = row["relay_counters"]
    return {
        "path": path_name, "seed": seed,
        "on_time": row["on_time"], "expired": row["expired"], "discovered": row["discovered"],
        "spent_bytes": row["spent_bytes"], "emissions": row["emissions"],
        "budget_refused_sends": counters["budget_refused_sends"],
        "capacity_refused_sends": counters["capacity_refused_sends"],
        "emissions_suppressed_terminal": counters["emissions_suppressed_terminal"],
        "deliveries_applied": counters["deliveries_applied"],
        "packets_lost": counters["packets_lost"],
        "ack_ledger": row["ack_ledger"],
        "behaviour_sha256": row["behaviour_sha256"],
    }


def write_deliverables() -> dict[str, Any]:
    gates = _gates()
    by_seed = {entry["seed"]: entry for entry in gates["seeds"]}

    equivalence: dict[str, Any] = {}
    closure: dict[str, Any] = {}
    repeatability: dict[str, Any] = {}
    for seed in common.SEEDS:
        entry = by_seed[seed]
        equivalence[str(seed)] = {
            "gate_b1": entry["gates"]["B1"],
            "compared_fields": ["emission_sequence", "refusal_sequence", "relay_counters",
                                "spent_bytes", "airtime_tap_total", "on_time", "completed",
                                "expired", "discovered", "behaviour_sha256", "ack_ledger"],
            "allowed_differences": list(common.ALLOWED_METADATA_DIFFERENCES),
            "native_facts": {"R_send": _cell_facts("R_send", seed),
                             "R_hold": _cell_facts("R_hold", seed)},
            "injected_facts": {"I_send": _cell_facts("I_send", seed),
                               "I_hold": _cell_facts("I_hold", seed)},
        }
        rows = {path: _row(path, seed) for path in common.PATHS}
        closure[str(seed)] = {
            "gate_b3": entry["gates"]["B3"],
            "gate_b3_math": entry["gates"]["B3_math"],
            "per_path": {path: {
                "spent_bytes": rows[path]["spent_bytes"],
                "normalised_cost_total": rows[path]["normalised_cost_total"],
                "airtime_tap_total": rows[path]["airtime_tap_total"],
                "task_reward_total": rows[path]["task_reward_total"],
                "on_time": rows[path]["on_time"],
                "b0_test_bytes": rows[path]["b0_test_bytes"],
            } for path in common.PATHS},
            "collector_summary": {path: rows[path]["collector_summary"]
                                  for path in ("P_init", "P_repeat")},
        }
        repeatability[str(seed)] = {
            "gate_b4": entry["gates"]["B4"],
            "P_init": {"spent_bytes": rows["P_init"]["spent_bytes"],
                       "on_time": rows["P_init"]["on_time"],
                       "behaviour_sha256": rows["P_init"]["behaviour_sha256"]},
            "P_repeat": {"spent_bytes": rows["P_repeat"]["spent_bytes"],
                         "on_time": rows["P_repeat"]["on_time"],
                         "behaviour_sha256": rows["P_repeat"]["behaviour_sha256"]},
        }

    common.write_json("EQUIVALENCE.json", {"round": "21", "gate": "B1 native/injected",
                                           "seeds": equivalence})
    common.write_json("CLOSURE_SUMMARY.json", {"round": "21", "gates": ["B3", "B3-math"],
                                               "seeds": closure})
    common.write_json("REPEATABILITY.json", {"round": "21", "gate": "B4", "seeds": repeatability})

    plan = common.read_json(common.HERE / "START_LEDGER.json")["plan"]
    report = common.ledger_report()
    common.write_json("START_LEDGER.json",
                      {"plan": plan, "live": {"used": report["used"],
                                              "by_kind": report["by_kind"],
                                              "results": report["results"]}},
                      refresh=True)
    return {"gates_all_passed": gates["all_passed"], "used": report["used"],
            "cap": common.LAUNCH_CAP}


def main() -> int:
    summary = write_deliverables()
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["gates_all_passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
