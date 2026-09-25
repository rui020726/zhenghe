"""SEARCH_MISS_TIMELINES.json + SEARCH_DETECTION_CORRECTNESS_VERDICT.json.

For every goal the reproduction cells left undetected, the actual timeline is rebuilt from the
B1 side file: where each robot was and where that target was *at the detection instant of each
step*, whether the environment's own predicate ever held for it, what the environment actually
deleted, and what the adapter actually published. Then each target is put in exactly one of the
four categories the round-18 protocol §五 names.

The three readings this round must not skip:

* a target that was visible at reset is not a target the detector saw -- ``Environment.step``
  moves the robot first and judges observation second, so only the post-move cell counts;
* the 8-connected component argument is a *necessary* condition for a legal observation, not a
  proof that a four-direction robot can walk there;
* comparing medians of a handful of missed targets is descriptive only; it cannot rule out a
  placement factor.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import sp_common as sc

CATEGORIES = {
    "probe_satisfied_no_env_deletion":
        "探测时刻满足环境自己的真实判据（observed_pairs 命中），但环境未登记发现",
    "env_deleted_but_event_missing_or_wrong":
        "环境已发现/删除，但适配器漏发、错 ID 或重复发",
    "never_visible_at_detection_step":
        "实际探测时刻从未满足可见条件（仅表示该次轨迹没有可见接触）",
    "still_undecidable":
        "日志或历史复现不足，仍不能判断",
}


def probe_records(path) -> list[dict[str, Any]]:
    return list(sc.iter_jsonl(path)) if path.exists() else []


def timeline_for(row: dict[str, Any]) -> dict[str, Any]:
    facts = row.get("probe") or {}
    records = probe_records(sc.REPO_ROOT / row["artefacts"]["search_probe"])
    steps = [record for record in records if record.get("kind") == "step"]
    placed = {record["target_id"]: record["cell"] for record in records
              if record.get("kind") == "placed"}
    deleted: dict[str, int] = {}
    for record in steps:
        for target in record.get("env_deleted_target_ids") or []:
            deleted.setdefault(str(target), int(record["step"]))
    seen_by_pairs: dict[str, list[int]] = {}
    for record in steps:
        for target in record.get("pair_observed_target_ids") or []:
            seen_by_pairs.setdefault(str(target), []).append(int(record["step"]))
    adapter_events: dict[str, list[int]] = {}
    for record in steps:
        for target in record.get("adapter_event_target_ids") or []:
            adapter_events.setdefault(str(target), []).append(int(record["step"]))
    disagreement_steps = [int(record["step"]) for record in steps
                          if not (record.get("closure") or {}).get("all_views_agree", True)]
    # nearest approach over the whole episode, from the per-step detection-instant positions
    closest: dict[str, dict[str, Any]] = {}
    for record in steps:
        agents = record.get("agents_after_move") or []
        for goal in record.get("goals_at_detection") or []:
            target = next((tid for tid in placed
                           if tid.endswith(f"-target-{goal['reset_index']}")), None)
            if target is None:
                continue
            for agent in agents:
                distance = max(abs(agent[0] - goal["cell"][0]), abs(agent[1] - goal["cell"][1]))
                best = closest.setdefault(target, {"chebyshev": distance, "step": record["step"]})
                if distance < best["chebyshev"]:
                    best.update({"chebyshev": distance, "step": int(record["step"])})
    final = next((record for record in records if record.get("kind") == "final"), {})
    out: dict[str, Any] = {}
    for target in sorted(row.get("undetected_ids") or []):
        if not facts.get("search_probe_present"):
            category, grade, why = "still_undecidable", "not_determinable", "no probe file"
        elif target in deleted:
            category, grade = "env_deleted_but_event_missing_or_wrong", "measured"
            why = "环境删除了它，但下游没有收到对应发现事件"
        elif seen_by_pairs.get(target):
            category, grade = "probe_satisfied_no_env_deletion", "measured"
            why = ("环境的判据在这些步为真，环境却没有把它列入删除集合："
                   f"steps={seen_by_pairs[target][:8]}")
        elif disagreement_steps:
            category, grade, why = "still_undecidable", "not_determinable", (
                f"三视图在这些步不一致：{disagreement_steps[:8]}")
        elif target in placed:
            category, grade = "never_visible_at_detection_step", "measured"
            why = ("逐步位置里没有任何一步让环境自己的判据对它为真；"
                   "这只说明该次轨迹没有可见接触，不说明它物理不可达")
        else:
            category, grade, why = "still_undecidable", "not_determinable", (
                "该目标不在探针的放置快照里，身份无法对齐")
        out[target] = {
            "cell_at_reset": placed.get(target),
            "category": category, "grade": grade, "why": why,
            "steps_env_predicate_was_true": (seen_by_pairs.get(target) or [])[:12],
            "step_environment_deleted_it": deleted.get(target),
            "step_adapter_emitted_event": (adapter_events.get(target) or [])[:5],
            "nearest_detection_instant_approach": closest.get(target),
            "episode_end_positions": final.get("agents"),
            "remaining_targets_at_end": final.get("remaining_targets"),
            "termination_reason": final.get("termination_reason"),
        }
    return {
        "arm": row["arm"], "seed": row["seed"], "batch": row["batch"],
        "goals_placed": len(placed) or row["goals_placed"],
        "undetected": sorted(row.get("undetected_ids") or []),
        "probe_steps": facts.get("probe_steps"),
        "probe_closure_disagreement_steps": disagreement_steps[:12],
        "probe_anomalies": facts.get("probe_anomalies"),
        "environment_deleted_ids": sorted(deleted),
        "adapter_event_ids": sorted({target for targets in adapter_events.values()
                                     for target in targets}),
        "historical_run_reproduced": None,
        "targets": out,
    }


def main(argv: list[str]) -> int:
    identity_path = sc.ROUND_DIR / "SEARCH_TRACE_IDENTITY_REPORT.json"
    identity = sc.read_json(identity_path) if identity_path.exists() else {}
    reproduction = (identity.get("reproduction_gate_Q") or {}).get("detail") or {}
    rows = [sc.read_json(path) for path in sorted(sc.ROWS.glob("repro_*.json"))
            if sc.read_json(path).get("ok")]
    timelines = []
    for row in rows:
        cell = timeline_for(row)
        key = f"{row['arm']}/s{row['seed']}"
        cell["historical_run_reproduced"] = (reproduction.get(key) or {}).get(
            "historical_run_reproduced")
        cell["reproduction_detail"] = {
            name: (reproduction.get(key) or {}).get(name) for name in
            ("map_and_layout_match", "detection_sequence_match", "detection_count_then",
             "detection_count_now", "fields_only_in_todays_record",
             "behaviour_fields_that_differ", "outcome_matches_then")}
        timelines.append(cell)

    counts: dict[str, int] = {name: 0 for name in CATEGORIES}
    per_target_rows = []
    for cell in timelines:
        for target, detail in cell["targets"].items():
            counts[detail["category"]] += 1
            per_target_rows.append({"seed": cell["seed"], "arm": cell["arm"],
                                    "batch": cell["batch"], "target_id": target,
                                    **{key: detail[key] for key in
                                       ("cell_at_reset", "category", "grade", "why",
                                        "nearest_detection_instant_approach",
                                        "termination_reason")}})
    disagreements = {f"{cell['arm']}/s{cell['seed']}": cell["probe_closure_disagreement_steps"]
                     for cell in timelines if cell["probe_closure_disagreement_steps"]}
    anomalies = {f"{cell['arm']}/s{cell['seed']}": cell["probe_anomalies"]
                 for cell in timelines if cell["probe_anomalies"]}
    undetected_total = sum(len(cell["undetected"]) for cell in timelines)
    placed_total = sum(cell["goals_placed"] for cell in timelines)
    payload = {
        "round": sc.ROUND, "start_sha": sc.START_SHA,
        "categories": CATEGORIES,
        "cells": timelines,
        "per_target_rows": per_target_rows,
        "category_counts": counts,
        "denominators": {"cells": len(timelines), "goals_placed": placed_total,
                         "undetected_goals": undetected_total},
        "closure_disagreement_cells": disagreements,
        "probe_anomaly_cells": anomalies,
        "readings_that_must_not_be_skipped": [
            "初始位置可见不等于第一次实际探测时仍可见：Environment.step 先移动机器人再判定观测",
            "8 连通结论不能直接替代四方向机器人运动的可达性证明：它只是合法观测的必要条件",
            "少量遗漏目标的距离中位数比较是描述性证据，不能据此彻底排除放置因素",
        ],
    }
    sc.write_json("SEARCH_MISS_TIMELINES.json", payload)

    passive = bool((identity.get("passivity_gate_P") or {}).get("all_checks_true"))
    reproduced = (identity.get("reproduction_gate_Q") or {}).get("reproduced") or []
    not_reproduced = (identity.get("reproduction_gate_Q") or {}).get("not_reproduced") or []
    defects = counts["probe_satisfied_no_env_deletion"] + counts[
        "env_deleted_but_event_missing_or_wrong"]
    verdict = {
        "round": sc.ROUND,
        "passivity_gate_P_passed": passive,
        "historical_cells_reproduced": len(reproduced),
        "historical_cells_not_reproduced": not_reproduced,
        "undetected_goals_examined": undetected_total,
        "category_counts": counts,
        "closure_disagreement_steps_total": sum(len(value) for value in disagreements.values()),
        "defect_count_category_1_and_2": defects,
        "undecidable_count_category_4": counts["still_undecidable"],
        "verdict": ("no_defect_found" if passive and defects == 0 and not counts["still_undecidable"]
                    else ("defect_found" if defects else "insufficient_evidence")),
        "s1_matrix_allowed": bool(passive and defects == 0
                                  and not counts["still_undecidable"]),
        "if_not_allowed": ("stop after B1: report the passivity and correctness evidence and "
                           "produce no search-side effect number"),
        "meanings_that_stay_out_of_this": [
            "类别 3 只表示该次实现轨迹没有发生可见接触，不表示目标物理不可达",
            "类别 3 不表示其他搜索策略必然能找到该目标",
            "本判定不评价学习型通信控制，也不改变任何通信配置",
        ],
    }
    sc.write_json("SEARCH_DETECTION_CORRECTNESS_VERDICT.json", verdict)
    print(json.dumps({"verdict": verdict["verdict"], "counts": counts,
                      "passive": passive, "reproduced": len(reproduced),
                      "s1_allowed": verdict["s1_matrix_allowed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
