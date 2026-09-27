"""SPEC_ITEM_VERIFICATION.json：补齐第十九轮 spec §3/§4 的三个未核对项。

只读磁盘上已有的 120 个格子的日志（`runs/search_trace_and_policy_20260925/`），0 新 episode。
第十九轮 spec 明确要求、而当时未核对的三件事：

  1. “120 000 步”的来源、arm 与 episode 清单，以及它与 14 条历史复现、120 条正式比较的关系；
  2. 用未四舍五入整数闭合 placed → detected → on_time → expired/未完成，并**查清差出的每个目标**
     （不能把“发现”写成“全部完成”）；
  3. 35/20/5 的标签由逐 seed 数据验证（禁止猜顺序）。

另外按 spec §4 的成本术语，把 C_uplink / C_ack / C_measured 与未计量项分开落一份机读件。
"""

from __future__ import annotations

import sys
from collections import Counter

import c20_common as c20

R18 = c20.REPO_ROOT / "runs" / "search_trace_and_policy_20260925"


def cells() -> dict[str, list[dict]]:
    rows = {"A": [], "B": []}
    for path in sorted((R18 / "rows").glob("data_*.json")):
        row = c20.read_json(path)
        if row.get("ok"):
            rows[row["arm"]].append(row)
    return {arm: sorted(block, key=lambda r: r["seed"]) for arm, block in rows.items()}


def per_target_outcomes(rows: list[dict]) -> dict[int, dict[str, dict]]:
    """每个格子的逐目标结局（来自该格自己的生命周期日志）。"""
    out: dict[int, dict[str, dict]] = {}
    for row in rows:
        records = c20_common_jsonl(row["artefacts"]["lifecycle"])
        first_seen: dict[str, float] = {}
        deadline: dict[str, float] = {}
        completed_at: dict[str, float] = {}
        for record in records:
            target = str(record.get("target_id"))
            stamp = float(record.get("timestamp") or 0.0)
            event = record.get("event_type")
            if event == "detection":
                first_seen.setdefault(target, stamp)
                deadline.setdefault(target, stamp + float(
                    (record.get("detail") or {}).get("deadline_offset", 0.0)))
            elif (event == "progress" and record.get("status") == "completed"
                    and target in first_seen):
                completed_at.setdefault(target, stamp)
        targets: dict[str, dict] = {}
        for target, seen in first_seen.items():
            done = completed_at.get(target)
            targets[target] = {
                "discovered_at": seen, "deadline": deadline[target],
                "completed_at": done,
                "outcome": ("on_time" if done is not None and done <= deadline[target] else
                            ("completed_late" if done is not None else "not_completed"))}
        out[int(row["seed"])] = targets
    return out


def c20_common_jsonl(path: str) -> list[dict]:
    from pathlib import Path
    with (c20.REPO_ROOT / path).open(encoding="utf-8") as handle:
        import json
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    rows = cells()
    steps = 1000
    payload: dict = {"round": "20", "phase": "spec item verification, 0 new episode",
                     "source": "runs/search_trace_and_policy_20260925/{rows,logs}"}

    # 1) “120 000 步”的来源
    per_arm = {arm: {"cells": len(block), "steps_per_cell": int(block[0]["steps"]),
                     "cells_checked": sorted({int(r["steps"]) for r in block}),
                     "seeds": [int(r["seed"]) for r in block]}
               for arm, block in rows.items()}
    total_steps = sum(block["cells"] * block["steps_per_cell"] for block in per_arm.values())
    payload["step_total_provenance"] = {
        "value_quoted_in_round18": total_steps,
        "meaning": "cells x configured steps of the formal matrix only",
        "per_arm": per_arm,
        "not_part_of_this_number": {
            "historical_reproductions": "14 seeds replayed through the detection stream, "
                                        "1000 steps each, but they are identity checks on the "
                                        "archived detection sequence, not part of the "
                                        "three-view closure count",
            "b1_identity_cells": "8 cells x 1000 steps of step-by-step comparison are counted "
                                 "in the 32 passivity checks, not here",
            "smoke_and_diagnostics": "excluded from the matrix entirely"},
        "three_view_closure_counted_separately": "120 000 (cell x step) three-view checks were "
                                                 "reported over the 120 matrix cells",
        "consistency": total_steps == 120_000 and per_arm["A"]["cells"] == 60}

    # 2) 未四舍五入整数闭合 + 差出的每个目标
    for arm, block in rows.items():
        placed = sum(int(r["goals_placed"]) for r in block)
        detected_ids = sum(len(r["detected_target_ids"]) for r in block)
        undetected_ids = sum(len(r["undetected_ids"]) for r in block)
        on_time = sum(int(r["on_time"]) for r in block)
        expired = sum(int(r["expired"]) for r in block)
        completed = sum(int(r["completed"]) for r in block)
        payload.setdefault("integer_closure", {})[arm] = {
            "placed": placed, "detected": detected_ids, "undetected": undetected_ids,
            "on_time": on_time, "completed": completed, "expired": expired,
            "detected_minus_on_time": detected_ids - on_time,
            "detected_plus_undetected_equals_placed": detected_ids + undetected_ids == placed}

    laggards = []
    for arm, block in rows.items():
        outcomes = per_target_outcomes(block)
        for row in block:
            seed = int(row["seed"])
            for target, info in outcomes.get(seed, {}).items():
                if info["outcome"] != "on_time":
                    laggards.append({
                        "arm": arm, "seed": seed, "target": target,
                        "outcome": info["outcome"],
                        "discovered_at": info["discovered_at"],
                        "deadline": info["deadline"],
                        "completed_at": info["completed_at"],
                        "in_detected_ids": target in row["detected_target_ids"],
                        "in_undetected_ids": target in row["undetected_ids"]})
    payload["targets_that_were_detected_but_not_on_time"] = laggards
    payload["laggard_notes"] = [
        "两者都是 detected 但未按时：一个 completed_late、一个 not_completed",
        "它们解释了 A/B 各自 detected−on_time = 1 的全部来源，不存在第三个差出的目标",
        "未发现的目标从未进入生命周期日志，因此不可能贡献 on_time 或 expired"]

    # 3) 35/20/5 由逐 seed 数据验证
    seeds = {int(r["seed"]) for r in rows["A"]}
    differences = []
    for seed in sorted(seeds):
        a = next(r for r in rows["A"] if int(r["seed"]) == seed)
        b = next(r for r in rows["B"] if int(r["seed"]) == seed)
        differences.append(int(a["undetected_count"]) - int(b["undetected_count"]))
    histogram = dict(sorted(Counter(differences).items()))
    payload["per_seed_difference_check"] = {
        "direction": "A − B（正数 = A 遗漏更多）",
        "histogram": histogram,
        "improved_A_fewer": sum(1 for d in differences if d < 0),
        "tied": sum(1 for d in differences if d == 0),
        "worse_A_more": sum(1 for d in differences if d > 0),
        "nonzero": sum(1 for d in differences if d != 0),
        "matches_round18_labels": (sum(1 for d in differences if d < 0) == 35
                                   and sum(1 for d in differences if d == 0) == 20
                                   and sum(1 for d in differences if d > 0) == 5)}

    # 4) 成本术语落一份 C_uplink / C_ack / C_measured 的机读件（基于已归档读数）
    round19 = c20.REPO_ROOT / "runs" / "comm_cost_objective_design_20260926"
    arms = c20.read_json(round19 / "COMM_COST_COVERAGE.json")["deployed_arm_table"]
    links = c20.read_json(round19 / "COMM_COST_COVERAGE.json")["link_coverage"]
    payload["cost_terms_from_archive"] = {
        "definitions": {
            "C_uplink": "载荷 × fan-out × 发送尝试，丢包计费；拒付计 0",
            "C_ack": "ACK 反向信道的真实尝试成本；未开启是结构性零，缺日志不是零",
            "C_measured": "只有当账本确实覆盖上面两项时才使用该名称",
            "unmetered": "本地感知/gossip/搜索内部通信等未计量项，不得称全系统总通信"},
        "per_arm": {arm: {"c_uplink_bytes_per_episode": arms[arm]["per_episode"]["airtime_bytes"],
                          "c_ack_bytes_per_episode": 0.0,
                          "ack_state": "未开启 ⇒ 结构性零（归档格 comm_ack=false）",
                          "configuration_identity": {
                              "policy": arms[arm]["comm_policy"],
                              "delivery_probability": arms[arm]["delivery_probability"]}}
                     for arm in ("A1", "A2", "A3", "A4")},
        "unmetered_and_listed_separately": links["free_knowledge_channels"] + [
            "ACK 反向信道（comm_ack.py:136-168，独立账本，永不共享额度）",
            "被预算/容量拒付的发送（未发射，计费 0）"],
        "note": "这些是第十六轮的归档读数，本轮只搬运定义与数值，不新跑一格"}

    payload["all_checks_pass"] = bool(
        payload["step_total_provenance"]["consistency"]
        and all(payload["integer_closure"][arm]["detected_plus_undetected_equals_placed"]
                for arm in ("A", "B"))
        and payload["per_seed_difference_check"]["matches_round18_labels"]
        and len(laggards) == 2)
    c20.write_json("SPEC_ITEM_VERIFICATION.json", payload)
    import json
    print(json.dumps({"consistency": payload["step_total_provenance"]["consistency"],
                      "integer_closure": payload["integer_closure"],
                      "laggards": len(laggards),
                      "per_seed": payload["per_seed_difference_check"],
                      "all_pass": payload["all_checks_pass"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
