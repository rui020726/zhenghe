"""BUCKET_ARITHMETIC_RECHECK.json: the design-revision recheck demanded by §二.3 and §二.4.

Nothing is re-run and no episode is created: the buckets are recomputed from the same stored
logs that produced COMM_COST_COVERAGE.json, and every percentage is reported together with its
denominator, because the round-19 documents quoted *byte* shares next to *target* counts.

It also records the identity that makes the whole-policy contrast well defined (the two arms see
the same 352 targets), and the provenance of the AUC/threshold numbers: which label, which
features, which split, which fitting step.

Run:  /d/project_env/python.exe runs/comm_cost_objective_design_20260926/c19_recheck.py
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import Any

import c19_common as c19

import c19_cost_coverage as cov

OLD_TO_NEW_LABEL = {
    "load_bearing_outcome_only_with_uplink": "outcome_differs_under_full_silence",
    "redundant_knowledge_and_outcome": "known_no_later_and_outcome_same_under_full_silence",
    "uplink_bought_time_same_outcome": "known_later_but_outcome_same_under_full_silence"}
LABEL_KIND = ("descriptive label of a whole-policy contrast (A1 immediate vs A2 always_hold); "
              "no single-message intervention was performed, so none of these is a causal "
              "contribution of one packet")


def recompute() -> tuple[Counter[str], Counter[str], dict[int, dict[str, Any]]]:
    counts: Counter[str] = Counter()
    cost_bytes: Counter[str] = Counter()
    per_seed: dict[int, dict[str, Any]] = {}
    a1_rows, a2_rows = cov.data_rows("A1"), cov.data_rows("A2")
    for seed in sorted(set(a1_rows) & set(a2_rows)):
        out_a1, _ = cov.per_target("A1", seed)
        out_a2, _ = cov.per_target("A2", seed)
        know_a1 = cov.knowledge_steps("A1", seed)
        know_a2 = cov.knowledge_steps("A2", seed)
        costs = {str(r["target"]): int(r["cost_bytes"]) for r in cov.emissions("A1", seed)
                 if r.get("first")}
        seed_counts: Counter[str] = Counter()
        for target, info in sorted(out_a1.items()):
            second = out_a2.get(target)
            known = target in know_a2
            if second is None or second["outcome"] != info["outcome"]:
                label = "outcome_differs_under_full_silence"
            elif known and know_a2[target]["first_step"] <= know_a1[target]["first_step"]:
                label = "known_no_later_and_outcome_same_under_full_silence"
            else:
                label = "known_later_but_outcome_same_under_full_silence"
            counts[label] += 1
            seed_counts[label] += 1
            cost_bytes[label] += costs.get(target, 0)
        per_seed[seed] = {
            "targets_in_A1": sorted(out_a1),
            "targets_in_A2": sorted(out_a2),
            "target_sets_identical": sorted(out_a1) == sorted(out_a2),
            "counts": dict(seed_counts),
            "on_time_A1": sum(1 for v in out_a1.values() if v["outcome"] == "on_time"),
            "on_time_A2": sum(1 for v in out_a2.values() if v["outcome"] == "on_time")}
    return counts, cost_bytes, per_seed


def shares(counts: Counter[str], cost_bytes: Counter[str], seeds: int) -> dict[str, Any]:
    total_targets = sum(counts.values())
    total_bytes = sum(cost_bytes.values())
    out: dict[str, Any] = {"denominators": {
        "targets": total_targets, "billed_bytes_over_the_batch": total_bytes,
        "seeds": seeds,
        "note": "count shares divide by targets, byte shares divide by billed bytes, "
                "per-episode values divide by seeds; the three must never be mixed in one "
                "sentence"},
        "per_bucket": {}}
    for label in sorted(counts):
        out["per_bucket"][label] = {
            "targets": counts[label],
            "count_share_of_targets": c19.ratio(counts[label], total_targets),
            "billed_bytes": cost_bytes[label],
            "byte_share_of_bill": c19.ratio(cost_bytes[label], total_bytes),
            "bytes_per_episode": c19.ratio(cost_bytes[label], seeds),
            "targets_per_episode": c19.ratio(counts[label], seeds)}
    out["totals"] = {"targets": total_targets, "billed_bytes": total_bytes,
                     "targets_per_episode": c19.ratio(total_targets, seeds),
                     "bytes_per_episode": c19.ratio(total_bytes, seeds),
                     "count_shares_sum": round(sum(
                         counts[v] / total_targets for v in counts), 6),
                     "byte_shares_sum": round(sum(
                         cost_bytes[v] / total_bytes for v in cost_bytes), 6)}
    return out


def main() -> int:
    counts, cost_bytes, per_seed = recompute()
    seeds = len(per_seed)
    share_block = shares(counts, cost_bytes, seeds)
    delivered = c19.read_json(c19.ROUND_DIR / "COMM_COST_COVERAGE.json")["redundancy_against_the_silence_counterfactual"]
    old_counts = {OLD_TO_NEW_LABEL[key]: value
                  for key, value in delivered["counts"].items()}
    old_bytes = {OLD_TO_NEW_LABEL[key]: value
                 for key, value in delivered["bytes_by_category"].items()}
    identical_sets = sum(1 for row in per_seed.values() if row["target_sets_identical"])
    on_time_a1 = sum(row["on_time_A1"] for row in per_seed.values())
    on_time_a2 = sum(row["on_time_A2"] for row in per_seed.values())
    c1_value = c19.read_json(c19.R16 / "D1_RESULTS.json")
    published_c1 = None
    for contrast in c1_value.get("contrasts", []):
        if contrast.get("name") == "C1":
            published_c1 = contrast.get("on_time_difference_left_minus_right", {}).get("mean")
    payload = {
        "round": "19-revision", "phase": "design revision §二.3 / §二.4 recheck, 0 episode",
        "label_semantics": {
            "kind": LABEL_KIND,
            "renamed_from": {new: old for old, new in OLD_TO_NEW_LABEL.items()},
            "why_renamed": ("the old names ('load bearing', 'strictly redundant') read as if one "
                            "packet had been switched off; what was actually compared is two "
                            "whole policies on the same seeds")},
        "recomputation_matches_delivered_artefact": {
            "counts_equal": dict(counts) == old_counts,
            "bytes_equal": dict(cost_bytes) == old_bytes,
            "recomputed_counts": dict(counts), "recomputed_bytes": dict(cost_bytes)},
        "arithmetic": share_block,
        "closure_checks": {
            "counts_sum_to_targets": sum(counts.values()) == share_block["denominators"]["targets"],
            "bytes_sum_to_bill": sum(cost_bytes.values())
            == share_block["denominators"]["billed_bytes_over_the_batch"],
            "per_episode_bill_matches_arm_A1": share_block["totals"]["bytes_per_episode"]
            == c19.read_json(c19.ROUND_DIR / "COMM_COST_COVERAGE.json")["deployed_arm_table"]
            ["A1"]["per_episode"]["airtime_bytes"],
            "both_arms_see_the_same_targets": identical_sets == seeds,
            "seeds_with_identical_target_sets": identical_sets,
            "outcome_difference_total": on_time_a1 - on_time_a2,
            "outcome_difference_per_episode": c19.ratio(on_time_a1 - on_time_a2, seeds),
            "published_C1_mean": published_c1,
            "matches_published_C1": published_c1 is not None
            and abs((on_time_a1 - on_time_a2) / seeds - published_c1) < 1e-5},
        "citation_corrections": [
            {"wrong_reading": "19 份（5.46 %）、110 份（31.24 %）、223 份（63.30 %）",
             "why_wrong": "5.46/31.24/63.30 是**字节份额**（分母 874 398 B），"
                          "不是目标份额（分母 352 个）",
             "correct": "按目标数 19/352 = 5.40 %、110/352 = 31.25 %、223/352 = 63.35 %；"
                        "按计费字节 47 706/874 398 = 5.46 %、273 162/874 398 = 31.24 %、"
                        "553 530/874 398 = 63.30 %",
             "occurrences": ["PROJECT_CHARTER.md v4.2 §一 与 §9 第十九轮段与 §13 v4.2 行",
                             "PROJECT_STATUS.md 2026-09-26 节",
                             "README.md 当前研究入口节",
                             "COMM_COST_COVERAGE.md §五 标题（'承重的那 5.5 %'）"],
             "action": "旧件不改，修订登记于 COMM_COST_OBJECTIVE_DESIGN_REVISED.md §二"},
            {"wrong_reading": "'承重'/'严格冗余' 当作单条消息的因果贡献",
             "why_wrong": "从未做单消息干预；比的是两个整策略（A1 全即时 vs A2 全静默）",
             "correct": "整策略对照下的描述性标签，见 label_semantics",
             "occurrences": ["COMM_COST_COVERAGE.md §四/§五", "CONSTRAINED_COMM_COST_DESIGN.md §五",
                             "CONSTRAINED_COMM_COST_PILOT.md §一"],
             "action": "同上"}],
        "auc_provenance": {
            "label": "the three descriptive buckets above, built from the A1/A2 whole-policy "
                     "contrast on seeds 1100000..1100059",
            "positive_class": "outcome_differs_under_full_silence (19 targets)",
            "negative_class": "the other 333 targets",
            "features_used": ["step_reported_at", "slack_at_report", "slack_over_horizon",
                              "deadline_offset", "severity", "confidence", "requirements_left",
                              "total_stages", "backlog_steps", "payload_bytes"],
            "features_source": "reconstructed from the A1 transcript's first-report records and "
                               "the A1 lifecycle detection records; not read from a decision tape "
                               "(no deployed cell in the archive has one)",
            "features_of_the_19_dim_schema_not_reconstructed": [
                "since_last_emission_time", "pending_event", "changed_view", "stage_changed",
                "requirements_changed", "members_changed", "progress_delta_abs", "revision_delta",
                "queue_backlog_targets"],
            "why_those_nine_are_constant_or_unused_here": (
                "all 352 archived emissions are first reports (revision == 1, members empty), so "
                "the change-detection dims have no previous view to compare against; "
                "queue_backlog_targets was not reconstructed and is assumed small under "
                "`immediate` but was NOT verified per cell"),
            "model_fitted": "none — a rank statistic (Mann-Whitney U / AUC) and an exhaustive "
                            "threshold sweep",
            "data_split": "NONE. The same 60 seeds produced the labels, the AUC and the threshold; "
                          "there is no held-out part and no cross-validation",
            "threshold_selection": "every distinct feature value swept in both directions, scored "
                                   "by counterfactual bookkeeping, cheapest rule at full recorded "
                                   "effect reported",
            "status": "development-set exploratory result. It is not a validated effect size and "
                      "must not be cited as one; seeds 1100000..1100059 are 'seen' under "
                      "PROJECT_CHARTER.md §10 and cannot serve as an unseen test set",
            "confound": "the outcome attributed to a dropped report is that arm's recorded "
                        "outcome; A2's robots moved differently because they were not told, so "
                        "the bookkeeping is not a per-message counterfactual"},
        "cost_nomenclature": {
            "what_the_number_is": "载荷尝试成本（当前计费模型口径）：payload-only 估算 × 被寻址的 "
                                  "订阅者数，按尝试计费、丢包照付",
            "what_it_is_not": ["真实无线空口时间", "含包头/成帧/链路层/MAC 的成本",
                               "能量或时延成本", "全系统通信成本"],
            "unmetered_and_listed_separately": [
                "环境自带视距真值刷新（old_task_env.py:477-492）",
                "机器人一跳 gossip（old_task_env.py:494-512）",
                "事件总线分发与 TaskRegistry 入库",
                "发送节点对 registry 的轮询",
                "协调器 advice 与分配命令",
                "ACK 反向信道（独立账本，永不共享额度）",
                "被额度拒掉的发送（未发出，不计费）"],
            "source": "integrated_sar/comm.py:97-123, 894-898, 953, 974; comm_budget.py:3-16"},
    }
    c19.write_json("BUCKET_ARITHMETIC_RECHECK.json", payload)
    import json
    print(json.dumps({
        "counts_equal_to_delivered": payload["recomputation_matches_delivered_artefact"]
        ["counts_equal"],
        "bytes_equal_to_delivered": payload["recomputation_matches_delivered_artefact"]
        ["bytes_equal"],
        "closure": payload["closure_checks"],
        "shares": {k: {kk: vv for kk, vv in v.items()}
                   for k, v in share_block["per_bucket"].items()},
        "totals": share_block["totals"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
