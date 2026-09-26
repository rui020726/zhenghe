"""COMM_COST_COVERAGE.json: what the deployed radio bill actually pays for, from stored runs.

Zero new episodes. Everything is re-read from round 16 (`comm_info_path_validation_20260923`,
four arms x 60 paired seeds on the deployed information path, with the B0 passive knowledge log
switched on) and from round 18's transcripts. Three questions:

  A. Which arm is the fair strong *rule* baseline, and at which cost? (the six comparators the
     charter demands, restricted to the deployed path, plus the isolated-path rules that are
     usually quoted as "the strong rule" and are not comparable here)
  B. What does the byte bill cover? Every billed packet of every deployed cell is re-priced from
     the transcript: payload x addressed robots, per first report, per refresh, lost or not.
  C. How much of that bill is *redundant with the environment's own sensing*? Answered by the
     only counterfactual the archive contains: arm A2 (``always_hold``) on the same 60 seeds,
     target by target. This bounds the savings a cost-minimising controller could chase and is
     the reason the design in step 3 is a constrained controller rather than a cheaper encoder.

Run:  /d/project_env/python.exe runs/comm_cost_objective_design_20260926/c19_cost_coverage.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from typing import Any

import c19_common as c19

sys.path.insert(0, str(c19.REPO_ROOT))

R16_ARMS = {"A1": "deployed, immediate, lossless",
            "A2": "deployed, always_hold (zero uplink traffic)",
            "A3": "deployed, immediate, 30 % packet loss",
            "A4": "as A3 but with the environment's own knowledge channels switched off: "
                  "D1 set gossip_interval=100000, and that counter gates "
                  "propagate_task_completion (old_task_env.py:1107-1110), which is the line-of-"
                  "sight truth refresh AND the one-hop gossip together (old_task_env.py:477-512) "
                  "— so no archived arm separates the two"}


def data_rows(arm: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in sorted((c19.R16 / "rows").glob(f"data_{arm}_s*.json")):
        row = c19.read_json(path)
        if row.get("ok"):
            out[int(row["seed"])] = row
    return out


def log_path(arm: str, seed: int, suffix: str) -> c19.Path:
    return c19.R16 / "logs" / f"data_{arm}_s{seed}_{suffix}.jsonl"


def per_target(arm: str, seed: int) -> tuple[dict[str, dict[str, Any]], float]:
    """Task outcome per target, read from that cell's own lifecycle log.

    The window is the log's own final timestamp: round 16 rows carry no ``steps`` field, and
    taking it from the run rather than from a constant keeps the expired/on-time split honest if
    a cell ever ran short.
    """
    records = list(c19.iter_jsonl(log_path(arm, seed, "rp")))
    window = max((float(record.get("timestamp") or 0.0) for record in records), default=0.0)
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
    out: dict[str, dict[str, Any]] = {}
    for target, seen in first_seen.items():
        done = completed_at.get(target)
        outcome = ("on_time" if done is not None and done <= deadline[target] else
                   ("completed_late" if done is not None else
                    ("expired" if deadline[target] <= window else "unresolved")))
        out[target] = {"discovered_at": seen, "deadline": deadline[target],
                       "completed_at": done, "outcome": outcome}
    return out, window


def knowledge_steps(arm: str, seed: int) -> dict[str, dict[str, Any]]:
    """First time any robot knew this target, and through which channel (B0 log)."""
    best: dict[str, dict[str, Any]] = {}
    path = log_path(arm, seed, "ip")
    if not path.exists():
        return {}
    for record in c19.iter_jsonl(path):
        if record.get("kind") != "knowledge":
            continue
        target = str(record["target_id"])
        step = int(record["step"])
        current = best.get(target)
        if current is None or step < current["first_step"]:
            best[target] = {"first_step": step, "channel": record["channel"]}
    return best


def emissions(arm: str, seed: int) -> list[dict[str, Any]]:
    if not log_path(arm, seed, "ct").exists():
        return []
    return [record for record in c19.iter_jsonl(log_path(arm, seed, "ct"))
            if record.get("kind") == "send"]


def arm_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm, meaning in R16_ARMS.items():
        rows = data_rows(arm)
        n = len(rows)
        total = lambda key: sum(int(row[key]) for row in rows.values())  # noqa: E731
        summary[arm] = {
            "meaning": meaning,
            "cells": n,
            "delivery_probability": float(next(iter(rows.values()))
                                          ["transport_parameters"]["delivery_probability"])
            if rows else None,
            "comm_policy": next(iter(rows.values()))["transport_parameters"]["policy"]
            if rows else None,
            "per_episode": {
                "placed": c19.ratio(total("goals_placed"), n),
                "discovered": c19.ratio(total("discovered"), n),
                "on_time": c19.ratio(total("on_time"), n),
                "expired": c19.ratio(total("expired"), n),
                "airtime_bytes": c19.ratio(total("airtime_bytes"), n),
                "emissions": c19.ratio(total("emissions"), n),
                "first_emissions": c19.ratio(total("first_emissions"), n),
                "refresh_emissions": c19.ratio(total("refresh_emissions"), n),
                "packets_lost": c19.ratio(total("packets_lost"), n),
                "targets_never_known": c19.ratio(total("targets_never_known"), n),
                "budget_refused_sends": c19.ratio(total("budget_refused_sends"), n)},
            "bytes_per_on_time_task": c19.ratio(total("airtime_bytes"), total("on_time"))}
    a1, a2 = summary["A1"], summary["A2"]
    summary["derived"] = {
        "value_of_the_uplink_C1": {
            "on_time_per_episode": round(a1["per_episode"]["on_time"]
                                         - a2["per_episode"]["on_time"], 6),
            "bytes_per_episode": round(a1["per_episode"]["airtime_bytes"]
                                       - a2["per_episode"]["airtime_bytes"], 6),
            "bytes_per_extra_on_time_task": c19.ratio(
                a1["per_episode"]["airtime_bytes"] - a2["per_episode"]["airtime_bytes"],
                a1["per_episode"]["on_time"] - a2["per_episode"]["on_time"]),
            "crosscheck": "round 16 published +0.31667 个/ep and 14 573.3 B/ep here"},
        "first_reports_are_the_whole_bill": all(
            summary[arm]["per_episode"]["refresh_emissions"] == 0.0
            and summary[arm]["per_episode"]["emissions"]
            == summary[arm]["per_episode"]["first_emissions"]
            for arm in ("A1", "A2", "A3", "A4")),
        "budget_was_never_enforced": all(
            row["budget_refused_sends"] == 0 and row["capacity_refused_sends"] == 0
            for arm in R16_ARMS for row in data_rows(arm).values())}
    return summary


def redundant_share() -> dict[str, Any]:
    """Target-by-target: what would the deployed bill have bought if nothing had been sent."""
    a1_rows, a2_rows = data_rows("A1"), data_rows("A2")
    paired = sorted(set(a1_rows) & set(a2_rows))
    categories: Counter[str] = Counter()
    bytes_by_category: Counter[str] = Counter()
    payload_by_category: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    emission_cost: dict[str, dict[str, int]] = {}
    per_seed_rows: list[dict[str, Any]] = []
    for seed in paired:
        out_a1, _ = per_target("A1", seed)
        out_a2, _ = per_target("A2", seed)
        know_a1 = knowledge_steps("A1", seed)
        know_a2 = knowledge_steps("A2", seed)
        costs = {}
        for record in emissions("A1", seed):
            costs[str(record["target"])] = {
                "cost_bytes": int(record["cost_bytes"]),
                "payload_bytes": int(record["payload_bytes"])}
        emission_cost[str(seed)] = costs
        seed_row = {"seed": seed, "categories": Counter()}
        for target, info in sorted(out_a1.items()):
            cost = costs.get(target, {"cost_bytes": 0, "payload_bytes": 0})
            known_in_a2 = target in know_a2
            outcome_a2 = out_a2.get(target, {}).get("outcome")
            same_outcome = outcome_a2 == info["outcome"]
            earlier_or_equal = (known_in_a2
                                and know_a2[target]["first_step"]
                                <= know_a1.get(target, {}).get("first_step", 10 ** 9))
            if known_in_a2 and same_outcome and earlier_or_equal:
                category = "redundant_knowledge_and_outcome"
            elif known_in_a2 and same_outcome:
                category = "uplink_bought_time_same_outcome"
            elif known_in_a2 and not same_outcome:
                category = "known_in_silence_but_different_outcome"
            elif not same_outcome:
                category = "load_bearing_outcome_only_with_uplink"
            else:
                category = "absent_from_both_arms"
            categories[category] += 1
            bytes_by_category[category] += cost["cost_bytes"]
            payload_by_category[category] += cost["payload_bytes"]
            seed_row["categories"][category] = seed_row["categories"].get(category, 0) + 1
            if len(examples[category]) < 4:
                examples[category].append({
                    "seed": seed, "target": target, "outcome_A1": info["outcome"],
                    "outcome_A2": outcome_a2, "knowledge_A1": know_a1.get(target),
                    "knowledge_A2": know_a2.get(target),
                    "cost_bytes": cost["cost_bytes"]})
        per_seed_rows.append({"seed": seed,
                              "categories": dict(seed_row["categories"]),
                              "strictly_removable_bytes": sum(
                                  costs.get(t, {"cost_bytes": 0})["cost_bytes"]
                                  for t, info in out_a1.items()
                                  if target_removable_strictly(t, info, out_a2, know_a1,
                                                               know_a2))})
    total_bytes = sum(bytes_by_category.values())
    strictly_removable = sum(row["strictly_removable_bytes"] for row in per_seed_rows)
    permissive = bytes_by_category["redundant_knowledge_and_outcome"] \
        + bytes_by_category["uplink_bought_time_same_outcome"]
    return {
        "method": ("paired by seed, target by target: arm A1 (immediate, lossless) against arm "
                   "A2 (always_hold). A2 is the only archived deployed-path run where the uplink "
                   "sent nothing, so it is the counterfactual the archive actually supports."),
        "seeds_paired": len(paired),
        "targets_compared": sum(categories.values()),
        "category_definitions": {
            "redundant_knowledge_and_outcome": (
                "some robot knew this target with no uplink at a step no later than with it, "
                "and the task ended the same way — the emission carried nothing"),
            "uplink_bought_time_same_outcome": (
                "known without the uplink too, but later; the outcome happened to survive the "
                "delay"),
            "known_in_silence_but_different_outcome": (
                "the target was still known in silence but the task ended differently — the "
                "uplink changed the outcome"),
            "load_bearing_outcome_only_with_uplink": (
                "never known at all in silence — the report was necessary for the task to exist "
                "for any robot"),
            "absent_from_both_arms": "residual bucket, expected empty"},
        "counts": dict(categories),
        "bytes_by_category": dict(bytes_by_category),
        "byte_share": {name: c19.ratio(value, total_bytes)
                       for name, value in sorted(bytes_by_category.items())},
        "per_recipient_payload_by_category": dict(payload_by_category),
        "savings_ceiling": {
            "strict_bytes_per_episode": c19.ratio(strictly_removable, len(paired)),
            "strict_share_of_bill": c19.ratio(strictly_removable, total_bytes),
            "permissive_bytes_per_episode": c19.ratio(permissive, len(paired)),
            "permissive_share_of_bill": c19.ratio(permissive, total_bytes),
            "note": ("strict = only emissions redundant in both knowledge and outcome; "
                     "permissive additionally counts emissions where silence delayed knowledge "
                     "but the recorded outcome survived. The permissive figure is NOT a savings "
                     "estimate: A2's robots moved on a different trajectory precisely because "
                     "they were not told, so 'later but still on time' is not a stable "
                     "counterfactual. Only the strict figure may be quoted, and only as a "
                     "ceiling on what a controller could aim at.")},
        "examples": {name: examples[name] for name in sorted(examples)},
        "per_seed_rows": per_seed_rows,
        "emission_costs_by_seed": emission_cost}


def target_removable_strictly(target: str, info_a1: dict[str, Any], out_a2: dict[str, Any],
                              know_a1: dict[str, dict[str, Any]],
                              know_a2: dict[str, dict[str, Any]]) -> bool:
    """The strict redundancy rule, factored out so the count and the total cannot drift."""
    second = out_a2.get(target)
    if second is None or target not in know_a2:
        return False
    if second["outcome"] != info_a1["outcome"]:
        return False
    return know_a2[target]["first_step"] <= know_a1.get(target, {}).get("first_step", 10 ** 9)


def per_packet_price() -> dict[str, Any]:
    """The price of one report, measured across the deployed archive."""
    payloads, costs, subscriber_counts = [], [], set()
    for arm in R16_ARMS:
        for seed in data_rows(arm):
            for record in emissions(arm, seed):
                payloads.append(int(record["payload_bytes"]))
                costs.append(int(record["cost_bytes"]))
                subscriber_counts.add(int(record["subscribers"]))
    if not payloads:
        return {}
    payloads.sort()
    costs.sort()
    return {"emissions": len(payloads),
            "payload_bytes": {"min": payloads[0], "median": payloads[len(payloads) // 2],
                              "mean": c19.ratio(sum(payloads), len(payloads)),
                              "max": payloads[-1]},
            "cost_bytes": {"min": costs[0], "median": costs[len(costs) // 2],
                           "mean": c19.ratio(sum(costs), len(costs)), "max": costs[-1]},
            "subscribers_observed": sorted(subscriber_counts),
            "formula": "cost_bytes = payload_bytes x subscribers, billed per attempt"}


def quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    ordered = sorted(values)
    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    return {"n": len(ordered), "min": round(ordered[0], 3), "p10": round(at(0.10), 3),
            "median": round(at(0.50), 3), "p90": round(at(0.90), 3),
            "max": round(ordered[-1], 3), "mean": round(sum(ordered) / len(ordered), 3)}


def deadline_slack() -> dict[str, Any]:
    """How much reporting delay the frozen deployed difficulty can absorb.

    This is the number a tolerance on task effect has to be written against: a report that is
    late by ``slack`` seconds costs nothing, and ``slack`` is measured here from the archive
    rather than assumed.
    """
    horizon: list[float] = []
    duration: list[float] = []
    late_completed: list[float] = []
    failures: Counter[str] = Counter()
    for seed, row in sorted(data_rows("A1").items()):
        targets, _ = per_target("A1", seed)
        for info in targets.values():
            horizon.append(info["deadline"] - info["discovered_at"])
            failures[info["outcome"]] += 1
            if info["completed_at"] is not None:
                taken = info["completed_at"] - info["discovered_at"]
                duration.append(taken)
                late_completed.append(info["deadline"] - info["completed_at"])
    return {"unit": "秒（dt = 1.0，1 步 = 1 拍）",
            "arm": "A1 deployed, immediate, lossless, 60 seeds",
            "deadline_horizon_per_task": quantiles(horizon),
            "detection_to_completion_duration": quantiles(duration),
            "remaining_slack_at_completion": quantiles(late_completed),
            "outcomes": dict(failures),
            "reading": ("a report can be late by the slack at completion without changing that "
                        "task's outcome; the distribution of that slack is what a pilot's "
                        "tolerance has to be set against, and the near-empty failure count is "
                        "why the frozen difficulty gives a cost constraint almost no contrast")}


def main() -> int:
    payload = {
        "round": "19", "phase": "step 2 — reuse deployed data, no new episode",
        "sources": {
            "deployed_arms": "runs/comm_info_path_validation_20260923/{rows,logs}/data_{A1..A4}_s*.json[l]",
            "seeded_by": "round 16 four-arm matrix, 60 paired seeds 1100000..1100059",
            "isolated_rules_cited_from": "runs/comm_task_pressure_learnability_20260922/"
                                         "RULE_MATRIX_g12.json and runs/comm_research_closure_20260923/",
            "round_18_cost": "runs/comm_cost_objective_design_20260926/ROUND18_ARCHIVE_CLOSURE.json"},
        "cost_model_from_source": {
            "pricer": "integrated_sar/comm.py:97-123 estimate_message_bytes (payload only)",
            "charged_at": "integrated_sar/comm.py:894-898, 953, 974 (payload x subscribers, per send event)",
            "budget": "integrated_sar/comm_budget.py:3-16, 83-94 (hard credit; a lost packet is billed in full)",
            "ack": "integrated_sar/comm_ack.py:136-168 (separate reverse ledger, never capped, does not share the credit)",
            "unbilled": ["PILOC sensing and DetectionEvent construction",
                         "event bus fan-out and TaskRegistry ingestion",
                         "registry polling by the sender",
                         "coordinator advice -> allocator logits",
                         "assignment commands -> executor",
                         "TaskEnv's own line-of-sight refresh and one-hop gossip",
                         "a send the budget refused"]},
        "link_coverage": {
            "only_billed_boundary": "sending node -> rescue robots' belief (the uplink)",
            "free_knowledge_channels": ["env line-of-sight truth refresh "
                                        "(task_allocation/old_task_env.py:477-492)",
                                        "robot one-hop gossip (old_task_env.py:494-512, "
                                        "gated by gossip_interval)"],
            "consequence": ("the bill covers one edge only, while the receivers can learn the "
                            "same fact through two unbilled environment channels")},
        "deployed_arm_table": arm_summary(),
        "redundancy_against_the_silence_counterfactual": redundant_share(),
        "deadline_slack_on_the_deployed_path": deadline_slack(),
        "price_of_one_report": per_packet_price(),
        "what_is_not_on_disk": {
            "budgeted_deployed_path": (
                "no archived run combines an enforced --comm-budget-bytes with the deployed "
                "information path (gossip_interval=2, comm_range=0.3). Round 16 is deployed but "
                "never binding (budget_refused_sends == 0 everywhere); rounds 12/15 are budgeted "
                "but isolated."),
            "interior_cost_points": (
                "the deployed archive holds two cost points only: 0 B (A2) and 14 573.3 B (A1). "
                "A cost-minimising controller needs the interior, and the interior cannot be "
                "obtained offline because silencing a report changes which robot is in line of "
                "sight later, which is what the unbilled channel depends on."),
            "recipient_heterogeneity": (
                "subscribers is a single scalar for the whole step "
                "(integrated_sar/comm_policy.py:112); per-recipient pricing does not exist, so "
                "'address fewer robots' is not an available lever (round 17: "
                "global_fanout_is_only_a_scalar)."),
            "the_two_free_channels_separately": (
                "the line-of-sight truth refresh and the one-hop gossip live in the same function "
                "(old_task_env.py:477-512) behind the same interval gate (:1107-1110), and every "
                "archived arm either leaves both on or switches both off. Which of the two "
                "carries the redundant knowledge is therefore not answerable from the archive, "
                "and the savings a controller could chase cannot be attributed to one channel.")},
    }
    c19.write_json("COMM_COST_COVERAGE.json", payload)
    coverage = payload["redundancy_against_the_silence_counterfactual"]
    print(json.dumps({
        "counts": coverage["counts"],
        "byte_share": coverage["byte_share"],
        "ceiling": coverage["savings_ceiling"],
        "value_of_uplink": payload["deployed_arm_table"]["derived"]["value_of_the_uplink_C1"],
        "per_ep_bytes": {arm: payload["deployed_arm_table"][arm]["per_episode"]["airtime_bytes"]
                         for arm in R16_ARMS if arm in payload["deployed_arm_table"]},
        "per_ep_on_time": {arm: payload["deployed_arm_table"][arm]["per_episode"]["on_time"]
                           for arm in R16_ARMS if arm in payload["deployed_arm_table"]},
        "first_reports_are_whole_bill":
            payload["deployed_arm_table"]["derived"]["first_reports_are_the_whole_bill"],
        "budget_never_enforced":
            payload["deployed_arm_table"]["derived"]["budget_was_never_enforced"],
        "slack": payload["deadline_slack_on_the_deployed_path"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
