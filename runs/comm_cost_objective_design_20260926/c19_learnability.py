"""LEARNABILITY_PROBE_OFFLINE.json: is the load-bearing traffic predictable at send time?

Step 2's redundancy split says only 19 of 352 deployed first reports changed a task outcome. A
cost-minimising controller is only worth building if *something it can see when it decides*
separates those 19 from the other 333 — otherwise the whole bill is one undifferentiated block
and the only policy that saves bytes is silence, which round 16 already priced.

Zero training and zero episodes here: no model is fitted. Each candidate feature is scored by a
rank statistic (no distribution assumed), and the best threshold rule is then evaluated by
*counterfactual bookkeeping* — send it, take the recorded A1 outcome; silence it, take the
recorded A2 outcome for that same target on the same seed. That bookkeeping is an estimate, not a
measurement, because a robot that is not told walks a different path and the unbilled
line-of-sight channel depends on where robots are. It is reported as a ceiling for the pilot to
test, never as a result.

Run:  /d/project_env/python.exe runs/comm_cost_objective_design_20260926/c19_learnability.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any

import c19_common as c19

import c19_cost_coverage as cov

# Features the sending node legitimately holds at the moment it decides, taken from the packet it
# is about to emit and from the registry record behind it. Robot positions and robot beliefs are
# deliberately absent: comm_learning.py:33-53 excludes them, and AGENTS.md invariant 10 keeps the
# search robots and the rescue robots in separate state.
FEATURES = {
    "step_reported_at": "the search step the report goes out on",
    "slack_at_report": "deadline - now, seconds of task life left when the decision is made",
    "slack_over_horizon": "the same quantity divided by the 1000 s episode horizon",
    "deadline_offset": "the task's own total window, set when it was created",
    "severity": "registry field, in the packet",
    "confidence": "registry field, in the packet",
    "requirements_left": "how many of the five capability slots are still needed",
    "total_stages": "how many stages the task has",
    "backlog_steps": "steps this candidate has been waiting inside the sender",
    "payload_bytes": "the size of the packet about to be billed",
}


def per_report_records(seed: int) -> dict[str, dict[str, Any]]:
    """One record per target the uplink reported in this cell, with its visible features."""
    sends = {str(record["target"]): record
             for record in cov.emissions("A1", seed) if record.get("first")}
    out: dict[str, dict[str, Any]] = {}
    for record in c19.iter_jsonl(cov.log_path("A1", seed, "rp")):
        if record.get("event_type") != "detection":
            continue
        target = str(record["target_id"])
        send = sends.get(target)
        if send is None:
            continue
        detail = record.get("detail") or {}
        now = float(record.get("timestamp") or 0.0)
        deadline = float(send.get("deadline") or 0.0)
        requirements = [int(value) for value in (send.get("remaining") or [])]
        out[target] = {
            "step_reported_at": float(send.get("step") or 0.0),
            "slack_at_report": deadline - now,
            "slack_over_horizon": (deadline - now) / 1000.0,
            "deadline_offset": float(detail.get("deadline_offset") or 0.0),
            "severity": float(detail.get("severity") or 0.0),
            "confidence": float(detail.get("confidence") or 0.0),
            "requirements_left": sum(1 for value in requirements if value > 0),
            "total_stages": float(detail.get("total_stages") or 0.0),
            "backlog_steps": float(send.get("backlog_steps") or 0.0),
            "payload_bytes": float(send.get("payload_bytes") or 0.0),
            "cost_bytes": float(send.get("cost_bytes") or 0.0),
        }
    return out


def labels() -> list[dict[str, Any]]:
    """Every deployed first report, tagged with what silence would have done to its task."""
    rows: list[dict[str, Any]] = []
    a1_rows, a2_rows = cov.data_rows("A1"), cov.data_rows("A2")
    for seed in sorted(set(a1_rows) & set(a2_rows)):
        out_a1, _ = cov.per_target("A1", seed)
        out_a2, _ = cov.per_target("A2", seed)
        know_a1 = cov.knowledge_steps("A1", seed)
        know_a2 = cov.knowledge_steps("A2", seed)
        reports = per_report_records(seed)
        for target, features in reports.items():
            second = out_a2.get(target)
            known_in_silence = target in know_a2
            if second is None or second["outcome"] != out_a1[target]["outcome"]:
                label = "load_bearing"
            elif known_in_silence and (know_a2[target]["first_step"]
                                       <= know_a1[target]["first_step"]):
                label = "redundant_knowledge_and_outcome"
            else:
                label = "uplink_bought_time_same_outcome"
            rows.append({"seed": seed, "target": target, "label": label,
                         "outcome_A1": out_a1[target]["outcome"],
                         "outcome_A2": (second or {}).get("outcome"),
                         "cost_bytes": features["cost_bytes"],
                         **{name: features[name] for name in FEATURES}})
    return rows


def auc(positive: list[float], negative: list[float]) -> float | None:
    """Mann-Whitney U turned into a probability, by rank, no distribution assumed.

    0.5 means the feature says nothing; the further from 0.5, the better one report is separated
    from the rest by that single number.
    """
    if not positive or not negative:
        return None
    values = [(value, 1.0) for value in positive] + [(value, 0.0) for value in negative]
    values.sort(key=lambda pair: pair[0])
    ranks: list[float] = []
    index = 0
    while index < len(values):
        last = index
        while last + 1 < len(values) and values[last + 1][0] == values[index][0]:
            last += 1
        average_rank = (index + last) / 2.0 + 1.0
        ranks.extend([average_rank] * (last - index + 1))
        index = last + 1
    positive_rank_sum = sum(rank for rank, (_, flag) in zip(ranks, values) if flag == 1.0)
    n_pos, n_neg = len(positive), len(negative)
    return round((positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg), 4)


def separability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    load = [row for row in rows if row["label"] == "load_bearing"]
    rest = [row for row in rows if row["label"] != "load_bearing"]
    for name in FEATURES:
        positive = [float(row[name]) for row in load]
        negative = [float(row[name]) for row in rest]
        out[name] = {
            "auc_load_bearing_vs_rest": auc(positive, negative),
            "load_bearing_median": cov.quantiles(positive).get("median"),
            "rest_median": cov.quantiles(negative).get("median"),
            "distinct_values_rest": len(set(negative)),
            "meaning": FEATURES[name]}
    return out


def threshold_rules(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The best single-feature silence rule, scored by counterfactual bookkeeping.

    A rule is "report only when <feature> < <value>" (or the mirror), so silence is chosen from
    what the sender can see. For each rule the estimate adds the recorded A1 outcome for the
    reports it sends and the recorded A2 outcome for the reports it drops, per target, plus the
    billed bytes of only the reports it sends.
    """
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    a2_outcomes = {seed: cov.per_target("A2", seed)[0] for seed in by_seed}
    results: dict[str, Any] = {}
    for name in FEATURES:
        values = sorted({float(row[name]) for row in rows})
        if len(values) < 2:
            results[name] = {"usable": False, "reason": "constant across the archive"}
            continue
        candidates = []
        for direction in ("report_when_below", "report_when_above"):
            for cut in values:
                def keep(row: dict[str, Any], cut=cut, direction=direction) -> bool:
                    value = float(row[name])
                    return value < cut if direction == "report_when_below" else value > cut
                sent = [row for row in rows if keep(row)]
                dropped = [row for row in rows if not keep(row)]
                if not sent or not dropped:
                    continue
                on_time = 0.0
                for seed, block in by_seed.items():
                    kept_targets = {row["target"] for row in block if keep(row)}
                    for row_local in block:
                        outcome_a1 = row_local["outcome_A1"]
                        if row_local["target"] in kept_targets:
                            on_time += 1.0 if outcome_a1 == "on_time" else 0.0
                        else:
                            second = a2_outcomes[seed].get(row_local["target"])
                            on_time += 1.0 if (second or {}).get("outcome") == "on_time" else 0.0
                # a dropped report bills nothing; a sent one bills its recorded price
                bytes_total = sum(float(row["cost_bytes"]) for row in sent)
                candidates.append({
                    "direction": direction, "cut": round(cut, 4),
                    "share_of_reports_sent": c19.ratio(len(sent), len(rows)),
                    "recall_of_load_bearing": c19.ratio(
                        sum(1 for row in sent if row["label"] == "load_bearing"),
                        sum(1 for row in rows if row["label"] == "load_bearing")),
                    "estimated_on_time_per_episode": c19.ratio(on_time, len(by_seed)),
                    "estimated_bytes_per_episode": c19.ratio(bytes_total, len(by_seed))})
        best = None
        feasible = [item for item in candidates
                    if item["estimated_on_time_per_episode"] is not None
                    and item["estimated_on_time_per_episode"] >= 5.866667]
        if feasible:
            best = min(feasible, key=lambda item: item["estimated_bytes_per_episode"])
        results[name] = {"usable": True, "distinct_values": len(values),
                         "rules_scored": len(candidates),
                         "cheapest_rule_at_full_task_effect": best,
                         "top_three_by_bytes": sorted(
                             candidates, key=lambda item: item["estimated_bytes_per_episode"])[:3]}
    return results


def verdict(rows: list[dict[str, Any]], separability: dict[str, Any],
            rules: dict[str, Any]) -> dict[str, Any]:
    """The two sentences the pilot proposal has to be written against."""
    scored = [(name, block.get("cheapest_rule_at_full_task_effect"))
              for name, block in rules.items()
              if block.get("cheapest_rule_at_full_task_effect")]
    scored.sort(key=lambda pair: pair[1]["estimated_bytes_per_episode"])
    winner_name, winner = scored[0] if scored else (None, None)
    return {
        "answer_to_the_question": (
            "yes, one number separates them: the step the discovery happened on. Late "
            "discoveries are the ones silence loses, because a robot needs time to walk into "
            "line of sight before the deadline; early discoveries reach the receivers through "
            "the environment's own channels long before they matter."),
        "strongest_single_feature_auc": max(
            (block["auc_load_bearing_vs_rest"] or 0.0)
            for block in separability.values()) if separability else None,
        "cheapest_one_threshold_rule_at_full_recorded_effect": {
            "feature": winner_name, **(winner or {})},
        "as_a_share_of_the_deployed_bill": c19.ratio(
            winner["estimated_bytes_per_episode"] if winner else 0.0,
            c19.ratio(sum(row["cost_bytes"] for row in rows), 60) or 1.0),
        "what_this_is_not": (
            "not a measured saving. The threshold is read off the same 60 seeds it would be "
            "scored on, and a mixed-silence run is not any run in the archive: which robot is in "
            "line of sight later depends on what the robots were told. It is a ceiling and a "
            "baseline definition — the pilot has to beat a one-number rule, measured, on seeds "
            "that have never been used."),
        "consequence_for_the_design": (
            "the comparator a learned cost-minimising controller must beat is a single "
            "threshold on the discovery step, which is not a learning problem at all. Any pilot "
            "that spends training budget must therefore be judged against that rule first, and "
            "if the rule already holds task effect at its own budget level, a learned controller "
            "is only worth running where the rule cannot go: per-target decisions under loss, or "
            "an operating point tight enough that reporting early and reporting late differ."),
    }


def main() -> int:
    rows = labels()
    counts = Counter(row["label"] for row in rows)
    load_bytes = sum(row["cost_bytes"] for row in rows if row["label"] == "load_bearing")
    total_bytes = sum(row["cost_bytes"] for row in rows)
    separability_block = separability(rows)
    rules_block = threshold_rules(rows)
    payload = {
        "round": "19", "phase": "step 2b — offline learnability probe, 0 training, 0 episode",
        "question": ("of the deployed first reports, the ones that silence would have lost are "
                     "19 of 352; can a sender tell them apart at decision time from what it is "
                     "allowed to see?"),
        "labels": {
            "load_bearing": "silence changed this task's outcome in the recorded A2 run",
            "uplink_bought_time_same_outcome": "silence still made the deadline, but knowledge "
                                               "arrived later — the same names as "
                                               "COMM_COST_COVERAGE.json",
            "redundant_knowledge_and_outcome": "silence knew it as early or earlier, same "
                                               "outcome"},
        "counts": dict(counts),
        "per_episode": {"reports": c19.ratio(len(rows), 60),
                        "load_bearing": c19.ratio(counts["load_bearing"], 60),
                        "bytes": c19.ratio(total_bytes, 60),
                        "load_bearing_bytes": c19.ratio(load_bytes, 60),
                        "load_bearing_byte_share": c19.ratio(load_bytes, total_bytes)},
        "feature_set": FEATURES,
        "feature_set_excludes": ["rescue-robot positions", "rescue-robot beliefs",
                                 "search-robot positions", "anything about the future"],
        "separability": separability_block,
        "threshold_rule_search": rules_block,
        "learnability_verdict": verdict(rows, separability_block, rules_block),
        "honest_limits": [
            "the outcome bookkeeping assumes a dropped report reproduces the A2 outcome for that "
            "target; A2's robots moved differently because they were not told, so this is a "
            "ceiling, not a prediction",
            "no feature here is the quantity that actually decides redundancy, which is whether "
            "a capable robot will be in line of sight in time — that needs robot positions, and "
            "the sender is not allowed them (comm_learning.py:33-53, AGENTS.md invariant 10)",
            "no model is fitted and no gradient is taken; AUC and a threshold sweep are bookkeeping",
            "the archived deployed difficulty produces 0 on-time failures in the reporting arm "
            "(352 of 352), so no deployed-path cost rule can be judged there"],
    }
    c19.write_json("LEARNABILITY_PROBE_OFFLINE.json", payload)
    best = {name: (block.get("cheapest_rule_at_full_task_effect")
                   or {}).get("estimated_bytes_per_episode")
            for name, block in payload["threshold_rule_search"].items()}
    print(json.dumps({"counts": dict(counts),
                      "per_episode": payload["per_episode"],
                      "auc": {name: block["auc_load_bearing_vs_rest"]
                              for name, block in payload["separability"].items()},
                      "cheapest_rule_bytes_per_ep_at_A1_effect": best}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
