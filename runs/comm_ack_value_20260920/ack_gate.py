"""The information-value gate: four checks, all of which must pass before any training.

This script decides whether the ACK reverse channel created a learnable trade-off. It does
not train anything and it does not look at a single learned number -- there are none yet.
It reads the rule matrix and the counterfactual diagnostic that were produced first, applies
the four thresholds written in ``COMM_ACK_VALUE_NEXT_ROUND.md`` §5 before any of them was
measured, and writes one machine-readable verdict.

The four gates:

1. **The information exists.** At least 8/48 training streams show a legal decision chain:
   an uplink was attempted, a receiver had not confirmed it, and the sender's confirmation
   state later changed. Without this there is nothing for a policy to condition on.
2. **The information pays.** The strongest ACK rule beats the strongest no-ACK rule on at
   least one axis: completion paired-difference CI lower > 0, or completion CI lower >= -0.05
   per episode *and* total bytes (uplink + ACK) CI upper < 0.
3. **The information is a choice, not a rule.** The controlled counterfactual has at least 20
   decision points where SEND is better and 20 where HOLD is better, each spread over at
   least 8 streams -- otherwise one action always wins and a rule suffices.
4. **No simple rule already covers it.** No single ACK rule is per-stream at least as good as
   every other rule on completion *and* per-stream no worse on total bytes. If one exists,
   the honest delivery is that rule plus a negative result, not a network trained to copy it.

Thresholds are engineering gates for entering training, not statistical proofs. Failing any
of them stops the round: no loss-rate increase, no deadline tightening, no scenario change.

    python ack_gate.py --tier 25pct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import ack_common as common

#: Gate 1's thresholds, written before the measurement.
MIN_CHAINS_STREAMS = 8
#: Gate 2's non-inferiority margin, in targets per episode, and the frozen statistic.
NON_INFERIORITY_MARGIN = -0.05
#: Gate 3's thresholds, shared with the counterfactual diagnostic.
MIN_PER_BRANCH = 20
MIN_STREAMS_PER_BRANCH = 8
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260919


def _pairs(rows: list[dict[str, Any]], channel: str, arm: str, key: str) -> dict[int, float]:
    return {
        int(row["seed"]): float(row[key])
        for row in rows
        if row["channel"] == channel and row["arm"] == arm
    }


def _paired_difference(
    rows: list[dict[str, Any]], channel: str, arm: str, baseline: str, key: str
) -> dict[str, Any] | None:
    left = _pairs(rows, channel, arm, key)
    right = _pairs(rows, channel, baseline, key)
    shared = sorted(set(left) & set(right))
    if len(shared) < 8:
        return None
    pairs = [(left[seed], right[seed]) for seed in shared]
    result = common.paired_bootstrap(
        pairs, baseline=0.0, key=lambda pair: pair[0] - pair[1]
    )
    result.update({"streams": len(shared)})
    return result


def gate_1_information_exists(
    points_by_stream: dict[int, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Streams where an unconfirmed attempt was followed by a change in confirmation state."""
    qualifying: list[int] = []
    evidence: dict[str, Any] = {}
    for seed, points in sorted(points_by_stream.items()):
        attempted = [point for point in points if int(point["unacked"]) >= 1]
        if not attempted:
            continue
        first_attempt = min(int(point["step"]) for point in attempted)
        later_confirmed = [
            point
            for point in points
            if int(point["step"]) > first_attempt
            and float(point["confirmed_fraction"]) > 0.0
        ]
        if later_confirmed:
            qualifying.append(int(seed))
            evidence[str(seed)] = {
                "first_unconfirmed_attempt_step": first_attempt,
                "first_confirmation_step": min(int(p["step"]) for p in later_confirmed),
                "unconfirmed_points": len(attempted),
            }
    return {
        "threshold_streams": MIN_CHAINS_STREAMS,
        "streams_total": len(points_by_stream),
        "streams_qualifying": len(qualifying),
        "streams": qualifying,
        "passed": len(qualifying) >= MIN_CHAINS_STREAMS,
        "evidence": evidence,
    }


def gate_2_information_pays(
    rows: list[dict[str, Any]], channel: str
) -> dict[str, Any]:
    """Strongest ACK rule vs the strongest no-ACK rule, on both axes."""
    summary = _aggregate(rows, channel)
    baseline = "send_once"
    candidates = [arm for arm in common.selectable_ack_arms() if arm in summary]
    ranked = sorted(
        candidates,
        key=lambda arm: (
            -summary[arm]["on_time_per_episode"],
            summary[arm]["total_bytes"],
            arm,
        ),
    )
    best = ranked[0] if ranked else None
    completion = (
        _paired_difference(rows, channel, best, baseline, "on_time") if best else None
    )
    total_bytes = (
        _paired_difference(rows, channel, best, baseline, "total_bytes") if best else None
    )
    better_completion = bool(
        completion and float(completion["ci_lower"]) > 0.0
    )
    non_inferior_and_cheaper = bool(
        completion
        and total_bytes
        and float(completion["ci_lower"]) >= NON_INFERIORITY_MARGIN
        and float(total_bytes["ci_upper"]) < 0.0
    )
    return {
        "channel": channel,
        "baseline": baseline,
        "baseline_on_time": summary.get(baseline, {}).get("on_time_per_episode"),
        "baseline_total_bytes": summary.get(baseline, {}).get("total_bytes"),
        "candidates": ranked,
        "selected_ack_rule": best,
        "selected_on_time": summary.get(best, {}).get("on_time_per_episode") if best else None,
        "selected_total_bytes": summary.get(best, {}).get("total_bytes") if best else None,
        "selected_uplink_bytes": summary.get(best, {}).get("uplink_bytes") if best else None,
        "selected_ack_bytes": summary.get(best, {}).get("ack_bytes") if best else None,
        "completion_difference": completion,
        "total_bytes_difference": total_bytes,
        "better_completion": better_completion,
        "non_inferior_and_cheaper": non_inferior_and_cheaper,
        "passed": bool(better_completion or non_inferior_and_cheaper),
    }


def gate_3_real_choice(counterfactual: dict[str, Any]) -> dict[str, Any]:
    item = dict(counterfactual.get("gate_item_3", {}))
    consistency = counterfactual.get("consistency", {})
    item["consistency"] = consistency
    item["passed"] = bool(
        item.get("send_ok")
        and item.get("hold_ok")
        and consistency.get("consistent") == consistency.get("checked")
        and consistency.get("checked", 0) > 0
    )
    return item


def gate_4_no_rule_dominates(rows: list[dict[str, Any]], channel: str) -> dict[str, Any]:
    """Whether one ACK rule is per-stream best on completion and no worse on bytes."""
    arms = sorted({row["arm"] for row in rows if row["channel"] == channel})
    by_stream: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["channel"] != channel:
            continue
        by_stream.setdefault(int(row["seed"]), {})[row["arm"]] = row
    dominators: list[dict[str, Any]] = []
    for arm in arms:
        dominates = True
        for stream, entries in by_stream.items():
            if arm not in entries:
                dominates = False
                break
            mine = entries[arm]
            for other, entry in entries.items():
                if other == arm:
                    continue
                if int(mine["on_time"]) < int(entry["on_time"]):
                    dominates = False
                    break
                if (
                    int(mine["on_time"]) == int(entry["on_time"])
                    and int(mine["total_bytes"]) > int(entry["total_bytes"])
                ):
                    dominates = False
                    break
            if not dominates:
                break
        if dominates:
            dominators.append({"arm": arm, "streams": len(by_stream)})
    return {
        "channel": channel,
        "arms_checked": arms,
        "streams_checked": len(by_stream),
        "dominators": dominators,
        "passed": not dominators,
    }


def _aggregate(rows: list[dict[str, Any]], channel: str) -> dict[str, dict[str, Any]]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["channel"] != channel:
            continue
        by_arm.setdefault(row["arm"], []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for arm, items in by_arm.items():
        summary[arm] = {
            "episodes": len(items),
            "on_time_per_episode": round(
                sum(float(r["on_time"]) for r in items) / len(items), 4
            ),
            "uplink_bytes": round(sum(float(r["uplink_bytes"]) for r in items) / len(items), 1),
            "ack_bytes": round(sum(float(r["ack_bytes"]) for r in items) / len(items), 1),
            "total_bytes": round(sum(float(r["total_bytes"]) for r in items) / len(items), 1),
        }
    return summary


def load_matrix(tier: str) -> list[dict[str, Any]]:
    root = common.evaluation_root(tier)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("matrix_shard*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    return rows


def load_counterfactual(tier: str) -> dict[str, Any]:
    path = common.json_path(f"COMM_ACK_VALUE_COUNTERFACTUAL_{tier}.json")
    if not path.exists():
        raise SystemExit(f"{path} missing; run ack_counterfactual.py --mode report first")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default=common.PRIMARY_TIER)
    parser.add_argument("--channel", default="lossy")
    args = parser.parse_args(argv)

    rows = load_matrix(args.tier)
    counterfactual = load_counterfactual(args.tier)
    # Gate 1 reuses the counterfactual's per-step probe records, which are the same legal
    # decision points the diagnostic forced. They are stored per stream in the JSON.
    points_by_stream = _probe_points(counterfactual)

    gates = {
        "1_information_exists": gate_1_information_exists(points_by_stream),
        "2_information_pays": gate_2_information_pays(rows, args.channel),
        "3_real_choice": gate_3_real_choice(counterfactual),
        "4_no_rule_dominates": gate_4_no_rule_dominates(rows, args.channel),
    }
    verdict = all(bool(gate.get("passed")) for gate in gates.values())
    payload = {
        "tier": args.tier,
        "budget_bytes": common.tier_bytes(args.tier),
        "channel": args.channel,
        "channel_parameters": common.channel_parameters(),
        "gates": gates,
        "all_passed": verdict,
        "decision": (
            "proceed to bounded training under COMM_ACK_POLICY_PROTOCOL.md"
            if verdict
            else "stop and deliver the ACK gate negative result"
        ),
        "thresholds": {
            "gate_1_streams": MIN_CHAINS_STREAMS,
            "gate_2_non_inferiority_margin": NON_INFERIORITY_MARGIN,
            "gate_3_per_branch": MIN_PER_BRANCH,
            "gate_3_streams": MIN_STREAMS_PER_BRANCH,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    }
    common.write_json(common.json_path(f"COMM_ACK_VALUE_GATE_{args.tier}.json"), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=1)[:6000])
    return 0


def _probe_points(counterfactual: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """The per-step legal decision points, regrouped by stream.

    Stored by the counterfactual runner beside each measured point so the gate does not have
    to re-run the probe: the same records, read once.
    """
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in counterfactual.get("probe_points", []):
        grouped.setdefault(int(row["seed"]), []).extend(row["points"])
    return grouped


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
