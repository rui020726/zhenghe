"""What a real recipient rule would do to the cost geometry -- projected, not replayed.

Run:  python topology_projection.py     # writes COMM_RECIPIENT_TOPOLOGY_PROJECTION.json

Label on every number here: **`offline_recipient_cost_projection`**. It re-prices stored messages
under a different recipient rule using only fields already recorded at send time. It does not modify
any trace, does not re-run an episode, and cannot say a task outcome would change.

Two recipient rules are testable from stored data at all, because they are the only ones whose
recipient set is named in the records:

* **coalition** -- send to the robots currently committed to that target, i.e. the message's own
  `members` field, read at send time.
* **broadcast** -- the current rule, everyone, as the reference.

The others (capability-and-deadline matching, radio neighbourhood, coordinator-only) are *not*
constructible from stored state: the relay is built with positions deliberately excluded
(`joint_driver.py:945-947`, invariant 10), and no per-message capability or range view is recorded.
They are reported as not-constructible rather than approximated with a random or synthetic set.

The interesting quantity is not whether coalition routing makes prices vary -- it obviously does --
but **which messages get priced up and which get priced to nothing.**
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from typing import Any

import topology_common as T

CALLS = T.install_env_tripwire()

LABEL = "offline_recipient_cost_projection"
BROADCAST_FANOUT = 6


def _quantiles(values: list[float]) -> dict[str, Any]:
    return T.quantiles(values)


def project(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-price the same messages under each rule and compare the resulting step geometry."""
    out: dict[str, Any] = {"label": LABEL, "messages": len(records)}
    for rule in ("broadcast", "coalition"):
        fanout: list[float] = []
        charges: list[float] = []
        degenerate: list[float] = []
        by_class: dict[str, list[float]] = defaultdict(list)
        for record in records:
            if rule == "broadcast":
                count = BROADCAST_FANOUT
            else:
                count = len(record["members"])
            fanout.append(float(count))
            charges.append(record["payload_bytes"] * count)
            by_class[record["class"]].append(float(count))
            if count == 0:
                degenerate.append(record["payload_bytes"])
        out[rule] = {
            "fanout": _quantiles(fanout),
            "fanout_distinct_values": sorted({int(value) for value in fanout}),
            "projected_charge_bytes": _quantiles(charges),
            "share_of_messages_with_empty_recipient_set": (
                round(len(degenerate) / len(records), 6) if records else None
            ),
            "bytes_that_would_never_be_sent_under_this_rule": int(sum(degenerate)),
            "fanout_by_message_class": {
                kind: {
                    "records": len(values),
                    "distinct": sorted({int(value) for value in values}),
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "median": _quantiles(values)["median"],
                }
                for kind, values in sorted(by_class.items())
            },
        }
    # The heterogeneity that actually matters is *within a step*: two candidates priced differently
    # is what forces a choice. A constant multiplier cancels out of the comparison.
    per_step: list[dict[str, float]] = []
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["layer"], record["seed"], record["step"])].append(record)
    for (_layer, _seed, _step), group in grouped.items():
        if len(group) < 2:
            continue
        for rule in ("broadcast", "coalition"):
            prices = [
                item["payload_bytes"] * (
                    BROADCAST_FANOUT if rule == "broadcast" else len(item["members"])
                )
                for item in group
            ]
            if not prices:
                continue
            mean = sum(prices) / len(prices)
            per_step.append(
                {
                    "rule": rule,
                    "spread": float(max(prices) - min(prices)),
                    "mean": mean,
                    "relative_spread": (max(prices) - min(prices)) / mean if mean else 0.0,
                }
            )
    for rule in ("broadcast", "coalition"):
        rows = [row for row in per_step if row["rule"] == rule]
        out[f"{rule}_within_step"] = {
            "steps_with_two_or_more_candidates": len(rows),
            "relative_spread_of_prices_within_a_step": _quantiles(
                [row["relative_spread"] for row in rows]
            ),
            "absolute_spread_bytes": _quantiles([row["spread"] for row in rows]),
        }
    return out


def first_report_choices(records: list[dict[str, Any]], rule: str) -> dict[str, Any]:
    """R4's actual question: do two *newly discovered* targets ever cost different amounts?

    Counted per g12 stream, among steps where at least two first-report candidates stand together.
    A rule that never puts two first reports in the same priced step cannot create a
    "which new target do we afford" decision, no matter how variable its prices are.
    """
    per_stream: dict[int, dict[str, Any]] = {}
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["class"] != "first_report":
            continue
        grouped[(record["seed"], record["step"])].append(record)
    for (seed, _step), group in grouped.items():
        if len(group) < 2:
            continue
        prices = [
            item["payload_bytes"] * (
                BROADCAST_FANOUT if rule == "broadcast" else len(item["members"])
            )
            for item in group
        ]
        bucket = per_stream.setdefault(
            seed, {"steps_with_two_first_reports": 0, "price_spreads": [], "empty_priced": 0}
        )
        bucket["steps_with_two_first_reports"] += 1
        bucket["price_spreads"].append(max(prices) - min(prices))
        if min(prices) == 0:
            bucket["empty_priced"] += 1
    return {
        "rule": rule,
        "streams_with_any_pair_of_first_reports": len(per_stream),
        "total_pairs_of_first_reports_in_one_step": sum(
            bucket["steps_with_two_first_reports"] for bucket in per_stream.values()
        ),
        "streams_where_two_first_reports_ever_cost_different_amounts": sum(
            1 for bucket in per_stream.values() if any(bucket["price_spreads"])
        ),
        "median_price_spread_between_two_first_reports": _quantiles(
            [
                float(spread)
                for bucket in per_stream.values()
                for spread in bucket["price_spreads"]
            ]
        )["median"],
        "steps_where_a_first_report_would_be_priced_to_zero": sum(
            bucket["empty_priced"] for bucket in per_stream.values()
        ),
    }


def main(argv: list[str]) -> int:
    records: list[dict[str, Any]] = []
    g12_records: list[dict[str, Any]] = []
    seen_classes: Counter[str] = Counter()
    for layer in T.LAYERS:
        for row in T.rows_of(layer):
            arm = str(row["arm"])
            seed = int(row["seed"])
            first_seen: dict[str, int] = {}
            for record in T.read_transcript(layer, arm, seed):
                if record.get("kind") != "send":
                    continue
                target = str(record["target"])
                emitted_before = target in first_seen
                kind = "first_report" if record.get("first") else "refresh"
                seen_classes[kind] += 1
                first_seen.setdefault(target, int(record["step"]))
                item = {
                    "layer": layer,
                    "arm": arm,
                    "seed": seed,
                    "step": int(record["step"]),
                    "target": target,
                    "class": kind,
                    "payload_bytes": int(record.get("payload_bytes") or 0),
                    "members": [str(member) for member in (record.get("members") or [])],
                    "subscribers_recorded": int(record.get("subscribers") or 0),
                }
                records.append(item)
                if layer == "g12":
                    g12_records.append(item)

    members_vs_subscribers = sum(
        1 for item in records if set(item["members"]) - {
            str(sub) for sub in range(BROADCAST_FANOUT)
        }
    )
    result = {
        "round": "comm_recipient_topology_feasibility",
        "label": LABEL,
        "not_a_replay": "no episode was run; no trace was modified; no outcome is claimed",
        "input": {
            "send_records": len(records),
            "g12_send_records": len(g12_records),
            "message_classes": dict(seen_classes),
            "member_ids_outside_the_subscriber_space": members_vs_subscribers,
        },
        "rules_testable_from_stored_state": ["broadcast", "coalition"],
        "rules_not_constructible_from_stored_state": {
            "capability_and_deadline_matching":
                "requires other robots' positions and schedules; the relay is built with positions "
                "excluded (joint_driver.py:945-947, invariant 10) and no such view is recorded",
            "radio_neighbourhood":
                "no per-robot position or range is recorded in the communication transcript",
            "coordinator_only":
                "there is no coordinator agent id in the subscriber space ({0..5} are rescue "
                "robots); this would be an architecture change, not a routing rule",
        },
        "projection_all_layers": project(records),
        "projection_g12": project(g12_records),
        "first_report_choice_g12": {
            rule: first_report_choices(g12_records, rule) for rule in ("broadcast", "coalition")
        },
        "environment_tripwire_calls": list(CALLS),
    }
    T.write_json(T.HERE / "COMM_RECIPIENT_TOPOLOGY_PROJECTION.json", result)
    print(json.dumps(
        {
            "input": result["input"],
            "g12_broadcast": result["projection_g12"]["broadcast"]["fanout"],
            "g12_coalition": result["projection_g12"]["coalition"]["fanout"],
            "g12_coalition_empty_share": result["projection_g12"]["coalition"][
                "share_of_messages_with_empty_recipient_set"
            ],
            "g12_coalition_fanout_by_class": result["projection_g12"]["coalition"][
                "fanout_by_message_class"
            ],
            "within_step_relative_spread": {
                rule: result["projection_g12"][f"{rule}_within_step"][
                    "relative_spread_of_prices_within_a_step"
                ]["median"]
                for rule in ("broadcast", "coalition")
            },
            "first_report_choice": result["first_report_choice_g12"],
            "tripwire_calls": result["environment_tripwire_calls"],
        },
        ensure_ascii=False,
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
