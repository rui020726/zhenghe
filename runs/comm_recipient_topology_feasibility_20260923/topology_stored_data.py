"""What the stored episodes actually say about subscriber counts and how a message is priced.

Run:  python topology_stored_data.py    # writes COMM_RECIPIENT_TOPOLOGY_STORED_DATA.json

The cost formula is `payload_bytes x subscriber_count`, so before any recipient-selection axis can be
considered, the second factor has to be shown to be a quantity that varies. This reads every send
record in the 576 audited episodes and asks, in order: does the count vary at all, does the pricing
identity close, and how much of the observed quote spread could the fan-out term explain.

No episode is run. Where a field is absent from the transcripts it is reported absent rather than
inferred from the formula that would have produced it.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from typing import Any

import topology_common as T

CALLS = T.install_env_tripwire()


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def classify(record: dict[str, Any], first_seen: dict[str, int]) -> str:
    """first report / refresh, by the audited round's own definition (``comm.py:802``).

    ``retry`` is a separate axis, not a third value here: it is keyed on ``revision > 1`` and is
    reported alongside, because a retry of a first report is both things at once.
    """
    target = str(record["target"])
    step = int(record["step"])
    earlier = first_seen.get(target)
    if record.get("first"):
        if earlier is not None and earlier < step:
            # The flag says first but an earlier send exists; keep the class by the flag and count
            # the disagreement rather than silently preferring one source.
            record["first_flag_conflict"] = True
        kind = "first_report"
    else:
        kind = "refresh"
    first_seen.setdefault(target, step)
    return kind


def analyse_layer(layer: str) -> dict[str, Any]:
    sends: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    per_episode_counts: dict[tuple[str, int], set[int]] = defaultdict(set)
    per_target_counts: dict[str, set[int]] = defaultdict(set)
    cells_with_sends = 0
    cells_seen = 0
    identity_violations: list[dict[str, Any]] = []
    missing_subscriber_field = 0
    first_flag_conflicts = 0

    for row in T.rows_of(layer):
        arm = str(row["arm"])
        seed = int(row["seed"])
        cells_seen += 1
        transcript = T.read_transcript(layer, arm, seed)
        records = [r for r in transcript if r.get("kind") == "send"]
        if records:
            cells_with_sends += 1
        first_seen: dict[str, int] = {}
        for record in sorted(records, key=lambda r: (int(r["step"]), str(r["target"]))):
            if "subscribers" not in record:
                missing_subscriber_field += 1
                continue
            count = int(record["subscribers"])
            payload = int(record.get("payload_bytes") or 0)
            cost = int(record.get("cost_bytes") or 0)
            item = {
                "arm": arm,
                "seed": seed,
                "step": int(record["step"]),
                "target": str(record["target"]),
                "subscribers": count,
                "payload_bytes": payload,
                "cost_bytes": cost,
                "revision": int(record.get("revision") or 1),
                "first": bool(record.get("first")),
                "class": classify(record, first_seen),
            }
            first_flag_conflicts += int(record.get("first_flag_conflict", False))
            sends.append(item)
            per_episode_counts[(arm, seed)].add(count)
            per_target_counts[(arm, seed, item["target"])].add(count)
            if payload * count != cost:
                identity_violations.append(
                    {"arm": arm, "seed": seed, "step": item["step"], "target": item["target"],
                     "payload": payload, "subscribers": count, "cost": cost,
                     "expected": payload * count}
                )
        for record in transcript:
            if record.get("kind") == "refused":
                refusals.append(
                    {"arm": arm, "seed": seed, "step": int(record["step"]),
                     "cost_bytes": int(record.get("cost_bytes") or 0),
                     "has_subscribers_field": "subscribers" in record}
                )

    counts = [item["subscribers"] for item in sends]
    payloads = [float(item["payload_bytes"]) for item in sends]
    costs = [float(item["cost_bytes"]) for item in sends]
    by_class: dict[str, list[int]] = defaultdict(list)
    for item in sends:
        by_class[item["class"]].append(item["subscribers"])
        # Reported as its own grouping rather than a third class: a retry is a re-send of
        # something that is already either a first report or a refresh.
        if item["revision"] > 1:
            by_class["retry(revision>1)"].append(item["subscribers"])
        else:
            by_class["first_attempt(revision==1)"].append(item["subscribers"])

    distinct_counts = sorted(set(counts))
    fanout_variance = _variance([float(value) for value in counts])
    cost_variance = _variance(costs)
    # cost = payload * k  =>  log cost = log payload + log k. With k constant the second term
    # carries no variance at all, so the whole spread of quotes belongs to the payload.
    log_cost = [math.log(value) for value in costs if value > 0]
    log_payload = [math.log(value) for value in payloads if value > 0]
    log_k = [math.log(value) for value in counts if value > 0]

    return {
        "layer": layer,
        "cells_seen": cells_seen,
        "cells_with_at_least_one_send": cells_with_sends,
        "send_records": len(sends),
        "refused_records": len(refusals),
        "refused_records_carrying_subscriber_count": sum(
            1 for record in refusals if record["has_subscribers_field"]
        ),
        "sends_missing_subscriber_field": missing_subscriber_field,
        "pricing_identity_violations": len(identity_violations),
        "pricing_identity_sample": identity_violations[:5],
        "first_flag_conflicts": first_flag_conflicts,
        "subscriber_count": {
            "distinct_values": distinct_counts,
            "min": min(counts) if counts else None,
            "median": T.quantiles([float(c) for c in counts])["median"],
            "max": max(counts) if counts else None,
            "variance": round(fanout_variance, 8),
            "episodes_with_a_constant_count": sum(
                1 for values in per_episode_counts.values() if len(values) == 1
            ),
            "episodes_with_varying_count": sum(
                1 for values in per_episode_counts.values() if len(values) > 1
            ),
            "episodes_observed": len(per_episode_counts),
            "distinct_targets_observed": len(per_target_counts),
            "targets_sharing_one_count": sum(
                1 for values in per_target_counts.values() if len(values) == 1
            ),
            "varies_across_targets_within_an_episode": any(
                len(values) > 1 for values in per_target_counts.values()
            ),
        },
        "by_message_class": {
            kind: {
                "records": len(values),
                "distinct_counts": sorted(set(values)),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            for kind, values in sorted(by_class.items())
        },
        "retried_records": sum(1 for item in sends if item["revision"] > 1),
        "revision_distribution": dict(
            sorted(Counter(item["revision"] for item in sends).items())
        ),
        "payload_bytes": T.quantiles(payloads),
        "cost_bytes": T.quantiles(costs),
        "cost_decomposition": {
            "cost_equals_payload_times_count_on_every_record": len(identity_violations) == 0,
            "fanout_variance": round(fanout_variance, 8),
            "cost_variance": round(cost_variance, 6),
            "share_of_log_cost_variance_from_fanout": _ratio(
                _variance(log_k), _variance(log_cost)
            ),
            "share_of_log_cost_variance_from_payload": _ratio(
                _variance(log_payload), _variance(log_cost)
            ),
            "cost_range": (max(costs) - min(costs)) if costs else None,
            "payload_range": (max(payloads) - min(payloads)) if payloads else None,
            "cost_range_divided_by_fanout_equals_payload_range": bool(
                costs and payloads and abs(
                    (max(costs) - min(costs)) / (distinct_counts[0] if distinct_counts else 1)
                    - (max(payloads) - min(payloads))
                ) < 1e-6
            ),
        },
    }


def main(argv: list[str]) -> int:
    layers = {layer: analyse_layer(layer) for layer in T.LAYERS}
    all_counts = sorted(
        {
            int(value)
            for data in layers.values()
            for value in data["subscriber_count"]["distinct_values"]
        }
    )
    result = {
        "round": "comm_recipient_topology_feasibility",
        "question": "is subscriber_count a real per-message quantity, or one constant "
                    "multiplied everywhere",
        "layers": layers,
        "aggregate": {
            "send_records": sum(data["send_records"] for data in layers.values()),
            "refused_records": sum(data["refused_records"] for data in layers.values()),
            "pricing_identity_violations": sum(
                data["pricing_identity_violations"] for data in layers.values()
            ),
            "distinct_subscriber_counts_across_all_layers": all_counts,
            "varying": len(all_counts) > 1,
            "episodes_with_varying_count": sum(
                data["subscriber_count"]["episodes_with_varying_count"]
                for data in layers.values()
            ),
            "episodes_observed": sum(
                data["subscriber_count"]["episodes_observed"] for data in layers.values()
            ),
        },
        "environment_tripwire_calls": list(CALLS),
    }
    T.write_json(T.HERE / "COMM_RECIPIENT_TOPOLOGY_STORED_DATA.json", result)
    print(json.dumps(
        {
            "aggregate": result["aggregate"],
            "per_layer_counts": {
                layer: {
                    "distinct": data["subscriber_count"]["distinct_values"],
                    "episodes_constant": data["subscriber_count"]["episodes_with_a_constant_count"],
                    "episodes_varying": data["subscriber_count"]["episodes_with_varying_count"],
                    "identity_violations": data["pricing_identity_violations"],
                    "fanout_share_of_log_variance": data["cost_decomposition"][
                        "share_of_log_cost_variance_from_fanout"
                    ],
                    "payload_share_of_log_variance": data["cost_decomposition"][
                        "share_of_log_cost_variance_from_payload"
                    ],
                    "by_class": {
                        kind: value["distinct_counts"]
                        for kind, value in data["by_message_class"].items()
                    },
                }
                for layer, data in layers.items()
            },
            "tripwire_calls": result["environment_tripwire_calls"],
        },
        ensure_ascii=False,
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
