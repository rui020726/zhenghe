"""Reading one episode's transcript the way round 3 asks: who decided to send, and in what unit.

Two things live here and neither one runs an episode.

**Send-reason attribution.** Every attempt is charged to exactly one primary reason --
``policy`` (the sender asked for it), ``forced_maxwait`` (the shared silence bound ran out and
the transport spoke), or ``other`` (nothing else exists today; the bucket is there so a future
mandatory protocol event cannot silently join either one). The split is what tells a controller
that learned when to talk apart from one that learned to shut up and let the rule talk for it.
Deliveries and refusals are joined back to the send that produced them through
``(sent_step, target)``, which is unique because the relay emits at most once per target per
step -- so the same join gives a per-reason read of what each kind of send achieved, not just
what it cost.

**Units.** A number without a unit is a rumour, and last round's "+0.083" was one. Each row
therefore carries the completion measure three separate ways -- targets per episode, a
per-episode rate over what that episode detected, and a pooled rate over all targets -- and the
cost measure three ways too: total airtime, per emission batch, and per receiver packet.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

REASONS = ("policy", "forced_maxwait", "other")

#: Two readings of "the same thing twice". ``redundant`` is the sender putting a view on the air
#: that it had already sent for that target unchanged -- visible only from the send side, and the
#: term that makes the generation identity close, because such an emission answers no news.
#: ``duplicate`` is a receiver refusing a packet it already believed -- which a lost earlier
#: packet can cause even when the sender was not redundant. Both are reported.
BUCKET_KEYS = (
    "emissions",
    "attempts",
    "bytes",
    "delivered",
    "applied",
    "duplicate",
    "superseded",
    "unknown_task",
    "refused_by_writer",
    "refused",
    "refused_bytes",
    "redundant",
    "wait_steps",
)

#: The relay's own words for why a packet reached a robot and changed nothing. Anything outside
#: this table is counted as a writer refusal but named in ``unexplained_verdicts``, because a new
#: refusal class appearing in a transcript is what a silent default would hide.
_VERDITS = {
    "delivery_duplicate": "duplicate",
    "delivery_superseded": "superseded",
    "delivery_unknown_task": "unknown_task",
    "delivery_refused": "refused_by_writer",
}


def _bucket() -> dict[str, int]:
    return {key: 0 for key in BUCKET_KEYS}


def _per_reason_rollup(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    sends = [record for record in records if record["kind"] == "send"]
    deliveries = [record for record in records if record["kind"] == "deliver"]
    refusals = [record for record in records if record["kind"] == "refused"]
    buckets: dict[str, dict[str, int]] = {reason: _bucket() for reason in REASONS}
    unexplained: set[str] = set()
    # One emission per target per step, so a delivery's own send is identifiable: the cost and
    # the outcome of the *same* decision can then be read per reason instead of per arm.
    reason_of_send = {
        (int(send["step"]), str(send["target"])): str(send.get("send_reason", "policy"))
        for send in sends
    }

    def bucket_for(reason: str) -> dict[str, int]:
        return buckets.setdefault(reason, _bucket())

    last_revision: dict[str, int] = {}
    for send in sends:
        bucket = bucket_for(str(send.get("send_reason", "policy")))
        target = str(send["target"])
        revision = int(send["revision"])
        bucket["emissions"] += 1
        bucket["attempts"] += int(send["attempts"])
        bucket["bytes"] += int(send["cost_bytes"])
        bucket["wait_steps"] += max(0, int(send["step"]) - int(send["ready_step"]))
        # Same registry revision as this target's previous send: the view on the air is one that
        # was already said, so this emission answers no generated item.
        bucket["redundant"] += 1 if last_revision.get(target) == revision else 0
        last_revision[target] = revision
    for delivery in deliveries:
        bucket = bucket_for(reason_of_send.get((int(delivery["sent_step"]), str(delivery["target"])), "policy"))
        bucket["delivered"] += 1
        verdict = str(delivery.get("reason"))
        if delivery["accepted"]:
            bucket["applied"] += 1
        else:
            bucket[_VERDITS.get(verdict, "refused_by_writer")] += 1
            if verdict not in _VERDITS:
                unexplained.add(verdict)
    for refusal in refusals:
        bucket = bucket_for(str(refusal.get("send_reason", "policy")))
        bucket["refused"] += 1
        bucket["refused_bytes"] += int(refusal["cost_bytes"])
    for bucket in buckets.values():
        emissions = bucket["emissions"]
        bucket["mean_wait_steps"] = round(bucket.pop("wait_steps") / emissions, 3) if emissions else 0.0
    total_bytes = sum(bucket["bytes"] for bucket in buckets.values())
    return {
        "by_reason": buckets,
        "reasons_seen": sorted(buckets),
        "unexplained_verdicts": sorted(unexplained),
        "autonomous_share_of_bytes": round(buckets["policy"]["bytes"] / total_bytes, 5) if total_bytes else None,
        "subscribers": int(sends[0]["subscribers"]) if sends else 0,
    }


def _closure(counters: Mapping[str, Any], rollup: Mapping[str, Any], *, spent_bytes: int) -> dict[str, Any]:
    """Every identity the ledger has to satisfy, each with its measured residual.

    A residual is reported, never hidden: an accounting claim that only holds because somebody
    subtracted the difference is not a claim.
    """
    by_reason = rollup["by_reason"]
    subscribers = int(rollup["subscribers"])
    emissions = int(counters["emissions"])
    attempts = int(counters["transmission_attempts"])
    airtime = int(counters["airtime_bytes"])
    delivered = int(counters["packets_delivered"])
    refused_classes = (
        int(counters["deliveries_applied"])
        + int(counters.get("delivery_duplicate", 0))
        + int(counters.get("delivery_superseded", 0))
        + int(counters.get("delivery_unknown_task", 0))
        + int(counters.get("delivery_refused", 0))
    )
    generated = int(counters["generated_items"])
    coalesced = int(counters["coalesced_changes"])
    suppressed = int(counters["emissions_suppressed_terminal"])
    return {
        "residuals": {
            "emissions_split": sum(bucket["emissions"] for bucket in by_reason.values()) - emissions,
            "attempts_split": sum(bucket["attempts"] for bucket in by_reason.values()) - attempts,
            "bytes_split": sum(bucket["bytes"] for bucket in by_reason.values()) - airtime,
            "airtime_is_the_bill": airtime - int(spent_bytes),
            "attempts_are_delivered_plus_lost": attempts - (delivered + int(counters["packets_lost"])),
            "delivered_split": delivered - refused_classes,
            "each_emission_reaches_every_receiver": attempts - emissions * subscribers,
        },
        "generated_balance": {
            "generated_items": generated,
            "emissions": emissions,
            "redundant_emissions": sum(bucket["redundant"] for bucket in by_reason.values()),
            "coalesced_changes": coalesced,
            "suppressed_terminal": suppressed,
            # What the sender knew and never said, by arithmetic. Reported, not assumed away:
            # news left on the stack at the end is a cost the arm avoided paying, and an arm
            # cannot be allowed to look cheap because the episode stopped asking.
            # What the sender knew and never said. One caveat, sized rather than hidden: the
            # transport's ``coalesced_changes`` fires on the event path only, so a second change
            # spotted by *polling* the registry while the first is still waiting is neither
            # generated nor coalesced there -- and shows up here as a shortfall of exactly that
            # many. See ``generation_balance_shortfall``.
            "unspoken_or_uncounted_at_end": generated
            - (emissions - sum(bucket["redundant"] for bucket in by_reason.values()))
            - coalesced
            - suppressed,
            "unsent_change_steps": int(counters["unsent_change_steps"]),
        },
    }


def _generation_balance_shortfall(closure: Mapping[str, Any]) -> int:
    """How far the generation identity fell short, or 0 if it held.

    A negative balance is not a strategy and not a cost error: it is the known blind spot of
    ``coalesced_changes``, which counts a merge when an event arrives but not when the sender
    polls a newer registry view while an older one is still waiting. The cost and outcome
    identities above are unaffected and close exactly.
    """
    balance = int(closure["generated_balance"]["unspoken_or_uncounted_at_end"])
    return min(0, balance)


def _closure_is_clean(closure: Mapping[str, Any]) -> bool:
    return all(int(value) == 0 for value in closure["residuals"].values())


#: Quantities the relay counts as it goes and the transcript reader recomputes from the log.
#: Two implementations of the same number, so the attribution is not resting on one of them.
SHARED_QUANTITIES = (
    "emissions",
    "first_emissions",
    "refresh_emissions",
    "transmission_attempts",
    "packets_lost",
    "packets_delivered",
    "deliveries_applied",
    "airtime_bytes",
)


def _implementations_agree(offline: Mapping[str, Any], live: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: int(offline[key]) - int(live[key])
        for key in SHARED_QUANTITIES
        if key in offline and key in live
    }


def _task_units(row: Mapping[str, Any], subscribers: int) -> dict[str, Any]:
    """The completion and cost numbers, one field per unit, spelled out in the field name."""
    ledger = row["ledger"]
    task = row["task"]
    targets = int(task["targets"])
    on_time = int(task["on_time"])
    emissions = int(ledger["emissions"])
    attempts = int(ledger["transmission_attempts"])
    airtime = int(row["spent_bytes"])
    return {
        "on_time_targets_per_episode": on_time,
        "detected_targets_this_episode": targets,
        "on_time_rate_within_detected_this_episode": round(on_time / targets, 5) if targets else None,
        "expired_targets_per_episode": int(task["expired"]),
        "unresolved_targets_per_episode": int(task["unresolved"]),
        "airtime_bytes": airtime,
        "credit_used_fraction": round(float(row["budget_used_fraction"] or 0.0), 5),
        "bytes_per_emission_batch": round(airtime / emissions, 1) if emissions else None,
        # One emission is addressed to the whole team and billed that many times, so a packet
        # is the unit a radio engineer would recognise as the thing that occupied the air.
        "bytes_per_receiver_packet": round(airtime / attempts, 1) if attempts else None,
        "emissions": emissions,
        "transmission_attempts": attempts,
        "deliveries_applied": int(ledger["deliveries_applied"]),
        "applied_share_of_attempts": round(
            int(ledger["deliveries_applied"]) / attempts, 5
        )
        if attempts
        else None,
        "belief_age_steps": float(row["belief"].get("belief_age_steps_mean") or 0.0),
        "unknown_to_everyone_steps": float(row["belief"].get("unknown_to_everyone_steps") or 0.0),
        "seconds": float(row["seconds"]),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def summarise(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-arm summary, with each unit labelled and none averaged into another."""
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[str(row["arm"])].append(row)
    table: dict[str, Any] = {}
    for arm, values in sorted(by_arm.items()):
        count = len(values)
        detected = sum(int(row["detected_targets_this_episode"]) for row in values)
        on_time = sum(int(row["on_time_targets_per_episode"]) for row in values)
        rates = [
            float(row["on_time_rate_within_detected_this_episode"])
            for row in values
            if row["on_time_rate_within_detected_this_episode"] is not None
        ]
        reasons: dict[str, dict[str, int]] = defaultdict(_bucket)
        for row in values:
            for reason, bucket in row["rollup"]["by_reason"].items():
                for key, amount in bucket.items():
                    if key != "mean_wait_steps":
                        reasons[reason][key] += int(amount)
        total_bytes = sum(sum(bucket["bytes"] for bucket in row["rollup"]["by_reason"].values()) for row in values)
        table[arm] = {
            "episodes": count,
            "on_time_targets_per_episode_mean": round(on_time / count, 4),
            "on_time_rate_per_episode_mean": round(sum(rates) / len(rates), 5) if rates else None,
            "pooled_on_time_over_detected_targets": round(on_time / detected, 5) if detected else None,
            "detected_targets_total": detected,
            "on_time_targets_total": on_time,
            "expired_total": sum(int(row["expired_targets_per_episode"]) for row in values),
            "unresolved_total": sum(int(row["unresolved_targets_per_episode"]) for row in values),
            "airtime_bytes_mean": _mean(row["airtime_bytes"] for row in values),
            "airtime_bytes_total": sum(int(row["airtime_bytes"]) for row in values),
            "credit_used_fraction_mean": _mean(row["credit_used_fraction"] for row in values),
            "bytes_per_emission_batch_mean": _mean(
                row["bytes_per_emission_batch"] or 0 for row in values
            ),
            "bytes_per_receiver_packet_mean": _mean(row["bytes_per_receiver_packet"] or 0 for row in values),
            "emissions_mean": _mean(row["emissions"] for row in values),
            "attempts_mean": _mean(row["transmission_attempts"] for row in values),
            "applied_mean": _mean(row["deliveries_applied"] for row in values),
            "applied_share_of_attempts_mean": _mean(
                row["applied_share_of_attempts"] or 0 for row in values
            ),
            "forced_share_of_bytes": round(reasons["forced_maxwait"]["bytes"] / total_bytes, 4)
            if total_bytes
            else None,
            "by_reason": {reason: dict(sorted(bucket.items())) for reason, bucket in sorted(reasons.items())},
            "belief_age_steps_mean": _mean(row["belief_age_steps"] for row in values),
            "unknown_to_everyone_steps_mean": _mean(row["unknown_to_everyone_steps"] for row in values),
            "seconds_mean": _mean(row["seconds"] for row in values),
            "ledger_closed": all(_closure_is_clean(row["closure"]) for row in values),
        }
    return table
