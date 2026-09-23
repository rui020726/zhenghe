"""The audit itself: candidate concurrency, quote geometry, budget timing, ledger-only savings.

Run:  python audit_run.py            # writes COMM_ACTION_GEOMETRY_AUDIT.json

Four sections, A-D, matching the handoff's questions. Each cell (layer, arm, seed) is read once and
reduced to a compact record; the transcripts are then dropped, because the refresh-heavy arms log
tens of thousands of per-step refusals each. Anything too large to keep is kept as a count;
anything small enough to keep is kept *raw*, so quantiles are computed once over real values rather
than over summary statistics of summaries.

Three rules the whole file obeys:

* Denominators are never merged. "Steps with a refusal", "steps with two candidates" and "steps
  where only a strict subset was payable" are different questions on different bases; the published
  round already had to say this once (extension protocol section 4).
* Where a number is a bound rather than a measurement, it is stored as a bound. The three delivered
  layers have no per-step candidate histogram, so their multi-candidate step count is bracketed.
* Section D is a ledger exercise. It re-adds one packet's bytes to the credit left and asks which
  logged refusals that would have cleared. Nothing is replayed, so it cannot say a task outcome
  would improve; the label `ledger_only_opportunity` is attached to every number in it.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import audit_cells as C
import audit_common as A

CALLS = C.CALLS
BUDGET = A.BUDGET_BYTES
CREDIT_FRACTIONS = ("0.50", "0.75", "0.90", "1.00")

#: The step classes the handoff asks for, each with the field it comes from, the fallback used when
#: that field does not exist in a delivered layer, and what the fallback actually means.
DENOMINATORS = (
    ("all_steps", "steps_all", "steps_all", "exact in every layer"),
    ("at_least_one_candidate", "steps_ge_1_candidate", "steps_ge_1_candidate",
     "exact in every layer (the probe is not called on an empty candidate list)"),
    ("at_least_two_candidates", "steps_ge_2_candidates_exact", "steps_ge_2_candidates_bound",
     "exact in g12; bracketed in g6/g8/g10, which stored no per-step histogram"),
    ("send_and_hold_both_executable", "steps_at_least_one_payable", "steps_at_least_one_payable",
     "exact in every layer"),
    ("all_payable", "steps_all_payable_within_credit", "steps_all_payable_within_credit",
     "exact in g12 only; missing in g6/g8/g10"),
    ("strict_subset_payable", "steps_strict_subset", "steps_strict_subset",
     "exact in every layer: this is the L3 reading"),
    ("none_payable", "steps_none_payable_two_or_more", "steps_none_payable_any_visible",
     "visible>=2 in g12; g6/g8/g10 only recorded visible>=1, a different base"),
)


def describe(values: list[float]) -> dict[str, Any]:
    return A.quantiles([float(value) for value in values])


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None or float(denominator) <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


# --------------------------------------------------------------------------- B: quote geometry --
def strict_step_measurements(sample: list[dict[str, Any]]) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Within-step quote geometry at the steps where only a strict subset was payable.

    These samples are the only stored per-candidate step views in the delivered layers, so they are
    also the only place their quote geometry can be measured at all. At most 20 steps per cell, so
    the raw per-step values are returned rather than summarised.
    """
    raw: dict[str, list[float]] = {
        key: []
        for key in (
            "charge_range_bytes",
            "charge_ratio_max_over_min",
            "charge_range_as_fraction_of_credit",
            "credit_minus_cheapest_charge",
            "credit_minus_dearest_charge",
            "position_of_credit_within_charge_band",
            "seam_bytes_to_next_unaffordable",
            "payable_count",
            "visible_count",
            "first_report_candidates",
            "refresh_candidates",
        )
    }
    checks = {"steps": 0, "credit_test_explains_payable": 0, "steps_below_two_priced": 0}
    for entry in sample:
        checks["steps"] += 1
        charges = sorted(
            int(item["charge"]) for item in entry["candidates"] if item["charge"] is not None
        )
        if len(charges) < 2:
            checks["steps_below_two_priced"] += 1
            continue
        remaining = float(entry["remaining_bytes"] or 0.0)
        low, high = float(charges[0]), float(charges[-1])
        paid = [charge for charge in charges if charge <= remaining]
        unpaid = [charge for charge in charges if charge > remaining]
        raw["charge_range_bytes"].append(high - low)
        raw["charge_ratio_max_over_min"].append(high / low if low else 0.0)
        raw["charge_range_as_fraction_of_credit"].append((high - low) / BUDGET)
        raw["credit_minus_cheapest_charge"].append(remaining - low)
        raw["credit_minus_dearest_charge"].append(remaining - high)
        raw["position_of_credit_within_charge_band"].append(
            0.5 if high == low else (remaining - low) / (high - low)
        )
        raw["seam_bytes_to_next_unaffordable"].append(
            float(unpaid[0] - paid[-1]) if paid and unpaid else 0.0
        )
        raw["payable_count"].append(float(entry["payable"]))
        raw["visible_count"].append(float(entry["visible"]))
        raw["first_report_candidates"].append(
            float(sum(1 for item in entry["candidates"] if int(item["first_emission"])))
        )
        raw["refresh_candidates"].append(
            float(sum(1 for item in entry["candidates"] if not int(item["first_emission"])))
        )
        if int(entry["payable"]) == len(paid):
            checks["credit_test_explains_payable"] += 1
    return raw, checks


# --------------------------------------------------------- C: what the refusals and timing say --
def transition_reading(cell: dict[str, Any]) -> dict[str, Any]:
    """Did the credit pass through the strict-subset band on its way to running out?

    A cell with zero strict-subset steps and a recorded starvation step went from something payable
    to nothing payable without ever offering a subset to choose between. The probe stores the first
    20 strict steps, so the earliest one is exact whenever the cell holds 20 points or fewer.
    """
    sample_steps = [int(entry["step"]) for entry in cell["strict_sample"]]
    first_strict = min(sample_steps) if sample_steps else None
    starvation = cell["first_none_payable_step"]
    points = int(cell["steps_strict_subset"])
    if points == 0:
        kind = "no_band_visited" if starvation is not None else "never_ran_out"
    elif starvation is None:
        kind = "band_visited_without_starvation"
    elif first_strict is not None and first_strict < starvation:
        kind = "band_visited_before_starvation"
    elif first_strict == starvation:
        kind = "band_and_starvation_same_step"
    else:
        kind = "band_only_after_starvation"
    return {
        "kind": kind,
        "first_strict_subset_step": first_strict,
        "first_none_payable_step": starvation,
        "strict_points_total": points,
        "first_strict_is_exact": cell["strict_sample_covers_all_points"],
    }


def refusal_composition(cell: dict[str, Any]) -> dict[str, Any]:
    """Which classes of message the refusals are, and what the first refusal looks like.

    The relay re-attempts a starved target every step, so attempts are not messages. Both are
    reported: the attempt count is what the ledger charged against, the distinct-target count is
    what the receivers never learned.
    """
    refusals = cell["refusals"]
    if not refusals:
        return {"attempts": 0}
    first_report = sum(1 for item in refusals if item["is_first_report"])
    earliest = min(refusals, key=lambda item: item["step"])
    targets = {item["target"] for item in refusals}
    return {
        "attempts": len(refusals),
        "first_report_attempts": first_report,
        "refresh_attempts": len(refusals) - first_report,
        "retry_attempts": sum(1 for item in refusals if item["retry"]),
        "distinct_targets": len(targets),
        "attempts_per_target": _ratio(len(refusals), len(targets)),
        "first_refusal_step": int(earliest["step"]),
        "first_refusal_is_first_report": bool(earliest["is_first_report"]),
        "first_refusal_charge": int(earliest["cost"]),
        "first_refusal_credit_left": float(earliest["remaining"]),
        "refused_but_recorded_as_payable": sum(
            1 for item in refusals if item["cost"] <= item["remaining"]
        ),
    }


def after_starvation(cell: dict[str, Any]) -> dict[str, Any]:
    """What was still asking to be sent once the credit covered nothing at all."""
    starvation = cell["first_none_payable_step"]
    if starvation is None:
        return {"starved": False}
    later = [item for item in cell["refusals"] if int(item["step"]) >= starvation]
    emitted = {item["target"] for item in cell["emissions"]}
    first_targets = {item["target"] for item in later if item["is_first_report"]}
    return {
        "starved": True,
        "step": int(starvation),
        "attempts_after": len(later),
        "steps_with_refusals_after": len({int(item["step"]) for item in later}),
        "targets_still_wanting_first_report": len(first_targets),
        "targets_still_wanting_refresh": len(
            {item["target"] for item in later if not item["is_first_report"]}
        ),
        "first_report_targets_never_emitted_in_the_episode": len(
            {target for target in first_targets if target not in emitted}
        ),
    }


# --------------------------------------------------------- D: the ledger-only counterfactual ---
def ledger_only_savings(cell: dict[str, Any]) -> list[dict[str, Any]]:
    """Hold each actually-sent refresh once in the ledger, and list what that would have cleared.

    Nothing is replayed. The candidate sequence, the charges, their order and every later event stay
    exactly as recorded; one refresh's bytes are added back to the credit the relay logged at each
    later refusal. The released credit is treated as still available at every later step, which is
    optimistic: in an environment the sender would have wanted that refresh again. So every number
    derived from here is an *opportunity*, never a gain, and never a claim about task outcomes.
    """
    savings: list[dict[str, Any]] = []
    refreshes = [item for item in cell["emissions"] if not item["first"]]
    if not refreshes or cell["budget_bytes"] is None:
        return savings
    for held in refreshes:
        released = int(held["cost"])
        unlocked: list[dict[str, Any]] = []
        for refusal in cell["refusals"]:
            if int(refusal["step"]) <= int(held["step"]):
                continue
            # Already covered by the credit it was refused against: not something this saving
            # explains, and its presence would mean the refusal log and the charge disagree.
            if not refusal["cost"] > refusal["remaining"]:
                continue
            if refusal["cost"] > refusal["remaining"] + released:
                continue
            target = cell["targets"].get(refusal["target"]) or {}
            deadline = target.get("deadline")
            unlocked.append(
                {
                    "step": int(refusal["step"]),
                    "target": str(refusal["target"]),
                    "is_first_report": bool(refusal["is_first_report"]),
                    "retry": bool(refusal["retry"]),
                    "delay_steps": int(refusal["step"]) - int(held["step"]),
                    "outcome": target.get("outcome"),
                    "slack_at_unlock": (
                        round(float(deadline) - int(refusal["step"]), 3)
                        if deadline is not None
                        else None
                    ),
                }
            )
        first_reports = {item["target"] for item in unlocked if item["is_first_report"]}
        refresh_targets = {item["target"] for item in unlocked if not item["is_first_report"]}
        earliest = min(unlocked, key=lambda item: item["step"]) if unlocked else None
        savings.append(
            {
                "held_step": int(held["step"]),
                "held_target": held["target"],
                "released_bytes": released,
                "unlocked_attempts": len(unlocked),
                "unlocked_first_report_targets": len(first_reports),
                "unlocked_refresh_targets": len(refresh_targets),
                "unlocked_retry_attempts": sum(1 for item in unlocked if item["retry"]),
                "unlock_is_refresh_only": bool(unlocked) and not first_reports,
                "earliest_delay_steps": earliest["delay_steps"] if earliest else None,
                "earliest_slack_at_unlock": earliest["slack_at_unlock"] if earliest else None,
                "earliest_is_first_report": earliest["is_first_report"] if earliest else None,
                "unlocked_targets_already_failed": sorted(
                    {
                        item["target"]
                        for item in unlocked
                        if item["outcome"] in ("expired", "unresolved")
                    }
                ),
            }
        )
    return savings


# ------------------------------------------------------------------------------ per cell ------
def compact_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Reduce one analysed cell into the record the aggregation works on."""
    record = {
        key: cell[key]
        for key in (
            "layer", "goals", "arm", "role", "seed", "budget_bytes", "spent_bytes",
            "budget_used_fraction", "on_time", "detected", "placed", "geometry_recorded",
            "steps_all", "steps_zero_candidate", "steps_ge_1_candidate",
            "steps_ge_2_candidates_exact", "steps_ge_2_candidates_lower_bound",
            "steps_ge_2_candidates_upper_bound", "steps_at_least_one_payable",
            "steps_at_least_one_blocked", "steps_both_payable_and_blocked",
            "steps_strict_subset", "steps_none_payable_any_visible",
            "steps_none_payable_two_or_more", "steps_all_payable_within_credit",
            "steps_joint_over_credit", "steps_held_while_payable", "steps_with_emission",
            "first_budget_refusal_step", "first_none_payable_step",
            "strict_sample_covers_all_points", "row_transcript_send_agreement",
            "first_flag_disagreements", "refusal_row_count", "refusal_transcript_count",
            "credit_fraction_steps", "first_none_payable_facts",
        )
    }
    probe = cell["probe"]
    #: How much of the candidate field a rule actually touches. A rule that never attempts a
    #: starved candidate logs no refusal for it, so "visible" and "attempted" are two different
    #: denominators and the gap between them is itself a finding.
    record["visible_total"] = int(probe.get("visible_total", 0))
    record["payable_total"] = int(probe.get("payable_total", 0))
    record["attempts_total"] = len(cell["emissions"]) + int(cell["refusal_transcript_count"])
    record["candidate_exercise_ratio"] = _ratio(
        record["attempts_total"], record["visible_total"]
    )
    record["steps_ge_2_candidates_bound"] = {
        "lower": record["steps_ge_2_candidates_lower_bound"],
        "upper": record["steps_ge_2_candidates_upper_bound"],
    }
    record["emissions"] = len(cell["emissions"])
    record["emission_first"] = sum(1 for item in cell["emissions"] if item["first"])
    record["emission_refresh"] = sum(1 for item in cell["emissions"] if not item["first"])
    record["emission_retry"] = sum(1 for item in cell["emissions"] if item["revision"] > 1)
    record["charge_bytes_median_emitted"] = describe(
        [float(item["cost"]) for item in cell["emissions"]]
    ).get("median")

    prices = cell["probe"].get("multi_candidate_prices") or {}
    count = int(prices.get("count", 0) or 0)
    if count:
        low, high = int(prices["min"]), int(prices["max"])
        record["multi_prices"] = {
            "count": count,
            "min": low,
            "max": high,
            "spread": high - low,
            "ratio": _ratio(high, low),
            "mean": round(int(prices["sum"]) / count, 1),
            "spread_as_fraction_of_credit": _ratio(high - low, BUDGET),
        }
    else:
        record["multi_prices"] = None

    raw, checks = strict_step_measurements(cell["strict_sample"])
    record["strict_raw"] = raw
    record["strict_checks"] = checks
    record["transition"] = transition_reading(cell)
    record["refusal_composition"] = refusal_composition(cell)
    record["after_starvation"] = after_starvation(cell)
    record["savings"] = ledger_only_savings(cell)
    return record


# ----------------------------------------------------------------------------- aggregation ----
def _denominators(records: list[dict[str, Any]]) -> dict[str, Any]:
    """One row per step class, over a base that is stated rather than implied.

    Three outcomes per stream, kept apart: measured exactly, replaced by the declared fallback
    (a different base or a bound), or missing outright. A missing class is reported as missing --
    it is never filled in by replaying anything.
    """
    out: dict[str, Any] = {}
    for name, exact_key, fallback_key, note in DENOMINATORS:
        values: list[float] = []
        exact_streams = 0
        fallback_streams = 0
        missing_streams = 0
        for record in records:
            value = record.get(exact_key)
            if value is not None:
                values.append(float(value))
                exact_streams += 1
                continue
            fallback = record.get(fallback_key)
            if isinstance(fallback, dict):
                values.append(float(fallback["lower"]))
                fallback_streams += 1
            elif fallback is not None:
                values.append(float(fallback))
                fallback_streams += 1
            else:
                missing_streams += 1
        out[name] = {
            "basis_note": note,
            "streams_measured_exactly": exact_streams,
            "streams_from_declared_fallback": fallback_streams,
            "streams_missing": missing_streams,
            "total_steps": round(sum(values), 1),
            "per_episode": describe(values),
            "streams_with_at_least_one": sum(1 for value in values if value > 0),
        }
    return out


def _quote(records: list[dict[str, Any]]) -> dict[str, Any]:
    priced = [record["multi_prices"] for record in records if record["multi_prices"]]
    out: dict[str, Any] = {
        "cells_with_recorded_multi_candidate_prices": len(priced),
        "note": "the per-step price summary exists only in g12; in g6/g8/g10 the quote geometry "
                "below is measured at the sampled strict-subset steps and nowhere else",
    }
    if priced:
        counts = [int(entry["count"]) for entry in priced]
        out["multi_candidate_price_bytes"] = {
            "candidates_priced": sum(counts),
            "min": min(int(entry["min"]) for entry in priced),
            "max": max(int(entry["max"]) for entry in priced),
            "mean": round(
                sum(entry["mean"] * entry["count"] for entry in priced) / sum(counts), 1
            ),
            "widest_within_episode_spread": max(int(entry["spread"]) for entry in priced),
            "narrowest_within_episode_spread": min(int(entry["spread"]) for entry in priced),
            "spread_as_fraction_of_credit_widest": max(
                entry["spread_as_fraction_of_credit"] for entry in priced
            ),
            "median_spread_as_fraction_of_credit": describe(
                [entry["spread_as_fraction_of_credit"] for entry in priced]
            )["median"],
        }
    pooled: dict[str, list[float]] = {}
    for record in records:
        for key, values in record["strict_raw"].items():
            pooled.setdefault(key, []).extend(values)
    out["steps_sampled"] = sum(record["strict_checks"]["steps"] for record in records)
    out["strict_points_total"] = sum(record["steps_strict_subset"] for record in records)
    out["streams_whose_sample_covers_every_point"] = sum(
        1 for record in records if record["strict_sample_covers_all_points"]
    )
    out["credit_test_explains_payable_on_sampled_steps"] = sum(
        record["strict_checks"]["credit_test_explains_payable"] for record in records
    )
    out["sampled_steps_with_fewer_than_two_priced_candidates"] = sum(
        record["strict_checks"]["steps_below_two_priced"] for record in records
    )
    out["at_strict_subset_steps"] = {
        key: describe(values) for key, values in sorted(pooled.items())
    }
    return out


def _timing(records: list[dict[str, Any]]) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    for record in records:
        kind = record["transition"]["kind"]
        kinds[kind] = kinds.get(kind, 0) + 1
    facts = [
        record["first_none_payable_facts"]
        for record in records
        if record["first_none_payable_facts"]
    ]
    compositions = [record["refusal_composition"] for record in records]
    starved = [record["after_starvation"] for record in records if record["after_starvation"].get("starved")]
    return {
        "streams_that_ran_out_of_credit": len(starved),
        "first_budget_refusal_step": describe(
            [
                float(record["first_budget_refusal_step"])
                for record in records
                if record["first_budget_refusal_step"] is not None
            ]
        ),
        "first_none_payable_step": describe(
            [
                float(record["first_none_payable_step"])
                for record in records
                if record["first_none_payable_step"] is not None
            ]
        ),
        "credit_fraction_crossing_steps": {
            fraction: describe(
                [
                    float(record["credit_fraction_steps"][fraction])
                    for record in records
                    if record["credit_fraction_steps"].get(fraction) is not None
                ]
            )
            for fraction in CREDIT_FRACTIONS
        },
        "transition_kinds": kinds,
        "first_starvation_candidate_composition_streams": len(facts),
        "first_starvation_all_candidates_were_refreshes_streams": sum(
            1 for fact in facts if int(fact.get("first_emissions", 0)) == 0
        ),
        "first_starvation_all_candidates_were_first_reports_streams": sum(
            1 for fact in facts if int(fact.get("refreshes", 0)) == 0
        ),
        "first_starvation_detail": [
            {
                "seed": record["seed"],
                "step": (record["first_none_payable_facts"] or {}).get("step"),
                "visible": (record["first_none_payable_facts"] or {}).get("visible"),
                "first_emissions": (record["first_none_payable_facts"] or {}).get("first_emissions"),
                "refreshes": (record["first_none_payable_facts"] or {}).get("refreshes"),
                "credit_left": (record["first_none_payable_facts"] or {}).get("remaining_bytes"),
                "cheapest_charge": min(
                    (
                        int(charge)
                        for charge in (record["first_none_payable_facts"] or {}).get("charges", [])
                    ),
                    default=None,
                ),
            }
            for record in records
            if record["first_none_payable_facts"]
        ],
        "refusals": {
            "attempts": sum(int(item.get("attempts", 0)) for item in compositions),
            "first_report_attempts": sum(
                int(item.get("first_report_attempts", 0)) for item in compositions
            ),
            "refresh_attempts": sum(int(item.get("refresh_attempts", 0)) for item in compositions),
            "retry_attempts": sum(int(item.get("retry_attempts", 0)) for item in compositions),
            "distinct_targets": sum(int(item.get("distinct_targets", 0)) for item in compositions),
            "streams_whose_first_refusal_was_a_first_report": sum(
                1 for item in compositions if item.get("first_refusal_is_first_report")
            ),
            "first_refusal_credit_left": describe(
                [
                    float(item["first_refusal_credit_left"])
                    for item in compositions
                    if item.get("first_refusal_credit_left") is not None
                ]
            ),
            "refused_yet_recorded_as_payable": sum(
                int(item.get("refused_but_recorded_as_payable", 0)) for item in compositions
            ),
        },
        "after_starvation": {
            "attempts": sum(int(item.get("attempts_after", 0)) for item in starved),
            "targets_still_wanting_first_report": sum(
                int(item.get("targets_still_wanting_first_report", 0)) for item in starved
            ),
            "targets_still_wanting_refresh": sum(
                int(item.get("targets_still_wanting_refresh", 0)) for item in starved
            ),
            "first_report_targets_never_emitted": sum(
                int(item.get("first_report_targets_never_emitted_in_the_episode", 0))
                for item in starved
            ),
        },
    }


def _counterfactual(records: list[dict[str, Any]]) -> dict[str, Any]:
    savings = [entry for record in records for entry in record["savings"]]
    useful = [entry for entry in savings if entry["unlocked_attempts"] > 0]
    streams_with = {
        (record["layer"], record["arm"], record["seed"])
        for record in records
        if any(entry["unlocked_attempts"] > 0 for entry in record["savings"])
    }
    streams_first = {
        (record["layer"], record["arm"], record["seed"])
        for record in records
        if any(entry["unlocked_first_report_targets"] > 0 for entry in record["savings"])
    }
    return {
        "label": "ledger_only_opportunity",
        "not_an_environment_causal_counterfactual": True,
        "refreshes_tested": len(savings),
        "refreshes_that_unlock_something": len(useful),
        "streams_tested": len(records),
        "streams_with_any_opportunity": len(streams_with),
        "streams_unlocking_a_first_report": len(streams_first),
        "streams_unlocking_refresh_only": len(streams_with) - len(streams_first),
        "streams_with_nothing_to_unlock": len(records) - len(streams_with),
        "unlocked_attempts_per_held_packet": describe(
            [float(entry["unlocked_attempts"]) for entry in savings]
        ),
        "max_unlocked_attempts_from_one_packet": max(
            (entry["unlocked_attempts"] for entry in savings), default=0
        ),
        "unlocked_first_report_targets_per_held_packet": describe(
            [float(entry["unlocked_first_report_targets"]) for entry in savings]
        ),
        "delay_to_earliest_unlock": describe(
            [float(entry["earliest_delay_steps"]) for entry in useful]
        ),
        "slack_at_earliest_unlock": describe(
            [
                float(entry["earliest_slack_at_unlock"])
                for entry in useful
                if entry["earliest_slack_at_unlock"] is not None
            ]
        ),
        "earliest_unlock_was_a_first_report": sum(
            1 for entry in useful if entry["earliest_is_first_report"]
        ),
        "released_bytes": describe([float(entry["released_bytes"]) for entry in savings]),
        "unlocked_targets_already_failed": sorted(
            {
                target
                for entry in useful
                for target in entry["unlocked_targets_already_failed"]
            }
        ),
    }


def arm_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "streams": len(records),
        "spent_bytes_total": sum(int(record["spent_bytes"]) for record in records),
        "on_time_total": sum(int(record["on_time"]) for record in records),
        "denominators": _denominators(records),
        #: What the rule actually engages with, as opposed to what the relay could see. Refusals are
        #: only logged for attempts, so a rule that ignores a starved candidate makes that
        #: candidate invisible to the refusal ledger while it stays fully visible to the probe.
        "candidate_exercise": {
            "visible_candidate_steps": sum(int(record["visible_total"]) for record in records),
            "payable_candidate_steps": sum(int(record["payable_total"]) for record in records),
            "emissions": sum(int(record["emissions"]) for record in records),
            "refused_attempts": sum(
                int(record["refusal_transcript_count"]) for record in records
            ),
            "attempts": sum(int(record["attempts_total"]) for record in records),
            "attempt_share_of_visible": _ratio(
                sum(int(record["attempts_total"]) for record in records),
                sum(int(record["visible_total"]) for record in records),
            ),
            "per_stream_exercise_ratio": describe(
                [
                    float(record["candidate_exercise_ratio"])
                    for record in records
                    if record["candidate_exercise_ratio"] is not None
                ]
            ),
        },
        "multi_step_partition_closes": all(
            (record["steps_ge_2_candidates_exact"] or 0)
            == (record["steps_all_payable_within_credit"] or 0)
            + record["steps_joint_over_credit"]
            + record["steps_strict_subset"]
            + (record["steps_none_payable_two_or_more"] or 0)
            for record in records
            if record["geometry_recorded"]
        ),
        "quote_geometry": _quote(records),
        "budget_time_structure": _timing(records),
        "ledger_only_opportunity": _counterfactual(records),
    }


def main(argv: list[str]) -> int:
    records: list[dict[str, Any]] = []
    for layer in A.LAYERS:
        for cell in C.iter_cells(layer):
            records.append(compact_cell(cell))

    layers: dict[str, Any] = {}
    for layer in A.LAYERS:
        arms = sorted({record["arm"] for record in records if record["layer"] == layer})
        by_arm = {
            arm: arm_block(
                [
                    record
                    for record in records
                    if record["layer"] == layer and record["arm"] == arm
                ]
            )
            for arm in arms
        }
        best = _best_fixed(by_arm)
        published_best = str(
            A.read_json(A.SOURCE_ROUND / f"GATES_{layer}.json")["gates"]["L3"]["evidence"][
                "best_fixed_rule"
            ]
        )
        layers[layer] = {
            "best_fixed_rule": best,
            "best_fixed_rule_published": published_best,
            "best_fixed_rule_matches_published": best == published_best,
            "arms": by_arm,
            "per_stream": [
                {
                    key: record[key]
                    for key in (
                        "arm", "seed", "steps_strict_subset", "steps_ge_1_candidate",
                        "steps_ge_2_candidates_exact", "steps_none_payable_two_or_more",
                        "steps_joint_over_credit", "steps_all_payable_within_credit",
                        "steps_at_least_one_payable", "steps_both_payable_and_blocked",
                        "steps_held_while_payable", "first_none_payable_step",
                        "first_budget_refusal_step", "budget_used_fraction", "spent_bytes",
                        "on_time", "transition",
                    )
                }
                for record in records
                if record["layer"] == layer
            ],
        }

    integrity = {
        "cells": len(records),
        "cells_per_layer": {
            layer: sum(1 for record in records if record["layer"] == layer)
            for layer in A.LAYERS
        },
        "row_transcript_send_agreement_all_cells": all(
            record["row_transcript_send_agreement"] for record in records
        ),
        "first_flag_disagreements_total": sum(
            int(record["first_flag_disagreements"]) for record in records
        ),
        "row_vs_transcript_refusal_counts": {
            "row_budget_refusal_records": sum(
                int(record["refusal_row_count"]) for record in records
            ),
            "transcript_refused_records": sum(
                int(record["refusal_transcript_count"]) for record in records
            ),
        },
        "refused_yet_recorded_as_payable": sum(
            int(record["refusal_composition"].get("refused_but_recorded_as_payable", 0))
            for record in records
        ),
        "environment_tripwire_calls": list(CALLS),
    }
    payload = {
        "round": "comm_action_geometry_audit",
        "status": "retrospective mechanism audit over stored episodes; 0 new episodes",
        "definitions": {
            "strict_subset_payable": "visible >= 2 and 1 <= payable < visible (the L3 reading)",
            "all_payable_within_credit": "visible >= 2 and payable == visible and "
                                         "sum(charge) <= credit left",
            "joint_over_credit": "visible >= 2 and payable == visible and sum(charge) > credit left",
            "none_payable": "visible >= 2 and payable == 0",
            "send_and_hold_both_executable": "a step with at least one payable candidate; HOLD is "
                                             "always available to a fixed rule, so a real binary "
                                             "choice exists exactly when something is payable",
            "first_report": "a target with no earlier successful emission (relay first_emission, "
                            "comm.py:802)",
            "refresh": "a target with at least one earlier successful emission",
            "retry": "an attempt repeating an earlier refused attempt at the same target; defined "
                     "by this audit because the source round stores no such counter",
            "not_a_competition_point": "a refusal on its own, and any joint-over-credit step; "
                                       "neither may be called strict-subset competition",
        },
        "integrity": integrity,
        "layers": layers,
        "per_cell": [
            {
                key: value
                for key, value in record.items()
                if key not in ("strict_raw", "savings")
            }
            for record in records
        ],
    }
    A.write_json(A.HERE / "COMM_ACTION_GEOMETRY_AUDIT.json", payload)
    print(json.dumps(_headline(layers, integrity), ensure_ascii=False, indent=1))
    return 0


def _best_fixed(by_arm: dict[str, Any]) -> str:
    """The protocol's own rule: most on-time per episode, then fewer bytes, then row order."""
    competitors = [
        arm for arm in A.COMPETITION_ORDER if arm in by_arm
    ]
    def key(arm: str) -> tuple[float, float, int]:
        block = by_arm[arm]
        return (
            -block["on_time_total"],
            block["spent_bytes_total"],
            A.COMPETITION_ORDER.index(arm),
        )

    return min(competitors, key=key)


def _headline(layers: dict[str, Any], integrity: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"integrity": integrity, "layers": {}}
    for layer, data in layers.items():
        best = data["best_fixed_rule"]
        block = data["arms"][best]
        denominators = block["denominators"]
        out["layers"][layer] = {
            "best_fixed_rule": best,
            "strict_subset": denominators["strict_subset_payable"],
            "two_or_more_candidates": denominators["at_least_two_candidates"],
            "none_payable": denominators["none_payable"],
            "counterfactual_streams_with_opportunity": block["ledger_only_opportunity"][
                "streams_with_any_opportunity"
            ],
            "counterfactual_streams_unlocking_first_report": block[
                "ledger_only_opportunity"
            ]["streams_unlocking_a_first_report"],
        }
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
