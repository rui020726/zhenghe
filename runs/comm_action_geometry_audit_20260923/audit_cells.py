"""One streaming pass per cell (layer, arm, seed) that extracts everything A-D need.

A cell is the 1 000-step episode stored in one row plus its transcript. Transcripts are read
record by record and reduced here, never kept: the refresh-heavy arms log tens of thousands of
per-step refusals for the same target, so a naive load of all 576 cells would be gigabytes of
Python objects for numbers that fit in a counter.

Every quantity is derived from stored artefacts. The episode runners are trip-wired by
``audit_common.install_env_tripwire`` before this module is even imported.

Message classes, stated once because the audit keeps returning to them:

``first report``  the target has no earlier successful emission  (relay: ``first_emission``,
                  ``comm.py:802`` -- ``sent is None``)
``refresh``       the target has at least one earlier successful emission
``retry``         an attempt that repeats an earlier *refused* attempt at the same target.
                  Orthogonal to the two above: a refused first report retried every step is both.
                  Not a counter the source round keeps, so this audit defines and states it.
"""

from __future__ import annotations

from typing import Any, Iterator

import audit_common as A

CALLS = A.install_env_tripwire()
A.enter_source_round()

STEP_COUNT = 1000


def iter_cells(layer: str, arms: tuple[str, ...] = A.ARMS) -> Iterator[dict[str, Any]]:
    for row in A.rows_of(layer):
        if str(row["arm"]) not in arms:
            continue
        yield analyse_cell(layer, row)


def analyse_cell(layer: str, row: dict[str, Any]) -> dict[str, Any]:
    """Reduce one cell into the numbers the four audit questions ask for."""
    probe = dict(row.get("probe") or {})
    block_reasons = {str(k): int(v) for k, v in (probe.get("block_reasons") or {}).items()}
    budget = row.get("budget_bytes")
    budget = int(budget) if budget is not None else None

    emissions = [
        {
            "step": int(item[0]),
            "target": str(item[1]),
            "revision": int(item[2]),
            "cost": int(item[3]),
            "subscribers": int(item[4]),
            "first": bool(item[5]),
        }
        for item in (row.get("emission_sequence") or [])
    ]
    emissions.sort(key=lambda item: (item["step"], item["target"]))

    # A first report is first because *no earlier emission for that target exists* -- the flag is
    # cross-checked against the sequence rather than trusted, because the whole audit turns on
    # which class a message belongs to.
    emitted_before: set[str] = set()
    for emission in emissions:
        was_first = emission["target"] not in emitted_before
        emission["first_by_history"] = was_first
        emitted_before.add(emission["target"])
    first_flag_disagreements = sum(
        1 for emission in emissions if emission["first"] != emission["first_by_history"]
    )

    spent_curve: list[tuple[int, int]] = []
    running = 0
    for emission in emissions:
        running += emission["cost"]
        spent_curve.append((emission["step"], running))

    step_thresholds: dict[str, int | None] = {}
    if budget:
        for fraction in (0.50, 0.75, 0.90, 1.00):
            target_bytes = fraction * budget
            step_thresholds[f"{fraction:.2f}"] = next(
                (step for step, total in spent_curve if total >= target_bytes), None
            )

    # ---- transcript: refusals carry the charge and the credit left, the row's copy does not ----
    transcript = A.read_transcript(layer, str(row["arm"]), int(row["seed"]))
    tx_sends = [record for record in transcript if record.get("kind") == "send"]
    refusals: list[dict[str, Any]] = []
    first_refusal_step_by_target: dict[str, int] = {}
    for record in transcript:
        if record.get("kind") != "refused":
            continue
        refusals.append(
            {
                "step": int(record["step"]),
                "target": str(record["target"]),
                "cost": int(record.get("cost_bytes") or 0),
                "remaining": float(record.get("remaining_bytes") or 0.0),
                "reason": str(record.get("reason")),
            }
        )
    refusals.sort(key=lambda item: (item["step"], item["target"]))
    first_emission_step = {
        emission["target"]: emission["step"] for emission in emissions if emission["first"]
    }
    refused_before: set[str] = set()
    for refusal in refusals:
        emission_step = first_emission_step.get(refusal["target"])
        refusal["is_first_report"] = emission_step is None or emission_step > refusal["step"]
        refusal["retry"] = refusal["target"] in refused_before
        refused_before.add(refusal["target"])
        first_refusal_step_by_target.setdefault(refusal["target"], refusal["step"])

    # Row-level and transcript-level send sets must agree, or the two artefacts describe
    # different episodes and every number below is unsafe.
    row_send_key = sorted(
        (item["step"], item["target"], item["cost"], item["first"]) for item in emissions
    )
    tx_send_key = sorted(
        (
            int(record["step"]),
            str(record["target"]),
            int(record["cost_bytes"]),
            bool(record["first"]),
        )
        for record in tx_sends
    )
    refusal_row_count = sum(
        1 for item in (row.get("refusal_sequence") or []) if str(item[2]) == "budget_exhausted"
    )

    steps_offered = int(probe.get("steps_offered", 0))
    visible_total = int(probe.get("visible_total", 0))
    max_visible = int(probe.get("max_visible", 0))
    allowed_steps = block_reasons.get("allowed", 0)
    blocked_steps = sum(
        count for reason, count in block_reasons.items() if reason != "allowed"
    )
    geometry_recorded = "steps_with_multiple_candidates" in probe

    multi_exact = int(probe.get("steps_with_multiple_candidates", 0)) if geometry_recorded else None
    # Without a stored histogram the multi-candidate count is bracketed, not known:
    #   visible_total <= (steps_offered - N2) + max_visible * N2   ->  lower bound
    #   visible_total >= (steps_offered - N2) + 2 * N2            ->  upper bound
    upper = max(visible_total - steps_offered, 0)
    lower = 0 if max_visible <= 1 else -(-(visible_total - steps_offered) // (max_visible - 1))
    steps_with_emission = len({emission["step"] for emission in emissions})

    strict_sample = [dict(entry) for entry in (probe.get("competition_sample") or [])]
    for entry in strict_sample:
        entry["candidates"] = [
            {
                "target": str(item[0]),
                "charge": item[1],
                "executable": bool(item[2]),
                "reason": str(item[3]),
                "first_emission": int(item[4]),
                "remaining_time": float(item[5]),
            }
            for item in entry.get("candidates") or []
        ]

    return {
        "layer": layer,
        "goals": int(row["goals"]),
        "arm": str(row["arm"]),
        "role": str(row.get("role")),
        "seed": int(row["seed"]),
        "budget_bytes": budget,
        "spent_bytes": int(row["spent_bytes"]),
        "budget_used_fraction": row.get("budget_used_fraction"),
        "on_time": int(row["on_time"]),
        "detected": int(row["detected"]),
        "placed": int(row["placed"]),
        "probe": probe,
        "geometry_recorded": geometry_recorded,
        "steps_all": STEP_COUNT,
        "steps_zero_candidate": STEP_COUNT - steps_offered,
        "steps_ge_1_candidate": steps_offered,
        "steps_ge_2_candidates_exact": multi_exact,
        "steps_ge_2_candidates_lower_bound": lower,
        "steps_ge_2_candidates_upper_bound": upper,
        "steps_at_least_one_payable": allowed_steps,
        "steps_at_least_one_blocked": blocked_steps,
        "steps_both_payable_and_blocked": max(allowed_steps + blocked_steps - steps_offered, 0),
        "steps_strict_subset": int(probe.get("competition_points", 0)),
        "steps_none_payable_any_visible": int(probe.get("credit_blocked_steps", 0)),
        "steps_none_payable_two_or_more": (
            int(probe.get("multi_none_payable_steps", 0)) if geometry_recorded else None
        ),
        "steps_all_payable_within_credit": (
            int(probe.get("multi_all_payable_within_credit_steps", 0)) if geometry_recorded else None
        ),
        "steps_joint_over_credit": int(probe.get("joint_only_points", 0)),
        "steps_held_while_payable": max(allowed_steps - steps_with_emission, 0),
        "steps_with_emission": steps_with_emission,
        "emissions": emissions,
        "emission_types": dict(row.get("emission_types") or {}),
        "first_flag_disagreements": first_flag_disagreements,
        "spent_curve": spent_curve,
        "credit_fraction_steps": step_thresholds,
        "refusals": refusals,
        "refusal_row_count": refusal_row_count,
        "refusal_transcript_count": len(refusals),
        "first_budget_refusal_step": (row.get("refusals") or {}).get("first_budget_refusal_step"),
        "first_none_payable_step": probe.get("first_credit_exhaustion_step"),
        "first_none_payable_facts": probe.get("first_credit_exhaustion_facts"),
        "strict_sample": strict_sample,
        "strict_sample_covers_all_points": int(probe.get("competition_points", 0))
        <= len(strict_sample),
        "row_transcript_send_agreement": row_send_key == tx_send_key,
        "row_transcript_send_counts": [len(row_send_key), len(tx_send_key)],
        "targets": {
            str(item["target"]): item for item in (row.get("targets") or [])
        },
        "ledger": dict(row.get("ledger") or {}),
    }


def describe(values: list[float]) -> dict[str, Any]:
    """Per-episode count summary: what the handoff asks for as median, quantiles and stream hits."""
    return A.quantiles([float(value) for value in values])
