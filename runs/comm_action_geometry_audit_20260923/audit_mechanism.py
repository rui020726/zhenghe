"""Why g12's 198 strict-subset steps sit in only 3 of 16 streams.

Run:  python audit_mechanism.py      # writes COMM_ACTION_GEOMETRY_MECHANISM.json

The handoff asks for the mechanism, not a restatement. The candidate explanation is arithmetic:
the credit is spent in a handful of packet-sized jumps, while the band of credit values in which a
strict subset is even possible is a few hundred bytes wide. A stream therefore either lands in that
band or it does not, and which one happens is decided by the residue of the charges it happened to
pay -- not by how many goals the layer has.

Two stored views make this checkable without replaying anything:

* each cell's own multi-candidate price envelope (``multi_candidate_prices``, g12 rows), and
* the exact candidates standing at the step where the credit first covered nothing at all
  (``first_credit_exhaustion_facts``, g12 rows), which gives the per-step band at the transition.

Both are joined against the terminal credit, which is just the budget minus the stored charges.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import audit_common as A

BUDGET = A.BUDGET_BYTES


def band_position(credit: float, low: float, high: float) -> str:
    if credit < low:
        return "below_band_nothing_payable"
    if credit >= high:
        return "above_band_everything_payable"
    return "inside_band_strict_subset_possible"


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None or float(denominator) <= 0:
        return None
    return round(float(numerator) / float(denominator), 5)


def analyse_layer(records: list[dict[str, Any]], best: str) -> dict[str, Any]:
    cells = [record for record in records if record["arm"] == best]
    rows: list[dict[str, Any]] = []
    for record in sorted(cells, key=lambda item: item["seed"]):
        prices = record.get("multi_prices")
        spent = int(record["spent_bytes"])
        credit = (BUDGET - spent) if record["budget_bytes"] else None
        facts = record.get("first_none_payable_facts")
        entry: dict[str, Any] = {
            "seed": record["seed"],
            "strict_subset_steps": int(record["steps_strict_subset"]),
            "multi_candidate_steps": record["steps_ge_2_candidates_exact"],
            "none_payable_steps": record["steps_none_payable_two_or_more"],
            "joint_over_credit_steps": int(record["steps_joint_over_credit"]),
            "all_payable_steps": record["steps_all_payable_within_credit"],
            "emissions": int(record["emissions"]),
            "charge_bytes_median_emitted": record.get("charge_bytes_median_emitted"),
            "spent_bytes": spent,
            "credit_left_bytes": credit,
        }
        if prices and credit is not None:
            low, high = int(prices["min"]), int(prices["max"])
            entry["envelope"] = {
                "min_charge": low,
                "max_charge": high,
                "band_width_bytes": high - low,
                "band_width_as_fraction_of_credit": round((high - low) / BUDGET, 5),
                "position_of_terminal_credit": band_position(credit, low, high),
                "distance_from_credit_to_cheapest_charge": round(credit - low, 1),
            }
        if facts:
            charges = [int(charge) for charge in (facts.get("charges") or [])]
            left = float(facts.get("remaining_bytes") or 0.0)
            if charges:
                packet = record.get("charge_bytes_median_emitted")
                low, high = min(charges), max(charges)
                entry["at_first_starvation_step"] = {
                    "step": int(facts["step"]),
                    "visible": int(facts["visible"]),
                    "credit_left_bytes": left,
                    "min_charge": low,
                    "max_charge": high,
                    "band_width_bytes": high - low,
                    "position": band_position(left, low, high),
                    "first_report_candidates": int(facts.get("first_emissions", 0)),
                    "refresh_candidates": int(facts.get("refreshes", 0)),
                }
                # ``position`` is below the band by construction -- a starvation step is defined as
                # the credit covering none of the candidates, so reporting it as a finding would be
                # circular. The informative question is where the credit came *from*: one packet
                # earlier, was the band still ahead of it, inside it, or already behind it?
                if packet:
                    previous = left + float(packet)
                    entry["at_first_starvation_step"]["credit_one_packet_earlier"] = round(
                        previous, 1
                    )
                    entry["at_first_starvation_step"]["one_packet_earlier_position"] = (
                        band_position(previous, low, high)
                    )
        rows.append(entry)

    starved_rows = [row for row in rows if row.get("at_first_starvation_step")]
    skips: dict[str, int] = {}
    for row in starved_rows:
        position = row["at_first_starvation_step"].get("one_packet_earlier_position")
        skips[position or "no_packet_charge_available"] = skips.get(
            position or "no_packet_charge_available", 0
        ) + 1

    inside = [row for row in rows if row.get("envelope", {}).get("position_of_terminal_credit")
              == "inside_band_strict_subset_possible"]
    strict_streams = [row for row in rows if row["strict_subset_steps"] > 0]
    band_widths = [
        float(row["at_first_starvation_step"]["band_width_bytes"]) for row in starved_rows
    ]
    packet_charges = [
        float(row["charge_bytes_median_emitted"])
        for row in rows
        if row.get("charge_bytes_median_emitted") is not None
    ]
    return {
        "best_fixed_rule": best,
        "streams": len(rows),
        "per_stream": rows,
        #: A zero below means "this layer stored no per-candidate price envelope", not "measured
        #: none": the three delivered layers kept no per-candidate prices, so the envelope test
        #: cannot run on them at all.
        "streams_with_price_envelope": sum(1 for row in rows if "envelope" in row),
        "streams_with_starvation_step_candidates": len(starved_rows),
        "terminal_credit_lands_inside_own_envelope": len(inside),
        "terminal_credit_lands_below_envelope": sum(
            1 for row in rows if row.get("envelope", {}).get("position_of_terminal_credit")
            == "below_band_nothing_payable"
        ),
        "terminal_credit_lands_above_envelope": sum(
            1 for row in rows if row.get("envelope", {}).get("position_of_terminal_credit")
            == "above_band_everything_payable"
        ),
        "streams_with_strict_subset_steps": len(strict_streams),
        "strict_subset_step_total": sum(int(row["strict_subset_steps"]) for row in rows),
        #: The credit always sits below its own band at the first starvation step, and that is true
        #: by construction: such a step is *defined* as the credit covering none of the candidates.
        #: It is kept only to show the partition closes. The evidence is the test below.
        "first_starvation_step_below_its_own_band_by_construction": sum(
            1 for row in starved_rows
            if row["at_first_starvation_step"]["position"] == "below_band_nothing_payable"
        ),
        "where_the_credit_came_from_one_packet_earlier": skips,
        "first_starvation_candidate_classes": {
            "all_refresh": sum(
                1 for row in starved_rows
                if row["at_first_starvation_step"]["first_report_candidates"] == 0
            ),
            "some_first_report": sum(
                1 for row in starved_rows
                if row["at_first_starvation_step"]["first_report_candidates"] > 0
            ),
            "streams_with_this_record": len(starved_rows),
        },
        #: The mechanism claim as one number: how wide the within-step choice window is compared
        #: with how far the credit actually moves per emission.
        "median_band_width_bytes_at_starvation_step": A.quantiles(band_widths).get("median"),
        "median_packet_charge_bytes": A.quantiles(packet_charges).get("median"),
        "band_width_as_fraction_of_one_packet": _ratio(
            A.quantiles(band_widths).get("median"), A.quantiles(packet_charges).get("median")
        ),
    }


def main(argv: list[str]) -> int:
    audit = A.read_json(A.HERE / "COMM_ACTION_GEOMETRY_AUDIT.json")
    records = audit["per_cell"]
    out: dict[str, Any] = {
        "question": "why does the strict-subset band appear in so few streams",
        "explanation_tested": "the credit is spent in packet-sized jumps while the band of credit "
                              "values that permits a strict subset is a few hundred bytes wide, so "
                              "landing in the band is decided by the residue of the charges paid, "
                              "not by the goal count",
        "layers": {},
        "cross_layer": {},
    }
    for layer in A.LAYERS:
        best = audit["layers"][layer]["best_fixed_rule"]
        out["layers"][layer] = analyse_layer(
            [record for record in records if record["layer"] == layer], best
        )
        out["cross_layer"][layer] = {
            "streams_with_band": out["layers"][layer]["streams_with_strict_subset_steps"],
            "band_steps_total": out["layers"][layer]["strict_subset_step_total"],
            "streams_terminal_credit_inside_envelope": out["layers"][layer][
                "terminal_credit_lands_inside_own_envelope"
            ],
        }
    #: The same test on every arm, because whether the band gets exercised depends on how a rule
    #: spends, and "this is residue arithmetic, not task pressure" is only supported if the arms
    #: that do refresh reach the band far more often at the same goal counts.
    out["per_arm_all_layers"] = {}
    for layer in A.LAYERS:
        layer_records = [record for record in records if record["layer"] == layer]
        arms = {}
        for arm in sorted({record["arm"] for record in layer_records}):
            analysed = analyse_layer(layer_records, arm)
            arms[arm] = {
                "streams_with_band": analysed["streams_with_strict_subset_steps"],
                "band_steps_total": analysed["strict_subset_step_total"],
                "terminal_credit_inside_envelope": analysed[
                    "terminal_credit_lands_inside_own_envelope"
                ],
            }
        out["per_arm_all_layers"][layer] = arms
    A.write_json(A.HERE / "COMM_ACTION_GEOMETRY_MECHANISM.json", out)
    print(json.dumps(
        {
            layer: {
                key: value
                for key, value in data.items()
                if key not in ("per_stream",)
            }
            for layer, data in out["layers"].items()
        },
        ensure_ascii=False,
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
