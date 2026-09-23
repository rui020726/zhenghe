"""Prove that the fan-out term is a scalar, not a new axis -- and show what it is equivalent to.

Run:  python topology_scalar_proof.py   # writes COMM_RECIPIENT_TOPOLOGY_SCALAR_PROOF.json

R1 asks whether `subscriber_count` varies. The stored data says it does not: one value, 6, on every
send in every layer. But "it does not vary today" is weaker than "it could not vary in a way that
matters", and the second claim is what the decision rests on, so it is proved here rather than
argued.

Two statements are checked numerically.

**Scale invariance of the choice window.** A candidate's charge is `k * payload`. The window in
which a strict subset is affordable is the spread of charges inside one step, `k * spread(payload)`,
and the distance the credit moves per emission is `k * mean(payload)`. Their ratio has no `k` in it,
so no global fan-out widens the window relative to the step that has to land in it.

**Fan-out is the budget knob, wearing a different name.** Rescaling `k` by `s` while rescaling the
budget by `1/s` leaves every step's payability classification exactly as it was. So "add subscribers"
is not a new experiment; it is the tier sweep the project already ran, and running it again cannot
produce structure that the payload spread does not already contain.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import topology_common as T

CALLS = T.install_env_tripwire()

#: Multipliers applied to the whole fan-out. 6 is the observed value; 1 and 12 bracket a team
#: half as small and twice as large, and 1/6 normalises charges down to raw payload.
SCALES = (1 / 6.0, 1 / 3.0, 1.0, 2.0, 3.0, 6.0)


def load_step_samples() -> list[dict[str, Any]]:
    """Every stored per-step candidate view: the strict-subset samples, all four layers, all arms.

    These are the only places the frozen rows kept per-candidate charges, so they are what the
    window-ratio claim can be checked against without running anything.
    """
    steps: list[dict[str, Any]] = []
    for layer in T.LAYERS:
        for row in T.rows_of(layer):
            for entry in (row.get("probe") or {}).get("competition_sample") or []:
                charges = [
                    int(item[1])
                    for item in (entry.get("candidates") or [])
                    if item[1] is not None
                ]
                if len(charges) < 2 or entry.get("remaining_bytes") is None:
                    continue
                steps.append(
                    {
                        "layer": layer,
                        "arm": str(row["arm"]),
                        "seed": int(row["seed"]),
                        "step": int(entry["step"]),
                        "charges": charges,
                        "remaining": float(entry["remaining_bytes"]),
                        "visible": int(entry["visible"]),
                        "payable": int(entry["payable"]),
                    }
                )
    return steps


def pay_at(charges: list[int], remaining: float) -> dict[str, Any]:
    """The stored probe's own four-way partition, recomputed from charges and credit."""
    payable = sum(1 for charge in charges if charge <= remaining)
    visible = len(charges)
    total = sum(charges)
    if payable == visible and total <= remaining:
        kind = "all_payable_within_credit"
    elif payable == visible:
        kind = "joint_over_credit"
    elif 1 <= payable < visible:
        kind = "strict_subset_payable"
    else:
        kind = "none_payable"
    return {
        "payable": payable,
        "visible": visible,
        "strict_subset": kind == "strict_subset_payable",
        "kind": kind,
    }


def main(argv: list[str]) -> int:
    samples = load_step_samples()
    base_fanout = 6  # the single value every stored send carried

    ratios: dict[str, dict[str, Any]] = {}
    class_counts: dict[str, dict[str, int]] = {}

    for scale in SCALES:
        widths: list[float] = []
        packets: list[float] = []
        counts: dict[str, int] = {}
        for sample in samples:
            # Re-express each stored step as if the fan-out had been `base_fanout * scale`; a
            # charge is `fanout * payload`, so scaling the fan-out scales every charge alike.
            charges = [float(charge) * scale for charge in sample["charges"]]
            span = max(charges) - min(charges)
            widths.append(span)
            packets.append(sum(charges) / len(charges))
            # Classes are taken with the recorded credit held fixed: this is the "add robots but
            # keep the budget" reading, which is what a real fan-out change would actually do.
            verdict = pay_at(charges, sample["remaining"])
            counts[verdict["kind"]] = counts.get(verdict["kind"], 0) + 1
        mean_width = sum(widths) / len(widths) if widths else None
        mean_packet = sum(packets) / len(packets) if packets else None
        ratios[f"x{scale:g}"] = {
            "mean_within_step_window_bytes": round(mean_width, 4) if mean_width else None,
            "mean_packet_bytes": round(mean_packet, 4) if mean_packet else None,
            "window_as_fraction_of_one_packet": (
                round(mean_width / mean_packet, 6) if mean_width and mean_packet else None
            ),
            "payability_classes": counts,
        }
        class_counts[f"x{scale:g}"] = counts

    # The informative equivalence. Raising the fan-out from k to s*k while leaving the budget at B
    # asks, per step, `s*k*payload <= credit`; lowering the budget to B/s at fan-out k asks
    # `k*payload <= credit/s`, which is the same inequality multiplied through by s. So "add
    # subscribers" is literally "tighten the budget", a knob this project already swept. Checked
    # numerically rather than asserted: the two routes must agree on every stored step.
    widening_mismatches: list[dict[str, Any]] = []
    for scale in SCALES:
        if scale == 1.0:
            continue
        for sample in samples:
            charges = [float(charge) for charge in sample["charges"]]
            widened = pay_at([charge * scale for charge in charges], sample["remaining"])
            tightened = pay_at(charges, sample["remaining"] / scale)
            if widened["kind"] != tightened["kind"] or widened["payable"] != tightened["payable"]:
                widening_mismatches.append(
                    {
                        "scale": scale,
                        "layer": sample["layer"],
                        "arm": sample["arm"],
                        "step": sample["step"],
                        "widened_fanout": widened["kind"],
                        "tightened_budget": tightened["kind"],
                    }
                )
    result = {
        "claim_1_scale_invariance_of_the_window": {
            "statement": "window/packet ratio carries no fan-out term, so no global fan-out can "
                         "widen the window relative to the jump that has to land in it",
            "ratio_at_every_scale": {
                key: value["window_as_fraction_of_one_packet"] for key, value in ratios.items()
            },
            "ratio_is_constant_across_scales": len(
                {
                    value["window_as_fraction_of_one_packet"]
                    for value in ratios.values()
                    if value["window_as_fraction_of_one_packet"] is not None
                }
            ) == 1,
            "per_scale": ratios,
        },
        "claim_2_raising_fanout_equals_lowering_the_budget": {
            "statement": "fan-out s*k at budget B and fan-out k at budget B/s ask the same "
                         "inequality at every step, so 'add subscribers' is the budget-tier sweep "
                         "this project already ran, not a new axis",
            "steps_checked": len(samples),
            "scales_checked": [scale for scale in SCALES if scale != 1.0],
            "classification_mismatches": len(widening_mismatches),
            "mismatch_sample": widening_mismatches[:5],
            "holds": not widening_mismatches,
        },
        "observed_fanout_values_in_stored_data": [base_fanout],
        "per_step_window_bytes_unscaled": T.quantiles(
            [float(max(s["charges"]) - min(s["charges"])) for s in samples]
        ),
        "per_step_payload_window_bytes": T.quantiles(
            [float((max(s["charges"]) - min(s["charges"])) / base_fanout) for s in samples]
        ),
        "mean_packet_bytes_unscaled": T.quantiles(
            [float(sum(s["charges"]) / len(s["charges"])) for s in samples]
        ),
        "samples": len(samples),
        "environment_tripwire_calls": list(CALLS),
    }
    T.write_json(T.HERE / "COMM_RECIPIENT_TOPOLOGY_SCALAR_PROOF.json", result)
    print(json.dumps(
        {
            "steps_checked": len(samples),
            "ratio_by_scale": result["claim_1_scale_invariance_of_the_window"][
                "ratio_at_every_scale"
            ],
            "ratio_constant": result["claim_1_scale_invariance_of_the_window"][
                "ratio_is_constant_across_scales"
            ],
            "equivalence_mismatches": result[
                "claim_2_raising_fanout_equals_lowering_the_budget"
            ]["classification_mismatches"],
            "per_step_window_bytes": result["per_step_window_bytes_unscaled"],
            "per_step_payload_window_bytes": result["per_step_payload_window_bytes"],
            "tripwire_calls": result["environment_tripwire_calls"],
        },
        ensure_ascii=False,
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
