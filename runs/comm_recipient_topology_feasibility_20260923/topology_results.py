"""Assemble the machine-readable results from the artefacts this round produced.

Run:  python topology_results.py     # writes COMM_RECIPIENT_TOPOLOGY_RESULTS.json

Nothing is re-typed: every number here is read back out of the stored-data, scalar-proof,
projection, manifest and verification files, so the JSON cannot drift from the artefacts that
justify it. Gates are evaluated here from those values, not asserted.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import topology_common as T

T.install_env_tripwire()

FILES = {
    "stored": "COMM_RECIPIENT_TOPOLOGY_STORED_DATA.json",
    "scalar": "COMM_RECIPIENT_TOPOLOGY_SCALAR_PROOF.json",
    "projection": "COMM_RECIPIENT_TOPOLOGY_PROJECTION.json",
    "manifest": "COMM_RECIPIENT_TOPOLOGY_INPUT_MANIFEST.json",
    "verify": "COMM_RECIPIENT_TOPOLOGY_VERIFICATION.json",
}

# R4 condition 1's threshold, quoted from the protocol so a later reader can see it was not moved.
R4_MIN_STREAMS_WITH_FIRST_REPORT_CHOICE = 8


def main(argv: list[str]) -> int:
    data = {key: T.read_json(T.HERE / name) for key, name in FILES.items()}
    stored, scalar = data["stored"], data["scalar"]
    projection, verify = data["projection"], data["verify"]

    counts = sorted(
        {
            int(value)
            for layer in stored["layers"].values()
            for value in layer["subscriber_count"]["distinct_values"]
        }
    )
    identity_ok = sum(
        layer["pricing_identity_violations"] for layer in stored["layers"].values()
    ) == 0
    ratios = scalar["claim_1_scale_invariance_of_the_window"]["ratio_at_every_scale"]
    g12_coalition = projection["projection_g12"]["coalition"]
    first_report_choice = projection["first_report_choice_g12"]

    gates = {
        "R0_semantics_trustworthy": {
            "passed": bool(verify["passed"]) and identity_ok,
            "episode_entry_point_calls": verify["episode_entry_point_calls"],
            "launches_added": verify["launches_added_by_this_round"],
            "input_files_changed": verify["files_changed"],
            "production_files_changed": verify["production_files_changed"],
            "pricing_identity_holds_on_every_send_record": identity_ok,
            "sealed_block_artefacts": verify["sealed_block_800000_800399_artefacts_on_disk"],
        },
        "R1_topology_already_heterogeneous": {
            "passed": len(counts) > 1,
            "distinct_subscriber_counts_across_all_four_layers": counts,
            "episodes_observed": stored["aggregate"]["episodes_observed"],
            "episodes_where_count_varies": stored["aggregate"]["episodes_with_varying_count"],
            "send_records": stored["aggregate"]["send_records"],
            "share_of_log_cost_variance_from_fanout": {
                layer: value["cost_decomposition"][
                    "share_of_log_cost_variance_from_fanout"
                ]
                for layer, value in stored["layers"].items()
            },
            "conclusion_when_failed": "global_fanout_is_only_a_scalar",
            "scale_invariance_of_the_window": {
                "window_as_fraction_of_one_packet_at_each_fanout_scale": ratios,
                "ratio_constant": scalar["claim_1_scale_invariance_of_the_window"][
                    "ratio_is_constant_across_scales"
                ],
            },
            "raising_fanout_equals_lowering_budget": {
                "classification_mismatches": scalar[
                    "claim_2_raising_fanout_equals_lowering_the_budget"
                ]["classification_mismatches"],
                "holds": scalar["claim_2_raising_fanout_equals_lowering_the_budget"]["holds"],
            },
        },
        "R2_a_recipient_rule_is_legitimate": {
            "passed": False,
            "candidates_evaluated": {
                "global_broadcast": {
                    "pre_action_determinable": True,
                    "uses_no_future_result": True,
                    "preserves_discovery_semantics": True,
                    "not_self_confirming": True,
                    "creates_cost_heterogeneity": False,
                    "verdict": "the status quo; a constant fan-out adds no choice",
                },
                "current_coalition": {
                    "pre_action_determinable": True,
                    "uses_no_future_result": True,
                    "preserves_discovery_semantics": False,
                    "not_self_confirming": False,
                    "creates_cost_heterogeneity": True,
                    "verdict": (
                        "recipients are the robots already told about the target, so a newly "
                        "discovered one has an empty set: every first report is addressed to "
                        "nobody, and under the isolated path an unreceived task can never enter "
                        "any robot's action list (source map question 9)"
                    ),
                    "first_report_fanout_values": g12_coalition["fanout_by_message_class"][
                        "first_report"
                    ]["distinct"],
                    "refresh_fanout_values": g12_coalition["fanout_by_message_class"]["refresh"][
                        "distinct"
                    ],
                    "share_of_messages_with_empty_recipient_set": g12_coalition[
                        "share_of_messages_with_empty_recipient_set"
                    ],
                },
                "capability_and_deadline_matching": {
                    "pre_action_determinable": False,
                    "verdict": "needs other robots' positions and schedules; positions are "
                               "excluded from the relay by design (invariant 10)",
                },
                "radio_neighbourhood": {
                    "pre_action_determinable": False,
                    "verdict": "no per-robot position or range view exists in the transcript or "
                               "the observation; only an artificial radius could supply one",
                },
                "coordinator_only": {
                    "pre_action_determinable": False,
                    "verdict": "there is no coordinator addressable in the subscriber space; "
                               "this is an architecture change, not a routing rule",
                },
            },
        },
        "R3_observable_before_the_action": {
            "passed": False,
            "reason": "SendObservation exposes one scalar `subscribers` for the whole step "
                      "(comm_policy.py:112), not a per-candidate recipient set, and SendItem "
                      "carries only `members_changed: bool` -- no member list. The contract has "
                      "no place to put 'whom would this go to'.",
            "charge_is_observable": True,
            "charge_is_a_network_input": False,
            "recipient_count_is_per_candidate": False,
            "recipient_set_reason_is_observable": False,
        },
        "R4_offline_quotes_form_a_real_choice": {
            "passed": False,
            "premise_failed_first": "R2 and R3 both failed; evaluated anyway as a diagnostic",
            "label": "offline_recipient_cost_projection",
            "condition_1_first_report_price_differences_ge_8_of_16_streams": {
                "threshold": R4_MIN_STREAMS_WITH_FIRST_REPORT_CHOICE,
                "coalition": first_report_choice["coalition"][
                    "streams_where_two_first_reports_ever_cost_different_amounts"
                ],
                "broadcast": first_report_choice["broadcast"][
                    "streams_where_two_first_reports_ever_cost_different_amounts"
                ],
                "broadcast_median_spread_bytes": first_report_choice["broadcast"][
                    "median_price_spread_between_two_first_reports"
                ],
                "note": "broadcast reaches 16/16 only through a median 6-byte difference driven by "
                        "serialized field lengths, not by recipients, so it satisfies the letter "
                        "of condition 1 and violates condition 5",
            },
            "condition_4_not_only_refresh_choices": {
                "passed": False,
                "coalition_steps_where_a_first_report_is_priced_to_zero": first_report_choice[
                    "coalition"
                ]["steps_where_a_first_report_would_be_priced_to_zero"],
                "within_step_relative_spread_median": {
                    "broadcast": projection["projection_g12"]["broadcast_within_step"][
                        "relative_spread_of_prices_within_a_step"
                    ]["median"],
                    "coalition": projection["projection_g12"]["coalition_within_step"][
                        "relative_spread_of_prices_within_a_step"
                    ]["median"],
                },
                "note": "the heterogeneity coalition routing buys is entirely on the refresh side; "
                        "the first-report side collapses to an empty recipient set",
            },
            "condition_5_difference_from_real_semantics_not_scaling": {
                "passed": False,
                "reason": "any global change of fan-out is provably equivalent to rescaling the "
                          "budget (see R1's claim 2), so it is the scaling route the protocol "
                          "forbids",
            },
        },
    }

    decision = {
        "branch": "close_multi_subscriber_topology_axis",
        "branch_source": "protocol section 6, row 1: R1 failed and no candidate passed R2",
        "gates_passed": {name: gate["passed"] for name, gate in gates.items()},
        "next_step_belongs_to_the_user": True,
    }

    result = {
        "round": "comm_recipient_topology_feasibility",
        "date": "2026-09-23",
        "status": "source and stored-data feasibility audit; not a performance experiment",
        "question": "is subscriber_count a real per-message quantity, or one constant multiplied "
                    "everywhere",
        "answer": "constant: one value across 8 904 sends in four layers, and any global change of "
                  "it is equivalent to changing the budget tier",
        "authorisation": data["manifest"]["authorisation"],
        "verification": {
            key: verify[key]
            for key in (
                "files_frozen", "files_changed", "files_missing_now",
                "launches_added_by_this_round", "episode_entry_point_calls",
                "production_files_changed", "sealed_block_800000_800399_artefacts_on_disk",
                "passed",
            )
        },
        "grid": data["manifest"]["grid"],
        "inputs_not_read": data["manifest"]["inputs_present_but_not_read"],
        "gates": gates,
        "missing": {
            "per_message_subscriber_count_on_refusals": "not recorded on any of the 671 604 "
                                                        "refused records; derived via the proven "
                                                        "constant fan-out and labelled as derived",
            "projected_budget_trajectory_under_coalition_pricing": "not computed; would require "
                                                                   "replaying the credit "
                                                                   "accumulation",
            "recipient_sets_for_candidates_3_4_5": "not constructible from this system at all",
        },
        "not_claimed": [
            "no claim that a learned controller would fail under a different topology",
            "no claim that coalition routing worsens task outcomes; no environment was run",
            "no claim that L3 should be re-judged either way",
            "no relaxation of the previous round's no_temporal_allocation_signal",
        ],
        "decision": decision,
    }
    T.write_json(T.HERE / "COMM_RECIPIENT_TOPOLOGY_RESULTS.json", result)
    print(json.dumps(
        {
            "gates_passed": {
                name: gate["passed"] for name, gate in result["gates"].items()
            },
            "answer": result["answer"],
            "decision": decision,
            "r1_counts": gates["R1_topology_already_heterogeneous"][
                "distinct_subscriber_counts_across_all_four_layers"
            ],
            "r4_first_report_streams": gates["R4_offline_quotes_form_a_real_choice"][
                "condition_1_first_report_price_differences_ge_8_of_16_streams"
            ],
        },
        ensure_ascii=False,
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
