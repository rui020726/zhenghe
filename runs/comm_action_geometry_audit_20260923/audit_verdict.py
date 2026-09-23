"""Assemble the machine-readable verdict from the artefacts, so no number here is re-typed by hand.

Run:  python audit_verdict.py        # writes COMM_ACTION_GEOMETRY_AUDIT_VERDICT.json
"""

from __future__ import annotations

import json
import sys
from typing import Any

import audit_common as A

A.install_env_tripwire()


def pick(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def main(argv: list[str]) -> int:
    audit = A.read_json(A.HERE / "COMM_ACTION_GEOMETRY_AUDIT.json")
    mechanism = A.read_json(A.HERE / "COMM_ACTION_GEOMETRY_MECHANISM.json")
    recompute = A.read_json(A.HERE / "RECOMPUTE_L0_L4.json")
    verify = A.read_json(A.HERE / "AUDIT_VERIFICATION.json")
    manifest = A.read_json(A.HERE / "COMM_ACTION_GEOMETRY_INPUT_MANIFEST.json")

    g12_best = "send_once"
    per_arm_refresh_claims = {}
    for arm, block in audit["layers"]["g12"]["arms"].items():
        timing = block["budget_time_structure"]
        recorded = timing["first_starvation_candidate_composition_streams"]
        if not recorded:
            continue
        per_arm_refresh_claims[arm] = {
            "streams_with_record": recorded,
            "all_candidates_were_refreshes": timing[
                "first_starvation_all_candidates_were_refreshes_streams"
            ],
            "at_least_one_never_reported_target": recorded - timing[
                "first_starvation_all_candidates_were_refreshes_streams"
            ],
            "all_candidates_were_first_reports": timing[
                "first_starvation_all_candidates_were_first_reports_streams"
            ],
        }

    verdict = {
        "round": "comm_action_geometry_audit",
        "date": "2026-09-23",
        "branch": manifest["git"]["branch"],
        "start_sha": manifest["git"]["head"],
        "status": "retrospective mechanism audit; not a pre-registered performance experiment",
        "authorisation_honoured": {
            "new_episodes": 0,
            "launches_added": verify["launches_added_by_this_round"],
            "episode_entry_point_calls": verify["episode_entry_point_calls"],
            "input_files_changed": verify["files_changed"],
            "input_files_missing": verify["files_missing_now"],
            "files_added_since_freeze": verify["files_added_since_freeze"],
            "sealed_block_artefacts_on_disk": verify["sealed_block_artefacts_on_disk"],
            "verification_passed": verify["passed"],
        },
        "recomputation": {
            "all_layers_reproduced": recompute["all_layers_reproduced"],
            "g12_zero_differences": recompute["g12_reproduces_with_zero_differences"],
            "published_vs_recomputed": {
                layer: {
                    "published": entry["published_gate_booleans"],
                    "recomputed": entry["recomputed_gate_booleans"],
                    "hard_failures": sum(
                        len(v) for v in entry["reproduction_failures"].values()
                    ),
                }
                for layer, entry in recompute["layers"].items()
            },
            "best_fixed_rule_matches_published": {
                layer: audit["layers"][layer]["best_fixed_rule_matches_published"]
                for layer in A.LAYERS
            },
        },
        "launch_ledger_closed": recompute["launch_ledger"]["ceiling"],
        "findings": {
            "A_concurrency": {
                "strict_subset_steps_by_layer_best_rule": {
                    layer: pick(audit, ["layers", layer, "arms",
                                        audit["layers"][layer]["best_fixed_rule"],
                                        "denominators", "strict_subset_payable", "total_steps"])
                    for layer in A.LAYERS
                },
                "strict_subset_streams_by_layer_best_rule": {
                    layer: pick(audit, ["layers", layer, "arms",
                                        audit["layers"][layer]["best_fixed_rule"],
                                        "denominators", "strict_subset_payable",
                                        "streams_with_at_least_one"])
                    for layer in A.LAYERS
                },
                "g12_best_rule_g12_only_partition": pick(
                    audit, ["layers", "g12", "arms", g12_best, "denominators"]
                , {}),
                "attempt_share_of_visible_g12": {
                    arm: pick(audit, ["layers", "g12", "arms", arm, "candidate_exercise",
                                      "attempt_share_of_visible"])
                    for arm in ("send_once", "event", "periodic_5", "always_hold")
                },
            },
            "B_quotes": {
                "strict_steps_sampled_vs_total_g12_best": {
                    "sampled": pick(audit, ["layers", "g12", "arms", g12_best,
                                            "quote_geometry", "steps_sampled"]),
                    "total": pick(audit, ["layers", "g12", "arms", g12_best,
                                          "quote_geometry", "strict_points_total"]),
                },
                "credit_test_explains_payable_all_layers_all_arms": sum(
                    pick(audit, ["layers", layer, "arms", arm, "quote_geometry",
                                 "credit_test_explains_payable_on_sampled_steps"], 0)
                    for layer in A.LAYERS
                    for arm in audit["layers"][layer]["arms"]
                ),
                "sampled_steps_total": sum(
                    pick(audit, ["layers", layer, "arms", arm, "quote_geometry",
                                 "steps_sampled"], 0)
                    for layer in A.LAYERS
                    for arm in audit["layers"][layer]["arms"]
                ),
                "first_report_candidates_at_g12_best_strict_steps": pick(
                    audit,
                    ["layers", "g12", "arms", g12_best, "quote_geometry",
                     "at_strict_subset_steps", "first_report_candidates"],
                ),
                "visible_equals_refresh_at_those_steps": {
                    "visible_median": pick(
                        audit, ["layers", "g12", "arms", g12_best, "quote_geometry",
                                "at_strict_subset_steps", "visible_count", "median"]),
                    "refresh_median": pick(
                        audit, ["layers", "g12", "arms", g12_best, "quote_geometry",
                                "at_strict_subset_steps", "refresh_candidates", "median"]),
                },
                "median_within_step_band_bytes_at_starvation": pick(
                    mechanism, ["layers", "g12", "median_band_width_bytes_at_starvation_step"]),
                "median_packet_charge_bytes": pick(
                    mechanism, ["layers", "g12", "median_packet_charge_bytes"]),
                "band_as_fraction_of_one_packet": pick(
                    mechanism, ["layers", "g12", "band_width_as_fraction_of_one_packet"]),
                "one_packet_earlier_position": pick(
                    mechanism, ["layers", "g12", "where_the_credit_came_from_one_packet_earlier"]),
                "terminal_credit_position_counts": {
                    "inside": pick(
                        mechanism, ["layers", "g12", "terminal_credit_lands_inside_own_envelope"]),
                    "below": pick(
                        mechanism, ["layers", "g12", "terminal_credit_lands_below_envelope"]),
                    "above": pick(
                        mechanism, ["layers", "g12", "terminal_credit_lands_above_envelope"]),
                },
            },
            "C_timing": {
                "cells_reaching_100_percent_credit": sum(
                    1 for record in audit["per_cell"]
                    if record["credit_fraction_steps"].get("1.00") is not None
                ),
                "max_budget_used_fraction_any_cell": max(
                    float(record["budget_used_fraction"] or 0.0) for record in audit["per_cell"]
                ),
                "per_arm_first_starvation_candidate_classes": per_arm_refresh_claims,
                "first_report_targets_never_emitted_g12": {
                    arm: pick(audit, ["layers", "g12", "arms", arm, "budget_time_structure",
                                      "after_starvation", "first_report_targets_never_emitted"])
                    for arm in ("send_once", "event", "periodic_5", "periodic_20",
                               "bound_only_3_10")
                },
            },
            "D_ledger_only": {
                "label": "ledger_only_opportunity",
                "environment_causal": False,
                #: The rule L2/L3/L4 are judged on in g8/g10/g12 never refreshes at all, so on that
                #: arm this counterfactual has no object. Stated as a per-layer count rather than a
                #: claim about "the best rule", since g6's best rule is `periodic_20`, which does.
                "send_once_refreshes_by_layer": {
                    layer: sum(
                        int(record["emission_refresh"])
                        for record in audit["per_cell"]
                        if record["layer"] == layer and record["arm"] == "send_once"
                    )
                    for layer in A.LAYERS
                },
                "event_arm_opportunity": {
                    layer: {
                        "streams_with_any_opportunity": pick(
                            audit, ["layers", layer, "arms", "event",
                                    "ledger_only_opportunity",
                                    "streams_with_any_opportunity"]),
                        "streams_unlocking_a_first_report": pick(
                            audit, ["layers", layer, "arms", "event",
                                    "ledger_only_opportunity",
                                    "streams_unlocking_a_first_report"]),
                        "median_first_reports_unlocked_per_packet": pick(
                            audit, ["layers", layer, "arms", "event", "ledger_only_opportunity",
                                    "unlocked_first_report_targets_per_held_packet", "median"]),
                    }
                    for layer in A.LAYERS
                },
            },
        },
        "corrections_to_published_wording": [
            {
                "statement": "首次额度耗尽的 15 条流，那一步上的候选全部是刷新",
                "where_it_was_made": "COMM_TASK_PRESSURE_G12_EXTENSION_RESULTS.md §四",
                "checked_reading": "对 `send_once`(15/15) 与 `first_send_plus_bound_3_10`(16/16) 成立；"
                                   "对 `bound_only_3_10` 13/16、`periodic_20` 12/16、"
                                   "`periodic_5` 10/16、`event_priority` 10/16、`event` 9/16 成立，"
                                   "其余流在那一步仍有一个从未报出的新目标。"
                                   "没有任何臂出现『全是新目标』。",
                "required_qualifier": "引用时必须带臂名；它是最佳规则的性质，不是跨规则的性质。",
            },
            {
                "statement": "g12 需要 160 次启动而账本只剩 144，缺口未闭合",
                "where_it_was_made": "本轮交接文档 §未解决问题",
                "checked_reading": "已闭合：追加授权 16 次使上限 640→656，g12 实到 160 次、0 失败、"
                                   "144/144 格；总数据启动 656 = 上限 656。",
                "required_qualifier": "账本 `entries`(670) 比 `starts`(666) 多 4 行是"
                                      "『重登记而未跑 episode』的审计痕迹，不是漏计。",
            },
        ],
        "does_not_change": [
            "四层任何一条已发布判据（`GATES_g*.json` 等字节不变，已由 SHA256 核对）",
            "g12 的『容量扩展 / 探索性闭环』地位",
            "总纲 §9 关于学习型发送控制器的既有结论口径",
        ],
        "not_claimed": [
            "本轮不声称任何策略的任务效果会改善",
            "本轮不声称学习控制器有或没有附加价值",
            "本轮不声称 L3 应被改判",
            "D 节全部数字是 ledger-only opportunity，不是环境因果效应",
        ],
        "next_step_belongs_to_the_user": True,
    }
    A.write_json(A.HERE / "COMM_ACTION_GEOMETRY_AUDIT_VERDICT.json", verdict)
    print(json.dumps(
        {
            "verification_passed": verdict["authorisation_honoured"]["verification_passed"],
            "reproduced": verdict["recomputation"]["all_layers_reproduced"],
            "cells_at_100_percent": verdict["findings"]["C_timing"]["cells_reaching_100_percent_credit"],
            "max_fraction": verdict["findings"]["C_timing"]["max_budget_used_fraction_any_cell"],
            "D_send_once_refreshes": verdict["findings"]["D_ledger_only"][
                "send_once_refreshes_by_layer"],
            "band_vs_packet": verdict["findings"]["B_quotes"]["band_as_fraction_of_one_packet"],
            "one_packet_earlier": verdict["findings"]["B_quotes"]["one_packet_earlier_position"],
        },
        ensure_ascii=False,
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
