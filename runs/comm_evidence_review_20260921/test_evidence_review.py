"""Synthetic regressions for the report pipeline this round changed.

The brief requires exactly these five behaviours to be pinned by tests, because each one was a
real defect in the archived report:

* a byte difference must never render as "better" (only 更高/相等/更低);
* the channel field is a *success* probability, so the labels must not read "丢包 1.0";
* the four units (个/ep, B/ep, B/新增任务, 完成率) must not be interchanged;
* a ratio whose denominator crosses zero stays undefined or conditional, never a clean CI;
* one statistic object, rendered twice: the judgement and the verdict must not re-draw.

Plus the two invariance guards this round must not violate: the 3,000-row summary is unchanged,
and the revision changes no pre-registered judgement.

    python -m unittest discover -s . -p 'test_*.py' -v
"""

from __future__ import annotations

import json
import re
import unittest

import numpy as np

import review_common as common
import revise_results as revise


def rows(overrides: list[tuple[str, int, int, int]]) -> list[dict]:
    """Minimal synthetic (arm, seed, on_time, total_bytes) rows, all other fields inert."""
    return [
        {
            "arm": arm,
            "seed": seed,
            "on_time": on_time,
            "total_bytes": total,
            "uplink_bytes": total,
            "ack_bytes": 0,
        }
        for arm, seed, on_time, total in overrides
    ]


def axes(effect_higher: bool = True) -> dict:
    block = {
        "arm": "a",
        "baseline": "b",
        "key": "on_time",
        "pairs": 200,
        "mean_difference": 0.8 if effect_higher else -0.2,
        "ci95_lower": 0.6 if effect_higher else -0.3,
        "ci95_upper": 1.0 if effect_higher else -0.1,
        "seeds_better": 150,
        "seeds_equal": 40,
        "seeds_worse": 10,
    }
    return {"on_time": revise.effect_entry(block), "total_bytes": block}


class CostDirectionTests(unittest.TestCase):
    def test_cost_axis_never_says_better(self) -> None:
        block = revise.cost_entry(
            {
                "arm": "ack_retry_2",
                "baseline": "noack_send_once",
                "key": "total_bytes",
                "pairs": 200,
                "mean_difference": 26354.1,
                "ci95_lower": 26002.3,
                "ci95_upper": 26687.8,
                "seeds_better": 200,
                "seeds_worse": 0,
                "seeds_equal": 0,
            }
        )
        self.assertEqual(revise._cost_counts(block), "200 更高 / 0 相等 / 0 更低")
        self.assertNotIn("更好", revise._cost_counts(block))
        self.assertEqual(block["direction"], "supported_dearer")
        self.assertEqual(revise._direction(block), "更贵（CI 下界 > 0）")

    def test_effect_axis_keeps_better_worse(self) -> None:
        block = revise.effect_entry(
            {
                "arm": "a",
                "baseline": "b",
                "key": "on_time",
                "pairs": 3,
                "mean_difference": 1.0,
                "ci95_lower": 0.5,
                "ci95_upper": 1.5,
                "seeds_better": 2,
                "seeds_worse": 0,
                "seeds_equal": 1,
            }
        )
        self.assertEqual(revise._effect_counts(block), "2 更好 / 1 相等 / 0 更差")

    def test_negative_byte_difference_is_lower_not_worse(self) -> None:
        block = revise.cost_entry(
            {
                "arm": "ack_periodic_20",
                "baseline": "ack_retry_2",
                "key": "total_bytes",
                "pairs": 200,
                "mean_difference": -3569.0,
                "ci95_lower": -4278.5,
                "ci95_upper": -2875.4,
                "seeds_better": 63,
                "seeds_worse": 137,
                "seeds_equal": 0,
            }
        )
        self.assertEqual(block["higher_streams"], 63)
        self.assertEqual(block["lower_streams"], 137)
        self.assertEqual(block["direction"], "supported_cheaper")


class ProbabilityLabelTests(unittest.TestCase):
    def test_labels_read_as_success_rates(self) -> None:
        self.assertEqual(
            common.channel_label("isolated-lossless"),
            "上行与反向投递成功率均 1.0（丢包率 0.0）",
        )
        self.assertEqual(
            common.channel_label("isolated-lossy"),
            "上行与反向投递成功率均 0.7（丢包率 0.3）",
        )

    def test_the_wording_defect_is_not_republished(self) -> None:
        text = common.json_path("COMM_RULE_CONFIRMATION_RESULTS_REVISED.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("丢包均", text)
        self.assertIn("成功率均 0.7（丢包率 0.3）", text)
        self.assertEqual(
            common.read_json("COMM_RULE_CONFIRMATION_RESULTS_REVISED.json")[
                "channel_semantics"
            ]["field"],
            "delivery_probability",
        )


class UnitTests(unittest.TestCase):
    def test_ratio_unit_is_per_additional_target(self) -> None:
        payload = common.read_json("COMM_RULE_CONFIRMATION_RESULTS_REVISED.json")
        check = payload["primary_ratio_check"]
        self.assertIn("B per additional on-time target", check["unit"])
        self.assertIn("不是 B/episode", check["unit"])
        # the point estimate must equal raw numerator / raw denominator, both printed
        self.assertAlmostEqual(
            check["raw_numerator_B_per_episode"] / check["raw_denominator_targets_per_episode"],
            check["point_estimate"],
            places=0,
        )

    def test_units_legend_names_all_four(self) -> None:
        legend = common.read_json("COMM_RULE_CONFIRMATION_RESULTS_REVISED.json")["units"]
        for token in ("个任务/episode", "B/episode", "B/新增按时任务", "完成率"):
            self.assertIn(token, legend)


class RatioDenominatorTests(unittest.TestCase):
    def test_mean_denominator_non_positive_stays_undefined(self) -> None:
        deltas = {
            "on_time": np.array([1.0, -1.0, 0.0, 0.0] * 50),
            "total_bytes": np.array([10.0, -5.0, 3.0, 2.0] * 50),
            "uplink_bytes": np.array([10.0, -5.0, 3.0, 2.0] * 50),
            "ack_bytes": np.zeros(200),
        }
        block = revise.ratio_block(deltas, axes())
        self.assertFalse(block["defined"])
        self.assertEqual(block["interval_kind"], "not_applicable")
        self.assertFalse(block["robustly_interpretable"])
        self.assertIn("undefined", block["note"])

    def test_dropped_replicates_make_a_conditional_interval(self) -> None:
        # A mean barely above zero, built from a few +/- unit swings: some resamples of it
        # land on the wrong side of zero, which is the case the interval must be labelled for.
        completion = np.array([1.0] * 3 + [-1.0] * 2 + [0.0] * 195)
        deltas = {
            "on_time": completion,
            "total_bytes": np.full(200, 1000.0),
            "uplink_bytes": np.full(200, 600.0),
            "ack_bytes": np.full(200, 400.0),
        }
        block = revise.ratio_block(deltas, axes())
        self.assertTrue(block["defined"])
        self.assertGreater(block["undefined_replicates"], 0)
        self.assertEqual(block["interval_kind"], "conditional")
        self.assertFalse(block["robustly_interpretable"])
        self.assertIn("条件区间", block["note"])

    def test_full_defined_and_supported_is_unconditional(self) -> None:
        deltas = {
            "on_time": np.abs(np.random.default_rng(7).integers(0, 3, 200)).astype(float) + 0.0,
            "total_bytes": np.full(200, 500.0),
            "uplink_bytes": np.full(200, 300.0),
            "ack_bytes": np.full(200, 200.0),
        }
        deltas["on_time"][0] = 1.0
        block = revise.ratio_block(deltas, axes())
        self.assertTrue(block["defined"])
        self.assertEqual(block["interval_kind"], "unconditional")
        self.assertEqual(block["undefined_replicates"], 0)
        self.assertTrue(block["robustly_interpretable"])


class SingleObjectRenderingTests(unittest.TestCase):
    """The §3.5 defect: the same quantity drawn twice and reported with two different CIs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = common.read_json("COMM_RULE_CONFIRMATION_RESULTS_REVISED.json")

    def test_judgement_slices_the_comparison_blocks(self) -> None:
        for setting, judgement in self.payload["judgement"].items():
            comparisons = self.payload["comparisons"][setting]
            for section, key in (
                ("lossy_confirmation", "ack_retry_2 - noack_send_once"),
                ("normal_path", "ack_retry_2 - noack_send_once"),
            ):
                if section not in judgement:
                    continue
                for axis, field in (("on_time", "completion"), ("total_bytes", "total_bytes")):
                    self.assertEqual(
                        judgement[section][field]["ci95_lower"],
                        comparisons[key]["axes"][axis]["ci95_lower"],
                        f"{setting}/{section}/{axis} lower bound drifted",
                    )
                    self.assertEqual(
                        judgement[section][field]["ci95_upper"],
                        comparisons[key]["axes"][axis]["ci95_upper"],
                        f"{setting}/{section}/{axis} upper bound drifted",
                    )
            for arm, entry in judgement["frontier"]["checks"].items():
                source = comparisons[f"{arm} - ack_retry_2"]["axes"]["on_time"]
                self.assertEqual(entry["completion"]["ci95_upper"], source["ci95_upper"])
            for arm, entry in judgement.get("negative_control", {}).get("arms", {}).items():
                source = comparisons[f"{arm} - noack_send_once"]["axes"]["on_time"]
                self.assertEqual(entry["completion"]["ci95_upper"], source["ci95_upper"])

    def test_the_archived_double_draw_is_recorded_not_denied(self) -> None:
        diff = self.payload["old_output_diff"]
        self.assertGreater(diff["checked_pairs"], 0)
        self.assertTrue(diff["differing"])
        self.assertFalse(diff["means_changed"])
        for item in diff["differing"]:
            self.assertEqual(item["old_mean"], item["new_mean"])
            self.assertNotEqual(item["old_ci"], item["new_ci"])

    def test_rebuild_is_deterministic(self) -> None:
        rebuilt = revise.build_payload()
        self.assertEqual(
            json.dumps(rebuilt["comparisons"], sort_keys=True),
            json.dumps(self.payload["comparisons"], sort_keys=True),
        )
        self.assertEqual(
            json.dumps(rebuilt["judgement"], sort_keys=True, default=str),
            json.dumps(self.payload["judgement"], sort_keys=True, default=str),
        )

    def test_aggregate_summary_unchanged(self) -> None:
        invariance = self.payload["aggregate_invariance"]
        self.assertTrue(invariance["identical"], invariance["drifted"][:3])
        self.assertGreater(invariance["fields_checked"], 500)
        self.assertEqual(self.payload["inputs"]["rows"], 3000)
        self.assertEqual(self.payload["episodes"], 3000)


class FrontierWordingTests(unittest.TestCase):
    def test_no_stability_label_and_no_equivalence_claim(self) -> None:
        text = common.json_path("COMM_RULE_CONFIRMATION_RESULTS_REVISED.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("规则前沿：稳定", text)
        self.assertNotIn("规则前沿：不稳定", text)
        self.assertIn("完成数未检出差异（CI 含 0）", text)
        self.assertIn("不能称效果相同", text)
        for definition in ("样本均值 Pareto 前沿", "逐流支配", "有统计支持的支配"):
            self.assertIn(definition, text)

    def test_non_inferiority_branch_reports_both_ways(self) -> None:
        payload = common.read_json("COMM_RULE_CONFIRMATION_RESULTS_REVISED.json")
        lossy = payload["judgement"]["isolated-lossy"]["frontier"]["checks"]["ack_periodic_20"]
        self.assertIn("未通过 −0.05 个/ep 的非劣界", lossy["reading"])
        self.assertFalse(lossy["completion"]["passes_non_inferiority"])
        normal = payload["judgement"]["normal-lossy"]["frontier"]["checks"]["ack_periodic_20"]
        self.assertIn("通过了 −0.05 个/ep 的非劣界", normal["reading"])
        self.assertIn("不建立等效", normal["reading"])


class AckObjectTests(unittest.TestCase):
    def test_round6_baseline_is_ack_on_and_unconsumed(self) -> None:
        payload = common.read_json("COMM_RULE_CONFIRMATION_RESULTS_REVISED.json")
        objects = {row["arm"]: row for row in payload["ack_comparison_objects"]}
        round6 = next(
            row for row in payload["ack_comparison_objects"] if "第六轮矩阵臂" in row["arm"]
        )
        self.assertEqual(round6["ack_switch"], "开")
        self.assertIn("否", round6["consumes_ack_feedback"])
        self.assertEqual(objects["noack_send_once（本轮基线）"]["ack_switch"], "关")

    def test_the_three_cost_differences_are_kept_apart(self) -> None:
        verdict = common.read_json("COMMUNICATION_LEARNING_VERDICT_REVISED.json")
        differences = verdict["conclusions"]["ack_task_gain_and_total_cost"][
            "cost_comparisons_are_different"
        ]
        self.assertEqual(len(differences), 3)
        baselines = {row["comparison"] for row in differences}
        self.assertEqual(len(baselines), 3)


class ForbiddenClaimsTests(unittest.TestCase):
    def test_packaged_claims_are_listed_and_absent_from_prose(self) -> None:
        verdict = common.read_json("COMMUNICATION_LEARNING_VERDICT_REVISED.json")
        claims = verdict["conclusions"]["learning_negative"]["not_claimed"]
        for claim in ("ACK 模型训练失败", "扩训已排除欠训练", "96.5 % 的所有决策没有学习空间"):
            self.assertIn(claim, claims)
        text = common.json_path("COMMUNICATION_LEARNING_VERDICT_REVISED.md").read_text(
            encoding="utf-8"
        )
        # the archived wording may only be quoted where the document is quoting it (the
        # forbidden list and the old-to-new table); it must not be asserted as a claim.
        claims = text.split("## 三、")[0]
        self.assertNotIn("已用扩训排除", claims)
        self.assertNotIn("欠训练（已用扩训排除）", claims)
        self.assertIn("不排除仍然欠训练", claims)

    def test_three_learning_facts_stay_separate(self) -> None:
        verdict = common.read_json("COMMUNICATION_LEARNING_VERDICT_REVISED.json")
        facts = verdict["conclusions"]["learning_negative"]["three_facts_stated_separately"]
        self.assertEqual(len(facts), 3)
        kinds = {fact["fact"] for fact in facts}
        self.assertTrue(any("no-ACK" in fact for fact in kinds))
        self.assertTrue(any("未训练" in fact for fact in kinds))
        self.assertTrue(any("没有学习臂" in fact or "无学习臂" in " ".join(kinds) for fact in kinds))

    def test_gate_3_is_scoped_as_a_sample(self) -> None:
        verdict = common.read_json("COMMUNICATION_LEARNING_VERDICT_REVISED.json")
        gate = verdict["conclusions"]["ack_task_gain_and_total_cost"]["gate_3_scope"]
        self.assertIn("资源门槛", gate["thresholds"]["nature"])
        self.assertIn("93", gate["sampling_history"])
        self.assertIn("425", gate["sampling_history"])
        self.assertNotIn("没有学习空间", gate["what_it_does_not_say"].replace(
            "不写'96.5 % 的所有决策没有学习空间'", ""
        ))


class DiagnosticTests(unittest.TestCase):
    def test_missing_items_are_marked_missing(self) -> None:
        data = common.read_json("COMM_LEARNING_DIAGNOSTIC.json")
        self.assertEqual(len(data["runs"]), 6)
        for run in data["runs"].values():
            self.assertEqual(run["value_loss_saved"], "missing")
            self.assertGreater(run["optimizer_steps_cumulative"], 0)
            self.assertNotEqual(
                run["environment_steps"], run["optimizer_steps_cumulative"]
            )
        self.assertTrue(data["ack_resend_question"]["listed_as_to_measure"])
        self.assertFalse(data["ack_resend_question"]["per_point_transcripts_retained"])

    def test_four_explanations_coexist(self) -> None:
        data = common.read_json("COMM_LEARNING_DIAGNOSTIC.json")
        keys = [key for key in data["explanations"] if not key.startswith("允许")]
        self.assertEqual(
            sorted(keys),
            sorted(
                [
                    "undertraining",
                    "insufficient_observation",
                    "temporal_credit_assignment",
                    "strong_rule_at_reachable_effect",
                ]
            ),
        )
        for key in keys:
            block = data["explanations"][key]
            self.assertTrue(block["supporting"] and block["contradicting"] and block["missing"])
            self.assertNotIn("是原因", block["status"])


class SourceUntouchedTests(unittest.TestCase):
    def test_revised_documents_do_not_claim_new_data(self) -> None:
        corrections = common.read_json("COMM_EVIDENCE_CORRECTIONS.json")
        self.assertEqual(corrections["episodes_run"], 0)
        self.assertEqual(corrections["streams_recorded"], 0)
        self.assertEqual(corrections["training_runs"], 0)
        verdict = common.read_json("COMMUNICATION_LEARNING_VERDICT_REVISED.json")
        self.assertFalse(verdict["preserved"]["gate_criteria_redefined"])
        self.assertFalse(verdict["preserved"]["production_defaults_changed"])
        self.assertEqual(verdict["preserved"]["new_episodes"], 0)


if __name__ == "__main__":
    unittest.main()
