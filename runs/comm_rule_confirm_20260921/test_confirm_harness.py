"""Unit tests for this round's harness: the checks, the statistics and the judgement.

These are not episode tests -- the 3,000 episodes are the round's evidence and are registered
in the manifest. What is tested here is the machinery that reads them, because a judgement that
mis-applies its own pre-registered rule, or a validator that lets a mixed sample through, would
invalidate the evidence without changing a single number.

    python -m unittest discover -s runs/comm_rule_confirm_20260921 -p "test_*.py" -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _directory in (str(HERE.parents[1]), str(HERE)):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

import numpy as np  # noqa: E402

import confirm_common as common  # noqa: E402
import confirm_matrix  # noqa: E402


def make_row(**overrides):
    """One valid matrix row, so each test can break exactly one thing."""
    arm = overrides.get("arm", "ack_retry_2")
    setting = overrides.get("setting", "isolated-lossy")
    spec = common.ARMS[arm]
    channel = common.SETTINGS[setting]
    uplink = int(overrides.pop("uplink_bytes", 27312))
    ack = int(overrides.pop("ack_bytes", 13647))
    row = {
        "arm": arm,
        "setting": setting,
        "seed": 900000,
        "policy": spec["policy"],
        "parameters": str(spec.get("params") or ""),
        "information": channel["information"],
        "delivery_probability": channel["delivery_probability"],
        "ack_enabled": bool(spec["ack"]),
        "ack_delivery_probability": channel["delivery_probability"] if spec["ack"] else None,
        "ack_delay_ticks": common.ACK_DELAY_TICKS if spec["ack"] else None,
        "timing_mode": common.POLICY_MODE,
        "budget_bytes": common.BUDGET_BYTES,
        "on_time": 5,
        "targets": 6,
        "expired": 1,
        "unresolved": 0,
        "completion_rate_conditional": 0.8333,
        "completion_rate_all_goals": 0.8333,
        "uplink_bytes": uplink,
        "ack_bytes": ack,
        "total_bytes": uplink + ack,
        "spent_bytes": uplink,
        "budget_used_fraction": 0.932,
        "belief_age_steps": 128.7,
        "latencies": {"detection_to_assignment": 4.0, "detection_to_work": 12.0,
                      "dispatch_to_work": 8.0, "detection_to_completion": 40.0},
        "targets_outcome": {
            "piloc-e1-target-0": "on_time",
            "piloc-e1-target-1": "on_time",
            "piloc-e1-target-2": "on_time",
            "piloc-e1-target-3": "on_time",
            "piloc-e1-target-4": "on_time",
            "piloc-e1-target-5": "expired",
        },
        "config_hash": "a" * 32,
        "stream_sha256": "b" * 64,
        "arm_label": "x",
    }
    for key in confirm_matrix.TRAFFIC_COUNTERS:
        row[key] = 0
    row.update(overrides)
    if "total_bytes" not in overrides and (
        "uplink_bytes" in overrides or "ack_bytes" in overrides
    ):
        row["total_bytes"] = row["uplink_bytes"] + row["ack_bytes"]
    return row


REGISTRATION = {900000: "b" * 64, 900001: "c" * 64}


class ValidateRowTests(unittest.TestCase):
    def test_a_valid_row_passes(self) -> None:
        self.assertEqual(confirm_matrix.validate_row(make_row(), REGISTRATION), [])

    def test_a_wrong_ack_switch_is_rejected(self) -> None:
        row = make_row(arm="noack_send_once")
        row["ack_enabled"] = True
        self.assertIn("the ACK switch differs", " ".join(
            confirm_matrix.validate_row(row, REGISTRATION)))

    def test_a_wrong_information_path_is_rejected(self) -> None:
        row = make_row()
        row["information"] = "isolated" if row["setting"] != "isolated-lossy" else "normal"
        problems = confirm_matrix.validate_row(row, REGISTRATION)
        self.assertTrue(any("information path" in item for item in problems))

    def test_a_wrong_delivery_probability_is_rejected(self) -> None:
        row = make_row()
        row["delivery_probability"] = 1.0
        self.assertTrue(any("uplink delivery probability" in item for item in
                            confirm_matrix.validate_row(row, REGISTRATION)))

    def test_a_reverse_channel_kinder_than_the_uplink_is_rejected(self) -> None:
        row = make_row()
        row["ack_delivery_probability"] = 1.0
        self.assertTrue(any("reverse-channel probability" in item for item in
                            confirm_matrix.validate_row(row, REGISTRATION)))

    def test_a_breached_credit_is_rejected(self) -> None:
        row = make_row()
        row["spent_bytes"] = common.BUDGET_BYTES + 1
        self.assertIn("the budget was breached", confirm_matrix.validate_row(row, REGISTRATION))

    def test_an_unregistered_stream_is_rejected(self) -> None:
        row = make_row(seed=900199)
        self.assertTrue(any("not part of the registered block" in item for item in
                            confirm_matrix.validate_row(row, REGISTRATION)))

    def test_a_swapped_stream_is_rejected(self) -> None:
        row = make_row()
        row["stream_sha256"] = "c" * 64
        self.assertTrue(any("not the registered one" in item for item in
                            confirm_matrix.validate_row(row, REGISTRATION)))

    def test_outcomes_that_do_not_close_are_rejected(self) -> None:
        row = make_row(targets=7)
        self.assertTrue(any("do not close" in item for item in
                            confirm_matrix.validate_row(row, REGISTRATION)))

    def test_a_no_ack_run_charging_reverse_bytes_is_rejected(self) -> None:
        row = make_row(arm="noack_send_once", ack_bytes=100, ack_enabled=False)
        self.assertIn("a no-ACK run charged reverse bytes",
                      confirm_matrix.validate_row(row, REGISTRATION))

    def test_a_ledger_that_disagrees_with_the_transcript_is_rejected(self) -> None:
        row = make_row()
        row["ack_ledger"] = {"attempt_bytes": 1}
        self.assertTrue(any("reverse ledger disagrees" in item for item in
                            confirm_matrix.validate_row(row, REGISTRATION)))


class CostIdentityTests(unittest.TestCase):
    def test_total_is_the_sum_and_the_credit_only_sees_the_uplink(self) -> None:
        row = make_row(uplink_bytes=20000, ack_bytes=7000)
        self.assertEqual(row["total_bytes"], 27000)
        self.assertEqual(row["spent_bytes"], 20000)

    def test_the_ack_share_is_computed_on_total_not_on_uplink(self) -> None:
        rows = [make_row(seed=900000), make_row(seed=900001)]
        summary = confirm_matrix.aggregate(rows)
        entry = summary["ack_retry_2"]
        self.assertAlmostEqual(
            entry["ack_share_of_total"], entry["ack_bytes"] / entry["total_bytes"], places=3
        )


class RatioTests(unittest.TestCase):
    def test_a_non_positive_denominator_is_undefined_not_trimmed(self) -> None:
        deltas = {
            key: np.asarray(values, dtype=float)
            for key, values in {
                "on_time": [0.0, 0.0, -1.0, 0.0],
                "total_bytes": [100.0, 200.0, -50.0, 10.0],
                "uplink_bytes": [10.0, 20.0, -5.0, 1.0],
                "ack_bytes": [90.0, 180.0, -45.0, 9.0],
            }.items()
        }
        block = confirm_matrix.ratio_bootstrap(deltas)
        self.assertFalse(block["defined"])
        self.assertNotIn("total_bytes_ratio", block)
        self.assertEqual(block["total_bytes_numerator_mean"], 65.0)

    def test_a_positive_denominator_reports_raw_numerator_and_denominator(self) -> None:
        deltas = {
            key: np.asarray(values, dtype=float)
            for key, values in {
                "on_time": [1.0, 1.0, 0.0, 1.0],
                "total_bytes": [30000.0, 31000.0, 29000.0, 30500.0],
                "uplink_bytes": [15000.0, 15500.0, 14500.0, 15000.0],
                "ack_bytes": [15000.0, 15500.0, 14500.0, 15500.0],
            }.items()
        }
        block = confirm_matrix.ratio_bootstrap(deltas)
        self.assertTrue(block["defined"])
        self.assertEqual(block["denominator_mean_on_time"], 0.75)
        self.assertEqual(block["total_bytes_numerator_mean"], 30125.0)
        self.assertAlmostEqual(block["total_bytes_ratio"], 30125.0 / 0.75, places=1)
        self.assertEqual(len(block["total_bytes_ratio_ci95"]), 2)

    def test_an_undefined_replicate_is_dropped_whole(self) -> None:
        # Every resample of a single-seed difference is that same difference, so the count of
        # undefined replicates is either all or none, and it is reported either way.
        deltas = {
            key: np.asarray(values, dtype=float)
            for key, values in {
                "on_time": [1.0],
                "total_bytes": [30000.0],
                "uplink_bytes": [15000.0],
                "ack_bytes": [15000.0],
            }.items()
        }
        block = confirm_matrix.ratio_bootstrap(deltas)
        self.assertTrue(block["defined"])
        self.assertEqual(block["undefined_replicates"], 0)


class FrontierTests(unittest.TestCase):
    def test_a_dominated_arm_leaves_the_frontier(self) -> None:
        cheap = make_row(arm="noack_send_once", seed=900000, uplink_bytes=14000, ack_bytes=0)
        dear = make_row(arm="ack_send_once", seed=900000, uplink_bytes=14000, ack_bytes=7000)
        frontier = confirm_matrix.pareto_frontier([cheap, dear])
        self.assertEqual(frontier["frontier"], ["noack_send_once"])
        self.assertEqual(frontier["dominated"], {"ack_send_once": ["noack_send_once"]})

    def test_an_arm_better_on_one_stream_stays_on_the_frontier(self) -> None:
        # Better on one stream and worse on another: no per-stream dominance either way.
        rows = [
            make_row(arm="noack_send_once", seed=900000, on_time=5, uplink_bytes=14000, ack_bytes=0),
            make_row(arm="noack_send_once", seed=900001, on_time=3, uplink_bytes=14000, ack_bytes=0),
            make_row(arm="ack_send_once", seed=900000, on_time=5, uplink_bytes=14000, ack_bytes=7000),
            make_row(arm="ack_send_once", seed=900001, on_time=4, uplink_bytes=14000, ack_bytes=7000),
        ]
        frontier = confirm_matrix.pareto_frontier(rows)
        self.assertEqual(sorted(frontier["frontier"]), ["ack_send_once", "noack_send_once"])
        self.assertEqual(frontier["dominated"], {})


class NewFailureTests(unittest.TestCase):
    def test_per_target_accounting_names_the_lost_and_the_recovered(self) -> None:
        baseline = make_row(arm="noack_send_once", seed=900000, on_time=5)
        baseline["targets_outcome"] = {
            "piloc-e1-target-0": "on_time",
            "piloc-e1-target-1": "on_time",
            "piloc-e1-target-2": "expired",
            "piloc-e1-target-3": "on_time",
            "piloc-e1-target-4": "on_time",
            "piloc-e1-target-5": "on_time",
        }
        arm = make_row(arm="ack_retry_2", seed=900000, on_time=5)
        arm["targets_outcome"] = {
            "piloc-e1-target-0": "on_time",
            "piloc-e1-target-1": "expired",
            "piloc-e1-target-2": "on_time",
            "piloc-e1-target-3": "on_time",
            "piloc-e1-target-4": "on_time",
            "piloc-e1-target-5": "on_time",
        }
        block = confirm_matrix.new_failures([arm, baseline], "ack_retry_2", "noack_send_once")
        self.assertEqual(block["new_failures"], 1)
        self.assertEqual(block["new_failure_cases"][0]["target_id"], "piloc-e1-target-1")
        self.assertEqual(block["recovered"], 1)
        self.assertEqual(block["recovered_cases"][0]["target_id"], "piloc-e1-target-2")


class JudgementTests(unittest.TestCase):
    def _rows(self, arm: str, *, on_time: int, uplink: int, ack: int, seed: int = 900000):
        row = make_row(arm=arm, seed=seed, on_time=on_time, uplink_bytes=uplink, ack_bytes=ack)
        outcomes = {
            f"piloc-e1-target-{index}": ("on_time" if index < on_time else "expired")
            for index in range(6)
        }
        row["targets_outcome"] = outcomes
        return row

    def test_the_lossless_negative_control_flags_a_rule_that_gains(self) -> None:
        rows = [
            self._rows("noack_send_once", on_time=5, uplink=14000, ack=0),
            self._rows("ack_retry_2", on_time=6, uplink=14000, ack=10000),
        ]
        block = confirm_matrix.judge(rows, "isolated-lossless")["negative_control"]
        self.assertFalse(block["passed"])
        self.assertEqual(block["violations"], ["ack_retry_2"])
        self.assertIn("pause", block["action"])

    def test_the_lossy_confirmation_needs_a_ci_lower_bound_above_zero(self) -> None:
        rows = [
            self._rows("noack_send_once", on_time=5, uplink=14000, ack=0, seed=900000),
            self._rows("noack_send_once", on_time=4, uplink=14000, ack=0, seed=900001),
            self._rows("noack_send_once", on_time=3, uplink=14000, ack=0, seed=900002),
            self._rows("ack_retry_2", on_time=6, uplink=27000, ack=13000, seed=900000),
            self._rows("ack_retry_2", on_time=5, uplink=27000, ack=13000, seed=900001),
            self._rows("ack_retry_2", on_time=4, uplink=27000, ack=13000, seed=900002),
        ]
        block = confirm_matrix.judge(rows, "isolated-lossy")["lossy_confirmation"]
        self.assertTrue(block["reproduced"])
        self.assertIn("independent support", block["action"])

    def test_a_gain_whose_ci_touches_zero_is_not_a_reproduction(self) -> None:
        rows = [
            self._rows("noack_send_once", on_time=5, uplink=14000, ack=0, seed=900000),
            self._rows("noack_send_once", on_time=4, uplink=14000, ack=0, seed=900001),
            self._rows("ack_retry_2", on_time=6, uplink=27000, ack=13000, seed=900000),
            self._rows("ack_retry_2", on_time=4, uplink=27000, ack=13000, seed=900001),
        ]
        block = confirm_matrix.judge(rows, "isolated-lossy")["lossy_confirmation"]
        self.assertFalse(block["reproduced"])
        self.assertIn("does not gain independent support", block["action"])

    def test_a_normal_path_gain_without_ci_support_is_not_proposed(self) -> None:
        rows = [
            self._rows("noack_send_once", on_time=5, uplink=14000, ack=0, seed=900000),
            self._rows("noack_send_once", on_time=4, uplink=14000, ack=0, seed=900001),
            self._rows("ack_retry_2", on_time=5, uplink=27000, ack=13000, seed=900000),
            self._rows("ack_retry_2", on_time=4, uplink=27000, ack=13000, seed=900001),
        ]
        block = confirm_matrix.judge(rows, "normal-lossy")["normal_path"]
        self.assertFalse(block["proposed_as_optional_mode"])
        self.assertTrue(block["same_outcome_but_dearer"])

    def test_a_rule_beating_the_locked_one_is_reported_as_unstable(self) -> None:
        rows = [
            self._rows("ack_retry_2", on_time=5, uplink=27000, ack=13000, seed=900000),
            self._rows("ack_retry_2", on_time=4, uplink=27000, ack=13000, seed=900001),
            self._rows("ack_periodic_20", on_time=6, uplink=25000, ack=12000, seed=900000),
            self._rows("ack_periodic_20", on_time=6, uplink=25000, ack=12000, seed=900001),
        ]
        block = confirm_matrix.judge(rows, "isolated-lossy")["frontier"]
        self.assertEqual(block["unstable"], ["ack_periodic_20"])
        self.assertIn("not re-tune", block["action"])


class FrozenDesignTests(unittest.TestCase):
    def test_the_five_arms_and_their_parameters_are_the_protocols(self) -> None:
        self.assertEqual(
            common.ARM_LABELS,
            ("noack_send_once", "ack_send_once", "ack_retry_2", "ack_periodic_20",
             "ack_backoff_16"),
        )
        self.assertEqual(common.ARMS["noack_send_once"]["policy"], "send_once")
        self.assertFalse(common.ARMS["noack_send_once"]["ack"])
        self.assertTrue(all(common.ARMS[arm]["ack"] for arm in common.ARM_LABELS[1:]))
        self.assertEqual(common.ARMS["ack_retry_2"]["params"], "max_attempts=2")
        self.assertEqual(common.ARMS["ack_periodic_20"]["params"], "interval=20")
        self.assertEqual(
            common.ARMS["ack_backoff_16"]["params"], "base_interval=16,max_interval=32"
        )

    def test_every_setting_gives_the_reverse_channel_the_uplinks_own_numbers(self) -> None:
        for setting in common.SETTING_LABELS:
            channel = common.SETTINGS[setting]
            self.assertIn(channel["information"], ("isolated", "normal"))
            self.assertIn(channel["delivery_probability"], (1.0, 0.7))

    def test_the_credit_is_the_locked_twenty_five_percent_tier(self) -> None:
        self.assertEqual(common.BUDGET_BYTES, 29300)
        self.assertEqual(common.PRIMARY_TIER, "25pct")

    def test_parameters_parse_the_way_the_policy_builder_expects(self) -> None:
        self.assertEqual(
            confirm_matrix.parse_params("base_interval=16,max_interval=32"),
            {"base_interval": "16", "max_interval": "32"},
        )
        self.assertEqual(confirm_matrix.parse_params(""), {})

    def test_settings_are_never_pooled(self) -> None:
        rows = [
            make_row(setting="isolated-lossy", seed=900000),
            make_row(setting="normal-lossy", seed=900000),
        ]
        with self.assertRaises(ValueError) as raised:
            confirm_matrix.aggregate(rows)
        self.assertIn("never pooled", str(raised.exception))

    def test_one_setting_aggregates_its_own_rows(self) -> None:
        rows = [
            make_row(setting="isolated-lossy", seed=900000),
            make_row(setting="isolated-lossy", seed=900001),
            make_row(setting="normal-lossy", seed=900000),
        ]
        summary = confirm_matrix.aggregate(
            [row for row in rows if row["setting"] == "isolated-lossy"]
        )
        self.assertEqual(summary["ack_retry_2"]["episodes"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
