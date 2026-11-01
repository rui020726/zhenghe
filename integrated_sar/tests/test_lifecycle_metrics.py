from __future__ import annotations

import unittest

from integrated_sar.lifecycle_metrics import parse_run_path, window_metrics


def _detection(step: float, target: str, deadline_offset: float = 100.0) -> dict:
    return {
        "event_type": "detection",
        "timestamp": step,
        "target_id": target,
        "detail": {"grid_position": [1, 2], "deadline_offset": deadline_offset},
    }


def _assignment(step: float, target: str) -> dict:
    return {"event_type": "assignment", "timestamp": step, "target_id": target}


def _completed(step: float, target: str) -> dict:
    return {"event_type": "progress", "timestamp": step, "target_id": target, "status": "completed"}


class WindowMetricsTests(unittest.TestCase):
    def test_discovered_is_the_detection_count_not_the_final_census(self) -> None:
        records = [_detection(5.0, "t0"), _completed(20.0, "t0")]
        metrics = window_metrics(records, window=500)
        self.assertEqual(metrics.discovered, 1)
        self.assertEqual(metrics.completed, 1)
        self.assertAlmostEqual(metrics.completion_rate, 1.0)

    def test_the_four_categories_partition_the_discovered_set(self) -> None:
        records = [
            _detection(1.0, "done"), _completed(10.0, "done"),
            _detection(2.0, "late-deadline", deadline_offset=5.0),  # expires at 7
            _detection(3.0, "assigned-active", deadline_offset=5000.0),
            _assignment(4.0, "assigned-active"),
            _detection(6.0, "never-assigned", deadline_offset=5000.0),
        ]
        metrics = window_metrics(records, window=500)
        self.assertEqual(metrics.discovered, 4)
        self.assertEqual(metrics.completed, 1)
        self.assertEqual(metrics.expired, 1)
        self.assertEqual(metrics.active, 1)
        self.assertEqual(metrics.unassigned, 1)
        self.assertEqual(
            metrics.completed + metrics.expired + metrics.active + metrics.unassigned,
            metrics.discovered,
        )

    def test_deadline_is_a_micro_average_over_completed_only(self) -> None:
        # A run that completed nothing has deadline_rate None, so the CLI's micro
        # sum skips it instead of counting a fake 0.0 the way a macro mean did.
        empty = window_metrics([_detection(1.0, "t0", deadline_offset=5.0)], window=500)
        self.assertEqual(empty.completed, 0)
        self.assertIsNone(empty.deadline_rate)

        mixed = window_metrics(
            [
                _detection(1.0, "ontime", deadline_offset=50.0), _completed(20.0, "ontime"),
                _detection(1.0, "late", deadline_offset=10.0), _completed(50.0, "late"),
            ],
            window=500,
        )
        self.assertEqual(mixed.completed, 2)
        self.assertEqual(mixed.on_time, 1)
        self.assertAlmostEqual(mixed.deadline_rate, 0.5)

    def test_events_after_the_window_are_invisible(self) -> None:
        records = [
            _detection(1.0, "early"), _completed(2.0, "early"),
            _detection(600.0, "late"), _completed(601.0, "late"),
        ]
        metrics = window_metrics(records, window=500)
        self.assertEqual(metrics.discovered, 1)
        self.assertEqual(metrics.completed, 1)
        self.assertEqual(metrics.assignment_commands, 0)

    def test_a_target_completed_after_its_deadline_is_not_on_time(self) -> None:
        records = [
            _detection(1.0, "slow", deadline_offset=10.0),
            _completed(50.0, "slow"),
        ]
        metrics = window_metrics(records, window=500)
        self.assertEqual(metrics.completed, 1)
        self.assertEqual(metrics.on_time, 0)

    def test_parse_run_path_extracts_arm_and_seed(self) -> None:
        self.assertEqual(parse_run_path("rp_rs0.25_s10003.jsonl"), ("rs0.25", 10003))
        self.assertEqual(parse_run_path("ab_nocoord_s7.jsonl"), ("nocoord", 7))
        with self.assertRaises(ValueError):
            parse_run_path("something_else.jsonl")


if __name__ == "__main__":
    unittest.main()
