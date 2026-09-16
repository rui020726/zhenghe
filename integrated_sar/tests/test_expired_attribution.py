"""归因与 oracle 上界的回归测试。

用合成生命周期日志（而非真实 run）验证：分类不会把"做完但迟到"误判成
"发现时就不可行"，进度倒扣（阶段重置）能被发现，oracle 曲线按定义单调。

``StagePlan`` 由测试直接注入，避免依赖位置哈希抽到的具体随机档位。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from integrated_sar.contracts import GridPosition
from integrated_sar.expired_attribution import (
    MAX_TRAVEL_STEPS,
    Attribution,
    ProgressPoint,
    StagePlan,
    TargetTimeline,
    attribute,
    build_timelines,
    classify,
    communication_facts,
    stage_plan_for,
)


def _plan(durations=(5.0, 5.0), offset=60.0) -> StagePlan:
    return StagePlan(
        durations=tuple(durations),
        requirements=tuple((1, 0, 0, 0, 0) for _ in durations),
        deadline_offset=offset,
        severity=0.5,
    )


def _provider(plan: StagePlan):
    def provider(target_id: str, position: GridPosition, detected_at: float) -> StagePlan:
        return plan

    return provider


def _detection(target: str, at: float, row: int = 3, col: int = 4, offset: float = 60.0) -> dict:
    return {
        "target_id": target,
        "event_type": "detection",
        "timestamp": at,
        "discovered_by": "agent_0",
        "detail": {"grid_position": [row, col], "deadline_offset": offset},
    }


def _progress(
    target: str,
    at: float,
    progress: float,
    *,
    stage: int = 0,
    members=("0",),
    status="in_progress",
    remaining=None,
) -> dict:
    """One progress row; ``remaining`` is the current stage's unmet requirement.

    Omitted by default so the older-log case (the field is absent) stays testable.
    """
    record = {
        "target_id": target,
        "event_type": "progress",
        "timestamp": at,
        "assigned_agents": list(members),
        "stage_index": stage,
        "progress": progress,
        "status": status,
        "detail": {"expired": False},
    }
    if remaining is not None:
        record["remaining_requirements"] = list(remaining)
    return record


def _assignment(target: str, at: float, agent: str = "0") -> dict:
    return {
        "target_id": target,
        "event_type": "assignment",
        "timestamp": at,
        "assigned_agents": [agent],
        "detail": {"step": int(at)},
    }


def _final(target: str, at: float, progress: float, status: str) -> dict:
    return {
        "target_id": target,
        "event_type": "final",
        "timestamp": at,
        "progress": progress,
        "status": status,
        "failure_reason": "deadline_expired" if status == "expired" else None,
    }


class TimelineTests(unittest.TestCase):
    def test_first_work_is_the_first_step_progress_actually_moved(self) -> None:
        records = [
            _detection("t0", 10.0),
            _progress("t0", 10.0, 0.0, members=()),
            _assignment("t0", 12.0),
            _progress("t0", 40.0, 0.0, members=("0",)),
            _progress("t0", 41.0, 0.25),
            _progress("t0", 42.0, 0.5),
        ]
        timeline = build_timelines(records, arm="a", seed=1, plan_provider=_provider(_plan()))["t0"]
        self.assertEqual(timeline.assignment_latency, 2.0)
        self.assertEqual(timeline.first_work, 41.0)
        self.assertEqual(timeline.delivery_latency, 31.0)
        self.assertEqual(timeline.worked_steps, 2)
        self.assertEqual(timeline.work_needed, 10.0)

    def test_a_stage_reset_is_visible_as_a_setback(self) -> None:
        records = [
            _detection("t0", 0.0),
            _progress("t0", 1.0, 0.0),
            _progress("t0", 2.0, 0.4),
            _progress("t0", 3.0, 0.0, status="assigned"),  # member left, stage refilled
            _progress("t0", 4.0, 0.2),
        ]
        timeline = build_timelines(records, arm="a", seed=1, plan_provider=_provider(_plan()))["t0"]
        self.assertEqual(timeline.progress_setbacks, 1)
        self.assertAlmostEqual(timeline.discarded_work_steps, 4.0)  # 0.4 * 10


class ShortGroupStallTests(unittest.TestCase):
    """A group that is on site but cannot work is idle, and must be measurable."""

    def _timeline(self, records: list[dict]) -> TargetTimeline:
        return build_timelines(records, arm="a", seed=1, plan_provider=_provider(_plan()))["t0"]

    def test_the_stall_is_integrated_between_rows_not_counted_per_row(self) -> None:
        records = [
            _detection("t0", 0.0),
            # Two members arrive and stay, but abilities 3 and 4 are still missing, so
            # the stage clock never starts: the run writes no further row for 51 steps.
            _progress("t0", 20.0, 0.0, members=("0", "1"), remaining=(0, 0, 0, 1, 1)),
            _progress("t0", 71.0, 0.0, members=("0", "1", "5"), remaining=(0, 0, 0, 0, 0)),
            _progress("t0", 80.0, 0.25, members=("0", "1", "5")),
        ]
        self.assertAlmostEqual(self._timeline(records).short_group_idle_steps, 51.0)

    def test_queueing_without_members_or_with_the_requirement_met_is_not_a_stall(self) -> None:
        records = [
            _detection("t0", 0.0),
            # Nobody on site yet: waiting for a robot is queueing, not a short group.
            _progress("t0", 10.0, 0.0, members=(), remaining=(1, 0, 0, 0, 0)),
            # On site with the requirement already met: this time is spent working.
            _progress("t0", 30.0, 0.5, members=("0",), remaining=(0, 0, 0, 0, 0)),
            _progress("t0", 40.0, 0.75, members=("0",), remaining=(0, 0, 0, 0, 0)),
        ]
        self.assertAlmostEqual(self._timeline(records).short_group_idle_steps, 0.0)

    def test_a_row_without_the_remaining_field_is_not_assumed_blocked(self) -> None:
        records = [
            _detection("t0", 0.0),
            _progress("t0", 10.0, 0.0, members=("0",)),  # no remaining_requirements
            _progress("t0", 40.0, 0.5, members=("0",)),
        ]
        self.assertAlmostEqual(self._timeline(records).short_group_idle_steps, 0.0)

    def test_stall_stats_keep_expired_and_completed_targets_apart(self) -> None:
        blocked = self._timeline(
            [
                _detection("t0", 0.0),
                _progress("t0", 10.0, 0.0, members=("0",), remaining=(0, 0, 0, 1, 1)),
                _progress("t0", 70.0, 0.0, members=("0",), remaining=(0, 0, 0, 1, 1)),
                _final("t0", 70.0, 0.0, "expired"),
            ]
        )
        healthy = self._timeline(
            [
                _detection("t0", 0.0),
                _progress("t0", 10.0, 0.0, members=("0",), remaining=(0, 0, 0, 0, 0)),
                _progress("t0", 12.0, 0.5, members=("0",), remaining=(0, 0, 0, 0, 0)),
                _final("t0", 20.0, 1.0, "completed"),
            ]
        )
        attribution = Attribution(
            arm="a",
            window=200.0,
            seeds=(1,),
            expired=(blocked,),
            completed=(healthy,),
            counts={},
        )
        stats = attribution.stall_stats()
        self.assertEqual(stats["expired"]["n"], 1.0)
        self.assertAlmostEqual(stats["expired"]["median"], 60.0)
        self.assertAlmostEqual(stats["expired"]["share_ge_30"], 1.0)
        self.assertAlmostEqual(stats["completed"]["share_ge_10"], 0.0)
        self.assertAlmostEqual(blocked.as_row()["short_group_idle_steps"], 60.0)


class ClassificationTests(unittest.TestCase):
    def _timeline(self, records: list[dict]) -> TargetTimeline:
        return build_timelines(records, arm="a", seed=1, plan_provider=_provider(_plan()))["t0"]

    def test_work_finished_past_the_deadline_is_not_called_infeasible(self) -> None:
        records = [
            _detection("t0", 0.0, offset=20.0),
            _assignment("t0", 1.0),
            _progress("t0", 30.0, 0.5),
            _progress("t0", 40.0, 1.0, stage=1, status="completed"),
            _final("t0", 500.0, 1.0, "expired"),
        ]
        self.assertEqual(classify(self._timeline(records)), "finished_after_deadline")

    def test_work_longer_than_the_budget_is_infeasible_at_discovery(self) -> None:
        plan = _plan(durations=(30.0, 30.0), offset=45.0)  # 60 s of work, 45 s budget
        timeline = build_timelines(
            [_detection("t0", 0.0, offset=45.0)], arm="a", seed=1, plan_provider=_provider(plan)
        )["t0"]
        self.assertEqual(classify(timeline), "infeasible_at_discovery")

    def test_a_target_nobody_was_ever_sent_to_is_never_assigned(self) -> None:
        records = [_detection("t0", 0.0), _final("t0", 500.0, 0.0, "expired")]
        self.assertEqual(classify(self._timeline(records)), "never_assigned")

    def test_members_on_site_without_progress_is_capability_blocked(self) -> None:
        records = [_detection("t0", 0.0), _assignment("t0", 1.0)]
        records += [_progress("t0", float(step), 0.0, members=("2",)) for step in range(2, 40)]
        records.append(_final("t0", 500.0, 0.0, "expired"))
        self.assertEqual(classify(self._timeline(records)), "capability_blocked")

    def test_work_that_stops_long_before_the_deadline_is_stopped_mid_work(self) -> None:
        records = [
            _detection("t0", 0.0, offset=200.0),
            _assignment("t0", 1.0),
            _progress("t0", 2.0, 0.4),
            _progress("t0", 3.0, 0.5),
            _final("t0", 500.0, 0.5, "expired"),
        ]
        self.assertEqual(classify(self._timeline(records)), "stopped_mid_work")


class OracleTests(unittest.TestCase):
    def _attribution(self, expired: list[TargetTimeline], completed: list[TargetTimeline]) -> Attribution:
        return Attribution(
            arm="a",
            window=1000.0,
            seeds=(1,),
            expired=tuple(expired),
            completed=tuple(completed),
            counts={},
        )

    def _timeline(self, offset: float, work: float, latency: float | None) -> TargetTimeline:
        plan = _plan(durations=(work / 2.0, work / 2.0), offset=offset)
        records = [_detection("t0", 0.0, offset=offset), _assignment("t0", 1.0)]
        if latency is not None:
            records.append(_progress("t0", latency, 0.5))
        records.append(_final("t0", 1000.0, 0.0, "expired"))
        return build_timelines(records, arm="a", seed=1, plan_provider=_provider(plan))["t0"]

    def test_the_oracle_curve_is_monotone_and_uses_the_work_budget(self) -> None:
        # budgets: 60-10=50 and 60-55=5
        expired = [self._timeline(60.0, 10.0, 30.0), self._timeline(60.0, 55.0, 30.0)]
        attribution = self._attribution(expired, [])
        self.assertEqual(attribution.oracle_completion(1.0), 1.0)
        self.assertAlmostEqual(attribution.oracle_completion(20.0), 0.5)
        self.assertAlmostEqual(attribution.oracle_completion(MAX_TRAVEL_STEPS), 0.5)

    def test_provably_infeasible_counts_only_work_beyond_the_budget(self) -> None:
        expired = [self._timeline(60.0, 10.0, 30.0), self._timeline(60.0, 70.0, 30.0)]
        attribution = self._attribution(expired, [])
        self.assertEqual(attribution.provably_infeasible, 1)
        self.assertEqual(attribution.infeasible_with_travel, 1)


class StagePlanTests(unittest.TestCase):
    def test_the_plan_is_recomputed_from_the_grid_cell(self) -> None:
        first = stage_plan_for("t0", GridPosition(row=7, col=11), 0.0)
        second = stage_plan_for("whatever-the-track-id", GridPosition(row=7, col=11), 30.0)
        self.assertEqual(first.durations, second.durations)
        self.assertEqual(first.requirements, second.requirements)
        self.assertAlmostEqual(first.deadline_offset, second.deadline_offset)
        self.assertTrue(1 <= len(first.durations) <= 3)


class LogIntegrationTests(unittest.TestCase):
    def test_attribute_reads_a_lifecycle_log_and_finds_the_expired_target(self) -> None:
        records = [
            {"target_id": "__run__", "event_type": "run_config", "timestamp": 0.0,
             "detail": {"configuration": {"comm_delivery_probability": 1.0, "comm_delay_ticks": 0}}},
            _detection("t0", 5.0, offset=30.0),
            _assignment("t0", 6.0),
            # The registry keeps an expired task expired even after the work is
            # finished, which is exactly how the real logs report this case.
            _progress("t0", 40.0, 1.0, stage=1, status="expired"),
            _final("t0", 500.0, 1.0, "expired"),
            _detection("t1", 6.0, row=9, col=9, offset=30.0),
            _assignment("t1", 7.0),
            _progress("t1", 8.0, 1.0, stage=1, status="completed"),
            _final("t1", 500.0, 1.0, "completed"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rp_demo_s7.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
            )
            attributions = attribute([path], window=500.0)

        attribution = attributions["demo"]
        self.assertEqual(attribution.seeds, (7,))
        self.assertEqual(len(attribution.expired), 1)
        self.assertEqual(len(attribution.completed), 1)
        self.assertEqual(attribution.counts["finished_after_deadline"], 1)
        self.assertEqual(communication_facts(records)["comm_delay_ticks"], 0)


class ProgressPointTests(unittest.TestCase):
    def test_advancing_ignores_repeats(self) -> None:
        timeline = TargetTimeline(
            target_id="t0", arm="a", seed=1, detected_at=0.0, plan=_plan(),
            progress=[
                ProgressPoint(1.0, 0, 0.0, (), "detected", False),
                ProgressPoint(2.0, 0, 0.5, ("0",), "in_progress", False),
                ProgressPoint(3.0, 0, 0.5, ("0",), "in_progress", False),
            ],
        )
        self.assertEqual([p.timestamp for p in timeline.advancing], [2.0])
        self.assertEqual(timeline.max_progress, 0.5)


if __name__ == "__main__":
    unittest.main()