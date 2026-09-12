from __future__ import annotations

import unittest
from dataclasses import fields

from integrated_sar.contracts import (
    AgentSnapshot,
    AgentTaskBelief,
    CoordinateTransform,
    CoordinationAdvice,
    CoordinatorInput,
    DetectionEvent,
    ExecutionStepResult,
    GridPosition,
    ProgressEvent,
    TaskStageSpec,
    TaskStatus,
    TargetProfile,
    WorldPosition,
)
from integrated_sar.coordinator import HeuristicCoordinator
from integrated_sar.demo import build_demo
from integrated_sar.event_bus import InMemoryEventBus
from integrated_sar.task_registry import TaskRegistry


def detection(event_id: str = "detect-1") -> DetectionEvent:
    return DetectionEvent(
        event_id=event_id,
        target_id="target-1",
        grid_position=GridPosition(3, 7),
        world_position=WorldPosition(0.7, 0.3),
        detected_at=1.0,
        discovered_by="search-0",
        confidence=0.8,
        profile=TargetProfile(
            stages=(TaskStageSpec((1, 0, 1, 0, 0), 4.0),),
            deadline_offset=20.0,
            severity=0.7,
        ),
    )


class ContractTests(unittest.TestCase):
    def test_coordinate_transform_does_not_swap_row_and_col(self) -> None:
        transform = CoordinateTransform(rows=11, cols=21)
        world = transform.to_world(GridPosition(row=5, col=20))
        self.assertEqual(world, WorldPosition(x=1.0, y=0.5))
        self.assertEqual(transform.to_grid(world), GridPosition(row=5, col=20))

    def test_execution_result_keeps_its_positional_argument_order(self) -> None:
        # Review finding 6: inserting a field before ``metrics`` silently re-bound
        # the second positional argument, so demo.py handed a metrics mapping to
        # belief_changed_agents and every tick looked like a belief revision.
        self.assertEqual(
            [field.name for field in fields(ExecutionStepResult)],
            ["progress_events", "metrics", "belief_changed_agents"],
        )
        result = ExecutionStepResult((), {"assigned_agents": 2})
        self.assertEqual(result.metrics, {"assigned_agents": 2})
        self.assertEqual(result.belief_changed_agents, ())

    def test_new_contract_fields_do_not_rebind_legacy_positional_calls(self) -> None:
        # Review finding 3: inserting ``confidence``/``revision`` ahead of
        # ``future_requirements`` and ``agent_task_bias`` ahead of ``pair_bias``
        # re-bound arguments that the earlier call sites pass by position.
        self.assertEqual(
            [field.name for field in fields(AgentTaskBelief)][:8],
            [
                "task_id",
                "position",
                "remaining_requirements",
                "deadline",
                "stage_index",
                "progress",
                "severity",
                "future_requirements",
            ],
        )
        self.assertEqual(
            [field.name for field in fields(CoordinationAdvice)][:5],
            ["revision", "generated_at", "valid_until", "task_bias", "pair_bias"],
        )

        belief = AgentTaskBelief(
            "t", WorldPosition(0.5, 0.5), (1, 0, 0, 0, 0), 10.0, 0, 0.0, 0.5,
            (2, 0, 0, 0, 0), ("a",), 1.0, False, True,
        )
        self.assertEqual(belief.future_requirements, (2, 0, 0, 0, 0))
        self.assertEqual(belief.members, ("a",))
        self.assertEqual(belief.updated_at, 1.0)
        self.assertFalse(belief.finished)
        self.assertTrue(belief.dynamic)
        self.assertEqual(belief.confidence, 0.5)
        self.assertEqual(belief.revision, 0)

        advice = CoordinationAdvice(
            1, 0.0, 2.0, {"t": 0.1}, {("a", "t"): 0.2}, frozenset({"b"}), {"policy": "x"},
        )
        self.assertEqual(advice.task_bias, {"t": 0.1})
        self.assertEqual(advice.pair_bias, {("a", "t"): 0.2})
        self.assertEqual(advice.replan_agents, frozenset({"b"}))
        self.assertEqual(advice.metadata, {"policy": "x"})
        self.assertEqual(advice.agent_task_bias, {})
        self.assertAlmostEqual(advice.bias_for("a", "t"), 0.3)

        # Both new fields are keyword-only, so a future argument cannot be mis-bound.
        with self.assertRaises(TypeError):
            AgentTaskBelief(
                "t", WorldPosition(0.5, 0.5), (1,), 10.0, 0, 0.0, 0.5,
                (), (), 0.0, False, False, 0.9,
            )
        with self.assertRaises(TypeError):
            CoordinationAdvice(1, 0.0, 2.0, {}, {}, frozenset(), {}, {})

    def test_advice_bias_sums_the_pool_pair_and_agent_terms(self) -> None:
        advice = CoordinationAdvice(
            revision=1,
            generated_at=0.0,
            valid_until=1.0,
            task_bias={"t": 0.1},
            pair_bias={("a", "t"): 0.2},
            agent_task_bias={("a", "t"): 0.3},
        )
        self.assertAlmostEqual(advice.bias_for("a", "t"), 0.6)
        # A pool-level term reaches every robot; the per-robot terms do not.
        self.assertAlmostEqual(advice.bias_for("b", "t"), 0.1)


class RegistryTests(unittest.TestCase):
    def test_detection_is_idempotent_and_progress_completes_task(self) -> None:
        registry = TaskRegistry()
        event = detection()
        registry.handle_detection(event)
        registry.handle_detection(event)
        self.assertEqual(len(registry), 1)
        self.assertEqual(registry.get("target-1").revision, 1)  # type: ignore[union-attr]

        registry.handle_progress(
            ProgressEvent(
                event_id="progress-1",
                task_id="target-1",
                observed_at=3.0,
                status=TaskStatus.COMPLETED,
                stage_index=0,
                remaining_requirements=(0, 0, 0, 0, 0),
                progress=1.0,
                members=("rescue-0",),
            )
        )
        task = registry.get("target-1")
        self.assertIsNotNone(task)
        self.assertEqual(task.status, TaskStatus.COMPLETED)  # type: ignore[union-attr]
        self.assertEqual(registry.active_snapshots(), ())

    def test_event_bus_fans_out_one_event(self) -> None:
        bus = InMemoryEventBus()
        first, second = [], []
        bus.subscribe(DetectionEvent, first.append)
        bus.subscribe(DetectionEvent, second.append)
        event = detection()
        bus.publish(event)
        self.assertEqual(first, [event])
        self.assertEqual(second, [event])


class CoordinatorTests(unittest.TestCase):
    def test_compatible_near_agent_receives_higher_pair_bias(self) -> None:
        registry = TaskRegistry()
        registry.handle_detection(detection())
        task = registry.get("target-1")
        assert task is not None
        # The coordinator scores from each robot's own belief (invariant 9), so a
        # robot is only biased towards tasks it has actually received.
        belief = AgentTaskBelief(
            task_id="target-1",
            position=WorldPosition(0.7, 0.3),
            remaining_requirements=(1, 0, 1, 0, 0),
            deadline=task.deadline,
            stage_index=0,
            progress=0.0,
            severity=0.7,
        )
        coordinator = HeuristicCoordinator()
        advice = coordinator.advise(
            CoordinatorInput(
                current_time=5.0,
                tasks=(task,),
                agents=(
                    AgentSnapshot(
                        "near", WorldPosition(0.7, 0.3), (1, 0, 1, 0, 0),
                        known_tasks=("target-1",), task_beliefs=(belief,),
                    ),
                    AgentSnapshot(
                        "far", WorldPosition(0.0, 1.0), (0, 1, 0, 0, 0),
                        known_tasks=("target-1",), task_beliefs=(belief,),
                    ),
                ),
            )
        )
        self.assertGreater(
            advice.pair_bias[("near", "target-1")],
            advice.pair_bias[("far", "target-1")],
        )

    def test_a_robot_without_a_belief_gets_no_bias_at_all(self) -> None:
        registry = TaskRegistry()
        registry.handle_detection(detection())
        task = registry.get("target-1")
        assert task is not None
        advice = HeuristicCoordinator().advise(
            CoordinatorInput(
                current_time=5.0,
                tasks=(task,),
                agents=(
                    AgentSnapshot("blind", WorldPosition(0.7, 0.3), (1, 0, 1, 0, 0)),
                ),
            )
        )
        self.assertEqual(advice.pair_bias, {})
        self.assertEqual(advice.task_bias, {})


class EndToEndTests(unittest.TestCase):
    def test_search_to_completion_loop(self) -> None:
        orchestrator = build_demo()
        reports = list(orchestrator.run(4))
        self.assertEqual(len(reports[0].detections), 1)
        self.assertEqual(len(reports[0].assignments), 2)
        task = orchestrator.registry.get("demo-target-1")
        self.assertIsNotNone(task)
        self.assertEqual(task.status, TaskStatus.COMPLETED)  # type: ignore[union-attr]
        self.assertEqual(task.progress, 1.0)  # type: ignore[union-attr]

    def test_the_demo_reports_no_belief_change_it_did_not_have(self) -> None:
        # The metrics mapping used to land in belief_changed_agents, which made the
        # orchestrator treat every tick as a belief revision and coordinate on each
        # one regardless of the configured interval.
        reports = list(build_demo().run(4))
        self.assertEqual([report.advice.revision for report in reports[:1]], [1])
        for report in reports:
            self.assertEqual(report.belief_changed_agents, ())
        # The last two ticks reuse one advice: no belief change, no expiry, and the
        # two-step coordination interval has not come round again.
        self.assertEqual(reports[-1].advice.revision, reports[-2].advice.revision)


if __name__ == "__main__":
    unittest.main()
