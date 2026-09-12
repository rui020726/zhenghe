from __future__ import annotations

import unittest

from integrated_sar.contracts import (
    AgentSnapshot,
    AgentTaskBelief,
    CoordinateTransform,
    CoordinatorInput,
    DetectionEvent,
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


if __name__ == "__main__":
    unittest.main()
