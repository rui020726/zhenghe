"""Deterministic tests for the joint PILOC + CTDE inference driver.

Everything here runs on numpy alone unless a test is explicitly gated on torch, so the
whole suite is usable on a CI machine that has no checkpoints and no GPU. Temporary
files are created with ``tempfile`` and never inside the repository.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from typing import Any
from unittest import mock

from integrated_sar.assignment_gate import AssignmentGate, ReplanCoordinator
from integrated_sar.clock import GlobalClock
from integrated_sar.contracts import (
    AssignmentCommand,
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
from integrated_sar.errors import AssetMissingError, ConfigurationError, IntegratedSARError
from integrated_sar.event_bus import InMemoryEventBus
from integrated_sar.joint_driver import (
    RunConfiguration,
    build_joint_run,
    main,
    run_from_configuration,
)
from integrated_sar.lifecycle_log import LIFECYCLE_FIELDS, RUN_TARGET_ID, LifecycleLogger
from integrated_sar.orchestrator import JointSAROrchestrator
from integrated_sar.scenarios import PILOC_OBS_RADIUS
from integrated_sar.task_registry import TaskRegistry

_REAL_FIND_SPEC = importlib.util.find_spec
HAS_TORCH = _REAL_FIND_SPEC("torch") is not None
# timemap needs gym and Pillow; the search policy needs torch and tensorflow on top.
HAS_PILOC_ENV = all(_REAL_FIND_SPEC(name) is not None for name in ("gym", "PIL"))


def make_configuration(directory: str, **overrides: Any) -> RunConfiguration:
    settings: dict[str, Any] = {
        "mock": True,
        "steps": 20,
        "seed": 42,
        "quiet": True,
        "output": os.path.join(directory, "lifecycle.jsonl"),
    }
    settings.update(overrides)
    return RunConfiguration(**settings)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_detection(
    target_id: str = "target-a",
    *,
    event_id: str | None = None,
    row: int = 20,
    col: int = 30,
    detected_at: float = 1.0,
    stages: tuple[TaskStageSpec, ...] | None = None,
) -> DetectionEvent:
    position = GridPosition(row, col)
    return DetectionEvent(
        event_id=event_id or f"detection:{target_id}:{row}:{col}",
        target_id=target_id,
        grid_position=position,
        world_position=WorldPosition(col / 59.0, row / 59.0),
        detected_at=detected_at,
        discovered_by="agent_0",
        confidence=1.0,
        profile=TargetProfile(
            stages=stages or (TaskStageSpec((1, 0, 0, 0, 0), 3.0),),
            deadline_offset=500.0,
            severity=0.8,
        ),
    )


class _RecordingExecutor:
    """Stand-in executor that records exactly which commands reached it."""

    def __init__(self) -> None:
        self.applied: list[AssignmentCommand] = []
        self.steps = 0

    def apply_assignments(self, commands: Any) -> None:
        self.applied.extend(commands)

    def step(self, current_time: float, dt: float) -> ExecutionStepResult:
        del current_time, dt
        self.steps += 1
        return ExecutionStepResult()


def make_command(
    *,
    agent_id: str = "0",
    task_id: str | None = "target-a",
    issued_at: float = 0.0,
    valid_until: float = 10.0,
    task_revision: int | None = 1,
    command_id: str = "assignment:1",
) -> AssignmentCommand:
    return AssignmentCommand(
        command_id=command_id,
        agent_id=agent_id,
        task_id=task_id,
        issued_at=issued_at,
        valid_until=valid_until,
        task_revision=task_revision,
    )


class JointRunTests(unittest.TestCase):
    """End-to-end behaviour of the mock pipeline."""

    def test_mock_mode_runs_a_complete_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(directory)
            summary = run_from_configuration(configuration)
            records = read_jsonl(configuration.output)

        self.assertEqual(summary["steps"], 20)
        self.assertEqual(summary["search_backend"], "mock-frontier")
        self.assertEqual(summary["allocation_backend"], "heuristic")
        self.assertGreater(summary["targets_discovered"], 0)
        self.assertGreater(summary["progress_events"], 0)
        self.assertGreater(summary["assignment_commands"], 0)
        self.assertEqual(summary["rejected_assignments"], 0)
        self.assertEqual(summary["current_time"], 20.0)

        kinds = [record["event_type"] for record in records]
        self.assertEqual(kinds[0], "run_config")
        self.assertIn("detection", kinds)
        self.assertIn("assignment", kinds)
        self.assertIn("progress", kinds)
        self.assertIn("final", kinds)
        # One closing record per discovered target, plus the run-level summary record.
        self.assertEqual(
            kinds.count("final"), summary["targets_discovered"] + 1
        )

    def test_every_record_carries_the_required_lifecycle_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(directory)
            run_from_configuration(configuration)
            records = read_jsonl(configuration.output)

        self.assertTrue(records)
        for record in records:
            absent = [field for field in LIFECYCLE_FIELDS if field not in record]
            self.assertEqual(absent, [], f"{record['event_type']} record lacks {absent}")

        progress = [record for record in records if record["event_type"] == "progress"]
        self.assertTrue(progress)
        for record in progress:
            self.assertIsNotNone(record["stage_index"])
            self.assertIsInstance(record["progress"], float)
            self.assertEqual(len(record["remaining_requirements"]), 5)
            self.assertIsNotNone(record["task_revision"])

    def test_mock_run_is_deterministic_for_a_seed(self) -> None:
        payloads: list[list[dict[str, Any]]] = []
        with tempfile.TemporaryDirectory() as directory:
            for index in (1, 2):
                configuration = make_configuration(
                    directory, output=os.path.join(directory, f"run{index}.jsonl")
                )
                run_from_configuration(configuration)
                records = read_jsonl(configuration.output)
                for record in records:
                    if record["event_type"] == "run_config":
                        record["detail"]["configuration"]["output"] = "<redacted>"
                payloads.append(records)
        self.assertEqual(payloads[0], payloads[1])

    def test_discovered_target_is_injected_into_task_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(directory)
            run = build_joint_run(configuration)
            try:
                run.run()
                adapter = run.task_adapter
                discovered = [
                    snapshot.task_id for snapshot in run.registry.snapshots()
                ]
            finally:
                run.logger.close()

        self.assertTrue(discovered)
        self.assertEqual(len(adapter.external_to_internal), len(discovered))
        for target_id in discovered:
            internal_id = adapter.external_to_internal[target_id]
            self.assertIn(internal_id, adapter.env.task_dic)
            # "discovered" is what makes TaskEnv set the dynamic_tasks observation
            # feature; provenance is carried by external_task_id instead.
            self.assertEqual(adapter.env.task_source[internal_id], "discovered")
            self.assertEqual(
                adapter.env.task_dic[internal_id]["external_task_id"], target_id
            )

    def test_world_coordinates_follow_the_shared_transform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(directory)
            run_from_configuration(configuration)
            records = read_jsonl(configuration.output)

        detections = [record for record in records if record["event_type"] == "detection"]
        self.assertTrue(detections)
        for record in detections:
            row, col = record["detail"]["grid_position"]
            x, y = record["detail"]["world_position"]
            self.assertAlmostEqual(x, col / 59.0, places=9)
            self.assertAlmostEqual(y, row / 59.0, places=9)

    def test_progress_flows_back_into_the_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(directory)
            run = build_joint_run(configuration)
            try:
                run.run()
                snapshots = run.registry.snapshots()
                revisions = {snapshot.task_id: snapshot.revision for snapshot in snapshots}
                progressed = [
                    snapshot for snapshot in snapshots if snapshot.progress > 0.0
                ]
            finally:
                run.logger.close()

        self.assertTrue(progressed, "no target advanced, so no progress reached the registry")
        for snapshot in progressed:
            self.assertGreater(revisions[snapshot.task_id], 1)
            self.assertLessEqual(snapshot.progress, 1.0)
            self.assertEqual(len(snapshot.remaining_requirements), 5)

    def test_missing_task_checkpoint_reports_a_clear_error(self) -> None:
        missing = os.path.join(tempfile.gettempdir(), "no-such-ctde-checkpoint.pth")
        self.assertFalse(os.path.exists(missing))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = main(["--task-checkpoint", missing, "--steps", "3", "--output", "-"])
        message = stderr.getvalue()
        self.assertEqual(exit_code, 2)
        self.assertIn("AssetMissingError", message)
        self.assertIn("no-such-ctde-checkpoint.pth", message)
        self.assertIn("--task-checkpoint", message)

    def test_missing_piloc_checkpoint_reports_a_clear_error(self) -> None:
        missing = os.path.join(tempfile.gettempdir(), "no-such-piloc-checkpoint")
        self.assertFalse(os.path.exists(missing))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = main(["--piloc-checkpoint", missing, "--steps", "3", "--output", "-"])
        message = stderr.getvalue()
        self.assertEqual(exit_code, 2)
        self.assertIn("AssetMissingError", message)
        self.assertIn("no-such-piloc-checkpoint", message)

    def test_missing_map_reports_a_clear_error(self) -> None:
        missing = os.path.join(tempfile.gettempdir(), "no-such-map.png")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = main(
                ["--mock", "--map", missing, "--steps", "3", "--output", "-"]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("AssetMissingError", stderr.getvalue())

    def test_mock_flag_conflicts_with_a_checkpoint(self) -> None:
        with self.assertRaises(ConfigurationError):
            make_configuration(tempfile.gettempdir(), task_checkpoint="x.pth").validate()

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = main(
                ["--mock", "--task-checkpoint", "x.pth", "--steps", "3", "--output", "-"]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("ConfigurationError", stderr.getvalue())

    def test_unsupported_device_is_rejected(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = main(
                ["--mock", "--device", "tpu", "--steps", "3", "--output", "-"]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("unsupported --device", stderr.getvalue())


class DetectionRadiusTests(unittest.TestCase):
    """One radius must decide observation, target removal and detection publication."""

    def test_every_removed_target_yields_exactly_one_stable_id(self) -> None:
        for radius in (1, 5, 10):
            with self.subTest(detection_radius=radius):
                with tempfile.TemporaryDirectory() as directory:
                    configuration = make_configuration(
                        directory, detection_radius=radius, steps=20
                    )
                    run = build_joint_run(configuration)
                    try:
                        run.run()
                        placed = len(run.scenario.goal_positions)
                        remaining = len(run.search_adapter.env.goals_pos)
                        task_ids = [
                            snapshot.task_id for snapshot in run.registry.snapshots()
                        ]
                        detections = run.counters.get("detections", 0)
                    finally:
                        run.logger.close()

                removed = placed - remaining
                self.assertLessEqual(removed, placed)
                # A target the environment removed without an event is lost forever;
                # more events than removed targets means one target is rescued twice.
                self.assertEqual(detections, removed)
                self.assertEqual(len(task_ids), removed)
                self.assertEqual(len(set(task_ids)), len(task_ids))

    def test_trained_search_rejects_a_radius_the_checkpoint_was_not_built_for(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(
                directory,
                mock=False,
                piloc_checkpoint="shared_mappo_model_21_9232.pth",
                detection_radius=PILOC_OBS_RADIUS + 2,
            )
            with self.assertRaises(ConfigurationError) as caught:
                configuration.validate()
            self.assertIn("--detection-radius", str(caught.exception))

            matching = make_configuration(
                directory,
                mock=False,
                piloc_checkpoint="shared_mappo_model_21_9232.pth",
                detection_radius=PILOC_OBS_RADIUS,
            )
            matching.validate()


class RegistryLifecycleTests(unittest.TestCase):
    """Deduplication, revision control and the active-task list."""

    def test_duplicate_detection_event_does_not_create_a_second_task(self) -> None:
        registry = TaskRegistry()
        bus = InMemoryEventBus()
        bus.subscribe(DetectionEvent, registry.handle_detection)
        event = make_detection()

        bus.publish(event)
        bus.publish(event)  # identical re-publication: dropped by event-id dedup
        # Same target, later observation, but a new event id: updates the task in place.
        bus.publish(make_detection(detected_at=2.0, event_id="detection:target-a:20:30:v2"))

        self.assertEqual(len(registry), 1)
        self.assertEqual(registry.accepted_event_count, 2)
        snapshot = registry.get("target-a")
        assert snapshot is not None
        # A re-detection updates the existing task in place instead of duplicating it.
        self.assertEqual(snapshot.revision, 2)
        self.assertEqual(snapshot.source_event_id, event.event_id)
        self.assertEqual(len(registry.active_snapshots()), 1)

    def test_completed_task_leaves_the_active_list(self) -> None:
        registry = TaskRegistry()
        bus = InMemoryEventBus()
        bus.subscribe(DetectionEvent, registry.handle_detection)
        bus.subscribe(ProgressEvent, registry.handle_progress)

        bus.publish(make_detection())
        self.assertEqual(len(registry.active_snapshots()), 1)

        bus.publish(
            ProgressEvent(
                event_id="progress:target-a:done",
                task_id="target-a",
                observed_at=5.0,
                status=TaskStatus.COMPLETED,
                stage_index=0,
                remaining_requirements=(0, 0, 0, 0, 0),
                progress=1.0,
                members=("0", "1"),
            )
        )

        self.assertEqual(registry.active_snapshots(), ())
        self.assertEqual(len(registry.snapshots()), 1)
        finished = registry.get("target-a")
        assert finished is not None
        self.assertEqual(finished.status, TaskStatus.COMPLETED)
        self.assertEqual(finished.progress, 1.0)
        self.assertFalse(finished.active)

        # A completed target must not be resurrected by a late detection.
        bus.publish(make_detection(event_id="detection:late", detected_at=6.0))
        self.assertEqual(registry.active_snapshots(), ())

    def test_completed_task_is_closed_inside_task_env(self) -> None:
        from integrated_sar.adapters.task03 import Task03Adapter
        from integrated_sar.task03_runtime import build_task_env, load_task_env_module

        module = load_task_env_module()
        env = build_task_env(module, seed=7)
        adapter = Task03Adapter(env=env, clear_initial_tasks=True)
        registry = TaskRegistry()
        registry.handle_detection(make_detection())

        adapter.sync_tasks(registry.active_snapshots())
        internal_id = adapter.external_to_internal["target-a"]
        self.assertFalse(env.task_dic[internal_id]["finished"])

        registry.handle_progress(
            ProgressEvent(
                event_id="progress:target-a:done",
                task_id="target-a",
                observed_at=5.0,
                status=TaskStatus.COMPLETED,
                stage_index=0,
                remaining_requirements=(0, 0, 0, 0, 0),
                progress=1.0,
            )
        )
        adapter.sync_tasks(registry.snapshots())
        self.assertTrue(env.task_dic[internal_id]["finished"])
        # The registry, not the environment, decides what is still active.
        self.assertEqual(registry.active_snapshots(), ())


class AssignmentGateTests(unittest.TestCase):
    """Command TTL and task-revision consistency."""

    def test_expired_command_is_not_executed(self) -> None:
        executor = _RecordingExecutor()
        clock = GlobalClock()
        gate = AssignmentGate(executor, clock=clock, revision_of=lambda _: 1)

        clock.advance_to(20.0)
        gate.apply_assignments(
            [make_command(issued_at=0.0, valid_until=10.0)]
        )

        self.assertEqual(executor.applied, [])
        rejections = gate.drain_rejections()
        self.assertEqual(len(rejections), 1)
        self.assertEqual(rejections[0].reason, "expired_command")
        self.assertEqual(gate.pending_replan, {"0"})
        self.assertEqual(executor.steps, 0)

    def test_stale_revision_is_rejected_and_triggers_a_replan(self) -> None:
        executor = _RecordingExecutor()
        clock = GlobalClock()
        gate = AssignmentGate(executor, clock=clock, revision_of=lambda _: 4)
        coordinator = ReplanCoordinator(HeuristicCoordinator(), gate.pending_replan)

        gate.apply_assignments([make_command(task_revision=2)])
        self.assertEqual(executor.applied, [])
        rejections = gate.drain_rejections()
        self.assertEqual(rejections[0].reason, "stale_task_revision")
        self.assertEqual(rejections[0].current_revision, 4)

        advice = coordinator.advise(CoordinatorInput(1.0, (), ()))
        self.assertIn("0", advice.replan_agents)
        # The pending set is consumed, so the next advice is unaffected.
        self.assertEqual(gate.pending_replan, set())
        self.assertEqual(coordinator.advise(CoordinatorInput(2.0, (), ())).replan_agents, frozenset())

    def test_current_revision_is_executed(self) -> None:
        executor = _RecordingExecutor()
        clock = GlobalClock()
        gate = AssignmentGate(executor, clock=clock, revision_of=lambda _: 3)

        clock.advance_to(1.0)
        command = make_command(issued_at=1.0, valid_until=11.0, task_revision=3)
        gate.apply_assignments([command])

        self.assertEqual(executor.applied, [command])
        self.assertEqual(gate.drain_rejections(), ())
        self.assertEqual(gate.pending_replan, set())

    def test_command_without_a_task_is_never_stale(self) -> None:
        executor = _RecordingExecutor()
        gate = AssignmentGate(executor, clock=GlobalClock(), revision_of=lambda _: None)
        command = make_command(task_id=None, task_revision=None, valid_until=10.0)
        gate.apply_assignments([command])
        self.assertEqual(executor.applied, [command])

    def test_gate_forwards_step_to_the_wrapped_executor(self) -> None:
        executor = _RecordingExecutor()
        gate = AssignmentGate(executor, clock=GlobalClock(), revision_of=lambda _: None)
        gate.step(0.0, 1.0)
        self.assertEqual(executor.steps, 1)


class ReplanSchedulingTests(unittest.TestCase):
    """A dropped command causes exactly one replan, on the next tick."""

    def test_rejection_replans_once_on_the_next_tick(self) -> None:
        executor = _RecordingExecutor()
        clock = GlobalClock()
        registry_revision = [1]
        gate = AssignmentGate(
            executor, clock=clock, revision_of=lambda _: registry_revision[0]
        )
        orchestrator = JointSAROrchestrator(
            search=_IdleSearch(),
            task_sink=_NullTaskSink(),
            agent_source=_NullAgentSource(),
            coordinator=ReplanCoordinator(HeuristicCoordinator(), gate.pending_replan),
            allocator=_FixedAllocator(task_revision=2),
            executor=gate,
            clock=clock,
            replan_source=gate,
            coordination_interval=5,
            allocation_interval=5,
        )

        counts: dict[int, int] = {}
        for report in orchestrator.run(8):
            counts[report.step] = len(report.assignments)
            if report.step == 1:
                # The registry catches up, so the next command matches and is accepted.
                registry_revision[0] = 2

        rejections = gate.drain_rejections()
        self.assertEqual([item.reason for item in rejections], ["stale_task_revision"])
        # assignment:1 was dropped by the gate; only the replan and the periodic
        # interval ever reach the executor.
        self.assertEqual(
            [command.command_id for command in executor.applied],
            ["assignment:2", "assignment:3"],
        )

        self.assertEqual(counts[1], 1, "first decision is issued, then dropped by the gate")
        self.assertEqual(counts[2], 1, "the replan must not wait for the coordination interval")
        self.assertEqual([counts[3], counts[4], counts[5]], [0, 0, 0])
        self.assertEqual(counts[6], 1, "the periodic allocation interval still fires")
        self.assertEqual([counts[7], counts[8]], [0, 0], "a consumed replan must not repeat")


class OrchestratorWiringTests(unittest.TestCase):
    """Regression coverage for collaborator injection."""

    def test_injected_registry_and_clock_are_used_not_replaced(self) -> None:
        # An empty TaskRegistry is falsy via __len__ and a freshly started GlobalClock is
        # falsy via __float__, so `x or Default()` used to discard both silently.
        registry = TaskRegistry()
        clock = GlobalClock()
        bus = InMemoryEventBus()
        orchestrator = JointSAROrchestrator(
            search=_IdleSearch(),
            task_sink=_NullTaskSink(),
            agent_source=_NullAgentSource(),
            coordinator=HeuristicCoordinator(),
            allocator=_NullAllocator(),
            executor=_RecordingExecutor(),
            registry=registry,
            event_bus=bus,
            clock=clock,
        )

        self.assertIs(orchestrator.registry, registry)
        self.assertIs(orchestrator.clock, clock)
        self.assertIs(orchestrator.event_bus, bus)

        bus.publish(make_detection())
        self.assertEqual(len(registry), 1, "the injected registry never saw the detection")
        orchestrator.step()
        self.assertEqual(clock.now, orchestrator.current_time)
        self.assertEqual(clock.now, 1.0)


class CoordinatorResidualTests(unittest.TestCase):
    """Invariant 5 and 6: the coordinator biases logits but never overrides the mask."""

    def test_residual_cannot_make_a_masked_action_selectable(self) -> None:
        from integrated_sar.adapters.task03 import Task03Adapter
        from integrated_sar.task03_runtime import build_task_env, load_task_env_module

        module = load_task_env_module()
        env = build_task_env(module, seed=7)
        adapter = Task03Adapter(env=env, clear_initial_tasks=True, residual_scale=1.0)

        # Drop the target on robot 0's own cell so it is inside comm_range: knowledge
        # now arrives through the environment's sensing and gossip, not a broadcast.
        origin = env.agent_dic[0]["location"]
        registry = TaskRegistry()
        registry.handle_detection(
            make_detection(row=int(round(float(origin[1]) * 59)), col=int(round(float(origin[0]) * 59)))
        )
        tasks = registry.active_snapshots()
        adapter.sync_tasks(tasks)
        env.update_communication_graph()
        env.propagate_task_completion()
        agents = adapter.agent_snapshots()

        agent_id = agents[0].agent_id
        _, _, _, mask, action_map = env.get_local_obs(int(agent_id))
        task_indices = [
            index
            for index, (kind, task_id) in enumerate(action_map)
            if kind == "task" and task_id is not None and not bool(mask[index])
        ]
        self.assertTrue(task_indices, "the scenario produced no selectable task action")

        # An extreme residual in favour of one legal task must still respect the mask.
        chosen_external = adapter.internal_to_external[int(action_map[task_indices[0]][1])]
        advice = CoordinationAdvice(
            revision=1,
            generated_at=0.0,
            valid_until=10.0,
            task_bias={chosen_external: 1.0},
            pair_bias={(agent_id, chosen_external): 1.0},
        )
        commands = adapter.allocate(tasks, agents, advice, 0.0)
        selected = {command.agent_id: command.task_id for command in commands}
        self.assertEqual(selected[agent_id], chosen_external)

        action_index_of = {
            int(task_id): index
            for index, (kind, task_id) in enumerate(action_map)
            if kind == "task" and task_id is not None
        }
        masked_indices = {index for index in range(len(mask)) if bool(mask[index])}
        for command in commands:
            if command.task_id is None:
                continue
            action_index = action_index_of.get(adapter.external_to_internal[command.task_id])
            self.assertIsNotNone(action_index, "an assigned task is absent from the action map")
            self.assertNotIn(action_index, masked_indices)


class UnifiedClockTests(unittest.TestCase):
    """One time base for the orchestrator, the search side and the rescue side."""

    def test_a_fractional_dt_keeps_every_clock_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(directory, dt=0.25, steps=8)
            run = build_joint_run(configuration)
            try:
                run.run()
                global_time = run.orchestrator.current_time
                search_time = float(run.search_adapter.env.current_time)
                task_time = float(run.task_adapter.env.current_time)
            finally:
                run.logger.close()

        self.assertAlmostEqual(global_time, 2.0, places=9)
        self.assertAlmostEqual(search_time, global_time, places=9)
        self.assertAlmostEqual(task_time, global_time, places=9)

    def test_trained_search_rejects_a_dt_the_environment_cannot_honour(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(
                directory,
                mock=False,
                piloc_checkpoint="shared_mappo_model_21_9232.pth",
                dt=0.25,
            )
            with self.assertRaises(ConfigurationError) as caught:
                configuration.validate()
        self.assertIn("--dt", str(caught.exception))


class BeliefBoundaryTests(unittest.TestCase):
    """Invariant 9: nothing may decide on a target it has not heard about."""

    @staticmethod
    def _wire(species: int, depot_positions: list[tuple[float, float]]) -> Any:
        from integrated_sar.adapters.task03 import Task03Adapter
        from integrated_sar.task03_runtime import build_task_env, load_task_env_module

        env = build_task_env(
            load_task_env_module(),
            seed=7,
            species=species,
            per_species=2,
            depot_positions=depot_positions,
        )
        return env, Task03Adapter(env=env, clear_initial_tasks=True)

    def test_only_robots_in_comm_range_learn_a_target(self) -> None:
        # Two staging points 1.13 apart, so they are out of comm_range of each other
        # and gossip cannot bridge them; the target sits next to the second one only.
        env, adapter = self._wire(3, [(0.1, 0.1), (0.9, 0.9)])
        registry = TaskRegistry()
        registry.handle_detection(make_detection(row=59, col=59))
        tasks = registry.active_snapshots()
        adapter.sync_tasks(tasks)
        env.update_communication_graph()
        env.propagate_task_completion()

        snapshots = adapter.agent_snapshots()
        knowing = sorted(item.agent_id for item in snapshots if item.known_tasks)
        self.assertEqual(len(knowing), 2, "expected one species of two robots to be in range")
        self.assertEqual(len(snapshots) - len(knowing), 4)
        for snapshot in snapshots:
            if snapshot.agent_id in knowing:
                self.assertEqual(snapshot.known_tasks, ("target-a",))

        advice = HeuristicCoordinator().advise(
            CoordinatorInput(0.0, tasks, snapshots)
        )
        self.assertEqual(sorted(agent for agent, _ in advice.pair_bias), knowing)

        commands = adapter.allocate(tasks, snapshots, advice, 0.0)
        for command in commands:
            if command.agent_id not in knowing:
                self.assertIsNone(
                    command.task_id,
                    f"{command.agent_id} was assigned a target it never heard about",
                )

    def test_a_target_outside_every_comm_range_reaches_nobody(self) -> None:
        env, adapter = self._wire(2, [(0.1, 0.1)])
        registry = TaskRegistry()
        registry.handle_detection(make_detection(row=59, col=59))
        tasks = registry.active_snapshots()
        adapter.sync_tasks(tasks)
        env.update_communication_graph()
        env.propagate_task_completion()

        snapshots = adapter.agent_snapshots()
        self.assertTrue(snapshots)
        for snapshot in snapshots:
            self.assertEqual(snapshot.known_tasks, ())

        advice = HeuristicCoordinator().advise(
            CoordinatorInput(0.0, tasks, snapshots)
        )
        self.assertEqual(advice.pair_bias, {})
        # The target still exists in the registry; it is simply unheard of downstream.
        self.assertIn("target-a", advice.task_bias)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class HardMaskOrderingTests(unittest.TestCase):
    """Invariant 6 must hold for an unbounded residual, not just a small one."""

    def test_an_unbounded_residual_cannot_select_a_masked_task(self) -> None:
        import numpy as np
        import torch
        from types import SimpleNamespace

        from integrated_sar.adapters.task03 import MASKED_LOGIT, Task03Adapter

        action_map = [("depot", None), ("task", 7), ("task", 8)]
        mask = np.array([False, True, False])

        class _StubEnv:
            def get_local_obs(self, agent_id: int, max_tasks: int | None = None) -> Any:
                del agent_id, max_tasks
                return (
                    np.zeros(4, dtype=np.float32),
                    np.zeros((2, 4), dtype=np.float32),
                    np.zeros((len(action_map), 4), dtype=np.float32),
                    mask,
                    action_map,
                )

        class _MaskingNetwork:
            """Mirrors old_network: the hard mask is applied inside get_action."""

            def parameters(self) -> Any:
                return iter([torch.zeros(1)])

            def get_action(self, state: Any, neighbors: Any, tasks: Any, hard_mask: Any) -> Any:
                del state, neighbors
                logits = torch.zeros(tasks.shape[0], tasks.shape[1])
                return SimpleNamespace(logits=logits.masked_fill(hard_mask.bool(), MASKED_LOGIT))

        adapter = Task03Adapter(
            env=_StubEnv(),
            network=_MaskingNetwork(),
            clear_initial_tasks=False,
            residual_scale=1.0,
        )
        adapter.internal_to_external = {7: "target-masked", 8: "target-legal"}
        advice = CoordinationAdvice(
            revision=1,
            generated_at=0.0,
            valid_until=1.0,
            task_bias={"target-masked": 1e12},
            pair_bias={("0", "target-masked"): 1e12},
        )

        chosen = adapter._network_action(0, action_map, advice)
        self.assertNotEqual(chosen, 1, "a masked task became selectable again")
        self.assertFalse(bool(mask[chosen]))


class _IdleSearch:
    def step(self, current_time: float, dt: float) -> Any:
        del current_time, dt
        from integrated_sar.contracts import SearchStepResult

        return SearchStepResult()


class _NullTaskSink:
    def sync_tasks(self, tasks: Any) -> None:
        del tasks


class _NullAgentSource:
    def agent_snapshots(self) -> tuple[Any, ...]:
        return ()


class _NullAllocator:
    def allocate(self, tasks: Any, agents: Any, advice: Any, current_time: float) -> tuple[Any, ...]:
        del tasks, agents, advice, current_time
        return ()


class _FixedAllocator:
    """Always offers the same task at a fixed revision, numbering each command."""

    def __init__(self, *, task_revision: int) -> None:
        self.task_revision = task_revision
        self.calls = 0

    def allocate(self, tasks: Any, agents: Any, advice: Any, current_time: float) -> tuple[Any, ...]:
        del tasks, agents, advice
        self.calls += 1
        return (
            make_command(
                issued_at=current_time,
                valid_until=current_time + 10.0,
                task_revision=self.task_revision,
                command_id=f"assignment:{self.calls}",
            ),
        )


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class CtdeCheckpointTests(unittest.TestCase):
    """Checkpoint loading is strict: a wrong model must never be used silently."""

    @classmethod
    def setUpClass(cls) -> None:
        import torch

        from integrated_sar.task03_runtime import _load_network_module

        cls.torch = torch
        cls.network_module = _load_network_module(None)

    def _save(self, directory: str, name: str, network: Any, episode: int = 1) -> str:
        path = os.path.join(directory, name)
        self.torch.save({"episode": episode, "model_state_dict": network.state_dict()}, path)
        return path

    def test_matching_checkpoint_loads_and_runs_a_joint_episode(self) -> None:
        from integrated_sar.task03_runtime import load_ctde_network

        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, "ctde_best.pth", self.network_module.CTDE_Network())
            network = load_ctde_network(path)
            self.assertEqual(network.checkpoint_episode, 1)
            self.assertEqual(os.path.abspath(network.checkpoint_path), os.path.abspath(path))

            configuration = make_configuration(
                directory,
                mock=False,
                task_checkpoint=path,
                output=os.path.join(directory, "ctde.jsonl"),
            )
            summary = run_from_configuration(configuration)

        self.assertEqual(summary["allocation_backend"], "ctde-network")
        self.assertGreater(summary["targets_discovered"], 0)
        self.assertGreater(summary["progress_events"], 0)

    def test_checkpoint_directory_resolution_prefers_named_files(self) -> None:
        from integrated_sar.task03_runtime import load_ctde_network, resolve_task_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            network = self.network_module.CTDE_Network()
            self._save(directory, "ctde_ep_5.pth", network, episode=5)
            self._save(directory, "ctde_ep_41.pth", network, episode=41)
            self.assertEqual(
                os.path.basename(resolve_task_checkpoint(directory)), "ctde_ep_41.pth"
            )
            self._save(directory, "ctde_best.pth", network, episode=9)
            self.assertEqual(
                os.path.basename(resolve_task_checkpoint(directory)), "ctde_best.pth"
            )
            self.assertIsNotNone(load_ctde_network(directory))

    def test_mismatched_architecture_is_rejected_not_partially_loaded(self) -> None:
        from integrated_sar.errors import CheckpointIncompatibleError
        from integrated_sar.task03_runtime import load_ctde_network

        with tempfile.TemporaryDirectory() as directory:
            path = self._save(
                directory,
                "ctde_wrong.pth",
                self.network_module.CTDE_Network(hidden=64),
            )
            with self.assertRaises(CheckpointIncompatibleError):
                load_ctde_network(path)

    def test_padding_only_neighbours_are_rejected_in_summary_mode(self) -> None:
        # Documents the known old_network.py issue from AGENTS.md: with every neighbour
        # slot padded, attention produces NaN. The mainline 'tokens' mode is unaffected
        # because the aggregated neighbour vector is unused there, but the load-time
        # probe must refuse a 'summary' checkpoint rather than decide with NaN logits.
        from integrated_sar.errors import CheckpointIncompatibleError
        from integrated_sar.task03_runtime import load_ctde_network

        with tempfile.TemporaryDirectory() as directory:
            path = self._save(
                directory,
                "ctde_summary.pth",
                self.network_module.CTDE_Network(actor_kv_mode="summary"),
            )
            with self.assertRaises(CheckpointIncompatibleError):
                load_ctde_network(path, actor_kv_mode="summary")

            tokens_path = self._save(
                directory, "ctde_tokens.pth", self.network_module.CTDE_Network()
            )
            self.assertIsNotNone(load_ctde_network(tokens_path))


class CheckpointResolutionTests(unittest.TestCase):
    """Path resolution needs no torch, so it is checked on every machine."""

    def test_directory_without_any_checkpoint_is_reported(self) -> None:
        from integrated_sar.task03_runtime import resolve_task_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "notes.txt"), "w", encoding="utf-8") as handle:
                handle.write("not a checkpoint")
            with self.assertRaises(AssetMissingError) as caught:
                resolve_task_checkpoint(directory)
        self.assertIn("--task-checkpoint", str(caught.exception))

    def test_missing_checkpoint_path_is_reported(self) -> None:
        from integrated_sar.task03_runtime import resolve_task_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            missing = os.path.join(directory, "ctde_final.pth")
            with self.assertRaises(AssetMissingError) as caught:
                resolve_task_checkpoint(missing)
        self.assertIn(missing, str(caught.exception))


@unittest.skipUnless(HAS_PILOC_ENV, "the PILOC environment needs gym and Pillow")
class PilocScenarioLoadingTests(unittest.TestCase):
    """``load`` must not leave state behind from the constructor's random starts."""

    def test_no_pheromone_or_history_survives_from_a_random_start(self) -> None:
        import numpy as np

        from integrated_sar.piloc_runtime import build_piloc_environment
        from integrated_sar.scenarios import build_scenario

        for seed in (11, 23):
            with self.subTest(seed=seed):
                scenario = build_scenario(seed=seed, num_agents=2, num_goals=3)
                environment = build_piloc_environment(scenario, seed=seed)

                expected = {
                    (int(position.row), int(position.col))
                    for position in scenario.agent_positions
                }
                marked = {
                    (int(row), int(col))
                    for row, col in zip(*np.nonzero(environment.global_pheromone))
                }
                self.assertEqual(marked, expected)

                for index, position in enumerate(scenario.agent_positions):
                    row, col = int(position.row), int(position.col)
                    agent_id = f"agent_{index}"
                    history = environment.agent_positions_history[agent_id]
                    self.assertEqual((int(history[0]), int(history[1])), (row, col))

                # timemap marks each start cell and measures that agent's window in
                # the same pass, so agent i only ever sees the marks of agents 0..i.
                staged = np.zeros(environment.map.shape, dtype=np.float32)
                for index, position in enumerate(scenario.agent_positions):
                    row, col = int(position.row), int(position.col)
                    staged[row, col] = 1.0
                    window = staged[
                        max(0, row - PILOC_OBS_RADIUS) : row + PILOC_OBS_RADIUS + 1,
                        max(0, col - PILOC_OBS_RADIUS) : col + PILOC_OBS_RADIUS + 1,
                    ]
                    self.assertEqual(
                        float(environment.local_pheromone[f"agent_{index}"]),
                        float(window.sum()),
                    )


class LifecycleCorrelationTests(unittest.TestCase):
    """The log must stay idempotent and let one target be traced by a single key."""

    def test_a_republished_event_is_logged_once(self) -> None:
        stream = io.StringIO()
        registry = TaskRegistry()
        bus = InMemoryEventBus()
        bus.subscribe(DetectionEvent, registry.handle_detection)
        logger = LifecycleLogger(stream, snapshot_of=registry.get)
        bus.subscribe(DetectionEvent, logger.handle_detection)

        event = make_detection()
        for _ in range(3):
            bus.publish(event)

        records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
        self.assertEqual(registry.accepted_event_count, 1)
        self.assertEqual(logger.records_written, 1)
        self.assertEqual(len(records), 1, "the registry is idempotent, so the log must be too")

    def test_every_record_for_a_target_shares_its_detection_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(directory)
            run_from_configuration(configuration)
            records = read_jsonl(configuration.output)

        sources: dict[str, set[Any]] = {}
        for record in records:
            if record["target_id"] == RUN_TARGET_ID:
                continue
            sources.setdefault(record["target_id"], set()).add(record["source_event_id"])

        self.assertTrue(sources)
        for target_id, ids in sources.items():
            self.assertEqual(len(ids), 1, f"{target_id} was logged under source ids {ids}")
            self.assertIsNotNone(next(iter(ids)))


class MissingDependencyTests(unittest.TestCase):
    """A missing dependency becomes an actionable message, never a bare ImportError."""

    @staticmethod
    def _torch_absent(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            return None
        return _REAL_FIND_SPEC(name, *args, **kwargs)

    def test_task_checkpoint_without_torch_is_reported(self) -> None:
        from integrated_sar.errors import DependencyMissingError
        from integrated_sar.task03_runtime import load_task_env_module

        with mock.patch.object(importlib.util, "find_spec", self._torch_absent):
            with self.assertRaises(DependencyMissingError) as caught:
                load_task_env_module(require_torch=True)
        self.assertIn("torch", str(caught.exception))

    def test_mock_pipeline_still_imports_without_torch(self) -> None:
        from integrated_sar.task03_runtime import load_task_env_module

        # The loader installs a stub for the three module-level torch statements in
        # old_task_env.py and pops it again, so put the real module back afterwards.
        installed = sys.modules.get("torch")
        try:
            with mock.patch.object(importlib.util, "find_spec", self._torch_absent):
                module = load_task_env_module()
        finally:
            if installed is not None:
                sys.modules["torch"] = installed
        self.assertTrue(hasattr(module, "TaskEnv"))


if __name__ == "__main__":
    unittest.main()
