from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np

from integrated_sar.adapters.task03 import Task03Adapter
from integrated_sar.contracts import (
    CoordinatorInput,
    DetectionEvent,
    GridPosition,
    TaskStageSpec,
    TargetProfile,
    WorldPosition,
)
from integrated_sar.coordinator import HeuristicCoordinator
from integrated_sar.task_registry import TaskRegistry


class _BareEnv:
    """Minimal stand-in: no depot_dic, no reset, no simulator state.

    Guards the adapter against assuming the full TaskEnv surface -- the mask
    ordering test builds one of these, and without coverage here the assumption
    would only fail on a machine that has torch installed.
    """

    def get_local_obs(self, agent_id, max_tasks=None):
        del agent_id, max_tasks
        return (
            np.zeros(4, dtype=np.float32),
            np.zeros((1, 4), dtype=np.float32),
            np.zeros((1, 4), dtype=np.float32),
            np.array([False]),
            [("depot", None)],
        )


class BareEnvironmentTests(unittest.TestCase):
    def test_adapter_tolerates_an_environment_without_depots(self) -> None:
        # The mask-ordering test builds this stub, and it is torch-gated, so
        # without coverage here an adapter constructor that assumes env.depot_dic
        # would only fail on a machine that has torch installed.
        adapter = Task03Adapter(env=_BareEnv(), clear_initial_tasks=False)
        self.assertEqual(adapter._depot_positions, {})
        self.assertIsNone(adapter.comm_relay)


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
TASK_ENV_PATH = next(
    (
        candidate
        for candidate in (
            REPOSITORY_ROOT / "task_allocation" / "old_task_env.py",
            REPOSITORY_ROOT / "task-03-修改为CTDE范式" / "old_task_env.py",
        )
        if candidate.exists()
    ),
    REPOSITORY_ROOT / "task_allocation" / "old_task_env.py",
)
@unittest.skipUnless(TASK_ENV_PATH.exists(), "task-03 source is not present")
class Task03AdapterSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        added_torch_stub = False
        if importlib.util.find_spec("torch") is None:
            torch_stub = types.ModuleType("torch")
            torch_stub.cuda = types.SimpleNamespace(empty_cache=lambda: None)
            torch_stub.backends = types.SimpleNamespace(
                cudnn=types.SimpleNamespace(benchmark=False, deterministic=True)
            )
            sys.modules["torch"] = torch_stub
            added_torch_stub = True
        spec = importlib.util.spec_from_file_location("task03_old_task_env", TASK_ENV_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        finally:
            if added_torch_stub:
                sys.modules.pop("torch", None)
        cls.TaskEnv = module.TaskEnv

    def test_external_detection_can_be_allocated_and_stepped(self) -> None:
        env = self.TaskEnv(
            tasks_range=(0, 0),
            species_range=(2, 2),
            per_species_range=(1, 1),
            enable_tidal_tasks=False,
            enable_discovered_tasks=False,
            seed=7,
        )
        adapter = Task03Adapter(env=env, clear_initial_tasks=True)
        registry = TaskRegistry()
        registry.handle_detection(
            DetectionEvent(
                event_id="smoke-detection",
                target_id="piloc-target-smoke",
                grid_position=GridPosition(20, 30),
                world_position=WorldPosition(30 / 59, 20 / 59),
                detected_at=1.0,
                discovered_by="search-0",
                confidence=1.0,
                profile=TargetProfile(
                    stages=(TaskStageSpec((1, 0, 0, 0, 0), 3.0),),
                    deadline_offset=50.0,
                ),
            )
        )
        tasks = registry.active_snapshots()
        adapter.sync_tasks(tasks)
        self.assertEqual(len(env.task_dic), 1)
        self.assertIn("piloc-target-smoke", adapter.external_to_internal)

        agents = adapter.agent_snapshots()
        advice = HeuristicCoordinator().advise(
            CoordinatorInput(1.0, tasks, agents)
        )
        commands = adapter.allocate(tasks, agents, advice, 1.0)
        self.assertTrue(commands)
        adapter.apply_assignments(commands)
        result = adapter.step(1.0, 1.0)
        self.assertTrue(result.progress_events)
        self.assertEqual(result.progress_events[0].task_id, "piloc-target-smoke")
