"""Pure event-bus regression for the ROUND34 task-reward tap.

The fixture uses the production event bus, registry, runner tap, and reward bundler, with fake
clock/relay/environment holders. It does not construct a SAR environment or JointRun.
"""

from __future__ import annotations

import sys
import random
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "comm_cost_v2_rerun_20260928"
STAGE_B = ROOT / "runs" / "comm_cost_stage_b_20260927"
sys.path[:0] = [str(RUN), str(STAGE_B)]

# The tap and pure bundler have no torch dependency. Keep their imports available in this
# runtime-only test when PyTorch is absent; no stub method is called by these tests.
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.optim = SimpleNamespace(Optimizer=object, Adam=object)
    torch_stub.manual_seed = lambda *args, **kwargs: None
    torch_stub.__path__ = []
    sys.modules["torch"] = torch_stub

collector_stub = types.ModuleType("sb_collector")
collector_stub.StageBCollector = type("StageBCollector", (), {})
sys.modules.setdefault("sb_collector", collector_stub)

import v2_run
import v2_trainer
from integrated_sar.contracts import (
    DetectionEvent,
    GridPosition,
    ProgressEvent,
    TargetProfile,
    TaskStageSpec,
    TaskStatus,
    WorldPosition,
)
from integrated_sar.event_bus import InMemoryEventBus
from integrated_sar.task_registry import TaskRegistry


# Hand-calculated expected ownership, independent of either implementation under test:
# event tick 1 belongs to action row 1; event tick 2 with no row 2 still belongs to row 1;
# an event before the first action is prefix; a late completion is not rewarded.
MANUAL_CASES = {
    "same_tick": {"event_tick": 1, "decision_steps": [1], "owner": 1},
    "next_action": {"event_tick": 1, "decision_steps": [1, 2], "owner": 1},
    "no_new_action": {"event_tick": 2, "decision_steps": [1], "owner": 1},
    "prefix": {"event_tick": 0, "decision_steps": [1], "owner": "prefix"},
}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0


class FakeRelay:
    def __init__(self) -> None:
        self._tick = 0
        self.counters = {key: 0 for key in v2_run.COST_KEYS}
        self.emitted_messages: list[object] = []

    def tick(self) -> tuple[str, ...]:
        self._tick += 1
        self.counters["airtime_bytes"] += 32
        self.counters["transmission_attempts"] += 1
        return ()


class FakeEnvironment:
    def __init__(self) -> None:
        self.current_time = 0.0

    def step(self, dt: float = 1.0) -> None:
        self.current_time += float(dt)


class TapHarness:
    def __init__(self) -> None:
        self.registry = TaskRegistry()
        self.bus = InMemoryEventBus()
        self.bus.subscribe(DetectionEvent, self.registry.handle_detection)
        self.bus.subscribe(ProgressEvent, self.registry.handle_progress)
        self.clock = FakeClock()
        self.relay = FakeRelay()
        self.env = FakeEnvironment()
        self.run = SimpleNamespace(
            task_adapter=SimpleNamespace(comm_relay=self.relay, env=self.env),
            registry=self.registry,
            orchestrator=SimpleNamespace(event_bus=self.bus, clock=self.clock),
        )
        self.taps = v2_run._install_taps(self.run)

    def detect(self, task_id: str, *, deadline_offset: float = 10.0) -> None:
        self.bus.publish(
            DetectionEvent(
                event_id=f"detect:{task_id}",
                target_id=task_id,
                grid_position=GridPosition(2, 3),
                world_position=WorldPosition(0.5, 0.5),
                detected_at=self.env.current_time,
                discovered_by="fixture-agent",
                confidence=1.0,
                profile=TargetProfile(
                    stages=(TaskStageSpec((1, 0, 0, 0, 0), 1.0),),
                    deadline_offset=deadline_offset,
                ),
            )
        )

    def progress(
        self,
        task_id: str,
        *,
        event_id: str,
        observed_at: float,
        status: TaskStatus = TaskStatus.COMPLETED,
        expired: bool = False,
    ) -> ProgressEvent:
        remaining = (0, 0, 0, 0, 0) if status is TaskStatus.COMPLETED else (1, 0, 0, 0, 0)
        return ProgressEvent(
            event_id=event_id,
            task_id=task_id,
            observed_at=observed_at,
            status=status,
            stage_index=0,
            remaining_requirements=remaining,
            progress=1.0 if status is TaskStatus.COMPLETED else 0.5,
            members=("fixture-agent",),
            expired=expired,
        )

    def tick(self, events: tuple[ProgressEvent, ...] = ()) -> None:
        # This is the audited production order: relay action/tick, raw environment step,
        # synchronous event publication, then GlobalClock advancement.
        self.relay.tick()
        self.env.step(1.0)
        for event in events:
            self.bus.publish(event)
        self.clock.now = self.env.current_time

    def bundle(self, decision_steps: list[int], horizon: int):
        return v2_trainer._bundle(self.taps["task_stream"], decision_steps, horizon)


class RewardTimingRegression(unittest.TestCase):
    def test_same_tick_completion_belongs_to_action_already_sampled(self) -> None:
        h = TapHarness()
        h.detect("same")
        event = h.progress("same", event_id="done:same:1", observed_at=1.0)
        h.tick((event,))

        bundled, prefix = h.bundle(MANUAL_CASES["same_tick"]["decision_steps"], 1)
        self.assertEqual(h.taps["task_stream"], {1: 1.0})
        self.assertEqual(bundled, {1: 1.0})
        self.assertEqual(prefix, 0.0)

    def test_prior_tick_reward_does_not_move_to_next_tick_action(self) -> None:
        h = TapHarness()
        h.detect("next")
        event = h.progress("next", event_id="done:next:1", observed_at=1.0)
        h.tick((event,))
        h.tick()  # decision row 2 is sampled after tick 1 has completed

        bundled, _prefix = h.bundle(MANUAL_CASES["next_action"]["decision_steps"], 2)
        self.assertEqual(h.taps["task_stream"], {1: 1.0})
        self.assertEqual(bundled, {1: 1.0, 2: 0.0})

    def test_no_new_action_next_tick_still_records_event_time_not_scan_time(self) -> None:
        h = TapHarness()
        h.detect("sparse")
        h.tick()
        event = h.progress("sparse", event_id="done:sparse:2", observed_at=2.0)
        h.tick((event,))
        h.tick()  # expose the legacy next-env-step scan; decision_steps still has only row 1

        # There is no decision row at tick 2. _bundle would assign a wrongly delayed tick 3 to
        # the same row 1, so assert the collector's source tick/time as well as the bucket.
        bundled, _prefix = h.bundle(MANUAL_CASES["no_new_action"]["decision_steps"], 3)
        self.assertEqual(bundled, {1: 1.0})  # same bucket even under the one-tick delay
        self.assertEqual(h.taps["task_stream"], {2: 1.0})
        self.assertEqual(bundled, {1: 1.0})
        self.assertEqual(h.taps["task_reward_events"][0]["event_time"], 2.0)
        self.assertEqual(h.taps["task_reward_events"][0]["relay_tick"], 2)
        self.assertEqual(h.taps["task_reward_events"][0]["clock_now"], 1.0)

    def test_completion_on_last_execution_tick_is_captured_without_another_step(self) -> None:
        h = TapHarness()
        h.detect("last")
        event = h.progress("last", event_id="done:last:1", observed_at=1.0)
        h.tick((event,))

        self.assertEqual(h.taps["task_stream"], {1: 1.0})

    def test_multiple_targets_completing_on_one_tick_each_get_one_unit(self) -> None:
        h = TapHarness()
        h.detect("multi-a")
        h.detect("multi-b")
        h.tick((
            h.progress("multi-a", event_id="done:multi-a:1", observed_at=1.0),
            h.progress("multi-b", event_id="done:multi-b:1", observed_at=1.0),
        ))
        h.tick()

        self.assertEqual(h.taps["task_stream"], {1: 2.0})

    def test_republication_of_same_completion_does_not_duplicate_reward(self) -> None:
        h = TapHarness()
        h.detect("repeat")
        event = h.progress("repeat", event_id="done:repeat:1", observed_at=1.0)
        h.tick((event, event))
        h.tick()

        self.assertEqual(h.taps["task_stream"], {1: 1.0})
        self.assertEqual(sum(h.taps["task_stream"].values()), 1.0)
        self.assertEqual(len(h.taps["task_reward_events"]), 2)
        self.assertEqual(
            [row["credited"] for row in h.taps["task_reward_events"]], [True, False]
        )

    def test_partial_stage_is_not_rewarded_but_later_completion_is(self) -> None:
        h = TapHarness()
        h.detect("partial")
        partial = h.progress(
            "partial", event_id="partial:1", observed_at=1.0, status=TaskStatus.PARTIAL
        )
        h.tick((partial,))
        self.assertEqual(h.taps["task_stream"], {})
        complete = h.progress("partial", event_id="done:partial:2", observed_at=2.0)
        h.tick((complete,))
        h.tick()
        self.assertEqual(h.taps["task_stream"], {2: 1.0})

    def test_completion_after_deadline_is_not_rewarded_even_if_expired_flag_is_false(self) -> None:
        h = TapHarness()
        h.detect("late", deadline_offset=1.0)
        h.tick()
        event = h.progress("late", event_id="done:late:2", observed_at=2.0, expired=False)
        h.tick((event,))
        h.tick()

        self.assertEqual(h.taps["task_stream"], {})
        self.assertEqual(h.taps["task_reward_events"][0]["on_time"], False)

    def test_completion_before_first_action_is_kept_as_prefix(self) -> None:
        h = TapHarness()
        h.detect("prefix")
        event = h.progress("prefix", event_id="done:prefix:0", observed_at=0.0)
        h.bus.publish(event)
        h.tick()  # any accidental prefix-to-future credit becomes visible in the old tap

        bundled, prefix = h.bundle(MANUAL_CASES["prefix"]["decision_steps"], 1)
        self.assertEqual(h.taps["task_stream"], {0: 1.0})
        self.assertEqual(bundled, {1: 0.0})
        self.assertEqual(prefix, 1.0)

    def test_observer_is_passive_and_existing_cost_tap_still_counts_tick_delta(self) -> None:
        h = TapHarness()
        h.detect("passive")
        observed_after_collector: list[tuple[bool, dict[int, float], TaskStatus]] = []
        h.bus.subscribe(
            ProgressEvent,
            lambda event: observed_after_collector.append((
                h.registry.was_accepted(event.event_id),
                dict(h.taps["task_stream"]),
                h.registry.get(event.task_id).status,
            )),
        )
        event = h.progress("passive", event_id="done:passive:1", observed_at=1.0)
        rng_before = random.getstate()
        h.tick((event,))

        # Original event order is retained: detection then progress, with no re-publication.
        self.assertEqual(
            [getattr(item, "event_id", None) for item in h.bus.history],
            ["detect:passive", "done:passive:1"],
        )
        self.assertEqual(observed_after_collector, [(True, {1: 1.0}, TaskStatus.COMPLETED)])
        self.assertEqual(h.registry.accepted_event_count, 2)
        self.assertEqual(h.registry.rejected_event_count, 0)
        self.assertEqual(random.getstate(), rng_before)
        self.assertEqual(h.relay.emitted_messages, [])
        self.assertEqual(h.taps["cost_stream"][1]["airtime_bytes"], 32)
        self.assertEqual(h.taps["cost_stream"][1]["transmission_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
