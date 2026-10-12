from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
R2 = HERE.parent
R61_RAW = R2 / "round61_early_report_value_intervention" / "raw"
MASK_SOURCE = HERE.parents[1] / "comm_budget_action_mask_fix_20260922"
ELIGIBILITY_SOURCE = HERE.parents[1] / "comm_first_report_training_ablation_20261003"
STAGE_B_SOURCE = HERE.parents[1] / "comm_cost_stage_b_20260927"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(R61_RAW))
sys.path.insert(0, str(MASK_SOURCE))
sys.path.insert(0, str(ELIGIBILITY_SOURCE))
sys.path.insert(0, str(STAGE_B_SOURCE))

from r64_execute import RAW, _cell_cursor, _normalized_lifecycle, persist_episode_row, setup_runtime
from r64_guard import R64Guard, R64GuardError
from r64_preflight import saved_focus_features_by_deadline_delta
from r64_runtime import authority_chain, install_counting_hooks, install_deadline_overlay
from r64_intervention import patched_choose_masked_source
from integrated_sar.contracts import (
    DetectionEvent,
    GridPosition,
    TargetProfile,
    TaskStageSpec,
    WorldPosition,
)
from integrated_sar.event_bus import InMemoryEventBus
from integrated_sar.task_registry import TaskRegistry
from r64_runtime import instrument_run

SCRATCH = HERE / ".test_scratch"
SCRATCH.mkdir(exist_ok=True)


class ProfileOverlayTests(unittest.TestCase):
    def test_provider_called_once_and_only_focus_offset_changes(self):
        calls = []
        profile = TargetProfile(
            deadline_offset=97.84999816742,
            stages=(TaskStageSpec((1, 0, 0, 0, 0), 3.0),),
            severity=0.65,
            metadata={"seed": 7},
        )

        class Provider:
            def profile_for(self, target, position, detected_at):
                calls.append((target, detected_at))
                return profile

        for delta in (-30, 0, 30, 60):
            local_calls = []
            provider = Provider()
            provider.profile_for = lambda target, position, detected_at: (
                local_calls.append((target, detected_at)) or profile)
            attempt = SimpleNamespace(delta=delta, profile_records=[])
            install_deadline_overlay(provider, attempt, "piloc-e1-target-5")
            changed = provider.profile_for("piloc-e1-target-5", SimpleNamespace(row=50, col=23), 134.0)
            self.assertEqual(changed.deadline_offset, profile.deadline_offset + delta)
            self.assertEqual(changed.stages, profile.stages)
            self.assertEqual(changed.severity, profile.severity)
            self.assertEqual(local_calls, [("piloc-e1-target-5", 134.0)])
            self.assertEqual(len(attempt.profile_records), 1)
            self.assertTrue(attempt.profile_records[0]["original_rng_generation_call_completed"])
            other = provider.profile_for("piloc-e1-target-4", SimpleNamespace(row=39, col=25), 5.0)
            self.assertIs(other, profile)
            self.assertEqual(other.deadline_offset, profile.deadline_offset)
            self.assertEqual(len(local_calls), 2)
            self.assertEqual(len(attempt.profile_records), 2)

    def test_source_anchor_patches_the_frozen_collector_once_and_compiles(self):
        from r42_one_shot import OneShotSendSkipCollector
        patched = patched_choose_masked_source(OneShotSendSkipCollector)
        self.assertEqual(patched.count("self._r64_enabled"), 1)
        compile(patched, "<r64-test>", "exec")

    def test_saved_focus_send_observation_recomputes_real_feature_function(self):
        result = saved_focus_features_by_deadline_delta()
        self.assertTrue(result["base_vector_exact_match"])
        self.assertEqual([x["delta"] for x in result["variants"]], [-30, 0, 30, 60])
        self.assertEqual(result["variants"][1]["features_19d"], result["saved_features_19d"])
        self.assertTrue(all(len(x["features_19d"]) == 19 for x in result["variants"]))
        self.assertTrue(all(x["changed_indices_from_saved"] == [] for x in result["variants"]))

    def test_log_only_or_double_deadline_consumer_mismatch_is_rejected(self):
        attempt = SimpleNamespace(
            profile_records=[{"target": "piloc-e1-target-5", "detected_at": 134.0,
                              "original_deadline_offset": 97.0, "actual_deadline_offset": 127.0}],
            registry_records=[{"event_id": "d1", "event_deadline": 261.0,
                               "event_offset": 127.0, "registry_deadline": 291.0,
                               "registry_accepted": True}],
            task_env_records=[{"task_env_deadline": 291.0}],
            message_records=[{"target": "piloc-e1-target-5", "deadline": 291.0}],
        )
        event = {"timestamp": 134.0, "detail": {"deadline_offset": 127.0}}
        with self.assertRaises(R64GuardError):
            authority_chain(attempt, event)

    def test_deadline_chain_requires_an_accepted_registry_receipt(self):
        attempt = SimpleNamespace(
            profile_records=[{"target": "piloc-e1-target-5", "detected_at": 134.0,
                              "original_deadline_offset": 100.0, "actual_deadline_offset": 100.0}],
            registry_records=[{"event_id": "d1", "event_deadline": 234.0,
                               "event_offset": 100.0, "registry_deadline": 234.0,
                               "registry_accepted": False}],
            task_env_records=[{"task_env_deadline": 234.0}],
            message_records=[{"target": "piloc-e1-target-5", "deadline": 234.0}],
        )
        with self.assertRaises(R64GuardError):
            authority_chain(attempt, {"timestamp": 134.0,
                                      "detail": {"deadline_offset": 100.0}})

    def test_receipt_observer_uses_live_event_bus_without_rebinding_registry(self):
        class CountingRegistry(TaskRegistry):
            def __init__(self):
                super().__init__()
                self.detection_calls = 0

            def handle_detection(self, event):
                self.detection_calls += 1
                super().handle_detection(event)

        registry = CountingRegistry()
        bus = InMemoryEventBus()
        subscribed_handler = registry.handle_detection
        bus.subscribe(DetectionEvent, subscribed_handler)
        run = SimpleNamespace(
            search_adapter=SimpleNamespace(profile_provider=SimpleNamespace()),
            registry=registry,
            orchestrator=SimpleNamespace(event_bus=bus),
            task_adapter=SimpleNamespace(
                sync_tasks=lambda tasks: None,
                external_to_internal={},
                env=SimpleNamespace(task_dic={}),
                comm_relay=SimpleNamespace(
                    _message_from_event=lambda event: event,
                    _message_from_snapshot=lambda target, event_id, snapshot: snapshot,
                ),
            ),
        )
        attempt = SimpleNamespace(
            registry_records=[], profile_records=[], task_env_records=[], message_records=[],
        )

        instrument_run(run, attempt, overlay_enabled=False)
        self.assertIs(registry.handle_detection.__func__, subscribed_handler.__func__)
        self.assertEqual(len(bus._subscriptions[DetectionEvent]), 2)
        event = DetectionEvent(
            event_id="r64-observer-spy",
            target_id="piloc-e1-target-5",
            grid_position=GridPosition(20, 30),
            world_position=WorldPosition(30 / 59, 20 / 59),
            detected_at=134.0,
            discovered_by="agent_0",
            confidence=1.0,
            profile=TargetProfile(
                stages=(TaskStageSpec((1, 0, 0, 0, 0), 3.0),),
                deadline_offset=100.0,
            ),
        )
        bus.publish(event)

        self.assertEqual(registry.detection_calls, 1)
        self.assertEqual(registry.accepted_event_count, 1)
        self.assertEqual(len(attempt.registry_records), 1)
        self.assertTrue(attempt.registry_records[0]["registry_accepted"])
        self.assertEqual(attempt.registry_records[0]["registry_deadline"], 234.0)
        self.assertEqual(bus.history, (event,))
        self.assertFalse(attempt.registry_observer_identity["registry_handler_reinvoked"])
        attempt.profile_records = [{
            "target": "piloc-e1-target-5", "detected_at": 134.0,
            "original_deadline_offset": 100.0, "actual_deadline_offset": 100.0,
        }]
        attempt.task_env_records = [{"task_env_deadline": 234.0}]
        attempt.message_records = [{"target": "piloc-e1-target-5", "deadline": 234.0}]
        chain = authority_chain(attempt, {
            "timestamp": 134.0, "detail": {"deadline_offset": 100.0},
        })
        self.assertTrue(chain["all_authority_consumers_equal"])


class GuardTests(unittest.TestCase):
    def temp_guard(self):
        tmp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(tmp.cleanup)
        return R64Guard(Path(tmp.name) / "raw")

    def test_returned_row_is_exclusively_persisted_inside_frozen_raw(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            root = Path(tmp) / "raw"
            row_path = root / "rows" / "row.json"
            row = {"artefacts": {"row": str(row_path)}, "steps": 1000}
            self.assertEqual(persist_episode_row(root, row), row_path)
            self.assertEqual(json.loads(row_path.read_text(encoding="utf-8")), row)
            with self.assertRaises(FileExistsError):
                persist_episode_row(root, row)
            outside = Path(tmp) / "outside.json"
            with self.assertRaisesRegex(RuntimeError, "escaped"):
                persist_episode_row(root, {"artefacts": {"row": str(outside)}})

    def test_lifecycle_gate_normalizes_only_explicit_transcript_output_path(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            paths = [Path(tmp) / "a.jsonl", Path(tmp) / "b.jsonl"]
            rows = [
                [{"event_type": "run_config", "detail": {"output": "a"}},
                 {"event_type": "progress", "detail": {"status": "completed"}},
                 {"event_type": "final", "target_id": "__run__", "detail": {"metrics": {
                     "steps": 1000, "comm_transcript": "a.ct"}}}],
                [{"event_type": "run_config", "detail": {"output": "b"}},
                 {"event_type": "progress", "detail": {"status": "completed"}},
                 {"event_type": "final", "target_id": "__run__", "detail": {"metrics": {
                     "steps": 1000, "comm_transcript": "b.ct"}}}],
            ]
            for path, values in zip(paths, rows):
                path.write_text("".join(json.dumps(row) + "\n" for row in values),
                                encoding="utf-8")
            self.assertEqual(_normalized_lifecycle(paths[0]), _normalized_lifecycle(paths[1]))
            rows[1][-1]["detail"]["metrics"]["steps"] = 999
            paths[1].write_text("".join(json.dumps(row) + "\n" for row in rows[1]),
                                encoding="utf-8")
            self.assertNotEqual(_normalized_lifecycle(paths[0]), _normalized_lifecycle(paths[1]))

    def test_ten_normal_then_eleventh_refused_and_restart_does_not_reset(self):
        guard = self.temp_guard()
        for n in range(10):
            with guard.attempt(f"normal-{n}") as attempt:
                attempt.finish()
        restarted = R64Guard(guard.root)
        self.assertEqual(restarted.snapshot()["slots_consumed"], 10)
        with self.assertRaises(R64GuardError):
            with restarted.attempt("normal-11"):
                self.fail("normal cap should refuse before body")

    def test_failure_consumes_slot_and_reserve_requires_interrupted_attempt(self):
        guard = self.temp_guard()
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with guard.attempt("failed"):
                raise RuntimeError("synthetic")
        self.assertEqual(guard.snapshot()["slots_consumed"], 1)
        with self.assertRaises(R64GuardError):
            with guard.attempt("not-failed", category="reserve", retry=True):
                pass
        with guard.attempt("failed", category="reserve", retry=True) as attempt:
            attempt.finish()
        self.assertEqual(guard.snapshot()["slots_consumed"], 2)

    def test_disqualified_result_cannot_be_reclassified_as_reserve(self):
        guard = self.temp_guard()
        with guard.attempt("disqualified") as attempt:
            attempt.qualify(False, "scientific result disqualified")
            attempt.finish()
        with self.assertRaises(R64GuardError):
            with guard.attempt("disqualified", category="reserve", retry=True):
                pass

    def test_ledger_slot_mismatch_fails_closed(self):
        guard = self.temp_guard()
        with guard.attempt("one") as attempt:
            attempt.finish()
        guard.ledger.write_text("", encoding="utf-8")
        with self.assertRaises(R64GuardError):
            guard.snapshot()

    def test_foreign_execution_identity_in_slot_fails_closed(self):
        guard = self.temp_guard()
        with guard.attempt("one") as attempt:
            attempt.finish()
        slot = next(guard.slots.glob("*.slot"))
        payload = json.loads(slot.read_text(encoding="utf-8"))
        payload["execution_version"] = "round61"
        slot.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(R64GuardError, "foreign task or execution version"):
            guard.snapshot()

    def test_frozen_prior_execution_version_remains_in_same_cumulative_ledger(self):
        guard = self.temp_guard()
        with guard.attempt("prior", category="normal") as attempt:
            attempt.finish()
        slot = next(guard.slots.glob("*.slot"))
        payload = json.loads(slot.read_text(encoding="utf-8"))
        payload["execution_version"] = "round64-controlled-deadline-intervention/2"
        slot.write_text(json.dumps(payload), encoding="utf-8")
        rows = guard.rows()
        for row in rows:
            row["execution_version"] = "round64-controlled-deadline-intervention/2"
        guard.ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.assertEqual(guard.snapshot()["slots_consumed"], 1)

    def test_alternate_live_output_root_is_refused_before_configuration(self):
        with self.assertRaisesRegex(RuntimeError, "single frozen raw root"):
            setup_runtime(HERE / "other-root")

    def test_resume_cursor_reuses_qualified_and_requires_explicit_failed_retry(self):
        guard = self.temp_guard()
        cells_dir = Path(guard.root) / "cells"
        cells_dir.mkdir()
        cfg = {"r64_guard": guard, "cells_dir": cells_dir}
        spec = {"logical": "done"}
        with guard.attempt("done") as attempt:
            attempt.qualify(True, "complete")
            attempt.finish()
        (cells_dir / "done.json").write_text(
            json.dumps({"logical": "done", "evidence": 1}), encoding="utf-8")
        with patch("r64_execute.run_cell", side_effect=AssertionError) as run_cell:
            self.assertEqual(_cell_cursor(cfg, spec)["evidence"], 1)
            run_cell.assert_not_called()

        failed_spec = {"logical": "failed"}
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with guard.attempt("failed"):
                raise RuntimeError("synthetic")
        with self.assertRaisesRegex(R64GuardError, "explicit reserve retry"):
            _cell_cursor(cfg, failed_spec)
        with patch("r64_execute.run_cell", return_value={"logical": "failed"}) as retry:
            self.assertEqual(_cell_cursor(cfg, failed_spec, "failed"), {"logical": "failed"})
            retry.assert_called_once_with(cfg, failed_spec, retry=True)

    def test_resume_cursor_fails_closed_on_slot_without_terminal_receipt(self):
        guard = self.temp_guard()
        cells_dir = Path(guard.root) / "cells"
        cells_dir.mkdir()
        cfg = {"r64_guard": guard, "cells_dir": cells_dir}
        guard._claim("unknown", "normal")
        with self.assertRaisesRegex(R64GuardError, "without a terminal receipt"):
            _cell_cursor(cfg, {"logical": "unknown"})

    def test_resume_cursor_fails_closed_on_unclaimed_cell_file(self):
        guard = self.temp_guard()
        cells_dir = Path(guard.root) / "cells"
        cells_dir.mkdir()
        cfg = {"r64_guard": guard, "cells_dir": cells_dir}
        (cells_dir / "stray.json").write_text(json.dumps({"logical": "stray"}), encoding="utf-8")
        with self.assertRaisesRegex(R64GuardError, "without a slot receipt"):
            _cell_cursor(cfg, {"logical": "stray"})


class LiveCounterHookSpyTests(unittest.TestCase):
    class FakeRun:
        def __init__(self):
            profile = TargetProfile(
                stages=(TaskStageSpec((1, 0, 0, 0, 0), 3.0),),
                deadline_offset=100.0,
            )
            self.search_adapter = SimpleNamespace(
                profile_provider=SimpleNamespace(profile_for=lambda *args: profile),
            )
            self.registry = SimpleNamespace(handle_detection=lambda event: None, get=lambda target: None)
            env = SimpleNamespace(task_dic={})
            self.task_adapter = SimpleNamespace(
                sync_tasks=lambda tasks: None, external_to_internal={}, env=env,
                comm_relay=SimpleNamespace(_message_from_event=lambda e: e,
                                           _message_from_snapshot=lambda t, e, s: s),
            )
            self.orchestrator = SimpleNamespace(
                current_time=1000.0,
                event_bus=InMemoryEventBus(),
            )

        def run(self, steps=None):
            return {"steps": 1000}

    def make_runtime(self, *, double_builder=False):
        guard = self.temp_guard()
        calls = {"builder": 0}

        class Config:
            def as_dict(self):
                return {"steps": 1000, "seed": 1}

        def builder(config):
            calls["builder"] += 1
            return self.FakeRun()

        runner = SimpleNamespace(build_joint_run=builder, __name__="r64_fake_runner")

        def episode(*args, **kwargs):
            run = runner.build_joint_run(Config())
            if double_builder:
                runner.build_joint_run(Config())
            return run.run()

        runner._run_episode = episode
        joint_driver = SimpleNamespace(build_joint_run=builder, JointRun=self.FakeRun,
                                       __name__="r64_fake_joint_driver")
        hooks = install_counting_hooks(runner, joint_driver, self.FakeRun, guard)
        return runner, guard, calls, hooks

    def temp_guard(self):
        tmp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(tmp.cleanup)
        return R64Guard(Path(tmp.name) / "raw")

    def test_live_factory_and_run_boundaries_count_exactly_once(self):
        runner, guard, calls, _ = self.make_runtime()
        with guard.attempt("spy") as attempt:
            row = runner._run_episode("tag", "eval", 1, policy="x")
            self.assertEqual(row["steps"], 1000)
            attempt.finish()
        snap = guard.snapshot()
        self.assertEqual(calls, {"builder": 1})
        self.assertEqual((snap["builder_calls"], snap["builder_returns"],
                          snap["run_calls"], snap["run_returns"]), (1, 1, 1, 1))

    def test_second_builder_is_rejected_before_delegate(self):
        runner, guard, calls, _ = self.make_runtime(double_builder=True)
        with self.assertRaises(R64GuardError):
            with guard.attempt("double"):
                runner._run_episode("tag", "eval", 1, policy="x")
        snap = guard.snapshot()
        self.assertEqual(calls["builder"], 1)
        self.assertEqual(snap["builder_returns"], 1)
        self.assertEqual(snap["run_calls"], 0)
        self.assertEqual(snap["interrupted"], 1)

    def test_unregistered_runner_is_refused_before_factory(self):
        runner, guard, calls, _ = self.make_runtime()
        with self.assertRaises(R64GuardError):
            runner._run_episode("unregistered", "eval", 1, policy="x")
        self.assertEqual(calls["builder"], 0)
        self.assertEqual(guard.snapshot()["slots_consumed"], 0)

    def test_r42_counted_proxy_records_run_call_at_joint_run_entry(self):
        guard = self.temp_guard()
        calls = {"factory": 0}

        class TargetRun:
            def __init__(self):
                self.search_adapter = SimpleNamespace(
                    profile_provider=SimpleNamespace(profile_for=lambda *args: None),
                )
                self.registry = SimpleNamespace(handle_detection=lambda event: None,
                                                 get=lambda target: None,
                                                 was_accepted=lambda event_id: False)
                self.task_adapter = SimpleNamespace(
                    sync_tasks=lambda tasks: None,
                    external_to_internal={},
                    env=SimpleNamespace(task_dic={}),
                    comm_relay=SimpleNamespace(
                        _message_from_event=lambda event: event,
                        _message_from_snapshot=lambda target, event_id, snapshot: snapshot,
                    ),
                )
                self.orchestrator = SimpleNamespace(event_bus=InMemoryEventBus())

            def run(self, steps=None):
                events = [row["event"] for row in guard.rows()]
                self_test.assertIn("run_called", events)
                return {"steps": 1000}

        class CountedProxy:
            def __init__(self, target):
                self._target = target

            def __getattr__(self, name):
                return getattr(self._target, name)

            def run(self, steps=None):
                return self._target.run(steps)

        self_test = self

        def original_builder(config):
            calls["factory"] += 1
            return TargetRun()

        def counted_build(config):
            return CountedProxy(original_builder(config))

        runner = SimpleNamespace(__name__="r64_proxy_runner", build_joint_run=counted_build)

        def episode(*args, **kwargs):
            return runner.build_joint_run(SimpleNamespace(as_dict=lambda: {"steps": 1000})).run()

        runner._run_episode = episode
        joint_driver = SimpleNamespace(__name__="r64_proxy_cli",
                                       build_joint_run=original_builder)
        hooks = install_counting_hooks(
            runner, joint_driver, TargetRun, guard, run_proxy_class=CountedProxy,
        )
        self.assertEqual(hooks["builder_binding"],
                         "r42-counted-builder-delegates-to-cli-joint-driver")
        with guard.attempt("counted-proxy") as attempt:
            result = runner._run_episode("tag", "eval", 1)
            self.assertEqual(result["steps"], 1000)
            attempt.finish()
        snapshot = guard.snapshot()
        self.assertEqual(calls["factory"], 1)
        self.assertEqual((snapshot["builder_calls"], snapshot["builder_returns"],
                          snapshot["run_calls"], snapshot["run_returns"]), (1, 1, 1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
