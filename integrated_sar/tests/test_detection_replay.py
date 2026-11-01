from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from integrated_sar.detection_replay import (
    DetectionStream,
    ReplaySearchModule,
    assert_bundle_matches_configuration,
    assert_streams_equal,
    load_detection_stream,
    scenario_map_sha256,
)
from integrated_sar.errors import ConfigurationError
from integrated_sar.joint_driver import RunConfiguration, run_from_configuration
from integrated_sar.scenarios import build_scenario


def _detections_from_log(path: Path) -> list[tuple[float, str, tuple[int, int]]]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(
        (e["timestamp"], e["target_id"], tuple(e["detail"]["grid_position"]))
        for e in events
        if e.get("event_type") == "detection"
    )


class MapHashTests(unittest.TestCase):
    def test_hash_tracks_the_map_and_the_placement(self) -> None:
        first = build_scenario(seed=3, num_agents=2, num_goals=4)
        same = build_scenario(seed=3, num_agents=2, num_goals=4)
        other = build_scenario(seed=4, num_agents=2, num_goals=4)
        self.assertEqual(scenario_map_sha256(first), scenario_map_sha256(same))
        self.assertNotEqual(scenario_map_sha256(first), scenario_map_sha256(other))


class RecordReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.tmp = Path(self._folder.name)
        self.bundle = self.tmp / "stream.json"

    def tearDown(self) -> None:
        self._folder.cleanup()

    def _config(self, output: str, **overrides) -> RunConfiguration:
        base = dict(
            seed=11,
            steps=40,
            mock=True,
            output=output,
            quiet=True,
            num_goals=4,
        )
        base.update(overrides)
        return RunConfiguration(**base)

    def test_record_then_replay_reproduces_the_detection_stream(self) -> None:
        run_from_configuration(self._config(str(self.tmp / "live.jsonl"), record_stream=str(self.bundle)))
        self.assertTrue(self.bundle.is_file())
        stream = load_detection_stream(self.bundle)
        self.assertEqual(stream.steps_recorded, 40)
        self.assertTrue(stream.events, "the mock search side must discover something in 40 steps")

        run_from_configuration(self._config(str(self.tmp / "replay.jsonl"), replay_stream=str(self.bundle)))
        self.assertEqual(
            _detections_from_log(self.tmp / "live.jsonl"),
            _detections_from_log(self.tmp / "replay.jsonl"),
        )

    def test_two_replays_of_one_bundle_are_identical(self) -> None:
        run_from_configuration(self._config(str(self.tmp / "live.jsonl"), record_stream=str(self.bundle)))
        for name in ("a.jsonl", "b.jsonl"):
            run_from_configuration(self._config(str(self.tmp / name), replay_stream=str(self.bundle)))
        self.assertEqual(_detections_from_log(self.tmp / "a.jsonl"), _detections_from_log(self.tmp / "b.jsonl"))

    def test_a_bundle_built_on_another_seed_is_refused(self) -> None:
        run_from_configuration(self._config(str(self.tmp / "live.jsonl"), record_stream=str(self.bundle)))
        with self.assertRaises(ConfigurationError) as caught:
            run_from_configuration(self._config(str(self.tmp / "x.jsonl"), seed=12, replay_stream=str(self.bundle)))
        self.assertIn("seed", str(caught.exception))

    def test_more_steps_than_recorded_is_refused(self) -> None:
        run_from_configuration(self._config(str(self.tmp / "live.jsonl"), record_stream=str(self.bundle)))
        with self.assertRaises(ConfigurationError) as caught:
            run_from_configuration(self._config(str(self.tmp / "x.jsonl"), steps=80, replay_stream=str(self.bundle)))
        self.assertIn("steps", str(caught.exception).lower())

    def test_recording_and_replaying_at_once_is_refused(self) -> None:
        with self.assertRaises(ConfigurationError):
            self._config("-", record_stream="a.json", replay_stream="b.json").validate()

    def test_replay_with_a_live_piloc_checkpoint_is_refused(self) -> None:
        with self.assertRaises(ConfigurationError):
            self._config(
                "-",
                mock=False,
                replay_stream="whatever.json",
                piloc_checkpoint="some.pth",
            ).validate()

    def test_clock_that_disagrees_with_the_recording_is_fatal(self) -> None:
        run_from_configuration(self._config(str(self.tmp / "live.jsonl"), record_stream=str(self.bundle)))
        stream = load_detection_stream(self.bundle)
        module = ReplaySearchModule(stream)
        with self.assertRaises(ConfigurationError) as caught:
            module.step(3.5, 1.0)
        self.assertIn("clock drift", str(caught.exception))

    def test_releasing_past_the_recording_is_fatal(self) -> None:
        run_from_configuration(self._config(str(self.tmp / "live.jsonl"), record_stream=str(self.bundle)))
        stream = load_detection_stream(self.bundle)
        module = ReplaySearchModule(stream)
        for step in range(stream.steps_recorded):
            module.step(float(step), 1.0)
        with self.assertRaises(ConfigurationError):
            module.step(float(stream.steps_recorded), 1.0)

    def test_detection_events_survive_a_json_round_trip(self) -> None:
        run_from_configuration(self._config(str(self.tmp / "live.jsonl"), record_stream=str(self.bundle)))
        stream = load_detection_stream(self.bundle)
        self.assertTrue(stream.events)
        rebuilt = DetectionStream(stream.as_bundle())
        assert_streams_equal(stream, rebuilt)
        for original, replayed in zip(stream.events, rebuilt.events):
            self.assertEqual(original.event_id, replayed.event_id)
            self.assertEqual(original.profile.stages, replayed.profile.stages)
            self.assertAlmostEqual(original.profile.deadline_offset, replayed.profile.deadline_offset)
            self.assertEqual(original.world_position, replayed.world_position)

    def test_assert_bundle_matches_configuration_checks_radius(self) -> None:
        run_from_configuration(self._config(str(self.tmp / "live.jsonl"), record_stream=str(self.bundle)))
        stream = load_detection_stream(self.bundle)
        configuration = self._config("-", detection_radius=3)
        with self.assertRaises(ConfigurationError) as caught:
            assert_bundle_matches_configuration(stream, configuration, configuration.steps)
        self.assertIn("detection_radius", str(caught.exception))

    def test_a_foreign_payload_is_rejected(self) -> None:
        path = self.tmp / "bad.json"
        path.write_text(json.dumps({"format": "something/else"}), encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            load_detection_stream(path)


class ReplayAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.tmp = Path(self._folder.name)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def _config(self, output: str, **overrides) -> RunConfiguration:
        base = dict(seed=11, steps=40, mock=True, output=output, quiet=True, num_goals=4)
        base.update(overrides)
        return RunConfiguration(**base)

    def test_audit_passes_for_arms_replaying_one_stream(self) -> None:
        from integrated_sar.replay_audit import audit

        bundle = self.tmp / "s11.json"
        run_from_configuration(self._config(str(self.tmp / "rp_nocoord_s11.jsonl"), record_stream=str(bundle)))
        run_from_configuration(
            self._config(str(self.tmp / "rp_rs0.25_s11.jsonl"), replay_stream=str(bundle))
        )
        verified, arms, problems = audit(
            sorted(str(p) for p in self.tmp.glob("rp_*_s*.jsonl")), [str(bundle)]
        )
        self.assertEqual(verified, 1)
        self.assertEqual(arms, 2)
        self.assertEqual(problems, [])

    def test_audit_flags_a_tampered_log(self) -> None:
        from integrated_sar.replay_audit import audit

        bundle = self.tmp / "s11.json"
        run_from_configuration(self._config(str(self.tmp / "rp_nocoord_s11.jsonl"), record_stream=str(bundle)))
        replay = self.tmp / "rp_rs0.25_s11.jsonl"
        run_from_configuration(self._config(str(replay), replay_stream=str(bundle)))
        # Drop the last detection line: the replayed arm no longer matches the stream.
        lines = replay.read_text(encoding="utf-8").splitlines()
        last = max(
            index for index, line in enumerate(lines) if '"event_type": "detection"' in line
        )
        del lines[last]
        replay.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _, _, problems = audit(
            sorted(str(p) for p in self.tmp.glob("rp_*_s*.jsonl")), [str(bundle)]
        )
        self.assertTrue(any("detections differ" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
