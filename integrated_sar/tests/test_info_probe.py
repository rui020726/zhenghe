"""B0's four claims, each tested as a claim and not as a description.

The probe exists so that a deployed-path experiment can answer "did this robot ever get
this goal, when, and was it out of date when it chose". That is only worth believing if
turning the probe on cannot change the experiment, so the load-bearing tests here are:
the lifecycle records are identical with the probe on, and the probe consumes no random
number. Path fields are compared separately on purpose — a probe path inside a
configuration record is an identity difference, not a behaviour difference, and the
tenth round of this project learned that conflating the two either fakes a failure or
hides a real one.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from integrated_sar.info_probe import (
    IDENTITY_FIELDS,
    InfoProbe,
    behaviour_view,
)
from integrated_sar.joint_driver import RunConfiguration, build_joint_run

#: Configuration fields that name where artefacts go. They cannot change behaviour.
IDENTITY_KEYS = IDENTITY_FIELDS


def probe_rows(raw: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def lifecycle_lines(directory: str, name: str = "lifecycle.jsonl") -> list[str]:
    return Path(os.path.join(directory, name)).read_text(encoding="utf-8").splitlines()


def behaviour_lines(lines: list[str]) -> list[dict[str, Any]]:
    """Every lifecycle record reduced to what it did, not where it was written."""
    return [behaviour_view(json.loads(line)) for line in lines]


def configuration_body(lines: list[str]) -> dict[str, Any]:
    for line in lines:
        record = json.loads(line)
        if record.get("event_type") == "run_config":
            body = dict(record["detail"]["configuration"])
            return {key: value for key, value in body.items() if key not in IDENTITY_KEYS}
    raise AssertionError("the lifecycle log carried no run_config record")


def run_episode(with_probe: bool = False, **overrides: Any
                ) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """One mock episode in a throwaway directory: summary, lifecycle lines, probe rows."""
    with tempfile.TemporaryDirectory() as directory:
        settings: dict[str, Any] = {
            "mock": True, "steps": 8, "seed": 42, "quiet": True,
            "output": os.path.join(directory, "lifecycle.jsonl"),
        }
        if with_probe:
            settings["info_probe"] = os.path.join(directory, "probe.jsonl")
        settings.update(overrides)
        configuration = RunConfiguration(**settings)
        run = build_joint_run(configuration)
        summary = run.run()
        run.logger.close()
        lines = lifecycle_lines(directory)
        rows: list[dict[str, Any]] = []
        if configuration.info_probe:
            rows = probe_rows(Path(configuration.info_probe).read_text(encoding="utf-8"))
        summary = {key: value for key, value in summary.items() if key not in IDENTITY_KEYS}
        return summary, lines, rows


class BehaviourCannotChange(unittest.TestCase):
    def test_lifecycle_records_are_identical_with_the_probe_on(self) -> None:
        quiet_summary, quiet_lines, quiet_rows = run_episode()
        loud_summary, loud_lines, loud_rows = run_episode(with_probe=True)
        self.assertEqual(quiet_rows, [], "a probe-less run must write nothing")
        self.assertTrue(loud_rows, "a run with the probe on wrote no records")
        self.assertEqual(behaviour_lines(quiet_lines), behaviour_lines(loud_lines),
                         "the probe changed a lifecycle record: it is not a passive observer")
        self.assertEqual(configuration_body(quiet_lines), configuration_body(loud_lines),
                         "only identity fields may differ, and they are compared apart")
        self.assertEqual(quiet_summary, loud_summary)

    def test_the_only_configuration_difference_is_the_probe_path_itself(self) -> None:
        _, quiet_lines, _ = run_episode()
        _, loud_lines, _ = run_episode(with_probe=True)
        quiet = json.loads(quiet_lines[0])["detail"]["configuration"]
        loud = json.loads(loud_lines[0])["detail"]["configuration"]
        self.assertIsNone(quiet["info_probe"])
        self.assertIsNotNone(loud["info_probe"])
        # Everything that is not an artefact path has to match, including the output path,
        # which the helper keeps identical by construction across the two runs.
        strip = {key: {k: v for k, v in body.items() if k not in IDENTITY_KEYS}
                 for key, body in (("quiet", quiet), ("loud", loud))}
        self.assertEqual(strip["quiet"], strip["loud"])

    def test_the_run_still_finishes_and_the_probe_file_lands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary, _, rows = run_episode(
                info_probe=os.path.join(directory, "probe.jsonl"), num_goals=6)
            self.assertEqual(summary["steps"], 8)
            self.assertGreater(summary["current_time"], 0.0)
            self.assertEqual(rows[-1]["kind"], "probe_summary")

    def test_the_dropped_block_really_is_the_transport_self_description(self) -> None:
        """behaviour_view may only discard a block that restates configuration.

        The gate drops the run summary's ``communication`` block, so this pins down that the
        block is where the relay writes its own knobs — otherwise dropping it would be
        sweeping real observations under a name.
        """
        with tempfile.TemporaryDirectory() as directory:
            configuration = RunConfiguration(
                mock=True, steps=4, seed=42, quiet=True,
                output=os.path.join(directory, "l.jsonl"),
                info_probe=os.path.join(directory, "p.jsonl"))
            run = build_joint_run(configuration)
            run.run()
            declared = set(run.task_adapter.comm_relay.transport_parameters())
            records = [json.loads(line) for line in
                       lifecycle_lines(directory, "l.jsonl")]
            final = next(record for record in records
                         if record["event_type"] == "final"
                         and record["target_id"] == "__run__")
            block = final["detail"]["metrics"]["communication"]
            self.assertTrue(declared <= set(block),
                            "the counters and the declared knobs must live in the block "
                            "the comparison drops, and only there")
            self.assertIsNone(behaviour_view(final)["detail"]["metrics"].get("communication"))
            run.logger.close()

    def test_probe_sampling_consumes_no_random_number(self) -> None:
        """Both streams must be untouched: a consumed draw would re-sequence packet loss."""
        with tempfile.TemporaryDirectory() as directory:
            configuration = RunConfiguration(
                mock=True, steps=3, seed=42, quiet=True,
                output=os.path.join(directory, "l.jsonl"),
                info_probe=os.path.join(directory, "p.jsonl"))
            run = build_joint_run(configuration)
            relay = run.task_adapter.comm_relay
            env = run.task_adapter.env
            before = (relay._rng.bit_generator.state, env.rng.bit_generator.state)
            run.info_probe.sample(step=1, now=1.0, env=env,
                                  external_to_internal=run.task_adapter.external_to_internal)
            after = (relay._rng.bit_generator.state, env.rng.bit_generator.state)
            run.logger.close()
            self.assertEqual(before, after, "the probe consumed a draw")


class RecordsSayWhatTheyClaim(unittest.TestCase):
    def test_every_placed_goal_is_recorded_even_if_nobody_finds_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, rows = run_episode(info_probe=os.path.join(directory, "p.jsonl"), num_goals=6)
            placed = [row for row in rows if row["kind"] == "placed"]
            summary = rows[-1]
            self.assertEqual(len(placed), 6)
            self.assertEqual(len({row["target_id"] for row in placed}), 6,
                             "placement ids must be unique or the denominator double-counts")
            self.assertEqual(summary["placed"], 6)
            self.assertEqual(summary["targets_never_known"],
                             6 - summary["targets_known_by_someone"])

    def test_delivery_records_close_against_the_transport_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probe_path = os.path.join(directory, "p.jsonl")
            configuration = RunConfiguration(
                mock=True, steps=6, seed=42, quiet=True,
                output=os.path.join(directory, "l.jsonl"),
                info_probe=probe_path, comm_delivery_probability=0.5)
            run = build_joint_run(configuration)
            run.run()
            run.logger.close()
            counters = run.task_adapter.comm_relay.counters
            rows = probe_rows(Path(probe_path).read_text(encoding="utf-8"))
            deliveries = [row for row in rows if row["kind"] == "delivery"]
            arrived = sum(1 for row in deliveries if row["arrived"])
            lost = sum(1 for row in deliveries if not row["arrived"])
            self.assertGreater(len(deliveries), 0, "a fair coin must lose something by step 6")
            self.assertEqual(len(deliveries), int(counters["transmission_attempts"]),
                             "one record per billed packet, no more and no less")
            self.assertEqual(lost, int(counters["packets_lost"]))
            self.assertEqual(arrived + lost, len(deliveries))

    def test_a_silenced_uplink_cannot_be_credited_with_any_knowledge(self) -> None:
        """The A2 arm: nothing goes on the air, so no belief may be attributed to the uplink."""
        with tempfile.TemporaryDirectory() as directory:
            _, _, rows = run_episode(info_probe=os.path.join(directory, "p.jsonl"),
                                     comm_policy="always_hold")
            self.assertFalse([row for row in rows if row["kind"] == "delivery"],
                             "an arm that never emits must never bill a packet")
            for row in rows:
                if row["kind"] == "knowledge":
                    self.assertNotEqual(row["channel"], "uplink_arrived")

    def test_ids_come_from_the_subscriber_and_placement_namespaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, rows = run_episode(info_probe=os.path.join(directory, "p.jsonl"))
            placed_ids = {row["target_id"] for row in rows if row["kind"] == "placed"}
            subscribers = {"0", "1", "2", "3", "4", "5"}
            for row in rows:
                if row["kind"] == "probe_summary":
                    continue
                if "robot_id" in row:
                    self.assertIn(str(row["robot_id"]), subscribers)
                if row["kind"] in ("delivery", "knowledge", "arrival"):
                    self.assertIn(row["target_id"], placed_ids,
                                  "a record named a target that was never placed")
            self.assertEqual(rows[-1]["anomaly_count"], 0)

    def test_turning_the_probe_off_costs_no_record_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = RunConfiguration(mock=True, steps=4, seed=42, quiet=True,
                                             output=os.path.join(directory, "l.jsonl"))
            run = build_joint_run(configuration)
            self.assertIsNone(run.info_probe)
            run.run()
            run.logger.close()
            self.assertEqual(list(Path(directory).glob("*.jsonl")),
                             [Path(directory) / "l.jsonl"])


class MappingIsNotGuessed(unittest.TestCase):
    def test_a_reused_internal_id_is_reported_instead_of_silently_overwritten(self) -> None:
        probe = InfoProbe("unused.jsonl")
        mapped = probe._map_ids({"a": 0, "b": 0, "c": 1})
        self.assertEqual(mapped[1], "c")
        self.assertIn("internal_id_reused:0:a:b", probe.anomalies)

    def test_an_empty_world_produces_a_summary_and_no_anomaly(self) -> None:
        probe = InfoProbe("unused.jsonl")

        class EmptyEnv:
            task_dic: dict = {}
            agent_knowledge: dict = {}
            agent_dic: dict = {}

        probe.sample(step=1, now=1.0, env=EmptyEnv(), external_to_internal={})
        self.assertEqual(probe.records()[-1]["anomalies"], [])


if __name__ == "__main__":
    unittest.main()
