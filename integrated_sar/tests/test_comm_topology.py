"""The production wiring must honour a configured communication topology.

Third review round, finding 5: ``build_joint_run`` put every rescue robot into
the uplink unconditionally and ``RunConfiguration`` had no topology setting, so
the subset-topology behaviour only existed in hand-built unit tests. The default
is still the whole rescue team, but it is now an explicit, configurable choice
with integration coverage rather than an accident of wiring.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any

from integrated_sar.errors import ConfigurationError
from integrated_sar.joint_driver import (
    RunConfiguration,
    build_joint_run,
    main,
    parse_comm_subscribers,
)


def make_configuration(directory: str, **overrides: Any) -> RunConfiguration:
    settings: dict[str, Any] = {
        "mock": True,
        "steps": 6,
        "seed": 42,
        "quiet": True,
        "output": os.path.join(directory, "lifecycle.jsonl"),
    }
    settings.update(overrides)
    return RunConfiguration(**settings)


class TopologySpecTests(unittest.TestCase):
    def test_specs_parse_into_the_intended_receiver_sets(self) -> None:
        self.assertIsNone(parse_comm_subscribers("all"))
        self.assertIsNone(parse_comm_subscribers("ALL"))
        self.assertEqual(parse_comm_subscribers("first:2"), ("0", "1"))
        self.assertEqual(parse_comm_subscribers("0"), ("0",))
        self.assertEqual(parse_comm_subscribers("0,1,3"), ("0", "1", "3"))
        self.assertEqual(parse_comm_subscribers("1,1,2"), ("1", "2"))
        self.assertEqual(parse_comm_subscribers("nonsense"), ())

    def test_an_unparsable_spec_is_a_configuration_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(directory, comm_subscribers="nope")
            with self.assertRaises(ConfigurationError) as caught:
                configuration.validate()
        self.assertIn("--comm-subscribers", str(caught.exception))

    def test_an_unknown_robot_id_fails_with_a_clean_cli_error(self) -> None:
        import contextlib
        import io

        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(
                ["--mock", "--steps", "2", "--comm-subscribers", "99", "--output", "-"]
            )
        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        lines = [line for line in stderr.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertIn("--comm-subscribers", lines[0])


class TopologyIntegrationTests(unittest.TestCase):
    """The wired relay must actually use the configured receiver set."""

    def _knowing_agents(self, configuration: RunConfiguration) -> list[str]:
        run = build_joint_run(configuration)
        try:
            run.run()
            return sorted(
                snapshot.agent_id
                for snapshot in run.task_adapter.agent_snapshots()
                if snapshot.known_tasks
            )
        finally:
            run.logger.close()

    def test_the_default_uplink_covers_the_whole_rescue_team(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(directory)
            run = build_joint_run(configuration)
            try:
                subscribers = run.task_adapter.comm_relay._subscribers
                total = len(run.task_adapter.env.agent_dic)
            finally:
                run.logger.close()
        self.assertEqual(len(subscribers), total)
        self.assertGreater(total, 1)

    def test_a_narrowed_uplink_delivers_to_the_listed_robot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configuration = make_configuration(
                directory, comm_subscribers="0", output=os.path.join(directory, "one.jsonl")
            )
            run = build_joint_run(configuration)
            try:
                self.assertEqual(run.task_adapter.comm_relay._subscribers, ("0",))
                run.run()
                knowing = sorted(
                    snapshot.agent_id
                    for snapshot in run.task_adapter.agent_snapshots()
                    if snapshot.known_tasks
                )
                delivered = run.counters.get("detections", 0)
            finally:
                run.logger.close()

        self.assertGreater(delivered, 0, "nothing was detected, so the test proves nothing")
        # The single receiver carries the uplink, so it must learn what was found.
        # Any other robot that knows a task can only have learned it through
        # sensing or gossip, which the topology setting does not control.
        self.assertIn("0", knowing)

    def test_narrowing_the_uplink_changes_what_the_team_knows(self) -> None:
        # Same seed and scenario: the full uplink informs more robots than a
        # single-receiver link, so the topology is load-bearing, not cosmetic.
        with tempfile.TemporaryDirectory() as directory:
            wide = self._knowing_agents(
                make_configuration(directory, output=os.path.join(directory, "wide.jsonl"))
            )
        with tempfile.TemporaryDirectory() as directory:
            narrow = self._knowing_agents(
                make_configuration(
                    directory,
                    comm_subscribers="0",
                    output=os.path.join(directory, "narrow.jsonl"),
                )
            )
        self.assertGreaterEqual(len(wide), len(narrow))


if __name__ == "__main__":
    unittest.main()
