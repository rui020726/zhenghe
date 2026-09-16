"""Tests for the switchable allocation decision trace and its analysis.

Two things must hold before a trace-enabled re-run can be trusted:

* attaching the trace must not move a single decision -- it is a sink written after the
  allocator has already chosen, so the same run with and without ``--decision-trace``
  issues the same commands;
* the analysis must read the questions the trace exists to answer off the recorded rows
  (depot despite a legal task, churn before arrival, group formation, logit margin), on
  synthetic rows as well as on a real writer's output.

Everything except the adapter integration test runs without torch.
"""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from integrated_sar.allocation_trace import (
    DEPOT_INDEX,
    TRACE_FORMAT,
    DecisionFacts,
    DecisionTraceWriter,
    analyse,
    ctde_margin,
    legal_task_indices,
    read_trace,
    summarise,
)
from integrated_sar.contracts import CoordinationAdvice

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _facts(**overrides: Any) -> DecisionFacts:
    values: dict[str, Any] = {
        "step": 1,
        "timestamp": 1.0,
        "agent_id": "0",
        "status": "idle",
        "abilities": (1.0, 0.0, 0.0, 0.0, 0.0),
        "position": (0.0, 0.0),
        "travel_dist": 0.0,
        "needs_decision": True,
        "replanned": False,
        "belief_task_count": 1,
        "action_task_ids": (None, "target-a"),
        "mask": (False, False),
        "chosen_index": 1,
    }
    values.update(overrides)
    return DecisionFacts(**values)


def _decision(**overrides: Any) -> dict[str, Any]:
    """One row shaped like a decision line read back from a trace file."""
    row: dict[str, Any] = {
        "event_type": "decision",
        "timestamp": 0.0,
        "agent_id": "0",
        "status": "idle",
        "action_task_ids": [None, "target-a"],
        "mask": [False, False],
        "chosen_index": DEPOT_INDEX,
        "current_kind": None,
        "current_task": None,
        "truth": {"targets": {}},
    }
    row.update(overrides)
    # The writer derives these two from the choice; a synthetic row must carry the same
    # redundancy or the analysis would read a row shape the real trace never produces.
    chosen = int(row["chosen_index"])
    ids = row["action_task_ids"]
    row["chosen_kind"] = "depot" if chosen == DEPOT_INDEX else "task"
    row["chosen_task"] = None if chosen == DEPOT_INDEX or chosen >= len(ids) else ids[chosen]
    return row


class DecisionFactsTests(unittest.TestCase):
    def test_masked_actions_are_recorded_as_null_not_as_the_sentinel(self) -> None:
        facts = _facts(
            action_task_ids=(None, "target-a", "target-b"),
            mask=(False, True, False),
            chosen_index=2,
            ctde_logits=(0.25, None, -0.5),
            bias=(0.0, None, 0.125),
            final_logits=(0.25, None, -0.375),
        )
        payload = facts.as_json({"arm": "rs0.25", "seed": 7})
        for key in ("ctde_logits", "bias", "final_logits"):
            self.assertEqual(payload[key][1], None, key)
        self.assertEqual(payload["ctde_logits"][0], 0.25)
        self.assertEqual(payload["chosen_kind"], "task")
        self.assertEqual(payload["chosen_task"], "target-b")
        self.assertEqual(facts.legal_task_indices, (2,))

    def test_a_depot_choice_carries_no_task_id(self) -> None:
        facts = _facts(chosen_index=DEPOT_INDEX)
        self.assertEqual(facts.chosen_kind, "depot")
        self.assertIsNone(facts.chosen_task)

    def test_a_backend_without_a_network_records_no_logits(self) -> None:
        payload = _facts().as_json({"arm": "nocoord"})
        self.assertIsNone(payload["ctde_logits"])
        self.assertIsNone(payload["bias"])
        self.assertIsNone(payload["final_logits"])


class DecisionTraceWriterTests(unittest.TestCase):
    def test_a_trace_can_be_read_back_with_its_identity(self) -> None:
        stream = io.StringIO()
        writer = DecisionTraceWriter(stream, meta={"arm": "rs0.25", "seed": 3})
        writer.record(_facts())
        writer.finish({"completed": 4})

        meta, decisions = _write_and_read(stream.getvalue())
        self.assertEqual(meta["arm"], "rs0.25")
        self.assertEqual(meta["seed"], 3)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["chosen_task"], "target-a")
        # Only the decision counts; the header and the closing line are not decisions.
        self.assertEqual(writer.records_written, 1)

    def test_the_header_is_written_once_per_file_not_once_per_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tr_s1.jsonl"
            with DecisionTraceWriter(path, meta={"arm": "nocoord", "seed": 1}) as writer:
                writer.record(_facts())
                writer.reset()
                writer.record(_facts(chosen_index=DEPOT_INDEX))
            meta, decisions = read_trace(path)
            self.assertEqual(meta["arm"], "nocoord")
            self.assertEqual(len(decisions), 2)

    def test_recording_after_close_is_refused(self) -> None:
        writer = DecisionTraceWriter(io.StringIO(), meta={})
        writer.close()
        with self.assertRaises(RuntimeError):
            writer.record(_facts())


def _write_and_read(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.jsonl"
        path.write_text(text, encoding="utf-8")
        return read_trace(path)


class MarginTests(unittest.TestCase):
    def test_the_margin_is_the_best_legal_task_against_the_depot(self) -> None:
        row = _decision(
            action_task_ids=[None, "target-a", "target-b"],
            mask=[False, False, True],
            ctde_logits=[0.2, 0.5, 4.0],
        )
        self.assertEqual(legal_task_indices(row), [1])
        self.assertAlmostEqual(ctde_margin(row), 0.3)

    def test_a_trace_without_logits_has_no_margin(self) -> None:
        self.assertIsNone(ctde_margin(_decision()))

    def test_a_trace_without_a_legal_task_has_no_margin(self) -> None:
        row = _decision(ctde_logits=[0.2, None], mask=[False, True])
        self.assertIsNone(ctde_margin(row))


class AnalyseTests(unittest.TestCase):
    def test_depot_with_a_legal_task_is_split_by_whether_the_robot_was_busy(self) -> None:
        rows = [
            _decision(timestamp=0.0, chosen_index=DEPOT_INDEX),
            _decision(
                timestamp=1.0,
                chosen_index=DEPOT_INDEX,
                current_kind="task",
                current_task="target-a",
                ctde_logits=[0.0, 1.0],
            ),
            _decision(timestamp=2.0, chosen_index=1, ctde_logits=[0.0, -1.0]),
            # No legal task at all: not part of the denominator.
            _decision(timestamp=3.0, mask=[False, True], action_task_ids=[None, "target-a"]),
        ]
        metrics = analyse(rows, arm="rs0.25", seed=1)
        self.assertEqual(metrics.decisions, 4)
        self.assertEqual(metrics.legal_decisions, 3)
        self.assertEqual(metrics.depot_with_legal, 2)
        self.assertEqual(metrics.depot_with_legal_idle, 1)
        self.assertEqual(metrics.depot_with_legal_committed, 1)
        self.assertEqual(metrics.margin_positive, 1)
        # The second row wanted a task (margin +1) and went home instead.
        self.assertEqual(metrics.depot_despite_margin, 1)
        # The third row wanted the depot (margin -1) and took the task -- the residual
        # overrode the actor, which is the coordinator working as intended.
        self.assertEqual(metrics.task_despite_margin, 1)
        self.assertEqual(metrics.margins, [1.0, -1.0])

    def test_the_action_space_composition_is_reported_next_to_the_choice(self) -> None:
        rows = [
            # Nothing but the depot in this robot's action space: not a preference.
            _decision(timestamp=0.0, action_task_ids=[None], mask=[False]),
            # A task was on the menu but this robot may not take it.
            _decision(timestamp=1.0, mask=[False, True]),
            # A legal task was on the menu and the depot was chosen anyway.
            _decision(timestamp=2.0),
            _decision(
                timestamp=3.0,
                chosen_index=1,
                status="moving",
                current_kind="task",
                current_task="target-a",
            ),
        ]
        metrics = analyse(rows, arm="rs0.25", seed=6)
        self.assertEqual(metrics.decisions, 4)
        self.assertEqual(metrics.no_task_on_menu, 1)
        self.assertEqual(metrics.all_tasks_masked, 1)
        # The masked-only row is not a legal-task decision; the other two are.
        self.assertEqual(metrics.legal_decisions, 2)
        self.assertEqual(metrics.depot_with_legal, 1)
        self.assertEqual(metrics.busy_decisions, 1)
        self.assertEqual(summarise([metrics])["rs0.25"]["no_task_on_menu_share"], 0.25)

    def test_a_robot_switched_before_arrival_is_churn(self) -> None:
        rows = [
            _decision(timestamp=0.0, agent_id="0"),
            _decision(
                timestamp=1.0,
                agent_id="0",
                status="moving",
                current_kind="task",
                current_task="target-a",
            ),
            # Re-tasked while still en route: never delivered a step of work.
            _decision(timestamp=4.0, agent_id="0", current_kind="depot", current_task=None),
            _decision(
                timestamp=5.0,
                agent_id="0",
                status="moving",
                current_kind="task",
                current_task="target-b",
            ),
            _decision(
                timestamp=6.0,
                agent_id="0",
                status="working",
                current_kind="task",
                current_task="target-b",
            ),
        ]
        metrics = analyse(rows, arm="rs0.25", seed=1)
        self.assertEqual(metrics.task_commitments, 2)
        self.assertEqual(metrics.switched_before_arrival, 1)
        self.assertEqual(metrics.switch_to_depot, 1)
        self.assertEqual(metrics.switch_to_task, 0)
        self.assertEqual(metrics.arrived_commitments, 1)
        self.assertEqual(metrics.durations, [3.0, 1.0])

    def test_a_commitment_still_open_when_the_run_ends_is_not_churn(self) -> None:
        rows = [
            _decision(
                timestamp=0.0,
                status="moving",
                current_kind="task",
                current_task="target-a",
            ),
            _decision(
                timestamp=9.0,
                status="moving",
                current_kind="task",
                current_task="target-a",
            ),
        ]
        metrics = analyse(rows, arm="nocoord", seed=2)
        self.assertEqual(metrics.task_commitments, 1)
        self.assertEqual(metrics.switched_before_arrival, 0)
        self.assertEqual(metrics.durations, [9.0])

    def test_arrival_is_read_from_status_or_from_the_target_being_reached(self) -> None:
        rows = [
            _decision(
                timestamp=0.0,
                agent_id="0",
                status="moving",
                current_kind="task",
                current_task="target-a",
            ),
            # Reached the navigation target without ever reporting "working".
            _decision(
                timestamp=2.0,
                agent_id="0",
                status="moving",
                current_kind="task",
                current_task="target-a",
                current_target_taken=True,
            ),
            _decision(
                timestamp=3.0,
                agent_id="0",
                status="moving",
                current_kind="task",
                current_task="target-b",
            ),
            _decision(timestamp=4.0, agent_id="0", current_kind="depot", current_task=None),
        ]
        metrics = analyse(rows, arm="nocoord", seed=3)
        self.assertEqual(metrics.task_commitments, 2)
        self.assertEqual(metrics.arrived_commitments, 1)
        self.assertEqual(metrics.switched_before_arrival, 1)
        self.assertEqual(metrics.switch_to_depot, 1)
        self.assertEqual(metrics.switch_to_task, 0)

    def test_group_formation_is_measured_from_the_first_dispatch(self) -> None:
        rows = [
            _decision(
                timestamp=0.0,
                chosen_index=1,
                truth={
                    "targets": {
                        "target-a": {"deadline": 500.0, "covered": False, "en_route_covers": True}
                    }
                },
            ),
            _decision(
                timestamp=30.0,
                truth={
                    "targets": {
                        "target-a": {"deadline": 500.0, "covered": True, "en_route_covers": True}
                    }
                },
            ),
        ]
        metrics = analyse(rows, arm="rs0.25", seed=4)
        summary = summarise([metrics])["rs0.25"]
        self.assertEqual(summary["targets_dispatched"], 1)
        self.assertEqual(summary["group_covered_share"], 1.0)
        self.assertEqual(summary["group_covered_gap_median"], 30.0)
        self.assertEqual(summary["group_filled_gap_median"], 0.0)
        self.assertEqual(summary["never_covered"], 0)

    def test_a_target_whose_group_never_assembles_is_counted_as_such(self) -> None:
        rows = [
            _decision(
                timestamp=0.0,
                chosen_index=1,
                truth={"targets": {"target-a": {"deadline": 50.0, "covered": False}}},
            ),
            _decision(timestamp=10.0),
        ]
        summary = summarise([analyse(rows, arm="nocoord", seed=5)])["nocoord"]
        self.assertEqual(summary["group_covered_share"], 0.0)
        self.assertEqual(summary["never_covered"], 1)
        self.assertEqual(summary["never_filled"], 1)

    def test_arms_are_pooled_separately_with_a_per_seed_interval(self) -> None:
        rows = [
            analyse([_decision(chosen_index=DEPOT_INDEX)], arm="nocoord", seed=1),
            analyse([_decision(chosen_index=1)], arm="nocoord", seed=2),
            analyse([_decision(chosen_index=DEPOT_INDEX)], arm="rs0.25", seed=1),
        ]
        summary = summarise(rows)
        self.assertEqual(sorted(summary), ["nocoord", "rs0.25"])
        self.assertEqual(summary["nocoord"]["depot_with_legal"], 1)
        self.assertEqual(summary["nocoord"]["legal_decisions"], 2)
        self.assertEqual(summary["nocoord"]["depot_with_legal_share"], 0.5)
        self.assertEqual(summary["rs0.25"]["depot_with_legal_share"], 1.0)
        self.assertEqual(summary["rs0.25"]["seeds"], [1])

    def test_the_writer_output_round_trips_through_the_analysis(self) -> None:
        stream = io.StringIO()
        with DecisionTraceWriter(stream, meta={"arm": "rs0.25", "seed": 11}) as writer:
            writer.record(
                _facts(
                    timestamp=2.0,
                    action_task_ids=(None, "target-a"),
                    mask=(False, True),
                    chosen_index=DEPOT_INDEX,
                    ctde_logits=(0.5, None),
                )
            )
            writer.record(
                _facts(
                    timestamp=3.0,
                    action_task_ids=(None, "target-a"),
                    mask=(False, False),
                    chosen_index=1,
                    ctde_logits=(0.5, -0.25),
                    bias=(0.0, 0.75),
                    final_logits=(0.5, 0.5),
                    current_kind="task",
                    current_task="target-a",
                )
            )
        meta, decisions = _write_and_read(stream.getvalue())
        self.assertEqual(meta["arm"], "rs0.25")
        self.assertEqual(decisions[0]["mask"], [False, True])
        self.assertEqual(ctde_margin(decisions[1]), -0.75)
        metrics = analyse(decisions, arm=str(meta["arm"]), seed=int(meta["seed"]))
        self.assertEqual(metrics.decisions, 2)
        self.assertEqual(metrics.legal_decisions, 1)
        # The first row had its only task masked, so it is not "depot despite a legal
        # task"; the second row took the task.
        self.assertEqual(metrics.depot_with_legal, 0)

    def test_a_trace_file_is_not_mistaken_for_a_lifecycle_log(self) -> None:
        stream = io.StringIO()
        with DecisionTraceWriter(stream, meta={"arm": "nocoord", "seed": 1}) as writer:
            writer.record(_facts())
        self.assertEqual(json.loads(stream.getvalue().splitlines()[0])["format"], TRACE_FORMAT)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TracedDecisionTests(unittest.TestCase):
    """The adapter records the decision it already made, and only that."""

    def _run(self, *, with_trace: bool) -> tuple[Any, list[dict[str, Any]]]:
        import numpy as np
        import torch

        from integrated_sar.adapters.task03 import Task03Adapter

        action_map = [("depot", None), ("task", 7), ("task", 8)]
        # The robot does not believe it may work on target-b (index 2); only target-a
        # (index 1) and the depot are legal.
        mask = np.array([False, False, True])

        class _StubEnv:
            def __init__(self) -> None:
                self.current_time = 3.0
                self._inner_step_counter = 3
                self.task_dic: dict[int, Any] = {}
                self.agent_dic = {
                    0: {
                        "location": np.array([0.4, 0.0]),
                        "target_pos": np.array([0.5, 0.0]),
                        "target_type": "task",
                        "target_id": 8,
                        "current_task": 8,
                        "abilities": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                        "status": "moving",
                        "travel_dist": 0.1,
                    }
                }

            def get_local_obs(self, agent_id: int, max_tasks: int | None = None) -> Any:
                del agent_id, max_tasks
                return (
                    np.zeros(4, dtype=np.float32),
                    np.zeros((2, 4), dtype=np.float32),
                    np.zeros((len(action_map), 4), dtype=np.float32),
                    mask,
                    action_map,
                )

            def apply_action(self, agent_id: int, action_index: int, actions: Any) -> None:
                kind, task_id = actions[action_index]
                self.agent_dic[agent_id]["target_type"] = kind
                self.agent_dic[agent_id]["target_id"] = task_id

        class _Network:
            """Depot slightly ahead of the legal task, so a bias can flip it."""

            def parameters(self) -> Any:
                return iter([torch.zeros(1)])

            def get_action(self, state: Any, neighbors: Any, tasks: Any, hard_mask: Any) -> Any:
                del state, neighbors
                logits = torch.tensor([[0.10, 0.05, -1e9]])
                return type("Distribution", (), {"logits": logits})()

        env = _StubEnv()
        adapter = Task03Adapter(
            env=env,
            network=_Network(),
            clear_initial_tasks=False,
            residual_scale=0.25,
        )
        adapter.internal_to_external = {7: "target-a", 8: "target-b"}
        stream = io.StringIO()
        if with_trace:
            adapter.decision_trace = DecisionTraceWriter(stream, meta={"arm": "rs0.25", "seed": 1})

        agent = type("Agent", (), {"agent_id": "0", "needs_decision": True, "known_tasks": (1, 2)})()
        advice = CoordinationAdvice(
            revision=1,
            generated_at=3.0,
            valid_until=13.0,
            agent_task_bias={("0", "target-a"): 1.0},
        )
        commands = adapter.allocate((), (agent,), advice, 3.0)
        adapter.apply_assignments(commands)

        _, decisions = _write_and_read(stream.getvalue()) if with_trace else ({}, [])
        return commands, decisions

    def test_the_trace_does_not_change_the_decision(self) -> None:
        traced, _ = self._run(with_trace=True)
        untraced, _ = self._run(with_trace=False)
        self.assertEqual(
            [(c.task_id, c.task_revision) for c in traced],
            [(c.task_id, c.task_revision) for c in untraced],
        )

    def test_the_recorded_choice_is_the_command_that_was_issued(self) -> None:
        commands, decisions = self._run(with_trace=True)
        self.assertEqual(len(decisions), 1)
        row = decisions[0]
        self.assertEqual(row["chosen_index"], 1)
        self.assertEqual(row["chosen_task"], commands[0].task_id)
        self.assertEqual(row["current_task"], "target-b")
        self.assertEqual(row["belief_task_count"], 2)
        self.assertTrue(row["needs_decision"])

    def test_the_residual_is_visible_in_the_logits_and_the_mask_still_wins(self) -> None:
        _, decisions = self._run(with_trace=True)
        row = decisions[0]
        # 0.25 * 1.0 of advice, on the task action only; the depot is never biased.
        self.assertEqual(row["bias"], [0.0, 0.25, None])
        self.assertAlmostEqual(row["ctde_logits"][0], 0.10)
        self.assertAlmostEqual(row["ctde_logits"][1], 0.05)
        self.assertAlmostEqual(row["final_logits"][0], 0.10)
        self.assertAlmostEqual(row["final_logits"][1], 0.30)
        # An illegal action is recorded as illegal rather than as "very unlikely": the
        # sentinel the network writes never reaches the trace.
        self.assertEqual(row["mask"], [False, False, True])
        self.assertIsNone(row["ctde_logits"][2])
        self.assertIsNone(row["final_logits"][2])

    def test_the_commitment_age_starts_when_the_target_was_applied(self) -> None:
        _, decisions = self._run(with_trace=True)
        self.assertEqual(decisions[0]["current_task_age"], 0.0)
        self.assertAlmostEqual(decisions[0]["current_target_distance"], 0.1)
        self.assertAlmostEqual(decisions[0]["current_target_eta"], 0.5)
        self.assertFalse(decisions[0]["current_target_taken"])


if __name__ == "__main__":
    unittest.main()