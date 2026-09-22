"""Synthetic tests for actionability classification and executor-shared SEND predicate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _directory in (str(ROOT), str(HERE)):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

from integrated_sar.comm_budget import CommBudget
from integrated_sar.comm_policy import SendItem
from send_actionability import (
    ActionabilityRow,
    attempt_charge_bytes,
    classify,
    classify_closed,
    identity_ok,
    message_class_of,
    send_block_from_budget,
)


def make_item(**kwargs) -> SendItem:
    values = dict(
        target_id="t0",
        ready_step=1,
        age_steps=1,
        pending_event=False,
        changed_view=True,
        first_emission=False,
        revision=3,
        revision_delta=1,
        stage_index=0,
        stage_changed=False,
        progress=0.2,
        progress_delta=0.1,
        requirements_changed=False,
        members_changed=False,
        deadline=100.0,
        remaining_time=50.0,
        sent_count=1,
        last_sent_step=0,
        since_last_emission_time=1.0,
        has_future_requirements=True,
        ack_confirmed_fraction=0.5,
        ack_unacked_broadcasts=1,
    )
    values.update(kwargs)
    return SendItem(**values)


def make_row(**kwargs) -> ActionabilityRow:
    values = dict(
        seed=1,
        step=1,
        communication_substep=0,
        target_id="t0",
        semantic_revision=3,
        content_fingerprint="fp",
        subscriber_set=("0", "1"),
        legal=True,
        terminal=False,
        subscriber_count=2,
        payload_bytes=100,
        attempt_charge_bytes=200,
        budget_spent=0,
        budget_remaining=1000.0,
        budget_bytes_per_episode=1000,
        sends_this_step=0,
        capacity_remaining=None,
        charge_over_remaining=0.2,
        sent_count=0,
        attempts_remaining=5,
        message_class="first",
        changed_view=True,
        first_emission=True,
        pending_event=True,
        stage_changed=False,
        requirements_changed=False,
        members_changed=False,
        progress_delta=0.0,
        age_steps=0,
        remaining_time=50.0,
        ack_confirmed_fraction=-1.0,
        ack_unacked_broadcasts=-1,
        rule_requested=True,
        send_executable=True,
        hold_executable=True,
        send_block_reason=None,
        hold_block_reason=None,
        protocol_forced=False,
        in_policy_interface=True,
    )
    values.update(kwargs)
    return ActionabilityRow(**values)


class PredicateMatchesExecutorTests(unittest.TestCase):
    def test_budget_predicate_matches_can_send(self):
        budget = CommBudget(bytes_per_episode=1000, sends_per_step=2)
        cases = [
            (200, 0, 0, None),
            (200, 900, 0, "budget_exhausted"),
            (100, 0, 2, "capacity_step"),
            (100, 0, 1, None),
        ]
        for cost, spent, sends, expected in cases:
            budget.spent_bytes = spent
            budget.sends_this_step = sends
            self.assertEqual(send_block_from_budget(budget, cost), expected)
            self.assertEqual(budget.can_send(cost), expected is None)

    def test_exactly_affordable_is_allowed(self):
        budget = CommBudget(bytes_per_episode=200, sends_per_step=None)
        budget.spent_bytes = 50
        self.assertIsNone(send_block_from_budget(budget, 150))
        self.assertEqual(send_block_from_budget(budget, 151), "budget_exhausted")

    def test_attempt_charge_is_payload_times_fanout(self):
        message = SimpleNamespace()

        class FakeMsg:
            event_id = "e"
            target_id = "t"
            sent_at = 0.0
            position = SimpleNamespace(x=0.0, y=0.0)
            stages = []
            current_stage = 0
            remaining_requirements = []
            deadline = 1.0
            severity = 1.0
            confidence = 1.0
            progress = 0.0
            members = []
            discovered_by = "d"
            revision = 1

        msg = FakeMsg()
        one = attempt_charge_bytes(msg, ["a"])
        three = attempt_charge_bytes(msg, ["a", "b", "c"])
        self.assertEqual(three, 3 * one)
        self.assertGreater(one, 0)


class ClassificationTests(unittest.TestCase):
    def test_both_actions_free(self):
        primary, _ = classify(make_row())
        self.assertEqual(primary, "both_actions_free")

    def test_credit_blocks(self):
        primary, secondary = classify(
            make_row(
                send_executable=False,
                send_block_reason="budget_exhausted",
                first_emission=False,
                pending_event=False,
                changed_view=True,
                message_class="update",
            )
        )
        self.assertEqual(primary, "send_blocked_credit")
        self.assertIn("budget_exhausted", secondary)

    def test_capacity_blocks(self):
        primary, _ = classify(
            make_row(send_executable=False, send_block_reason="capacity_step")
        )
        self.assertEqual(primary, "send_blocked_capacity_or_queue")

    def test_terminal_blocks(self):
        primary, _ = classify(make_row(terminal=True))
        self.assertEqual(primary, "send_blocked_content_or_terminal")

    def test_retry_exhausted(self):
        primary, _ = classify(
            make_row(
                first_emission=False,
                pending_event=False,
                changed_view=False,
                message_class="same_view_retry",
                attempts_remaining=0,
                sent_count=5,
            )
        )
        self.assertEqual(primary, "send_blocked_retry_exhausted")

    def test_protocol_forced(self):
        primary, _ = classify(make_row(protocol_forced=True, hold_executable=False))
        self.assertEqual(primary, "hold_not_free_protocol_forced")

    def test_not_policy_decision(self):
        primary, _ = classify(make_row(in_policy_interface=False))
        self.assertEqual(primary, "not_policy_decision")

    def test_other_requires_explanation_path(self):
        primary, secondary = classify(
            make_row(send_executable=False, send_block_reason="weird")
        )
        self.assertEqual(primary, "other")
        self.assertTrue(secondary)

    def test_identity_closes(self):
        rows = [
            make_row(seed=1),
            make_row(seed=2, send_executable=False, send_block_reason="budget_exhausted"),
            make_row(seed=3, terminal=True),
            make_row(seed=4, protocol_forced=True),
        ]
        counts = classify_closed(rows)
        self.assertTrue(identity_ok(rows, counts))
        self.assertEqual(sum(counts.values()), 4)

    def test_message_classes(self):
        self.assertEqual(
            message_class_of(first_emission=True, pending_event=False, changed_view=True),
            "first",
        )
        self.assertEqual(
            message_class_of(first_emission=False, pending_event=True, changed_view=True),
            "first",
        )
        self.assertEqual(
            message_class_of(first_emission=False, pending_event=False, changed_view=True),
            "update",
        )
        self.assertEqual(
            message_class_of(first_emission=False, pending_event=False, changed_view=False),
            "same_view_retry",
        )

    def test_no_outcome_fields_in_row(self):
        banned = {"on_time", "expired", "send", "hold", "free", "total_bytes", "status"}
        row = make_row()
        self.assertFalse(banned & set(row.as_dict().keys()))


if __name__ == "__main__":
    unittest.main()
