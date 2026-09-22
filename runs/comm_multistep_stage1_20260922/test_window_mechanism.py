"""Synthetic mechanism tests for S(H)/H(H) window interventions.

No real episodes. A minimal stand-in relay exercises the emit entry so the window's
accounting rules are fixed before any simulation start is spent.
"""

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

from integrated_sar.comm_policy import SendItem, SendObservation  # noqa: E402
from window_intervention import (  # noqa: E402
    WindowIntervention,
    digests_equal,
    snapshot_relay_state,
)


def make_item(target: str = "t0", revision: int = 1, **kwargs) -> SendItem:
    values = dict(
        target_id=target,
        ready_step=1,
        age_steps=0,
        pending_event=False,
        changed_view=True,
        first_emission=False,
        revision=revision,
        revision_delta=1,
        stage_index=0,
        stage_changed=False,
        progress=0.2,
        progress_delta=0.1,
        requirements_changed=False,
        members_changed=False,
        deadline=100.0,
        remaining_time=80.0,
        sent_count=1,
        last_sent_step=0,
        since_last_emission_time=1.0,
        has_future_requirements=True,
        ack_confirmed_fraction=0.5,
        ack_unacked_broadcasts=1,
    )
    values.update(kwargs)
    return SendItem(**values)


class InnerPolicy:
    name = "inner"
    orders_selection = True

    def __init__(self, choose: set[str] | None = None):
        self.choose = set(choose or ())
        self.calls = 0

    def parameters(self):
        return {"choose": sorted(self.choose)}

    def decide(self, observation):
        self.calls += 1
        return tuple(
            item.target_id for item in observation.items if item.target_id in self.choose
        )


class FakeRelay:
    """Enough of CommRelay for the window wrapper's emit backstop."""

    def __init__(self, policy, *, budget_bytes=10_000):
        self._policy = policy
        self._tick = 0
        self._sequence = 0
        self._pending = []
        self._ack_pending = []
        self._ack_ordinal = 0
        self._ack_enabled = True
        self._rng = SimpleNamespace()
        self._ledger = {}
        self._message_calls = []
        self._emitted = []
        self.counters = {
            "emissions": 0,
            "transmission_attempts": 0,
            "airtime_bytes": 0,
            "budget_refused_sends": 0,
        }
        self.ack = SimpleNamespace(states={}, broadcast_attempts={})
        self.budget = SimpleNamespace(
            spent_bytes=0,
            bytes_per_episode=budget_bytes,
            remaining_bytes=budget_bytes,
            sends_this_step=0,
            can_send=lambda cost: self.counters["airtime_bytes"] + cost <= budget_bytes,
            why_blocked=lambda cost: "budget_episode",
            note_refusal=lambda blocked, cost: None,
            charge=lambda cost: self._charge(cost),
            begin_step=lambda: None,
            reset=lambda: None,
        )

    def _charge(self, cost: int) -> None:
        self.counters["airtime_bytes"] += int(cost)
        self.budget.spent_bytes = self.counters["airtime_bytes"]
        self.budget.remaining_bytes = int(self.budget.bytes_per_episode) - int(
            self.budget.spent_bytes
        )

    @property
    def policy(self):
        return self._policy

    def attach_policy(self, policy) -> None:
        self._policy = policy

    def _message(self, target, ledger, snapshot):
        self._message_calls.append(target)
        return SimpleNamespace(target_id=target, revision=ledger.get("revision", 1))

    def _emit(self, item, *, first=False, reason="policy"):
        ledger = self._ledger.setdefault(
            item.target_id,
            SimpleNamespace(
                last_sent=None,
                last_sent_step=None,
                sent_count=0,
                ready_step=1,
                pending_event=False,
                closed=False,
            ),
        )
        cost = 100
        if not self.budget.can_send(cost):
            self.counters["budget_refused_sends"] += 1
            return
        self.budget.charge(cost)
        self.counters["emissions"] += 1
        self.counters["transmission_attempts"] += 1
        ledger.last_sent = SimpleNamespace(revision=item.revision)
        ledger.last_sent_step = self._tick
        ledger.sent_count += 1
        self._emitted.append((self._tick, item.target_id, item.revision, reason))

    def _emit_selected(self):
        observation = self._observation()
        chosen = self._policy.decide(observation)
        by_target = {item.target_id: item for item in observation.items}
        for name in chosen:
            if name in by_target:
                self._emit(by_target[name], reason="policy")

    def _observation(self, items=None):
        if items is None:
            items = [make_item()]
        return SendObservation(
            step=self._tick,
            current_time=float(self._tick),
            items=tuple(items),
            remaining_budget_bytes=self.budget.remaining_bytes,
            budget_bytes_per_episode=self.budget.bytes_per_episode,
            capacity_remaining=10,
            backlog_targets=1,
            subscribers=1,
        )


class WindowMechanismTests(unittest.TestCase):
    def test_h1_boundary_blocks_only_the_window(self):
        relay = FakeRelay(InnerPolicy({"t0"}))
        window = WindowIntervention(mode="H", at_step=5, target="t0", horizon=1)
        try:
            window.wrap_relay(relay)
            for step in range(1, 9):
                relay._tick = step
                relay._emit_selected()
            emitted_steps = [row[0] for row in relay._emitted]
            self.assertEqual(emitted_steps, [1, 2, 3, 4, 6, 7, 8])
            self.assertGreaterEqual(window.stats.window_blocked_requests, 1)
        finally:
            window.uninstall()

    def test_h20_blocks_every_retry_in_window(self):
        relay = FakeRelay(InnerPolicy({"t0"}))
        window = WindowIntervention(mode="H", at_step=2, target="t0", horizon=20)
        try:
            window.wrap_relay(relay)
            for step in range(1, 25):
                relay._tick = step
                relay._emit_selected()
            emitted_steps = [row[0] for row in relay._emitted]
            self.assertEqual(emitted_steps, [1] + list(range(22, 25)))
            self.assertEqual(window.stats.window_blocked_requests, 20)
        finally:
            window.uninstall()

    def test_s_forces_once_then_window_silences(self):
        inner = InnerPolicy(set())  # rule would hold
        relay = FakeRelay(inner)
        window = WindowIntervention(mode="S", at_step=3, target="t0", horizon=5)
        try:
            window.wrap_relay(relay)
            for step in range(1, 10):
                relay._tick = step
                items = [make_item()]
                # emulate emit_selected with a custom observation via policy decide path
                observation = relay._observation(items)
                chosen = relay.policy.decide(observation)
                by_target = {item.target_id: item for item in observation.items}
                for name in chosen:
                    relay._emit(by_target[name], first=False, reason="policy")
            self.assertEqual(window.stats.forced_sends, 1)
            self.assertTrue(window.stats.force_applied)
            emitted = [row for row in relay._emitted if row[1] == "t0"]
            self.assertEqual(emitted, [(3, "t0", 1, "other")])
        finally:
            window.uninstall()

    def test_new_revision_truncates_window(self):
        relay = FakeRelay(InnerPolicy({"t0"}))
        window = WindowIntervention(mode="H", at_step=2, target="t0", horizon=20)
        try:
            window.wrap_relay(relay)
            relay._tick = 2
            relay.policy.decide(relay._observation([make_item(revision=1)]))
            relay._tick = 5
            relay.policy.decide(relay._observation([make_item(revision=2)]))
            self.assertTrue(window.stats.window_truncated)
            self.assertEqual(window.stats.window_truncated_at, 5)
            self.assertEqual(
                window.stats.window_truncated_reason, "window_truncated_by_revision"
            )
            self.assertEqual(window.stats.effective_window_steps, 3)
            # After truncation the new revision may send.
            relay._emit(make_item(revision=2), first=True, reason="policy")
            self.assertEqual(relay.counters["emissions"], 1)
        finally:
            window.uninstall()

    def test_does_not_lock_revision_before_tstar(self):
        relay = FakeRelay(InnerPolicy({"t0"}))
        window = WindowIntervention(mode="H", at_step=33, target="t0", horizon=5)
        try:
            window.wrap_relay(relay)
            # Earlier revisions of the same target must not become the locked view.
            for step, revision in ((1, 1), (10, 4), (32, 7)):
                relay._tick = step
                relay.policy.decide(relay._observation([make_item(revision=revision)]))
            self.assertIsNone(window.stats.locked_revision)
            self.assertFalse(window.stats.window_truncated)
            relay._tick = 33
            relay.policy.decide(relay._observation([make_item(revision=8)]))
            self.assertEqual(window.stats.locked_revision, 8)
            self.assertFalse(window.stats.window_truncated)
        finally:
            window.uninstall()

    def test_blocked_request_is_not_billed(self):
        relay = FakeRelay(InnerPolicy({"t0"}), budget_bytes=250)
        window = WindowIntervention(mode="H", at_step=1, target="t0", horizon=3)
        try:
            window.wrap_relay(relay)
            for step in (1, 2, 3):
                relay._tick = step
                relay._emit_selected()
            self.assertEqual(relay.counters["emissions"], 0)
            self.assertEqual(relay.counters["airtime_bytes"], 0)
            self.assertEqual(window.stats.window_blocked_requests, 3)
            # Outside the window a send is billed once.
            relay._tick = 4
            relay._emit_selected()
            self.assertEqual(relay.counters["emissions"], 1)
            self.assertEqual(relay.counters["airtime_bytes"], 100)
        finally:
            window.uninstall()

    def test_budget_refusal_is_hard_not_window(self):
        relay = FakeRelay(InnerPolicy({"t0"}), budget_bytes=50)
        window = WindowIntervention(mode="S", at_step=1, target="t0", horizon=5)
        try:
            window.wrap_relay(relay)
            relay._tick = 1
            relay._emit_selected()
            self.assertEqual(relay.counters["emissions"], 0)
            self.assertTrue(window.stats.force_refused)
            self.assertEqual(window.stats.forced_sends, 0)
            self.assertEqual(window.stats.window_blocked_requests, 0)
            self.assertEqual(relay.counters["budget_refused_sends"], 1)
        finally:
            window.uninstall()

    def test_capacity_style_refusal_counted_as_hard(self):
        # Same path as budget: original emit refuses before charging.
        relay = FakeRelay(InnerPolicy({"t0"}), budget_bytes=0)
        window = WindowIntervention(mode="H", at_step=1, target="t0", horizon=1)
        try:
            window.wrap_relay(relay)
            relay._tick = 1
            # Inner chooses, policy may suppress; force nothing. Use S-less H and call emit
            # directly to hit the hard path with a present view outside suppression timing.
            window.stats.locked_revision = 99  # different revision => not same view
            window.stats.window_truncated = False
            relay._emit(make_item(revision=1), first=False, reason="policy")
            self.assertEqual(relay.counters["budget_refused_sends"], 1)
            self.assertEqual(window.stats.window_blocked_requests, 0)
        finally:
            window.uninstall()

    def test_multiple_point_files_do_not_overwrite(self):
        import tempfile

        from ms_common import write_json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "cf_s700000_p1_H1_S.json", {"seed": 700000, "point": 1})
            write_json(root / "cf_s700000_p2_H1_S.json", {"seed": 700000, "point": 2})
            left = (root / "cf_s700000_p1_H1_S.json").read_text(encoding="utf-8")
            right = (root / "cf_s700000_p2_H1_S.json").read_text(encoding="utf-8")
            self.assertNotEqual(left, right)
            self.assertEqual(len(list(root.glob("cf_s700000_*.json"))), 2)

    def test_prefix_digest_reproducible_after_reset_like(self):
        relay_a = FakeRelay(InnerPolicy({"t0"}))
        relay_b = FakeRelay(InnerPolicy({"t0"}))
        for relay in (relay_a, relay_b):
            for step in (1, 2, 3):
                relay._tick = step
                relay._emit_selected()
        left = snapshot_relay_state(relay_a)
        right = snapshot_relay_state(relay_b)
        ok, problems = digests_equal(left, right)
        self.assertTrue(ok, problems)

    def test_uninstall_restores_normal_path(self):
        inner = InnerPolicy({"t0"})
        relay = FakeRelay(inner)
        window = WindowIntervention(mode="H", at_step=1, target="t0", horizon=10)
        window.wrap_relay(relay)
        relay._tick = 1
        relay._emit_selected()
        self.assertEqual(relay.counters["emissions"], 0)
        window.uninstall()
        relay._tick = 1
        relay._emit_selected()
        self.assertEqual(relay.counters["emissions"], 1)
        self.assertEqual(relay.counters["airtime_bytes"], 100)

    def test_selection_helper_rejects_outcome_fields(self):
        from ms_common import eligible, strip_outcome_fields

        ok_point = {
            "step": 3,
            "target": "t",
            "confirmed_fraction": 0.5,
            "changed_view": True,
            "rule_sent": True,
            "send": {"on_time": 99},
            "hold": {"on_time": 0},
        }
        stripped = strip_outcome_fields(ok_point)
        self.assertNotIn("send", stripped)
        self.assertNotIn("hold", stripped)
        self.assertTrue(eligible(stripped))
        self.assertFalse(eligible({"confirmed_fraction": 0.0, "changed_view": True}))
        self.assertFalse(eligible({"confirmed_fraction": 1.0, "changed_view": True}))
        self.assertFalse(eligible({"confirmed_fraction": 0.5, "changed_view": False}))


if __name__ == "__main__":
    unittest.main()
