"""Synthetic tests for target-scoped window (T0 building blocks)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _d in (str(ROOT), str(HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from integrated_sar.comm_policy import SendItem, SendObservation
from target_window import BoundaryComplete, TargetWindow


def make_item(target="t0", revision=1, **kw):
    values = dict(
        target_id=target,
        ready_step=1,
        age_steps=1,
        pending_event=False,
        changed_view=True,
        first_emission=False,
        revision=revision,
        revision_delta=1,
        stage_index=0,
        stage_changed=False,
        progress=0.1,
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
    values.update(kw)
    return SendItem(**values)


class Inner:
    name = "inner"
    orders_selection = True

    def __init__(self, choose=None):
        self.choose = set(choose or {"t0", "t1"})

    def parameters(self):
        return {}

    def decide(self, observation):
        return tuple(i.target_id for i in observation.items if i.target_id in self.choose)


class FakeRelay:
    def __init__(self, policy=None, budget=10000, subscribers=1):
        self._policy = policy or Inner()
        self._tick = 0
        self._sequence = 0
        self._pending = []
        self._ack_pending = []
        self._ack_ordinal = 0
        self._ack_enabled = False
        self._rng = SimpleNamespace()
        self._ledger = {}
        self._subscribers = [str(i) for i in range(subscribers)]
        self._snapshot_of = lambda tid: SimpleNamespace(
            status=SimpleNamespace(terminal=False), revision=1
        )
        self.counters = {
            "transmission_attempts": 0,
            "airtime_bytes": 0,
            "ack_attempt_bytes": 0,
            "emissions": 0,
        }
        self.ack = SimpleNamespace(states={}, broadcast_attempts={})
        self.budget = SimpleNamespace(
            spent_bytes=0,
            bytes_per_episode=budget,
            remaining_bytes=float(budget),
            sends_this_step=0,
            can_send=lambda c: self.counters["airtime_bytes"] + c <= budget,
            why_blocked=lambda c: (
                "capacity_step"
                if False
                else (
                    "budget_exhausted"
                    if self.counters["airtime_bytes"] + c > budget
                    else "allowed"
                )
            ),
            note_refusal=lambda r, c: None,
            charge=lambda c: self._charge(c),
            begin_step=lambda: None,
            reset=lambda: None,
        )

    def _charge(self, cost):
        self.counters["airtime_bytes"] += int(cost)
        self.budget.spent_bytes = self.counters["airtime_bytes"]
        self.budget.remaining_bytes = float(self.budget.bytes_per_episode) - float(
            self.budget.spent_bytes
        )

    def _message(self, target, ledger, snapshot):
        return SimpleNamespace(
            target_id=target,
            revision=2,
            sent_at=0.0,
            position=SimpleNamespace(x=0.0, y=0.0),
            stages=[],
            current_stage=0,
            remaining_requirements=[0],
            deadline=10.0,
            severity=1.0,
            confidence=1.0,
            progress=0.2,
            members=["0"],
            discovered_by="d",
            event_id="e",
        )

    @property
    def policy(self):
        return self._policy

    def attach_policy(self, policy):
        self._policy = policy

    def tick(self):
        self._tick += 1
        self._emit_selected()
        return ()

    def _emit(self, item, *, first=False, reason="policy"):
        cost = 100
        if not self.budget.can_send(cost):
            return
        self.budget.charge(cost)
        self.counters["emissions"] += 1
        self.counters["transmission_attempts"] += 1
        self._ledger[item.target_id] = SimpleNamespace(
            last_sent=self._message(item.target_id, None, None),
            last_sent_step=self._tick,
            sent_count=1,
            ready_step=1,
            pending_event=False,
            closed=False,
        )

    def _observation(self):
        items = [make_item("t0", 1), make_item("t1", 1)]
        return SendObservation(
            step=self._tick,
            current_time=float(self._tick),
            items=tuple(items),
            remaining_budget_bytes=self.budget.remaining_bytes,
            budget_bytes_per_episode=self.budget.bytes_per_episode,
            capacity_remaining=10,
            backlog_targets=2,
            subscribers=len(self._subscribers),
        )

    def _emit_selected(self):
        obs = self._observation()
        chosen = self._policy.decide(obs)
        by = {i.target_id: i for i in obs.items}
        for name in chosen:
            if name in by:
                self._emit(by[name], first=False, reason="policy")


class TargetWindowTests(unittest.TestCase):
    def test_early_sends_once_then_blocks_all_revisions(self):
        relay = FakeRelay(budget=100000, subscribers=1)
        w = TargetWindow(mode="EARLY_FLUSH", at_step=1, target="t0", horizon=5)
        try:
            w.wrap_relay(relay)
            raised = None
            for _ in range(10):
                try:
                    relay.tick()
                except BoundaryComplete as exc:
                    raised = exc
                    break
            self.assertIsNotNone(raised)
            self.assertTrue(w.stats.start_send_ok)
            self.assertGreaterEqual(w.stats.policy_requests_blocked, 1)
            # direct emit backstop during suppress window
            relay._tick = 3
            relay._emit(make_item("t0", 9), first=False, reason="policy")
            self.assertGreaterEqual(w.stats.emit_entry_blocked, 1)
            self.assertEqual(raised.record["boundary_status"], "flushed_latest")
        finally:
            w.uninstall()

    def test_defer_blocks_at_tstar(self):
        relay = FakeRelay(budget=100000)
        w = TargetWindow(mode="DEFER_FLUSH", at_step=1, target="t0", horizon=1)
        try:
            w.wrap_relay(relay)
            with self.assertRaises(BoundaryComplete) as ctx:
                for _ in range(5):
                    relay.tick()
            self.assertFalse(w.stats.start_send_ok)
            self.assertGreaterEqual(w.stats.policy_requests_blocked + w.stats.emit_entry_blocked, 1)
            self.assertEqual(ctx.exception.record["boundary_status"], "flushed_latest")
            self.assertGreaterEqual(w.stats.other_target_emissions, 1)
        finally:
            w.uninstall()

    def test_terminal_before_boundary(self):
        relay = FakeRelay(budget=100000)
        relay._snapshot_of = lambda tid: SimpleNamespace(
            status=SimpleNamespace(terminal=(tid == "t0")), revision=2
        )
        w = TargetWindow(mode="DEFER_FLUSH", at_step=1, target="t0", horizon=2)
        try:
            w.wrap_relay(relay)
            with self.assertRaises(BoundaryComplete) as ctx:
                for _ in range(5):
                    relay.tick()
            self.assertEqual(ctx.exception.record["boundary_status"], "terminal_before_boundary")
        finally:
            w.uninstall()

    def test_boundary_refused_credit(self):
        relay = FakeRelay(budget=50)
        relay._policy = Inner({"t0"})
        w = TargetWindow(mode="DEFER_FLUSH", at_step=1, target="t0", horizon=1)
        try:
            w.wrap_relay(relay)
            with self.assertRaises(BoundaryComplete) as ctx:
                for _ in range(4):
                    relay.tick()
            self.assertEqual(ctx.exception.record["boundary_status"], "refused_credit")
            self.assertEqual(ctx.exception.record["boundary_attempt_bytes"], 0)
        finally:
            w.uninstall()

    def test_exactly_affordable_flush(self):
        relay = FakeRelay(budget=100)
        relay._policy = Inner({"t0"})
        w = TargetWindow(mode="DEFER_FLUSH", at_step=1, target="t0", horizon=1)
        try:
            w.wrap_relay(relay)
            with self.assertRaises(BoundaryComplete) as ctx:
                for _ in range(4):
                    relay.tick()
            self.assertEqual(ctx.exception.record["boundary_status"], "flushed_latest")
            self.assertEqual(ctx.exception.record["boundary_attempt_bytes"], 100)
        finally:
            w.uninstall()

    def test_uninstall_restores(self):
        relay = FakeRelay()
        w = TargetWindow(mode="EARLY_FLUSH", at_step=1, target="t0", horizon=3)
        w.wrap_relay(relay)
        w.uninstall()
        relay._tick = 1
        relay._emit_selected()
        self.assertGreaterEqual(relay.counters["emissions"], 1)

    def test_blocked_requests_not_billed(self):
        relay = FakeRelay(budget=100000)
        w = TargetWindow(mode="DEFER_FLUSH", at_step=1, target="t0", horizon=3)
        try:
            w.wrap_relay(relay)
            with self.assertRaises(BoundaryComplete):
                for i in range(6):
                    if i == 1:
                        relay._tick = 2
                        relay._emit(make_item("t0", 5), first=False, reason="policy")
                    relay.tick()
            self.assertGreaterEqual(w.stats.policy_requests_blocked, 1)
            self.assertGreaterEqual(w.stats.emit_entry_blocked, 1)
        finally:
            w.uninstall()


if __name__ == "__main__":
    unittest.main()
