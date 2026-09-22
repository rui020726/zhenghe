"""Target-scoped EARLY_FLUSH / DEFER_FLUSH window at the real _emit entry.

Does not modify integrated_sar. Blocks only one target's uplink task emissions;
ACK and other targets continue. Boundary flush uses the live observation item
and the real _message/_emit path with real billing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from integrated_sar.comm_ack import content_fingerprint
from integrated_sar.comm_policy import SendItem, SendObservation


class BoundaryComplete(Exception):
    """Raised after the boundary flush attempt so the episode stops immediately."""

    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__("boundary complete")
        self.record = record


@dataclass
class WindowStats:
    mode: str
    at_step: int
    horizon: int
    target: str
    start_send_requested: bool = False
    start_send_ok: bool = False
    start_send_refused_reason: str | None = None
    start_send_bytes: int = 0
    start_revision: int | None = None
    start_fingerprint: str | None = None
    policy_requests_blocked: int = 0
    emit_entry_blocked: int = 0
    revisions_seen: list[int] = field(default_factory=list)
    coalesced_ready_updates: int = 0
    other_target_emissions: int = 0
    other_target_bytes: int = 0
    boundary_status: str = "not_reached"
    boundary_revision: int | None = None
    boundary_fingerprint: str | None = None
    boundary_payload_bytes: int | None = None
    boundary_attempt_bytes: int | None = None
    boundary_budget_before: float | None = None
    boundary_budget_after: float | None = None
    boundary_ack_bytes: int | None = None
    terminal_before_boundary: bool = False
    terminal_at: int | None = None
    pre_action_digest: dict[str, Any] = field(default_factory=dict)
    forward_draws_at_fork: int | None = None
    forward_draws_after: int | None = None
    steps_executed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "at_step": self.at_step,
            "horizon": self.horizon,
            "target": self.target,
            "start_send_requested": self.start_send_requested,
            "start_send_ok": self.start_send_ok,
            "start_send_refused_reason": self.start_send_refused_reason,
            "start_send_bytes": self.start_send_bytes,
            "start_revision": self.start_revision,
            "start_fingerprint": self.start_fingerprint,
            "policy_requests_blocked": self.policy_requests_blocked,
            "emit_entry_blocked": self.emit_entry_blocked,
            "revisions_seen": list(self.revisions_seen),
            "coalesced_ready_updates": self.coalesced_ready_updates,
            "other_target_emissions": self.other_target_emissions,
            "other_target_bytes": self.other_target_bytes,
            "boundary_status": self.boundary_status,
            "boundary_revision": self.boundary_revision,
            "boundary_fingerprint": self.boundary_fingerprint,
            "boundary_payload_bytes": self.boundary_payload_bytes,
            "boundary_attempt_bytes": self.boundary_attempt_bytes,
            "boundary_budget_before": self.boundary_budget_before,
            "boundary_budget_after": self.boundary_budget_after,
            "boundary_ack_bytes": self.boundary_ack_bytes,
            "terminal_before_boundary": self.terminal_before_boundary,
            "terminal_at": self.terminal_at,
            "pre_action_digest": self.pre_action_digest,
            "forward_draws_at_fork": self.forward_draws_at_fork,
            "forward_draws_after": self.forward_draws_after,
            "steps_executed": self.steps_executed,
        }


class TargetWindow:
    def __init__(self, *, mode: str, at_step: int, target: str, horizon: int) -> None:
        if mode not in ("EARLY_FLUSH", "DEFER_FLUSH"):
            raise ValueError(mode)
        if int(horizon) < 1:
            raise ValueError(horizon)
        self.mode = mode
        self.at_step = int(at_step)
        self.target = str(target)
        self.horizon = int(horizon)
        self.stats = WindowStats(
            mode=self.mode, at_step=self.at_step, horizon=self.horizon, target=self.target
        )
        self._relay: Any | None = None
        self._original_emit: Any | None = None
        self._original_emit_selected: Any | None = None
        self._original_policy: Any | None = None
        self._original_tick: Any | None = None
        self._installed = False
        self._boundary_attempted = False
        self._start_done = False

    @property
    def boundary_step(self) -> int:
        return self.at_step + self.horizon

    @property
    def in_suppress_window(self) -> bool:
        """[t*, t*+H) exclusive of boundary flush step after start/defer semantics."""
        return False

    def _suppress(self, step: int) -> bool:
        # Suppress all target emissions for t* .. t*+H-1 inclusive of t* after EARLY's one send.
        return self.at_step <= step < self.boundary_step

    def wrap_relay(self, relay: Any) -> None:
        if self._installed:
            raise RuntimeError("already installed")
        self._relay = relay
        self._original_emit = relay._emit
        self._original_emit_selected = relay._emit_selected
        self._original_policy = relay.policy
        self._original_tick = relay.tick
        outer = self
        original_emit = self._original_emit

        def emit_with_window(item: SendItem, *, first: bool = False, reason: str = "policy") -> None:
            step = int(relay._tick)
            if str(item.target_id) != outer.target:
                result = original_emit(item, first=first, reason=reason)
                outer.stats.other_target_emissions += 1
                return result
            # track revisions
            rev = int(item.revision)
            if rev not in outer.stats.revisions_seen:
                outer.stats.revisions_seen.append(rev)
            if step == outer.at_step and not outer._start_done and outer.mode == "EARLY_FLUSH":
                outer.stats.start_send_requested = True
                before = int(relay.budget.spent_bytes)
                ack_before = int(relay.counters.get("ack_attempt_bytes", 0))
                result = original_emit(item, first=first, reason="other")
                after = int(relay.budget.spent_bytes)
                if after > before:
                    outer.stats.start_send_ok = True
                    outer.stats.start_send_bytes = after - before
                    outer.stats.start_revision = rev
                    try:
                        outer.stats.start_fingerprint = content_fingerprint(
                            relay._ledger[outer.target].last_sent
                        )
                    except Exception:
                        outer.stats.start_fingerprint = f"rev:{rev}"
                else:
                    outer.stats.start_send_ok = False
                    outer.stats.start_send_refused_reason = "budget_or_capacity"
                outer._start_done = True
                return result
            if step == outer.at_step and outer.mode == "DEFER_FLUSH":
                outer._start_done = True
                outer.stats.emit_entry_blocked += 1
                return None
            if outer._suppress(step):
                outer.stats.emit_entry_blocked += 1
                return None
            return original_emit(item, first=first, reason=reason)

        def emit_selected_with_window() -> None:
            step = int(relay._tick)
            if step == outer.at_step and not outer.stats.pre_action_digest:
                outer.stats.pre_action_digest = _digest(relay)
                outer.stats.forward_draws_at_fork = int(
                    relay.counters.get("transmission_attempts", 0)
                )
            if step == outer.boundary_step and not outer._boundary_attempted:
                # Boundary step: attempt the real latest-view flush first (or terminal),
                # then let the rest of the step run so delivery/ACK stay real.
                outer._force_boundary_flush()
            outer._original_emit_selected()

        def tick_stopping() -> tuple[str, ...]:
            changed = outer._original_tick()
            outer.stats.steps_executed = int(relay._tick)
            step = int(relay._tick)
            if step == outer.boundary_step and outer._boundary_attempted:
                outer.stats.forward_draws_after = int(
                    relay.counters.get("transmission_attempts", 0)
                )
                raise BoundaryComplete(outer.stats.as_dict())
            return changed

        def policy_wrap(inner: Any) -> Any:
            class W:
                name = f"target_window_{outer.mode}"
                orders_selection = bool(getattr(inner, "orders_selection", False))

                def parameters(self) -> dict[str, Any]:
                    v = dict(inner.parameters())
                    v.update(
                        {
                            "window_mode": outer.mode,
                            "window_at_step": outer.at_step,
                            "window_horizon": outer.horizon,
                            "window_target": outer.target,
                        }
                    )
                    return v

                def decide(self, observation: SendObservation) -> tuple[str, ...]:
                    step = int(observation.step)
                    chosen = list(inner.decide(observation))
                    if step == outer.at_step and outer.mode == "EARLY_FLUSH":
                        if outer.target not in chosen:
                            chosen.append(outer.target)
                        return tuple(dict.fromkeys(chosen))
                    if outer._suppress(step):
                        before = len(chosen)
                        chosen = [n for n in chosen if n != outer.target]
                        if before != len(chosen):
                            outer.stats.policy_requests_blocked += 1
                        return tuple(chosen)
                    return tuple(chosen)

            return W()

        relay._emit = emit_with_window
        relay._emit_selected = emit_selected_with_window
        relay.tick = tick_stopping
        relay.attach_policy(policy_wrap(self._original_policy))
        self._installed = True

    def _boundary_flush(self, item: SendItem, *, first: bool, reason: str) -> Any:
        if self._boundary_attempted:
            return None
        self._boundary_attempted = True
        return self._do_flush(item, first=first, reason="other")

    def _force_boundary_flush(self) -> None:
        if self._boundary_attempted or self._relay is None:
            return
        self._boundary_attempted = True
        relay = self._relay
        step = int(relay._tick)
        if step != self.boundary_step:
            self.stats.boundary_status = "implementation_error"
            return
        snapshot = relay._snapshot_of(self.target) if relay._snapshot_of else None
        if snapshot is not None and getattr(snapshot.status, "terminal", False):
            self.stats.boundary_status = "terminal_before_boundary"
            self.stats.terminal_before_boundary = True
            self.stats.terminal_at = step
            return
        try:
            observation = relay._observation()
        except Exception:
            self.stats.boundary_status = "missing_legal_view"
            return
        item = None
        for cand in observation.items:
            if str(cand.target_id) == self.target:
                item = cand
                break
        if item is None:
            self.stats.boundary_status = "missing_legal_view"
            return
        self._do_flush(item, first=False, reason="other")

    def _do_flush(self, item: SendItem, *, first: bool, reason: str) -> Any:
        relay = self._relay
        assert relay is not None
        try:
            ledger = relay._ledger.get(item.target_id)
            snapshot = relay._snapshot_of(item.target_id) if relay._snapshot_of else None
            message = relay._message(item.target_id, ledger, snapshot)
            self.stats.boundary_revision = int(message.revision)
            self.stats.boundary_fingerprint = content_fingerprint(message)
            from integrated_sar.comm import estimate_message_bytes

            payload = int(estimate_message_bytes(message))
            self.stats.boundary_payload_bytes = payload
            attempt = payload * max(1, len(relay._subscribers))
        except Exception:
            self.stats.boundary_status = "implementation_error"
            return None
        self.stats.boundary_budget_before = float(relay.budget.remaining_bytes)
        ack_before = int(relay.counters.get("ack_attempt_bytes", 0))
        spent_before = int(relay.budget.spent_bytes)
        self._original_emit(item, first=first, reason="other")
        spent_after = int(relay.budget.spent_bytes)
        self.stats.boundary_budget_after = float(relay.budget.remaining_bytes)
        self.stats.boundary_ack_bytes = int(relay.counters.get("ack_attempt_bytes", 0)) - ack_before
        if spent_after > spent_before:
            self.stats.boundary_status = "flushed_latest"
            self.stats.boundary_attempt_bytes = spent_after - spent_before
        else:
            blocked = relay.budget.why_blocked(attempt)
            if blocked == "capacity_step":
                self.stats.boundary_status = "refused_capacity"
            else:
                self.stats.boundary_status = "refused_credit"
            self.stats.boundary_attempt_bytes = 0
        return None

    def uninstall(self) -> None:
        if not self._installed or self._relay is None:
            return
        self._relay._emit = self._original_emit
        self._relay._emit_selected = self._original_emit_selected
        self._relay.tick = self._original_tick
        try:
            self._relay._policy = self._original_policy
        except Exception:
            pass
        self._installed = False


def _digest(relay: Any) -> dict[str, Any]:
    budget = relay.budget
    return {
        "tick": int(relay._tick),
        "budget_spent": int(budget.spent_bytes),
        "budget_remaining": float(budget.remaining_bytes),
        "pending_packets": len(getattr(relay, "_pending", []) or []),
        "ack_pending": len(getattr(relay, "_ack_pending", []) or []),
        "transmission_attempts": int(relay.counters.get("transmission_attempts", 0)),
        "ack_attempt_bytes": int(relay.counters.get("ack_attempt_bytes", 0)),
    }
