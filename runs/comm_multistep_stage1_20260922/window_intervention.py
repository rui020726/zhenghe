"""S(H) / H(H) window interventions at the actual communication send entry.

The wrapper never edits production ``comm.py``. It is installed on a live relay before the
episode starts and removed in a ``finally`` when a harness is done with a synthetic case.

View identity is ``(target_id, revision, content_fingerprint)``. A new revision or a new
content fingerprint truncates the window; the new view is free to send under the baseline
rule. Window-blocked requests are counted separately from budget/capacity refusals and are
not billed, do not consume retries, and do not increment ``sent_count``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from integrated_sar.comm_ack import content_fingerprint
from integrated_sar.comm_policy import SendItem, SendObservation


@dataclass
class WindowStats:
    mode: str
    at_step: int
    horizon: int
    target: str
    locked_revision: int | None = None
    locked_fingerprint: str | None = None
    forced_sends: int = 0
    window_blocked_requests: int = 0
    hard_refusals: int = 0
    window_truncated: bool = False
    window_truncated_at: int | None = None
    window_truncated_reason: str | None = None
    effective_window_steps: int | None = None
    next_same_view_send_step: int | None = None
    next_revision_send_step: int | None = None
    force_requested: bool = False
    force_applied: bool = False
    force_refused: bool = False
    pre_action_digest: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "at_step": self.at_step,
            "horizon": self.horizon,
            "target": self.target,
            "locked_revision": self.locked_revision,
            "locked_fingerprint": self.locked_fingerprint,
            "forced_sends": self.forced_sends,
            "window_blocked_requests": self.window_blocked_requests,
            "hard_refusals": self.hard_refusals,
            "window_truncated": self.window_truncated,
            "window_truncated_at": self.window_truncated_at,
            "window_truncated_reason": self.window_truncated_reason,
            "effective_window_steps": self.effective_window_steps,
            "next_same_view_send_step": self.next_same_view_send_step,
            "next_revision_send_step": self.next_revision_send_step,
            "force_requested": self.force_requested,
            "force_applied": self.force_applied,
            "force_refused": self.force_refused,
            "pre_action_digest": self.pre_action_digest,
        }


class WindowIntervention:
    """Force one send (S) or suppress a view for H steps (H) around ``at_step``."""

    def __init__(
        self,
        *,
        mode: str,
        at_step: int,
        target: str,
        horizon: int,
    ) -> None:
        if mode not in ("S", "H"):
            raise ValueError(f"mode must be S or H, got {mode!r}")
        if int(horizon) < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.mode = str(mode)
        self.at_step = int(at_step)
        self.target = str(target)
        self.horizon = int(horizon)
        self.stats = WindowStats(
            mode=self.mode,
            at_step=self.at_step,
            horizon=self.horizon,
            target=self.target,
        )
        self._relay: Any | None = None
        self._original_emit: Any | None = None
        self._original_emit_selected: Any | None = None
        self._original_policy: Any | None = None
        self._installed = False

    # ------------------------------------------------------------------ view identity
    def _view_key(self, item: SendItem) -> tuple[str, int]:
        return (str(item.target_id), int(item.revision))

    def _lock_from_item(self, item: SendItem) -> None:
        """Lock the intervened view only at t*, never from an earlier revision."""
        if self.stats.locked_revision is None:
            self.stats.locked_revision = int(item.revision)

    def _maybe_truncate(self, item: SendItem, step: int) -> None:
        if self.stats.window_truncated or self.stats.locked_revision is None:
            return
        if int(step) < self.at_step:
            return
        if self._view_key(item) != (self.target, int(self.stats.locked_revision)):
            self.stats.window_truncated = True
            self.stats.window_truncated_at = int(step)
            self.stats.window_truncated_reason = "window_truncated_by_revision"
            self.stats.effective_window_steps = max(0, int(step) - self.at_step)

    def _same_locked_view(self, item: SendItem) -> bool:
        return (
            str(item.target_id) == self.target
            and self.stats.locked_revision is not None
            and int(item.revision) == int(self.stats.locked_revision)
        )

    def in_window(self, step: int) -> bool:
        if self.stats.window_truncated:
            return False
        if self.stats.locked_revision is None:
            return False
        return self.at_step <= int(step) < self.at_step + self.horizon

    # ------------------------------------------------------------------ policy wrapper
    def wrap_policy(self, inner: Any) -> Any:
        outer = self

        class WindowPolicy:
            name = f"window_{outer.mode}{outer.horizon}"
            orders_selection = bool(getattr(inner, "orders_selection", False))

            def parameters(self) -> dict[str, Any]:
                values = dict(inner.parameters())
                values.update(
                    {
                        "window_mode": outer.mode,
                        "window_at_step": outer.at_step,
                        "window_horizon": outer.horizon,
                        "window_target": outer.target,
                    }
                )
                return values

            def decide(self, observation: SendObservation) -> tuple[str, ...]:
                step = int(observation.step)
                items = {str(item.target_id): item for item in observation.items}
                target_item = items.get(outer.target)
                if target_item is not None and step >= outer.at_step:
                    outer._lock_from_item(target_item)
                    outer._maybe_truncate(target_item, step)
                chosen = list(inner.decide(observation))
                if target_item is None:
                    return tuple(chosen)
                if step == outer.at_step and not outer.stats.window_truncated:
                    outer._lock_from_item(target_item)
                    if outer.mode == "S":
                        # Request one send of the locked view. The emit gate still applies
                        # budget, capacity and terminal legality.
                        outer.stats.force_requested = True
                        if outer.target not in chosen:
                            chosen.append(outer.target)
                        return tuple(dict.fromkeys(chosen))
                if outer.in_window(step) and outer._same_locked_view(target_item):
                    before = len(chosen)
                    chosen = [name for name in chosen if name != outer.target]
                    if before != len(chosen):
                        outer.stats.window_blocked_requests += 1
                    return tuple(chosen)
                return tuple(chosen)

        return WindowPolicy()

    # ------------------------------------------------------------------ emit backstop
    def wrap_relay(self, relay: Any) -> None:
        if self._installed:
            raise RuntimeError("window already installed on a relay")
        self._relay = relay
        self._original_emit = relay._emit
        self._original_emit_selected = relay._emit_selected
        self._original_policy = relay.policy
        outer = self
        original_emit = self._original_emit

        def emit_with_window(
            item: SendItem, *, first: bool = False, reason: str = "policy"
        ) -> None:
            step = int(relay._tick)
            if str(item.target_id) == outer.target:
                if step >= outer.at_step:
                    outer._lock_from_item(item)
                    outer._maybe_truncate(item, step)
                if (
                    outer.mode == "S"
                    and step == outer.at_step
                    and outer.stats.forced_sends == 0
                    and not outer.stats.window_truncated
                ):
                    # The one forced emission of the locked view. Still goes through the
                    # original emit so budget/capacity/billing stay identical.
                    result = original_emit(item, first=first, reason="other")
                    if outer._did_emit(item):
                        outer.stats.forced_sends += 1
                        outer.stats.force_applied = True
                        if outer.stats.locked_fingerprint is None:
                            outer.stats.locked_fingerprint = outer._fingerprint_of(item)
                        if outer.stats.next_same_view_send_step is None:
                            outer.stats.next_same_view_send_step = step
                    else:
                        outer.stats.force_refused = True
                    return result
                if outer.in_window(step) and outer._same_locked_view(item):
                    # Hard backstop: no bill, no retry consumption, no sent_count bump.
                    outer.stats.window_blocked_requests += 1
                    return None
                if (
                    outer.stats.next_same_view_send_step is None
                    and step > outer.at_step
                    and outer._same_locked_view(item)
                    and outer.stats.forced_sends > 0
                ):
                    outer.stats.next_same_view_send_step = step
                if (
                    outer.stats.next_revision_send_step is None
                    and step > outer.at_step
                    and str(item.target_id) == outer.target
                    and not outer._same_locked_view(item)
                ):
                    outer.stats.next_revision_send_step = step
            return original_emit(item, first=first, reason=reason)

        def emit_selected_with_window() -> None:
            # Pre-action digest is taken before any send decision at t*.
            step = int(relay._tick) + 1  # tick() increments then calls _emit_selected
            # relay._tick is already incremented inside tick() before _emit_selected.
            current = int(relay._tick)
            if current == outer.at_step and not outer.stats.pre_action_digest:
                outer.stats.pre_action_digest = snapshot_relay_state(relay)
            return outer._original_emit_selected()

        relay._emit = emit_with_window
        relay._emit_selected = emit_selected_with_window
        relay.attach_policy(self.wrap_policy(self._original_policy))
        self._installed = True

    def _did_emit(self, item: SendItem) -> bool:
        ledger = self._relay._ledger.get(item.target_id) if self._relay is not None else None
        return ledger is not None and ledger.last_sent_step == int(self._relay._tick)

    def _fingerprint_of(self, item: SendItem) -> str | None:
        if self._relay is None:
            return None
        ledger = self._relay._ledger.get(item.target_id)
        if ledger is None or ledger.last_sent is None:
            return None
        try:
            return content_fingerprint(ledger.last_sent)
        except Exception:
            return None

    def uninstall(self) -> None:
        if not self._installed or self._relay is None:
            return
        self._relay._emit = self._original_emit
        self._relay._emit_selected = self._original_emit_selected
        # Policy may only be swapped before the first step; after a run the relay is done.
        # For synthetic tests we restore the original policy object for cleanliness.
        try:
            self._relay._policy = self._original_policy
        except Exception:
            pass
        self._installed = False

    def finalize(self, *, episode_end_step: int | None = None) -> dict[str, Any]:
        if (
            not self.stats.window_truncated
            and self.stats.effective_window_steps is None
        ):
            self.stats.effective_window_steps = self.horizon
        if episode_end_step is not None and self.stats.next_same_view_send_step is None:
            # Distinguish never-sent from truncated/ended: leave as None and report as never.
            pass
        return self.stats.as_dict()


def snapshot_relay_state(relay: Any) -> dict[str, Any]:
    """Canonical pre-action digest: no object addresses, no log paths."""
    ledger_rows = []
    for target in sorted(getattr(relay, "_ledger", {}) or {}):
        entry = relay._ledger[target]
        last = entry.last_sent
        last_view = None
        if last is not None:
            try:
                last_view = content_fingerprint(last)
            except Exception:
                last_view = f"rev:{int(getattr(last, 'revision', -1))}"
        ledger_rows.append(
            {
                "target": str(target),
                "ready_step": entry.ready_step,
                "sent_count": int(entry.sent_count),
                "last_sent_step": entry.last_sent_step,
                "last_revision": int(getattr(last, "revision", -1)) if last is not None else None,
                "last_view": last_view,
                "pending_event": bool(entry.pending_event),
                "closed": bool(getattr(entry, "closed", False)),
            }
        )
    budget = relay.budget
    ack_states = {}
    if getattr(relay, "_ack_enabled", False):
        for key, state in sorted((relay.ack.states or {}).items()):
            ack_states[str(key)] = {
                "highest": list(state.highest) if state.highest else None,
                "last_arrival_step": state.last_arrival_step,
                "reaffirmations": int(state.reaffirmations),
            }
    counters = {
        key: int(value)
        for key, value in sorted((relay.counters or {}).items())
        if isinstance(value, (int, float))
    }
    return {
        "tick": int(relay._tick),
        "sequence": int(getattr(relay, "_sequence", 0)),
        "pending_packets": int(len(getattr(relay, "_pending", []) or [])),
        "ack_pending": int(len(getattr(relay, "_ack_pending", []) or [])),
        "ack_ordinal": int(getattr(relay, "_ack_ordinal", 0)),
        "budget_spent": int(budget.spent_bytes),
        "budget_remaining": (
            None if budget.bytes_per_episode is None else int(budget.remaining_bytes)
        ),
        "budget_sends_this_step": int(getattr(budget, "sends_this_step", 0)),
        "forward_rng_draws": int(
            getattr(getattr(relay, "_rng", None), "_bit_generator", None) is not None
            and _rng_draw_estimate(relay)
            or 0
        ),
        "ledger": ledger_rows,
        "ack_states": ack_states,
        "counters": counters,
    }


def _rng_draw_estimate(relay: Any) -> int:
    """Numpy Generator does not expose a draw counter; use the relay's own attempt count.

    ``transmission_attempts`` is the number of per-subscriber uplink draws this relay has
    taken, which is the forward sequential RNG consumption under the production emit path.
    """
    return int((relay.counters or {}).get("transmission_attempts", 0))


def digests_equal(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, list[str]]:
    """Compare two pre-action digests field by field."""
    problems: list[str] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if type(a) is not type(b) and not (
            isinstance(a, (int, float)) and isinstance(b, (int, float))
        ):
            problems.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
            return
        if isinstance(a, dict):
            keys = set(a) | set(b)
            for key in sorted(keys):
                if key not in a:
                    problems.append(f"{path}.{key}: missing on left")
                elif key not in b:
                    problems.append(f"{path}.{key}: missing on right")
                else:
                    walk(a[key], b[key], f"{path}.{key}")
            return
        if isinstance(a, list):
            if len(a) != len(b):
                problems.append(f"{path}: length {len(a)} vs {len(b)}")
                return
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{index}]")
            return
        if a != b:
            problems.append(f"{path}: {a!r} vs {b!r}")

    walk(left, right, "")
    return (not problems, problems)
