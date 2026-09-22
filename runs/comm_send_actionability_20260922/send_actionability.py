"""Shared SEND/HOLD actionability predicate and mutually exclusive classification.

The predicate calls production ``estimate_message_bytes`` and ``CommBudget.can_send`` /
``why_blocked`` so it cannot drift from the executor's arithmetic. Classification is a
pure function of the actionability record fields (no outcome columns).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from integrated_sar.comm import estimate_message_bytes
from integrated_sar.comm_ack import content_fingerprint

MAX_ATTEMPTS = 5  # event_retry_until_acked_5, frozen in Stage1 FROZEN_CONFIG


@dataclass
class ActionabilityRow:
    seed: int
    step: int
    communication_substep: int
    target_id: str
    semantic_revision: int
    content_fingerprint: str
    subscriber_set: tuple[str, ...]
    # legality / billing
    legal: bool
    terminal: bool
    subscriber_count: int
    payload_bytes: int
    attempt_charge_bytes: int
    # budget / capacity
    budget_spent: int
    budget_remaining: float
    budget_bytes_per_episode: int | None
    sends_this_step: int
    capacity_remaining: int | None
    charge_over_remaining: float | None
    # history / class
    sent_count: int
    attempts_remaining: int
    message_class: str  # first | update | same_view_retry
    # content deltas
    changed_view: bool
    first_emission: bool
    pending_event: bool
    stage_changed: bool
    requirements_changed: bool
    members_changed: bool
    progress_delta: float
    age_steps: int
    remaining_time: float
    # ACK (explicit unknown = -1)
    ack_confirmed_fraction: float
    ack_unacked_broadcasts: int
    # actions
    rule_requested: bool
    send_executable: bool
    hold_executable: bool
    send_block_reason: str | None
    hold_block_reason: str | None
    protocol_forced: bool
    in_policy_interface: bool
    primary_class: str = ""
    secondary_reasons: list[str] = field(default_factory=list)

    def dedup_key(self) -> tuple:
        return (
            self.seed,
            self.step,
            self.communication_substep,
            self.target_id,
            self.semantic_revision,
            self.content_fingerprint,
            self.subscriber_set,
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["subscriber_set"] = list(self.subscriber_set)
        return data


def classify(row: ActionabilityRow) -> tuple[str, list[str]]:
    """Priority 1-8 mutually exclusive primary class plus secondary reasons."""
    secondary: list[str] = []
    if not row.in_policy_interface:
        return "not_policy_decision", secondary
    if row.protocol_forced:
        secondary.append("protocol_forced")
        return "hold_not_free_protocol_forced", secondary
    if row.terminal or not row.legal:
        return "send_blocked_content_or_terminal", secondary
    if not row.send_executable:
        reason = row.send_block_reason or "unknown"
        secondary.append(reason)
        if reason == "budget_exhausted":
            return "send_blocked_credit", secondary
        if reason == "capacity_step":
            return "send_blocked_capacity_or_queue", secondary
        return "other", secondary
    if row.attempts_remaining <= 0 and row.message_class == "same_view_retry":
        secondary.append("retry_exhausted")
        return "send_blocked_retry_exhausted", secondary
    if row.send_executable and row.hold_executable:
        return "both_actions_free", secondary
    if not row.hold_executable:
        return "hold_not_free_protocol_forced", secondary
    secondary.append("send_ok_hold_blocked_unexplained")
    return "other", secondary


def message_class_of(*, first_emission: bool, pending_event: bool, changed_view: bool) -> str:
    if first_emission or pending_event:
        return "first"
    if changed_view:
        return "update"
    return "same_view_retry"


def attempt_charge_bytes(message: Any, subscribers: Sequence[str] | Iterable[str]) -> int:
    """Identical to CommRelay._emit: payload estimate times fan-out."""
    return int(estimate_message_bytes(message) * len(list(subscribers)))


def send_block_from_budget(budget: Any, cost_bytes: int) -> str | None:
    """Same decision as CommRelay._emit's pre-charge check."""
    if budget.can_send(cost_bytes):
        return None
    return budget.why_blocked(cost_bytes)


def classify_closed(rows: Sequence[ActionabilityRow]) -> dict[str, int]:
    counts = {
        "both_actions_free": 0,
        "send_blocked_credit": 0,
        "send_blocked_capacity_or_queue": 0,
        "send_blocked_content_or_terminal": 0,
        "send_blocked_retry_exhausted": 0,
        "hold_not_free_protocol_forced": 0,
        "not_policy_decision": 0,
        "other": 0,
    }
    for row in rows:
        primary, secondary = classify(row)
        row.primary_class = primary
        row.secondary_reasons = secondary
        counts[primary] = counts.get(primary, 0) + 1
    return counts


def identity_ok(rows: Sequence[ActionabilityRow], counts: dict[str, int]) -> bool:
    return sum(counts.values()) == len(rows)


def fingerprint_of_message(message: Any) -> str:
    try:
        return content_fingerprint(message)
    except Exception:
        return f"rev:{getattr(message, 'revision', -1)}"
