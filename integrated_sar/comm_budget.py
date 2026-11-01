"""The transmission budget the uplink has to live inside.

One executor, shared by every policy including the learned one: a byte credit for the episode
plus a per-step emission capacity. Nothing about it is policy-specific -- same initial credit,
same cost model, same merge and queue rules, same deadline for the episode.

What a packet costs is decided here and nowhere else:

* cost = the payload estimate of the emitted view, multiplied by the number of subscribed
  robots the sender addressed, **counted whether or not the packet survived**;
* a lost packet is billed in full. Radio time spent on a packet the receiver never got is
  exactly the cost a deployment pays, so an arm that "saves" traffic by dropping it is
  not saving anything;
* a send that cannot be paid for does not happen. The credit is a hard constraint: it is
  never driven negative, never waived for an urgent task, and every refusal is counted so a
  reader can see how much urgency was turned away by arithmetic.
"""

from __future__ import annotations

from typing import Any


class CommBudget:
    """A byte credit plus a per-step emission capacity, reset once per episode."""

    def __init__(
        self,
        *,
        bytes_per_episode: int | None = None,
        sends_per_step: int | None = None,
    ) -> None:
        if bytes_per_episode is not None and int(bytes_per_episode) < 1:
            raise ValueError(
                f"bytes_per_episode must be >= 1 or None (unbounded), got {bytes_per_episode}"
            )
        if sends_per_step is not None and int(sends_per_step) < 1:
            raise ValueError(
                f"sends_per_step must be >= 1 or None (unbounded), got {sends_per_step}"
            )
        self.bytes_per_episode = None if bytes_per_episode is None else int(bytes_per_episode)
        self.sends_per_step = None if sends_per_step is None else int(sends_per_step)
        self.spent_bytes = 0
        self.attempts = 0
        self.sends_this_step = 0
        self.counters: dict[str, int] = {
            "budget_refused_sends": 0,
            "budget_refused_bytes": 0,
            "capacity_refused_sends": 0,
        }

    @property
    def unbounded(self) -> bool:
        return self.bytes_per_episode is None and self.sends_per_step is None

    @property
    def remaining_bytes(self) -> float:
        if self.bytes_per_episode is None:
            return float("inf")
        return float(self.bytes_per_episode - self.spent_bytes)

    def budgeted(self) -> bool:
        return self.bytes_per_episode is not None

    def begin_step(self) -> None:
        self.sends_this_step = 0

    def can_send(self, cost_bytes: int) -> bool:
        """Whether one more emission to the whole subscriber set fits right now."""
        if self.sends_per_step is not None and self.sends_this_step >= self.sends_per_step:
            return False
        if self.bytes_per_episode is not None and self.spent_bytes + cost_bytes > self.bytes_per_episode:
            return False
        return True

    def why_blocked(self, cost_bytes: int) -> str:
        if self.sends_per_step is not None and self.sends_this_step >= self.sends_per_step:
            return "capacity_step"
        if self.bytes_per_episode is not None and self.spent_bytes + cost_bytes > self.bytes_per_episode:
            return "budget_exhausted"
        return "allowed"

    def charge(self, cost_bytes: int) -> None:
        if not self.can_send(cost_bytes):
            # Refusing here rather than going negative is the point of the class: a forced or
            # urgent send is billed like any other, and if it cannot be paid for it does not
            # happen.
            raise RuntimeError(
                f"budget violation: {self.counters} cannot charge {cost_bytes} bytes "
                f"against {self.bytes_per_episode}"
            )
        self.spent_bytes += int(cost_bytes)
        self.attempts += 1
        self.sends_this_step += 1

    def note_refusal(self, reason: str, cost_bytes: int) -> None:
        if reason == "capacity_step":
            self.counters["capacity_refused_sends"] += 1
        elif reason == "budget_exhausted":
            self.counters["budget_refused_sends"] += 1
            self.counters["budget_refused_bytes"] += int(cost_bytes)

    def usage(self) -> dict[str, Any]:
        return {
            "bytes_per_episode": self.bytes_per_episode,
            "sends_per_step": self.sends_per_step,
            "spent_bytes": self.spent_bytes,
            "remaining_bytes": None if self.bytes_per_episode is None else self.remaining_bytes,
            "used_fraction": (
                None
                if self.bytes_per_episode is None
                else round(self.spent_bytes / self.bytes_per_episode, 5)
            ),
            **self.counters,
        }

    def reset(self) -> None:
        """New episode: full credit again, same declared capacity."""
        self.spent_bytes = 0
        self.attempts = 0
        self.sends_this_step = 0
        for key in self.counters:
            self.counters[key] = 0
