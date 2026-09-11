"""The single global clock shared by search, registry, coordination and execution."""

from __future__ import annotations


class GlobalClock:
    """
    Monotonic episode clock owned by :class:`~integrated_sar.orchestrator.JointSAROrchestrator`.

    Other components hold a reference instead of reading ``orchestrator.current_time``
    so that command validity and event timestamps are always judged against the same
    time base as the simulators.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._start = float(start)
        self._now = float(start)

    @property
    def now(self) -> float:
        return self._now

    def advance_to(self, value: float) -> None:
        value = float(value)
        if value < self._now:
            raise ValueError(f"global clock cannot move backwards: {self._now} -> {value}")
        self._now = value

    def reset(self, start: float | None = None) -> None:
        if start is not None:
            self._start = float(start)
        self._now = self._start

    def __float__(self) -> float:
        return self._now

    def __repr__(self) -> str:
        return f"GlobalClock(now={self._now})"
