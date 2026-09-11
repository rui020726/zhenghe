"""A deterministic synchronous event bus with fan-out and idempotent consumers."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Callable, Deque, TypeVar


EventT = TypeVar("EventT")
EventHandler = Callable[[object], None]


class InMemoryEventBus:
    def __init__(self, max_history: int = 1_000) -> None:
        self._subscriptions: dict[type, list[EventHandler]] = defaultdict(list)
        self._queue: Deque[object] = deque()
        self._dispatching = False
        self._max_history = max(0, int(max_history))
        self._history: Deque[object] = deque(maxlen=self._max_history or None)

    def subscribe(self, event_type: type[EventT], handler: Callable[[EventT], None]) -> None:
        handlers = self._subscriptions[event_type]
        if handler not in handlers:
            handlers.append(handler)  # type: ignore[arg-type]

    def unsubscribe(self, event_type: type[EventT], handler: Callable[[EventT], None]) -> None:
        handlers = self._subscriptions.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)  # type: ignore[arg-type]

    def publish(self, event: object) -> None:
        """Publish once; matching subscribers receive the same immutable event."""
        self._queue.append(event)
        if self._dispatching:
            return

        self._dispatching = True
        try:
            while self._queue:
                current = self._queue.popleft()
                if self._max_history:
                    self._history.append(current)
                for event_type, handlers in tuple(self._subscriptions.items()):
                    if isinstance(current, event_type):
                        for handler in tuple(handlers):
                            handler(current)
        finally:
            self._dispatching = False

    @property
    def history(self) -> tuple[object, ...]:
        return tuple(self._history)

    def clear_history(self) -> None:
        self._history.clear()
