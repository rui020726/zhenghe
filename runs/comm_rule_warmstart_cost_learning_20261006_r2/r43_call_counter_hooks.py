"""Identity-preserving wrappers for the actual PPO and Adam call boundaries."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable


COUNTER_HOOK_VERSION = "round43-training-call-counter/2"


def count_at_entry(original: Callable[..., Any], record_entry: Callable[..., Any]):
    """Record one invocation before delegating, preserving args/results/exceptions."""
    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        record_entry(*args, **kwargs)
        return original(*args, **kwargs)

    return wrapped
