"""Two pure-communication budget-pacing rules over the frozen 19-dim candidate vector.

Both rules read only indices 5, 11, 13, 15 and 18 of the vector produced by the frozen
``integrated_sar.comm_learning.features`` function, so a rule and the network see the same
numbers. Candidate order is preserved: ``orders_selection`` is False and the transport trims
the selected set exactly as it does for every other rule.

The budget term is the tick-start balance shared by all candidates of that tick. A send may
push the real balance below the floor inside the same tick; the hard budget remains the only
thing that stops traffic, and nothing here claims the budget will be spent in full.
"""

from __future__ import annotations

from . import constants as C
from .paths import ensure_repo_path


class _FrozenFeatures:
    """Lazy reader of the frozen feature function; the package is never imported at module load."""

    def __init__(self):
        self._function = None

    def vector(self, observation, item):
        if self._function is None:
            # Resolve the frozen repository root from this file, never from the process cwd.
            ensure_repo_path()
            from integrated_sar.comm_learning import FEATURE_NAMES, features
            if tuple(FEATURE_NAMES) != tuple(C.FEATURE_NAMES):
                raise RuntimeError("frozen feature layout moved: %r" % (tuple(FEATURE_NAMES),))
            self._function = features
        return self._function(observation, item, horizon=float(C.STEPS))


_READER = _FrozenFeatures()


def _fields(vector):
    """Extract the five allowed dims by their frozen positions and nothing else."""
    if len(vector) != C.FEATURE_DIM:
        raise RuntimeError("candidate vector has %d dims, expected %d"
                           % (len(vector), C.FEATURE_DIM))
    return {
        "first_emission": float(vector[C.IDX_FIRST_EMISSION]),
        "remaining_time": float(vector[C.IDX_REMAINING_TIME]),
        "has_future": float(vector[C.IDX_HAS_FUTURE]),
        "budget_fraction": float(vector[C.IDX_BUDGET_FRACTION]),
        "tau": float(vector[C.IDX_STEP_FRACTION]),
    }


def admit_vec(vector, priority):
    """Pure decision for one candidate vector; identical logic to ``decide``."""
    fields = _fields(vector)
    if fields["first_emission"] != 1.0:
        return False
    floor = C.budget_floor(fields["tau"])
    if fields["budget_fraction"] < floor:
        return False
    if not priority:
        return True
    if fields["tau"] >= C.PRIORITY_TAIL_CUTOFF:
        return True
    if fields["budget_fraction"] >= floor + C.PRIORITY_BAND:
        return True
    return fields["has_future"] == 0.0 and fields["remaining_time"] > C.PRIORITY_MIN_REMAINING


class _PacingBase:
    """Shared plumbing; subclasses only choose whether the priority filter is active."""

    name = "pacing_abstract"
    orders_selection = False
    priority = False

    def parameters(self):
        return {
            "name": self.name,
            "budget_floors": [list(pair) for pair in C.BUDGET_FLOORS],
            "priority_band": C.PRIORITY_BAND,
            "priority_tail_cutoff": C.PRIORITY_TAIL_CUTOFF,
            "priority_min_remaining": C.PRIORITY_MIN_REMAINING,
            "features_used": [C.IDX_FIRST_EMISSION, C.IDX_REMAINING_TIME, C.IDX_HAS_FUTURE,
                              C.IDX_BUDGET_FRACTION, C.IDX_STEP_FRACTION],
            "orders_selection": False,
            "priority": self.priority,
        }

    def decide(self, observation):
        chosen = []
        for item in observation.items:
            vector = _READER.vector(observation, item)
            if admit_vec(vector, self.priority):
                chosen.append(item.target_id)
        return tuple(chosen)


class WindowedPacingPolicy(_PacingBase):
    """First reports only, released by the three-window budget ladder."""

    name = "comm_windowed_pacing"
    priority = False


class PriorityPacingPolicy(_PacingBase):
    """Windowed pacing plus a scarcity filter while the balance sits near the floor."""

    name = "comm_priority_pacing"
    priority = True


def build_pacing_policy(name):
    if name == "comm_windowed_pacing":
        return WindowedPacingPolicy()
    if name == "comm_priority_pacing":
        return PriorityPacingPolicy()
    raise ValueError("unknown pacing arm %r" % (name,))

