"""Dimensionless rewrite of the three scale-carrying dims, for arbitrary (G, R, N).

The 19-dim ``comm-send-state/1`` layout does not change. Three of its dims carry the size of
the world, and leaving them raw would make the same policy see "24 pending targets" and "48
pending targets" as different situations when they are the same situation at a different
scale:

  #2  sent_count             -> sent_count / G
  #14 queue_backlog_targets  -> backlog / G
  #17 subscribers            -> N / 6

Everything else is untouched, including #11. Earlier notes proposed unclipping #11, but the
task's registered change list names exactly these three dims; changing a fourth would make
this revision's rows incomparable to every archived one without buying anything the task asked
for.

Where the rewrite hooks in: ``controller.inner.trunk``. Three places read the row --
the sampler's ``controller.forward``, the cost critic's ``controller.cost_values``, and
``trainer.build_samples`` re-reading ``tape["features"]`` -- and only the first goes through
``forward``. Wrapping the trunk covers all three with one rewrite, which is what makes the
importance ratio at update 0 exactly 1 instead of the 1.589 it was when P4 shadowed ``forward``
and the update saw raw rows. ``tests/test_revision_02_gates.py`` re-proves this.

``install`` writes the shadow into the instance ``__dict__`` and restores by type: a plain
class (``train_pilot.Controller``) gets the real object written back, an ``nn.Module`` gets the
key popped so lookup falls through to ``_modules``. Getting that branch wrong deletes the
trunk outright, which is not a hypothetical -- the first P4 smoke died on it.
"""

from __future__ import annotations

import math

import torch

SUBSCRIBERS_REF = 6
INDEX_SENT_COUNT = 2
INDEX_BACKLOG = 14
INDEX_SUBSCRIBERS = 17
SCALED_INDICES = (INDEX_SENT_COUNT, INDEX_BACKLOG, INDEX_SUBSCRIBERS)
MODES = ("off", "scaled")


class _RunningDim:
    """Streaming min/max/mean/sd for one column, so 1000 steps x N candidates stay cheap."""

    __slots__ = ("count", "total", "total_sq", "low", "high")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.low = math.inf
        self.high = -math.inf

    def add(self, values) -> None:
        for value in values:
            value = float(value)
            self.count += 1
            self.total += value
            self.total_sq += value * value
            if value < self.low:
                self.low = value
            if value > self.high:
                self.high = value

    def summary(self) -> dict:
        if self.count == 0:
            return {"count": 0}
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {"count": self.count, "min": self.low, "max": self.high,
                "mean": mean, "sd": math.sqrt(variance)}


class ScaleFeatures:
    """Rewrite #2, #14 and #17 so one policy reads every cell the same way."""

    def __init__(self, *, mode: str = "scaled", goals: int = 24,
                 subscribers: int = 6) -> None:
        if mode not in MODES:
            raise ValueError("SCALE_FEATURES must be one of %r, got %r" % (MODES, mode))
        goals = int(goals)
        subscribers = int(subscribers)
        if goals < 1:
            raise ValueError("goals must be >= 1 for the #2/#14 divisor, got %d" % goals)
        if subscribers < 1:
            raise ValueError("subscribers must be >= 1 for #17, got %d" % subscribers)
        self.mode = mode
        self.goals = goals
        self.subscribers = subscribers
        self.original = {index: _RunningDim() for index in SCALED_INDICES}
        self.rewritten = {index: _RunningDim() for index in SCALED_INDICES}
        self.matrix_calls = 0
        self.changed = {index: 0 for index in SCALED_INDICES}
        self._restore = None

    # ---------------------------------------------------------------- the transform
    def rewrite(self, matrix):
        """Return what the trunk should see. ``off`` returns the caller's own object."""
        import numpy as np

        array = np.asarray(matrix, dtype="float32")
        if array.ndim != 2 or array.shape[1] <= INDEX_SUBSCRIBERS:
            raise RuntimeError("expected a [candidates, 19] matrix, got %r" % (array.shape,))
        self.matrix_calls += 1
        for index in SCALED_INDICES:
            self.original[index].add(array[:, index])
        if self.mode == "off":
            # Identity on purpose, and provably so: the same accounting runs, nothing is
            # written back, and the caller's object is returned untouched.
            for index in SCALED_INDICES:
                self.rewritten[index].add(array[:, index])
            return matrix

        out = array.copy()
        out[:, INDEX_SENT_COUNT] = out[:, INDEX_SENT_COUNT] / float(self.goals)
        out[:, INDEX_BACKLOG] = out[:, INDEX_BACKLOG] / float(self.goals)
        out[:, INDEX_SUBSCRIBERS] = out[:, INDEX_SUBSCRIBERS] / float(SUBSCRIBERS_REF)

        for index in SCALED_INDICES:
            before, after = array[:, index], out[:, index]
            self.rewritten[index].add(after)
            for old_value, new_value in zip(before, after):
                if float(old_value) != float(new_value):
                    self.changed[index] += 1
        # The trunk is a torch module: the array must go back as a tensor of the same dtype
        # the frozen collector produced, or ``Linear`` raises on a numpy input.
        return torch.as_tensor(out)

    # ---------------------------------------------------------------- install / restore
    def install(self, controller):
        """Wrap ``controller.inner.trunk``; returns the original so the caller can restore."""
        inner = getattr(controller, "inner", None)
        trunk = getattr(inner, "trunk", None)
        if trunk is None:
            raise AttributeError("controller has no inner.trunk to wrap")
        original = trunk
        shadowed_module = isinstance(inner, torch.nn.Module)

        def tapped(batch, *args, **kwargs):
            return original(self.rewrite(batch), *args, **kwargs)

        object.__setattr__(inner, "trunk", tapped)
        self._restore = (inner, original, shadowed_module)
        return original

    @staticmethod
    def restore(installed):
        """Undo :meth:`install` using what it returned. Idempotent."""
        target = installed.get("scale_features")
        if target is None or target._restore is None:
            return
        inner, original, shadowed_module = target._restore
        if shadowed_module:
            inner.__dict__.pop("trunk", None)
        else:
            object.__setattr__(inner, "trunk", original)
        target._restore = None

    # ---------------------------------------------------------------- evidence
    def audit(self) -> dict:
        return {
            "mode": self.mode,
            "goals": self.goals,
            "subscribers": self.subscribers,
            "matrix_calls": self.matrix_calls,
            "divisors": {"#2 sent_count": self.goals, "#14 backlog": self.goals,
                         "#17 subscribers": SUBSCRIBERS_REF},
            "dims": {("dim_%d" % index): {
                "original": self.original[index].summary(),
                "rewritten": self.rewritten[index].summary(),
                "values_changed": self.changed[index],
            } for index in SCALED_INDICES},
        }
