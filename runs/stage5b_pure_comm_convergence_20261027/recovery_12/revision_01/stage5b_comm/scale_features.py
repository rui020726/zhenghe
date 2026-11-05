"""Shadow feature transform for the arbitrary-N probe (P3).

Two dimensions of the 19-dim ``comm-send-state/1`` row carry the agent count or scale
with it, and both are rewritten here:

  #17 subscribers            -> n / 6          (identity at n=6, where the dim is always 6.0)
  #14 queue_backlog_targets  -> backlog / 24   (NOT identity at n=6: this is a real change)

``#2 sent_count`` is left alone: it is a live decision input, and rescaling it would make
every archived checkpoint's on-time rate uninterpretable.

Where this hooks in, and why there rather than in a feature function: the pinned collector
computes its rows with ``features(observation, item)`` inside ``budget_masked.py`` and hands
the resulting matrix straight to ``self.controller.forward(matrix)``. Both files are in the
99-file pin list, so neither may be edited. The controller object is ours, however, and
``forward`` is looked up on the instance at call time, so shadowing it with an instance
attribute is enough to change exactly what the network sees -- and enough to put it back.

The install/restore shape mirrors ``causal_credit.py``: capture the bound callable, replace
it, and return the original so the caller can restore it on the single finally path.
"""

from __future__ import annotations

import math

SUBSCRIBERS_REF = 6
NUM_GOALS = 24
INDEX_SUBSCRIBERS = 17
INDEX_BACKLOG = 14
MODES = ("off", "norm_only")


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


class ScaleInvariantFeatures:
    """Rewrite the two N-carrying feature dims on the way into the network."""

    def __init__(self, *, mode: str = "off", subscribers: int | None = None) -> None:
        if mode not in MODES:
            raise ValueError("SCALE_INVARIANT must be one of %r, got %r" % (MODES, mode))
        self.mode = mode
        self.subscribers = subscribers
        self.original = {INDEX_BACKLOG: _RunningDim(), INDEX_SUBSCRIBERS: _RunningDim()}
        self.rewritten = {INDEX_BACKLOG: _RunningDim(), INDEX_SUBSCRIBERS: _RunningDim()}
        self.matrix_calls = 0
        self.diff_histogram = {INDEX_BACKLOG: {}, INDEX_SUBSCRIBERS: {}}
        self.changed = {INDEX_BACKLOG: 0, INDEX_SUBSCRIBERS: 0}
        self._restore = None

    # ---------------------------------------------------------------- the transform
    def rewrite(self, matrix):
        """Return (matrix handed to the network, ). ``off`` returns the input unchanged."""
        import numpy as np

        array = np.asarray(matrix, dtype="float32")
        if array.ndim != 2 or array.shape[1] <= INDEX_SUBSCRIBERS:
            raise RuntimeError("expected a [candidates, 19] matrix, got %r"
                               % (array.shape,))
        self.matrix_calls += 1
        self.original[INDEX_BACKLOG].add(array[:, INDEX_BACKLOG])
        self.original[INDEX_SUBSCRIBERS].add(array[:, INDEX_SUBSCRIBERS])
        if self.mode == "off":
            # Identity on purpose: the same accounting runs, nothing is written back, so
            # "off" is provably the untouched path rather than a second code path.
            self.rewritten[INDEX_BACKLOG].add(array[:, INDEX_BACKLOG])
            self.rewritten[INDEX_SUBSCRIBERS].add(array[:, INDEX_SUBSCRIBERS])
            return matrix

        out = array.copy()
        divisor = float(SUBSCRIBERS_REF)
        if self.subscribers is not None:
            # #17 holds the real subscriber count; n/6 makes N=6 exactly 1.0
            out[:, INDEX_SUBSCRIBERS] = out[:, INDEX_SUBSCRIBERS] / divisor
        out[:, INDEX_BACKLOG] = out[:, INDEX_BACKLOG] / float(NUM_GOALS)

        for index in (INDEX_BACKLOG, INDEX_SUBSCRIBERS):
            before = array[:, index]
            after = out[:, index]
            self.rewritten[index].add(after)
            for old_value, new_value in zip(before, after):
                if float(old_value) != float(new_value):
                    self.changed[index] += 1
                key = round(float(new_value) - float(old_value), 4)
                bucket = self.diff_histogram[index]
                bucket[key] = bucket.get(key, 0) + 1
        # The caller hands this straight to the trunk, which is a torch module: the array must
        # go back to a tensor with the exact dtype the frozen collector produced.
        import torch
        return torch.as_tensor(out)

    # ---------------------------------------------------------------- install / restore
    def install(self, controller):
        """Shadow ``controller.forward``; returns the original so the caller can restore."""
        original = controller.forward

        def tapped(matrix):
            return original(self.rewrite(matrix))

        controller.forward = tapped
        self._restore = (controller, original)
        return original

    @staticmethod
    def restore(installed):
        """Undo :meth:`install` using what it returned."""
        target = installed.get("scale_features")
        if target is None:
            return
        controller, original = target._restore
        try:
            del controller.forward
        except AttributeError:
            controller.forward = original

    # ---------------------------------------------------------------- evidence
    def audit(self) -> dict:
        return {
            "mode": self.mode,
            "subscribers": self.subscribers,
            "matrix_calls": self.matrix_calls,
            "dims": {
                "queue_backlog_targets(#14)": {
                    "original": self.original[INDEX_BACKLOG].summary(),
                    "rewritten": self.rewritten[INDEX_BACKLOG].summary(),
                    "values_changed": self.changed[INDEX_BACKLOG],
                    "diff_histogram": {str(k): v for k, v in
                                       sorted(self.diff_histogram[INDEX_BACKLOG].items())},
                },
                "subscribers(#17)": {
                    "original": self.original[INDEX_SUBSCRIBERS].summary(),
                    "rewritten": self.rewritten[INDEX_SUBSCRIBERS].summary(),
                    "values_changed": self.changed[INDEX_SUBSCRIBERS],
                    "diff_histogram": {str(k): v for k, v in
                                       sorted(self.diff_histogram[INDEX_SUBSCRIBERS].items())},
                },
            },
        }
