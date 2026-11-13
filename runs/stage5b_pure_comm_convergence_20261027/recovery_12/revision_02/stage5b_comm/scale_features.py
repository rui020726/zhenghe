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

from . import constants as C

SUBSCRIBERS_REF = 6
INDEX_SENT_COUNT = 2
INDEX_REMAINING_TIME = 11
INDEX_BACKLOG = 14
INDEX_CAPACITY = 16
INDEX_SUBSCRIBERS = 17
ROBOTS_REF = 6
GOALS_REF = 24
SCALED_INDICES = (INDEX_SENT_COUNT, INDEX_BACKLOG, INDEX_CAPACITY, INDEX_SUBSCRIBERS)
#: Dimensions experiment J is allowed to force to a constant. #11 is in the list because its
#: ablation is "put the shared 50-step clipping back", which is a source-level change, not a
#: column overwrite; it is named here so a caller cannot mask an arbitrary column by accident.
MASKABLE_INDICES = (INDEX_SENT_COUNT, INDEX_REMAINING_TIME, INDEX_BACKLOG, INDEX_CAPACITY,
                    INDEX_SUBSCRIBERS)
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


def capacity_ratio(robots, goals):
    """#16: rescue capacity against task load, ``(R / 6) / (G / 24)``.

    The frozen builder computed ``capacity_remaining / (capacity_remaining + 1)``, which is
    0.5 for one free slot and 0.99 for a hundred -- effectively a constant 1.0 at every step of
    every world, so the policy could not tell R=6 from R=12 by reading its own input. This puts
    the roster-to-workload ratio there instead: 1.0 at the reference (R=6, G=24), 2.0 at
    (12, 6), 0.5 at (48, 6). It is a cell-level constant, which is exactly what R and G are.
    """
    robots = int(robots)
    goals = int(goals)
    if robots < 1 or goals < 1:
        raise ValueError("robots and goals must be >= 1, got %d and %d" % (robots, goals))
    return (float(robots) / ROBOTS_REF) / (float(goals) / GOALS_REF)


def apply_rewrite(matrix, *, mode, goals, subscribers, robots=ROBOTS_REF):
    """The transform as a pure function: no state, no hook, same answer anywhere.

    This exists because the trunk wrap is **episode-scoped**: ``world`` restores it when the
    episode ends, and the PPO update runs afterwards. An update that read the raw
    ``tape["features"]`` would therefore recompute log-probabilities from a different row than
    the one the policy actually saw -- P4's 1.589 importance-ratio defect, relocated into the
    trainer. Measured on this revision: max |logp_sampled - logp_update| = 0.030 when the
    transform is installed only for the episode.

    With this function the update reproduces the exact row without needing the wrap alive, so
    the ratio at update 0 is again ~0.

    Dim #11 is left alone here on purpose: its denominator is replaced at the source, in
    :func:`install_horizon_scale`, because the raw ``remaining_time`` is not recoverable from
    the stored column once the shared builder has clipped it.
    """
    import numpy as np

    array = np.asarray(matrix, dtype="float32")
    if array.ndim != 2 or array.shape[1] <= INDEX_SUBSCRIBERS:
        raise RuntimeError("expected a [candidates, 19] matrix, got %r" % (array.shape,))
    if mode not in MODES:
        raise ValueError("unknown scale mode %r" % (mode,))
    if mode == "off":
        return torch.as_tensor(array)
    goals = int(goals)
    subscribers = int(subscribers)
    if goals < 1 or subscribers < 1:
        raise ValueError("goals and subscribers must be >= 1, got %d and %d"
                         % (goals, subscribers))
    out = array.copy()
    out[:, INDEX_SENT_COUNT] = out[:, INDEX_SENT_COUNT] / float(goals)
    out[:, INDEX_BACKLOG] = out[:, INDEX_BACKLOG] / float(goals)
    out[:, INDEX_SUBSCRIBERS] = out[:, INDEX_SUBSCRIBERS] / float(SUBSCRIBERS_REF)
    out[:, INDEX_CAPACITY] = capacity_ratio(robots, goals)
    return torch.as_tensor(out)


def install_horizon_scale(denominator=None):
    """Shadow ``integrated_sar.comm_learning.features`` so #11 divides by 200, not 50.

    The shared builder writes ``clip(remaining_time / 50, -1, 1)``, and that clipping is lossy:
    once a target is more than 50 steps from its deadline its column reads exactly 1.0, so
    "tight" and "far" are indistinguishable for the bulk of an episode. A pure matrix rewrite
    cannot undo it (``1.0`` carries no information about how far past 50 it was), so the column
    has to be produced correctly at the source.

    No shared file is edited: this replaces the module attribute in memory, exactly like the
    trunk shadow, and :func:`restore_horizon_scale` puts the original back. ``LearnedSendPolicy
    .decide`` resolves ``features`` from module globals on every call, so the shadow is live for
    both the row the policy samples from and the row the collector records.

    Returns ``(original, denominator)``; idempotent.
    """
    import integrated_sar.comm_learning as comm_learning

    denominator = float(denominator if denominator is not None
                        else C.HORIZON_SCALE_DENOMINATOR)
    current = comm_learning.features
    existing = getattr(current, "_revision02_horizon_denominator", None)
    if existing is not None:
        return current._revision02_horizon_original, float(existing)

    def shadowed(observation, item, *, horizon=comm_learning.EPISODE_HORIZON_STEPS,
                 ack_features=None):
        values = list(current(observation, item, horizon=horizon, ack_features=ack_features))
        remaining = float(item.remaining_time)
        values[INDEX_REMAINING_TIME] = max(-1.0, min(1.0, remaining / denominator))
        return tuple(values)

    shadowed._revision02_horizon_denominator = denominator
    shadowed._revision02_horizon_original = current
    comm_learning.features = shadowed
    return current, denominator


def restore_horizon_scale():
    """Put the shared builder back. Idempotent; returns the denominator that was live."""
    import integrated_sar.comm_learning as comm_learning

    current = comm_learning.features
    original = getattr(current, "_revision02_horizon_original", None)
    if original is None:
        return None
    comm_learning.features = original
    return float(current._revision02_horizon_denominator)


class ScaleFeatures:
    """Rewrite #2, #14, #16 and #17 so one policy reads every cell the same way."""

    def __init__(self, *, mode: str = "scaled", goals: int = 24,
                 subscribers: int = 6, robots: int = ROBOTS_REF,
                 mask: dict | None = None, horizon_denominator: float | None = None) -> None:
        if mode not in MODES:
            raise ValueError("SCALE_FEATURES must be one of %r, got %r" % (MODES, mode))
        goals = int(goals)
        subscribers = int(subscribers)
        robots = int(robots)
        if goals < 1:
            raise ValueError("goals must be >= 1 for the #2/#14 divisor, got %d" % goals)
        if subscribers < 1:
            raise ValueError("subscribers must be >= 1 for #17, got %d" % subscribers)
        if robots < 1:
            raise ValueError("robots must be >= 1 for #16, got %d" % robots)
        #: Feature ablation (experiment J): ``{column index: forced value}`` applied **after**
        #: the scale rewrite. Only the four scale-carrying dims and #11 are addressable, because
        #: those are the columns this revision's story names; masking any other column would be
        #: a different probe than the one registered.
        self.mask = {}
        for index, value in (mask or {}).items():
            index = int(index)
            if index not in MASKABLE_INDICES:
                raise ValueError("feature #%d is not maskable; registered dims are %r"
                                 % (index, sorted(MASKABLE_INDICES)))
            self.mask[index] = float(value)
        self.horizon_denominator = (None if horizon_denominator is None
                                    else float(horizon_denominator))
        if self.horizon_denominator is not None and self.horizon_denominator <= 0:
            raise ValueError("horizon denominator must be positive, got %r"
                             % (horizon_denominator,))
        self.mode = mode
        self.goals = goals
        self.subscribers = subscribers
        self.robots = robots
        self.original = {index: _RunningDim() for index in SCALED_INDICES}
        self.rewritten = {index: _RunningDim() for index in SCALED_INDICES}
        self.matrix_calls = 0
        self.changed = {index: 0 for index in SCALED_INDICES}
        self._restore = None
        self._horizon = None
        #: What was installed, kept across ``restore``: the row is assembled **after** the
        #: world releases the seams, so an audit that reads ``_horizon`` reports null for every
        #: world. This field is the evidence that the shadow really was in force.
        self._installed = {"horizon_denominator": None}

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

        out = apply_rewrite(array, mode=self.mode, goals=self.goals,
                            subscribers=self.subscribers, robots=self.robots).numpy()
        for index, value in self.mask.items():
            out[:, index] = value

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
        # Dim #11's denominator cannot be fixed downstream (the shared builder already clipped
        # it), so the source is shadowed here and un-shadowed in ``restore``. The experiment-J
        # horizon arm passes 50.0, which reproduces the shared clipping exactly.
        self._horizon = (install_horizon_scale(self.horizon_denominator)
                         if self.mode != "off" else None)
        self._installed["horizon_denominator"] = (
            None if self._horizon is None else float(self._horizon[1]))
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
        if target._horizon is not None:
            restore_horizon_scale()
            target._horizon = None

    # ---------------------------------------------------------------- evidence
    def audit(self) -> dict:
        return {
            "mode": self.mode,
            "goals": self.goals,
            "subscribers": self.subscribers,
            "robots": self.robots,
            "matrix_calls": self.matrix_calls,
            "divisors": {"#2 sent_count": self.goals, "#14 backlog": self.goals,
                         "#17 subscribers": SUBSCRIBERS_REF,
                         "#16 capacity_ratio": [self.robots, self.goals]},
            "horizon_scale": {
                # ``denominator`` is null once ``restore`` has run, which is always by the time
                # a row is assembled; ``installed_denominator`` is the value that was in force
                # while the policy was deciding.
                "denominator": (None if self._horizon is None
                                else float(self._horizon[1])),
                "installed_denominator": self._installed["horizon_denominator"],
                "shadow": ("integrated_sar.comm_learning.features"
                           if self._horizon is not None else None),
                "shadow_was_installed": self._installed["horizon_denominator"] is not None,
                "unshadowed_original": (None if self._horizon is None
                                        else type(self._horizon[0]).__name__),
            },
            "feature_mask": {("dim_%d" % index): value
                             for index, value in sorted(self.mask.items())},
            "dims": {("dim_%d" % index): {
                "original": self.original[index].summary(),
                "rewritten": self.rewritten[index].summary(),
                "values_changed": self.changed[index],
            } for index in SCALED_INDICES},
        }
