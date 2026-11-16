"""Four named classical algorithms at the *policy* layer: no environment change, no training of
the frozen network, one evaluation world per (arm, cell, scene).

Every arm wraps the ``immediate`` rule, so the candidate list at each decision is exactly the
one the full-broadcast baseline sees; the arms differ only in which of those candidates are
admitted.  All four read the same dimensionless state the learned controller reads:

  #10 stage_index          #11 remaining_time / 200        #14 backlog / G
  #16 capacity_ratio       #17 subscribers / 6

``RED_AQM`` (Floyd & Jacobson, 1993)
    Random Early Detection on the equivalent queue load ``x = (backlog / G) / capacity_ratio``.
    Below ``min_th`` every candidate is admitted; above ``max_th`` admission is suppressed with
    probability ``max_p``; between them the drop probability rises linearly.  The admission
    decision is taken once, at the target's **first offer**, and then remembered for the episode
    -- the arrival-time drop semantics of an AQM.  A fresh coin per decision would only delay
    an unsent target, because the relay re-offers it until it is sent (the same defect measured
    on the random throttle in ``paper_baselines``).

``Token_Bucket`` (Turner, 1986)
    A shaper, not an admission rule.  Tokens refill at ``rate`` per step up to ``depth``; a page
    costs one token; candidates go out earliest-deadline-first while tokens last.  ``rate`` is
    calibrated per cell from ``immediate``'s measured pages per world, so the shaper runs at the
    same reference budget as the learned arm instead of at an arbitrary one.

``AHP_Utility`` (Saaty; the multi-attribute method of Han Cheng's §4)
    Five indicators -- urgency (#11), stage progress (#10), backlog pressure (#14), capacity
    headroom (#16), fan-out (#17) -- scored by a fixed eigenvector weight vector.  The pairwise
    matrix and its consistency ratio are computed below and asserted ``CR < 0.1``.  A target is
    admitted at its first offer when the composite score reaches ``threshold``.

``LinUCB_Bandit``
    The same 19-dim state and the same training world stream as the learned controller, but a
    one-step (myopic) linear UCB instead of the long-horizon dual-critic PPO.  Parameters are
    ridge estimates from the first-offer decisions of ``immediate`` on the *training* scenes;
    the UCB bonus uses the stored ``(X'X + lI)^-1``.  Evaluation freezes the parameters: the
    training scenes are disjoint from the eight DEV evaluation scenes.

Nothing here writes to the environment, the relay or the frozen 98 files.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from . import constants as C
from .paper_baselines import _PerEpisode
from .scale_features import (INDEX_BACKLOG, INDEX_CAPACITY, INDEX_REMAINING_TIME,
                             INDEX_SUBSCRIBERS, apply_rewrite, capacity_ratio)

CLASSIC_ARMS = ("red_aqm", "token_bucket", "ahp_utility", "linucb_bandit")

STAGE_REFERENCE = 3.0
FANOUT_REFERENCE = 12.0

#: Saaty pairwise judgments over (urgency, stage, backlog, capacity, fan-out).  Chosen to encode
#: "deadline first, then congestion, then progress/roster", and verified consistent below.
AHP_PAIRWISE = (
    (1.0, 3.0, 2.0, 3.0, 4.0),
    (1.0 / 3.0, 1.0, 1.0, 1.0, 1.0),
    (1.0 / 2.0, 1.0, 1.0, 1.0, 2.0),
    (1.0 / 3.0, 1.0, 1.0, 1.0, 1.0),
    (1.0 / 4.0, 1.0, 1.0 / 2.0, 1.0, 1.0),
)
AHP_RI5 = 1.12


def ahp_weights():
    """Principal eigenvector of :data:`AHP_PAIRWISE`, plus ``lambda_max``, CI and CR."""
    import numpy as np

    matrix = np.asarray(AHP_PAIRWISE, dtype="float64")
    values, vectors = np.linalg.eig(matrix)
    index = int(np.argmax(values.real))
    lambda_max = float(values.real[index])
    vector = np.abs(vectors[:, index].real)
    vector = vector / vector.sum()
    count = matrix.shape[0]
    consistency = (lambda_max - count) / (count - 1.0) / AHP_RI5
    return vector.tolist(), lambda_max, consistency


_AHP_WEIGHTS, _AHP_LAMBDA, _AHP_CR = ahp_weights()
if not _AHP_CR < 0.1:
    raise RuntimeError("AHP pairwise matrix is inconsistent: CR=%.4f" % _AHP_CR)


def _uniform(seed, target):
    """Stable [0,1) draw per (seed, target): independent of iteration order."""
    digest = hashlib.sha256(("%d:%s" % (int(seed), target)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _items(observation):
    return {str(item.target_id): item for item in observation.items}


def _urgency(remaining):
    """#11 exactly as ``install_horizon_scale`` writes it into the learned arm's row."""
    return max(-1.0, min(1.0, float(remaining) / float(C.HORIZON_SCALE_DENOMINATOR)))


def core_context(observation, item, *, goals, robots, subscribers):
    """The five dimensionless indicators the two non-learning selectors share."""
    remaining = _urgency(item.remaining_time)
    return {
        "urgency": (1.0 - remaining) / 2.0,
        "stage": min(1.0, float(item.stage_index) / STAGE_REFERENCE),
        "backlog": min(1.0, float(observation.backlog_targets) / float(goals)),
        "capacity": min(1.0, capacity_ratio(robots, goals) / 2.0),
        "fanout": min(1.0, float(subscribers) / FANOUT_REFERENCE),
    }


def state_vector(observation, item, *, goals, robots, subscribers):
    """The 19-dim row the learned controller sees, for the bandit baseline.

    ``comm_learning.features`` still carries the shared 50-step clip in #11, so #11 is
    overwritten here exactly as the revision's horizon shadow does; the other three scale dims
    go through the same pure rewrite the trunk uses.
    """
    from integrated_sar.comm_learning import features

    raw = list(features(observation, item, horizon=C.HORIZON_SCALE_DENOMINATOR))
    raw[INDEX_REMAINING_TIME] = _urgency(item.remaining_time)
    rewritten = apply_rewrite([raw], mode="scaled", goals=goals, subscribers=subscribers,
                              robots=robots).numpy()[0]
    return [float(value) for value in rewritten]


class _AdmissionBase(_PerEpisode):
    """Admit or drop each target once, at its first offer."""

    orders_selection = False

    def __init__(self, inner, *, goals, robots, subscribers):
        super().__init__()
        self._inner = inner
        self.goals = int(goals)
        self.robots = int(robots)
        self.subscribers = int(subscribers)
        self._admitted = {}
        self.dropped = 0

    def on_reset(self):
        self.pages = 0
        self._admitted = {}
        self.dropped = 0

    def admit(self, observation, item):  # pragma: no cover - overridden
        raise NotImplementedError

    def decide(self, observation):
        self._tick(observation)
        chosen = list(self._inner.decide(observation))
        index = _items(observation)
        kept = []
        for target in chosen:
            decision = self._admitted.get(target)
            if decision is None:
                item = index.get(str(target))
                decision = False if item is None else bool(self.admit(observation, item))
                self._admitted[target] = decision
            if decision:
                kept.append(target)
            else:
                self.dropped += 1
        self.pages += len(kept)
        return tuple(kept)


class RedAqmPolicy(_AdmissionBase):
    """Random Early Detection on the equivalent queue load ``(backlog/G) / capacity_ratio``."""

    name = "red_aqm"

    def __init__(self, inner, *, goals, robots, subscribers, min_th, max_th, max_p, seed=0):
        super().__init__(inner, goals=goals, robots=robots, subscribers=subscribers)
        self.min_th = float(min_th)
        self.max_th = float(max_th)
        self.max_p = float(max_p)
        if not 0.0 <= self.min_th <= self.max_th:
            raise ValueError("RED thresholds must satisfy 0 <= min_th <= max_th")
        if not 0.0 <= self.max_p <= 1.0:
            raise ValueError("max_p must be in [0, 1], got %r" % (max_p,))
        self.seed = int(seed)
        self.loads = []
        self.drop_probabilities = []

    def on_reset(self):
        super().on_reset()
        self.loads = []
        self.drop_probabilities = []

    def drop_probability(self, observation):
        load = (float(observation.backlog_targets) / float(self.goals)) / capacity_ratio(
            self.robots, self.goals)
        self.loads.append(load)
        if load <= self.min_th:
            return 0.0
        if load >= self.max_th:
            return self.max_p
        return self.max_p * (load - self.min_th) / (self.max_th - self.min_th)

    def admit(self, observation, item):
        probability = self.drop_probability(observation)
        self.drop_probabilities.append(probability)
        return _uniform(self.seed, item.target_id) >= probability

    def parameters(self):
        return {"min_th": self.min_th, "max_th": self.max_th, "max_p": self.max_p,
                "seed": self.seed, "load": "((backlog/G) / capacity_ratio)",
                "decision": "sticky_first_offer", "inner": "immediate"}


class TokenBucketPolicy(_PerEpisode):
    """Fixed-rate token shaper: a page costs a token; earliest deadline first while tokens last."""

    name = "token_bucket"
    orders_selection = True

    def __init__(self, inner, *, rate, depth, per_step=None):
        super().__init__()
        self._inner = inner
        self.rate = float(rate)
        self.depth = float(depth)
        self.per_step = None if per_step is None else max(1, int(per_step))
        if self.rate < 0.0:
            raise ValueError("token rate cannot be negative")
        if self.depth < 1.0:
            raise ValueError("bucket depth must hold at least one page")
        self.tokens = 0.0
        self._refill_step = None
        self.starved = 0

    def on_reset(self):
        self.pages = 0
        self.tokens = self.depth
        self._refill_step = None
        self.starved = 0

    def decide(self, observation):
        step = self._tick(observation)
        if self._refill_step is None:
            self.tokens = self.depth
        else:
            self.tokens = min(self.depth, self.tokens + self.rate * (step - self._refill_step))
        self._refill_step = step
        chosen = list(self._inner.decide(observation))
        slack = {}
        for item in observation.items:
            slack[str(item.target_id)] = float(item.remaining_time)
        ordered = sorted(chosen, key=lambda target: (slack.get(str(target), float("inf")),
                                                     chosen.index(target)))
        room = int(math.floor(self.tokens + 1e-9))
        if self.per_step is not None:
            room = min(room, self.per_step)
        room = max(0, room)
        picked = tuple(ordered[:room])
        self.starved += max(0, len(ordered) - len(picked))
        self.tokens -= len(picked)
        self.pages += len(picked)
        return picked

    def parameters(self):
        return {"rate": self.rate, "depth": self.depth, "per_step": self.per_step,
                "order": "remaining_time_ascending", "charge": "1 page = 1 token",
                "inner": "immediate"}


class AhpUtilityPolicy(_AdmissionBase):
    """Fixed AHP weights over five indicators, with an admission threshold."""

    name = "ahp_utility"

    def __init__(self, inner, *, goals, robots, subscribers, threshold):
        super().__init__(inner, goals=goals, robots=robots, subscribers=subscribers)
        self.threshold = float(threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("AHP threshold must be in [0, 1], got %r" % (threshold,))
        self.scores = []

    def on_reset(self):
        super().on_reset()
        self.scores = []

    def utility(self, observation, item):
        context = core_context(observation, item, goals=self.goals, robots=self.robots,
                               subscribers=self.subscribers)
        names = ("urgency", "stage", "backlog", "capacity", "fanout")
        return float(sum(weight * context[name]
                         for weight, name in zip(_AHP_WEIGHTS, names)))

    def admit(self, observation, item):
        score = self.utility(observation, item)
        self.scores.append(score)
        return score >= self.threshold

    def parameters(self):
        return {"threshold": self.threshold, "weights": list(_AHP_WEIGHTS),
                "lambda_max": _AHP_LAMBDA, "consistency_ratio": _AHP_CR,
                "indicators": ["#11 urgency", "#10 stage", "#14 backlog/G",
                               "#16 capacity_ratio", "#17 subscribers"],
                "decision": "sticky_first_offer", "inner": "immediate"}


class LinUcbBanditPolicy(_AdmissionBase):
    """Myopic linear UCB over the 19-dim state, parameters frozen from the training stream."""

    name = "linucb_bandit"

    def __init__(self, inner, *, goals, robots, subscribers, model):
        super().__init__(inner, goals=goals, robots=robots, subscribers=subscribers)
        if isinstance(model, (str, Path)):
            model = json.loads(Path(model).read_text(encoding="utf-8"))
        self.model = model
        self.theta = [float(value) for value in model["theta"]]
        self.alpha = float(model["alpha"])
        self.threshold = float(model["threshold"])
        self.mean = [float(value) for value in model["mean"]]
        self.sd = [float(value) for value in model["sd"]]
        self.a_inv = [[float(value) for value in row] for row in model["a_inv"]]
        if len(self.theta) != 20 or len(self.a_inv) != 20:
            raise ValueError("the frozen model must carry a 20-dim intercept + 19 features")
        self.scores = []

    def on_reset(self):
        super().on_reset()
        self.scores = []

    def _standardise(self, vector):
        return [1.0] + [(value - mean) / sd
                        for value, mean, sd in zip(vector, self.mean, self.sd)]

    def score(self, row):
        mean = sum(theta * value for theta, value in zip(self.theta, row))
        quad = 0.0
        for i, left in enumerate(self.a_inv):
            inner = sum(left[j] * row[j] for j in range(len(row)))
            quad += row[i] * inner
        return mean + self.alpha * math.sqrt(max(0.0, quad))

    def admit(self, observation, item):
        vector = state_vector(observation, item, goals=self.goals, robots=self.robots,
                              subscribers=self.subscribers)
        value = self.score(self._standardise(vector))
        self.scores.append(value)
        return value >= self.threshold

    def parameters(self):
        return {"alpha": self.alpha, "threshold": self.threshold,
                "features": "19-dim comm-send-state/1 with the revision's scaling",
                "training": self.model.get("training"),
                "decision": "sticky_first_offer", "inner": "immediate"}


class FirstOfferLogger(_PerEpisode):
    """Sends everything and records the 19-dim row at each target's first offer.

    Used once, on the training scenes, to build the bandit's labelled sample.  Because the inner
    rule is ``immediate`` every detection is emitted, so the labels carry no admission-selection
    bias.
    """

    name = "feature_log"

    def __init__(self, inner, *, trace_path, goals, robots, subscribers, extras=None):
        super().__init__()
        self._inner = inner
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.goals = int(goals)
        self.robots = int(robots)
        self.subscribers = int(subscribers)
        self.extras = dict(extras or {})
        self._seen = set()
        self._handle = None
        self.rows = 0

    def on_reset(self):
        self.pages = 0
        self._seen = set()
        if self._handle is not None:
            self._handle.close()
        self._handle = self.trace_path.open("w", encoding="utf-8")
        header = {"schema": "stage5b-classic-trace/1"}
        header.update(self.extras)
        self._handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        self.rows = 0

    def decide(self, observation):
        self._tick(observation)
        chosen = list(self._inner.decide(observation))
        index = _items(observation)
        for target in chosen:
            if target in self._seen:
                continue
            self._seen.add(target)
            item = index.get(str(target))
            if item is None:
                continue
            context = core_context(observation, item, goals=self.goals, robots=self.robots,
                                   subscribers=self.subscribers)
            self._handle.write(json.dumps({
                "target": str(target), "step": int(observation.step),
                "state": state_vector(observation, item, goals=self.goals, robots=self.robots,
                                      subscribers=self.subscribers),
                "core": context,
                "remaining_time": float(item.remaining_time),
            }, sort_keys=True) + "\n")
            self._handle.flush()
            self.rows += 1
        self.pages += len(chosen)
        return tuple(chosen)

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def parameters(self):
        return {"trace": str(self.trace_path), "rows": self.rows,
                "decision": "first_offer_of_immediate"}


def build_classic_policy(arm, *, goals, robots, subscribers, params=None, model_path=None):
    """Construct one arm; all numeric parameters must be explicit (no hidden defaults)."""
    from integrated_sar.comm_policy import build_comm_policy

    params = dict(params or {})
    inner = build_comm_policy("immediate")
    if arm == "red_aqm":
        return RedAqmPolicy(inner, goals=goals, robots=robots, subscribers=subscribers,
                            min_th=params["min_th"], max_th=params["max_th"],
                            max_p=params["max_p"], seed=params.get("seed", 0))
    if arm == "token_bucket":
        return TokenBucketPolicy(inner, rate=params["rate"], depth=params["depth"],
                                 per_step=params.get("per_step"))
    if arm == "ahp_utility":
        return AhpUtilityPolicy(inner, goals=goals, robots=robots, subscribers=subscribers,
                                threshold=params["threshold"])
    if arm == "linucb_bandit":
        model = json.loads(Path(model_path).read_text(encoding="utf-8"))
        if params.get("threshold") is not None:
            model["threshold"] = float(params["threshold"])
        return LinUcbBanditPolicy(inner, goals=goals, robots=robots, subscribers=subscribers,
                                  model=model)
    if arm == "feature_log":
        return FirstOfferLogger(inner, trace_path=params["trace"], goals=goals, robots=robots,
                                subscribers=subscribers, extras=params.get("extras"))
    raise ValueError("unknown classic arm %r" % (arm,))
