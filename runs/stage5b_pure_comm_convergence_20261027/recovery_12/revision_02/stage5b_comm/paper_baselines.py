"""Paper baselines that need no training: two throttles over the immediate rule.

Both wrap ``immediate`` (send every accepted detection once, at once) so the only difference from
the learned arms is *which* of those candidate messages survive. Neither reads a network or a
checkpoint, so they cost one world each and are reproducible from the same seeds.

``random_matched_rate``  thins each decision's candidate list at a fixed per-cell keep
                        probability ``(P - 1) / P``, where ``P`` is ``normal_immediate``'s pages
                        per episode in that cell. The expected page count therefore matches what
                        ``v4t``'s page budget allows, which is what makes the comparison
                        "same bytes, different choice" rather than "same rule, different budget".

``slack_heuristic_cap`` keeps the same page budget but orders candidates by deadline slack
                        (``remaining_time`` ascending, earliest first) instead of at random. It is
                        the single-feature heuristic a practitioner would reach for, so it tests
                        whether the 19-dim joint state buys anything over one dimension.

Both spend a per-episode page budget, but at most ``per_step`` pages per decision. Without that
rate, ``immediate`` empties the budget on the first decision and the two throttles become
indistinguishable -- measured on two scenes: identical pages and identical on-time for both arms.
One page per decision is what makes "which message gets the page" a real choice.

Both reset per episode on the same rule the page budget uses: a decision whose ``observation.step``
does not increase starts a new episode.
"""

from __future__ import annotations

from random import Random


class _PerEpisode:
    """Shared bookkeeping: reset when the relay's step counter goes backwards."""

    def __init__(self):
        self._last_step = None
        self.pages = 0

    def _tick(self, observation):
        step = int(getattr(observation, "step", 0))
        # Only a *backwards* step means a new episode. Several decisions can share one step, and
        # re-seeding on equality restarted the coin stream every call: measured on the first pass,
        # the same first draw was reused for every candidate and the arm sent exactly what
        # `immediate` sent (1245/1245 sends, byte ratio 1.000).
        if self._last_step is None or step < self._last_step:
            self.on_reset()
        self._last_step = step
        return step

    def on_reset(self):  # pragma: no cover - overridden
        raise NotImplementedError


class RandomThinPolicy(_PerEpisode):
    """Give each target one coin flip per episode: heads it is sent, tails it never is.

    The decision must be **sticky per target**, not a fresh coin per step. A relay re-offers a
    target until it has been sent, so an un-throttled coin at rate ``p`` only delays each send: at
    ``p = 0.40`` the measured arm still emitted 142 of 142 pages, just later than ``immediate``,
    and spent exactly ``immediate``'s bytes. One flip per target is what makes the byte count
    actually fall, and it is also the arm the paper needs: "random choice among candidates at the
    same byte budget".
    """

    name = "random_matched_rate"
    orders_selection = False

    def __init__(self, inner, keep_probability, *, seed=0):
        super().__init__()
        self._inner = inner
        self.keep_probability = float(keep_probability)
        if not 0.0 <= self.keep_probability <= 1.0:
            raise ValueError("keep_probability must be in [0, 1], got %r"
                             % (keep_probability,))
        self.seed = int(seed)
        self._rng = None
        self._kept = {}
        self.dropped = 0

    def on_reset(self):
        self._rng = Random(self.seed)
        self._kept = {}
        self.pages = 0
        self.dropped = 0

    def parameters(self):
        return {"keep_probability": self.keep_probability, "seed": self.seed,
                "decision": "sticky_per_target",
                "inner": getattr(self._inner, "name", "immediate")}

    def decide(self, observation):
        self._tick(observation)
        chosen = list(self._inner.decide(observation))
        kept = []
        for target in chosen:
            decision = self._kept.get(target)
            if decision is None:
                decision = self._rng.random() < self.keep_probability
                self._kept[target] = decision
            if decision:
                kept.append(target)
            else:
                self.dropped += 1
        self.pages += len(kept)
        return tuple(kept)


class FifoCapPolicy(_PerEpisode):
    """Pure quota: keep the rule's own order, send the first ``cap`` pages of the episode.

    No reordering and no selection at all -- this is the control that separates "the quota did it"
    from "the selector did it". A learned arm or a deadline heuristic that beats this arm is
    contributing a choice; one that matches it is only contributing a budget.
    """

    name = "fifo_cap_k1"
    orders_selection = False

    def __init__(self, inner, cap, *, per_step=1):
        super().__init__()
        self._inner = inner
        self.cap = int(cap)
        self.per_step = max(1, int(per_step))
        if self.cap < 0:
            raise ValueError("a page budget cannot be negative, got %r" % (cap,))

    def on_reset(self):
        self.pages = 0

    def parameters(self):
        return {"cap": self.cap, "per_step": self.per_step, "order": "rule_order",
                "inner": getattr(self._inner, "name", "immediate")}

    def decide(self, observation):
        self._tick(observation)
        chosen = list(self._inner.decide(observation))
        room = min(self.per_step, max(0, self.cap - self.pages))
        picked = tuple(chosen[:room])
        self.pages += len(picked)
        return picked


class SlackMatchedPolicy(_PerEpisode):
    """Deadline order at exactly the learned arm's page count -- the matched-depth control.

    ``SlackCapPolicy`` runs at the registered ``k = 1`` budget (about 3 % fewer bytes), so it is
    not comparable to a 20 % -cutting arm. This one keeps the same ordering rule but caps the
    episode at ``v4t``'s measured pages per episode in that cell, which is what makes
    "heuristic vs learned at the same bytes" a real comparison.
    """

    name = "slack_matched_rate"
    orders_selection = True

    def __init__(self, inner, cap, *, per_step=1):
        super().__init__()
        self._inner = inner
        self.cap = int(cap)
        self.per_step = max(1, int(per_step))

    def on_reset(self):
        self.pages = 0

    def parameters(self):
        return {"cap": self.cap, "per_step": self.per_step, "order": "remaining_time_ascending",
                "matched_to": "v4t pages per episode",
                "inner": getattr(self._inner, "name", "immediate")}

    def decide(self, observation):
        self._tick(observation)
        chosen = list(self._inner.decide(observation))
        slack = {}
        for item in observation.items:
            remaining = getattr(item, "remaining_time", None)
            slack[item.target_id] = (float("inf") if remaining is None
                                     else float(remaining))
        ordered = sorted(chosen, key=lambda target: (slack.get(target, float("inf")),
                                                     chosen.index(target)))
        room = min(self.per_step, max(0, self.cap - self.pages))
        picked = tuple(ordered[:room])
        self.pages += len(picked)
        return picked


class SlackCapPolicy(_PerEpisode):
    """Same page budget as the learned arms, ordered by earliest deadline slack."""

    name = "slack_heuristic_cap"
    orders_selection = True

    def __init__(self, inner, cap, *, per_step=1):
        super().__init__()
        self._inner = inner
        self.cap = int(cap)
        self.per_step = max(1, int(per_step))
        if self.cap < 0:
            raise ValueError("a page budget cannot be negative, got %r" % (cap,))

    def on_reset(self):
        self.pages = 0

    def parameters(self):
        return {"cap": self.cap, "per_step": self.per_step,
                "order": "remaining_time_ascending",
                "inner": getattr(self._inner, "name", "immediate")}

    def decide(self, observation):
        self._tick(observation)
        chosen = list(self._inner.decide(observation))
        slack = {}
        for item in observation.items:
            remaining = getattr(item, "remaining_time", None)
            slack[item.target_id] = (float("inf") if remaining is None
                                     else float(remaining))
        # Ties fall back to the inner policy's own order, so the arm is deterministic.
        ordered = sorted(chosen, key=lambda target: (slack.get(target, float("inf")),
                                                     chosen.index(target)))
        room = min(self.per_step, max(0, self.cap - self.pages))
        picked = tuple(ordered[:room])
        self.pages += len(picked)
        return picked
