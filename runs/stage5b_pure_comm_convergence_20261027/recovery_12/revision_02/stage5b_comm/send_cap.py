"""Per-episode page budget: an integer actuator for "cut exactly k pages".

Why a second actuator exists. The scalar price was measured (v4d–v4n) to move the send/hold
decision in steps, not in pages: at λ≈4 one cell cut one page while another cut ninety-seven,
because the λ→pages map differs per cell and is steep. The registered verdict, however, is
decided in whole pages (bytes strictly below ``normal_immediate``) and whole goals (on-time at
least 95 % of it). A budget counts exactly those units, so one integer step of the dual is one
page — the same granularity as the verdict.

The policy is unchanged: the learned controller still ranks its candidate targets and the
relay still sends what it gets. The budget only truncates the ordered list, i.e. it drops the
targets the policy itself ranked lowest. No rule is substituted for the ranking.

Reset rule: a decision step whose ``observation.step`` does not increase starts a new episode.
The relay numbers steps from 0/1 upward and a world runs exactly one episode, so this is exact;
counting the pages sent is what matters, not the episode identity.
"""

from __future__ import annotations


class SendCapPolicy:
    """Wrap a policy and allow at most ``cap`` pages per episode."""

    def __init__(self, inner, cap):
        self._inner = inner
        self.cap = int(cap)
        if self.cap < 0:
            raise ValueError("a send cap cannot be negative, got %r" % (cap,))
        self.sent = 0
        self.dropped = 0
        self._last_step = None
        self.decisions = 0

    @property
    def name(self):
        return getattr(self._inner, "name", "learned")

    @property
    def orders_selection(self):
        return getattr(self._inner, "orders_selection", True)

    @property
    def last_features(self):
        return getattr(self._inner, "last_features", {})

    def parameters(self):
        base = {}
        if hasattr(self._inner, "parameters"):
            base = dict(self._inner.parameters())
        base.update({"send_cap": self.cap, "pages_sent": self.sent,
                     "pages_dropped": self.dropped})
        return base

    def decide(self, observation):
        step = int(getattr(observation, "step", 0))
        if self._last_step is None or step <= self._last_step:
            self.sent = 0
            self.dropped = 0
        self._last_step = step
        chosen = list(self._inner.decide(observation))
        remaining = max(0, self.cap - self.sent)
        picked = tuple(chosen[:remaining])
        self.dropped += len(chosen) - len(picked)
        self.sent += len(picked)
        self.decisions += 1
        return picked

    def summary(self):
        return {"cap": self.cap, "sent": self.sent, "dropped": self.dropped}


def update_cap(current, q_ema, q_ref, *, step=1, measured=None, q_target_fraction=0.965):
    """One scan of the integer dual. Returns ``(new_cap, detail)``.

    ``current`` is the cap in force (pages per episode); ``measured`` is what the policy spent
    without a cap, which is the ceiling the dual may climb back to. Above the water line the cap
    loses ``step`` pages, below it gains ``step``; the cap never goes below one page (a cell that
    sends nothing is ``always_hold``, a different arm) and never above ``measured + 1`` (so the
    dual cannot invent pages the policy would not have sent).

    The water line is the same ``0.965 * q_ref`` the price used, so a reader compares like with
    like: only the actuator changed.
    """
    if q_ref is None:
        raise ValueError("q_ref is required; a cell without a normal_immediate rate cannot "
                         "set its own constraint")
    q_ema = float(q_ema)
    q_ref = float(q_ref)
    for name, value in (("q_ema", q_ema), ("q_ref", q_ref)):
        if not 0.0 <= value <= 1.0:
            raise ValueError("%s must be in [0, 1], got %r" % (name, value))
    step = int(step)
    if step < 1:
        raise ValueError("the cap step is whole pages and must be >= 1, got %r" % (step,))
    if current is None:
        current = int(measured if measured is not None else 1)
    current = int(current)
    ceiling = None if measured is None else int(measured) + 1
    target = float(q_target_fraction) * q_ref
    if q_ema >= target:
        proposed = current - step
        direction = "cut_page"
    else:
        proposed = current + step
        direction = "add_page"
    new = max(1, proposed)
    if ceiling is not None:
        new = min(ceiling, new)
    if new == current:
        direction = "hold"
    return new, {
        "cap_before": current, "cap_after": new, "direction": direction,
        "q_ema": q_ema, "q_ref": q_ref, "q_target": target,
        "q_target_fraction": float(q_target_fraction),
        "step": step, "measured_pages": ceiling if ceiling is None else ceiling - 1,
        "clamped": new != proposed,
    }


class SendWindowPolicy:
    """Two-sided page window: at most ``cap`` pages, and at least ``floor`` **paced linearly**.

    The cap alone can only say "stop"; when the policy itself declines to send, the cap cannot
    make it send, and measured on ``v4t`` three cells therefore fell below the on-time target
    (G24_R6_N3, G48_R12_N3, G48_R12_N6). The floor restores exactly those cells: at each decision
    it compares the pages spent so far against ``floor * step / horizon`` and, when behind,
    appends the next-best candidates the inner policy did **not** pick.

    Those extra pages are taken in the inner policy's own order first (the ones it ranked but
    dropped are not available -- it never returned them), then in the relay's transport order,
    which is deterministic and identical for every arm. The floor never overrides the cap, and a
    floor of 0 reproduces ``SendCapPolicy`` byte for byte.
    """

    def __init__(self, inner, cap, floor=0, *, horizon=1000.0):
        self._inner = inner
        self.cap = int(cap)
        self.floor = int(floor)
        if self.cap < 1:
            raise ValueError("a page cap must be at least one page, got %r" % (cap,))
        if not 0 <= self.floor <= self.cap:
            raise ValueError("floor %r must be within [0, cap=%d]"
                             % (floor, self.cap))
        self.horizon = float(horizon)
        self.sent = 0
        self.dropped = 0
        self.forced = 0
        self._last_step = None
        self._sent_ids = set()
        self.decisions = 0

    @property
    def name(self):
        return getattr(self._inner, "name", "learned")

    @property
    def orders_selection(self):
        # The inner policy ranks; this wrapper preserves its order for everything it picks.
        return getattr(self._inner, "orders_selection", True)

    @property
    def last_features(self):
        return getattr(self._inner, "last_features", {})

    def parameters(self):
        base = {}
        if hasattr(self._inner, "parameters"):
            base = dict(self._inner.parameters())
        base.update({"send_cap": self.cap, "send_floor": self.floor,
                     "paced_over_horizon": self.horizon})
        return base

    def decide(self, observation):
        step = int(getattr(observation, "step", 0))
        if self._last_step is None or step <= self._last_step:
            self.sent = 0
            self.dropped = 0
            self.forced = 0
            self._sent_ids = set()
        self._last_step = step
        chosen = list(self._inner.decide(observation))
        room = max(0, self.cap - self.sent)
        picked = chosen[:room]
        self.dropped += len(chosen) - len(picked)
        self.sent += len(picked)
        self._sent_ids.update(str(target) for target in picked)
        if self.floor > 0:
            fraction = min(1.0, (step + 1.0) / max(1.0, self.horizon))
            target = int(round(self.floor * fraction))
            if target > self.sent and self.sent < self.cap:
                # A page already on the air this episode is not a candidate. Without this the
                # floor re-emitted delivered targets: measured on the first run, G48_R6_N3 sent
                # 152 pages against a cap of 138 (three scenes), i.e. more than `immediate`.
                already = set(picked) | self._sent_ids
                ceiling = min(target, self.cap) - self.sent
                extra = []
                for item in observation.items:
                    target_id = str(item.target_id)
                    if target_id in already:
                        continue
                    extra.append(target_id)
                    if len(extra) >= ceiling:
                        break
                picked = picked + extra
                self.sent += len(extra)
                self.forced += len(extra)
                self._sent_ids.update(extra)
        self.decisions += 1
        return tuple(picked)

    def summary(self):
        return {"cap": self.cap, "floor": self.floor, "sent": self.sent,
                "forced": self.forced, "dropped": self.dropped}
