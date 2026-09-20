"""When the search side puts a task message on the uplink.

The relay (:class:`integrated_sar.comm.CommRelay`) is the only channel from the
registry-side task view into a rescue robot's belief. Until now the relay emitted exactly
one packet per accepted detection, immediately, and every later change to that task reached
robots through TaskEnv's own sensing and gossip instead. That makes "how often should task
information be sent, and when" an unanswerable question, so the relay now asks a policy.

A policy is a pure function from what the *sending node* holds to a set of targets to emit
this step. It never sees a robot's position, belief or identity:

* the sending node is the ground station that already runs the registry, so the task's
  stage, progress, requirements and deadline are its own state, not a receiver's truth;
* which robots already know what is *not* observable, because the uplink carries no
  acknowledgements -- that asymmetry is the whole difficulty of the problem;
* the payload is always built from the newest registry snapshot at the moment of sending,
  so a held item is superseded rather than delayed-and-stale, and a policy that sends less
  sends *later and fresher*, never "less information per packet".

:func:`immediate_policy` reproduces the historical behaviour exactly: one emission per
accepted detection, at the step that detection was accepted, no refresh traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class SendItem:
    """One target the sending node could put on the air this step.

    ``pending_event`` means an accepted detection has not been emitted yet -- the only
    condition the historical relay acted on. ``changed_view`` is the umbrella: the sending
    node's current view differs from the last packet it emitted for this target, for any
    reason. The narrower flags say *which* part moved, so a policy can decide that a roster
    edit is worth a packet and a confidence tweak is not.

    ``ack_confirmed_fraction`` and ``ack_unacked_broadcasts`` are the reverse channel's
    contribution, and both are the sender's own arithmetic over ACKs that actually arrived:
    the first is how many configured receivers have confirmed the view it last put on the
    air, the second how many emissions it has made for this target since the view last
    changed without everyone confirming. Both are ``-1`` when no ACK has ever arrived, which
    is the explicit unknown encoding -- a sender that has never been told anything must not
    read as a sender that has been told "nobody confirmed".
    """

    target_id: str
    ready_step: int
    age_steps: int
    pending_event: bool
    changed_view: bool
    first_emission: bool
    revision: int
    revision_delta: int
    stage_index: int
    stage_changed: bool
    progress: float
    progress_delta: float
    requirements_changed: bool
    members_changed: bool
    deadline: float
    remaining_time: float
    sent_count: int
    last_sent_step: int | None
    since_last_emission_time: float | None
    has_future_requirements: bool
    ack_confirmed_fraction: float = -1.0
    ack_unacked_broadcasts: int = -1


@dataclass(frozen=True)
class SendObservation:
    """Everything the sending node can act on at one decision point.

    The budget fields are what the executor still allows, not a forecast of what will fit:
    a policy that asks for more than the remaining credit is refused and stays queued, so a
    learned controller sees its own constraint rather than a privileged view of the future.
    """

    step: int
    current_time: float
    items: tuple[SendItem, ...]
    remaining_budget_bytes: float = float("inf")
    budget_bytes_per_episode: int | None = None
    capacity_remaining: int | None = None
    backlog_targets: int = 0
    subscribers: int = 1


def transport_trim_key(item: SendItem) -> tuple[float, int, str]:
    """The one order the transport trims an unranked selection by: slack, staleness, id.

    Published because a rule that has to reproduce the transport's own emission sequence must not
    copy this key into its own body: two definitions of one order drift apart, and the drift
    shows up as a protocol change nobody made.
    """
    return (item.remaining_time, -item.age_steps, item.target_id)


def transport_order(items: Iterable[SendItem]) -> list[SendItem]:
    """The candidates in the order the transport would take them."""
    return sorted(items, key=transport_trim_key)


class CommSendPolicy:
    """Base class: a policy names itself and reports its parameters verbatim.

    ``orders_selection`` says whether the executor should keep the order this policy returns.
    Rules that do not claim an ordering are trimmed by the transport in one fixed order, so
    they are compared on what they ask to say rather than on queue-pushing skill.
    """

    name = "abstract"
    orders_selection = False

    def parameters(self) -> dict[str, Any]:
        return {}

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        """The targets to emit this step; the transport trims them to any budget."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.parameters()})"


class ImmediatePolicy(CommSendPolicy):
    """Emit each accepted detection at once, and nothing else (the historical relay)."""

    name = "immediate"

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        return tuple(item.target_id for item in observation.items if item.pending_event)


class PeriodicPolicy(CommSendPolicy):
    """Full table broadcast on a fixed clock: every ``interval`` steps, everything known.

    Slots are keyed to the global step counter, not to when a target was discovered, so the
    schedule is visible in advance and cannot be tuned per target. Because every slot carries
    the newest snapshot, a periodic policy also recovers a packet the uplink dropped.
    """

    name = "periodic"

    def __init__(self, *, interval: int = 5) -> None:
        interval = int(interval)
        if interval < 1:
            raise ValueError(f"a periodic interval must be >= 1, got {interval}")
        self.interval = interval

    def parameters(self) -> dict[str, Any]:
        return {"interval": self.interval}

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        if observation.step % self.interval != 0:
            return ()
        return tuple(item.target_id for item in observation.items)


def _changed(item: SendItem) -> bool:
    return (
        item.pending_event
        or item.stage_changed
        or item.requirements_changed
        or item.members_changed
    )


class EventPolicy(CommSendPolicy):
    """Emit when a fact a robot could act on changed, or when the clock says it is urgent.

    The event set is fixed here and shared by every event-based arm: a new task (its first
    emission), a stage transition, a change in the remaining capability requirement, a
    change in who is on the roster, a progress jump of at least ``progress_threshold``, and
    a task inside ``urgency_time`` of its deadline. Everything else is held: a packet whose
    payload would repeat what the last one said carries no information.
    """

    name = "event"

    def __init__(
        self,
        *,
        progress_threshold: float = 0.2,
        urgency_time: float = 25.0,
    ) -> None:
        progress_threshold = float(progress_threshold)
        if not 0.0 < progress_threshold <= 1.0:
            raise ValueError(
                f"progress_threshold must be in (0, 1], got {progress_threshold}"
            )
        if float(urgency_time) < 0.0:
            raise ValueError(f"urgency_time cannot be negative, got {urgency_time}")
        self.progress_threshold = progress_threshold
        self.urgency_time = float(urgency_time)

    def parameters(self) -> dict[str, Any]:
        return {"progress_threshold": self.progress_threshold, "urgency_time": self.urgency_time}

    def urgent_now(self, item: SendItem) -> bool:
        return 0.0 <= item.remaining_time <= self.urgency_time

    def triggered(self, item: SendItem) -> bool:
        if item.first_emission or _changed(item):
            return True
        if item.progress_delta >= self.progress_threshold:
            return True
        if not self.urgent_now(item):
            return False
        # "Near the deadline" is a crossing, not a state: say it on the step the horizon is
        # reached. Re-announcing every following step would be a periodic policy wearing an
        # event policy's name, and it would double-count the same news.
        return (
            item.since_last_emission_time is None
            or item.remaining_time + item.since_last_emission_time > self.urgency_time
        )

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        return tuple(item.target_id for item in observation.items if self.triggered(item))


class EventMaxWaitPolicy(EventPolicy):
    """Event triggers with a deadline on silence: never hold a change for more than
    ``max_wait_steps`` steps, and never hold an urgent task for more than ``max_urgent_steps``.

    The bound is what keeps a rule that "waits for something worth saying" from becoming an
    occupancy lease on a robot's knowledge: an emergency can only be hidden for a declared
    number of steps, so the worst case stays analysable without the policy being lucky.
    """

    name = "event_maxwait"

    def __init__(
        self,
        *,
        max_wait_steps: int = 10,
        max_urgent_steps: int = 3,
        progress_threshold: float = 0.2,
        urgency_time: float = 25.0,
    ) -> None:
        super().__init__(
            progress_threshold=progress_threshold,
            urgency_time=urgency_time,
        )
        max_wait_steps = int(max_wait_steps)
        max_urgent_steps = int(max_urgent_steps)
        if max_wait_steps < 1:
            raise ValueError(f"max_wait_steps must be >= 1, got {max_wait_steps}")
        if max_urgent_steps < 0:
            raise ValueError(f"max_urgent_steps cannot be negative, got {max_urgent_steps}")
        self.max_wait_steps = max_wait_steps
        self.max_urgent_steps = max_urgent_steps

    def parameters(self) -> dict[str, Any]:
        values = super().parameters()
        values.update(
            {"max_wait_steps": self.max_wait_steps, "max_urgent_steps": self.max_urgent_steps}
        )
        return values

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        selected = []
        for item in observation.items:
            urgent = self.urgent_now(item)
            overdue = item.age_steps >= (
                self.max_urgent_steps if urgent else self.max_wait_steps
            )
            if overdue or self.triggered(item):
                selected.append(item.target_id)
        return tuple(selected)


class PriorityEventPolicy(EventPolicy):
    """The event triggers, and when the airtime will not cover them all, say the most
    actionable first: a score of urgency, how long the change has been held, and how much the
    packet would actually say.

    With unbounded airtime this emits exactly what ``event`` emits -- the ordering only matters
    once something has to be refused. That is the point: it is the strongest simple rule a
    learned controller has to beat under a budget, and it is a rule, not a ceiling, because
    every term in the score is visible state.
    """

    name = "event_priority"
    orders_selection = True

    #: Three pre-registered quantities, each normalised to roughly [0, 1], summed with equal
    #: weight. Nothing here was tuned against a result: urgency is 1/(1+slack), staleness is
    #: held-steps over 50 (a command lifetime), and information content counts a fresh
    #: discovery, a stage or requirement change, or a full progress jump as one unit each.
    @staticmethod
    def score(item: SendItem) -> float:
        urgency = 1.0 / (1.0 + max(0.0, item.remaining_time) / 50.0)
        staleness = min(1.0, item.age_steps / 50.0)
        information = (
            1.0 if item.pending_event else 0.0
        ) + (
            1.0 if (item.stage_changed or item.requirements_changed or item.members_changed) else 0.0
        ) + min(1.0, abs(item.progress_delta) / 0.5)
        return urgency + staleness + information

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        triggered = [item for item in observation.items if self.triggered(item)]
        triggered.sort(key=lambda item: (-self.score(item), item.target_id))
        return tuple(item.target_id for item in triggered)


class HoldToBoundPolicy(CommSendPolicy):
    """Ask for nothing, ever, and let the shared silence bound do all the talking.

    This is the floor of the design space, and it is a *rule*, not an ablation switch: the
    bound lives in the transport, so an empty decision here is not radio silence, it is "only
    the protocol's own deadline sends". A learned controller that merely learned to hold has
    become this arm, and the only way to see that is to run it.
    """

    name = "hold_to_bound"

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        return ()


#: The declared silence bound, as numbers an arm can be held to. ``hold_to_bound`` and
#: ``immediate`` never see them: the transport enforces them on every arm. The arms below ask
#: for the same timings themselves, which is what a transport that does not speak for everyone
#: needs in order to keep the comparison honest.
BOUND_MAX_WAIT_STEPS = 10
BOUND_URGENT_WAIT_STEPS = 3
BOUND_URGENCY_TIME = 25.0

FIRST_SEND_PLUS_BOUND = f"first_send_plus_bound_{BOUND_URGENT_WAIT_STEPS}_{BOUND_MAX_WAIT_STEPS}"
BOUND_ONLY = f"bound_only_{BOUND_URGENT_WAIT_STEPS}_{BOUND_MAX_WAIT_STEPS}"


class SilenceBoundPolicy(CommSendPolicy):
    """Plumbing for arms that *request* the declared bound instead of being spoken for by it.

    The three numbers are the same 3/10/25 the transport has always enforced, and the arm's
    registered name quotes two of them, so the constructor refuses a parameter set the name does
    not describe. An arm whose label says 3/10 and whose body says 4/20 would otherwise be a
    protocol change nobody made.
    """

    orders_selection = True

    def __init__(
        self,
        *,
        max_wait_steps: int = BOUND_MAX_WAIT_STEPS,
        urgent_wait_steps: int = BOUND_URGENT_WAIT_STEPS,
        urgency_time: float = BOUND_URGENCY_TIME,
    ) -> None:
        max_wait_steps = int(max_wait_steps)
        urgent_wait_steps = int(urgent_wait_steps)
        if max_wait_steps < 1 or urgent_wait_steps < 0:
            raise ValueError(
                f"a silence bound needs max_wait_steps >= 1 and urgent_wait_steps >= 0, got "
                f"{max_wait_steps} and {urgent_wait_steps}"
            )
        if float(urgency_time) < 0.0:
            raise ValueError(f"urgency_time cannot be negative, got {urgency_time}")
        quoted = f"{urgent_wait_steps}_{max_wait_steps}"
        if not self.name.endswith(quoted):
            raise ValueError(
                f"{type(self).__name__} is registered as {self.name!r}, which quotes a different "
                f"bound than the one it was given ({quoted!r}); an arm may not be named for a "
                "timing it does not obey"
            )
        self.max_wait_steps = max_wait_steps
        self.urgent_wait_steps = urgent_wait_steps
        self.urgency_time = float(urgency_time)

    def parameters(self) -> dict[str, Any]:
        return {
            "max_wait_steps": self.max_wait_steps,
            "urgent_wait_steps": self.urgent_wait_steps,
            "urgency_time": self.urgency_time,
        }

    def urgent_now(self, item: SendItem) -> bool:
        return 0.0 <= item.remaining_time <= self.urgency_time

    def wait_limit(self, item: SendItem) -> int:
        return self.urgent_wait_steps if self.urgent_now(item) else self.max_wait_steps

    def overdue(self, item: SendItem) -> bool:
        """Whether this held change has run out the bound. Same predicate the transport uses."""
        return int(item.age_steps) >= self.wait_limit(item)


class BoundOnlyPolicy(SilenceBoundPolicy):
    """Say nothing except what the declared bound has run out on, and say it as a request.

    Under a transport that enforces the bound, this is what an empty decision does by default.
    Under a transport that does not, it has to ask -- and asking is the point: the same timings
    are then available to a learned controller without a hidden mechanism handing them out, and
    the arm that asks for exactly these is the floor the controller has to beat.
    """

    name = BOUND_ONLY

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        return tuple(item.target_id for item in observation.items if self.overdue(item))


class FirstSendPlusBoundPolicy(SilenceBoundPolicy):
    """Report news as it arrives, and refresh anything the declared bound has run out on.

    The two bands are requested in the order the transport used to assemble them -- its own
    trim order for the news it asked for, then the overdue changes behind it -- because when the
    credit cannot pay for everything, position in that list decides who speaks. An arm that is
    meant to be the same *behaviour* under a different protocol has to reproduce that order, not
    merely the same set.
    """

    name = FIRST_SEND_PLUS_BOUND

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        asked = [item for item in observation.items if item.pending_event]
        ranked = transport_order(asked)
        following = [
            item.target_id
            for item in observation.items
            if self.overdue(item) and item.target_id not in {ranked_target.target_id for ranked_target in ranked}
        ]
        return tuple(item.target_id for item in ranked) + tuple(following)


class SendOncePolicy(CommSendPolicy):
    """One report per accepted detection, and no refresh traffic at all.

    This is what the relay did before any bound existed, and the cheapest arm that still tells
    the truth about a discovery: the receiver learns a task exists and never learns how it has
    moved. Refreshing is not this arm's job, so nothing speaks for it when a stage changes.
    """

    name = "send_once"

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        return tuple(item.target_id for item in observation.items if item.pending_event)


class AlwaysHoldPolicy(CommSendPolicy):
    """Radio silence, chosen rather than imposed.

    Under the shared bound this arm was impossible: the transport spoke for it every 10 steps.
    Here it can actually do what it says, which is why it is the control that tells a reader
    what the uplink was worth -- and why a task left unknown is allowed to fail.
    """

    name = "always_hold"

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        return ()


class RandomSendPolicy(CommSendPolicy):
    """Flip a coin per candidate, on a seed of its own.

    A trained sender has to beat more than "not the always-sender": an untrained network emits
    a sparse, structured-looking pattern by accident. This arm is the unstructured sparse
    control, and it is reproducible because the coin is seeded and per-episode -- the same
    property the transport's loss rolls have.
    """

    name = "random_send"

    def __init__(self, *, send_probability: float = 0.5, seed: int = 0) -> None:
        if not 0.0 <= float(send_probability) <= 1.0:
            raise ValueError(
                f"send_probability must stay in [0, 1], got {send_probability}"
            )
        self.send_probability = float(send_probability)
        self.seed = int(seed)
        self._rng: Random | None = None
        self._last_step = -1

    def parameters(self) -> dict[str, Any]:
        return {"send_probability": self.send_probability, "seed": self.seed}

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        from random import Random

        step = int(observation.step)
        if self._rng is None or step <= self._last_step:
            # A relay is reused across episodes and calls reset(), which the policy does not
            # see. Re-seeding when the step counter goes back is how this arm restarts its
            # stream, so episode N always draws the same coins as episode N did before.
            self._rng = Random(self.seed)
        self._last_step = step
        return tuple(
            item.target_id
            for item in observation.items
            if self._rng.random() < self.send_probability
        )


class _AckAwarePolicy(CommSendPolicy):
    """Plumbing for rules that read the reverse channel.

    The two facts an ACK rule may use are on every :class:`SendItem` already, so a rule stays
    a pure function of its observation. ``unknown`` is what a sender that has never received
    an ACK sees, and it is deliberately *not* zero: "nobody has confirmed" and "I have not
    been told anything" are different states, and a rule that confuses them would retry a
    view that was confirmed long ago.
    """

    #: The value ``ack_confirmed_fraction`` carries when no ACK has arrived.
    UNKNOWN = -1.0

    @staticmethod
    def confirmed(item: SendItem) -> float | None:
        value = float(item.ack_confirmed_fraction)
        return None if value < 0.0 else value

    @staticmethod
    def unacked(item: SendItem) -> int | None:
        value = int(item.ack_unacked_broadcasts)
        return None if value < 0 else value


class RetryUntilAckedPolicy(_AckAwarePolicy):
    """Say each view again until every configured receiver has confirmed it, up to k attempts.

    The simplest rule that can only exist with a reverse channel: without an ACK there is no
    "everyone has it" to wait for, and without the cap there is no bound on what a lost
    acknowledgement costs. The first emission of a view always goes out; the retries are
    driven by the sender's own unacknowledged-broadcast count, never by a receiver's state.
    """

    name = "retry_until_acked"

    def __init__(self, *, max_attempts: int = 5) -> None:
        max_attempts = int(max_attempts)
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self.max_attempts = max_attempts

    def parameters(self) -> dict[str, Any]:
        return {"max_attempts": self.max_attempts}

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        chosen: list[str] = []
        for item in observation.items:
            if item.first_emission or item.pending_event:
                chosen.append(item.target_id)
                continue
            confirmed = self.confirmed(item)
            unacked = self.unacked(item)
            if confirmed is None:
                # No ACK has ever arrived for this view: the rule cannot tell "unconfirmed"
                # from "unknown", so it falls back to its attempt budget rather than
                # pretending it knows.
                if unacked is None or unacked < self.max_attempts:
                    chosen.append(item.target_id)
                continue
            if confirmed >= 1.0:
                continue
            if unacked is None or unacked < self.max_attempts:
                chosen.append(item.target_id)
        return tuple(chosen)


class EventRetryUntilAckedPolicy(_AckAwarePolicy):
    """The event triggers, plus a bounded retry for the views they have not confirmed.

    This is the arm the brief names as the realistic ACK rule: it does not resend everything,
    it resends what a receiver has not confirmed yet, and it stops when the event set is
    exhausted and the confirmations are in. It shares the event trigger with ``event`` so the
    only difference between the two arms is the retry band.
    """

    name = "event_retry_until_acked"
    orders_selection = True

    def __init__(
        self,
        *,
        max_attempts: int = 5,
        progress_threshold: float = 0.2,
        urgency_time: float = 25.0,
    ) -> None:
        max_attempts = int(max_attempts)
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self.max_attempts = max_attempts
        self._event = EventPolicy(
            progress_threshold=progress_threshold, urgency_time=urgency_time
        )

    def parameters(self) -> dict[str, Any]:
        values = self._event.parameters()
        values.update({"max_attempts": self.max_attempts})
        return values

    def triggered(self, item: SendItem) -> bool:
        return self._event.triggered(item)

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        triggered = [item for item in observation.items if self.triggered(item)]
        retries = []
        for item in observation.items:
            if item in triggered:
                continue
            confirmed = self.confirmed(item)
            if confirmed is not None and confirmed >= 1.0:
                continue
            unacked = self.unacked(item)
            if unacked is not None and unacked >= self.max_attempts:
                continue
            retries.append(item)
        ordered = transport_order(triggered) + transport_order(retries)
        return tuple(item.target_id for item in ordered)


class ExponentialBackoffPolicy(_AckAwarePolicy):
    """Retry on a doubling clock, capped, until the view is confirmed.

    The backoff interval is measured in emissions for this target, so the clock is the
    sender's own history rather than the global step counter: a target that changed rarely
    does not inherit a schedule from a neighbour that changed often.
    """

    name = "exponential_backoff"

    def __init__(self, *, base_interval: int = 2, max_interval: int = 32) -> None:
        base_interval = int(base_interval)
        max_interval = int(max_interval)
        if base_interval < 1:
            raise ValueError(f"base_interval must be >= 1, got {base_interval}")
        if max_interval < base_interval:
            raise ValueError(
                f"max_interval must be >= base_interval, got {max_interval} < {base_interval}"
            )
        self.base_interval = base_interval
        self.max_interval = max_interval

    def parameters(self) -> dict[str, Any]:
        return {"base_interval": self.base_interval, "max_interval": self.max_interval}

    def _interval(self, attempts: int) -> int:
        """The gap after the ``attempts``-th emission, doubling up to the cap."""
        if attempts <= 0:
            return self.base_interval
        return min(self.max_interval, self.base_interval * (2 ** max(0, attempts - 1)))

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        chosen: list[str] = []
        for item in observation.items:
            if item.first_emission or item.pending_event:
                chosen.append(item.target_id)
                continue
            confirmed = self.confirmed(item)
            if confirmed is not None and confirmed >= 1.0:
                continue
            unacked = self.unacked(item)
            attempts = 0 if unacked is None else unacked
            since = item.since_last_emission_time
            if since is None:
                chosen.append(item.target_id)
                continue
            if float(since) >= float(self._interval(attempts)):
                chosen.append(item.target_id)
        return tuple(chosen)


class DeadlineTighteningRetryPolicy(_AckAwarePolicy):
    """Retry spacing shrinks as the deadline approaches, capped by an attempt budget.

    The explicit version of "matter more, say it more often": the gap between retries is the
    base interval scaled by how much slack is left, so a task inside its urgency window is
    retried every step and one with months of slack keeps its spacing. Every term is visible
    state -- remaining time and the sender's own attempt count -- so the rule stays a rule.
    """

    name = "deadline_tightening_retry"

    def __init__(
        self,
        *,
        max_attempts: int = 10,
        urgency_time: float = 25.0,
        base_interval: int = 8,
    ) -> None:
        max_attempts = int(max_attempts)
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        if float(urgency_time) < 0.0:
            raise ValueError(f"urgency_time cannot be negative, got {urgency_time}")
        if int(base_interval) < 1:
            raise ValueError(f"base_interval must be >= 1, got {base_interval}")
        self.max_attempts = max_attempts
        self.urgency_time = float(urgency_time)
        self.base_interval = int(base_interval)

    def parameters(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "urgency_time": self.urgency_time,
            "base_interval": self.base_interval,
        }

    def _gap(self, item: SendItem) -> int:
        """Steps to wait before the next retry: 1 when urgent, growing with slack."""
        slack = float(item.remaining_time)
        if 0.0 <= slack <= self.urgency_time:
            return 1
        # One extra step of waiting per full urgency window of slack, capped at the base.
        windows = max(0.0, slack) / max(1.0, self.urgency_time)
        return max(1, min(self.base_interval, int(windows) + 1))

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        chosen: list[str] = []
        for item in observation.items:
            if item.first_emission or item.pending_event:
                chosen.append(item.target_id)
                continue
            confirmed = self.confirmed(item)
            if confirmed is not None and confirmed >= 1.0:
                continue
            unacked = self.unacked(item)
            if unacked is not None and unacked >= self.max_attempts:
                continue
            since = item.since_last_emission_time
            if since is None or float(since) >= float(self._gap(item)):
                chosen.append(item.target_id)
        return tuple(chosen)


_RULE_CLASSES: tuple[type[CommSendPolicy], ...] = (
    ImmediatePolicy,
    PeriodicPolicy,
    EventPolicy,
    EventMaxWaitPolicy,
    PriorityEventPolicy,
    HoldToBoundPolicy,
    BoundOnlyPolicy,
    FirstSendPlusBoundPolicy,
    SendOncePolicy,
    AlwaysHoldPolicy,
    RandomSendPolicy,
    # The ACK-aware rules. They read the same two reverse-channel facts the ``/2`` features
    # read, so a rule and a learned sender compete on the same information rather than one of
    # them holding a private channel.
    RetryUntilAckedPolicy,
    EventRetryUntilAckedPolicy,
    ExponentialBackoffPolicy,
    DeadlineTighteningRetryPolicy,
)

POLICIES: dict[str, type[CommSendPolicy]] = {
    policy.name: policy for policy in _RULE_CLASSES
}

_RULE_NAMES = [policy.name for policy in _RULE_CLASSES]
if len(set(_RULE_NAMES)) != len(_RULE_NAMES):
    # A colliding name would drop an arm out of the catalogue silently, and a dropped arm is a
    # comparison that quietly stops including the thing it was supposed to include.
    repeated = sorted(name for name in set(_RULE_NAMES) if _RULE_NAMES.count(name) > 1)
    raise ImportError(f"two communication rules claim the same name: {repeated}")


def parse_parameters(spec: str | None) -> dict[str, Any]:
    """Parse ``key=value`` pairs: ``"interval=5"`` or ``"urgency_time=20,max_wait=6"``.

    Integers stay integers, a bare decimal is a float, everything else is an error: a typo
    in a pre-registered protocol must fail loudly rather than quietly run at a default.
    """
    parameters: dict[str, Any] = {}
    text = str(spec or "").strip()
    if not text:
        return parameters
    for token in text.split(","):
        key, separator, value = token.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key:
            raise ValueError(
                f"--comm-policy-params expects key=value pairs, got {token!r} in {spec!r}"
            )
        if not value:
            raise ValueError(f"--comm-policy-params key {key!r} has no value")
        try:
            parameters[key] = int(value)
        except ValueError:
            try:
                parameters[key] = float(value)
            except ValueError:
                raise ValueError(
                    f"--comm-policy-params value for {key!r} must be a number, got {value!r}"
                ) from None
    return parameters


def build_comm_policy(name: str, parameters: dict[str, Any] | str | None = None) -> CommSendPolicy:
    """Instantiate a named policy; unknown names and unknown keys are hard errors."""
    normalized = str(name or "").strip().lower()
    if normalized == "learned":
        # The learned controller is not a rule and its weights are not a string parameter: it
        # is injected onto the relay (CommRelay.attach_policy) after construction. Building the
        # placeholder here is what makes "learned" a legal, label-visible arm; running it
        # without injecting a decision function fails loudly instead of silently sending all.
        from .comm_learning import LearnedSendPolicy

        return LearnedSendPolicy(None)
    if normalized not in POLICIES:
        raise ValueError(
            f"unknown communication policy {name!r}; available: {', '.join(sorted(POLICIES))}"
        )
    if isinstance(parameters, str) or parameters is None:
        values = parse_parameters(parameters)
    else:
        values = dict(parameters)
    policy = POLICIES[normalized]()
    signature = _accepted_keywords(policy)
    unknown = sorted(set(values) - set(signature))
    if unknown:
        raise ValueError(
            f"policy {normalized!r} accepts {sorted(signature)}, not {unknown}; "
            f"parameters were {values}"
        )
    return POLICIES[normalized](**values)


def _accepted_keywords(policy: CommSendPolicy) -> dict[str, str]:
    init = getattr(type(policy), "__init__", None)
    if init is None:
        return {}
    from inspect import signature as call_signature

    parameters = call_signature(init).parameters
    return {
        key: str(value.annotation)
        for key, value in parameters.items()
        if key != "self" and value.kind in (value.KEYWORD_ONLY, value.POSITIONAL_OR_KEYWORD)
    }


def policy_descriptors() -> Sequence[dict[str, Any]]:
    """The rule menu, for a protocol document that must be reviewable before any run.

    "learned" is deliberately absent: it has no parameters of its own to publish, only a
    network and a checkpoint that the pilot records separately.
    """
    return [
        {
            "name": policy.name,
            "parameters": policy().parameters(),
            "doc": " ".join((policy.__doc__ or "").split()),
        }
        for policy in POLICIES.values()
    ]
