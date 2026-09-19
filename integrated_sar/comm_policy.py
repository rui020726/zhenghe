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
from typing import Any, Sequence


@dataclass(frozen=True)
class SendItem:
    """One target the sending node could put on the air this step.

    ``pending_event`` means an accepted detection has not been emitted yet -- the only
    condition the historical relay acted on. ``changed_view`` is the umbrella: the sending
    node's current view differs from the last packet it emitted for this target, for any
    reason. The narrower flags say *which* part moved, so a policy can decide that a roster
    edit is worth a packet and a confidence tweak is not.
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


POLICIES: dict[str, type[CommSendPolicy]] = {
    policy.name: policy
    for policy in (
        ImmediatePolicy,
        PeriodicPolicy,
        EventPolicy,
        EventMaxWaitPolicy,
        PriorityEventPolicy,
        HoldToBoundPolicy,
        RandomSendPolicy,
    )
}


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
