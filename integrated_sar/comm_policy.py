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
    step: int
    current_time: float
    items: tuple[SendItem, ...]


class CommSendPolicy:
    """Base class: a policy names itself and reports its parameters verbatim."""

    name = "abstract"

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


POLICIES: dict[str, type[CommSendPolicy]] = {
    policy.name: policy
    for policy in (
        ImmediatePolicy,
        PeriodicPolicy,
        EventPolicy,
        EventMaxWaitPolicy,
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
    """The policy menu, for a protocol document that must be reviewable before any run."""
    return [
        {
            "name": policy.name,
            "parameters": policy().parameters(),
            "doc": " ".join((policy.__doc__ or "").split()),
        }
        for policy in POLICIES.values()
    ]
