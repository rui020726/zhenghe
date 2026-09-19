"""The state a learned sender is allowed to see, in one place and named.

Keeping this separate from the policy classes is the point: the feature vector is the audit
surface. A controller that predicts well because it saw a robot's position, a receiver's
belief, or the task identifier is not solving the communication problem, so every input here
is either a property of the message the sender is already holding or a property of the
sender's own ledger and airtime.

Nothing in this module imports torch. The pilot's network lives with the pilot; what is
production-relevant is the boundary -- these fields, in this order, are what a sender may read.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .comm_policy import SendItem, SendObservation

# Order matters: this is the observation layout a checkpoint was trained against. Adding a
# field means a new layout name, not a silent reshape of an old one.
FEATURE_LAYOUT = "comm-send-state/1"

FEATURE_NAMES: tuple[str, ...] = (
    "age_steps",
    "since_last_emission_time",
    "sent_count",
    "pending_event",
    "changed_view",
    "first_emission",
    "stage_changed",
    "requirements_changed",
    "members_changed",
    "progress_delta_abs",
    "revision_delta",
    "remaining_time_over_horizon",
    "urgency_now",
    "has_future_requirements",
    "queue_backlog_targets",
    "budget_remaining_fraction",
    "capacity_remaining_fraction",
    "subscribers",
    "step_over_horizon",
)

#: What "near the deadline" means for the feature scale, and the episode length used to
#: normalise the position in the episode. Both are declared constants of the *state
#: representation*, not tuned per run: changing either invalidates a checkpoint.
URGENCY_SCALE_STEPS = 50.0
EPISODE_HORIZON_STEPS = 1000.0


def features(
    observation: SendObservation,
    item: SendItem,
    *,
    horizon: float = EPISODE_HORIZON_STEPS,
) -> tuple[float, ...]:
    """The named feature vector for one candidate at one decision point.

    ``since_last_emission_time`` and ``age_steps`` are the sender's own silence clocks. The
    change flags compare the view it would send with the view it last sent. The budget and
    capacity terms are what the executor will allow now, never a forecast of future arrivals.
    """
    remaining = float(item.remaining_time)
    budget_fraction = 1.0
    if observation.budget_bytes_per_episode:
        budget_fraction = max(
            0.0,
            min(1.0, float(observation.remaining_budget_bytes) / observation.budget_bytes_per_episode),
        )
    capacity_fraction = 1.0
    if observation.capacity_remaining is not None:
        capacity_fraction = float(observation.capacity_remaining) / float(
            max(1, observation.capacity_remaining + 1)
        )
    return (
        float(item.age_steps),
        float(item.since_last_emission_time or 0.0),
        float(item.sent_count),
        1.0 if item.pending_event else 0.0,
        1.0 if item.changed_view else 0.0,
        1.0 if item.first_emission else 0.0,
        1.0 if item.stage_changed else 0.0,
        1.0 if item.requirements_changed else 0.0,
        1.0 if item.members_changed else 0.0,
        abs(float(item.progress_delta)),
        float(item.revision_delta),
        max(-1.0, min(1.0, remaining / URGENCY_SCALE_STEPS)),
        1.0 if 0.0 <= remaining <= URGENCY_SCALE_STEPS / 2.0 else 0.0,
        1.0 if item.has_future_requirements else 0.0,
        float(observation.backlog_targets),
        budget_fraction,
        capacity_fraction,
        float(observation.subscribers),
        min(1.0, float(observation.step) / max(1.0, float(horizon))),
    )


class LearnedSendPolicy:
    """A policy object that asks an injected function what to send, in what order.

    ``orders_selection`` is True: a controller that ranks its own candidates is compared
    against ``event_priority``, which is given the same privilege, and not against a rule that
    was trimmed by the transport. The injected function returns the targets it wants on the
    air, best first; it may be a sampled action, a greedy argmax, or a plain python rule --
    nothing here assumes a framework.
    """

    name = "learned"
    orders_selection = True

    def __init__(
        self,
        decide_fn: Callable[[SendObservation], list[str]] | None,
        *,
        horizon: float = EPISODE_HORIZON_STEPS,
    ) -> None:
        self._decide_fn = decide_fn
        self.horizon = float(horizon)
        self.last_features: dict[str, tuple[float, ...]] = {}
        self.decisions = 0

    def parameters(self) -> dict[str, Any]:
        return {"horizon": self.horizon, "feature_layout": FEATURE_LAYOUT}

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        if self._decide_fn is None:
            raise RuntimeError(
                "the learned policy was selected but no decision function is attached; build "
                "the policy with a controller and pass it to CommRelay.attach_policy()"
            )
        self.last_features = {
            item.target_id: features(observation, item, horizon=self.horizon)
            for item in observation.items
        }
        self.decisions += 1
        chosen = self._decide_fn(observation)
        known = set(self.last_features)
        ordered = [target for target in chosen if target in known]
        return tuple(dict.fromkeys(ordered))


def feature_matrix(state: Mapping[str, tuple[float, ...]]) -> list[list[float]]:
    """The stored features as a matrix, ordered by feature name for stable columns."""
    return [[state[key][index] for index in range(len(FEATURE_NAMES))] for key in sorted(state)]
