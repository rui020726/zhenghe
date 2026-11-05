"""``dedup_immediate``: send on first discovery or a substantive view change, never rehash.

Why this arm exists. P4's learned sender wasted 11 of 91 sends re-broadcasting targets that
were already on the air, and the P5 probe showed the score is essentially "how many DISTINCT
targets the budget reached". A rule that never repeats is therefore the right low-cost
strong baseline: it isolates "does the learner know what it already said" from "can it rank".

The definition is deliberately narrower than ``immediate``:

    immediate      -> every candidate with ``pending_event``
    dedup_immediate -> every candidate that is either
                       (a) on the air for the first time, or
                       (b) has had a substantive view change since it was last sent

"Substantive" is not "any flag moved". A confidence tweak or a revision bump with identical
members and requirements is not worth a packet, and treating it as one would collapse this
arm back into ``immediate``. The rule reads only sender-visible fields on ``SendItem``, so it
obeys the same belief boundary as the learner (Invariant 9): nothing here looks at world
truth, robot positions, or another node's belief.

The 0.7 delivery probability is why (b) exists at all: a dropped first packet is worth
resending once the view genuinely changes, but not by re-sending the identical payload.
"""

from __future__ import annotations

from integrated_sar.comm_policy import CommSendPolicy, SendItem, SendObservation


def sender_namespace(node):
    """An identifier for the sending node.

    ``dedup_immediate`` is one sender, so its own memory is enough. The function exists so an
    explicit namespace can be passed in tests without touching the policy body.
    """
    return str(node)


class DedupImmediatePolicy(CommSendPolicy):
    """First sighting, or a material change since the last packet. Never a blind repeat."""

    name = "dedup_immediate"
    #: The rule ranks nothing: the transport keeps its own trim order, so this arm is judged on
    #: which targets it asks for and not on queue-pushing skill.
    orders_selection = False

    def __init__(self, *, stage_only: bool = True) -> None:
        #: Stage transitions only. The broader "stage or requirements or members" form was
        #: measured at 138 sends over 24 targets in one world, which is not a cheap arm.
        self.stage_only = bool(stage_only)
        #: Evidence of what this sender has asked for; the decision reads the relay's own
        #: ``first_emission`` / ``changed_view`` instead, so a disagreement shows up here.
        self._seen: set[str] = set()
        self.sends = 0
        self.suppressed_repeats = 0

    def _material(self, item: SendItem) -> bool:
        """A stage transition: the one structural change that always needs a packet.

        Measured, not assumed. Including ``requirements_changed`` and ``members_changed`` made
        this arm send 138 times over 24 targets in one world -- the member list churns on every
        assignment, so "members moved" is not rare information. Under 0.7 delivery a stage
        transition is the case worth paying for; a re-shuffled member list is not. The
        alternative rule (send only on ``first_emission``) is also kept behind a flag, because
        it cannot recover a dropped packet at all.
        """
        if self.stage_only:
            return bool(item.stage_changed)
        return bool(item.stage_changed or item.requirements_changed or item.members_changed)

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        chosen: list[str] = []
        for item in observation.items:
            target = item.target_id
            # ``first_emission`` is the relay's own "this sender has never emitted for this
            # target"; ``changed_view`` is its own comparison against the last packet. Both are
            # computed from the sender's ledger, not re-derived here.
            if item.first_emission:
                chosen.append(target)
            elif item.changed_view and self._material(item):
                chosen.append(target)
            else:
                self.suppressed_repeats += 1
            self._seen.add(target)
        self.sends += len(chosen)
        return tuple(chosen)

    def parameters(self) -> dict:
        return {
            "rule": "first_emission_or_stage_change",
            "material": "stage_changed" if self.stage_only else
                        "stage_changed | requirements_changed | members_changed",
            "requires": "the relay's own changed_view",
            "orders_selection": False,
            "sends": self.sends,
            "suppressed_repeats": self.suppressed_repeats,
        }


def build_dedup_policy():
    """The policy object the runner attaches; kept as a function so imports stay lazy."""
    return DedupImmediatePolicy()
