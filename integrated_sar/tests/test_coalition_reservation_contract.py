"""The coalition reservation contract, and where the allocator is allowed to read it.

A ``CoalitionReservation`` names one robot, one task, one capability slot and one
expiry. It is the only coordination output that is *not* a per-pair score, so the
properties that make the rest of the interface safe have to be re-established for
it:

* the field is off by default, and an advice without reservations behaves exactly
  like one from before the field existed;
* a commitment is agent-specific -- it can move the robot that holds it and no one
  else, which is what keeps a belief-derived number from reaching a robot that
  never received the message (invariant 9);
* it lands on legal task actions only, before the environment's hard mask, so it
  can reorder a decision but never legalise one (invariant 6);
* it rides the ordinary bounded residual channel, so the traced ``bias`` vector
  still satisfies ``final = ctde + offset + bias``.
"""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace
from typing import Any

from integrated_sar.contracts import CoalitionReservation, CoordinationAdvice

HAS_TORCH = importlib.util.find_spec("torch") is not None

ACTION_MAP = (("depot", None), ("task", 7), ("task", 8))


def reservation(
    *,
    agent_id: str = "0",
    task_id: str = "target-a",
    ability_slot: int = 0,
    slot_dimension: int = 0,
    task_revision: int = 1,
    issued_at: float = 0.0,
    valid_until: float = 10.0,
    score: float = 0.5,
) -> CoalitionReservation:
    return CoalitionReservation(
        agent_id=agent_id,
        task_id=task_id,
        ability_slot=ability_slot,
        slot_dimension=slot_dimension,
        task_revision=task_revision,
        issued_at=issued_at,
        valid_until=valid_until,
        score=score,
    )


class ReservationContractTests(unittest.TestCase):
    """The field is additive, keyword-only, and validated at construction."""

    def test_an_advice_without_reservations_keeps_the_old_bias_exactly(self) -> None:
        advice = CoordinationAdvice(
            revision=1,
            generated_at=0.0,
            valid_until=2.0,
            task_bias={"t": 0.1},
            pair_bias={("a", "t"): 0.2},
            agent_task_bias={("a", "t"): 0.3},
        )
        self.assertEqual(advice.reservations, {})
        # The reservation is a fourth channel, not a term inside ``bias_for``: the
        # sum the pre-existing code relies on must not silently grow a member.
        self.assertAlmostEqual(advice.bias_for("a", "t"), 0.6)

    def test_the_reservation_field_is_keyword_only(self) -> None:
        # Nine positional arguments: the ninth must not bind to ``reservations``.
        with self.assertRaises(TypeError):
            CoordinationAdvice(1, 0.0, 2.0, {}, {}, frozenset(), {}, 0.0, {})

    def test_a_score_outside_the_bounded_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reservation(score=1.5)
        with self.assertRaises(ValueError):
            reservation(score=-1.5)
        for bound in (-1.0, 0.0, 1.0):
            self.assertEqual(reservation(score=bound).score, bound)

    def test_an_expiry_before_the_issue_time_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reservation(issued_at=5.0, valid_until=4.0)
        self.assertEqual(reservation(issued_at=5.0, valid_until=5.0).valid_until, 5.0)

    def test_slot_indices_and_revisions_cannot_be_negative(self) -> None:
        with self.assertRaises(ValueError):
            reservation(ability_slot=-1)
        with self.assertRaises(ValueError):
            reservation(slot_dimension=-1)
        with self.assertRaises(ValueError):
            reservation(task_revision=-1)
        with self.assertRaises(ValueError):
            reservation(agent_id="")
        with self.assertRaises(ValueError):
            reservation(task_id="")

    def test_active_at_includes_the_expiry_instant(self) -> None:
        held = reservation(issued_at=0.0, valid_until=10.0)
        self.assertTrue(held.active_at(9.999))
        self.assertTrue(held.active_at(10.0))
        self.assertFalse(held.active_at(10.001))

    def test_a_reservation_without_an_explicit_id_names_itself_by_slot(self) -> None:
        held = reservation(agent_id="0", task_id="target-a", ability_slot=2)
        self.assertEqual(held.commitment_id, "0|target-a|2")
        self.assertEqual(held.created_at, held.issued_at)

    def test_an_explicit_identity_is_kept_exactly(self) -> None:
        held = CoalitionReservation(
            agent_id="0",
            task_id="target-a",
            ability_slot=0,
            slot_dimension=0,
            task_revision=1,
            issued_at=5.0,
            valid_until=10.0,
            score=0.5,
            commitment_id="rp0007",
            created_at=1.0,
        )
        self.assertEqual(held.commitment_id, "rp0007")
        self.assertEqual(held.created_at, 1.0)

    def test_a_reservation_cannot_be_created_after_it_was_issued(self) -> None:
        with self.assertRaises(ValueError):
            CoalitionReservation(
                agent_id="0",
                task_id="target-a",
                ability_slot=0,
                slot_dimension=0,
                task_revision=1,
                issued_at=5.0,
                valid_until=10.0,
                score=0.5,
                created_at=6.0,
            )

    def test_a_renewal_keeps_the_identity_and_moves_only_the_facts(self) -> None:
        first = CoalitionReservation(
            agent_id="0",
            task_id="target-a",
            ability_slot=1,
            slot_dimension=0,
            task_revision=3,
            issued_at=5.0,
            valid_until=10.0,
            score=0.4,
            commitment_id="rp0002",
            created_at=5.0,
        )
        second = first.renewed(
            issued_at=15.0, valid_until=20.0, task_revision=4, score=0.6
        )
        self.assertEqual(second.commitment_id, first.commitment_id)
        self.assertEqual(second.created_at, first.created_at)
        self.assertEqual(second.ability_slot, first.ability_slot)
        self.assertEqual(second.issued_at, 15.0)
        self.assertEqual(second.valid_until, 20.0)
        self.assertEqual(second.task_revision, 4)
        self.assertEqual(second.score, 0.6)


def _stub_env(mask: Any) -> Any:
    import numpy as np

    class _StubEnv:
        """Just enough TaskEnv for both the network and the heuristic verdict."""

        def __init__(self) -> None:
            self.current_time = 3.0
            self._inner_step_counter = 3
            self.task_dic: dict[int, Any] = {}
            self.agent_dic = {
                0: {
                    "location": np.array([0.4, 0.0]),
                    "target_pos": np.array([0.5, 0.0]),
                    "target_type": None,
                    "target_id": None,
                    "current_task": -1,
                    "abilities": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                    "status": "idle",
                    "travel_dist": 0.0,
                },
                1: {
                    "location": np.array([0.4, 0.0]),
                    "target_pos": np.array([0.5, 0.0]),
                    "target_type": None,
                    "target_id": None,
                    "current_task": -1,
                    "abilities": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                    "status": "idle",
                    "travel_dist": 0.0,
                },
            }
            self.agent_knowledge = {
                0: {
                    "task_current_status": {7: (1, 0, 0, 0, 0), 8: (1, 0, 0, 0, 0)},
                    "task_locations": {7: np.array([0.5, 0.0]), 8: np.array([0.5, 0.0])},
                    "task_deadlines": {7: 100.0, 8: 100.0},
                },
                1: {
                    "task_current_status": {7: (1, 0, 0, 0, 0), 8: (1, 0, 0, 0, 0)},
                    "task_locations": {7: np.array([0.5, 0.0]), 8: np.array([0.5, 0.0])},
                    "task_deadlines": {7: 100.0, 8: 100.0},
                },
            }

        def get_local_obs(self, agent_id: int, max_tasks: int | None = None) -> Any:
            del agent_id, max_tasks
            return (
                np.zeros(4, dtype=np.float32),
                np.zeros((2, 4), dtype=np.float32),
                np.zeros((len(ACTION_MAP), 4), dtype=np.float32),
                mask,
                list(ACTION_MAP),
            )

    return _StubEnv()


def _fixed_network(logits: Any) -> Any:
    import torch

    from integrated_sar.adapters.task03 import MASKED_LOGIT

    class _Network:
        def parameters(self) -> Any:
            return iter([torch.zeros(1)])

        def get_action(self, state: Any, neighbors: Any, tasks: Any, hard_mask: Any) -> Any:
            del state, neighbors, tasks
            values = logits.clone()
            return SimpleNamespace(logits=values.masked_fill(hard_mask.bool(), MASKED_LOGIT))

    return _Network()


def _make_adapter(
    *,
    mask: Any,
    logits: Any,
    residual_scale: float = 0.0,
    network: bool = True,
    **kwargs: Any,
) -> Any:
    from integrated_sar.adapters.task03 import Task03Adapter

    adapter = Task03Adapter(
        env=_stub_env(mask),
        network=_fixed_network(logits) if network else None,
        deterministic=True,
        clear_initial_tasks=False,
        residual_scale=residual_scale,
        ctde_task_logit_offset=0.0,
        **kwargs,
    )
    adapter.internal_to_external = {7: "target-a", 8: "target-b"}
    return adapter


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class ReservationAllocatorTests(unittest.TestCase):
    """What a commitment does to a logit, and what it must never do."""

    def _advice(self, **kwargs: Any) -> CoordinationAdvice:
        return CoordinationAdvice(
            revision=1,
            generated_at=0.0,
            valid_until=10.0,
            pair_bias={("0", "target-a"): 0.2, ("0", "target-b"): 0.0},
            **kwargs,
        )

    def test_the_commitment_rides_the_bounded_residual_channel(self) -> None:
        import numpy as np
        import torch

        mask = np.array([False, False, True], dtype=bool)
        logits = torch.tensor([[0.50, 0.10, 0.05]])
        plain = _make_adapter(mask=mask, logits=logits, residual_scale=0.5)._network_verdict(
            0, list(ACTION_MAP), self._advice()
        )
        committed = _make_adapter(
            mask=mask, logits=logits, residual_scale=0.5
        )._network_verdict(
            0,
            list(ACTION_MAP),
            self._advice(reservations={("0", "target-a"): reservation(score=0.4)}),
        )
        # final = ctde + offset + bias, with the commitment folded into ``bias``.
        self.assertAlmostEqual(committed.final[1], 0.10 + 0.5 * (0.2 + 0.4))
        self.assertAlmostEqual(committed.bias[1], 0.5 * 0.6)
        self.assertAlmostEqual(plain.bias[1], 0.5 * 0.2)
        # The depot is never biased, with or without a commitment.
        self.assertAlmostEqual(committed.final[0], 0.50)
        self.assertEqual(committed.bias[0], 0.0)
        self.assertAlmostEqual(committed.ctde[1], 0.10)

    def test_a_masked_task_is_never_re_legalised_by_a_commitment(self) -> None:
        import numpy as np
        import torch

        mask = np.array([False, False, True], dtype=bool)
        logits = torch.tensor([[0.30, 0.05, 0.05]])
        advice = CoordinationAdvice(
            revision=1,
            generated_at=0.0,
            valid_until=10.0,
            reservations={
                ("0", "target-a"): reservation(score=1.0),
                ("0", "target-b"): reservation(task_id="target-b", score=1.0),
            },
        )
        adapter = _make_adapter(mask=mask, logits=logits, residual_scale=1.0)
        verdict = adapter._network_verdict(0, list(ACTION_MAP), advice)
        # The full legal residual still leaves the masked task out: the hard mask is
        # applied after the commitment, not before it.
        self.assertIsNone(verdict.final[2])
        self.assertNotEqual(verdict.index, 2)
        self.assertAlmostEqual(verdict.final[0], 0.30)

    def test_a_commitment_moves_only_the_robot_that_holds_it(self) -> None:
        import numpy as np
        import torch

        mask = np.array([False, False, True], dtype=bool)
        logits = torch.tensor([[0.50, 0.10, 0.05]])
        advice = self._advice(reservations={("0", "target-a"): reservation(score=0.9)})
        adapter = _make_adapter(mask=mask, logits=logits, residual_scale=0.5)
        holder = adapter._network_verdict(0, list(ACTION_MAP), advice)
        other = adapter._network_verdict(1, list(ACTION_MAP), advice)
        self.assertAlmostEqual(holder.bias[1], 0.5 * (0.2 + 0.9))
        self.assertEqual(other.bias[1], 0.0)

    def test_a_commitment_for_another_task_leaves_this_task_alone(self) -> None:
        import numpy as np
        import torch

        mask = np.array([False, False, True], dtype=bool)
        logits = torch.tensor([[0.50, 0.10, 0.05]])
        advice = self._advice(
            reservations={("0", "target-b"): reservation(task_id="target-b", score=0.9)}
        )
        adapter = _make_adapter(mask=mask, logits=logits, residual_scale=0.5)
        verdict = adapter._network_verdict(0, list(ACTION_MAP), advice)
        self.assertAlmostEqual(verdict.bias[1], 0.5 * 0.2)

    def test_the_heuristic_path_reads_the_same_commitment(self) -> None:
        import numpy as np

        mask = np.array([False, False, True], dtype=bool)
        plain = _make_adapter(
            mask=mask, logits=None, residual_scale=0.5, network=False
        )._heuristic_verdict(0, mask, list(ACTION_MAP), 3.0, self._advice())
        committed = _make_adapter(
            mask=mask, logits=None, residual_scale=0.5, network=False
        )._heuristic_verdict(
            0,
            mask,
            list(ACTION_MAP),
            3.0,
            self._advice(reservations={("0", "target-a"): reservation(score=0.4)}),
        )
        self.assertAlmostEqual(committed.final[1] - plain.final[1], 0.5 * 0.4)
        # The masked and depot entries are untouched by the commitment.
        self.assertEqual(committed.final[0], plain.final[0])
        self.assertIsNone(committed.final[2])

    def test_the_reservation_lookup_is_a_pure_read(self) -> None:
        # A frozen advice stays usable across verdicts: reading a commitment must not
        # consume it, or the second robot to decide would lose the team plan.
        advice = self._advice(reservations={("0", "target-a"): reservation(score=0.4)})
        self.assertIn(("0", "target-a"), advice.reservations)
        self.assertEqual(advice.reservations[("0", "target-a")].score, 0.4)


if __name__ == "__main__":
    unittest.main()
