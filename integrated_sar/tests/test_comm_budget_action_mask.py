"""The candidate's price tag must be the transport's own arithmetic, not an echo of it.

``SendItem.attempt_charge_bytes`` / ``send_executable`` / ``send_block_reason`` exist so a learned
sender can be told which actions the executor could honour. The moment they are computed by a
second formula they become a *claim* about the budget rather than a reading of it, and the drift
shows up only later as a controller trained on an action set the relay never offered. Every test
here therefore compares the metadata against what ``CommRelay._emit`` actually charged, or against
``CommBudget`` -- the same gate the emitter passes a send through.

The fields are execution and tape metadata only. The last test pins the feature layout at 19 and
23 dimensions, because putting the price into the observation is a different experiment and would
invalidate every checkpoint trained on the current one.
"""

from __future__ import annotations

import unittest

from integrated_sar.comm import CommRelay, estimate_message_bytes
from integrated_sar.comm_budget import CommBudget
from integrated_sar.comm_learning import features
from integrated_sar.comm_policy import (
    CommSendPolicy,
    PeriodicPolicy,
    SendItem,
    SendObservation,
    transport_order,
)
from integrated_sar.contracts import (
    DetectionEvent,
    GridPosition,
    ProgressEvent,
    TargetProfile,
    TaskStageSpec,
    TaskStatus,
    WorldPosition,
)
from integrated_sar.task_registry import TaskRegistry

ACK_VECTOR = (0.5, 3.0, 1.0, 0.0)


class RecordingWriter:
    """Applies the real freshness rule and keeps every message it was handed."""

    def __init__(self) -> None:
        self.believed: dict[tuple[str, str], tuple[float, int]] = {}

    def deliver_detection(self, message, agent_id: str) -> bool:
        key = (str(agent_id), str(message.target_id))
        known = self.believed.get(key)
        if known is not None and (float(message.sent_at), int(message.revision)) <= known:
            return False
        self.believed[key] = (float(message.sent_at), int(message.revision))
        return True

    def belief_freshness(self, agent_id: str, target_id: str):
        return self.believed.get((str(agent_id), str(target_id)))


def detection(
    target_id: str = "target-1",
    *,
    detected_at: float = 1.0,
    stages: tuple[TaskStageSpec, ...] | None = None,
    deadline_offset: float = 500.0,
) -> DetectionEvent:
    return DetectionEvent(
        event_id=f"detection:{target_id}",
        target_id=target_id,
        grid_position=GridPosition(59, 59),
        world_position=WorldPosition(1.0, 1.0),
        detected_at=detected_at,
        discovered_by="scout_0",
        confidence=1.0,
        profile=TargetProfile(
            stages=stages or (TaskStageSpec((1, 0, 0, 0, 0), 3.0),),
            deadline_offset=deadline_offset,
        ),
    )


def progress(target_id: str, *, observed_at: float, stage_index: int = 0) -> ProgressEvent:
    return ProgressEvent(
        event_id=f"progress:{target_id}:{observed_at}",
        task_id=target_id,
        observed_at=observed_at,
        status=TaskStatus.IN_PROGRESS,
        stage_index=stage_index,
        remaining_requirements=(0, 1, 0, 0, 0),
        progress=0.5,
        members=("0",),
    )


class CapturingPolicy(CommSendPolicy):
    """Records the observation it was shown; emits only when told to."""

    name = "charge_capturer"
    orders_selection = True

    def __init__(self, *, emit: bool = False) -> None:
        self.observations: list[SendObservation] = []
        self.emit = emit

    def decide(self, observation: SendObservation) -> tuple[str, ...]:
        self.observations.append(observation)
        return tuple(item.target_id for item in observation.items) if self.emit else ()


class Harness:
    """A registry, a relay and a writer, wired the way the driver wires them."""

    def __init__(
        self,
        *,
        subscribers: tuple[str, ...] = ("0", "1"),
        budget: CommBudget | None = None,
        policy: CommSendPolicy | None = None,
        with_snapshot: bool = True,
    ) -> None:
        self.registry = TaskRegistry()
        self.writer = RecordingWriter()
        self.policy = policy if policy is not None else CapturingPolicy()
        self.now = 0.0
        self.relay = CommRelay(
            subscribers=subscribers,
            writer=self.writer,
            snapshot_of=self.registry.get if with_snapshot else None,
            accepted_of=self.registry.was_accepted,
            delivery_probability=1.0,
            delay_ticks=0,
            seed=11,
            policy=self.policy,
            budget=budget if budget is not None else CommBudget(),
            now_of=lambda: self.now,
            freshness_of=self.writer.belief_freshness,
        )

    # -- driving ---------------------------------------------------------------
    def publish(self, event: DetectionEvent) -> None:
        self.registry.handle_detection(event)
        self.relay.handle_detection(event)

    def report(self, event: ProgressEvent) -> None:
        self.registry.handle_progress(event)

    def tick(self) -> None:
        self.now += 1.0
        self.relay.tick()

    def step(self) -> SendObservation:
        self.tick()
        assert self.policy.observations, "this policy does not record the candidates it was shown"
        return self.policy.observations[-1]

    # -- reading the metadata back --------------------------------------------
    @staticmethod
    def charge(observation: SendObservation, target_id: str) -> int:
        item = next(entry for entry in observation.items if entry.target_id == target_id)
        assert item.attempt_charge_bytes is not None, "the relay offered a candidate it did not price"
        return int(item.attempt_charge_bytes)

    @staticmethod
    def item(observation: SendObservation, target_id: str) -> SendItem:
        return next(entry for entry in observation.items if entry.target_id == target_id)

    @property
    def airtime(self) -> int:
        return int(self.relay.counters["airtime_bytes"])


class ChargeEqualsEmitTests(unittest.TestCase):
    """What a candidate is told it costs is what the transport then spends."""

    def test_single_candidate_charge_equals_the_bytes_spent(self) -> None:
        harness = Harness(policy=CapturingPolicy(emit=True))
        harness.publish(detection("target-1"))
        observation = harness.step()
        charge = harness.charge(observation, "target-1")
        self.assertEqual(harness.airtime, charge)
        self.assertEqual(int(harness.relay.counters["bytes_policy"]), charge)
        self.assertEqual(int(harness.relay.counters["emissions"]), 1)

    def test_refresh_is_priced_by_the_same_arithmetic_as_the_first_emission(self) -> None:
        harness = Harness(policy=CapturingPolicy(emit=True))
        harness.publish(detection("target-1"))
        first = harness.charge(harness.step(), "target-1")
        harness.report(progress("target-1", observed_at=2.0))
        second = harness.charge(harness.step(), "target-1")
        self.assertEqual(harness.airtime, first + second)
        self.assertEqual(int(harness.relay.counters["refresh_emissions"]), 1)

    def test_charge_is_the_payload_times_the_subscriber_fan_out(self) -> None:
        prices = {}
        for count in (1, 2, 5):
            harness = Harness(subscribers=tuple(str(index) for index in range(count)))
            harness.publish(detection("target-1"))
            prices[count] = harness.charge(harness.step(), "target-1")
        self.assertEqual(prices[2], 2 * prices[1])
        self.assertEqual(prices[5], 5 * prices[1])

    def test_a_bigger_view_costs_more(self) -> None:
        small = Harness()
        small.publish(detection("target-1"))
        small_charge = small.charge(small.step(), "target-1")
        big = Harness()
        big.publish(
            detection(
                "target-1",
                stages=tuple(TaskStageSpec((1, 0, 0, 0, 0), 3.0) for _ in range(4)),
            )
        )
        self.assertGreater(big.charge(big.step(), "target-1"), small_charge)

    def test_charge_matches_estimate_message_bytes_for_the_built_message(self) -> None:
        harness = Harness(subscribers=("0", "1", "2"))
        harness.publish(detection("target-1"))
        observation = harness.step()
        ledger = harness.relay._ledger["target-1"]
        built = harness.relay._message("target-1", ledger, harness.registry.get("target-1"))
        self.assertEqual(
            harness.charge(observation, "target-1"),
            estimate_message_bytes(built) * 3,
        )


class GateVerdictTests(unittest.TestCase):
    """``send_executable`` is the emitter's own answer to ``can_send``, at the same price."""

    def price_with(self, budget: CommBudget) -> tuple[int, bool | None, str | None]:
        harness = Harness(budget=budget)
        harness.publish(detection("target-1"))
        item = harness.item(harness.step(), "target-1")
        return int(item.attempt_charge_bytes or 0), item.send_executable, item.send_block_reason

    def test_credit_exactly_equal_to_the_charge_is_executable(self) -> None:
        charge = self.price_with(CommBudget())[0]
        priced, executable, reason = self.price_with(CommBudget(bytes_per_episode=charge))
        self.assertEqual(priced, charge)
        self.assertTrue(executable)
        self.assertEqual(reason, "allowed")

    def test_one_byte_short_is_budget_exhausted(self) -> None:
        charge = self.price_with(CommBudget())[0]
        priced, executable, reason = self.price_with(CommBudget(bytes_per_episode=charge - 1))
        self.assertEqual(priced, charge)
        self.assertFalse(executable)
        self.assertEqual(reason, "budget_exhausted")

    def test_reasons_are_the_transport_vocabulary(self) -> None:
        charge = self.price_with(CommBudget())[0]
        gate = CommBudget(bytes_per_episode=charge - 1)
        self.assertEqual(gate.why_blocked(charge), "budget_exhausted")
        gate = CommBudget(bytes_per_episode=10**6, sends_per_step=1)
        gate.charge(1)
        self.assertEqual(gate.why_blocked(charge), "capacity_step")
        self.assertEqual(CommBudget().why_blocked(charge), "allowed")

    def test_capacity_is_judged_as_the_next_send_not_after_the_whole_step(self) -> None:
        # Both candidates can be *the next* emission when capacity is one. Which one actually
        # gets it is the order-dependent question, and the collector answers it with a shadow.
        harness = Harness(budget=CommBudget(bytes_per_episode=10**6, sends_per_step=1))
        for index in (1, 2):
            harness.publish(detection(f"target-{index}"))
        observation = harness.step()
        self.assertEqual(
            [item.send_block_reason for item in observation.items], ["allowed", "allowed"]
        )
        self.assertTrue(all(item.send_executable for item in observation.items))
        self.assertEqual(observation.capacity_remaining, 1)

    def test_unbounded_budget_reports_allowed(self) -> None:
        _, executable, reason = self.price_with(CommBudget())
        self.assertTrue(executable)
        self.assertEqual(reason, "allowed")

    def test_an_unpriceable_candidate_says_so_instead_of_guessing(self) -> None:
        # No registry lookup and no stored event left after the first emission: exactly the
        # condition under which _emit itself would raise, so the metadata reports it.
        policy = CapturingPolicy(emit=True)
        harness = Harness(with_snapshot=False, policy=policy)
        harness.publish(detection("target-1"))
        first = harness.item(harness.step(), "target-1")
        self.assertIsNotNone(first.attempt_charge_bytes)
        policy.emit = False
        second = harness.item(harness.step(), "target-1")
        self.assertIsNone(second.attempt_charge_bytes)
        self.assertIsNone(second.send_executable)
        self.assertEqual(second.send_block_reason, "charge_unavailable")
        # The metadata says "I cannot price this", not "this is free": asking for it is exactly
        # the request the transport refuses to serve, for the same reason.
        policy.emit = True
        with self.assertRaises(RuntimeError):
            harness.tick()


class CompatibilityTests(unittest.TestCase):
    """Old construction sites keep working, and the new fields stay out of the network."""

    def test_price_metadata_does_not_enter_the_feature_vector(self) -> None:
        harness = Harness()
        harness.publish(detection("target-1"))
        observation = harness.step()
        item = harness.item(observation, "target-1")
        self.assertEqual(len(features(observation, item)), 19)
        self.assertEqual(
            len(features(observation, item, ack_features=ACK_VECTOR)), 23
        )
        priced = SendItem(**{**item.__dict__, "attempt_charge_bytes": 999999,
                             "send_executable": False, "send_block_reason": "budget_exhausted"})
        self.assertEqual(features(observation, priced), features(observation, item))

    def test_a_hand_built_item_without_the_new_fields_still_works(self) -> None:
        values = dict(
            target_id="target-9",
            ready_step=0,
            age_steps=3,
            pending_event=True,
            changed_view=True,
            first_emission=True,
            revision=1,
            revision_delta=1,
            stage_index=0,
            stage_changed=False,
            progress=0.0,
            progress_delta=0.0,
            requirements_changed=False,
            members_changed=False,
            deadline=500.0,
            remaining_time=400.0,
            sent_count=0,
            last_sent_step=None,
            since_last_emission_time=None,
            has_future_requirements=False,
        )
        legacy = SendItem(**values)
        self.assertIsNone(legacy.attempt_charge_bytes)
        self.assertIsNone(legacy.send_executable)
        self.assertIsNone(legacy.send_block_reason)
        observation = SendObservation(step=3, current_time=3.0, items=(legacy,))
        self.assertEqual([item.target_id for item in transport_order(observation.items)], ["target-9"])
        self.assertEqual(len(features(observation, legacy)), 19)

    def test_a_rule_arm_still_emits_and_reports_no_refusal(self) -> None:
        harness = Harness(policy=PeriodicPolicy(interval=1), budget=CommBudget(bytes_per_episode=10**6))
        harness.publish(detection("target-1"))
        harness.tick()
        self.assertEqual(int(harness.relay.counters["emissions"]), 1)
        self.assertEqual(int(harness.relay.counters["budget_refused_sends"]), 0)
        self.assertEqual(int(harness.relay.counters["capacity_refused_sends"]), 0)


if __name__ == "__main__":
    unittest.main()
