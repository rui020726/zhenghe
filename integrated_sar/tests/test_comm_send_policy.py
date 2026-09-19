"""The uplink send policy: what goes on the air, when, and what the numbers say about it.

These tests pin the transport's own promises (one emission per target per step, a payload
built from the newest snapshot, uniform budget trimming, no announcement of a closed task) and
each policy's decision rule. They also pin the one thing the whole round depends on: with the
default policy and no delay or loss, nothing new goes out that did not go out before.
"""

from __future__ import annotations

import unittest

from integrated_sar.comm import CommRelay, CommTranscript, estimate_message_bytes
from integrated_sar.comm_policy import (
    EventMaxWaitPolicy,
    EventPolicy,
    ImmediatePolicy,
    PeriodicPolicy,
    build_comm_policy,
    parse_parameters,
    policy_descriptors,
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


class RecordingWriter:
    """The receiver side: remember what arrived, accept everything."""

    def __init__(self) -> None:
        self.received: list[tuple[str, str, int, float]] = []
        self.messages: list[object] = []

    def deliver_detection(self, message, agent_id: str) -> bool:
        self.received.append(
            (
                str(agent_id),
                message.target_id,
                int(message.revision),
                float(message.progress),
            )
        )
        self.messages.append(message)
        return True


def detection(
    target_id: str = "target-1",
    *,
    event_id: str | None = None,
    detected_at: float = 1.0,
    stages: tuple[TaskStageSpec, ...] = (TaskStageSpec((1, 0, 0, 0, 0), 3.0),),
    deadline_offset: float = 500.0,
) -> DetectionEvent:
    return DetectionEvent(
        event_id=event_id or f"detection:{target_id}",
        target_id=target_id,
        grid_position=GridPosition(59, 59),
        world_position=WorldPosition(1.0, 1.0),
        detected_at=detected_at,
        discovered_by="scout_0",
        confidence=1.0,
        profile=TargetProfile(stages=stages, deadline_offset=deadline_offset),
    )


def progress(
    target_id: str,
    *,
    observed_at: float,
    stage_index: int = 0,
    remaining: tuple[int, ...] = (1, 0, 0, 0, 0),
    value: float = 0.0,
    members: tuple[str, ...] = (),
    status: TaskStatus = TaskStatus.IN_PROGRESS,
    event_id: str | None = None,
) -> ProgressEvent:
    return ProgressEvent(
        event_id=event_id or f"progress:{target_id}:{observed_at}",
        task_id=target_id,
        observed_at=observed_at,
        status=status,
        stage_index=stage_index,
        remaining_requirements=remaining,
        progress=value,
        members=members,
    )


class Harness:
    """A registry, a relay and a writer, wired the way the driver wires them."""

    def __init__(
        self,
        *,
        policy=None,
        subscribers=("0", "1"),
        delay_ticks: int = 0,
        delivery_probability: float = 1.0,
        sends_per_tick: int | None = None,
        seed: int = 7,
    ) -> None:
        self.registry = TaskRegistry()
        self.writer = RecordingWriter()
        self.now = 0.0
        self.relay = CommRelay(
            subscribers=subscribers,
            writer=self.writer,
            snapshot_of=self.registry.get,
            accepted_of=self.registry.was_accepted,
            delivery_probability=delivery_probability,
            delay_ticks=delay_ticks,
            seed=seed,
            policy=policy,
            sends_per_tick=sends_per_tick,
            now_of=lambda: self.now,
        )

    def publish(self, event: DetectionEvent) -> None:
        self.registry.handle_detection(event)
        self.relay.handle_detection(event)

    def report(self, event: ProgressEvent) -> None:
        self.registry.handle_progress(event)

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            self.now += 1.0
            self.relay.tick()

    @property
    def emissions(self) -> int:
        return self.relay.counters["emissions"]

    @property
    def delivered(self):
        return self.writer.received


class TransportPromiseTests(unittest.TestCase):
    """What the relay owes every policy alike."""

    def test_an_emission_at_zero_delay_lands_in_the_same_step(self) -> None:
        run = Harness(delay_ticks=0)
        run.publish(detection())
        run.step()
        self.assertEqual({item[1] for item in run.delivered}, {"target-1"})
        self.assertEqual(run.emissions, 1)

    def test_a_one_step_delay_arrives_one_step_later(self) -> None:
        run = Harness(delay_ticks=1)
        run.publish(detection())
        run.step()
        self.assertEqual(run.delivered, [])
        run.step()
        self.assertEqual(len(run.delivered), 2, "one packet per subscribed robot")

    def test_the_historical_policy_emits_once_per_detection_and_never_again(self) -> None:
        run = Harness(policy=ImmediatePolicy())
        run.publish(detection())
        run.step()
        for step in range(2, 8):
            run.report(
                progress("target-1", observed_at=float(step), value=step / 10.0, members=("0",))
            )
            run.step()
        self.assertEqual(run.emissions, 1, "a refresh the old relay never sent stays unsent")
        self.assertEqual(len(run.delivered), 2)

    def test_news_the_registry_refused_is_not_news(self) -> None:
        run = Harness()
        run.publish(detection())
        stale = detection(event_id="detection:late", detected_at=0.0)
        run.registry.handle_detection(stale)
        run.relay.handle_detection(stale)
        run.step(3)
        self.assertEqual(run.registry.rejection_reason("detection:late"), "late_event")
        self.assertEqual(run.emissions, 1, "only the accepted detection went out")

    def test_a_repeated_event_id_is_not_counted_twice(self) -> None:
        run = Harness()
        event = detection()
        run.publish(event)
        run.relay.handle_detection(event)
        run.step()
        self.assertEqual(run.relay.counters["events_accepted"], 1)
        self.assertEqual(run.emissions, 1)

    def test_a_closed_task_is_never_announced(self) -> None:
        run = Harness(policy=PeriodicPolicy(interval=5))
        run.publish(detection())
        run.report(progress("target-1", observed_at=2.0, status=TaskStatus.COMPLETED, value=1.0))
        run.step(6)
        self.assertEqual(run.emissions, 0)
        self.assertEqual(run.relay.counters["emissions_suppressed_terminal"], 1)
        self.assertEqual(run.delivered, [])

    def test_a_budget_trims_in_one_order_whatever_the_policy(self) -> None:
        # Two targets discovered in the same step, one much closer to its deadline: the
        # transport keeps the urgent one, identically for every policy.
        for policy in (ImmediatePolicy(), PeriodicPolicy(interval=1)):
            run = Harness(policy=policy, sends_per_tick=1)
            run.publish(detection("far", deadline_offset=500.0))
            run.publish(detection("near", deadline_offset=20.0))
            run.step()
            self.assertEqual(run.emissions, 1, policy.name)
            self.assertEqual([item[1] for item in run.delivered], ["near", "near"], policy.name)
            self.assertEqual(run.relay.counters["emissions_suppressed_budget"], 1, policy.name)

    def test_reset_replays_the_same_packet_outcome(self) -> None:
        run = Harness(delivery_probability=0.5)
        run.publish(detection())
        run.step(3)
        first, first_sent = list(run.delivered), run.relay.counters["packets_sent"]
        run.relay.reset()
        run.writer.received.clear()
        run.publish(detection())
        run.step(3)
        self.assertEqual(run.delivered, first)
        self.assertEqual(run.relay.counters["packets_sent"], first_sent)


class PeriodicPolicyTests(unittest.TestCase):
    def test_nothing_goes_on_the_air_between_slots(self) -> None:
        run = Harness(policy=PeriodicPolicy(interval=5))
        run.publish(detection())
        run.step(4)
        self.assertEqual(run.emissions, 0)
        run.step()
        self.assertEqual(run.emissions, 1)

    def test_a_slot_carries_the_newest_view_not_the_one_that_waited(self) -> None:
        # This is the difference between fewer packets and less information: holding a change
        # and then sending the current state loses nothing the delayed send could have said.
        run = Harness(policy=PeriodicPolicy(interval=5))
        run.publish(detection())
        run.report(progress("target-1", observed_at=2.0, value=0.5, members=("3",)))
        run.step(5)
        self.assertEqual(run.emissions, 1, "one coalesced packet, not two")
        self.assertEqual(run.delivered[0][3], 0.5)
        self.assertEqual(run.delivered[0][2], 2, "the registry revision after the progress event")

    def test_a_periodic_policy_keeps_offering_so_a_loss_is_recoverable(self) -> None:
        run = Harness(policy=PeriodicPolicy(interval=2), delivery_probability=0.5)
        run.publish(detection())
        run.step(20)
        self.assertEqual(run.emissions, 10)
        self.assertEqual(
            run.relay.counters["packets_sent"] + run.relay.counters["packets_lost"],
            run.emissions * 2,
            "every emission is one packet per subscriber, lost or not",
        )
        self.assertGreater(len(run.delivered), 0)


class EventPolicyTests(unittest.TestCase):
    def test_a_view_that_has_not_changed_does_not_go_on_the_air(self) -> None:
        run = Harness(policy=EventPolicy())
        run.publish(detection())
        run.step()
        self.assertEqual(run.emissions, 1)
        run.step(10)
        self.assertEqual(run.emissions, 1, "silence is free when nothing moved")

    def test_a_stage_change_triggers(self) -> None:
        stages = (TaskStageSpec((1, 0, 0, 0, 0), 3.0), TaskStageSpec((0, 1, 0, 0, 0), 3.0))
        run = Harness(policy=EventPolicy())
        run.publish(detection(stages=stages))
        run.step()
        run.report(progress("target-1", observed_at=5.0, stage_index=1, value=0.5))
        run.step()
        self.assertEqual(run.emissions, 2)

    def test_a_requirement_change_triggers(self) -> None:
        run = Harness(policy=EventPolicy())
        run.publish(detection())
        run.step()
        run.report(progress("target-1", observed_at=5.0, remaining=(0, 0, 0, 0, 0)))
        run.step()
        self.assertEqual(run.emissions, 2)

    def test_a_roster_change_triggers(self) -> None:
        run = Harness(policy=EventPolicy())
        run.publish(detection())
        run.step()
        run.report(progress("target-1", observed_at=5.0, members=("4",)))
        run.step()
        self.assertEqual(run.emissions, 2)

    def test_a_progress_jump_triggers_but_a_small_one_does_not(self) -> None:
        run = Harness(policy=EventPolicy(progress_threshold=0.2))
        run.publish(detection())
        run.step()
        run.report(progress("target-1", observed_at=5.0, value=0.05))
        run.step()
        self.assertEqual(run.emissions, 1, "5% more done is not news")
        run.report(progress("target-1", observed_at=7.0, value=0.4, event_id="p2"))
        run.step()
        self.assertEqual(run.emissions, 2)

    def test_the_deadline_horizon_triggers_once_on_the_crossing(self) -> None:
        run = Harness(policy=EventPolicy(urgency_time=25.0))
        run.publish(detection(deadline_offset=40.0))
        run.step()
        self.assertEqual(run.emissions, 1)
        run.step(14)
        self.assertEqual(run.emissions, 1, "still 26 steps of slack")
        run.step()
        self.assertEqual(run.emissions, 2, "the step it enters the horizon")
        run.step(3)
        self.assertEqual(run.emissions, 2, "and not again for the same news")

    def test_a_change_nothing_reads_is_not_a_reason_to_send(self) -> None:
        # A progress event that leaves every applied field alone bumps the registry revision
        # but says nothing a robot would act on: a packet for it is traffic, not information.
        run = Harness(policy=EventPolicy())
        run.publish(detection())
        run.step()
        run.report(progress("target-1", observed_at=5.0))
        run.step(4)
        self.assertEqual(run.emissions, 1)
        self.assertEqual(run.relay.counters["unsent_change_steps"], 0, "nothing was held")

    def test_the_max_wait_bounds_silence_about_a_real_change(self) -> None:
        run = Harness(policy=EventMaxWaitPolicy(max_wait_steps=3, urgency_time=0.0))
        run.publish(detection())
        run.step()
        self.assertEqual(run.emissions, 1)
        run.report(progress("target-1", observed_at=5.0, value=0.02))
        run.step(2)
        self.assertEqual(run.emissions, 1, "held, because holding is cheaper")
        run.step(2)
        self.assertEqual(run.emissions, 2, "but never beyond the declared bound")


class MeasurementTests(unittest.TestCase):
    def test_bytes_count_payload_and_fan_out_not_just_batches(self) -> None:
        single = Harness()
        single.publish(detection())
        single.step()
        many = Harness()
        many.publish(detection(stages=tuple(TaskStageSpec((1, 0, 0, 0, 0), 3.0) for _ in range(4))))
        many.step()
        self.assertEqual(single.emissions, many.emissions, "one batch each")
        self.assertEqual(len(single.writer.messages[0].stages), 1)
        self.assertGreater(
            many.relay.counters["payload_bytes"],
            single.relay.counters["payload_bytes"],
            "more task structure in the letter is more information, and it has to show up",
        )
        per_packet = estimate_message_bytes(single.writer.messages[0])
        self.assertEqual(single.relay.counters["payload_bytes"], per_packet * 2)
        self.assertEqual(single.relay.counters["packets_sent"], 2, "two robots, two packets")

    def test_the_transcript_names_every_stage_of_a_packet_life(self) -> None:
        run = Harness(policy=PeriodicPolicy(interval=3), delay_ticks=1)
        run.relay.transcript = CommTranscript()
        run.publish(detection())
        run.step(4)
        records = run.relay.transcript.records
        kinds = [record["kind"] for record in records]
        self.assertEqual(kinds.count("event"), 1)
        self.assertEqual(kinds.count("send"), 1)
        self.assertEqual(kinds.count("deliver"), 2)
        send = next(record for record in records if record["kind"] == "send")
        self.assertEqual(send["packets"], 2)
        self.assertEqual(send["subscribers"], 2)
        self.assertEqual(send["ready_step"], 1, "the step the news became sendable")
        self.assertEqual(send["backlog_steps"], 2, "it waited for the next slot")
        delivered = [record for record in records if record["kind"] == "deliver"]
        self.assertTrue(all(record["accepted"] for record in delivered))
        self.assertEqual({record["step"] for record in delivered}, {send["due_step"]})

    def test_a_transcript_changes_nothing_about_what_is_delivered(self) -> None:
        plain = Harness(policy=EventPolicy())
        plain.publish(detection())
        plain.step(2)
        recorded = Harness(policy=EventPolicy())
        recorded.relay.transcript = CommTranscript()
        recorded.publish(detection())
        recorded.step(2)
        self.assertEqual(recorded.writer.received, plain.writer.received)
        self.assertEqual(recorded.emissions, plain.emissions)

    def test_a_held_change_is_visible_as_a_wait_and_not_as_a_packet(self) -> None:
        run = Harness(policy=ImmediatePolicy())
        run.publish(detection())
        run.step()
        run.report(progress("target-1", observed_at=5.0, value=0.5))
        run.step(4)
        self.assertEqual(run.emissions, 1)
        self.assertEqual(run.relay.backlog, 1)
        # The relay looks at the registry once per step, at the start of its tick, so a change
        # reported between ticks first ages on the following one: four steps of holding, three
        # steps of age.
        self.assertEqual(run.relay.counters["max_backlog_age"], 3)
        self.assertGreaterEqual(run.relay.counters["unsent_change_steps"], 4)
        self.assertEqual(run.relay.counters["backlog_steps"], 0, "what went out went at once")


class ConstructionTests(unittest.TestCase):
    def test_an_unknown_policy_name_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            build_comm_policy("telepathy")

    def test_an_unknown_parameter_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            build_comm_policy("periodic", "period=5")

    def test_parameters_are_typed_before_a_run_starts(self) -> None:
        self.assertEqual(
            parse_parameters("interval=5,urgency_time=2.5"),
            {"interval": 5, "urgency_time": 2.5},
        )
        with self.assertRaises(ValueError):
            parse_parameters("interval=five")
        with self.assertRaises(ValueError):
            parse_parameters("interval")

    def test_the_default_policy_is_the_historical_relay(self) -> None:
        run = Harness()
        self.assertEqual(run.relay.policy.name, "immediate")
        self.assertIsInstance(run.relay.policy, ImmediatePolicy)

    def test_every_published_policy_can_be_built_from_its_published_parameters(self) -> None:
        for descriptor in policy_descriptors():
            policy = build_comm_policy(descriptor["name"], dict(descriptor["parameters"]))
            self.assertEqual(policy.name, descriptor["name"])
            self.assertEqual(policy.parameters(), descriptor["parameters"])


class DeliveryRefusalTests(unittest.TestCase):
    """The refusal side of the ledger, against the real rescue adapter.

    The relay classifies a refusal so "we sent 30 packets, 12 changed nothing" is a number
    rather than a guess. This uses the far-away single-robot layout from the comm-bridge
    tests, where no line-of-sight refresh can move a belief behind the relay's back.
    """

    def setUp(self) -> None:
        from integrated_sar.adapters.task03 import Task03Adapter
        from integrated_sar.task03_runtime import build_task_env, load_task_env_module

        self.registry = TaskRegistry()
        self.env = build_task_env(
            load_task_env_module(),
            seed=7,
            species=1,
            per_species=2,
            depot_positions=[(0.0, 0.0)],
        )
        self.adapter = Task03Adapter(env=self.env, clear_initial_tasks=True, seed=7)
        self.event = detection()
        self.registry.handle_detection(self.event)
        self.adapter.sync_tasks(self.registry.snapshots())
        self.internal_id = self.adapter.external_to_internal["target-1"]

    def relay(self) -> CommRelay:
        return CommRelay(
            subscribers=("0",),
            writer=self.adapter,
            snapshot_of=self.registry.get,
            accepted_of=self.registry.was_accepted,
            seed=7,
            freshness_of=self.adapter.belief_freshness,
        )

    def test_freshness_reports_what_the_robot_believes_after_a_delivery(self) -> None:
        relay = self.relay()
        relay.handle_detection(self.event)
        relay.tick()
        believed = self.adapter.belief_freshness("0", "target-1")
        self.assertIsNotNone(believed)
        self.assertEqual(believed[1], 1, "the registry revision it just learned")
        self.assertIn(self.internal_id, self.env.agent_knowledge[0]["known_tasks"])

    def test_a_refused_packet_leaves_the_belief_where_it_was(self) -> None:
        relay = self.relay()
        relay.handle_detection(self.event)
        relay.tick()
        before = self.adapter.belief_freshness("0", "target-1")
        message = relay._ledger["target-1"].last_sent
        self.assertFalse(self.adapter.deliver_detection(message, "0"))
        self.assertEqual(self.adapter.belief_freshness("0", "target-1"), before)

    def test_the_relay_tells_a_duplicate_from_a_superseded_packet(self) -> None:
        relay = self.relay()
        relay.handle_detection(self.event)
        relay.tick()
        first = relay._ledger["target-1"].last_sent

        self.registry.handle_progress(progress("target-1", observed_at=20.0, value=0.6))
        again = detection(event_id="detection:target-1:again", detected_at=20.0)
        self.registry.handle_detection(again)
        relay.handle_detection(again)
        relay.tick()
        second = relay._ledger["target-1"].last_sent
        self.assertGreater(second.revision, first.revision)

        self.assertEqual(relay._deliver(first, "0"), (False, "delivery_superseded"))
        self.assertEqual(relay._deliver(second, "0"), (False, "delivery_duplicate"))

        newcomer = detection("target-2", event_id="detection:target-2", detected_at=21.0)
        self.registry.handle_detection(newcomer)
        self.adapter.sync_tasks(self.registry.snapshots())
        relay.handle_detection(newcomer)
        relay.tick()
        fresh = relay._ledger["target-2"].last_sent
        # Robot 0 is the only subscriber, so robot 1 has heard nothing about target-2 yet.
        self.assertEqual(relay._deliver(fresh, "1"), (True, "accepted"))
        self.assertEqual(
            relay._deliver(fresh, "1"), (False, "delivery_duplicate"), "the same packet twice"
        )


if __name__ == "__main__":
    unittest.main()
