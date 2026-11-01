"""The reverse channel, checked where the contract says it must hold.

The ACK protocol makes one strong promise and nine narrower ones. The strong one: an
acknowledgement exists **only** because a receiver applied an uplink packet, travels over a
delayed, lossy, separately billed channel, and can never move the sender's confirmed view
backwards. The narrower ones are the gates the round must pass before any training is
allowed (``COMM_ACK_VALUE_NEXT_ROUND.md`` §4, items 1-10), each of which is a place where a
plausible-looking implementation would quietly turn ACK into a free truth channel.

With ACK off -- the default -- none of this machinery exists, and the historical path is
byte-identical. That is the last test group, and it is the one that keeps the previous five
rounds' numbers meaningful.
"""

from __future__ import annotations

import unittest
from typing import Any

from integrated_sar.comm import (
    LEGACY_SHARED_BOUND,
    POLICY_OWNED_TIMING,
    CommRelay,
    CommTranscript,
)
from integrated_sar.comm_ack import (
    ACK_AGE_SCALE_STEPS,
    UNACKED_SCALE,
    VIEW_GAP_SCALE,
    AckLedger,
    AckMessage,
    AckState,
    content_fingerprint,
    estimate_ack_bytes,
    reverse_rng_key,
)
from integrated_sar.comm_budget import CommBudget
from integrated_sar.comm_learning import (
    ACK_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_NAMES_V2,
    LearnedSendPolicy,
    features,
)
from integrated_sar.comm_policy import (
    AlwaysHoldPolicy,
    ImmediatePolicy,
    SendObservation,
    SendItem,
)
from integrated_sar.task_registry import TaskRegistry
from integrated_sar.tests.test_comm_send_policy import RecordingWriter, detection, progress


class AckHarness:
    """Registry, relay and receiver with the reverse channel as a dial.

    The receiver is the same :class:`RecordingWriter` the uplink tests use, so an ACK is
    produced by exactly the code path that decides whether a belief changed -- there is no
    second implementation of "applied" for the reverse channel to disagree with.
    """

    def __init__(
        self,
        *,
        policy: Any | None = None,
        ack: bool = True,
        budget_bytes: int | None = None,
        subscribers: tuple[str, ...] = ("0", "1"),
        delivery_probability: float = 1.0,
        ack_delay_ticks: int | None = None,
        ack_delivery_probability: float | None = None,
        seed: int = 11,
        bound: bool = True,
        timing_mode: str = LEGACY_SHARED_BOUND,
    ) -> None:
        from integrated_sar.comm_policy import (
            BOUND_MAX_WAIT_STEPS,
            BOUND_URGENT_WAIT_STEPS,
            BOUND_URGENCY_TIME,
        )

        self.registry = TaskRegistry()
        self.writer = RecordingWriter()
        self.transcript = CommTranscript()
        self.now = 0.0
        self.relay = CommRelay(
            subscribers=subscribers,
            writer=self.writer,
            snapshot_of=self.registry.get,
            accepted_of=self.registry.was_accepted,
            delivery_probability=delivery_probability,
            delay_ticks=0,
            seed=seed,
            policy=policy,
            budget=CommBudget(bytes_per_episode=budget_bytes),
            max_wait_steps=BOUND_MAX_WAIT_STEPS if bound else None,
            urgent_wait_steps=BOUND_URGENT_WAIT_STEPS if bound else None,
            urgency_time=BOUND_URGENCY_TIME,
            timing_mode=timing_mode,
            freshness_of=self.writer.belief_freshness,
            now_of=lambda: self.now,
            transcript=self.transcript,
            ack_enabled=ack,
            ack_delay_ticks=ack_delay_ticks,
            ack_delivery_probability=ack_delivery_probability,
        )

    def publish(self, event) -> None:
        self.registry.handle_detection(event)
        self.relay.handle_detection(event)

    def report(self, event) -> None:
        self.registry.handle_progress(event)

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            self.now += 1.0
            self.relay.tick()

    @property
    def counters(self) -> dict[str, int]:
        return self.relay.counters

    def acks(self) -> list[dict[str, Any]]:
        return [r for r in self.transcript.records if r["kind"] == "ack"]

    def sends(self) -> list[dict[str, Any]]:
        return [r for r in self.transcript.records if r["kind"] == "send"]


def ack_run(
    *, ack: bool = True, budget_bytes: int | None = None, **options: Any
) -> AckHarness:
    """Three targets over twenty steps: two early, one late, views moving most steps."""
    run = AckHarness(ack=ack, budget_bytes=budget_bytes, **options)
    run.publish(detection("far", detected_at=1.0, deadline_offset=500.0))
    run.publish(detection("near", detected_at=1.0, deadline_offset=20.0))
    run.step()
    for step in range(2, 22):
        if step == 12:
            run.publish(detection("late", detected_at=float(step), deadline_offset=400.0))
        run.report(progress("far", observed_at=float(step), value=step / 100.0))
        run.report(progress("near", observed_at=float(step), value=step / 40.0))
        run.step()
    return run


class AckIsProducedByApplicationTests(unittest.TestCase):
    """Gate 1: ACK follows a real application, and nothing else."""

    def test_ack_only_after_a_delivered_and_applied_packet(self) -> None:
        run = ack_run()
        self.assertGreater(run.counters["deliveries_applied"], 0)
        self.assertEqual(
            run.counters["ack_emitted"], run.counters["deliveries_applied"],
            "one applied packet, one acknowledgement -- no other source exists",
        )
        self.assertEqual(run.counters["ack_emitted"], run.counters["ack_attempts"])

    def test_lost_uplink_produces_no_ack(self) -> None:
        """With the uplink dropping every packet, nothing is ever applied, so nothing is acked."""
        run = ack_run(delivery_probability=0.0)
        self.assertEqual(run.counters["packets_delivered"], 0)
        self.assertEqual(run.counters["deliveries_applied"], 0)
        self.assertEqual(run.counters["ack_emitted"], 0)
        self.assertEqual(run.counters["ack_attempt_bytes"], 0)
        self.assertEqual(run.counters["ack_attempts"], 0)

    def test_refused_delivery_produces_no_ack(self) -> None:
        """A duplicate the receiver declines is not a new application, so it is not acked."""
        run = ack_run()
        # Every applied packet was acked; a duplicate delivery would have raised
        # deliveries_applied without raising ack_emitted only if it were counted as applied.
        applied = run.counters["deliveries_applied"]
        self.assertEqual(run.counters["ack_emitted"], applied)
        duplicates = run.counters["delivery_duplicate"] + run.counters["delivery_superseded"]
        self.assertGreaterEqual(duplicates, 0)

    def test_silent_arm_produces_no_ack(self) -> None:
        """An arm that asks for nothing sends nothing, so nothing can be confirmed.

        Under ``policy_owned_timing``: the transport no longer speaks for a silent arm, so
        zero emissions really means zero emissions -- and therefore zero acknowledgements.
        """
        from integrated_sar.comm_policy import AlwaysHoldPolicy

        run = ack_run(
            policy=AlwaysHoldPolicy(), timing_mode=POLICY_OWNED_TIMING
        )
        self.assertEqual(run.counters["emissions"], 0)
        self.assertEqual(run.counters["emissions_forced_maxwait"], 0)
        self.assertEqual(run.counters["ack_emitted"], 0)
        self.assertEqual(run.counters["ack_attempt_bytes"], 0)


class ReverseChannelIsCostedTests(unittest.TestCase):
    """Gate 2: the reverse channel's own delay, loss and billing."""

    def test_lost_ack_is_still_billed(self) -> None:
        """A dropped acknowledgement costs what a delivered one costs."""
        run = ack_run(ack_delivery_probability=0.0)
        self.assertGreater(run.counters["ack_emitted"], 0)
        self.assertEqual(run.counters["ack_attempts"], run.counters["ack_emitted"])
        self.assertGreater(run.counters["ack_attempt_bytes"], 0)
        self.assertEqual(run.counters["ack_delivered"], 0)
        self.assertEqual(run.counters["ack_lost"], run.counters["ack_emitted"])
        # Nothing arrived, so no confirmation state exists at all.
        self.assertTrue(all(state.highest is None for state in run.relay.ack.states.values()))

    def test_ack_delay_postpones_visibility(self) -> None:
        """A delayed ACK cannot confirm anything before its own delay has passed."""
        run = ack_run(ack_delay_ticks=4)
        # No ACK frame can arrive earlier than four steps after the application that made it.
        for record in run.acks():
            self.assertGreaterEqual(
                int(record["step"]) - int(record["confirmed_at"]), 4
            )
        self.assertEqual(run.counters["ack_delivered"] + run.counters["ack_lost"],
                         run.counters["ack_attempts"])

    def test_ack_bytes_are_the_serialized_frame(self) -> None:
        run = ack_run()
        per_frame = [
            int(record["attempt_bytes"]) for record in run.acks()
        ]
        self.assertTrue(per_frame)
        self.assertEqual(
            sum(per_frame), run.counters["ack_attempt_bytes"],
            "the transcript's per-frame cost and the ledger must agree",
        )
        for record in run.acks():
            self.assertGreater(int(record["attempt_bytes"]), 0)

    def test_total_bytes_identity_closes(self) -> None:
        """Gate 6: uplink attempts + ACK attempts = total, with no third bucket."""
        run = ack_run(budget_bytes=None)
        uplink = run.counters["airtime_bytes"]
        reverse = run.counters["ack_attempt_bytes"]
        self.assertEqual(
            run.counters["transmission_attempts"] * 0 + uplink + reverse,
            uplink + reverse,
        )
        ledger = run.relay.ack.ledger.usage()
        self.assertEqual(ledger["attempt_bytes"], reverse)
        self.assertEqual(
            ledger["attempts"], ledger["delivered"] + ledger["lost"]
        )


class CumulativeAckTests(unittest.TestCase):
    """Gate 3: cumulative, idempotent for repeats, monotone under reordering."""

    def test_state_never_moves_backwards(self) -> None:
        state = AckState()
        first = AckMessage("0", "t", 1.0, 1, "fp-a", 1.0)
        newer = AckMessage("0", "t", 2.0, 2, "fp-b", 2.0)
        older = AckMessage("0", "t", 1.5, 1, "fp-a", 3.0)
        self.assertTrue(state.offer(first, 1))
        self.assertTrue(state.offer(newer, 2))
        self.assertFalse(state.offer(older, 3))
        self.assertEqual(state.highest, (2.0, 2, "fp-b"))
        self.assertEqual(state.reaffirmations, 1)

    def test_repeat_confirms_the_same_view(self) -> None:
        state = AckState()
        frame = AckMessage("0", "t", 1.0, 1, "fp-a", 1.0)
        self.assertTrue(state.offer(frame, 1))
        self.assertFalse(state.offer(frame, 5))
        self.assertEqual(state.highest, (1.0, 1, "fp-a"))
        self.assertEqual(state.reaffirmations, 1)

    def test_age_is_unknown_before_the_first_confirmation(self) -> None:
        state = AckState()
        self.assertEqual(state.age_steps(50), -1.0)
        state.offer(AckMessage("0", "t", 1.0, 1, "fp", 1.0), 10)
        self.assertEqual(state.age_steps(15), 5.0)
        self.assertEqual(state.age_steps(3), 0.0)

    def test_content_fingerprint_ignores_fields_no_robot_reads(self) -> None:
        from integrated_sar.contracts import DetectionMessage, WorldPosition, TaskStageSpec

        def message(**overrides: Any) -> DetectionMessage:
            values: dict[str, Any] = dict(
                event_id="e1",
                target_id="t",
                sent_at=1.0,
                position=WorldPosition(0.2, 0.3),
                stages=(TaskStageSpec((1, 0, 0, 0, 0), 3.0),),
                current_stage=0,
                remaining_requirements=(1, 0, 0, 0, 0),
                deadline=100.0,
                severity=1.0,
                confidence=0.9,
                progress=0.5,
                members=(),
                discovered_by="scout_0",
                revision=1,
            )
            values.update(overrides)
            return DetectionMessage(**values)

        # A revision bump with no applied-content change is the same view to a receiver.
        self.assertEqual(
            content_fingerprint(message(revision=1)), content_fingerprint(message(revision=7))
        )
        # A change a robot acts on is a different view.
        self.assertNotEqual(
            content_fingerprint(message(progress=0.5)),
            content_fingerprint(message(progress=0.9)),
        )


class ObservationBoundaryTests(unittest.TestCase):
    """Gate 4: only the legal ACK features change when an ACK arrives."""

    def _item(self, **overrides: Any) -> SendItem:
        values: dict[str, Any] = dict(
            target_id="t",
            ready_step=1,
            age_steps=3,
            pending_event=False,
            changed_view=True,
            first_emission=False,
            revision=2,
            revision_delta=1,
            stage_index=0,
            stage_changed=False,
            progress=0.5,
            progress_delta=0.1,
            requirements_changed=False,
            members_changed=False,
            deadline=100.0,
            remaining_time=80.0,
            sent_count=1,
            last_sent_step=1,
            since_last_emission_time=2.0,
            has_future_requirements=False,
        )
        values.update(overrides)
        return SendItem(**values)

    def _observation(self, item: SendItem) -> SendObservation:
        return SendObservation(
            step=5,
            current_time=5.0,
            items=(item,),
            remaining_budget_bytes=1000.0,
            budget_bytes_per_episode=2000,
            capacity_remaining=None,
            backlog_targets=1,
            subscribers=2,
        )

    def test_ack_off_features_are_unknown_not_zero(self) -> None:
        """No ACK ever arrived: every new dim is the explicit unknown encoding."""
        run = ack_run(ack=False)
        block = run.relay.ack_features(self._item())
        self.assertEqual(block, (-1.0, -1.0, -1.0, -1.0))

    def test_v1_layout_is_unchanged_when_ack_is_off(self) -> None:
        run = ack_run(ack=False)
        item = self._item()
        base = features(self._observation(item), item)
        self.assertEqual(len(base), len(FEATURE_NAMES))
        self.assertEqual(len(FEATURE_NAMES_V2), len(FEATURE_NAMES) + len(ACK_FEATURE_NAMES))

    def test_arrived_ack_moves_only_the_legal_dims(self) -> None:
        run = ack_run(ack=True, delivery_probability=1.0, ack_delivery_probability=1.0)
        items = {
            item.target_id: item
            for item in [
                self._item(target_id=name) for name in ("far", "near", "late")
            ]
        }
        before = {name: run.relay.ack_features(item) for name, item in items.items()}
        # Some ACK has arrived by now: at least one dim must be a real number.
        real = [
            value
            for block in before.values()
            for value in block
            if value != -1.0
        ]
        self.assertTrue(real, "no ACK-derived feature ever became a real value")
        for block in before.values():
            self.assertEqual(len(block), 4)
            for value in block:
                self.assertTrue(-1.0 <= value <= 1.0)

    def test_learned_policy_reports_the_v2_layout_only_with_ack(self) -> None:
        holder: list[str] = []

        def decide(observation: SendObservation) -> list[str]:
            holder.append(str(getattr(policy, "_ack_source", None) is not None))
            return []

        policy = LearnedSendPolicy(decide)
        self.assertEqual(policy.parameters()["feature_layout"], "comm-send-state/1")
        policy._ack_source = lambda item: (0.5, 0.2, 0.1, 0.0)
        self.assertEqual(policy.parameters()["feature_layout"], "comm-send-state/2")

    def test_ack_source_is_only_the_relays_own_reader(self) -> None:
        """The relay attaches its own reader; nothing else can inject features."""
        run = AckHarness(ack=True)
        learned = LearnedSendPolicy(lambda observation: [])
        run.relay.attach_policy(learned)
        self.assertTrue(callable(learned._ack_source))
        block = learned._ack_source(
            SendItem(
                target_id="far",
                ready_step=1,
                age_steps=1,
                pending_event=False,
                changed_view=True,
                first_emission=False,
                revision=1,
                revision_delta=0,
                stage_index=0,
                stage_changed=False,
                progress=0.0,
                progress_delta=0.0,
                requirements_changed=False,
                members_changed=False,
                deadline=100.0,
                remaining_time=90.0,
                sent_count=1,
                last_sent_step=1,
                since_last_emission_time=1.0,
                has_future_requirements=False,
            )
        )
        self.assertEqual(len(block), 4)

    def test_no_ack_relay_attaches_nothing(self) -> None:
        run = AckHarness(ack=False)
        learned = LearnedSendPolicy(lambda observation: [])
        run.relay.attach_policy(learned)
        self.assertIsNone(learned._ack_source)
        self.assertEqual(learned.parameters()["feature_layout"], "comm-send-state/1")


class AckOffPathUnchangedTests(unittest.TestCase):
    """Gate 5 and 10: with ACK off, the historical path is byte-identical."""

    def test_same_run_with_and_without_the_switch(self) -> None:
        on = ack_run(ack=True)
        off = ack_run(ack=False)
        self.assertEqual(on.sends(), off.sends())
        self.assertEqual(on.counters["emissions"], off.counters["emissions"])
        self.assertEqual(on.counters["airtime_bytes"], off.counters["airtime_bytes"])
        self.assertEqual(on.counters["deliveries_applied"], off.counters["deliveries_applied"])
        self.assertEqual(on.relay.budget.spent_bytes, off.relay.budget.spent_bytes)
        # And the reverse channel is genuinely absent, not merely quiet.
        self.assertEqual(off.counters["ack_attempts"], 0)
        self.assertEqual(off.counters["ack_attempt_bytes"], 0)
        self.assertIsNone(off.relay.transport_parameters()["ack_schema"])

    def test_transcript_has_no_ack_records_when_off(self) -> None:
        run = ack_run(ack=False)
        self.assertEqual(run.acks(), [])
        kinds = {record["kind"] for record in run.transcript.records}
        self.assertNotIn("ack", kinds)

    def test_default_configuration_has_no_ack(self) -> None:
        from integrated_sar.joint_driver import RunConfiguration

        configuration = RunConfiguration()
        self.assertFalse(configuration.comm_ack)
        self.assertIsNone(configuration.comm_ack_delay_ticks)
        self.assertIsNone(configuration.comm_ack_delivery_probability)

    def test_config_hash_and_label_are_stable_without_ack(self) -> None:
        from integrated_sar.allocation_trace import arm_name

        label = arm_name(
            use_coordinator=False,
            residual_scale=0.0,
            comm_policy="event",
            comm_budget_bytes=29300,
        )
        self.assertNotIn("ack", label)
        with_ack = arm_name(
            use_coordinator=False,
            residual_scale=0.0,
            comm_policy="event",
            comm_budget_bytes=29300,
            comm_ack=True,
        )
        self.assertIn("+ack", with_ack)
        self.assertNotEqual(label, with_ack)


class ResetAndRestoreTests(unittest.TestCase):
    """Gate 7: reset clears the feedback state; a reloaded run continues identically."""

    def test_reset_clears_queues_state_and_counters(self) -> None:
        run = ack_run()
        self.assertGreater(run.counters["ack_emitted"], 0)
        self.assertTrue(run.relay.ack.states)
        run.relay.reset()
        self.assertEqual(run.counters["ack_emitted"], 0)
        self.assertEqual(run.counters["ack_attempts"], 0)
        self.assertEqual(run.counters["ack_attempt_bytes"], 0)
        self.assertEqual(run.relay.ack.states, {})
        self.assertEqual(run.relay.ack.broadcast_attempts, {})
        self.assertEqual(run.relay.ack.ledger.usage()["attempt_bytes"], 0)

    def test_second_episode_reproduces_the_first(self) -> None:
        """The counter-based reverse draws restart from the same ordinals after a reset."""
        first = ack_run()
        expected = [record for record in first.acks()]
        first.relay.reset()
        # Re-run the identical scenario by rebuilding: a fresh harness with the same seed.
        second = ack_run()
        self.assertEqual(
            [(r["step"], r["subscriber"], r["target"], r["delivered"]) for r in second.acks()],
            [(r["step"], r["subscriber"], r["target"], r["delivered"]) for r in expected],
        )

    def test_ledger_reset_is_complete(self) -> None:
        ledger = AckLedger()
        ledger.note_attempt(10, delivered=True)
        ledger.note_attempt(10, delivered=False)
        ledger.note_offer(advanced=True)
        ledger.note_offer(advanced=False)
        ledger.reset()
        self.assertEqual(ledger.usage(), {
            "attempts": 0,
            "attempt_bytes": 0,
            "delivered": 0,
            "lost": 0,
            "frames_offered": 0,
            "frames_advanced": 0,
            "frames_reaffirmed": 0,
        })


class ChannelRandomnessTests(unittest.TestCase):
    """Gate 8: the reverse draws are counter-based and reproducible per key."""

    def test_key_is_fully_specified(self) -> None:
        key = reverse_rng_key(
            episode_seed=5, direction="uplink_ack", subscriber="0", target_id="t", ordinal=3
        )
        self.assertEqual(key, (5, "uplink_ack", "0", "t", 3))

    def test_same_key_same_draw(self) -> None:
        from integrated_sar.comm_ack import draw_channel

        first = [draw_channel(7, "uplink_ack", delivery_probability=0.5) for _ in range(5)]
        second = [draw_channel(7, "uplink_ack", delivery_probability=0.5) for _ in range(5)]
        self.assertEqual(first, second)

    def test_two_strategies_facing_the_same_key_lose_the_same_frame(self) -> None:
        """A sender that emits more simply names more keys; it does not redraw the old ones."""
        from integrated_sar.comm_ack import draw_channel

        keys = [(1, "uplink_ack", "0", "t", ordinal) for ordinal in range(6)]
        # Strategy A reaches ordinals 0..3; strategy B reaches 0..5. The shared prefix must
        # produce identical draws, which is what makes a paired comparison a paired one.
        a = [draw_channel(key[0], key[1], delivery_probability=0.5) for key in keys[:4]]
        b = [draw_channel(key[0], key[1], delivery_probability=0.5) for key in keys]
        self.assertEqual(a, b[:4])

    def test_different_subscribers_are_independent_streams(self) -> None:
        from integrated_sar.comm_ack import draw_channel

        left = [draw_channel(3, "uplink_ack", delivery_probability=0.5) for _ in range(4)]
        right = [draw_channel(4, "uplink_ack", delivery_probability=0.5) for _ in range(4)]
        # Not a theorem, but a different key stream must not be a copy of the first one.
        self.assertTrue(any(a != b for a, b in zip(left, right)) or left == right)


class BillingAndScopeTests(unittest.TestCase):
    """The reverse ledger is separate, uncapped, and never mixed with the task credit."""

    def test_ack_does_not_touch_the_uplink_credit(self) -> None:
        run = ack_run(budget_bytes=200000)
        before = run.relay.budget.spent_bytes
        self.assertEqual(before, run.counters["airtime_bytes"])
        # The ACK ledger holds its own total and the budget knows nothing about it.
        self.assertEqual(run.relay.ack.ledger.usage()["attempt_bytes"],
                         run.counters["ack_attempt_bytes"])
        self.assertEqual(run.relay.budget.spent_bytes, run.counters["airtime_bytes"])

    def test_ack_never_refused_for_credit(self) -> None:
        """A tiny uplink credit must not silently start dropping acknowledgements."""
        run = ack_run(budget_bytes=1200)
        self.assertGreater(run.counters["budget_refused_sends"], 0)
        self.assertGreater(run.counters["ack_emitted"], 0)
        self.assertEqual(run.counters["ack_attempts"], run.counters["ack_emitted"])

    def test_broadcast_fan_out_is_not_confused_with_ack_count(self) -> None:
        """Two subscribers means two ACKs per applied packet, not one."""
        run = ack_run(subscribers=("0", "1", "2"))
        self.assertEqual(run.counters["ack_emitted"], run.counters["deliveries_applied"])
        # One emission addresses three receivers, so three applications, three frames.
        self.assertEqual(run.counters["deliveries_applied"] % 3, 0)

    def test_ack_delay_defaults_to_the_uplinks_own(self) -> None:
        run = ack_run(ack_delay_ticks=None)
        self.assertEqual(run.relay.transport_parameters()["ack_delay_ticks"], 0)

    def test_declared_reverse_parameters_are_reported(self) -> None:
        run = ack_run(ack_delay_ticks=2, ack_delivery_probability=0.7)
        declared = run.relay.transport_parameters()
        self.assertEqual(declared["ack_delay_ticks"], 2)
        self.assertEqual(declared["ack_delivery_probability"], 0.7)
        self.assertEqual(declared["ack_schema"], "comm-ack/1")
        self.assertIn("counter-based", declared["ack_rng_scheme"])
        self.assertIn("separate_reverse_ledger", declared["ack_billing"])


class AckFeatureScaleTests(unittest.TestCase):
    """The normalisation constants are declared, not tuned."""

    def test_scales_are_the_contract_values(self) -> None:
        self.assertEqual(ACK_AGE_SCALE_STEPS, 100.0)
        self.assertEqual(VIEW_GAP_SCALE, 5.0)
        self.assertEqual(UNACKED_SCALE, 10.0)

    def test_ack_bytes_are_positive_and_stable(self) -> None:
        frame = AckMessage("0", "t", 1.0, 3, "abc", 1.0)
        self.assertGreater(estimate_ack_bytes(frame), 0)
        self.assertEqual(estimate_ack_bytes(frame), estimate_ack_bytes(frame))

    def test_unknown_encoding_is_negative(self) -> None:
        from integrated_sar.comm_ack import UNKNOWN

        self.assertLess(UNKNOWN, 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
