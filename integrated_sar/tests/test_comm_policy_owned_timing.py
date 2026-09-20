"""Two timing protocols on one transport, and what the new one must not touch.

``policy_owned_timing`` differs from the protocol every recorded experiment used in exactly one
mechanism: nothing goes on the air because a silence clock ran out. That makes the tests below a
pair of promises, and the second is the harder to keep.

**The new protocol really does hand the timing over.** An arm that asks for nothing is silent --
no forced refresh, no billed byte, no ``last_sent``, no belief update -- and a held change keeps
ageing instead of being spoken for.

**Everything else is untouched.** When the arm already asks for every candidate, the two protocols
produce the same transcript; the byte credit, the per-step capacity, the content and terminal
rules, message merging and the loss process behave the same either way; and the legacy protocol
still reproduces the forced-emission behaviour the archived rounds were measured on.

The last group is the equivalence claim the round rests on: ``first_send_plus_bound_3_10`` and
``bound_only_3_10`` must emit the same packets, at the same steps, in the same order, for the same
money as ``immediate`` and ``hold_to_bound`` did under the bound. Reason labels are expected to
differ -- a send the arm asked for is ``policy``, whatever the protocol -- so the comparison is
made on the emission sequence, with the labels reported beside it.
"""

from __future__ import annotations

import unittest
from typing import Any

from integrated_sar.comm import (
    TIMING_MODES,
    LEGACY_SHARED_BOUND,
    POLICY_OWNED_TIMING,
    CommRelay,
    CommTranscript,
)
from integrated_sar.comm_budget import CommBudget
from integrated_sar.comm_learning import FEATURE_NAMES
from integrated_sar.comm_policy import (
    BOUND_MAX_WAIT_STEPS,
    BOUND_URGENT_WAIT_STEPS,
    BOUND_URGENCY_TIME,
    AlwaysHoldPolicy,
    BoundOnlyPolicy,
    CommSendPolicy,
    FirstSendPlusBoundPolicy,
    HoldToBoundPolicy,
    ImmediatePolicy,
    PeriodicPolicy,
    POLICIES,
    SendOncePolicy,
    build_comm_policy,
    policy_descriptors,
    transport_order,
    transport_trim_key,
)
from integrated_sar.contracts import TaskStatus
from integrated_sar.joint_driver import RunConfiguration
from integrated_sar.task_registry import TaskRegistry
from integrated_sar.tests.test_comm_send_policy import RecordingWriter, detection, progress


class Observer(CommSendPolicy):
    """Records the candidates the relay offered, then asks the wrapped policy to decide.

    Read-only by construction: it never draws a packet, spends credit or touches the ledger, so a
    run with an observer attached has to be indistinguishable from the same run without one.
    """

    def __init__(self, inner: CommSendPolicy, sink: list[tuple[int, tuple[str, ...]]]) -> None:
        self._inner = inner
        self._sink = sink
        self.name = inner.name
        self.orders_selection = getattr(inner, "orders_selection", False)

    def parameters(self) -> dict[str, Any]:
        return self._inner.parameters()

    def decide(self, observation) -> tuple[str, ...]:
        self._sink.append((int(observation.step), tuple(item.target_id for item in observation.items)))
        return self._inner.decide(observation)


class Harness:
    """Registry, relay and receiver in one place, with the timing protocol as a dial.

    A copy of the round-1 harness rather than an import of it: that one builds a relay with no
    silence bound at all, and every test here is about the bound.
    """

    def __init__(
        self,
        *,
        policy: CommSendPolicy | None = None,
        timing_mode: str = LEGACY_SHARED_BOUND,
        bound: bool = True,
        budget_bytes: int | None = None,
        sends_per_tick: int | None = None,
        subscribers: tuple[str, ...] = ("0", "1"),
        delivery_probability: float = 1.0,
        seed: int = 7,
        observe: bool = True,
    ) -> None:
        self.registry = TaskRegistry()
        self.writer = RecordingWriter()
        self.transcript = CommTranscript()
        self.now = 0.0
        self.offered: list[tuple[int, tuple[str, ...]]] = []
        self.relay = CommRelay(
            subscribers=subscribers,
            writer=self.writer,
            snapshot_of=self.registry.get,
            accepted_of=self.registry.was_accepted,
            delivery_probability=delivery_probability,
            delay_ticks=0,
            seed=seed,
            policy=policy,
            budget=CommBudget(bytes_per_episode=budget_bytes, sends_per_step=sends_per_tick),
            max_wait_steps=BOUND_MAX_WAIT_STEPS if bound else None,
            urgent_wait_steps=BOUND_URGENT_WAIT_STEPS if bound else None,
            urgency_time=BOUND_URGENCY_TIME,
            timing_mode=timing_mode,
            freshness_of=self.writer.belief_freshness,
            now_of=lambda: self.now,
            transcript=self.transcript,
        )
        if observe:
            self.relay.attach_policy(Observer(self.relay.policy, self.offered))

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

    @property
    def emissions(self) -> int:
        return int(self.relay.counters["emissions"])

    def sends(self) -> list[tuple[int, str, int]]:
        """(step, target, index-within-step) per emission, in the order the transport took them."""
        counted: dict[int, int] = {}
        trace: list[tuple[int, str, int]] = []
        for record in self.transcript.records:
            if record["kind"] != "send":
                continue
            step = int(record["step"])
            position = counted.get(step, 0)
            counted[step] = position + 1
            trace.append((step, str(record["target"]), position))
        return trace

    def refusals(self) -> list[tuple[int, str, str, int]]:
        """(step, target, why, send_reason) for every send the transport would not pay for."""
        return [
            (int(r["step"]), str(r["target"]), str(r["reason"]), str(r.get("send_reason", "policy")))
            for r in self.transcript.records
            if r["kind"] == "refused"
        ]

    def belief(self) -> list[tuple[str, str, int, float]]:
        return list(self.writer.received)


def scripted_run(
    policy: CommSendPolicy,
    timing_mode: str,
    *,
    budget_bytes: int | None = None,
    bound: bool = True,
    **options: Any,
) -> Harness:
    """Two early targets, a late third, twenty steps of registry movement, one credit.

    ``far`` sits outside the urgency window and so obeys the 10-step bound; ``near`` is inside it
    from the first step and so obeys the 3-step bound. Both get a fresh view most steps, which is
    what a silence bound exists to bound. ``late`` is discovered halfway through on purpose: it
    makes a step where the arm has something to ask for *and* something else is overdue, which is
    the only kind of step where emission order decides who the credit pays for.
    """
    run = Harness(
        policy=policy, timing_mode=timing_mode, budget_bytes=budget_bytes, bound=bound, **options
    )
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


class ModeIsExplicitTests(unittest.TestCase):
    """The protocol a relay runs is a stated choice, never an accident of construction."""

    def test_a_default_relay_and_a_default_configuration_run_the_legacy_protocol(self) -> None:
        run = Harness(policy=HoldToBoundPolicy())
        parameters = run.relay.transport_parameters()
        self.assertEqual(parameters["timing_mode"], LEGACY_SHARED_BOUND)
        self.assertTrue(parameters["bound_enforced_by_transport"])
        self.assertEqual(RunConfiguration().comm_timing_mode, LEGACY_SHARED_BOUND)

    def test_the_new_mode_is_declared_by_the_relay_even_though_it_does_not_enforce(self) -> None:
        run = Harness(policy=AlwaysHoldPolicy(), timing_mode=POLICY_OWNED_TIMING)
        parameters = run.relay.transport_parameters()
        self.assertEqual(parameters["timing_mode"], POLICY_OWNED_TIMING)
        self.assertEqual(parameters["max_wait_steps"], BOUND_MAX_WAIT_STEPS)
        self.assertFalse(parameters["bound_enforced_by_transport"])

    def test_an_unknown_mode_is_refused_by_the_relay_and_by_the_configuration(self) -> None:
        with self.assertRaises(ValueError):
            Harness(timing_mode="policy_owned")
        with self.assertRaises(Exception) as caught:
            RunConfiguration(comm_timing_mode="whatever").validate()
        self.assertIn("--comm-timing-mode", str(caught.exception))
        self.assertEqual(sorted(TIMING_MODES), ["legacy_shared_bound", "policy_owned_timing"])

    def test_the_protocol_is_part_of_the_arm_label_and_of_the_hashed_configuration(self) -> None:
        from integrated_sar.allocation_trace import arm_name

        def label(mode: str) -> str:
            return arm_name(
                use_coordinator=True,
                residual_scale=0.5,
                comm_policy="event",
                comm_budget_bytes=29_300,
                comm_max_wait_steps=10,
                comm_urgent_wait_steps=3,
                comm_timing_mode=mode,
            )

        # A legacy label must not change: it is the name archived artefacts are filed under.
        self.assertEqual(label(LEGACY_SHARED_BOUND), label("legacy_shared_bound"))
        self.assertNotIn("policy_owned", label(LEGACY_SHARED_BOUND))
        self.assertEqual(label(POLICY_OWNED_TIMING), label(LEGACY_SHARED_BOUND) + "+policy_owned_timing")

        legacy = RunConfiguration(comm_policy="event")
        owned = RunConfiguration(comm_policy="event", comm_timing_mode=POLICY_OWNED_TIMING)
        self.assertEqual(
            set(legacy.as_dict()) - {"comm_timing_mode"},
            set(owned.as_dict()) - {"comm_timing_mode"},
        )
        self.assertNotEqual(legacy.as_dict()["comm_timing_mode"], owned.as_dict()["comm_timing_mode"])


class SilenceIsRealTests(unittest.TestCase):
    """Under the new protocol an arm that says nothing is silent, not secretly speaking."""

    def test_a_policy_that_asks_for_nothing_emits_nothing(self) -> None:
        legacy = scripted_run(HoldToBoundPolicy(), LEGACY_SHARED_BOUND)
        owned = scripted_run(AlwaysHoldPolicy(), POLICY_OWNED_TIMING)
        self.assertGreater(legacy.emissions, 0, "the bound speaks for an arm that holds")
        self.assertGreater(legacy.counters["emissions_forced_maxwait"], 0)
        self.assertEqual(owned.emissions, 0)
        self.assertEqual(owned.counters["emissions_forced_maxwait"], 0)
        self.assertEqual(owned.counters["emissions_policy"], 0)
        self.assertEqual(owned.counters["airtime_bytes"], 0)
        self.assertEqual(owned.refusals(), [], "nothing was asked, so nothing was refused")
        self.assertEqual(owned.relay.budget.spent_bytes, 0)
        self.assertEqual(owned.belief(), [], "no packet, so no belief update")

    def test_a_held_change_keeps_ageing_instead_of_being_spoken_for(self) -> None:
        owned = scripted_run(AlwaysHoldPolicy(), POLICY_OWNED_TIMING)
        self.assertGreater(owned.counters["unsent_change_steps"], 0)
        self.assertGreater(owned.counters["max_backlog_age"], 3)
        self.assertEqual(owned.emissions, 0)
        self.assertGreater(owned.counters["generated_items"], 0, "the sender still learns")

    def test_a_refresh_the_arm_did_not_ask_for_is_never_sent(self) -> None:
        # ``send_once`` asks only when a detection is accepted. Everything after that sits unsaid;
        # the same request set under the legacy transport gets refreshed by the bound.
        owned = scripted_run(SendOncePolicy(), POLICY_OWNED_TIMING)
        legacy = scripted_run(ImmediatePolicy(), LEGACY_SHARED_BOUND)
        self.assertEqual(owned.counters["first_emissions"], 3, "one per discovered target")
        self.assertEqual(owned.counters["refresh_emissions"], 0)
        self.assertEqual(owned.counters["emissions_forced_maxwait"], 0)
        self.assertGreater(legacy.counters["emissions_forced_maxwait"], 0)
        self.assertGreater(legacy.counters["refresh_emissions"], 0)

    def test_the_transport_never_rewrites_a_hold_into_a_send(self) -> None:
        """Half the bound's job is a floor, and the arm's answer must be able to be "no"."""
        run = scripted_run(AlwaysHoldPolicy(), POLICY_OWNED_TIMING)
        asked = [
            (step, targets) for step, targets in run.offered
        ]
        self.assertTrue(asked, "the policy was still consulted, so it had real chances to decline")
        self.assertEqual(run.sends(), [])

    def test_a_lost_packet_is_still_billed_and_is_never_retransmitted(self) -> None:
        run = Harness(
            policy=SendOncePolicy(), timing_mode=POLICY_OWNED_TIMING, delivery_probability=0.0, seed=3
        )
        run.publish(detection())
        run.step(2)
        self.assertEqual(run.counters["emissions"], 1)
        self.assertEqual(run.counters["transmission_attempts"], 2, "one attempt per subscriber")
        self.assertEqual(run.counters["packets_lost"], 2)
        self.assertEqual(run.counters["packets_delivered"], 0)
        self.assertEqual(run.belief(), [])
        self.assertGreater(run.relay.budget.spent_bytes, 0, "a lost packet is radio time spent")
        run.step(8)
        self.assertEqual(run.counters["emissions"], 1, "the relay does not re-send on its own")


class ConstraintsSurviveTests(unittest.TestCase):
    """What the new protocol keeps: credit, capacity, content, terminal state, the pipeline."""

    def test_a_send_that_cannot_be_paid_for_is_refused_and_changes_nothing(self) -> None:
        reference = scripted_run(FirstSendPlusBoundPolicy(), POLICY_OWNED_TIMING)
        payload = reference.counters["airtime_bytes"] // max(1, reference.emissions)
        run = scripted_run(FirstSendPlusBoundPolicy(), POLICY_OWNED_TIMING, budget_bytes=payload)
        self.assertEqual(run.emissions, 1, "only the first packet fits")
        self.assertLessEqual(run.relay.budget.spent_bytes, payload)
        self.assertGreater(run.counters["budget_refused_sends"], 0)
        self.assertEqual(run.counters["refused_forced_maxwait"], 0, "nothing is forced in this mode")
        self.assertGreaterEqual(run.counters["refused_policy"], 1)
        self.assertEqual(run.counters["emissions_policy"], 1)
        for _, target, why, reason in run.refusals():
            self.assertEqual(why, "budget_exhausted", f"{target} refused for {why} as {reason}")

    def test_the_per_step_capacity_still_trims_in_both_modes(self) -> None:
        for mode in (LEGACY_SHARED_BOUND, POLICY_OWNED_TIMING):
            run = Harness(policy=PeriodicPolicy(interval=1), timing_mode=mode, sends_per_tick=1)
            run.publish(detection("far", deadline_offset=500.0))
            run.publish(detection("near", deadline_offset=20.0))
            run.step()
            self.assertEqual(run.emissions, 1, mode)
            self.assertEqual(run.counters["capacity_refused_sends"], 1, mode)

    def test_a_closed_task_is_still_never_announced(self) -> None:
        for mode in (LEGACY_SHARED_BOUND, POLICY_OWNED_TIMING):
            run = Harness(policy=PeriodicPolicy(interval=2), timing_mode=mode)
            run.publish(detection())
            run.report(progress("target-1", observed_at=2.0, status=TaskStatus.COMPLETED, value=1.0))
            run.step(8)
            self.assertEqual(run.emissions, 0, mode)
            self.assertEqual(run.counters["emissions_suppressed_terminal"], 1, mode)

    def test_an_arm_that_asks_for_everything_sees_one_identical_transport(self) -> None:
        """The mode must change only the takeover, so with nothing to take over both are equal.

        ``periodic interval=1`` requests every candidate on every step in both protocols, so the
        bound has no overdue change left to add. Everything from candidate generation through
        merging, cost, delivery and belief therefore has to agree step for step; if the new mode
        had also cancelled a poll or a merge, this is where it would show.
        """
        legacy = scripted_run(PeriodicPolicy(interval=1), LEGACY_SHARED_BOUND)
        owned = scripted_run(PeriodicPolicy(interval=1), POLICY_OWNED_TIMING)
        self.assertEqual(legacy.sends(), owned.sends())
        self.assertEqual(legacy.offered, owned.offered)
        self.assertEqual(legacy.belief(), owned.belief())
        self.assertEqual(legacy.refusals(), owned.refusals())
        shared = (
            "emissions",
            "first_emissions",
            "refresh_emissions",
            "redundant_emissions",
            "generated_items",
            "coalesced_changes",
            "events_accepted",
            "transmission_attempts",
            "packets_delivered",
            "deliveries_applied",
            "airtime_bytes",
            "unsent_change_steps",
            "max_backlog_age",
        )
        self.assertEqual(
            {key: legacy.counters[key] for key in shared},
            {key: owned.counters[key] for key in shared},
        )

    def test_the_feature_layout_names_no_protocol_at_all(self) -> None:
        """The 19 inputs keep their meaning when the protocol moves.

        An arm whose features encoded who enforced the bound would not be comparing timing with
        timing. The layout is a name and a field list, and neither mentions the mechanism.
        """
        self.assertEqual(len(FEATURE_NAMES), 19)
        self.assertFalse(
            [name for name in FEATURE_NAMES if "mode" in name or "bound" in name or "forced" in name],
            "a feature that encodes the protocol would make the comparison about the observation",
        )

    def test_a_relay_without_a_bound_is_the_same_relay_in_either_mode(self) -> None:
        owned = scripted_run(SendOncePolicy(), POLICY_OWNED_TIMING, bound=False)
        legacy = scripted_run(SendOncePolicy(), LEGACY_SHARED_BOUND, bound=False)
        self.assertEqual(owned.sends(), legacy.sends())
        self.assertEqual(owned.counters["emissions"], legacy.counters["emissions"])


class LegacyUnchangedTests(unittest.TestCase):
    """The old protocol still does what the archived rounds were measured on."""

    def test_the_bound_still_spokes_for_every_arm_and_is_billed_per_reason(self) -> None:
        run = scripted_run(HoldToBoundPolicy(), LEGACY_SHARED_BOUND)
        self.assertGreater(run.counters["emissions_forced_maxwait"], 0)
        self.assertEqual(run.counters["emissions_policy"], 0, "the arm asked for nothing")
        self.assertEqual(
            run.counters["bytes_policy"] + run.counters["bytes_forced_maxwait"] + run.counters["bytes_other"],
            run.counters["airtime_bytes"],
        )
        self.assertEqual(
            run.counters["emissions_policy"]
            + run.counters["emissions_forced_maxwait"]
            + run.counters["emissions_other"],
            run.emissions,
        )

    def test_an_overdue_change_the_credit_cannot_pay_for_is_a_refusal_not_an_emission(self) -> None:
        reference = scripted_run(HoldToBoundPolicy(), LEGACY_SHARED_BOUND)
        payload = reference.counters["airtime_bytes"] // max(1, reference.emissions)
        run = scripted_run(HoldToBoundPolicy(), LEGACY_SHARED_BOUND, budget_bytes=payload)
        self.assertGreater(run.counters["refused_forced_maxwait"], 0)
        self.assertLess(run.counters["emissions_forced_maxwait"], reference.counters["emissions_forced_maxwait"])
        self.assertLessEqual(run.relay.budget.spent_bytes, payload)
        self.assertTrue(
            all(reason == "forced_maxwait" for _, _, _, reason in run.refusals()),
            "a bound arm asks for nothing, so every refusal there is a refused takeover",
        )

    def test_the_forced_send_goes_behind_the_requests_of_the_same_step(self) -> None:
        """Position in a step's emission list is part of the behaviour being reproduced.

        The bound appends what it takes over after the arm's own asks, so when the credit cannot
        pay for the lot, it is the takeover that loses. An equivalent rule has to reproduce that
        order, not merely the same set.
        """
        run = scripted_run(ImmediatePolicy(), LEGACY_SHARED_BOUND)
        steps_with_both = 0
        by_step: dict[int, list[tuple[int, str]]] = {}
        for step, target, position in run.sends():
            by_step.setdefault(step, []).append((position, target))
        reasons = {
            (int(record["step"]), str(record["target"])): str(record.get("send_reason"))
            for record in run.transcript.records
            if record["kind"] == "send"
        }
        for step, emitted in sorted(by_step.items()):
            forced = [
                position for position, target in emitted if reasons[(step, target)] == "forced_maxwait"
            ]
            asked = [
                position for position, target in emitted if reasons[(step, target)] == "policy"
            ]
            if forced and asked:
                steps_with_both += 1
                self.assertGreater(
                    min(forced), max(asked), f"step {step}: a takeover preceded an own request"
                )
        self.assertGreater(steps_with_both, 0, "the scenario must produce a step with both")




class RuleEquivalenceTests(unittest.TestCase):
    """The two named equivalents, the self-naming controls, and one order definition."""

    def test_first_send_plus_bound_reproduces_immediate_packet_for_packet(self) -> None:
        for budget in (None, 6_000, 12_000):
            with self.subTest(budget=budget):
                legacy = scripted_run(ImmediatePolicy(), LEGACY_SHARED_BOUND, budget_bytes=budget)
                owned = scripted_run(
                    FirstSendPlusBoundPolicy(), POLICY_OWNED_TIMING, budget_bytes=budget
                )
                self.assertEqual(legacy.sends(), owned.sends())
                self.assertEqual(legacy.counters["airtime_bytes"], owned.counters["airtime_bytes"])
                self.assertEqual(legacy.counters["transmission_attempts"], owned.counters["transmission_attempts"])
                self.assertEqual(legacy.counters["deliveries_applied"], owned.counters["deliveries_applied"])
                self.assertEqual(legacy.counters["budget_refused_sends"], owned.counters["budget_refused_sends"])
                self.assertEqual(legacy.refusals(), owned.refusals())
                self.assertEqual(legacy.belief(), owned.belief())
                self.assertGreater(legacy.counters["emissions_forced_maxwait"], 0)
                self.assertEqual(owned.counters["emissions_forced_maxwait"], 0)
                self.assertEqual(owned.counters["emissions_policy"], owned.emissions)

    def test_bound_only_reproduces_hold_to_bound_packet_for_packet(self) -> None:
        for budget in (None, 6_000, 12_000):
            with self.subTest(budget=budget):
                legacy = scripted_run(HoldToBoundPolicy(), LEGACY_SHARED_BOUND, budget_bytes=budget)
                owned = scripted_run(BoundOnlyPolicy(), POLICY_OWNED_TIMING, budget_bytes=budget)
                self.assertEqual(legacy.sends(), owned.sends())
                self.assertEqual(legacy.counters["airtime_bytes"], owned.counters["airtime_bytes"])
                self.assertEqual(legacy.refusals(), owned.refusals())
                self.assertEqual(legacy.belief(), owned.belief())
                self.assertEqual(owned.emissions, legacy.emissions)

    def test_a_bound_named_rule_refuses_parameters_it_does_not_quote(self) -> None:
        with self.assertRaises(ValueError):
            BoundOnlyPolicy(max_wait_steps=20)
        with self.assertRaises(ValueError):
            FirstSendPlusBoundPolicy(urgent_wait_steps=4)
        self.assertEqual(
            BoundOnlyPolicy().name, f"bound_only_{BOUND_URGENT_WAIT_STEPS}_{BOUND_MAX_WAIT_STEPS}"
        )
        self.assertEqual(
            build_comm_policy(
                "first_send_plus_bound_3_10",
                "max_wait_steps=10,urgent_wait_steps=3,urgency_time=25",
            ).name,
            "first_send_plus_bound_3_10",
        )
        with self.assertRaises(ValueError):
            build_comm_policy("first_send_plus_bound_3_10", "max_wait_steps=11")

    def test_send_once_and_always_hold_do_what_their_names_say(self) -> None:
        once = scripted_run(SendOncePolicy(), POLICY_OWNED_TIMING)
        self.assertEqual(once.counters["first_emissions"], 3, "one per discovered target")
        self.assertEqual(once.counters["refresh_emissions"], 0)
        held = scripted_run(AlwaysHoldPolicy(), POLICY_OWNED_TIMING)
        self.assertEqual(held.emissions, 0)
        self.assertEqual(held.counters["generated_items"], once.counters["generated_items"])

    def test_the_transport_order_is_one_definition_not_two(self) -> None:
        class Stub:
            def __init__(self, target: str, remaining: float, age: int) -> None:
                self.target_id, self.remaining_time, self.age_steps = target, remaining, age

        items = [Stub("b", 10.0, 1), Stub("a", 30.0, 5), Stub("c", 10.0, 4)]
        self.assertEqual(
            [item.target_id for item in transport_order(items)],
            [item.target_id for item in sorted(items, key=transport_trim_key)],
        )
        self.assertEqual([item.target_id for item in transport_order(items)], ["c", "b", "a"])

    def test_every_registered_rule_can_be_built_by_name(self) -> None:
        for expected in (
            "immediate",
            "hold_to_bound",
            "send_once",
            "always_hold",
            "bound_only_3_10",
            "first_send_plus_bound_3_10",
            "periodic",
            "event",
            "event_priority",
            "event_maxwait",
            "random_send",
        ):
            with self.subTest(rule=expected):
                self.assertIn(expected, POLICIES)
                self.assertEqual(build_comm_policy(expected).name, expected)
        self.assertNotIn(
            "abstract",
            {descriptor["name"] for descriptor in policy_descriptors()},
            "shared plumbing is not a selectable arm",
        )

    def test_an_observer_attached_policy_changes_nothing_about_the_run(self) -> None:
        """The diagnostic that reads the masks must not be a participant.

        Every harness here carries a recorder that sees the offered candidates and forwards the
        decision. The same script built without it has to emit the same packets for the same money
        and leave the receiver believing the same things.
        """
        observed = scripted_run(ImmediatePolicy(), LEGACY_SHARED_BOUND)
        quiet = scripted_run(ImmediatePolicy(), LEGACY_SHARED_BOUND, observe=False)
        self.assertEqual(observed.sends(), quiet.sends())
        self.assertEqual(observed.refusals(), quiet.refusals())
        self.assertEqual(observed.belief(), quiet.belief())
        self.assertEqual(observed.counters["airtime_bytes"], quiet.counters["airtime_bytes"])
        self.assertTrue(observed.offered, "the observed run really was offered candidates")
        self.assertEqual(quiet.offered, [], "and the quiet one was never consulted")




if __name__ == "__main__":  # pragma: no cover
    unittest.main()
