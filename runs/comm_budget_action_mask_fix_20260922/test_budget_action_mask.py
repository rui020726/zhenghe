"""The sequential budget mask, tested through the seam it is meant to fix.

These are correctness tests for shipped behaviour, so they drive the real objects: a real
``CommRelay`` producing the real candidate metadata, the real ``OnPolicyCollector`` it is meant to
replace, the real ``CommBudget`` gate, and the real PPO alignment check that consumes the tape.
Synthetic-only coverage would miss exactly the failure this round is about -- a collector whose
notion of "affordable" and a transport's notion of "affordable" disagreeing.

    python -m unittest discover -s . -p 'test_*.py' -v
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _directory in (
    str(ROOT),
    str(ROOT / "runs" / "comm_budget_pilot_20260919"),
    str(ROOT / "runs" / "comm_stability_20260919"),
    str(ROOT / "runs" / "comm_tight_budget_20260919"),
    str(ROOT / "runs" / "comm_protocol_v2_20260920"),
    str(HERE),
):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

import train_budget_masked as entry  # noqa: E402
from budget_masked import BudgetMaskedCollector  # noqa: E402
from integrated_sar.comm import CommRelay, estimate_message_bytes  # noqa: E402
from integrated_sar.comm_budget import CommBudget  # noqa: E402
from integrated_sar.comm_learning import LearnedSendPolicy  # noqa: E402
from integrated_sar.comm_policy import (  # noqa: E402
    SendItem,
    SendObservation,
)
from integrated_sar.contracts import (  # noqa: E402
    DetectionEvent,
    GridPosition,
    ProgressEvent,
    TargetProfile,
    TaskStageSpec,
    TaskStatus,
    WorldPosition,
)
from integrated_sar.task_registry import TaskRegistry  # noqa: E402
from onpolicy import OnPolicyCollector, action_entropy, joint_log_prob  # noqa: E402

SEND, HOLD = 1, 0


class FixedController:
    """A two-logit network stand-in: it makes preferences exact instead of sampling luck."""

    def __init__(self, *, send_bias: float = 0.0, value: float = 0.5) -> None:
        self.send_bias = float(send_bias)
        self.value = float(value)
        self.calls = 0

    def forward(self, matrix: torch.Tensor):
        self.calls += 1
        rows = int(np.asarray(matrix).shape[0])
        logits = torch.tensor([[-self.send_bias, self.send_bias] for _ in range(rows)], dtype=torch.float32)
        values = torch.full((rows,), self.value, dtype=torch.float32)
        return logits, values

    @property
    def parameters(self):
        return [torch.nn.Parameter(torch.zeros(1))]


def observation_of(items, *, budget_bytes=None, capacity=None, spent=0, step=1) -> SendObservation:
    budget = CommBudget(bytes_per_episode=budget_bytes, sends_per_step=capacity)
    if spent:
        budget.spent_bytes = int(spent)
    return SendObservation(
        step=step,
        current_time=float(step),
        items=tuple(items),
        remaining_budget_bytes=budget.remaining_bytes,
        budget_bytes_per_episode=budget_bytes,
        capacity_remaining=(
            None if capacity is None else max(0, int(capacity) - int(budget.sends_this_step))
        ),
        backlog_targets=len(items),
        subscribers=2,
    )


def candidate(
    target_id: str,
    *,
    charge: int,
    budget_bytes: int | None,
    executable: bool | None = None,
    reason: str | None = None,
    remaining: float = 400.0,
) -> SendItem:
    budget = CommBudget(bytes_per_episode=budget_bytes)
    if executable is None:
        executable = None if charge < 0 else budget.can_send(charge)
    if reason is None:
        reason = "allowed" if charge < 0 else budget.why_blocked(charge)
    return SendItem(
        target_id=target_id,
        ready_step=0,
        age_steps=1,
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
        deadline=remaining,
        remaining_time=remaining,
        sent_count=0,
        last_sent_step=None,
        since_last_emission_time=None,
        has_future_requirements=False,
        attempt_charge_bytes=(None if charge < 0 else charge),
        send_executable=executable,
        send_block_reason=reason,
    )


def masked(controller, **kwargs):
    kwargs.setdefault("forced", lambda item: False)
    return BudgetMaskedCollector(controller, **kwargs)


class SequentialMaskTests(unittest.TestCase):
    """An unaffordable SEND is not an action, so it must not be sampled or logged."""

    def test_unaffordable_candidate_is_held_and_carries_no_likelihood(self) -> None:
        controller = FixedController(send_bias=+50.0)  # wants to send, loudly
        observation = observation_of(
            [candidate("target-1", charge=1000, budget_bytes=999)], budget_bytes=999
        )
        asked, actions, free, log_prob, value, tape = masked(controller).choose(observation)
        self.assertEqual(asked, [])
        self.assertEqual(actions["target-1"], HOLD)
        self.assertEqual(free, [])
        self.assertEqual(log_prob, 0.0)
        self.assertEqual(tape["masked"], ["target-1"])
        self.assertEqual(tape["masked_reasons"]["target-1"], "budget_exhausted")
        self.assertEqual(tape["actionable"], [])
        self.assertFalse(tape["requested"]["target-1"])
        self.assertEqual(tape["shadow_budget_after"], tape["shadow_budget_before"])
        self.assertEqual(value, 0.5, "the critic still sees the state")

    def test_the_first_send_decides_whether_the_second_can_send(self) -> None:
        observation = observation_of(
            [
                candidate("target-1", charge=600, budget_bytes=1000),
                candidate("target-2", charge=600, budget_bytes=1000),
            ],
            budget_bytes=1000,
        )
        asked, _actions, free, _lp, _v, tape = masked(
            FixedController(send_bias=+50.0)
        ).choose(observation)
        self.assertEqual(asked, ["target-1"], "the credit covers exactly one of the two")
        self.assertEqual(free, ["target-1"])
        self.assertEqual(tape["masked"], ["target-2"])
        self.assertEqual(tape["masked_reasons"]["target-2"], "budget_exhausted")

        holding = masked(FixedController(send_bias=-50.0)).choose(observation)
        self.assertEqual(holding[0], [])
        self.assertEqual(holding[2], ["target-1", "target-2"], "both stay actionable if nothing is spent")

    def test_capacity_one_is_an_order_constraint_not_a_permanent_block(self) -> None:
        observation = observation_of(
            [
                candidate("target-1", charge=10, budget_bytes=None),
                candidate("target-2", charge=10, budget_bytes=None),
            ],
            budget_bytes=None,
            capacity=1,
        )
        _asked, _actions, free, _lp, _v, tape = masked(FixedController(send_bias=+50.0)).choose(
            observation
        )
        self.assertEqual(free, ["target-1"])
        self.assertEqual(tape["masked_reasons"]["target-2"], "capacity_step")

    def test_requested_charges_never_outrun_the_credit_and_execution_refuses_nothing(self) -> None:
        controller = FixedController(send_bias=+8.0)
        collector = masked(controller, greedy=False)
        for seed in (1, 2, 3, 4, 5):
            torch.manual_seed(seed)
            random.seed(seed)
            observation = observation_of(
                [
                    candidate(f"target-{index}", charge=300, budget_bytes=1000)
                    for index in (1, 2, 3, 4)
                ],
                budget_bytes=1000,
            )
            asked, _actions, _free, _lp, _v, tape = collector.choose(observation)
            spent = sum(tape["attempt_charge_bytes"][target] for target in asked)
            self.assertLessEqual(
                spent, float(tape["shadow_budget_before"]), "asked for more than the step had"
            )
            self.assertTrue(all(tape["requested"][target] for target in asked))

    def test_greedy_and_stochastic_paths_share_the_actionable_set(self) -> None:
        observation = observation_of(
            [
                candidate("target-1", charge=600, budget_bytes=1000),
                candidate("target-2", charge=600, budget_bytes=1000),
            ],
            budget_bytes=1000,
        )
        greedy = masked(FixedController(send_bias=+50.0), greedy=True).choose(observation)
        torch.manual_seed(3)
        stochastic = masked(FixedController(send_bias=+50.0), greedy=False).choose(observation)
        self.assertEqual(greedy[2], stochastic[2])
        self.assertEqual(greedy[5]["actionable"], stochastic[5]["actionable"])
        self.assertEqual(greedy[5]["masked"], stochastic[5]["masked"])


class ProbabilityTests(unittest.TestCase):
    """The mask is part of the action's probability, and the update can recompute it."""

    def test_logged_and_recomputed_log_prob_agree_to_one_in_a_million(self) -> None:
        controller = FixedController(send_bias=+1.0)
        collector = masked(controller)
        torch.manual_seed(11)
        for step in range(1, 6):
            # Each step starts from the previous step's spending, so the credit is what the
            # shadow has to reason about, not a fresh allowance per decision point.
            observation = observation_of(
                [
                    candidate(f"target-{index}", charge=400, budget_bytes=1200 * step)
                    for index in (1, 2, 3)
                ],
                budget_bytes=1200 * step,
                spent=0,
                step=step,
            )
            collector.decide(observation)
        check = entry.drawn_versus_logged(controller, collector.records)
        self.assertGreaterEqual(check["records"], 5)
        self.assertLessEqual(check["max_abs_gap"], 1e-6)

    def test_the_check_fails_when_the_actionable_set_is_a_lie(self) -> None:
        controller = FixedController(send_bias=+1.0)
        collector = masked(controller)
        torch.manual_seed(5)
        observation = observation_of(
            [
                candidate("target-1", charge=600, budget_bytes=1000),
                candidate("target-2", charge=600, budget_bytes=1000),
            ],
            budget_bytes=1000,
        )
        collector.decide(observation)
        self.assertEqual(collector.records[0]["masked"], ["target-2"])
        record = dict(collector.records[0])
        # Claiming target-2 was a free choice when the credit had already decided it must make
        # the recomputed likelihood differ from the one that was logged.
        record["free"] = sorted(set(record["free"]) | {"target-2"})
        check = entry.drawn_versus_logged(controller, [record])
        self.assertGreater(check["max_abs_gap"], 1e-6)

    def test_a_fully_masked_step_contributes_no_actor_gradient_but_keeps_the_critic_row(self) -> None:
        controller = FixedController(send_bias=+50.0)
        observation = observation_of(
            [candidate("target-1", charge=5000, budget_bytes=1000)], budget_bytes=1000
        )
        _asked, _actions, _free, log_prob, value, tape = masked(controller).choose(observation)
        logits = torch.tensor([[-50.0, 50.0]], dtype=torch.float32)
        usable = torch.tensor([0.0])
        picked = torch.tensor([HOLD], dtype=torch.long)
        self.assertEqual(log_prob, 0.0)
        self.assertEqual(float(joint_log_prob(logits, picked, usable).item()), 0.0)
        self.assertEqual(float(action_entropy(logits, usable).item()), 0.0)
        self.assertGreater(value, 0.0, "the critic keeps a sample for a step with no actor choice")
        episodes = [{
            "controller": controller,
            "decisions": [tape],
            "rewards": [1.0],
        }]
        flat, checks = entry.flatten(episodes)
        self.assertEqual(len(flat), 1, "the step is not dropped from the buffer")
        self.assertEqual(flat[0]["free"], set())
        self.assertAlmostEqual(float(flat[0]["return"]), 1.0, places=3)


class LegacyIsolationTests(unittest.TestCase):
    """``budget_mask=False`` is the archived sampler, not a look-alike of it."""

    def test_archived_collector_and_mask_off_agree_on_every_field(self) -> None:
        items = [
            candidate("target-1", charge=100, budget_bytes=10_000),
            candidate("target-2", charge=100, budget_bytes=10_000),
        ]
        controller = FixedController(send_bias=+0.4)
        torch.manual_seed(17)
        archived = OnPolicyCollector(controller, forced=lambda item: False, mask=True).choose(
            observation_of(items, budget_bytes=10_000)
        )
        torch.manual_seed(17)
        passthrough = masked(controller, budget_mask=False).choose(
            observation_of(items, budget_bytes=10_000)
        )
        self.assertEqual(archived[0], passthrough[0])
        self.assertEqual(archived[1], passthrough[1])
        self.assertEqual(archived[2], passthrough[2])
        self.assertAlmostEqual(archived[3], passthrough[3], places=12)
        self.assertEqual(archived[5]["free"], passthrough[5]["free"])
        self.assertNotIn("actionable", archived[5], "the archived tape keeps its own schema")
        self.assertNotIn("attempt_charge_bytes", archived[5])

    def test_a_candidate_without_a_price_is_refused_not_assumed_free(self) -> None:
        legacy = SendItem(
            target_id="target-9",
            ready_step=0,
            age_steps=1,
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
            deadline=400.0,
            remaining_time=400.0,
            sent_count=0,
            last_sent_step=None,
            since_last_emission_time=None,
            has_future_requirements=False,
        )
        with self.assertRaisesRegex(RuntimeError, "no attempt charge metadata"):
            masked(FixedController()).choose(observation_of([legacy], budget_bytes=1000))


class RelaySeamTests(unittest.TestCase):
    """relay observation -> collector -> relay _emit, with no stub in the middle."""

    def setUp(self) -> None:
        self.registry = TaskRegistry()
        self.now = 0.0
        self.applied: list[tuple[str, str]] = []

        class Writer:
            def __init__(self, owner):
                self.owner = owner

            def deliver_detection(self, message, agent_id):
                self.owner.applied.append((str(agent_id), str(message.target_id)))
                return True

            def belief_freshness(self, agent_id, target_id):
                return None

        self.writer = Writer(self)
        self.controller = FixedController(send_bias=+6.0)
        self.collector = masked(self.controller)
        self.relay = CommRelay(
            subscribers=("0", "1"),
            writer=self.writer,
            snapshot_of=self.registry.get,
            accepted_of=self.registry.was_accepted,
            delivery_probability=1.0,
            delay_ticks=0,
            seed=23,
            policy=LearnedSendPolicy(self.collector.decide),
            budget=CommBudget(bytes_per_episode=29_300, sends_per_step=None),
            max_wait_steps=None,
            urgent_wait_steps=None,
            timing_mode="policy_owned_timing",
            now_of=lambda: self.now,
        )
        for index, stages in ((1, 1), (2, 3), (10, 2)):
            target_id = f"target-{index}"
            event = DetectionEvent(
                event_id=f"detection:{target_id}",
                target_id=target_id,
                grid_position=GridPosition(59, 59),
                world_position=WorldPosition(1.0, 1.0),
                detected_at=1.0,
                discovered_by="scout_0",
                confidence=1.0,
                profile=TargetProfile(
                    stages=tuple(TaskStageSpec((1, 0, 0, 0, 0), 3.0) for _ in range(stages)),
                    deadline_offset=500.0,
                ),
            )
            self.registry.handle_detection(event)
            self.relay.handle_detection(event)

    def tick(self) -> None:
        self.now += 1.0
        self.relay.tick()

    def test_the_transport_never_refuses_a_send_the_collector_asked_for(self) -> None:
        for _ in range(12):
            self.tick()
        ledger = self.relay.transport_parameters()
        self.assertEqual(int(self.relay.counters["budget_refused_sends"]), 0)
        self.assertEqual(int(self.relay.counters["capacity_refused_sends"]), 0)
        self.assertEqual(int(self.relay.counters["refused_policy"]), 0)
        self.assertGreater(int(self.relay.counters["emissions"]), 0, "the seam did run")
        self.assertTrue(ledger["budget"]["bytes_per_episode"] <= 29_300)

    def test_emission_order_is_the_candidate_order_the_mask_walked(self) -> None:
        requested = 0
        for _ in range(3):
            self.tick()
        for record in self.collector.records:
            # The relay offers candidates in sorted-target order and executes the policy's
            # selection in that same order, which is what lets a shadow credit walk the list.
            self.assertEqual(record["targets"], sorted(record["targets"]))
            self.assertEqual(
                [target for target in record["actionable"] if record["requested"][target]],
                sorted(target for target in record["actionable"] if record["requested"][target]),
            )
            self.assertTrue(all(record["actions"][target] == HOLD for target in record["masked"]))
            requested += sum(1 for on in record["requested"].values() if on)
        self.assertGreater(requested, 0)
        self.assertEqual(int(self.relay.counters["emissions"]), requested)
        self.assertEqual(self.relay.counters["emissions_suppressed_terminal"], 0)

    def test_tape_invariants_hold_on_a_real_episode(self) -> None:
        for _ in range(10):
            self.tick()
        report = entry.tape_invariants(self.collector.records)
        self.assertTrue(report["passed"], report["violations"][:3])
        self.assertGreaterEqual(report["tapes"], 5)

    def test_credit_exhaustion_masks_everything_after_it_and_actor_terms_stop(self) -> None:
        for _ in range(60):
            self.tick()
        spent = float(self.relay._budget.spent_bytes)
        self.assertLessEqual(spent, 29_300)
        tail = self.collector.records[-5:]
        self.assertTrue(all(not record["actionable"] for record in tail),
                        "once the credit is gone no candidate may remain actionable")
        self.assertTrue(all(record["log_prob"] == 0.0 for record in tail))
        self.assertEqual(int(self.relay.counters["budget_refused_sends"]), 0)


LEGACY_CHECKPOINT = (
    ROOT / "runs" / "comm_protocol_v2_20260920" / "train" / "25pct_seed1"
    / "v2_25pct_seed1_at400000.pt"
)


class LegacyCheckpointTests(unittest.TestCase):
    """An old checkpoint is a forward-pass artefact, not a place to continue optimising from."""

    @unittest.skipUnless(LEGACY_CHECKPOINT.exists(), "the round-5 400k checkpoint is not here")
    def test_old_checkpoint_loads_forward_and_is_refused_for_resume(self) -> None:
        import torch

        info = entry.load_forward_only(LEGACY_CHECKPOINT, training_seed=1)
        self.assertTrue(info["loaded"])
        self.assertFalse(info["optimizer_resumable"])
        self.assertGreater(info["parameters"], 0)
        self.assertIn("unaffordable", info["reason"])
        payload = torch.load(LEGACY_CHECKPOINT, map_location="cpu", weights_only=False)
        problems = entry.resume_identity(payload)
        # The pre-fix run says nothing about this round's mask, so resuming it would carry the
        # defect forward under a label that claims otherwise.
        self.assertIn("trainer_tag", problems)
        self.assertIn("action_mask_schema", problems)
        self.assertIn("tape_schema", problems)
        def elements(state: Any) -> int:
            if torch.is_tensor(state):
                return int(state.numel())
            if isinstance(state, dict):
                return sum(elements(value) for value in state.values())
            return 0

        # The saved tensor count is the same shape the forward-only loader rebuilt, and the
        # 5 635-parameter network is unchanged by this round: no new input, no new head.
        self.assertEqual(elements(payload["controller"]), info["parameters"])
        self.assertEqual(info["parameters"], 5635)


class TrainerIdentityTests(unittest.TestCase):
    def test_resume_requires_the_same_mask_story(self) -> None:
        good = {
            "trainer_tag": "onpolicy-v3-budget-mask",
            "action_mask_schema": "budget-sequential-mask/1",
            "tape_schema": "onpolicy-tape/2",
            "timing_mode": "policy_owned_timing",
        }
        self.assertEqual(entry.resume_identity(dict(good)), {})
        bad = dict(good, action_mask_schema="none/1")
        problems = entry.resume_identity(bad)
        self.assertIn("action_mask_schema", problems)
        bad = dict(good, timing_mode="legacy_shared_bound")
        self.assertIn("timing_mode", entry.resume_identity(bad))
        bad = {key: value for key, value in good.items() if key != "trainer_tag"}
        self.assertIn("trainer_tag", entry.resume_identity(bad))

    def test_the_feature_layout_is_still_19(self) -> None:
        from integrated_sar.comm_learning import features

        item = candidate("target-1", charge=100, budget_bytes=1000)
        observation = observation_of([item], budget_bytes=1000)
        self.assertEqual(len(features(observation, item)), 19)
        self.assertEqual(
            len(features(observation, item, ack_features=(-1.0, -1.0, -1.0, -1.0))), 23
        )


if __name__ == "__main__":
    unittest.main()
