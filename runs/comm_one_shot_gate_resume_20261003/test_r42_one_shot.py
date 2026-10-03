"""Prelaunch pure-compute checks for ROUND42's actual collector/trainer seams.

The relay fixtures use the production CommRelay and TaskRegistry with synthetic events.  No
Scenario, JointRun, SAR simulator, training rollout, or optimizer step is constructed here.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parent
ROUND42_PROJECT_ROOT = RUN_DIR.parents[1]
WORKSPACE_ROOT = ROUND42_PROJECT_ROOT.parent
PROJECT_ROOT = WORKSPACE_ROOT / "zhenghe"
ROUND40_ROOT = WORKSPACE_ROOT / "zhenghe-round40"
ROUND40_OUTPUT = ROUND40_ROOT / "runs" / "comm_first_report_training_ablation_20261003"
ROUND40_RECONCILIATION = json.loads(
    (ROUND40_OUTPUT / "CHECKPOINT_RECONCILIATION.json").read_text(encoding="utf-8")
)
ROUND40_FREEZE = json.loads(
    (ROUND40_OUTPUT / "ROUND40_FROZEN_CONFIG.json").read_text(encoding="utf-8")
)["round40"]["freeze_identity"]

_MODULE_DIRS = [
    PROJECT_ROOT / "runs" / "comm_cost_v2_rerun_20260928",
    PROJECT_ROOT / "runs" / "comm_cost_stage_b_20260927",
    PROJECT_ROOT / "runs" / "comm_budget_action_mask_fix_20260922",
    PROJECT_ROOT / "runs" / "comm_budget_pilot_20260919",
    PROJECT_ROOT / "runs" / "comm_cost_controller_stage_a_20260927",
    PROJECT_ROOT / "runs" / "comm_tight_budget_20260919",
    PROJECT_ROOT / "runs" / "search_trace_and_policy_20260925",
    ROUND40_OUTPUT,
    PROJECT_ROOT,
]
for _directory in reversed(_MODULE_DIRS):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import v2_common as common  # noqa: E402
import v2_controller  # noqa: E402
import v2_trainer  # noqa: E402
import v2_run  # noqa: E402
from budget_masked import HOLD, SEND  # noqa: E402
from integrated_sar.comm_budget import CommBudget  # noqa: E402
from integrated_sar.comm_policy import SendItem, SendObservation  # noqa: E402
from onpolicy import action_entropy, joint_log_prob  # noqa: E402
from r42_one_shot import (  # noqa: E402
    COMM_ACTION_SEMANTICS,
    OneShotSendSkipCollector,
    UnexpectedPolicySendRefusal,
    stop_on_policy_refusal,
)
from r42_versions import TRAINING_VERSION, install as install_versions  # noqa: E402

common.DEVICE = torch.device("cpu")

R42_FREEZE = dict(ROUND40_FREEZE)
R42_FREEZE.update({
    "kind": "round42-one-shot-send-skip-freeze/1",
    "training_version": TRAINING_VERSION,
    "comm_action_semantics": COMM_ACTION_SEMANTICS,
})
install_versions(v2_trainer, v2_run, common=common, freeze=R42_FREEZE,
                 round40_freeze=ROUND40_FREEZE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _item(target: str, *, first: bool = True, revision: int = 1,
          charge: int = 4, executable: bool = True,
          reason: str = "allowed", step: int = 1) -> SendItem:
    return SendItem(
        target_id=target,
        ready_step=step,
        age_steps=max(0, step - 1),
        pending_event=first,
        changed_view=True,
        first_emission=first,
        revision=revision,
        revision_delta=revision,
        stage_index=0,
        stage_changed=False,
        progress=0.0,
        progress_delta=0.0,
        requirements_changed=False,
        members_changed=False,
        deadline=500.0,
        remaining_time=500.0 - step,
        sent_count=0 if first else 1,
        last_sent_step=None if first else step - 1,
        since_last_emission_time=None if first else 1.0,
        has_future_requirements=False,
        attempt_charge_bytes=charge,
        send_executable=executable,
        send_block_reason=reason,
    )


def _observation(items: list[SendItem], *, remaining: int = 100,
                 budget: int = 100, step: int = 1,
                 capacity: int | None = None) -> SendObservation:
    return SendObservation(
        step=step,
        current_time=float(step),
        items=tuple(items),
        remaining_budget_bytes=float(remaining),
        budget_bytes_per_episode=budget,
        capacity_remaining=capacity,
        backlog_targets=len(items),
        subscribers=1,
    )


class _FixedController:
    def __init__(self, *, hold_logit: float = 0.0, send_logit: float = 0.0) -> None:
        self.logits = torch.tensor([hold_logit, send_logit], dtype=torch.float32)

    def forward(self, batch: torch.Tensor):
        return self.logits.repeat((len(batch), 1)), torch.zeros(len(batch))

    def cost_values(self, batch: torch.Tensor):
        return torch.zeros(len(batch))


class _SequenceController(_FixedController):
    def __init__(self, rows: list[tuple[float, float]]) -> None:
        super().__init__()
        self.rows = torch.tensor(rows, dtype=torch.float32)

    def forward(self, batch: torch.Tensor):
        return self.rows[:len(batch)], torch.zeros(len(batch))


def _training_episode(collector: OneShotSendSkipCollector, *, steps: int,
                      task_stream: dict[int, float],
                      cost_stream: dict[int, int] | None = None,
                      episode_index: int = 0) -> dict:
    costs = cost_stream or {}
    return {
        "collector": collector,
        "steps": steps,
        "cost_stream": {step: {"airtime_bytes": value}
                        for step, value in costs.items()},
        "task_stream": task_stream,
        "lambda": 1.0,
        "episode_index": episode_index,
        "scenario_seed": 1_300_000 + int(episode_index),
        "on_time": int(sum(task_stream.values())),
    }


def _actor_critic_probe(controller, sample: dict) -> dict[str, float]:
    """Run the exact logged joint-likelihood, entropy, PPO ratio and twin-critic loss path."""
    matrix = torch.as_tensor(sample["features"], dtype=torch.float32)
    hidden = controller.inner.trunk(matrix)
    logits = controller.inner.action_head(hidden)
    values_task = controller.inner.value_head(hidden).squeeze(-1)
    values_cost = controller.cost_head(hidden).squeeze(-1)
    picked = torch.tensor([sample["actions"][target] for target in sample["targets"]],
                          dtype=torch.long)
    usable = torch.tensor([float(target in sample["free"]) for target in sample["targets"]],
                          dtype=torch.float32)
    new_log_prob = joint_log_prob(logits, picked, usable)
    entropy = action_entropy(logits, usable)
    old_log_prob = torch.tensor(sample["log_prob"], dtype=torch.float32)
    advantage = torch.tensor(sample["advantage"], dtype=torch.float32)
    ratio = torch.exp(new_log_prob - old_log_prob)
    surrogate = torch.minimum(
        ratio * advantage,
        torch.clamp(ratio, 1.0 - common.CLIP, 1.0 + common.CLIP) * advantage,
    )
    target_task = torch.tensor(sample["return_task"], dtype=torch.float32)
    target_cost = torch.tensor(sample["return_cost"], dtype=torch.float32)
    value_task_loss = ((values_task.mean() - target_task) ** 2)
    value_cost_loss = ((values_cost.mean() - target_cost) ** 2)
    loss = (-surrogate + common.VALUE_COEFFICIENT * (value_task_loss + value_cost_loss)
            - common.ENTROPY_BONUS * entropy)
    for module in (controller.inner.trunk, controller.inner.action_head,
                   controller.inner.value_head, controller.cost_head):
        module.zero_grad(set_to_none=True)
    loss.backward()

    def grad_l1(module) -> float:
        return sum(float(parameter.grad.detach().abs().sum())
                   for parameter in module.parameters() if parameter.grad is not None)

    return {
        "loss": float(loss.detach()),
        "entropy": float(entropy.detach()),
        "action_head_grad_l1": grad_l1(controller.inner.action_head),
        "task_critic_grad_l1": grad_l1(controller.inner.value_head),
        "cost_critic_grad_l1": grad_l1(controller.cost_head),
    }


class Round42OneShotTests(unittest.TestCase):
    def test_01_free_choice_latches_once_and_revision_cannot_reopen(self):
        collector = OneShotSendSkipCollector(_FixedController(hold_logit=1, send_logit=0),
                                             greedy=True)
        first = _observation([_item("target", charge=4)], step=1)
        collector.decide(first)
        first_tape = collector.records[-1]
        self.assertEqual(first_tape["free"], ["target"])
        self.assertEqual(first_tape["actions"]["target"], HOLD)
        self.assertEqual(collector.target_state["target"], "skip_episode_chosen")

        revision = _observation([_item("target", revision=2, charge=4, step=2)], step=2)
        collector.decide(revision)
        second_tape = collector.records[-1]
        self.assertEqual(second_tape["free"], [])
        self.assertEqual(second_tape["actions"]["target"], HOLD)
        self.assertEqual(second_tape["fixed_hold_one_shot_closed"], ["target"])
        self.assertEqual(len(collector.target_state), 1)

    def test_02_skip_is_a_real_likelihood_entropy_and_actor_action(self):
        controller = v2_controller.fresh_controller(991)
        with torch.no_grad():
            controller.inner.action_head.weight.zero_()
            controller.inner.action_head.bias.copy_(torch.tensor([0.05, 0.0]))
            controller.inner.value_head.weight.zero_()
            controller.inner.value_head.bias.zero_()
            controller.cost_head.weight.zero_()
            controller.cost_head.bias.zero_()
        collector = OneShotSendSkipCollector(controller, greedy=True)
        collector.decide(_observation([_item("skip-target")], step=1))
        tape = collector.records[-1]
        self.assertEqual(tape["actions"]["skip-target"], HOLD)
        self.assertIn("skip-target", tape["free"])
        self.assertLess(tape["log_prob"], 0.0)
        self.assertGreater(tape["send_probabilities"]["skip-target"], 0.0)
        self.assertNotEqual(tape["decision_class"]["skip-target"], "existing_budget_or_capacity_mask")
        self.assertGreater(float(action_entropy(
            controller.forward(torch.as_tensor(tape["features"], dtype=torch.float32))[0],
            torch.ones(1)).detach()), 0.0)

        samples = v2_trainer.batch_samples([
            _training_episode(collector, steps=1, task_stream={1: 1.0})
        ])
        self.assertEqual(samples[0]["free"], {"skip-target"})
        alignment = v2_trainer.likelihood_alignment(controller, samples)
        self.assertTrue(alignment["passed"], alignment)
        probe = _actor_critic_probe(controller, samples[0])
        self.assertGreater(probe["action_head_grad_l1"], 0.0)

    def test_03_closed_skip_stays_in_features_but_never_sends_or_charges(self):
        from integrated_sar.tests.test_comm_budget_action_mask import Harness, detection, progress

        collector = OneShotSendSkipCollector(_FixedController(hold_logit=1, send_logit=0),
                                             greedy=True)
        harness = Harness(policy=collector.policy(),
                          budget=CommBudget(bytes_per_episode=100_000))
        harness.publish(detection("skip-live"))
        harness.tick()
        self.assertEqual(harness.relay.counters["emissions"], 0)
        spent_before = harness.airtime
        harness.report(progress("skip-live", observed_at=2.0))
        harness.tick()
        self.assertEqual(collector.records[-1]["free"], [])
        self.assertEqual(collector.records[-1]["actions"]["skip-live"], HOLD)
        self.assertEqual(harness.relay.counters["emissions"], 0)
        self.assertEqual(harness.airtime, spent_before)
        self.assertEqual(collector.records[-1]["targets"], ["skip-live"])
        self.assertEqual(len(collector.records[-1]["features"][0]), 19)

    def test_04_real_send_costs_even_when_lost_and_closes_refreshes(self):
        from integrated_sar.tests.test_comm_budget_action_mask import Harness, detection, progress

        collector = OneShotSendSkipCollector(_FixedController(hold_logit=0, send_logit=1),
                                             greedy=True)
        harness = Harness(policy=collector.policy(),
                          budget=CommBudget(bytes_per_episode=100_000))
        harness.relay._delivery_probability = 0.0
        harness.publish(detection("send-live"))
        harness.tick()
        quote = int(collector.records[-1]["attempt_charge_bytes"]["send-live"])
        self.assertEqual(collector.records[-1]["actions"]["send-live"], SEND)
        self.assertEqual(harness.airtime, quote)
        self.assertGreater(harness.relay.counters["packets_lost"], 0)
        self.assertEqual(harness.relay.counters["emissions"], 1)

        harness.report(progress("send-live", observed_at=2.0))
        harness.tick()
        self.assertEqual(collector.records[-1]["free"], [])
        self.assertEqual(collector.records[-1]["actions"]["send-live"], HOLD)
        self.assertEqual(harness.relay.counters["emissions"], 1)
        self.assertEqual(harness.relay.counters["refresh_emissions"], 0)
        self.assertEqual(harness.airtime, quote)

    def test_05_masks_do_not_decide_and_new_episode_resets_latch(self):
        collector = OneShotSendSkipCollector(_FixedController(hold_logit=0, send_logit=1),
                                             greedy=True)
        torch.manual_seed(453)
        rng_before = torch.get_rng_state().clone()
        collector.decide(_observation([
            _item("masked", charge=11, executable=False, reason="budget_exhausted")
        ], remaining=10, budget=10, step=1))
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertEqual(collector.target_state, {})
        self.assertEqual(collector.records[-1]["free"], [])

        collector.decide(_observation([_item("masked", revision=2, charge=4, step=2)],
                                      remaining=10, budget=10, step=2))
        self.assertEqual(collector.records[-1]["free"], ["masked"])
        self.assertEqual(collector.target_state["masked"], "send_now_chosen")

        new_episode = OneShotSendSkipCollector(
            _FixedController(hold_logit=0, send_logit=1), greedy=True)
        new_episode.decide(_observation([_item("masked", step=1)], step=1))
        self.assertEqual(new_episode.records[-1]["free"], ["masked"])

    def test_06_multi_target_latches_are_independent_and_budget_is_sequential(self):
        multi = OneShotSendSkipCollector(
            _SequenceController([(0.1, 0.0), (0.0, 0.1)]), greedy=True)
        multi.decide(_observation([
            _item("a-skip", charge=4), _item("b-send", charge=4)
        ], remaining=20, budget=20))
        tape = multi.records[-1]
        self.assertEqual(tape["actions"], {"a-skip": HOLD, "b-send": SEND})
        self.assertEqual(multi.target_state["a-skip"], "skip_episode_chosen")
        self.assertEqual(multi.target_state["b-send"], "send_now_chosen")

        sequential = OneShotSendSkipCollector(
            _FixedController(hold_logit=0, send_logit=1), greedy=True)
        sequential.decide(_observation([
            _item("a", charge=6), _item("b", charge=6)
        ], remaining=10, budget=10, step=1))
        first_tape = sequential.records[-1]
        self.assertEqual(first_tape["free"], ["a"])
        self.assertEqual(first_tape["masked"], ["b"])
        self.assertEqual(sequential.target_state, {"a": "send_now_chosen"})
        sequential.decide(_observation([
            _item("a", first=False, revision=2, charge=6, step=2),
            _item("b", revision=2, charge=4, step=2),
        ], remaining=4, budget=10, step=2))
        self.assertEqual(sequential.records[-1]["free"], ["b"])
        self.assertEqual(sequential.target_state["b"], "send_now_chosen")

    def test_07_batch_likelihood_matches_independent_action_prefix_product(self):
        controller = v2_controller.fresh_controller(992)
        with torch.no_grad():
            controller.inner.action_head.weight.zero_()
            controller.inner.action_head.bias.copy_(torch.tensor([0.1, 0.0]))
        collector = OneShotSendSkipCollector(controller, greedy=True)
        collector.decide(_observation([_item("first")], step=1))
        collector.decide(_observation([_item("first", revision=2, step=2)], step=2))
        samples = v2_trainer.batch_samples([
            _training_episode(collector, steps=2, task_stream={1: 1.0})
        ])
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["free"], {"first"})
        self.assertEqual(samples[1]["free"], set())
        self.assertTrue(v2_trainer.likelihood_alignment(controller, samples)["passed"])

        sample = samples[0]
        logits, _ = controller.forward(torch.as_tensor(sample["features"], dtype=torch.float32))
        picked = torch.tensor([sample["actions"][target] for target in sample["targets"]])
        probability = torch.softmax(logits, dim=-1)[0, int(picked[0])]
        independently_computed = float(torch.log(probability).detach())
        self.assertAlmostEqual(independently_computed, sample["log_prob"], places=7)
        self.assertEqual(samples[1]["log_prob"], 0.0)
        no_free = torch.zeros(len(samples[1]["targets"]))
        closed_logits, _ = controller.forward(
            torch.as_tensor(samples[1]["features"], dtype=torch.float32))
        self.assertEqual(float(joint_log_prob(
            closed_logits,
            torch.tensor([samples[1]["actions"][target]
                          for target in samples[1]["targets"]]),
            no_free,
        ).detach()), 0.0)
        self.assertEqual(float(action_entropy(closed_logits, no_free).detach()), 0.0)

    def test_08_all_closed_rows_keep_both_critics_and_zero_actor_gradient(self):
        controller = v2_controller.fresh_controller(993)
        collector = OneShotSendSkipCollector(controller, greedy=True)
        collector.target_state["closed"] = "skip_episode_chosen"
        collector.decide(_observation([_item("closed", step=2)], step=2))
        samples = v2_trainer.batch_samples([
            _training_episode(collector, steps=2, task_stream={2: 1.0},
                              cost_stream={2: 1000})
        ])
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["free"], set())
        self.assertTrue(v2_trainer.likelihood_alignment(controller, samples)["passed"])
        probe = _actor_critic_probe(controller, samples[0])
        self.assertEqual(probe["action_head_grad_l1"], 0.0)
        self.assertGreater(probe["task_critic_grad_l1"], 0.0)
        self.assertGreater(probe["cost_critic_grad_l1"], 0.0)

    def test_09_old_training_restore_refuses_before_state_restore_old_readonly_succeeds(self):
        old = ROUND40_RECONCILIATION["checkpoints"]["NEW_training_outputs"]
        checkpoint = Path(old["11"]["ep128"]["path"])
        before = _sha256(checkpoint)
        with patch.object(v2_trainer, "_restore_controller",
                          side_effect=AssertionError("controller restore must not run")), \
             patch.object(torch.optim, "Adam",
                          side_effect=AssertionError("optimizer construction must not run")), \
             patch.object(v2_trainer, "restore_rng",
                          side_effect=AssertionError("RNG restore must not run")):
            with self.assertRaisesRegex(RuntimeError, "training_version mismatch"):
                v2_trainer.load_checkpoint(checkpoint, freeze=R42_FREEZE)
        self.assertEqual(_sha256(checkpoint), before)

        torch.manual_seed(238)
        ambient_rng_before = torch.get_rng_state().clone()
        controller, state, identity = v2_trainer.load_checkpoint_read_only(checkpoint)
        self.assertEqual(state["episodes_done"], 128)
        self.assertEqual(identity["load_mode"], "read_only")
        self.assertFalse(identity["optimizer_restored"])
        self.assertFalse(identity["rng_restored"])
        self.assertTrue(identity["ambient_rng_preserved"])
        self.assertTrue(torch.equal(ambient_rng_before, torch.get_rng_state()))
        self.assertEqual(identity["action_semantics_interpretation"],
                         "round40-first-report-only/1_adapted_to_one-shot-send-skip/1")
        self.assertEqual(_sha256(checkpoint), before)
        self.assertEqual(controller.parameter_summary()["parameter_sha256"],
                         old["11"]["ep128"]["parameter_sha256"])

    def test_10_round40_ep0_matches_fresh_same_seed_initialization(self):
        checkpoints = ROUND40_RECONCILIATION["checkpoints"]["NEW_training_outputs"]
        comparisons = {}
        for seed in (11, 22, 33):
            record = checkpoints[str(seed)]["ep0"]
            path = Path(record["path"])
            self.assertEqual(_sha256(path), record["file_sha256"])
            restored, state, identity = v2_trainer.load_checkpoint_read_only(
                path, freeze=ROUND40_FREEZE)
            fresh = v2_controller.fresh_controller(seed)
            self.assertEqual(state["episodes_done"], 0)
            self.assertFalse(identity["optimizer_restored"])
            self.assertFalse(identity["rng_restored"])
            restored_id = restored.parameter_summary()["parameter_sha256"]
            fresh_id = fresh.parameter_summary()["parameter_sha256"]
            self.assertEqual(restored_id, record["parameter_sha256"])
            self.assertEqual(fresh_id, record["parameter_sha256"])
            comparisons[str(seed)] = restored_id
        self.assertEqual(comparisons, ROUND40_FREEZE["initial_parameter_identities"])

    def test_11_unexpected_real_emit_refusal_stops_before_next_tick(self):
        from integrated_sar.comm_policy import CommSendPolicy
        from integrated_sar.tests.test_comm_budget_action_mask import Harness, detection

        collector = OneShotSendSkipCollector(_FixedController(hold_logit=0, send_logit=1),
                                             greedy=True)
        delegate = collector.policy()

        class SpendAfterQuote(CommSendPolicy):
            name = "r42-spend-after-quote-fixture"
            orders_selection = True

            def __init__(self):
                self.harness = None

            def decide(self, observation):
                selected = delegate.decide(observation)
                if selected:
                    # Drain the real remaining credit after collection and before _emit.
                    self.harness.relay.budget.charge(
                        int(self.harness.relay.budget.remaining_bytes))
                return selected

        policy = SpendAfterQuote()
        harness = Harness(policy=policy, budget=CommBudget(bytes_per_episode=100_000))
        policy.harness = harness
        harness.publish(detection("unexpected-refusal"))
        with stop_on_policy_refusal(), self.assertRaises(UnexpectedPolicySendRefusal):
            harness.tick()
        self.assertEqual(collector.records[-1]["actions"]["unexpected-refusal"], SEND)
        self.assertEqual(collector.target_state["unexpected-refusal"], "send_now_chosen")
        self.assertEqual(harness.relay.counters["budget_refused_sends"], 1)
        self.assertEqual(harness.relay._tick, 1)
        self.assertEqual(harness.relay.counters["emissions"], 0)

    def test_12_hand_derived_expected_objective_directions(self):
        # Independent hand derivation at p=1/2: dJ/d(send-logit) = p(1-p) * ΔU.
        p = 0.5
        # A: equal task result, SEND costs .2 more than SKIP.
        expected_a = p * (1.0 - p) * (0.0 - 0.2)
        x_a = torch.tensor(0.0, requires_grad=True)
        logits_a = torch.stack([torch.zeros_like(x_a), x_a]).reshape(1, 2)
        p_send_a = torch.exp(joint_log_prob(logits_a, torch.tensor([SEND]), torch.ones(1)))
        p_skip_a = torch.exp(joint_log_prob(logits_a, torch.tensor([HOLD]), torch.ones(1)))
        objective_a = p_send_a * (1.0 - 0.2) + p_skip_a * 1.0
        objective_a.backward()
        self.assertAlmostEqual(float(x_a.grad), expected_a, places=7)
        self.assertLess(float(x_a.grad), 0.0)

        # B: SEND gets task=1/cost=.2, SKIP gets task=0/cost=0.
        expected_b = {lam: p * (1.0 - p) * (lam - 0.2) for lam in (0.0, 1.0)}
        observed_b = {}
        for lam in (0.0, 1.0):
            x = torch.tensor(0.0, requires_grad=True)
            logits = torch.stack([torch.zeros_like(x), x]).reshape(1, 2)
            p_send = torch.exp(joint_log_prob(logits, torch.tensor([SEND]), torch.ones(1)))
            p_skip = torch.exp(joint_log_prob(logits, torch.tensor([HOLD]), torch.ones(1)))
            objective = p_send * (lam * 1.0 - 0.2) + p_skip * 0.0
            objective.backward()
            observed_b[lam] = float(x.grad)
            self.assertAlmostEqual(observed_b[lam], expected_b[lam], places=7)
        self.assertAlmostEqual(expected_b[0.0], -0.05, places=12)
        self.assertAlmostEqual(expected_b[1.0], 0.20, places=12)


if __name__ == "__main__":
    import integrated_sar.joint_driver as joint_driver

    counters = {"joint_run_constructors": 0, "optimizer_step_calls": 0,
                "checkpoint_write_calls": 0}
    restorers = []
    seen: set[type] = set()

    def descendants(base: type) -> list[type]:
        result = [base]
        for child in base.__subclasses__():
            if child not in seen:
                seen.add(child)
                result.extend(descendants(child))
        return result

    for cls in descendants(torch.optim.Optimizer):
        original = cls.__dict__.get("step")
        if original is None:
            continue

        def blocked_step(*_args, _cls=cls, **_kwargs):
            counters["optimizer_step_calls"] += 1
            raise AssertionError(f"pure-compute tests forbid {_cls.__name__}.step")

        restorers.append((cls, "step", original))
        setattr(cls, "step", blocked_step)

    original_save = torch.save

    def blocked_save(*_args, **_kwargs):
        counters["checkpoint_write_calls"] += 1
        raise AssertionError("pure-compute tests forbid checkpoint writes")

    torch.save = blocked_save
    original_build = joint_driver.build_joint_run

    def blocked_build(*_args, **_kwargs):
        counters["joint_run_constructors"] += 1
        raise AssertionError("pure-compute tests forbid JointRun construction")

    joint_driver.build_joint_run = blocked_build
    v2_run.build_joint_run = blocked_build
    try:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(Round42OneShotTests))
    finally:
        v2_run.build_joint_run = original_build
        joint_driver.build_joint_run = original_build
        torch.save = original_save
        for cls, name, original in reversed(restorers):
            setattr(cls, name, original)
    print("ROUND42_TEST_GUARD " + json.dumps({
        "test_count": result.testsRun,
        "passed": result.wasSuccessful(),
        **counters,
    }, sort_keys=True))
    if not result.wasSuccessful() or any(counters.values()):
        raise SystemExit(1)
