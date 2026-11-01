"""Regression tests for the collector's probability bookkeeping.

Round 2 sampled from ``mu(a|s) = 0.9 pi(a|s) + 0.1 Uniform(a)`` and logged ``log pi(a|s)``, and
round 3 recorded that in its own protocol as a live defect it chose not to fix. These tests are
the reason it is a defect rather than a style choice: each one compares a drawn action with the
number stored beside it, which is the check that was missing.

The module under test lives with the experiments, not in the shipped system, but it is imported
here because a correctness property belongs in the suite that runs on every change -- and it
imports nothing but ``integrated_sar`` and torch, so it does not need a checkpoint or a stream.
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ROUND_DIR = ROOT / "runs" / "comm_tight_budget_20260919"
if str(ROUND_DIR) not in sys.path:
    sys.path.insert(0, str(ROUND_DIR))

import onpolicy  # noqa: E402
from integrated_sar.comm_policy import SendItem, SendObservation  # noqa: E402

FEATURE_DIM = 19


def item(target_id: str, *, age: int = 0, remaining: float = 400.0) -> SendItem:
    return SendItem(
        target_id=target_id,
        ready_step=0,
        age_steps=age,
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
    )


def observation(items: tuple[SendItem, ...], *, step: int = 1) -> SendObservation:
    return SendObservation(
        step=step,
        current_time=float(step),
        items=items,
        remaining_budget_bytes=1000,
        budget_bytes_per_episode=1000,
        capacity_remaining=None,
        backlog_targets=len(items),
        subscribers=2,
    )


class FixedLogitsController(torch.nn.Module):
    """A decision function whose probabilities are known exactly, so sampling is checkable.

    The real network cannot be used for this: an arbitrary initialisation gives probabilities
    nobody can write down, and a test that compares two opaque numbers proves nothing. The logits
    are a trainable parameter, so the gradient tests flow through the same tensor the sampler
    read, which is the only honest place to ask whether a forced row contributes to the actor.
    """

    def __init__(self, send_probability: float | list[float]) -> None:
        super().__init__()
        probabilities = (
            [float(send_probability)] if isinstance(send_probability, float) else list(send_probability)
        )
        self.action_logits = torch.nn.Parameter(
            torch.tensor(
                [[float(np.log((1.0 - p) / p)), 0.0] for p in probabilities],
                dtype=torch.float32,
            ),
            requires_grad=True,
        )
        self.value_weights = torch.nn.Parameter(torch.zeros(len(probabilities), 1))

    def forward(self, batch: torch.Tensor):
        count = int(batch.shape[0])
        return self.action_logits[:count], self.value_weights[:count].squeeze(-1)


class NeverForced:
    def __call__(self, _item: SendItem) -> bool:
        return False


class AlwaysForced:
    def __call__(self, _item: SendItem) -> bool:
        return True


class SamplerMatchesItsOwnLogProbTests(unittest.TestCase):
    def test_drawn_actions_match_the_distribution_that_is_logged(self) -> None:
        torch.manual_seed(20260919)
        random.seed(20260919)
        probability = 0.75
        controller = FixedLogitsController(probability)
        collector = onpolicy.OnPolicyCollector(controller, forced=NeverForced())
        observation_ = observation((item("a"),))
        draws = 4000
        for _ in range(draws):
            collector.decide(observation_)
        wanted = collector.distribution_reads["wanted_send"]
        rate = wanted / draws
        standard_error = float(np.sqrt(probability * (1 - probability) / draws))
        self.assertLess(
            abs(rate - probability), 4 * standard_error,
            f"the sampler drew at {rate}, not at the policy's {probability}",
        )
        logged = [record["log_prob"] for record in collector.records]
        expected = probability * np.log(probability) + (1 - probability) * np.log(1 - probability)
        self.assertAlmostEqual(
            float(np.mean(logged)), float(expected), delta=4 * standard_error
        )

    def test_a_mixture_sampler_would_have_failed_this_same_check(self) -> None:
        """The check has to be able to fail: epsilon-greedy, measured with the identical test.

        Without this, "our sampler matches our log-probabilities" is only evidence that nobody
        ever tried the mixture. Round 2's collector is reproduced here by overlaying a uniform
        draw on the same fixed-probability controller, and the empirical rate moves away from the
        logged distribution by far more than sampling error allows.
        """
        torch.manual_seed(20260919)
        random.seed(20260919)
        probability, epsilon, draws = 0.75, 0.10, 20_000
        controller = FixedLogitsController(probability)
        sent = 0
        for _ in range(draws):
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                action = int(
                    torch.multinomial(
                        torch.softmax(controller.action_logits, dim=-1)[0], num_samples=1
                    ).item()
                )
            sent += int(action == onpolicy.SEND)
        rate = sent / draws
        standard_error = float(np.sqrt(probability * (1 - probability) / draws))
        self.assertGreater(
            abs(rate - probability), 4 * standard_error,
            "the reproduction did not separate the mixture from the policy, so it tests nothing",
        )
        self.assertAlmostEqual(rate, (1 - epsilon) * probability + epsilon * 0.5, delta=4 * standard_error)

    def test_the_joint_log_prob_is_the_sum_of_the_free_terms(self) -> None:
        torch.manual_seed(7)
        probabilities = [0.7, 0.25]
        controller = FixedLogitsController(probabilities)
        collector = onpolicy.OnPolicyCollector(controller, forced=NeverForced())
        asked, actions, free, log_prob, _value, tape = collector.choose(
            observation((item("a"), item("b")))
        )
        self.assertEqual(sorted(free), ["a", "b"])
        self.assertEqual(
            sorted(asked), sorted(target for target, action in actions.items() if action == onpolicy.SEND)
        )
        expected = sum(
            float(np.log(probabilities[index] if actions[target] == onpolicy.SEND else 1 - probabilities[index]))
            for index, target in enumerate(["a", "b"])
        )
        self.assertAlmostEqual(log_prob, expected, places=5)
        self.assertAlmostEqual(tape["log_prob"], expected, places=5)

    def test_ratio_is_exactly_one_when_nothing_has_moved(self) -> None:
        torch.manual_seed(11)
        controller = FixedLogitsController([0.6, 0.4])
        collector = onpolicy.OnPolicyCollector(controller, forced=NeverForced())
        _asked, actions, free, logged, _value, _tape = collector.choose(
            observation((item("a"), item("b")))
        )
        targets = ["a", "b"]
        logits, _values = controller.forward(torch.zeros(len(targets), FEATURE_DIM))
        new_log_prob = onpolicy.joint_log_prob(
            logits,
            torch.tensor([actions[target] for target in targets], dtype=torch.long),
            torch.tensor([target in free for target in targets], dtype=torch.float32),
        )
        ratio = float(torch.exp(new_log_prob - logged).item())
        self.assertAlmostEqual(ratio, 1.0, places=5)
        self.assertAlmostEqual(float(new_log_prob.item()), logged, places=5)

    def test_the_ratio_follows_the_analytic_probability_after_a_change(self) -> None:
        """Send probability 0.5 -> 0.25: the ratio is new/old, i.e. 0.5 for a send, 1.5 for a hold."""
        torch.manual_seed(3)
        before = FixedLogitsController([0.5])
        collector = onpolicy.OnPolicyCollector(before, forced=NeverForced())
        _asked, actions, free, logged, _value, tape = collector.choose(observation((item("a"),)))
        picked = torch.tensor([actions[target] for target in tape["targets"]], dtype=torch.long)
        usable = torch.tensor([target in free for target in tape["targets"]], dtype=torch.float32)
        after = FixedLogitsController([0.25])
        logits_after, _ = after.forward(torch.zeros(1, FEATURE_DIM))
        new_log_prob = float(onpolicy.joint_log_prob(logits_after, picked, usable).item())
        expected = 0.5 if actions["a"] == onpolicy.SEND else 1.5
        self.assertAlmostEqual(float(np.exp(new_log_prob - logged)), expected, places=5)


class ForcedActionsLeaveNoTraceTests(unittest.TestCase):
    def test_a_bound_taken_over_candidate_carries_no_likelihood_and_no_entropy(self) -> None:
        forced = {"b"}

        def predicate(candidate: SendItem) -> bool:
            return candidate.target_id in forced

        torch.manual_seed(5)
        controller = FixedLogitsController([0.3, 0.8])
        collector = onpolicy.OnPolicyCollector(controller, forced=predicate)
        asked, actions, free, logged, _value, tape = collector.choose(
            observation((item("a"), item("b", age=99, remaining=5.0)))
        )
        self.assertEqual(actions["b"], onpolicy.SEND, "the bound still puts it on the air")
        self.assertNotIn("b", free)
        self.assertNotIn("b", asked, "but the collector does not claim the decision")
        self.assertEqual(tape["forced"], ["b"])
        only_a = torch.tensor([actions["a"]], dtype=torch.long)
        logits, _ = controller.forward(torch.zeros(2, FEATURE_DIM))
        self.assertAlmostEqual(
            logged,
            float(onpolicy.joint_log_prob(logits[:1], only_a, torch.tensor([1.0])).item()),
            places=5,
        )
        self.assertAlmostEqual(
            float(onpolicy.action_entropy(logits[1:], torch.tensor([0.0])).item()), 0.0, places=9
        )

    def test_a_forced_row_contributes_nothing_to_the_actor_but_the_value_head_still_learns(self) -> None:
        forced = {"b"}

        def predicate(candidate: SendItem) -> bool:
            return candidate.target_id in forced

        torch.manual_seed(13)
        controller = FixedLogitsController([0.3, 0.9])
        collector = onpolicy.OnPolicyCollector(controller, forced=predicate)
        _asked, actions, free, _logged, _value, tape = collector.choose(
            observation((item("a"), item("b", age=99, remaining=5.0)))
        )
        logits, values = controller.forward(torch.zeros(2, FEATURE_DIM))
        picked = torch.tensor([actions[target] for target in tape["targets"]], dtype=torch.long)
        usable = torch.tensor([target in free for target in tape["targets"]], dtype=torch.float32)
        likelihood = onpolicy.joint_log_prob(logits, picked, usable)
        surrogate_loss = -torch.exp(likelihood - likelihood.detach()) * 1.0
        actor_gradient = torch.autograd.grad(surrogate_loss, logits, retain_graph=True)[0]
        self.assertEqual(float(actor_gradient[1].abs().sum().item()), 0.0)
        self.assertGreater(float(actor_gradient[0].abs().sum().item()), 0.0)
        critic_loss = ((values.sum() - 1.0) ** 2)
        critic_gradient = torch.autograd.grad(critic_loss, values)[0]
        self.assertGreater(float(critic_gradient.abs().sum().item()), 0.0, "the same tick still trains the critic")

    def test_an_all_forced_step_has_zero_log_prob_and_zero_entropy(self) -> None:
        torch.manual_seed(17)
        controller = FixedLogitsController([0.4, 0.9])
        collector = onpolicy.OnPolicyCollector(controller, forced=AlwaysForced())
        asked, actions, free, logged, _value, tape = collector.choose(
            observation((item("a", age=99), item("b", age=99)))
        )
        self.assertEqual(free, [])
        self.assertEqual(asked, [], "nothing here was the policy's doing")
        self.assertEqual(logged, 0.0)
        self.assertEqual(set(actions.values()), {onpolicy.SEND})
        logits, _ = controller.forward(torch.zeros(2, FEATURE_DIM))
        usable = torch.tensor([0.0, 0.0])
        picked = torch.tensor([actions["a"], actions["b"]], dtype=torch.long)
        self.assertAlmostEqual(float(onpolicy.joint_log_prob(logits, picked, usable).item()), 0.0, places=9)
        self.assertAlmostEqual(float(onpolicy.action_entropy(logits, usable).item()), 0.0, places=9)
        self.assertEqual(tape["send_probabilities"], {}, "no free candidate, so no logged probability")


class RestoreEverythingTests(unittest.TestCase):
    def test_the_three_random_streams_survive_a_round_trip(self) -> None:
        random.seed(101)
        np.random.seed(102)
        torch.manual_seed(103)
        state = onpolicy.rng_state()
        first = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
        restored = onpolicy.restore_rng(state)
        self.assertTrue(restored)
        self.assertEqual(
            (random.random(), float(np.random.random()), float(torch.rand(1).item())), first
        )

    def test_a_missing_state_is_reported_as_not_restored(self) -> None:
        self.assertFalse(onpolicy.restore_rng(None))

    def test_a_saved_run_carries_the_optimizer_moments_and_the_scenario_cursor(self) -> None:
        import tempfile

        torch.manual_seed(29)
        module = FixedLogitsController([0.5])
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
        (module.action_logits.sum() * 2).backward()
        optimizer.step()
        random.seed(4)
        np.random.seed(5)
        torch.manual_seed(6)
        state = {"cursor": 17, "environment_steps": 100_000, "rng": onpolicy.rng_state()}
        draw_before = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "checkpoint.pt"
            torch.save({"controller": module.state_dict(), "optimizer": optimizer.state_dict(), "state": state}, path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        restored_optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
        restored_optimizer.load_state_dict(payload["optimizer"])
        self.assertEqual(
            {key: int(value["step"]) for key, value in restored_optimizer.state.items()},
            {key: int(value["step"]) for key, value in optimizer.state.items()},
        )
        self.assertEqual(payload["state"]["cursor"], 17, "the scenario cursor has to survive")
        self.assertTrue(onpolicy.restore_rng(payload["state"]["rng"]))
        self.assertEqual(
            (random.random(), float(np.random.random()), float(torch.rand(1).item())), draw_before
        )

    def test_the_actionable_set_is_recorded_with_the_decision(self) -> None:
        torch.manual_seed(23)
        controller = FixedLogitsController([0.5, 0.5, 0.5])
        collector = onpolicy.OnPolicyCollector(
            controller, forced=lambda candidate: candidate.target_id == "b"
        )
        collector.decide(observation((item("a"), item("b", age=99), item("c"))))
        record = collector.records[0]
        self.assertEqual(sorted(record["free"]), ["a", "c"])
        self.assertEqual(sorted(record["targets"]), ["a", "b", "c"])
        self.assertEqual(sorted(record["send_probabilities"]), ["a", "c"])


if __name__ == "__main__":
    unittest.main()
