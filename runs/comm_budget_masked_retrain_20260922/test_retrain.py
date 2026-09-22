"""Tests for the from-scratch retraining run: identity, the real seam, and refusal to inherit.

Three things are checked here that a protocol document cannot check by itself.

*The starting point.* Every arm must be ``Controller(seed)``. That is proved by making `torch.load`
raise while building the arms, so a silently-imported checkpoint cannot pass, and by requiring the
three fingerprints to differ from each other and to match a fresh construction.

*The seam, on a real episode.* One rollout through the relay, the masked collector and the
transport's ledger: requests equal executions, no policy send is refused, no emission is forced,
the emission classes close, the logged likelihood recomputes to 1e-6, and the greedy replay's
candidate classes add up.

*The refusal.* A resume must reject a different freeze, a different configuration hash, a missing
RNG block, and the archived 400k checkpoint from the unfixed action space -- while the same
weights remain available for a forward pass, which is what the previous round's `load_forward_only`
is for.

    python -m unittest discover -s . -p 'test_*.py'
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

import retrain_common as common
import train_retrain as entry

import train_pilot  # noqa: E402

HERE = Path(__file__).resolve().parent
SCRATCH = common.EPISODE_OUT / "unittest"
#: The archived run whose action space still contained unaffordable sends. Forward-only evidence.
LEGACY_CHECKPOINT = (
    common.ROOT
    / "runs"
    / "comm_protocol_v2_20260920"
    / "train"
    / "25pct_seed1"
    / "v2_25pct_seed1_at400000.pt"
)


def _rollout(controller, stream_seed, *, training_seed=1, index=0):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return entry.rollout(
        controller,
        stream_seed,
        training_seed=training_seed,
        episode_index=index,
        directory=SCRATCH,
    )


class InitialWeightTests(unittest.TestCase):
    def test_no_checkpoint_can_be_read_while_building_an_arm(self) -> None:
        """``Controller(seed)`` is the only source of weights, and torch.load is not on the path."""
        original = torch.load
        calls: list[str] = []

        def spy(*arguments, **keywords):
            calls.append(str(arguments[0]))
            return original(*arguments, **keywords)

        torch.load = spy
        try:
            fingerprints = [
                common.weight_fingerprint(entry.initial_controller(seed))
                for seed in common.TRAINING_SEEDS
            ]
        finally:
            torch.load = original
        self.assertEqual([], calls, "building an arm read a checkpoint")
        self.assertEqual(3, len(set(fingerprints)), "two arms started from the same weights")

    def test_identity_report_says_all_three_are_distinct(self) -> None:
        report = entry.identity_report()
        self.assertTrue(report["all_distinct"])
        self.assertEqual(3, report["distinct"])
        for seed, item in report["initial_weights"].items():
            self.assertIsNone(item["loaded_from"])
            self.assertEqual(5635, item["parameters"])

    def test_configuration_hash_is_stable_and_setting_sensitive(self) -> None:
        first = entry.configuration_hash()
        self.assertEqual(first, entry.configuration_hash())
        original = entry.BUDGET_BYTES
        try:
            entry.BUDGET_BYTES = original + 1
            self.assertNotEqual(first, entry.configuration_hash())
        finally:
            entry.BUDGET_BYTES = original


class RealSeamTests(unittest.TestCase):
    """One episode through the transport, because every property here is a joint one."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        controller = entry.initial_controller(1)
        cls.episode = _rollout(controller, common.TRAIN_SEEDS[0])
        cls.detail = entry.per_episode_row(cls.episode)
        cls.tapes = cls.episode["collector"].records

    def test_episode_ran_the_configured_length(self) -> None:
        self.assertEqual(common.STEPS_PER_EPISODE, self.detail["steps_run"])

    def test_requested_sends_are_the_executed_ones(self) -> None:
        self.assertEqual(
            self.detail["requested"], self.detail["executed_policy_emissions"],
            "the relay did not honour exactly what the arm asked for",
        )

    def test_no_policy_send_was_refused_and_nothing_was_forced(self) -> None:
        self.assertEqual(0, self.detail["policy_budget_refusals"])
        self.assertEqual(0, self.detail["policy_capacity_refusals"])
        self.assertEqual(0, self.detail["forced_emissions"])

    def test_emission_classes_close(self) -> None:
        self.assertEqual(
            self.detail["message_emissions"],
            self.detail["message_first"] + self.detail["message_refresh"],
        )
        self.assertLessEqual(
            self.detail["message_retry_same_view"], self.detail["message_refresh"]
        )
        self.assertEqual(
            self.detail["message_emissions"],
            self.detail["message_first"]
            + self.detail["message_update"]
            + self.detail["message_retry_same_view"],
        )

    def test_logged_likelihood_recomputes(self) -> None:
        alignment = entry.masked.drawn_versus_logged(
            self.episode["controller"], self.tapes
        )
        self.assertLessEqual(
            float(alignment["max_abs_gap"]), entry.LOG_PROB_TOLERANCE, str(alignment)
        )
        self.assertEqual(len(self.tapes), int(alignment["records"]))

    def test_tape_invariants_hold_on_a_real_episode(self) -> None:
        report = entry.masked.tape_invariants(self.tapes)
        self.assertTrue(report["passed"], str(report["violations"]))

    def test_candidate_classes_close_in_the_greedy_replay(self) -> None:
        greedy = entry.greedy_replay(self.episode["controller"], self.tapes)
        for record in self.tapes:
            total = (
                len(record["actionable"])
                + len(record["masked"])
                + len(record["forced"])
            )
            self.assertEqual(len(record["targets"]), total)
        self.assertLessEqual(greedy["greedy_actionable_candidates"], self.detail["candidates"])
        self.assertGreater(greedy["tapes_replayed"], 0)

    def test_every_masked_candidate_is_held_and_explained(self) -> None:
        """A masked candidate must be held, must not be requested, and must name its blocker.

        These three together are what "the constraint was applied before the draw" means in
        observable terms. If a masked candidate were ever requested for emission, the tape would be
        describing a decision the transport then overruled -- the exact shape of the old defect.
        """
        for record in self.tapes:
            for target in record["masked"]:
                self.assertEqual(0, record["actions"][target], "a masked candidate was not held")
                self.assertFalse(record["requested"][target], "a masked candidate was requested")
                self.assertIn(
                    record["masked_reasons"][target], {"budget_exhausted", "capacity_step"}
                )
            for target in record["actionable"]:
                self.assertIn(target, record["free"])

    def test_actionable_count_matches_the_actor_terms(self) -> None:
        self.assertEqual(
            self.detail["actionable_candidates"],
            sum(len(record["actionable"]) for record in self.tapes),
        )
        self.assertEqual(self.detail["actor_terms"], self.detail["actionable_candidates"])


class GreedyReplayArithmeticTests(unittest.TestCase):
    """The replay is tested on a synthetic tape too: the credit boundary is order-dependent."""

    def _tape(self, charges, remaining):
        targets = [f"t{index}" for index in range(len(charges))]
        return {
            "step": 5,
            "targets": targets,
            "features": [[0.0] * 19 for _ in targets],
            "actions": {target: 0 for target in targets},
            "actionable": [],
            "masked": [],
            "forced": [],
            "attempt_charge_bytes": dict(zip(targets, charges)),
            "send_executable_initial": {target: True for target in targets},
            "shadow_budget_before": remaining,
            "shadow_capacity_before": None,
            "shadow_budget_after": remaining,
            "shadow_capacity_after": None,
            "requested": {target: False for target in targets},
            "log_prob": 0.0,
            "value": 0.0,
            "send_probabilities": {},
        }

    def test_gate_starts_from_the_remaining_credit(self) -> None:
        gate = entry._gate_from_tape(self._tape([100, 100], 500.0))
        self.assertEqual(entry.BUDGET_BYTES - 500, gate.spent_bytes)
        self.assertTrue(gate.can_send(500))
        self.assertFalse(gate.can_send(501))

    def test_an_impossible_remaining_credit_refuses(self) -> None:
        with self.assertRaises(RuntimeError):
            entry._gate_from_tape(self._tape([100], entry.BUDGET_BYTES + 10.0))

    def test_replay_masks_only_what_the_credit_cannot_pay(self) -> None:
        """One byte of headroom: the cheap candidate is a choice, the expensive one is not.

        The assertion holds whatever the argmax says, because a 29,300-byte packet cannot fit in
        one remaining byte. That is the point -- the mask is about arithmetic, not about the
        policy's preference, and a greedy walk must reach the same conclusion the sampler did.
        """
        controller = entry.initial_controller(7)
        tape = self._tape([1, entry.BUDGET_BYTES], 1.0)
        greedy = entry.greedy_replay(controller, [tape])
        self.assertEqual(1, greedy["greedy_actionable_candidates"])
        self.assertEqual(1, greedy["greedy_masked_by_credit"])
        self.assertLessEqual(greedy["greedy_send"], 1)
        self.assertEqual(5, greedy["greedy_first_credit_masked_step"])


class ResumeGuardTests(unittest.TestCase):
    """A checkpoint must state this run's identity in every field that changes the sample."""

    def _payload(self, directory: Path, **overrides):
        from onpolicy import rng_state

        controller = entry.initial_controller(1)
        optimizer = torch.optim.Adam(controller.parameters, lr=train_pilot.LEARNING_RATE)
        state = {
            "seed": 1,
            "environment_steps": 0,
            "episodes": 0,
            "cursor": 0,
            "updates": [],
            "rng": rng_state(),
        }
        path = directory / "guard.pt"
        entry.save(controller, optimizer, state, path, freeze_sha="f" * 40)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.update(overrides)
        torch.save(payload, path)
        return path

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = SCRATCH / "guards"
        cls.directory.mkdir(parents=True, exist_ok=True)

    def test_matching_identity_resumes(self) -> None:
        path = self._payload(self.directory)
        payload = entry.load_for_resume(path, training_seed=1, freeze_sha="f" * 40)
        self.assertEqual(1, int(payload["state"]["seed"]))

    def test_a_different_freeze_is_refused(self) -> None:
        path = self._payload(self.directory)
        with self.assertRaises(RuntimeError) as caught:
            entry.load_for_resume(path, training_seed=1, freeze_sha="a" * 40)
        self.assertIn("code_freeze_sha", str(caught.exception))

    def test_a_missing_rng_block_is_refused(self) -> None:
        path = self._payload(self.directory)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["state"]["rng"] = None
        torch.save(payload, path)
        with self.assertRaises(RuntimeError) as caught:
            entry.load_for_resume(path, training_seed=1, freeze_sha="f" * 40)
        self.assertIn("rng", str(caught.exception))

    def test_a_different_trainer_tag_is_refused(self) -> None:
        path = self._payload(self.directory, trainer_tag="onpolicy-v2")
        with self.assertRaises(RuntimeError) as caught:
            entry.load_for_resume(path, training_seed=1, freeze_sha="f" * 40)
        self.assertIn("trainer_tag", str(caught.exception))

    def test_a_different_budget_is_refused(self) -> None:
        path = self._payload(self.directory, budget_bytes=entry.BUDGET_BYTES + 1000)
        with self.assertRaises(RuntimeError) as caught:
            entry.load_for_resume(path, training_seed=1, freeze_sha="f" * 40)
        self.assertIn("budget_bytes", str(caught.exception))

    def test_the_wrong_training_seed_is_refused(self) -> None:
        path = self._payload(self.directory)
        with self.assertRaises(RuntimeError) as caught:
            entry.load_for_resume(path, training_seed=2, freeze_sha="f" * 40)
        self.assertIn("seed", str(caught.exception))

    @unittest.skipUnless(LEGACY_CHECKPOINT.exists(), "the archived 400k checkpoint is not on disk")
    def test_the_archived_checkpoint_cannot_resume_but_can_run_forward(self) -> None:
        """Forward-only is the whole boundary: the weights may speak, the optimiser may not return.

        The archived arm was trained while an unaffordable SEND was still an action, so its
        gradients -- not just its starting point -- belong to a different experiment.
        """
        path = LEGACY_CHECKPOINT
        with self.assertRaises(RuntimeError) as caught:
            entry.load_for_resume(path, training_seed=1, freeze_sha="f" * 40)
        message = str(caught.exception)
        self.assertIn("trainer_tag", message)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        controller = entry.initial_controller(1)
        controller.load_state_dict(payload["controller"])
        self.assertNotEqual(
            common.weight_fingerprint(controller),
            common.weight_fingerprint(entry.initial_controller(1)),
            "the archived file carried initial weights, so this test proves nothing",
        )
        logits, values = controller.forward(torch.zeros(3, 19))
        self.assertTrue(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(values).all()))


class DeterministicResumeTests(unittest.TestCase):
    """Two updates in a row against save-after-one, restore, then the second."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.directory = SCRATCH / "determinism"
        cls.directory.mkdir(parents=True, exist_ok=True)

    def _step(self, controller, optimizer, state) -> dict:
        stream_seed = common.TRAIN_SEEDS[state["cursor"] % len(common.TRAIN_SEEDS)]
        state["cursor"] += 1
        episode = _rollout(controller, stream_seed, training_seed=1, index=state["episodes"])
        state["episodes"] += 1
        report, _details = entry.update_record(
            controller,
            optimizer,
            [episode],
            at_steps=state["environment_steps"] + common.STEPS_PER_EPISODE,
            batch_seeds=[stream_seed],
            elapsed=0.0,
        )
        state["environment_steps"] = report["at_steps"]
        from onpolicy import rng_state

        state["rng"] = rng_state()
        return report

    def _fresh(self):
        import random

        random.seed(1)
        np.random.seed(1)
        controller = entry.initial_controller(1)
        optimizer = torch.optim.Adam(controller.parameters, lr=train_pilot.LEARNING_RATE)
        from onpolicy import rng_state

        return controller, optimizer, {
            "seed": 1,
            "environment_steps": 0,
            "episodes": 0,
            "cursor": 0,
            "updates": [],
            "rng": rng_state(),
        }

    def test_resumed_second_update_matches_the_continuous_one(self) -> None:
        first_controller, first_optimizer, first_state = self._fresh()
        self._step(first_controller, first_optimizer, first_state)
        continuous = self._step(first_controller, first_optimizer, first_state)

        split_controller, split_optimizer, split_state = self._fresh()
        self._step(split_controller, split_optimizer, split_state)
        path = self.directory / "determinism.pt"
        entry.save(split_controller, split_optimizer, split_state, path, freeze_sha="f" * 40)
        payload = entry.load_for_resume(path, training_seed=1, freeze_sha="f" * 40)
        restored_controller = entry.initial_controller(1)
        restored_optimizer = torch.optim.Adam(
            restored_controller.parameters, lr=train_pilot.LEARNING_RATE
        )
        restored_controller.load_state_dict(payload["controller"])
        restored_optimizer.load_state_dict(payload["optimizer"])
        restored_state = dict(payload["state"])
        from onpolicy import restore_rng

        self.assertTrue(restore_rng(restored_state["rng"]))
        resumed = self._step(restored_controller, restored_optimizer, restored_state)

        self.assertEqual(
            entry._batch_signature(continuous), entry._batch_signature(resumed)
        )
        self.assertEqual(
            common.weight_fingerprint(first_controller),
            common.weight_fingerprint(restored_controller),
        )
        self.assertEqual(
            entry._optimizer_signature(first_optimizer),
            entry._optimizer_signature(restored_optimizer),
        )

    def test_report_carries_the_absolute_sample_accounting(self) -> None:
        """§5's list, checked as fields rather than as a proportion."""
        controller, optimizer, state = self._fresh()
        report = self._step(controller, optimizer, state)
        for key in (
            "candidates",
            "initial_executable_candidates",
            "actionable_candidates",
            "actor_terms",
            "masked_by_credit",
            "masked_by_capacity",
            "steps_with_candidates",
            "steps_with_zero_actionable",
            "episodes_without_actionable",
            "episodes_without_decisions",
            "sampled_send",
            "requested_by_policy",
            "executed_policy_emissions",
            "policy_budget_refusals",
            "stochastic_send_rate",
            "stochastic_entropy_per_actionable_candidate",
            "greedy_send_rate",
            "greedy_entropy_per_actionable_candidate",
            "greedy_first_credit_masked_step",
            "message_first",
            "message_update",
            "message_retry_same_view",
            "optimizer_steps",
            "minibatches",
            "value_loss",
            "approx_kl",
            "clip_fraction",
            "gradient_norm_mean",
            "value_explained_variance",
            "nonfinite",
        ):
            self.assertIn(key, report, f"{key} missing from the update report")
        self.assertEqual(
            report["actionable_candidates"]
            + report["masked_by_credit"]
            + report["masked_by_capacity"],
            report["candidates"],
        )
        self.assertLessEqual(
            float(report["likelihood_alignment"]["max_abs_gap"]), entry.LOG_PROB_TOLERANCE
        )
        self.assertEqual(0, report["nonfinite"])


class PreflightHelpersTests(unittest.TestCase):
    def test_streams_match_the_registry_they_were_cut_against(self) -> None:
        report = common.verify_streams()
        self.assertTrue(report["passed"], str(report["mismatches"]))
        self.assertEqual(72, report["streams_checked"])
        self.assertEqual(48, len(report["training_block"]))
        self.assertEqual(24, len(report["development_block"]))

    def test_execution_graph_has_no_phantom_and_every_member_exists(self) -> None:
        graph = common.executed_modules()
        self.assertNotIn("integrated_sar/comm_transcript.py", graph)
        for path in graph:
            self.assertTrue((common.ROOT / path).exists(), path)

    def test_blob_identity_calls_absence_absence(self) -> None:
        report = common.blob_identity(["integrated_sar/comm_transcript.py"])
        entry_ids = report["integrated_sar/comm_transcript.py"]
        self.assertFalse(entry_ids["in_head"])
        self.assertFalse(entry_ids["on_disk"])
        self.assertIsNone(entry_ids["head_blob"])

    def test_launch_ledger_caps_and_counts_failures(self) -> None:
        original = common.LAUNCH_LEDGER
        temporary = SCRATCH / "ledger.json"
        if temporary.exists():
            temporary.unlink()
        common.LAUNCH_LEDGER = temporary
        try:
            ledger = common.LaunchLedger()
            self.assertEqual(common.LAUNCH_CAP, ledger.remaining())
            ledger.reserve("probe", 700000)
            ledger.record("probe", 700000, False, "simulated technical failure")
            stored = json.loads(temporary.read_text(encoding="utf-8"))
            self.assertEqual(1, stored["starts"])
            self.assertEqual(1, stored["failures"], "a failure must still spend the ceiling")
        finally:
            common.LAUNCH_LEDGER = original

    def test_shipped_code_has_not_moved_since_the_mask_fix(self) -> None:
        report = common.production_diff()
        self.assertTrue(report["empty"], str(report))
        self.assertEqual([], report["integrated_sar_changed"])

    def test_production_defaults_are_still_the_ones_every_round_assumed(self) -> None:
        defaults = common.setting()["production_defaults"]
        self.assertFalse(defaults["comm_ack"])
        self.assertEqual("legacy_shared_bound", defaults["comm_timing_mode"])
        self.assertEqual("immediate", defaults["comm_policy"])
        self.assertIsNone(defaults["comm_budget_bytes"])

    def test_sealed_block_still_has_no_artefacts(self) -> None:
        report = common.sealed_block_scan()
        self.assertEqual(0, report["count"], str(report["artefacts_found"]))

    def test_the_round_has_a_budget_and_only_one(self) -> None:
        setting = common.setting()
        self.assertEqual(29300, setting["budget_bytes"])
        self.assertFalse(setting["ack"], "the reverse channel is not authorised this round")
        self.assertEqual(19, setting["feature_dimension"])
        self.assertEqual(1500, setting["launch_cap"])
        self.assertEqual(12, setting["probe_episodes"])
        self.assertEqual(4, setting["episodes_per_update"])
        self.assertEqual(1_200_000, setting["total_step_cap"])


class JudgeTests(unittest.TestCase):
    """The preregistered gates, exercised on synthetic intervals before any result exists.

    A judgement rule that has never been seen to fire is not a rule; it is a sentence. These cases
    are the boundary conditions §7 names -- two seeds over the line and one within the margin, two
    over the line and one outside it, and a correctness violation overriding both.
    """

    def _stats(self, arms: dict[str, dict]) -> dict:
        return {"per_arm": arms, "development_rows": []}

    def _arm(self, completion: list[float], bytes_: list[float]) -> dict:
        return {
            "on_time": {
                "mean_difference": (completion[0] + completion[1]) / 2.0,
                "ci95": completion,
            },
            "airtime_bytes": {
                "mean_difference": (bytes_[0] + bytes_[1]) / 2.0,
                "ci95": bytes_,
            },
        }

    def _gates(self, passed: bool) -> dict:
        return {"passed": passed}

    def test_two_seeds_over_the_line_and_one_within_margin_passes(self) -> None:
        import report_retrain as report

        arms = {
            "masked_s1_at400000": self._arm([0.04, 0.6], [-100.0, 50.0]),
            "masked_s2_at400000": self._arm([0.01, 0.4], [-20.0, 20.0]),
            "masked_s3_at400000": self._arm([-0.04, 0.3], [-10.0, 40.0]),
        }
        judgement = report.judge(self._stats(arms), self._gates(True))
        self.assertTrue(judgement["primary_effect"]["passed"])
        self.assertEqual(2, judgement["primary_effect"]["seeds"])

    def test_the_third_seed_outside_the_margin_blocks_the_gate(self) -> None:
        import report_retrain as report

        arms = {
            "masked_s1_at400000": self._arm([0.04, 0.6], [-100.0, 50.0]),
            "masked_s2_at400000": self._arm([0.01, 0.4], [-20.0, 20.0]),
            "masked_s3_at400000": self._arm([-0.2, -0.05], [-10.0, 5.0]),
        }
        judgement = report.judge(self._stats(arms), self._gates(True))
        self.assertFalse(judgement["primary_effect"]["passed"])

    def test_cost_needs_completion_non_inferior_and_strictly_cheaper(self) -> None:
        import report_retrain as report

        arms = {
            "masked_s1_at400000": self._arm([-0.02, 0.30], [-900.0, -120.0]),
            "masked_s2_at400000": self._arm([-0.04, 0.10], [-500.0, -10.0]),
            "masked_s3_at400000": self._arm([-0.04, 0.20], [-100.0, 100.0]),
        }
        judgement = report.judge(self._stats(arms), self._gates(True))
        self.assertFalse(judgement["primary_effect"]["passed"])
        self.assertTrue(judgement["non_inferiority_cost"]["passed"])

    def test_a_correctness_violation_blocks_both_paths(self) -> None:
        import report_retrain as report

        arms = {
            "masked_s1_at400000": self._arm([0.1, 0.9], [-900.0, -100.0]),
            "masked_s2_at400000": self._arm([0.1, 0.9], [-900.0, -100.0]),
            "masked_s3_at400000": self._arm([0.1, 0.9], [-900.0, -100.0]),
        }
        judgement = report.judge(self._stats(arms), self._gates(False))
        self.assertFalse(judgement["primary_effect"]["passed"])
        self.assertFalse(judgement["non_inferiority_cost"]["passed"])

    def test_a_missing_arm_is_not_read_as_zero(self) -> None:
        import report_retrain as report

        arms = {"masked_s1_at400000": self._arm([0.1, 0.9], [-900.0, -100.0])}
        judgement = report.judge(self._stats(arms), self._gates(True))
        self.assertFalse(judgement["primary_effect"]["passed"])
        self.assertFalse(judgement["per_seed"][2]["available"])


RESULTS_FILE = common.HERE / "COMM_BUDGET_MASKED_RETRAIN_RESULTS.md"


@unittest.skipUnless(RESULTS_FILE.exists(), "the results document is written after the run")
class RenderedReportTests(unittest.TestCase):
    """Readings of the delivered report, so a wording defect cannot survive a rerender."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = RESULTS_FILE.read_text(encoding="utf-8")
        cls.data = json.loads(
            (common.HERE / "COMM_BUDGET_MASKED_RETRAIN_RESULTS.json").read_text(encoding="utf-8")
        )

    def test_units_are_named_wherever_a_difference_is_quoted(self) -> None:
        self.assertIn("个/ep", self.text)
        self.assertIn("B/ep", self.text)
        self.assertIn("B/episode", self.text)

    def test_the_share_is_always_beside_its_absolute_counts(self) -> None:
        share = self.data["measured"]["actionable_share"]
        self.assertIn(f"{share:.2%}", self.text)
        self.assertGreater(self.data["measured"]["actionable"], 0)
        self.assertIn("actor 项", self.text)

    def test_the_seeds_are_never_pooled_into_seventy_two(self) -> None:
        per_seed = self.data["judgement"]["per_seed"]
        self.assertEqual(3, len(per_seed))
        for item in per_seed.values():
            if item.get("available"):
                self.assertEqual(24, int(item["completion"]["pairs"]))

    def test_a_failed_gate_is_not_worded_as_equivalence(self) -> None:
        for arm in ("主效果", "非劣降本"):
            self.assertIn(arm, self.text)
        if not self.data["judgement"]["primary_effect"]["passed"]:
            self.assertNotIn("证明学习优于规则", self.text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
