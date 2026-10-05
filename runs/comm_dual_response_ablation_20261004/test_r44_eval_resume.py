"""Pure schedule and eval-only guard regressions for ROUND44."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import r44_eval_resume as resume


class _Index:
    def __init__(self, complete=()):
        self.complete = set(complete)

    def get(self, seed, mode, scene):
        key = (seed, mode, scene)
        return {"key": key} if key in self.complete else None


class EvalScheduleTests(unittest.TestCase):
    def test_fixed_matrix_has_96_cells_and_retry_is_first(self):
        schedule = resume.evaluation_schedule()
        self.assertEqual(len(schedule), 96)
        self.assertEqual(schedule[0], resume.RETRY_KEY)
        self.assertEqual(len(set(schedule)), 96)
        self.assertEqual(schedule[-1], (33, "sampled", 1_301_015))

    def test_retry_requires_attempt_two_and_reserve_metadata(self):
        metadata = {
            "arm": "NEW", "training_seed": 11,
            "mode": "greedy", "scenario_seed": 1_301_000,
        }
        self.assertEqual(
            resume.validate_eval_claim(
                "model_eval_new_s11_greedy_scene1301000", 2, metadata),
            (resume.RETRY_KEY, "reserve"),
        )
        with self.assertRaises(resume.EvalResumeError):
            resume.validate_eval_claim(
                "model_eval_new_s11_greedy_scene1301000", 1, metadata)

    def test_other_fixed_cell_requires_normal_attempt_one(self):
        metadata = {
            "arm": "NEW", "training_seed": 22,
            "mode": "sampled", "scenario_seed": 1_301_004,
        }
        self.assertEqual(
            resume.validate_eval_claim(
                "model_eval_new_s22_sampled_scene1301004", 1, metadata),
            ((22, "sampled", 1_301_004), "normal"),
        )
        with self.assertRaises(resume.EvalResumeError):
            resume.validate_eval_claim(
                "model_eval_new_s22_sampled_scene1301004", 2, metadata)

    def test_claim_rejects_wrong_metadata_and_foreign_keys(self):
        logical = "model_eval_new_s11_greedy_scene1301000"
        wrong_metadata = {
            "arm": "OLD", "training_seed": 11,
            "mode": "greedy", "scenario_seed": 1_301_000,
        }
        with self.assertRaises(resume.EvalResumeError):
            resume.validate_eval_claim(logical, 2, wrong_metadata)
        with self.assertRaises(resume.EvalResumeError):
            resume.validate_eval_claim(
                "model_eval_new_s11_greedy_scene1301016", 1,
                {"arm": "NEW", "training_seed": 11,
                 "mode": "greedy", "scenario_seed": 1_301_016})

    def test_retry_queue_must_precede_remaining_fixed_cells(self):
        queue = resume._make_claim_guard_queue(_Index())
        self.assertEqual(queue[0], resume.RETRY_KEY)
        self.assertEqual(len(queue), 96)
        completed_retry = resume._make_claim_guard_queue(
            _Index(complete=(resume.RETRY_KEY,)))
        self.assertEqual(len(completed_retry), 95)
        self.assertEqual(completed_retry[0], (11, "greedy", 1_301_001))

    def test_queue_advances_from_retry_to_next_fixed_cell(self):
        queue = resume._make_claim_guard_queue(_Index())
        resume._consume_claim_queue_key(queue, resume.RETRY_KEY)
        self.assertEqual(queue[0], (11, "greedy", 1_301_001))
        self.assertEqual(len(queue), 95)

    def test_queue_rejects_consumption_out_of_order(self):
        queue = resume._make_claim_guard_queue(_Index())
        with self.assertRaises(resume.EvalResumeError):
            resume._consume_claim_queue_key(queue, (11, "greedy", 1_301_001))


class EvalOnlyGuardTests(unittest.TestCase):
    def test_forbidden_training_optimizer_and_checkpoint_calls_raise(self):
        class Optimizer:
            def step(self):
                raise AssertionError("underlying optimizer must not be reached")

        class Adam(Optimizer):
            pass

        def untouched(*_args, **_kwargs):
            raise AssertionError("underlying operation must not be reached")

        run_module = SimpleNamespace(
            train_seed=untouched,
            run_training_episode=untouched,
            run_reference=untouched,
            run_rule=untouched,
        )
        trainer = SimpleNamespace(
            ppo_update=untouched,
            save_checkpoint=untouched,
            load_checkpoint=untouched,
            restore_rng=untouched,
        )
        r40_train = SimpleNamespace(
            train_seed=untouched,
            _save_milestone=untouched,
            _write_immutable_snapshot=untouched,
            _save_preupdate_checkpoint=untouched,
        )
        torch_module = SimpleNamespace(
            save=untouched,
            optim=SimpleNamespace(Optimizer=Optimizer, Adam=Adam),
        )
        guarded = resume.install_read_only_guards(
            run_module, {"trainer": trainer, "r40_train": r40_train}, torch_module)

        self.assertIn("r42_run.train_seed", guarded)
        self.assertIn("Adam.step", guarded)
        with self.assertRaises(resume.EvalOnlyViolation):
            run_module.train_seed()
        with self.assertRaises(resume.EvalOnlyViolation):
            Adam().step()
        with self.assertRaises(resume.EvalOnlyViolation):
            torch_module.save(object(), "forbidden.pt")


if __name__ == "__main__":
    unittest.main()
