# -*- coding: utf-8 -*-
"""ROUND73 zero-environment gates for the pure-cost actor objective.

No real environment construction, no model training, no trunk forward, no optimizer
step.  These gates exercise the REAL frozen trainer ``batch_samples`` and the REAL
log-softmax to prove that the single intervention (actor advantage = -A_cost) is wired
through the actual actor path and that A_task / lambda no longer reach it.
"""
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _d in (str(HERE),
           str(REPO / "runs" / "comm_cost_controller_stage_a_20260927"),
           str(REPO / "runs" / "comm_tight_budget_20260919"),
           str(REPO / "runs" / "comm_cost_v2_rerun_20260928"),
           str(REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
                / "round51_frozen_source_gradient_close" / "frozen_sources")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np
import torch
import cost_math as cm
import canonical_v2_trainer as trainer
import v2_common as common
import r73_pure_cost_objective as r73

R71_DIR = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2" / "round71_critic_initialization_calibration"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _FakeCollector:
    def __init__(self, records):
        self.records = records


def _episode(lam, task_val, cost_air, idx=0):
    records = [
        {"step": 0, "features": np.zeros((1, 19), dtype=np.float32), "targets": [0],
         "actions": {0: 0}, "actionable": {0}, "log_prob": -0.60, "value": 0.10, "cost_value": 0.02},
        {"step": 4, "features": np.zeros((1, 19), dtype=np.float32), "targets": [0],
         "actions": {0: 1}, "actionable": {0}, "log_prob": -0.70, "value": 0.20, "cost_value": 0.03},
    ]
    return {"collector": _FakeCollector(records), "steps": 6,
            "cost_stream": {1: {"airtime_bytes": cost_air}, 5: {"airtime_bytes": cost_air * 0.5}},
            "task_stream": {1: task_val, 5: task_val * 0.5},
            "lambda": lam, "episode_index": idx}


class TestR73PureCostGates(unittest.TestCase):
    def setUp(self):
        self._gamma, self._gae = common.GAMMA, common.GAE_LAMBDA
        self._orig_batch_samples = trainer.batch_samples
        self._orig_actor_advantage = cm.actor_advantage
        common.GAMMA, common.GAE_LAMBDA = r73.LIVE_GAMMA, r73.LIVE_GAE_LAMBDA  # live 1.0/1.0

    def tearDown(self):
        common.GAMMA, common.GAE_LAMBDA = self._gamma, self._gae
        trainer.batch_samples = self._orig_batch_samples
        cm.actor_advantage = self._orig_actor_advantage

    def test_pure_cost_value_invariant_to_task_and_lambda(self):
        at = torch.tensor([1.7, -0.3, 0.0, 4.2], dtype=torch.float64)
        ac = torch.tensor([0.5, 0.2, 0.9, 0.0], dtype=torch.float64)
        base = r73.pure_cost_actor_advantage(at, ac, 1.0)
        self.assertTrue(torch.allclose(base, -ac))
        for lam in (0.0, 1.0, 1.625, 9.0):
            self.assertTrue(torch.equal(r73.pure_cost_actor_advantage(at, ac, lam), base))
        self.assertTrue(torch.equal(r73.pure_cost_actor_advantage(at + 1000.0, ac, 3.3), base))

    def test_real_batch_samples_actor_gradient_inputs_invariant(self):
        receipt = r73.install_pure_cost_objective(
            trainer, cm=cm, freeze={"actor_objective": r73.R73_ACTOR_OBJECTIVE})
        self.assertEqual(receipt["actor_objective"], r73.R73_ACTOR_OBJECTIVE)
        e1 = _episode(1.625, task_val=1.0, cost_air=200.0)
        e2 = _episode(5.000, task_val=-3.0, cost_air=200.0)   # non-zero A_task, different lambda
        s1 = trainer.batch_samples([e1])
        s2 = trainer.batch_samples([e2])
        # The scoped swap is restored after each call: cost_math is untouched outside it.
        self.assertIs(cm.actor_advantage, self._orig_actor_advantage)
        self.assertEqual(len(s1), 2)
        # (a) R73 actor advantage is bit-identical across A_task and lambda changes.
        for a, b in zip((x["advantage"] for x in s1), (x["advantage"] for x in s2)):
            self.assertEqual(a, b)
        # (b) The other actor-surrogate inputs (log_prob, features) are unchanged too, so the
        #     actor surrogate and its gradient are invariant, not merely the scalar.
        for x, y in zip(s1, s2):
            self.assertEqual(x["log_prob"], y["log_prob"])
            self.assertTrue(np.array_equal(x["features"], y["features"]))
        # (c) R73 advantage equals -A_cost (A_cost = return_cost - behaviour_value_cost).
        for x in s1:
            self.assertAlmostEqual(x["advantage"], -(x["return_cost"] - x["behaviour_value_cost"]), places=6)
        # (d) A_task is genuinely non-zero here, so invariance is not the trivial task==0 case.
        self.assertTrue(any(abs(x["return_task"] - x["behaviour_value_task"]) > 1e-9 for x in s1))
        # (e) CONTROL (lambda*A_task - A_cost) would differ -> the override is what removed the term.
        lam = 1.625
        control = [lam * (x["return_task"] - x["behaviour_value_task"])
                   - (x["return_cost"] - x["behaviour_value_cost"]) for x in s1]
        self.assertTrue(any(abs(cv - x["advantage"]) > 1e-9 for cv, x in zip(control, s1)))

    def test_hand_cost_example_expected_actor_gradient(self):
        p, c = 0.58, 3.0
        z = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            z[0, 0] = 0.0
            z[0, 1] = math.log((1.0 - p) / p)   # softmax -> [p, 1-p]
        logp = torch.log_softmax(z, dim=-1)[0]
        pi = torch.softmax(z.detach(), dim=-1)[0]
        v_cost = p * c
        r = torch.tensor([c, 0.0], dtype=torch.float64)          # SEND costs c, SKIP costs 0
        a_cost = r - v_cost                                        # action-dependent cost advantage
        a_actor = r73.pure_cost_actor_advantage(torch.zeros(2, dtype=torch.float64), a_cost, r73.LIVE_LAMBDA_INIT)
        ascent = 0.0
        for a in range(2):
            score = torch.autograd.grad(logp[a], z, retain_graph=True)[0][0, 0].item()
            ascent += float(pi[a]) * float(a_actor[a]) * score
        self.assertAlmostEqual(ascent, -c * p * (1.0 - p), places=9)   # -0.2436 c
        self.assertLess(ascent, 0.0)          # ascent lowers the SEND logit (less costly sending)
        self.assertGreater(-ascent, 0.0)      # the minimized loss has the opposite sign

    def test_init_file_identity_matches_round71_index(self):
        index = json.loads((R71_DIR / "ROUND71_CHECKPOINT_INDEX.json").read_text(encoding="utf-8"))
        ckpt_dir = R71_DIR / "raw" / "checkpoints"
        for seed_key, node in index["seeds"].items():
            init = node["calibrated_init"]
            path = ckpt_dir / init["file"]
            self.assertTrue(path.is_file(), f"missing INIT {path}")
            self.assertEqual(_sha256_file(path), init["file_sha256"], seed_key)

    def test_gated_module_is_the_live_trainer_bytes(self):
        """The module these gates exercise IS the module the live runtime imports, byte for byte."""
        live = REPO.parent / "zhenghe" / "runs" / "comm_cost_v2_rerun_20260928" / "v2_trainer.py"
        gated = (REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
                 / "round51_frozen_source_gradient_close" / "frozen_sources"
                 / "canonical_v2_trainer.py")
        self.assertTrue(live.is_file(), f"the live canonical trainer is missing: {live}")
        self.assertEqual(_sha256_file(live), _sha256_file(gated))
        # the cost_math the trainer itself imports is the object these gates import
        self.assertIs(trainer.cm, cm)
        self.assertIs(trainer.cm.actor_advantage, cm.actor_advantage)

    def test_version_identity_and_install_refuses_wrong_freeze(self):
        self.assertEqual(r73.R73_ACTOR_OBJECTIVE, "pure-cost/1")
        self.assertIn("pure-cost-actor-objective/1", r73.R73_TRAINING_VERSION)
        with self.assertRaises(RuntimeError):
            r73.install_pure_cost_objective(trainer, cm=cm, freeze={"actor_objective": "lagrangian/1"})
        with self.assertRaises(RuntimeError):
            r73.install_pure_cost_objective(trainer, cm=cm, freeze={})


if __name__ == "__main__":
    unittest.main(verbosity=2)