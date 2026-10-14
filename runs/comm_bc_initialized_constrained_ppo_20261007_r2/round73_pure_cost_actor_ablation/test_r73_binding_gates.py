# -*- coding: utf-8 -*-
"""ROUND73 zero-environment binding/contract gates.

No environment construction, no episode, no optimizer step, no trunk forward.  These gates
prove the ROUND73 identity wiring (licence, run identity, checkpoint contract) and the
model-only INIT read path against the REAL ROUND71 calibrated INIT payload (a parameter read
only: torch.load + sha256, never a forward).
"""
import copy
import json
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

import cost_math as cm
import canonical_v2_trainer as trainer
import r73_pure_cost_objective as r73
import r73_checkpoint as ckpt
import r73_trial_binding as binding


class TestR73BindingGates(unittest.TestCase):
    def test_archived_licence_opens_gate_and_lookalikes_do_not(self):
        receipt = binding.licence_receipt()
        self.assertEqual(receipt["status"], "passed")
        for key, value in receipt.items():
            if isinstance(value, bool):
                self.assertTrue(value, key)

    def test_run_identity_is_round73_and_not_round71(self):
        ident = binding.trial_run_identity().as_dict()
        self.assertEqual(ident["task_id"], "ROUND73-PURE-COST-ACTOR-ABLATION")
        self.assertEqual(ident["branch"], "exp/round73-pure-cost-actor-ablation")
        self.assertEqual(ident["quota_identity"], binding.R73_QUOTA_IDENTITY)
        self.assertIn("579-normal", ident["quota_identity"])
        self.assertTrue(ident["training_version"].endswith("+r73-pure-cost-actor-objective/1"))
        self.assertNotEqual(ident["training_version"], ckpt.readiness.R71_TRAINING_VERSION)

    def test_checkpoint_contract_extends_round71_and_refuses_foreign(self):
        self.assertTrue(ckpt.R73_TRAINING_VERSION.startswith(ckpt.readiness.R71_TRAINING_VERSION))
        good = {"schema": ckpt.R73_STATE_SCHEMA, "trainer_tag": ckpt.R73_TRAINER_TAG,
                "loss_version": ckpt.R73_LOSS_VERSION, "actor_objective": ckpt.R73_ACTOR_OBJECTIVE,
                "gae_schema": "per-decision-next-value/1",
                "optimizer_schema": ckpt.R73_OPTIMIZER_SCHEMA,
                "structure": ckpt.R73_STRUCTURE, "freeze": ckpt.R73_FREEZE,
                "training_state": {"loss_version": ckpt.R73_LOSS_VERSION,
                                   "state_schema": ckpt.R73_STATE_SCHEMA,
                                   "optimizer_schema": ckpt.R73_OPTIMIZER_SCHEMA}}
        state = ckpt._validate_r73_base(Path("v2_train_s11.pt"), copy.deepcopy(good))
        self.assertEqual(state["optimizer_schema"], ckpt.R73_OPTIMIZER_SCHEMA)
        foreign = copy.deepcopy(good)
        foreign["trainer_tag"] = ckpt.readiness.R71_TRAINER_TAG
        with self.assertRaises(RuntimeError):
            ckpt._validate_r73_base(Path("v2_train_s11.pt"), foreign)
        foreign2 = copy.deepcopy(good)
        foreign2["schema"] = ckpt.readiness.R71_STATE_SCHEMA
        with self.assertRaises(RuntimeError):
            ckpt._validate_r73_base(Path("v2_train_s11.pt"), foreign2)
        no_task_term = copy.deepcopy(good)
        no_task_term["actor_objective"] = "lagrangian/1"
        with self.assertRaises(RuntimeError):
            ckpt._validate_r73_base(Path("v2_train_s11.pt"), no_task_term)

    def test_init_payload_identity_and_refusals(self):
        """Payload-level proof of the staged INIT (torch.load + sha only, no controller, no forward).

        The controller-level parameter/module identity is proven by the REAL first-batch gate on
        the configured runtime (``r73_load_init_read_only`` inside ``_first_batch_gate``); here we
        stay at the payload level so this suite remains zero-consumption by construction.
        """
        import hashlib
        import torch
        import r73_trial_execute as execute
        registered = execute._registered_init_identity(11)
        path = execute.R71_CHECKPOINTS / registered["source_file"]
        self.assertTrue(path.is_file())
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        self.assertEqual(digest.hexdigest(), registered["file_sha256"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["training_version"], ckpt.readiness.R71_TRAINING_VERSION)
        self.assertEqual(payload["structure"], ckpt.readiness.R71_STRUCTURE)
        self.assertEqual(payload["schema"], ckpt.readiness.R71_STATE_SCHEMA)
        self.assertEqual(set(payload["controller"]), set(ckpt.MODULES))
        self.assertEqual(int(payload["training_state"]["episodes_done"]), 0)
        self.assertEqual(int(payload["training_state"]["updates"]), 0)
        # the registered identity carries the per-module calibration digests of that seed
        cal = registered["calibration"]
        for key in ("trunk_sha256", "early_head_sha256", "late_head_sha256",
                    "value_task_head_sha256", "value_cost_head_sha256"):
            self.assertTrue(cal[key])
        self.assertEqual(registered["init_parameter_sha256"], registered["parameter_sha256"])
        # a tampered file identity is refused BEFORE any controller is constructed
        tampered = copy.deepcopy(registered)
        tampered["file_sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            ckpt.r73_load_init_read_only(path, registered=tampered)

    def test_scoped_install_is_idempotent_and_restores_cost_math(self):
        original = cm.actor_advantage
        original_batch = trainer.batch_samples
        r73.install_pure_cost_objective(trainer, cm=cm,
                                        freeze={"actor_objective": r73.R73_ACTOR_OBJECTIVE})
        first = trainer.batch_samples
        receipt = r73.install_pure_cost_objective(
            trainer, cm=cm, freeze={"actor_objective": r73.R73_ACTOR_OBJECTIVE})
        # a second install must not silently stack a wrapper around the same original
        self.assertIs(trainer.batch_samples, first)
        self.assertIn("already installed", receipt["patched"])
        self.assertEqual(trainer.batch_samples.__name__, "_scoped_batch_samples")
        self.assertIs(cm.actor_advantage, original)
        trainer.batch_samples = original_batch
        del trainer.R73_ACTOR_OBJECTIVE
        del trainer.R73_OBJECTIVE_VERSION


if __name__ == "__main__":
    unittest.main(verbosity=2)