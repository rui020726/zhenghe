from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import torch

import r58_committed_prefix_bridge as bridge


class CommittedPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.claim_root = cls.root / "round58_gen3/raw"
        cls.source_root = cls.root / "round58_gen3_resume_v22/raw"
        cls.checkpoint = cls.source_root / "checkpoints/v2_train_s11.pt"
        cls.commit_path = cls.source_root / "checkpoints/r40_batch_commit_s11_b14.json"
        payload = torch.load(cls.checkpoint, map_location="cpu", weights_only=False)
        cls.trainer = bridge.prior_probe._frozen_trainer()
        cls.previous_common = {name: getattr(cls.trainer.common, name)
                               for name in ("GAE_LAMBDA", "GAMMA", "B0_BYTES")}
        state = payload["training_state"]
        for name, key in (("GAE_LAMBDA", "gae_lambda"), ("GAMMA", "gamma"),
                          ("B0_BYTES", "b0_bytes")):
            setattr(cls.trainer.common, name, float(state[key]))
        cls.controller = cls.trainer._restore_controller(payload, state)

    @classmethod
    def tearDownClass(cls):
        for name, value in cls.previous_common.items():
            setattr(cls.trainer.common, name, value)

    def validator(self, path, seed):
        self.assertEqual(seed, 11)
        return {"sha256": bridge.digest(path),
                "commit_sha256": bridge.digest(self.commit_path), "batch": 14}

    def test_exact_batch14_prefix_and_previous_rng_control(self):
        result = bridge.materialize(
            self.controller, seed=11, claim_root=self.claim_root,
            source_root=self.source_root, source_checkpoint=self.checkpoint,
            validate_current_source=self.validator)
        self.assertEqual(result["proof"]["episodes"], [60, 61])
        self.assertEqual(result["proof"]["samples"], 169)
        self.assertTrue(result["proof"]["previous_episode_rng_equals_commit"])
        self.assertEqual(result["last_episode"], 61)

    def test_bad_commit_or_tape_digest_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "commit"):
            bridge.materialize(
                self.controller, seed=11, claim_root=self.claim_root,
                source_root=self.source_root, source_checkpoint=self.checkpoint,
                validate_current_source=lambda *_: {"sha256": bridge.digest(self.checkpoint),
                                                      "commit_sha256": "0" * 64, "batch": 14})
        original = bridge.digest
        def damaged(path):
            if "e60_" in Path(path).name and Path(path).name.endswith("_tape.jsonl"):
                return "0" * 64
            return original(path)
        with mock.patch.object(bridge, "digest", side_effect=damaged):
            with self.assertRaisesRegex(RuntimeError, "tape source bytes"):
                bridge.materialize(
                    self.controller, seed=11, claim_root=self.claim_root,
                    source_root=self.source_root, source_checkpoint=self.checkpoint,
                    validate_current_source=self.validator)
