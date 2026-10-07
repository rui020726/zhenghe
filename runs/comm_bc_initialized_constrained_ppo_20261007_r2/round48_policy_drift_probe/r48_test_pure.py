"""Pure-computation tests for the ROUND48 probe (no environments, no training)."""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("r48_probe", HERE / "r48_probe.py")
assert SPEC is not None and SPEC.loader is not None
R48 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R48
SPEC.loader.exec_module(R48)
RUN_DIR = R48.RUN_DIR


def zero_action_head_controller(seed: int) -> dict:
    """A controller whose action head outputs equal logits -> p(SEND) = 0.5 exactly."""
    import torch
    ep128 = torch.load(RUN_DIR / "v6" / "checkpoints" / f"v2_train_s{seed}_ep128.pt",
                       map_location="cpu", weights_only=False)["controller"]
    ep128["action_head"]["weight"] = torch.zeros_like(ep128["action_head"]["weight"])
    ep128["action_head"]["bias"] = torch.zeros_like(ep128["action_head"]["bias"])
    return ep128


class ActionMappingTests(unittest.TestCase):
    def test_send_and_hold_indices_come_from_the_real_mapping(self):
        self.assertEqual(R48.SEND, 1)
        self.assertEqual(R48.HOLD, 0)

    def test_greedy_tie_sends(self):
        import torch
        controller = zero_action_head_controller(11)
        batch = torch.zeros(1, 19)
        p = R48.p_send(controller, batch)
        self.assertEqual(float(p[0]), 0.5)
        greedy = R48.SEND if float(p[0]) >= 0.5 else R48.HOLD
        self.assertEqual(greedy, R48.SEND)


class InputValidationTests(unittest.TestCase):
    def test_wrong_feature_width_is_rejected(self):
        import torch
        controller = R48.load_controller(
            RUN_DIR / "v6" / "checkpoints" / "v2_train_s11_ep128.pt")
        with self.assertRaises(Exception):
            R48.forward_logits(controller, torch.zeros(1, 18))


class ModelIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity = R48.build_model_identity()

    def test_ep0_is_the_bc_post_model_and_matches_registered_identity(self):
        for seed in ("11", "22", "33"):
            block = self.identity["roles"]["ep0_training_init"][seed]
            self.assertTrue(block["registered_matches_recomputed"], seed)
            self.assertTrue(block["bc_control_rows_use_same_parameter_identity"], seed)
            self.assertTrue(block["r43_era_ep0_file_not_the_round46_init"]
                            ["four_module_parameter_sha256"]
                            != block["round46_initializer_parameter_sha256"])

    def test_ep128_file_and_parameter_hashes_verified(self):
        for seed in ("11", "22", "33"):
            block = self.identity["roles"]["ep128_final"][seed]
            self.assertTrue(block["file_sha256_verified"], seed)
            self.assertTrue(block["parameter_sha256_verified"], seed)

    def test_loading_guards_recorded(self):
        loading = self.identity["loading"]
        self.assertEqual(loading["environment_constructor_calls"], 0)
        self.assertEqual(loading["checkpoint_write_calls"], 0)
        self.assertFalse(loading["optimizer_restored"])
        self.assertFalse(loading["rng_restored"])

    def test_r43_era_ep0_is_a_different_policy_not_the_round46_init(self):
        for seed in ("11", "22", "33"):
            block = self.identity["roles"]["ep0_training_init"][seed]
            self.assertNotEqual(block["r43_era_ep0_file_not_the_round46_init"]
                                ["four_module_parameter_sha256"],
                                block["round46_initializer_parameter_sha256"])


class PanelAndDriftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((HERE / "PROBE_INPUT_MANIFEST.json").read_text(encoding="utf-8"))
        cls.results = json.loads((HERE / "POLICY_DRIFT_RESULTS.json").read_text(encoding="utf-8"))

    def test_panel_records_are_all_free_first_report_decisions(self):
        self.assertGreater(self.manifest["panel_records"], 0)
        for record in self.manifest.get("exclusions", []):
            self.assertNotEqual(record["source"], "free_one_shot_choice")

    def test_duplicate_vectors_are_retained_per_target(self):
        self.assertGreater(self.results["feature_observations"]["records"], 0)
        self.assertGreater(self.results["feature_observations"]["distinct_feature_vectors"], 0)
        self.assertLessEqual(self.results["feature_observations"]["distinct_feature_vectors"],
                             self.results["feature_observations"]["records"])

    def test_transitions_sum_to_panel_size_per_seed(self):
        for seed in ("11", "22", "33"):
            block = self.results["per_seed"][seed]
            total = sum(block["greedy_transitions_ep0_to_ep128"].values())
            self.assertEqual(total, block["panel_records"], seed)

    def test_drift_groups_partition_the_panel_within_each_bench_family(self):
        for seed in ("11", "22", "33"):
            block = self.results["per_seed"][seed]
            for bench in ("vs_late", "vs_bc"):
                total = sum(g["count"] for label, g in block["cost_group_metadata"].items()
                            if label.endswith(bench))
                self.assertEqual(total, block["panel_records"], f"{seed}/{bench}")

    def test_ep128_reproduction_validated(self):
        self.assertEqual(self.results["validation"]["ep128_reproduction_mismatches"], 0)
        self.assertEqual(self.results["validation"]["action_mapping"]["SEND"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
