import json
import math
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from round50_audit import EXPECTED_TRAINER, OUT, build, read_json


class Round50PureTests(unittest.TestCase):
    def test_real_action_mapping_and_greedy_tie(self):
        replay = read_json(OUT.parent / "round49_bc_input_policy_closure" / "BC_PATH_REPLAY.json")
        self.assertEqual(replay["action_mapping"], {"SEND": 1, "SKIP": 0})
        self.assertEqual(replay["greedy_tie_rule"], "p_send >= 0.5 -> SEND")
        self.assertEqual("SEND" if 0.5 >= 0.5 else "SKIP", "SEND")

    def test_wrong_source_role_cannot_be_substituted(self):
        replay = read_json(OUT.parent / "round49_bc_input_policy_closure" / "BC_PATH_REPLAY.json")
        self.assertTrue(all(replay["source_model_identity"][str(s)]["bc_row_parameter_identity_match"] for s in (11, 22, 33)))
        self.assertTrue(all(not replay["round48_ep0_loader_audit"]["per_seed"][str(s)]["registered_file_matches_round46_initializer"] for s in (11, 22, 33)))

    def test_frozen_source_hash_mismatch_blocks_numeric_attribution(self):
        result = build()
        identity = result["identity"]["frozen_trainer"]
        self.assertEqual(identity["frozen_sha256"], EXPECTED_TRAINER)
        self.assertFalse(identity["source_bytes_available"])
        self.assertFalse(result["components"]["numeric_component_gradients_computed"])
        self.assertEqual(result["sensitivity"]["status"], "NOT_COMPUTED_COMPONENT_GRADIENT_UNAVAILABLE")

    def test_gate_records_are_seed_scoped_and_closed(self):
        result = build()
        self.assertEqual(set(result["components"]["batch0_records"]), {"11", "22", "33"})
        for seed, row in result["components"]["batch0_records"].items():
            self.assertEqual(row["gate_status"], "passed")
            self.assertEqual(row["samples"], 265)
            self.assertEqual(row["optimizer_steps_recorded"], 8)
            self.assertTrue(row["recorded_likelihood_alignment"]["passed"])
            self.assertFalse(result["components"]["gate_checks"][seed]["minibatch_indices_persisted"])

    def test_closed_form_shared_trunk_critic_can_move_actor_logit(self):
        # h=w*x; z=v*h; y=u*h; L=.5*(y-target)^2
        x, w, v, u, target = 3.0, 2.0, 4.0, 5.0, 0.0
        h = w * x
        y = u * h
        grad_w = (y - target) * u * x
        grad_v = 0.0  # critic loss has no direct path into the actor-only head
        dlogit_dw = v * x
        descent_directional_change = dlogit_dw * (-grad_w)
        self.assertEqual(grad_w, 450.0)
        self.assertEqual(grad_v, 0.0)
        self.assertEqual(descent_directional_change, -5400.0)
        self.assertTrue(math.isfinite(descent_directional_change))

    def test_round50_zero_side_effect_guard(self):
        result = build()
        self.assertEqual(set(result["components"]["zero_side_effect_counts"].values()), {0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
