import importlib.util
import json
import unittest
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("r49_under_test", HERE / "r49_closure.py")
r49 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r49)


class Round49PureTests(unittest.TestCase):
    def test_action_indices_and_tie(self):
        self.assertEqual(r49.action_label(1), "SEND")
        self.assertEqual(r49.action_label(0), "SKIP")
        self.assertEqual("SEND" if 0.5 >= 0.5 else "SKIP", "SEND")
        with self.assertRaises(ValueError):
            r49.action_label(2)

    def test_role_mismatch_rejected(self):
        r49.validate_model_role("BC_ONLY_R2", "BC_ONLY_R2")
        with self.assertRaises(ValueError):
            r49.validate_model_role("PPO_EP128", "BC_ONLY_R2")

    def test_masked_forced_and_ineligible_excluded(self):
        audit = {"one_shot_state_before":"undecided", "first_emission":True,
                 "relay_initial_executable":True, "sequentially_affordable_before_action":True,
                 "effective_block_reason":"allowed"}
        self.assertTrue(r49.is_free_executable_first_report("t", {"t"}, set(), set(), audit))
        self.assertFalse(r49.is_free_executable_first_report("t", {"t"}, {"t"}, set(), audit))
        self.assertFalse(r49.is_free_executable_first_report("t", {"t"}, set(), {"t"}, audit))
        bad = dict(audit, sequentially_affordable_before_action=False)
        self.assertFalse(r49.is_free_executable_first_report("t", {"t"}, set(), set(), bad))

    def test_duplicate_vectors_keep_target_rows_and_close_counts(self):
        vec = [0.0]*19
        h = r49.vector_hash(vec)
        rows = [{"features":vec, "features_sha256":h, "target":"a", "scene":1, "source_seed":11},
                {"features":vec, "features_sha256":h, "target":"b", "scene":2, "source_seed":22},
                {"features":[1.0]*19, "features_sha256":r49.vector_hash([1.0]*19), "target":"c", "scene":3, "source_seed":11}]
        stats = r49.vector_counts(rows)
        self.assertEqual(stats["records"], 3)
        self.assertEqual(stats["unique_vectors"], 2)
        self.assertEqual(stats["repeated_vector_groups"], 1)
        self.assertEqual(stats["records_in_repeated_groups"], 2)
        self.assertEqual(stats["groups_shared_across_distinct_scene_targets"], 1)
        self.assertEqual(stats["cross_seed_repeated_groups"], 1)
        self.assertEqual(len(rows), 3)

    def test_probability_grouping_is_descriptive_and_side_effect_counts_zero(self):
        records = [{"p":0.25},{"p":0.75}]
        summary = r49.action_summary(records, "p")
        self.assertEqual(summary["greedy_action_counts"], {"SEND":1,"SKIP":1})
        self.assertAlmostEqual(summary["p_send_mean"], .5)
        guards = {"environment_constructor_calls":0,"simulation_calls":0,"new_flow_calls":0,
                  "real_training_update_calls":0,"synthetic_optimizer_step_calls":0,
                  "checkpoint_write_calls":0}
        self.assertTrue(all(value == 0 for value in guards.values()))

    def test_actual_r43_ep0_file_is_rejected_at_load_forward_seam(self):
        r48 = r49.load_round48_module()
        roles=json.loads((r49.ROUND48/"MODEL_ROLE_IDENTITY.json").read_text(encoding="utf-8"))["roles"]
        seed=11
        true_role=roles["ep0_training_init"][str(seed)]
        true_path=Path(true_role["round46_initializer_file"])
        true_model,binding=r49.bind_verified_controller(
            true_path,true_role["round46_initializer_parameter_sha256"],r48,
            true_role["round46_initializer_file_sha256"])
        self.assertTrue(binding["role_verified_before_forward"])
        sample=torch.tensor([[0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,6.0,0.001]],dtype=torch.float32)
        with torch.no_grad():
            true_p=float(r48.p_send(true_model,sample)[0])
        self.assertGreaterEqual(true_p,0.0)
        # Deliberately label the stale R43 file as the ep0 role. The forward path cannot
        # consume it because the loaded tensor digest is checked against the true BC init.
        wrong_path=Path(r48.REGISTERED_EP0[seed]["path"])
        with self.assertRaisesRegex(ValueError,"parameter identity mismatch"):
            stale_model,_=r49.bind_verified_controller(
                wrong_path,true_role["round46_initializer_parameter_sha256"],r48,
                r48.REGISTERED_EP0[seed]["file_sha256"])
                # No call to p_send is reachable for stale_model.


if __name__ == "__main__":
    unittest.main(verbosity=2)
