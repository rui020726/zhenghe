import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r59_density_audit as audit


class ZeroEnvironmentCounterexamples(unittest.TestCase):
    def test_discovery_and_completion_are_different(self):
        result = audit.discovery_counts(6, ["a", "b", "c"], ["a", "b"])
        self.assertEqual((result["discovered"], result["on_time"], result["undiscovered_at_window_end"]), (3, 2, 3))
        with self.assertRaises(ValueError):
            audit.discovery_counts(6, ["a"], ["a", "b"])

    def test_candidates_free_rows_and_unique_targets_are_different(self):
        tape = {"targets": ["a", "b", "c"], "free": ["a"], "forced": [], "masked": ["b"],
                "fixed_hold_one_shot_closed": ["c"], "actionable": ["a"], "free_one_shot_choices": ["a"]}
        partitions = audit.partition(tape)
        exposures = ["a", "a"]  # Separate records; not two distinct episode-targets.
        self.assertEqual((len(tape["targets"])*2, len(partitions["free"])*2, len(set(exposures))), (6, 2, 1))
        tape["masked"] = ["a", "b"]
        with self.assertRaises(ValueError):
            audit.partition(tape)

    def test_equal_task_counts_can_exchange_target_ids(self):
        result = audit.id_difference(["a", "c"], ["a", "b"])
        self.assertEqual(result, {"lost": ["b"], "rescued": ["c"], "net_count": 0})

    def test_repeated_scene_and_seed_do_not_create_independent_scenes(self):
        observations = [(11, 101), (11, 101), (22, 101), (33, 102)]
        self.assertEqual(len(observations), 4)
        self.assertEqual(len(set(observations)), 3)
        self.assertEqual(len({scene for seed, scene in observations}), 2)

    def test_ready_recovery_uses_age_and_strict_boundary(self):
        vector = [0.] * 19
        vector[5], vector[18] = 1, .148
        self.assertEqual(audit.gate(vector, 148), (True, 148, True))
        vector[0] = 1
        self.assertEqual(audit.gate(vector, 148), (True, 147, False))

    def test_reward_can_be_credited_to_fixed_row_without_free_action(self):
        bundled, prefix = audit.credit_task_stream({0: 1, 4: 1, 7: 2}, [{"step": 2}, {"step": 5}])
        self.assertEqual((bundled, prefix), ([1., 2.], 1.))


if __name__ == "__main__":
    unittest.main()
