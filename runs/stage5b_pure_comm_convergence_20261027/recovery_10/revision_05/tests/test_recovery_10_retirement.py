import unittest

from stage5b_comm import recovery_descendant as D

ROSTER = (("T1", 91001), ("T1", 91002), ("T1", 91003),
          ("T2", 91001), ("T2", 91002), ("T2", 91003))


def make_runs():
    return [{"arm": arm, "training_seed": seed} for arm, seed in ROSTER]


def make_states(completed):
    states = {}
    for key in ROSTER:
        cursor = completed[key]
        states[key] = {
            "cursor": cursor, "next_update": cursor // 4, "certified": False,
            "loss_stability_stable": False, "completed_episode_count": cursor,
            "registered_missing_episode_count": 0,
            "donor": {"path": "donor.pth", "sha256": "0" * 64, "bytes": 1},
        }
    return states


class RetirementTests(unittest.TestCase):
    def test_long_target_without_retirement_continues_every_run(self):
        runs = D.next_runs(make_runs(), make_states({k: 5000 for k in ROSTER}),
                           include_completion_counts=True, include_minimum_target=True,
                           target_completed_episodes=10000)
        self.assertEqual([r["new_world_cap"] for r in runs], [5000] * 6)
        self.assertTrue(all(r["active"] for r in runs))
        self.assertTrue(all("revision_retired" not in r for r in runs))

    def test_retired_runs_stop_at_parent_target_while_others_continue(self):
        retired = frozenset({("T1", 91001), ("T1", 91003), ("T2", 91001), ("T2", 91003)})
        runs = D.next_runs(make_runs(), make_states({k: 5000 for k in ROSTER}),
                           include_completion_counts=True, include_minimum_target=True,
                           target_completed_episodes=10000, retired_runs=retired)
        by_key = {(r["arm"], r["training_seed"]): r for r in runs}
        for key in retired:
            run = by_key[key]
            self.assertEqual(run["new_world_cap"], 0)
            self.assertFalse(run["active"])
            self.assertIs(run.get("revision_retired"), True)
            self.assertEqual(run["end_episode_cap"], run["start_episode"])
        for key in (("T1", 91002), ("T2", 91002)):
            run = by_key[key]
            self.assertEqual(run["new_world_cap"], 5000)
            self.assertTrue(run["active"])
            self.assertNotIn("revision_retired", run)

    def test_retirement_is_ignored_outside_long_target_mode(self):
        runs = D.next_runs(make_runs(), make_states({k: 5000 for k in ROSTER}),
                           retired_runs=frozenset({("T1", 91001)}))
        self.assertEqual([r["new_world_cap"] for r in runs], [120] * 6)
        self.assertTrue(all(r["active"] for r in runs))


if __name__ == "__main__":
    unittest.main()
