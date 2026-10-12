import copy
import unittest

import round63_analysis as analysis


def candidate(scene, target, ready, backlog, future=True, budget=1.0, **extra):
    row = {
        "scene": scene,
        "target": target,
        "ready_step": ready,
        "backlog": backlog,
        "has_future_requirements": future,
        "budget_fraction": budget,
        "remaining_time": 40.0,
    }
    row.update(extra)
    return row


class MatchingContractTests(unittest.TestCase):
    def setUp(self):
        self.focus = candidate(90, "focus", 10, 2.0)
        self.rows = [
            candidate(90, "same-scene-exact", 10, 2.0),
            candidate(5, "ready-backlog-exact", 10, 2.0),
            candidate(3, "ready-one-later", 11, 2.0),
            candidate(2, "ready-one-earlier", 9, 2.0),
            candidate(4, "same-ready-other-backlog", 10, 3.0),
        ]

    def test_exact_groups_never_mix_ready_or_backlog(self):
        groups = analysis.group_summary(self.rows, ["ready_step", "backlog"])
        keys = {tuple(g["key"].values()) for g in groups}
        self.assertEqual(keys, {(10, 2.0), (11, 2.0), (9, 2.0), (10, 3.0)})
        exact = next(g for g in groups if g["key"] == {"ready_step": 10, "backlog": 2.0})
        self.assertEqual(exact["scenes"], [5, 90])

    def test_missing_remaining_is_null_and_not_zero(self):
        rows = [candidate(1, "known", 10, 2.0), candidate(2, "missing", 10, 2.0, remaining_time=None)]
        group = analysis.group_summary(rows, ["ready_step", "backlog"])[0]
        self.assertEqual(group["remaining_time_known_count"], 1)
        self.assertEqual(group["remaining_time_missing_count"], 1)
        self.assertEqual(group["remaining_time_min_within_group"], 40.0)
        self.assertEqual(rows[1]["remaining_time"], None)

    def test_matcher_excludes_focus_scene(self):
        ordered = analysis.nearest_covariate_candidates(self.focus, self.rows)
        self.assertNotIn(90, {row["scene"] for row in ordered})
        self.assertEqual(ordered[0]["scene"], 5)

    def test_order_and_ties_are_deterministic(self):
        first = analysis.nearest_covariate_candidates(self.focus, self.rows)
        second = analysis.nearest_covariate_candidates(self.focus, list(reversed(self.rows)))
        keys1 = [(r["scene"], r["target"]) for r in first]
        keys2 = [(r["scene"], r["target"]) for r in second]
        self.assertEqual(keys1, [(5, "ready-backlog-exact"), (4, "same-ready-other-backlog"),
                                 (2, "ready-one-earlier"), (3, "ready-one-later")])
        self.assertEqual(keys1, keys2)

    def test_outcome_mutation_does_not_change_match_output(self):
        baseline = analysis.nearest_covariate_candidates(self.focus, self.rows)
        altered = copy.deepcopy(self.rows)
        for index, row in enumerate(altered):
            row["target_delta"] = 1000 - index
            row["cost_delta_bytes"] = -999999 * (index + 1)
            row["final_task_ids"] = ["outcome-dependent-decoy"]
        after = analysis.nearest_covariate_candidates(self.focus, altered)
        self.assertEqual([(r["scene"], r["target"]) for r in baseline],
                         [(r["scene"], r["target"]) for r in after])


if __name__ == "__main__":
    unittest.main()
