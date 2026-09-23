"""0-episode unit tests for the undetected-goal attribution harness.

Nothing here runs an environment: the geometry tests work on grids rebuilt by the same pure
scenario function the driver uses, and the data tests read the artefacts this round wrote.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import rs_common as rc

rc.enter_repo()

from integrated_sar.detection_replay import scenario_map_sha256  # noqa: E402
from integrated_sar.scenarios import build_scenario, has_line_of_sight  # noqa: E402

import rs_attribute  # noqa: E402
import rs_geometry  # noqa: E402

import numpy as np  # noqa: E402

ARTIFACTS = ("BATCH_RECONCILIATION.json", "FIELD_INVENTORY.json", "SCENARIO_GEOMETRY.json",
             "UNDETECTED_GOAL_ATTRIBUTION.json", "ATTRIBUTION_VERIFICATION.json")


def load(name: str) -> dict:
    return json.loads((rc.ROUND_DIR / name).read_text(encoding="utf-8"))


class GridGeometry(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = build_scenario(map_path=None, size=60, seed=1100000, num_agents=2,
                                       num_goals=6, obstacle_density=0.15)

    def test_a_rebuilt_scenario_reproduces_the_hash_the_run_logged(self) -> None:
        logged = load("SCENARIO_GEOMETRY.json")["batches"]["B1_d1_20260923"]["per_seed"]["1100000"]
        self.assertTrue(logged["certified"])
        self.assertEqual(scenario_map_sha256(self.scenario), logged["logged_map_sha256"])

    def test_line_of_sight_helpers_agree_with_piloc_s_own_rule(self) -> None:
        grid = self.scenario.obstacles
        rows, cols = grid.shape
        disagreements = 0
        checked = 0
        for row in range(0, rows, 3):
            for col in range(0, cols, 3):
                if int(grid[row][col]) == 1:
                    continue
                for other in ((row + 1, col + 1), (row + 2, col - 1), (row - 3, col + 2)):
                    if not (0 <= other[0] < rows and 0 <= other[1] < cols):
                        continue
                    if int(grid[other[0]][other[1]]) == 1:
                        continue
                    checked += 1
                    if bool(has_line_of_sight(grid, (row, col), other)) != \
                            rc.piloc_line_of_sight(grid, (row, col), other):
                        disagreements += 1
        self.assertGreater(checked, 50)
        self.assertEqual(disagreements, 0)

    def test_every_legal_observer_of_a_goal_shares_its_eight_component(self) -> None:
        """The reachability lemma this round leans on, tested on certified real terrain."""
        grid = self.scenario.obstacles
        rows, cols = grid.shape
        index = rc.component_index_map(rc.components(grid, eight=True))
        for cell in rs_geometry._as_lists(self.scenario.goal_positions):
            cell = (int(cell[0]), int(cell[1]))
            visible = rs_geometry.visibility_set(grid, cell, rows, cols)
            self.assertGreaterEqual(len(visible), 1)
            self.assertIn(cell, visible)
            self.assertTrue({index.get(probe) for probe in visible} == {index.get(cell)},
                            f"an observer in another region could see {cell}")

    def test_a_diagonal_pinch_is_one_eight_region_but_two_four_regions(self) -> None:
        """The distinction the reachability lemma needs: legality follows 8-adjacency,
        movement follows 4-adjacency, and a diagonal touch separates them."""
        grid = np.zeros((4, 4), dtype=np.int32)
        grid[0][1] = 1
        grid[1][0] = 1
        four = rc.components(grid, eight=False)
        eight = rc.components(grid, eight=True)
        self.assertEqual(len(four), len(eight) + 1)
        index4 = rc.component_index_map(four)
        index8 = rc.component_index_map(eight)
        self.assertNotEqual(index4.get((0, 0)), index4.get((1, 1)))
        self.assertEqual(index8.get((0, 0)), index8.get((1, 1)))
        distances = rc.geodesic_distance(grid, (0, 0))
        self.assertIsNone(distances.get((1, 1)))          # no walkable path
        self.assertIsNotNone(distances.get((0, 0)))
        self.assertEqual(rc.geodesic_distance(grid, (2, 2)).get((2, 3)), 1)

    def test_a_walled_region_makes_the_far_side_unreachable_and_unobservable(self) -> None:
        grid = np.zeros((12, 12), dtype=np.int32)
        grid[:, 6] = 1                                     # full wall, no gap at all
        index4 = rc.component_index_map(rc.components(grid, eight=False))
        self.assertNotEqual(index4.get((5, 2)), index4.get((5, 9)))
        distances = rc.geodesic_distance(grid, (5, 2))
        self.assertIsNone(distances.get((5, 9)))
        visible = rs_geometry.visibility_set(grid, (5, 9), 12, 12)
        self.assertTrue(all(index4.get(cell) == index4.get((5, 9)) for cell in visible),
                        "an observer from the other side of the wall would be legal")

    def test_chebyshev_and_bresenham_edge_cases(self) -> None:
        self.assertEqual(rc.chebyshev((3, 4), (0, 0)), 4)
        self.assertEqual(rc.bresenham((2, 2), (2, 2)), [(2, 2)])
        self.assertEqual(rc.bresenham((0, 0), (0, 2)), [(0, 0), (0, 1), (0, 2)])


class Classifiers(unittest.TestCase):
    def goal(self, **overrides):
        base = {"goal_index": 0, "target_id": "piloc-e1-target-0", "placed_at": [5, 5],
                "missed": True, "component_4": 0, "component_8": 0,
                "same_8_component_as_an_agent": True, "on_obstacle": False,
                "chebyshev_to_nearest_agent_start": 6, "geodesic_to_nearest_agent_start": 8,
                "visibility_cells_at_placement": 60, "geodesic_start_to_visibility_set": 4,
                "in_legal_range_of_a_start_at_reset": False,
                "chebyshev_to_nearest_reached_cell": 6}
        base.update(overrides)
        return base

    def test_a_goal_outside_the_agents_region_is_a_scene_problem(self) -> None:
        verdict = rs_attribute.classify(self.goal(same_8_component_as_an_agent=False), True)
        self.assertEqual(verdict["bucket"], "unreachable_or_scene_problem")
        self.assertEqual(rs_attribute.classify(self.goal(on_obstacle=True), True)["bucket"],
                         "unreachable_or_scene_problem")

    def test_start_in_range_is_a_candidate_not_a_verdict(self) -> None:
        verdict = rs_attribute.classify(self.goal(in_legal_range_of_a_start_at_reset=True), True)
        self.assertEqual(verdict["bucket"], "insufficient_logs")
        self.assertNotIn("entered_range_without_event", json.dumps(verdict))

    def test_the_default_reading_stays_conditional_on_the_detector(self) -> None:
        verdict = rs_attribute.classify(self.goal(), True)
        self.assertEqual(verdict["bucket"], "reachable_but_not_covered")
        self.assertIn("conditional", verdict["grade"])
        self.assertIn("detector", verdict["why"])

    def test_bucket_labels_are_exactly_the_four_asked_for(self) -> None:
        self.assertEqual(set(rs_attribute.BUCKETS), {
            "unreachable_or_scene_problem", "entered_range_without_event",
            "reachable_but_not_covered", "insufficient_logs"})


class Harness(unittest.TestCase):
    def test_cell_names_are_unique_across_batch_arm_and_seed(self) -> None:
        seen = set()
        for batch in rc.BATCHES.values():
            for arm in batch["arms"]:
                for seed in batch["seeds"]:
                    for path in rc.batch_paths(batch, arm, seed).values():
                        self.assertNotIn(path, seen, f"{path.name} reused")
                        seen.add(path)
        self.assertEqual(len(seen), 4 * 60 * 2 + 2 * 60 * 1)

    def test_write_json_stays_inside_the_round_and_refuses_silent_rewrites(self) -> None:
        payload = {"probe": 1}
        path = rc.write_json("_test_probe.json", payload)
        try:
            self.assertTrue(path.exists())
            with self.assertRaises(RuntimeError):
                rc.write_json("_test_probe.json", {"probe": 2})
            with self.assertRaises(RuntimeError):
                rc.write_json("../outside.json", payload)
        finally:
            path.unlink(missing_ok=True)

    def test_the_analysis_never_imports_a_backend(self) -> None:
        self.assertEqual(rc.forbidden_imports_seen(), [])

    def test_goal_labels_use_the_adapters_episode_scoped_form(self) -> None:
        self.assertEqual(rc.goal_label(3), "piloc-e1-target-3")

    def test_iter_jsonl_refuses_a_truncated_record(self) -> None:
        temporary = rc.ROUND_DIR / "_test_truncate.jsonl"
        temporary.write_text('{"a": 1}\n{"a": ', encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                list(rc.iter_jsonl(temporary))
        finally:
            temporary.unlink(missing_ok=True)


class DeliveredArtefacts(unittest.TestCase):
    @unittest.skipIf(not (rc.ROUND_DIR / "ATTRIBUTION_VERIFICATION.json").exists(),
                     "the verification artefact is written after the analysis")
    def test_verification_passed_and_nothing_stray_was_touched(self) -> None:
        checks = load("ATTRIBUTION_VERIFICATION.json")
        self.assertTrue(checks["passed"])
        self.assertEqual(checks["scope"]["protected_paths_touched"], [])
        self.assertEqual(checks["scope"]["d1_frozen_artefacts_changed"], [])
        self.assertEqual(checks["scope"]["outside_the_allowed_set"], [])
        self.assertEqual(checks["inputs_unchanged"]["changed"], [])
        self.assertEqual(checks["sealed"]["sealed_seed_count_seen"], 0)

    def test_targets_move_and_the_mobility_bound_holds(self) -> None:
        mobility = load("BATCH_RECONCILIATION.json")["target_mobility"]
        self.assertGreater(mobility["detections"], mobility["cell_equals_placement"])
        self.assertEqual(mobility["drift_over_step_violations"], [])
        self.assertGreater(mobility["drift_quantiles"]["late_gt_10_steps"]["max"], 10)

    def test_the_two_batches_keep_their_own_denominators(self) -> None:
        attribution = load("UNDETECTED_GOAL_ATTRIBUTION.json")
        self.assertEqual(attribution["batches"]["B1_d1_20260923"]["denominators"]["placed"], 360)
        self.assertEqual(attribution["batches"]["B1_d1_20260923"]["denominators"]["undetected"], 8)
        self.assertEqual(attribution["batches"]["B2_systemloop_20260918"]["denominators"]["detected"],
                         356)
        self.assertEqual(attribution["batches"]["B2_systemloop_20260918"]["denominators"]["undetected"],
                         4)
        self.assertEqual(len(attribution["rows"]), 12)
        for key, row in attribution["batches"].items():
            self.assertTrue(row["remaining_search_targets_sum_matches_misses"], key)
            self.assertEqual(sum(row["bucket_counts"].values()),
                             row["denominators"]["undetected"], key)

    def test_no_required_position_field_exists_in_the_archived_logs(self) -> None:
        inventory = load("FIELD_INVENTORY.json")
        for name, field in inventory["wanted_fields"].items():
            self.assertFalse(field["present_in_the_archived_logs"], name)
        self.assertEqual(
            inventory["batches"]["B1_d1_20260923"]["files_scanned"]
            + inventory["batches"]["B2_systemloop_20260918"]["files_scanned"], 360)


if __name__ == "__main__":
    unittest.main(verbosity=2)
