"""Rebuild each audited scenario, certify it against the hash the run logged, and measure
what the placement cells say about reachability, distance and legal observability.

The obstacle grid is not stored by any run, only its hash, so the grid is re-derived from
the same pure function the driver used (``scenarios.build_scenario``) and is accepted only
if it reproduces that hash. A seed whose hash does not match contributes no geometry.
"""

from __future__ import annotations

import json
from typing import Any

import rs_common as rc

rc.enter_repo()
rc.install_env_tripwire()

from integrated_sar.scenarios import build_scenario, has_line_of_sight  # noqa: E402
from integrated_sar.detection_replay import scenario_map_sha256  # noqa: E402

DETECTION_RADIUS = 5  # the run's logged value; asserted per seed below


def _as_lists(cells) -> list[list[int]]:
    """GridPosition tuples to plain lists, so logged and rebuilt placements can be compared."""
    return [[int(cell.row), int(cell.col)] for cell in cells]


def visibility_set(obstacles, cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    """Free cells from which ``cell`` is legally observable (Chebyshev <= 5 and clear line).

    The cell itself counts: a robot standing on a target satisfies the recorded rule.
    """
    out = []
    for row in range(max(0, cell[0] - DETECTION_RADIUS), min(rows, cell[0] + DETECTION_RADIUS + 1)):
        for col in range(max(0, cell[1] - DETECTION_RADIUS), min(cols, cell[1] + DETECTION_RADIUS + 1)):
            if int(obstacles[row][col]) == 1:
                continue
            if (row, col) == cell or has_line_of_sight(obstacles, (row, col), cell):
                out.append((row, col))
    return out


def cross_check_line_predicate(maps: list[Any], samples: int = 6) -> dict[str, int]:
    """``scenarios.has_line_of_sight`` must agree with PILOC's own rule, cell pair by pair."""
    compared = disagreements = 0
    for grid in maps[:samples]:
        rows, cols = grid.shape
        for row in range(0, rows, 7):
            for col in range(0, cols, 7):
                if int(grid[row][col]) == 1:
                    continue
                for other_row in range(max(0, row - 5), min(rows, row + 6)):
                    for other_col in range(max(0, col - 5), min(cols, col + 6)):
                        if int(grid[other_row][other_col]) == 1:
                            continue
                        compared += 1
                        if bool(has_line_of_sight(grid, (row, col), (other_row, other_col))) \
                                != rc.piloc_line_of_sight(grid, (row, col), (other_row, other_col)):
                            disagreements += 1
    return {"pairs_compared": compared, "disagreements": disagreements}


def measure_seed(configuration: dict, scenario_logged: dict, placements: list[tuple[int, int]],
                 agents: list[tuple[int, int]], anchors: list[tuple[int, int]],
                 missed_ids: dict[int, str]) -> dict[str, Any]:
    rebuilt = build_scenario(
        map_path=None,
        size=int(configuration["grid_size"]),
        seed=int(configuration["seed"]),
        num_agents=int(configuration["search_agents"]),
        num_goals=int(configuration["num_goals"]),
        obstacle_density=float(configuration["obstacle_density"]),
    )
    logged_hash = str(scenario_logged.get("map_sha256"))
    rebuilt_hash = scenario_map_sha256(rebuilt)
    obstacles = rebuilt.obstacles
    rows, cols = obstacles.shape
    record: dict[str, Any] = {
        "certified": rebuilt_hash == logged_hash,
        "logged_map_sha256": logged_hash, "rebuilt_map_sha256": rebuilt_hash,
        "grid": [int(rows), int(cols)],
        "logged_obstacle_cells": int(scenario_logged.get("obstacle_cells", -1)),
        "rebuilt_obstacle_cells": int((obstacles == 1).sum()),
        "logged_free_cells": int(scenario_logged.get("free_cells", -1)),
        "rebuilt_free_cells": int((obstacles != 1).sum()),
        "logged_placements_match_rebuilt": _as_lists(rebuilt.goal_positions) == [
            list(cell) for cell in placements],
        "logged_agent_starts_match_rebuilt": _as_lists(rebuilt.agent_positions) == [
            list(cell) for cell in agents],
        "detection_radius_logged": int(configuration.get("detection_radius", DETECTION_RADIUS)),
    }
    if not record["certified"]:
        return record

    comps4 = rc.components(obstacles, eight=False)
    comps8 = rc.components(obstacles, eight=True)
    index4 = rc.component_index_map(comps4)
    index8 = rc.component_index_map(comps8)
    record["component_sizes_4"] = [len(block) for block in comps4[:5]]
    record["component_sizes_8"] = [len(block) for block in comps8[:5]]
    agent_components_8 = sorted({index8.get(cell, -1) for cell in agents})
    record["agent_components_8"] = agent_components_8
    record["agents_reachable_set_4"] = None

    geodesic: dict[tuple[int, int], dict[tuple[int, int], Any]] = {}
    for start in agents:
        geodesic[start] = rc.geodesic_distance(obstacles, start)

    per_goal: list[dict[str, Any]] = []
    for index, cell in enumerate(placements):
        visible = visibility_set(obstacles, cell, rows, cols)
        row = {
            "goal_index": index,
            "target_id": rc.goal_label(index),
            "placed_at": list(cell),
            "missed": index in missed_ids,
            "component_4": index4.get(cell),
            "component_8": index8.get(cell),
            "same_8_component_as_an_agent": index8.get(cell) in agent_components_8,
            "on_obstacle": int(obstacles[cell[0]][cell[1]]) == 1,
            "chebyshev_to_nearest_agent_start": min(rc.chebyshev(cell, a) for a in agents),
            "geodesic_to_nearest_agent_start": min(
                (geodesic[start].get(cell) for start in agents if geodesic[start].get(cell) is not None),
                default=None),
            "visibility_cells_at_placement": len(visible),
            "geodesic_start_to_visibility_set": min(
                (value for start in agents for v in visible
                 if (value := geodesic[start].get(v)) is not None), default=None),
            "in_legal_range_of_a_start_at_reset": any(
                rc.chebyshev(cell, a) <= DETECTION_RADIUS
                and has_line_of_sight(obstacles, a, cell) for a in agents),
            "chebyshev_to_nearest_reached_cell": (
                min(rc.chebyshev(cell, anchor) for anchor in anchors) if anchors else None),
        }
        per_goal.append(row)
    for goal in per_goal:
        if not goal["missed"] or not anchors:
            continue
        # a walkable-path distance for the cases this round is actually about
        from_placement = rc.geodesic_distance(obstacles, tuple(goal["placed_at"]))
        reachable = [from_placement.get(anchor) for anchor in anchors]
        goal["geodesic_to_nearest_reached_cell"] = min(
            (value for value in reachable if value is not None), default=None)
        goal["anchors_unreachable_from_placement"] = sum(
            1 for value in reachable if value is None)
    record["goals"] = per_goal
    record["all_goals_in_one_8_component_with_agents"] = all(
        goal["same_8_component_as_an_agent"] and not goal["on_obstacle"] for goal in per_goal)
    return record


def run() -> int:
    reconciliation = json.loads((rc.ROUND_DIR / "BATCH_RECONCILIATION.json").read_text(
        encoding="utf-8"))
    manifest: dict[str, Any] = {}
    out: dict[str, Any] = {"round": rc.ROUND, "start_sha": rc.START_SHA,
                           "reading": (
                               "positions below are simulation truth recorded for the audit; the "
                               "search policy at step t only had its own observation, so every "
                               "distance here is a post-hoc statement about the scenario, not a "
                               "statement about what the policy could see"),
                           "batches": {}}

    for key, batch in rc.BATCHES.items():
        arm = batch["search_record_arm"]
        by_seed: dict[int, list[dict[str, Any]]] = {}
        for row in reconciliation["batches"][key]["undetected_goals"]:
            by_seed.setdefault(int(row["seed"]), []).append(row)
        seeds: dict[str, Any] = {}
        grids = []
        for seed in batch["seeds"]:
            missed_ids = {int(row["goal_index"]): str(row["target_id"])
                          for row in by_seed.get(seed, [])}
            paths = rc.batch_paths(batch, arm, seed)
            records = list(rc.iter_jsonl(paths["lifecycle"]))
            configuration: dict[str, Any] = {}
            logged_scenario: dict[str, Any] = {}
            placements: list[tuple[int, int]] = []
            agents: list[tuple[int, int]] = []
            anchors: list[tuple[int, int]] = []
            for record in records:
                if record.get("event_type") == "run_config":
                    detail = record.get("detail") or {}
                    configuration = dict(detail.get("configuration") or {})
                    logged_scenario = dict(detail.get("scenario") or {})
                    placements = [tuple(cell) for cell in logged_scenario.get("goal_positions") or []]
                    agents = [tuple(cell) for cell in logged_scenario.get("agent_positions") or []]
                elif record.get("event_type") == "detection":
                    anchors.append(tuple((record.get("detail") or {})["grid_position"]))
            grids.append(build_scenario(map_path=None, size=int(configuration["grid_size"]),
                                        seed=int(seed), num_agents=int(configuration["search_agents"]),
                                        num_goals=int(configuration["num_goals"]),
                                        obstacle_density=float(configuration["obstacle_density"])
                                        ).obstacles)
            seeds[str(seed)] = measure_seed(configuration, logged_scenario, placements, agents,
                                           anchors, missed_ids)
            rc.register_input(paths["lifecycle"], manifest)

        certified = [row for row in seeds.values() if row["certified"]]
        goals = [goal for row in certified for goal in row.get("goals", [])]
        missed = [goal for goal in goals if goal["missed"]]
        found = [goal for goal in goals if not goal["missed"]]

        def summary(rows: list[dict]) -> dict[str, Any]:
            return {
                "n": len(rows),
                "chebyshev_to_start": rc.quantiles([r["chebyshev_to_nearest_agent_start"]
                                                    for r in rows]),
                "geodesic_to_start": rc.quantiles(
                    [r["geodesic_to_nearest_agent_start"] for r in rows
                     if r["geodesic_to_nearest_agent_start"] is not None]),
                "visibility_cells": rc.quantiles([r["visibility_cells_at_placement"] for r in rows]),
                "start_to_visibility_set": rc.quantiles(
                    [r["geodesic_start_to_visibility_set"] for r in rows
                     if r["geodesic_start_to_visibility_set"] is not None]),
                "chebyshev_to_nearest_reached_cell": rc.quantiles(
                    [r["chebyshev_to_nearest_reached_cell"] for r in rows
                     if r["chebyshev_to_nearest_reached_cell"] is not None]),
                "in_legal_range_of_a_start_at_reset": sum(
                    r["in_legal_range_of_a_start_at_reset"] for r in rows),
                "not_in_any_agent_component": sum(not r["same_8_component_as_an_agent"] for r in rows),
                "standing_on_an_obstacle": sum(r["on_obstacle"] for r in rows),
            }

        # the found-goal detail is a control, not a deliverable: its aggregates stay, the
        # 708 rows are dropped so the artefact keeps the 12 that this round is about
        for row in seeds.values():
            if "goals" in row:
                row["found_goal_rows_dropped"] = sum(1 for goal in row["goals"]
                                                     if not goal["missed"])
                row["goals"] = [goal for goal in row["goals"] if goal["missed"]]

        out["batches"][key] = {
            "seeds_total": len(seeds), "seeds_certified_by_map_hash": len(certified),
            "uncertified_seeds": [seed for seed, row in seeds.items() if not row["certified"]],
            "logged_and_rebuilt_placements_agree": all(
                row["logged_placements_match_rebuilt"] for row in certified),
            "logged_and_rebuilt_agent_starts_agree": all(
                row["logged_agent_starts_match_rebuilt"] for row in certified),
            "obstacle_cell_counts_agree": all(row["logged_obstacle_cells"] == row["rebuilt_obstacle_cells"]
                                              for row in certified),
            "every_goal_shares_agent_8_component": all(
                row["all_goals_in_one_8_component_with_agents"] for row in certified),
            "line_of_sight_predicate_cross_check": cross_check_line_predicate(grids),
            "missed_goals": summary(missed), "found_goals": summary(found),
            "per_seed": seeds,
        }

    out["inputs_read"] = {"count": len(manifest), "files": manifest}
    path = rc.write_json("SCENARIO_GEOMETRY.json", out)
    for key, row in out["batches"].items():
        print(f"== {key}: certified {row['seeds_certified_by_map_hash']}/{row['seeds_total']} seeds")
        print("   placements agree:", row["logged_and_rebuilt_placements_agree"],
              "| agent starts agree:", row["logged_and_rebuilt_agent_starts_agree"],
              "| obstacle counts agree:", row["obstacle_cell_counts_agree"])
        print("   every goal shares an 8-connected component with an agent start:",
              row["every_goal_shares_agent_8_component"])
        print("   line-of-sight predicate cross-check:", row["line_of_sight_predicate_cross_check"])
        for label in ("missed_goals", "found_goals"):
            s = row[label]
            print(f"   {label}: n={s['n']} cheb_to_start={s['chebyshev_to_start'].get('median')} "
                  f"geodesic_to_start={s['geodesic_to_start'].get('median')} "
                  f"visibility_cells={s['visibility_cells'].get('median')} "
                  f"start_to_vis_set={s['start_to_visibility_set'].get('median')} "
                  f"in_range_at_reset={s['in_legal_range_of_a_start_at_reset']} "
                  f"other_component={s['not_in_any_agent_component']} "
                  f"on_obstacle={s['standing_on_an_obstacle']}")
    print(f"wrote {path.name}; inputs hashed: {len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
