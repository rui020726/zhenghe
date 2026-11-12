"""The crossed G x R x S grid: deep goal counts against rescue rosters and search-agent counts.

This is the first sweep that CROSSES the three axes instead of moving one at a time. The earlier
rounds answered "what happens as G grows" (R,S fixed), "as R grows" (G,S fixed) and "as S grows"
(G,R fixed); this one asks what the *combination* does at the deep end, where G exceeds 100.

Grid (N held at 6 so the comparison is not confounded by fan-out):

    G in {72, 84, 96, 108, 120, 144}      six goal counts, four of them new (> 72)
    R in {6, 12, 18}                      scarce / trained / abundant rescue capacity
    S in {1, 2, 4}                        few / trained / many search agents

= 54 (G,R,S) points x 3 anchor scenes = 162 worlds per arm. ``G=72`` at ``R in {6,12}`` with
``S=2`` is already on disk from the G sweep, so 156 are new per arm.

Arms (six): ``normal_immediate`` and ``slack_heuristic_cap`` (rule), ``slack_matched_rate``
(matched to v4t's own page count in the same G,R at S=2), ``v4f@200`` / ``v4s@300`` /
``v4t@100`` (sealed checkpoints). ``v4t`` and ``v4s`` take a page budget of
``normal_immediate's pages per episode at the same (G, R), S=2, minus one``, held fixed across S
so an S difference is the environment's rather than a re-derived quota's.

    D:/project_env/python.exe analysis/g_rs_grid.py --stage immediate --workers 8
    D:/project_env/python.exe analysis/g_rs_grid.py --stage rules --workers 8
    D:/project_env/python.exe analysis/g_rs_grid.py --stage v4f --workers 8
    D:/project_env/python.exe analysis/g_rs_grid.py --stage v4s --workers 8
    D:/project_env/python.exe analysis/g_rs_grid.py --stage v4t --workers 8
    D:/project_env/python.exe analysis/g_rs_grid.py --stage slack_matched --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
STAGE = BASE.parents[1]
OUT = BASE / "execution/g_rs_grid"

ANCHOR = (2301901, 2301902, 2301903)
GRID_GOALS = (72, 84, 96, 108, 120, 144)
GRID_ROBOTS = (6, 12, 18)
GRID_AGENTS = (1, 2, 4)
SUBSCRIBERS = 6

LEARNED = {
    "v4f@200": {"tag": "train_v4f_anchor", "episode": 200},
    "v4s@300": {"tag": "train_v4s_cap300", "episode": 300},
    "v4t@100": {"tag": "train_v4t_cap_frozen", "episode": 100},
}
RULE_WORLD_ARMS = ("normal_immediate", "slack_heuristic_cap", "slack_matched_rate")
ALL_ARMS = ("normal_immediate", "slack_heuristic_cap", "slack_matched_rate",
            "v4f@200", "v4s@300", "v4t@100")


def grid_points():
    """``[((G, R, 6), S)]`` in a fixed order, so a report can key on the position."""
    return [((goal, robots, SUBSCRIBERS), agents)
            for goal in GRID_GOALS for robots in GRID_ROBOTS for agents in GRID_AGENTS]


def cell_label(cell, search_agents=None):
    label = "G%d_R%d_N%d" % (int(cell[0]), int(cell[1]), int(cell[2]))
    if search_agents is not None and int(search_agents) != 2:
        label += "_S%d" % int(search_agents)
    return label


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(str(tmp), str(path))


def arm_root(arm):
    return OUT / arm.replace("@", "_")


def arm_path(arm):
    return arm_root(arm) / "RESULTS.json"


#: Archived worlds this grid may reuse. Each source is (file, arm filter); a row is reusable
#: when its (cell, scene, S) is inside this grid.
ARCHIVE_SOURCES = {
    "normal_immediate": (("execution/g_scaling/normal_immediate/RESULTS.json", None),),
    "slack_heuristic_cap": (("execution/g_scaling/slack_heuristic_cap/RESULTS.json", None),),
    "slack_matched_rate": (("execution/g_scaling/slack_matched_rate/RESULTS.json", None),),
    "v4f@200": (("execution/g_scaling/v4f_200/RESULTS.json", None),),
    "v4s@300": (("execution/g_scaling/v4s_300/RESULTS.json", None),),
    "v4t@100": (("execution/g_scaling/v4t_100/RESULTS.json", None),),
}


def archived_rows(arm):
    rows = []
    for relative, arm_filter in ARCHIVE_SOURCES.get(arm, ()):
        path = BASE / relative
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("results") or []:
            if row.get("status") != "ok":
                continue
            if arm_filter is not None and row.get("arm") != arm_filter:
                continue
            rows.append(row)
    return rows


def archived_keys(arm):
    wanted = {(cell, agents) for cell, agents in grid_points()}
    keys = set()
    for row in archived_rows(arm):
        cell = tuple(int(v) for v in row["cell"])
        scene = int(row["scene"])
        agents = int(row.get("search_agents") or 2)
        if (cell, agents) in wanted and scene in ANCHOR:
            keys.add((cell, scene, agents))
    return keys


def pages_from_rows(rows):
    totals = {}
    for row in rows:
        key = "G%d_R%d_N%d" % (int(row["cell"][0]), int(row["cell"][1]), int(row["cell"][2]))
        entry = totals.setdefault(key, [0, 0])
        entry[0] += int(row["emissions"])
        entry[1] += 1
    return {key: int(round(pages / worlds)) for key, (pages, worlds) in totals.items()}


def immediate_pages():
    """``{cell_label: pages per episode}`` at S=2, archives plus this grid's own immediate rows."""
    rows = [row for row in archived_rows("normal_immediate")
            if int(row["scene"]) in ANCHOR and int(row.get("search_agents") or 2) == 2]
    path = arm_path("normal_immediate")
    if path.exists():
        rows += [row for row in json.loads(path.read_text(encoding="utf-8")).get("results") or []
                 if row.get("status") == "ok" and int(row.get("search_agents") or 2) == 2]
    return pages_from_rows(rows)


def v4t_pages():
    """``{cell_label: v4t pages per episode}`` **at S=2 only**.

    The matched arm has to be matched to a depth measured in the same conditions it runs in;
    averaging in the S=1/S=4 worlds would give it a budget that belongs to neither.
    """
    rows = [row for row in archived_rows("v4t@100")
            if int(row.get("search_agents") or 2) == 2]
    path = arm_path("v4t@100")
    if path.exists():
        rows += [row for row in
                 json.loads(path.read_text(encoding="utf-8")).get("results") or []
                 if row.get("status") == "ok"
                 and int(row.get("search_agents") or 2) == 2]
    return pages_from_rows(rows)


def worker(task):
    sys.dont_write_bytecode = True
    result = {"arm": task["arm"], "cell": task["cell"], "scene": int(task["scene"]),
              "search_agents": task.get("search_agents"),
              "cell_label": task["cell_label"]}
    started = time.time()
    try:
        root, run_dir = Path(task["root"]), Path(task["run_dir"])
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from stage5b_comm import episode, trainer
        from stage5b_comm.paths import Layout

        layout = Layout(root)
        layout.execution = run_dir
        layout.world_logs = run_dir / "world_logs"
        layout.rows = run_dir / "rows"
        layout.ledger = run_dir / "ledger"
        layout.receipts = run_dir / "receipts"
        for path in (layout.world_logs, layout.rows, layout.ledger, layout.receipts):
            path.mkdir(parents=True, exist_ok=True)
        episode.ensure_import_paths(layout)
        episode.native_import_preflight(layout)

        cell = tuple(int(value) for value in task["cell"])
        agents = task.get("search_agents")
        if task["kind"] == "rule":
            row = episode.run_rule_world(
                layout, task["arm"], int(task["scene"]), phase="eval", cell=cell,
                immediate_pages=task.get("immediate_pages"),
                keep_probability=task.get("keep_probability"),
                matched_pages_by_cell=task.get("matched_pages"),
                search_agents=agents)
        else:
            controller, _opt, _state = trainer.load_checkpoint(
                Path(task["checkpoint"]), arm="T2", identity_sha256=task["identity_sha"],
                expected_seed=91002, expected_episode=int(task["checkpoint_episode"]))
            row = episode.run_eval_world(
                layout, "T2", 91002, int(task["scene"]), controller, cell=cell,
                episode_index=int(task["checkpoint_episode"]),
                cap=task.get("cap"), floor=None, search_agents=agents)
        sends = list(row.get("emission_sequence") or [])
        distinct = {str(entry[1]) for entry in sends}
        result.update({
            "status": "ok",
            "on_time": int(row["on_time"]),
            "generated_goals": int(row.get("generated_goals") or 0),
            "emissions": int(row["emissions"]),
            "distinct_targets": len(distinct),
            "duplicate_sends": int(row["emissions"]) - len(distinct),
            "spent_bytes": int(row["spent_bytes"]),
            "subscribers_live": row.get("subscribers_live"),
            "rescue_robots": row.get("rescue_robots"),
            "search_agents_live": agents,
            "cap": task.get("cap"),
            "rule_parameters": row.get("rule_parameters"),
            "budget_restored": row.get("budget_authority_restored"),
        })
    except Exception as error:  # a failed world is reported, never counted as a result
        result["status"] = "error"
        result["error"] = "%s: %s" % (type(error).__name__, error)
        result["traceback"] = traceback.format_exc()[-2500:]
    result["elapsed_seconds"] = round(time.time() - started, 2)
    return result


def merge(arm, rows):
    path = arm_path(arm)
    merged = {}
    if path.exists():
        for row in json.loads(path.read_text(encoding="utf-8")).get("results") or []:
            if row.get("status") == "ok":
                merged[(tuple(row["cell"]), int(row["scene"]),
                        int(row.get("search_agents") or 2))] = row
    for row in rows:
        if row.get("status") == "ok":
            merged[(tuple(row["cell"]), int(row["scene"]),
                    int(row.get("search_agents") or 2))] = row
    failures = [row for row in rows if row.get("status") != "ok"]
    dump(path, {
        "schema": "stage5b-g-rs-grid-arm/1",
        "arm": arm,
        "root": str(BASE),
        "goals": list(GRID_GOALS),
        "robots": list(GRID_ROBOTS),
        "search_agents": list(GRID_AGENTS),
        "subscribers": SUBSCRIBERS,
        "scenes": list(ANCHOR),
        "worlds": len(merged),
        "failures": failures,
        "results": sorted(merged.values(),
                          key=lambda row: (tuple(row["cell"]), int(row["scene"]),
                                           int(row.get("search_agents") or 2))),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return len(merged), len(failures)


def build_tasks(arm):
    identity = sha256(STAGE / "recovery_09b" / "revision_01" / "protocol"
                      / "RECOVERY_EXPERIMENT_IDENTITY.json")
    run_dir = arm_root(arm)
    pages = {}
    matched = {}
    if arm in ("v4t@100", "v4s@300", "slack_heuristic_cap", "slack_matched_rate"):
        pages = immediate_pages()
    if arm == "slack_matched_rate":
        matched = v4t_pages()
    tasks = []
    for cell, agents in grid_points():
        label = "G%d_R%d_N%d" % cell
        base_label = label
        if agents != 2:
            label = cell_label(cell, agents)
        cap = None
        if arm in ("v4t@100", "v4s@300"):
            if base_label not in pages:
                raise SystemExit("%s needs immediate pages for %s (run --stage immediate)"
                                 % (arm, base_label))
            cap = max(1, int(pages[base_label]) - 1)
        for scene in ANCHOR:
            task = {"root": str(BASE), "run_dir": str(run_dir), "arm": arm,
                    "cell": list(cell), "cell_label": label, "scene": int(scene),
                    "search_agents": agents, "cap": cap}
            if arm in RULE_WORLD_ARMS:
                task["kind"] = "rule"
                if base_label not in pages:
                    # Never default to 1: a missing page count silently turns
                    # slack_heuristic_cap into a one-page broadcast for the whole episode.
                    raise SystemExit("%s needs immediate pages for %s (run --stage immediate)"
                                     % (arm, base_label))
                task["immediate_pages"] = int(pages[base_label])
                task["keep_probability"] = max(
                    0.0, (pages[base_label] - 1) / float(pages[base_label]))
                task["matched_pages"] = (
                    {base_label: matched.get(base_label)}
                    if arm == "slack_matched_rate" else None)
            else:
                spec = LEARNED[arm]
                task["kind"] = "learned"
                task["checkpoint"] = str(BASE / "execution" / spec["tag"] / "checkpoints"
                                         / ("milestone_%d.pth" % spec["episode"]))
                task["checkpoint_episode"] = spec["episode"]
                task["identity_sha"] = identity
            tasks.append(task)
    archived = archived_keys(arm)
    return [task for task in tasks
            if (tuple(task["cell"]), int(task["scene"]),
                int(task.get("search_agents") or 2)) not in archived]


def run_stage(arm, workers):
    tasks = build_tasks(arm)
    print("%s: %d new worlds (%d grid points x 3 scenes, archives excluded)"
          % (arm, len(tasks), len(grid_points())), flush=True)
    started = time.time()
    results, done = [], 0
    with ProcessPoolExecutor(max_workers=int(workers),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(worker, task) for task in tasks}
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                row = future.result()
                results.append(row)
                done += 1
                if row["status"] != "ok":
                    print("  [%d/%d] %-20s s%d FAILED: %s"
                          % (done, len(tasks), row["cell_label"], row["scene"],
                             row.get("error")), flush=True)
                elif done % 25 == 0:
                    print("  [%d/%d] %s" % (done, len(tasks),
                                            time.strftime("%H:%M:%S", time.gmtime())), flush=True)
    worlds, failures = merge(arm, results)
    print("%s: %d worlds on disk, %d failures, %.0fs"
          % (arm, worlds, failures, time.time() - started), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["immediate", "rules", "v4f", "v4s", "v4t",
                                 "slack_matched", "all"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.stage == "all":
        for arm in ALL_ARMS:
            run_stage(arm, args.workers)
        return
    arm = {"immediate": "normal_immediate", "rules": "slack_heuristic_cap",
           "v4f": "v4f@200", "v4s": "v4s@300", "v4t": "v4t@100",
           "slack_matched": "slack_matched_rate"}[args.stage]
    run_stage(arm, args.workers)


if __name__ == "__main__":
    main()
