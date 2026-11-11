"""The R and S sweeps: the sealed weights at rescue rosters and search-agent counts never trained.

Two orthogonal axes beside G. Neither is inside the ``(G, R, N)`` training cell: R is admitted
by ``scale_program.RESCUE_ACCEPTED`` and S travels beside the cell, so ``CELLS`` stays the 15
rows and every archived artifact keeps its name.

R axis (``--axis r``). ``R in {3,6,9,12,15,18}`` x ``G in {24,36,48}``, at N=3 (all six R) and
N=6 (R>=6, i.e. five). 33 cells x 3 anchor scenes = 99 worlds per arm. ``R=6/12`` at
``G=24/48`` are already archived by the G sweep, so the incremental work is
``R in {3,9,15,18}`` = 63 worlds per arm.

S axis (``--axis s``). ``S in {1,2,3,4,6}`` x ``R in {6,12}`` x ``G in {24,48}`` at N=6.
4 cells x 3 anchor scenes x 5 counts = 60 worlds per arm; ``S=2`` is archived, so 48 are new.

Arms (six): ``normal_immediate`` and ``slack_heuristic_cap`` (rule), ``slack_matched_rate``
(matched to the v4t page count in the same cell, S=2), ``v4f@200`` / ``v4s@300`` / ``v4t@100``
(sealed checkpoints). ``v4t`` and ``v4s`` take a page budget of
``normal_immediate's pages per episode in the same cell, minus one``, measured at S=2 and held
fixed across S so the S-axis differences come from the environment, not from a re-derived cap.

    D:/project_env/python.exe analysis/rs_scaling.py --axis r --stage immediate --workers 8
    D:/project_env/python.exe analysis/rs_scaling.py --axis r --stage rules --workers 8
    D:/project_env/python.exe analysis/rs_scaling.py --axis r --stage v4f --workers 8
    D:/project_env/python.exe analysis/rs_scaling.py --axis r --stage v4s --workers 8
    D:/project_env/python.exe analysis/rs_scaling.py --axis r --stage v4t --workers 8
    D:/project_env/python.exe analysis/rs_scaling.py --axis r --stage slack_matched --workers 8
    ... and the same six with --axis s
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
OUT = BASE / "execution/r_s_scaling"

ANCHOR = (2301901, 2301902, 2301903)
R_VALUES = (3, 6, 9, 12, 15, 18)
S_VALUES = (1, 2, 3, 4, 6)
R_AXIS_GOALS = (24, 36, 48)
S_AXIS_GOALS = (24, 48)
R_AXIS_PAIRS = ((3, 3), (6, 3), (9, 3), (12, 3), (15, 3), (18, 3),
                (6, 6), (9, 6), (12, 6), (15, 6), (18, 6))
S_AXIS_PAIRS = ((6, 6), (12, 6))

LEARNED = {
    "v4f@200": {"tag": "train_v4f_anchor", "episode": 200},
    "v4s@300": {"tag": "train_v4s_cap300", "episode": 300},
    "v4t@100": {"tag": "train_v4t_cap_frozen", "episode": 100},
}
RULE_WORLD_ARMS = ("normal_immediate", "slack_heuristic_cap", "slack_matched_rate")


def axis_cells(axis):
    """``[(cell, search_agents)]`` for one axis; S is ``None`` on the R axis (frozen 2)."""
    if axis == "r":
        return [((goal, robots, subscribers), None)
                for goal in R_AXIS_GOALS for robots, subscribers in R_AXIS_PAIRS]
    return [((goal, robots, subscribers), agents)
            for agents in S_VALUES for goal in S_AXIS_GOALS
            for robots, subscribers in S_AXIS_PAIRS]


def cell_key(cell, search_agents=None):
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


def arm_root(axis, arm):
    return OUT / axis / arm.replace("@", "_")


def arm_path(axis, arm):
    return arm_root(axis, arm) / "RESULTS.json"


#: Where each arm's already-archived worlds live, for the goal counts this round reuses.
#: G=24/48 come from the training-cell archives; G=36 exists only in the G sweep, which ran
#: every arm at that goal count.
ARCHIVE_SOURCES = {
    "normal_immediate": (
        ("execution/baselines/grid_v2/BASELINES.json", "normal_immediate"),
        ("execution/g_scaling/normal_immediate/RESULTS.json", None),
    ),
    "slack_heuristic_cap": (
        ("execution/baselines/paper_throttles_v1/BASELINES.json", "slack_heuristic_cap"),
        ("execution/g_scaling/slack_heuristic_cap/RESULTS.json", None),
    ),
    "slack_matched_rate": (
        ("execution/baselines/paper_slack_matched/BASELINES.json", "slack_matched_rate"),
        ("execution/g_scaling/slack_matched_rate/RESULTS.json", None),
    ),
    "v4f@200": (
        ("execution/train_v4f_anchor/eval_200_s8/EVAL.json", None),
        ("execution/g_scaling/v4f_200/RESULTS.json", None),
    ),
    "v4s@300": (
        ("execution/train_v4s_cap300/eval_300_s8/EVAL.json", None),
        ("execution/g_scaling/v4s_300/RESULTS.json", None),
    ),
    "v4t@100": (
        ("execution/train_v4t_cap_frozen/eval_100_s8/EVAL.json", None),
        ("execution/g_scaling/v4t_100/RESULTS.json", None),
    ),
}


def archived_rows(arm):
    """Every archived world of ``arm`` that this round could reuse, as result-shaped dicts."""
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
            if arm_filter is None and row.get("arm") is not None and row["arm"] != arm:
                continue
            rows.append(row)
    return rows


def _reusable(axis, cell, scene, agents):
    """Whether ``(cell, scene, S)`` is inside what this axis needs and is already archived."""
    if int(scene) not in ANCHOR:
        return False
    if axis == "r":
        if agents is not None and int(agents) != 2:
            return False
        if tuple(cell) not in set(axis_cells("r")[i][0] for i in range(len(axis_cells("r")))):
            return False
    else:
        if tuple(cell) not in S_AXIS_PAIRS and (cell[1], cell[2]) not in S_AXIS_PAIRS:
            return False
    return True


def archived_immediate_rows(axis):
    """Immediate worlds already on disk for the cells this axis reuses at S=2."""
    wanted = {cell for cell, agents in axis_cells(axis)}
    rows = []
    for row in archived_rows("normal_immediate"):
        cell = tuple(int(value) for value in row["cell"])
        if cell in wanted and int(row["scene"]) in ANCHOR:
            rows.append(row)
    return rows


def pages_from(rows):
    totals = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = cell_key(row["cell"], row.get("search_agents"))
        entry = totals.setdefault(key, [0, 0])
        entry[0] += int(row["emissions"])
        entry[1] += 1
    return {key: int(round(pages / worlds)) for key, (pages, worlds) in totals.items()}


def immediate_pages(axis):
    """``{cell_key: pages per episode}`` for every cell this axis uses (S=2 reference)."""
    rows = archived_immediate_rows(axis)
    path = arm_path(axis, "normal_immediate")
    if path.exists():
        rows += [row for row in json.loads(path.read_text(encoding="utf-8")).get("results") or []
                 if row.get("status") == "ok"]
    return pages_from(rows)


def v4t_pages(axis):
    path = arm_path(axis, "v4t@100")
    if not path.exists():
        return {}
    return pages_from(json.loads(path.read_text(encoding="utf-8")).get("results") or [])


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


def merge(axis, arm, rows):
    path = arm_path(axis, arm)
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
        "schema": "stage5b-rs-scaling-arm/1",
        "axis": axis,
        "arm": arm,
        "root": str(BASE),
        "scenes": list(ANCHOR),
        "worlds": len(merged),
        "failures": failures,
        "results": sorted(merged.values(),
                          key=lambda row: (tuple(row["cell"]), int(row["scene"]),
                                           int(row.get("search_agents") or 2))),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return len(merged), len(failures)


def archived_world_keys(axis):
    """``{arm: {(cell, scene, S)}}`` the archives already hold for this axis.

    The training-cell archives carry G=24/48 at R=6/12 with S=2 and the same ``immediate - 1``
    page rule, and the G sweep carries G=36 for every arm. Those worlds are reused; anything
    the archives do not hold is run here, which is why the R axis adds 63 worlds and the S axis
    48 rather than 99 and 60.
    """
    wanted = {cell for cell, _agents in axis_cells(axis)}
    keys = {}
    for arm in ("normal_immediate", "slack_heuristic_cap", "slack_matched_rate",
                "v4f@200", "v4s@300", "v4t@100"):
        keys[arm] = {(tuple(int(v) for v in row["cell"]), int(row["scene"]), 2)
                     for row in archived_rows(arm)
                     if tuple(int(v) for v in row["cell"]) in wanted
                     and int(row["scene"]) in ANCHOR}
    return keys


def build_tasks(axis, arm):
    identity = sha256(STAGE / "recovery_09b" / "revision_01" / "protocol"
                      / "RECOVERY_EXPERIMENT_IDENTITY.json")
    run_dir = arm_root(axis, arm)
    pages = {}
    matched = {}
    if arm in ("v4t@100", "v4s@300", "slack_heuristic_cap", "slack_matched_rate"):
        pages = immediate_pages(axis)
    if arm == "slack_matched_rate":
        matched = v4t_pages(axis)
    tasks = []
    for cell, agents in axis_cells(axis):
        label = "G%d_R%d_N%d" % cell
        if agents is not None:
            label = cell_key(cell, agents)
        cap = None
        if arm in ("v4t@100", "v4s@300"):
            # The budget is measured at S=2 and held fixed across S, so an S-axis difference
            # is the environment's, not a re-derived quota's.
            base_label = "G%d_R%d_N%d" % cell
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
                base_label = "G%d_R%d_N%d" % cell
                task["immediate_pages"] = int(pages.get(base_label, 1))
                task["keep_probability"] = max(
                    0.0, (pages.get(base_label, 1) - 1) / float(pages.get(base_label, 1)))
                # ``run_rule_world`` looks the matched depth up by the *base* cell label
                # because the cell tuple does not carry S. Build a one-entry table keyed that
                # way, holding this S's own v4t page count.
                task["matched_pages"] = (
                    {base_label: matched.get(label, matched.get(base_label))}
                    if arm == "slack_matched_rate" else None)
            else:
                spec = LEARNED[arm]
                task["kind"] = "learned"
                task["checkpoint"] = str(BASE / "execution" / spec["tag"] / "checkpoints"
                                         / ("milestone_%d.pth" % spec["episode"]))
                task["checkpoint_episode"] = spec["episode"]
                task["identity_sha"] = identity
            tasks.append(task)
    archived = archived_world_keys(axis).get(arm, set())
    kept = [task for task in tasks
            if (tuple(task["cell"]), int(task["scene"]),
                int(task.get("search_agents") or 2)) not in archived]
    return kept


def run_stage(axis, arm, workers):
    tasks = build_tasks(axis, arm)
    print("[%s] %s: %d worlds" % (axis, arm, len(tasks)), flush=True)
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
                    print("  [%d/%d] %-18s s%d FAILED: %s"
                          % (done, len(tasks), row["cell_label"], row["scene"],
                             row.get("error")), flush=True)
                elif done % 30 == 0:
                    print("  [%d/%d] %s" % (done, len(tasks),
                                            time.strftime("%H:%M:%S", time.gmtime())), flush=True)
    worlds, failures = merge(axis, arm, results)
    print("[%s] %s: %d worlds on disk, %d failures, %.0fs"
          % (axis, arm, worlds, failures, time.time() - started), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", required=True, choices=["r", "s"])
    parser.add_argument("--stage", required=True,
                        choices=["immediate", "rules", "v4f", "v4s", "v4t",
                                 "slack_matched", "all"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.stage == "all":
        for arm in ("normal_immediate", "slack_heuristic_cap", "v4f@200", "v4s@300",
                    "v4t@100", "slack_matched_rate"):
            run_stage(args.axis, arm, args.workers)
        return
    arm = {"immediate": "normal_immediate", "rules": "slack_heuristic_cap",
           "v4f": "v4f@200", "v4s": "v4s@300", "v4t": "v4t@100",
           "slack_matched": "slack_matched_rate"}[args.stage]
    run_stage(args.axis, arm, args.workers)


if __name__ == "__main__":
    main()
