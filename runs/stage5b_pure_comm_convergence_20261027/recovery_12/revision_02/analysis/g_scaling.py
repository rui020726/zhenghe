"""The G sweep: the sealed weights evaluated at ten goal counts, staged so each arm runs alone.

Grid. Ten goal counts ``{12, 18, 24, 30, 36, 42, 48, 54, 60, 72}`` crossed with the five
``(R, N)`` pairs. ``R=6_N=6`` and ``R=12_N=6`` run all eight DEV scenes; the other three pairs
run the three anchor scenes. That is 25 worlds per goal count, 175 per arm for the seven new
goal counts.

Reuse. ``G = 12 / 24 / 48`` are the training cells, and their arms are already archived:

* ``normal_immediate``             -> ``execution/baselines/grid_v2``
* ``slack_heuristic_cap`` / ``fifo_cap_k1`` -> ``execution/baselines/paper_throttles_v1``
* ``slack_matched_rate``           -> ``execution/baselines/paper_slack_matched``
* ``v4f@200`` / ``v4s@300`` / ``v4t@100`` -> the matching ``eval_*_s8`` directories

Only the seven new goal counts are run here, and the driver writes one file per arm so a long
stage can be stopped and resumed without re-running what is already on disk.

Caps. The task fixes one rule for the G sweep: ``v4t``'s page budget is
``normal_immediate's pages per episode in the same cell, minus one``. ``v4s`` (which has no
trained budget at these goals) uses the same rule. ``slack_matched_rate`` is matched to the
pages ``v4t`` actually spent in the same cell. At the three archived goal counts the sealed
arms keep their own trained caps; the report prints both the sealed cap and what
``immediate - 1`` would have been, so the size of that discontinuity is visible.

    D:/project_env/python.exe analysis/g_scaling.py --stage immediate --workers 8
    D:/project_env/python.exe analysis/g_scaling.py --stage rules --workers 8
    D:/project_env/python.exe analysis/g_scaling.py --stage v4t --workers 8
    D:/project_env/python.exe analysis/g_scaling.py --stage slack_matched --workers 8
    D:/project_env/python.exe analysis/g_scaling.py --report
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
OUT = BASE / "execution/g_scaling"

NEW_GOALS = (18, 30, 36, 42, 54, 60, 72)
ARCHIVED_GOALS = (12, 24, 48)
GOALS_ALL = (12, 18, 24, 30, 36, 42, 48, 54, 60, 72)
PAIRS = ((6, 3), (6, 6), (12, 3), (12, 6), (12, 12))
FULL_SCENE_PAIRS = ((6, 6), (12, 6))
ANCHOR = (2301901, 2301902, 2301903)
ALL_SCENES = tuple(range(2301901, 2301909))

LEARNED = {
    "v4f@200": {"tag": "train_v4f_anchor", "episode": 200},
    "v4s@300": {"tag": "train_v4s_cap300", "episode": 300},
    "v4t@100": {"tag": "train_v4t_cap_frozen", "episode": 100},
}
RULE_ARMS = ("normal_immediate", "slack_heuristic_cap")
#: Arms whose world is built by ``run_rule_world`` (rule or throttle) rather than a checkpoint.
RULE_WORLD_ARMS = ("normal_immediate", "slack_heuristic_cap", "slack_matched_rate")


def cells_for(goals=NEW_GOALS):
    return tuple((goal, robots, subscribers)
                 for goal in goals for robots, subscribers in PAIRS)


def scenes_for(cell):
    return ALL_SCENES if (int(cell[1]), int(cell[2])) in FULL_SCENE_PAIRS else ANCHOR


def cell_label(cell):
    return "G%d_R%d_N%d" % (int(cell[0]), int(cell[1]), int(cell[2]))


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


def arm_slug(arm):
    return arm.replace("@", "_")


def arm_path(arm):
    return OUT / arm_slug(arm) / "RESULTS.json"


def pages_from_rows(rows, goals=None):
    """``{cell_label: (pages per world, worlds)}`` from result rows, filtered by goal count."""
    totals = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        cell = tuple(int(value) for value in row["cell"])
        if goals is not None and cell[0] not in goals:
            continue
        label = cell_label(cell)
        entry = totals.setdefault(label, [0, 0])
        entry[0] += int(row["emissions"])
        entry[1] += 1
    return {label: (pages / worlds, worlds)
            for label, (pages, worlds) in totals.items()}


def archived_immediate_rows():
    """The three archived goal counts' immediate rows, over exactly the scenes this sweep uses."""
    payload = json.loads((BASE / "execution/baselines/grid_v2/BASELINES.json")
                         .read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("results") or []:
        if row.get("arm") != "normal_immediate" or row.get("status") != "ok":
            continue
        cell = tuple(int(value) for value in row["cell"])
        if cell[0] not in ARCHIVED_GOALS:
            continue
        if int(row["scene"]) not in scenes_for(cell):
            continue
        rows.append(row)
    return rows


def immediate_pages():
    """``{cell_label: pages per episode}`` over all ten goal counts (archived + this sweep)."""
    pages = {label: int(round(value))
             for label, (value, _worlds) in pages_from_rows(archived_immediate_rows()).items()}
    path = arm_path("normal_immediate")
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        pages.update({label: int(round(value)) for label, (value, _worlds)
                      in pages_from_rows(payload.get("results") or []).items()})
    return pages


def v4t_pages(arm="v4t@100"):
    """``{cell_label: pages per episode actually spent by the capped learned arm}``."""
    path = arm_path(arm)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {label: int(round(value)) for label, (value, _worlds)
            in pages_from_rows(payload.get("results") or []).items()}


def worker(task):
    sys.dont_write_bytecode = True
    result = {"arm": task["arm"], "cell": task["cell"], "scene": int(task["scene"]),
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
        if task["kind"] == "rule":
            row = episode.run_rule_world(
                layout, task["arm"], int(task["scene"]), phase="eval", cell=cell,
                immediate_pages=task.get("immediate_pages"),
                keep_probability=task.get("keep_probability"),
                matched_pages_by_cell=task.get("matched_pages"))
        else:
            controller, _opt, _state = trainer.load_checkpoint(
                Path(task["checkpoint"]), arm="T2", identity_sha256=task["identity_sha"],
                expected_seed=91002, expected_episode=int(task["checkpoint_episode"]))
            row = episode.run_eval_world(
                layout, "T2", 91002, int(task["scene"]), controller, cell=cell,
                episode_index=int(task["checkpoint_episode"]),
                cap=task.get("cap"), floor=None)
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
                merged[(tuple(row["cell"]), int(row["scene"]))] = row
    for row in rows:
        if row.get("status") == "ok":
            merged[(tuple(row["cell"]), int(row["scene"]))] = row
    failures = [row for row in rows if row.get("status") != "ok"]
    dump(path, {
        "schema": "stage5b-g-scaling-arm/1",
        "arm": arm,
        "root": str(BASE),
        "goals": list(GOALS_ALL),
        "archived_goals": list(ARCHIVED_GOALS),
        "new_goals": list(NEW_GOALS),
        "pairs": [list(pair) for pair in PAIRS],
        "worlds": len(merged),
        "failures": failures,
        "results": sorted(merged.values(),
                          key=lambda row: (tuple(row["cell"]), int(row["scene"]))),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return len(merged), len(failures)


def build_tasks(arm):
    identity = sha256(STAGE / "recovery_09b" / "revision_01" / "protocol"
                      / "RECOVERY_EXPERIMENT_IDENTITY.json")
    run_dir = OUT / arm_slug(arm)
    pages = {}
    matched = {}
    # Every capped or throttled arm needs its cell's immediate page count: v4t/v4s for the
    # budget, slack_heuristic_cap for its k=1 cut, slack_matched_rate for the matched depth.
    if arm in ("v4t@100", "v4s@300", "slack_heuristic_cap", "slack_matched_rate"):
        pages = immediate_pages()
        if len(pages) < 50:
            raise SystemExit("%s needs immediate pages for all 50 cells; have %d "
                             "(run --stage immediate first)" % (arm, len(pages)))
    if arm == "slack_matched_rate":
        # Only the seven new goal counts are run here; the archived three keep their own
        # matched arm in ``paper_slack_matched``. So 35 cells (7 goals x 5 pairs) is complete.
        matched = v4t_pages()
        if len(matched) < len(NEW_GOALS) * len(PAIRS):
            raise SystemExit("slack_matched_rate needs v4t pages for the %d new cells; have %d "
                             "(run --stage v4t first)"
                             % (len(NEW_GOALS) * len(PAIRS), len(matched)))
    tasks = []
    for cell in cells_for():
        label = cell_label(cell)
        cap = None
        if arm in ("v4t@100", "v4s@300"):
            cap = max(1, int(pages[label]) - 1)
        for scene in scenes_for(cell):
            task = {"root": str(BASE), "run_dir": str(run_dir), "arm": arm,
                    "cell": list(cell), "cell_label": label, "scene": int(scene),
                    "cap": cap}
            if arm in RULE_WORLD_ARMS:
                task["kind"] = "rule"
                task["immediate_pages"] = int(pages.get(label, 1))
                task["keep_probability"] = max(
                    0.0, (pages.get(label, 1) - 1) / float(pages.get(label, 1)))
                task["matched_pages"] = matched if arm == "slack_matched_rate" else None
            else:
                spec = LEARNED[arm]
                task["kind"] = "learned"
                task["checkpoint"] = str(BASE / "execution" / spec["tag"] / "checkpoints"
                                         / ("milestone_%d.pth" % spec["episode"]))
                task["checkpoint_episode"] = spec["episode"]
                task["identity_sha"] = identity
            tasks.append(task)
    return tasks


def run_stage(arm, workers):
    tasks = build_tasks(arm)
    print("%s: %d worlds = %d new goals x 5 pairs x (2 pairs x 8 scenes + 3 pairs x 3 scenes)"
          % (arm, len(tasks), len(NEW_GOALS)), flush=True)
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
                    print("  [%d/%d] %-13s s%d FAILED: %s"
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
    parser.add_argument("--stage", default="",
                        choices=["", "immediate", "rules", "v4f", "v4s", "v4t", "slack_matched",
                                 "all"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.stage == "immediate":
        run_stage("normal_immediate", args.workers)
    elif args.stage == "rules":
        run_stage("slack_heuristic_cap", args.workers)
    elif args.stage == "v4f":
        run_stage("v4f@200", args.workers)
    elif args.stage == "v4s":
        run_stage("v4s@300", args.workers)
    elif args.stage == "v4t":
        run_stage("v4t@100", args.workers)
    elif args.stage == "slack_matched":
        run_stage("slack_matched_rate", args.workers)
    elif args.stage == "all":
        for arm in ("normal_immediate", "slack_heuristic_cap", "v4f@200", "v4s@300",
                    "v4t@100", "slack_matched_rate"):
            run_stage(arm, args.workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
