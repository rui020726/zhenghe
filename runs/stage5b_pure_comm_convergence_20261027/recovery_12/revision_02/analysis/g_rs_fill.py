"""Fill the two deep-grid points that a label collision left out of every arm.

``(G=72, R=6, S=2)`` and ``(G=72, R=12, S=2)`` are the only two of the 54 points missing from
``execution/g_rs_grid/<arm>/RESULTS.json``. They are missing for a naming reason, not a physics
reason: the first pass wrote S=2 worlds under a label without the ``_S`` suffix
(``G72_R6_N6``), the resume pass tried to write the same leaf again and every arm died with
``FileExistsError``, and the survivors were quarantined instead of merged.

This driver re-runs exactly those 36 worlds (2 points x 3 anchor scenes x 6 arms) under an
explicit ``_S2`` label, which cannot collide with the legacy leaf, and merges them into the same
``RESULTS.json`` the report reads. Nothing else is re-run.

    D:/project_env/python.exe analysis/g_rs_fill.py --workers 6
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "analysis"))
sys.dont_write_bytecode = True

import g_rs_grid as grid  # noqa: E402  (the driver this fill feeds)

MISSING = ((72, 6), (72, 12))
AGENTS = 2


def fill_tasks(arm, identity):
    tasks = []
    pages = grid.immediate_pages()
    matched = grid.v4t_pages()
    run_dir = grid.arm_root(arm)
    for goals, robots in MISSING:
        cell = (goals, robots, grid.SUBSCRIBERS)
        base_label = "G%d_R%d_N%d" % cell
        label = base_label + "_S2"
        cap = None
        if arm in ("v4t@100", "v4s@300"):
            if base_label not in pages:
                raise SystemExit("%s has no immediate page count for %s" % (arm, base_label))
            cap = max(1, int(pages[base_label]) - 1)
        for scene in grid.ANCHOR:
            task = {"root": str(BASE), "run_dir": str(run_dir), "arm": arm,
                    "cell": list(cell), "cell_label": label, "scene": int(scene),
                    "search_agents": AGENTS, "cap": cap}
            if arm in grid.RULE_WORLD_ARMS:
                task["kind"] = "rule"
                task["immediate_pages"] = int(pages[base_label])
                task["keep_probability"] = max(
                    0.0, (pages[base_label] - 1) / float(pages[base_label]))
                task["matched_pages"] = ({base_label: matched.get(base_label)}
                                         if arm == "slack_matched_rate" else None)
            else:
                spec = grid.LEARNED[arm]
                task["kind"] = "learned"
                task["checkpoint"] = str(BASE / "execution" / spec["tag"] / "checkpoints"
                                         / ("milestone_%d.pth" % spec["episode"]))
                task["checkpoint_episode"] = spec["episode"]
                task["identity_sha"] = identity
            tasks.append(task)
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--arms", default=",".join(grid.ALL_ARMS))
    args = parser.parse_args()
    sys.dont_write_bytecode = True

    identity = grid.sha256(grid.STAGE / "recovery_09b" / "revision_01" / "protocol"
                           / "RECOVERY_EXPERIMENT_IDENTITY.json")
    arms = [arm for arm in args.arms.split(",") if arm]
    for arm in arms:
        tasks = fill_tasks(arm, identity)
        print("%s: %d worlds (%s at S=2)"
              % (arm, len(tasks), ", ".join("G%d_R%d" % pair for pair in MISSING)),
              flush=True)
        results, done = [], 0
        started = time.time()
        with ProcessPoolExecutor(max_workers=int(args.workers),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = {pool.submit(grid.worker, task) for task in tasks}
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
        worlds, failures = grid.merge(arm, results)
        print("%s: %d worlds on disk, %d failures this pass, %.0fs"
              % (arm, worlds, failures, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
