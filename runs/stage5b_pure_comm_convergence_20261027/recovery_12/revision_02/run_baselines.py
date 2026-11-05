"""Per-cell baselines: ``normal_immediate``, ``always_hold`` and ``dedup_immediate``.

Every cell in the grid is measured on every DEV scene before any training, for two reasons:

* ``q_ref`` -- the ``normal_immediate`` on-time rate in that exact cell -- is what the dual
  compares against. It cannot be borrowed from another cell: a 48-goal world with 3
  subscribers and a 12-goal world with 12 subscribers are different problems, and using one's
  rate for the other's constraint would silently re-target the lambda rule; and
* the byte counts are the "cheaper than" side of the improvement test. The task's criterion is
  "same-cell success >= 95% of immediate AND fewer bytes", so both numbers must come from the
  same cell and the same seed.

Seeds. Each (scene, cell) world uses ``scale_program.eval_seed``, so the learned arm's greedy
evaluation on that cell -- which uses the same function -- is paired with these baselines
world for world rather than merely statistically comparable.

Nothing here trains and nothing is copied from an earlier revision: a cell is a new
configuration, a new scenario and a new world, so its baselines must be its own.
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

BASE = Path(__file__).resolve().parent          # .../recovery_12/revision_02
ROOT = BASE                                     # the revision package root
OUT = ROOT / "execution" / "baselines"
SCENES = tuple(range(2301901, 2301909))
ARMS = ("normal_immediate", "always_hold", "dedup_immediate")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def worker(task):
    sys.dont_write_bytecode = True
    scene = int(task["scene"])
    cell = tuple(int(v) for v in task["cell"])
    arm = str(task["arm"])
    result = {"scene": scene, "cell": list(cell), "arm": arm}
    started = time.time()
    try:
        root, run_dir = Path(task["root"]), Path(task["run_dir"])
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from stage5b_comm import episode, scale_program
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

        # A dedup arm needs its own namespace: its worlds are eval-category but must never
        # collide with an immediate world on the same scene.
        row = episode.run_rule_world(layout, arm, scene, phase="eval", cell=cell)
        sends = list(row.get("emission_sequence") or [])
        distinct = {str(entry[1]) for entry in sends}
        result.update({
            "status": "ok",
            "on_time": int(row["on_time"]),
            "generated_goals": int(row.get("generated_goals") or 0),
            "completed": int(row.get("completed") or 0),
            "expired": int(row.get("expired") or 0),
            "emissions": int(row["emissions"]),
            "distinct_targets": len(distinct),
            "duplicate_sends": int(row["emissions"]) - len(distinct),
            "spent_bytes": int(row["spent_bytes"]),
            "bytes_per_on_time": row.get("bytes_per_on_time"),
            "on_time_rate": row.get("on_time_rate"),
            "budget_refused": int(row.get("masked_by_credit_total") or 0),
            "capacity_refused": int(row.get("masked_by_capacity_total") or 0),
            "subscribers_live": row.get("subscribers_live"),
            "num_goals_configured": row.get("num_goals_configured"),
            "rescue_robots": row.get("rescue_robots"),
            "cell_label": row.get("cell_label") or scale_program.cell_label(cell),
            "budget_bytes": int(row.get("budget_bytes") or 0),
            "budget_restored": row.get("budget_authority_restored"),
        })
    except Exception as error:
        result["status"] = "error"
        result["error"] = "%s: %s" % (type(error).__name__, error)
        result["traceback"] = traceback.format_exc()[-3000:]
    result["elapsed_seconds"] = round(time.time() - started, 2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="", help="comma-separated G,R,N triples, or empty for all")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tag", default="baselines_v1")
    args = ap.parse_args()

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    from stage5b_comm import scale_program

    if args.cells:
        cells = []
        for chunk in args.cells.split(";"):
            cells.append(tuple(int(part) for part in chunk.split(",")))
    else:
        cells = list(scale_program.CELLS)
    arms = tuple(a for a in args.arms.split(",") if a)
    scenes = (tuple(int(x) for x in args.scenes.split(",")) if args.scenes else SCENES)

    run_dir = OUT / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for cell in cells:
        scale_program.validate_cell(cell)
        for arm in arms:
            for scene in scenes:
                tasks.append({"root": str(ROOT), "run_dir": str(run_dir), "scene": scene,
                              "cell": cell, "arm": arm})
    print("baselines: %d cells x %d arms x %d scenes = %d worlds"
          % (len(cells), len(arms), len(scenes), len(tasks)))
    print("out:", run_dir)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(worker, task): task for task in tasks}
        done_count = 0
        while pending:
            done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = {"scene": task["scene"], "cell": list(task["cell"]),
                              "arm": task["arm"], "status": "executor_error",
                              "error": "%s: %s" % (type(error).__name__, error)}
                results.append(result)
                done_count += 1
                print("  [%d/%d] %-22s s%d %-17s on_time=%-3s/%s sends=%-3s dup=%-2s "
                      "bytes=%-7s"
                      % (done_count, len(tasks),
                         scale_program.cell_label(tuple(task["cell"])), result["scene"],
                         task["arm"], result.get("on_time"), result.get("generated_goals"),
                         result.get("emissions"), result.get("duplicate_sends"),
                         result.get("spent_bytes")), flush=True)

    ok = [r for r in results if r.get("status") == "ok"]
    per_cell = {}
    for cell in cells:
        label = scale_program.cell_label(cell)
        entry = {}
        for arm in arms:
            rows = [r for r in ok if r["arm"] == arm and tuple(r["cell"]) == tuple(cell)]
            on_time = sum(r["on_time"] for r in rows)
            generated = sum(r["generated_goals"] for r in rows)
            entry[arm] = {
                "worlds": len(rows),
                "on_time": on_time,
                "generated_goals": generated,
                "q": (on_time / generated) if generated else None,
                "spent_bytes": sum(r["spent_bytes"] for r in rows),
                "emissions": sum(r["emissions"] for r in rows),
                "distinct_targets": sum(r["distinct_targets"] for r in rows),
                "duplicate_sends": sum(r["duplicate_sends"] for r in rows),
                "bytes_per_on_time": (sum(r["spent_bytes"] for r in rows) / on_time
                                      if on_time else None),
            }
        per_cell[label] = {"cell": list(cell), "arms": entry}

    failures = [r for r in results if r.get("status") != "ok"]
    payload = {
        "schema": "stage5b-revision02-baselines/1",
        "root": str(ROOT),
        "scenes": list(scenes),
        "arms": list(arms),
        "cells": [list(cell) for cell in cells],
        "worlds": len(results),
        "worlds_ok": len(ok),
        "failures": failures,
        "per_cell": per_cell,
        "results": sorted(results, key=lambda r: (tuple(r["cell"]), r["arm"], r["scene"])),
        "ts": utc(),
    }
    target = run_dir / "BASELINES.json"
    dump(target, payload)

    print()
    print("per-cell summary")
    print("  %-16s %-18s %10s %10s %12s" % ("cell", "arm", "q", "sends", "bytes"))
    for label, entry in per_cell.items():
        for arm in arms:
            data = entry["arms"][arm]
            print("  %-16s %-18s %10s %10d %12d"
                  % (label, arm,
                     "n/a" if data["q"] is None else "%.4f" % data["q"],
                     data["emissions"], data["spent_bytes"]))
    print()
    print("wrote", target)
    print("sha256", sha256(target))
    if failures:
        print("FAILURES: %d" % len(failures))
        for failure in failures[:10]:
            print("  ", failure.get("cell"), failure.get("arm"), failure.get("scene"),
                  failure.get("error"))


if __name__ == "__main__":
    main()
