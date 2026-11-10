"""Run the three no-network paper baselines on all 15 cells x 8 DEV scenes.

``dedup_immediate`` is not re-run: it is already in ``grid_v2`` and this script only reads it.

Every arm here wraps the ``immediate`` rule, so no network, no checkpoint and no training is
involved. Each (arm, cell, scene) world uses ``scale_program.eval_seed`` -- the same seed a
learned arm uses -- so the comparison is paired world for world.

    D:/project_env/python.exe analysis/paper_baselines.py --workers 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

ARMS = ("fifo_cap_k1", "random_matched_rate", "slack_heuristic_cap")
SCENES = tuple(range(2301901, 2301909))
OUT = BASE / "execution/baselines/paper_throttles"


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


def immediate_stats():
    """``{cell_label: pages per episode}`` and the same with bytes, from the frozen baseline."""
    payload = json.loads((BASE / "execution/baselines/grid_v2/BASELINES.json")
                         .read_text(encoding="utf-8"))
    pages, bytes_per_world = {}, {}
    for label, entry in (payload.get("per_cell") or {}).items():
        arm = (entry.get("arms") or {}).get("normal_immediate") or {}
        worlds = arm.get("worlds")
        if not worlds:
            continue
        pages[label] = max(1, int(round(float(arm["emissions"]) / float(worlds))))
        bytes_per_world[label] = float(arm["spent_bytes"]) / float(worlds)
    return pages, bytes_per_world


def matched_rates():
    """``{cell_label: v4t pages / immediate pages}``, measured on the three anchor scenes.

    The random arm must spend the same bytes as the learned arm it is compared against, so its
    keep probability is read off ``v4t``'s own evaluation rather than from the page budget.
    """
    immediate = {}
    baseline = json.loads((BASE / "execution/baselines/grid_v2/BASELINES.json")
                          .read_text(encoding="utf-8"))
    for row in baseline.get("results") or []:
        if row.get("arm") != "normal_immediate" or row.get("status") != "ok":
            continue
        if int(row["scene"]) not in (2301901, 2301902, 2301903):
            continue
        immediate[tuple(row["cell"])] = immediate.get(tuple(row["cell"]), 0) + int(row["emissions"])
    summary = json.loads((BASE / "execution/train_v4t_cap_frozen/TRAIN_SUMMARY.json")
                         .read_text(encoding="utf-8"))
    point = summary["evaluation_points"][0]
    rates = {}
    for cell_row in point["cells"]:
        parts = cell_row["cell"].split("_")
        cell = (int(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:]))
        base = immediate.get(cell)
        if base:
            rates[cell_row["cell"]] = min(1.0, float(cell_row["emissions"]) / float(base))
    return rates


def matched_page_counts():
    """``{cell_label: v4t pages per episode}``: the depth the matched amortised arms must hit."""
    summary = json.loads((BASE / "execution/train_v4t_cap_frozen/TRAIN_SUMMARY.json")
                         .read_text(encoding="utf-8"))
    point = summary["evaluation_points"][0]
    scenes_ok = max(1, len(point.get("scenes") or (1,)))
    return {row["cell"]: max(1, int(round(float(row["emissions"]) / scenes_ok)))
            for row in point["cells"] if row.get("status") == "ok"}


def worker(task):
    sys.dont_write_bytecode = True
    result = {"cell": task["cell"], "scene": int(task["scene"]), "arm": task["arm"],
              "cell_label": task["cell_label"]}
    started = time.time()
    try:
        root = Path(task["root"])
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from stage5b_comm import episode, scale_program
        from stage5b_comm.paths import Layout

        layout = Layout(root)
        layout.execution = Path(task["run_dir"])
        layout.world_logs = layout.execution / "world_logs"
        layout.rows = layout.execution / "rows"
        layout.ledger = layout.execution / "ledger"
        layout.receipts = layout.execution / "receipts"
        for path in (layout.world_logs, layout.rows, layout.ledger, layout.receipts):
            path.mkdir(parents=True, exist_ok=True)
        episode.ensure_import_paths(layout)
        episode.native_import_preflight(layout)

        pages = int(task["immediate_pages"])
        keep = max(0.0, min(1.0, (pages - 1) / float(pages))) if pages > 1 else 0.0
        row = episode.run_rule_world(
            layout, task["arm"], int(task["scene"]), phase="eval",
            cell=tuple(int(v) for v in task["cell"]),
            immediate_pages=pages, keep_probability=keep,
            keep_by_cell=task.get("keep_by_cell"),
            matched_pages_by_cell=task.get("matched_pages"))
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
            "bytes_per_on_time": row.get("bytes_per_on_time"),
            "subscribers_live": row.get("subscribers_live"),
            "rescue_robots": row.get("rescue_robots"),
            "budget_restored": row.get("budget_authority_restored"),
            "throttle_parameters": row.get("rule_parameters"),
        })
    except Exception as error:  # a failed world is reported, never counted as a result
        result["status"] = "error"
        result["error"] = "%s: %s" % (type(error).__name__, error)
        result["traceback"] = traceback.format_exc()[-2500:]
    result["elapsed_seconds"] = round(time.time() - started, 2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--scenes", default=",".join(str(s) for s in SCENES))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tag", default="paper_throttles_v1")
    args = ap.parse_args()

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(BASE))
    from stage5b_comm import scale_program

    pages, bytes_per_world = immediate_stats()
    rates = matched_rates()
    matched_pages = matched_page_counts()
    print("v4t matched keep rates: %s"
          % ", ".join("%s=%.3f" % (k, rates[k]) for k in sorted(rates)), flush=True)
    scenes = tuple(int(s) for s in args.scenes.split(",") if s)
    arms = tuple(a for a in args.arms.split(",") if a)
    run_dir = BASE / "execution/baselines" / args.tag
    tasks = []
    for arm in arms:
        for scene in scenes:
            for cell in scale_program.CELLS:
                label = scale_program.cell_label(cell)
                tasks.append({"root": str(BASE), "run_dir": str(run_dir), "arm": arm,
                              "scene": scene, "cell": list(cell), "cell_label": label,
                              "immediate_pages": pages[label],
                              "keep_by_cell": rates,
                              "matched_pages": matched_pages})
    print("%d worlds (%d arms x %d scenes x %d cells), %d workers"
          % (len(tasks), len(arms), len(scenes), len(scale_program.CELLS), args.workers),
          flush=True)

    results, done = [], 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        pending = {pool.submit(worker, task) for task in tasks}
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                row = future.result()
                results.append(row)
                done += 1
                if row["status"] != "ok":
                    print("  [%d/%d] %-22s %-12s s%d FAILED: %s"
                          % (done, len(tasks), row["arm"], row["cell_label"], row["scene"],
                             row.get("error")), flush=True)
                elif done % 20 == 0:
                    print("  [%d/%d] %s" % (done, len(tasks), utc()), flush=True)

    failures = [r for r in results if r["status"] != "ok"]
    per_cell = {}
    for label in {r["cell_label"] for r in results}:
        per_cell[label] = {}
        for arm in arms:
            own = [r for r in results if r["cell_label"] == label and r["arm"] == arm
                   and r["status"] == "ok"]
            if not own:
                continue
            per_cell[label][arm] = {
                "worlds": len(own),
                "on_time": sum(r["on_time"] for r in own),
                "generated_goals": sum(r["generated_goals"] for r in own),
                "emissions": sum(r["emissions"] for r in own),
                "spent_bytes": sum(r["spent_bytes"] for r in own),
                "q": (sum(r["on_time"] for r in own)
                      / max(1, sum(r["generated_goals"] for r in own))),
                "bytes_per_world": (sum(r["spent_bytes"] for r in own) / len(own)),
                "immediate_pages": pages[label],
                "matched_byte_ratio": ((sum(r["spent_bytes"] for r in own) / len(own))
                                       / bytes_per_world[label]),
            }
    payload = {
        "schema": "stage5b-paper-throttle-baselines/1",
        "tag": args.tag,
        "root": str(BASE),
        "baselines": str(BASE / "execution/baselines/grid_v2/BASELINES.json"),
        "baselines_sha256": sha256(BASE / "execution/baselines/grid_v2/BASELINES.json"),
        "arms": list(arms),
        "matched_keep_rates": rates,
        "matched_pages": matched_pages,
        "scenes": list(scenes),
        "cells": len(scale_program.CELLS),
        "worlds": len(results),
        "failures": len(failures),
        "failure_detail": failures[:10],
        "per_cell": per_cell,
        "results": results,
        "wall_seconds": round(time.time() - started, 1),
    }
    dump(run_dir / "BASELINES.json", payload)
    print("wrote %s  failures %d/%d  wall %.0fs"
          % (run_dir / "BASELINES.json", len(failures), len(results),
             time.time() - started))


def utc():
    return time.strftime("%H:%M:%S", time.gmtime())


if __name__ == "__main__":
    main()
