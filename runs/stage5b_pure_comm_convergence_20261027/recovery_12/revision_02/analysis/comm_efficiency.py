"""Four communication-efficiency readings, re-derived from the archived world logs only.

Nothing here opens a world. Every number is read back from the two files a finished world
already wrote -- ``lifecycle.jsonl.gz`` (who was detected, when, with what deadline, and when it
completed) and ``relay_transcript.jsonl.gz`` (what was actually put on the air) -- so the four
columns can be added to the paper table without re-running a single episode.

The four readings, all over one scene group (anchor(3), unseen(5), or all(8)):

``channel_uses_per_world``
    ``emissions x subscribers``, summed over the group and divided by its world count. One
    emission addresses every subscriber, and the relay bills it per receiver, so this is the
    receiver-side airtime the arm actually occupied.

``bytes_per_on_time``
    ``sum(spent_bytes) / sum(on_time)``. Bytes are the relay's own billed total, already
    per-receiver. The denominator is the number of tasks that finished inside their deadline,
    so this is the price of a delivered task, not the price of an attempt.

``messages_per_on_time``
    ``sum(emissions) / sum(on_time)``. Its reciprocal is the broadcast conversion rate
    ``on_time / emissions``: of every message put on the air, what fraction corresponds to a
    task that finished on time. Values above 1 mean the arm spends more than one message per
    delivered task (retransmission, refreshes, or broadcasts that were not needed).

``broadcast_waste_ratio``
    Of the distinct targets a group put on the air, the fraction that did **not** finish inside
    their deadline. Denominator = distinct broadcast targets; numerator = those among them not
    in the on-time set. A target that was never broadcast is outside this ratio by construction
    (it is a discovery or local-knowledge outcome, not a broadcast outcome), so the ratio can be
    read as "how much of what we said was too late or useless".

Consistency is asserted, not assumed: for every world the extracted ``emissions`` and
``on_time`` must equal the archived row that the paper table already prints. A world whose logs
disagree with its own summary aborts the extraction instead of producing a second, quieter set
of numbers.

    D:/project_env/python.exe analysis/comm_efficiency.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "execution/COMM_EFFICIENCY.json"

ANCHOR = (2301901, 2301902, 2301903)
UNSEEN = (2301904, 2301905, 2301906, 2301907, 2301908)
GROUPS = (("anchor(3)", ANCHOR), ("unseen(5)", UNSEEN), ("all(8)", ANCHOR + UNSEEN))

CELL_TAG = re.compile(r"(G\d+R\d+N\d+)")
#: arm -> (root that holds ``world_logs/``, archived results file, root holds several arms?)
#: A shared root stores every arm's worlds side by side, so a directory is that arm's only when
#: its ``<arm>_<cell>`` prefix names it; a dedicated root (one learned checkpoint per run) has
#: exactly one arm's worlds and every leaf belongs to it.
SOURCES = (
    ("normal_immediate", "execution/baselines/grid_v2", "execution/baselines/grid_v2/BASELINES.json", True),
    ("dedup_immediate", "execution/baselines/grid_v2", "execution/baselines/grid_v2/BASELINES.json", True),
    ("always_hold", "execution/baselines/grid_v2", "execution/baselines/grid_v2/BASELINES.json", True),
    ("fifo_cap_k1", "execution/baselines/paper_throttles_v1",
     "execution/baselines/paper_throttles_v1/BASELINES.json", True),
    ("slack_heuristic_cap", "execution/baselines/paper_throttles_v1",
     "execution/baselines/paper_throttles_v1/BASELINES.json", True),
    ("random_matched_rate", "execution/baselines/paper_random_matched",
     "execution/baselines/paper_random_matched/BASELINES.json", False),
    ("slack_matched_rate", "execution/baselines/paper_slack_matched",
     "execution/baselines/paper_slack_matched/BASELINES.json", False),
    ("periodic_k5", "execution/baselines/paper_traditional_v1",
     "execution/baselines/paper_traditional_v1/BASELINES.json", True),
    ("event_maxwait", "execution/baselines/paper_traditional_v1",
     "execution/baselines/paper_traditional_v1/BASELINES.json", True),
    ("red_aqm", "execution/baselines/classic_v1",
     "execution/baselines/classic_v1/BASELINES.json", True),
    ("token_bucket", "execution/baselines/classic_v1",
     "execution/baselines/classic_v1/BASELINES.json", True),
    ("ahp_utility", "execution/baselines/classic_v1",
     "execution/baselines/classic_v1/BASELINES.json", True),
    ("linucb_bandit", "execution/baselines/classic_v1",
     "execution/baselines/classic_v1/BASELINES.json", True),
    ("v4f@200", "execution/train_v4f_anchor/eval_200_s8",
     "execution/train_v4f_anchor/eval_200_s8/EVAL.json", False),
    ("v4g@300", "execution/train_v4g_3scene_anchor/eval_300_s8",
     "execution/train_v4g_3scene_anchor/eval_300_s8/EVAL.json", False),
    ("v4s@300", "execution/train_v4s_cap300/eval_300_s8",
     "execution/train_v4s_cap300/eval_300_s8/EVAL.json", False),
    ("v4t@100", "execution/train_v4t_cap_frozen/eval_100_s8",
     "execution/train_v4t_cap_frozen/eval_100_s8/EVAL.json", False),
    ("v4u@100", "execution/train_v4u2_window_fixed/eval_100_s8",
     "execution/train_v4u2_window_fixed/eval_100_s8/EVAL.json", False),
)


def cell_label(cell):
    goals, robots, subscribers = (int(value) for value in cell)
    return "G%d_R%d_N%d" % (goals, robots, subscribers)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_gz(path):
    with gzip.open(str(path), "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def world_key(directory):
    """``(cell, scene)`` out of ``eval__<anything>_<GxRyNz>[_c<ep>]__<seed>__<scene>``."""
    name = Path(directory).name
    parts = name.split("__")
    if len(parts) != 4:
        raise RuntimeError("%s is not a two-part world leaf name" % (name,))
    cell = CELL_TAG.search(parts[1])
    if cell is None:
        raise RuntimeError("%s carries no cell tag" % (name,))
    numbers = [int(value) for value in re.findall(r"\d+", cell.group(1))]
    return tuple(numbers), int(parts[3])


def measure(directory, window=1000):
    """Everything the four readings need, from one world's two logs."""
    directory = Path(directory)
    lifecycle = read_jsonl_gz(directory / "lifecycle.jsonl.gz")
    transcript = read_jsonl_gz(directory / "relay_transcript.jsonl.gz")
    discovered = {}
    completed_at = {}
    for record in lifecycle:
        event_type = record.get("event_type")
        timestamp = float(record.get("timestamp") or 0.0)
        target = str(record.get("target_id"))
        if event_type == "detection" and timestamp <= window:
            deadline_offset = float((record.get("detail") or {}).get("deadline_offset", 0.0))
            discovered[target] = timestamp + deadline_offset
        elif (event_type == "progress" and record.get("status") == "completed"
              and timestamp <= window):
            completed_at.setdefault(target, timestamp)
    on_time = {target for target, deadline in discovered.items()
               if target in completed_at and completed_at[target] <= deadline}
    sends = [record for record in transcript if record.get("kind") == "send"]
    broadcast = {str(record["target"]) for record in sends}
    return {
        "emissions": len(sends),
        "distinct_broadcast": len(broadcast),
        "on_time": len(on_time),
        "broadcast_on_time": len(broadcast & on_time),
        "waste_n": len(broadcast - on_time),
        "waste_d": len(broadcast),
    }


def archived_rows(path, arm):
    """``{(cell, scene): (on_time, emissions, spent_bytes)}`` from a BASELINES/EVAL file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {}
    for row in payload.get("results") or []:
        if row.get("status") != "ok":
            continue
        if row.get("arm") is not None and row["arm"] != arm:
            continue
        rows[(tuple(int(v) for v in row["cell"]), int(row["scene"]))] = (
            int(row["on_time"]), int(row["emissions"]), int(row["spent_bytes"]))
    return rows


def worker(task):
    sys.dont_write_bytecode = True
    try:
        readings = measure(task["directory"])
        readings["spent_bytes"] = int(task["archived"][2])
        return dict(task, status="ok", **readings)
    except Exception as error:  # a failed world is reported, never counted as a result
        return dict(task, status="error", error="%s: %s" % (type(error).__name__, error))


def aggregate(worlds, scenes):
    """One group's four readings plus the plain totals they are ratios of."""
    chosen = [w for w in worlds if int(w["scene"]) in scenes]
    if not chosen:
        return None
    emissions = sum(w["emissions"] for w in chosen)
    on_time = sum(w["on_time"] for w in chosen)
    spent = sum(w["spent_bytes"] for w in chosen)
    channel_uses = sum(w["emissions"] * w["subscribers"] for w in chosen)
    broadcast = sum(w["distinct_broadcast"] for w in chosen)
    waste = sum(w["waste_n"] for w in chosen)
    goals = sum(w["generated_goals"] for w in chosen)
    return {
        "worlds": len(chosen),
        "goals": goals,
        "on_time": on_time,
        "q": on_time / goals if goals else 0.0,
        "emissions": emissions,
        "spent_bytes": spent,
        "channel_uses": channel_uses,
        "channel_uses_per_world": channel_uses / len(chosen),
        "bytes_per_on_time": spent / on_time if on_time else None,
        "messages_per_on_time": emissions / on_time if on_time else None,
        "conversion": on_time / emissions if emissions else None,
        "distinct_broadcast": broadcast,
        "waste_n": waste,
        "broadcast_waste_ratio": waste / broadcast if broadcast else None,
        "broadcast_on_time": sum(w["broadcast_on_time"] for w in chosen),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    sys.dont_write_bytecode = True

    tasks, archives = [], {}
    for arm, root, archived, shared in SOURCES:
        root = BASE / root
        logs = root / "world_logs"
        archived_path = BASE / archived
        if not logs.is_dir() or not archived_path.exists():
            print("skip %-20s (no world_logs or no archive under %s)" % (arm, root), flush=True)
            continue
        archives[arm] = archived_rows(archived_path, arm)
        directories = sorted(path for path in logs.iterdir() if path.is_dir())
        if shared:
            directories = [path for path in directories
                           if path.name.startswith("eval__%s_" % arm)]
        for directory in directories:
            inner = [path for path in directory.iterdir() if path.is_dir()]
            if len(inner) != 1:
                raise RuntimeError("%s holds %d inner directories" % (directory, len(inner)))
            cell, scene = world_key(directory)
            row = archives[arm].get((cell, scene))
            if row is None:
                raise RuntimeError("%s: no archived row for %s scene %d"
                                   % (arm, cell, scene))
            tasks.append({"arm": arm, "cell": list(cell), "scene": scene,
                          "directory": str(inner[0]),
                          "subscribers": int(cell[2]),
                          "generated_goals": None,
                          "archived": list(row)})

    # The generated-goal count is a property of the cell, not of the arm: read it off the
    # baseline rows rather than off any learned file.
    goals_by_cell = {}
    for arm, root, archived, _shared in SOURCES:
        if arm not in ("normal_immediate", "always_hold"):
            continue
        payload = json.loads((BASE / archived).read_text(encoding="utf-8"))
        for row in payload.get("results") or []:
            if row.get("status") == "ok":
                goals_by_cell[tuple(int(v) for v in row["cell"])] = int(row["generated_goals"])
    for task in tasks:
        task["generated_goals"] = goals_by_cell[tuple(task["cell"])]
        if task["generated_goals"] != 12 * (task["cell"][0] // 12):
            raise RuntimeError("%s: unexpected goal count %d"
                               % (task["cell"], task["generated_goals"]))

    print("%d worlds across %d arms, %d workers"
          % (len(tasks), len({t["arm"] for t in tasks}), args.workers), flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        for done, row in enumerate(pool.map(worker, tasks), start=1):
            results.append(row)
            if row["status"] != "ok":
                print("  FAILED %s %s s%d: %s"
                      % (row["arm"], row["cell"], row["scene"], row["error"]), flush=True)
            elif done % 200 == 0:
                print("  [%d/%d]" % (done, len(tasks)), flush=True)

    failures = [row for row in results if row["status"] != "ok"]
    if failures:
        raise SystemExit("%d worlds failed to yield a reading; first: %s"
                         % (len(failures), failures[0]))

    mismatches = []
    for row in results:
        on_time, emissions, spent = (int(v) for v in row["archived"])
        if int(row["on_time"]) != on_time or int(row["emissions"]) != emissions:
            mismatches.append({"arm": row["arm"], "cell": row["cell"], "scene": row["scene"],
                               "archived": row["archived"],
                               "extracted": [row["on_time"], row["emissions"]]})
    if mismatches:
        raise SystemExit("logs disagree with the archive in %d worlds; first: %s"
                         % (len(mismatches), mismatches[0]))

    per_arm = {}
    for arm, _root, _archived, _shared in SOURCES:
        worlds = [row for row in results if row["arm"] == arm]
        if not worlds:
            continue
        by_cell, by_cell_anchor = {}, {}
        for cell in sorted({tuple(row["cell"]) for row in worlds}):
            own = [row for row in worlds if tuple(row["cell"]) == cell]
            by_cell[cell_label(cell)] = aggregate(
                own, {int(row["scene"]) for row in worlds})
            by_cell_anchor[cell_label(cell)] = aggregate(own, set(ANCHOR))
        per_arm[arm] = {
            "groups": {name: aggregate(worlds, set(scenes)) for name, scenes in GROUPS},
            "cells": by_cell,
            "cells_anchor": by_cell_anchor,
        }

    payload = {
        "schema": "stage5b-comm-efficiency/1",
        "root": str(BASE),
        "rules": {
            "channel_uses_per_world": "sum(emissions * subscribers) / worlds",
            "bytes_per_on_time": "sum(spent_bytes) / sum(on_time)",
            "messages_per_on_time": "sum(emissions) / sum(on_time)",
            "conversion": "sum(on_time) / sum(emissions)  (reciprocal of messages_per_on_time)",
            "broadcast_waste_ratio":
                "sum(distinct broadcast not on time) / sum(distinct broadcast)",
        },
        "sources": {arm: sha256(BASE / archived)
                    for arm, _root, archived, _shared in SOURCES
                    if (BASE / archived).exists()},
        "groups": {name: list(scenes) for name, scenes in GROUPS},
        "worlds": len(results),
        "failures": failures,
        "per_arm": per_arm,
        "per_world": [
            {"arm": row["arm"], "cell": row["cell"], "scene": row["scene"],
             "subscribers": row["subscribers"], "generated_goals": row["generated_goals"],
             "emissions": row["emissions"], "distinct_broadcast": row["distinct_broadcast"],
             "on_time": row["on_time"], "broadcast_on_time": row["broadcast_on_time"],
             "waste_n": row["waste_n"], "waste_d": row["waste_d"],
             "spent_bytes": int(row["spent_bytes"])}
            for row in results],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(str(tmp), str(out))

    print()
    print("| arm | group | channel uses/world | B / on-time | msg / on-time | conv | waste |")
    print("|---|---|---|---|---|---|---|")
    for arm, _root, _archived, _shared in SOURCES:
        if arm not in per_arm:
            continue
        for name, _scenes in GROUPS:
            stats = per_arm[arm]["groups"][name]

            def number(key, fmt="%.4f"):
                value = stats[key]
                return "n/a" if value is None else fmt % value

            print("| %s | %s | %s | %s | %s | %s | %s |"
                  % (arm, name, number("channel_uses_per_world", "%.1f"),
                     number("bytes_per_on_time", "%.1f"),
                     number("messages_per_on_time"), number("conversion"),
                     number("broadcast_waste_ratio")))
    print()
    print("wrote %s  sha256 %s" % (out, sha256(out)))


if __name__ == "__main__":
    main()
