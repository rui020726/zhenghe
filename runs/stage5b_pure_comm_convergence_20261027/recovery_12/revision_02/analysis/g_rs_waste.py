"""Per-world broadcast-waste readings for the crossed G x R x S grid.

Sources: the archived G-sweep worlds this grid reuses plus every world written under
``execution/g_rs_grid/<arm>/``. The key carries the search-agent count because the grid runs the
same (G, R) cell at three values of S.

    D:/project_env/python.exe analysis/g_rs_waste.py --workers 8
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
OUT = BASE / "execution/g_rs_grid"
CELL_TAG = re.compile(r"(G\d+R\d+N\d+)")
#: Both spellings occur: rule arms write ``...G24R6N6S4``, learned arms ``..._c100_s4``.
S_TAG = re.compile(r"_?[sS](\d+)$")
ANCHOR = (2301901, 2301902, 2301903)
GRID_GOALS = (72, 84, 96, 108, 120, 144)
GRID_ROBOTS = (6, 12, 18)

ARCHIVE_ROOTS = {
    "normal_immediate": "execution/g_scaling/normal_immediate",
    "slack_heuristic_cap": "execution/g_scaling/slack_heuristic_cap",
    "slack_matched_rate": "execution/g_scaling/slack_matched_rate",
    "v4f@200": "execution/g_scaling/v4f_200",
    "v4s@300": "execution/g_scaling/v4s_300",
    "v4t@100": "execution/g_scaling/v4t_100",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_gz(path):
    with gzip.open(str(path), "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key_of(directory):
    name = Path(directory).name
    parts = name.split("__")
    if len(parts) != 4:
        raise RuntimeError("%s is not a four-part world leaf" % (name,))
    match = CELL_TAG.search(parts[1])
    if match is None:
        raise RuntimeError("%s carries no cell tag" % (name,))
    cell = tuple(int(value) for value in re.findall(r"\d+", match.group(1)))
    s_match = S_TAG.search(parts[1])
    agents = int(s_match.group(1)) if s_match else 2
    return cell, int(parts[3]), agents


def measure(directory, window=1000):
    directory = Path(directory)
    lifecycle = read_jsonl_gz(directory / "lifecycle.jsonl.gz")
    transcript = read_jsonl_gz(directory / "relay_transcript.jsonl.gz")
    discovered, completed = {}, {}
    for record in lifecycle:
        kind = record.get("event_type")
        stamp = float(record.get("timestamp") or 0.0)
        target = str(record.get("target_id"))
        if kind == "detection" and stamp <= window:
            discovered[target] = stamp + float((record.get("detail") or {}).get(
                "deadline_offset", 0.0))
        elif (kind == "progress" and record.get("status") == "completed"
              and stamp <= window):
            completed.setdefault(target, stamp)
    on_time = {target for target, deadline in discovered.items()
               if target in completed and completed[target] <= deadline}
    broadcast = {str(record["target"]) for record in transcript
                 if record.get("kind") == "send"}
    return len(broadcast - on_time), len(broadcast)


def worker(task):
    sys.dont_write_bytecode = True
    try:
        waste_n, waste_d = measure(task["directory"])
        return {"arm": task["arm"], "cell": list(task["cell"]), "scene": task["scene"],
                "search_agents": task["search_agents"],
                "waste_n": waste_n, "waste_d": waste_d, "status": "ok",
                "source": task["source"]}
    except Exception as error:
        return {"arm": task["arm"], "cell": list(task["cell"]), "scene": task["scene"],
                "search_agents": task["search_agents"], "status": "error",
                "error": "%s: %s" % (type(error).__name__, error),
                "source": task["source"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sys.dont_write_bytecode = True

    tasks, added = [], set()
    for arm, root in ARCHIVE_ROOTS.items():
        logs = BASE / root / "world_logs"
        if not logs.is_dir():
            continue
        for directory in sorted(logs.iterdir()):
            if not directory.is_dir():
                continue
            cell, scene, agents = key_of(directory)
            if cell[0] not in GRID_GOALS or cell[1] not in GRID_ROBOTS or cell[2] != 6:
                continue
            if scene not in ANCHOR or agents != 2:
                continue
            key = (arm, cell, scene, agents)
            if key in added:
                continue
            added.add(key)
            inner = [path for path in directory.iterdir() if path.is_dir()][0]
            tasks.append({"arm": arm, "cell": cell, "scene": scene,
                          "search_agents": agents, "directory": str(inner),
                          "source": "archived:%s" % root})
    for arm in ARCHIVE_ROOTS:
        logs = OUT / arm.replace("@", "_") / "world_logs"
        if not logs.is_dir():
            continue
        for directory in sorted(logs.iterdir()):
            if not directory.is_dir():
                continue
            cell, scene, agents = key_of(directory)
            key = (arm, cell, scene, agents)
            if key in added:
                continue
            added.add(key)
            inner = [path for path in directory.iterdir() if path.is_dir()][0]
            tasks.append({"arm": arm, "cell": cell, "scene": scene,
                          "search_agents": agents, "directory": str(inner),
                          "source": "grid"})

    print("%d worlds, %d workers" % (len(tasks), args.workers), flush=True)
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        for row in pool.map(worker, tasks):
            rows.append(row)
            done += 1
            if row["status"] != "ok":
                print("  FAILED %s %s s%d S%s: %s"
                      % (row["arm"], row["cell"], row["scene"], row["search_agents"],
                         row["error"]), flush=True)
            elif done % 250 == 0:
                print("  [%d/%d]" % (done, len(tasks)), flush=True)
    failures = [row for row in rows if row["status"] != "ok"]
    if failures:
        raise SystemExit("%d worlds failed; first: %s" % (len(failures), failures[0]))
    out = OUT / "WASTE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "stage5b-g-rs-grid-waste/1",
        "root": str(BASE),
        "worlds": len(rows),
        "rule": "waste = distinct broadcast targets that did not finish inside their deadline"
                " / distinct broadcast targets",
        "results": sorted(rows, key=lambda row: (row["arm"], tuple(row["cell"]),
                                                 int(row["scene"]),
                                                 int(row["search_agents"]))),
    }
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(str(tmp), str(out))
    print("wrote %s  sha256 %s" % (out, sha256(out)))


if __name__ == "__main__":
    main()
