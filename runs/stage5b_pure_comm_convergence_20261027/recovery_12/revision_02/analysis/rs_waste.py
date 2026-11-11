"""Per-world broadcast-waste readings for the R and S sweeps.

The waste ratio needs the lifecycle (which targets finished on time) and the relay transcript
(which distinct targets went on the air), so it cannot come from the runner's RESULT rows.

Sources per axis: every archived world the axis reuses plus every world this round wrote under
``execution/r_s_scaling/<axis>/``. A search-agent count is part of the key, because the S axis
runs the same (G, R, N) cell at five values of S.

    D:/project_env/python.exe analysis/rs_waste.py --axis r --workers 8
    D:/project_env/python.exe analysis/rs_waste.py --axis s --workers 8
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
CELL_TAG = re.compile(r"(G\d+R\d+N\d+)")
#: Two namespaces carry S differently: rule arms write ``...G24R6N6S1`` (uppercase, no
#: separator) and learned arms write ``..._c100_s1`` (lowercase, underscore). One pattern
#: matches both; without the case fold the learned arms all read as the frozen S=2.
S_TAG = re.compile(r"_?[sS](\d+)$")

#: arm -> (world-log root, directory prefix, arm filter)
ARCHIVED_ROOTS = {
    "normal_immediate": ("execution/baselines/grid_v2", "eval__normal_immediate_"),
    "slack_heuristic_cap": ("execution/baselines/paper_throttles_v1",
                            "eval__slack_heuristic_cap_"),
    "slack_matched_rate": ("execution/baselines/paper_slack_matched",
                           "eval__slack_matched_rate_"),
    "v4f@200": ("execution/train_v4f_anchor/eval_200_s8", "eval__"),
    "v4s@300": ("execution/train_v4s_cap300/eval_300_s8", "eval__"),
    "v4t@100": ("execution/train_v4t_cap_frozen/eval_100_s8", "eval__"),
}
SWEEP_ROOTS = {
    "normal_immediate": "execution/g_scaling/normal_immediate",
    "slack_heuristic_cap": "execution/g_scaling/slack_heuristic_cap",
    "slack_matched_rate": "execution/g_scaling/slack_matched_rate",
    "v4f@200": "execution/g_scaling/v4f_200",
    "v4s@300": "execution/g_scaling/v4s_300",
    "v4t@100": "execution/g_scaling/v4t_100",
}
ANCHOR = (2301901, 2301902, 2301903)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_gz(path):
    with gzip.open(str(path), "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key_of(directory, axis):
    name = Path(directory).name
    parts = name.split("__")
    if len(parts) != 4:
        raise RuntimeError("%s is not a four-part world leaf" % (name,))
    match = CELL_TAG.search(parts[1])
    if match is None:
        raise RuntimeError("%s carries no cell tag" % (name,))
    cell = tuple(int(value) for value in re.findall(r"\d+", match.group(1)))
    # The R axis runs at the frozen S=2; only the S axis carries a search-agent count in the
    # leaf name.
    agents = 2
    if axis == "s":
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


def wanted_cells(axis):
    from analysis.rs_scaling import axis_cells
    return {cell for cell, _agents in axis_cells(axis)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", required=True, choices=["r", "s"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(BASE))

    wanted = wanted_cells(args.axis)
    tasks = []
    added = set()
    for arm, (root, prefix) in ARCHIVED_ROOTS.items():
        logs = BASE / root / "world_logs"
        if not logs.is_dir():
            continue
        for directory in sorted(logs.iterdir()):
            if not directory.is_dir() or not directory.name.startswith(prefix):
                continue
            inner = [path for path in directory.iterdir() if path.is_dir()]
            if len(inner) != 1:
                raise RuntimeError("%s holds %d inner directories" % (directory, len(inner)))
            cell, scene, agents = key_of(directory, args.axis)
            if cell not in wanted or scene not in ANCHOR:
                continue
            if args.axis == "s" and agents != 2:
                continue
            key = (arm, cell, scene, agents)
            if key in added:
                continue
            added.add(key)
            tasks.append({"arm": arm, "cell": cell, "scene": scene,
                          "search_agents": agents, "directory": str(inner[0]),
                          "source": "archived:%s" % root})
    for arm, root in SWEEP_ROOTS.items():
        logs = BASE / root / "world_logs"
        if not logs.is_dir():
            continue
        for directory in sorted(logs.iterdir()):
            if not directory.is_dir():
                continue
            cell, scene, agents = key_of(directory, "s")
            if cell not in wanted or scene not in ANCHOR:
                continue
            key = (arm, cell, scene, agents)
            if key in added:
                continue
            added.add(key)
            inner = [path for path in directory.iterdir() if path.is_dir()][0]
            tasks.append({"arm": arm, "cell": cell, "scene": scene,
                          "search_agents": agents, "directory": str(inner),
                          "source": "g_sweep"})
    for arm in ARCHIVED_ROOTS:
        logs = BASE / "execution/r_s_scaling" / args.axis / arm.replace("@", "_") / "world_logs"
        if not logs.is_dir():
            continue
        for directory in sorted(logs.iterdir()):
            if not directory.is_dir():
                continue
            cell, scene, agents = key_of(directory, args.axis)
            key = (arm, cell, scene, agents)
            if key in added:
                continue
            added.add(key)
            inner = [path for path in directory.iterdir() if path.is_dir()][0]
            tasks.append({"arm": arm, "cell": cell, "scene": scene,
                          "search_agents": agents, "directory": str(inner),
                          "source": "rs_%s" % args.axis})

    print("[%s] %d worlds, %d workers" % (args.axis, len(tasks), args.workers), flush=True)
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        for row in pool.map(worker, tasks):
            rows.append(row)
            done += 1
            if row["status"] != "ok":
                print("  FAILED %s %s s%d S%s: %s"
                      % (row["arm"], row["cell"], row["scene"], row["search_agents"],
                         row["error"]), flush=True)
            elif done % 200 == 0:
                print("  [%d/%d]" % (done, len(tasks)), flush=True)
    failures = [row for row in rows if row["status"] != "ok"]
    if failures:
        raise SystemExit("%d worlds failed; first: %s" % (len(failures), failures[0]))
    out = BASE / "execution/r_s_scaling" / args.axis / "WASTE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "stage5b-rs-waste/1",
        "axis": args.axis,
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
