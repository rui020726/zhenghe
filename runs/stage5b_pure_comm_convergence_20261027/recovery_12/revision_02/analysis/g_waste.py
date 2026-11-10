"""Per-world broadcast-waste readings for the G sweep, from the archived and new world logs.

The waste ratio needs the lifecycle (which targets finished on time) and the relay transcript
(which distinct targets went on the air), so it cannot come from the runner's RESULT rows. This
reads both logs for every (arm, cell, scene) the sweep uses and writes one table; the report
joins it with the rest of the readings.

    D:/project_env/python.exe analysis/g_waste.py --workers 8
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
OUT = BASE / "execution/g_scaling/WASTE.json"
CELL_TAG = re.compile(r"(G\d+R\d+N\d+)")

#: arm -> the world-log roots that hold its archived worlds (the 15 training cells).
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
    return tuple(int(value) for value in re.findall(r"\d+", match.group(1))), int(parts[3])


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
                "waste_n": waste_n, "waste_d": waste_d, "status": "ok",
                "source": task["source"]}
    except Exception as error:
        return {"arm": task["arm"], "cell": list(task["cell"]), "scene": task["scene"],
                "status": "error", "error": "%s: %s" % (type(error).__name__, error),
                "source": task["source"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    sys.dont_write_bytecode = True

    tasks = []
    for arm, (root, prefix) in ARCHIVED_ROOTS.items():
        logs = BASE / root / "world_logs"
        if not logs.is_dir():
            print("skip archived %s (no %s)" % (arm, logs), flush=True)
            continue
        for directory in sorted(logs.iterdir()):
            if not directory.is_dir() or not directory.name.startswith(prefix):
                continue
            inner = [path for path in directory.iterdir() if path.is_dir()]
            if len(inner) != 1:
                raise RuntimeError("%s holds %d inner directories" % (directory, len(inner)))
            cell, scene = key_of(directory)
            if cell[0] not in (12, 24, 48):
                continue
            tasks.append({"arm": arm, "cell": cell, "scene": scene,
                          "directory": str(inner[0]),
                          "source": "archived:%s" % root})
    for arm in ARCHIVED_ROOTS:
        logs = BASE / "execution/g_scaling" / arm.replace("@", "_") / "world_logs"
        if not logs.is_dir():
            print("skip new %s (no %s)" % (arm, logs), flush=True)
            continue
        for directory in sorted(logs.iterdir()):
            if not directory.is_dir():
                continue
            inner = [path for path in directory.iterdir() if path.is_dir()]
            if len(inner) != 1:
                raise RuntimeError("%s holds %d inner directories" % (directory, len(inner)))
            cell, scene = key_of(directory)
            tasks.append({"arm": arm, "cell": cell, "scene": scene,
                          "directory": str(inner[0]), "source": "sweep"})

    print("%d worlds, %d workers" % (len(tasks), args.workers), flush=True)
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        for row in pool.map(worker, tasks):
            rows.append(row)
            done += 1
            if row["status"] != "ok":
                print("  FAILED %s %s s%d: %s"
                      % (row["arm"], row["cell"], row["scene"], row["error"]), flush=True)
            elif done % 200 == 0:
                print("  [%d/%d]" % (done, len(tasks)), flush=True)
    failures = [row for row in rows if row["status"] != "ok"]
    if failures:
        raise SystemExit("%d worlds failed; first: %s" % (len(failures), failures[0]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "stage5b-g-waste/1",
        "root": str(BASE),
        "worlds": len(rows),
        "rule": "waste = distinct broadcast targets that did not finish inside their deadline"
                " / distinct broadcast targets",
        "results": sorted(rows, key=lambda row: (row["arm"], tuple(row["cell"]),
                                                 int(row["scene"]))),
    }
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(str(tmp), str(out))
    print("wrote %s  sha256 %s" % (out, sha256(out)))


if __name__ == "__main__":
    main()
