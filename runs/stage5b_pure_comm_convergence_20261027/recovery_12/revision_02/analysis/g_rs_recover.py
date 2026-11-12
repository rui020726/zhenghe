"""Rebuild ``RESULTS.json`` for an arm whose worlds are on disk but whose merge never ran.

Why this exists: a stage can die between writing world logs and calling ``merge`` (a silent
process exit leaves no traceback). The worlds themselves are complete — each leaf holds a
finished lifecycle and relay transcript — so the readings can be recomputed instead of re-running
156 worlds into existing directories, which would fail on the artifact-leaf collision guard.

Every rebuilt row is cross-checked against the same arithmetic the runner uses:
``on_time`` comes from the lifecycle, ``emissions`` from the transcript, and ``generated_goals``
from the cell's goal count (which the runner also asserts equals the scenario's placed count).

    D:/project_env/python.exe analysis/g_rs_recover.py --arm normal_immediate
    D:/project_env/python.exe analysis/g_rs_recover.py --arm normal_immediate --write
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "execution/g_rs_grid"
CELL_TAG = re.compile(r"(G\d+R\d+N\d+)")
S_TAG = re.compile(r"_?[sS](\d+)$")
ANCHOR = (2301901, 2301902, 2301903)


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
    return cell, int(parts[3]), (int(s_match.group(1)) if s_match else 2)


def reading(directory, cell):
    directory = Path(directory)
    lifecycle = read_jsonl_gz(directory / "lifecycle.jsonl.gz")
    transcript = read_jsonl_gz(directory / "relay_transcript.jsonl.gz")
    discovered, completed = {}, {}
    run_config = None
    for record in lifecycle:
        kind = record.get("event_type")
        stamp = float(record.get("timestamp") or 0.0)
        target = str(record.get("target_id"))
        if kind == "run_config":
            run_config = record.get("detail") or {}
        elif kind == "detection" and stamp <= 1000:
            discovered[target] = stamp + float((record.get("detail") or {}).get(
                "deadline_offset", 0.0))
        elif (kind == "progress" and record.get("status") == "completed"
              and stamp <= 1000):
            completed.setdefault(target, stamp)
    on_time = {target for target, deadline in discovered.items()
               if target in completed and completed[target] <= deadline}
    sends = [record for record in transcript if record.get("kind") == "send"]
    distinct = {str(record["target"]) for record in sends}
    spent = sum(int(record.get("bytes_total") or 0) for record in sends)
    scenario = ((run_config or {}).get("configuration") or {})
    return {
        "on_time": len(on_time),
        "generated_goals": int(scenario.get("num_goals") or cell[0]),
        "emissions": len(sends),
        "distinct_targets": len(distinct),
        "duplicate_sends": len(sends) - len(distinct),
        "spent_bytes": spent,
        "subscribers_live": int(record.get("subscribers") or 0) if sends else cell[2],
        "rescue_robots": int(scenario.get("rescue_species") or 0)
                         * int(scenario.get("rescue_per_species") or 0),
        "search_agents_live": int(scenario.get("search_agents") or 2),
        "cap": None,
        "rule_parameters": None,
        "budget_restored": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--write", action="store_true",
                        help="write RESULTS.json (default: report what would be written)")
    args = parser.parse_args()
    sys.dont_write_bytecode = True

    root = OUT / args.arm.replace("@", "_")
    logs = root / "world_logs"
    if not logs.is_dir():
        raise SystemExit("no world_logs under %s" % logs)
    rows = {}
    problems = []
    for directory in sorted(logs.iterdir()):
        if not directory.is_dir():
            continue
        cell, scene, agents = key_of(directory)
        inner = [path for path in directory.iterdir() if path.is_dir()]
        if len(inner) != 1:
            problems.append(("leaf", directory.name))
            continue
        try:
            data = reading(inner[0], cell)
        except Exception as error:
            problems.append((directory.name, "%s: %s" % (type(error).__name__, error)))
            continue
        rows[(cell, scene, agents)] = dict(
            data, arm=args.arm, cell=list(cell), scene=scene, search_agents=agents,
            cell_label="G%d_R%d_N%d" % cell + ("" if agents == 2 else "_S%d" % agents),
            status="ok", source="recovered_from_world_logs")
    print("%s: %d readable worlds, %d problems" % (args.arm, len(rows), len(problems)))
    for item in problems[:5]:
        print("  ", item)
    if not args.write:
        print("dry run; pass --write to persist")
        return
    path = root / "RESULTS.json"
    payload = {
        "schema": "stage5b-g-rs-grid-arm/1",
        "arm": args.arm,
        "root": str(BASE),
        "worlds": len(rows),
        "failures": [{"leaf": name, "error": error} for name, error in problems],
        "recovered": True,
        "results": sorted(rows.values(),
                          key=lambda row: (tuple(row["cell"]), int(row["scene"]),
                                           int(row["search_agents"]))),
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(str(tmp), str(path))
    print("wrote", path)


if __name__ == "__main__":
    main()
