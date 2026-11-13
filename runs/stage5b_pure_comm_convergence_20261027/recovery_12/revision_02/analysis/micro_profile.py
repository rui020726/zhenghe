"""Experiment L: the micro time-space decision profile, from the archived 120-world logs only.

Two questions, both answerable without opening a world:

(a) *When* does each arm send, and does the timing matter? For every 100-tick bucket, count the
    distinct targets whose **first** broadcast fell in that bucket and the share of them that
    finished inside their own deadline. A target sent early and one sent late are different
    bets; this is the reading that shows which arm makes which bet.

(b) Which targets does v4t decline to broadcast? Compare the set it put on the air with the set
    it never mentioned, on the three quantities a deadline-aware filter should act on:
    discovery tick, the deadline slack available at discovery, and the task's stage count.

The "remaining time" here is measured **at discovery**, for sent and suppressed targets alike,
because that is the only moment both sets share in the archive. Measuring it at the send tick
would compare a broadcast target's later time with a suppressed target's discovery time and
would manufacture a difference by construction. The send-time slack is reported separately for
the broadcast set, and is labelled as such.

Nothing is re-run and no shared file is read for RNG: the two inputs are the gzipped lifecycle
and relay transcripts this revision already published.

    D:/project_env/python.exe analysis/micro_profile.py --workers 8
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
OUT = BASE / "execution/extended/micro/MICRO.json"
CELL_TAG = re.compile(r"(G\d+R\d+N\d+)")
BUCKET = 100
STEPS = 1000
#: Every event inside the episode counts; the same 1000-step window the paper's other readings
#: use. A target detected at tick 900 is inside it, so the detected set is the full generated set.
WINDOW = STEPS

SOURCES = {
    "normal_immediate": ("execution/baselines/grid_v2", "eval__normal_immediate_", True),
    "v4t@100": ("execution/train_v4t_cap_frozen/eval_100_s8", "eval__", False),
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
        raise RuntimeError("%s is not a four-part world leaf name" % (name,))
    match = CELL_TAG.search(parts[1])
    if match is None:
        raise RuntimeError("%s carries no cell tag" % (name,))
    return tuple(int(value) for value in re.findall(r"\d+", match.group(1))), int(parts[3])


def _quantile(sorted_values, fraction):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def _describe(values):
    ordered = sorted(float(value) for value in values if value is not None)
    if not ordered:
        return {"n": 0, "mean": None, "p25": None, "median": None, "p75": None}
    return {"n": len(ordered), "mean": sum(ordered) / len(ordered),
            "p25": _quantile(ordered, 0.25), "median": _quantile(ordered, 0.50),
            "p75": _quantile(ordered, 0.75)}


def measure(directory):
    """One world: bucket table, plus the sent / suppressed target contrast."""
    directory = Path(directory)
    lifecycle = read_jsonl_gz(directory / "lifecycle.jsonl.gz")
    transcript = read_jsonl_gz(directory / "relay_transcript.jsonl.gz")

    discovery, deadline, stages, completed = {}, {}, {}, {}
    for record in lifecycle:
        target = str(record.get("target_id"))
        stamp = float(record.get("timestamp") or 0.0)
        kind = record.get("event_type")
        detail = record.get("detail") or {}
        if kind == "detection" and stamp <= WINDOW:
            discovery[target] = stamp
            deadline[target] = stamp + float(detail.get("deadline_offset", 0.0))
            stages[target] = int(detail.get("total_stages", 0) or 0)
        elif kind == "progress" and stamp <= WINDOW and str(record.get("status")) == "completed":
            completed.setdefault(target, stamp)

    on_time = {target for target, due in deadline.items()
               if target in completed and completed[target] <= due}

    first_send = {}
    for record in transcript:
        if record.get("kind") != "send":
            continue
        target = str(record.get("target"))
        step = float(record.get("step"))
        if target not in first_send or step < first_send[target]:
            first_send[target] = step

    buckets = []
    for index in range(STEPS // BUCKET):
        low, high = index * BUCKET, (index + 1) * BUCKET
        targets = [target for target, step in first_send.items() if low <= step < high]
        buckets.append({"bucket": index, "low": low, "high": high,
                        "first_sends": len(targets),
                        "on_time": sum(1 for target in targets if target in on_time)})

    broadcast = set(first_send) & set(discovery)
    suppressed = set(discovery) - broadcast

    def block(targets):
        return {
            "n": len(targets),
            "discovery_tick": _describe(discovery.get(t) for t in targets),
            "slack_at_discovery": _describe(deadline.get(t, 0.0) - discovery.get(t, 0.0)
                                            for t in targets),
            "total_stages": _describe(stages.get(t) for t in targets),
            "on_time": sum(1 for t in targets if t in on_time),
            "on_time_rate": (sum(1 for t in targets if t in on_time) / len(targets)
                             if targets else None),
        }

    return {
        "cell": list(key_of(directory.parent)[0]),
        "scene": int(key_of(directory.parent)[1]),
        "generated": len(discovery),
        "emissions": sum(1 for record in transcript if record.get("kind") == "send"),
        "on_time_total": len(on_time),
        "buckets": buckets,
        "broadcast": block(broadcast),
        "suppressed": block(suppressed),
        "send_slack": _describe(deadline.get(t, 0.0) - first_send[t] for t in broadcast),
        "send_delay": _describe(first_send[t] - discovery.get(t, 0.0) for t in broadcast),
    }


def worker(task):
    sys.dont_write_bytecode = True
    try:
        return dict(task, status="ok", **measure(task["directory"]))
    except Exception as error:
        return dict(task, status="error", error="%s: %s" % (type(error).__name__, error))


def group(worlds, label):
    buckets = []
    for index in range(STEPS // BUCKET):
        sends = sum(w["buckets"][index]["first_sends"] for w in worlds)
        on_time = sum(w["buckets"][index]["on_time"] for w in worlds)
        buckets.append({"bucket": index, "low": index * BUCKET, "high": (index + 1) * BUCKET,
                        "worlds": len(worlds), "first_sends": sends,
                        "first_sends_per_world": sends / len(worlds) if worlds else None,
                        "on_time": on_time,
                        "on_time_rate": on_time / sends if sends else None})
    broadcast = {"n": sum(w["broadcast"]["n"] for w in worlds),
                 "on_time": sum(w["broadcast"]["on_time"] for w in worlds)}
    suppressed = {"n": sum(w["suppressed"]["n"] for w in worlds),
                  "on_time": sum(w["suppressed"]["on_time"] for w in worlds)}
    merged = {}
    for key in ("discovery_tick", "slack_at_discovery", "total_stages"):
        values = {"broadcast": [], "suppressed": []}
        for world in worlds:
            values["broadcast"].extend(world["broadcast"].get("_raw", {}).get(key, []))
            values["suppressed"].extend(world["suppressed"].get("_raw", {}).get(key, []))
        merged[key] = {side: _describe(partition) for side, partition in values.items()}
    return {
        "label": label,
        "worlds": len(worlds),
        "generated": sum(w["generated"] for w in worlds),
        "emissions": sum(w["emissions"] for w in worlds),
        "on_time": sum(w["on_time_total"] for w in worlds),
        "buckets": buckets,
        "broadcast": dict(broadcast,
                          on_time_rate=(broadcast["on_time"] / broadcast["n"]
                                        if broadcast["n"] else None)),
        "suppressed": dict(suppressed,
                           on_time_rate=(suppressed["on_time"] / suppressed["n"]
                                         if suppressed["n"] else None)),
        "contrast": merged,
        "send_slack": _describe(
            value for world in worlds for value in world.get("_send_slack", [])),
        "send_delay": _describe(
            value for world in worlds for value in world.get("_send_delay", [])),
    }


def carry_raw(world, raw):
    """Attach the per-target values the grouped description needs, then strip them."""
    world["broadcast"]["_raw"] = raw["broadcast"]
    world["suppressed"]["_raw"] = raw["suppressed"]
    world["_send_slack"] = raw["send_slack"]
    world["_send_delay"] = raw["send_delay"]
    return world


def raw_lists(directory):
    """Per-target values for both sets, so the group can describe them without re-reading."""
    directory = Path(directory)
    lifecycle = read_jsonl_gz(directory / "lifecycle.jsonl.gz")
    transcript = read_jsonl_gz(directory / "relay_transcript.jsonl.gz")
    discovery, deadline, stages = {}, {}, {}
    for record in lifecycle:
        target = str(record.get("target_id"))
        stamp = float(record.get("timestamp") or 0.0)
        if record.get("event_type") != "detection" or stamp > WINDOW:
            continue
        detail = record.get("detail") or {}
        discovery[target] = stamp
        deadline[target] = stamp + float(detail.get("deadline_offset", 0.0))
        stages[target] = int(detail.get("total_stages", 0) or 0)
    first_send = {}
    for record in transcript:
        if record.get("kind") != "send":
            continue
        target, step = str(record.get("target")), float(record.get("step"))
        if target not in first_send or step < first_send[target]:
            first_send[target] = step
    broadcast = set(first_send) & set(discovery)
    suppressed = set(discovery) - broadcast

    def collect(targets):
        return {
            "discovery_tick": [discovery[t] for t in sorted(targets)],
            "slack_at_discovery": [deadline[t] - discovery[t] for t in sorted(targets)],
            "total_stages": [stages[t] for t in sorted(targets)],
        }

    return {"broadcast": collect(broadcast), "suppressed": collect(suppressed),
            "send_slack": [deadline.get(t, 0.0) - first_send[t] for t in sorted(broadcast)],
            "send_delay": [first_send[t] - discovery.get(t, 0.0) for t in sorted(broadcast)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sys.dont_write_bytecode = True

    tasks = []
    for arm, (root, prefix, shared) in SOURCES.items():
        logs = BASE / root / "world_logs"
        directories = sorted(path for path in logs.iterdir() if path.is_dir())
        if shared:
            directories = [path for path in directories if path.name.startswith(prefix)]
        for directory in directories:
            inner = [path for path in directory.iterdir() if path.is_dir()]
            if len(inner) != 1:
                raise RuntimeError("%s holds %d inner directories" % (directory, len(inner)))
            cell, scene = key_of(directory)
            tasks.append({"arm": arm, "cell": list(cell), "scene": scene,
                          "directory": str(inner[0]),
                          "raw": str(inner[0])})

    print("%d worlds over %d arms" % (len(tasks), len({t["arm"] for t in tasks})), flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        for row in pool.map(worker, tasks):
            results.append(row)
    failures = [row for row in results if row["status"] != "ok"]
    if failures:
        raise SystemExit("%d worlds failed; first: %s" % (len(failures), failures[0]))

    payload = {"schema": "stage5b-micro-profile/1", "bucket_ticks": BUCKET,
               "window": WINDOW, "arms": {},
               "inputs": {arm: {"root": root, "worlds": sum(1 for row in results
                                                           if row["arm"] == arm)}
                          for arm, (root, _prefix, _shared) in SOURCES.items()},
               "sha256": {}}
    for arm in SOURCES:
        worlds = [row for row in results if row["arm"] == arm]
        for world in worlds:
            carry_raw(world, raw_lists(Path(world["directory"])))
        payload["arms"][arm] = {
            "all": group(worlds, "all(8)"),
            "R=6": group([w for w in worlds if int(w["cell"][1]) == 6], "R=6"),
            "R=12": group([w for w in worlds if int(w["cell"][1]) == 12], "R=12"),
        }
        payload["arms"][arm]["per_cell"] = {
            "G%d_R%d_N%d" % tuple(w["cell"]): {"worlds": 1, "scene": w["scene"],
                                               "sent": w["broadcast"]["n"],
                                               "suppressed": w["suppressed"]["n"],
                                               "on_time": w["on_time_total"]}
            for w in worlds}

    for arm, (root, _prefix, _shared) in SOURCES.items():
        path = BASE / root / "world_logs"
        payload["sha256"][arm] = {"world_logs": str(path)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(str(tmp), str(OUT))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
