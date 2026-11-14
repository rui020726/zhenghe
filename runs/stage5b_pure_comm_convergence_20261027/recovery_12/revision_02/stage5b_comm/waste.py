"""Per-world broadcast waste, for the training loop's scan.

The scan already knows each cell's greedy success rate; experiment v4v additionally needs to
know **how much of what the policy put on the air was useless** (broadcast but did not finish
inside its deadline). That number is not in the runner row -- it needs the two logs the world
wrote -- so this module reads them at scan time, from the leaf the evaluation just produced.

Definition is the paper's, unchanged (``analysis/comm_efficiency.measure``): denominator is the
distinct targets put on the air, numerator those among them that are not in the on-time set.
Nothing here opens a world or touches a shared file.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

WINDOW = 1000


def read_jsonl_gz(path):
    with gzip.open(str(path), "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def world_waste(directory, window=WINDOW):
    """``{distinct_broadcast, waste_n, waste_ratio}`` for one finished world."""
    directory = Path(directory)
    lifecycle = read_jsonl_gz(directory / "lifecycle.jsonl.gz")
    transcript = read_jsonl_gz(directory / "relay_transcript.jsonl.gz")
    discovered, completed = {}, {}
    for record in lifecycle:
        target = str(record.get("target_id"))
        stamp = float(record.get("timestamp") or 0.0)
        kind = record.get("event_type")
        detail = record.get("detail") or {}
        if kind == "detection" and stamp <= window:
            discovered[target] = stamp + float(detail.get("deadline_offset", 0.0))
        elif kind == "progress" and stamp <= window and str(record.get("status")) == "completed":
            completed.setdefault(target, stamp)
    on_time = {target for target, due in discovered.items()
               if target in completed and completed[target] <= due}
    broadcast = {str(record["target"]) for record in transcript
                 if record.get("kind") == "send"}
    waste = broadcast - on_time
    return {"distinct_broadcast": len(broadcast), "waste_n": len(waste),
            "waste_ratio": (len(waste) / len(broadcast)) if broadcast else None}
