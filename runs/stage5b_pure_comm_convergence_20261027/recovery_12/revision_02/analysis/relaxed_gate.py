"""Re-score the archived arms under the relaxed gate, from ``COMM_EFFICIENCY.json`` only.

No world is opened. The old gate demands, in every cell, ``on_time >= ceil(0.95 * immediate)``
**and** ``bytes < immediate``. The relaxed gate keeps the per-cell on-time clause and moves the
cost clause to the whole grid:

    per cell   on_time >= ceil(0.95 * immediate)
    global     total bytes / immediate bytes < 0.85
               broadcast waste ratio < 0.10

The unit of "bytes" is the same one every other table uses: the relay's billed total over the
fifteen training cells and their three anchor scenes (45 worlds per arm). Waste is the paper's
own definition (distinct broadcast targets that did not finish in time / distinct broadcast
targets), so the global clause is not a new metric, only a new place to apply it.

    D:/project_env/python.exe analysis/relaxed_gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
EFFICIENCY = BASE / "execution/COMM_EFFICIENCY.json"
OUT = BASE / "execution/relaxed_gate/RELAXED_GATE.json"

ARMS = ("v4f@200", "v4g@300", "v4s@300", "v4t@100", "v4u@100",
        "slack_heuristic_cap", "slack_matched_rate", "fifo_cap_k1", "dedup_immediate",
        "random_matched_rate", "normal_immediate")
BYTES_LIMIT = 0.85
WASTE_LIMIT = 0.10


def need(base):
    return (95 * int(base) + 99) // 100


def main():
    payload = json.loads(EFFICIENCY.read_text(encoding="utf-8"))
    per_arm = payload["per_arm"]
    base = per_arm["normal_immediate"]["cells_anchor"]
    cells = sorted(base)

    state = {"schema": "stage5b-relaxed-gate/1",
             "gate": {"per_cell": "on_time >= ceil(0.95 * immediate)",
                      "global": {"bytes_ratio_below": BYTES_LIMIT,
                                 "waste_below": WASTE_LIMIT}},
             "cells": cells, "arms": {}, "source": EFFICIENCY.name}

    for arm in ARMS:
        cells_arm = per_arm[arm]["cells_anchor"]
        per_cell = {}
        for cell in cells:
            own, ref = cells_arm[cell], base[cell]
            threshold = need(ref["on_time"])
            per_cell[cell] = {
                "on_time": own["on_time"], "immediate_on_time": ref["on_time"],
                "threshold": threshold, "on_time_ok": own["on_time"] >= threshold,
                "bytes": own["spent_bytes"], "immediate_bytes": ref["spent_bytes"],
                "bytes_ok": own["spent_bytes"] < ref["spent_bytes"],
                "bytes_equal": own["spent_bytes"] == ref["spent_bytes"],
            }
        bytes_ratio = (sum(cells_arm[c]["spent_bytes"] for c in cells)
                       / sum(base[c]["spent_bytes"] for c in cells))
        broadcast = sum(cells_arm[c]["distinct_broadcast"] for c in cells)
        waste = sum(cells_arm[c]["waste_n"] for c in cells)
        waste_ratio = waste / broadcast if broadcast else None
        old_pass = sum(1 for c in cells if per_cell[c]["on_time_ok"] and per_cell[c]["bytes_ok"])
        on_time_pass = sum(1 for c in cells if per_cell[c]["on_time_ok"])
        global_ok = bytes_ratio < BYTES_LIMIT and (waste_ratio is not None
                                                   and waste_ratio < WASTE_LIMIT)
        # Cells the old gate rejected on bytes alone while the on-time clause held.
        blocked_by_bytes = [c for c in cells
                            if per_cell[c]["on_time_ok"] and not per_cell[c]["bytes_ok"]]
        state["arms"][arm] = {
            "old_passes": old_pass,
            "on_time_passes": on_time_pass,
            "relaxed_passes": on_time_pass if global_ok else 0,
            "global_bytes_ratio": bytes_ratio,
            "global_waste_ratio": waste_ratio,
            "global_ok": global_ok,
            "blocked_by_bytes_only": blocked_by_bytes,
            "bytes_equal_cells": [c for c in cells if per_cell[c]["bytes_equal"]],
            "on_time": sum(cells_arm[c]["on_time"] for c in cells),
            "immediate_on_time": sum(base[c]["on_time"] for c in cells),
            "goals": sum(cells_arm[c]["goals"] for c in cells),
            "emissions": sum(cells_arm[c]["emissions"] for c in cells),
            "bytes": sum(cells_arm[c]["spent_bytes"] for c in cells),
            "immediate_bytes": sum(base[c]["spent_bytes"] for c in cells),
            "messages_per_on_time": (sum(cells_arm[c]["emissions"] for c in cells)
                                     / sum(cells_arm[c]["on_time"] for c in cells)),
            "per_cell": per_cell,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")

    print("| arm | old passes | on-time passes | global bytes | global waste | relaxed passes |")
    print("|---|---|---|---|---|---|")
    for arm in ARMS:
        record = state["arms"][arm]
        print("| %s | %d/15 | %d/15 | %.3f | %s | %s |" % (
            arm, record["old_passes"], record["on_time_passes"],
            record["global_bytes_ratio"],
            "n/a" if record["global_waste_ratio"] is None
            else "%.3f" % record["global_waste_ratio"],
            ("%d/15" % record["relaxed_passes"]) if record["global_ok"] else "0/15 (global fails)"))
    print()
    for arm in ARMS:
        record = state["arms"][arm]
        if record["blocked_by_bytes_only"]:
            print("%s blocked by bytes alone: %s" % (arm, ", ".join(record["blocked_by_bytes_only"])))
    print()
    print("wrote %s" % OUT.relative_to(BASE).as_posix())


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
