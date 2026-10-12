#!/usr/bin/env python
"""Emit the SLIM numeric copy of an already-generated ROUND68 projection.

Pure extraction: reads raw/ROUND68_PROJECTION.json and writes a compact copy holding only the
scene-ordered numbers. No statistics function, forward, model or environment is invoked. The full
projection (with per-cell paths and row hashes) stays on disk untouched; only the slim copy is
echoed so the observer can audit the arithmetic without a large log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def triples(cells: dict) -> list:
    """32 (or 6) scene-ordered [Y, C, mask] rows, sorted by scene."""
    return [[c["on_time"], c["spent_bytes"], c["mask_bin"]]
            for _, c in sorted(cells.items(), key=lambda kv: kv[1]["scene"])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND68 slim projection export")
    ap.add_argument("--projection", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    src = Path(args.projection).resolve()
    d = json.loads(src.read_text(encoding="utf-8"))

    groups: dict[str, list] = {}
    tmp: dict[str, dict] = {}
    for key, cell in d["model_cells"].items():
        arm, seed, mode, scene = key.split("|")
        tmp.setdefault(f"{arm}_s{seed}_{mode}", {})[scene] = cell
    for g, cells in tmp.items():
        groups[g] = triples(cells)
    for key, cell in d["rule_cells"].items():
        arm, scene = key.split("|")
        tmp.setdefault(f"RULE_{arm}", {})[scene] = cell
    for g in [g for g in tmp if g.startswith("RULE_")]:
        groups[g] = triples(tmp[g])

    slim = {
        "kind": "round68-projection-slim/1",
        "source_projection": str(src),
        "source_projection_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "scene_order": [c["scene"] for _, c in
                        sorted(d["model_cells"].items(), key=lambda kv: kv[1]["scene"])
                        if _.split("|")[0] == "INIT_ALPHA" and _.split("|")[1] == "11"
                        and _.split("|")[2] == "greedy"],
        "row_format": "[on_time, spent_bytes, mask_bin]",
        "target_slots": d["target_slots"],
        "bootstrap": d["bootstrap"],
        "groups": groups,
        "focal_scene_1700027": {k: [c["on_time"], c["spent_bytes"], c["mask_bin"]]
                                for k, c in sorted(d["focal_cells"].items())},
        "primary_trained_minus_init": {
            k: {"task_mean": v["task_delta"]["mean"], "task_ci95": v["task_delta"]["ci95"],
                "task_lost": v["task_delta"]["lost_count"],
                "task_gained": v["task_delta"]["gained_count"],
                "task_gained_cells": v["task_delta"]["gained"],
                "cost_mean": v["cost_delta_B"]["mean"], "cost_ci95": v["cost_delta_B"]["ci95"],
                "trained_on_time_total": v["trained_on_time_total"],
                "init_on_time_total": v["init_on_time_total"],
                "trained_bytes_per_scene": v["trained_bytes_per_scene"],
                "init_bytes_per_scene": v["init_bytes_per_scene"]}
            for k, v in d["primary_trained_minus_init"].items()},
        "rule_absolute": d["rule_absolute"],
        "budget": d["budget"],
        "training_journals": d["training_journals"],
        "cross_check_all_match": d.get("cross_check_all_match"),
        "bootstrap_index_sha_match": d.get("bootstrap_index_sha_match"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmpf = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(tmpf, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(slim, fh, separators=(",", ":"), ensure_ascii=False)
    tmpf.replace(args.out)
    sys.stdout.write(json.dumps(slim, separators=(",", ":"), ensure_ascii=False))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())