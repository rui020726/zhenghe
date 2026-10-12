#!/usr/bin/env python
"""ROUND68 read-only rule-context post-processor.

Adds the immediate / late / always_hold absolute task and metered-cost readings over the SAME
32 scenes (1800000..1800031) that ROUND68's frozen statistics use, together with the exact
on-time target-ID sets, so the closure does not report the model arms against nothing.

Constraints honoured by construction:
  * read-only: no environment is constructed, no model is loaded, no PPO/optimizer call is made,
    no checkpoint is written, no claim or slot is consumed;
  * it does NOT modify r68_eval.py / r68_execute.py / r68_frozen.py; it only imports them and
    reuses their qualified-claim row loader, evidenced target-id rebuild and frozen paired
    bootstrap index verbatim, so the interval method and index are unchanged;
  * rule rows are ROUND66's own, identified by claim eligibility and row SHA, never by the
    literal label; no new rule simulation is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r68_eval as ev  # noqa: E402  read-only reuse of the frozen loaders / bootstrap index

RULE_TAG = "r66eval2"
RULE_PREFIX = RULE_TAG + "_rule_"
RULE_ARMS = ("IMMEDIATE", "LATE_DISCOVERY_GATE", "ALWAYS_HOLD")
SCENES = ev.SCENES_NEW


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rule_rows() -> tuple[dict[tuple[str, int], dict], dict]:
    """Qualified ROUND66 rule rows keyed (ARM, scene). Refuses to guess when a cell is ambiguous."""
    r66_raw = ev.base.R66_DIR / "raw"
    qualified = ev._qualified_row_paths(r66_raw, RULE_TAG)
    rows: dict[tuple[str, int], dict] = {}
    identity: dict[str, Any] = {}
    skipped: list[dict] = []
    for logical in sorted(qualified):
        if not logical.startswith(RULE_PREFIX):
            continue
        paths = qualified[logical]
        if len(paths) != 1:
            raise RuntimeError(f"{logical}: {len(paths)} sound complete receipts name different rows "
                               f"({paths}); refusing to choose one by order")
        row = ev.base._read_json(Path(paths[0]))
        arm = str(row.get("model_arm", "")).upper()
        scene = row.get("scenario_seed")
        if arm not in RULE_ARMS:
            skipped.append({"logical_job": logical, "reason": "arm_not_a_rule_arm", "arm": arm})
            continue
        if not isinstance(scene, int) or scene not in SCENES:
            skipped.append({"logical_job": logical, "reason": "scene_not_in_used_block",
                            "scene": scene})
            continue
        key = (arm, int(scene))
        if key in rows:
            raise RuntimeError(f"{logical}: duplicate rule cell {key}")
        # Same projection as the frozen loader: on_time / spent_bytes / the evidenced on_time_ids,
        # so the shared _paired_delta sees exactly the shape it expects. The raw row is never
        # mutated and no field is invented: on_time_ids comes from on_time_target_ids().
        rows[key] = {"on_time": int(row["on_time"]),
                     "spent_bytes": int(row["spent_bytes"]),
                     "on_time_ids": ev.on_time_target_ids(row, int(scene)),
                     "row_path": paths[0], "logical_job": logical}
        identity[f"{arm}/{scene}"] = {
            "logical_job": logical,
            "row_path": paths[0],
            "row_sha256": _sha256(Path(paths[0])),
            "behaviour_sha256": row.get("behaviour_sha256"),
            "training_version": row.get("training_version"),
            "policy": row.get("policy"),
            "eval_mode": row.get("eval_mode"),
            "attempt": row.get("attempt"),
        }
    return rows, {"source_root": str(r66_raw), "row_tag": RULE_TAG,
                  "qualified_rule_cells": len(rows), "cells": identity, "skipped": skipped}


def rule_summary(rows: dict[tuple[str, int], dict]) -> dict:
    out: dict[str, Any] = {}
    for arm in RULE_ARMS:
        missing = [s for s in SCENES if (arm, s) not in rows]
        if missing:
            out[arm] = {"unavailable": f"{len(missing)} scenes missing", "missing_scenes": missing[:8]}
            continue
        cells = [rows[(arm, s)] for s in SCENES]
        on_time_ids = {s: rows[(arm, s)]["on_time_ids"] for s in SCENES}
        union = sorted({t for ids in on_time_ids.values() for t in ids})
        out[arm] = {
            "scenes": len(SCENES),
            "on_time_total": int(sum(c["on_time"] for c in cells)),
            "on_time_per_scene_mean": float(sum(c["on_time"] for c in cells) / len(SCENES)),
            "spent_bytes_total": int(sum(c["spent_bytes"] for c in cells)),
            "spent_bytes_per_scene_mean": float(
                sum(c["spent_bytes"] for c in cells) / len(SCENES)),
            "on_time_target_ids_union": union,
            "on_time_target_ids_union_count": len(union),
            "on_time_target_ids_per_scene": {str(s): on_time_ids[s] for s in SCENES},
            "policy": sorted({str(c.get("policy")) for c in cells}),
            "eval_mode": sorted({str(c.get("eval_mode")) for c in cells}),
        }
    return out


def build(raw_root: Path) -> dict:
    idx, idx_sha = ev._bootstrap_index()
    rule_rows, rule_audit = load_rule_rows()
    r68_rows, r68_audit = ev._load_r68_model_rows(Path(raw_root))
    scenes = list(SCENES)

    per_rule = rule_summary(rule_rows)
    # Context only: each ROUND68 arm against each rule arm, same frozen paired method/index.
    context: dict[str, Any] = {}
    for seed in ev.SEEDS:
        for mode in ev.MODES:
            for rule_arm in RULE_ARMS:
                if any((rule_arm, s) not in rule_rows for s in scenes):
                    continue
                if any((a, seed, mode, s) not in r68_rows for a in ev.MODEL_ARMS for s in scenes):
                    continue
                for arm in ev.MODEL_ARMS:
                    context[f"{arm}_s{seed}_{mode}_vs_{rule_arm}"] = ev._paired_delta(
                        lambda s, a=arm, sd=seed, md=mode: r68_rows[(a, sd, md, s)],
                        lambda s, ra=rule_arm: rule_rows[(ra, s)],
                        scenes, idx, arm, rule_arm)
    return {
        "kind": "round68-rule-context/1",
        "task_id": ev.base.TASK_ID,
        "branch": "exp/round68-late-initial-confidence-ablation",
        "eval_version": ev.EVAL_VERSION,
        "training_version": ev.r68.R68_TRAINING_VERSION,
        "purpose": ("absolute immediate/late/always_hold task and metered cost over the same used "
                    "development block 1800000..1800031, plus exact on-time target-ID sets"),
        "block_status": "1800000..1800031 is an ALREADY-USED development block, not a confirmation set",
        "rule_audit": rule_audit,
        "rule_absolute": per_rule,
        "arm_vs_rule_context": context,
        "bootstrap_index_sha256": idx_sha,
        "bootstrap_index_source": "reused verbatim from r68_eval._bootstrap_index (seed 20261012, 10000x32)",
        "environment_constructions": 0,
        "model_forwards": 0,
        "ppo_or_optimizer_calls": 0,
        "checkpoint_writes": 0,
        "new_rule_simulations": 0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND68 read-only rule-context post-processor")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    result = build(Path(args.raw_root).resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    tmp.replace(args.out)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("rule_absolute", "arm_vs_rule_context", "rule_audit")},
                     indent=1))
    for arm, s in result["rule_absolute"].items():
        if "unavailable" in s:
            print(arm, "UNAVAILABLE", s["unavailable"])
        else:
            print(f"{arm}: on_time_total={s['on_time_total']} "
                  f"bytes/scene={s['spent_bytes_per_scene_mean']:.1f} "
                  f"targets_union={s['on_time_target_ids_union_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())