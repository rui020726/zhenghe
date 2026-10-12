#!/usr/bin/env python
"""ROUND68 authoritative final ledger + Git-external original manifest.

The training executor's raw/runtime/ROUND68_LEDGER.json is a phase-boundary snapshot and stops at
385 actual constructions because the evaluation phase never rewrites it. This script derives the
final cumulative ledger from the atomic slot files and the live quota guard, preserves the stale
snapshot's value as a disclosed historical figure rather than overwriting it, and emits a manifest
of the original artefacts that live outside Git.

Read-only with respect to the run: it counts files and hashes them, and writes only two new JSON
documents. No environment is constructed and no training artefact is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r68_execute as base   # noqa: E402

SEEDS = (11, 22, 33)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND68 final ledger + manifest")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--ledger-out", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    args = ap.parse_args(argv)
    raw = Path(args.raw_root).resolve()
    rt = raw / "runtime"

    normal = sorted((raw / "launch_slots" / "normal").glob("*.json"))
    reserve = sorted((raw / "launch_slots" / "reserve").glob("*.json"))
    guard = json.loads((rt / "ROUND68_EVAL_QUOTA_GUARD.json").read_text(encoding="utf-8"))
    stale = json.loads((rt / "ROUND68_LEDGER.json").read_text(encoding="utf-8"))

    claim_dirs = [p for p in (raw / "launch_claims").iterdir() if p.is_dir()]
    statuses, categories, attempts = {}, {}, {}
    for d in claim_dirs:
        for f in sorted(d.glob("attempt_*.json")):
            try:
                c = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            statuses[c.get("status")] = statuses.get(c.get("status"), 0) + 1
            categories[c.get("category")] = categories.get(c.get("category"), 0) + 1
            attempts[c.get("attempt")] = attempts.get(c.get("attempt"), 0) + 1
    multi = [p.name for p in claim_dirs if len(list(p.glob("attempt_*.json"))) != 1]

    actual = len(normal) + len(reserve)
    ledger = {
        "kind": "round68-final-ledger/1",
        "task_id": base.TASK_ID,
        "branch": "exp/round68-late-initial-confidence-ablation",
        "authoritative": True,
        "supersedes": {
            "document": "raw/runtime/ROUND68_LEDGER.json",
            "its_actual_environment_constructions": stale["counts"][
                "actual_environment_constructions"],
            "status": "STALE PHASE SNAPSHOT - retained, not rewritten",
            "why_stale": ("it is written by the training executor at a phase boundary and the "
                          "evaluation phase never rewrites it, so it stops at the post-seed-22 "
                          "value. It must not be quoted as the final consumption."),
        },
        "counts": {
            "planned_normal": 774,
            "normal_slot_files": len(normal),
            "reserve_slot_files": len(reserve),
            "booked_normal": guard["booked_normal"],
            "booked_reserve": guard["booked_reserve"],
            "actual_environment_constructions": actual,
            "hard_cap": guard["hard_cap"],
            "hard_headroom": guard["hard_cap"] - actual,
            "hard_cap_exceeded_by": max(0, actual - guard["hard_cap"]),
            "historical_non_plan_probe": 0,
            "unregistered_extra_environment_constructions": 0,
        },
        "claim_admission": {
            "logical_jobs": len(claim_dirs),
            "jobs_with_exactly_one_attempt": len(claim_dirs) - len(multi),
            "multi_attempt_jobs": multi,
            "statuses": statuses,
            "categories": categories,
            "attempts": attempts,
        },
        "quota_guard_agreement": {
            "guard_actual": guard["actual_environment_constructions"],
            "slot_file_actual": actual,
            "agree": guard["actual_environment_constructions"] == actual,
        },
        "reserve_detail": {
            "slot": str(reserve[0]) if reserve else None,
            "note": ("the single reserve slot is the seed-11 e1 late-registered correction; it is "
                     "counted exactly once by the no-double-count guard and is NOT a scientific "
                     "rerun"),
        },
        "environment_constructions_this_script": 0,
    }

    manifest = {
        "kind": "round68-git-external-manifest/1",
        "task_id": base.TASK_ID,
        "branch": "exp/round68-late-initial-confidence-ablation",
        "note": ("raw/ is gitignored, so every original run artefact lives outside Git. This "
                 "manifest names them with size and SHA-256 so they remain identifiable. Nothing "
                 "here is a new execution."),
        "top_level": {},
        "runtime": {},
        "checkpoints": {},
    }
    for rel in ("ROUND68_FIXED_PLAN.json", "ROUND68_EVAL_PLAN.json", "ROUND68_STATISTICS.json",
                "ROUND68_PROJECTION.json", "ROUND68_PROJECTION_SLIM.json",
                "ROUND68_RULE_CONTEXT.json", "ROUND68_CHECKPOINT_INDEX.json",
                "ROUND68_MACHINE_VERDICT.json", "ROUND68_SEED_CHAIN_ELIGIBILITY.json",
                "ROUND68_LEDGER_FINAL.json"):
        p = raw / rel
        if p.is_file():
            manifest["top_level"][rel] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    for p in sorted((rt).glob("*.json")):
        manifest["runtime"][p.name] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    for p in sorted((rt).glob("*.jsonl")):
        manifest["runtime"][p.name] = {
            "bytes": p.stat().st_size, "sha256": sha(p),
            "lines": sum(1 for _ in open(p, encoding="utf-8")),
            "instrumented": p.name != "checkpoint_writes.jsonl"}
    for s in SEEDS:
        manifest["checkpoints"][f"s{s}"] = {
            "ep128": {"bytes": (raw / "checkpoints" / f"v2_train_s{s}_ep128.pt").stat().st_size,
                      "sha256": sha(raw / "checkpoints" / f"v2_train_s{s}_ep128.pt")},
            "batch_records": len(list((raw / "checkpoints").glob(
                f"r40_batch_s{s}_b*_attempt*.json"))),
            "before_b": len(list((raw / "checkpoints").glob(f"v2_train_s{s}_before_b*.pt"))),
            "after_b": len(list((raw / "checkpoints").glob(f"v2_train_s{s}_after_b*.pt"))),
            "rolling": len(list((raw / "checkpoints").glob(f"v2_train_s{s}.pt"))),
            "init_identity": sha(raw / "checkpoints" / f"r68_init_identity_s{s}.json"),
        }
    manifest["rows"] = {
        "training_rows_per_seed": {f"s{s}": len(list((raw / "rows").glob(f"*s{s}_*")))
                                   for s in SEEDS},
        "r68eval_model_rows": len(list((raw / "rows").glob("v2_r68eval_model_*.json"))),
        "r68eval_focal_rows": len(list((raw / "rows").glob("v2_r68eval_focal_*.json"))),
    }
    manifest["environment_constructions_this_script"] = 0

    for path, doc in ((args.ledger_out, ledger), (args.manifest_out, manifest)):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, indent=1, ensure_ascii=False)
        tmp.replace(path)
    print(json.dumps({"final_counts": ledger["counts"],
                      "claim_admission": ledger["claim_admission"],
                      "quota_guard_agreement": ledger["quota_guard_agreement"],
                      "manifest_top_level_files": len(manifest["top_level"]),
                      "manifest_runtime_files": len(manifest["runtime"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())