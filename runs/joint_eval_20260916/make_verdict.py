"""Diagnose the expired-task gap and archive the three-window verdict.

Reads the replay batch (rp_<arm>_s<seed>.jsonl + streams/) and writes
verdict.json next to this script, then prints a short report:

* the three-window paired table (from integrated_sar.lifecycle_metrics);
* per-target detail for every task that expired by window 1000, classified as
  barely-missed (last progress >= 0.8), stalled (0 < progress < 0.8), or
  never-started (no progress event) -- this separates a path-cost / deadline
  problem from an allocation problem;
* the overlap between the nocoord and rs0.25 expired sets (same tasks failing
  under both arms means the difficulty is not the allocator's choice);
* SHA256 of the two frozen checkpoints and the exact commands used.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from integrated_sar.lifecycle_metrics import collect, report  # noqa: E402

ARMS = ("nocoord", "rs0.25", "rs0.5")
SEEDS = list(range(10000, 10030))
WINDOWS = (500, 750, 1000)
PILOC_CKPT = REPO / "runs" / "cur_v2rep05_s2" / "best" / "shared_mappo_model_21_3100.pth"
CTDE_CKPT = HERE / "ctde_final_7_1.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def per_target(path: Path) -> dict[str, dict]:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out: dict[str, dict] = {}
    for e in events:
        kind = e.get("event_type")
        target = str(e.get("target_id"))
        ts = float(e.get("timestamp") or 0.0)
        if kind == "detection":
            detail = e.get("detail") or {}
            out[target] = {
                "detected_at": ts,
                "deadline": ts + float(detail.get("deadline_offset", 0.0)),
                "deadline_offset": float(detail.get("deadline_offset", 0.0)),
                "grid": list(detail.get("grid_position", [])),
                "completed_at": None,
                "last_progress": None,
                "last_progress_ts": None,
                "first_assignment": None,
                "assignments": 0,
            }
        elif kind == "progress" and target in out:
            if e.get("status") == "completed" and out[target]["completed_at"] is None:
                out[target]["completed_at"] = ts
            value = e.get("progress")
            if value is not None:
                out[target]["last_progress"] = float(value)
                out[target]["last_progress_ts"] = ts
        elif kind == "assignment" and target in out:
            out[target]["assignments"] += 1
            if out[target]["first_assignment"] is None:
                out[target]["first_assignment"] = ts
    return out


def classify(record: dict, window: int) -> str:
    if record["completed_at"] is not None and record["completed_at"] <= window:
        return "completed"
    if record["deadline"] > window:
        return "in_time_budget"
    last = record["last_progress"]
    if last is None:
        return "never_started"
    if last >= 0.8:
        return "barely_missed"
    return "stalled"


def main() -> int:
    logs = {arm: {s: HERE / f"rp_{arm}_s{s}.jsonl" for s in SEEDS} for arm in ARMS}
    missing = [str(p) for arm in ARMS for s, p in logs[arm].items() if not p.is_file()]
    if missing:
        print("missing logs:", missing[:5], "...")
        return 1

    table = collect([str(p) for arm in ARMS for p in logs[arm].values()], WINDOWS)
    text_report = report(table, WINDOWS, "nocoord")
    print(text_report)

    # per-target diagnosis at window 1000
    classes: dict[str, dict[str, int]] = {arm: defaultdict(int) for arm in ARMS}
    expired_sets: dict[str, set[tuple[int, str]]] = {arm: set() for arm in ARMS}
    offsets_expired: list[float] = []
    offsets_completed: list[float] = []
    for arm in ARMS:
        for seed in SEEDS:
            for target, record in per_target(logs[arm][seed]).items():
                label = classify(record, 1000)
                classes[arm][label] += 1
                if label in ("never_started", "barely_missed", "stalled"):
                    expired_sets[arm].add((seed, target))
                    offsets_expired.append(record["deadline_offset"])
                elif label == "completed":
                    offsets_completed.append(record["deadline_offset"])

    both = expired_sets["nocoord"] & expired_sets["rs0.25"]
    only_nocoord = expired_sets["nocoord"] - expired_sets["rs0.25"]
    only_rs025 = expired_sets["rs0.25"] - expired_sets["nocoord"]
    print("\n== expired diagnosis @ window 1000 ==")
    for arm in ARMS:
        c = classes[arm]
        total = sum(c.values())
        print(
            f"{arm:<8} completed={c['completed']} barely_missed={c['barely_missed']} "
            f"stalled={c['stalled']} never_started={c['never_started']} "
            f"in_time_budget={c['in_time_budget']} (total {total})"
        )
    print(f"expired overlap nocoord&rs0.25 = {len(both)}, "
          f"only-nocoord(rescued by coord) = {len(only_nocoord)}, "
          f"only-rs0.25(broken by coord) = {len(only_rs025)}")
    if offsets_expired:
        print(f"deadline_offset mean: expired={sum(offsets_expired)/len(offsets_expired):.1f} "
              f"(n={len(offsets_expired)}), completed={sum(offsets_completed)/len(offsets_completed):.1f} "
              f"(n={len(offsets_completed)})")

    verdict = {
        "stage": "joint-coordinator-ablation (replay, corrected metrics)",
        "date": "2026-09-16",
        "git_commit": _git_commit(),
        "frozen_checkpoints": {
            "piloc": {
                "path": str(PILOC_CKPT.relative_to(REPO)).replace("\\", "/"),
                "sha256": sha256(PILOC_CKPT),
                "note": "cur_v2rep05_s2 best @ episode 3100 (best independent-test discovery)",
            },
            "ctde": {
                "path": "runs/joint_eval_20260916/ctde_final_7_1.pth",
                "sha256": sha256(CTDE_CKPT),
                "note": "task-03 checkpoints_old/7/1/ctde_final.pth @ episode 20002, actor_kv_mode=tokens",
            },
        },
        "protocol": {
            "seeds": SEEDS,
            "steps_recorded": 1000,
            "windows": list(WINDOWS),
            "arms": list(ARMS),
            "search_backend": "nocoord arm records the live PILOC stream; rs0.25/rs0.5 replay it",
            "consistency": "30 seeds x 3 arms, detections + map hash identical, 0 failures",
            "causality": "1000-step prefix @500 == independent 500-step run (verified)",
        },
        "three_window_table": text_report,
        "expired_diagnosis_window1000": {
            arm: dict(classes[arm]) for arm in ARMS
        },
        "expired_overlap": {
            "nocoord_and_rs0.25": len(both),
            "only_nocoord": len(only_nocoord),
            "only_rs0.25": len(only_rs025),
        },
        "commands": {
            "record_nocoord": (
                "python -m integrated_sar.joint_driver --piloc-checkpoint <piloc> "
                "--task-checkpoint <ctde> --actor-kv-mode tokens --steps 1000 --seed <s> "
                "--no-coordinator --quiet --record-stream streams/s<s>.json "
                "--output rp_nocoord_s<s>.jsonl"
            ),
            "replay_arm": (
                "python -m integrated_sar.joint_driver --task-checkpoint <ctde> "
                "--actor-kv-mode tokens --steps 1000 --seed <s> --replay-stream streams/s<s>.json "
                "--residual-scale <a> --quiet --output rp_rs<a>_s<s>.jsonl"
            ),
            "audit": "python -m integrated_sar.replay_audit --logs rp_*_s*.jsonl --bundles streams/s*.json",
            "metrics": "python -m integrated_sar.lifecycle_metrics rp_*_s*.jsonl --windows 500 750 1000",
        },
    }
    (HERE / "verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nverdict written to {HERE / 'verdict.json'}")
    return 0


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
