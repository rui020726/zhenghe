"""Every number this round quotes, with its unit and its closure.

Three separate quantities kept getting read as one: the pre-registered stress block (180
paired scenarios, 360 replays), the 17 stress episodes that actually contained one of the 19
harmful releases, and the 15 development episodes. This script prints what each cell covers,
in units, and closes every target count so a ratio cannot hide a residue.

    D:/project_env/python.exe runs/corrected_env_final_20260918/metric_accounting.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (REPO, HERE, REPO / "runs" / "membership_consistency_20260918",
             REPO / "runs" / "release_validation_20260918"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

import membership_measurement as mm  # noqa: E402
import ghost_recheck  # noqa: E402

BLOCKS = ("dev", "stress")


def one_block(kind: str) -> dict:
    # The reference detection counts come from the four-cell summary of the previous round:
    # the closure statement is about those published ratios, not about a fresh reading.
    reference = json.loads(
        (REPO / "runs" / "membership_consistency_20260918" / f"CELL_SUMMARY_{kind}.json").read_text(
            encoding="utf-8"
        )
    )
    rows: dict[str, list[dict]] = {}
    for cell in mm.CELLS:
        for path in sorted((ghost_recheck.OUT / cell).glob(f"recheck_{cell}_s*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["scenario"] == kind or (kind == "stress" and payload["scenario"] != "dev"):
                rows.setdefault(cell, []).append(payload)
    table = {}
    for cell, values in sorted(rows.items()):
        if not values:
            continue
        reasons: Counter[str] = Counter()
        logged: Counter[str] = Counter()
        statuses: Counter[str] = Counter()
        on_time = expired = unresolved = 0
        for payload in values:
            records = load_records(
                ghost_recheck.OUT / cell / f"{cell}_rp_s{payload['seed']}.jsonl"
            )
            timelines = build_timelines(records, arm=cell, seed=payload["seed"])
            for timeline in timelines.values():
                if timeline.final_progress >= 1.0 and timeline.final_failure_reason != "deadline_expired":
                    on_time += 1
                elif timeline.final_failure_reason == "deadline_expired":
                    expired += 1
                else:
                    unresolved += 1
                    statuses[str(timeline.final_failure_reason or "still_running")] += 1
            logged["assignment_rows"] += sum(
                1 for row in records if row.get("event_type") == "assignment"
            )
            logged["rejection_rows"] += sum(
                1 for row in records if row.get("event_type") == "assignment_rejected"
            )
        table[cell] = {
            "episodes_unique_scenario_seed_pairs": len(values),
            "environment_ticks": int(sum(v["ticks"] for v in values)),
            "unique_targets_detected": on_time + expired + unresolved,
            "on_time": on_time,
            "deadline_expired": expired,
            "unresolved_at_episode_end": unresolved,
            "unresolved_reasons": dict(sorted(statuses.items())),
            "assignment_rows_logged": int(logged["assignment_rows"]),
            "rejection_rows_logged": int(logged["rejection_rows"]),
            "recorded_targets_in_the_published_ratio": int(
                reference[cell]["detections"]
            ),
            "count_closes": (on_time + expired + unresolved)
            == int(reference[cell]["detections"]),
            "published_on_time_ratio": f"{reference[cell]['on_time']}/{reference[cell]['detections']}",
            "published_expired": int(reference[cell]["expired"]),
        }
    return table


def main() -> int:
    out = {kind: one_block(kind) for kind in BLOCKS}
    (HERE / "METRIC_ACCOUNTING.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    for kind, table in out.items():
        print(f"== {kind}")
        for cell, values in table.items():
            print(f"  {cell}: {json.dumps(values, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
