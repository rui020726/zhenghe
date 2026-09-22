"""Repair actionability rows from Stage1 free transcripts (0 new launches).

The first replay pass overwrote decision-time budget/terminal with end-of-episode
relay state. Charge numbers were already correct (matched executor sends). This
rebuilds remaining-by-step from the free comm transcript's send/refused records,
clears bogus terminal flags on offered candidates, and reclassifies.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import send_actionability as SA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE1 = ROOT / "runs" / "comm_multistep_stage1_20260922"
FREE_CT = STAGE1 / "episodes" / "free"
BUDGET = 29300


def load_transcript(seed: int) -> list[dict[str, Any]]:
    path = FREE_CT / f"ct_ms_free_s{seed}_t{{step}}_s{seed}.jsonl"
    # filenames embed the Stage1 point step; glob instead
    matches = sorted(FREE_CT.glob(f"ct_ms_free_s{seed}_t*_s{seed}.jsonl"))
    if not matches:
        raise FileNotFoundError(f"no free transcript for seed {seed}")
    rows = []
    with matches[0].open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def budget_timeline(transcript: list[dict[str, Any]]) -> tuple[dict[int, float], float]:
    """Return remaining at the *start* of each step, and total spent at end."""
    spent_after_by_step: dict[int, int] = {}
    refused_remaining: dict[int, float] = {}
    events: list[tuple[int, int, dict[str, Any]]] = []
    for row in transcript:
        kind = row.get("kind")
        if kind not in ("send", "refused"):
            continue
        # sends before refusals within a step (executor tries chosen then may refuse)
        order = 0 if kind == "send" else 1
        events.append((int(row["step"]), order, row))
    events.sort(key=lambda x: (x[0], x[1]))
    spent = 0
    end_spent: dict[int, int] = {}
    for step, _order, row in events:
        if row["kind"] == "send":
            spent = int(row.get("spent_after_bytes", spent + int(row.get("cost_bytes", 0))))
        else:
            # refusal: remaining_bytes is remaining *before* the refused charge
            refused_remaining[step] = float(row.get("remaining_bytes", BUDGET - spent))
        end_spent[step] = spent
    remaining_at_start: dict[int, float] = {}
    prev_end = 0
    steps = sorted(set(list(end_spent) + list(refused_remaining)))
    if not steps:
        return {1: float(BUDGET)}, 0.0
    for step in range(1, max(steps) + 2):
        remaining_at_start[step] = float(BUDGET - prev_end)
        if step in end_spent:
            prev_end = end_spent[step]
    return remaining_at_start, float(spent)


def repair_stream(stream: dict[str, Any]) -> dict[str, Any]:
    seed = int(stream["seed"])
    remaining_at_start, total_spent = budget_timeline(load_transcript(seed))
    # per-step running remaining as we walk candidates in recorded order
    running = dict(remaining_at_start)
    rows = []
    for raw in stream["rows"]:
        row = dict(raw)
        step = int(row["step"])
        rem0 = running.get(step, float(BUDGET - total_spent))
        charge = int(row.get("attempt_charge_bytes") or 0)
        row["budget_bytes_per_episode"] = BUDGET
        row["budget_remaining"] = rem0
        row["budget_spent"] = int(BUDGET - rem0)
        if rem0 > 0 and charge > 0:
            row["charge_over_remaining"] = charge / rem0
        else:
            row["charge_over_remaining"] = None
        # Offered candidates are non-terminal by construction (_observation suppresses terminal).
        row["terminal"] = False
        row["legal"] = True
        # Shared predicate at decision time.
        if charge <= 0:
            row["send_executable"] = True
            row["send_block_reason"] = None
        elif rem0 + 1e-9 < charge:
            row["send_executable"] = False
            row["send_block_reason"] = "budget_exhausted"
        else:
            row["send_executable"] = True
            row["send_block_reason"] = None
        # If this candidate was actually emitted, consume from running remaining.
        # rule_requested + send_executable approximates executor attempts; use transcript
        # consumption only via running remaining update when rule requested and not blocked
        # earlier — conservative: deduct charge when rule_requested and send_executable.
        if row.get("rule_requested") and row.get("send_executable") and charge > 0:
            running[step] = rem0 - charge
        rows.append(row)
    stream = dict(stream)
    stream["rows"] = rows
    stream["budget_total_spent_from_transcript"] = total_spent
    stream["budget_repair"] = "transcript_timeline_v1"
    return stream


def main() -> int:
    rows_path = HERE / "COMM_SEND_ACTIONABILITY_ROWS.json"
    payload = json.loads(rows_path.read_text(encoding="utf-8"))
    repaired = []
    for stream in payload.get("streams", []):
        fixed = repair_stream(stream)
        # classify
        for raw in fixed["rows"]:
            obj = SA.ActionabilityRow(
                **{
                    k: (tuple(v) if k == "subscriber_set" else v)
                    for k, v in raw.items()
                    if k in SA.ActionabilityRow.__dataclass_fields__
                }
            )
            primary, secondary = SA.classify(obj)
            raw["primary_class"] = primary
            raw["secondary_reasons"] = secondary
        counts = {}
        for raw in fixed["rows"]:
            counts[raw["primary_class"]] = counts.get(raw["primary_class"], 0) + 1
        fixed["class_counts"] = counts
        repaired.append(fixed)
        print(
            f"s{fixed['seed']} rows={len(fixed['rows'])} counts={counts} "
            f"spent={fixed['budget_total_spent_from_transcript']}"
        )
    out = {"streams": repaired, "repair": "transcript_timeline_v1"}
    rows_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("repaired", rows_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
