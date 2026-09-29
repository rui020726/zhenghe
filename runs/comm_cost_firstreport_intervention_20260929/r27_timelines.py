"""ROUND27 §5: the two named targets' five-path timelines and message chains.

Zero launches: reads the five rows, their lifecycle logs, transcripts and tapes that already
exist. For each named target and each path: discovery, first-report opportunity/action/emission,
per-receiver accepted/superseded/drop, belief-bearing deliveries, assignment chain, work
segments and interruptions, on-time/expired outcome and the measured cost. The goal is the
mechanism story behind the E0-only failure and the E5/E05 successes, with every field sourced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import r27_common as common

TARGETS = common.INTERVENTION_TARGETS
PATHS = common.PATH_ORDER


def _row(name: str) -> dict[str, Any]:
    return common.read_json(common.row_path(name, 1))


def _records(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(common.iter_jsonl(Path(row["artefacts"]["lifecycle"])))


def _transcript(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(common.iter_jsonl(Path(row["artefacts"]["transcript"])))


def _tape(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(common.iter_jsonl(Path(row["artefacts"]["tape"])))


def target_chain(name: str, target: str) -> dict[str, Any]:
    row = _row(name)
    records = _records(row)
    transcript = _transcript(row)
    tape = _tape(row)

    detection = next((r for r in records if r.get("target_id") == target
                      and r.get("event_type") == "detection"), None)
    assignments = [{"t": float(r["timestamp"]), "agents": list(r.get("assigned_agents") or []),
                    "valid_until": (r.get("detail") or {}).get("valid_until"),
                    "command_id": (r.get("detail") or {}).get("command_id")}
                   for r in records
                   if r.get("target_id") == target and r.get("event_type") == "assignment"]
    work: list[dict[str, Any]] = []
    for record in records:
        if record.get("target_id") != target or record.get("event_type") != "progress":
            continue
        agents = list(record.get("assigned_agents") or [])
        stamp = float(record["timestamp"])
        if work and work[-1]["agents"] == agents and work[-1]["to"] == stamp - 1.0:
            work[-1]["to"] = stamp
            work[-1]["progress_at_end"] = float(record.get("progress") or 0.0)
            work[-1]["stage_at_end"] = int(record.get("stage_index") or 0)
        else:
            work.append({"agents": agents, "from": stamp, "to": stamp,
                         "progress_at_end": float(record.get("progress") or 0.0),
                         "stage_at_end": int(record.get("stage_index") or 0)})
    sends = [r for r in transcript if r["kind"] == "send" and r.get("target") == target]
    deliveries = [r for r in transcript if r["kind"] == "deliver" and r.get("target") == target]
    refusals = [r for r in transcript if r["kind"] == "refused" and r.get("target") == target]

    opportunities: list[dict[str, Any]] = []
    for line in tape:
        note_block = line.get("first_report_overrides")
        if not note_block:
            continue
        for note in note_block.get("opportunities", []):
            if note.get("target") == target:
                opportunities.append(note)

    outcome = row["target_outcomes"].get(target, {})
    evidence = row["intervention_evidence"]["per_target"].get(target, {})
    return {
        "path": name,
        "target": target,
        "discovery": (None if detection is None else {
            "t": float(detection["timestamp"]), "discovered_by": detection.get("discovered_by"),
            "observed_by": (detection.get("detail") or {}).get("observed_by"),
            "deadline_offset": (detection.get("detail") or {}).get("deadline_offset")}),
        "outcome": {key: value for key, value in outcome.items() if key != "final_record"},
        "final_record": outcome.get("final_record"),
        "intervention": {
            "opportunities_recorded": evidence.get("opportunities_recorded"),
            "first_opportunity_step": evidence.get("first_opportunity_step"),
            "override_applied": evidence.get("override_applied"),
            "override_steps": evidence.get("override_steps"),
            "override_details": evidence.get("override_details"),
            "non_override_outcomes": evidence.get("non_override_outcomes"),
        },
        "sends": [{"step": int(r["step"]), "revision": int(r["revision"]),
                   "cost_bytes": int(r["cost_bytes"]), "packets": int(r["packets"]),
                   "packets_lost": int(r["packets_lost"]),
                   "subscribers": int(r["subscribers"]), "first": bool(r["first"]),
                   "send_reason": r.get("send_reason"), "ready_step": int(r.get("ready_step", -1)),
                   "progress": r.get("progress")} for r in sends],
        "refusals": [{"step": int(r["step"]), "reason": r.get("reason")} for r in refusals],
        "deliveries": [{"step": int(r["step"]), "agent": r.get("agent"),
                        "accepted": bool(r.get("accepted")), "reason": r.get("reason"),
                        "sent_step": int(r.get("sent_step", -1)),
                        "sent_at": r.get("sent_at")} for r in deliveries],
        "assignment_events": assignments,
        "work_segments": work,
        "measured_cost_bytes": row["spent_bytes"],
        "whole_scene": {"on_time": row["on_time"], "expired": row["expired"],
                        "emissions": row["emissions"]},
    }


def build() -> dict[str, Any]:
    chains = {name: {target: target_chain(name, target) for target in TARGETS} for name in PATHS}
    summary = {}
    for name in PATHS:
        summary[name] = {}
        for target in TARGETS:
            chain = chains[name][target]
            intervention = chain["intervention"]
            accepted = sorted({d["agent"] for d in chain["deliveries"] if d["accepted"]})
            superseded = sorted({d["agent"] for d in chain["deliveries"]
                                 if d["reason"] == "delivery_superseded"})
            dropped = [d["agent"] for d in chain["deliveries"] if not d["accepted"]]
            first_send = chain["sends"][0]["step"] if chain["sends"] else None
            first_work = chain["work_segments"][0]["from"] if chain["work_segments"] else None
            last_work_with_agents = max((segment["to"] for segment in chain["work_segments"]
                                         if segment["agents"]), default=None)
            summary[name][target] = {
                "intervention_applied_at": intervention["override_steps"],
                "first_send_step": first_send,
                "first_send_accepted_agents": accepted,
                "superseded_agents": superseded,
                "dropped_agents": dropped,
                "first_work_step": first_work,
                "last_work_with_agents": last_work_with_agents,
                "completed_at": chain["outcome"].get("completed_at"),
                "on_time": chain["outcome"].get("on_time"),
                "deadline": chain["outcome"].get("deadline"),
                "measured_cost_bytes": chain["measured_cost_bytes"],
            }
    return {"round": "27", "section": "five-path per-target timelines (zero launches)",
            "scene": common.SCENE, "targets": list(TARGETS), "paths": list(PATHS),
            "summary": summary, "chains": chains}


def main() -> None:
    payload = build()
    common.write_json("PER_TARGET_TIMELINES.json", payload)
    for name in PATHS:
        for target in TARGETS:
            entry = payload["summary"][name][target]
            print(f"{name:>4} {target}: override@{entry['intervention_applied_at']} "
                  f"first_send={entry['first_send_step']} accepted={entry['first_send_accepted_agents']} "
                  f"first_work={entry['first_work_step']} last_work={entry['last_work_with_agents']} "
                  f"done={entry['completed_at']} on_time={entry['on_time']}")


if __name__ == "__main__":
    main()
