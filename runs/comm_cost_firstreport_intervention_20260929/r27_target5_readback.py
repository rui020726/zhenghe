"""ROUND27 §1: target-5 work during steps 3-11 of the fixed scene 1400031, from existing logs.

Zero environment launches: every field below is read back from the ROUND25 artefacts of scene
1400031 (lifecycle, comm transcript, decision tape). Fields the logs do not carry are marked
"missing" rather than inferred. No new simulation is run to fill the gaps.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN25 = REPO / "runs/comm_cost_greedy_replication_20260928"
OUT = Path(__file__).resolve().parent

TARGET = "piloc-e1-target-5"
TARGET0 = "piloc-e1-target-0"
SCENE = 1400031
WINDOW = (1, 54)  # discovery at 1.0; the next assignment after the interruption is at 54.0


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _lifecycle(arm: str) -> list[dict]:
    return _jsonl(RUN25 / "logs" / f"r25_{arm}_s{SCENE}_a1_rp.jsonl")


def _transcript(arm: str) -> list[dict]:
    return _jsonl(RUN25 / "logs" / f"r25_{arm}_s{SCENE}_a1_ct.jsonl")


def _tape() -> list[dict]:
    return _jsonl(RUN25 / "logs" / f"r25_learned_s33_greedy_s{SCENE}_a1_tape.jsonl")


def _row(arm: str) -> dict:
    return json.loads((RUN25 / "rows" / f"r25_{arm}_s{SCENE}_a1.json").read_text(encoding="utf-8"))


def target_readback(arm: str, target: str) -> dict:
    records = _lifecycle(arm)
    events = [r for r in records
              if r.get("target_id") == target
              and WINDOW[0] <= float(r.get("timestamp") or 0.0) <= WINDOW[1]]
    detection = [r for r in events if r["event_type"] == "detection"]
    progress = [r for r in events if r["event_type"] == "progress"]
    assignments = [r for r in events if r["event_type"] == "assignment"]
    trans = _transcript(arm)
    sends = [r for r in trans if r["kind"] == "send" and r.get("target") == target
             and WINDOW[0] <= int(r["step"]) <= WINDOW[1]]
    deliveries = [r for r in trans if r["kind"] == "deliver" and r.get("target") == target
                  and WINDOW[0] <= int(r["step"]) <= WINDOW[1]]
    work = []
    for r in progress:
        work.append({
            "t": float(r["timestamp"]),
            "assigned_agents": list(r.get("assigned_agents") or []),
            "progress": float(r.get("progress") or 0.0),
            "stage_index": int(r.get("stage_index") or 0),
            "status": r.get("status"),
            "remaining_requirements": list(r.get("remaining_requirements") or []),
        })
    return {
        "arm": arm,
        "target": target,
        "window": list(WINDOW),
        "detection": [
            {"t": float(r["timestamp"]), "discovered_by": r.get("discovered_by"),
             "detail_observed_by": (r.get("detail") or {}).get("observed_by"),
             "detail_grid_position": (r.get("detail") or {}).get("grid_position"),
             "detail_deadline_offset": (r.get("detail") or {}).get("deadline_offset")}
            for r in detection],
        "assignment_events": [
            {"t": float(r["timestamp"]), "agents": list(r.get("assigned_agents") or []),
             "reason": (r.get("detail") or {}).get("reason"),
             "command_id": (r.get("detail") or {}).get("command_id"),
             "valid_until": (r.get("detail") or {}).get("valid_until")}
            for r in assignments],
        "progress_events": work,
        "uplink_sends_window": [
            {"step": int(r["step"]), "target": r.get("target"), "revision": r.get("revision"),
             "cost_bytes": r.get("cost_bytes"), "payload_bytes": r.get("payload_bytes"),
             "packets": r.get("packets"), "packets_lost": r.get("packets_lost"),
             "subscribers": r.get("subscribers"), "first": r.get("first"),
             "send_reason": r.get("send_reason"),
             "ready_step": r.get("ready_step"), "progress": r.get("progress")}
            for r in sends],
        "uplink_deliveries_window": [
            {"step": int(r["step"]), "agent": r.get("agent"), "revision": r.get("revision"),
             "accepted": r.get("accepted"), "reason": r.get("reason"),
             "sent_step": r.get("sent_step"), "sent_at": r.get("sent_at"),
             "now": r.get("now")}
            for r in deliveries],
    }


def tape_decisions() -> list[dict]:
    rows = []
    for tape in _tape():
        if TARGET in tape["targets"] and WINDOW[0] <= int(tape["step"]) <= WINDOW[1]:
            rows.append({
                "step": int(tape["step"]),
                "action": "SEND" if tape["actions"][TARGET] == 1 else "HOLD",
                "send_probability": round(float(tape["send_probabilities"].get(TARGET, float("nan"))), 6),
                "class": "actionable" if TARGET in tape["actionable"] else (
                    "masked" if TARGET in tape["masked"] else
                    "forced" if TARGET in tape["forced"] else "not_a_candidate"),
                "honours_this_candidate": (tape["candidate_shadow"].get(TARGET) or {}).get(
                    "honours_this_candidate"),
                "shadow_budget_before": (tape["candidate_shadow"].get(TARGET) or {}).get(
                    "budget_before"),
                "attempt_charge_bytes": tape["attempt_charge_bytes"].get(TARGET),
                "first_emission_flag_missing": "SendItem.first_emission is not stored on the tape; "
                                               "the transcript's send record carries `first`",
            })
    return rows


def who_knew() -> dict:
    """Who could have known target-5 during steps 3-11, by evidence class, no inference."""
    learned = target_readback("learned_s33_greedy", TARGET)
    immediate = target_readback("immediate", TARGET)
    detection = learned["detection"][0]
    assign_agents = sorted({agent for event in learned["assignment_events"]
                            for agent in event["agents"]})
    deliver_agents_learned = sorted({d["agent"] for d in learned["uplink_deliveries_window"]})
    deliver_agents_immediate = sorted({d["agent"] for d in immediate["uplink_deliveries_window"]
                                       if d["accepted"]})
    # Evidence about receivers whose packets were refused as superseded at t=34 proves they
    # already held a belief for this target with an update time later than the message's own
    # sent_at (11.0): _deliver returns delivery_superseded only when known is not None and
    # known.update_time > message.sent_at. "Unknown" is therefore never claimed for an agent
    # that received a superseded verdict.
    superseded_agents = sorted({d["agent"] for d in learned["uplink_deliveries_window"]})
    superseded_detail = [
        {"agent": d["agent"], "step": d["step"], "reason": d["reason"],
         "sent_at_of_message": d["sent_at"]}
        for d in learned["uplink_deliveries_window"]]
    return {
        "known_by_own_detection": [detection["discovered_by"]],
        "assigned_during_window": assign_agents,
        "uplink_deliveries_learned_window": deliver_agents_learned,
        "uplink_sends_learned_window_count": len(learned["uplink_sends_window"]),
        "uplink_accepted_deliveries_immediate_window": deliver_agents_immediate,
        "receivers_that_held_a_belief_by_t34": superseded_agents,
        "superseded_evidence": superseded_detail,
        "info_source_evidence": {
            "measured_uplink": "the learned arm's transcript holds exactly one send for this "
                               "target inside steps 1-53 (step 34, first=true), so no receiver "
                               "could have learned about target-5 from the measured uplink "
                               "between its discovery at t=1.0 and that step-34 send",
            "unmeasured_bypass": "assignment of agents 3 and 4 at t=2.0 and their work from t=3.0 "
                                 "exist without any measured uplink traffic for this target, and "
                                 "the step-34 send was refused as superseded for agents 2, 3, 4 "
                                 "and 5, so the knowledge must have arrived over the "
                                 "line-of-sight / gossip bypass, which the current cost "
                                 "accounting does not measure",
            "bypass_separation": "missing: line-of-sight refresh and one-hop gossip share one code "
                                 "path and one switch in old_task_env, and no log separates them; "
                                 "no per-robot belief log exists in this configuration "
                                 "(info_probe is null)",
            "allocator_belief_source": "missing: the assignment records name the reason "
                                       "'ctde+coordination' but not whether the allocator consulted "
                                       "robot beliefs or the registry truth",
            "exact_time_agents_2_and_5_learned": "missing: only the t=34 superseded verdicts bound "
                                                 "it to 'by t=34', with no finer timestamp",
        },
        "still_unknown_during_steps_3_to_11": {
            "agents_without_any_logged_path_to_know": ["agent_1", "agent_2", "agent_5"],
            "note": "during the 3-11 work window the logs hold no detection, no accepted uplink "
                    "delivery and no assignment for these three agents for this target; the "
                    "unmeasured bypass leaves no per-agent record, so this is the absence of "
                    "evidence rather than a logged fact (agents 2 and 5 are later proven to have "
                    "held a belief by t=34)",
        },
        "learners_during_steps_3_to_11": {
            "agent_0": "own search detection at t=1.0 (lifecycle detail.observed_by)",
            "agent_3": "assigned at t=2.0, worked t=3.0-10.0 (assignment + progress records)",
            "agent_4": "assigned at t=2.0, worked t=3.0-10.0 (assignment + progress records)",
        },
    }


def interruption_evidence() -> dict:
    learned = target_readback("learned_s33_greedy", TARGET)
    work = learned["progress_events"]
    last_with_agents = max((w for w in work if w["assigned_agents"]),
                           key=lambda w: w["t"])
    first_empty_after = next((w for w in work if w["t"] > last_with_agents["t"]
                              and not w["assigned_agents"]), None)
    later_assignments = [e for e in learned["assignment_events"] if e["t"] > first_empty_after["t"]]
    others = {}
    records = _lifecycle("learned_s33_greedy")
    for record in records:
        if record["event_type"] == "assignment" and record.get("target_id") != TARGET:
            t = float(record["timestamp"])
            if 11.0 <= t <= 54.0:
                others.setdefault(str(record["target_id"]), []).append(
                    {"t": t, "agents": list(record.get("assigned_agents") or [])})
    return {
        "last_work_with_agents": last_with_agents,
        "first_record_without_agents_after": first_empty_after,
        "stage_change_at_interruption": {
            "stage_index_before": last_with_agents["stage_index"],
            "stage_index_after": first_empty_after["stage_index"],
            "remaining_requirements_before": last_with_agents["remaining_requirements"],
            "remaining_requirements_after": first_empty_after["remaining_requirements"],
            "status_after": first_empty_after["status"],
        },
        "assignment_gap": {
            "from_t": first_empty_after["t"],
            "next_assignment": later_assignments[0] if later_assignments else None,
            "gap_note": "no assignment record for this target between these two events",
        },
        "progress_gap": {
            "from_t": first_empty_after["t"],
            "next_progress_t": min((w["t"] for w in work if w["t"] > first_empty_after["t"]),
                                   default=None),
        },
        "other_targets_assigned_during_gap": others,
        "capability_read": "missing: the logs carry no per-agent capability schedule, so why the "
                           "stage-1 requirement was not re-assigned to anyone is not decidable "
                           "from these records",
    }


def full_episode_messages(arm: str, target: str) -> dict:
    """Every send, delivery and assignment for one target across the whole episode, one arm.

    Used for the L-versus-immediate contrast: the immediate arm's own first-report timing and the
    receivers that actually accepted it.
    """
    trans = _transcript(arm)
    sends = [r for r in trans if r["kind"] == "send" and r.get("target") == target]
    deliveries = [r for r in trans if r["kind"] == "deliver" and r.get("target") == target]
    assignments = [r for r in _lifecycle(arm)
                   if r["event_type"] == "assignment" and r.get("target_id") == target]
    finished = [r for r in _lifecycle(arm)
                if r.get("target_id") == target and r["event_type"] == "final"]
    return {
        "arm": arm, "target": target,
        "sends": [{"step": int(r["step"]), "revision": r.get("revision"),
                   "cost_bytes": r.get("cost_bytes"), "packets": r.get("packets"),
                   "packets_lost": r.get("packets_lost"), "subscribers": r.get("subscribers"),
                   "first": r.get("first"), "send_reason": r.get("send_reason"),
                   "ready_step": r.get("ready_step"), "progress": r.get("progress")}
                  for r in sends],
        "deliveries": [{"step": int(r["step"]), "agent": r.get("agent"),
                        "accepted": r.get("accepted"), "reason": r.get("reason"),
                        "sent_step": r.get("sent_step"), "sent_at": r.get("sent_at")}
                       for r in deliveries],
        "assignments": [{"t": float(r["timestamp"]), "agents": list(r.get("assigned_agents") or []),
                         "valid_until": (r.get("detail") or {}).get("valid_until")}
                        for r in assignments],
        "final": [{"t": float(r["timestamp"]), "status": r.get("status"),
                   "detail": r.get("detail")} for r in finished],
    }


def main() -> None:
    payload = {
        "round": "27",
        "scene": SCENE,
        "section": "§1 target-5 steps 3-11 read-back from existing ROUND25 logs (zero launches)",
        "sources": {
            "lifecycle": str((RUN25 / "logs" / f"r25_learned_s33_greedy_s{SCENE}_a1_rp.jsonl")
                             .relative_to(REPO)),
            "transcript": str((RUN25 / "logs" / f"r25_learned_s33_greedy_s{SCENE}_a1_ct.jsonl")
                              .relative_to(REPO)),
            "tape": str((RUN25 / "logs" / f"r25_learned_s33_greedy_s{SCENE}_a1_tape.jsonl")
                        .relative_to(REPO)),
            "rows": str((RUN25 / "rows" / f"r25_learned_s33_greedy_s{SCENE}_a1.json")
                        .relative_to(REPO)),
        },
        "target5_learned_window": target_readback("learned_s33_greedy", TARGET),
        "target5_immediate_window": target_readback("immediate", TARGET),
        "target5_full_episode_messages": {
            "learned_s33_greedy": full_episode_messages("learned_s33_greedy", TARGET),
            "immediate": full_episode_messages("immediate", TARGET),
            "late_discovery_gate_147": full_episode_messages("late_discovery_gate_147", TARGET),
        },
        "target0_full_episode_messages": {
            "learned_s33_greedy": full_episode_messages("learned_s33_greedy", TARGET0),
            "immediate": full_episode_messages("immediate", TARGET0),
            "late_discovery_gate_147": full_episode_messages("late_discovery_gate_147", TARGET0),
        },
        "target5_tape_decisions_window": tape_decisions(),
        "who_knew": who_knew(),
        "interruption_evidence": interruption_evidence(),
    }
    (OUT / "TARGET5_STEPS3_11_READBACK.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    known = payload["who_knew"]
    print("detection:", known["known_by_own_detection"])
    print("assigned during window:", known["assigned_during_window"])
    print("uplink sends for target-5 in window:", known["uplink_sends_learned_window_count"])
    print("immediate accepted deliveries in window:", known["uplink_accepted_deliveries_immediate_window"])
    ie = payload["interruption_evidence"]
    print("last work with agents:", ie["last_work_with_agents"]["t"], ie["last_work_with_agents"]["assigned_agents"])
    print("first empty after:", ie["first_record_without_agents_after"]["t"],
          "stage", ie["first_record_without_agents_after"]["stage_index"])
    print("next assignment:", ie["assignment_gap"]["next_assignment"])
    print("next progress:", ie["progress_gap"]["next_progress_t"])


if __name__ == "__main__":
    main()
