"""ROUND26 §3：场景 1400031 两个丢失目标的三臂机制诊断（只读，0 新启动）。

从已落盘的 lifecycle / transcript / tape 重建：发现 → 消息 → 策略请求/HOLD → 发射/投递/应用 →
接收端 belief → 分配/到场 → 推进 → 结局。每个字段标注来源；没有日志写 missing；
同 seed 不等于同一批丢包，"少发先于失败"不直接断言补发必然救回。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "runs" / "comm_cost_greedy_replication_20260928"
ARMS = ("learned_s33_greedy", "immediate", "late_discovery_gate_147")
LOST = ("piloc-e1-target-0", "piloc-e1-target-5")
SCENE = 1400031


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def lifecycle(arm: str) -> list[dict]:
    return read_jsonl(RUN / "logs" / f"r25_{arm}_s{SCENE}_a1_rp.jsonl")


def transcript(arm: str) -> list[dict]:
    return read_jsonl(RUN / "logs" / f"r25_{arm}_s{SCENE}_a1_ct.jsonl")


def tape(arm: str) -> list[dict]:
    path = RUN / "logs" / f"r25_{arm}_s{SCENE}_a1_tape.jsonl"
    return read_jsonl(path) if path.exists() else []


def target_timeline(arm: str, target: str) -> dict:
    """One target's lifecycle chain: discovery, assignments, work spans, outcome, sources."""
    records = [r for r in lifecycle(arm) if r.get("target_id") == target]
    detections = [r for r in records if r["event_type"] == "detection"]
    assignments = [r for r in records if r["event_type"] == "assignment"]
    progress = [r for r in records if r["event_type"] == "progress"]
    finals = [r for r in records if r["event_type"] == "final" and r.get("target_id") == target]
    discovered_at = detections[0]["timestamp"] if detections else None
    deadline = (discovered_at + detections[0]["detail"]["deadline_offset"]) if detections else None
    work_spans = []
    current: list[str] | None = None
    span_start = None
    for record in progress:
        agents = tuple(sorted(record.get("assigned_agents") or []))
        stamp = record["timestamp"]
        if agents != current:
            if current:
                work_spans.append({"agents": list(current), "from": span_start, "to": stamp})
            current = agents if agents else None
            span_start = stamp
    if current:
        work_spans.append({"agents": list(current), "from": span_start,
                           "to": progress[-1]["timestamp"]})
    working = [span for span in work_spans if span["agents"]]
    first_assignment = assignments[0]["timestamp"] if assignments else None
    outcome = finals[0]["status"] if finals else "missing"
    completed_at = next((r["timestamp"] for r in progress if r.get("status") == "completed"), None)
    expired_at = next((r["timestamp"] for r in progress if r.get("status") == "expired"), None)
    return {
        "target": target,
        "sources": {"lifecycle": f"logs/r25_{arm}_s{SCENE}_a1_rp.jsonl",
                    "transcript": f"logs/r25_{arm}_s{SCENE}_a1_ct.jsonl"},
        "discovered_at": discovered_at,
        "deadline": round(deadline, 3) if deadline is not None else None,
        "evaluation_window": 1000,
        "first_assignment_at": first_assignment,
        "assignment_events": [{"t": a["timestamp"], "agents": a.get("assigned_agents"),
                               "reason": (a.get("detail") or {}).get("reason"),
                               "release_from_task": (a.get("detail") or {}).get("release_from_task")}
                              for a in assignments],
        "work_spans_with_agents": working,
        "idle_gaps_between_work_spans": [
            {"from": working[index]["to"], "to": working[index + 1]["from"]}
            for index in range(len(working) - 1)
            if working[index + 1]["from"] > working[index]["to"]],
        "completed_at": completed_at, "expired_at": expired_at, "outcome": outcome,
        "on_time": (outcome == "completed" and completed_at is not None
                    and deadline is not None and completed_at <= deadline),
    }


def messages_for(arm: str, target: str) -> dict:
    sends = [r for r in transcript(arm) if r["kind"] == "send" and r.get("target") == target]
    delivers = [r for r in transcript(arm) if r["kind"] == "deliver" and r.get("target") == target]
    return {
        "sends": [{"step": r["step"], "revision": r["revision"], "first": r["first"],
                   "charge_bytes": r["cost_bytes"], "payload_bytes": r["payload_bytes"],
                   "packets": r["packets"], "packets_lost": r["packets_lost"],
                   "subscribers": r["subscribers"], "ready_step": r["ready_step"],
                   "sender_view_progress": r["progress"], "send_reason": r["send_reason"],
                   "source": f"logs/r25_{arm}_s{SCENE}_a1_ct.jsonl"} for r in sends],
        "deliveries": [{"delivered_at_step": r["step"], "sent_step": r.get("sent_step"),
                        "revision": r.get("revision"), "accepted": r.get("accepted"),
                        "reason": r.get("reason"), "agent": r.get("agent")}
                       for r in delivers],
    }


def policy_decisions(arm: str, target: str) -> list[dict]:
    """The frozen policy's per-decision record for one target from the saved tape."""
    rows = []
    for record in tape(arm):
        if target not in record["targets"]:
            continue
        index = record["targets"].index(target)
        entry = record["candidate_shadow"].get(target, {})
        rows.append({
            "step": record["step"],
            "class": ("actionable" if target in record["actionable"]
                      else "masked" if target in record["masked"]
                      else "forced" if target in record["forced"] else "absent"),
            "send_probability": record["send_probabilities"].get(target),
            "action": ("SEND" if record["actions"].get(target) == 1 else
                       "HOLD" if record["actions"].get(target) == 0 else "missing"),
            "attempt_charge_bytes": record["attempt_charge_bytes"].get(target),
            "shadow_budget_before": entry.get("budget_before"),
            "honours_this_candidate": entry.get("honours_this_candidate"),
            "masked_reason": (record.get("masked_reasons") or {}).get(target),
            "source": f"logs/r25_{arm}_s{SCENE}_a1_tape.jsonl",
        })
    return rows


def arm_episode_summary(arm: str) -> dict:
    row = json.loads((RUN / "rows" / f"r25_{arm}_s{SCENE}_a1.json").read_text(encoding="utf-8"))
    return {"arm": arm, "on_time": row["on_time"], "spent_bytes": row["spent_bytes"],
            "emissions": row["emissions"], "on_time_goal_ids": row["goal_identity"]["on_time_goal_ids"],
            "expired_goal_ids": row["goal_identity"]["expired_goal_ids"],
            "relay_counters": {key: row["relay_counters"][key] for key in
                               ("emissions", "packets_lost", "packets_delivered",
                                "deliveries_applied", "budget_refused_sends",
                                "capacity_refused_sends")}}


def main() -> int:
    payload: dict[str, object] = {
        "round": "26",
        "section": "scene 1400031 diagnosis: the two lost targets across all three arms",
        "scene": SCENE,
        "data_sources": {
            "lifecycle": "runs/comm_cost_greedy_replication_20260928/logs/r25_<arm>_s1400031_a1_rp.jsonl",
            "transcript": "runs/comm_cost_greedy_replication_20260928/logs/r25_<arm>_s1400031_a1_ct.jsonl",
            "tape": "runs/comm_cost_greedy_replication_20260928/logs/r25_learned_s33_greedy_s1400031_a1_tape.jsonl",
            "rows": "runs/comm_cost_greedy_replication_20260928/rows/r25_<arm>_s1400031_a1.json",
        },
        "episode_summaries": [arm_episode_summary(arm) for arm in ARMS],
        "targets": {},
    }
    for target in LOST:
        entry = {"arms": {}}
        for arm in ARMS:
            entry["arms"][arm] = {
                "timeline": target_timeline(arm, target),
                "messages": messages_for(arm, target),
            }
        entry["learned_policy_decisions"] = policy_decisions("learned_s33_greedy", target)
        payload["targets"][target] = entry

    # cross-arm comparison of the two targets' timelines (identical chains are evidence)
    for target in LOST:
        signatures = {}
        for arm in ARMS:
            timeline = payload["targets"][target]["arms"][arm]["timeline"]
            signatures[arm] = json.dumps(
                {"work": timeline["work_spans_with_agents"],
                 "outcome": timeline["outcome"],
                 "completed_at": timeline["completed_at"],
                 "expired_at": timeline["expired_at"],
                 "assignment_ts": [a["t"] for a in timeline["assignment_events"]]},
                sort_keys=True)
        payload["targets"][target]["timeline_signatures"] = signatures
        payload["targets"][target]["timelines_identical"] = len(set(signatures.values())) == 1

    (HERE / "SCENE_1400031_DIAGNOSIS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "episode_summaries": payload["episode_summaries"],
        "targets_identical": {target: payload["targets"][target]["timelines_identical"]
                              for target in LOST},
        "learned_first_sends": {
            target: [r["step"] for r in payload["targets"][target]["arms"]
                     ["learned_s33_greedy"]["messages"]["sends"]] for target in LOST},
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
