"""ROUND26 §4：相对 late 改善的四个场景的信息链对照（只读，0 新启动）。

对每个改善场景：找出获得按时完成的目标，比较三臂的 发现→首报→投递结果→分配/作业→结局；
回答"候选发了什么 late 没发的信息、何时发、是否生效"，并检查旁路/信道替代解释。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "runs" / "comm_cost_greedy_replication_20260928"
ARMS = ("learned_s33_greedy", "immediate", "late_discovery_gate_147")
IMPROVED_SCENES = (1400010, 1400014, 1400017, 1400020)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def target_chain(arm: str, scene: int, target: str) -> dict:
    lifecycle = read_jsonl(RUN / "logs" / f"r25_{arm}_s{scene}_a1_rp.jsonl")
    records = [r for r in lifecycle if r.get("target_id") == target]
    detections = [r for r in records if r["event_type"] == "detection"]
    discovered_at = detections[0]["timestamp"] if detections else None
    deadline = (discovered_at + detections[0]["detail"]["deadline_offset"]) if detections else None
    progress = [r for r in records if r["event_type"] == "progress"]
    assignments = [r for r in records if r["event_type"] == "assignment"]
    spans, current, start = [], None, None
    for record in progress:
        agents = tuple(sorted(record.get("assigned_agents") or []))
        if agents != current:
            if current:
                spans.append({"agents": list(current), "from": start, "to": record["timestamp"]})
            current = agents if agents else None
            start = record["timestamp"]
    if current:
        spans.append({"agents": list(current), "from": start, "to": progress[-1]["timestamp"]})
    finals = [r for r in records if r["event_type"] == "final" and r.get("target_id") == target]
    completed_at = next((r["timestamp"] for r in progress if r.get("status") == "completed"), None)
    expired_at = next((r["timestamp"] for r in progress if r.get("status") == "expired"), None)
    outcome = finals[0]["status"] if finals else "missing"
    return {
        "discovered_at": discovered_at,
        "deadline": round(deadline, 3) if deadline is not None else None,
        "first_assignment_at": assignments[0]["timestamp"] if assignments else None,
        "work_spans_with_agents": [s for s in spans if s["agents"]],
        "completed_at": completed_at, "expired_at": expired_at, "outcome": outcome,
        "on_time": outcome == "completed" and completed_at is not None
        and deadline is not None and completed_at <= deadline,
    }


def target_messages(arm: str, scene: int, target: str) -> dict:
    path = RUN / "logs" / f"r25_{arm}_s{scene}_a1_ct.jsonl"
    if not path.exists():
        return {"sends": [], "deliveries": []}
    records = read_jsonl(path)
    sends = [r for r in records if r["kind"] == "send" and r.get("target") == target]
    delivers = [r for r in records if r["kind"] == "deliver" and r.get("target") == target]
    return {
        "sends": [{"step": r["step"], "revision": r["revision"], "first": r["first"],
                   "charge_bytes": r["cost_bytes"], "ready_step": r["ready_step"],
                   "sender_view_progress": r["progress"]} for r in sends],
        "deliveries": [{"delivered_at_step": r["step"], "accepted": r.get("accepted"),
                        "reason": r.get("reason"), "agent": r.get("agent")} for r in delivers],
        "accepted_count": sum(1 for r in delivers if r.get("accepted")),
        "superseded_count": sum(1 for r in delivers if r.get("reason") == "delivery_superseded"),
    }


def all_sends(arm: str, scene: int) -> list[dict]:
    path = RUN / "logs" / f"r25_{arm}_s{scene}_a1_ct.jsonl"
    if not path.exists():
        return []
    return [{"step": r["step"], "target": r["target"], "first": r["first"],
             "charge_bytes": r["cost_bytes"]} for r in read_jsonl(path) if r["kind"] == "send"]


def main() -> int:
    identity = json.loads((RUN / "TASK_IDENTITY_TRADEOFFS.json").read_text(encoding="utf-8"))
    rows = {scene: [r for r in identity["comparators"]["late_discovery_gate_147"]
                    if r["scene"] == scene][0] for scene in IMPROVED_SCENES}
    payload: dict[str, object] = {"round": "26",
                                  "section": "the four scenes where the candidate gained on-time "
                                             "targets versus late",
                                  "scenes": {}}
    for scene in IMPROVED_SCENES:
        gained = rows[scene]["gained_on_time_goal_ids"]
        entry = {"gained_targets": gained, "arms": {}}
        for target in gained:
            chains = {}
            for arm in ARMS:
                chains[arm] = {"chain": target_chain(arm, scene, target),
                               "messages": target_messages(arm, scene, target)}
            entry[target] = chains
        entry["arm_send_lists"] = {arm: all_sends(arm, scene) for arm in ARMS}
        entry["arm_on_time"] = {}
        for arm in ARMS:
            row = json.loads((RUN / "rows" / f"r25_{arm}_s{scene}_a1.json").read_text(
                encoding="utf-8"))
            entry["arm_on_time"][arm] = {"on_time": row["on_time"],
                                         "spent_bytes": row["spent_bytes"],
                                         "emissions": row["emissions"]}
        payload["scenes"][str(scene)] = entry

    (HERE / "IMPROVED_SCENES.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    summary = {}
    for scene, entry in payload["scenes"].items():
        lines = []
        for target in entry["gained_targets"]:
            chains = entry[target]
            learned = chains["learned_s33_greedy"]
            late = chains["late_discovery_gate_147"]
            lines.append({
                "target": target,
                "learned": {"discovered": learned["chain"]["discovered_at"],
                            "first_send_step": (learned["messages"]["sends"][0]["step"]
                                                if learned["messages"]["sends"] else None),
                            "accepted": learned["messages"]["accepted_count"],
                            "superseded": learned["messages"]["superseded_count"],
                            "first_assignment": learned["chain"]["first_assignment_at"],
                            "completed": learned["chain"]["completed_at"],
                            "on_time": learned["chain"]["on_time"]},
                "late": {"first_send_step": (late["messages"]["sends"][0]["step"]
                                             if late["messages"]["sends"] else None),
                         "accepted": late["messages"]["accepted_count"],
                         "first_assignment": late["chain"]["first_assignment_at"],
                         "completed": late["chain"]["completed_at"],
                         "on_time": late["chain"]["on_time"]},
            })
        summary[scene] = {"gained": entry["gained_targets"], "targets": lines,
                          "arm_on_time": entry["arm_on_time"]}
    (HERE / "IMPROVED_SCENES_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
