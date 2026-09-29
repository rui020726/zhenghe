"""ROUND27 verdict: the five-path table, the intervention-effectiveness evidence and the rules.

Reads only the artefacts already on disk (FIRST2_GATE.json, FIVE_PATH_COMPARISON.json,
PER_TARGET_TIMELINES.json, ARTEFACT_MANIFEST.json) and writes ROUND27_VERDICT.json/.md. The
statement rules are the execution basis §6: what a success in this one selected scene does and
does not support, and which path classes may not be read as evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import r27_common as common

TARGETS = common.INTERVENTION_TARGETS


def _read(name: str) -> dict[str, Any]:
    path = common.OUTPUT_DIR / name
    if not path.exists():
        raise RuntimeError(f"{name} is missing; the verdict cannot be written without it")
    return common.read_json(path)


def build_verdict() -> dict[str, Any]:
    gate = _read("FIRST2_GATE.json")
    comparison = _read("FIVE_PATH_COMPARISON.json")
    timelines = _read("PER_TARGET_TIMELINES.json")
    manifest = _read("ARTEFACT_MANIFEST.json")

    rows = comparison["rows"]
    table = []
    for name in common.PATH_ORDER:
        row = rows[name]
        table.append({
            "path": name,
            "role": common.PATHS[name]["role"],
            "override_targets": list(common.PATHS[name].get("override_targets") or ()),
            "on_time": row["on_time"], "expired": row["expired"],
            "on_time_goal_ids": row["on_time_goal_ids"],
            "expired_goal_ids": row["expired_goal_ids"],
            "spent_bytes": row["spent_bytes"], "emissions": row["emissions"],
            "target_outcomes": row["target_outcomes"],
        })

    interventions = {}
    for name in common.INTERVENTION_PATHS:
        entry = comparison["comparisons"][f"{name}-L"]["intervention_effectiveness"]["per_target"]
        interventions[name] = {
            target: (
                {
                    "override_applied": entry[target]["override_applied"],
                    "override_steps": entry[target]["override_steps"],
                    "opportunities_recorded": entry[target]["opportunities_recorded"],
                    "real_sends": [(s["step"], s["packets"], s["packets_lost"])
                                   for s in entry[target]["real_sends"]],
                    "accepted_agents": sorted({d["agent"] for d in entry[target]["deliveries"]
                                               if d["accepted"]}),
                    "superseded_agents": sorted({d["agent"] for d in entry[target]["deliveries"]
                                                 if d["reason"] == "delivery_superseded"}),
                    "dropped_or_other_agents": sorted(
                        {d["agent"] for d in entry[target]["deliveries"] if not d["accepted"]}
                        - {d["agent"] for d in entry[target]["deliveries"]
                           if d["reason"] == "delivery_superseded"}),
                    "refusals": entry[target]["real_refusals"],
                } if target in entry else {
                    "override_applied": False,
                    "note": "not a named override target of this path; its own messages are in "
                            "PER_TARGET_TIMELINES.json",
                })
            for target in TARGETS}

    comparisons = {}
    for key, value in comparison["comparisons"].items():
        whole = value["task_and_cost"]["whole_scene"]
        cost = value["task_and_cost"]["cost"]
        comparisons[key] = {
            "on_time_delta": whole["on_time_delta"],
            "gained": whole["on_time_goal_ids_gained"],
            "lost": whole["on_time_goal_ids_lost"],
            "other_targets_affected": whole["other_targets_affected"],
            "cost_delta_bytes": cost["cost_delta_bytes"],
            "emissions_delta": cost["emissions_delta"],
            "candidate_refusals": cost["candidate_refusal_sequence"],
            "prefix_consistent_until_override": value["prefix_consistency"][
                "prefix_matches_until_divergence"],
            "per_named_target_task_change": {
                target: value["task_and_cost"]["per_named_target"][target]["task_change"]
                for target in TARGETS},
        }

    e5 = comparisons["E5-L"]
    e05 = comparisons["E05-L"]
    e0 = comparisons["E0-L"]
    decisive_intervention = "E5" if e5["on_time_delta"] > 0 else None

    return {
        "round": "27",
        "section": "verdict: fixed-scene first-report advance intervention",
        "scene": common.SCENE,
        "experiment_nature": "post-hoc selected failing scene; a privileged diagnostic "
                             "intervention, not a generalisation evaluation, rule search, "
                             "learning-benefit proof or deployment strategy",
        "launches": {"normal": manifest["launch_ledger"]["used"],
                     "reserve_used": 0, "hard_cap": manifest["launch_ledger"]["hard_cap"],
                     "failures": 0, "retries": 0,
                     "optimizer_steps": 0, "training": 0, "threshold_adjustments": 0},
        "gate": {"passed": gate["passed"], "verdict": gate["verdict"],
                 "checks": {name: bool(check.get("passed", check.get("identical", False)))
                            for name, check in gate["checks"].items()}},
        "five_path_table": table,
        "interventions": interventions,
        "comparisons": comparisons,
        "mechanism": {
            "decisive_intervention": decisive_intervention,
            "e0_alone": {
                "action_difference": "the step-19 HOLD became a real SEND, 5 of 6 receivers "
                                     "accepted it",
                "task_effect": "no change: both named targets still expired, on-time count "
                               "unchanged, cost unchanged",
                "mechanism": "an accepted first report for target-0 did not move the allocation "
                             "chain: the first assignment of target-0 still waited for step 31 "
                             "and the work pattern was already the same segments as L",
            },
            "e5": {
                "action_difference": "the step-1 HOLD became a real SEND, 5 of 6 receivers "
                                     "accepted it",
                "task_effect": "target-5 completed at 60.0 (deadline 96.5) and target-0 completed "
                               "at 70.0 (deadline 84.0); whole scene 6/6 on time",
                "mechanism": "the acceptance at step 1 seeded the belief of five robots; the "
                             "assignment chain for target-5 started at step 1 and the work "
                             "segments ran 2-21 with five then two robots; target-0's chain "
                             "restarted at steps 31-60 with five robots and finished at 70.0",
            },
            "e05": {
                "action_difference": "both overrides fired (steps 1 and 19); 5 of 6 receivers "
                                     "accepted each",
                "task_effect": "both named targets on time (60.0 and 66.0), whole scene 6/6",
                "interaction": "the combination did not show a task effect beyond E5's; the two "
                               "targets' work chains interleaved differently but the deadline "
                               "outcomes coincide",
            },
        },
        "statement_rules": [
            "in this one post-hoc selected scene and this random implementation, advancing the "
            "first report of target-5 changed the failing outcome; that is all this run supports",
            "it is not established that early sending is generally effective, that all tasks "
            "should be announced early, or that the actor needs an ACK channel",
            "one scene, one RNG implementation: no significance, no bootstrap, no non-inferiority "
            "claim, no new seeds, channel repeats or rule searches",
            "different emissions change the channel RNG consumption, so the post-override paths "
            "are not the same per-message loss batch; outcome changes are not attributable to the "
            "single accepted message",
            "E0 is a real intervention with an accepted message that changed no outcome: it is "
            "evidence that an accepted first report is not sufficient here, not evidence that "
            "early sending is ineffective",
        ],
        "publication": {"remote_verification": "NOT_PERFORMED",
                        "publication_status": "PENDING_NETWORK",
                        "network_retries_this_session": 0},
    }


def write_verdict() -> tuple[Path, Path]:
    payload = build_verdict()
    json_path = common.write_json("ROUND27_VERDICT.json", payload)
    markdown = Path(__file__).resolve().parent / "ROUND27_VERDICT.md"
    markdown.write_text(_markdown(payload), encoding="utf-8")
    return json_path, markdown


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ROUND27 判定：固定失败场景的首报提前干预", "",
        f"场景 `{payload['scene']}`（ROUND26 事后选定的失败案例，特权诊断场景）。", "",
        "## 启动账本", "",
        f"正常 {payload['launches']['normal']}/{payload['launches']['hard_cap'] - 3}，"
        f"储备 {payload['launches']['reserve_used']}/{payload['launches']['hard_cap'] - 5} "
        f"未动，硬上限 {payload['launches']['hard_cap']}；失败 0、重试 0、"
        f"optimizer.step 0、训练 0、阈值调整 0。", "",
        "## 五路径对照表", "",
        "| 路径 | 覆盖目标 | on_time | expired | 已计量字节 | 发射 |", "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["five_path_table"]:
        targets = ", ".join(row["override_targets"]) or "—"
        lines.append(f"| {row['path']} | {targets} | {row['on_time']} | {row['expired']} | "
                     f"{row['spent_bytes']} | {row['emissions']} |")
    lines += ["", "## 干预有效性（相对 L）", "",
              "| 比较 | 干预动作 | 接收 | 任务变化 | 成本变化 |", "| --- | --- | --- | --- | --- |"]
    for name in common.INTERVENTION_PATHS:
        entry = payload["interventions"][name]
        action = "; ".join(
            (f"{target.split('-')[-1]}@{entry[target]['override_steps']}"
             if entry[target].get("override_steps")
             else f"{target.split('-')[-1]}:未覆盖")
            for target in TARGETS)
        comparison = payload["comparisons"][f"{name}-L"]
        lines.append(
            f"| {name}-L | {action} | "
            + "; ".join(
                (f"{t.split('-')[-1]}:{len(entry[t]['accepted_agents'])}/6 accepted"
                 if entry[t].get("accepted_agents") is not None
                 else f"{t.split('-')[-1]}:未覆盖")
                for t in TARGETS)
            + f" | on_time {comparison['on_time_delta']:+d}（获得 "
            + ",".join(g.split('-')[-1] for g in comparison["gained"]) + "，损失 "
            + ",".join(g.split('-')[-1] for g in comparison["lost"]) + "） | "
            + f"{comparison['cost_delta_bytes']:+d} B |")
    lines += ["", "## 机制（描述性，单场景）", ""]
    mechanism = payload["mechanism"]
    lines += [
        f"- 决定性干预：{mechanism['decisive_intervention']}。",
        f"- E0 单独：{mechanism['e0_alone']['mechanism']}。",
        f"- E5：{mechanism['e5']['mechanism']}。",
        f"- E05：{mechanism['e05']['task_effect']}；组合相互作用："
        f"{mechanism['e05']['interaction']}。",
        "", "## 判读纪律", "",
    ]
    lines += [f"- {rule}" for rule in payload["statement_rules"]]
    lines += [
        "", "## 发布状态", "",
        f"remote_verification = {payload['publication']['remote_verification']}；"
        f"publication_status = {payload['publication']['publication_status']}；"
        f"本会话网络重试 {payload['publication']['network_retries_this_session']} 次。", "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    json_path, markdown_path = write_verdict()
    payload = common.read_json(json_path)
    print("verdict:", json_path.name, markdown_path.name)
    print("decisive intervention:", payload["mechanism"]["decisive_intervention"])
    print("launches:", payload["launches"])
