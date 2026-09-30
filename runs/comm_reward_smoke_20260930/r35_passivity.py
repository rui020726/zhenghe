"""ROUND35 任务 5：真实被动性证据（0 环境启动，只读已落盘件）。

分级给出证据，不含混：

A 级（直接、可字节比对）：episodes 0..3。这批 rollout 全部在**任何权重移动之前**采样，
  `fresh_controller(11)` 的参数哈希已在 P5 里与归档 ROUND32 的批 0 更新前参数、git 里的启动前
  配对件、ROUND24 ep0 里程碑恢复结果四源相同；动作流是按 (seed, episode) 登记的 sha256。
  因此这些格与归档修复前记录**必须逐字节相同**，除了 task_stream 的键（早一个 tick）与新增的
  事件账本字段。逐项比对：behaviour_sha256、发射序列、成本流、relay 计数器、已花字节、
  on_time/completed/expired、tape 每行的 features/actions/log_prob 与目标集合。

B 级（间接）：episodes 4..7。这些集的采样权重已经经过一次**由修复后奖励驱动的更新**，
  与归档记录**按构造必然可分叉**；分叉本身不是被动性反证。此处的间接证据是：
  同一 scenario、同一 horizon、同一 transport 声明、同一 `_assert_transport` 门槛、
  以及 collector 只在事件总线上读、不写环境。

C 级（结构性）：collector 源码只读属性 + 事件账本里 `credited` 与 registry 提交状态一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import r35_common as smoke

LAUNCH = "launch1"
COMPARISON_FIELDS = ("behaviour_sha256", "emission_sequence", "refusal_sequence",
                     "relay_counters", "spent_bytes", "airtime_tap_total", "emissions",
                     "on_time", "completed", "expired", "discovered", "active", "unassigned",
                     "transport_parameters", "steps", "budget_bytes", "ack_ledger")
TAPE_FIELDS = ("step", "targets", "actions", "actionable", "log_prob", "features")
RESULTS: dict[str, Any] = {}
PROBLEMS: list[str] = []


def gate(name: str, payload: dict[str, Any]) -> None:
    payload["passed"] = bool(payload["passed"])
    RESULTS[name] = payload
    if not payload["passed"]:
        PROBLEMS.append(f"{name}: {payload.get('failure', 'comparison failed')}")


def smoke_row(episode: int) -> dict[str, Any]:
    paths = sorted((smoke.HERE / LAUNCH / "rows").glob(
        f"v2_train_s11_e{episode}_s*_a{episode}.json"))
    if len(paths) != 1:
        raise RuntimeError(f"want one smoke row for e{episode}, got {len(paths)}")
    return json.loads(paths[0].read_text("utf-8"))


def smoke_tape(episode: int) -> list[dict[str, Any]]:
    path = sorted((smoke.HERE / LAUNCH / "logs").glob(
        f"v2_train_s11_e{episode}_s*_a{episode}_tape.jsonl"))[0]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def archived_row(episode: int) -> dict[str, Any]:
    paths = sorted((smoke.ROUND32 / "rows").glob(
        f"r32_train_ALL_ROWS_s11_e{episode}_s*_a{episode}.json"))
    if len(paths) != 1:
        raise RuntimeError(f"want one archived row for e{episode}, got {len(paths)}")
    return json.loads(paths[0].read_text("utf-8"))


def archived_tape(episode: int) -> list[dict[str, Any]]:
    path = sorted((smoke.ROUND32 / "logs").glob(
        f"r32_train_ALL_ROWS_s11_e{episode}_s*_a{episode}_tape.jsonl"))[0]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def a_grade() -> None:
    per_episode: dict[str, Any] = {}
    problems: list[str] = []
    for episode in range(4):
        ours, theirs = smoke_row(episode), archived_row(episode)
        fields = {key: {"smoke": ours[key], "archived": theirs[key],
                        "equal": ours[key] == theirs[key]} for key in COMPARISON_FIELDS}
        tape_ours, tape_theirs = smoke_tape(episode), archived_tape(episode)
        tape_compare = {"rows_smoke": len(tape_ours), "rows_archived": len(tape_theirs),
                        "row_by_row_identical": (
                            len(tape_ours) == len(tape_theirs)
                            and all(ours_row.get(key) == theirs_row.get(key)
                                    for ours_row, theirs_row in zip(tape_ours, tape_theirs)
                                    for key in TAPE_FIELDS))}
        stream_equal = ours["task_stream"] == theirs["task_stream"]
        shifted = sorted(int(key) for key in ours["task_stream"]) == sorted(
            int(key) - 1 for key in theirs["task_stream"])
        bad = [key for key, value in fields.items() if not value["equal"]]
        if bad:
            problems.extend(f"e{episode}/{key}" for key in bad)
        if not tape_compare["row_by_row_identical"]:
            problems.append(f"e{episode}: the per-decision tapes differ")
        if stream_equal:
            problems.append(f"e{episode}: the reward stream is unchanged, so the fix had no "
                            "effect on this cell")
        if not shifted:
            problems.append(f"e{episode}: the new keys are not exactly the archived keys minus 1")
        per_episode[f"e{episode}"] = {
            "scenario_smoke": int(ours["scenario_seed"]),
            "scenario_archived": int(theirs["scenario_seed"]),
            "same_scenario": int(ours["scenario_seed"]) == int(theirs["scenario_seed"]),
            "field_equalities": {key: value["equal"] for key, value in fields.items()},
            "differing_fields": bad,
            "rewarded_field_difference_expected": ["task_stream", "task_reward_events",
                                                   "training_version", "reward_timing_version"],
            "task_stream": {"smoke_keys": sorted(int(k) for k in ours["task_stream"]),
                            "archived_keys": sorted(int(k) for k in theirs["task_stream"]),
                            "unchanged": stream_equal, "archived_keys_minus_one": shifted},
            "tape": tape_compare,
            "behaviour_sha256_smoke": ours["behaviour_sha256"],
            "behaviour_sha256_archived": theirs["behaviour_sha256"],
            "config_hash_note": "config_hash covers each round's own output paths, so it is not "
                                "part of the comparison; the observable behaviour is carried by "
                                "behaviour_sha256, the emission sequence, the relay counters and "
                                "the per-decision tapes",
        }
    gate("A_grade_direct_byte_comparison_episodes_0_to_3", {
        "episodes": per_episode, "problems": problems,
        "evidence_level": "A (direct): the same registered initial weights, the same scenario, "
                          "the same registered action stream, and the environment observed through "
                          "byte-identical tapes, emissions, cost stream, relay counters and "
                          "lifecycle behaviour hash",
        "passed": not problems})


def b_grade() -> None:
    per_episode: dict[str, Any] = {}
    for episode in range(4, smoke.SMOKE_EPISODES):
        ours, theirs = smoke_row(episode), archived_row(episode)
        per_episode[f"e{episode}"] = {
            "scenario_smoke": int(ours["scenario_seed"]),
            "scenario_archived": int(theirs["scenario_seed"]),
            "same_scenario_and_horizon": int(ours["scenario_seed"]) == int(theirs["scenario_seed"])
                                         and ours["steps"] == theirs["steps"],
            "transport_declaration_equal": ours["transport_parameters"]
                                           == theirs["transport_parameters"],
            "behaviour_sha256_equal": ours["behaviour_sha256"] == theirs["behaviour_sha256"],
            "spent_bytes_smoke": int(ours["spent_bytes"]),
            "spent_bytes_archived": int(theirs["spent_bytes"]),
            "note": "these cells were sampled after an update that the corrected reward changes, "
                    "so divergence here is the intervention, not a passivity violation",
        }
    gate("B_grade_indirect_episodes_4_to_7", {
        "episodes": per_episode,
        "all_cells_share_scenario_horizon_and_transport": all(
            value["same_scenario_and_horizon"] and value["transport_declaration_equal"]
            for value in per_episode.values()),
        "cells_that_still_match_the_archived_bytes": [key for key, value in per_episode.items()
                                                      if value["behaviour_sha256_equal"]],
        "evidence_level": "B (indirect)",
        "passed": all(value["same_scenario_and_horizon"]
                      and value["transport_declaration_equal"]
                      for value in per_episode.values())})


def closure() -> None:
    per_episode = {}
    problems = []
    for episode in range(smoke.SMOKE_EPISODES):
        ours = smoke_row(episode)
        credited = int(sum(float(value) for value in ours["task_stream"].values()))
        if credited != int(ours["on_time"]):
            problems.append(f"e{episode}: credited {credited} != on_time {ours['on_time']}")
        if credited != int(ours["task_reward_total"]):
            problems.append(f"e{episode}: credited != task_reward_total")
        ledger_ids = [event["event_id"] for event in ours["task_reward_events"]]
        if len(ledger_ids) != len(set(ledger_ids)):
            problems.append(f"e{episode}: duplicate event ids in the ledger")
        per_episode[f"e{episode}"] = {
            "credited": credited, "on_time": int(ours["on_time"]),
            "ledger_rows": len(ledger_ids), "unique_event_ids": len(set(ledger_ids)),
            "credited_rows": sum(1 for event in ours["task_reward_events"] if event["credited"]),
            "non_credited_rows": sum(1 for event in ours["task_reward_events"]
                                     if not event["credited"]),
            "cost_tap_equals_ledger": int(ours["airtime_tap_total"]) == int(ours["spent_bytes"])}
    gate("C_grade_ledger_and_metric_closure", {
        "per_episode": per_episode, "problems": problems,
        "note": "the runner already refuses to book a cell whose credited total differs from the "
                "lifecycle metric; this re-derives the same closure from the finished rows",
        "passed": not problems})


def main() -> int:
    a_grade()
    b_grade()
    closure()
    payload = {
        "schema": "round35-passivity-evidence/1", "round": 35, "launch": LAUNCH,
        "environment_launches_used_by_this_script": 0,
        "checks": RESULTS, "problems": PROBLEMS, "clean": not PROBLEMS,
        "grading": {"A": "byte comparison against the archived legacy run on cells that share "
                         "the same registered initial weights and action stream",
                    "B": "same scenario, horizon and transport declaration, but the policy has "
                         "already been updated under the corrected reward, so divergence there "
                         "is the intervention",
                    "C": "closure and structural properties re-derived from the finished rows"},
        "claim_this_supports": "the observer does not change what the environment did in the "
                               "cells where the weights are provably the same",
        "claim_this_does_not_support": "a whole-system proof of passivity for every episode, and "
                                       "any statement about learning quality",
    }
    out = smoke.write_json("PASSIVITY_EVIDENCE.json", payload, refresh=True)
    print(json.dumps({"clean": payload["clean"], "problems": PROBLEMS,
                      "a_grade_fields_checked": len(COMPARISON_FIELDS) + 1,
                      "summary": {key: value["passed"] for key, value in RESULTS.items()}},
                     ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0 if payload["clean"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
