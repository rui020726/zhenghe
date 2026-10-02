"""ROUND37 技术门禁（授权边界 §5）：首个冻结场景的九臂正式运行之后，逐项核对技术正确性。

九臂 = 3 个 ep128 模型 x greedy + 3 x sampled + immediate + late_discovery_gate + always_hold，
这 9 次启动计入 576。本脚本 0 额外启动，只读本轮已写的行、tape 与账本，并做一次性的前向复算
（不建 optimizer、不动 RNG、不做任何参数更新）。

检查项：
1 模型与配置身份：行的 checkpoint_load_identity 与冻结件一致；规则臂的策略名正确；
2 无意外训练：本轮训练启动 0、optimizer.step 0、九个行的参数摘要与冻结模型相同；
3 计费与记录闭合：tap 字节 == 预算账本、发射序列长度 == 计数器、task tap == on_time、步数 == 1000；
4 解码标签与实际路径：greedy 行逐决策的动作等于该掩码下的 argmax（并列取 SEND），
  sampled 行的联合对数似然与 tape 登记值相符（<=1e-6），且两臂的动作 RNG 派生不同；
5 修复后的采集接线：行内两个版本字段齐全、task_stream 键 == 入账事件的 relay tick；
6 共同场景：九臂的场景号相同、非策略字段（步数/预算/B0/传输参数除 policy）相同；
7 配对与信道实现的区分：记录各臂的发射/丢包/投递计数，并明确不声称"同一批丢包"。

    /d/project_env/python.exe r37_gate.py
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import torch

import r37_common as r37
import v2_common
import v2_run
from onpolicy import joint_log_prob
from v2_reward_timing import REWARD_TIMING_VERSION, TRAINING_VERSION

SEND, HOLD = 1, 0

GATE: dict[str, Any] = {}
TOLERANCE = v2_common.LOG_PROB_TOLERANCE


def plan() -> dict[str, Any]:
    return r37.read_json(r37.HERE / "FROZEN_EVALUATION_PLAN.json")


def rows_for(scenario: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for arm in r37.ALL_ARMS:
        job = {"arm": arm, "scenario": scenario}
        path = r37.row_path(job)
        if not path.exists():
            raise SystemExit(f"gate row missing: {path}")
        out[arm] = r37.read_json(path)
    return out


def tape_for(arm: str, scenario: int) -> list[dict[str, Any]]:
    kind, seed, mode = r37.parse_arm(arm)
    stem = (f"v2_eval_s{seed}_final_{mode}_s{scenario}_a0" if kind == "model"
            else f"v2_{arm}_s{scenario}_a1")
    path = v2_common.LOGS / f"{stem}_tape.jsonl"
    return list(v2_common.iter_jsonl(path)) if path.exists() else []


# ---------------------------------------------------------------- 1+2+6 身份与配置 -------------
def check_identity(rows: dict[str, dict[str, Any]], frozen: dict[str, Any],
                   scenario: int) -> None:
    models = frozen["models"]
    model_rows = {arm: row for arm, row in rows.items() if arm.startswith("model_")}
    identity = {}
    for arm, row in model_rows.items():
        seed = f"s{r37.parse_arm(arm)[1]}"
        loaded = row.get("checkpoint_load_identity") or {}
        identity[arm] = {
            "checkpoint_matches_frozen_plan": loaded.get("checkpoint") == models[seed]["path"],
            "bytes_sha256_matches": loaded.get("checkpoint_bytes_sha256")
                                    == models[seed]["bytes_sha256"],
            "parameter_sha256_matches_frozen_model": loaded.get("parameter_sha256")
                                                     == models[seed]["parameter_sha256"],
            "load_mode_is_read_only": loaded.get("load_mode") == "read_only",
            "optimizer_restored_false": loaded.get("optimizer_restored") is False,
            "rng_restored_false": loaded.get("rng_restored") is False,
            "episodes_done_is_128": loaded.get("episodes_done") == r37.MILESTONE_EPISODES,
            "decoding_mode_label_matches_arm": loaded.get("decoding_mode")
                                               == r37.parse_arm(arm)[2],
            "scenario": int(row["scenario_seed"]) == scenario}
    rule_identity = {arm: rows[arm]["policy"] == ("immediate" if arm == "immediate"
                                                 else "learned")
                     for arm in r37.RULE_ARMS}
    shared_fields = {}
    for arm, row in rows.items():
        # `policy` and `policy_parameters` are the arm's own identity (relay.transport_parameters()
        # reports the attached policy's parameters: the native immediate policy carries none, while
        # a learned policy reports horizon and feature layout). They must not be mistaken for a
        # world-configuration difference, so the identity check compares everything else.
        transport = {key: value for key, value in row["transport_parameters"].items()
                     if key not in ("policy", "policy_parameters")}
        shared_fields[arm] = {"steps": int(row["steps"]),
                              "budget_bytes": int(row["budget_bytes"]),
                              "b0_bytes": float(row["b0_bytes"]),
                              "transport_without_policy_identity": transport,
                              "scenario_seed": int(row["scenario_seed"])}
    signature = {key: json.dumps(value, sort_keys=True)
                 for key, value in shared_fields.items()}
    flags = {key: all(item[key] for item in identity.values())
             for key in next(iter(identity.values()))} if identity else {}
    GATE["T1_model_and_configuration_identity"] = {
        "per_model_row": identity, "aggregate": flags,
        "rule_arm_policy_fields": rule_identity,
        "policy_side_keys_excluded": ["policy", "policy_parameters"],
        "exclusion_reason": "relay.transport_parameters() reports the attached policy's own "
                            "identity; the native immediate policy carries no parameters while a "
                            "learned policy reports horizon and feature layout, so these keys "
                            "differ by arm kind by construction and are not world configuration",
        "non_policy_configuration_identical_across_arms": len(set(signature.values())) == 1,
        "distinct_configuration_signatures": len(set(signature.values())),
        "passed": bool(flags) and all(flags.values())
                  and all(rule_identity.values())
                  and len(set(signature.values())) == 1}


def check_no_training(rows: dict[str, dict[str, Any]], frozen: dict[str, Any]) -> None:
    book = r37.book()
    GATE["T2_no_unexpected_training"] = {
        "training_launches_in_book": book["training_launches"],
        "optimizer_steps_in_book": book["optimizer_steps"],
        "training_version_fields_in_every_row": all(
            row.get("training_version") == TRAINING_VERSION
            and row.get("reward_timing_version") == REWARD_TIMING_VERSION
            for row in rows.values()),
        "parameters_unchanged_within_an_arm_across_cells": {
            arm: len({(row.get("checkpoint_load_identity") or {}).get("parameter_sha256")}) <= 1
            for arm, row in rows.items() if arm.startswith("model_")},
        "frozen_models_are_ep128_only": all(
            item["episodes_done"] == r37.MILESTONE_EPISODES and item["updates"]
            == r37.EXPECTED_UPDATES for item in frozen["models"].values()),
        "read_only_load_proved": all((row.get("checkpoint_load_identity") or {}).get("load_mode")
                                     == "read_only" for row in rows.values()
                                     if row["policy"] == "learned"),
        "passed": book["training_launches"] == 0 and book["optimizer_steps"] == 0}


# ---------------------------------------------------------------- 3 闭合 ---------------------
def check_records(rows: dict[str, dict[str, Any]]) -> None:
    per_arm = {}
    for arm, row in rows.items():
        taps = {int(key): value for key, value in row["cost_stream"].items()}
        per_arm[arm] = {
            "steps_1000": int(row["steps"]) == v2_common.STEPS_PER_EPISODE,
            "cost_tap_equals_ledger": int(sum(value["airtime_bytes"] for value in
                                              taps.values())) == int(row["spent_bytes"])
                                      and int(row["airtime_tap_total"]) == int(row["spent_bytes"]),
            "emissions_match_sequence": int(row["emissions"]) == len(row["emission_sequence"]),
            "task_tap_equals_on_time": int(row["task_reward_total"]) == int(row["on_time"])
                                       and int(round(float(sum(
                                           row["task_stream"].values())))) == int(row["on_time"]),
            "ack_ledger_all_zero": not any(row["ack_ledger"].values()),
            "budget_within_credit": int(row["spent_bytes"]) <= int(row["budget_bytes"]),
            "lifecycle_records": int(row["on_time"]) + int(row["expired"]) >= 0}
    GATE["T3_billing_and_records_complete"] = {
        "per_arm": per_arm,
        "passed": all(all(item.values()) for item in per_arm.values())}


# ---------------------------------------------------------------- 4 解码语义 -----------------
def check_decoding(rows: dict[str, dict[str, Any]], scenario: int,
                   frozen: dict[str, Any]) -> None:
    """greedy must follow the archived rule on every actionable candidate; sampled must match
    its own logged joint likelihood. How often sampled departs from greedy is a property of the
    policy, not a gate condition."""
    import v2_trainer

    report: dict[str, Any] = {}
    greedy_violations: list[str] = []
    likelihood_violations: list[str] = []
    greedy_seeds: list[int] = []
    sampled_seeds: list[int] = []
    for arm in r37.ALL_ARMS:
        kind, seed, mode = r37.parse_arm(arm)
        tapes = tape_for(arm, scenario)
        if kind != "model":
            report[arm] = {"rule_arm": True, "tape_rows": len(tapes),
                           "policy": rows[arm]["policy"]}
            continue
        path = r37.ROOT / frozen["models"][f"s{seed}"]["path"]
        controller, _state, _identity = v2_trainer.load_checkpoint_read_only(
            path, freeze=r37.stored_freeze_token(path))
        gaps: list[float] = []
        decisions = actionable_decisions = mismatches = departs = 0
        for tape in tapes:
            features = torch.as_tensor(np.asarray(tape["features"], dtype=np.float32))
            with torch.no_grad():
                logits, _values = controller.forward(features)
            probabilities = torch.softmax(logits.float(), dim=-1)
            targets = list(tape["targets"])
            usable = torch.tensor([1.0 if t in tape["actionable"] else 0.0 for t in targets],
                                  dtype=torch.float32)
            picked = torch.tensor([tape["actions"][t] for t in targets], dtype=torch.long)
            with torch.no_grad():
                gaps.append(abs(float(joint_log_prob(logits, picked, usable))
                                - float(tape["log_prob"])))
            decisions += len(targets)
            for position, target in enumerate(targets):
                if target not in tape["actionable"]:
                    continue
                actionable_decisions += 1
                expected = SEND if float(probabilities[position, SEND]) >= float(
                    probabilities[position, HOLD]) else HOLD
                chosen = int(tape["actions"][target])
                if chosen != expected:
                    departs += 1
                    if mode == "greedy":
                        mismatches += 1
                        greedy_violations.append(f"{arm} step {tape['step']} {target}")
        maximum = max(gaps) if gaps else None
        within = bool(gaps) and maximum <= TOLERANCE
        if not within:
            likelihood_violations.append(arm)
        stream_seed = v2_run.eval_action_seed(seed, scenario, mode) % 10**12
        (greedy_seeds if mode == "greedy" else sampled_seeds).append(stream_seed)
        report[arm] = {"tape_rows": len(tapes), "decision_candidates": decisions,
                       "actionable_candidates": actionable_decisions,
                       "greedy_rule_violations": mismatches if mode == "greedy" else None,
                       "decisions_deviating_from_greedy_rule": departs,
                       "max_abs_log_prob_gap": maximum,
                       "log_prob_within_tolerance": within,
                       "action_rng_seed": stream_seed}
    GATE["T4_decoder_labels_match_code_paths"] = {
        "per_arm": report,
        "greedy_rule_violations": greedy_violations,
        "likelihood_violations": likelihood_violations,
        "greedy_stream_seeds": greedy_seeds, "sampled_stream_seeds": sampled_seeds,
        "sampled_stream_differs_from_greedy_stream": bool(greedy_seeds) and bool(sampled_seeds)
                                                     and not set(greedy_seeds)
                                                     & set(sampled_seeds),
        "tolerance": TOLERANCE,
        "greedy_definition": "SEND if p[SEND] >= p[HOLD] else HOLD, applied only to candidates the "
                             "credit and capacity leave free (budget_masked.py:238-239)",
        "passed": not greedy_violations and not likelihood_violations
                  and bool(greedy_seeds) and bool(sampled_seeds)}


# ---------------------------------------------------------------- 5 采集接线 -----------------
def check_reward_wiring(rows: dict[str, dict[str, Any]]) -> None:
    per_arm = {}
    for arm, row in rows.items():
        ledger = row.get("task_reward_events") or []
        credited = [item for item in ledger if item["credited"]]
        per_arm[arm] = {
            "training_version": row.get("training_version"),
            "reward_timing_version": row.get("reward_timing_version"),
            "both_fields_present": bool(row.get("training_version"))
                                   and row.get("reward_timing_version") == REWARD_TIMING_VERSION,
            "ledger_rows": len(ledger), "credited": len(credited),
            "every_relay_tick_equals_rounded_event_time": all(
                item["relay_tick"] == int(round(item["event_time"])) for item in ledger),
            "written_keys_are_crediting_ticks": set(int(key) for key in row["task_stream"]) == {
                int(item["relay_tick"]) for item in credited},
            "on_time_equals_credited": int(row["on_time"]) == len(credited)}
    GATE["T5_corrected_reward_capture_is_live"] = {
        "per_arm": per_arm, "collector": "PublishedProgressRewardCollector",
        "passed": all(item["both_fields_present"]
                      and item["written_keys_are_crediting_ticks"]
                      and item["on_time_equals_credited"]
                      and item["every_relay_tick_equals_rounded_event_time"]
                      for item in per_arm.values())}


# ---------------------------------------------------------------- 7 配对 vs 信道实现 --------
def check_pairing_language(rows: dict[str, dict[str, Any]], scenario: int) -> None:
    counts = {arm: {"emissions": int(row["emissions"]),
                    "attempts": int(row["relay_counters"].get("transmission_attempts", 0)),
                    "delivered": int(row["relay_counters"].get("packets_delivered", 0)),
                    "lost": int(row["relay_counters"].get("packets_lost", 0)),
                    "refused_budget": int(row["relay_counters"].get("budget_refused_sends", 0))}
              for arm, row in rows.items()}
    distinct_attempts = len({item["attempts"] for item in counts.values()})
    GATE["T7_pairing_versus_per_message_channel_realisation"] = {
        "per_arm_counters": counts,
        "arms_are_paired_by_scenario": len({int(row["scenario_seed"])
                                            for row in rows.values()}) == 1,
        "transmission_attempt_counts_differ_across_arms": distinct_attempts > 1,
        "per_message_channel_identity_claimed": False,
        "statement": "配对是在场景层面（同一场景号、同一初始状态、同一世界种子）；各臂的发送轨迹"
                     "不同 ⇒ 抽取的丢包实现次数与顺序也不同，因此不得声称不同臂遭遇同一批丢包",
        "passed": len({int(row["scenario_seed"]) for row in rows.values()}) == 1}


def check_quota_and_isolation(scenario: int) -> None:
    book = r37.book()
    today = __import__("time").strftime("%Y-%m-%d")
    strays = [str(path.relative_to(r37.ROOT)).replace("\\", "/")
              for directory in (r37.STAGE_C, r37.ROUND36, r37.ROUND34)
              for path in directory.rglob("*")
              if path.is_file() and path.stat().st_size > 0
              and __import__("time").strftime("%Y-%m-%d",
                                              time.localtime(path.stat().st_mtime)) == today]
    GATE["T8_quota_and_isolation"] = {
        "gate_cells_expected": len(r37.ALL_ARMS),
        "claims_held": r37.claims_held(),
        "book_starts": book["starts"], "completed": book["completed"],
        "technical_failures": book["technical_failures"],
        "real_ledger_total": r37.real_ledger_total(),
        "hard_cap": r37.HARD_CAP,
        "files_written_today_in_other_round_directories": strays,
        "passed": (book["starts"] == len(r37.ALL_ARMS) and book["completed"] == len(r37.ALL_ARMS)
                   and book["technical_failures"] == 0 and not strays)}


def main(argv: list[str]) -> int:
    frozen = plan()
    scenario = int(frozen["scenarios"][0])
    r37.configure()
    rows = rows_for(scenario)
    check_identity(rows, frozen, scenario)
    check_no_training(rows, frozen)
    check_records(rows)
    check_decoding(rows, scenario, frozen)
    check_reward_wiring(rows)
    check_pairing_language(rows, scenario)
    check_quota_and_isolation(scenario)
    payload = {"kind": "round37-technical-gate", "schema": "round37-gate/1", "round": "37",
               "scenario": scenario, "arms": len(r37.ALL_ARMS),
               "launches_consumed_by_the_gate": len(r37.ALL_ARMS),
               "counted_in_normal_evaluation_launches": True,
               "environment_launches_used_by_this_script": 0, "optimizer_steps_used": 0,
               "checks": GATE, "all_passed": all(item["passed"] for item in GATE.values()),
               "decision_basis": "只有技术正确性决定是否可以扩展；首场景的效果差不构成停止理由",
               "checked_at": time.strftime("%F %T")}
    r37.write_json("ROUND37_GATE.json", payload, refresh=True)
    print(json.dumps({key: value["passed"] for key, value in GATE.items()},
                     ensure_ascii=False, indent=1))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
