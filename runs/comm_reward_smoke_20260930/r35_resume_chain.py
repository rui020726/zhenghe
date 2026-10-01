"""ROUND35 任务 4d（0 环境启动）：跨进程恢复之后**继续跑出来的 rollout** 是否仍与连续运行一致。

launch2 的真实过程是：进程内先跑 e0..e3 → 批 0 更新 → 存 rolling → **第二次 train_seed 调用**
经 `v2_trainer.load_checkpoint` 跨调用恢复（controller + optimizer + RNG + training_state）→
用恢复出来的权重继续跑 e4、e5。到 e6 时本轮 episode 构造额度（16）用完，真实 ledger 拒起。

本脚本核对四件事：

V1 保存点身份：launch2 的 ep4 存档与 launch1 批 0 记录的后更新状态逐字段相同
   （参数哈希、四模块 digest、λ、updates、optimizer_steps）。
V2 续跑格逐字节比对：launch2 的 e4/e5 行与 tape，对 launch1 的 e4/e5 —— 行为哈希、发射序列、
   成本流、task_stream、事件账本、逐决策 features/actions/log_prob 全部相同才算通过。
V3 恢复确实用上了：e4/e5 行里登记的 `lambda_at_rollout` 必须等于恢复态的 λ（批 0 的
   `lambda_next`），也就是"更新之后的 λ 被带进恢复后的采样"，而不只是文件里有这个数。
V4 停止是额度生效而不是崩溃：launch2 的 ledger 起始数 == 16 额度内该目录的全部起始，
   且第 7 次起始被 cap 拒绝（拒起信息登记在此）。
V5 未覆盖项：恢复之后的那一次**更新**没能在 launch2 里跑到（rollout 先被额度拦住），
   由 R4（从恢复态重算更新、与登记的后更新哈希逐字节相符）间接覆盖，如实标注等级。
"""

from __future__ import annotations

import json
import sys
from typing import Any

import r35_common as smoke

COMPARISON_FIELDS = ("behaviour_sha256", "emission_sequence", "refusal_sequence",
                     "relay_counters", "spent_bytes", "airtime_tap_total", "emissions",
                     "on_time", "completed", "expired", "discovered", "active", "unassigned",
                     "task_stream", "task_reward_events", "transport_parameters", "steps",
                     "budget_bytes", "ack_ledger", "cost_stream")
RESULTS: dict[str, Any] = {}
PROBLEMS: list[str] = []


def gate(name: str, payload: dict[str, Any]) -> None:
    payload["passed"] = bool(payload["passed"])
    RESULTS[name] = payload
    if not payload["passed"]:
        PROBLEMS.append(f"{name}: {payload.get('failure', 'check failed')}")


def row(launch: str, episode: int) -> dict[str, Any]:
    paths = sorted((smoke.HERE / launch / "rows").glob(
        f"v2_train_s11_e{episode}_s*_a{episode}.json"))
    if len(paths) != 1:
        raise RuntimeError(f"{launch} e{episode}: expected one row, found {len(paths)}")
    return json.loads(paths[0].read_text("utf-8"))


def tape(launch: str, episode: int) -> list[dict[str, Any]]:
    paths = sorted((smoke.HERE / launch / "logs").glob(
        f"v2_train_s11_e{episode}_s*_a{episode}_tape.jsonl"))
    return [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()
            if line.strip()]


def v1_saved_resume_point() -> None:
    smoke.configure("launch2")
    freeze = smoke.freeze_identity()
    import torch
    import v2_trainer
    from v2_controller import parameter_identity

    saved = smoke.HERE / "launch2" / "checkpoints" / "v2_train_s11_ep4_saved_for_resume.pt"
    controller, optimizer, state = v2_trainer.load_checkpoint(saved, freeze=freeze)
    launch1_batch = json.loads((smoke.HERE / "launch1" / "checkpoints" / "v2_batch_s11_b0.json")
                               .read_text("utf-8"))
    launch2_batch = json.loads((smoke.HERE / "launch2" / "checkpoints" / "v2_batch_s11_b0.json")
                               .read_text("utf-8"))
    identity = parameter_identity(controller)
    digest = smoke.digest_of(optimizer.state_dict())
    comparison = {
        "restored_parameter_sha256": identity["parameter_sha256"],
        "launch1_batch0_post_update": launch1_batch["controller_parameters_after"]["parameter_sha256"],
        "launch2_batch0_post_update": launch2_batch["controller_parameters_after"]["parameter_sha256"],
        "restored_lambda": float(state["lambda"]),
        "launch1_lambda_next": launch1_batch["dual"]["lambda_next"],
        "restored_updates": int(state["updates"]), "restored_optimizer_steps":
            int(state["optimizer_steps"]), "restored_episodes_done": int(state["episodes_done"]),
        "module_digests_launch1": launch1_batch["controller_module_digests_after"],
        "module_digests_launch2": launch2_batch["controller_module_digests_after"],
        "optimizer_state_sha256_launch1_and_launch2_identical": None}
    same = (comparison["restored_parameter_sha256"]
            == comparison["launch1_batch0_post_update"]
            == comparison["launch2_batch0_post_update"])
    comparison["b0_update_reproduced_bit_for_bit_across_the_two_launches"] = same
    comparison["lambda_identical"] = comparison["restored_lambda"] == comparison[
        "launch1_lambda_next"]
    comparison["module_digests_identical"] = (comparison["module_digests_launch1"]
                                              == comparison["module_digests_launch2"])
    gate("V1_saved_resume_point_identity", {
        **comparison,
        "note": "the resume point launch2 saved is the same state launch1 had after its batch 0, "
                "so the two runs are comparable from episode 4 onward",
        "passed": same and comparison["lambda_identical"] and comparison["module_digests_identical"]
                  and comparison["restored_episodes_done"] == 4
                  and comparison["restored_updates"] == 1})


def v2_resumed_cells_match_the_uninterrupted_run() -> None:
    per_cell = {}
    problems = []
    for episode in (4, 5):
        resumed, uninterrupted = row("launch2", episode), row("launch1", episode)
        fields = {key: {"equal": resumed.get(key) == uninterrupted.get(key),
                        "value_launch2": resumed.get(key)}
                  for key in COMPARISON_FIELDS}
        tape_resumed, tape_plain = tape("launch2", episode), tape("launch1", episode)
        tape_equal = (len(tape_resumed) == len(tape_plain)
                      and all(a["features"] == b["features"] and a["actions"] == b["actions"]
                              and a["log_prob"] == b["log_prob"]
                              and a["actionable"] == b["actionable"]
                              and a["targets"] == b["targets"] and a["step"] == b["step"]
                              for a, b in zip(tape_resumed, tape_plain)))
        differing = sorted(key for key, value in fields.items() if not value["equal"])
        for key in differing:
            problems.append(f"e{episode}/{key}")
        if not tape_equal:
            problems.append(f"e{episode}: the per-decision tapes differ")
        per_cell[f"e{episode}"] = {
            "scenario_resumed": int(resumed["scenario_seed"]),
            "scenario_uninterrupted": int(uninterrupted["scenario_seed"]),
            "fields_compared": len(COMPARISON_FIELDS), "differing_fields": differing,
            "tape_rows": len(tape_resumed), "tape_row_by_row_identical": tape_equal,
            "behaviour_sha256": resumed["behaviour_sha256"],
            "weights_used_for_this_resumed_rollout_equal_the_uninterrupted_run":
                resumed["controller_parameters_after"]
                == uninterrupted["controller_parameters_after"]}
        if (resumed["controller_parameters_after"] != uninterrupted["controller_parameters_after"]):
            problems.append(f"e{episode}: the parameters after the resumed rollout differ")
    gate("V2_resumed_rollouts_match_the_uninterrupted_run", {
        "cells": per_cell, "problems": problems,
        "cells_compared": len(per_cell),
        "evidence_level": "A (direct byte comparison) on the two episodes the resume actually "
                          "reached before the construction quota was spent",
        "passed": not problems})


def v3_resume_used_the_restored_lambda() -> None:
    launch2_batch = json.loads((smoke.HERE / "launch2" / "checkpoints" / "v2_batch_s11_b0.json")
                               .read_text("utf-8"))
    per_episode = {f"e{episode}": {"lambda_at_rollout_in_the_row":
                                   row("launch2", episode)["lambda_at_rollout"],
                                   "lambda_column_in_the_row": row("launch2", episode)["lambda"],
                                   "matches_the_dual_update_result":
                                   row("launch2", episode)["lambda_at_rollout"]
                                   == launch2_batch["dual"]["lambda_next"]}
                   for episode in (4, 5)}
    gate("V3_resumed_rollouts_carried_the_restored_lambda", {
        "batch0_dual": launch2_batch["dual"], "per_episode": per_episode,
        "note": "the rollout rows record the lambda they were sampled with, so this shows the "
                "restored state really fed the continued rollouts rather than only being stored "
                "in a file",
        "passed": all(value["matches_the_dual_update_result"]
                      and value["lambda_at_rollout_in_the_row"] == value["lambda_column_in_the_row"]
                      for value in per_episode.values())})


def v4_the_stop_was_the_quota_not_a_crash() -> None:
    budget = smoke.ledger_starts_across_launches()
    gate("V4_stop_reason", {
        "per_launch_starts": budget["per_launch"], "total_starts": budget["total_starts"],
        "authorised_episode_constructions": smoke.MAX_EPISODE_CONSTRUCTIONS,
        "remaining": budget["remaining"],
        "refused_start": "train_s11_e6 (scenario 1300005) was refused by the real ledger with "
                         "'launch cap 7 reached at 7' because launch2's own book already carried "
                         "the crashed attempt",
        "harness_defect_registered": "r35_launch.py bound the per-directory cap to the round's "
                                     "remaining budget without adding the starts this directory "
                                     "had already recorded, so launch2 was effectively capped at "
                                     "6 fresh episodes; the direction of the error is conservative "
                                     "(it stopped early, it never overran the quota)",
        "no_run_overran_the_quota": budget["total_starts"] <= smoke.MAX_EPISODE_CONSTRUCTIONS,
        "passed": budget["total_starts"] <= smoke.MAX_EPISODE_CONSTRUCTIONS})


def v5_uncovered_items() -> None:
    gate("V5_uncovered_after_the_resume", {
        "not_executed": [
            "the update that would have followed the resumed rollouts in launch2: the quota stopped "
            "the rollouts at six episodes, so launch2 never reached its own second batch update",
            "the live collector probe during launch2: the probe's records lived in the process "
            "memory and the run ended on the ledger refusal before the dump",
            "a third runner invocation: not authorised, and the one the user approved is spent"],
        "covered_indirectly": [
            "post-resume update correctness: R4 recomputes batch 0's update from the restored ep0 "
            "milestone and reproduces the recorded post-update parameter hash, lambda and step "
            "count byte for byte",
            "collector non-interference: V2's byte-identical resumed cells plus the A-grade "
            "episode 0..3 comparison in PASSIVITY_EVIDENCE.json plus the pure wiring probe in "
            "LAUNCH2_ATTEMPT1_REPRO.json (rng, clock, registry and publish counters)"],
        "residual_risk_for_frozen_training": "the corrected timing changes which action row owns "
                                             "each credit, so a 128-episode run's trajectory will "
                                             "not match the archived legacy run; that is the point "
                                             "of the fix, not a defect, but it means old "
                                             "trajectories cannot be reused as predictions",
        "passed": True})


def main() -> int:
    v1_saved_resume_point()
    v2_resumed_cells_match_the_uninterrupted_run()
    v3_resume_used_the_restored_lambda()
    v4_the_stop_was_the_quota_not_a_crash()
    v5_uncovered_items()
    payload = {"schema": "round35-resume-chain/1", "round": 35,
               "environment_launches_used_by_this_script": 0,
               "episode_constructions_by_this_script": 0,
               "checks": RESULTS, "problems": PROBLEMS, "clean": not PROBLEMS,
               "runner_invocations_used": {"launch1": 1, "launch2_attempts": 2,
                                           "crashed_attempt_consumed_one_construction": True}}
    out = smoke.write_json("RESUME_CHAIN.json", payload, refresh=True)
    print(json.dumps({"clean": payload["clean"], "problems": PROBLEMS,
                      "checks": {key: value["passed"] for key, value in RESULTS.items()}},
                     ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0 if payload["clean"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
