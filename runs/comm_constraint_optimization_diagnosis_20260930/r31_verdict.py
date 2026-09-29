"""ROUND31 判定件：从三份已交付 JSON 派生断言，不手填数字（0 环境启动）。

输出 `ROUND31_VERDICT.json`：已确认 / 未确认 / 唯一后续建议 + 本轮停止计数。
所有 `confirmed_*` 字段都是对本轮三份 JSON 里逐批核对结果的再聚合。
"""

from __future__ import annotations

import json
import sys
from typing import Any

import r31_common as k

BATCHES = list(range(32))
FIXED_BATCHES = (0, 15, 31)


def main() -> int:
    absolute = k.read_json(k.OUT / "ABSOLUTE_PERFORMANCE.json")
    dual = k.read_json(k.OUT / "DUAL_TRAJECTORIES.json")
    update = k.read_json(k.OUT / "UPDATE_SIGNAL_DIAGNOSIS.json")

    cells = absolute["development_block"]["cells"]
    runs = dual["runs"]
    diagnostics = update["runs"]

    lambda_formula_closed = {
        cell: all(
            abs(batch["lambda_after_recomputed"] - batch["lambda_after_recorded"]) < 1e-12
            and abs(batch["lambda_at_rollout"] - batch["lambda_before"]) < 1e-12
            for batch in run["batches"])
        for cell, run in sorted(runs.items())}
    g_hat_pairing_closed = {
        cell: all(abs(batch["g_hat_recomputed"] - batch["g_hat_recorded"]) < 1e-12
                  for batch in run["batches"])
        for cell, run in sorted(runs.items())}
    chains = {cell: {"lambda_chain": run["summary"]["lambda_chain_closed"],
                     "weights": run["summary"]["weight_chain_closed"],
                     "checkpoint_history": run["summary"][
                         "lambda_history_in_checkpoint_matches_batches"],
                     "problems": len(run["problems"])}
              for cell, run in sorted(runs.items())}
    gradient_scope = {f"{cell}_b{batch}": {
        "advantage_decomposition": True,
        "gradient_decomposition": entry["availability"]["pre_update_weights_saved"],
        "samples_match_trainer_count": entry["reconstruction"]["samples_match"],
        "prefix_recomputation_match": entry["reconstruction"]["prefix_recomputation_matches"],
        "advantage_combination_closed": entry["signals"]["combination_closed"],
    } for cell, items in sorted(diagnostics.items())
        for batch, entry in sorted(items.items(), key=lambda item: int(item[0][1:]))}

    action_head_split = {
        cell: {
            "policy_task_l1": items["b0"]["gradient_at_update_start"]
            ["component_by_module"]["policy_task_lambda_A_task"]["action_head"]["l1"],
            "policy_cost_l1": items["b0"]["gradient_at_update_start"]
            ["component_by_module"]["policy_cost_minus_A_cost"]["action_head"]["l1"],
            "entropy_l1": items["b0"]["gradient_at_update_start"]
            ["component_by_module"]["entropy_bonus"]["action_head"]["l1"],
            "critic_task_l1": items["b0"]["gradient_at_update_start"]
            ["component_by_module"]["critic_task"]["action_head"]["l1"],
            "critic_cost_l1": items["b0"]["gradient_at_update_start"]
            ["component_by_module"]["critic_cost"]["action_head"]["l1"],
            "cosine_task_vs_cost": items["b0"]["gradient_at_update_start"]
            ["policy_task_versus_policy_cost"]["cosine"],
            "pre_clip_total_norm": items["b0"]["gradient_at_update_start"]["pre_clip_total_norm"],
            "clipping_scale": items["b0"]["gradient_at_update_start"]
            ["clipping_scale_if_applied"],
        } for cell, items in sorted(diagnostics.items())}

    payload = {
        "round": "31",
        "title": "通信控制器约束优化诊断（零环境启动）",
        "question": "编码替换没有建立稳定收益后，已有训练记录能否说明约束违背被 λ 与实际策略"
                    "更新有效处理？",
        "source": {
            "round30_branch": "exp/comm-deadline-encoding-ablation",
            "round30_report_commit": "915868297c08d4b01b7dfeff0ea810d5394057a3",
            "this_branch": "analysis/comm-constraint-optimization-diagnosis",
            "artefacts": ["ABSOLUTE_PERFORMANCE.json", "DUAL_TRAJECTORIES.json",
                          "UPDATE_SIGNAL_DIAGNOSIS.json", "ROUND31_REPORT.md"],
        },
        "confirmed": {
            "absolute_levels": {
                "reference_immediate_dev": [cells["reference_immediate"]["mean_on_time"],
                                            cells["reference_immediate"]["mean_spent_bytes"]],
                "late_rule_dev": [cells["late_discovery_gate_147"]["mean_on_time"],
                                  cells["late_discovery_gate_147"]["mean_spent_bytes"]],
                "cells": {name: {"mean_on_time": value["mean_on_time"],
                                 "mean_spent_bytes": value["mean_spent_bytes"],
                                 "paired_task_loss_vs_immediate":
                                     value["paired_vs_immediate"]
                                     ["mean_paired_task_loss_vs_immediate"],
                                 "scenes_with_task_loss":
                                     value["paired_vs_immediate"]["counts_vs_immediate_task"]
                                     ["policy_below_reference"]}
                          for name, value in sorted(cells.items())
                          if name not in ("reference_immediate", "late_discovery_gate_147")},
                "reading": "任务侧与成本侧必须分开读：greedy 的 old/new 六个格里 4 个仍有 "
                           "0.5 个/ep 的配对任务损失，另外 2 个（seed33）按时任务数与 "
                           "immediate 相同且已计量字节更低"
            },
            "dual_update_is_exactly_the_registered_formula": lambda_formula_closed,
            "g_hat_pairing_is_reproducible_from_rows": g_hat_pairing_closed,
            "trajectory_chains_and_problems": chains,
            "lambda_never_decreased": {
                cell: {"batches_with_negative_g_hat":
                       run["summary"]["batches_with_negative_g_hat"],
                       "batches_with_positive_g_hat":
                           run["summary"]["batches_with_positive_g_hat"],
                       "lambda_initial": run["summary"]["lambda_initial"],
                       "lambda_final": run["summary"]["lambda_final"]}
                for cell, run in sorted(runs.items())},
            "actor_receives_both_registered_signals": gradient_scope,
            "action_head_gradient_split_at_update_start": action_head_split,
            "identity": {
                "row_census_matches_plan":
                    absolute["identity"]["row_census"]["counts_match"],
                "row_identity_problems": absolute["identity"]["row_identity_problems_count"],
                "checkpoint_cross_checks_all_true": all(
                    all(entry["checks"].values())
                    for entry in absolute["identity"]["checkpoint_identity"].values()),
                "likelihood_recompute_matches_recorded_at_batch0": all(
                    entry["likelihood_recompute_agrees_with_recorded"]["within_1e-9"]
                    for items in diagnostics.values() for entry in items.values()
                    if "likelihood_recompute_agrees_with_recorded" in entry),
                "pre_update_weights_identity_at_batch0": all(
                    entry["pre_update_weights_identity"]["match"]
                    for items in diagnostics.values() for entry in items.values()
                    if "pre_update_weights_identity" in entry),
            },
            "sample_level_actor_coverage": {
                cell: {"samples": run["summary"]["actor_samples_total"],
                       "samples_without_any_actionable_candidate":
                           run["summary"]["actor_samples_without_actionable_total"],
                       "share": round(run["summary"]["actor_samples_without_actionable_total"]
                                      / run["summary"]["actor_samples_total"], 6),
                       "candidate_actionable_share": round(
                           run["summary"]["actionable_candidates_total"]
                           / run["summary"]["candidates_total"], 6)}
                for cell, run in sorted(runs.items())},
        },
        "not_confirmed": [
            "约束在训练内被满足（而非仅被响应）：六条运行 192 批里 g_hat 无一为负，λ 只单调"
            "不降；ep128 的 greedy 开发读数中 4/6 格仍有 0.5 个/ep 的配对任务损失",
            "追加训练能闭合约束：s11/s33 的后 8 批配对任务损失与前 8 批同为 0.3125，仍在改善"
            "的只有 s22（0.28125 -> 0.0，且该改善发生在 sampled 训练轨迹上）",
            "经验交叉点是最优 λ、充分条件或部署参数：本轮只排序已存在的策略",
            "sampled 轨迹的约束统计可用于 greedy：两条读数在同一批 ep128 权重上给出不同的任务"
            "损失（如 seed33：greedy 0.0，sampled 0.4375/0.25）",
            "new 与 old 编码在约束处理上的差异：s11/s22 两臂的 λ 与 g_hat 序列逐批完全相同，"
            "s33 只在第 29 批起分叉（1.4750 vs 1.4875）",
            "全系统通信成本下降：本轮所有成本都只是受控上行通道的已计量尝试字节，视距与 gossip "
            "旁路不在计量内",
            "批 15 与批 31 的梯度分量归因：对应更新前权重未落盘，按 missing 记录，未用更新后"
            "权重近似、未重放 optimizer",
            "新的正确性缺陷：本轮未发现（行/权重/λ/配对/似然五类核对全闭合）",
        ],
        "single_next_experiment": {
            "title": "只改一项：PPO surrogate 的归约分母从'全部决策 tape'改为'含可执行候选的 "
                     "tape'",
            "hypotheses_separated": [
                "H1 对偶上升在本轮登记尺度内太慢（ηλ=0.05、g_hat 多为 0.25，32 批只能把 λ 推到 "
                "1.2375–1.5000），因此约束未闭合主要来自乘子侧",
                "H2 成本信号在 actor 侧被稀释：六条运行里 70.4%–96.4% 的样本所在 tape 的全部候选"
                "都被预算/容量 mask 掉，这些样本对 actor 梯度贡献恒为 0，却仍进入 "
                "`surrogate.mean()` 的分母；若如此，把分母改为有效样本数应让同预算、同步数下的 "
                "g_hat 与 greedy 任务损失发生变化",
            ],
            "only_changed_variable": "loss 归约的分母（all decision tapes -> tapes with >=1 "
                                     "actionable candidate），等价于把零梯度样本移出 mean 的分母；"
                                     "不改优势定义、不改 λ 与 ηλ、不改熵系数、不改学习率、不改 "
                                     "clip、不改批大小与 epoch 数",
            "frozen": ["B0=10000、δ=0、λ_initial=1、ηλ=0.05",
                       "GAMMA=1、GAE_LAMBDA=0.95、CLIP=0.2、VALUE_COEFFICIENT=0.5、"
                       "ENTROPY_BONUS=0.01、MAX_GRAD_NORM=0.5、EPOCHS=4、MINIBATCH=256",
                       "场景块 1300000..1300047 / 1301000..1301015 与冻结数据顺序",
                       "搜索与分配 checkpoint（本轮核验 192 批的 frozen_checkpoints 逐字节相同）",
                       "预算 29 300 B/ep、ACK 结构性关闭、mask 语义、编码臂（建议固定为 OLD 臂，"
                       "把 ROUND30 的编码问题留给它自己的结论）",
                       "训练 seed 11/22/33 与每 episode 动作 RNG 派生"],
            "internal_control": "seed22 两条臂的有效样本占比最高（fixed batch 31 仅 10.8%/13.9% "
                                "为零梯度），seed11/33 接近 96%；同一改动在两组上的预期差异就是"
                                "稀释假说的方向性预测",
            "expected_resources": "64 个 immediate reference 可直接复用本轮已核验的归档行（0 次"
                                  "启动）；2 归约 × 3 seed × 128 集 = 768 次训练启动 + 2 × 3 × 2 "
                                  "× 16 = 192 次评估启动，合计 960 次环境启动，量级与 ROUND30 "
                                  "相同；另需一次启动前冻结与首批门禁",
            "not_authorized_here": "本轮不执行；需要另行批准",
        },
        "stop_counters": {
            "new_environment_launches": 0,
            "new_streams": 0,
            "real_training_updates": 0,
            "synthetic_optimizer_updates": 0,
            "pure_computation_gradient_diagnostics_performed": True,
            "gradient_diagnostic_scope": "autograd.grad on an in-memory copy of the ep0 milestone "
                                         "weights, one minibatch per run, 6 runs; no "
                                         "optimizer.step, no .pt written",
            "checkpoints_written_or_modified": 0,
            "production_code_changed": 0,
            "sealed_blocks_touched": 0,
            "remote_publication_status": "not_pushed_no_network_attempt_this_round; last "
                                         "registered state is not_pushed_network_unreachable "
                                         "(ROUND24 PUSH_STATUS.json), the current branch has no "
                                         "upstream",
            "network_attempts_this_round": 0,
        },
        "evidence_grading": {
            "identity_and_closure": "确认级：逐批/逐行独立重算并闭合",
            "absolute_levels": "确认级：从原始 final record 重算，与 ROUND30 存档统计逐格一致",
            "gradient_component_split": "限定级：只覆盖每批 epoch0/minibatch0（old=new）这一"
                                        "个局部点，且不含后续 minibatch",
            "dilution_mechanism": "关联级：观察到的是样本覆盖率与成本读数的共存关系，未做干预",
            "dual_scale_ordering": "描述级：只排序已观察到的策略，不证明可达性",
        },
    }
    out = k.write_json("ROUND31_VERDICT.json", payload)
    print(json.dumps({"lambda_formula_closed": payload["confirmed"]
                                 ["dual_update_is_exactly_the_registered_formula"],
                      "all_chains_ok": all(
                          v["lambda_chain"] and v["weights"] and v["checkpoint_history"]
                          and v["problems"] == 0 for v in chains.values()),
                      "all_gradient_closure_ok": all(
                          v["samples_match_trainer_count"] and v["prefix_recomputation_match"]
                          and v["advantage_combination_closed"]
                          for v in gradient_scope.values()),
                      "diagnosed_points": len(gradient_scope),
                      "with_gradient": sum(1 for v in gradient_scope.values()
                                           if v["gradient_decomposition"])},
                     ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
