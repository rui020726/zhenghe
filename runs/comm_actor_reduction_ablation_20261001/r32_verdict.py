"""ROUND32 判定件：从已交付机读件派生结论，不手填数字（0 环境启动）。

按执行依据 §十 的七问顺序输出，并把"已确认 / 未确认 / 唯一后续建议"与停止计数分开登记。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import r32_common as common

BASE, TEST = common.ARM_ORDER


def load(name: str) -> dict[str, Any]:
    path = common.OUTPUT_DIR / name
    return common.read_json(path) if path.exists() else {"missing": True}


def main() -> int:
    stats = load("PAIRED_COMPARISON.json")
    curves = load("TRAINING_CURVES.json")
    mechanism = load("UPDATE_MECHANISM.json")
    recheck = load("INDEPENDENT_RECHECK.json")
    math_note = load("REDUCTION_MATH.json")
    gates = {f"{reduction}_s11": load(f"FIRST4_GATE_{reduction}_s11.json")
             for reduction in common.ARM_ORDER}
    cross = load("FIRST4_CROSS_s11.json")
    ledger = load("START_LEDGER.json")
    checkpoint_index = load("CHECKPOINT_INDEX.json")

    per_cell = {}
    for key, block in sorted((stats.get("comparisons") or {}).items()):
        per_cell[key] = {
            "delta_y_mean": block["delta_y"]["mean"], "delta_y_ci95": block["delta_y"]["ci95"],
            "delta_y_counts": block["delta_y"]["counts"],
            "delta_c_mean": block["delta_c"]["mean"], "delta_c_ci95": block["delta_c"]["ci95"],
            "delta_c_counts": block["delta_c"]["counts"],
            "absolute": {arm: {"mean_on_time": block["absolute"][arm]["mean_on_time"],
                               "mean_spent_bytes": block["absolute"][arm]["mean_spent_bytes"],
                               "paired_task_loss_vs_immediate":
                                   block["absolute"][arm]["mean_paired_task_loss_vs_immediate"],
                               "scenes_with_task_loss": len(
                                   block["absolute"][arm]["scenes_with_task_loss"])}
                        for arm in common.ARM_ORDER},
            "cheaper_but_lost_task_vs_base": block["joint_counts"]["cheaper_but_lost_task_vs_base"],
            "goal_set_swap": block["goal_set_swaps"]["identity_swap_despite_net_tie"],
        }

    summaries = {key: value["summary"] for key, value in (curves.get("runs") or {}).items()}

    def outcome_direction() -> dict[str, Any]:
        cells = list(per_cell.values())
        if not cells:
            return {"status": "no comparison cells yet"}
        task_gain = sum(1 for cell in cells if cell["delta_y_mean"] > 0)
        task_loss = sum(1 for cell in cells if cell["delta_y_mean"] < 0)
        cost_down = sum(1 for cell in cells if cell["delta_c_mean"] < 0)
        loss_reduced = sum(1 for cell in cells
                           if (cell["absolute"][TEST]["paired_task_loss_vs_immediate"]
                               < cell["absolute"][BASE]["paired_task_loss_vs_immediate"]))
        absolute_satisfies = sum(1 for cell in cells
                                 if cell["absolute"][TEST]["paired_task_loss_vs_immediate"] == 0)
        return {"cells": len(cells), "cells_with_task_gain": task_gain,
                "cells_with_task_loss": task_loss, "cells_cheaper": cost_down,
                "cells_where_absolute_loss_reduced": loss_reduced,
                "cells_where_test_arm_has_zero_absolute_loss": absolute_satisfies}

    direction = outcome_direction()
    amplification_holds = bool((mechanism.get("answers") or {}).get(
        "amplification_holds_at_every_analysed_point"))
    not_established = [
        "更多批数或不同的 lambda 能否闭合约束：两者本轮都没有做任何干预",
        "actor 梯度被放大是否等于参数步被放大：Adam 对梯度非线性，entropy 与两个 critic 也在同一"
        "总向量里，全局裁剪在两侧同时触发时只改变方向不改变大小",
        "独立泛化：训练块 1300000..1300047 与开发块 1301000..1301015 都已被多轮反复使用",
        "全系统通信成本下降：本轮计量只是受控上行通道的已计量尝试字节，视距与 gossip 旁路不在内",
        "delta=0 的部署级保证：δ=0 是训练目标，ACTIVE_ROWS 的绝对任务损失只要大于 0，约束就没有"
        "满足，无论相对改善多大",
        "把 3 个训练 seed 当总体：三个 seed 分别展示、未池化",
        "把 sampled 训练轨迹当成 greedy 的约束轨迹",
    ]
    if direction.get("cells") and (direction.get("cells_where_test_arm_has_zero_absolute_loss")
                                   or direction.get("cells_with_task_gain", 0) >= 2):
        next_step = {
            "title": "在未用过的场景块上做一次预登记的确认实验（同一 harness，只换场景块）",
            "reason": "本轮已出现方向一致的改善，但训练块与开发块都是反复使用过的，任何收益目前"
                      "只能是开发性描述",
            "only_changed_variable": "评估场景块（不新增训练量、不改任何超参）",
            "estimated_resources": "2 臂 × 3 seed × 64 未使用场景 = 384 次评估启动 + 64 次 "
                                   "reference；训练复用本轮 ep128 checkpoint（0 次训练启动）"}
    else:
        next_step = {
            "title": "只改一个量：在同一权重、同一 tape 上把 actor 项的梯度贡献与总梯度的其余项"
                     "分开量化（不训练）",
            "reason": "本轮归约已在同状态算术比较中确认放大 N/M（见 UPDATE_MECHANISM），但"
                      "'actor 份额上升是否改变实际参数步'仍未测；这一步只测这件事",
            "only_changed_variable": "诊断量本身（在已落盘的更新前 checkpoint 上做分量分解），"
                                      "不改任何训练设置",
            "estimated_resources": "0 次环境启动；纯计算 forward 与 autograd.grad；"
                                   "不新增 optimizer 更新"}
    payload = {
        "round": "32",
        "title": "Actor 归约分母消融：ALL_ROWS vs ACTIVE_ROWS",
        "question": "仅把 PPO actor surrogate 从'所有记录行平均'改为'有自由决策的记录行平均'，"
                    "是否改善有限训练内的任务—成本表现？",
        "kind": "optimisation-design ablation, not claimed as a bug fix",
        "answers": {
            "1_only_change_is_the_actor_reduction": {
                "verdict": None,
                "evidence": {
                    "trainer_diff_unexplained_blocks": (
                        (load("CONFIG_AND_MODEL_MANIFEST.json").get("trainer_source_diff") or {})
                        .get("unexplained_blocks")),
                    "runner_seam_unexplained": (
                        (load("CONFIG_AND_MODEL_MANIFEST.json").get("runner_seam_diff") or {})
                        .get("unexplained_lines")),
                    "math_checks_passed": math_note.get("passed"),
                    "all_rows_reproduces_frozen_trainer": (
                        (math_note.get("T9_all_rows_equals_frozen_trainer") or {}).get("passed")),
                    "identity_sections_unchanged_vs_round30": (
                        recheck.get("ledger_recheck") is not None
                        and (load("RUN_PREFLIGHT.json").get("identity") or {}).get(
                            "code_identity_sections")),
                }},
            "2_what_actually_changed_in_the_update": {
                "evidence": (mechanism.get("answers") or {}),
                "per_arm_active_row_share_run": {
                    key: value.get("active_row_share_run") for key, value in summaries.items()},
                "note": "the actor term's own gradient scales by N/M; the total pre-clip norm and "
                        "the post-clip vector move much less because the two critics dominate and "
                        "global clipping rescales everything"},
            "3_greedy_and_sampled_results": per_cell,
            "4_seed_consistency": {
                "delta_y_by_seed_and_mode": {
                    key: value["delta_y_mean"] for key, value in per_cell.items()},
                "delta_c_by_seed_and_mode": {
                    key: value["delta_c_mean"] for key, value in per_cell.items()},
                "seeds_never_pooled": True,
                "best_mode_never_selected": True},
            "5_did_the_violation_shrink_or_did_sending_rise": {
                "per_run": {key: {
                    "lambda_final": value["lambda_final"],
                    "g_hat_positive_batches": value["batches_with_positive_g_hat"],
                    "g_hat_zero_batches": value["batches_with_zero_g_hat"],
                    "g_hat_negative_batches": value["batches_with_negative_g_hat"],
                    "mean_paired_task_loss_all_episodes":
                        value["mean_paired_task_loss_all_episodes"],
                    "mean_cost_c_all_episodes": value["mean_cost_c_all_episodes"],
                    "emissions_total": None} for key, value in summaries.items()},
                "note": "violation is judged against the paired immediate reference, never by the "
                        "message count"},
            "6_not_established": not_established,
            "7_next_single_experiment": dict(next_step, **{"not_executed_here": True,
                                                           "requires_approval": True}),
        },
        "outcome_direction_read": direction,
        "confirmed": {
            "reduction_mathematics": {
                "all_checks_passed": math_note.get("passed"),
                "gradient_identity": "grad(ACTIVE actor term) = (N/M) * grad(ALL actor term)",
                "all_rows_equals_frozen_trainer": bool(
                    (math_note.get("T9_all_rows_equals_frozen_trainer") or {}).get("passed")),
                "other_terms_unchanged": bool(
                    (math_note.get("T7_other_losses_unchanged") or {}).get("passed"))},
            "amplification_observed_on_this_rounds_own_artefacts": amplification_holds,
            "first4_gates": {key: (value.get("passed") if not value.get("missing") else None)
                             for key, value in gates.items()},
            "cross_arm_gate": cross.get("passed"),
            "independent_recheck_clean": recheck.get("clean"),
            "ledger": recheck.get("ledger_recheck"),
        },
        "not_confirmed": not_established,
        "counters": {
            "new_environment_launches": (ledger.get("live") or {}).get("used"),
            "launch_plan_normal": common.NORMAL_LAUNCHES,
            "hard_cap": common.LAUNCH_CAP,
            "technical_failures": (recheck.get("ledger_recheck") or {}).get("failures"),
            "synthetic_optimizer_updates": (recheck.get("ledger_recheck") or {}).get(
                "synthetic_updates", {}).get("used") if isinstance(
                    (recheck.get("ledger_recheck") or {}).get("synthetic_updates"), dict) else None,
            "synthetic_cap": common.SYNTHETIC_UPDATE_CAP,
            "real_training_updates_performed_by_this_analysis_round": 0,
            "checkpoints_written_by_diagnosis": 0,
            "production_code_changed": 0,
            "sealed_blocks_touched": 0,
            "pure_computation_gradient_diagnostics": bool(
                (mechanism.get("answers") or {}).get("analysed_points")),
            "remote_publication_status": "not_pushed_no_network_attempt_this_round; the branch has "
                                         "no upstream; the last registered state is "
                                         "not_pushed_network_unreachable",
            "network_attempts": 0,
        },
        "inputs": {"paired_comparison": bool(stats.get("comparisons")),
                   "training_curves": bool(curves.get("runs")),
                   "update_mechanism": bool(mechanism.get("runs")),
                   "independent_recheck": bool(recheck.get("clean") is not None),
                   "reduction_math": bool(math_note.get("passed") is not None),
                   "first4_gates": {key: value.get("passed") for key, value in gates.items()},
                   "cross_arm_gate_s11": cross.get("passed"),
                   "checkpoint_index_pre_update_all_match": None},
    }
    if not checkpoint_index.get("missing"):
        payload["inputs"]["checkpoint_index_pre_update_all_match"] = all(
            entry["pre_update_checkpoints"][batch]["batch_record_parameters_before_matches"]
            for entry in checkpoint_index.values()
            for batch in entry.get("pre_update_checkpoints", {}))
    out = common.OUTPUT_DIR / "ROUND32_VERDICT.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({"cells": list(per_cell), "answers_6": payload["answers"]["6_not_established"],
                      "counters": payload["counters"]}, ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
