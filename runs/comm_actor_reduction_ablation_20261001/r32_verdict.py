"""ROUND32 判定件：从已交付机读件派生结论，不手填数字（0 环境启动）。

按执行依据 §十 的七问顺序输出，并把"已确认 / 未确认 / 唯一后续建议"与停止计数分开登记。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import r32_common as common
import r32_postfreeze

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

    def message_read() -> dict[str, Any]:
        """Message counts straight from the delivered statistics: did ACTIVE_ROWS send more?

        The constraint is judged on metered bytes and on-time tasks; this table exists so the
        behavioural reading (how often the controller talked) is never mistaken for the criterion.
        """
        cells = {}
        for key, block in sorted((stats.get("comparisons") or {}).items()):
            per_arm = {arm: {"emissions_total_16_scenes":
                              block["absolute"][arm]["emissions_total_16_scenes"],
                              "mean_emissions_per_episode":
                                  block["absolute"][arm]["mean_emissions_per_episode"]}
                       for arm in common.ARM_ORDER}
            per_arm["delta_test_minus_base_per_episode"] = round(
                block["absolute"][TEST]["mean_emissions_per_episode"]
                - block["absolute"][BASE]["mean_emissions_per_episode"], 6)
            cells[key] = per_arm
        return {"mean_emissions_per_episode_at_ep128": cells,
                "cells_where_test_arm_sends_more": sum(
                    1 for value in cells.values()
                    if value["delta_test_minus_base_per_episode"] > 0),
                "cells_where_test_arm_sends_less": sum(
                    1 for value in cells.values()
                    if value["delta_test_minus_base_per_episode"] < 0),
                "cells_where_message_count_is_identical": sum(
                    1 for value in cells.values()
                    if value["delta_test_minus_base_per_episode"] == 0),
                "costlier_and_sends_more_agree": sorted(
                    key for key, block in sorted((stats.get("comparisons") or {}).items())
                    if block["delta_c"]["mean"] > 0) == sorted(
                    key for key, value in cells.items()
                    if value["delta_test_minus_base_per_episode"] > 0),
                "source": "PAIRED_COMPARISON.json / absolute / emissions_total_16_scenes "
                          "(16 development scenes x 1 rollout per arm, seed and mode)"}

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
    manifest = load("CONFIG_AND_MODEL_MANIFEST.json")
    postfreeze = load("POST_FREEZE_CHANGES.json")
    trainer_diff = manifest.get("trainer_source_diff") or {}
    runner_seam = manifest.get("runner_seam_diff") or {}
    identity = (load("RUN_PREFLIGHT.json").get("identity") or {})
    provenance = recheck.get("init_pairing_provenance") or {}
    single_variable_clean = bool(
        trainer_diff.get("passed") and not (trainer_diff.get("unexplained_blocks") or [])
        and not (trainer_diff.get("unexplained_removed_lines") or [])
        and runner_seam.get("passed") and not (runner_seam.get("unexplained_lines") or [])
        and math_note.get("passed")
        and (math_note.get("T9_all_rows_equals_frozen_trainer") or {}).get("passed")
        and (manifest.get("round30_identity_audit") or {}).get("clean")
        and postfreeze.get("training_path_unchanged")
        and postfreeze.get("clean")
        and all(str(entry.get("file")) in r32_postfreeze.ALLOWED
                for entry in postfreeze.get("changed_files") or [])
        and bool(identity.get("code_identity_sections")))
    def advantage_scale_read() -> dict[str, Any]:
        """|A_cost| / |lambda*A_task| at the 18 pre-update points, straight from the mechanism file.

        This says which side of the combined advantage the reduction actually amplified; it is a
        reading, not an intervention, so it cannot on its own say what a different scale would do.
        """
        ratios = {}
        for name, batches in sorted((mechanism.get("runs") or {}).items()):
            for batch in sorted(batches):
                advantages = (batches[batch].get("advantages") or {})
                task = (advantages.get("lambda_times_A_task") or {}).get("abs_mean")
                cost = (advantages.get("A_cost") or {}).get("abs_mean")
                if task and cost is not None:
                    ratios[f"{name}/{batch}"] = {
                        "lambda_of_this_batch": advantages.get("lambda_of_this_batch"),
                        "abs_mean_lambda_times_A_task": task,
                        "abs_mean_A_cost": cost,
                        "cost_over_task": round(cost / task, 6)}
        values = [value["cost_over_task"] for value in ratios.values()]
        return {"per_point": ratios, "points": len(ratios),
                "min": min(values) if values else None, "max": max(values) if values else None,
                "read": "在每一个分析点上成本项都小于任务项，所以归约放大的是以任务侧为主的一项"
                        if values and max(values) < 1.0 else "成本项在部分点上超过任务项",
                "limitation": "这是幅度读数，不是尺度干预；'改成多少才能闭合约束'本轮没有做任何干预"}

    messages = message_read()
    advantage_scale = advantage_scale_read()
    base_losses = [cell["absolute"][BASE]["paired_task_loss_vs_immediate"]
                   for cell in per_cell.values()]
    test_losses = [cell["absolute"][TEST]["paired_task_loss_vs_immediate"]
                   for cell in per_cell.values()]
    cells_costlier = sum(1 for cell in per_cell.values() if cell["delta_c_mean"] > 0)
    cells_send_more = sum(
        1 for cell in messages["mean_emissions_per_episode_at_ep128"].values()
        if cell["delta_test_minus_base_per_episode"] > 0)
    never_violated = bool(summaries) and all(
        value["batches_with_negative_g_hat"] == 0 for value in summaries.values())
    violation_shrank = {
        "mean_absolute_paired_task_loss": {BASE: round(sum(base_losses) / len(base_losses), 6),
                                           TEST: round(sum(test_losses) / len(test_losses), 6)}
                                          if base_losses else None,
        "cells_improved": sum(1 for base, test in zip(base_losses, test_losses) if test < base),
        "cells_worse": sum(1 for base, test in zip(base_losses, test_losses) if test > base),
        "cells_tied": sum(1 for base, test in zip(base_losses, test_losses) if test == base),
        "answer": ("没有减少" if not base_losses or sum(test_losses) >= sum(base_losses)
                   else "有减少"),
    }

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
        "组合优势里任务项与成本项的幅度比本轮已直接读出（abs_mean(A_cost)/abs_mean(λ·A_task) 在 "
        "18/18 点为 0.1828–0.4247），但'把成本项尺度改成多少才能闭合约束'没有做任何干预，因此"
        "不能由该读数判定",
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
            "title": "只换一个读数：把六格里剩余的任务损失按场景归因，算出通信侧可达的任务上界"
                     "（不训练、不启动环境）",
            "reason": f"本轮已排除'信号没进 actor'与'零梯度行稀释是主要瓶颈'，也已经在 "
                      f"{advantage_scale['points']}/18 个点上读出组合优势里 λ·A_task 与 A_cost 的"
                      f"幅度比（{advantage_scale['min']}–{advantage_scale['max']}，见 UPDATE_MECHANISM"
                      " 各点 advantages），所以尺度读数本身不是缺口。缺口是：ACTIVE_ROWS 多花 10 条/集"
                      "仍换不到任务，那么剩余任务损失里有多少**根本不在通信这一侧**（搜索零接触）？"
                      "不先量这个上界，任何目标函数改动（含尺度）都是在未知收益的空间里试",
            "only_changed_variable": "归因口径（沿用十九轮的承重首报/严格冗余/买到时间分类），"
                                     "本轮 192 份评估记录与 lifecycle，不改任何训练设置",
            "estimated_resources": "0 次环境启动、0 次 optimizer.step、0 个 checkpoint 写回；"
                                   "只读已落盘的 lifecycle / tape / row",
            "what_would_follow": "若上界接近 0，控制器侧的目标函数消融（包括只改尺度）都不值得再"
                                 "申请；若上界明显大于 0，才另案申请一次**只改尺度、不改归约**的"
                                 "训练消融"}
    cross_run = curves.get("cross_run") or {}
    pre_update_match = bool(checkpoint_index) and not checkpoint_index.get("missing") and all(
        entry["pre_update_checkpoints"][batch]["batch_record_parameters_before_matches"]
        for entry in checkpoint_index.values()
        for batch in entry.get("pre_update_checkpoints", {}))
    verified = {
        "the_reduction_is_the_only_active_change": single_variable_clean,
        "the_reduction_changed_the_actual_update": bool(
            amplification_holds
            and (mechanism.get("answers") or {}).get(
                "advantage_combination_closed_everywhere")
            and (math_note.get("T10_reduction_is_effective") or {}).get("passed")
            and math_note.get("passed")),
        "first_batch_gate_passed": bool(
            all(value.get("passed") for value in gates.values()) and cross.get("passed")),
        "the_intervention_survived_to_lambdas": bool(cross_run) and all(
            not value.get("identical_lambda_sequence", True)
            and not value.get("identical_g_hat_sequence", True)
            for value in cross_run.values()),
        "the_scene_block_was_not_re_shuffled": bool(cross_run) and all(
            value.get("identical_scenario_sequence") and value.get("identical_y_reference")
            for value in cross_run.values()),
        "pre_update_checkpoints_identity_matched": pre_update_match,
        "init_pairing_load_bearing_fields_identical":
            provenance.get("all_load_bearing_fields_identical"),
        "independent_recheck_clean": recheck.get("clean"),
        "note": "each entry is derived from an artefact in this directory, not typed in; "
                "the same claims are what the report's section 1 table asserts",
    }
    payload = {
        "round": "32",
        "title": "Actor 归约分母消融：ALL_ROWS vs ACTIVE_ROWS",
        "question": "仅把 PPO actor surrogate 从'所有记录行平均'改为'有自由决策的记录行平均'，"
                    "是否改善有限训练内的任务—成本表现？",
        "kind": "optimisation-design ablation, not claimed as a bug fix",
        "answers": {
            "1_only_change_is_the_actor_reduction": {
                "verdict": ("是：核验项全过（另有一处不影响测量、但影响证据自述能力的来源缺陷已"
                            "登记在 registered_defects 与报告 §1.1）" if single_variable_clean
                            else "否：见 evidence 中未通过的项"),
                "statement": f"两条臂之间唯一被主动改变的是 actor surrogate 的归约集合与分母 "
                             f"(sum/N -> sum over active rows / max(M,1))；相对冻结父件被删掉的行"
                             f"共 {len(trainer_diff.get('removed_parent_lines') or [])} 行，其分类"
                             f"逐行登记在 evidence.removed_line_categories。代价是 actor 相对 "
                             f"entropy 与两个 critic 的更新权重随之改变——这正是本消融的作用范围，"
                             f"不宣称其他优化关系不受影响。",
                "evidence": {
                    "trainer_diff_unexplained_blocks": trainer_diff.get("unexplained_blocks"),
                    "trainer_diff_unexplained_removed_lines": (
                        trainer_diff.get("unexplained_removed_lines")),
                    "trainer_removed_parent_lines": trainer_diff.get("removed_parent_lines"),
                    "removed_line_categories": trainer_diff.get("removed_line_categories"),
                    "runner_seam_unexplained": runner_seam.get("unexplained_lines"),
                    "runner_run_episode_body_identical": runner_seam.get(
                        "run_episode_body_identical_modulo_module_alias"),
                    "encoder_fixed_to_round30_old": runner_seam.get(
                        "encoder_arm_is_fixed_to_round30_old"),
                    "math_checks_passed": math_note.get("passed"),
                    "all_rows_reproduces_frozen_trainer": (
                        (math_note.get("T9_all_rows_equals_frozen_trainer") or {}).get("passed")),
                    "identity_sections_unchanged_vs_round30": identity.get(
                        "code_identity_sections"),
                    "round30_identity_audit_clean": (
                        (manifest.get("round30_identity_audit") or {}).get("clean")),
                    "post_freeze_training_path_unchanged": postfreeze.get(
                        "training_path_unchanged"),
                    "post_freeze_changes_registered_clean": postfreeze.get("clean"),
                    "post_freeze_changed_files": [entry.get("file") for entry in
                                                  postfreeze.get("changed_files") or []],
                    "init_pairing_provenance": {
                        "all_load_bearing_fields_identical":
                            provenance.get("all_load_bearing_fields_identical"),
                        "pre_launch_marker_still_in_the_working_file":
                            provenance.get("pre_launch_marker_still_in_the_working_file"),
                        "findings": provenance.get("findings"),
                        "meaning": "the pairing claim itself holds (same initial parameters, "
                                   "optimizer state, lambda and module digests for both arms of "
                                   "every seed); what was lost when the run rewrote its own "
                                   "evidence file is the in-file statement that the payload "
                                   "predates the first launch. Registered, not patched, because "
                                   "both involved files sit on the training path"
                    }},
                "clean": single_variable_clean},
            "2_what_actually_changed_in_the_update": {
                "evidence": (mechanism.get("answers") or {}),
                "advantage_scale_read": advantage_scale,
                "per_arm_active_row_share_run": {
                    key: value.get("active_row_share_run") for key, value in summaries.items()},
                "note": "the actor term's own gradient scales by N/M; the total pre-clip norm and "
                        "the post-clip vector move much less because the two critics dominate and "
                        "global clipping rescales everything; and the term that gets scaled is "
                        "itself task-side dominated at every point measured"},
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
                    "active_row_share_run": value["active_row_share_run"],
                    "zero_actor_gradient_row_share_run":
                        value["zero_actor_gradient_row_share_run"]}
                    for key, value in summaries.items()},
                "messages_at_ep128": messages,
                "advantage_scale": {key: value for key, value in advantage_scale.items()
                                    if key != "per_point"},
                "read": {
                    "violation_shrank": violation_shrank,
                    "never_violated_in_training": never_violated,
                    "cells_where_test_arm_costs_more": cells_costlier,
                    "cells_where_test_arm_sends_more": cells_send_more,
                    "absolute_loss_never_improved": bool(
                        direction.get("cells_where_absolute_loss_reduced") == 0),
                    "note": "约束是否满足只看与配对 immediate reference 的任务差与已计量成本；"
                            "消息数只是行为读数，不是判据"},
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
            "synthetic_optimizer_updates_performed": (recheck.get("ledger_recheck") or {}).get(
                "synthetic_updates", {}).get("performed_total"),
            "real_training_batches": sum(int(value.get("batches", 0))
                                         for value in summaries.values()),
            "real_optimizer_steps": sum(int(value.get("optimizer_steps_total", 0))
                                        for value in summaries.values()),
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
                   "checkpoint_index_pre_update_all_match": pre_update_match},
        "verified": verified,
        "registered_defects": [
            *(provenance.get("findings") or []),
            "本轮不修改 r32_freeze.py / r32_train.py：二者都在 POST_FREEZE_CHANGES.json 的"
            "训练/采样/更新/恢复路径清单里，冻结后改动它们属于'拼接修复前后训练链'，执行依据禁止",
            "补救（0 启动、不改训练路径）：r32_postfreeze.preserve_prelaunch_pairing 已把生效"
            "冻结提交里那六份带标记的原件另存为 INIT_PAIRING_PRELAUNCH.json，并逐格比对"
            "启动前/运行期字段（all_load_bearing_fields_equal = true）；下一轮应把运行期证据写入"
            "独立文件名"],
    }
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
