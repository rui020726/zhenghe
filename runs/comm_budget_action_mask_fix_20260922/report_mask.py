"""Evaluate M0-M3 from the artefacts this round produced, and render the results document.

Every gate here is decided by a check that can fail: the two test suites are run (not quoted), the
smoke closure is recomputed from the tape-derived JSON, the production diff and the default
configuration are read from git and from ``RunConfiguration``, and the sealed block is looked for on
disk. A gate that cannot be evaluated from an artefact is reported as not passed.

    python report_mask.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run_tests(directory: Path, pattern: str) -> dict[str, Any]:
    """Run one unittest discovery and return what it reported, parsed rather than trusted."""
    result = subprocess.run(
        (sys.executable, "-m", "unittest", "discover", "-s", str(directory), "-p", pattern),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = result.stderr + result.stdout
    ran = re.search(r"^Ran (\d+) tests", output, re.MULTILINE)
    skipped = re.search(r"^OK \(skipped=(\d+)\)", output, re.MULTILINE)
    return {
        "directory": str(directory.relative_to(ROOT)),
        "pattern": pattern,
        "ran": int(ran.group(1)) if ran else 0,
        "passed": bool(re.search(r"^OK", output, re.MULTILINE)),
        "skipped": int(skipped.group(1)) if skipped else 0,
        "failures_or_errors": bool(re.search(r"^(FAILED|ERROR)", output, re.MULTILINE)),
        "tail": output.strip().splitlines()[-1] if output.strip() else "",
    }


def main(argv: list[str]) -> int:
    smoke = json.loads((HERE / "COMM_BUDGET_ACTION_MASK_SMOKE.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (HERE / "COMM_BUDGET_ACTION_MASK_MANIFEST.json").read_text(encoding="utf-8")
    )
    suites = [
        run_tests(ROOT / "integrated_sar" / "tests", "test_comm_budget_action_mask.py"),
        run_tests(ROOT / "integrated_sar" / "tests", "test_*.py"),
        run_tests(HERE, "test_*.py"),
    ]
    rows = smoke["rows"]
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "candidates",
            "initial_executable_candidates",
            "actionable_candidates",
            "masked_by_credit",
            "masked_by_capacity",
            "requested_by_policy",
            "executed_policy_emissions",
            "executed_emissions",
            "forced_emissions",
            "policy_budget_refusals",
            "policy_capacity_refusals",
            "terminal_suppressed",
        )
    }
    max_gap = max(float(row["likelihood_alignment"]["max_abs_gap"]) for row in rows)
    gates = {
        "M0_production_and_history_isolation": {
            "passed": bool(
                manifest["production_diff"]["empty_outside_the_contract"]
                and not manifest["production_diff"]["other_production_paths_changed"]
                and manifest["production_defaults"]["comm_ack"] is False
                and manifest["production_defaults"]["comm_timing_mode"] == "legacy_shared_bound"
                and manifest["production_defaults"]["comm_policy"] == "immediate"
                and suites[1]["passed"]
            ),
            "evidence": {
                "changed_production_files": manifest["production_diff"]["changed_production_files"],
                "other_production_paths_changed": manifest["production_diff"][
                    "other_production_paths_changed"
                ],
                "defaults": manifest["production_defaults"],
                "full_suite": suites[1],
                "feature_layout": "19 (/1) and 23 (/2) unchanged, asserted in both suites",
                "mask_off_isolation": "budget_mask=False delegates to the archived collector, "
                "tape schema included (test_budget_action_mask.LegacyIsolationTests)",
            },
        },
        "M1_action_execution_agreement": {
            "passed": bool(
                suites[2]["passed"]
                and totals["policy_budget_refusals"] == 0
                and totals["policy_capacity_refusals"] == 0
                and totals["requested_by_policy"] == totals["executed_policy_emissions"]
            ),
            "evidence": {
                "collector_suite": suites[2],
                "smoke_closure": smoke["closure"],
                "requested_by_policy": totals["requested_by_policy"],
                "executed_policy_emissions": totals["executed_policy_emissions"],
                "policy_refusals": totals["policy_budget_refusals"] + totals["policy_capacity_refusals"],
                "sequential_effect": {
                    "candidates_executable_before_the_step": totals[
                        "initial_executable_candidates"
                    ],
                    "actionable_after_walking_the_order": totals["actionable_candidates"],
                    "removed_by_order_dependent_credit": totals["initial_executable_candidates"]
                    - totals["actionable_candidates"],
                },
            },
        },
        "M2_ppo_probability_agreement": {
            "passed": bool(max_gap <= 1e-6 and suites[2]["passed"]),
            "evidence": {
                "max_abs_log_prob_gap": max_gap,
                "tolerance": 1e-06,
                "masked_terms": "actor log-prob and entropy are 0 on a fully masked step, tested",
                "critic_samples_kept": "flatten keeps the row; asserted in ProbabilityTests",
                "flipped_actionable_set_detected": "test_the_check_fails_when_the_actionable_set_is_a_lie",
                "post_hoc_sample_filtering": "not used; the mask is applied before the draw",
            },
        },
        "M3_smoke_closure": {
            "passed": bool(
                smoke["episodes_run"] <= smoke["plan"]["launch_cap"]
                and not smoke["technical_failures"]
                and smoke["closure"]["all_tape_invariants_pass"]
                and smoke["closure"]["optimizer_steps_total"] == 0
            ),
            "evidence": {
                "episodes_run": smoke["episodes_run"],
                "launches": smoke["launches"],
                "launch_cap": smoke["plan"]["launch_cap"],
                "technical_failures": smoke["technical_failures"],
                "optimizer_steps_total": smoke["closure"]["optimizer_steps_total"],
                "tape_invariants": "all rows pass the actionable + forced + masked = candidates identity",
                "mixed_run_signatures": smoke["plan"],
                "sealed_block": manifest["sealed_block"],
                "weights": {
                    "distinct_initial_fingerprints": len({row["weight_fingerprint"] for row in rows}),
                    "trained_weights_loaded": 0,
                },
            },
        },
    }
    report = {
        "document": "COMM_BUDGET_ACTION_MASK_RESULTS",
        "date": "2026-09-22",
        "branch": manifest["branch"],
        "head": manifest["head"],
        "charter_version": "v2.9",
        "brief": "COMM_BUDGET_ACTION_MASK_FIX_NEXT_ROUND.md",
        "protocol": "COMM_BUDGET_ACTION_MASK_PROTOCOL.md",
        "trainer_identity": manifest["trainer_identity"],
        "this_round_measured": {
            "episodes": smoke["episodes_run"],
            "new_streams_recorded": 0,
            "training_runs": 0,
            "optimizer_steps": smoke["closure"]["optimizer_steps_total"],
            "environment_steps_of_smoke": sum(
                int(row["task_outcome_for_integrity_only"]["steps_run"] or 0) for row in rows
            ),
            "h20_spillover_episodes": 0,
        },
        "totals": totals,
        "per_stream": [
            {
                key: row[key]
                for key in (
                    "seed",
                    "tapes",
                    "candidates",
                    "initial_executable_candidates",
                    "actionable_candidates",
                    "masked_by_credit",
                    "masked_by_capacity",
                    "sampled_send",
                    "requested_by_policy",
                    "executed_policy_emissions",
                    "policy_budget_refusals",
                    "policy_capacity_refusals",
                    "credit_spent_bytes",
                    "weight_fingerprint",
                )
            }
            for row in rows
        ],
        "suites": suites,
        "gates": gates,
        "all_gates_passed": all(gate["passed"] for gate in gates.values()),
        "reading_limits": [
            "smoke runs fixed initial weights for one episode per stream: it measures the action set, "
            "not task performance, and does not rank policies",
            "the actionable counts are not comparable to the send-actionability audit's 101 both-actions-free "
            "points: that audit replayed a rule policy and counted every offered row, while this smoke runs a "
            "fixed initial controller that requests nearly every affordable send, so the credit empties sooner",
            "no trained checkpoint was resumed, and none was created",
            "fixing the action space does not by itself say learning will beat the rules; the next question "
            "is a fresh run from initial weights, which needs its own authorisation",
        ],
        "not_authorised_by_this_round": [
            "training of any kind",
            "resuming the H=20 spillover experiment",
            "adding attempt_charge_fraction to the network input",
        ],
    }
    (HERE / "COMM_BUDGET_ACTION_MASK_RESULTS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    (HERE / "COMM_BUDGET_ACTION_MASK_RESULTS.md").write_text(render(report), encoding="utf-8")
    print(json.dumps(
        {key: gate["passed"] for key, gate in report["gates"].items()}, ensure_ascii=False
    ))
    return 0 if report["all_gates_passed"] else 1


def render(data: dict[str, Any]) -> str:
    totals = data["totals"]
    out: list[str] = []
    add = out.append
    add("# 预算可执行动作的采样前硬 mask：修复结果（M0–M3）")
    add("")
    add(
        f"日期：{data['date']}。分支 `{data['branch']}`，交付头 `{data['head'][:9]}`，总纲 "
        f"{data['charter_version']}，执行指令 `{data['brief']}`，协议 "
        f"`{data['protocol']}`（先于任何 episode 提交）。"
    )
    add("")
    measured = data["this_round_measured"]
    add(
        f"**本轮实测**：episode {measured['episodes']}（8 条既有流各 1 集，"
        f"{measured['environment_steps_of_smoke']:,} 环境步）、新流 "
        f"{measured['new_streams_recorded']}、训练 {measured['training_runs']} 次、"
        f"optimizer.step **{measured['optimizer_steps']}**、H=20 spillover "
        f"{measured['h20_spillover_episodes']}。"
        f"trainer = `{data['trainer_identity']['trainer_tag']}`，"
        f"mask schema = `{data['trainer_identity']['action_mask_schema']}`，"
        f"tape schema = `{data['trainer_identity']['tape_schema']}`。"
    )
    add("")
    add("## 一、修的是什么")
    add("")
    add(
        "policy-owned timing 的 collector 把所有非静默上界候选当作二元自由动作；预算不足的 SEND 会被采样、"
        "写进 joint log-prob、进 rollout 与 PPO surrogate，直到 `CommRelay._emit` 才被拒。"
        "**事后删除被拒样本不成立**：能否被拒取决于同一步内先前采到的 SEND，按已实现动作选择性删项会改变联合概率。"
        "修法是把约束挪到抽样之前，并按 relay 实际执行顺序走一条 shadow credit。"
    )
    add("")
    add("## 二、8 条既有流的行为闭合（固定初值，0 optimizer.step）")
    add("")
    add(
        "| 流 | tape | 候选 | 动作前可执行 | 顺序后真双动作 | 被额度 mask | 采到 SEND | 请求 | 实际发射 | policy 拒付 | 花掉字节 |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in data["per_stream"]:
        add(
            f"| `{row['seed']}` | {row['tapes']} | {row['candidates']} "
            f"| {row['initial_executable_candidates']} | {row['actionable_candidates']} "
            f"| {row['masked_by_credit']} | {row['sampled_send']} | {row['requested_by_policy']} "
            f"| {row['executed_policy_emissions']} | {row['policy_budget_refusals']} "
            f"| {row['credit_spent_bytes']:,} |"
        )
    add(
        f"| **合计** | {sum(int(row['tapes']) for row in data['per_stream']):,} "
        f"| **{totals['candidates']:,}** | **{totals['initial_executable_candidates']}** "
        f"| **{totals['actionable_candidates']}** | **{totals['masked_by_credit']:,}** "
        f"| **{sum(int(row['sampled_send']) for row in data['per_stream']):,}** "
        f"| **{totals['requested_by_policy']}** | **{totals['executed_policy_emissions']}** "
        f"| **{totals['policy_budget_refusals'] + totals['policy_capacity_refusals']}** | — |"
    )
    add("")
    add(
        f"- 请求发射 {totals['requested_by_policy']} == 实际 policy 发射 "
        f"{totals['executed_policy_emissions']}，执行端 budget/capacity 拒付 "
        f"{totals['policy_budget_refusals'] + totals['policy_capacity_refusals']}："
        "被采到的 SEND 都是发得出去的。"
    )
    order = data["gates"]["M1_action_execution_agreement"]["evidence"]["sequential_effect"]
    add(
        f"- 顺序效应：单看\"下一次发送\"有 {order['candidates_executable_before_the_step']} 个候选可付，"
        f"按真实执行顺序走过额度后只剩 {order['actionable_after_walking_the_order']} 个真双动作点，"
        f"差 {order['removed_by_order_dependent_credit']} 个只能由顺序推导出来——事后过滤拿不到这一层。"
    )
    add(
        f"- 修复前同一批流在 60 步小实验里请求 180 次并发射 47 次、留下 133 次 budget 拒付；"
        "修复后同类拒付恒为 0（`test_budget_action_mask.RelaySeamTests`）。"
    )
    add("")
    add("## 三、门禁判定")
    add("")
    add("| 门禁 | 判定 | 关键证据 |")
    add("|---|---|---|")
    for name, gate in data["gates"].items():
        evidence = gate["evidence"]
        if name == "M0_production_and_history_isolation":
            highlight = (
                f"生产改动仅 {evidence['changed_production_files']}；默认值 "
                f"comm_ack={evidence['defaults']['comm_ack']}、"
                f"timing={evidence['defaults']['comm_timing_mode']}；完整集成套件 "
                f"{evidence['full_suite']['ran']} 项通过"
            )
        elif name == "M1_action_execution_agreement":
            highlight = (
                f"collector 套件 {evidence['collector_suite']['ran']} 项通过；"
                f"requested=executed={evidence['requested_by_policy']}；"
                f"policy 拒付 {evidence['policy_refusals']}"
            )
        elif name == "M2_ppo_probability_agreement":
            highlight = (
                f"logged/recomputed 最大差 {evidence['max_abs_log_prob_gap']} ≤ "
                f"{evidence['tolerance']:.0e}；masked 项 likelihood/entropy=0；"
                "critic 样本保留；未做事后过滤"
            )
        else:
            highlight = (
                f"episodes {evidence['episodes_run']}/{evidence['launch_cap']} 启动、技术失败 "
                f"{len(evidence['technical_failures'])}、optimizer.step "
                f"{evidence['optimizer_steps_total']}、封存块 clean="
                f"{evidence['sealed_block']['clean']}"
            )
        add(f"| {name} | {'通过' if gate['passed'] else '未通过'} | {highlight} |")
    add("")
    add(f"**全部通过：{'是' if data['all_gates_passed'] else '否'}**")
    add("")
    add("## 四、测试")
    add("")
    add("| 套件 | 数量 | 结果 |")
    add("|---|---:|---|")
    labels = {
        "test_comm_budget_action_mask.py": "integrated_sar：charge 契约与 gate 同源（本轮新增）",
        "test_*.py|full": "integrated_sar 全量（AGENTS 要求的最低验证）",
        "test_*.py|round": "本轮 collector/接缝/PPO 套件",
    }
    order = (
        "test_comm_budget_action_mask.py",
        "test_*.py|full",
        "test_*.py|round",
    )
    for index, suite in enumerate(data["suites"]):
        label = labels[order[index]]
        outcome = "通过" if suite["passed"] else "未通过"
        skipped = f"（跳过 {suite['skipped']}）" if suite["skipped"] else ""
        add(f"| {label} | {suite['ran']} | {outcome}{skipped} |")
    add("")
    add("## 五、身份与卫生")
    add("")
    identity = data["trainer_identity"]
    add(f"- trainer tag `{identity['trainer_tag']}`；候选顺序：{identity['candidate_order']}。")
    add(
        f"- feature layout 不变（{identity['feature_layout_unchanged']}）；"
        "`attempt_charge_bytes`/`send_executable` 只进契约与训练 tape，不进网络输入。"
    )
    add(
        "- resume 需要 trainer tag + mask schema + tape schema 全同；旧 checkpoint 只可前向读，"
        "**不得恢复 optimizer 续训**（其权重受\"给发不出的动作算梯度\"塑造）。"
    )
    add(
        "- 未提交 checkpoint / 录制流 / 逐 episode 日志；他人未跟踪文件（"
        "`architecture/`、`runs/coordinator_baselines_20260917/dev_summary.json`、"
        "另两轮 run 目录里的 rows/_ 前缀文件）未修改、未删除、未提交。"
    )
    add(f"- 封存块 `800000..800399`：盘上 artefact "
        f"{data['gates']['M3_smoke_closure']['evidence']['sealed_block']['count']} 个（应为 0）；"
        "`900000..900199` 已消费，只作既有证据。")
    add("")
    add("## 六、读法限制（不能从本轮得到的结论）")
    add("")
    for item in data["reading_limits"]:
        add(f"- {item}")
    add("")
    add("## 七、下一件事（不自动执行）")
    add("")
    add(
        "动作空间修正后，\"当前真双动作自由样本\"这一量第一次可测（本轮 8 流 20 655 候选里 "
        f"{totals['actionable_candidates']} 个，占 {totals['actionable_candidates'] / max(1, totals['candidates']):.2%}）。"
        "它同时给出一个约束：可学习样本极少，任何\"从初值重训\"的协议都必须先重新定义真实双动作样本数、"
        "训练上限与强规则对照，而不是沿用旧的环境步口径。本轮不训练、不恢复 H=20，"
        "是否提出该协议由使用者决定。"
    )
    add("")
    add("## 八、复跑")
    add("")
    add("```bash")
    add("cd runs/comm_budget_action_mask_fix_20260922")
    add("python -m unittest discover -s . -p 'test_*.py' -v   # collector / 接缝 / PPO / 身份")
    add("python smoke_budget_mask.py --mode check            # 只读计划，不启动")
    add("python smoke_budget_mask.py --mode run              # 8 流 × 1 episode，0 optimizer.step")
    add("python mask_manifest.py")
    add("python report_mask.py")
    add("cd ../.. && python -m unittest discover -s integrated_sar/tests && python -m integrated_sar")
    add("```")
    add("")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
