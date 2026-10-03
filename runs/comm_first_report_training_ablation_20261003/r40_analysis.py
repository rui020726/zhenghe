"""ROUND40 fixed-matrix analysis. Descriptive only; no inference or threshold fitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

import r40_common as r40


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_claims() -> dict[str, dict[str, Any]]:
    ledger = r40.ledger_snapshot()
    by_job: dict[str, list[dict[str, Any]]] = {}
    for claim in ledger["claims"]:
        by_job.setdefault(str(claim["logical_job"]), []).append(claim)
    return {job: sorted(rows, key=lambda row: int(row["attempt"]))[-1]
            for job, rows in by_job.items()}


def _load_row(common: Any, claims: dict[str, dict[str, Any]], logical: str,
              path: Path) -> dict[str, Any]:
    claim = claims.get(logical)
    if claim is None or claim.get("status") != "complete":
        raise RuntimeError(f"launch is not complete: {logical}")
    if not path.is_file():
        raise RuntimeError(f"completed row is missing: {path}")
    row = _json(path)
    if int(row.get("attempt", -1)) != int(claim["attempt"]):
        raise RuntimeError(f"row attempt does not match its claim: {logical}")
    if row.get("training_version") != "constrained-cost-v2/first-report-only/1/" \
            "published-completion-event-at-relay-tick/2":
        raise RuntimeError(f"row training version mismatch: {path}")
    if row.get("comm_action_eligibility") != "first-report-only/1":
        raise RuntimeError(f"row eligibility identity mismatch: {path}")
    return row


def _targets(row: dict[str, Any]) -> set[str]:
    return {
        str(event["task_id"])
        for event in row.get("task_reward_events", [])
        if event.get("credited") and event.get("on_time")
    }


def _emission_stats(row: dict[str, Any]) -> dict[str, Any]:
    emissions = row.get("emission_sequence", [])
    first = [entry for entry in emissions if bool(entry[5])]
    refresh = [entry for entry in emissions if not bool(entry[5])]
    refusals = row.get("refusal_sequence", [])
    collector = row.get("collector_summary") or {}
    reasons = collector.get("masking_reasons", {})
    eligibility = collector.get("eligibility_counts", {})
    return {
        "first_report_emissions": len(first),
        "first_report_bytes": sum(int(entry[3]) for entry in first),
        "refresh_emissions": len(refresh),
        "refresh_bytes": sum(int(entry[3]) for entry in refresh),
        "budget_masked": int(reasons.get("budget_exhausted", 0)
                              + reasons.get("masked_by_credit", 0)),
        "capacity_masked": int(reasons.get("capacity_step", 0)
                                + reasons.get("masked_by_capacity", 0)),
        "budget_refusals": sum(1 for event in refusals
                               if event[2] == "budget_exhausted"),
        "capacity_refusals": sum(1 for event in refusals if event[2] == "capacity_step"),
        "free_first_reports": int(eligibility.get("free_first_reports", 0)),
        "fixed_hold_refreshes": int(eligibility.get("fixed_hold_refreshes", 0)),
        "requested_first_report_sends": int(
            eligibility.get("requested_first_report_sends", 0)),
    }


def _eval_path(common: Any, arm: str, seed: int, mode: str,
               scenario: int, attempt: int) -> Path:
    return common.ROWS / (
        f"v2_r40_eval_{arm.lower()}_s{seed}_ep128_{mode}_s{scenario}_a{attempt}.json"
    )


def _write_report(results: dict[str, Any], *, artifacts_manifest: dict[str, Any]) -> None:
    ledger = results["launch_ledger"]
    lines = [
        "# ROUND40 报告：first-report-only 通信策略训练消融",
        "",
        "## 执行状态",
        "",
        f"- 分支：`exp/comm-first-report-training-ablation`。",
        f"- 实际环境启动：{ledger['total_used']} 次；正常计划 {ledger['normal_used']}/672，"
        f"技术储备 {ledger['reserve_used']}/28，上限 700。",
        f"- 未构造环境的启动前 worker 失败 {ledger['preconstruction_failure_count']} 次；"
        "不计入环境启动额度，具体异常见 Git 外原件清单。",
        f"- 训练：NEW 三个 seed 各 128 集、32 个 PPO 批；实际 PPO 批数 "
        f"{sum(int(results['training_diagnostics'][str(seed)]['summary']['updates']) for seed in r40.TRAIN_SEEDS)}；"
        f"训练 optimizer.step 次数 "
        f"{sum(int(results['training_diagnostics'][str(seed)]['summary']['optimizer_steps']) for seed in r40.TRAIN_SEEDS)}；"
        "纯合成 optimizer 更新 0。",
        "- OLD checkpoint 只读评估；源 checkpoint 字节哈希与启动前清单一致。",
        "- 评估固定在已使用的 16 个开发场景，不作显著性、总体非劣或泛化声明。",
        f"- Git 外原始产物 {artifacts_manifest['raw_artifact_count']} 个，路径、大小和 SHA256"
        "见 `GIT_OUTSIDE_ARTIFACTS.json`。",
        "",
        "## 预定问题的回答",
        "",
        "### 1. 训练期 refresh 限制是否生效？",
        "",
        "纯计算门禁及每个 seed 的首 4 集门禁均验证 `first_emission` 资格、固定 HOLD、"
        "候选类别闭合、free 集合、似然和 entropy 排除。训练行的 refresh 实际发射计数须为 0。",
        "逐 seed 门禁数据见下表。",
        "",
        "| Seed | 首 4 集 free 首报动作 | 固定 HOLD refresh 候选 | refresh 发射 | 似然最大误差 | 批 0 PPO steps |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in r40.TRAIN_SEEDS:
        gate = _json(r40.ROUND_DIR / f"SEED{seed}_FIRST4_GATE.json")
        lines.append(
            f"| {seed} | {gate['free_first_report_decisions']} | "
            f"{gate['fixed_hold_refresh_decisions']} | {gate['refresh_emissions']} | "
            f"{gate['likelihood_alignment']['max_abs_gap']:.3g} | "
            f"{gate['batch_optimizer_steps']} |"
        )
    lines += [
        "",
        "### 2–4. NEW 相对 OLD：首报、全 HOLD、解码模式与 seed",
        "",
        "| Seed | 模式 | OLD 按时任务 | NEW 按时任务 | NEW−OLD 任务 | OLD 字节 | NEW 字节 | NEW−OLD 字节 | 任务改善/持平/退化场景 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in r40.TRAIN_SEEDS:
        for mode in ("greedy", "sampled"):
            row = results["paired_new_minus_old"][f"s{seed}/{mode}"]
            lines.append(
                f"| {seed} | {mode} | {row['old_on_time_total']} | "
                f"{row['new_on_time_total']} | {row['new_minus_old_on_time_total']:+d} | "
                f"{row['old_bytes_total']} | {row['new_bytes_total']} | "
                f"{row['new_minus_old_bytes_total']:+d} | "
                f"{row['task_improved_scenes']}/{row['task_tied_scenes']}/"
                f"{row['task_regressed_scenes']} |"
            )
    lines += [
        "",
        "NEW 获得和失去的按时目标 ID、逐场景任务差、最大损失和字节差保存在"
        " `PAIRED_RESULTS.json`。按 seed 与模式分别阅读，不合并成独立样本。",
        "",
        "### 5–6. 与 immediate、late147 和 always_hold 的绝对取舍",
        "",
        "| Arm | 开发集按时任务 | 平均任务/场景 | 计量字节 | 平均字节/场景 | 首报发射 | refresh 发射 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in (("immediate", {
            "on_time_total": sum(int(results["absolute_vs_immediate"][
                f"NEW/s{seed}/greedy"]["immediate_on_time_total"])
                for seed in r40.TRAIN_SEEDS) // len(r40.TRAIN_SEEDS),
            "bytes_total": sum(int(results["absolute_vs_immediate"][
                f"NEW/s{seed}/greedy"]["immediate_bytes_total"])
                for seed in r40.TRAIN_SEEDS) // len(r40.TRAIN_SEEDS),
            "first_report_emissions": "—", "refresh_emissions": "—",
        }), ("late_discovery_gate_147", results["rule_totals"]["late_discovery_gate"]),
            ("always_hold", results["rule_totals"]["always_hold"])):
        count = len(r40.DEV_SCENARIOS)
        task_total = row["on_time_total"]
        byte_total = row["bytes_total"]
        lines.append(f"| {arm} | {task_total} | {task_total / count:.3f} | "
                     f"{byte_total} | {byte_total / count:.1f} | "
                     f"{row['first_report_emissions']} | {row['refresh_emissions']} |")
    lines += [
        "",
        "NEW/OLD 对 immediate 与两个规则臂的逐 seed、逐模式比较已列入"
        " `PAIRED_RESULTS.json`。这组开发复用数据只描述本轮固定场景；不据此选择新阈值。",
        "",
        "## 训练和 checkpoint 身份",
        "",
        f"- PPO 批完成数：{sum(int(results['training_diagnostics'][str(seed)]['summary']['updates']) for seed in r40.TRAIN_SEEDS)}；"
        f"实际 optimizer.step：{sum(int(results['training_diagnostics'][str(seed)]['summary']['optimizer_steps']) for seed in r40.TRAIN_SEEDS)}。",
        "- ROUND36 的 ep0、ep64、ep128 文件均在启动前按字节哈希核验，并只读载入计算参数摘要。",
        "- NEW ep0 参数与相同 seed 的 ROUND36 ep0 逐参数摘要一致；NEW ep0/64/128 版本字段、"
        "episode、update、optimizer steps、lambda 和参数摘要见 `PAIRED_RESULTS.json`。",
        "- 每个 PPO 批都有 immutable pre-update 与 post-update checkpoint、batch 参数前后链、"
        "optimizer/lambda/RNG 恢复状态和 commit receipt。OLD 未写回。",
        "- 旧训练版本恢复到 NEW 被真实 `load_checkpoint` 入口拒绝；OLD 仅通过 read-only 入口评估。",
        "",
        "## 限制与后续",
        "",
        "ROUND39 的刷新抑制结果不能推出刷新普遍无用。本轮只检验训练候选动作资格变更，"
        "并且 OLD 也在相同的 first-report-only 外部运行限制下评估。开发场景此前已使用，"
        "结果不能替代独立确认。",
        "",
        "建议由决策 agent 基于逐 seed 配对结果决定是否设计一轮独立确认；本轮不自动续训、"
        "调参、改默认值或推送。",
        "",
    ]
    (r40.ROUND_DIR / "ROUND40_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _verdict(results: dict[str, Any]) -> dict[str, Any]:
    paired = results["paired_new_minus_old"]
    seed_mode_findings = {}
    for key, row in paired.items():
        seed_mode_findings[key] = {
            "new_minus_old_on_time_total": row["new_minus_old_on_time_total"],
            "new_minus_old_bytes_total": row["new_minus_old_bytes_total"],
            "all_hold_scenes": results["behavior_diagnostics"][key]["NEW"][
                "all_hold_scenes"],
            "immediate_equivalent_scenes": results["behavior_diagnostics"][key]["NEW"][
                "immediate_equivalent_scenes"],
        }
    per_seed_mode_task_signs = {
        mode: [paired[f"s{seed}/{mode}"]["new_minus_old_on_time_total"]
               for seed in r40.TRAIN_SEEDS]
        for mode in ("greedy", "sampled")
    }
    return {
        "kind": "round40-verdict/1",
        "training_refresh_suppression": "passed pure-compute, relay and all three first-four gates",
        "new_vs_old_by_seed_and_mode": seed_mode_findings,
        "greedy_and_sampled_task_signs": per_seed_mode_task_signs,
        "three_seed_consistency": {
            mode: ("all_improved" if all(value > 0 for value in values) else
                   "all_tied_or_improved" if all(value >= 0 for value in values) else
                   "all_tied_or_regressed" if all(value <= 0 for value in values) else
                   "mixed")
            for mode, values in per_seed_mode_task_signs.items()
        },
        "new_vs_immediate": {
            key: {"task_delta": row["policy_minus_immediate_on_time"],
                  "byte_delta": row["policy_minus_immediate_bytes"],
                  "same_task_lower_cost": row["policy_minus_immediate_on_time"] == 0
                      and row["policy_minus_immediate_bytes"] < 0}
            for key, row in results["absolute_vs_immediate"].items() if key.startswith("NEW/")
        },
        "development_block_confirmatory": False,
        "generalization_claim": False,
        "recommendation": "Use the fixed per-seed results to decide whether to design a separately authorized independent confirmation; do not reuse this development block for confirmation.",
    }


def analyze(common: Any, *, write_artifacts: bool = True) -> dict[str, Any]:
    claims = _completed_claims()
    references: dict[int, dict[str, Any]] = {}
    for scenario in sorted((*r40.TRAIN_SCENARIOS, *r40.DEV_SCENARIOS)):
        logical = f"reference_immediate_s{scenario}"
        claim = claims.get(logical)
        if claim is None:
            raise RuntimeError(f"immediate launch is missing: {scenario}")
        path = common.ROWS / f"v2_reference_s{scenario}_a{claim['attempt']}.json"
        row = _load_row(common, claims, logical, path)
        if row.get("reference_arm") != "immediate":
            raise RuntimeError(f"reference row is not immediate: {path}")
        references[scenario] = row
    if len(references) != 64:
        raise RuntimeError(f"expected 64 immediate references, got {len(references)}")

    evaluations: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for arm in ("OLD", "NEW"):
        for seed in r40.TRAIN_SEEDS:
            for mode in ("greedy", "sampled"):
                for scenario in r40.DEV_SCENARIOS:
                    logical = f"model_eval_{arm.lower()}_s{seed}_{mode}_scene{scenario}"
                    claim = claims.get(logical)
                    if claim is None:
                        raise RuntimeError(f"model evaluation is missing: {logical}")
                    row = _load_row(common, claims, logical,
                                    _eval_path(common, arm, seed, mode, scenario,
                                               int(claim["attempt"])))
                    if (row.get("model_arm") != arm or int(row["training_seed"]) != seed
                            or row.get("eval_mode") != mode
                            or int(row["scenario_seed"]) != scenario
                            or row.get("timepoint") != "ep128"):
                        raise RuntimeError(f"model evaluation identity mismatch: {logical}")
                    if row.get("checkpoint_identity", {}).get("arm") != arm:
                        raise RuntimeError(f"checkpoint identity is absent/mislabeled: {logical}")
                    if row.get("checkpoint_identity", {}).get("load_mode") != "read_only":
                        raise RuntimeError(f"model was not read-only loaded: {logical}")
                    if _emission_stats(row)["refresh_emissions"] != 0:
                        raise RuntimeError(f"refresh emissions were nonzero in {logical}")
                    evaluations[(arm, seed, mode, scenario)] = row
    if len(evaluations) != 192:
        raise RuntimeError(f"expected 192 model evaluation rows, got {len(evaluations)}")

    rules: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in r40.RULE_ARMS:
        for scenario in r40.DEV_SCENARIOS:
            logical = f"rule_{arm}_scene{scenario}"
            claim = claims.get(logical)
            if claim is None:
                raise RuntimeError(f"rule evaluation is missing: {logical}")
            path = common.ROWS / f"v2_r40_rule_{arm}_s{scenario}_a{claim['attempt']}.json"
            row = _load_row(common, claims, logical, path)
            if row.get("rule_identity", {}).get("arm") != arm:
                raise RuntimeError(f"rule arm identity mismatch: {logical}")
            rules[(arm, scenario)] = row
    if len(rules) != 32:
        raise RuntimeError(f"expected 32 rule rows, got {len(rules)}")

    paired: dict[str, Any] = {}
    versus_reference: dict[str, Any] = {}
    behavior: dict[str, Any] = {}
    for seed in r40.TRAIN_SEEDS:
        for mode in ("greedy", "sampled"):
            key = f"s{seed}/{mode}"
            pairs = []
            for scenario in r40.DEV_SCENARIOS:
                old = evaluations[("OLD", seed, mode, scenario)]
                new = evaluations[("NEW", seed, mode, scenario)]
                immediate = references[scenario]
                old_targets, new_targets = _targets(old), _targets(new)
                old_target_diff = old_targets - new_targets
                new_target_diff = new_targets - old_targets
                old_seed = old.get("action_rng_seed")
                new_seed = new.get("action_rng_seed")
                if old_seed != new_seed:
                    raise RuntimeError(f"OLD/NEW action RNG seed mismatch: {key}, {scenario}")
                pairs.append({
                    "scenario_seed": scenario,
                    "old_on_time": int(old["on_time"]),
                    "new_on_time": int(new["on_time"]),
                    "new_minus_old_on_time": int(new["on_time"]) - int(old["on_time"]),
                    "old_on_time_targets": sorted(old_targets),
                    "new_on_time_targets": sorted(new_targets),
                    "new_gained_targets": sorted(new_target_diff),
                    "new_lost_targets": sorted(old_target_diff),
                    "old_spent_bytes": int(old["spent_bytes"]),
                    "new_spent_bytes": int(new["spent_bytes"]),
                    "new_minus_old_bytes": int(new["spent_bytes"]) - int(old["spent_bytes"]),
                    "immediate_on_time": int(immediate["on_time"]),
                    "immediate_bytes": int(immediate["spent_bytes"]),
                    "old_minus_immediate_on_time": int(old["on_time"])
                        - int(immediate["on_time"]),
                    "new_minus_immediate_on_time": int(new["on_time"])
                        - int(immediate["on_time"]),
                    "old_minus_immediate_bytes": int(old["spent_bytes"])
                        - int(immediate["spent_bytes"]),
                    "new_minus_immediate_bytes": int(new["spent_bytes"])
                        - int(immediate["spent_bytes"]),
                    "action_rng_seed": old_seed,
                })
            dt = [row["new_minus_old_on_time"] for row in pairs]
            dc = [row["new_minus_old_bytes"] for row in pairs]
            paired[key] = {
                "scenes": len(pairs),
                "old_on_time_total": sum(row["old_on_time"] for row in pairs),
                "new_on_time_total": sum(row["new_on_time"] for row in pairs),
                "new_minus_old_on_time_total": sum(dt),
                "new_minus_old_on_time_mean_per_scene": sum(dt) / len(pairs),
                "task_improved_scenes": sum(value > 0 for value in dt),
                "task_tied_scenes": sum(value == 0 for value in dt),
                "task_regressed_scenes": sum(value < 0 for value in dt),
                "largest_scene_task_loss": min(dt),
                "old_bytes_total": sum(row["old_spent_bytes"] for row in pairs),
                "new_bytes_total": sum(row["new_spent_bytes"] for row in pairs),
                "new_minus_old_bytes_total": sum(dc),
                "new_minus_old_bytes_mean_per_scene": sum(dc) / len(pairs),
                "task_and_cost_both_improved_scenes": sum(
                    row["new_minus_old_on_time"] > 0 and row["new_minus_old_bytes"] < 0
                    for row in pairs),
                "task_tied_and_cost_lower_scenes": sum(
                    row["new_minus_old_on_time"] == 0 and row["new_minus_old_bytes"] < 0
                    for row in pairs),
                "paired_rows": pairs,
            }
            for arm in ("OLD", "NEW"):
                rows = [evaluations[(arm, seed, mode, scenario)]
                        for scenario in r40.DEV_SCENARIOS]
                immediate_rows = [references[scenario] for scenario in r40.DEV_SCENARIOS]
                rule_summaries = {}
                for rule in r40.RULE_ARMS:
                    rule_rows = [rules[(rule, scenario)] for scenario in r40.DEV_SCENARIOS]
                    rule_summaries[rule] = {
                        "rule_on_time_total": sum(int(row["on_time"]) for row in rule_rows),
                        "rule_bytes_total": sum(int(row["spent_bytes"]) for row in rule_rows),
                        "policy_on_time_total": sum(int(row["on_time"]) for row in rows),
                        "policy_bytes_total": sum(int(row["spent_bytes"]) for row in rows),
                        "policy_minus_rule_on_time": sum(int(row["on_time"]) for row in rows)
                            - sum(int(row["on_time"]) for row in rule_rows),
                        "policy_minus_rule_bytes": sum(int(row["spent_bytes"]) for row in rows)
                            - sum(int(row["spent_bytes"]) for row in rule_rows),
                        "per_scene": [{
                            "scenario_seed": scenario,
                            "policy_on_time": int(evaluations[(arm, seed, mode, scenario)]["on_time"]),
                            "rule_on_time": int(rules[(rule, scenario)]["on_time"]),
                            "policy_bytes": int(evaluations[(arm, seed, mode, scenario)]["spent_bytes"]),
                            "rule_bytes": int(rules[(rule, scenario)]["spent_bytes"]),
                        } for scenario in r40.DEV_SCENARIOS],
                    }
                versus_reference[f"{arm}/s{seed}/{mode}"] = {
                    "policy_on_time_total": sum(int(row["on_time"]) for row in rows),
                    "immediate_on_time_total": sum(int(row["on_time"])
                                                    for row in immediate_rows),
                    "policy_minus_immediate_on_time": sum(int(row["on_time"]) for row in rows)
                        - sum(int(row["on_time"]) for row in immediate_rows),
                    "policy_bytes_total": sum(int(row["spent_bytes"]) for row in rows),
                    "immediate_bytes_total": sum(int(row["spent_bytes"])
                                                 for row in immediate_rows),
                    "policy_minus_immediate_bytes": sum(int(row["spent_bytes"]) for row in rows)
                        - sum(int(row["spent_bytes"]) for row in immediate_rows),
                    "rule_comparisons": rule_summaries,
                }
            behavior[key] = {}
            for arm in ("OLD", "NEW"):
                arm_rows = [evaluations[(arm, seed, mode, scene)]
                            for scene in r40.DEV_SCENARIOS]
                immediate_equivalent = 0
                all_hold_scenes = 0
                delayed_task_preserved = 0
                diagnostics = []
                for scene, row in zip(r40.DEV_SCENARIOS, arm_rows):
                    immediate = references[scene]
                    sequence = [(int(entry[0]), str(entry[1]), int(entry[2]), int(entry[3]))
                                for entry in row["emission_sequence"] if bool(entry[5])]
                    immediate_sequence = [(int(entry[0]), str(entry[1]), int(entry[2]),
                                           int(entry[3]))
                                          for entry in immediate["emission_sequence"]
                                          if bool(entry[5])]
                    same = sequence == immediate_sequence
                    immediate_equivalent += int(same)
                    stats = _emission_stats(row)
                    all_hold = stats["first_report_emissions"] == 0
                    all_hold_scenes += int(all_hold)
                    delayed = (not same and bool(sequence)
                               and int(row["on_time"]) == int(immediate["on_time"]))
                    delayed_task_preserved += int(delayed)
                    diagnostics.append({"scenario_seed": scene,
                                        "immediate_equivalent_first_sequence": same,
                                        "no_first_report_emission": all_hold,
                                        "delayed_first_report_with_task_count_preserved": delayed,
                                        "first_report_emissions": stats["first_report_emissions"],
                                        "first_report_bytes": stats["first_report_bytes"],
                                        "budget_masked": stats["budget_masked"],
                                        "budget_refusals": stats["budget_refusals"],
                                        "capacity_refusals": stats["capacity_refusals"],
                                        "refresh_emissions": stats["refresh_emissions"]})
                behavior[key][arm] = {
                    "immediate_equivalent_scenes": immediate_equivalent,
                    "all_hold_scenes": all_hold_scenes,
                    "delayed_first_report_task_preserved_scenes": delayed_task_preserved,
                    "per_scene": diagnostics,
                }

    rule_totals = {}
    for arm in r40.RULE_ARMS:
        rows = [rules[(arm, scene)] for scene in r40.DEV_SCENARIOS]
        rule_totals[arm] = {
            "scenes": len(rows),
            "on_time_total": sum(int(row["on_time"]) for row in rows),
            "on_time_mean_per_scene": sum(int(row["on_time"]) for row in rows) / len(rows),
            "bytes_total": sum(int(row["spent_bytes"]) for row in rows),
            "bytes_mean_per_scene": sum(int(row["spent_bytes"]) for row in rows) / len(rows),
            "first_report_emissions": sum(_emission_stats(row)["first_report_emissions"]
                                          for row in rows),
            "refresh_emissions": sum(_emission_stats(row)["refresh_emissions"]
                                     for row in rows),
            "per_scene": [{"scenario_seed": scene, "on_time": int(rules[(arm, scene)]["on_time"]),
                           "bytes": int(rules[(arm, scene)]["spent_bytes"])}
                          for scene in r40.DEV_SCENARIOS],
        }

    import sys

    trainer = sys.modules["v2_trainer"]
    freeze = _json(r40.ROUND_DIR / "ROUND40_FROZEN_CONFIG.json")["round40"][
        "freeze_identity"]
    training: dict[str, Any] = {}
    for seed in r40.TRAIN_SEEDS:
        summary_path = r40.ROUND_DIR / "training_progress" / f"TRAINING_SUMMARY_s{seed}_ep128.json"
        summary = _json(summary_path)
        if (int(summary["episodes_done"]) != 128 or int(summary["updates"]) != 32
                or summary["training_version"] != "constrained-cost-v2/first-report-only/1/"
                "published-completion-event-at-relay-tick/2"
                or summary["comm_action_eligibility"] != "first-report-only/1"):
            raise RuntimeError(f"seed {seed} final training identity/cursor is incomplete")
        batches = []
        frozen = _json(r40.ROUND_DIR / "ROUND40_FROZEN_CONFIG.json")["round40"][
            "freeze_identity"]
        initial_parameter = frozen["initial_parameter_identities"][str(seed)]
        for batch_index in range(32):
            commit_path = common.CHECKPOINTS / f"r40_batch_commit_s{seed}_b{batch_index}.json"
            commit = _json(commit_path)
            batch_path = common.CHECKPOINTS / commit["batch_record"]
            postupdate = common.CHECKPOINTS / commit["postupdate_checkpoint"]
            if (_sha256(batch_path) != commit["batch_record_sha256"]
                    or _sha256(postupdate) != commit["postupdate_checkpoint_sha256"]):
                raise RuntimeError(f"seed {seed} batch {batch_index} immutable identity mismatch")
            batch = _json(batch_path)
            expected_before = (initial_parameter if batch_index == 0 else
                               batches[-1]["parameter_identity_after"])
            if (int(batch["batch"]) != batch_index
                    or batch["controller_parameters_before"]["parameter_sha256"]
                    != expected_before):
                raise RuntimeError(f"seed {seed} parameter chain breaks at batch {batch_index}")
            preupdate = common.CHECKPOINTS / batch["preupdate_checkpoint"]
            if _sha256(preupdate) != batch["preupdate_checkpoint_sha256"]:
                raise RuntimeError(f"seed {seed} pre-update checkpoint changed: {preupdate}")
            pre_model, pre_state, _pre_identity = trainer.load_checkpoint_read_only(
                preupdate, freeze=freeze)
            post_model, post_state, _post_identity = trainer.load_checkpoint_read_only(
                postupdate, freeze=freeze)
            if (pre_model.parameter_summary()["parameter_sha256"] != expected_before
                    or post_model.parameter_summary()["parameter_sha256"]
                    != batch["controller_parameters_after"]["parameter_sha256"]
                    or int(pre_state["updates"]) != batch_index
                    or int(pre_state["episodes_done"]) != batch_index * r40.EPISODES_PER_UPDATE
                    or int(post_state["updates"]) != batch_index + 1
                    or int(post_state["episodes_done"])
                    != (batch_index + 1) * r40.EPISODES_PER_UPDATE):
                raise RuntimeError(f"seed {seed} pre/post checkpoint state chain breaks at {batch_index}")
            free = 0
            refresh = 0
            for episode_artifact in batch["episode_artifacts"]:
                tape_path = Path(episode_artifact["tape_path"])
                if _sha256(tape_path) != episode_artifact["tape_sha256"]:
                    raise RuntimeError(f"seed {seed} training tape changed: {tape_path}")
                for line in tape_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    tape = json.loads(line)
                    free += len(tape["actionable"])
                    refresh += len(tape["fixed_hold_refresh"])
            update = batch["update"]
            batches.append({
                "batch": batch_index,
                "episodes": batch["episodes"],
                "scenarios": batch["scenarios"],
                "g_hat": batch["constraint"]["g_hat"],
                "lambda_before": batch["dual"]["lambda_before"],
                "lambda_after": batch["dual"]["lambda_next"],
                "free_first_report_decisions": free,
                "fixed_hold_refresh_decisions": refresh,
                "optimizer_steps": int(update["optimizer_steps"]),
                "likelihood_max_abs_gap": update["likelihood_alignment"]["max_abs_gap"],
                "module_gradient_l1_mean": update["module_gradient_l1_mean"],
                "parameter_identity_before": batch["controller_parameters_before"][
                    "parameter_sha256"],
                "parameter_identity_after": batch["controller_parameters_after"][
                    "parameter_sha256"],
                "preupdate_checkpoint": batch["preupdate_checkpoint"],
                "postupdate_checkpoint": postupdate.name,
                "postupdate_checkpoint_sha256": _sha256(postupdate),
            })
        if len(batches) != 32:
            raise RuntimeError(f"seed {seed} does not have 32 committed batch records")
        training[str(seed)] = {"summary": summary, "batches": batches}

    freeze_doc = _json(r40.ROUND_DIR / "ROUND40_FROZEN_CONFIG.json")
    freeze = freeze_doc["round40"]["freeze_identity"]
    preflight = _json(r40.ROUND_DIR / "ROUND40_PREFLIGHT.json")
    checkpoint_results: dict[str, Any] = {
        "OLD_read_only_inputs": {}, "NEW_training_outputs": {},
        "preserved_preconstruction_checkpoints": preflight.get(
            "preserved_preconstruction_checkpoints", []),
    }
    for seed in r40.TRAIN_SEEDS:
        old_record = preflight["checkpoint_identities"]["old_checkpoint_milestones"][str(seed)]
        old_ep128 = Path(old_record["ep128"]["path"])
        old_sha = _sha256(old_ep128)
        if old_sha != old_record["ep128"]["file_sha256"]:
            raise RuntimeError(f"OLD checkpoint bytes changed during ROUND40: {old_ep128}")
        checkpoint_results["OLD_read_only_inputs"][str(seed)] = {
            "path": str(old_ep128), "size_bytes": old_ep128.stat().st_size,
            "file_sha256": old_sha,
            "parameter_sha256": old_record["ep128"]["parameter_sha256"],
            "bytes_unchanged_from_prelaunch": True,
            "read_only_evaluation_only": True,
        }
        new_points = {}
        milestone_paths = summary.get("milestone_checkpoints", {})
        for episodes in (0, 64, 128):
            milestone_value = milestone_paths.get(str(episodes))
            if not milestone_value:
                raise RuntimeError(f"seed {seed} lacks a versioned ep{episodes} milestone path")
            path = Path(milestone_value)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            controller, state, identity = trainer.load_checkpoint_read_only(path, freeze=freeze)
            if (payload.get("training_version") != "constrained-cost-v2/first-report-only/1/"
                    "published-completion-event-at-relay-tick/2"
                    or payload.get("comm_action_eligibility") != "first-report-only/1"
                    or int(state["episodes_done"]) != episodes
                    or int(state["updates"]) != episodes // 4
                    or (episodes == 0 and int(state["optimizer_steps"]) != 0)
                    or (episodes == 64 and int(state["optimizer_steps"]) <= 0)
                    or (episodes == 128 and int(state["optimizer_steps"])
                        != int(training[str(seed)]["summary"]["optimizer_steps"]))
                    or identity.get("read_only_for_round40") is not True):
                raise RuntimeError(f"NEW checkpoint identity/cursor mismatch: {path}")
            if episodes == 0 and controller.parameter_summary()["parameter_sha256"] \
                    != old_record["ep0"]["parameter_sha256"]:
                raise RuntimeError(f"NEW seed {seed} ep0 differs from the registered OLD ep0")
            if episodes == 128 and controller.parameter_summary()["parameter_sha256"] \
                    != training[str(seed)]["summary"]["controller_parameters"]["parameter_sha256"]:
                raise RuntimeError(f"NEW seed {seed} final checkpoint differs from training summary")
            if episodes in (64, 128):
                batch_index = episodes // 4 - 1
                batch = _json(common.CHECKPOINTS / _json(
                    common.CHECKPOINTS / f"r40_batch_commit_s{seed}_b{batch_index}.json")[
                        "batch_record"])
                if batch["controller_parameters_after"]["parameter_sha256"] \
                        != controller.parameter_summary()["parameter_sha256"]:
                    raise RuntimeError(f"NEW ep{episodes} parameter identity breaks batch chain")
            new_points[f"ep{episodes}"] = {
                "path": str(path), "size_bytes": path.stat().st_size,
                "file_sha256": _sha256(path),
                "parameter_sha256": controller.parameter_summary()["parameter_sha256"],
                "episodes_done": int(state["episodes_done"]),
                "updates": int(state["updates"]),
                "optimizer_steps": int(state["optimizer_steps"]),
                "lambda": float(state["lambda"]),
                "training_version": payload["training_version"],
                "comm_action_eligibility": payload["comm_action_eligibility"],
                "read_only_identity": identity,
            }
        checkpoint_results["NEW_training_outputs"][str(seed)] = new_points

    ledger = r40.ledger_snapshot()
    preconstruction_dir = r40.ROUND_DIR / "preconstruction_failures"
    preconstruction_records = [
        _json(path) for path in sorted(preconstruction_dir.glob("failure_*.json"))
    ] if preconstruction_dir.exists() else []
    ledger["preconstruction_failure_count"] = len(preconstruction_records)
    ledger["preconstruction_failures"] = preconstruction_records
    if ledger["normal_used"] != r40.NORMAL_CAP or ledger["total_used"] != r40.NORMAL_CAP + ledger[
            "reserve_used"]:
        raise RuntimeError(f"ROUND40 launch ledger does not reconcile to the 672 plan: {ledger}")
    if ledger["reserve_used"] > r40.RESERVE_CAP:
        raise RuntimeError("ROUND40 technical reserve was exceeded")
    by_logical: dict[str, list[dict[str, Any]]] = {}
    for claim in ledger["claims"]:
        by_logical.setdefault(str(claim["logical_job"]), []).append(claim)
    latest_by_logical = {logical: sorted(attempts, key=lambda row: int(row["attempt"]))[-1]
                         for logical, attempts in by_logical.items()}
    if len(latest_by_logical) != r40.NORMAL_CAP or any(
            item.get("status") != "complete" for item in latest_by_logical.values()):
        raise RuntimeError("ROUND40 plan has missing logical jobs or unfinished latest attempts")
    technical_failures = sum(item.get("status") == "failed_technical"
                             for item in ledger["claims"])
    nonterminal = [item for item in ledger["claims"]
                   if item.get("status") not in ("complete", "failed_technical")]
    if nonterminal:
        raise RuntimeError(f"ROUND40 has unfinished launch attempts: {nonterminal[:3]}")
    ledger["technical_failures"] = technical_failures
    ledger["completed_attempts"] = sum(item.get("status") == "complete"
                                       for item in ledger["claims"])
    ledger["completed_logical_jobs"] = len(latest_by_logical)

    # Hash every raw run artifact outside Git and list every frozen checkpoint input.
    raw_files = []
    for folder in (common.LOGS, common.ROWS, common.CHECKPOINTS,
                   r40.ROUND_DIR / "launch_claims", r40.ROUND_DIR / "launch_slots",
                   r40.ROUND_DIR / "training_progress",
                   r40.ROUND_DIR / "preconstruction_failures"):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                raw_files.append({"path": str(path), "size_bytes": path.stat().st_size,
                                  "sha256": _sha256(path)})
    artifacts_manifest = {
        "kind": "round40-git-external-artifact-manifest/1",
        "raw_artifact_count": len(raw_files),
        "raw_artifacts": raw_files,
        "input_root": str(common.ROOT),
        "round36_and_frozen_input_hashes": _json(
            r40.ROUND_DIR / "ROUND40_FROZEN_CONFIG.json")["round40"]["freeze_identity"][
                "round36_dependency_hashes"],
        "prelaunch_checkpoint_identity_table": _json(
            r40.ROUND_DIR / "ROUND40_PREFLIGHT.json")["checkpoint_identities"][
                "old_checkpoint_milestones"],
    }

    results = {
        "kind": "round40-fixed-development-comparison/1",
        "development_block": list(r40.DEV_SCENARIOS),
        "independent_test_or_generalization_claim": False,
        "inferential_statistics": "not performed",
        "pooled_seed_claim": False,
        "paired_new_minus_old": paired,
        "absolute_vs_immediate": versus_reference,
        "behavior_diagnostics": behavior,
        "rule_totals": rule_totals,
        "training_diagnostics": training,
        "checkpoint_results": checkpoint_results,
        "launch_ledger": ledger,
        "reading_rules": [
            "NEW matching immediate only supports learning rule behavior; it does not establish additional learned value.",
            "NEW all-HOLD means this action-space change did not resolve the all-HOLD behavior.",
            "A single improved seed does not establish training stability.",
            "Lower bytes with lost tasks is a tradeoff, not goal completion.",
            "This development block has prior use and is not confirmatory.",
        ],
    }
    verdict = _verdict(results)
    if write_artifacts:
        (r40.ROUND_DIR / "PAIRED_RESULTS.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (r40.ROUND_DIR / "GIT_OUTSIDE_ARTIFACTS.json").write_text(
            json.dumps(artifacts_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        (r40.ROUND_DIR / "TRAINING_DIAGNOSTICS.json").write_text(
            json.dumps({"kind": "round40-training-diagnostics/1",
                        "by_seed": training}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        (r40.ROUND_DIR / "CHECKPOINT_RECONCILIATION.json").write_text(
            json.dumps({"kind": "round40-checkpoint-reconciliation/1",
                        "validation_layers": {
                            "file_hash": "passed",
                            "metadata_and_parameter_identity": "passed",
                            "training_cursor_and_parameter_chain": "passed",
                            "forward_and_likelihood": "see per-batch training diagnostics and preflight",
                        },
                        "checkpoints": checkpoint_results,
                        "batch_count_by_seed": {seed: len(row["batches"])
                                                for seed, row in training.items()},
                        "old_checkpoint_writeback_count": 0},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (r40.ROUND_DIR / "VERDICT.json").write_text(
            json.dumps(verdict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_report(results, artifacts_manifest=artifacts_manifest)
    return results


__all__ = ["analyze"]
