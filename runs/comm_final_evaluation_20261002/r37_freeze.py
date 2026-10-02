"""ROUND37 冻结件：在首次环境启动之前把评估对象、场景块、九臂、配置与统计方案全部登记。

本脚本 0 环境启动。它只做核验与登记，并写下两份交付件：

- `EVALUATION_IDENTITY.json`：起点核验（分支、完整头、ROUND36 报告与模型索引、通过项引用）；
- `FROZEN_EVALUATION_PLAN.json`：三个模型的路径/字节 SHA256/参数身份/两个版本字段、解码语义、
  搜索与分配模型哈希、完整环境与通信配置、64 场景块、九臂清单与额度算术、固定统计方案。

一旦任何评估产物存在，本脚本拒绝重写冻结件（授权边界 §3"模型、规则、场景清单、配置和统计方案
全部先冻结，再生成或运行场景"）。

    /d/project_env/python.exe r37_freeze.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy
import torch

import r37_common as r37

ROUND36_DELIVERABLES = {
    "report": r37.ROUND36 / "ROUND36_TRAIN_REPORT.md",
    "verification": r37.ROUND36 / "ROUND36_VERIFICATION.json",
    "recheck": r37.ROUND36 / "ROUND36_INDEPENDENT_RECHECK.json",
    "launch_book": r37.ROUND36 / "TRAIN_LAUNCH_BOOK.json",
    "registry": r37.ROUND36 / "delivery_draft" / "ARTEFACT_HASH_REGISTRY.json",
    "frozen_config": r37.ROUND36 / "FROZEN_CONFIG_MANIFEST.json"}


def identity_block() -> dict[str, Any]:
    branch = r37.git("rev-parse", "--abbrev-ref", "HEAD").strip()
    head = r37.git("rev-parse", "HEAD").strip()
    dirty = [line for line in r37.git("status", "--porcelain", "--untracked-files=no")
             .splitlines() if line.strip()]
    return {"branch": branch, "head": head,
            "expected_branch": r37.EXPECTED_BRANCH, "expected_head": r37.EXPECTED_HEAD,
            "head_is_the_round36_delivery_head": head == r37.EXPECTED_HEAD,
            "tracked_modifications_at_start": dirty,
            "round36_deliverables_present": {
                key: path.exists() for key, path in ROUND36_DELIVERABLES.items()},
            "round34_contract_present": (r37.ROUND34 / "TASK_REWARD_TIMING_CONTRACT.md").exists(),
            "round35_smoke_report_present": (r37.ROUND35 / "ROUND35_SMOKE_REPORT.md").exists()}


def carried_forward() -> dict[str, Any]:
    """Read the previous rounds' passing items and residual risks instead of re-running them."""
    verification = r37.read_json(ROUND36_DELIVERABLES["verification"])
    recheck = r37.read_json(ROUND36_DELIVERABLES["recheck"])
    smoke = r37.read_json(r37.ROUND35 / "ROUND35_PREFLIGHT.json")
    boundary = r37.read_json(r37.ROUND35 / "RESTORE_BOUNDARY.json")
    content = r37.read_json(r37.ROUND35 / "CHECKPOINT_CONTENT_LAYER.json")
    report = ROUND36_DELIVERABLES["report"].read_text("utf-8")
    return {
        "round36": {"all_verification_checks_passed": verification["all_passed"],
                    "checks": {key: value["passed"]
                               for key, value in verification["checks"].items()},
                    "independent_recheck_passed": recheck["all_passed"],
                    "recheck_items": {key: value["passed"]
                                      for key, value in recheck["results"].items()},
                    "quota_used": {key: r37.read_json(ROUND36_DELIVERABLES["launch_book"])[key]
                                   for key in ("training_launches", "reserve_reruns",
                                               "episode_constructions", "optimizer_steps")}},
        "round35": {"preflight_clean": smoke["clean"], "gate_count": len(smoke["gates"]),
                    "restore_boundary_clean": boundary["clean"],
                    "checkpoint_content_layer_clean": content["clean"],
                    "smoke_verdict": "go"},
        "round34": {"contract": "TASK_REWARD_TIMING_CONTRACT.md",
                    "training_version_manifest": "TRAINING_VERSION_MANIFEST.json",
                    "restore_policy": "legacy checkpoints are refused by the version gate and "
                                      "are only readable through load_checkpoint_read_only"},
        "residual_risks_carried": [line.strip() for line in report.splitlines()
                                   if line.strip().startswith(("1. **", "2. **", "3. **", "4. **",
                                                               "5. **", "6. **", "7. **", "8. **"))
                                   and "残余风险" not in line][:8],
        "not_re_executed": ["ROUND34 回归套件", "ROUND35 smoke 与恢复边界",
                            "ROUND36 的 G1-G12 / V1-V6 / R1-R6"],
        "note": "本轮按授权只做必要身份核验：读取上一轮已交付的通过项，不重跑历史套件"}


def environment_block() -> dict[str, Any]:
    import v2_common

    fields, substituted = v2_common.stage_b_inherited_config(r37.SCENARIOS[0])
    return {"interpreter": sys.executable, "python": sys.version.split()[0],
            "torch": torch.__version__, "numpy": numpy.__version__,
            "new_virtualenv": False, "dependencies_installed": 0,
            "steps_per_episode": v2_common.STEPS_PER_EPISODE,
            "inherited_world_fields": {key: fields[key] for key in sorted(fields)},
            "substituted_fields_note": substituted[-1],
            "transport_overrides": v2_common.STAGE_B_INHERITED_OVERRIDES,
            "budget_bytes": v2_common.STAGE_B_INHERITED_OVERRIDES["comm_budget_bytes"],
            "b0_bytes": v2_common.B0_BYTES, "delta": v2_common.DELTA,
            "delivery_probability": v2_common.STAGE_B_INHERITED_OVERRIDES["comm_delivery_probability"],
            "ack_enabled": v2_common.STAGE_B_INHERITED_OVERRIDES["comm_ack"],
            "replay_or_online": "每个 cell 独立在线运行一集（1000 步），不做回放；与训练块同一运行方式"}


def configuration_block() -> dict[str, Any]:
    code = r36_code_identity()
    return {"frozen_search_and_allocation_weights": code["frozen_checkpoints"],
            "production_files": code["production"],
            "stage_c_harness": code["stage_c_harness"],
            "training_version": r37.TRAINING_VERSION,
            "reward_timing_version": r37.REWARD_TIMING_VERSION,
            "trainer_tag": r37.v2_trainer.TRAINER_TAG,
            "loss_version": r37.v2_trainer.LOSS_VERSION,
            "inherited_freeze_token": r37.freeze_identity(),
            "reward_capture_in_eval_rows": "评估不做更新；每行必须带 "
                                           "reward_timing_version="
                                           "published-completion-event-at-relay-tick/2"
                                           "（技术门禁检查项）"}


def r36_code_identity() -> dict[str, Any]:
    production = {name: r37.sha256_file(r37.ROOT / name)
                  for name in r37.v2_common.PRODUCTION_DEPENDENCIES}
    harness = {name: r37.sha256_file(r37.STAGE_C / name) for name in r37.STAGE_C_MODULES
               if (r37.STAGE_C / name).exists()}
    checkpoints = {str(path.relative_to(r37.ROOT)).replace("\\", "/"):
                   r37.sha256_file(path) for path in r37.v2_common.CHECKPOINT_FILES}
    return {"production": production, "stage_c_harness": harness,
            "frozen_checkpoints": checkpoints}


def quota_block() -> dict[str, Any]:
    return {"arms": list(r37.ALL_ARMS), "arm_count": len(r37.ALL_ARMS),
            "scenarios": [r37.SCENARIOS[0], r37.SCENARIOS[-1]], "scenario_count": r37.SCENARIO_COUNT,
            "normal_evaluation_launches": r37.NORMAL_EVALUATION_LAUNCHES,
            "recording_reserve": r37.RECORDING_RESERVE,
            "recording_reserve_needed": False,
            "recording_reserve_reason": "immediate 本身就是九臂之一且其行即为配对参照，不需要另录"
                                        "独立参照流；因此正常评估就是 576 次，未用额度不得挪用",
            "failure_reserve": r37.FAILURE_RESERVE,
            "hard_cap": r37.HARD_CAP,
            "max_parallel_workers": r37.MAX_PARALLEL_WORKERS,
            "training_launches_allowed": r37.TRAINING_LAUNCHES_ALLOWED,
            "optimizer_steps_allowed": r37.OPTIMIZER_STEPS_ALLOWED,
            "arithmetic": f"{len(r37.ALL_ARMS)} arms x {r37.SCENARIO_COUNT} scenarios = "
                          f"{r37.NORMAL_EVALUATION_LAUNCHES}; + {r37.RECORDING_RESERVE} + "
                          f"{r37.FAILURE_RESERVE} <= {r37.HARD_CAP}",
            "first_scenario_gate": {"scenario": r37.SCENARIOS[0], "arms": len(r37.ALL_ARMS),
                                    "counted_in_normal_launches": True,
                                    "effect_based_stop": "禁止：只按技术正确性决定是否扩展"}}


def main(argv: list[str]) -> int:
    rows_exist = (r37.HERE / "rows").exists() and any((r37.HERE / "rows").glob("*.json"))
    if rows_exist and (r37.HERE / "FROZEN_EVALUATION_PLAN.json").exists():
        raise SystemExit("evaluation rows already exist and the plan is frozen; refusing to "
                         "rewrite the frozen registration -- the plan must precede the launches")
    identity = identity_block()
    if identity["branch"] != identity["expected_branch"]:
        raise SystemExit(f"wrong branch: {identity['branch']}")
    if identity["head"] != identity["expected_head"]:
        raise SystemExit(f"wrong head: {identity['head']} != {identity['expected_head']}")
    if identity["tracked_modifications_at_start"]:
        raise SystemExit(f"working tree is not clean: {identity['tracked_modifications_at_start']}")
    if not all(identity["round36_deliverables_present"].values()):
        raise SystemExit(f"missing ROUND36 deliverables: {identity['round36_deliverables_present']}")

    r37.configure()
    audit = r37.read_json(r37.HERE / "SCENE_USAGE_AUDIT.json")
    if not audit["passed"] or audit["chosen_block"] != [r37.SCENARIOS[0], r37.SCENARIOS[-1]]:
        raise SystemExit(f"the registered block is not what the audit chose: {audit['passed']}")
    models = r37.model_registry()
    if not models["all_match_round36_registry"]:
        raise SystemExit("a model identity does not match ROUND36's registry")
    if r37.book()["starts"] or r37.claims_held():
        raise SystemExit("launches already booked before the freeze; the plan must precede them")

    r37.write_json("EVALUATION_IDENTITY.json", {
        "kind": "round37-evaluation-identity", "schema": "round37-eval-identity/1",
        "round": "37", "written_at": __import__("time").strftime("%F %T"),
        "start": identity, "carried_forward": carried_forward(),
        "numbering_convention": r37.NUMBERING_CONVENTION,
        "environment": environment_block(),
        "scenario_audit": {"chosen_block": audit["chosen_block"],
                           "occupied_members_in_block": audit["candidate_blocks_tested_in_fixed_order"][0],
                           "history_and_outside_runs": audit["history_and_outside_runs_checks"],
                           "outcome_data_read": False},
        "model_registry": models,
        "all_passed": True}, refresh=True)
    r37.write_json("FROZEN_EVALUATION_PLAN.json", {
        "kind": "round37-frozen-evaluation-plan", "schema": "round37-frozen-plan/1",
        "round": "37", "status": "frozen before the first environment start",
        "written_at": __import__("time").strftime("%F %T"),
        "branch": identity["branch"], "head": identity["head"],
        "models": models["models"], "episodes_vs_batches": r37.NUMBERING_CONVENTION,
        "scenarios": list(r37.SCENARIOS), "scenario_provenance": {
            "block": audit["chosen_block"],
            "basis": "occupancy only: no artefact, registration, claim or committed content uses "
                     "1600000..1600063; outcomes were never read when choosing it",
            "audit_artefact": "SCENE_USAGE_AUDIT.json",
            "historical_or_development_blocks_reused": False,
            "why_not_the_old_blocks": "封存块与开发块不能冒充新测试；本块在路径、跟踪文本、"
                                      "全 ref git 历史与 runs/ 之外均无占用记录"},
        "arms": [{"id": arm, "kind": r37.parse_arm(arm)[0],
                  "seed": r37.parse_arm(arm)[1], "mode": r37.parse_arm(arm)[2],
                  "rule_params": (r37.v2_common.RULE_ARMS.get(arm, {}).get("params")
                                  if r37.parse_arm(arm)[0] == "rule" else None)}
                 for arm in r37.ALL_ARMS],
        "decoder_semantics": r37.DECODER_SEMANTICS,
        "configuration": configuration_block(),
        "statistics": r37.STATISTICS,
        "quota": quota_block(),
        "reporting_rules": [
            "greedy 是主表，sampled 是辅助表，两者不合并成一个均值",
            "三个训练 seed 分别报告，不池化成独立场景样本",
            "所有表格与文字引用同一份统计结果件",
            "区间包含零不等于等效；未观察到任务损失不等于 δ=0 的总体保证",
            "单个 seed 成功不代表训练算法稳定；不得挑一个显著结果宣称整体方法成立",
            "sampled 每场景一次不是完整采样不确定性评估",
            "场景配对不等于逐消息信道实现相同：不同发送轨迹遭遇的是各自独立的丢包实现"],
        "prohibitions": ["0 训练、0 optimizer.step", "不挑最佳 seed、不改用中间里程碑",
                        "不调阈值、不改默认协议", "不因首场景效果换模型或停止扩展",
                        "性能差不算技术失败", "不部署、不推送、不替用户决定最终部署选择"]},
        refresh=True)
    print(__import__("json").dumps({
        "branch": identity["branch"], "head": identity["head"][:12],
        "models_registered": len([k for k in models["models"] if k.startswith("s")]),
        "all_model_identities_match_round36": models["all_match_round36_registry"],
        "scenario_block": audit["chosen_block"], "arms": len(r37.ALL_ARMS),
        "normal_launches": r37.NORMAL_EVALUATION_LAUNCHES,
        "hard_cap": r37.HARD_CAP, "frozen_plan_written": True}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
