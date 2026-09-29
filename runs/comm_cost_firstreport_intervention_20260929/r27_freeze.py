"""ROUND27 启动前冻结：路径定义、干预定义、比较计划、启动账本计划与模型/配置身份。

零环境、零启动：模型身份通过 ROUND24 冻结负载只读加载 checkpoint 计算（不构造 driver），
配置身份逐字段继承并与 ROUND25 该场景行按 path-independent 摘要核对。产出 ROUND27_FREEZE.json
与 ROUND27_PROTOCOL.md，在首次环境启动前随执行依据一并提交。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import r27_common as common


def _model_identity() -> dict[str, Any]:
    import r27_run

    controller, identity = r27_run.load_candidate_controller()
    parameters = controller.parameter_summary()
    return {"identity": identity, "parameters": parameters,
            "checkpoint_sha256": common.sha256_file(common.LEARNED_CHECKPOINT)}


def _config_identity() -> dict[str, Any]:
    fields, substituted = common.stage_b_inherited_config(common.SCENE)
    import r27_run

    digest = r27_run._behaviour_config_sha256(fields)
    round25_row = common.read_json(common.ROUND25_DIR / "rows"
                                   / "r25_learned_s33_greedy_s1400031_a1.json")
    return {
        "behaviour_config_sha256": digest,
        "round25_learned_behaviour_config_sha256":
            round25_row["behaviour_config_sha256"],
        "matches_round25": digest == round25_row["behaviour_config_sha256"],
        "substituted_fields": substituted,
        "behaviour_fields": {key: value for key, value in sorted(fields.items())
                             if key not in {"seed", "output", "comm_transcript", "info_probe",
                                            "search_probe", "decision_trace", "record_stream",
                                            "replay_stream", "comm_policy", "comm_policy_params"}},
    }


def freeze_payload() -> dict[str, Any]:
    model = _model_identity()
    config = _config_identity()
    return {
        "round": "27",
        "run_id": "comm_cost_firstreport_intervention_20260929",
        "execution_basis": {
            "file": "runs/comm_cost_firstreport_intervention_20260929/EXECUTION_BASIS.md",
            "sha256": "eeb982dee928aa5c605a724404eb61c0758806b5cad9faa5b4dc66110f2ff259",
        },
        "parent_head": "e22db99ee1e1db9b536f4d8f9161b6538adf7ea8",
        "scene": common.SCENE,
        "scenario_seminar": "post-hoc selected failing scene from the ROUND26 diagnosis; a "
                            "privileged diagnostic scenario, not a random sample",
        "paths": {name: {"kind": spec["kind"],
                         "interceptor": bool(spec.get("interceptor", False)),
                         "override_enabled": bool(spec.get("override_enabled", False)),
                         "override_targets": list(spec.get("override_targets") or ()),
                         "role": spec["role"]}
                  for name, spec in common.PATHS.items()},
        "path_order": list(common.PATH_ORDER),
        "model": model,
        "config": config,
        "launch_budget": {
            "normal_launches": common.NORMAL_LAUNCHES,
            "reserve_launches": common.RESERVE_LAUNCHES,
            "hard_cap": common.LAUNCH_CAP,
            "counting_rule": "every failure, retry and smoke run counts; no throughput probe or "
                             "confirmation run beyond the five paths",
            "training": 0, "optimizer_steps": 0, "threshold_adjustments": 0,
        },
        "comparison_plan": {
            "comparisons": ["E0-L", "E5-L", "E05-L"],
            "reference_for_task_and_cost": "L (the original learned path of this scene)",
            "per_comparison": [
                "task change of each named target (on-time / expired / window-end status)",
                "whole-scene task change (on-time count, goal identity gained/lost)",
                "other targets damaged (goal identity)",
                "cost change (measured payload-attempt bytes, refusals listed separately)",
                "allocation continuity (assignment chain gaps for the named targets)",
            ],
            "interaction_note": "whether the combined intervention shows an interaction is "
                                "observed, never generalised from one scene",
            "no_significance_testing": "no single-scene significance, no bootstrap, no "
                                       "non-inferiority claim",
        },
        "intervention_definition": {
            "advance_rule": "at the first decision opportunity after discovery where the "
                            "candidate is legal and payable in the real sequential budget walk, "
                            "the original greedy HOLD of a named target is overridden to SEND",
            "already_sending": "if the original action is already SEND there is no second send; "
                               "the no-action-difference is recorded",
            "one_per_target": "at most one override per target; a lost packet adds no retransmission",
            "restore_after_first_emission": "after the first emission the original learned logic "
                                            "is fully restored; other targets are never "
                                            "overridden; later actions are recomputed in the real "
                                            "closed-loop state",
            "real_path_only": "the override lives inside the real sequential budget decision "
                              "path, uses the actual balance and quote, keeps the candidate "
                              "order, never appends an emission after the collector decided, "
                              "never sends free, reserves extra budget or bypasses the mask",
            "not_deployable": "the target IDs come from a failing case; this is a privileged "
                              "diagnostic intervention, not a general deployable rule",
        },
        "rng_registration": {
            "action_rng": "torch.manual_seed(eval_action_seed(33, 1400031, 'greedy')) per path, "
                          "identical label and formula to ROUND25",
            "environment_and_channel_rng": "derived inside the driver from the scenario seed; "
                                           "the override draws nothing, but an extra emission "
                                           "consumes channel loss draws, so post-intervention "
                                           "paths legitimately diverge",
            "forbidden_claims": "not the same per-message loss batch; outcome changes are not "
                                "attributable to one accepted message",
        },
        "stopping_rules": [
            "stop after the five paths complete, or on a technical-health/quota stop",
            "an unexplained behavioural drift between L and L0 stops the round before E0/E5/E05",
            "a behavioural-semantics defect stops the round; no mixed implementations in one panel",
            "no continued training, candidate swap, confirmation or deployment",
        ],
        "start_ledger_plan": {
            "cells": [{"path": name, "attempt": 1, "kind": "normal"}
                      for name in common.PATH_ORDER],
            "reserve": "3 launches reserved for technical failures and retries of a failed cell "
                       "only; reserve consumption is booked like any launch",
        },
        "publication": {"remote_verification": "NOT_PERFORMED",
                        "publication_status": "PENDING_NETWORK",
                        "network_retries_this_session": 0},
    }


def write_freeze() -> tuple[Path, Path]:
    payload = freeze_payload()
    if not payload["config"]["matches_round25"]:
        raise RuntimeError("the inherited configuration digest does not match the ROUND25 row; "
                           "refusing to launch on a drifted configuration")
    path = common.write_json("ROUND27_FREEZE.json", payload)
    protocol = Path(__file__).resolve().parent / "ROUND27_PROTOCOL.md"
    protocol.write_text(_protocol_markdown(payload), encoding="utf-8")
    return path, protocol


def _protocol_markdown(payload: dict[str, Any]) -> str:
    model = payload["model"]["identity"]
    lines = [
        "# ROUND27 协议：固定失败场景的首报提前干预", "",
        "执行依据：`EXECUTION_BASIS.md`（sha256 "
        f"`{payload['execution_basis']['sha256'][:16]}…`），起点 `e22db99`。", "",
        "## 实验性质", "",
        "事后选定失败案例（场景 1400031）的机制实验：在保持已冻结 learned 策略其余决策逻辑不变的",
        "前提下，检验提前两个目标的首报是否改变接收信息、分配连续性及任务结局。不是泛化评估、",
        "规则筛选、学习收益证明或部署策略。", "",
        "## 固定不变项", "",
        f"- 场景：`{payload['scene']}`（唯一场景，来自 ROUND26 三臂诊断的失败案例）。",
        f"- 模型：seed {model['training_seed']} ep{model['episodes_done']} 里程碑，"
        f"checkpoint `{model['checkpoint']}`，sha256 `{model['checkpoint_sha256'][:16]}…`。",
        "- 解码：原 greedy（SEND iff p(SEND) ≥ p(HOLD)，平手归 SEND），不混 sampled。",
        "- 搜索、分配、预算、信道、旁路、ACK、场景、horizon 均不变；配置摘要 "
        f"`{payload['config']['behaviour_config_sha256'][:16]}…` 与 ROUND25 一致。",
        "- 0 训练、0 optimizer.step、0 阈值调整。", "",
        "## 五条路径", "",
        "| 路径 | 干预器 | 覆盖 | 目标 |", "| --- | --- | --- | --- |",
    ]
    for name in payload["path_order"]:
        spec = payload["paths"][name]
        lines.append(
            f"| {name} | {'安装' if spec['interceptor'] else '不安装'} | "
            f"{'启用' if spec['override_enabled'] else '禁用'} | {', '.join(spec['override_targets']) or '—'} |")
    lines += [
        "", f"启动预算：{payload['launch_budget']['normal_launches']} 正常 + "
        f"{payload['launch_budget']['reserve_launches']} 技术储备 = "
        f"{payload['launch_budget']['hard_cap']} 硬上限；失败、重试、smoke 全部计入。", "",
        "## 干预定义", "",
        "- 在发现后第一个真实合法、可付的首报决策机会，把原 greedy 的 HOLD 覆盖为 SEND。",
        "- 覆盖发生在真实顺序预算决策路径内：同一 shadow gate、同一报价、同一候选顺序；"
        "不免费、不预留、不绕 mask；不追加发射。",
        "- 若原动作已是 SEND：不加第二次发送，记录无动作差异。",
        "- 每个目标最多一次覆盖；丢包不新增重传；首次发射后完全恢复原逻辑。",
        "- 目标 ID 来自失败案例：特权诊断干预，不是可部署通用规则。", "",
        "## 门禁", "",
        "1. 零环境合成测试六项（只命中指定目标 / 禁用即委托 / 已发送不重复 / 顺序预算正确 / "
        "无合法机会不绕硬门 / 新日志不改动作不耗环境信道 RNG）。",
        "2. 先跑 L、L0（计入 5 次正常启动）：动作、发射顺序、成本、任务结局、关键行为记录一致，"
        "并与 ROUND25 该场景行按共同字段核对；漂移则停止。",
        "3. 门禁通过后执行 E0、E5、E05。", "",
        "## 判读纪律", "",
        "- 成功救回目标 ⇒ 支持在该选定场景、该次随机实现中干预能改变结局；不证明早发必然有效。",
        "- 首报 accepted 仍失败 ⇒ 结合分配链报告为何不足；不自动归咎通信或分配器。",
        "- 丢包 / 拒绝 / 无有效动作差异 ⇒ 该路径不能作为“提前发送无效”的证据。",
        "- 不计算单场景显著性；不新增 seed、信道重复或规则搜索。",
        "- 不同发射改变后续随机数消耗：不宣称逐消息同一批丢包，不把结局变化全部归因一次 accepted。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    freeze_path, protocol_path = write_freeze()
    payload = common.read_json(freeze_path)
    print("freeze:", freeze_path.name)
    print("model checkpoint sha256:", payload["model"]["identity"]["checkpoint_sha256"])
    print("config digest matches ROUND25:", payload["config"]["matches_round25"])
    print("protocol:", protocol_path.name)
