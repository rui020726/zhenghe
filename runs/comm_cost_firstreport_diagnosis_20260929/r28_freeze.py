"""ROUND28 启动前冻结：场景登记、选择算法、运行计划、模型/配置身份与停止规则。

零环境、零启动：模型身份通过 ROUND24 冻结负载只读加载 checkpoint 计算（不构造 driver），
配置身份逐字段继承并与 ROUND27 该协议的配置摘要核对。产出 ROUND28_FREEZE.json 与
ROUND28_PROTOCOL.md，在首次环境启动前随执行依据一并提交。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import r28_common as common


def _model_identity() -> dict[str, Any]:
    import r28_run

    controller, identity = r28_run.load_candidate_controller()
    return {"identity": identity, "parameters": controller.parameter_summary(),
            "checkpoint_sha256": common.sha256_file(common.LEARNED_CHECKPOINT)}


def _config_identity() -> dict[str, Any]:
    import r28_run

    fields, substituted = common.stage_b_inherited_config(common.GATE_SCENARIO)
    digest = r28_run._behaviour_config_sha256(fields)
    round27_row = common.read_json(common.ROUND27_DIR / "rows" / "r27_L_s1400031_a1.json")
    return {
        "behaviour_config_sha256": digest,
        "round27_L_behaviour_config_sha256": round27_row["behaviour_config_sha256"],
        "matches_round27": digest == round27_row["behaviour_config_sha256"],
        "substituted_fields": substituted,
    }


def _scene_registration() -> dict[str, Any]:
    registry = common.read_json(common.ROUND24_DIR / "SCENE_REGISTRY.json")
    training = list(registry["training_scenarios"])
    requested = list(common.SCENARIOS)
    return {
        "requested_scenes": requested,
        "source_registry": "runs/comm_cost_v2_rerun_20260928/SCENE_REGISTRY.json",
        "registered_training_block": registry["training_block"],
        "registered_training_scenes_count": len(training),
        "requested_equals_first_16_of_registered_training_block": training[:16] == requested,
        "registered_by": registry["collision_audit"]["reuse_registration"]["registered_by"],
        "prior_user": registry["collision_audit"]["reuse_registration"]["prior_user"],
        "sealed_blocks_untouched": [item for item in
                                    registry["collision_audit"]["claimed_ranges"]
                                    if item["low"] in (800000, 900000)],
        "training_set_diagnosis_not_validation": True,
        "scene_selection_rule": "the fixed first 16 of the registered training block; no new "
                                "selection rule, no outcome-based scene swap",
    }


def freeze_payload() -> dict[str, Any]:
    model = _model_identity()
    config = _config_identity()
    scenes = _scene_registration()
    return {
        "round": "28",
        "run_id": common.OUTPUT_DIR.name,
        "execution_basis": {
            "file": "runs/comm_cost_firstreport_diagnosis_20260929/EXECUTION_BASIS.md",
            "sha256": "f91858127f2f9ab664f161351bf2b187be247e655e0f16033a3581c5ad5b6943",
        },
        "parent_head": "5982677c9370a9a73acf082000e6865bf0d86a72",
        "parent_branch": "exp/comm-cost-firstreport-intervention",
        "branch": "exp/comm-cost-firstreport-diagnosis",
        "scenes": scenes,
        "model": model,
        "config": config,
        "launch_budget": {
            "baselines": 16,
            "disabled_gate_paths": 1,
            "interventions_max": 16 * common.MAX_TARGETS_PER_SCENE,
            "normal_cap": common.NORMAL_LAUNCHES,
            "reserve_launches": common.RESERVE_LAUNCHES,
            "hard_cap": common.LAUNCH_CAP,
            "counting_rule": "every failure, retry and smoke run counts; fewer interventions are "
                             "fine, no top-up with other targets or scenes",
            "training": 0, "optimizer_steps": 0, "threshold_adjustments": 0,
            "extra_recording_or_throughput_probes": 0,
        },
        "selection_algorithm": {
            "rule": "per baseline target: the first tape step where the target is actionable "
                    "(sequential shadow gate: payable in transport order); an intervention point "
                    "exists only if the original greedy action there is HOLD",
            "exclusion_reasons": ["first_legal_opportunity_greedy_send",
                                  "no_legal_payable_opportunity", "never_discovered",
                                  "terminal_before_first_candidate"],
            "inputs_allowed": ["pre-action candidate set", "send history (requested flags)",
                               "effective sequential mask (actionable/masked)",
                               "original greedy action", "time identity (step, discovery time)"],
            "inputs_forbidden": ["completion", "expiry", "on-time", "spent bytes", "final cost",
                                 "any receiver-side truth"],
            "frozen_before_launch": True,
            "redistribution_allowed_after_effects_seen": False,
        },
        "intervention": {
            "mechanism": "ROUND27's FirstReportAdvancer, imported read-only; one HOLD->SEND "
                         "override per cell at the frozen (scene, target, step)",
            "same_scene_restart": "each intervention cell re-runs the whole episode from the "
                                  "scene's initial state; the prefix must reproduce the baseline "
                                  "until the intervention step",
            "not_allowed": ["free send", "forced accepted", "mask bypass", "extra budget",
                            "second send for the same target", "receiver-truth injection",
                            "target id as a new actor feature"],
        },
        "comparison_plan": {
            "unit": "per intervention cell against the same scene's baseline",
            "deltas": "intervention minus baseline (ΔY on-time count, ΔC measured bytes)",
            "per_sample": ["ΔY", "ΔC", "intervened target outcome", "other targets gained/lost",
                           "real emission", "accepted/superseded/drop", "allocation and work "
                           "continuity change", "intervention effectiveness", "prefix identity",
                           "actor-visible features and analysis metadata (kept separate)"],
            "labels_kept_distinct": ["emitted", "accepted", "allocation changed",
                                     "task outcome changed"],
            "descriptive_only": "counts and distributions; no significance test, no classifier, "
                                "no threshold fitting, no new policy",
            "independence_warning": "up to 96 interventions over only 16 scenes; interventions "
                                    "are not 96 independent scenes and per-scene gains are not "
                                    "added up to predict joint sends",
        },
        "gate": {
            "first_scene": "baseline(1300000) must equal its disabled-interceptor cell on every "
                           "behaviour field and tape line",
            "per_sample_prefix": "the intervention tape before the intervention step must be the "
                                 "baseline tape, and the frozen point's state must reproduce the "
                                 "baseline row at that step",
            "synthetic_only": "zero-environment, zero-optimizer synthetic checks; the full "
                              "production unittest suite is NOT re-run this round (its scope was "
                              "registered in ROUND27_SCOPE_REGISTRATION.json)",
        },
        "stopping_rules": [
            "stop after the frozen plan completes, or on a technical-health/quota stop",
            "an unexplained baseline/disabled drift stops the round",
            "a failed prefix reproduction is reported as a technical problem; no nearby-moment "
            "substitution",
            "no continued training, no candidate swap, no confirmation, no deployment",
        ],
        "publication": {"remote_verification": "NOT_PERFORMED",
                        "publication_status": "PENDING_NETWORK",
                        "network_retries_this_session": 0},
    }


def write_freeze() -> tuple[Path, Path]:
    payload = freeze_payload()
    if not payload["scenes"]["requested_equals_first_16_of_registered_training_block"]:
        raise RuntimeError("the 16 requested scenes are not the registered training block's first "
                           "16; refusing to launch")
    if not payload["config"]["matches_round27"]:
        raise RuntimeError("the inherited configuration digest does not match ROUND27's; "
                           "refusing to launch on a drifted configuration")
    path = common.write_json("ROUND28_FREEZE.json", payload)
    protocol = Path(__file__).resolve().parent / "ROUND28_PROTOCOL.md"
    protocol.write_text(_protocol_markdown(payload), encoding="utf-8")
    return path, protocol


def _protocol_markdown(payload: dict[str, Any]) -> str:
    model = payload["model"]["identity"]
    scenes = payload["scenes"]
    lines = [
        "# ROUND28 协议：训练场景上的首报反事实样本诊断", "",
        f"执行依据 `EXECUTION_BASIS.md`（sha256 `{payload['execution_basis']['sha256'][:16]}…`），"
        f"起点 `{payload['parent_head'][:7]}`。", "",
        "## 性质", "",
        "训练集机制诊断：把 ROUND27 的单案例发现扩展为一小批有明确来源的"
        "“发送端观测—首报动作—全局任务/成本变化”样本。不是独立验证集、不是训练、"
        "不是部署。", "",
        "## 固定不变项", "",
        f"- 场景：{scenes['requested_scenes'][0]}..{scenes['requested_scenes'][-1]}"
        f"（原登记训练块 {scenes['registered_training_block'][0]}.."
        f"{scenes['registered_training_block'][1]} 的前 16 个，"
        f"`{scenes['source_registry']}` 登记）。",
        f"- 模型：seed {model['training_seed']} ep{model['episodes_done']}，"
        f"checkpoint `{model['checkpoint']}`，sha256 `{model['checkpoint_sha256'][:16]}…`。",
        "- 解码：原 greedy（SEND iff p(SEND) ≥ p(HOLD)，平手归 SEND）。",
        f"- 研究配置摘要 `{payload['config']['behaviour_config_sha256'][:16]}…` 与 ROUND27 一致。",
        "- 搜索、分配权重、环境、信道、预算、旁路、ACK、候选顺序全部不变。",
        "- 不读接收端真值；目标 ID 只作记录和干预定位，不作新增特征。", "",
        "## 启动预算", "",
        f"16 baseline + 1 禁用覆盖门禁 + 每场景每目标最多 1 条干预 = 正常上限 "
        f"{payload['launch_budget']['normal_cap']}；技术储备 {payload['launch_budget']['reserve_launches']}；"
        f"硬上限 {payload['launch_budget']['hard_cap']}。0 训练、0 optimizer.step、0 阈值调整。", "",
        "## 干预点选择（冻结算法）", "",
        "- 对 baseline 中每个目标：找它**首次真实合法可付**的首报决策机会"
        "（tape 中该目标第一次进入 actionable：顺序 shadow gate 判可付）。",
        "- 只有该机会原 greedy 动作为 HOLD 才登记干预点。",
        "- 排除原因分别登记：首个合法机会已 SEND / 无合法可付机会 / 从未发现 / 首发前终态。",
        "- 选择器只读动作前候选、发送历史、有效顺序 mask、原 greedy 动作与时间身份；"
        "不读完成/过期/成本等结局；不许挑失败目标或“看起来最可能获益”的目标。",
        "- baseline 产生后先冻结清单，再跑干预路径；不允许根据已见效果改算法。", "",
        "## 门禁", "",
        "- 首场景 baseline 与禁用覆盖路径行为逐字段一致、tape 逐行相同。",
        "- 每条干预路径：干预步之前的 tape 与 baseline 逐字段相同；冻结点的状态"
        "（步号、特征、p(SEND)、报价、shadow 余额）必须在该场景 baseline 里逐字重现。",
        "- 合成测试只做零环境、零 optimizer 检查；本轮不重跑完整生产套件（其范围已在 "
        "`ROUND27_SCOPE_REGISTRATION.json` 单列）。", "",
        "## 判读纪律", "",
        "- 每样本差值 = 干预路径 − baseline；保留正、零、负结果。",
        "- 发射成功、消息被接受、改变分配、改变任务结局是四个不同标签。",
        "- 最多 96 个干预只有 16 个场景：不当作 96 个独立场景，不把单目标收益相加预测联合发送。",
        "- 新增发射改变后续随机数消耗：不宣称逐消息同批丢包；结果含闭环总效应。",
        "- 不拟合阈值/分类器/新策略，不做显著性筛选，不凭小样本宣布特征足够或不足。", "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    freeze_path, protocol_path = write_freeze()
    payload = common.read_json(freeze_path)
    print("freeze:", freeze_path.name, "| protocol:", protocol_path.name)
    print("model checkpoint sha256:", payload["model"]["identity"]["checkpoint_sha256"])
    print("config matches ROUND27:", payload["config"]["matches_round27"])
    print("scenes ok:", payload["scenes"]["requested_equals_first_16_of_registered_training_block"])
