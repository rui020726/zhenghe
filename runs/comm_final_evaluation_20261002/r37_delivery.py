"""ROUND37 交付登记（0 环境启动）：模型/配置/版本清单 + Git 外原件逐件哈希 + 代码身份。

产出两份：

- `delivery_draft/MODEL_AND_CONFIG_INVENTORY.md`：清单文档，全部数字直接从冻结件与磁盘重算，
  不从报告正文抄；
- `delivery_draft/ARTEFACT_HASH_REGISTRY.json`：Git 内交付件清单 + 每个原件（rows / tapes /
  lifecycles / transcripts / claims / shard 摘要 / 账本）的相对路径、字节数与 SHA256。

    /d/project_env/python.exe r37_delivery.py
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import r37_common as r37

RAW_KINDS = (("rows", "*.json"), ("logs", "*.jsonl"), ("claims", "*.claim"),
             ("shards", "*.json"), ("checkpoints", "*.pt"))


def raw_inventory() -> dict[str, Any]:
    files: dict[str, str] = {}
    sizes: dict[str, int] = {}
    counts: dict[str, int] = {}
    for directory, pattern in RAW_KINDS:
        paths = sorted((r37.HERE / directory).glob(pattern)) if (r37.HERE / directory).exists() \
            else []
        counts[f"{directory}/{pattern}"] = len(paths)
        for path in paths:
            key = f"{directory}/{path.name}"
            files[key] = r37.sha256_file(path)
            sizes[key] = path.stat().st_size
    ledgers = {}
    for path in sorted((r37.HERE / "logs").glob("START_LEDGER_p*.json")):
        payload = r37.read_json(path)
        ledgers[path.name] = {"starts": int(payload.get("starts") or 0),
                             "results": len(payload.get("results") or []),
                             "sha256": r37.sha256_file(path)}
    return {"counts": counts, "files_with_sha256": files, "bytes": sizes,
            "total_bytes": sum(sizes.values()), "episode_start_ledgers": ledgers}


def inventory_markdown(plan: dict[str, Any], gate: dict[str, Any], stats: dict[str, Any],
                       reconcile: dict[str, Any], raw: dict[str, Any]) -> str:
    models = plan["models"]
    lines = ["# 模型、配置与版本清单",
             "",
             "本文件由 `r37_delivery.py` 直接从冻结件与磁盘重算生成，不从报告正文抄数字。",
             "",
             "## 1. 被评估的三个模型（全部评估、全部报告）",
             "",
             "| 训练 seed | checkpoint（Git 外） | 字节 SHA256 | whole-controller 参数摘要 | "
             "episodes / 更新批 | λ 终值 | 版本字段 |",
             "|---|---|---|---|---|---|---|"]
    for seed in r37.TRAINING_SEEDS:
        item = models[f"s{seed}"]
        lines.append("| {} | `{}` | `{}` | `{}` | {} / {} | {} | 两处齐全 |".format(
            seed, item["path"], item["bytes_sha256"][:16] + "…",
            item["parameter_sha256"][:16] + "…", item["episodes_done"], item["updates"],
            f"{item['lambda_final']:.4f}"))
    lines += ["",
              f"- 装载方式：`load_checkpoint_read_only`，`load_mode={models['s11']['load_mode']}、"
              f"optimizer_restored={models['s11']['optimizer_restored']}、"
              f"rng_restored={models['s11']['rng_restored']}`（不建 optimizer、不动 RNG）",
              f"- 采集语义标签：`{models['s11']['timing_interpretation']}`（缺版本字段的旧件会标 "
              "`legacy_unversioned_capture_semantics` 并禁止续训）",
              "- 中间里程碑 ep64 与 rolling 文件**未参与评估**，只作审计存在",
              "",
              "## 2. 版本与代码身份",
              "",
              "| 项 | 值 |",
              "|---|---|"]
    version_fields = {
        "training_version": r37.TRAINING_VERSION,
        "reward_timing_version": r37.REWARD_TIMING_VERSION,
        "trainer_tag": r37.v2_trainer.TRAINER_TAG,
        "loss_version": r37.v2_trainer.LOSS_VERSION,
        "state_schema": r37.v2_trainer.STATE_SCHEMA,
        "gae_schema": r37.v2_trainer.GAE_SCHEMA,
        "optimizer_schema": r37.v2_trainer.OPTIMIZER_SCHEMA,
        "reward_timing_schema": r37.v2_trainer.REWARD_TIMING_SCHEMA}
    for key, value in version_fields.items():
        lines.append(f"| {key} | `{value}` |")
    code = plan["configuration"]
    lines += [f"| 评估分支 / 头 | `{plan['branch']}` @ `{plan['head']}` |",
              f"| 生产件（integrated_sar） | {len(code['production_files'])} 个哈希登记 |",
              f"| Stage C harness | {len(code['stage_c_harness'])} 个哈希登记 |",
              "",
              "## 3. 搜索 / 分配模型哈希（冻结，不改）",
              ""]
    for path, digest in sorted(code["frozen_search_and_allocation_weights"].items()):
        lines.append(f"- `{path}` — `{digest}`")
    lines += ["", "## 4. 运行环境与完整配置快照", "",
              "| 字段 | 值 |", "|---|---|"]
    identity = r37.read_json(r37.HERE / "EVALUATION_IDENTITY.json")["environment"]
    for key in ("interpreter", "python", "torch", "numpy", "steps_per_episode", "budget_bytes",
                "b0_bytes", "delta", "delivery_probability", "ack_enabled", "replay_or_online"):
        lines.append(f"| {key} | `{identity[key]}` |")
    lines += ["", f"- 继承的世界字段共 {len(identity['inherited_world_fields'])} 项，全部登记在 "
              "`EVALUATION_IDENTITY.json`",
              f"- 传输覆盖字段：`{json.dumps(identity['transport_overrides'], ensure_ascii=False)}`",
              "",
              "## 5. 场景块与九臂",
              "",
              f"- 场景：{plan['scenarios'][0]}..{plan['scenarios'][-1]}（{len(plan['scenarios'])} "
              f"个），占用依据 `SCENE_USAGE_AUDIT.json`（不看结局选块）",
              f"- 臂：`{', '.join(item['id'] for item in plan['arms'])}`",
              f"- 额度：正常 {plan['quota']['normal_evaluation_launches']} / 录制储备 "
              f"{plan['quota']['recording_reserve']}（判定不需要）/ 失败储备 "
              f"{plan['quota']['failure_reserve']} / 硬上限 {plan['quota']['hard_cap']}",
              "",
              "## 6. 产物位置（Git 外，逐件哈希在本轮登记表内）",
              ""]
    for key, value in sorted(raw["counts"].items()):
        lines.append(f"- `{key}`：{value} 件")
    lines += [f"- 合计 {len(raw['files_with_sha256'])} 件、{raw['total_bytes']:,} 字节",
              "",
              "## 7. 门禁与对账状态",
              "",
              f"- 技术门禁：`all_passed={gate['all_passed']}`，"
              f"核对项 {len(gate['checks'])} 项，计入正常评估 {gate['launches_consumed_by_the_gate']} 次",
              f"- 账本四方闭合：`all_passed={reconcile['all_passed']}`，"
              f"行 {reconcile['rows']['written']} / claim {reconcile['claims']['held']} / "
              f"账本完成 {reconcile['this_round_book']['completed']} / "
              f"真实 ledger {reconcile['real_stage_c_ledger']['total']}",
              f"- 技术失败 {reconcile['this_round_book']['technical_failures']} 次、"
              f"复用行 {reconcile['this_round_book']['reused_rows']} 次、"
              f"独立流录制 {reconcile['this_round_book']['recording_launches']} 次",
              f"- 统计矩阵完整性：{stats['matrix_complete']}（cells={stats['cells']}）"]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    r37.configure()
    plan = r37.read_json(r37.HERE / "FROZEN_EVALUATION_PLAN.json")
    gate = r37.read_json(r37.HERE / "ROUND37_GATE.json")
    stats = r37.read_json(r37.HERE / "ROUND37_STATISTICS.json")
    reconcile = r37.read_json(r37.HERE / "EVALUATION_LEDGER_RECONCILIATION.json")
    raw = raw_inventory()
    tracked = [line for line in r37.git("ls-files", "--",
                                        "runs/comm_final_evaluation_20261002").splitlines() if line]
    registry = {
        "kind": "round37-artefact-hash-registry", "schema": "round37-delivery-registry/1",
        "round": "37", "registered_at": time.strftime("%F %T"),
        "branch": r37.git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "head_at_registration": r37.git("rev-parse", "HEAD").strip(),
        "start_head_from_round36": plan["head"],
        "git_tracked_deliverables": tracked,
        "raw_artefacts_kept_outside_git": raw,
        "frozen_plan_identity": {"sha256": r37.sha256_file(r37.HERE / "FROZEN_EVALUATION_PLAN.json"),
                                 "models": {key: value["bytes_sha256"] for key, value in
                                            plan["models"].items()},
                                 "scenarios": [plan["scenarios"][0], plan["scenarios"][-1]],
                                 "statistics": plan["statistics"]},
        "model_byte_hashes_verified_against_round36": all(
            value["matches_round36_registry"] for value in plan["models"].values()),
        "training_or_optimizer_performed": False, "optimizer_steps": 0,
        "old_round_artefacts_rewritten": 0,
        "note": "Git 外的原件全部在此逐件登记路径与 SHA256，审阅者无需取回原件即可核对结论所依据的"
                "是同一批数据；本轮不写入任何旧轮目录"}
    (r37.HERE / "delivery_draft" / "ARTEFACT_HASH_REGISTRY.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    text = inventory_markdown(plan, gate, stats, reconcile, raw)
    (r37.HERE / "delivery_draft" / "MODEL_AND_CONFIG_INVENTORY.md").write_text(text,
                                                                              encoding="utf-8")
    print(json.dumps({"tracked": len(tracked), "raw_files": len(raw["files_with_sha256"]),
                      "raw_bytes": raw["total_bytes"],
                      "inventory_markdown": len(text)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
