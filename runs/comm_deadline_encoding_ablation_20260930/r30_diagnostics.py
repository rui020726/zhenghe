"""ROUND30 §9 训练诊断：只读既有落盘件（batch 记录 + 被动首报记录），0 环境启动。

输出 DIAGNOSTICS.json：
- 每臂每 seed 的逐批 task/cost/g_hat/λ、有效双动作样本数、masked 数、两个 critic 的损失与
  逐模块梯度读数（ppo_update 已登记的部分）；
- 首报决策点的原始剩余期限、实际编码列值、p(SEND) 与原动作（r30_train 的被动记录）；
- greedy 与 sampled 的差异、三个训练 seed 的一致性/不一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import r30_common as common


def load_batch_records(arm: str, seed: int) -> list[dict[str, Any]]:
    records = []
    for path in sorted(common.CHECKPOINTS.glob(f"r30_batch_{arm}_s{seed}_b*.json"),
                       key=lambda item: int(item.stem.rsplit("_b", 1)[1])):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def seed_summary(arm: str, seed: int) -> dict[str, Any]:
    records = load_batch_records(arm, seed)
    per_batch = []
    for record in records:
        update = record.get("update") or {}
        timing = None
        per_batch.append({
            "batch": record["batch"],
            "lambda_at_rollout": record["lambda_at_rollout"],
            "samples": int(update.get("samples", 0)),
            "optimizer_steps": int(update.get("optimizer_steps", 0)),
            "g_hat": (record.get("dual") or {}).get("g_hat"),
            "lambda_next": (record.get("dual") or {}).get("lambda_next"),
            "value_loss_task": update.get("value_loss_task"),
            "value_loss_cost": update.get("value_loss_cost"),
            "policy_surrogate": update.get("policy_surrogate"),
            "entropy": update.get("entropy"),
            "approx_kl": update.get("approx_kl"),
            "clip_fraction": update.get("clip_fraction"),
            "gradient_norm_mean": update.get("gradient_norm_mean"),
            "module_gradient_l1_mean": update.get("module_gradient_l1_mean"),
            "module_gradient_l1_min": update.get("module_gradient_l1_min"),
            "likelihood_max_abs_gap": (update.get("likelihood_alignment") or {}).get(
                "max_abs_gap"),
            "controller_parameters_before": (record.get("controller_parameters_before")
                                             or {}).get("parameter_sha256"),
            "controller_parameters_after": (record.get("controller_parameters_after")
                                            or {}).get("parameter_sha256"),
        })
    first_reports: list[dict[str, Any]] = []
    for record in records:
        for episode in record.get("first_report_passive_records") or []:
            for entry in episode["records"]:
                first_reports.append({
                    "episode_index": episode["episode_index"],
                    "scenario": episode["scenario"], **entry})
    encoded_values = [round(float(entry["encoded_deadline_column"]), 6)
                      for entry in first_reports]
    raw_values = [float(entry["raw_remaining_time"]) for entry in first_reports]
    return {
        "batches": len(records),
        "per_batch": per_batch,
        "first_report_points": len(first_reports),
        "first_report_raw_remaining": {
            "min": min(raw_values) if raw_values else None,
            "max": max(raw_values) if raw_values else None,
            "distinct": len(set(round(value, 6) for value in raw_values)),
        },
        "first_report_encoded_deadline_column": {
            "values": sorted(set(encoded_values)),
            "distinct": len(set(encoded_values)),
        },
        "first_report_send_probability": {
            "min": min((entry["send_probability"] for entry in first_reports), default=None),
            "max": max((entry["send_probability"] for entry in first_reports), default=None),
        },
        "first_report_original_actions": {
            action: sum(1 for entry in first_reports
                        if entry["original_action"] == action)
            for action in ("SEND", "HOLD")},
        "first_report_sample": first_reports[:5],
    }


def main() -> int:
    payload: dict[str, Any] = {"round": "30", "section": "training diagnostics (passive)",
                               "arms": {}}
    for arm in ("old", "new"):
        payload["arms"][arm] = {f"s{seed}": seed_summary(arm, seed)
                                for seed in common.TRAIN_SEEDS}
    consistency = {}
    for seed in common.TRAIN_SEEDS:
        old_summary = payload["arms"]["old"][f"s{seed}"]
        new_summary = payload["arms"]["new"][f"s{seed}"]
        consistency[f"s{seed}"] = {
            "same_batch_count": old_summary["batches"] == new_summary["batches"],
            "old_lambda_final": old_summary["per_batch"][-1]["lambda_next"]
                               if old_summary["per_batch"] else None,
            "new_lambda_final": new_summary["per_batch"][-1]["lambda_next"]
                               if new_summary["per_batch"] else None,
            "old_first_report_encoded_distinct":
                old_summary["first_report_encoded_deadline_column"]["distinct"],
            "new_first_report_encoded_distinct":
                new_summary["first_report_encoded_deadline_column"]["distinct"],
            "old_first_report_action_counts": old_summary["first_report_original_actions"],
            "new_first_report_action_counts": new_summary["first_report_original_actions"],
        }
    payload["seed_consistency"] = consistency
    payload["note"] = ("accepted share is not a task-benefit measure; no diagnostic classifier "
                       "and no threshold were fitted; all fields are passive records frozen "
                       "before the first launch")
    out = common.OUTPUT_DIR / "DIAGNOSTICS.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({arm: {f"s{seed}": {"batches": value["batches"],
                                         "encoded_distinct":
                                             value["first_report_encoded_deadline_column"][
                                                 "distinct"],
                                         "raw_distinct":
                                             value["first_report_raw_remaining"]["distinct"]}
                                for seed, value in payload["arms"][arm].items()}
                       for arm in payload["arms"]}, ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
