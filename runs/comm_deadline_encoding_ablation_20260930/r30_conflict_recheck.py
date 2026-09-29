"""ROUND30 §3.1：开发块任务数冲突的机读身份表与裁定（0 环境启动）。

冲突：ROUND24 判定称 seed33 最终 greedy 在 16 个开发场景全部按时；ROUND29 的场景归属
更正文件称 seed33 greedy 在同一开发块有 5/16 场景失败。两侧都不预设正确，全部从原始行
与 checkpoint 重建身份表后裁定。

身份表字段（执行依据 §3.1 要求）：实验轮次及 v1/v2、训练 seed、checkpoint 里程碑、
greedy/sampled 模式、checkpoint SHA256、场景 seed、配置身份、源文件路径、按时任务数。
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
R24 = ROOT / "runs/comm_cost_v2_rerun_20260928"
DEV = tuple(range(1301000, 1301016))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def controller_param_sha(payload: dict[str, Any]) -> str:
    c = payload["controller"]
    tensors = {}
    for prefix, key in (("trunk", "trunk"), ("action_head", "action_head"),
                        ("value_task_head", "value_task_head"),
                        ("value_cost_head", "value_cost_head")):
        for name, tensor in c[key].items():
            tensors[f"{prefix}.{name}"] = tensor.detach().cpu()
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        digest.update(tensors[name].contiguous().numpy().tobytes())
    return digest.hexdigest()


def milestone_identity(seed: int) -> dict[str, Any]:
    import torch

    path = R24 / "checkpoints" / f"v2_train_s{seed}_ep128.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["training_state"]
    return {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": sha256_file(path),
        "controller_parameter_sha256": controller_param_sha(payload),
        "episodes_done": state["episodes_done"],
        "updates": state["updates"],
        "optimizer_steps": state["optimizer_steps"],
        "lambda": state["lambda"],
        "trainer_tag": payload["trainer_tag"],
        "loss_version": payload["loss_version"],
        "freeze_sha256": payload["freeze"]["freeze_sha256"],
    }


def main() -> int:
    # 1) 每 (seed, scene, block) 的行身份
    identity_rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(R24 / "rows/v2_eval_s*_s13*.json"))):
        row = json.loads(Path(path).read_text(encoding="utf-8"))
        if row.get("timepoint") != "final" or row["scenario_seed"] not in DEV:
            continue
        identity_rows.append({
            "experiment_round": "ROUND24 v2 rerun (spec version v2; v1 = ROUND22/Stage C, a "
                                "different trainer contract and not comparable)",
            "training_seed": row["training_seed"],
            "checkpoint_milestone": f"ep{128} (rolling replaced it; the row names timepoint=final)",
            "mode": "greedy" if row["eval_block"] == "final_greedy" else "sampled",
            "checkpoint_sha256": None,   # filled per seed below
            "scenario_seed": row["scenario_seed"],
            "config_identity": {"config_hash": row["config_hash"],
                                "behaviour_sha256": row["behaviour_sha256"],
                                "transport_budget_bytes": row["transport_parameters"]["budget"][
                                    "bytes_per_episode"]},
            "source_file": str(Path(path).relative_to(ROOT)),
            "on_time": row["on_time"],
            "expired": row["expired"],
            "spent_bytes": row["spent_bytes"],
            "emissions": row["emissions"],
            "controller_parameter_sha256_in_row": row["controller_parameters_after"][
                "parameter_sha256"],
        })
    milestones = {seed: milestone_identity(seed) for seed in (11, 22, 33)}
    for entry in identity_rows:
        entry["checkpoint_sha256"] = milestones[entry["training_seed"]]["file_sha256"]
        entry["controller_matches_milestone"] = (
            entry["controller_parameter_sha256_in_row"]
            == milestones[entry["training_seed"]]["controller_parameter_sha256"])
    # 2) immediate reference 行（配对基准）
    references = {}
    for path in sorted(glob.glob(str(R24 / "rows/v2_reference_s13*.json"))):
        row = json.loads(Path(path).read_text(encoding="utf-8"))
        if row["scenario_seed"] in DEV:
            references[row["scenario_seed"]] = {
                "on_time": row["on_time"], "spent_bytes": row["spent_bytes"],
                "source_file": str(Path(path).relative_to(ROOT))}
    # 3) ROUND29 曾打印的值（逐值指纹）与三个 seed 的 final_greedy 行比对
    printed_by_scene = {  # ROUND29 会话为"seed33 final greedy"打印的 on_time/spent/emissions
        1301000: (6, 28188, 11), 1301001: (5, 29208, 11), 1301002: (2, 28104, 11),
        1301003: (5, 29214, 11), 1301004: (6, 27312, 10), 1301005: (6, 27180, 11),
        1301006: (6, 27672, 11), 1301007: (6, 28836, 11), 1301008: (6, 27666, 10),
        1301009: (6, 28908, 11), 1301010: (6, 28896, 12), 1301011: (6, 27894, 12),
        1301012: (6, 28368, 11), 1301013: (5, 29196, 12), 1301014: (6, 27240, 11),
        1301015: (5, 27444, 11),
    }
    greedy_by_seed: dict[int, dict[int, Any]] = {11: {}, 22: {}, 33: {}}
    for entry in identity_rows:
        if entry["mode"] != "greedy":
            continue
        greedy_by_seed[entry["training_seed"]][entry["scenario_seed"]] = entry
    fingerprint: dict[str, Any] = {}
    for seed in (11, 22, 33):
        matches = 0
        for scene, printed in printed_by_scene.items():
            entry = greedy_by_seed[seed].get(scene)
            if entry and (entry["on_time"], entry["spent_bytes"],
                          entry["emissions"]) == printed:
                matches += 1
        fingerprint[f"seed{seed}_final_greedy"] = {
            "scenes_matching_printed_values": matches, "scenes": len(printed_by_scene)}
    # 4) 裁定
    s33 = greedy_by_seed[33]
    s33_failures = sorted(scene for scene, entry in s33.items() if entry["on_time"] < 6)
    payload = {
        "round": "30",
        "section": "zero-launch resolution of the development-block task-count conflict",
        "conflict": {
            "statement_a": "ROUND24 verdict/COMPARISON.json: seed33 final greedy has zero on-time "
                           "degradation on the 16 development scenarios (mean_on_time_delta 0.0, "
                           "degraded 0, improved 16)",
            "statement_b": "ROUND29 SCENE_BLOCK_CORRECTION.json (from ROUND29's audit): seed33 "
                           "greedy had 5/16 development scenes failing (1301001/1301002/1301003/"
                           "1301013/1301015)",
        },
        "identity_table_rows": len(identity_rows),
        "identity_table": identity_rows,
        "immediate_reference_rows_dev": references,
        "milestones": milestones,
        "value_fingerprint_against_round29_printed_table": fingerprint,
        "resolution": {
            "root_cause": "file selection in ROUND29's audit read: the values ROUND29 presented "
                          "under the seed33 label match the seed-11 final_greedy rows scene by "
                          "scene and value by value (on_time/spent/emissions, 16/16); no "
                          "milestone, mode, version or config difference is involved — the same "
                          "v2 trainer contract, the same ep128 milestone convention, the same "
                          "greedy decoder, rows whose controller parameter hash equals their own "
                          "seed's ep128 checkpoint",
            "round24_verdict_status": "stands: seed33 final greedy is 6/6 on-time on all 16 dev "
                                      "scenes; ROUND24's zero-degradation claim is what the raw "
                                      "rows say",
            "round29_audit_status": "its dev-block task numbers (5/16 failures) were seed-11 rows "
                                    "and are corrected here; the block-ownership core of the "
                                    "ROUND29 correction (1301000..1301015 is the ROUND22/24 "
                                    "development block, 1400031 lives in 1400000..1400063) is "
                                    "unaffected and remains registered",
            "dev_block_failures_under_seed33_final_greedy": s33_failures,
            "not_used_for": "no simulation was run to settle this historical question; the rows "
                            "and checkpoints already on disk were the only evidence",
        },
        "old_reports_preserved": [
            "runs/comm_cost_v2_rerun_20260928/ROUND24_VERDICT.json",
            "runs/comm_deadline_encoding_audit_20260929/SCENE_BLOCK_CORRECTION.json",
        ],
    }
    out = Path(__file__).resolve().parent / "CONFLICT_RECHECK_3_1.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "fingerprint": fingerprint,
        "dev_block_failures_seed33_final_greedy": s33_failures,
        "identity_rows": len(identity_rows),
        "all_controller_matches_milestone": all(e["controller_matches_milestone"]
                                                for e in identity_rows),
    }, ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
