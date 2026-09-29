"""ROUND32 交付登记：checkpoint 索引、更新前 checkpoint 索引、账本刷新、artefact manifest。

0 环境启动。原始 logs/rows/checkpoints 按既有规则留在 Git 外，这里只登记路径、大小与
sha256，使任何一个后续会话都能核验"读的就是当时写的"。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

import r32_common as common
import r32_trainer as trainer
from r32_train import batch_record_path, milestone_path, pre_update_path, rolling_path


def parameter_sha(payload: dict[str, Any]) -> str:
    """Same digest recipe as StageBController.parameter_summary, recomputed independently."""
    import hashlib

    tensors: dict[str, Any] = {}
    for prefix in ("trunk", "action_head", "value_task_head", "value_cost_head"):
        for name, tensor in payload["controller"][prefix].items():
            tensors[f"{prefix}.{name}"] = tensor
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        digest.update(tensors[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _entry(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(common.ROOT)), "size_bytes": path.stat().st_size,
            "sha256": common.sha256_file(path)}


def checkpoint_index() -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for reduction in common.ARM_ORDER:
        for seed in common.TRAIN_SEEDS:
            key = f"{reduction}_s{seed}"
            rolling = rolling_path(reduction, int(seed))
            payload = torch.load(rolling, map_location="cpu", weights_only=False)
            state = payload["training_state"]
            milestones = {}
            for episodes in common.MILESTONE_EPISODES:
                candidate = milestone_path(reduction, int(seed), int(episodes))
                if candidate.exists():
                    milestones[str(episodes)] = {**_entry(candidate),
                                                  "parameter_sha256": parameter_sha(
                                                      torch.load(candidate, map_location="cpu",
                                                                 weights_only=False))}
            pre_updates = {}
            for batch in common.PRE_UPDATE_BATCHES:
                candidate = pre_update_path(reduction, int(seed), int(batch))
                if candidate.exists():
                    pre_payload = torch.load(candidate, map_location="cpu", weights_only=False)
                    pre_updates[str(batch)] = {
                        **_entry(candidate),
                        "parameter_sha256": parameter_sha(pre_payload),
                        "episodes_done": int(pre_payload["training_state"]["episodes_done"]),
                        "updates": int(pre_payload["training_state"]["updates"]),
                        "lambda": float(pre_payload["training_state"]["lambda"]),
                        "scenarios_of_batch": common.read_json(
                            batch_record_path(reduction, int(seed), int(batch)))["scenarios"],
                        "reduction_mode": pre_payload["reduction_mode"],
                        "has_rng": bool({"python", "numpy", "torch"}
                                        <= set(pre_payload["rng"])),
                        "batch_record_parameters_before_matches": parameter_sha(
                            pre_payload) == common.read_json(batch_record_path(
                                reduction, int(seed), int(batch)))
                        ["controller_parameters_before"]["parameter_sha256"]}
            entries[key] = {
                "reduction": reduction, "training_seed": int(seed),
                "rolling_path": str(rolling.relative_to(common.ROOT)),
                "rolling_sha256": common.sha256_file(rolling),
                "milestones": milestones, "pre_update_checkpoints": pre_updates,
                "episodes_done": int(state["episodes_done"]), "updates": int(state["updates"]),
                "optimizer_steps": int(state["optimizer_steps"]),
                "lambda": float(state["lambda"]),
                "trainer_tag": payload["trainer_tag"], "loss_version": payload["loss_version"],
                "reduction_mode": payload.get("reduction_mode"),
                "freeze_sha256": payload["freeze"]["freeze_sha256"],
                "feature_layout": payload["freeze"]["feature_layout"],
                "encoder_arm": payload["freeze"]["encoder_arm"],
                "batch_records": sorted(str(path.relative_to(common.ROOT))
                                        for path in common.CHECKPOINTS.glob(
                                            f"r32_batch_{reduction}_s{seed}_b*.json")),
            }
    return entries


def artefact_manifest() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for directory in (common.LOGS, common.ROWS, common.CHECKPOINTS):
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files.append(_entry(path))
    by_kind: dict[str, int] = {}
    for item in files:
        name = Path(item["path"]).name
        kind = ("tape" if name.endswith("_tape.jsonl") else
                "lifecycle" if name.endswith("_rp.jsonl") else
                "transcript" if name.endswith("_ct.jsonl") else
                "row" if name.endswith(".json") and "/rows/" in item["path"] else
                "batch_record" if "/checkpoints/r32_batch_" in item["path"] else
                "checkpoint" if item["path"].endswith(".pt") else "other")
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return {"files": len(files), "by_kind": dict(sorted(by_kind.items())),
            "total_bytes": sum(int(item["size_bytes"]) for item in files),
            "note": "raw logs, tapes, rows and weights stay outside git exactly as in rounds "
                    "24/28/29/30/31; this manifest is the path+size+sha256 register for them",
            "items": files}


def main() -> int:
    from r32_blocks import refresh_ledger
    ledger = refresh_ledger()
    index = checkpoint_index()
    (common.OUTPUT_DIR / "CHECKPOINT_INDEX.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    manifest = artefact_manifest()
    (common.OUTPUT_DIR / "ARTEFACT_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"ledger": ledger, "artefacts": {key: manifest[key] for key in
                                                     ("files", "by_kind", "total_bytes")},
                      "pre_update_coverage": {
                          key: sorted(value["pre_update_checkpoints"])
                          for key, value in index.items()},
                      "pre_update_identity_all_match": all(
                          entry["pre_update_checkpoints"][batch][
                              "batch_record_parameters_before_matches"]
                          for entry in index.values() for batch in entry["pre_update_checkpoints"])},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
