"""ROUND30 交付登记：checkpoint 索引与哈希、账本刷新、artefact manifest（0 环境启动）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

import r30_common as common


def checkpoint_index() -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for arm in ("old", "new"):
        for seed in common.TRAIN_SEEDS:
            state_path = common.CHECKPOINTS / f"r30_train_{arm}_s{seed}.pt"
            payload = torch.load(state_path, map_location="cpu", weights_only=False)
            state = payload["training_state"]
            milestones = {}
            for episodes in common.MILESTONE_EPISODES:
                candidate = common.CHECKPOINTS / f"r30_train_{arm}_s{seed}_ep{episodes}.pt"
                if candidate.exists():
                    milestones[str(episodes)] = {
                        "path": str(candidate.relative_to(common.ROOT)),
                        "sha256": common.sha256_file(candidate)}
            entries[f"{arm}_s{seed}"] = {
                "arm": arm, "training_seed": seed,
                "rolling_path": str(state_path.relative_to(common.ROOT)),
                "rolling_sha256": common.sha256_file(state_path),
                "milestones": milestones,
                "episodes_done": state["episodes_done"], "updates": state["updates"],
                "optimizer_steps": state["optimizer_steps"], "lambda": state["lambda"],
                "trainer_tag": payload["trainer_tag"], "loss_version": payload["loss_version"],
                "freeze_sha256": payload["freeze"]["freeze_sha256"],
                "freeze_arm": payload["freeze"]["arm"],
                "feature_layout": payload["freeze"]["feature_layout"],
                "batch_records": sorted(
                    str(path.relative_to(common.ROOT)) for path in common.CHECKPOINTS.glob(
                        f"r30_batch_{arm}_s{seed}_b*.json")),
            }
    return entries


def training_curves() -> dict[str, Any]:
    curves: dict[str, Any] = {}
    for arm in ("old", "new"):
        for seed in common.TRAIN_SEEDS:
            payload = torch.load(common.CHECKPOINTS / f"r30_train_{arm}_s{seed}.pt",
                                 map_location="cpu", weights_only=False)
            state = payload["training_state"]
            curves[f"{arm}_s{seed}"] = {
                "lambda_start": common.LAMBDA_INIT, "lambda_final": state["lambda"],
                "lambda_history": state["lambda_history"],
                "updates": state["updates"], "optimizer_steps": state["optimizer_steps"],
                "episodes_done": state["episodes_done"],
            }
    return curves


def artefact_manifest() -> dict[str, Any]:
    sections: dict[str, Any] = {}
    for name, base in (("LOGS", common.LOGS), ("ROWS", common.ROWS),
                       ("CHECKPOINTS", common.CHECKPOINTS), ("REJECTED", common.REJECTED)):
        entries: dict[str, dict[str, Any]] = {}
        if base.exists():
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    entries[path.relative_to(common.ROOT).as_posix()] = {
                        "bytes": path.stat().st_size, "sha256": common.sha256_file(path)}
        sections[name] = entries
    totals = {name: {"files": len(entries),
                     "bytes": sum(int(entry["bytes"]) for entry in entries.values())}
              for name, entries in sections.items()}
    return {"round": "30", "run_id": common.OUTPUT_DIR.name, "sections": sections,
            "totals": totals,
            "git_policy": "rows, logs, tapes, checkpoints and batch records stay out of version "
                          "control; this manifest is the external hash list that lets a remote "
                          "reviewer find and verify the originals",
            "read_only_sources": [
                "runs/comm_cost_v2_rerun_20260928 (ROUND24 execution artefacts, never written)",
                "runs/comm_deadline_encoding_audit_20260929 (ROUND29 audit artefacts, read only)",
            ]}


def main() -> int:
    import r30_blocks

    index = checkpoint_index()
    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (common.OUTPUT_DIR / "CHECKPOINT_INDEX.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    (common.OUTPUT_DIR / "TRAINING_CURVES.json").write_text(
        json.dumps({"round": "30", "curves": training_curves()}, ensure_ascii=False, indent=1,
                   sort_keys=True) + "\n", encoding="utf-8")
    manifest = artefact_manifest()
    (common.OUTPUT_DIR / "ARTEFACT_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"totals": manifest["totals"],
                      "checkpoint_entries": len(index)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
