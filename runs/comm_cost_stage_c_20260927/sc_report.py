"""Stage C 汇总：训练曲线、checkpoint 索引、账本刷新（只读落盘件，0 启动）。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

import sc_common as common


def training_curves() -> dict[str, Any]:
    """Per-seed training-time episodes, eval-time dev-scene means, lambda trajectory."""
    payload: dict[str, Any] = {}
    for training_seed in common.TRAIN_SEEDS:
        episodes: list[dict[str, Any]] = []
        for index in range(common.EPISODES_PER_SEED):
            matches = sorted(common.ROWS.glob(
                f"sc_train_s{training_seed}_e{index}_s*_a{index}.json"))
            if not matches:
                continue
            row = json.loads(matches[0].read_text(encoding="utf-8"))
            episodes.append({"episode_index": index, "scenario": row["scenario_seed"],
                             "on_time": row["on_time"], "spent_bytes": row["spent_bytes"],
                             "emissions": row["emissions"],
                             "decisions": (row.get("collector_summary") or {}).get(
                                 "decision_steps"),
                             "lambda_at_rollout": row["lambda_at_rollout"]})
        evals: dict[str, Any] = {}
        for timepoint in ("0", "64", "128"):
            scenes = []
            for scenario in common.DEVELOPMENT_SCENARIOS:
                tag = f"eval_s{training_seed}_{timepoint}"
                path = common.ROWS / f"sc_{tag}_s{scenario}_a0.json"
                if path.exists():
                    row = json.loads(path.read_text(encoding="utf-8"))
                    scenes.append({"scenario": scenario, "on_time": row["on_time"],
                                   "spent_bytes": row["spent_bytes"]})
            if scenes:
                evals[timepoint] = {
                    "scenes": len(scenes),
                    "mean_on_time": round(sum(s["on_time"] for s in scenes) / len(scenes), 6),
                    "mean_bytes": round(sum(s["spent_bytes"] for s in scenes) / len(scenes), 3)}
        state = torch.load(common.CHECKPOINTS / f"sc_train_s{training_seed}.pt",
                           map_location="cpu", weights_only=False)["training_state"]
        per_quarter = []
        for start in range(0, common.EPISODES_PER_SEED, 32):
            block = episodes[start:start + 32]
            if block:
                per_quarter.append({
                    "episodes": f"{start}..{start + len(block) - 1}",
                    "mean_on_time": round(sum(e["on_time"] for e in block) / len(block), 4),
                    "mean_bytes": round(sum(e["spent_bytes"] for e in block) / len(block), 1),
                    "mean_emissions": round(sum(e["emissions"] for e in block) / len(block), 3)})
        payload[str(training_seed)] = {
            "training_episodes": len(episodes),
            "per_32_episodes": per_quarter,
            "dev_eval": evals,
            "lambda_start": common.LAMBDA_INIT,
            "lambda_final": state["lambda"],
            "lambda_history": state["lambda_history"],
            "updates": state["updates"], "optimizer_steps": state["optimizer_steps"]}
    return payload


def checkpoint_index() -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for training_seed in common.TRAIN_SEEDS:
        path = common.CHECKPOINTS / f"sc_train_s{training_seed}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["training_state"]
        entries[str(training_seed)] = {
            "path": str(path.relative_to(common.ROOT)),
            "sha256": common.sha256_file(path),
            "episodes_done": state["episodes_done"], "updates": state["updates"],
            "optimizer_steps": state["optimizer_steps"], "lambda": state["lambda"],
            "trainer_tag": payload["trainer_tag"], "loss_version": payload["loss_version"],
            "freeze_sha256": payload["freeze"]["freeze_sha256"],
            "controller_parameter_sha256": hashlib.sha256(json.dumps(
                {key: {name: list(tensor.shape) for name, tensor in value.items()}
                 for key, value in payload["controller"].items()},
                sort_keys=True).encode("utf-8")).hexdigest()[:16],
            "batch_records": sorted(str(path.relative_to(common.ROOT)) for path in
                                    common.CHECKPOINTS.glob(
                                        f"sc_batch_s{training_seed}_b*.json")),
        }
    return entries


def main() -> int:
    curves = training_curves()
    index = checkpoint_index()
    common.write_json("TRAINING_CURVES.json", {"round": "23", "curves": curves,
                                               "note": "dev-scene means are greedy rollouts of "
                                                       "the frozen checkpoints; training-time "
                                                       "rows are sampled rollouts"})
    common.write_json("CHECKPOINT_INDEX.json", index)
    report = common.ledger_report()
    plan = common.read_json(common.OUTPUT_DIR / "START_LEDGER.json")["plan"]
    common.write_json("START_LEDGER.json",
                      {"plan": plan, "live": {"used": report["used"],
                                              "by_kind": report["by_kind"],
                                              "results": report["results"]}},
                      refresh=True)
    print(json.dumps({"seeds": sorted(curves), "ledger_used": report["used"],
                      "cap": common.LAUNCH_CAP}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
