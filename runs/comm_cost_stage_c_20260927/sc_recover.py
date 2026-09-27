"""Stage C 首批次恢复：从落盘 episode 件确定性地重算 batch 0 的更新（0 新启动）。

背景：batch 0 的 4 个训练 episode 正常跑完并落盘（row + tape + transcript），PPO 更新在内存
完成后，写批记录时因 `checkpoints/` 目录不存在而崩溃，checkpoint 未保存。更新是落盘件与
**登记初值权重**的确定函数（shuffle 种子派生、CPU 前向无随机性），因此可从磁盘完整重算，
不需要也不允许重跑这 4 个 episode。

恢复脚本先证明这 4 个 episode 确实由登记初值采样（log_prob 用初值权重逐 tape 重算 ≤1e-6），
再重算 update / g_hat / 对偶步，写批记录与 checkpoint，并把恢复 provenance 登记进批记录。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sc_common as common
import sc_trainer as trainer
from onpolicy import joint_log_prob
from sc_controller import fresh_controller, parameter_identity


class TapeShim:
    """The collector view batch_samples expects, reconstructed from the tape file."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records


def load_batch_episodes(training_seed: int, batch: int) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for episode_index in range(batch * common.EPISODES_PER_UPDATE,
                               (batch + 1) * common.EPISODES_PER_UPDATE):
        tag = f"train_s{training_seed}_e{episode_index}"
        row_path = common.ROWS / f"sc_{tag}_s*_a{episode_index}.json"
        matches = sorted(common.ROWS.glob(f"sc_{tag}_s*_a{episode_index}.json"))
        if not matches:
            raise FileNotFoundError(f"no row for {tag}")
        row = json.loads(matches[0].read_text(encoding="utf-8"))
        tape_path = common.LOGS / (matches[0].stem.replace(".json", "") + "_tape.jsonl")
        if not tape_path.exists():
            raise FileNotFoundError(f"no tape for {tag}: {tape_path}")
        records = list(common.iter_jsonl(tape_path))
        episode = dict(row)
        episode["collector"] = TapeShim(records)
        episode["lambda"] = float(row["lambda_at_rollout"])
        episode["steps"] = int(row["steps"])
        episode["episode_index"] = int(row["episode_index"])
        episode["scenario_seed"] = int(row["scenario_seed"])
        episode["on_time"] = int(row["on_time"])
        episodes.append(episode)
    return episodes


def verify_init_alignment(controller: Any, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """The four episodes must have been sampled by the registered initialisation."""
    gaps: list[float] = []
    for episode in episodes:
        for tape in episode["collector"].records:
            logits, _values = controller.forward(
                torch.as_tensor(np.asarray(tape["features"], dtype=np.float32)))
            with torch.no_grad():
                picked = torch.tensor([tape["actions"][t] for t in tape["targets"]],
                                      dtype=torch.long)
                usable = torch.tensor([1.0 if t in tape["actionable"] else 0.0
                                       for t in tape["targets"]], dtype=torch.float32)
                gaps.append(abs(float(joint_log_prob(logits, picked, usable))
                                - float(tape["log_prob"])))
    return {"tapes": len(gaps), "max_abs_gap": max(gaps) if gaps else 0.0,
            "tolerance": common.LOG_PROB_TOLERANCE,
            "passed": (not gaps) or max(gaps) <= common.LOG_PROB_TOLERANCE}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="recompute batch 0 from disk artefacts")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch", type=int, default=0)
    args = parser.parse_args(argv)

    import subprocess

    import sc_check

    freeze = sc_check.freeze_record()
    # The four episodes ran under the freeze registered before the mkdir fix; that identity is
    # read back from git so the record states which code produced the rollouts.
    episodes_identity = None
    try:
        manifest_text = subprocess.run(
            ["git", "show", "6a1ea0f:runs/comm_cost_stage_c_20260927/"
                             "CONFIG_AND_MODEL_MANIFEST.json"],
            cwd=common.ROOT, capture_output=True, text=True, encoding="utf-8").stdout
        if manifest_text.strip():
            episodes_identity = json.loads(manifest_text)["code_identity"]["freeze_sha256"]
    except Exception:
        episodes_identity = None
    common.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    controller = fresh_controller(args.seed)
    optimizer = torch.optim.Adam(controller.parameters, lr=common.LEARNING_RATE)
    episodes = load_batch_episodes(args.seed, args.batch)
    alignment = verify_init_alignment(controller, episodes)
    if not alignment["passed"]:
        raise SystemExit(f"the episodes were not sampled by the registered initialisation: "
                         f"{alignment}")

    references = {}
    for row_path in sorted(common.ROWS.glob("sc_reference_s*_a1.json")):
        row = json.loads(row_path.read_text(encoding="utf-8"))
        references[int(row["scenario_seed"])] = float(row["on_time"])

    before = parameter_identity(controller)
    update_report = trainer.ppo_update(controller, optimizer, episodes,
                                       update_index=args.batch)
    g_record = trainer.batch_g_hat(episodes, references)
    dual_record = trainer.dual_update(common.LAMBDA_INIT, g_record["g_hat"])
    state = trainer.fresh_training_state(args.seed)
    state["episodes_done"] = len(episodes)
    state["lambda"] = dual_record["lambda_next"]
    state["lambda_history"] = [{"batch": args.batch, "g_hat": dual_record["g_hat"],
                                "lambda_before": dual_record["lambda_before"],
                                "lambda_next": dual_record["lambda_next"]}]
    state["updates"] = 1
    state["optimizer_steps"] = int(update_report.get("optimizer_steps", 0))
    after = parameter_identity(controller)

    batch_record = {
        "batch": args.batch, "training_seed": args.seed,
        "episodes": [int(episode["episode_index"]) for episode in episodes],
        "scenarios": [int(episode["scenario_seed"]) for episode in episodes],
        "lambda_at_rollout": common.LAMBDA_INIT, "update": update_report,
        "constraint": g_record, "dual": dual_record,
        "controller_parameters_before": before, "controller_parameters_after": after,
        "optimizer_parameter_ids": [id(parameter) for group in optimizer.param_groups
                                    for parameter in group["params"]],
        "controller_parameter_ids": [id(parameter) for parameter in controller.parameters],
        "frozen_checkpoints": {str(path.relative_to(common.ROOT)): common.sha256_file(path)
                               for path in common.CHECKPOINT_FILES},
        "recovery": {
            "kind": "batch_recomputed_from_disk",
            "reason": "the in-memory batch 0 update was lost when the batch-record write "
                      "failed (missing checkpoints/ directory); the four episodes and their "
                      "tapes are on disk and the update is a deterministic function of them "
                      "and the registered initialisation, so it is recomputed rather than "
                      "re-run (no new launches, no episode is trained twice)",
            "init_alignment": alignment,
            "launches_spent_by_recovery": 0,
            "episodes_code_identity": episodes_identity,
            "recompute_code_identity": freeze["freeze_sha256"],
        },
    }
    record_path = common.CHECKPOINTS / f"sc_batch_s{args.seed}_b{args.batch}.json"
    record_path.write_text(json.dumps(batch_record, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    trainer.save_checkpoint(controller, optimizer, state,
                            common.CHECKPOINTS / f"sc_train_s{args.seed}.pt", freeze=freeze)
    print(json.dumps({"recovered_batch": args.batch, "training_seed": args.seed,
                      "alignment_max_gap": alignment["max_abs_gap"],
                      "optimizer_steps": update_report["optimizer_steps"],
                      "g_hat": g_record["g_hat"],
                      "lambda_next": dual_record["lambda_next"],
                      "controller_changed": before["parameter_sha256"] != after["parameter_sha256"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
