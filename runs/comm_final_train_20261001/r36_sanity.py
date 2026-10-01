"""ROUND36 每次训练结束后的一次最小 sanity 核对（0 环境启动、0 optimizer.step、不产出结果数字）。

只做四件事，全部在真实文件与真实加载器上：
1. 用真实 `v2_trainer.load_checkpoint` 恢复 ep128 里程碑（两个版本字段不齐就会被恢复门拒绝）；
2. 读回 ep0 / ep64 / ep128 与 rolling 的两个版本字段、字节哈希与状态字段；
3. 对本轮自己的第 0 集 tape 做**一次**前向，只断言输出有限且形状正确；
4. 从本轮批记录核对训练量（批数、集覆盖、更新数、优化器步数、likelihood 对齐门）。

这里不登记任何按时任务数、成本字节或 loss 水平：本轮不做正式评估（授权边界 §三）。

    /d/project_env/python.exe r36_sanity.py --seed 11
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import r36_common as r36
import v2_common
import v2_controller
import v2_train
import v2_trainer

MILESTONES = (0, 64, 128)


def tape_path(seed: int) -> Path:
    """This round's own episode-0 tape, named the way the real writer names it."""
    directory = r36.seed_directory(seed) / "logs"
    matches = sorted(directory.glob(f"v2_train_s{seed}_e0_s*_a0_tape.jsonl"))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one episode-0 tape for seed {seed}, got {matches}")
    return matches[0]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="post-launch minimal sanity forward (no eval)")
    parser.add_argument("--seed", type=int, required=True, choices=list(r36.SEEDS))
    arguments = parser.parse_args(argv)
    seed = arguments.seed
    r36.configure(seed)
    freeze = r36.freeze_identity()

    milestones = {episodes: r36.seed_directory(seed) / "checkpoints"
                  / f"v2_train_s{seed}_ep{episodes}.pt" for episodes in MILESTONES}
    missing = [path.name for path in milestones.values() if not path.exists()]
    if missing:
        raise SystemExit(f"milestones missing for seed {seed}: {missing}")

    layer = {str(episodes): r36.checkpoint_version_fields(milestones[episodes])
             for episodes in MILESTONES}
    rolling = v2_train.rolling_path(seed)
    layer["rolling"] = r36.checkpoint_version_fields(rolling) if rolling.exists() else None

    # the real, version-gated resume path
    controller, _optimizer, state = v2_trainer.load_checkpoint(milestones[128], freeze=freeze)
    identity = v2_controller.parameter_identity(controller)

    # one forward pass over this round's own episode-0 tape; nothing is decided from it
    tape = list(v2_common.iter_jsonl(tape_path(seed)))
    features = torch.as_tensor(np.asarray([item["features"] for item in tape[:8]],
                                          dtype=np.float32))
    with torch.no_grad():
        logits, values = controller.forward(features)
    forward = {"tape_rows_in_episode_0": len(tape), "rows_used": int(min(8, len(tape))),
               "logits_shape": list(logits.shape), "values_shape": list(values.shape),
               "all_finite": bool(torch.isfinite(logits).all() and torch.isfinite(values).all()),
               "note": "shapes and finiteness only; no action is chosen, no outcome is read"}

    # the registered initialisation must be exactly what the ep0 milestone holds
    ep0_payload = torch.load(milestones[0], map_location="cpu", weights_only=False)
    ep0_hash = v2_controller.parameter_identity(
        v2_trainer._restore_controller(ep0_payload, ep0_payload["training_state"])
    )["parameter_sha256"]
    init_matches = ep0_hash == r36.fresh_controller_hash(seed)

    batches = sorted((r36.seed_directory(seed) / "checkpoints").glob(
        f"v2_batch_s{seed}_b*.json"), key=lambda path: int(path.stem.split("_b")[-1]))
    records = [r36.read_json(path) for path in batches]
    volume = {"batches": len(records), "expected_batches": r36.BATCHES_PER_SEED,
              "episodes_covered": sorted(int(episode) for item in records
                                         for episode in item["episodes"]),
              "episodes_done_in_state": int(state["episodes_done"]),
              "updates_in_state": int(state["updates"]),
              "optimizer_steps_in_state": int(state["optimizer_steps"]),
              "optimizer_steps_sum_of_batches": sum(int(item["update"]["optimizer_steps"])
                                                    for item in records),
              "likelihood_alignment_all_passed": all(
                  item["update"]["likelihood_alignment"]["passed"] for item in records),
              "controller_changed_in_every_batch": all(
                  item["controller_parameters_before"]["parameter_sha256"]
                  != item["controller_parameters_after"]["parameter_sha256"]
                  for item in records),
              "lambda_final": float(state["lambda"]),
              "parameters_after": identity["parameter_sha256"],
              "ep0_milestone_equals_the_registered_initialisation": init_matches}
    if volume["episodes_covered"] != list(range(r36.EPISODES_PER_SEED)):
        raise SystemExit(f"seed {seed} did not cover episodes 0..127 in its batch records")
    if not volume["likelihood_alignment_all_passed"] or not volume["batches"]:
        raise SystemExit(f"seed {seed} batch records failed their own gates: {volume}")

    payload: dict[str, Any] = {
        "kind": "round36-post-launch-sanity", "round": "36", "training_seed": int(seed),
        "episode_constructions_used": 0, "optimizer_steps_used": 0,
        "milestones_present": {str(episodes): True for episodes in MILESTONES},
        "checkpoint_version_and_hash_layer": layer,
        "real_loader_restore_of_ep128": {
            "succeeded": True,
            "training_state_fields": {key: state.get(key) for key in
                                      ("training_version", "reward_timing_version", "loss_version",
                                       "state_schema", "training_seed", "episodes_done",
                                       "updates", "optimizer_steps", "lambda")},
            "parameter_sha256_after_restore": identity["parameter_sha256"]},
        "single_forward_check": forward,
        "training_volume": volume,
        "lambda_history": [item["dual"]["lambda_next"] for item in records],
        "g_hat_history": [item["dual"]["g_hat"] for item in records],
        "evaluation_performed": False,
        "note": "the authorised minimal sanity forward after a training launch: it proves the "
                "fixed checkpoint loads under the version gate and can run a forward pass. It "
                "produces no performance numbers and is not an evaluation",
        "written_at": time.strftime("%F %T")}
    r36.write_seed_json(seed, "SANITY.json", payload, refresh=True)
    r36.write_seed_json(seed, "CHECKPOINT_HASHES.json", {
        "kind": "round36-checkpoint-hash-list", "training_seed": int(seed),
        "files": {**{f"milestone_ep{episodes}": layer[str(episodes)] for episodes in MILESTONES},
                  "rolling": layer["rolling"],
                  "batch_records": {path.name: r36.sha256_file(path) for path in batches}},
        "log_and_tape_files_kept_outside_git": sorted(
            path.name for path in (r36.seed_directory(seed) / "logs").glob("*.jsonl")),
        "row_files_kept_outside_git": len(list((r36.seed_directory(seed) / "rows").glob("*.json"))),
        "both_version_fields_complete_everywhere": all(
            item["both_fields_complete"] for item in layer.values() if item),
        "written_at": time.strftime("%F %T")}, refresh=True)
    r36.write_seed_json(seed, "LAMBDA_TRAJECTORY.json", {
        "kind": "round36-lambda-trajectory", "training_seed": int(seed),
        "batches": [item["batch"] for item in records],
        "lambda_before": [item["dual"]["lambda_before"] for item in records],
        "lambda_next": [item["dual"]["lambda_next"] for item in records],
        "g_hat": [item["dual"]["g_hat"] for item in records],
        "lambda_at_rollout_per_batch": [item["lambda_at_rollout"] for item in records],
        "samples_per_batch": [int(item["update"]["samples"]) for item in records],
        "optimizer_steps_per_batch": [int(item["update"]["optimizer_steps"]) for item in records],
        "parameters_before_batch0": records[0]["controller_parameters_before"]["parameter_sha256"],
        "parameters_after_last_batch":
            records[-1]["controller_parameters_after"]["parameter_sha256"],
        "written_at": time.strftime("%F %T")}, refresh=True)
    complete = all(item["both_fields_complete"] for item in layer.values() if item)
    print(json.dumps({"seed": seed,
                      "ep128_restored_under_version_gate": True,
                      "both_fields_complete_everywhere": complete,
                      "forward_all_finite": forward["all_finite"],
                      "batches": volume["batches"],
                      "optimizer_steps": volume["optimizer_steps_in_state"],
                      "lambda_final": volume["lambda_final"],
                      "init_matches_ep0": init_matches},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
