"""ROUND24 v2 重跑故障恢复：只从本轮合法 v2 落盘件重算丢失的批次更新（0 新启动）。

ROUND24 §2.5：旧 `sc_recover.py` 硬编码旧提交/旧 manifest 的恢复入口未证明适用于 v2，已禁用；
不从旧 tape 重建新训练。本入口只读**本轮** `runs/comm_cost_v2_rerun_20260928/` 的 rows/tapes/
rolling checkpoint 与本轮 manifest，拒绝任何其他来源：

- 每个 row 的 run config 用本轮登记的配置重新构造比对（不启动环境），证明 tape 属于本轮配置；
- 每个 tape 的 log_prob 用该批次的 rollout 权重（batch 0 = 登记初值；batch k>0 = 本轮 rolling
  checkpoint 的 k-1 批后状态）重算，≤1e-6 才承认这些 episode 由该权重采样；
- 更新是落盘件与 rollout 权重的确定函数（shuffle 种子派生、CPU 前向无随机性），因此重算而不
  重跑；4 次启动不重复消耗，也不挑选更好结果。

    /d/project_env/python.exe v2_recover.py --seed 11 --batch 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import v2_common as common
import v2_trainer as trainer
from onpolicy import joint_log_prob
from v2_controller import fresh_controller, parameter_identity
from v2_train import milestone_path, rolling_path


class TapeShim:
    """The collector view batch_samples expects, reconstructed from the tape file."""

    def __init__(self, records: list[dict[str, Any]]):
        self.records = records


def load_batch_episodes(training_seed: int, batch: int) -> list[dict[str, Any]]:
    """Only this round's rows and tapes, matched to the frozen scenario order."""
    order = common.scenario_order()
    episodes: list[dict[str, Any]] = []
    for episode_index in range(batch * common.EPISODES_PER_UPDATE,
                               (batch + 1) * common.EPISODES_PER_UPDATE):
        tag = f"train_s{training_seed}_e{episode_index}"
        matches = sorted(common.ROWS.glob(f"v2_{tag}_s*_a{episode_index}.json"))
        if not matches:
            raise FileNotFoundError(f"no this-round row for {tag}")
        row = json.loads(matches[0].read_text(encoding="utf-8"))
        if not str(matches[0].resolve()).startswith(str(common.OUTPUT_DIR.resolve())):
            raise RuntimeError(f"row {matches[0]} is outside this round's output directory")
        if int(row["scenario_seed"]) != int(order[episode_index]):
            raise RuntimeError(f"{tag}: scenario {row['scenario_seed']} != frozen order "
                               f"{order[episode_index]}")
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


def verify_configuration(episodes: list[dict[str, Any]],
                         manifest: dict[str, Any]) -> dict[str, Any]:
    """Rebuild each episode's run configuration from this round's registered settings.

    ``build_joint_run`` is constructed but never stepped: this compares the row's recorded
    config hash against the configuration this round froze, proving the tapes belong to it.
    """
    from integrated_sar.joint_driver import RunConfiguration, build_joint_run

    expected_hashes = set()
    for episode in episodes:
        fields, _substituted = common.stage_b_inherited_config(int(episode["scenario_seed"]))
        fields["comm_policy"] = "learned"
        paths = common.cell_paths("train", int(episode["scenario_seed"]),
                                  attempt=int(episode["episode_index"]),
                                  tag=f"train_s{episode.get('training_seed')}"
                                      f"_e{episode['episode_index']}")
        fields["output"] = str(paths["lifecycle"])
        fields["comm_transcript"] = str(paths["transcript"])
        configuration = RunConfiguration(**fields)
        configuration.validate()
        run = build_joint_run(configuration)
        expected_hashes.add(str(run.run_signature["config_hash"]))
    recorded = {str(episode["config_hash"]) for episode in episodes}
    return {"expected": sorted(expected_hashes), "recorded": sorted(recorded),
            "passed": recorded == expected_hashes and len(recorded) == 1,
            "note": "configuration rebuilt from this round's inherited fields under the frozen "
                    "manifest; no environment is stepped"}


def verify_alignment(controller: Any, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """The batch's episodes must have been sampled by the controller state they resume from."""
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
    parser = argparse.ArgumentParser(description="recompute a lost in-memory batch from this "
                                                 "round's own disk artefacts")
    parser.add_argument("--seed", type=int, required=True, choices=list(common.TRAIN_SEEDS))
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args(argv)

    manifest = json.loads(
        (common.OUTPUT_DIR / "CONFIG_AND_MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    freeze = {"freeze_sha256": manifest["code_identity"]["freeze_sha256"],
              "config_hash": manifest["configuration_hash"],
              "branch_start": manifest["branch_start"],
              "head_at_freeze": manifest["head_at_freeze"]}

    record_path = common.CHECKPOINTS / f"v2_batch_s{args.seed}_b{args.batch}.json"
    if record_path.exists():
        raise SystemExit(f"batch {args.batch} already has a record on disk; nothing to recover")

    rolling = rolling_path(args.seed)
    if rolling.exists():
        controller, optimizer, state = trainer.load_checkpoint(rolling, freeze=freeze)
        if int(state["episodes_done"]) != args.batch * common.EPISODES_PER_UPDATE:
            raise SystemExit(
                f"the rolling checkpoint holds {state['episodes_done']} episodes; batch "
                f"{args.batch} expects {args.batch * common.EPISODES_PER_UPDATE} -- the "
                "in-memory loss cannot be recomputed from this state")
        rollout_state = "rolling checkpoint after batch %d" % (args.batch - 1)
    elif args.batch == 0:
        controller = fresh_controller(args.seed)
        optimizer = torch.optim.Adam(controller.parameters, lr=common.LEARNING_RATE)
        state = trainer.fresh_training_state(args.seed)
        rollout_state = "registered initialisation"
    else:
        raise SystemExit(f"no rolling checkpoint for batch {args.batch} and it is not batch 0; "
                         "only a batch whose rollout weights are reproducible can be recovered")

    episodes = load_batch_episodes(args.seed, args.batch)
    configuration = verify_configuration(episodes, manifest)
    if not configuration["passed"]:
        raise SystemExit(f"the episodes do not match this round's frozen configuration: "
                         f"{configuration}")
    alignment = verify_alignment(controller, episodes)
    if not alignment["passed"]:
        raise SystemExit(f"the episodes were not sampled by {rollout_state}: {alignment}")

    references = {}
    for row_path in sorted(common.ROWS.glob("v2_reference_s*_a1.json")):
        row = json.loads(row_path.read_text(encoding="utf-8"))
        references[int(row["scenario_seed"])] = float(row["on_time"])
    missing = sorted({int(episode["scenario_seed"]) for episode in episodes}
                     - set(references))
    if missing:
        raise SystemExit(f"this round's reference cache is missing scenarios {missing[:8]}")

    before = parameter_identity(controller)
    lambda_at_rollout = float(state["lambda"])   # the lambda the rollouts were drawn with
    update_report = trainer.ppo_update(controller, optimizer, episodes, update_index=args.batch)
    g_record = trainer.batch_g_hat(episodes, references)
    dual_record = trainer.dual_update(lambda_at_rollout, g_record["g_hat"])
    state["lambda"] = dual_record["lambda_next"]
    state["lambda_history"].append({"batch": args.batch, "g_hat": dual_record["g_hat"],
                                    "lambda_before": dual_record["lambda_before"],
                                    "lambda_next": dual_record["lambda_next"]})
    state["updates"] += 1
    state["optimizer_steps"] += int(update_report.get("optimizer_steps", 0))
    state["episodes_done"] += len(episodes)
    after = parameter_identity(controller)

    from v2_train import module_digests

    batch_record = {
        "batch": args.batch, "training_seed": args.seed,
        "episodes": [int(episode["episode_index"]) for episode in episodes],
        "scenarios": [int(episode["scenario_seed"]) for episode in episodes],
        "lambda_at_rollout": lambda_at_rollout,
        "update": update_report,
        "constraint": g_record, "dual": dual_record,
        "controller_parameters_before": before, "controller_parameters_after": after,
        "controller_module_digests_before": None,
        "controller_module_digests_after": module_digests(controller),
        "frozen_checkpoints": {str(path.relative_to(common.ROOT)): common.sha256_file(path)
                               for path in common.CHECKPOINT_FILES},
        "optimizer_parameter_shapes": [list(parameter.shape) for group in optimizer.param_groups
                                       for parameter in group["params"]],
        "controller_parameter_shapes": [list(parameter.shape)
                                        for parameter in controller.parameters],
        "recovery": {
            "kind": "batch_recomputed_from_this_round_disk_artefacts",
            "reason": "the in-memory batch update was lost (crash between the rollout phase and "
                      "the batch-record write); the update is a deterministic function of the "
                      "on-disk rows/tapes and the rollout weights, so it is recomputed instead "
                      "of re-run (no new launches, no episode trained twice)",
            "rollout_weights": rollout_state,
            "init_or_rolling_alignment": alignment,
            "configuration_verification": configuration,
            "launches_spent_by_recovery": 0,
            "recovery_code_identity": freeze["freeze_sha256"],
            "refusals": "old-round tapes, old commits and old manifests are never read; any row "
                        "outside this round's output directory is refused"},
    }
    common.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(batch_record, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    trainer.save_checkpoint(controller, optimizer, state, rolling, freeze=freeze)
    if state["episodes_done"] in common.MILESTONE_EPISODES:
        from v2_train import save_milestone

        save_milestone(controller, optimizer, state, args.seed, int(state["episodes_done"]),
                       freeze=freeze)
    print(json.dumps({"recovered_batch": args.batch, "training_seed": args.seed,
                      "rollout_weights": rollout_state,
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
