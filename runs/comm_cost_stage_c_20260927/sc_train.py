"""Stage C 训练驱动：三 seed × 128 集的约束式成本学习（ROUND22 spec §5/§7）。

每个训练 seed：Controller(训练种子) 初值 → 32 批 ×（4 集 rollout + 一次 PPO 更新 +
一次对偶 lambda 更新）→ 每批存 checkpoint（崩溃恢复只用本轮合法完整 checkpoint，恢复 RNG
与顺序）。lambda 在同一批 PPO 期间固定，批后更新；g_hat 用配对的 immediate reference
缓存标量。只训练通信模块（optimizer 只含 controller.parameters）；搜索/分配权重来自两个
冻结 checkpoint 文件，逐阶段核对 SHA256。

    python sc_train.py --seed 11 --episodes 4         # 首4集 + 正确性门禁
    python sc_train.py --seed 11 --episodes 128       # 恢复后继续
    python sc_train.py --seed 22 --episodes 128
    python sc_train.py --seed 33 --episodes 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

import sc_common as common
import sc_run as runner
import sc_trainer as trainer
from sc_controller import fresh_controller, parameter_identity


def load_reference_cache() -> dict[int, dict[str, Any]]:
    """The cached immediate references, keyed by scenario seed."""
    cache: dict[int, dict[str, Any]] = {}
    for path in sorted(common.ROWS.glob("sc_reference_s*_a1.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        scenario = int(row["scenario_seed"])
        if scenario in cache and cache[scenario]["behaviour_sha256"] != row["behaviour_sha256"]:
            raise RuntimeError(f"scenario {scenario} has two different cached references")
        cache[scenario] = row
    return cache


def train_seed(training_seed: int, *, episodes: int, freeze: dict[str, Any],
               reference_cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Train one communication seed for ``episodes`` episodes, resuming from its checkpoint."""
    checkpoint = common.CHECKPOINTS / f"sc_train_s{training_seed}.pt"
    if checkpoint.exists():
        controller, optimizer, state = trainer.load_checkpoint(checkpoint, freeze=freeze)
        print(f"resumed seed {training_seed} at episode {state['episodes_done']} "
              f"lambda={state['lambda']:.4f} updates={state['updates']}", flush=True)
    else:
        controller = fresh_controller(training_seed)
        optimizer = torch.optim.Adam(controller.parameters, lr=common.LEARNING_RATE)
        state = trainer.fresh_training_state(training_seed)
        print(f"seed {training_seed}: fresh init Controller({training_seed}), "
              f"optimizer Adam lr={common.LEARNING_RATE}", flush=True)

    references = {scenario: float(row["on_time"]) for scenario, row in reference_cache.items()}
    order = common.scenario_order()
    before = parameter_identity(controller)
    history: list[dict[str, Any]] = []
    started = time.time()

    while state["episodes_done"] < episodes:
        batch_index = state["episodes_done"] // common.EPISODES_PER_UPDATE
        lam = float(state["lambda"])          # frozen for the whole batch (spec §2)
        episodes_in_batch: list[dict[str, Any]] = []
        for offset in range(common.EPISODES_PER_UPDATE):
            episode_index = state["episodes_done"]
            if episode_index >= episodes:
                break
            scenario = order[episode_index]
            if scenario not in references:
                raise RuntimeError(
                    f"seed {training_seed} episode {episode_index}: scenario {scenario} has no "
                    "cached immediate reference; the reference cache must be complete before "
                    "training starts")
            episode = runner.run_training_episode(
                controller, training_seed=training_seed, episode_index=episode_index,
                scenario_seed=scenario, lam=lam)
            episodes_in_batch.append(episode)
            state["episodes_done"] += 1
            history.append({"episode_index": episode_index, "scenario": scenario,
                            "on_time": episode["on_time"], "spent_bytes": episode["spent_bytes"],
                            "emissions": episode["emissions"],
                            "decisions": (episode.get("collector_summary") or {}).get(
                                "decision_steps")})
            print(f"seed {training_seed} episode {episode_index}: scenario {scenario} "
                  f"on_time={episode['on_time']} bytes={episode['spent_bytes']} "
                  f"lambda={lam:.4f}", flush=True)

        if not episodes_in_batch:
            break
        update_report = trainer.ppo_update(controller, optimizer, episodes_in_batch,
                                           update_index=batch_index)
        g_record = trainer.batch_g_hat(episodes_in_batch, references)
        dual_record = trainer.dual_update(state["lambda"], g_record["g_hat"])
        state["lambda"] = dual_record["lambda_next"]
        state["lambda_history"].append({"batch": batch_index,
                                        "g_hat": dual_record["g_hat"],
                                        "lambda_before": dual_record["lambda_before"],
                                        "lambda_next": dual_record["lambda_next"]})
        state["updates"] += 1
        state["optimizer_steps"] += int(update_report.get("optimizer_steps", 0))
        after = parameter_identity(controller)
        if after["parameter_sha256"] == before["parameter_sha256"]:
            raise RuntimeError(
                f"seed {training_seed} batch {batch_index}: the optimizer did not change the "
                "communication controller")
        batch_record = {
            "batch": batch_index, "training_seed": training_seed,
            "episodes": [int(episode["episode_index"]) for episode in episodes_in_batch],
            "scenarios": [int(episode["scenario_seed"]) for episode in episodes_in_batch],
            "lambda_at_rollout": lam, "update": update_report,
            "constraint": g_record, "dual": dual_record,
            "controller_parameters_before": before, "controller_parameters_after": after,
            "frozen_checkpoints": {
                str(path.relative_to(common.ROOT)): common.sha256_file(path)
                for path in common.CHECKPOINT_FILES},
            "optimizer_parameter_ids": [id(parameter) for group in optimizer.param_groups
                                        for parameter in group["params"]],
            "controller_parameter_ids": [id(parameter) for parameter in controller.parameters],
        }
        common.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        (common.CHECKPOINTS / f"sc_batch_s{training_seed}_b{batch_index}.json").write_text(
            json.dumps(batch_record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        trainer.save_checkpoint(controller, optimizer, state, checkpoint, freeze=freeze)
        before = after
        print(f"seed {training_seed} batch {batch_index}: samples={update_report['samples']} "
              f"opt_steps={update_report['optimizer_steps']} g_hat={dual_record['g_hat']:+.4f} "
              f"lambda {dual_record['lambda_before']:.4f} -> {dual_record['lambda_next']:.4f} "
              f"({update_report['policy_surrogate']:.5f} surrogate)", flush=True)

    return {"training_seed": training_seed, "episodes_done": state["episodes_done"],
            "updates": state["updates"], "optimizer_steps": state["optimizer_steps"],
            "lambda": state["lambda"], "lambda_history": state["lambda_history"],
            "seconds": round(time.time() - started, 2),
            "checkpoint": str(checkpoint),
            "controller_parameters": parameter_identity(controller),
            "history": history}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="train one communication seed")
    parser.add_argument("--seed", type=int, required=True, choices=list(common.TRAIN_SEEDS))
    parser.add_argument("--episodes", type=int, default=common.EPISODES_PER_SEED)
    args = parser.parse_args(argv)
    import sc_check

    freeze = sc_check.freeze_record()
    cache = load_reference_cache()
    needed = set(common.scenario_order()[:args.episodes])
    missing = sorted(scenario for scenario in needed if scenario not in cache)
    if missing:
        raise SystemExit(f"reference cache is missing scenarios {missing[:8]}; run the "
                         "reference block first (it is part of the 672 normal launches)")
    summary = train_seed(args.seed, episodes=args.episodes, freeze=freeze,
                         reference_cache=cache)
    print(json.dumps({key: summary[key] for key in
                      ("training_seed", "episodes_done", "updates", "optimizer_steps",
                       "lambda", "seconds")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
