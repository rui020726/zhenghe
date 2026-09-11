# -*- coding: utf-8 -*-
"""
实验 B：消融实验
固定 config: large/(20,30)，跑 5 个变体 × 20 episodes
变体：
  1. baseline:    完整系统（v2 默认配置）
  2. -gossip:     COMM_RANGE=0（无通信）
  3. -tidal:      ENABLE_TIDAL_TASKS=False（无潮汐注入）
  4. -soft-deadline: HARD_EXPIRE=True（过期任务硬删除，对照软约束设计的价值）
  5. +tight-deadline: DEADLINE_TIGHTNESS=(1.0, 1.3)（更紧 deadline，压力测试）
"""
import os
import sys
import time
import json
import copy
import argparse
import numpy as np
import torch

from old_test import TestConfig, MetricsCollector, build_env, load_model, pad_to_batch_max


# large 规模配置（与 paper_compare.py 一致）
SPECIES_RANGE = (4, 5)
PER_SPECIES_RANGE = (5, 7)
TASK_RANGE = (20, 30)


def make_config(variant: str) -> TestConfig:
    """根据变体名返回修改后的 TestConfig"""
    config = TestConfig()
    if variant == "baseline":
        pass
    elif variant == "-gossip":
        config.COMM_RANGE = 0.0
    elif variant == "-tidal":
        config.ENABLE_TIDAL_TASKS = False
    elif variant == "-soft-deadline":
        config.HARD_EXPIRE = True
    elif variant == "+tight-deadline":
        config.DEADLINE_TIGHTNESS = (1.0, 1.3)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return config


@torch.no_grad()
def run_rl_episode(env, net, device, max_steps=3000, env_time_limit=3000.0):
    """镜像 paper_compare.run_rl_episode"""
    step_count = 0
    t0 = time.time()

    while not env.finished and step_count < max_steps:
        if env.current_time >= env_time_limit:
            env.finished = True
            break

        decision_ids = [aid for aid, a in env.agent_dic.items() if a.get("need_decision", False)]
        if decision_ids:
            batch_s, batch_n, batch_t, batch_m, action_maps = [], [], [], [], []
            for aid in decision_ids:
                s, n, t, m, amap = env.get_local_obs(aid)
                batch_s.append(torch.as_tensor(s, dtype=torch.float32))
                batch_n.append(torch.as_tensor(n, dtype=torch.float32))
                batch_t.append(torch.as_tensor(t, dtype=torch.float32))
                batch_m.append(torch.as_tensor(m, dtype=torch.bool))
                action_maps.append(amap)

            ts = torch.stack(batch_s).to(device)
            tn = torch.stack(batch_n).to(device)
            tt, _ = pad_to_batch_max(batch_t, pad_value=0.0)
            tt = tt.to(device)
            tm_exp = [m.unsqueeze(-1).float() for m in batch_m]
            tm, _ = pad_to_batch_max(tm_exp, pad_value=1.0)
            tm = tm.squeeze(-1).bool().to(device)

            dist = net.get_action(ts, tn, tt, tm)
            actions = dist.sample()

            for i, aid in enumerate(decision_ids):
                env.apply_action(aid, int(actions[i].item()), action_maps[i])

        env.step(dt=1.0)
        if step_count % 10 == 0:
            env.cleanup_failed_tasks_memory()
        step_count += 1

    wall_time = time.time() - t0
    episode_reward, _ = env.get_episode_reward()
    return episode_reward, wall_time, env.current_time, step_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_old/7/1/ctde_best.pth")
    parser.add_argument("--output", type=str, default="./paper_results/expB_ablation.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {args.checkpoint}")
    net = load_model(args.checkpoint, device)

    variants = ["baseline", "-gossip", "-tidal", "-soft-deadline", "+tight-deadline"]
    print(f"\nVariants: {variants}")
    print(f"Task range: {TASK_RANGE} | species={SPECIES_RANGE} | per_species={PER_SPECIES_RANGE}")
    print(f"Episodes per variant: {args.episodes}")
    print()

    results = []
    start_time = time.time()

    for variant in variants:
        config = make_config(variant)
        variant_metrics = []

        for ep in range(args.episodes):
            seed = ep
            env = build_env(
                task_range=TASK_RANGE,
                config=config,
                seed=seed,
                species_range=SPECIES_RANGE,
                per_species_range=PER_SPECIES_RANGE,
            )
            try:
                r, wt, et, sc = run_rl_episode(env, net, device)
                collector = MetricsCollector(initial_task_range=TASK_RANGE)
                collector.add_episode(env, r, wt, et, sc)
                summary = collector.get_summary()
                variant_metrics.append({
                    "reward": r,
                    "dl_sr": summary["deadline_success_rate_mean"],
                    "completion": summary["task_completion_rate_mean"],
                    "expired": summary["expired_rate_mean"],
                    "slack": summary["avg_slack_mean"],
                    "res_util": summary["resource_utilization_mean"],
                    "wall": wt,
                })
            except Exception as e:
                print(f"  [ERR] {variant} ep{ep}: {e}")
                variant_metrics.append(None)

        valid = [m for m in variant_metrics if m is not None]
        if not valid:
            print(f"  {variant:20s}: ALL FAILED")
            continue

        result = {
            "variant": variant,
            "num_episodes": len(valid),
            "reward_mean": float(np.mean([m["reward"] for m in valid])),
            "reward_std": float(np.std([m["reward"] for m in valid])),
            "dl_sr_mean": float(np.mean([m["dl_sr"] for m in valid])),
            "dl_sr_std": float(np.std([m["dl_sr"] for m in valid])),
            "completion_mean": float(np.mean([m["completion"] for m in valid])),
            "completion_std": float(np.std([m["completion"] for m in valid])),
            "expired_mean": float(np.mean([m["expired"] for m in valid])),
            "expired_std": float(np.std([m["expired"] for m in valid])),
            "slack_mean": float(np.mean([m["slack"] for m in valid])),
            "slack_std": float(np.std([m["slack"] for m in valid])),
            "res_util_mean": float(np.mean([m["res_util"] for m in valid])),
            "res_util_std": float(np.std([m["res_util"] for m in valid])),
            "wall_mean": float(np.mean([m["wall"] for m in valid])),
            "wall_std": float(np.std([m["wall"] for m in valid])),
        }
        results.append(result)

        print(f"  {variant:20s} | DL-SR={result['dl_sr_mean']:6.2%}±{result['dl_sr_std']:.2%} | "
              f"Comp={result['completion_mean']:6.2%} | "
              f"Exp={result['expired_mean']:5.2%} | "
              f"Slack={result['slack_mean']:5.1f} | "
              f"R={result['reward_mean']:7.1f} | "
              f"wall={result['wall_mean']:5.1f}s")

    total_time = time.time() - start_time
    print(f"\n=== Total time: {total_time:.1f}s ({total_time/60:.1f}min) ===")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
