# -*- coding: utf-8 -*-
"""
基线对比测试：RL (CTDE+PPO) vs 5种经典启发式算法
- Hungarian / Greedy-Nearest / Greedy-Priority / Greedy-Coalition / CBBA
- 相同环境 seed，公平对比
- 复用 old_test.py 的 build_env / load_model / run_episode / MetricsCollector
"""

import os
import sys
import csv
import time
import json
import argparse
import numpy as np
import torch

from old_task_env import TaskEnv
from old_network import CTDE_Network
from baselines import (
    HungarianBaseline,
    GreedyNearestBaseline,
    GreedyPriorityBaseline,
    GreedyCoalitionBaseline,
    CBBABaseline,
)
from old_test import (
    TestConfig,
    MetricsCollector,
    build_env,
    load_model,
    run_episode,
    pad_to_batch_max,
)


class BaselineTestConfig(TestConfig):
    CHECKPOINT_PATH = "./checkpoints_old/6/01"
    NUM_TEST_EPISODES = 50
    OUTPUT_DIR = "./comparison_results"


BASELINE_CLASSES = {
    "Hungarian":        HungarianBaseline,
    "Greedy-Nearest":   GreedyNearestBaseline,
    "Greedy-Priority":  GreedyPriorityBaseline,
    "Greedy-Coalition": GreedyCoalitionBaseline,
    "CBBA":             CBBABaseline,
}

METHODS_ORDER = ["RL (CTDE)", "Hungarian", "Greedy-Nearest", "Greedy-Priority", "Greedy-Coalition", "CBBA"]


def enrich_priority(env):
    """基于 deadline urgency 计算 search_priority，供 baselines 使用"""
    ct = env.current_time
    for task in env.task_dic.values():
        if task.get("finished") or task.get("expired"):
            task["search_priority"] = 0.0
            continue
        dl = task.get("deadline", ct + 999)
        time_remaining = max(0.0, dl - ct)
        task["search_priority"] = 1.5 - float(np.tanh(time_remaining / 50.0))


def run_baseline_episode(env, baseline, config, seed=None):
    """用 baseline 算法跑一个 episode，返回 (reward, wall_time, env_time, step_count)"""
    env.reset(seed=seed)
    baseline.env = env  # re-bind 因为 reset 重建了 task_dic/agent_dic

    step_count = 0
    t0 = time.time()

    while not env.finished and step_count < config.MAX_STEP_PER_EPISODE:
        if env.current_time >= config.ENV_TIME_LIMIT:
            env.finished = True
            break

        decision_ids = [aid for aid, a in env.agent_dic.items()
                        if a.get("need_decision", False)]

        if decision_ids:
            enrich_priority(env)
            assignment = baseline.assign_tasks(decision_ids)

            max_tid = max(env.task_dic.keys()) if env.task_dic else 0
            action_map = [("depot", None)] + [("task", tid) for tid in range(max_tid + 1)]

            for aid in decision_ids:
                task_id_plus_1 = assignment.get(aid, 0)
                action_idx = min(task_id_plus_1, max_tid + 1)
                env.apply_action(aid, action_idx, action_map)

        env.step(dt=config.DT)
        if step_count % 10 == 0:
            env.cleanup_failed_tasks_memory()
        step_count += 1

    wall_time = time.time() - t0
    episode_reward, _ = env.get_episode_reward()
    return episode_reward, wall_time, env.current_time, step_count


def run_comparison(config, device):
    """跑所有 15 个条件 × 6 个方法的对比"""
    net = load_model(config.CHECKPOINT_PATH, device)

    all_results = []
    raw_records = []

    for scale in config.AGENT_SCALES:
        s_name = scale["name"]
        s_species = scale["species_range"]
        s_per_species = scale["per_species_range"]

        for task_range in config.TASK_RANGES:
            cond_tag = f"{s_name} x tasks{task_range}"
            print(f"\n{'=' * 60}")
            print(f"Condition: {cond_tag}")
            print(f"{'=' * 60}")

            # 为每个方法独立测（共享 seeds）
            for method_name in METHODS_ORDER:
                collector = MetricsCollector(initial_task_range=task_range)

                for ep in range(config.NUM_TEST_EPISODES):
                    env = build_env(
                        task_range=task_range, config=config, seed=ep,
                        species_range=s_species, per_species_range=s_per_species,
                    )

                    if method_name == "RL (CTDE)":
                        r, wt, et, steps = run_episode(env, net, device, config, seed=ep)
                    else:
                        baseline_cls = BASELINE_CLASSES[method_name]
                        baseline = baseline_cls(env)
                        r, wt, et, steps = run_baseline_episode(env, baseline, config, seed=ep)

                    collector.add_episode(env, r, wt, et, steps)

                    raw_records.append({
                        "method": method_name,
                        "agent_scale": s_name,
                        "task_range": str(task_range),
                        "episode": ep,
                        "reward": r,
                        "wall_time": wt,
                        "env_time": et,
                        "steps": steps,
                    })

                summary = collector.get_summary()
                summary["method"] = method_name
                summary["agent_scale"] = s_name
                summary["task_range"] = list(task_range)
                all_results.append(summary)

                dl_sr = summary.get("deadline_success_rate_mean", 0)
                tc = summary.get("task_completion_rate_mean", 0)
                rw = summary.get("episode_reward_mean", 0)
                exp = summary.get("expired_rate_mean", 0)
                util = summary.get("resource_utilization_mean", 0)
                print(f"  {method_name:20s}  DL-SR={dl_sr:.2%}  "
                      f"Comp={tc:.2%}  Exp={exp:.1%}  Util={util:.1%}  Reward={rw:.1f}")

    return all_results, raw_records


def save_and_report(all_results, raw_records, config):
    """输出 JSON / CSV 并打印对比表"""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(config.OUTPUT_DIR, f"baseline_comparison_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(config.OUTPUT_DIR, f"baseline_per_episode_{ts}.csv")
    if raw_records:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=raw_records[0].keys())
            writer.writeheader()
            writer.writerows(raw_records)

    # 打印 DL-SR 对比表
    print(f"\n{'=' * 100}")
    print("DL-SR Comparison Table")
    print(f"{'=' * 100}")

    for s_name in ["small", "medium", "large"]:
        header = f"\n--- {s_name.upper()} ---"
        print(header)
        hdr = f"{'TaskRange':<14}" + "".join(f"{m:>18}" for m in METHODS_ORDER)
        print(hdr)
        print("-" * len(hdr))

        for tr in config.TASK_RANGES:
            row = [r for r in all_results
                   if r["agent_scale"] == s_name and r["task_range"] == list(tr)]
            if not row:
                continue
            values = {r["method"]: r.get("deadline_success_rate_mean", 0) for r in row}
            line = f"{str(tr):<14}" + "".join(f"{values.get(m, 0):18.2%}" for m in METHODS_ORDER)
            print(line)

    print(f"\nResults: {json_path}")
    print(f"Raw:     {csv_path}")
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_old/6/01")
    parser.add_argument("--gpu", action="store_true", default=True)
    args = parser.parse_args()

    config = BaselineTestConfig()
    config.NUM_TEST_EPISODES = args.episodes
    config.CHECKPOINT_PATH = args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Device: {device} | Episodes/condition: {config.NUM_TEST_EPISODES}")
    print(f"Checkpoint: {config.CHECKPOINT_PATH}")
    print(f"Total episodes: 15 conditions × 6 methods × {config.NUM_TEST_EPISODES} = "
          f"{15 * 6 * config.NUM_TEST_EPISODES}")

    all_results, raw_records = run_comparison(config, device)
    save_and_report(all_results, raw_records, config)


if __name__ == "__main__":
    main()
