# -*- coding: utf-8 -*-
"""
CTDE 模型测试程序（消融实验版 Ablation Test）
- ✅ 一键切换测试 Baseline (15维) 或 Proposed (20维) 模型。
- ✅ 自动读取对应的权重文件夹，避免混淆。
- ✅ 新增核心指标：Stage Gap Time (空窗期时间)，用于直观展现预判带来的时间节省。
- ✅ 结果自动保存到独立文件 (baseline.json / prepos.json)。
"""

import os
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# ✅ 引入你刚才修改的消融版环境和网络
from Ab_task_env import TaskEnv
from Ab_old_network import CTDE_Network


def pad_to_batch_max(batch_data_list):
    if len(batch_data_list) == 0:
        return None, None

    batch_size = len(batch_data_list)
    max_len = max(x.shape[0] for x in batch_data_list)
    feature_dim = batch_data_list[0].shape[-1] if batch_data_list[0].dim() > 1 else 1

    if batch_data_list[0].dim() == 1:
        padded = torch.zeros(batch_size, max_len, dtype=batch_data_list[0].dtype, device=batch_data_list[0].device)
    else:
        padded = torch.zeros(batch_size, max_len, feature_dim, dtype=batch_data_list[0].dtype,
                             device=batch_data_list[0].device)

    lengths = []
    for i, data in enumerate(batch_data_list):
        actual_len = data.shape[0]
        padded[i, :actual_len] = data
        lengths.append(actual_len)

    return padded, torch.tensor(lengths)


class TestConfig:
    # ========================================================
    # 🌟 核心消融开关：测试时，你要测哪个，就改哪个！
    # ========================================================
    PRE_POSITIONING = False  # True: 测带预判的 20维模型; False: 测基线的 15维模型

    USE_GPU = True
    DT = 1.0
    MAX_STEP_PER_EPISODE = 4000
    ENV_TIME_LIMIT = 4000.0
    NUM_TEST_EPISODES = 100
    EVAL_GREEDY = False

    # 环境参数对齐
    PRIO_RANGE = (0.3, 1.0)
    PER_SPECIES_RANGE = (3, 5)
    SPECIES_RANGE = (2, 3)
    COMM_RANGE = 0.28
    DYNAMIC_TASK_DISCOVERY = True
    DISCOVERY_PROBABILITY = 0.07
    MAX_DYNAMIC_TASKS = 5
    DURATION_SCALE = 9

    SHAPING_CROWD_PENALTY = 0.15
    SHAPING_LONELY_BONUS = 0.10
    SHAPING_REDUNDANT_PENALTY = 0.30
    SHAPING_DISTANCE_PENALTY = 0.05

    WORKING_TIMEOUT_RANGE = (50.0, 120.0)
    WORKING_TIMEOUT_DECAY = 0.7
    WORKING_TIMEOUT_MIN = 15.0
    WORKING_TIMEOUT_PENALTY = -0.5
    COOLDOWN_DURATION = 10.0

    SCALABILITY_AGENTS_FIXED = {
        "agent_species_range": (2, 3),
        "agent_per_species_range": (3, 5),
        "task_ranges": [
            (10, 20), (20, 35), (35, 50), (50, 64), (65, 80)
        ]
    }

    OUTPUT_DIR = "./test_results_ablation"

    # ✅ 自动找对应的权重
    @classmethod
    def get_checkpoint_path(cls):
        suffix = "prepos" if cls.PRE_POSITIONING else "baseline"
        return f"./checkpoints_ablation/{suffix}/ctde_final.pth"

    # ✅ 自动分文件保存结果
    @classmethod
    def get_result_file_name(cls):
        suffix = "prepos" if cls.PRE_POSITIONING else "baseline"
        return f"test_results_{suffix}.json"


class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)

    def add_episode(self, env, episode_reward, wall_time, env_time, step_count):
        task_ids = list(env.task_dic.keys())
        finished_flags = [env.task_dic[tid]["finished"] for tid in task_ids]

        num_total_tasks = len(task_ids)
        num_finished_tasks = sum(finished_flags)
        task_completion = num_finished_tasks / num_total_tasks if num_total_tasks > 0 else 0.0
        success = 1.0 if task_completion >= 0.9999 else 0.0

        total_stages = sum(task.get("total_stages", 1) for task in env.task_dic.values())
        completed_stages = sum(task.get("current_stage_index", 0) for task in env.task_dic.values())
        stage_completion = completed_stages / total_stages if total_stages > 0 else 0.0

        # ✅ 新增：获取这段游戏的平均 Gap Time
        avg_gap = env.get_avg_stage_gap()

        self.metrics["num_agents"].append(len(env.agent_dic))
        self.metrics["num_tasks"].append(num_total_tasks)
        self.metrics["task_completion_rate"].append(task_completion)
        self.metrics["stage_completion_rate"].append(stage_completion)
        self.metrics["stage_gap_time"].append(avg_gap)  # 记录空窗期
        self.metrics["episode_success"].append(success)
        self.metrics["episode_reward"].append(episode_reward)
        self.metrics["env_time"].append(env_time)
        self.metrics["wall_time"].append(wall_time)
        self.metrics["step_count"].append(step_count)

    def get_summary(self):
        out = {}
        for k, v in self.metrics.items():
            out[f"{k}_mean"] = float(np.mean(v)) if v else 0.0
            out[f"{k}_std"] = float(np.std(v)) if v else 0.0
        out["success_rate"] = out.get("episode_success_mean", 0.0)
        out["success_count"] = int(np.sum(self.metrics["episode_success"])) if self.metrics["episode_success"] else 0
        out["total_episodes"] = len(self.metrics["episode_success"])
        return out


def build_env(task_range, config: TestConfig, seed=None):
    env = TaskEnv(
        prio_range=config.PRIO_RANGE,
        per_species_range=config.PER_SPECIES_RANGE,
        species_range=config.SPECIES_RANGE,
        tasks_range=task_range,
        duration_scale=config.DURATION_SCALE,
        comm_range=config.COMM_RANGE,
        dynamic_task_discovery=config.DYNAMIC_TASK_DISCOVERY,
        discovery_probability=config.DISCOVERY_PROBABILITY,
        max_dynamic_tasks=config.MAX_DYNAMIC_TASKS,
        dt=config.DT,
        seed=seed,
        shaping_crowd_penalty=config.SHAPING_CROWD_PENALTY,
        shaping_lonely_bonus=config.SHAPING_LONELY_BONUS,
        shaping_redundant_penalty=config.SHAPING_REDUNDANT_PENALTY,
        shaping_distance_penalty=config.SHAPING_DISTANCE_PENALTY,
        working_timeout_range=config.WORKING_TIMEOUT_RANGE,
        working_timeout_decay=config.WORKING_TIMEOUT_DECAY,
        working_timeout_min=config.WORKING_TIMEOUT_MIN,
        working_timeout_penalty=config.WORKING_TIMEOUT_PENALTY,
        cooldown_duration=config.COOLDOWN_DURATION,
        pre_positioning=config.PRE_POSITIONING,  # ✅ 环境同步切换
    )
    return env


def load_model(checkpoint_path, device, config):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ✅ 自动匹配网络维度
    t_dim = 20 if config.PRE_POSITIONING else 15
    gt_dim = 23 if config.PRE_POSITIONING else 18

    net = CTDE_Network(
        self_dim=10,
        neighbor_dim=10,
        task_dim=t_dim,
        global_agent_dim=13,
        global_task_dim=gt_dim,
        hidden=128,
        dropout=0.15,
        use_pos_encoding=True,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    print(f"✅ Loaded: {checkpoint_path}")
    print(f"   Architecture: task_dim={t_dim}, global_task_dim={gt_dim}")
    print(f"   episode: {ckpt.get('episode', 'N/A')}")
    if "best_metric" in ckpt:
        print(f"   best_metric: {ckpt['best_metric']:.4f}")
    return net


@torch.no_grad()
def run_episode(env, net, device, config: TestConfig, seed=None):
    env.reset(seed=seed)

    step_count = 0
    t0 = time.time()

    while not env.finished and step_count < config.MAX_STEP_PER_EPISODE:
        if env.current_time > config.ENV_TIME_LIMIT:
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

            tt, _ = pad_to_batch_max(batch_t)
            tt = tt.to(device)

            tm_exp = [m.unsqueeze(-1).float() for m in batch_m]
            tm, _ = pad_to_batch_max(tm_exp)
            tm = tm.squeeze(-1).bool().to(device)

            dist = net.get_action(ts, tn, tt, tm)

            if config.EVAL_GREEDY:
                actions = torch.argmax(dist.logits, dim=-1)
            else:
                actions = dist.sample()

            for i, aid in enumerate(decision_ids):
                env.apply_action(aid, int(actions[i].item()), action_maps[i])

        env.step(dt=config.DT)
        if step_count % 10 == 0:
            env.cleanup_failed_tasks_memory()
        step_count += 1

    wall_time = time.time() - t0
    episode_reward, _ = env.get_episode_reward()
    return episode_reward, wall_time, env.current_time, step_count


def test_scalability_fixed_agents(net, device, config: TestConfig):
    print("\n=== Test: Fixed Agents, Increasing Tasks (Ablation) ===")
    results = []

    for task_range in config.SCALABILITY_AGENTS_FIXED["task_ranges"]:
        print(f"\nTask range: {task_range}")
        collector = MetricsCollector()

        for ep in range(config.NUM_TEST_EPISODES):
            env = build_env(task_range=task_range, config=config, seed=ep)
            r, wt, et, steps = run_episode(env, net, device, config, seed=ep)
            collector.add_episode(env, r, wt, et, steps)

            if (ep + 1) % 10 == 0:
                sc = collector.metrics['stage_completion_rate'][-1]
                gap = collector.metrics['stage_gap_time'][-1]
                print(f"  Ep {ep + 1}/{config.NUM_TEST_EPISODES} | R={r:.1f} | "
                      f"StageComp={sc:.1%} | Gap={gap:.1f}s | env_t={et:.0f}")

        s = collector.get_summary()
        s["task_range"] = list(task_range)
        results.append(s)
        print(f"  Summary -> Succ={s['success_rate']:.1%}, "
              f"Gap={s['stage_gap_time_mean']:.2f}s, "
              f"Time={s['env_time_mean']:.1f}")

    return results


def main():
    config = TestConfig()

    mode_str = "PROPOSED (预判开启, 20/23维)" if config.PRE_POSITIONING else "BASELINE (预判关闭, 15/18维)"
    print("=" * 60)
    print(f"🚀 CTDE Ablation Test 🚀")
    print(f"测试模式: {mode_str}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 获取路径
    ckpt_path = config.get_checkpoint_path()
    result_file = config.get_result_file_name()

    net = load_model(ckpt_path, device, config)

    results = test_scalability_fixed_agents(net, device, config)

    output_path = os.path.join(config.OUTPUT_DIR, result_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()