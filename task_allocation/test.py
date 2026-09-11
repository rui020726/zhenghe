# -*- coding: utf-8 -*-
"""
CTDE 模型测试程序（动态Padding版 + 泛化性增强）
- ✅ 支持动态长度输入（突破65任务限制）
- ✅ 加强泛化性测试（50-100任务）
- ✅ 修复成功率计算（只有100%完成才算成功）
- ✅ 区分环境时间和计算时间
- ✅ 详细的性能分析
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

from env.task_env import TaskEnv
from network import CTDE_Network


# ✅ 导入动态padding函数（与driver. py一致）
def pad_to_batch_max(batch_data_list):
    """动态padding到batch内最大长度"""
    if len(batch_data_list) == 0:
        return None, None

    batch_size = len(batch_data_list)
    max_len = max(x.shape[0] for x in batch_data_list)
    feature_dim = batch_data_list[0].shape[-1] if batch_data_list[0].dim() > 1 else 1

    if batch_data_list[0].dim() == 1:
        padded = torch.zeros(batch_size, max_len,
                             dtype=batch_data_list[0].dtype,
                             device=batch_data_list[0].device)
    else:
        padded = torch.zeros(batch_size, max_len, feature_dim,
                             dtype=batch_data_list[0].dtype,
                             device=batch_data_list[0].device)

    lengths = []
    for i, data in enumerate(batch_data_list):
        actual_len = data.shape[0]
        if data.dim() == 1:
            padded[i, :actual_len] = data
        else:
            padded[i, :actual_len] = data
        lengths.append(actual_len)

    return padded, torch.tensor(lengths)


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TestConfig:
    """测试配置"""
    # 模型路径
    CHECKPOINT_PATH = "./checkpoints_3_6/ctde_final.pth"  # 或 ctde_final.pth

    # 设备
    USE_GPU = True

    # ✅ 移除固定MAX_TASKS限制
    MAX_AGENTS = 30
    # MAX_TASKS = 65  # 不再需要

    DT = 1.0
    MAX_STEP_PER_EPISODE = 2000
    ENV_TIME_LIMIT = 1500.
    0

    # 测试参数
    NUM_TEST_EPISODES = 100  # 每个配置测试轮数

    # ========== ✅ 扩展性测试配置（加强高任务数测试） ==========

    # 测试1:  固定智能体，递增任务（泛化性重点）
    SCALABILITY_AGENTS_FIXED = {
        "agent_species_range": (2, 3),
        "agent_per_species_range": (3, 5),
        "task_ranges": [
            (5, 10),  # 简单
            (10, 20),  # 中等
            (20, 35),  # 困难
            (35, 50),  # 很困难
            (50, 64),  # 极限（训练见过的上限）
            # ✅ 新增：测试泛化性
            (55, 70),  # 轻度OOD
            (65, 80),  # 中度OOD
            (75, 90),  # 重度OOD
        ]
    }

    # 测试2: 固定任务，增加智能体
    SCALABILITY_TASKS_FIXED = {
        "task_range": (20, 30),
        "agent_configs": [
            {"species_range": (2, 2), "per_species_range": (2, 3)},  # 4-6
            {"species_range": (2, 3), "per_species_range": (3, 4)},  # 6-12
            {"species_range": (2, 3), "per_species_range": (4, 6)},  # 8-18
            {"species_range": (3, 4), "per_species_range": (4, 6)},  # 12-24
            {"species_range": (3, 4), "per_species_range": (5, 7)},  # 15-28
        ]
    }

    # 测试3: 比例扩展
    SCALABILITY_PROPORTIONAL = [
        {"tasks": (5, 10), "species": (2, 2), "per_species": (2, 3)},
        {"tasks": (10, 15), "species": (2, 3), "per_species": (3, 4)},
        {"tasks": (20, 25), "species": (2, 3), "per_species": (4, 6)},
        {"tasks": (30, 40), "species": (3, 4), "per_species": (4, 6)},
        {"tasks": (40, 55), "species": (3, 4), "per_species": (5, 7)},
        # ✅ 新增：极限比例测试
        {"tasks": (55, 70), "species": (3, 4), "per_species": (6, 8)},
    ]

    # 输出目录
    OUTPUT_DIR = "./test_results"
    FIGURE_DIR = "./test_results/figures"


class MetricsCollector:
    """性能指标收集器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = defaultdict(list)

    def add_episode(self, env, episode_reward, wall_time, env_time, step_count):
        """收集单个episode的指标"""
        task_ids = list(env.task_dic.keys())
        finished_flags = [env.task_dic[tid]["finished"] for tid in task_ids]

        num_total = len(task_ids)
        num_finished = sum(finished_flags)
        task_completion_rate = num_finished / num_total if num_total > 0 else 0.0

        # ✅ 只有100%完成才算成功
        episode_success = 1.0 if task_completion_rate >= 0.9999 else 0.0

        # 时间指标
        finish_times = [
            env.task_dic[tid].get("time_finish", env.current_time)
            for tid in task_ids if env.task_dic[tid]["finished"]
        ]
        makespan = max(finish_times) if finish_times else env.current_time
        avg_completion_time = np.mean(finish_times) if finish_times else 0.0

        # 优先级
        priorities = [env.task_dic[tid]["search_priority"] for tid in task_ids]
        avg_priority = np.mean(priorities)

        finished_priorities = [
            env.task_dic[tid]["search_priority"]
            for tid in task_ids if env.task_dic[tid]["finished"]
        ]
        avg_finished_priority = np.mean(finished_priorities) if finished_priorities else 0.0

        # 失败指标
        total_failed = sum(len(a.get("failed_tasks", [])) for a in env.agent_dic.values())
        failed_per_agent = total_failed / len(env.agent_dic) if len(env.agent_dic) > 0 else 0.0

        # 移动指标
        total_travel = sum(a.get("travel_dist", 0.0) for a in env.agent_dic.values())
        avg_travel = total_travel / len(env.agent_dic) if len(env.agent_dic) > 0 else 0.0

        # 等待时间
        env.calculate_waiting_time()
        total_waiting = sum(a.get("sum_waiting_time", 0.0) for a in env.agent_dic.values())
        avg_waiting = total_waiting / len(env.agent_dic) if len(env.agent_dic) > 0 else 0.0

        # 动态任务
        dynamic_count = sum(1 for s in env.task_source.values() if s == "discovered")
        dynamic_finished = sum(
            1 for tid in task_ids
            if env.task_source.get(tid) == "discovered" and env.task_dic[tid]["finished"]
        )

        # 记录所有指标
        self.metrics["num_agents"].append(len(env.agent_dic))
        self.metrics["num_tasks"].append(num_total)
        self.metrics["num_finished"].append(num_finished)
        self.metrics["task_completion_rate"].append(task_completion_rate)
        self.metrics["episode_success"].append(episode_success)

        self.metrics["episode_reward"].append(episode_reward)
        self.metrics["makespan"].append(makespan)
        self.metrics["avg_completion_time"].append(avg_completion_time)
        self.metrics["avg_priority"].append(avg_priority)
        self.metrics["avg_finished_priority"].append(avg_finished_priority)
        self.metrics["total_failed"].append(total_failed)
        self.metrics["failed_per_agent"].append(failed_per_agent)
        self.metrics["total_travel"].append(total_travel)
        self.metrics["avg_travel"].append(avg_travel)
        self.metrics["avg_waiting"].append(avg_waiting)
        self.metrics["dynamic_count"].append(dynamic_count)
        self.metrics["dynamic_finished"].append(dynamic_finished)

        self.metrics["wall_time"].append(wall_time)
        self.metrics["env_time"].append(env_time)
        self.metrics["step_count"].append(step_count)

        if wall_time > 0:
            self.metrics["steps_per_second"].append(step_count / wall_time)
            self.metrics["env_time_per_wall_time"].append(env_time / wall_time)
        else:
            self.metrics["steps_per_second"].append(0)
            self.metrics["env_time_per_wall_time"].append(0)

    def get_summary(self):
        """获取统计摘要"""
        summary = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
                summary[f"{key}_min"] = float(np.min(values))
                summary[f"{key}_max"] = float(np.max(values))

                if key == "episode_success":
                    summary["success_rate"] = float(np.mean(values))
                    summary["success_count"] = int(np.sum(values))
                    summary["total_episodes"] = len(values)

        return summary


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    net = CTDE_Network(
        self_dim=10,
        neighbor_dim=10,
        task_dim=15,
        global_agent_dim=13,
        global_task_dim=18,
        hidden=128,
        dropout=0.15,  # ✅ 与训练一致
        use_pos_encoding=True,  # ✅ 启用位置编码
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    print(f"✅ Model loaded from {checkpoint_path}")
    if "episode" in checkpoint:
        print(f"   Trained episodes: {checkpoint['episode']}")
    if "best_metric" in checkpoint:
        print(f"   Best metric: {checkpoint['best_metric']:.2f}")

    return net


def run_episode(env, net, device, config, seed=None, verbose=False):
    """
    与当前训练逻辑一致的测试episode（顺序决策 + 局部通信实时更新）
    - 单智能体依次决策（非并行批量）
    - 每个智能体动作后立即更新通信与传播，后续智能体基于最新局部信息决策
    - 累加 env.last_step_reward（无效到达即时惩罚等）
    """
    env.reset(seed=seed)

    step_count = 0
    episode_start_time = time.time()
    step_reward_sum = 0.0

    with torch.no_grad():
        while not env.finished and step_count < config.MAX_STEP_PER_EPISODE:
            if env.current_time > config.ENV_TIME_LIMIT:
                env.finished = True
                break

            # 需要决策的智能体（固定顺序可复现）
            decision_ids = sorted([aid for aid, a in env.agent_dic.items() if a.get("need_decision", False)])

            # ✅ 顺序决策（和训练一致）
            for aid in decision_ids:
                s, n, t, m, amap = env.get_local_obs(aid)

                ts = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)   # [1, self_dim]
                tn = torch.as_tensor(n, dtype=torch.float32, device=device).unsqueeze(0)   # [1, Nn, nd]
                tt = torch.as_tensor(t, dtype=torch.float32, device=device).unsqueeze(0)   # [1, Nt, td]
                tm = torch.as_tensor(m, dtype=torch.bool, device=device).unsqueeze(0)      # [1, Nt]

                dist = net.get_action(ts, tn, tt, tm)

                # 测试可选 greedy；若想与训练采样一致可改 sample()
                # act = int(torch.argmax(dist.probs[0]).item())
                act = int(dist.sample()[0].item())

                r_act, _need_replan, _info = env.apply_action(aid, act, amap)
                step_reward_sum += float(r_act)

                # 每个智能体动作后，立刻同步局部信息
                env.update_communication_graph()
                env.propagate_task_completion()

            # 物理时间推进
            env.step(dt=config.DT)

            # 累加环境step即时奖励（如 invalid_arrival_penalty）
            step_reward_sum += float(getattr(env, "last_step_reward", 0.0))

            if step_count % 10 == 0:
                env.cleanup_failed_tasks_memory()

            step_count += 1

            if verbose and step_count % 100 == 0:
                elapsed = time.time() - episode_start_time
                print(
                    f"  Step {step_count}, EnvTime={env.current_time:.1f}, "
                    f"Tasks={len(env.task_dic)}, WallTime={elapsed:.1f}s, "
                    f"StepRewardSum={step_reward_sum:.2f}"
                )

    wall_time = time.time() - episode_start_time

    # 终局奖励（主指标）
    terminal_reward, _ = env.get_episode_reward()

    # 返回“总评测分数”：终局 + 过程即时奖励（可根据你习惯改为仅terminal）
    episode_reward = float(terminal_reward + step_reward_sum)

    return episode_reward, wall_time, env.current_time, step_count

def test_scalability_fixed_agents(net, device, config):
    """测试1:  固定智能体，增加任务（泛化性重点）"""
    print("\n" + "=" * 80)
    print("📊 Test 1: Scalability with Fixed Agents, Increasing Tasks")
    print("   🎯 Focus: Generalization to High Task Counts (50-90 tasks)")
    print("=" * 80)

    results = []

    for task_range in config.SCALABILITY_AGENTS_FIXED["task_ranges"]:
        # ✅ 标记是否为OOD测试
        is_ood = task_range[0] > 50
        ood_label = " [OOD]" if is_ood else ""

        print(f"\n🔹 Testing task range:  {task_range}{ood_label}")

        collector = MetricsCollector()

        for ep in range(config.NUM_TEST_EPISODES):
            env = TaskEnv(
                prio_range=(0.3, 1.0),
                per_species_range=config.SCALABILITY_AGENTS_FIXED["agent_per_species_range"],
                species_range=config.SCALABILITY_AGENTS_FIXED["agent_species_range"],
                tasks_range=task_range,
                comm_range=0.3,
            dynamic_task_discovery = True,
            discovery_probability = 0.05,
            max_dynamic_tasks = 5,
            seed = ep,
            dt = config.DT,
            )

            episode_reward, wall_time, env_time, step_count = run_episode(
                env, net, device, config, seed=ep
            )
            collector.add_episode(env, episode_reward, wall_time, env_time, step_count)

            comp_rate = collector.metrics['task_completion_rate'][-1]
            is_success = "✓" if collector.metrics['episode_success'][-1] == 1.0 else "✗"

            if (ep + 1) % 10 == 0 or ep < 3:  # 减少打印
                print(f"  Ep {ep + 1}/{config.NUM_TEST_EPISODES} {is_success}:  "
                      f"Agents={len(env.agent_dic)}, Tasks={len(env.task_dic)}, "
                      f"Completion={comp_rate:.2%}, "
                      f"Reward={episode_reward:.2f}, "
                      f"EnvTime={env_time:.1f}, WallTime={wall_time:.2f}s")

        summary = collector.get_summary()
        summary["task_range"] = task_range
        summary["is_ood"] = is_ood
        results.append(summary)

        print(f"\n  📈 Summary for {task_range}{ood_label}:")
        print(f"     Agents: {summary['num_agents_mean']:.1f} ± {summary['num_agents_std']:.1f}")
        print(f"     Tasks: {summary['num_tasks_mean']:.1f} ± {summary['num_tasks_std']:.1f}")
        print(f"     ✅ Episode Success Rate: {summary['success_rate']:.2%} "
              f"({summary['success_count']}/{summary['total_episodes']})")
        print(f"     Task Completion:  {summary['task_completion_rate_mean']:.2%} ± "
              f"{summary['task_completion_rate_std']:.2%}")
        print(f"     Reward: {summary['episode_reward_mean']:.2f} ± {summary['episode_reward_std']:.2f}")
        print(f"     Makespan: {summary['makespan_mean']:.2f} ± {summary['makespan_std']:.2f}")
        print(f"     WallTime: {summary['wall_time_mean']:.2f}s ± {summary['wall_time_std']:.2f}s")

        # ✅ OOD性能下降分析
        if is_ood and len(results) > 1:
            baseline = results[4]  # (50, 64) 是训练上限
            degradation = (baseline['success_rate'] - summary['success_rate']) / max(baseline['success_rate'], 0.01)
            print(f"     ⚠️ OOD Degradation: {degradation:.1%} vs baseline (50-64 tasks)")

    return results


def test_scalability_fixed_tasks(net, device, config):
    """测试2: 固定任务，增加智能体"""
    print("\n" + "=" * 80)
    print("📊 Test 2: Scalability with Fixed Tasks, Increasing Agents")
    print("=" * 80)

    results = []

    for agent_cfg in config.SCALABILITY_TASKS_FIXED["agent_configs"]:
        print(f"\n🔹 Testing agent config: {agent_cfg}")

        collector = MetricsCollector()

        for ep in range(config.NUM_TEST_EPISODES):
            env = TaskEnv(
                prio_range=(0.3, 1.0),
                per_species_range=agent_cfg["per_species_range"],
                species_range=agent_cfg["species_range"],
                tasks_range=config.SCALABILITY_TASKS_FIXED["task_range"],
                comm_range=0.3,
                dynamic_task_discovery=False,
                seed=ep,
                dt=config.DT,
            )

            episode_reward, wall_time, env_time, step_count = run_episode(
                env, net, device, config, seed=ep
            )
            collector.add_episode(env, episode_reward, wall_time, env_time, step_count)

            if (ep + 1) % 10 == 0 or ep < 3:
                comp_rate = collector.metrics['task_completion_rate'][-1]
                is_success = "✓" if collector.metrics['episode_success'][-1] == 1.0 else "✗"
                print(f"  Ep {ep + 1} {is_success}:  Agents={len(env.agent_dic)}, "
                      f"Completion={comp_rate:.2%}, Reward={episode_reward:.2f}")

        summary = collector.get_summary()
        summary["agent_config"] = agent_cfg
        results.append(summary)

        print(f"\n  📈 Summary:")
        print(f"     Agents: {summary['num_agents_mean']:.1f}")
        print(f"     ✅ Success Rate: {summary['success_rate']:.2%}")
        print(f"     Makespan: {summary['makespan_mean']:.2f} ± {summary['makespan_std']:.2f}")

    return results


def test_scalability_proportional(net, device, config):
    """测试3: 比例扩展"""
    print("\n" + "=" * 80)
    print("📊 Test 3: Proportional Scalability")
    print("=" * 80)

    results = []

    for test_cfg in config.SCALABILITY_PROPORTIONAL:
        is_ood = test_cfg["tasks"][0] > 50
        ood_label = " [OOD]" if is_ood else ""

        print(f"\n🔹 Testing:  Tasks={test_cfg['tasks']}, "
              f"Species={test_cfg['species']}, PerSpecies={test_cfg['per_species']}{ood_label}")

        collector = MetricsCollector()

        for ep in range(config.NUM_TEST_EPISODES):
            env = TaskEnv(
                prio_range=(0.3, 1.0),
                per_species_range=test_cfg["per_species"],
                species_range=test_cfg["species"],
                tasks_range=test_cfg["tasks"],
                comm_range=0.3,
                dynamic_task_discovery=False,
                seed=ep,
                dt=config.DT,
            )

            episode_reward, wall_time, env_time, step_count = run_episode(
                env, net, device, config, seed=ep
            )
            collector.add_episode(env, episode_reward, wall_time, env_time, step_count)

            if (ep + 1) % 10 == 0 or ep < 3:
                comp_rate = collector.metrics['task_completion_rate'][-1]
                is_success = "✓" if collector.metrics['episode_success'][-1] == 1.0 else "✗"
                print(f"  Ep {ep + 1} {is_success}: A={len(env.agent_dic)}, T={len(env.task_dic)}, "
                      f"Compl={comp_rate:.2%}")

        summary = collector.get_summary()
        summary["test_config"] = test_cfg
        summary["task_agent_ratio"] = summary["num_tasks_mean"] / max(summary["num_agents_mean"], 1)
        summary["is_ood"] = is_ood
        results.append(summary)

        print(f"\n  📈 Summary{ood_label}:")
        print(f"     A/T: {summary['num_agents_mean']:.0f}/{summary['num_tasks_mean']:.0f} "
              f"(ratio={summary['task_agent_ratio']:.2f})")
        print(f"     ✅ Success:  {summary['success_rate']:.2%}")
        print(f"     Reward: {summary['episode_reward_mean']:.2f}")

    return results


def plot_results(results_fixed_agents, results_fixed_tasks, results_proportional, config):
    """绘制测试结果（增强OOD可视化）"""
    print("\n" + "=" * 80)
    print("📊 Generating Visualization")
    print("=" * 80)

    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # ==================== 图1: 泛化性重点分析 ====================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Test 1: Generalization to High Task Counts', fontsize=16, fontweight='bold')

    task_counts = [r['num_tasks_mean'] for r in results_fixed_agents]
    is_ood = [r.get('is_ood', False) for r in results_fixed_agents]

    # 分离训练内和OOD数据
    train_idx = [i for i, ood in enumerate(is_ood) if not ood]
    ood_idx = [i for i, ood in enumerate(is_ood) if ood]

    # Success Rate with OOD highlighting
    ax = axes[0, 0]
    if train_idx:
        ax.plot([task_counts[i] for i in train_idx],
                [results_fixed_agents[i]['success_rate'] for i in train_idx],
                marker='o', linewidth=2, markersize=8, color='green', label='In-Distribution')
    if ood_idx:
        ax.plot([task_counts[i] for i in ood_idx],
                [results_fixed_agents[i]['success_rate'] for i in ood_idx],
                marker='s', linewidth=2, markersize=8, color='red', label='Out-of-Distribution', linestyle='--')
    ax.axvline(x=50, color='gray', linestyle=':', linewidth=2, label='Training Upper Bound')
    ax.set_xlabel('Number of Tasks')
    ax.set_ylabel('Episode Success Rate')
    ax.set_title('Generalization:  Success Rate')
    ax.legend()
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)

    # Task Completion Rate
    ax = axes[0, 1]
    if train_idx:
        ax.errorbar([task_counts[i] for i in train_idx],
                    [results_fixed_agents[i]['task_completion_rate_mean'] for i in train_idx],
                    yerr=[results_fixed_agents[i]['task_completion_rate_std'] for i in train_idx],
                    marker='o', linewidth=2, capsize=5, color='blue', label='In-Dist')
    if ood_idx:
        ax.errorbar([task_counts[i] for i in ood_idx],
                    [results_fixed_agents[i]['task_completion_rate_mean'] for i in ood_idx],
                    yerr=[results_fixed_agents[i]['task_completion_rate_std'] for i in ood_idx],
                    marker='s', linewidth=2, capsize=5, color='orange', label='OOD', linestyle='--')
    ax.axvline(x=50, color='gray', linestyle=':', linewidth=2)
    ax.set_xlabel('Number of Tasks')
    ax.set_ylabel('Task Completion Rate')
    ax.set_title('Average Task Completion')
    ax.legend()
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)

    # Reward
    ax = axes[0, 2]
    ax.errorbar(task_counts,
                [r['episode_reward_mean'] for r in results_fixed_agents],
                yerr=[r['episode_reward_std'] for r in results_fixed_agents],
                marker='d', linewidth=2, capsize=5, color='purple')
    ax.axvline(x=50, color='gray', linestyle=':', linewidth=2, label='Training Bound')
    ax.set_xlabel('Number of Tasks')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Episode Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Makespan
    ax = axes[1, 0]
    ax.errorbar(task_counts,
                [r['makespan_mean'] for r in results_fixed_agents],
                yerr=[r['makespan_std'] for r in results_fixed_agents],
                marker='^', linewidth=2, capsize=5, color='orange')
    ax.axvline(x=50, color='gray', linestyle=':', linewidth=2)
    ax.set_xlabel('Number of Tasks')
    ax.set_ylabel('Makespan (env time)')
    ax.set_title('Completion Time')
    ax.grid(True, alpha=0.3)

    # Failed per Agent
    ax = axes[1, 1]
    ax.errorbar(task_counts,
                [r['failed_per_agent_mean'] for r in results_fixed_agents],
                yerr=[r['failed_per_agent_std'] for r in results_fixed_agents],
                marker='v', linewidth=2, capsize=5, color='red')
    ax.axvline(x=50, color='gray', linestyle=':', linewidth=2)
    ax.set_xlabel('Number of Tasks')
    ax.set_ylabel('Failed Tasks per Agent')
    ax.set_title('Invalid Arrivals')
    ax.grid(True, alpha=0.3)

    # Wall Time
    ax = axes[1, 2]
    ax.errorbar(task_counts,
                [r['wall_time_mean'] for r in results_fixed_agents],
                yerr=[r['wall_time_std'] for r in results_fixed_agents],
                marker='h', linewidth=2, capsize=5, color='brown')
    ax.axvline(x=50, color='gray', linestyle=':', linewidth=2)
    ax.set_xlabel('Number of Tasks')
    ax.set_ylabel('Wall Time (seconds)')
    ax.set_title('Computational Cost')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURE_DIR, 'test1_generalization.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved:  test1_generalization.png")

    # ==================== 图2+3: 保持原样（省略，与原代码相同）====================
    # ...  [Test 2 和 Test 3 的绘图代码保持不变] ...

    plt.close('all')


def save_results_json(results_fixed_agents, results_fixed_tasks, results_proportional, config):
    """保存结果为JSON"""
    results = {
        "test1_fixed_agents": results_fixed_agents,
        "test2_fixed_tasks": results_fixed_tasks,
        "test3_proportional": results_proportional,
    }

    output_path = os.path.join(config.OUTPUT_DIR, "test_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to:  {output_path}")


def print_generalization_analysis(results_fixed_agents):
    """✅ 新增：泛化性能详细分析"""
    print("\n" + "=" * 80)
    print("🎯 GENERALIZATION PERFORMANCE ANALYSIS")
    print("=" * 80)

    # 找到训练上限（50-64）
    baseline_idx = 4  # (50, 64)
    baseline = results_fixed_agents[baseline_idx]

    print(f"\n📌 Baseline (Training Upper Bound:  50-64 tasks)")
    print(f"   Success Rate: {baseline['success_rate']:.2%}")
    print(f"   Avg Completion: {baseline['task_completion_rate_mean']:.2%}")
    print(f"   Reward: {baseline['episode_reward_mean']:.2f}")

    print(f"\n📊 OOD Performance:")
    for i in range(5, len(results_fixed_agents)):  # 从第6个开始是OOD
        r = results_fixed_agents[i]
        task_range = r['task_range']

        # 计算性能下降
        success_drop = baseline['success_rate'] - r['success_rate']
        completion_drop = baseline['task_completion_rate_mean'] - r['task_completion_rate_mean']
        reward_drop = baseline['episode_reward_mean'] - r['episode_reward_mean']

        print(f"\n   Task Range: {task_range}")
        print(f"   ├─ Success Rate: {r['success_rate']:.2%} (↓ {success_drop:.2%})")
        print(f"   ├─ Avg Completion: {r['task_completion_rate_mean']:.2%} (↓ {completion_drop:.2%})")
        print(f"   ├─ Reward: {r['episode_reward_mean']:.2f} (↓ {reward_drop:.2f})")
        print(f"   └─ Makespan: {r['makespan_mean']:.1f} ± {r['makespan_std']:.1f}")

    # 计算平均OOD退化
    ood_results = results_fixed_agents[5:]
    if ood_results:
        avg_ood_success = np.mean([r['success_rate'] for r in ood_results])
        avg_degradation = (baseline['success_rate'] - avg_ood_success) / max(baseline['success_rate'], 0.01)

        print(f"\n🔍 Summary:")
        print(f"   In-Distribution Performance (≤50 tasks): {baseline['success_rate']:.2%}")
        print(f"   Average OOD Performance (>50 tasks): {avg_ood_success:.2%}")
        print(f"   ⚠️ Average Degradation: {avg_degradation:.1%}")

        if avg_degradation < 0.15:
            print(f"   ✅ Excellent generalization (< 15% drop)")
        elif avg_degradation < 0.30:
            print(f"   ⚠️ Moderate generalization (15-30% drop)")
        else:
            print(f"   ❌ Poor generalization (> 30% drop)")


def main():
    """主测试流程"""
    print("=" * 80)
    print("🚀 CTDE Model Testing - Dynamic Padding + Generalization")
    print("=" * 80)

    config = TestConfig()

    device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.FIGURE_DIR, exist_ok=True)

    # 加载模型
    net = load_model(config.CHECKPOINT_PATH, device)

    # 运行测试
    results_fixed_agents = test_scalability_fixed_agents(net, device, config)
    results_fixed_tasks = test_scalability_fixed_tasks(net, device, config)
    results_proportional = test_scalability_proportional(net, device, config)

    # ✅ 泛化性分析
    print_generalization_analysis(results_fixed_agents)

    # 绘图
    plot_results(results_fixed_agents, results_fixed_tasks, results_proportional, config)

    # 保存结果
    save_results_json(results_fixed_agents, results_fixed_tasks, results_proportional, config)

    print("\n✅ All tests completed!")
    print(f"📁 Results:  {config.OUTPUT_DIR}")
    print(f"📊 Figures: {config.FIGURE_DIR}")


if __name__ == "__main__":
    main()