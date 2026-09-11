import time
import numpy as np
import torch
from torch.distributions import Categorical
from src.DDQN.Agent import MAPPO
from src.Map.timemap import Environment
from src.DDQN.rescue import RescueAlgorithm
import configs
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image
import re
import pandas as pd
import json

# === 新增：复访率与覆盖指标采集 ===
from src.metrics.revisit_metrics import RevisitCoverageMeter


class CoverageTracker:
    """包装 RevisitCoverageMeter 以追踪覆盖率（只用于生成单回合的 coverage history）"""

    def __init__(self, meter):
        self.meter = meter
        self.coverage_history = []

    def update(self, agents_pos, env):
        self.meter.update(agents_pos, env)
        free_cells = np.sum(self.meter.free_mask)
        if hasattr(self.meter, 'visited'):
            observed_cells = np.sum(self.meter.visited)
        else:
            observed_cells = 0
        coverage_rate = (observed_cells / free_cells) if free_cells > 0 else 0.0
        self.coverage_history.append(coverage_rate)

    def reset(self):
        self.coverage_history = []

    # 不再保存每回合图表的方法，节省时间。
    def save_plots(self, *args, **kwargs):
        return None

    def save_summary_csv(self, *args, **kwargs):
        return None


# 通信配置
COMMUNICATION_CONDITIONS = {
    'perfect': {
        'name': 'Perfect Communication',
        'comm_failure_prob': 0.0,
        'comm_partial_loss_prob': 0.0,
        'comm_delay_prob': 0.0,
        'enable_comm_noise': False
    },
    'good': {
        'name': 'Good Communication',
        'comm_failure_prob': 0.02,
        'comm_partial_loss_prob': 0.05,
        'comm_delay_prob': 0.03,
        'enable_comm_noise': True
    },
    'medium': {
        'name': 'Medium Communication',
        'comm_failure_prob': 0.05,
        'comm_partial_loss_prob': 0.10,
        'comm_delay_prob': 0.05,
        'enable_comm_noise': True
    },
    'poor': {
        'name': 'Poor Communication',
        'comm_failure_prob': 0.10,
        'comm_partial_loss_prob': 0.20,
        'comm_delay_prob': 0.10,
        'enable_comm_noise': True
    },
    'very_poor': {
        'name': 'Very Poor Communication',
        'comm_failure_prob': 0.20,
        'comm_partial_loss_prob': 0.30,
        'comm_delay_prob': 0.15,
        'enable_comm_noise': True
    }
}


def _is_line_of_sight_blocked(x1, y1, x2, y2, env_map):
    points = _bresenham_line(x1, y1, x2, y2)
    for px, py in points:
        if env_map[px, py] == 1 and (px != x2 or py != y2):
            return False
    return True


def _bresenham_line(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points


def initialize_obs(env, num_agents=configs.num_agents, map_size=(60, 60)):
    obs = {}
    for i in range(num_agents):
        agent_obs = np.full(map_size, 0.5, dtype=np.float32)
        x, y = env.agents_pos[i]
        agent_obs[x, y] = -4.5
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                nx, ny = x + dx, y + dy
                if _is_line_of_sight_blocked(x, y, nx, ny, env.map):
                    if 0 <= nx < env.map_size[0] and 0 <= ny < env.map_size[1]:
                        if env.map[nx, ny] == 1:
                            agent_obs[nx, ny] = 1
                        elif [nx, ny] != [x, y] and env.map[nx, ny] != 1 and agent_obs[nx, ny] not in [-4.5, -5.5, -6.5]:
                            agent_obs[nx, ny] = 0
        obs[f'agent_{i}'] = agent_obs
    return obs


def get_local_ph(global_ph, x, y, view_radius=5):
    h, w = global_ph.shape
    x_min = max(0, x - view_radius)
    x_max = min(h, x + view_radius + 1)
    y_min = max(0, y - view_radius)
    y_max = min(w, y + view_radius + 1)
    local_ph = global_ph[x_min:x_max, y_min:y_max]
    target_size = 2 * view_radius + 1
    pad_top = max(0, target_size - (x_max - x_min)) if x - view_radius < 0 else 0
    pad_bottom = max(0, target_size - (x_max - x_min)) if x + view_radius + 1 > h else 0
    pad_left = max(0, target_size - (y_max - y_min)) if y - view_radius < 0 else 0
    pad_right = max(0, target_size - (y_max - y_min)) if y + view_radius + 1 > w else 0
    return np.pad(local_ph, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')


def select_action(two_channelmap, policy, env, env_agents, device):
    actions = {}
    action_log_pis = {}
    for agent_id in env_agents:
        obstacle_map = two_channelmap[agent_id]['obstacle_map']
        explore_map = two_channelmap[agent_id]['explore_map']
        agent_intid = int(agent_id.split("_")[-1])
        x, y = env.agents_pos[agent_intid]
        pheromone_map = get_local_ph(env.global_pheromone, x, y)
        pheromone_map_tensor = torch.tensor(pheromone_map, dtype=torch.float32).unsqueeze(0).to(device)
        obs = np.stack([obstacle_map, explore_map], axis=0)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action_probs = policy.agents.get_action(obs_tensor, pheromone_map_tensor, batch_size=1)
        action_probs = action_probs[1]
        dist = Categorical(probs=action_probs)
        action = dist.sample()
        action_log_pi = dist.log_prob(action)
        actions[agent_id] = action.item()
        action_log_pis[agent_id] = action_log_pi.item()
    return actions, action_log_pis


def plot_mean_std_curve(mean_curve, std_curve, save_dir, max_steps=250, title_suffix=""):
    steps = np.arange(1, max_steps + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, mean_curve, 'b-', linewidth=2, label='Mean Coverage')
    ax.fill_between(steps, np.clip(mean_curve - std_curve, 0.0, 1.0), np.clip(mean_curve + std_curve, 0.0, 1.0),
                    alpha=0.3, color='blue', label='±1 Std Dev')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Coverage Rate')
    ax.set_title(f'Average Coverage Rate {title_suffix}')
    ax.set_xlim(1, max_steps)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, 'average_coverage_curve.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    # save csv
    df = pd.DataFrame({'step': steps, 'mean_coverage': mean_curve, 'std_coverage': std_curve})
    csv_path = os.path.join(save_dir, 'average_coverage_data.csv')
    df.to_csv(csv_path, index=False)
    return fig_path, csv_path


def test_single_condition(condition_name, condition_params, model_dir, num_episodes=50, max_steps=250):
    """
    主要改动点：
    - 不再保存/记录每回合的 coverage_curve（节省IO和内存）
    - 使用逐步在线算法（Welford）统计每步的均值与方差（无需保存所有回合数据）
    - 仍保存每个 condition 的最终平均覆盖率曲线（mean ± std）到文件
    """
    print(f"\n=== Start test: {condition_params['name']} ===")
    device = torch.device(configs.DEVICE) if torch.cuda.is_available() else torch.device('cpu')
    trick = configs.TRICK

    condition_dir = os.path.join('metrics_out', condition_name)
    os.makedirs(condition_dir, exist_ok=True)

    # 统计变量
    success = 0
    success_steps = []
    steps_per_episode = []
    get_goals = []
    res_t = 0
    res_step = 0

    # Welford online accumulators for each timestep
    mean_acc = np.zeros(max_steps, dtype=np.float64)
    M2_acc = np.zeros(max_steps, dtype=np.float64)
    count = 0

    dim_info = {f'agent_{i}': [9, 5] for i in range(configs.num_agents)}
    policy = MAPPO.load(dim_info, False, model_dir, trick=trick)

    for ep in range(num_episodes):
        env = Environment(
            test=True,
            comm_failure_prob=condition_params['comm_failure_prob'],
            comm_partial_loss_prob=condition_params['comm_partial_loss_prob'],
            comm_delay_prob=condition_params['comm_delay_prob'],
            enable_comm_noise=condition_params['enable_comm_noise']
        )

        env_agents = [agent_id for agent_id in env.agents]
        obs = initialize_obs(env)
        two_channelmap = env.two_channel_map(obs, {})
        rescue_policy = RescueAlgorithm(env.map_size, env, obs)
        rescue_actions = {}

        # tracker 用于本回合在线计算 coverage_history（但不保存）
        base_meter = RevisitCoverageMeter(free_mask=(env.map == 0), radius=5, window=100, use_los=True)
        tracker = CoverageTracker(base_meter)
        tracker.reset()

        step_t = 0
        episode_reward = {agent_id: 0 for agent_id in env_agents}

        while step_t < max_steps:
            step_t += 1
            action, action_log_pi = select_action(two_channelmap, policy, env, env_agents, device)

            # apply rescue actions if any
            if len(rescue_actions) > 0:
                for agent_id, rescue_action_sequence in list(rescue_actions.items()):
                    if rescue_action_sequence:
                        action[agent_id] = rescue_action_sequence[0]
                        rescue_action_sequence.pop(0)
                        res_step += 1
                    if len(rescue_action_sequence) == 0:
                        del rescue_actions[agent_id]

            next_obs, reward, terminated, repeat_val, next_two_channelmap = env.step(
                action, obs, rescue_actions, two_channelmap, ep
            )

            # update coverage tracker (per-step), but we won't store the full dataset beyond this episode
            tracker.update(env.agents_pos, env)

            done = {agent_id: terminated[agent_id] for agent_id in env_agents}
            episode_reward = {agent_id: episode_reward[agent_id] + reward[agent_id] for agent_id in env_agents}
            obs = next_obs
            two_channelmap = next_two_channelmap

            if repeat_val and len(repeat_val) != 0:
                rescue_actions_fu = repeat_check(repeat_val, env, obs, rescue_policy)
                rescue_actions.update(rescue_actions_fu)
                res_t += 1

            if any(done.values()):
                success += 1
                success_steps.append(step_t)
                get_goals.append(6)
                break

        # 回合结束：记录步数和捕获目标数
        steps_per_episode.append(step_t)
        if not any(done.values()):
            num_goals = len(env.goals_pos)
            get_goals.append(6 - num_goals)

        # 使用本回合的 coverage_history 更新全局均值和方差（Welford）
        # 将本回合曲线 pad/truncate 到 max_steps
        ep_curve = np.array(tracker.coverage_history, dtype=np.float64)
        if ep_curve.size == 0:
            ep_curve = np.zeros(max_steps, dtype=np.float64)
        elif ep_curve.size < max_steps:
            pad_len = max_steps - ep_curve.size
            ep_curve = np.concatenate([ep_curve, np.full(pad_len, ep_curve[-1] if ep_curve.size > 0 else 0.0)])
        elif ep_curve.size > max_steps:
            ep_curve = ep_curve[:max_steps]

        # Welford update (vectorized)
        count += 1
        if count == 1:
            mean_acc[:] = ep_curve
            M2_acc[:] = 0.0
        else:
            delta = ep_curve - mean_acc
            mean_acc += delta / count
            delta2 = ep_curve - mean_acc
            M2_acc += delta * delta2

        # Print progress occasionally
        if (ep + 1) % 10 == 0:
            print(f"[{condition_name}] Episode {ep+1}/{num_episodes}  success_rate={success/(ep+1):.3f}")

    # 结束所有回合，计算最终统计量
    comm_stats = env.get_communication_stats()
    success_rate = success / num_episodes if num_episodes > 0 else 0.0
    steps_mean = float(np.mean(steps_per_episode)) if steps_per_episode else 0.0
    steps_var = float(np.var(steps_per_episode)) if steps_per_episode else 0.0
    avg_goals = float(np.mean(get_goals)) if get_goals else 0.0
    goals_var = float(np.var(get_goals)) if get_goals else 0.0

    # 计算 mean / std from Welford
    mean_curve = mean_acc
    if count > 1:
        var_curve = M2_acc / (count - 1)
        std_curve = np.sqrt(var_curve)
    else:
        std_curve = np.zeros_like(mean_curve)

    # 保存最终平均覆盖率曲线（mean ± std），不保存每回合曲线
    fig_path, csv_path = plot_mean_std_curve(mean_curve, std_curve, condition_dir, max_steps=max_steps,
                                             title_suffix=f"({condition_params['name']})")

    # 构建结果字典并保存 JSON（不包含每回合曲线）
    results = {
        'condition_name': condition_name,
        'condition_params': condition_params,
        'num_episodes': num_episodes,
        'success': success,
        'success_rate': success_rate,
        'steps_mean': steps_mean,
        'steps_var': steps_var,
        'avg_goals': avg_goals,
        'goals_var': goals_var,
        'rescue_triggers': res_t,
        'rescue_steps': res_step,
        'comm_stats': comm_stats,
        'success_steps': success_steps,
        'final_mean_curve_csv': csv_path,
        'final_mean_curve_png': fig_path
    }

    with open(os.path.join(condition_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Finished test for {condition_name}: success_rate={success_rate:.3f}, avg_steps={steps_mean:.2f}, avg_goals={avg_goals:.2f}")
    return results


def repeat_check(repeat_mark, env, obs, rescue_policy):
    all_paths = {}
    if len(repeat_mark) != 0:
        for agent_id in repeat_mark:
            agent_index = int(agent_id.split('_')[1])
            path = rescue_policy.rescue_action(agent_index, env, obs)
            all_paths[agent_id] = path
    return all_paths


def plot_communication_comparison(all_conditions_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    conditions = list(all_conditions_results.keys())
    colors = ['green', 'blue', 'orange', 'red', 'darkred']

    # 成功率
    fig, ax = plt.subplots(figsize=(10, 6))
    success_rates = [all_conditions_results[c]['success_rate'] for c in conditions]
    bars = ax.bar(range(len(conditions)), success_rates, color=colors[:len(conditions)], alpha=0.7)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([COMMUNICATION_CONDITIONS[c]['name'] for c in conditions], rotation=45, ha='right')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1.0)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{success_rates[i]:.3f}', ha='center', va='bottom')
    fig.savefig(os.path.join(output_dir, 'success_rate_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 平均步数和方差
    fig, ax = plt.subplots(figsize=(10, 6))
    steps_mean = [all_conditions_results[c]['steps_mean'] for c in conditions]
    steps_var = [all_conditions_results[c]['steps_var'] for c in conditions]
    bars = ax.bar(range(len(conditions)), steps_mean, color=colors[:len(conditions)], alpha=0.7)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([COMMUNICATION_CONDITIONS[c]['name'] for c in conditions], rotation=45, ha='right')
    ax.set_ylabel('Average Steps')
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{steps_mean[i]:.1f}\n(var={steps_var[i]:.1f})',
                ha='center', va='bottom', fontsize=9)
    fig.savefig(os.path.join(output_dir, 'steps_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 平均捕获目标数与方差
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_goals = [all_conditions_results[c]['avg_goals'] for c in conditions]
    goals_var = [all_conditions_results[c]['goals_var'] for c in conditions]
    bars = ax.bar(range(len(conditions)), avg_goals, color=colors[:len(conditions)], alpha=0.7)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([COMMUNICATION_CONDITIONS[c]['name'] for c in conditions], rotation=45, ha='right')
    ax.set_ylabel('Average Captured Goals')
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{avg_goals[i]:.2f}\n(var={goals_var[i]:.2f})',
                ha='center', va='bottom', fontsize=9)
    fig.savefig(os.path.join(output_dir, 'goals_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Comparison plots saved to {output_dir}")


def test():
    print("Start multi-condition tests")
    model_dir = 'path_to_save_model_1'
    num_episodes_per_condition = 250
    max_steps = 250

    all_conditions_results = {}
    for condition_name, condition_params in COMMUNICATION_CONDITIONS.items():
        results = test_single_condition(
            condition_name=condition_name,
            condition_params=condition_params,
            model_dir=model_dir,
            num_episodes=num_episodes_per_condition,
            max_steps=max_steps
        )
        all_conditions_results[condition_name] = results

    comparison_dir = 'metrics_out/comparison'
    plot_communication_comparison(all_conditions_results, comparison_dir)

    # summary table
    summary_rows = []
    for cname in COMMUNICATION_CONDITIONS.keys():
        r = all_conditions_results[cname]
        summary_rows.append({
            'Condition': COMMUNICATION_CONDITIONS[cname]['name'],
            'Success Rate': f"{r['success_rate']:.3f}",
            'Avg Steps': f"{r['steps_mean']:.1f}",
            'Steps Var': f"{r['steps_var']:.2f}",
            'Avg Goals': f"{r['avg_goals']:.2f}"
        })
    df_summary = pd.DataFrame(summary_rows)
    os.makedirs(comparison_dir, exist_ok=True)
    summary_path = os.path.join(comparison_dir, 'summary_table.csv')
    df_summary.to_csv(summary_path, index=False)
    print("All tests done. Summary saved to", summary_path)
    with open(os.path.join(comparison_dir, 'full_results.json'), 'w') as f:
        json.dump(all_conditions_results, f, indent=4)


if __name__ == '__main__':
    start_time = time.time()
    test()
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")