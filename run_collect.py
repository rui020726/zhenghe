import os
import json
import numpy as np
import torch
from torch.distributions import Categorical
from typing import List, Dict
import types

import configs
from src.Map.timemap import Environment                 # PILOC（带信息素）
from Abl_del_ph.timemap import Environment as EnvNoPh   # PILOC-ph（去信息素）
from src.DDQN.Agent import MAPPO                        # PILOC 模型
from Abl_del_ph.Agent import MAPPO as MAPPO_NoPh        # PILOC-ph 模型
from src.DDQN.rescue import RescueAlgorithm             # 混合/救援策略
from src.metrics.revisit_metrics import RevisitCoverageMeter

DEVICE = torch.device(configs.DEVICE) if torch.cuda.is_available() else torch.device("cpu")

# 可配置：是否关闭通信噪声（建议在显著性检验或上限评估时关闭）
DISABLE_COMM_NOISE = True

# ---------------- 视线判断（与环境一致的 Bresenham 线判挡） ----------------
def _bresenham_line(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1); dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x1 += sx
        if e2 < dx:
            err += dx; y1 += sy
    return points

def _is_line_of_sight_clear(x1, y1, x2, y2, env_map):
    for px, py in _bresenham_line(x1, y1, x2, y2):
        if env_map[px, py] == 1 and (px != x2 or py != y2):
            return False
    return True

# ---------------- 对齐训练分布的初始观测（含 LoS） ----------------
def initialize_obs(env, num_agents=configs.num_agents, map_size=(60, 60), view_radius=5):
    obs = {}
    for i in range(num_agents):
        agent_obs = np.full(map_size, 0.5, dtype=np.float32)
        x, y = env.agents_pos[i]
        agent_obs[x, y] = -4.5
        for dx in range(-view_radius, view_radius + 1):
            for dy in range(-view_radius, view_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < map_size[0] and 0 <= ny < map_size[1]:
                    if _is_line_of_sight_clear(x, y, nx, ny, env.map):
                        if env.map[nx, ny] == 1:
                            agent_obs[nx, ny] = 1
                        elif [nx, ny] != [x, y] and agent_obs[nx, ny] not in [-4.5, -5.5, -6.5]:
                            agent_obs[nx, ny] = 0
        obs[f"agent_{i}"] = agent_obs
    return obs

def get_local_ph(global_ph, x, y, view_radius=5):
    h, w = global_ph.shape
    x_min = max(0, x - view_radius); x_max = min(h, x + view_radius + 1)
    y_min = max(0, y - view_radius); y_max = min(w, y + view_radius + 1)
    local_ph = global_ph[x_min:x_max, y_min:y_max]
    target_size = 2 * view_radius + 1
    pad_top = max(0, target_size - (x_max - x_min)) if x - view_radius < 0 else 0
    pad_bottom = max(0, target_size - (x_max - x_min)) if x + view_radius + 1 > h else 0
    pad_left = max(0, target_size - (y_max - y_min)) if y - view_radius < 0 else 0
    pad_right = max(0, target_size - (y_max - y_min)) if y + view_radius + 1 > w else 0
    return np.pad(local_ph, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

# ---------------- 选择动作（区分是否使用信息素分支） ----------------
def select_action(two_channelmap, policy, env, env_agents, use_pheromone=True):
    actions = {}
    for agent_id in env_agents:
        obstacle_map = two_channelmap[agent_id]['obstacle_map']
        explore_map = two_channelmap[agent_id]['explore_map']
        agent_intid = int(agent_id.split("_")[-1])
        x, y = env.agents_pos[agent_intid]

        obs = np.stack([obstacle_map, explore_map], axis=0)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        if use_pheromone:
            ph = get_local_ph(env.global_pheromone, x, y)
            ph_tensor = torch.tensor(ph, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            action_probs = policy.agents.get_action(obs_tensor, ph_tensor, batch_size=1)[1]
        else:
            action_probs = policy.agents.get_action(obs_tensor, batch_size=1)[1]

        dist = Categorical(probs=action_probs)
        action = dist.sample()
        actions[agent_id] = action.item()
    return actions

# ---------------- 关闭通信噪声（Monkey Patch，不改源码） ----------------
def disable_comm_noise(env):
    # 强制关闭 merge_timestamp_maps 的通信噪声
    original_merge = env.merge_timestamp_maps
    def _merge_no_noise(map1, map2, enable_comm_noise=True):
        return original_merge(map1, map2, enable_comm_noise=False)
    env.merge_timestamp_maps = types.MethodType(lambda self, m1, m2, enable_comm_noise=True: _merge_no_noise(m1, m2), env)

# ---------------- 覆盖曲线工具 ----------------
def pad_curve(curve: List[float], max_steps: int) -> np.ndarray:
    if len(curve) == 0:
        return np.zeros(max_steps, dtype=np.float32)
    if len(curve) < max_steps:
        return np.concatenate([curve, np.full(max_steps - len(curve), curve[-1])])
    return np.array(curve[:max_steps], dtype=np.float32)

def compute_auc(curve: np.ndarray) -> float:
    return float(np.mean(curve))  # 步长归一时 AUC 等于均值

# ---------------- 主评测函数（在你的基础上：加入救援策略、关闭噪声、对齐 dim_info） ----------------
def run_variant(manifest_file: str,
                model_dir: str,
                use_pheromone: bool,
                num_episodes: int,
                max_steps: int,
                save_json: str,
                E: int):
    # 对齐 test.py / test—ph.py 的输入维度设定
    dim_info = {f'agent_{i}': [9, 5] for i in range(configs.num_agents)}
    if E == 1:
        policy = MAPPO.load(dim_info, False, model_dir, trick=configs.TRICK)        # PILOC
    else:
        policy = MAPPO_NoPh.load(dim_info, False, model_dir, trick=configs.TRICK)   # PILOC-ph

    with open(manifest_file, 'r', encoding='utf-8') as f:
        map_files = [line.strip() for line in f if line.strip()]

    if len(map_files) < num_episodes:
        raise ValueError(f"Manifest has {len(map_files)} maps but num_episodes={num_episodes}")

    # 数据容器
    success_flags: List[int] = []
    steps_list: List[int] = []
    targets_found: List[int] = []
    final_coverages: List[float] = []
    coverage_aucs: List[float] = []

    for ep_idx in range(num_episodes):
        map_file = map_files[ep_idx]
        env = (Environment if E == 1 else EnvNoPh)(test=True, map_filename=map_file)

        if DISABLE_COMM_NOISE:
            disable_comm_noise(env)

        env_agents = list(env.agents)
        obs = initialize_obs(env)
        two_channelmap = env.two_channel_map(obs, {})

        # 救援策略（混合决策）
        rescue_policy = RescueAlgorithm(env.map_size, env, obs)
        rescue_actions: Dict[str, List[int]] = {}

        # 覆盖率追踪
        meter = RevisitCoverageMeter(free_mask=(env.map == 0), radius=5, window=100, use_los=True)
        coverage_history: List[float] = []

        step_t = 0
        success = False

        while step_t < max_steps:
            step_t += 1

            # 神经策略动作
            actions = select_action(two_channelmap, policy, env, env_agents, use_pheromone=use_pheromone)

            # 覆盖救援动作（若存在）
            if rescue_actions:
                for agent_id, seq in list(rescue_actions.items()):
                    if seq:
                        actions[agent_id] = seq[0]
                        seq.pop(0)
                    if not seq:
                        del rescue_actions[agent_id]

            # 环境前进一步
            next_obs, reward, terminated, repeat_mark, next_two_channelmap = env.step(
                actions, obs, rescue_actions, two_channelmap, ep_idx
            )

            # 覆盖率更新
            meter.update(env.agents_pos, env)
            free_cells = np.sum(meter.free_mask)
            observed_cells = np.sum(meter.visited) if hasattr(meter, 'visited') else 0
            coverage_rate = (observed_cells / free_cells) if free_cells > 0 else 0.0
            coverage_history.append(coverage_rate)

            # 触发救援：对重复陷入的智能体生成 A* 序列
            if repeat_mark:
                for agent_id in repeat_mark:
                    agent_index = int(agent_id.split('_')[1])
                    seq = rescue_policy.rescue_action(agent_index, env, next_obs)
                    rescue_actions[agent_id] = seq if seq is not None else []

            # 状态推进
            obs = next_obs
            two_channelmap = next_two_channelmap

            # 终止（所有目标被观测到）
            if any(terminated.values()):
                success = True
                targets_found.append(configs.num_goals)
                break

        if not success:
            remaining = len(env.goals_pos)
            targets_found.append(configs.num_goals - remaining)

        success_flags.append(1 if success else 0)
        steps_list.append(step_t)
        cov_curve = pad_curve(coverage_history, max_steps)
        final_coverages.append(float(cov_curve[-1]))
        coverage_aucs.append(compute_auc(cov_curve))

    results = {
        "variant": "PILOC" if use_pheromone else "PILOC-ph",
        "model_dir": model_dir,
        "use_pheromone": use_pheromone,
        "num_episodes": num_episodes,
        "max_steps": max_steps,
        "success_flags": success_flags,
        "steps": steps_list,
        "targets_found": targets_found,
        "final_coverages": final_coverages,
        "coverage_aucs": coverage_aucs
    }

    os.makedirs(os.path.dirname(save_json), exist_ok=True)
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {results['variant']} raw data -> {save_json}")

if __name__ == "__main__":
    # 示例调用：先运行 PILOC 再运行 PILOC-ph（同一 manifest，保证严格配对）
    manifest = "test_manifest.txt"
    num_eps = 250
    max_steps = 250

    # 替换为你实际的模型路径
    run_variant(manifest, model_dir="path_to_save_model_1", use_pheromone=True,
                num_episodes=num_eps, max_steps=max_steps,
                save_json="raw_metrics/piloc.json", E=1)

    run_variant(manifest, model_dir=r'E:/Desktop/代码/修改6--在修改5的基础上，所有网络使用相同的参数/path_to_save_model_Abl', use_pheromone=False,
                num_episodes=num_eps, max_steps=max_steps,
                save_json="raw_metrics/piloc_ph.json", E=0)