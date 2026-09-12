# -*- coding: utf-8 -*-
"""
CTDE 训练主程序（时间驱动 + 动态Padding版）
- ✅ Batch内动态padding（突破固定MAX_TASKS限制）
- ✅ 改进的采样策略（更高OOD概率）
- ✅ 特征归一化
- ✅ 任务��量监控
- ✅ P0: 全量buffer × K_EPOCHS mini-batch PPO
- ✅ P0: GAE(λ) 替代 MC returns
- ✅ P0: 按episode分段计算returns，消除跨episode污染
- ✅ 思路3: shaping reward + 终末全局reward 结合
- ✅ 持续调度版本：小规模开局 + 高频潮汐注入
"""

import os
import re
import glob as _glob
import time
import numpy as np
import torch
import torch.optim as optim
from contextlib import nullcontext
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from old_task_env import TaskEnv
from old_network import CTDE_Network
from torch.utils.tensorboard import SummaryWriter


class TrainParams:
    LR = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    K_EPOCHS = 3
    EPS_CLIP = 0.2
    MAX_EPISODES = 20000          # ✅ 2W
    USE_GPU = True

    # === 硬件档位（i5-14600KF + 12GB GPU + 32GB RAM）===
    N_ENVS = 6                       # ✅ 6，折中并行数平衡CPU和GPU利用率
    UPDATE_STEP = 4096               # ✅ 4096，更快触发GPU更新
    MINI_BATCH_SIZE = 1024
    CPU_THREADS = 6

    # 计算后端加速
    USE_TF32 = True
    MATMUL_PRECISION = "high"  # "high" / "medium"
    USE_INFERENCE_MODE = True
    USE_AMP = True
    AMP_DTYPE = "bfloat16"  # "bfloat16" / "float16"
    # Entropy annealing（A方案：更激进的衰减，让模型变 deterministic）
    ENTROPY_START = 0.05            # A方案：0.02 → 0.05（前期更多探索）
    ENTROPY_END = 0.0005            # A方案：0.003 → 0.0005（后期接近 deterministic）
    ENTROPY_DECAY_END_EP = 12000    # A方案：8000 → 12000（衰减再慢一点）

    GRAD_NORM = 1.0
    MAX_AGENTS = 30

    MAX_STEP_PER_EPISODE = 3000
    STEP_TIMEOUT = 300.0
    EPISODE_TIMEOUT = 120.0
    ENV_TIME_LIMIT = 3000.0
    STEP_LOG_INTERVAL = 200
    PERF_LOG_INTERVAL_STEPS = 1000

    DT = 1.0

    # ========== ✅ 5阶段课程学习 ==========
    CURR_STAGE_1 = 0
    CURR_STAGE_2 = 500
    CURR_STAGE_3 = 5000
    CURR_STAGE_4 = 11000
    CURR_STAGE_5 = 15000

    # --- 初始积压任务数范围（模拟早班交接时的存量订单） ---
    # 核心逻辑：初始只给少量任务，大头全靠后续的“潮汐定时爆发”源源不断地喂给小车
    TASKS_RANGE_STAGE_1 = (8, 15)    # 阶段1：热身，极少任务，让小车学会基础移动和干活
    TASKS_RANGE_STAGE_2 = (15, 25)   # 阶段2：起步，稍微增加一点积压
    TASKS_RANGE_STAGE_3 = (20, 30)   # 阶段3：进阶，常规早高峰积压
    TASKS_RANGE_STAGE_4 = (25, 35)   # 阶段4：主训练区，中度早高峰积压
    TASKS_RANGE_STAGE_5 = (30, 40)   # 阶段5：高压区，重度早高峰积压

    # --- 泛化测试（OOD）---
    # 用于偶尔抽查模型能否应对极其恶劣的开局
    OOD_TASKS_RANGE = (35, 45)
    OOD_STAGE5_RANGE = (40, 50)

    # --- 任务持续时间缩放 ---
    DURATION_SCALE_STAGE_1 = 3
    DURATION_SCALE_STAGE_2 = 5
    DURATION_SCALE_STAGE_3 = 7
    DURATION_SCALE_STAGE_4 = 9
    DURATION_SCALE_STAGE_5 = 10

    # --- 动态任务发现概率 ---
    DISCOVERY_PROB_STAGE_1 = 0.01
    DISCOVERY_PROB_STAGE_2 = 0.03
    DISCOVERY_PROB_STAGE_3 = 0.05
    DISCOVERY_PROB_STAGE_4 = 0.07
    DISCOVERY_PROB_STAGE_5 = 0.08

    # --- Deadline 松紧度 ---
    DEADLINE_TIGHTNESS_STAGE_1 = (1.5, 2.5)
    DEADLINE_TIGHTNESS_STAGE_2 = (1.2, 2.0)
    DEADLINE_TIGHTNESS_STAGE_3 = (1.0, 1.5)
    DEADLINE_TIGHTNESS_STAGE_4 = (1.0, 1.5)
    DEADLINE_TIGHTNESS_STAGE_5 = (1.0, 1.3)

    # --- Agent 数量课程（渐进增加）---
    AGENT_SPECIES_STAGE_1 = (2, 3)
    AGENT_SPECIES_STAGE_2 = (2, 4)
    AGENT_SPECIES_STAGE_3 = (3, 4)
    AGENT_SPECIES_STAGE_4 = (3, 5)
    AGENT_SPECIES_STAGE_5 = (4, 5)

    AGENT_PER_SPECIES_STAGE_1 = (2, 3)
    AGENT_PER_SPECIES_STAGE_2 = (3, 4)
    AGENT_PER_SPECIES_STAGE_3 = (4, 5)
    AGENT_PER_SPECIES_STAGE_4 = (4, 5)
    AGENT_PER_SPECIES_STAGE_5 = (5, 6)

    # --- 通信范围 ---
    COMM_RANGE_STAGE_1 = 0.5
    COMM_RANGE_STAGE_2 = 0.4
    COMM_RANGE_STAGE_3 = 0.35
    COMM_RANGE_STAGE_4 = 0.28
    COMM_RANGE_STAGE_5 = 0.25

    # Domain randomization
    COMM_RANGE_JITTER = 0.06
    COMM_RANGE_MIN = 0.12
    COMM_RANGE_MAX = 0.60

    DISCOVERY_JITTER = 0.02
    DISCOVERY_MIN = 0.0
    DISCOVERY_MAX = 0.10

    DURATION_SCALE_JITTER_FRAC = 0.20

    TASK_RANGE_JITTER = 4
    MIX_SIMPLE_PROB = 0.25
    MIX_SIMPLE_RANGE = (10, 20)

    OOD_SPRINKLE_PROB = 0.20
    OOD_STAGE5_PROB = 0.35

    HIGH_TASK_BOOST = True
    HIGH_TASK_THRESHOLD = 50
    HIGH_TASK_PROB = 0.25

    # 终末奖励分摊
    TERMINAL_REWARD_MODE = "weighted_spread"

    # 模型保存（A方案：新目录，避免覆盖 v1.0）
    CHECKPOINT_DIR = "./checkpoints_old/7/1"
    SAVE_EVERY_EP = 1000
    SAVE_BEST_METRIC = "reward_total"

    # 新增：学习率衰减
    LR_DECAY = True
    LR_DECAY_START = 10000
    LR_DECAY_END = 15000
    LR_MIN = 5e-5

    # Actor KV 消融开关: "summary"(单向量KV) / "tokens"(多agent KV)
    ACTOR_KV_MODE = "tokens"

    # 断点续训：从 7/1 的 10000 episode checkpoint 继续训
    RESUME = True
    RESUME_CKPT = None  # None = 自动从 CHECKPOINT_DIR 找最新 checkpoint


# ========== 动态Padding核心函数 ==========

def pad_to_batch_max(batch_data_list, pad_value=0.0):
    if len(batch_data_list) == 0:
        return None, None

    batch_size = len(batch_data_list)
    max_len = max(x.shape[0] for x in batch_data_list)
    feature_dim = batch_data_list[0].shape[-1] if batch_data_list[0].dim() > 1 else 1

    if batch_data_list[0].dim() == 1:
        padded = torch.full((batch_size, max_len), pad_value,
                             dtype=batch_data_list[0].dtype,
                             device=batch_data_list[0].device)
    else:
        padded = torch.full((batch_size, max_len, feature_dim), pad_value,
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


def padding_mask_from_lengths(lengths, max_len, device):
    """Build the [B, max_len] padding mask (True = padding) that get_value expects.

    ``pad_to_batch_max`` already returns the real lengths, but both call sites threw
    them away. Without the mask, ``get_value`` attends over the zero rows and divides
    its mean pooling by the padded count, so the value of one stored global state
    depends on how many other samples happened to land in the same batch. Rollout
    pads to the round maximum and update pads to a random 1024-sample minibatch
    maximum, which made the two disagree for identical input -- and since the critic
    target is ``returns = advantages + rollout_values``, it was trained towards a
    value it cannot represent.
    """
    if lengths is None:
        return None
    positions = torch.arange(int(max_len), device=device)
    return positions.unsqueeze(0) >= lengths.to(device).unsqueeze(1)


def collate_batch_dynamic(buffer, indices, device):
        batch_states = []
        batch_neighbors = []
        batch_tasks = []
        batch_masks = []
        batch_actions = []
        batch_logprobs = []
        batch_values = []
        batch_rewards = []
        batch_dones = []
        batch_g_agents = []
        batch_g_tasks = []

        for idx in indices:
            batch_states.append(buffer["states"][idx])
            batch_neighbors.append(buffer["neighbors"][idx])
            batch_tasks.append(buffer["tasks"][idx])
            batch_masks.append(buffer["masks"][idx])
            batch_actions.append(buffer["actions"][idx])
            batch_logprobs.append(buffer["logprobs"][idx])
            batch_values.append(buffer["values"][idx])
            batch_rewards.append(buffer["rewards"][idx])
            batch_dones.append(buffer["dones"][idx])
            batch_g_agents.append(buffer["g_agents"][idx])
            batch_g_tasks.append(buffer["g_tasks"][idx])

        states = torch.stack(batch_states).to(device)
        neighbors = torch.stack(batch_neighbors).to(device)

        tt_list = [t for t in batch_tasks]
        tasks, task_lengths = pad_to_batch_max(tt_list, pad_value=0.0)
        tasks = tasks.to(device)

        masks_expanded = [m.unsqueeze(-1).float() for m in batch_masks]
        masks, _ = pad_to_batch_max(masks_expanded, pad_value=1.0)
        masks = masks.squeeze(-1).bool().to(device)

        actions = torch.tensor(batch_actions, dtype=torch.long).to(device)
        old_logprobs = torch.tensor(batch_logprobs, dtype=torch.float32).to(device)
        old_values = torch.tensor(batch_values, dtype=torch.float32).to(device)
        rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(batch_dones, dtype=torch.float32).to(device)

        g_agents, agent_lengths = pad_to_batch_max(batch_g_agents, pad_value=0.0)
        g_tasks, task_lengths_global = pad_to_batch_max(batch_g_tasks, pad_value=0.0)
        g_agents = g_agents.to(device)
        g_tasks = g_tasks.to(device)
        agent_padding_mask = padding_mask_from_lengths(agent_lengths, g_agents.shape[1], device)
        task_padding_mask = padding_mask_from_lengths(task_lengths_global, g_tasks.shape[1], device)

        return {
            'states': states,
            'neighbors': neighbors,
            'tasks': tasks,
            'masks': masks,
            'actions': actions,
            'old_logprobs': old_logprobs,
            'old_values': old_values,
            'rewards': rewards,
            'dones': dones,
            'g_agents': g_agents,
            'g_tasks': g_tasks,
            'agent_padding_mask': agent_padding_mask,
            'task_padding_mask': task_padding_mask,
            'task_lengths': task_lengths,
        }

# ========== 辅助函数 ==========

def get_entropy_weight(ep: int, params: TrainParams) -> float:
    if ep <= 0:
        return float(params.ENTROPY_START)
    if ep >= params.ENTROPY_DECAY_END_EP:
        return float(params.ENTROPY_END)
    a = ep / float(params.ENTROPY_DECAY_END_EP)
    return float(params.ENTROPY_START * (1 - a) + params.ENTROPY_END * a)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def jitter_int_range(rng, jitter, lo=1, hi=200):
    a, b = int(rng[0]), int(rng[1])
    a2 = clamp(a - jitter, lo, hi)
    b2 = clamp(b + jitter, lo, hi)
    if a2 > b2:
        a2, b2 = b2, a2
    return (int(a2), int(b2))

def sample_curriculum_settings(ep: int, params: TrainParams, rng: np.random.Generator):
    if ep < params.CURR_STAGE_2:
        stage = 1
        base_comm = params.COMM_RANGE_STAGE_1
        base_tasks = params.TASKS_RANGE_STAGE_1
        base_dur = params.DURATION_SCALE_STAGE_1
        base_disc = params.DISCOVERY_PROB_STAGE_1
        base_deadline = params.DEADLINE_TIGHTNESS_STAGE_1
        species_range = params.AGENT_SPECIES_STAGE_1
        per_species_range = params.AGENT_PER_SPECIES_STAGE_1
    elif ep < params.CURR_STAGE_3:
        stage = 2
        base_comm = params.COMM_RANGE_STAGE_2
        base_tasks = params.TASKS_RANGE_STAGE_2
        base_dur = params.DURATION_SCALE_STAGE_2
        base_disc = params.DISCOVERY_PROB_STAGE_2
        base_deadline = params.DEADLINE_TIGHTNESS_STAGE_2
        species_range = params.AGENT_SPECIES_STAGE_2
        per_species_range = params.AGENT_PER_SPECIES_STAGE_2
    elif ep < params.CURR_STAGE_4:
        stage = 3
        base_comm = params.COMM_RANGE_STAGE_3
        base_tasks = params.TASKS_RANGE_STAGE_3
        base_dur = params.DURATION_SCALE_STAGE_3
        base_disc = params.DISCOVERY_PROB_STAGE_3
        base_deadline = params.DEADLINE_TIGHTNESS_STAGE_3
        species_range = params.AGENT_SPECIES_STAGE_3
        per_species_range = params.AGENT_PER_SPECIES_STAGE_3
    elif ep < params.CURR_STAGE_5:
        stage = 4
        base_comm = params.COMM_RANGE_STAGE_4
        base_tasks = params.TASKS_RANGE_STAGE_4
        base_dur = params.DURATION_SCALE_STAGE_4
        base_disc = params.DISCOVERY_PROB_STAGE_4
        base_deadline = params.DEADLINE_TIGHTNESS_STAGE_4
        species_range = params.AGENT_SPECIES_STAGE_4
        per_species_range = params.AGENT_PER_SPECIES_STAGE_4
    else:
        stage = 5
        base_comm = params.COMM_RANGE_STAGE_5
        base_tasks = params.TASKS_RANGE_STAGE_5
        base_dur = params.DURATION_SCALE_STAGE_5
        base_disc = params.DISCOVERY_PROB_STAGE_5
        base_deadline = params.DEADLINE_TIGHTNESS_STAGE_5
        species_range = params.AGENT_SPECIES_STAGE_5
        per_species_range = params.AGENT_PER_SPECIES_STAGE_5

    # ✅ 均匀任务池：直接在全区间内均匀采样，避免阶段偏置与灾难性遗忘
    min_tasks = int(params.TASKS_RANGE_STAGE_1[0])
    max_tasks = int(params.OOD_STAGE5_RANGE[1])
    tasks_rng = (min_tasks, max_tasks)
    tasks_rng = jitter_int_range(tasks_rng, params.TASK_RANGE_JITTER, lo=min_tasks, hi=max_tasks)

    comm = base_comm + rng.uniform(-params.COMM_RANGE_JITTER, params.COMM_RANGE_JITTER)
    comm = float(clamp(comm, params.COMM_RANGE_MIN, params.COMM_RANGE_MAX))

    disc = base_disc + rng.uniform(-params.DISCOVERY_JITTER, params.DISCOVERY_JITTER)
    disc = float(clamp(disc, params.DISCOVERY_MIN, params.DISCOVERY_MAX))

    dur = base_dur * (1.0 + rng.uniform(-params.DURATION_SCALE_JITTER_FRAC, params.DURATION_SCALE_JITTER_FRAC))
    dur = float(max(1.0, dur))

    return stage, comm, tasks_rng, dur, disc, base_deadline, species_range, per_species_range

def inject_terminal_reward(buffer, terminal_reward, ep_start_idx, mode="spread"):
    n = len(buffer["rewards"])
    ep_len = n - ep_start_idx

    if ep_len <= 0 or terminal_reward == 0.0:
        return

    if mode == "last":
        buffer["rewards"][-1] += terminal_reward
    elif mode == "spread":
        per_step = terminal_reward / ep_len
        for i in range(ep_start_idx, n):
            buffer["rewards"][i] += per_step
    elif mode == "weighted_spread":
        weights = np.arange(1, ep_len + 1, dtype=np.float64)
        weights = weights / weights.sum()
        for i, w in enumerate(weights):
            buffer["rewards"][ep_start_idx + i] += terminal_reward * w
    else:
        per_step = terminal_reward / ep_len
        for i in range(ep_start_idx, n):
            buffer["rewards"][i] += per_step

def compute_gae_segmented(buffer, gamma, gae_lambda, device, net):
    n = len(buffer["states"])
    if n == 0:
        return [], []

    rewards = buffer["rewards"]
    dones = buffer["dones"]
    values = buffer["values"]

    advantages = [0.0] * n
    returns = [0.0] * n

    last_gae = 0.0

    for t in reversed(range(n)):
        if t == n - 1:
            next_value = 0.0
            last_gae = 0.0
        else:
            if dones[t]:
                next_value = 0.0
                last_gae = 0.0
            else:
                next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1.0 - float(dones[t])) - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - float(dones[t])) * last_gae

        advantages[t] = last_gae
        returns[t] = last_gae + values[t]

    return advantages, returns

def update(net, optimizer, buffer, device, params: TrainParams, entropy_weight: float):
    n = len(buffer["states"])
    if n < params.MINI_BATCH_SIZE:
        return {}

    advantages_list, returns_list = compute_gae_segmented(
        buffer, params.GAMMA, params.GAE_LAMBDA, device, net
    )

    buffer["advantages"] = advantages_list
    buffer["returns"] = returns_list

    advantages_tensor = torch.as_tensor(advantages_list, dtype=torch.float32, device=device)
    returns_tensor = torch.as_tensor(returns_list, dtype=torch.float32, device=device)

    amp_enabled = bool(params.USE_AMP and device.type == "cuda")
    amp_dtype = torch.bfloat16 if str(params.AMP_DTYPE).lower() == "bfloat16" else torch.float16

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    for epoch in range(params.K_EPOCHS):
        perm = np.random.permutation(n)

        for start in range(0, n, params.MINI_BATCH_SIZE):
            end = min(start + params.MINI_BATCH_SIZE, n)
            if end - start < 8:
                continue

            mb_indices = perm[start:end]

            try:
                batch = collate_batch_dynamic(buffer, mb_indices, device)
            except Exception as e:
                print(f"[ERROR] Error in collate_batch_dynamic: {e}")
                continue

            mb_idx_t = torch.as_tensor(mb_indices, dtype=torch.long, device=device)
            mb_advantages = advantages_tensor.index_select(0, mb_idx_t)
            mb_returns = returns_tensor.index_select(0, mb_idx_t)

            if len(mb_advantages) > 1:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
            with amp_ctx:
                dist = net.get_action(
                    batch['states'],
                    batch['neighbors'],
                    batch['tasks'],
                    batch['masks']
                )

                logprobs = dist.log_prob(batch['actions'])
                entropy = dist.entropy().mean()

                values = net.get_value(
                    batch['g_agents'],
                    batch['g_tasks'],
                    batch['agent_padding_mask'],
                    batch['task_padding_mask'],
                ).squeeze(-1)

                ratio = torch.exp(logprobs - batch['old_logprobs'])
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - params.EPS_CLIP, 1 + params.EPS_CLIP) * mb_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * ((values - mb_returns) ** 2).mean()
                entropy_loss = -float(entropy_weight) * entropy

                loss = actor_loss + critic_loss + entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), params.GRAD_NORM)
            optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            num_updates += 1

    for key in buffer.keys():
        buffer[key] = []

    if num_updates > 0:
        return {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }
    return {}

def get_lr(ep: int, params: TrainParams) -> float:
    if not params.LR_DECAY or ep < params.LR_DECAY_START:
        return params.LR
    if ep >= params.LR_DECAY_END:
        return params.LR_MIN

    progress = (ep - params.LR_DECAY_START) / (params.LR_DECAY_END - params.LR_DECAY_START)
    return params.LR * (1.0 - progress) + params.LR_MIN * progress

def save_checkpoint(path, net, optimizer, ep, best_metric=None, extra=None):
    ckpt = {
        "episode": ep,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if best_metric is not None:
        ckpt["best_metric"] = float(best_metric)
    if extra is not None:
        ckpt.update(extra)
    torch.save(ckpt, path)


def main():
    print("=" * 60)
    print(f"[START] CTDE High-Performance Training (N_ENVS={TrainParams.N_ENVS})")
    print("=" * 60)

    # 针对 i5-14600KF 设定 CPU 线程，避免过度争抢
    torch.set_num_threads(int(TrainParams.CPU_THREADS))
    torch.set_num_interop_threads(1)

    # RTX 50 系列可用的通用加速开关
    if TrainParams.USE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision(TrainParams.MATMUL_PRECISION)

    device = torch.device("cuda" if torch.cuda.is_available() and TrainParams.USE_GPU else "cpu")

    log_dir = os.path.join(r"D:\tb_logs", time.strftime("ctde_dynamic_%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs(TrainParams.CHECKPOINT_DIR, exist_ok=True)

    net = CTDE_Network(self_dim=10, neighbor_dim=10, task_dim=20, global_agent_dim=13, global_task_dim=23, hidden=128,
                       dropout=0.15, use_pos_encoding=True, actor_kv_mode=TrainParams.ACTOR_KV_MODE).to(device)
    optimizer = optim.Adam(net.parameters(), lr=TrainParams.LR)

    global_episodes_done = 0
    best_reward = -1e18
    total_env_steps = 0

    # ===== 断点续训 =====
    if TrainParams.RESUME:
        resume_path = TrainParams.RESUME_CKPT
        if resume_path is None:
            # 自动找最新 checkpoint（按 episode 编号）
            ckpt_dir = TrainParams.CHECKPOINT_DIR
            pattern = os.path.join(ckpt_dir, "ctde_ep_*.pth")
            candidates = _glob.glob(pattern)
            if candidates:
                def _ep_num(p):
                    m = re.search(r'ctde_ep_(\d+)\.pth', p)
                    return int(m.group(1)) if m else 0
                resume_path = max(candidates, key=_ep_num)
        if resume_path and os.path.isfile(resume_path):
            print(f"[RESUME] Loading checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            net.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            global_episodes_done = ckpt.get("episode", 0)
            best_reward = ckpt.get("best_metric", None)
            # 如果当前 ckpt 不含 best_metric，从 ctde_best.pth 读取
            if best_reward is None:
                best_path = os.path.join(TrainParams.CHECKPOINT_DIR, "ctde_best.pth")
                if os.path.isfile(best_path):
                    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
                    best_reward = best_ckpt.get("best_metric", -1e18)
                else:
                    best_reward = -1e18
            print(f"[RESUME] Restored episode={global_episodes_done}, best_reward={best_reward:.2f}")
        else:
            print("[RESUME] No checkpoint found, starting from scratch.")

    # ✅ 核心架构跃迁：开启高频潮汐持续注入
    envs = [TaskEnv(
        deadline_tightness=TrainParams.DEADLINE_TIGHTNESS_STAGE_1,
        per_species_range=TrainParams.AGENT_PER_SPECIES_STAGE_1,
        species_range=TrainParams.AGENT_SPECIES_STAGE_1,
        tasks_range=(TrainParams.TASKS_RANGE_STAGE_1[0], TrainParams.OOD_STAGE5_RANGE[1]),
        comm_range=TrainParams.COMM_RANGE_STAGE_1,
        dynamic_task_discovery=True, discovery_probability=TrainParams.DISCOVERY_PROB_STAGE_1, max_dynamic_tasks=5,
        enable_tidal_tasks=True,
        # ✅ 修改爆发间隔为 200 步，每次爆发 8~15 个
        tidal_interval=200.0, tidal_burst_range=(8, 15), tidal_notify_ratio=0.2, dt=TrainParams.DT,
        enable_discovered_tasks=False,
    ) for _ in range(TrainParams.N_ENVS)]

    # ✅ 修改：加入 agent_ids 字段用于阶段奖励回填
    _buf_keys = ["states", "neighbors", "tasks", "masks", "actions", "logprobs",
                 "values", "rewards", "dones", "g_agents", "g_tasks", "agent_ids"]
    env_buffers = [{k: [] for k in _buf_keys} for _ in range(TrainParams.N_ENVS)]
    global_buffer = {k: [] for k in
                     ["states", "neighbors", "tasks", "masks", "actions", "logprobs",
                      "values", "rewards", "dones", "g_agents", "g_tasks"]}

    rng = np.random.default_rng(seed=42)

    # 性能监控（total_env_steps 已在 resume 前初始化）
    total_samples = 0
    last_perf_time = time.perf_counter()
    last_perf_steps = 0
    last_perf_samples = 0
    sample_time_accum = 0.0
    env_step_time_accum = 0.0
    update_time_accum = 0.0
    update_count = 0

    for i in range(TrainParams.N_ENVS):
        envs[i].reset(seed=i)

    while global_episodes_done < TrainParams.MAX_EPISODES:
        entropy_w = get_entropy_weight(global_episodes_done, TrainParams)
        current_lr = get_lr(global_episodes_done, TrainParams)
        for param_group in optimizer.param_groups: param_group['lr'] = current_lr

        batch_s, batch_n, batch_t, batch_m = [], [], [], []
        meta_info = []

        t_sample0 = time.perf_counter()
        for env_idx, env in enumerate(envs):
            if env.finished: continue
            decision_ids = [aid for aid, a in env.agent_dic.items() if a.get("need_decision", False)]
            if not decision_ids: continue

            g_agents_np, g_tasks_np = env.get_global_state()
            g_a_cpu = torch.as_tensor(g_agents_np, dtype=torch.float32)
            g_t_cpu = torch.as_tensor(g_tasks_np, dtype=torch.float32)

            for aid in decision_ids:
                s, n, t, m, amap = env.get_local_obs(aid)
                batch_s.append(torch.as_tensor(s, dtype=torch.float32))
                batch_n.append(torch.as_tensor(n, dtype=torch.float32))
                batch_t.append(torch.as_tensor(t, dtype=torch.float32))
                batch_m.append(torch.as_tensor(m, dtype=torch.bool))
                meta_info.append((env_idx, aid, amap, g_a_cpu, g_t_cpu))

        if len(meta_info) > 0:
            total_samples += len(meta_info)
            ts = torch.stack(batch_s).to(device)
            tn = torch.stack(batch_n).to(device)
            tt, _ = pad_to_batch_max(batch_t, pad_value=0.0);
            tt = tt.to(device)
            tm_expanded = [m.unsqueeze(-1).float() for m in batch_m]
            tm, _ = pad_to_batch_max(tm_expanded, pad_value=1.0)
            tm = tm.squeeze(-1).bool().to(device)

            bg_a_list = [item[3] for item in meta_info]
            bg_t_list = [item[4] for item in meta_info]
            tg_a, tg_a_lengths = pad_to_batch_max(bg_a_list, pad_value=0.0);
            tg_a = tg_a.to(device)
            tg_t, tg_t_lengths = pad_to_batch_max(bg_t_list, pad_value=0.0);
            tg_t = tg_t.to(device)
            tg_a_padding = padding_mask_from_lengths(tg_a_lengths, tg_a.shape[1], device)
            tg_t_padding = padding_mask_from_lengths(tg_t_lengths, tg_t.shape[1], device)

            if TrainParams.USE_INFERENCE_MODE:
                ctx = torch.inference_mode()
            else:
                ctx = torch.no_grad()

            with ctx:
                dist = net.get_action(ts, tn, tt, tm)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)
                values = net.get_value(tg_a, tg_t, tg_a_padding, tg_t_padding).squeeze(-1)

            for j, (env_idx, aid, amap, g_a_cpu, g_t_cpu) in enumerate(meta_info):
                act = int(actions[j].item())
                reward, _, _ = envs[env_idx].apply_action(aid, act, amap)

                env_buffers[env_idx]["states"].append(batch_s[j])
                env_buffers[env_idx]["neighbors"].append(batch_n[j])
                env_buffers[env_idx]["tasks"].append(batch_t[j])
                env_buffers[env_idx]["masks"].append(batch_m[j])
                env_buffers[env_idx]["actions"].append(act)
                env_buffers[env_idx]["logprobs"].append(float(logprobs[j].item()))
                env_buffers[env_idx]["values"].append(float(values[j].item()))
                env_buffers[env_idx]["rewards"].append(float(reward))
                env_buffers[env_idx]["dones"].append(False)
                env_buffers[env_idx]["g_agents"].append(g_a_cpu)
                env_buffers[env_idx]["g_tasks"].append(g_t_cpu)
                env_buffers[env_idx]["agent_ids"].append(aid)  # ✅ 新增
        sample_time_accum += (time.perf_counter() - t_sample0)

        t_env0 = time.perf_counter()
        # ✅ ThreadPoolExecutor 并行 env.step()（numpy 操作释放 GIL）
        with ThreadPoolExecutor(max_workers=TrainParams.N_ENVS) as executor:
            future_to_env = {executor.submit(env.step, TrainParams.DT): (env_idx, env)
                             for env_idx, env in enumerate(envs) if not env.finished}
            for future in as_completed(future_to_env):
                env_idx, env = future_to_env[future]
                future.result()
                total_env_steps += 1
                if env.current_time >= TrainParams.ENV_TIME_LIMIT or env._inner_step_counter >= TrainParams.MAX_STEP_PER_EPISODE:
                    env.finished = True
                if env._inner_step_counter % 10 == 0:
                    env.cleanup_failed_tasks_memory()
                # ✅ 阶段完成奖励回填
                if env._step_events:
                    buf = env_buffers[env_idx]
                    buf_len = len(buf["rewards"])
                    if buf_len > 0:
                        n_agents = max(1, env.agents_num)
                        scan_start = max(0, buf_len - n_agents * 2)
                        aid_to_last_idx = {}
                        for bi in range(scan_start, buf_len):
                            aid_to_last_idx[buf["agent_ids"][bi]] = bi
                        for event in env._step_events:
                            if event["type"] == "stage_complete":
                                for aid, r in event["rewards"].items():
                                    if aid in aid_to_last_idx:
                                        buf["rewards"][aid_to_last_idx[aid]] += float(r)
                    env._step_events.clear()
        for env_idx, env in enumerate(envs):
            if env.finished:
                if len(env_buffers[env_idx]["dones"]) > 0:
                    env_buffers[env_idx]["dones"][-1] = True
                try:
                    final_reward, finished_tasks = env.get_episode_reward()
                    total_tasks = len(env.task_dic)
                    deadline_done = sum(1 for t in env.task_dic.values()
                                        if t.get("finished") and not t.get("expired"))
                    deadline_sr = deadline_done / total_tasks if total_tasks > 0 else 0.0
                    res_util = getattr(env, "get_resource_utilization", lambda: 0.0)()
                except Exception:
                    final_reward, deadline_sr, res_util = 0.0, 0.0, 0.0
                inject_terminal_reward(env_buffers[env_idx], terminal_reward=final_reward, ep_start_idx=0,
                                       mode=TrainParams.TERMINAL_REWARD_MODE)
                ep_reward = sum(env_buffers[env_idx]["rewards"])
                for k in env_buffers[env_idx].keys():
                    # ✅ agent_ids 不需要进 global_buffer（只是辅助字段），跳过
                    if k == "agent_ids":
                        env_buffers[env_idx][k] = []
                        continue
                    global_buffer[k].extend(env_buffers[env_idx][k])
                    env_buffers[env_idx][k] = []

                global_episodes_done += 1

                writer.add_scalar("episode/reward_total", ep_reward, global_episodes_done)
                writer.add_scalar("episode/deadline_success_rate", deadline_sr, global_episodes_done)
                writer.add_scalar("episode/resource_utilization", res_util, global_episodes_done)

                # deadline 指标
                n_expired = sum(1 for t in env.task_dic.values() if t.get("expired", False))
                writer.add_scalar("episode/expired_rate", n_expired / max(1, len(env.task_dic)), global_episodes_done)
                slack_sum = 0.0
                slack_count = 0
                for t in env.task_dic.values():
                    if t.get("finished") and not t.get("expired"):
                        slack_sum += max(0.0, t["deadline"] - t.get("time_finish", env.current_time))
                        slack_count += 1
                avg_slack = slack_sum / max(1, slack_count)
                writer.add_scalar("episode/avg_slack", avg_slack, global_episodes_done)

                print(
                    f"Env[{env_idx}] Ep {global_episodes_done} | R:{ep_reward:.1f} | Term:{final_reward:.1f} | DL_SR:{deadline_sr:.2%} | Exp:{n_expired} | Slack:{avg_slack:.1f} | Time:{env.current_time:.0f}")

                if ep_reward > best_reward:
                    best_reward = ep_reward
                    save_checkpoint(os.path.join(TrainParams.CHECKPOINT_DIR, "ctde_best.pth"), net, optimizer,
                                    global_episodes_done, best_metric=best_reward)

                if global_episodes_done % TrainParams.SAVE_EVERY_EP == 0:
                    save_path = os.path.join(TrainParams.CHECKPOINT_DIR, f"ctde_ep_{global_episodes_done}.pth")
                    torch.save({
                        "episode": global_episodes_done,
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()
                    }, save_path)
                    print(f"[SAVE] Checkpoint saved: {save_path}")

                stage, comm, tasks_rng, dur_scale, disc_prob, deadline_tightness, species_range, per_species_range = sample_curriculum_settings(
                    global_episodes_done, TrainParams, rng)
                env.comm_range, env.ranges["tasks"], env.duration_scale, env.discovery_probability, env.ranges["deadline_tightness"] = \
                    comm, tasks_rng, dur_scale, disc_prob, deadline_tightness
                env.ranges["species"] = species_range
                env.ranges["per_species"] = per_species_range
                env.reset(seed=global_episodes_done)
        env_step_time_accum += (time.perf_counter() - t_env0)

        if len(global_buffer["states"]) >= TrainParams.UPDATE_STEP:
            t_up0 = time.perf_counter()
            update_stats = update(net, optimizer, global_buffer, device, TrainParams, entropy_weight=entropy_w)
            update_time_accum += (time.perf_counter() - t_up0)
            update_count += 1
            if update_stats:
                writer.add_scalar("train/actor_loss", update_stats.get("actor_loss", 0), global_episodes_done)
                writer.add_scalar("train/critic_loss", update_stats.get("critic_loss", 0), global_episodes_done)

        if total_env_steps - last_perf_steps >= TrainParams.PERF_LOG_INTERVAL_STEPS:
            now = time.perf_counter()
            dt_perf = max(1e-6, now - last_perf_time)
            steps_delta = total_env_steps - last_perf_steps
            samples_delta = total_samples - last_perf_samples

            steps_per_sec = steps_delta / dt_perf
            samples_per_sec = samples_delta / dt_perf
            avg_update_time = (update_time_accum / update_count) if update_count > 0 else 0.0

            stage_total = sample_time_accum + env_step_time_accum + update_time_accum
            if stage_total > 1e-9:
                sample_pct = 100.0 * sample_time_accum / stage_total
                env_pct = 100.0 * env_step_time_accum / stage_total
                update_pct = 100.0 * update_time_accum / stage_total
            else:
                sample_pct = env_pct = update_pct = 0.0

            print(
                f"[Perf] env_steps={total_env_steps} | eps_done={global_episodes_done} | "
                f"steps/s={steps_per_sec:.1f} | samples/s={samples_per_sec:.1f} | "
                f"update_time={avg_update_time:.3f}s | "
                f"stage_ms(sample/env/update)="
                f"{sample_time_accum * 1000:.0f}/{env_step_time_accum * 1000:.0f}/{update_time_accum * 1000:.0f} | "
                f"stage_pct(sample/env/update)={sample_pct:.1f}%/{env_pct:.1f}%/{update_pct:.1f}%"
            )

            writer.add_scalar("perf/steps_per_sec", steps_per_sec, total_env_steps)
            writer.add_scalar("perf/samples_per_sec", samples_per_sec, total_env_steps)
            writer.add_scalar("perf/update_time", avg_update_time, total_env_steps)
            writer.add_scalar("perf/sample_time_sec", sample_time_accum, total_env_steps)
            writer.add_scalar("perf/env_step_time_sec", env_step_time_accum, total_env_steps)
            writer.add_scalar("perf/update_time_sec_window", update_time_accum, total_env_steps)
            writer.add_scalar("perf/sample_time_pct", sample_pct, total_env_steps)
            writer.add_scalar("perf/env_step_time_pct", env_pct, total_env_steps)
            writer.add_scalar("perf/update_time_pct", update_pct, total_env_steps)

            last_perf_time = now
            last_perf_steps = total_env_steps
            last_perf_samples = total_samples
            sample_time_accum = 0.0
            env_step_time_accum = 0.0
            update_time_accum = 0.0
            update_count = 0

    writer.close()
    torch.save({
        "episode": global_episodes_done,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, os.path.join(TrainParams.CHECKPOINT_DIR, "ctde_final.pth"))

    print("\n[DONE] High-Performance Training completed! Final model saved.")


if __name__ == "__main__":
    main()
