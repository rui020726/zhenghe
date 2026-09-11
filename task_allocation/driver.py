# -*- coding: utf-8 -*-
"""
CTDE 训练主程序（时间驱动 + 动态Padding版）
- ✅ Batch内动态padding（突破固定MAX_TASKS限制）
- ✅ 改进的采样策略（更高OOD概率）
- ✅ 特征归一化
- ✅ 任务数量监控
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict
from collections import deque

from env.task_env import TaskEnv
from network import CTDE_Network
from torch.utils.tensorboard import SummaryWriter


class TrainParams:
    LR = 1e-4
    GAMMA = 0.99
    K_EPOCHS = 4
    EPS_CLIP = 0.1
    MAX_EPISODES = 30000  # ✅ 增加训练量
    BATCH_SIZE = 64
    UPDATE_STEP = 128
    USE_GPU = True

    # Entropy annealing
    ENTROPY_START = 0.02
    ENTROPY_END = 0.005
    ENTROPY_DECAY_END_EP = 4000

    GRAD_NORM = 1.0
    MAX_AGENTS = 30

    # ✅ 移除固定MAX_TASKS限制（仅用于初始化，实际运行时动态）
    # MAX_TASKS = 65  # 不再硬限制

    MAX_STEP_PER_EPISODE = 1000
    STEP_TIMEOUT = 30.0
    EPISODE_TIMEOUT = 120.0
    ENV_TIME_LIMIT = 1000.0
    STEP_LOG_INTERVAL = 200
    TARGET_KL = 0.3

    DT = 1.0

    # ========== ✅ 改进的Curriculum策略 ==========
    CURR_STAGE_1 = 0
    CURR_STAGE_2 = 1500
    CURR_STAGE_3 = 6000
    CURR_STAGE_4 = 15000

    # ✅ Stage3上限扩展到58，Stage4到64
    TASKS_RANGE_STAGE_1 = (8, 15)
    TASKS_RANGE_STAGE_2 = (12, 30)
    TASKS_RANGE_STAGE_3 = (25, 58)  # ⭐ 关键修改
    TASKS_RANGE_STAGE_4 = (45, 64)  # ⭐ 新增

    DURATION_SCALE_STAGE_1 = 3
    DURATION_SCALE_STAGE_2 = 5
    DURATION_SCALE_STAGE_3 = 8
    DURATION_SCALE_STAGE_4 = 10

    DISCOVERY_PROB_STAGE_1 = 0.01
    DISCOVERY_PROB_STAGE_2 = 0.03
    DISCOVERY_PROB_STAGE_3 = 0.06
    DISCOVERY_PROB_STAGE_4 = 0.08

    COMM_RANGE_STAGE_1 = 0.5
    COMM_RANGE_STAGE_2 = 0.4
    COMM_RANGE_STAGE_3 = 0.3
    COMM_RANGE_STAGE_4 = 0.25

    DEADLINE_TIGHTNESS_STAGE_1 = (1.0, 2.5)
    DEADLINE_TIGHTNESS_STAGE_2 = (0.7, 2.0)
    DEADLINE_TIGHTNESS_STAGE_3 = (0.5, 1.5)
    DEADLINE_TIGHTNESS_STAGE_4 = (0.3, 1.2)




    # Domain randomization
    COMM_RANGE_JITTER = 0.06
    COMM_RANGE_MIN = 0.12
    COMM_RANGE_MAX = 0.60

    DISCOVERY_JITTER = 0.02
    DISCOVERY_MIN = 0.0
    DISCOVERY_MAX = 0.10

    DURATION_SCALE_JITTER_FRAC = 0.20

    TASK_RANGE_JITTER = 4
    MIX_SIMPLE_PROB = 0.08  # Stage3/4时偶尔采样简单任务
    MIX_SIMPLE_RANGE = (10, 25)

    # ✅ 大幅提升OOD采样概率
    OOD_SPRINKLE_PROB = 0.25  # 从0.08提升到0.25
    OOD_TASKS_RANGE = (50, 64)

    # ✅ 新增：高任务数强化采样
    HIGH_TASK_BOOST = True
    HIGH_TASK_THRESHOLD = 50
    HIGH_TASK_PROB = 0.30  # 30%概率采样50+任务

    # 模型保存
    CHECKPOINT_DIR = "./checkpoints_3_6"
    SAVE_EVERY_EP = 500
    SAVE_BEST_METRIC = "reward_total"


# ========== ✅ 动态Padding核心函数 ==========

def pad_to_batch_max(batch_data_list):
    """
    将一个batch内的变长数据pad到batch内最大长度

    Args:
        batch_data_list: List[Tensor], 每个元素shape=[N_i, D]
    Returns:
        padded_tensor: [B, max_N, D]
        lengths: [B] - 每个样本的实际长度
    """
    if len(batch_data_list) == 0:
        return None, None

    batch_size = len(batch_data_list)
    max_len = max(x.shape[0] for x in batch_data_list)
    feature_dim = batch_data_list[0].shape[-1] if batch_data_list[0].dim() > 1 else 1

    # 创建padding后的张量
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


def collate_batch_dynamic(buffer, indices, device):
    batch_states = []
    batch_neighbors = []
    batch_tasks = []
    batch_masks = []
    batch_actions = []
    batch_logprobs = []
    batch_rewards = []
    batch_dones = []
    batch_g_agents = []
    batch_g_tasks = []
    batch_task_pad_lengths = []

    for idx in indices:
        batch_states.append(buffer["states"][idx])
        batch_neighbors.append(buffer["neighbors"][idx])
        batch_tasks.append(buffer["tasks"][idx])
        batch_masks.append(buffer["masks"][idx])
        batch_actions.append(buffer["actions"][idx])
        batch_logprobs.append(buffer["logprobs"][idx])
        batch_rewards.append(buffer["rewards"][idx])
        batch_dones.append(buffer["dones"][idx])
        batch_g_agents.append(buffer["g_agents"][idx])
        batch_g_tasks.append(buffer["g_tasks"][idx])
        batch_task_pad_lengths.append(buffer["task_pad_lengths"][idx])

    # ✅ 诊断1：检查从 buffer 取出的原始长度 vs 记录的 pad 长度
    original_lens = [t.shape[0] for t in batch_tasks]
    """
    print(f"  [collate诊断1] 从buffer取出的原始任务数: "
          f"min={min(original_lens)}, max={max(original_lens)}, mean={np.mean(original_lens):.1f}")
    print(f"  [collate诊断1] 记录的task_pad_lengths: "
          f"min={min(batch_task_pad_lengths)}, max={max(batch_task_pad_lengths)}, "
          f"mean={np.mean(batch_task_pad_lengths):.1f}")
    print(f"  [collate诊断1] 原始长度==pad长度的样本数: "
          f"{sum(a==b for a,b in zip(original_lens, batch_task_pad_lengths))}/{len(indices)}")
    """
    states = torch.stack(batch_states).to(device)
    neighbors = torch.stack(batch_neighbors).to(device)

    # ✅ 核心修复：按采样时的 pad 长度还原每个样本的 tasks 和 masks
    tasks_restored = []
    masks_restored = []
    for t, m, pad_len in zip(batch_tasks, batch_masks, batch_task_pad_lengths):
        actual_len = t.shape[0]

        if actual_len >= pad_len:
            tasks_restored.append(t[:pad_len])
            masks_restored.append(m[:pad_len])
        else:
            pad_size = pad_len - actual_len
            t_restored = torch.cat([t, torch.zeros(pad_size, t.shape[1], dtype=torch.float32)], dim=0)
            m_restored = torch.cat([m, torch.ones(pad_size, dtype=torch.bool)], dim=0)
            tasks_restored.append(t_restored)
            masks_restored.append(m_restored)

    # ✅ 诊断2：检查还原后的长度是否与记录的 pad 长度完全一致
    restored_lens = [t.shape[0] for t in tasks_restored]
    match = all(a == b for a, b in zip(restored_lens, batch_task_pad_lengths))

    # 原始就是变长，直接pad到batch内最大长度
    tasks, task_lengths = pad_to_batch_max(batch_tasks)
    tasks = tasks.to(device)

    masks_expanded = [m.unsqueeze(-1).float() for m in batch_masks]
    masks, _ = pad_to_batch_max(masks_expanded)
    masks = masks.squeeze(-1).bool().to(device)

    # ✅ 掩码-任务维度一致性断言
    assert tasks.shape[:2] == masks.shape[:2], \
        f"tasks/masks shape mismatch: {tasks.shape} vs {masks.shape}"

    actions = torch.tensor(batch_actions, dtype=torch.long).to(device)
    old_logprobs = torch.tensor(batch_logprobs, dtype=torch.float32).to(device)
    rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(batch_dones, dtype=torch.float32).to(device)

    g_agents, agent_lengths = pad_to_batch_max(batch_g_agents)
    g_tasks, task_lengths_global = pad_to_batch_max(batch_g_tasks)
    g_agents = g_agents.to(device)
    g_tasks = g_tasks.to(device)

    B = len(indices)
    max_agent_len = g_agents.shape[1]
    max_task_len = g_tasks.shape[1]

    agent_padding_mask = torch.ones(B, max_agent_len, dtype=torch.bool)
    for i, l in enumerate(agent_lengths):
        agent_padding_mask[i, :l] = False

    task_padding_mask = torch.ones(B, max_task_len, dtype=torch.bool)
    for i, l in enumerate(task_lengths_global):
        task_padding_mask[i, :l] = False

    critic_padding_mask = torch.cat(
        [agent_padding_mask, task_padding_mask], dim=1
    ).to(device)

    return {
        'states': states,
        'neighbors': neighbors,
        'tasks': tasks,
        'masks': masks,
        'actions': actions,
        'old_logprobs': old_logprobs,
        'rewards': rewards,
        'dones': dones,
        'g_agents': g_agents,
        'g_tasks': g_tasks,
        'task_lengths': task_lengths,
        'critic_padding_mask': critic_padding_mask,
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
    """✅ 改进的采样策略：4个Stage + 更高OOD概率"""

    # Stage判断
    if ep < params.CURR_STAGE_2:
        stage = 1
        base_comm = params.COMM_RANGE_STAGE_1
        base_tasks = params.TASKS_RANGE_STAGE_1
        base_dur = params.DURATION_SCALE_STAGE_1
        base_disc = params.DISCOVERY_PROB_STAGE_1
        base_tightness = params.DEADLINE_TIGHTNESS_STAGE_1
    elif ep < params.CURR_STAGE_3:
        stage = 2
        base_comm = params.COMM_RANGE_STAGE_2
        base_tasks = params.TASKS_RANGE_STAGE_2
        base_dur = params.DURATION_SCALE_STAGE_2
        base_disc = params.DISCOVERY_PROB_STAGE_2
        base_tightness = params.DEADLINE_TIGHTNESS_STAGE_2
    elif ep < params.CURR_STAGE_4:
        stage = 3
        base_comm = params.COMM_RANGE_STAGE_3
        base_tasks = params.TASKS_RANGE_STAGE_3
        base_dur = params.DURATION_SCALE_STAGE_3
        base_disc = params.DISCOVERY_PROB_STAGE_3
        base_tightness = params.DEADLINE_TIGHTNESS_STAGE_3
    else:
        stage = 4
        base_comm = params.COMM_RANGE_STAGE_4
        base_tasks = params.TASKS_RANGE_STAGE_4
        base_dur = params.DURATION_SCALE_STAGE_4
        base_disc = params.DISCOVERY_PROB_STAGE_4
        base_tightness = params.DEADLINE_TIGHTNESS_STAGE_4

    tasks_rng = base_tasks

    # ✅ Stage3/4: 混合采样（防止遗忘简单任务）
    if stage >= 3:
        r = rng.random()
        if r < params.MIX_SIMPLE_PROB:                          # 8%
            tasks_rng = params.MIX_SIMPLE_RANGE
        elif r < params.MIX_SIMPLE_PROB + params.OOD_SPRINKLE_PROB:  # 8%~33%
            tasks_rng = params.OOD_TASKS_RANGE
        elif r < params.MIX_SIMPLE_PROB + params.OOD_SPRINKLE_PROB + params.HIGH_TASK_PROB:  # 33%~63%
            tasks_rng = (params.HIGH_TASK_THRESHOLD, base_tasks[1])
        # else: 使用 base_tasks（37%）

    # Jitter
    tasks_rng = jitter_int_range(tasks_rng, params.TASK_RANGE_JITTER, lo=5, hi=100)

    comm = base_comm + rng.uniform(-params.COMM_RANGE_JITTER, params.COMM_RANGE_JITTER)
    comm = float(clamp(comm, params.COMM_RANGE_MIN, params.COMM_RANGE_MAX))

    disc = base_disc + rng.uniform(-params.DISCOVERY_JITTER, params.DISCOVERY_JITTER)
    disc = float(clamp(disc, params.DISCOVERY_MIN, params.DISCOVERY_MAX))

    dur = base_dur * (1.0 + rng.uniform(-params.DURATION_SCALE_JITTER_FRAC, params.DURATION_SCALE_JITTER_FRAC))
    dur = float(max(1.0, dur))

    return stage, comm, tasks_rng, dur, disc, base_tightness


def update(net, optimizer, buffer, device, params, entropy_weight):
    """
    PPO update (dynamic padding) - KL稳健版
    关键改动：
    1) 不再因 approx_kl > TARGET_KL 直接中断整个epoch
    2) 使用 KL penalty 而不是激进 early-stop
    3) 仅当 KL 极端离谱时跳过当前 batch（不影响其它batch）
    """

    buffer_size = len(buffer["states"])
    if buffer_size < params.BATCH_SIZE:
        return

    # 兼容旧参数：如果 TrainParams 未定义这些字段，给默认值
    target_kl = float(getattr(params, "TARGET_KL", 0.3))
    kl_coef = float(getattr(params, "KL_COEF", 0.02))
    max_kl_for_skip = float(getattr(params, "MAX_KL_FOR_SKIP", 2.0))

    update_success = False
    total_batches_updated = 0

    try:
        # ================================================================
        # 1) 全量 buffer 按时序计算 returns
        # ================================================================
        all_returns = []
        discounted = 0.0
        for rew, done in zip(reversed(buffer["rewards"]), reversed(buffer["dones"])):
            if done:
                discounted = 0.0
            discounted = rew + params.GAMMA * discounted
            all_returns.append(discounted)
        all_returns.reverse()
        all_returns_t = torch.tensor(all_returns, dtype=torch.float32, device=device)

        # returns normalize
        returns_mean = all_returns_t.mean()
        returns_std = all_returns_t.std() + 1e-8
        all_returns_normalized = (all_returns_t - returns_mean) / returns_std

        # ================================================================
        # 2) 全量 old values + advantages
        # ================================================================
        all_old_values = []
        full_indices = list(range(buffer_size))

        net.eval()
        for start in range(0, buffer_size, params.BATCH_SIZE):
            idx_chunk = full_indices[start: start + params.BATCH_SIZE]
            chunk = collate_batch_dynamic(buffer, idx_chunk, device)
            with torch.no_grad():
                v = net.get_value(
                    chunk["g_agents"],
                    chunk["g_tasks"],
                    padding_mask=chunk["critic_padding_mask"],
                ).squeeze(-1)
            all_old_values.append(v)

        all_old_values_t = torch.cat(all_old_values, dim=0)

        advantages_all = all_returns_normalized - all_old_values_t.detach()
        if buffer_size > 1:
            advantages_all = (advantages_all - advantages_all.mean()) / (advantages_all.std() + 1e-8)
        advantages_all = advantages_all.detach()

        # ================================================================
        # 3) K_EPOCHS mini-batch update
        # ================================================================
        for epoch_i in range(params.K_EPOCHS):
            perm = np.random.permutation(buffer_size)

            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0
            epoch_entropy = 0.0
            epoch_kl_sum = 0.0
            epoch_kl_pen_sum = 0.0
            n_batches = 0
            skipped_batches = 0

            for start in range(0, buffer_size, params.BATCH_SIZE):
                indices = perm[start: start + params.BATCH_SIZE]
                if len(indices) < 4:
                    continue

                batch = collate_batch_dynamic(buffer, indices, device)

                row = torch.arange(batch["actions"].shape[0], device=device)
                illegal_chosen = batch["masks"][row, batch["actions"]]  # True表示选了被mask动作
                if illegal_chosen.any():
                    bad_idx = torch.where(illegal_chosen)[0][:8].tolist()
                    raise RuntimeError(f"Masked action found in replay batch, idx={bad_idx}")

                returns_chunk = all_returns_normalized[indices]
                advantages_chunk = advantages_all[indices]
                old_logprobs_fixed = batch["old_logprobs"].detach()

                net.eval()
                dist = net.get_action(
                    batch["states"],
                    batch["neighbors"],
                    batch["tasks"],
                    batch["masks"],
                )
                new_logprobs = dist.log_prob(batch["actions"])
                entropy = dist.entropy().sum(-1).mean()

                new_values = net.get_value(
                    batch["g_agents"],
                    batch["g_tasks"],
                    padding_mask=batch["critic_padding_mask"],
                ).squeeze(-1)

                # 数值稳定
                log_ratio = torch.clamp(new_logprobs - old_logprobs_fixed, min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(torch.clamp(ratio, min=1e-8))).mean()

                # 仅极端异常的batch跳过，不终止训练
                if torch.isnan(approx_kl) or torch.isinf(approx_kl) or (approx_kl.item() > max_kl_for_skip):
                    skipped_batches += 1
                    continue

                surr1 = ratio * advantages_chunk
                surr2 = torch.clamp(ratio, 1 - params.EPS_CLIP, 1 + params.EPS_CLIP) * advantages_chunk

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = ((new_values - returns_chunk) ** 2).mean()
                entropy_loss = -float(entropy_weight) * entropy

                # KL penalty（超过target才罚）
                kl_excess = torch.clamp(approx_kl - target_kl, min=0.0)
                kl_penalty = kl_coef * kl_excess

                loss = actor_loss + critic_loss + entropy_loss + kl_penalty

                net.train()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), params.GRAD_NORM)
                optimizer.step()

                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()
                epoch_entropy += entropy.item()
                epoch_kl_sum += approx_kl.item()
                epoch_kl_pen_sum += kl_penalty.item() if torch.is_tensor(kl_penalty) else float(kl_penalty)
                n_batches += 1
                total_batches_updated += 1

            if n_batches > 0:
                print(
                    f"  [Epoch {epoch_i}] batches={n_batches}, skipped={skipped_batches}, "
                    f"actor={epoch_actor_loss / n_batches:.4f}, "
                    f"critic={epoch_critic_loss / n_batches:.4f}, "
                    f"entropy={epoch_entropy / n_batches:.4f}, "
                    f"kl={epoch_kl_sum / n_batches:.4f}, "
                    f"kl_pen={epoch_kl_pen_sum / n_batches:.6f}"
                )
            else:
                print(f"  [Epoch {epoch_i}] ⚠️ 无有效batch（全部跳过），提前结束本次update")
                break

        update_success = True

    except Exception as e:
        print(f"❌ Update error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # on-policy: 一次update后清空buffer，防止陈旧样本反复训练
        if update_success:
            for key in buffer.keys():
                buffer[key] = []

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
    print("CTDE Training with Dynamic Padding")
    print("=" * 60)
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() and TrainParams.USE_GPU else "cpu")
    print(f"Using device: {device}")

    log_dir = os.path.join(r"D:\tb_logs", time.strftime("ctde_dynamic_%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard log:  {log_dir}")

    os.makedirs(TrainParams.CHECKPOINT_DIR, exist_ok=True)

    # ✅ 初始化环境（使用Stage1设置）
    env = TaskEnv(
        deadline_tightness=TrainParams.DEADLINE_TIGHTNESS_STAGE_1,
        per_species_range=(3, 5),
        species_range=(2, 3),
        tasks_range=TrainParams.TASKS_RANGE_STAGE_1,
        comm_range=TrainParams.COMM_RANGE_STAGE_1,
        dynamic_task_discovery=True,
        discovery_probability=TrainParams.DISCOVERY_PROB_STAGE_1,
        max_dynamic_tasks=5,
        seed=None,
        dt=TrainParams.DT,
    )
    print("✅ Environment created")

    # ✅ 网络（启用动态长度支持）
    net = CTDE_Network(
        self_dim=10,
        neighbor_dim=10,
        task_dim=15,
        global_agent_dim=13,
        global_task_dim=18,
        hidden=128,
        dropout=0.15,
        use_pos_encoding=True,
    ).to(device)
    print("✅ Network created with dynamic length support")

    optimizer = optim.Adam(net.parameters(), lr=TrainParams.LR)

    buffer = {
        "states": [],
        "neighbors": [],
        "tasks": [],
        "masks": [],
        "actions": [],
        "logprobs": [],
        "rewards": [],
        "dones": [],
        "g_agents": [],
        "g_tasks": [],
        "task_pad_lengths": [],
    }

    timestep = 0
    best_reward = -1e18

    # ✅ 任务数量统计
    task_count_history = deque(maxlen=100)

    try:
        for ep in range(TrainParams.MAX_EPISODES):
            rng = np.random.default_rng(seed=ep * 9973 + 123)

            # ✅ 采样curriculum设置
            stage, comm, tasks_rng, dur_scale, disc_prob, tightness = sample_curriculum_settings(
                ep, TrainParams, rng
            )

            env.comm_range = comm
            env.ranges["tasks"] = tasks_rng
            env.ranges["deadline_tightness"] = tightness
            env.duration_scale = dur_scale
            env.discovery_probability = disc_prob

            if ep % 100 == 0 or ep in (0, TrainParams.CURR_STAGE_2, TrainParams.CURR_STAGE_3, TrainParams.CURR_STAGE_4):
                print(
                    f"\n📈 Episode {ep} | Stage {stage} | "
                    f"Tasks:{tasks_rng} | Comm:{comm:.2f} | "
                    f"Dur:{dur_scale:.1f} | Disc:{disc_prob:.3f}"
                )

            env.reset(seed=ep)
            ep_step_reward = 0.0   # step-wise 累加（用于 debug）
            step_count = 0
            episode_start_time = time.time()
            ep_terminal_reward = 0.0
            finished_tasks = []
            success_rate = 0.0
            ep_reward = 0.0

            entropy_w = get_entropy_weight(ep, TrainParams)

            # ✅ 记录本episode的任务数
            actual_tasks = len(env.task_dic)
            task_count_history.append(actual_tasks)

            #  用于更新最后buffer[dones]最后一个的状态
            ep_start_idx = len(buffer["states"])
            # 用于计算timestep的步数
            prev_buffer_size = len(buffer["states"])

            while not env.finished and step_count < TrainParams.MAX_STEP_PER_EPISODE:
                if env.current_time > TrainParams.ENV_TIME_LIMIT:
                    env.finished = True
                    break
                if time.time() - episode_start_time > TrainParams.EPISODE_TIMEOUT:
                    env.finished = True
                    break

                decision_ids = [aid for aid, a in env.agent_dic.items() if a.get("need_decision", False)]

                if len(decision_ids) > 0:
                    decision_ids = sorted(decision_ids)

                    # ========== ⭐ Batch 优化 ==========
                    # 1) 批量收集所有 agent 的局部观测
                    all_obs = []
                    for aid in decision_ids:
                        s, n, t, m, amap = env.get_local_obs(aid)
                        all_obs.append((aid, s, n, t, m, amap))

                    # 2) 获取全局状态（只需取一次，步内不会变化）
                    g_agents, g_tasks = env.get_global_state()
                    g_agents_t = torch.as_tensor(g_agents, dtype=torch.float32)
                    g_tasks_t = torch.as_tensor(g_tasks, dtype=torch.float32)

                    # 3) 组装 batch 观测（动态 padding 到 batch 内最大任务数）
                    max_tasks = max(obs[3].shape[0] for obs in all_obs)

                    states_np = np.stack([obs[1] for obs in all_obs], axis=0)       # [B, self_dim]
                    neighbors_np = np.stack([obs[2] for obs in all_obs], axis=0)     # [B, 20, 10]

                    tasks_list, masks_list = [], []
                    for obs in all_obs:
                        t, m = obs[3], obs[4]
                        if t.shape[0] < max_tasks:
                            pad_len = max_tasks - t.shape[0]
                            t = np.vstack([t, np.zeros((pad_len, t.shape[1]), dtype=np.float32)])
                            m = np.concatenate([m, np.ones(pad_len, dtype=bool)])
                        tasks_list.append(t)
                        masks_list.append(m)

                    # 一次性搬到 GPU（4 次传输，代替原来的 4×N 次）
                    ts_batch = torch.as_tensor(states_np, dtype=torch.float32, device=device)
                    tn_batch = torch.as_tensor(neighbors_np, dtype=torch.float32, device=device)
                    tt_batch = torch.as_tensor(np.stack(tasks_list), dtype=torch.float32, device=device)
                    tm_batch = torch.as_tensor(np.stack(masks_list), dtype=torch.bool, device=device)

                    # 4) 单次前向传播（所有 agent 并行决策）
                    net.eval()
                    with torch.no_grad():
                        dist = net.get_action(ts_batch, tn_batch, tt_batch, tm_batch)
                        actions = dist.sample()          # [B]
                        logprobs = dist.log_prob(actions)  # [B]
                    net.train()

                    # 5) 逐个执行动作（不重复通信）
                    for i, aid in enumerate(decision_ids):
                        act = int(actions[i].item())
                        _, _, _, _, _, amap = all_obs[i]
                        reward, _need_replan, _info = env.apply_action(aid, act, amap)
                        ep_step_reward += float(reward)

                        # 存 buffer（直接 CPU tensor，避免 GPU→CPU 搬运）
                        buffer["states"].append(torch.as_tensor(all_obs[i][1], dtype=torch.float32))
                        buffer["neighbors"].append(torch.as_tensor(all_obs[i][2], dtype=torch.float32))
                        t_cpu = torch.as_tensor(all_obs[i][3], dtype=torch.float32)
                        m_cpu = torch.as_tensor(all_obs[i][4], dtype=torch.bool)
                        buffer["tasks"].append(t_cpu)
                        buffer["masks"].append(m_cpu)
                        buffer["actions"].append(act)
                        buffer["logprobs"].append(float(logprobs[i].item()))
                        buffer["rewards"].append(0.0)
                        buffer["dones"].append(False)
                        buffer["g_agents"].append(g_agents_t)
                        buffer["g_tasks"].append(g_tasks_t)
                        buffer["task_pad_lengths"].append(int(t_cpu.shape[0]))

                    # 6) 所有 agent 决策完成，统一通信一次
                    env.update_communication_graph()
                    env.propagate_task_completion()

                env.step(dt=TrainParams.DT)
                ep_step_reward += float(getattr(env, "last_step_reward", 0.0))
                if step_count % 10 == 0:
                    env.cleanup_failed_tasks_memory()

                """
                在回合结束后更新网络
                if timestep >= TrainParams.UPDATE_STEP:
                    update(net, optimizer, buffer, device, TrainParams, entropy_weight=entropy_w)
                    timestep = 0
                """
                step_count += 1

            #  更新buffer[dones]最后一个的状态
            ep_end_idx = len(buffer["states"])
            if ep_end_idx > ep_start_idx:
                buffer["dones"][ep_end_idx - 1] = True
                # 先计算终局奖励
                ep_terminal_reward, finished_tasks = env.get_episode_reward()
                ep_reward = float(ep_terminal_reward)
                success_rate = (sum(finished_tasks) / len(finished_tasks)) if finished_tasks else 0.0
                # 方案A（推荐）：整段都赋同一个终局奖励
                for i in range(ep_start_idx, ep_end_idx):
                    buffer["rewards"][i] = ep_reward
            else:
                ep_terminal_reward = 0.0
                ep_reward = 0.0
                success_rate = 0.0



            # ✅ 积累足够数据后触发一次完整的 PPO 更新
            if len(buffer["states"]) >= TrainParams.UPDATE_STEP:
                update(net, optimizer, buffer, device, TrainParams, entropy_weight=entropy_w)
                # buffer 在 update 内部清空，无需手动置零 timestep


            failed_count = sum(len(a.get("failed_tasks", [])) for a in env.agent_dic.values())
            dynamic_count = sum(1 for s in env.task_source.values() if s == "discovered")
            expired_count = sum(1 for t in env.task_dic.values() if t.get("expired", False))
            expired_rate = expired_count / max(1, len(env.task_dic))

            # TensorBoard
            writer.add_scalar("episode/reward_total", ep_reward, ep)
            writer.add_scalar("episode/success_rate", success_rate, ep)
            writer.add_scalar("episode/env_time", env.current_time, ep)
            writer.add_scalar("episode/failed_count", failed_count, ep)
            writer.add_scalar("episode/dynamic_tasks", dynamic_count, ep)
            writer.add_scalar("episode/expired_count", expired_count, ep)
            writer.add_scalar("episode/expired_rate", expired_rate, ep)
            writer.add_scalar("episode/reward_step_sum", ep_step_reward, ep)


            # ✅ 任务数量监控
            writer.add_scalar("episode/actual_task_count", actual_tasks, ep)
            writer.add_scalar("curriculum/stage", stage, ep)
            writer.add_scalar("curriculum/entropy_weight", float(entropy_w), ep)
            writer.add_scalar("curriculum/comm_range", float(env.comm_range), ep)
            writer.add_scalar("curriculum/duration_scale", float(env.duration_scale), ep)
            writer.add_scalar("curriculum/discovery_probability", float(env.discovery_probability), ep)
            writer.add_scalar("curriculum/tasks_min", float(env.ranges["tasks"][0]), ep)
            writer.add_scalar("curriculum/tasks_max", float(env.ranges["tasks"][1]), ep)

            # ✅ 任务数分布（每100 episodes）
            if ep > 0 and ep % 100 == 0:
                recent_tasks = list(task_count_history)
                bins = [0, 20, 35, 50, 64, 100]
                hist, _ = np.histogram(recent_tasks, bins=bins)
                print(
                    f"  Task distribution (last 100): <20={hist[0]}, 20-35={hist[1]}, 35-50={hist[2]}, 50-64={hist[3]}, 64+={hist[4]}")

            # Save models
            if ep_reward > best_reward:
                best_reward = ep_reward
                save_checkpoint(
                    path=os.path.join(TrainParams.CHECKPOINT_DIR, "ctde_best.pth"),
                    net=net,
                    optimizer=optimizer,
                    ep=ep,
                    best_metric=best_reward,
                    extra={
                        "success_rate": float(success_rate),
                        "final_reward": float(ep_terminal_reward),
                        "env_time": float(env.current_time),
                    },
                )

            if ep % TrainParams.SAVE_EVERY_EP == 0 and ep > 0:
                save_checkpoint(
                    path=os.path.join(TrainParams.CHECKPOINT_DIR, f"ctde_ep_{ep}.pth"),
                    net=net,
                    optimizer=optimizer,
                    ep=ep,
                    best_metric=None,
                    extra={"success_rate": float(success_rate), "final_reward": float(ep_terminal_reward)},
                )

            if ep % 20 == 0:
                print(
                    f"Ep {ep} | R:{ep_reward:.1f} | Succ:{success_rate:.2%} | "
                    f"Time:{env.current_time:.0f} | Fail:{failed_count} | Exp:{expired_count} | "
                    f"Tasks:{actual_tasks} | Best:{best_reward:.1f}"
                )

        save_checkpoint(
            path=os.path.join(TrainParams.CHECKPOINT_DIR, "ctde_final.pth"),
            net=net,
            optimizer=optimizer,
            ep=TrainParams.MAX_EPISODES - 1,
        )

    finally:
        writer.close()
        print("\n✅ Training completed!")


if __name__ == "__main__":
    main()