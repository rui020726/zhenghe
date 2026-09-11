# -*- coding: utf-8 -*-
"""
CTDE 训练主程序（消融实验一键切换版）
- ✅ 新增：PRE_POSITIONING 开关，一键切换 Baseline 和 Proposed
- ✅ 自动隔离权重文件：保存在 checkpoints_ablation/baseline 或 prepos 文件夹下
- ✅ 自动匹配 15/18 维 或 20/23 维网络
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict

# ✅ 引入消融版环境和网络
from Ab_task_env import TaskEnv
from Ab_old_network import CTDE_Network
from torch.utils.tensorboard import SummaryWriter


class TrainParams:
    # ========================================================
    # 🌟 核心消融开关：训练时只要改这个 True / False 即可！
    # ========================================================
    PRE_POSITIONING = False  # True: 开启预判机制; False: 关闭预判(Baseline)

    LR = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    K_EPOCHS = 4
    EPS_CLIP = 0.2
    MAX_EPISODES = 20000
    MINI_BATCH_SIZE = 256
    UPDATE_STEP = 2048
    USE_GPU = True

    ENTROPY_START = 0.02
    ENTROPY_END = 0.003
    ENTROPY_DECAY_END_EP = 8000

    GRAD_NORM = 1.0
    MAX_AGENTS = 30

    MAX_STEP_PER_EPISODE = 1000
    STEP_TIMEOUT = 30.0
    EPISODE_TIMEOUT = 120.0
    ENV_TIME_LIMIT = 1000.0
    STEP_LOG_INTERVAL = 200
    DT = 1.0

    # 5阶段课程学习
    CURR_STAGE_1 = 0
    CURR_STAGE_2 = 500
    CURR_STAGE_3 = 1500
    CURR_STAGE_4 = 5000
    CURR_STAGE_5 = 11000

    TASKS_RANGE_STAGE_1 = (8, 15)
    TASKS_RANGE_STAGE_2 = (12, 30)
    TASKS_RANGE_STAGE_3 = (25, 50)
    TASKS_RANGE_STAGE_4 = (40, 64)
    TASKS_RANGE_STAGE_5 = (50, 80)

    DURATION_SCALE_STAGE_1 = 3
    DURATION_SCALE_STAGE_2 = 5
    DURATION_SCALE_STAGE_3 = 7
    DURATION_SCALE_STAGE_4 = 9
    DURATION_SCALE_STAGE_5 = 10

    DISCOVERY_PROB_STAGE_1 = 0.01
    DISCOVERY_PROB_STAGE_2 = 0.03
    DISCOVERY_PROB_STAGE_3 = 0.05
    DISCOVERY_PROB_STAGE_4 = 0.07
    DISCOVERY_PROB_STAGE_5 = 0.08

    COMM_RANGE_STAGE_1 = 0.5
    COMM_RANGE_STAGE_2 = 0.4
    COMM_RANGE_STAGE_3 = 0.35
    COMM_RANGE_STAGE_4 = 0.28
    COMM_RANGE_STAGE_5 = 0.25

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
    OOD_TASKS_RANGE = (55, 75)
    OOD_STAGE5_PROB = 0.35
    OOD_STAGE5_RANGE = (65, 90)
    HIGH_TASK_BOOST = True
    HIGH_TASK_THRESHOLD = 50
    HIGH_TASK_PROB = 0.25
    TERMINAL_REWARD_MODE = "spread"

    SAVE_EVERY_EP = 1000
    SAVE_BEST_METRIC = "reward_total"

    LR_DECAY = True
    LR_DECAY_START = 14000
    LR_DECAY_END = 20000
    LR_MIN = 5e-5

    # ✅ 自动分离保存路径，不会互相覆盖
    @classmethod
    def get_checkpoint_dir(cls):
        base = "./checkpoints_ablation"
        suffix = "prepos" if cls.PRE_POSITIONING else "baseline"
        return os.path.join(base, suffix)


def pad_to_batch_max(batch_data_list):
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
    tasks, task_lengths = pad_to_batch_max(tt_list)
    tasks = tasks.to(device)

    masks_expanded = [m.unsqueeze(-1).float() for m in batch_masks]
    masks, _ = pad_to_batch_max(masks_expanded)
    masks = masks.squeeze(-1).bool().to(device)

    actions = torch.tensor(batch_actions, dtype=torch.long).to(device)
    old_logprobs = torch.tensor(batch_logprobs, dtype=torch.float32).to(device)
    old_values = torch.tensor(batch_values, dtype=torch.float32).to(device)
    rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(batch_dones, dtype=torch.float32).to(device)

    g_agents, agent_lengths = pad_to_batch_max(batch_g_agents)
    g_tasks, task_lengths_global = pad_to_batch_max(batch_g_tasks)
    g_agents = g_agents.to(device)
    g_tasks = g_tasks.to(device)

    return {
        'states': states, 'neighbors': neighbors, 'tasks': tasks, 'masks': masks,
        'actions': actions, 'old_logprobs': old_logprobs, 'old_values': old_values,
        'rewards': rewards, 'dones': dones, 'g_agents': g_agents, 'g_tasks': g_tasks,
        'task_lengths': task_lengths,
    }


def get_entropy_weight(ep: int, params: TrainParams) -> float:
    if ep <= 0: return float(params.ENTROPY_START)
    if ep >= params.ENTROPY_DECAY_END_EP: return float(params.ENTROPY_END)
    a = ep / float(params.ENTROPY_DECAY_END_EP)
    return float(params.ENTROPY_START * (1 - a) + params.ENTROPY_END * a)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def jitter_int_range(rng, jitter, lo=1, hi=200):
    a, b = int(rng[0]), int(rng[1])
    a2 = clamp(a - jitter, lo, hi)
    b2 = clamp(b + jitter, lo, hi)
    if a2 > b2: a2, b2 = b2, a2
    return (int(a2), int(b2))


def sample_curriculum_settings(ep: int, params: TrainParams, rng: np.random.Generator):
    if ep < params.CURR_STAGE_2:
        stage, base_comm, base_tasks, base_dur, base_disc = 1, params.COMM_RANGE_STAGE_1, params.TASKS_RANGE_STAGE_1, params.DURATION_SCALE_STAGE_1, params.DISCOVERY_PROB_STAGE_1
    elif ep < params.CURR_STAGE_3:
        stage, base_comm, base_tasks, base_dur, base_disc = 2, params.COMM_RANGE_STAGE_2, params.TASKS_RANGE_STAGE_2, params.DURATION_SCALE_STAGE_2, params.DISCOVERY_PROB_STAGE_2
    elif ep < params.CURR_STAGE_4:
        stage, base_comm, base_tasks, base_dur, base_disc = 3, params.COMM_RANGE_STAGE_3, params.TASKS_RANGE_STAGE_3, params.DURATION_SCALE_STAGE_3, params.DISCOVERY_PROB_STAGE_3
    elif ep < params.CURR_STAGE_5:
        stage, base_comm, base_tasks, base_dur, base_disc = 4, params.COMM_RANGE_STAGE_4, params.TASKS_RANGE_STAGE_4, params.DURATION_SCALE_STAGE_4, params.DISCOVERY_PROB_STAGE_4
    else:
        stage, base_comm, base_tasks, base_dur, base_disc = 5, params.COMM_RANGE_STAGE_5, params.TASKS_RANGE_STAGE_5, params.DURATION_SCALE_STAGE_5, params.DISCOVERY_PROB_STAGE_5

    tasks_rng = base_tasks
    if stage >= 3 and rng.random() < params.MIX_SIMPLE_PROB:
        tasks_rng = params.MIX_SIMPLE_RANGE
    elif stage in (3, 4) and rng.random() < params.OOD_SPRINKLE_PROB:
        tasks_rng = params.OOD_TASKS_RANGE
    elif stage == 5 and rng.random() < params.OOD_STAGE5_PROB:
        tasks_rng = params.OOD_STAGE5_RANGE

    if stage in (3, 4) and params.HIGH_TASK_BOOST and rng.random() < params.HIGH_TASK_PROB:
        tasks_rng = (params.HIGH_TASK_THRESHOLD, base_tasks[1])

    tasks_rng = jitter_int_range(tasks_rng, params.TASK_RANGE_JITTER, lo=5, hi=100)
    comm = float(
        clamp(base_comm + rng.uniform(-params.COMM_RANGE_JITTER, params.COMM_RANGE_JITTER), params.COMM_RANGE_MIN,
              params.COMM_RANGE_MAX))
    disc = float(clamp(base_disc + rng.uniform(-params.DISCOVERY_JITTER, params.DISCOVERY_JITTER), params.DISCOVERY_MIN,
                       params.DISCOVERY_MAX))
    dur = float(
        max(1.0, base_dur * (1.0 + rng.uniform(-params.DURATION_SCALE_JITTER_FRAC, params.DURATION_SCALE_JITTER_FRAC))))

    return stage, comm, tasks_rng, dur, disc


def inject_terminal_reward(buffer, terminal_reward, ep_start_idx, mode="spread"):
    n = len(buffer["rewards"])
    ep_len = n - ep_start_idx
    if ep_len <= 0 or terminal_reward == 0.0: return
    if mode == "last":
        buffer["rewards"][-1] += terminal_reward
    elif mode == "spread":
        per_step = terminal_reward / ep_len
        for i in range(ep_start_idx, n): buffer["rewards"][i] += per_step
    elif mode == "weighted_spread":
        weights = np.arange(1, ep_len + 1, dtype=np.float64)
        weights = weights / weights.sum()
        for i, w in enumerate(weights): buffer["rewards"][ep_start_idx + i] += terminal_reward * w
    else:
        per_step = terminal_reward / ep_len
        for i in range(ep_start_idx, n): buffer["rewards"][i] += per_step


def compute_gae_segmented(buffer, gamma, gae_lambda, device, net):
    n = len(buffer["states"])
    if n == 0: return [], []
    rewards, dones, values = buffer["rewards"], buffer["dones"], buffer["values"]
    advantages, returns = [0.0] * n, [0.0] * n
    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1 or dones[t]:
            next_value, last_gae = 0.0, 0.0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1.0 - float(dones[t])) - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - float(dones[t])) * last_gae
        advantages[t] = last_gae
        returns[t] = last_gae + values[t]
    return advantages, returns


def update(net, optimizer, buffer, device, params: TrainParams, entropy_weight: float):
    n = len(buffer["states"])
    if n < params.MINI_BATCH_SIZE: return {}
    advantages_list, returns_list = compute_gae_segmented(buffer, params.GAMMA, params.GAE_LAMBDA, device, net)
    buffer["advantages"], buffer["returns"] = advantages_list, returns_list
    total_actor_loss, total_critic_loss, total_entropy, num_updates = 0.0, 0.0, 0.0, 0

    for epoch in range(params.K_EPOCHS):
        perm = np.random.permutation(n)
        for start in range(0, n, params.MINI_BATCH_SIZE):
            end = min(start + params.MINI_BATCH_SIZE, n)
            if end - start < 8: continue
            mb_indices = perm[start:end]
            try:
                batch = collate_batch_dynamic(buffer, mb_indices, device)
            except Exception as e:
                print(f"❌ Error in collate_batch_dynamic: {e}")
                continue

            mb_advantages = torch.tensor([buffer["advantages"][i] for i in mb_indices], dtype=torch.float32,
                                         device=device)
            mb_returns = torch.tensor([buffer["returns"][i] for i in mb_indices], dtype=torch.float32, device=device)
            if len(mb_advantages) > 1: mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8)

            dist = net.get_action(batch['states'], batch['neighbors'], batch['tasks'], batch['masks'])
            logprobs, entropy = dist.log_prob(batch['actions']), dist.entropy().mean()
            values = net.get_value(batch['g_agents'], batch['g_tasks']).squeeze(-1)

            ratio = torch.exp(logprobs - batch['old_logprobs'])
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - params.EPS_CLIP, 1 + params.EPS_CLIP) * mb_advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * ((values - mb_returns) ** 2).mean()
            entropy_loss = -float(entropy_weight) * entropy
            loss = actor_loss + critic_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), params.GRAD_NORM)
            optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            num_updates += 1

    for key in buffer.keys(): buffer[key] = []
    if num_updates > 0: return {"actor_loss": total_actor_loss / num_updates,
                                "critic_loss": total_critic_loss / num_updates, "entropy": total_entropy / num_updates}
    return {}


def get_lr(ep: int, params: TrainParams) -> float:
    if not params.LR_DECAY or ep < params.LR_DECAY_START: return params.LR
    if ep >= params.LR_DECAY_END: return params.LR_MIN
    progress = (ep - params.LR_DECAY_START) / (params.LR_DECAY_END - params.LR_DECAY_START)
    return params.LR * (1.0 - progress) + params.LR_MIN * progress


def save_checkpoint(path, net, optimizer, ep, best_metric=None, extra=None):
    ckpt = {"episode": ep, "model_state_dict": net.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
    if best_metric is not None: ckpt["best_metric"] = float(best_metric)
    if extra is not None: ckpt.update(extra)
    torch.save(ckpt, path)


def main():
    mode_str = "PROPOSED (预判开启, 20/23维)" if TrainParams.PRE_POSITIONING else "BASELINE (预判关闭, 15/18维)"
    print("=" * 60)
    print(f"🔥 CTDE Ablation Training 🔥")
    print(f"当前模式: {mode_str}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() and TrainParams.USE_GPU else "cpu")
    print(f"Using device: {device}")

    # 获取自动隔离的保存路径
    checkpoint_dir = TrainParams.get_checkpoint_dir()
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    # TensorBoard 日志也按模式分开
    log_suffix = "prepos" if TrainParams.PRE_POSITIONING else "baseline"
    log_dir = os.path.join(r"D:\tb_logs", time.strftime(f"ctde_ablation_{log_suffix}_%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard log:  {log_dir}")

    # ✅ 把开关传递给环境
    env = TaskEnv(
        prio_range=(0.3, 1.0),
        per_species_range=(3, 5),
        species_range=(2, 3),
        tasks_range=TrainParams.TASKS_RANGE_STAGE_1,
        comm_range=TrainParams.COMM_RANGE_STAGE_1,
        dynamic_task_discovery=True,
        discovery_probability=TrainParams.DISCOVERY_PROB_STAGE_1,
        max_dynamic_tasks=5,
        seed=None,
        dt=TrainParams.DT,
        pre_positioning=TrainParams.PRE_POSITIONING  # 核心开关
    )
    print("✅ Environment created")

    # ✅ 自动匹配网络维度
    t_dim = 20 if TrainParams.PRE_POSITIONING else 15
    gt_dim = 23 if TrainParams.PRE_POSITIONING else 18

    net = CTDE_Network(
        self_dim=10,
        neighbor_dim=10,
        task_dim=t_dim,  # 自动变化
        global_agent_dim=13,
        global_task_dim=gt_dim,  # 自动变化
        hidden=128,
        dropout=0.15,
        use_pos_encoding=True,
    ).to(device)
    print(f"✅ Network created with Task Dim: {t_dim}, Global Task Dim: {gt_dim}")

    optimizer = optim.Adam(net.parameters(), lr=TrainParams.LR)

    buffer = {k: [] for k in
              ["states", "neighbors", "tasks", "masks", "actions", "logprobs", "values", "rewards", "dones", "g_agents",
               "g_tasks"]}

    timestep = 0
    best_reward = -1e18
    task_count_history = []

    try:
        for ep in range(TrainParams.MAX_EPISODES):
            rng = np.random.default_rng(seed=ep * 9973 + 123)

            stage, comm, tasks_rng, dur_scale, disc_prob = sample_curriculum_settings(ep, TrainParams, rng)

            env.comm_range = comm
            env.ranges["tasks"] = tasks_rng
            env.duration_scale = dur_scale
            env.discovery_probability = disc_prob

            env.reset(seed=ep)
            ep_reward = 0.0
            step_count = 0
            episode_start_time = time.time()

            entropy_w = get_entropy_weight(ep, TrainParams)
            current_lr = get_lr(ep, TrainParams)
            for param_group in optimizer.param_groups: param_group['lr'] = current_lr

            actual_tasks = len(env.task_dic)
            task_count_history.append(actual_tasks)
            ep_start_idx = len(buffer["states"])

            while not env.finished and step_count < TrainParams.MAX_STEP_PER_EPISODE:
                if env.current_time > TrainParams.ENV_TIME_LIMIT: break
                if time.time() - episode_start_time > TrainParams.EPISODE_TIMEOUT: break

                decision_ids = [aid for aid, a in env.agent_dic.items() if a.get("need_decision", False)]

                if len(decision_ids) > 0:
                    batch_s, batch_n, batch_t, batch_m, action_maps = [], [], [], [], []

                    for aid in decision_ids:
                        s, n, t, m, amap = env.get_local_obs(aid)
                        batch_s.append(s);
                        batch_n.append(n);
                        batch_t.append(t);
                        batch_m.append(m.astype(np.bool_));
                        action_maps.append(amap)

                    g_agents, g_tasks = env.get_global_state()

                    ts = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in batch_s]).to(device)
                    tn = torch.stack([torch.as_tensor(n, dtype=torch.float32) for n in batch_n]).to(device)

                    tt_list = [torch.as_tensor(t, dtype=torch.float32) for t in batch_t]
                    tt, _ = pad_to_batch_max(tt_list)
                    tt = tt.to(device)

                    tm_list = [torch.as_tensor(m, dtype=torch.bool) for m in batch_m]
                    tm_expanded = [m.unsqueeze(-1).float() for m in tm_list]
                    tm, _ = pad_to_batch_max(tm_expanded)
                    tm = tm.squeeze(-1).bool().to(device)

                    tg_a = torch.as_tensor(g_agents, dtype=torch.float32, device=device).unsqueeze(0)
                    tg_t = torch.as_tensor(g_tasks, dtype=torch.float32, device=device).unsqueeze(0)

                    with torch.no_grad():
                        dist = net.get_action(ts, tn, tt, tm)
                        actions = dist.sample()
                        logprobs = dist.log_prob(actions)
                        v_scalar = float(net.get_value(tg_a, tg_t).squeeze(-1).item())

                    for i, aid in enumerate(decision_ids):
                        act = int(actions[i].item())
                        reward, _need_replan, _info = env.apply_action(aid, act, action_maps[i])
                        ep_reward += float(reward)

                        buffer["states"].append(ts[i].detach().cpu())
                        buffer["neighbors"].append(tn[i].detach().cpu())
                        buffer["tasks"].append(tt_list[i].detach().cpu())
                        buffer["masks"].append(tm_list[i].detach().cpu())
                        buffer["actions"].append(act)
                        buffer["logprobs"].append(float(logprobs[i].item()))
                        buffer["values"].append(v_scalar)
                        buffer["rewards"].append(float(reward))
                        buffer["dones"].append(False)
                        buffer["g_agents"].append(tg_a.detach().cpu().squeeze(0))
                        buffer["g_tasks"].append(tg_t.detach().cpu().squeeze(0))
                        timestep += 1

                env.step(dt=TrainParams.DT)
                if step_count % 10 == 0: env.cleanup_failed_tasks_memory()
                step_count += 1

            if len(buffer["dones"]) > 0: buffer["dones"][-1] = True

            try:
                final_reward, finished_tasks = env.get_episode_reward()
            except Exception:
                final_reward, finished_tasks = 0.0, []
            success_rate = (sum(finished_tasks) / len(finished_tasks)) if finished_tasks else 0.0

            inject_terminal_reward(buffer, terminal_reward=final_reward, ep_start_idx=ep_start_idx,
                                   mode=TrainParams.TERMINAL_REWARD_MODE)
            ep_reward += float(final_reward)

            if timestep >= TrainParams.UPDATE_STEP:
                update_stats = update(net, optimizer, buffer, device, TrainParams, entropy_weight=entropy_w)
                timestep = 0
                if update_stats:
                    writer.add_scalar("train/actor_loss", update_stats.get("actor_loss", 0), ep)
                    writer.add_scalar("train/critic_loss", update_stats.get("critic_loss", 0), ep)
                    writer.add_scalar("train/entropy", update_stats.get("entropy", 0), ep)

            # Logging
            writer.add_scalar("episode/reward_total", ep_reward, ep)
            writer.add_scalar("episode/success_rate", success_rate, ep)
            writer.add_scalar("episode/env_time", env.current_time, ep)

            if ep_reward > best_reward:
                best_reward = ep_reward
                save_checkpoint(os.path.join(checkpoint_dir, "ctde_best.pth"), net, optimizer, ep,
                                best_metric=best_reward)

            if ep % TrainParams.SAVE_EVERY_EP == 0 and ep > 0:
                save_checkpoint(os.path.join(checkpoint_dir, f"ctde_ep_{ep}.pth"), net, optimizer, ep)

            if ep % 20 == 0:
                print(
                    f"Ep {ep} | R:{ep_reward:.1f} | Succ:{success_rate:.2%} | Time:{env.current_time:.0f} | Tasks:{actual_tasks} | Best:{best_reward:.1f}")

        save_checkpoint(os.path.join(checkpoint_dir, "ctde_final.pth"), net, optimizer, ep=TrainParams.MAX_EPISODES - 1)

    finally:
        writer.close()
        print("\n✅ Training completed!")


if __name__ == "__main__":
    main()