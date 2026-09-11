# 完整修正后的 CentralWorker（替换原文件中的类实现）
import pickle
import time
import torch
import numpy as np
import random
from env.task_env import TaskEnv
from attention import CentralAttentionNet
from central_controller import CentralController
import scipy.signal as signal
from parameters import *
import copy
from torch.nn import functional as F
from torch.distributions import Categorical
import warnings


def discount(x, gamma):
    """折扣奖励计算（用于优势函数）"""
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class CentralWorker:
    def __init__(self, meta_agent_id=None, local_network=None, local_baseline=None, global_step=0,
                 device=None, save_image=False, seed=None, env_params=None, is_test=False, **kwargs):
        """
        兼容性说明：
          - 支持旧拼写 mete_agent_id（调用处若使用该名也不会崩溃）
          - 支持 central_net 作为 local_network 的别名
          - 接受 **kwargs 以提高容错性，警告未使用的键
        """
        # 兼容旧参数名（如果调用方传了 mete_agent_id）
        if 'mete_agent_id' in kwargs and meta_agent_id is None:
            meta_agent_id = kwargs.pop('mete_agent_id')

        # 兼容 central_net/centralNet / local_net 等别名
        if local_network is None:
            local_network = kwargs.pop('central_net', None) or kwargs.pop('local_net', None)

        if local_baseline is None:
            local_baseline = kwargs.pop('baseline_net', None) or kwargs.pop('local_baseline_net', None)

        # 设备处理：优先使用调用者传入的 device；否则自动检测
        try:
            import torch as _torch
            if device is None:
                device = 'cuda' if _torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'
        # 保证为 torch.device 类型
        try:
            self.device = device if isinstance(device, _torch.device) else _torch.device(device)
        except Exception:
            # 如果 device 字符串非法，退回 cpu
            self.device = _torch.device('cpu')

        # 基本属性
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.is_test = is_test
        self.max_time = getattr(EnvParams, "MAX_TIME", 100)  # 兜底
        self.seed = seed

        # 处理 env_params 的默认值
        if env_params is None:
            env_params = {
                "per_species_agents": getattr(EnvParams, "SPECIES_AGENTS_RANGE", (1, 10)),
                "species_num": getattr(EnvParams, "SPECIES_RANGE", (1, 5)),
                "tasks_num": getattr(EnvParams, "TASKS_RANGE", (1, 20)),
                "prio_range": (0.3, 1.0)
            }
        self.env_params = env_params

        # 初始化环境
        self.env = TaskEnv(
            per_species_range=env_params["per_species_agents"],
            species_range=env_params["species_num"],
            tasks_range=env_params["tasks_num"],
            traits_dim=getattr(EnvParams, "TRAIT_DIM", 8),
            decision_dim=getattr(EnvParams, "DECISION_DIM", 4),
            seed=seed,
            plot_figure=save_image,

        )
        self.baseline_env = copy.deepcopy(self.env)

        # 初始化集中式网络与控制器
        # local_network/local_baseline 可能为 None（调用方需传入），若为 None 发出警告
        if local_network is None:
            warnings.warn("CentralWorker initialized without local_network (policy). Pass local_network for training.")
        if local_baseline is None:
            # 若没有独立 baseline，则使用 local_network 作为 baseline 的占位
            warnings.warn("CentralWorker initialized without local_baseline; using local_network as baseline fallback.")
            local_baseline = local_network

        self.local_net = local_network
        self.local_baseline = local_baseline
        # 控制器负责把 env 的全局观测转换成网络可用输入
        self.controller = CentralController(self.local_net, self.device) if self.local_net is not None else None
        self.baseline_controller = CentralController(self.local_baseline, self.device) if self.local_baseline is not None else None

        # 经验缓冲区
        self.experience = {
            "global_agents": [],
            "global_tasks": [],
            "global_mask": [],
            "allocations": [],
            "rewards": [],
            "high_prio_success": [],
            "advantages": []
        }
        self.episode_number = None
        self.perf_metrics = {}

        # 记录未使用的 kwargs（便于排查）
        if kwargs:
            warnings.warn(f"CentralWorker.__init__ received unused kwargs: {list(kwargs.keys())}")

    def obs_padding(self, x, target_dim, dim=0):
        """
        通用补零实现：对任意维度的 tensor 做行/列补零（使用 concat 实现，避免 ZeroPad2d 维度限制）
        :param x: Tensor，至少 1D
        :param target_dim: 目标大小（沿 dim）
        :param dim: 需要补零的维度索引
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        current_dim = x.shape[dim]
        if current_dim >= target_dim:
            return x
        pad_size = target_dim - current_dim
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_size
        pad_tensor = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad_tensor], dim=dim)

    def run_episode(self, training=True, max_waiting=False):
        """
        集中式 episode 运行
        返回：buffer_dict, perf_metrics
        """
        buffer_dict = copy.deepcopy(self.experience)
        current_action_index = 0
        decision_step = 0

        prev_high_prio_finish = 0
        prev_low_prio_finish = 0

        # 初始化环境状态
        self.env.init_state()

        while not self.env.finished and self.env.current_time < self.max_time and current_action_index < 300:
            # 获取全局观察
            if self.controller is None:
                raise RuntimeError("Controller (and policy network) is not set for CentralWorker.")
            global_obs = self.controller.get_global_observation(self.env)
            num_agents = int(global_obs["agents"].shape[0])
            num_tasks = int(global_obs["tasks"].shape[0])

            # 补零
            max_agents = getattr(EnvParams, "SPECIES_AGENTS_RANGE", (1, 10))[1] * getattr(EnvParams, "SPECIES_RANGE", (1, 5))[1]
            global_agents_padded = self.obs_padding(global_obs["agents"], target_dim=max_agents, dim=0)
            global_tasks_padded = self.obs_padding(global_obs["tasks"], target_dim=getattr(EnvParams, "TASKS_RANGE", (1, 20))[1], dim=0)
            global_mask = global_obs["global"][:, 1].unsqueeze(0).unsqueeze(0)  # 保持原逻辑

            # 分配
            with torch.no_grad() if not training else torch.enable_grad():
                alloc_probs, _ = self.local_net(
                    global_tasks_padded.unsqueeze(0),
                    global_agents_padded.unsqueeze(0),
                    global_mask
                )
                if training and not self.is_test:
                    alloc_dist = Categorical(alloc_probs.squeeze(0))
                    allocations = alloc_dist.sample()
                else:
                    allocations = torch.argmax(alloc_probs.squeeze(0), dim=1)

                valid_allocations = allocations[:num_agents]
                valid_allocations = torch.clamp(valid_allocations, 0, max(num_tasks - 1, 0))

            # 执行分配
            doable_count = 0
            for agent_id in range(num_agents):
                task_id = int(valid_allocations[agent_id].item())
                r, doable, _ = self.env.agent_step(agent_id, task_id, decision_step)
                if doable:
                    doable_count += 1
            current_action_index += doable_count
            decision_step += 1

            # 计算奖励（传入两个 prev）
            reward, curr_high_prio_success = self.compute_priority_reward(prev_high_prio_finish, prev_low_prio_finish)

            # 更新 prev counts（用于下一步 delta 计算）
            prev_high_prio_finish = sum(1 for t in self.env.task_dic.values() if t["search_priority"] >= 0.6 and t["finished"])
            prev_low_prio_finish = sum(1 for t in self.env.task_dic.values() if t["search_priority"] < 0.6 and t["finished"])

            # 存储经验（只在训练时）
            if training:
                buffer_dict["global_agents"].append(global_agents_padded.cpu())
                buffer_dict["global_tasks"].append(global_tasks_padded.cpu())
                buffer_dict["global_mask"].append(global_mask.cpu())
                buffer_dict["allocations"].append(valid_allocations.cpu())
                buffer_dict["rewards"].append(torch.tensor([reward], dtype=torch.float32).cpu())
                buffer_dict["high_prio_success"].append(torch.tensor([curr_high_prio_success], dtype=torch.float32).cpu())

            # 更新环境状态
            self.env.finished = self.env.check_finished()
            self.env.current_time += 1

        # 优势计算
        if training:
            buffer_dict["advantages"] = self.compute_advantages(buffer_dict["rewards"])

        # 统计
        total_finished = sum(1 for t in self.env.task_dic.values() if t["finished"])
        high_prio_tasks = [t for t in self.env.task_dic.values() if t["search_priority"] >= 0.6]
        high_prio_finished = sum(1 for t in high_prio_tasks if t["finished"])
        high_prio_success_rate = high_prio_finished / len(high_prio_tasks) if high_prio_tasks else 1.0

        perf_metrics = {
            'success_rate': [total_finished / len(self.env.task_dic) if self.env.task_dic else 0.0],
            'high_prio_success_rate': [high_prio_success_rate],
            'makespan': [self.env.current_time],
            'time_cost': [np.nanmean(self.env.get_matrix(self.env.task_dic, 'time_start'))],
            'waiting_time': [np.mean(self.env.get_matrix(self.env.agent_dic, 'sum_waiting_time'))],
            'travel_dist': [np.sum(self.env.get_matrix(self.env.agent_dic, 'travel_dist'))],
            'efficiency': [self.env.get_efficiency()]
        }

        return buffer_dict, perf_metrics

    def compute_priority_reward(self, prev_high_prio_finish, prev_low_prio_finish):
        """
        修正后的加权奖励：
          - 接收上一步高/低优先级完成数，计算 delta
          - 高优先级奖励/惩罚权重可调整
        """
        high_prio_tasks = [t for t in self.env.task_dic.values() if t["search_priority"] >= 0.6]
        low_prio_tasks = [t for t in self.env.task_dic.values() if t["search_priority"] < 0.6]

        num_high_prio = len(high_prio_tasks)
        num_low_prio = len(low_prio_tasks)

        curr_high_finish = sum(1 for t in high_prio_tasks if t["finished"])
        curr_low_finish = sum(1 for t in low_prio_tasks if t["finished"])

        delta_high = curr_high_finish - prev_high_prio_finish
        delta_low = curr_low_finish - prev_low_prio_finish

        high_reward = max(0.0, delta_high) * 20.0
        low_reward = max(0.0, delta_low) * 10.0

        timeout_penalty = 0.0
        if self.env.current_time > self.max_time * 0.5:
            for t in high_prio_tasks:
                if not t["finished"]:
                    timeout_penalty -= (1 - t["search_priority"]) * 2.0

        final_reward = 0.0
        if self.env.finished and (num_high_prio == 0 or curr_high_finish == num_high_prio):
            final_reward = (self.max_time - self.env.current_time) * 3.0

        total_reward = high_reward + low_reward + timeout_penalty + final_reward
        high_prio_success_rate = curr_high_finish / num_high_prio if num_high_prio else 1.0
        return total_reward, high_prio_success_rate

    def compute_advantages(self, rewards):
        """GAE 简化实现（若 rewards 为空，返回空列表）"""
        if not rewards:
            return []
        # rewards 是 list of tensors [ [r], [r], ... ]
        rewards_np = torch.stack(rewards).squeeze(1).cpu().numpy()
        baseline = np.mean(rewards_np)
        discounted_rewards = discount(rewards_np, gamma = 0.99)
        advantages = discounted_rewards - baseline
        if advantages.size == 0:
            return []
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return [torch.tensor([adv], dtype=torch.float32).cpu() for adv in advantages]

    def baseline_test(self):
        """基线评估（贪心，无探索）"""
        self.baseline_env.plot_figure = False
        current_action_index = 0
        decision_step = 0
        start_time = time.time()
        prev_high_prio_finish = 0

        while not self.baseline_env.finished and self.baseline_env.current_time < self.max_time and current_action_index < 300:
            if time.time() - start_time > 30:
                break

            if self.baseline_controller is None:
                raise RuntimeError("Baseline controller not set.")

            baseline_obs = self.baseline_controller.get_global_observation(self.baseline_env)
            num_agents = int(baseline_obs["agents"].shape[0])
            baseline_agents_padded = self.obs_padding(
                baseline_obs["agents"],
                target_dim=getattr(EnvParams, "SPECIES_AGENTS_RANGE", (1, 10))[1] * getattr(EnvParams, "SPECIES_RANGE", (1, 5))[1],
                dim=0
            )
            baseline_tasks_padded = self.obs_padding(
                baseline_obs["tasks"],
                target_dim=getattr(EnvParams, "TASKS_RANGE", (1, 20))[1],
                dim=0
            )
            baseline_mask = baseline_obs["global"][:, 1].unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                baseline_probs, _ = self.local_baseline(
                    baseline_tasks_padded.unsqueeze(0),
                    baseline_agents_padded.unsqueeze(0),
                    baseline_mask
                )
                baseline_allocations = torch.argmax(baseline_probs.squeeze(0), dim=1)[:num_agents]

            for agent_id in range(num_agents):
                task_id = int(baseline_allocations[agent_id].item())
                self.baseline_env.agent_step(agent_id, task_id, decision_step)
            current_action_index += num_agents
            decision_step += 1

            self.baseline_env.finished = self.baseline_env.check_finished()
            self.baseline_env.current_time += 1

        total_reward, _ = self.compute_priority_reward(prev_high_prio_finish, 0)
        high_prio_tasks = [t for t in self.baseline_env.task_dic.values() if t["search_priority"] >= 0.6]
        high_prio_finished = sum(1 for t in high_prio_tasks if t["finished"])
        high_prio_success_rate = high_prio_finished / len(high_prio_tasks) if high_prio_tasks else 1.0
        return total_reward, high_prio_success_rate

    def work(self, episode_number):
        """主训练循环（POMO 多次采样合并经验）"""
        buffers = []
        metrics = []
        max_waiting = getattr(TrainParams, "FORCE_MAX_OPEN_TASK", False)

        for _ in range(getattr(TrainParams, "POMO_SIZE", 1)):
            # env.init_state() 已在 run_episode 内部调用，移除重复调用
            buffer, perf_metrics = self.run_episode(training=True, max_waiting=max_waiting)
            if not buffer["rewards"]:
                max_waiting = True
                continue
            buffers.append(buffer)
            metrics.append(perf_metrics)

        # 合并
        for buffer in buffers:
            for key in self.experience.keys():
                self.experience[key].extend(buffer[key])

        for metric in metrics:
            for key, val in metric.items():
                if key not in self.perf_metrics:
                    self.perf_metrics[key] = val
                else:
                    self.perf_metrics[key].extend(val)

        if self.save_image:
            try:
                self.env.plot_animation(SaverParams.GIFS_PATH, episode_number)
            except Exception as e:
                print(f"Save animation failed: {e}")

        self.episode_number = episode_number

    def convert_torch(self, args):
        """兼容旧接口：转换数据为Tensor"""
        out = []
        for arg in args:
            t = torch.tensor(arg, dtype=torch.float)
            try:
                out.append(t.to(self.device))
            except Exception:
                out.append(t)
        return out


if __name__ == '__main__':
    # 本地快速测试：初始化网络与 worker（注意使用 CentralWorker）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_env_params = {
        "per_species_agents": (3, 3),
        "species_num": (5, 5),
        "tasks_num": (20, 20),
        "prio_range": (0.3, 1.0)
    }

    local_net = CentralAttentionNet(
        TrainParams.AGENT_INPUT_DIM,
        TrainParams.TASK_INPUT_DIM,
        TrainParams.EMBEDDING_DIM
    ).to(device)

    for i in range(3):
        worker = CentralWorker(
            meta_agent_id=0,
            local_network=local_net,
            local_baseline=local_net,
            global_step=0,
            device=device,
            seed=i,
            save_image=False,
            env_params=test_env_params,
            is_test=False
        )
        worker.work(episode_number=i)
        sr = np.mean(worker.perf_metrics.get('success_rate', [0.0]))
        hp = np.mean(worker.perf_metrics.get('high_prio_success_rate', [0.0]))
        print(f"Test episode {i} finished. Success rate: {sr:.2f}, High prio success: {hp:.2f}")
