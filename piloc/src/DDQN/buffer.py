import numpy as np
import torch
import numpy as np
import torch

class Buffer_for_PPO:
    """replay buffer for each agent"""

    def __init__(self, capacity, obs_dim, act_dim, device, trick=None):
        self.capacity = capacity = int(capacity)

        # 修改obs的维度为 (capacity, 60, 60) 以存储 60x60 的地图
        self.obs1 = np.zeros((capacity, 60, 60))  # batch_size x (60, 60) 通道1
        self.obs2 = np.zeros((capacity, 60, 60))  # batch_size x (60, 60) 通道2
        self.obs_pheromone = np.zeros((capacity, 11, 11))  # batch_size x (11, 11) 通道3

        self.actions = np.zeros((capacity, act_dim))  # batch_size x action_dim
        self.rewards = np.zeros(capacity)  # just a tensor with length: batch
        self.next_obs1 = np.zeros((capacity, 60, 60))  # batch_size x (60, 60) 通道1
        self.next_obs2 = np.zeros((capacity, 60, 60))  # batch_size x (60, 60) 通道2
        self.next_obs_pheromone = np.zeros((capacity, 11, 11))  # batch_size x (11, 11) 通道3

        self.dones = np.zeros(capacity, dtype=bool)  # just a tensor with length: batch_size
        # 合法动作掩码（1=合法，0=越界/撞墙）：训练采样、argmax 验证与 PPO 更新
        # 重算 log-prob 必须使用同一份掩码，否则新旧策略分布不一致，ratio 有偏。
        self.legal_masks = np.ones((capacity, 4), dtype=np.float32)
        if trick is not None and trick['decaystd']:
            self.action_log_probs = np.zeros((capacity))
        else:
            self.action_log_probs = np.zeros((capacity, act_dim))
        self.adv_dones = np.zeros(capacity, dtype=bool)

        self._index = 0
        self._size = 0

        self.device = device

    def add(self, obs1,obs2, obs_pheromone,action, reward, next_obs1,next_obs2,next_obs_pheromone, done, action_log_probs, adv_done, legal_mask):
        """ add an experience to the memory """
        # 这里将 obs 和 next_obs 存储为形状为 (60, 60) 的二维数组
        self.obs1[self._index] = obs1
        self.obs2[self._index] = obs2
        self.obs_pheromone[self._index] = obs_pheromone

        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self.next_obs1[self._index] = next_obs1
        self.next_obs2[self._index] = next_obs2
        self.next_obs_pheromone[self._index] = next_obs_pheromone
        self.dones[self._index] = done

        self.action_log_probs[self._index] = action_log_probs
        self.adv_dones[self._index] = adv_done
        self.legal_masks[self._index] = legal_mask

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    # __len__ is a magic method in Python 可以让对象实现len()方法
    def __len__(self):
        return self._size

    # 清空buffer
    def clear(self):
        self._index = 0
        self._size = 0

    def state_dict(self):
        """导出断点恢复所需的 buffer 状态（全量数组）。

        PPO 的 learn 通过 ``all()`` 消费全部 capacity 行：连续训练中
        "未填充" 行保留着上一窗口的残留数据，同样参与梯度更新（learn 后
        的 clear 只重置索引，不清数组内容）。checkpoint 若只存前
        ``_size`` 行，恢复后这些残留行会变成零，与连续训练不一致
        （N+M ≠ N+resume M）。因此必须全量存取，size==0 且刚 learn 完
        时残留数据同样不能丢。数组训练时按 float32 使用（``all()`` 里
        ``dtype=torch.float32``），落盘也用 float32 不损失精度。
        """
        return {
            "index": int(self._index),
            "size": int(self._size),
            "obs1": self.obs1.astype(np.float32),
            "obs2": self.obs2.astype(np.float32),
            "obs_pheromone": self.obs_pheromone.astype(np.float32),
            "actions": self.actions.astype(np.float32),
            "rewards": self.rewards.astype(np.float32),
            "next_obs1": self.next_obs1.astype(np.float32),
            "next_obs2": self.next_obs2.astype(np.float32),
            "next_obs_pheromone": self.next_obs_pheromone.astype(np.float32),
            "dones": self.dones.copy(),
            "action_log_probs": self.action_log_probs.astype(np.float32),
            "adv_dones": self.adv_dones.copy(),
            "legal_masks": self.legal_masks.astype(np.float32),
        }

    def load_state_dict(self, state):
        """恢复 ``state_dict`` 导出的 buffer 状态（None 表示全新空窗口）。"""
        self.clear()
        if state is None:
            return
        self.obs1[:] = state["obs1"]
        self.obs2[:] = state["obs2"]
        self.obs_pheromone[:] = state["obs_pheromone"]
        self.actions[:] = state["actions"]
        self.rewards[:] = state["rewards"]
        self.next_obs1[:] = state["next_obs1"]
        self.next_obs2[:] = state["next_obs2"]
        self.next_obs_pheromone[:] = state["next_obs_pheromone"]
        self.dones[:] = state["dones"]
        self.action_log_probs[:] = state["action_log_probs"]
        self.adv_dones[:] = state["adv_dones"]
        self.legal_masks[:] = state["legal_masks"]
        self._index = int(state["index"])
        self._size = int(state["size"])

    def all(self):
        # 这里的obs和next_obs是 (batch_size, 60, 60)，可以使用flatten将它们展平为一维数组
        obs1 = torch.as_tensor(self.obs1, dtype=torch.float32).to(self.device)  # torch.Size([batch_size, 60, 60])
        obs2 = torch.as_tensor(self.obs2, dtype=torch.float32).to(self.device)  # torch.Size([batch_size, 60, 60])
        obs_pheromone = torch.as_tensor(self.obs_pheromone, dtype=torch.float32).to(self.device)

        actions = torch.as_tensor(self.actions, dtype=torch.float32).to(
            self.device)  # torch.Size([batch_size, action_dim])
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32).reshape(-1, 1).to(
            self.device)  # torch.Size([batch_size]) -> torch.Size([batch_size, 1])
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)

        next_obs1 = torch.as_tensor(self.next_obs1, dtype=torch.float32).to(
            self.device)  # torch.Size([batch_size, 60, 60])
        next_obs2 = torch.as_tensor(self.next_obs2, dtype=torch.float32).to(
            self.device)  # torch.Size([batch_size, 60, 60])
        next_obs_pheromone = torch.as_tensor(self.next_obs_pheromone, dtype=torch.float32).to(self.device)

        dones = torch.as_tensor(self.dones, dtype=torch.float32).reshape(-1, 1).to(self.device)

        actions_log_probs = torch.as_tensor(self.action_log_probs, dtype=torch.float32).to(
            self.device)  # torch.Size([batch_size, action_dim])
        adv_dones = torch.as_tensor(self.adv_dones, dtype=torch.float32).reshape(-1, 1).to(self.device)
        legal_masks = torch.as_tensor(self.legal_masks, dtype=torch.float32).to(self.device)

        return obs1,obs2,obs_pheromone, actions, rewards, next_obs1,next_obs2,next_obs_pheromone ,dones, actions_log_probs, adv_dones, legal_masks
