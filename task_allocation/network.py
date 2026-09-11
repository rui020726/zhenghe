# -*- coding: utf-8 -*-
"""
CTDE 网络架构（分布式决策 + 集中式评估）
- ✅ 支持动态长度输入（无固定MAX_TASKS限制）
- ✅ Sinusoidal位置编码（任意长度）
- ✅ Dropout正则化
- ✅ 任务数量自适应Attention
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical


class CTDE_Network(nn.Module):
    def __init__(
            self,
            self_dim=10,
            neighbor_dim=10,
            task_dim=15,
            global_agent_dim=13,
            global_task_dim=18,
            hidden=128,
            dropout=0.15,
            use_pos_encoding=True,
    ):
        super().__init__()

        self.hidden = hidden
        self.use_pos_encoding = use_pos_encoding
        self.dropout_rate = dropout

        # ========== Actor ==========
        self.actor_self_enc = nn.Linear(self_dim, hidden)
        self.actor_neighbor_enc = nn.Linear(neighbor_dim, hidden)
        self.actor_task_enc = nn.Linear(task_dim, hidden)

        self.actor_neighbor_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, batch_first=True, dropout=dropout
        )
        self.actor_task_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, batch_first=True, dropout=dropout
        )

        self.actor_head = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

        # ✅ 任务数量条件化（帮助网络感知任务规模）
        self.task_count_embed = nn.Sequential(
            nn.Linear(1, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, hidden)
        )

        # ========== Critic ==========
        self.critic_agent_enc = nn.Linear(global_agent_dim, hidden)
        self.critic_task_enc = nn.Linear(global_task_dim, hidden)
        self.critic_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, batch_first=True, dropout=dropout
        )
        self.critic_val = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def get_sinusoidal_encoding(self, seq_len, d_model, device):
        """
        生成任��长度的Sinusoidal位置编码
        Args:
            seq_len: 序列长度
            d_model: 编码维度
            device: 设备
        Returns:
            [seq_len, d_model] 位置编码
        """
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
            -(np.log(10000.0) / d_model)
        )

        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        return pe

    def get_action(self, self_state, neighbors, tasks, mask):
        """
        ✅ 支持动态长度输入
        Args:
            self_state: [B, self_dim]
            neighbors:   [B, N_neighbors, neighbor_dim]
            tasks:      [B, N_tasks, task_dim]  ⭐ N_tasks可以是任意值
            mask:       [B, N_tasks]  (True=masked)
        Returns:
            Categorical distribution over tasks
        """
        B, N_tasks, _ = tasks.shape

        # 1) Encode self
        h_self = self.actor_self_enc(self_state).unsqueeze(1)  # [B, 1, H]

        # 2) Encode neighbors + attention
        h_neighbors = self.actor_neighbor_enc(neighbors)  # [B, Nn, H]
        h_neighbors = self.dropout(h_neighbors)
        h_neighbor_agg, _ = self.actor_neighbor_attn(h_self, h_neighbors, h_neighbors)

        # 3) Agent context
        h_agent = h_self + h_neighbor_agg  # [B, 1, H]

        # 4) Encode tasks
        h_tasks = self.actor_task_enc(tasks)  # [B, N_tasks, H]
        h_tasks = self.dropout(h_tasks)

        # 5) ✅ 添加位置编码（支持任意长度）
        if self.use_pos_encoding:
            pe = self.get_sinusoidal_encoding(N_tasks, self.hidden, h_tasks.device)
            pe = pe.unsqueeze(0).expand(B, -1, -1)  # [B, N_tasks, H]
            h_tasks = h_tasks + 0.1 * pe  # 小权重混合

        # 6) ✅ 任务数量条件化
        # 统计有效任务数（非masked）
        valid_counts = (~mask).sum(dim=1, keepdim=True).float()  # [B, 1]
        count_feat = self.task_count_embed(valid_counts / 100.0)  # 归一化到[0,1]
        h_agent_modulated = h_agent + count_feat.unsqueeze(1) * 0.1  # [B, 1, H]

        # 7) Task attention
        q = h_agent_modulated.expand(-1, N_tasks, -1)  # [B, N_tasks, H]
        h_task_attn, _ = self.actor_task_attn(
            h_tasks,  # Query: [B, N_tasks, H]（每个 task 问"我值得被选吗?"）
            h_agent_modulated.expand(-1, 1, -1),  # Key:   [B, 1, H]（agent context）
            h_agent_modulated.expand(-1, 1, -1)  # Value: [B, 1, H]
        )
        h_task_attn = self.dropout(h_task_attn)

        # 8) Logits
        logits = self.actor_head(h_task_attn).squeeze(-1)  # [B, N_tasks]

        # 9) Mask (True = illegal)
        logits = logits.masked_fill(mask.bool(), -1e9)

        return Categorical(logits=logits)

    def get_value(self, global_agents, global_tasks, padding_mask=None):
        h_agents = self.critic_agent_enc(global_agents)
        h_tasks = self.critic_task_enc(global_tasks)

        all_entities = torch.cat([h_agents, h_tasks], dim=1)  # [B, Na+Nt, H]

        # ✅ 传入 padding_mask，attention 时忽略 padding 位置
        h_global, _ = self.critic_attn(
            all_entities, all_entities, all_entities,
            key_padding_mask=padding_mask  # ✅ 新增
        )

        # ✅ mean pooling 时也排除 padding 位置
        if padding_mask is not None:
            valid_mask = ~padding_mask  # True = 真实位置
            valid_mask_f = valid_mask.unsqueeze(-1).float()  # [B, Na+Nt, 1]
            h_pooled = (h_global * valid_mask_f).sum(dim=1) \
                       / valid_mask_f.sum(dim=1).clamp(min=1)  # [B, H]
        else:
            h_pooled = h_global.mean(dim=1)

        return self.critic_val(h_pooled)


if __name__ == "__main__":
    print("Testing dynamic length support...")
    net = CTDE_Network()

    B = 4
    self_state = torch.randn(B, 10)
    neighbors = torch.randn(B, 20, 10)

    # ✅ 测试不同长度
    for N_tasks in [10, 35, 80, 120]:
        print(f"\nTesting with {N_tasks} tasks...")
        tasks = torch.randn(B, N_tasks, 15)
        mask = torch.zeros(B, N_tasks, dtype=torch.bool)
        mask[:, -5:] = True  # 最后5个mask掉

        global_agents = torch.randn(B, 30, 13)
        global_tasks = torch.randn(B, N_tasks, 18)

        dist = net.get_action(self_state, neighbors, tasks, mask)
        a = dist.sample()
        print(f"  Action shape: {a.shape}, sampled actions: {a}")
        assert a.max().item() < N_tasks - 5, "Sampled masked action!"

        v = net.get_value(global_agents, global_tasks)
        print(f"  Value shape: {v.shape}")

    print("\n✅ Network dynamic length test passed!")