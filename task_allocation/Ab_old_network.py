# -*- coding: utf-8 -*-
"""
CTDE 网络架构（消融实验兼容版）
- 支持动态长度输入
- 支持动态维度：根据消融开关，可接收 15/18维 或 20/23维 输入
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
            task_dim=20,           # ✅ 支持外部动态传入 (15 或 20)
            global_agent_dim=13,
            global_task_dim=23,    # ✅ 支持外部动态传入 (18 或 23)
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

        # 任务数量条件化
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
        B, N_tasks, _ = tasks.shape

        h_self = self.actor_self_enc(self_state).unsqueeze(1)
        h_neighbors = self.actor_neighbor_enc(neighbors)
        h_neighbors = self.dropout(h_neighbors)
        h_neighbor_agg, _ = self.actor_neighbor_attn(h_self, h_neighbors, h_neighbors)

        h_agent = h_self + h_neighbor_agg

        h_tasks = self.actor_task_enc(tasks)
        h_tasks = self.dropout(h_tasks)

        if self.use_pos_encoding:
            pe = self.get_sinusoidal_encoding(N_tasks, self.hidden, h_tasks.device)
            pe = pe.unsqueeze(0).expand(B, -1, -1)
            h_tasks = h_tasks + 0.1 * pe

        valid_counts = (~mask).sum(dim=1, keepdim=True).float()
        count_feat = self.task_count_embed(valid_counts / 100.0)
        h_agent_modulated = h_agent + count_feat.unsqueeze(1) * 0.1

        q = h_agent_modulated.expand(-1, N_tasks, -1)
        h_task_attn, _ = self.actor_task_attn(q, h_tasks, h_tasks)
        h_task_attn = self.dropout(h_task_attn)

        logits = self.actor_head(h_task_attn).squeeze(-1)
        logits = logits.masked_fill(mask.bool(), -1e9)

        return Categorical(logits=logits)

    def get_value(self, global_agents, global_tasks):
        h_agents = self.critic_agent_enc(global_agents)
        h_tasks = self.critic_task_enc(global_tasks)
        all_entities = torch.cat([h_agents, h_tasks], dim=1)
        h_global, _ = self.critic_attn(all_entities, all_entities, all_entities)
        return self.critic_val(h_global.mean(dim=1))


if __name__ == "__main__":
    print("Testing Ablation Network with Baseline (15-dim)...")
    net_base = CTDE_Network(task_dim=15, global_task_dim=18)
    B, N = 4, 10
    tasks = torch.randn(B, N, 15)
    g_tasks = torch.randn(B, N, 18)
    mask = torch.zeros(B, N, dtype=torch.bool)
    net_base.get_action(torch.randn(B, 10), torch.randn(B, 20, 10), tasks, mask)
    print("✅ Baseline pass!")

    print("Testing Ablation Network with Pre-positioning (20-dim)...")
    net_prepos = CTDE_Network(task_dim=20, global_task_dim=23)
    tasks = torch.randn(B, N, 20)
    g_tasks = torch.randn(B, N, 23)
    net_prepos.get_action(torch.randn(B, 10), torch.randn(B, 20, 10), tasks, mask)
    print("✅ Pre-positioning pass!")