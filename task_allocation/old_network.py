# -*- coding: utf-8 -*-
"""
CTDE 网络架构（分布式决策 + 集中式评估）
- ✅ 修复 Attention 方向：Task 做 Query，Agent context 做 Key/Value
- ✅ Critic 加 Padding Mask + Masked Mean Pooling
- ✅ 支持动态长度输入
- ✅ Sinusoidal位置编码
- ✅ Dropout正则化
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
            task_dim=20,
            global_agent_dim=13,
            global_task_dim=23,
            hidden=128,
            dropout=0.15,
            use_pos_encoding=True,
            actor_kv_mode="tokens",
    ):
        super().__init__()

        self.hidden = hidden
        self.use_pos_encoding = use_pos_encoding
        self.dropout_rate = dropout
        self.actor_kv_mode = str(actor_kv_mode).lower()
        if self.actor_kv_mode not in ("summary", "tokens"):
            raise ValueError("actor_kv_mode must be 'summary' or 'tokens'")

        # ========== Actor ==========
        self.actor_self_enc = nn.Linear(self_dim, hidden)
        self.actor_neighbor_enc = nn.Linear(neighbor_dim, hidden)
        self.actor_task_enc = nn.Linear(task_dim, hidden)

        self.actor_neighbor_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, batch_first=True, dropout=dropout
        )
        # ✅ 修复：Task(Q) attend to Agent context(KV)
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
        """
    ✅ 修复后的 Attention 方向（可切换 KV 模式）：
        - Task embeddings 做 Query
    - Agent context 做 Key/Value（summary 或 multi-agent tokens）
        - 每个 task 位置的 Q 不同，能有效区分任务

        Args:
            self_state: [B, self_dim]
            neighbors:  [B, N_neighbors, neighbor_dim]
            tasks:      [B, N_tasks, task_dim]
            mask:       [B, N_tasks] (True=masked)
        Returns:
            Categorical distribution over tasks
        """
        B, N_tasks, _ = tasks.shape

        # 1) Encode self
        h_self = self.actor_self_enc(self_state).unsqueeze(1)  # [B, 1, H]

        # 2) Encode neighbors + attention (agent attends to neighbors)
        h_neighbors = self.actor_neighbor_enc(neighbors)  # [B, N_nb, H]
        h_neighbors = self.dropout(h_neighbors)
        nb_padding_mask = (neighbors.abs().sum(dim=-1) < 1e-8)  # [B, N_nb]
        h_neighbor_agg, _ = self.actor_neighbor_attn(
            h_self, h_neighbors, h_neighbors, key_padding_mask=nb_padding_mask
        )  # [B, 1, H]

        # 3) Agent context = self + neighbor aggregation
        h_agent = h_self + h_neighbor_agg  # [B, 1, H]

        # 4) 任务数量条件化调制
        valid_counts = (~mask).sum(dim=1, keepdim=True).float()
        count_feat = self.task_count_embed(valid_counts / 100.0)  # [B, H]
        h_agent = h_agent + count_feat.unsqueeze(1) * 0.1  # [B, 1, H]

        # 5) Encode tasks
        h_tasks = self.actor_task_enc(tasks)  # [B, N_tasks, H]
        h_tasks = self.dropout(h_tasks)

        # 6) 位置编码
        if self.use_pos_encoding:
            pe = self.get_sinusoidal_encoding(N_tasks, self.hidden, h_tasks.device)
            pe = pe.unsqueeze(0).expand(B, -1, -1)
            h_tasks = h_tasks + 0.1 * pe

        # 7) ✅ Task(Q) attend to Agent context(KV)
        #    - summary: K=V=[B,1,H]（旧版/单向量）
        #    - tokens : K=V=[B,1+N_nb,H]（多agent token）
        if self.actor_kv_mode == "tokens":
            kv_tokens = torch.cat([h_self, h_neighbors], dim=1)  # [B, 1+N_nb, H]
            nb_pad = (neighbors.abs().sum(dim=-1) < 1e-8)  # [B, N_nb], True=padding
            self_pad = torch.zeros(B, 1, dtype=torch.bool, device=neighbors.device)
            kv_padding_mask = torch.cat([self_pad, nb_pad], dim=1)  # [B, 1+N_nb]
            h_task_attn, _ = self.actor_task_attn(
                h_tasks, kv_tokens, kv_tokens, key_padding_mask=kv_padding_mask
            )
        else:
            h_task_attn, _ = self.actor_task_attn(h_tasks, h_agent, h_agent)  # [B, N_tasks, H]
        h_task_attn = self.dropout(h_task_attn)

        # 8) 融合：task 自身特征 + attention 输出
        h_fused = h_tasks + h_task_attn  # [B, N_tasks, H]

        # 9) Logits
        logits = self.actor_head(h_fused).squeeze(-1)  # [B, N_tasks]

        # 10) Mask (True = illegal)
        logits = logits.masked_fill(mask.bool(), -1e9)

        return Categorical(logits=logits)

    def get_value(self, global_agents, global_tasks, agent_padding_mask=None, task_padding_mask=None):
        """
        ✅ 修复：支持 padding mask + masked mean pooling

        Args:
            global_agents: [B, N_agents, global_agent_dim]
            global_tasks:  [B, N_tasks, global_task_dim]
            agent_padding_mask: [B, N_agents] bool, True=padding (可选)
            task_padding_mask:  [B, N_tasks] bool, True=padding (可选)
        """
        h_agents = self.critic_agent_enc(global_agents)  # [B, N_a, H]
        h_tasks = self.critic_task_enc(global_tasks)      # [B, N_t, H]

        all_entities = torch.cat([h_agents, h_tasks], dim=1)  # [B, N_a+N_t, H]

        # 构建 key_padding_mask
        key_padding_mask = None
        if agent_padding_mask is not None or task_padding_mask is not None:
            B = all_entities.shape[0]
            if agent_padding_mask is None:
                agent_padding_mask = torch.zeros(B, h_agents.shape[1], dtype=torch.bool, device=all_entities.device)
            if task_padding_mask is None:
                task_padding_mask = torch.zeros(B, h_tasks.shape[1], dtype=torch.bool, device=all_entities.device)
            key_padding_mask = torch.cat([agent_padding_mask, task_padding_mask], dim=1)  # [B, N_a+N_t]

        h_global, _ = self.critic_attn(
            all_entities, all_entities, all_entities,
            key_padding_mask=key_padding_mask
        )

        # ✅ Masked mean pooling（排除 padding 位置）
        if key_padding_mask is not None:
            valid_mask = ~key_padding_mask  # [B, N], True=valid
            valid_mask_f = valid_mask.unsqueeze(-1).float()  # [B, N, 1]
            h_pooled = (h_global * valid_mask_f).sum(dim=1) / valid_mask_f.sum(dim=1).clamp(min=1.0)
        else:
            h_pooled = h_global.mean(dim=1)

        return self.critic_val(h_pooled)


if __name__ == "__main__":
    print("Testing dynamic length support...")
    net = CTDE_Network(actor_kv_mode="summary")

    B = 4
    self_state = torch.randn(B, 10)
    neighbors = torch.randn(B, 20, 10)

    for N_tasks in [10, 35, 80, 120]:
        print(f"\nTesting with {N_tasks} tasks...")
        tasks = torch.randn(B, N_tasks, 20)
        mask = torch.zeros(B, N_tasks, dtype=torch.bool)
        mask[:, -5:] = True

        global_agents = torch.randn(B, 30, 13)
        global_tasks = torch.randn(B, N_tasks, 23)

        # 测试带 padding mask 的 critic
        agent_pad = torch.zeros(B, 30, dtype=torch.bool)
        agent_pad[:, -5:] = True
        task_pad = torch.zeros(B, N_tasks, dtype=torch.bool)
        task_pad[:, -3:] = True

        dist = net.get_action(self_state, neighbors, tasks, mask)
        a = dist.sample()
        print(f"  Action shape: {a.shape}, sampled actions: {a}")

        v = net.get_value(global_agents, global_tasks, agent_pad, task_pad)
        print(f"  Value shape: {v.shape}")

    print("\nTesting tokens KV mode...")
    net_tokens = CTDE_Network(actor_kv_mode="tokens")
    dist_tokens = net_tokens.get_action(self_state, neighbors, tasks, mask)
    print(f"  Tokens mode probs shape: {dist_tokens.probs.shape}")

    print("\n✅ Network test passed!")
