# -*- coding: utf-8 -*-
"""
CTDE 网络架构 —— 线性注意力版 (Linear Attention)

与 old_network.py 架构对齐，但将所有 nn.MultiheadAttention 替换为
O(N) 线性注意力：φ(Q) (φ(K)ᵀ V) / (φ(Q)ᵀ (φ(K)ᵀ 1))

核函数: φ(x) = elu(x) + 1  (Linear Transformer, Katharopoulos et al. 2020)

优势：
- 复杂度从 O(N²) 降为 O(N)，适合大规模任务场景
- 无 softmax 竞争的数值稳定性
- 因果/双向均可，此处全部使用双向注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


# ═══════════════════════════════════════════════════════════════════════
# 线性注意力底层实现
# ═══════════════════════════════════════════════════════════════════════

class MultiHeadLinearAttention(nn.Module):
    """
    多头线性注意力。

    标准注意力:  V' = softmax(QKᵀ/√d) V          ← O(N²)
    线性注意力:  V' = φ(Q) (φ(K)ᵀ V) / (φ(Q)ᵀ φ(K)ᵀ 1)  ← O(N)

    通过先计算 φ(K)ᵀ V ∈ R^{d×d}，再与 φ(Q) 相乘，避免显式 N×N 矩阵。
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} 必须被 num_heads {num_heads} 整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _kernel(x: torch.Tensor) -> torch.Tensor:
        """φ(x) = elu(x) + 1，保证正值，避免分母退化"""
        return F.elu(x) + 1.0

    def forward(self, query, key, value, key_padding_mask=None):
        """
        Args:
            query: [B, N_q, D]
            key:   [B, N_kv, D]
            value: [B, N_kv, D]
            key_padding_mask: [B, N_kv] bool, True=padding
        Returns:
            out: [B, N_q, D]
        """
        B, N_q, _ = query.shape
        N_kv = key.shape[1]

        # 投影 → 多头
        q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim)   # [B,N_q,H,d]
        k = self.k_proj(key).view(B, N_kv, self.num_heads, self.head_dim)     # [B,N_kv,H,d]
        v = self.v_proj(value).view(B, N_kv, self.num_heads, self.head_dim)   # [B,N_kv,H,d]

        # 核映射
        q = self._kernel(q)
        k = self._kernel(k)
        # v 不经过核映射（标准做法）

        # Padding mask：在核映射之后置零，确保 φ(0) 不会贡献到 sum
        if key_padding_mask is not None:
            # [B, N_kv] → [B, N_kv, 1, 1]
            mask = (~key_padding_mask).float().unsqueeze(-1).unsqueeze(-1)
            k = k * mask
            v = v * mask

        # ── 线性注意力核心 ──
        # KV: [B, H, d, d] = Σⱼ φ(kⱼ) ⊗ vⱼ
        kv = torch.einsum('bshd,bshe->bhde', k, v)

        # K_sum: [B, H, d] = Σⱼ φ(kⱼ)，用于归一化
        k_sum = k.sum(dim=1)  # [B, H, d]

        # Q·KV: [B, N_q, H, d]
        qkv = torch.einsum('bnhd,bhde->bnhe', q, kv)

        # 归一化分母: [B, N_q, H]
        qk_norm = torch.einsum('bnhd,bhd->bnh', q, k_sum).clamp(min=1e-8)

        out = qkv / qk_norm.unsqueeze(-1)  # [B, N_q, H, d]

        # 合并多头
        out = out.reshape(B, N_q, self.embed_dim)
        out = self.dropout(out)
        out = self.out_proj(out)

        return out


# ═══════════════════════════════════════════════════════════════════════
# CTDE 网络主体
# ═══════════════════════════════════════════════════════════════════════

class CTDE_Network_Linear(nn.Module):
    """
    CTDE 网络（线性注意力版）。

    与 CTDE_Network 接口完全兼容：
    - get_action(self_state, neighbors, tasks, mask) → Categorical
    - get_value(global_agents, global_tasks, ...) → Tensor

    架构差异：
    - 所有 MultiheadAttention → MultiHeadLinearAttention
    - 其余组件（编码器、PE、task_count 调制、输出头）保持一致
    """

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
        self.actor_kv_mode = str(actor_kv_mode).lower()
        if self.actor_kv_mode not in ("summary", "tokens"):
            raise ValueError("actor_kv_mode must be 'summary' or 'tokens'")

        # ========== Actor ==========
        self.actor_self_enc = nn.Linear(self_dim, hidden)
        self.actor_neighbor_enc = nn.Linear(neighbor_dim, hidden)
        self.actor_task_enc = nn.Linear(task_dim, hidden)

        # 线性注意力替代标准 MHA
        self.actor_neighbor_attn = MultiHeadLinearAttention(
            embed_dim=hidden, num_heads=4, dropout=dropout
        )
        self.actor_task_attn = MultiHeadLinearAttention(
            embed_dim=hidden, num_heads=4, dropout=dropout
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
        self.critic_attn = MultiHeadLinearAttention(
            embed_dim=hidden, num_heads=4, dropout=dropout
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
        Task-as-Query attention with linear complexity.

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

        # 2) Encode neighbors + linear attention
        h_neighbors = self.actor_neighbor_enc(neighbors)  # [B, N_nb, H]
        h_neighbors = self.dropout(h_neighbors)
        nb_padding_mask = (neighbors.abs().sum(dim=-1) < 1e-8)  # [B, N_nb]
        h_neighbor_agg = self.actor_neighbor_attn(
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

        # 7) Task(Q) attend to Agent context(KV) —— 线性注意力
        if self.actor_kv_mode == "tokens":
            kv_tokens = torch.cat([h_self, h_neighbors], dim=1)  # [B, 1+N_nb, H]
            nb_pad = (neighbors.abs().sum(dim=-1) < 1e-8)  # [B, N_nb]
            self_pad = torch.zeros(B, 1, dtype=torch.bool, device=neighbors.device)
            kv_padding_mask = torch.cat([self_pad, nb_pad], dim=1)  # [B, 1+N_nb]
            h_task_attn = self.actor_task_attn(
                h_tasks, kv_tokens, kv_tokens, key_padding_mask=kv_padding_mask
            )
        else:
            h_task_attn = self.actor_task_attn(h_tasks, h_agent, h_agent)  # [B, N_tasks, H]
        h_task_attn = self.dropout(h_task_attn)

        # 8) 融合
        h_fused = h_tasks + h_task_attn  # [B, N_tasks, H]

        # 9) Logits
        logits = self.actor_head(h_fused).squeeze(-1)  # [B, N_tasks]

        # 10) Mask
        logits = logits.masked_fill(mask.bool(), -1e9)

        return Categorical(logits=logits)

    def get_value(self, global_agents, global_tasks, agent_padding_mask=None, task_padding_mask=None):
        """
        Critic: 全局 self-attention → masked mean pooling → MLP → scalar value.

        Args:
            global_agents: [B, N_agents, global_agent_dim]
            global_tasks:  [B, N_tasks, global_task_dim]
            agent_padding_mask: [B, N_agents] bool, True=padding
            task_padding_mask:  [B, N_tasks] bool, True=padding
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

        h_global = self.critic_attn(
            all_entities, all_entities, all_entities,
            key_padding_mask=key_padding_mask
        )

        # Masked mean pooling
        if key_padding_mask is not None:
            valid_mask = ~key_padding_mask  # [B, N], True=valid
            valid_mask_f = valid_mask.unsqueeze(-1).float()
            h_pooled = (h_global * valid_mask_f).sum(dim=1) / valid_mask_f.sum(dim=1).clamp(min=1.0)
        else:
            h_pooled = h_global.mean(dim=1)

        return self.critic_val(h_pooled)


# ═══════════════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing CTDE_Network_Linear (linear attention)...")

    for kv_mode in ["summary", "tokens"]:
        print(f"\n  KV mode: {kv_mode}")
        net = CTDE_Network_Linear(actor_kv_mode=kv_mode)

        B = 4
        self_state = torch.randn(B, 10)
        neighbors = torch.randn(B, 20, 10)

        for N_tasks in [10, 35, 80, 120]:
            tasks = torch.randn(B, N_tasks, 20)
            mask = torch.zeros(B, N_tasks, dtype=torch.bool)
            mask[:, -5:] = True

            global_agents = torch.randn(B, 30, 13)
            global_tasks = torch.randn(B, N_tasks, 23)

            agent_pad = torch.zeros(B, 30, dtype=torch.bool)
            agent_pad[:, -5:] = True
            task_pad = torch.zeros(B, N_tasks, dtype=torch.bool)
            task_pad[:, -3:] = True

            dist = net.get_action(self_state, neighbors, tasks, mask)
            a = dist.sample()
            print(f"    N_tasks={N_tasks:3d}: action={a.shape}, entropy={dist.entropy().mean():.3f}")

            v = net.get_value(global_agents, global_tasks, agent_pad, task_pad)
            print(f"             value={v.shape}, mean={v.mean():.3f}")

    print("\nAll tests passed.")