import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from parameters import *


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 辅助函数：掩码生成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_global_attn_pad_mask(seq_q, seq_k):
    """
    生成全局注意力掩码：屏蔽填充/无效的任务/智能体
    seq_q: 查询序列（如智能体特征），形状 [batch_size, num_queries, dim]
    seq_k: 键序列（如任务特征），形状 [batch_size, num_keys, dim]
    返回：全局掩码 [batch_size, num_queries, num_keys]，True表示屏蔽
    """
    batch_size, num_q, _ = seq_q.size()
    batch_size, num_k, _ = seq_k.size()

    # 屏蔽填充的查询（如无效智能体）
    pad_mask_q = seq_q.eq(0).all(dim=-1).unsqueeze(2)  # [batch, num_q, 1]
    # 屏蔽填充的键（如已完成任务）
    pad_mask_k = seq_k.eq(0).all(dim=-1).unsqueeze(1)  # [batch, 1, num_k]

    # 扩展为全局掩码：[batch, num_q, num_k]
    global_mask = pad_mask_q.expand(batch_size, num_q, num_k) | pad_mask_k.expand(batch_size, num_q, num_k)
    return global_mask


def global_padding_inputs(inputs_list, pad_value=0):
    """
    对批量全局数据（如多批任务/智能体）进行零填充，统一序列长度
    inputs_list: 列表，每个元素为 [num_tasks/agents, input_dim]
    返回：填充后的数据 [batch, max_seq_len, input_dim]、掩码 [batch, max_seq_len]
    """
    padded = pad_sequence(inputs_list, batch_first=True, padding_value=pad_value)
    mask = (padded.sum(dim=-1) == 0).unsqueeze(2)  # [batch, max_seq_len, 1]
    return padded, mask


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 基础模块：注意力机制
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SingleHeadAttention(nn.Module):
    """单头注意力：适配集中式，支持多查询（所有智能体）对多键（所有任务）"""

    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.key_dim = embedding_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)
        self.tanh_clipping = 10  # 防止注意力分数过大

        # 线性变换矩阵
        self.w_query = nn.Parameter(torch.Tensor(embedding_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(embedding_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h, mask=None):
        """
        q: 多查询（所有智能体特征）→ [batch, num_agents, embedding_dim]
        h: 键值（所有任务特征）→ [batch, num_tasks, embedding_dim]
        mask: 全局掩码 → [batch, num_agents, num_tasks]
        返回：注意力概率、对数概率
        """
        batch_size, num_agents, _ = q.size()
        num_tasks = h.size(1)

        # 线性变换
        Q = torch.matmul(q, self.w_query)  # [batch, num_agents, key_dim]
        K = torch.matmul(h, self.w_key)  # [batch, num_tasks, key_dim]

        # 全局注意力分数
        attn_scores = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        attn_scores = self.tanh_clipping * torch.tanh(attn_scores)

        # 应用掩码
        if mask is not None:
            mask_expanded = mask
            if mask.dim() != attn_scores.dim():
                mask_expanded = mask.expand_as(attn_scores)
            attn_scores = attn_scores.masked_fill(mask_expanded.bool(), -1e8)

        # 计算概率
        probs = torch.softmax(attn_scores, dim=-1)
        logps = torch.log_softmax(attn_scores, dim=-1)

        return probs, logps


class MultiHeadAttention(nn.Module):
    """多头注意力：强化全局特征交互"""

    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"

        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // n_heads
        self.key_dim = self.head_dim
        self.value_dim = self.head_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        # 多头线性变换
        self.w_query = nn.Parameter(torch.Tensor(embedding_dim, n_heads * self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(embedding_dim, n_heads * self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(embedding_dim, n_heads * self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(n_heads * self.value_dim, embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q

        batch_size, num_q, _ = q.size()
        num_k = h.size(1)

        # 线性映射和多头拆分
        Q = torch.matmul(q, self.w_query)
        K = torch.matmul(h, self.w_key)
        V = torch.matmul(h, self.w_value)

        Q = Q.view(batch_size, num_q, self.n_heads, self.key_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, num_k, self.n_heads, self.key_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, num_k, self.n_heads, self.value_dim).permute(0, 2, 1, 3)

        # 注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.norm_factor

        # 应用掩码
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [B, 1, N_q, N_k]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        heads = torch.matmul(attn_weights, V)
        heads_merged = heads.permute(0, 2, 1, 3).reshape(batch_size, num_q, -1)
        out = torch.matmul(heads_merged, self.w_out)

        return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 前馈网络模块
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class GateFFNDense(nn.Module):
    """门控前馈网络"""

    def __init__(self, model_dim, hidden_unit=512):
        super(GateFFNDense, self).__init__()
        self.W = nn.Linear(model_dim, hidden_unit, bias=False)
        self.V = nn.Linear(model_dim, hidden_unit, bias=False)
        self.W2 = nn.Linear(hidden_unit, model_dim, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, hidden_states):
        hidden_act = self.act(self.W(hidden_states))
        hidden_linear = self.V(hidden_states)
        hidden_states = hidden_act * hidden_linear
        hidden_states = self.W2(hidden_states)
        return hidden_states


class GateFFNLayer(nn.Module):
    """门控前馈层"""

    def __init__(self, model_dim):
        super(GateFFNLayer, self).__init__()
        self.DenseReluDense = GateFFNDense(model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        return forwarded_states


class Normalization(nn.Module):
    """归一化层"""

    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Encoder 模块
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EncoderLayer(nn.Module):
    """单层编码器"""

    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(embedding_dim, n_head)
        self.norm1 = Normalization(embedding_dim)
        self.ffn = GateFFNLayer(embedding_dim)
        self.norm2 = Normalization(embedding_dim)

    def forward(self, x, mask=None):
        # 残差连接+多头注意力
        x = x + self.mha(q=self.norm1(x), h=None, mask=mask)
        # 残差连接+前馈网络
        x = x + self.ffn(self.norm2(x))
        return x


class GlobalEncoder(nn.Module):
    """全局编码器（多层）"""

    def __init__(self, embedding_dim=128, n_head=8, n_layer=2):
        super(GlobalEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, n_head) for _ in range(n_layer)
        ])

    def forward(self, x, mask=None):
        """x: 全局序列 → [batch, seq_len, embedding_dim]"""
        for layer in self.layers:
            x = layer(x, mask)
        return x


# 为兼容性添加 Encoder 别名
Encoder = GlobalEncoder


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Decoder 模块
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CentralDecoderLayer(nn.Module):
    """集中式解码器层"""

    def __init__(self, embedding_dim, n_head):
        super(CentralDecoderLayer, self).__init__()
        self.agent_self_mha = MultiHeadAttention(embedding_dim, n_head)
        self.agent_task_mha = MultiHeadAttention(embedding_dim, n_head)
        self.norm1 = Normalization(embedding_dim)
        self.norm2 = Normalization(embedding_dim)
        self.ffn = GateFFNLayer(embedding_dim)
        self.norm3 = Normalization(embedding_dim)

    def forward(self, agents_x, tasks_x, mask=None):
        # 智能体间自注意力
        agents_x = agents_x + self.agent_self_mha(q=self.norm1(agents_x), h=None)
        # 智能体-任务交叉注意力
        agents_x = agents_x + self.agent_task_mha(q=self.norm2(agents_x), h=tasks_x, mask=mask)
        # 前馈网络
        agents_x = agents_x + self.ffn(self.norm3(agents_x))
        return agents_x


class CentralDecoder(nn.Module):
    """集中式解码器"""

    def __init__(self, embedding_dim=128, n_head=8, n_layer=2):
        super(CentralDecoder, self).__init__()
        self.layers = nn.ModuleList([
            CentralDecoderLayer(embedding_dim, n_head) for _ in range(n_layer)
        ])

    def forward(self, agents_x, tasks_x, mask=None):
        """
        agents_x: [batch, num_agents, embedding_dim]
        tasks_x: [batch, num_tasks, embedding_dim]
        mask: [batch, num_agents, num_tasks]
        """
        for layer in self.layers:
            agents_x = layer(agents_x, tasks_x, mask)
        return agents_x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 集中式注意力网络（并行版本）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CentralAttentionNet(nn.Module):
    """
    集中式网络（并行决策版本）
    所有智能体同时决策
    """

    def __init__(self, agent_input_dim=13, task_input_dim=18, embedding_dim=128):
        super(CentralAttentionNet, self).__init__()

        # 输入嵌入
        self.agent_embedding = nn.Linear(agent_input_dim, embedding_dim)
        self.task_embedding = nn.Linear(task_input_dim, embedding_dim)

        # 位置编码
        self.agent_pos_encoding = nn.Parameter(torch.randn(1, 100, embedding_dim) * 0.02)
        self.task_pos_encoding = nn.Parameter(torch.randn(1, 100, embedding_dim) * 0.02)

        # 编码器
        self.agentEncoder = GlobalEncoder(embedding_dim=embedding_dim, n_head=8, n_layer=3)
        self.taskEncoder = GlobalEncoder(embedding_dim=embedding_dim, n_head=8, n_layer=3)

        # 跨模态交互
        self.agent_task_cross_attn = MultiHeadAttention(embedding_dim, n_heads=8)
        self.task_agent_cross_attn = MultiHeadAttention(embedding_dim, n_heads=8)

        # 决策智能体特征提取
        self.decision_agent_transform = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

        # 输出层
        self.pointer = SingleHeadAttention(embedding_dim)

    def forward(self, tasks, agents, global_mask, agent_indices):
        """
        Args:
            tasks: [batch, max_tasks, task_dim]
            agents: [batch, max_agents, agent_dim]
            global_mask: [batch, max_tasks]
            agent_indices: [batch, num_valid_agents]

        Returns:
            probs: [batch, num_valid_agents, max_tasks]
            logps: [batch, num_valid_agents, max_tasks]
        """
        batch_size = tasks.shape[0]
        num_tasks = tasks.shape[1]
        num_agents = agents.shape[1]
        num_valid = agent_indices.shape[1]

        # 嵌入 + 位置编码
        task_embed = self.task_embedding(tasks)
        agent_embed = self.agent_embedding(agents)

        task_embed = task_embed + self.task_pos_encoding[:, :num_tasks, :]
        agent_embed = agent_embed + self.agent_pos_encoding[:, :num_agents, :]

        # 全局编码
        task_encoded = self.taskEncoder(task_embed, mask=None)
        agent_encoded = self.agentEncoder(agent_embed, mask=None)

        # 跨模态交互
        agent_with_task = self.agent_task_cross_attn(
            q=agent_encoded, h=task_encoded, mask=None
        )
        agent_encoded = agent_encoded + agent_with_task

        task_with_agent = self.task_agent_cross_attn(
            q=task_encoded, h=agent_encoded, mask=None
        )
        task_encoded = task_encoded + task_with_agent

        # 提取需要决策的智能体
        expanded_indices = agent_indices.unsqueeze(-1).expand(
            batch_size, num_valid, agent_encoded.shape[-1]
        )

        valid_agents_encoded = torch.gather(agent_encoded, 1, expanded_indices)
        decision_agents = self.decision_agent_transform(valid_agents_encoded)

        # 计算分配分数
        probs, logps = self.pointer(
            q=decision_agents,
            h=task_encoded,
            mask=global_mask.unsqueeze(1).expand(batch_size, num_valid, num_tasks)
        )

        return probs, logps


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 自回归集中式网络（顺序决策版本）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AutoregressiveCentralNet(nn.Module):
    """
    自回归集中式网络
    每次生成一个智能体的动作，状态实时更新
    """

    def __init__(self, agent_input_dim=13, task_input_dim=18, embedding_dim=128):
        super(AutoregressiveCentralNet, self).__init__()

        # 编码器（只运行一次）
        self.agent_embedding = nn.Linear(agent_input_dim, embedding_dim)
        self.task_embedding = nn.Linear(task_input_dim, embedding_dim)

        self.agent_pos_encoding = nn.Parameter(torch.randn(1, 100, embedding_dim) * 0.02)
        self.task_pos_encoding = nn.Parameter(torch.randn(1, 100, embedding_dim) * 0.02)

        self.agentEncoder = GlobalEncoder(embedding_dim=embedding_dim, n_head=8, n_layer=3)
        self.taskEncoder = GlobalEncoder(embedding_dim=embedding_dim, n_head=8, n_layer=3)

        # 跨模态交互
        self.agent_task_cross_attn = MultiHeadAttention(embedding_dim, n_heads=8)
        self.task_agent_cross_attn = MultiHeadAttention(embedding_dim, n_heads=8)

        # 解码器（每个智能体运行一次）
        self.state_embedding = nn.Sequential(
            nn.Linear(embedding_dim + 1, embedding_dim),  # +1 for is_assigned flag
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

        self.query_generator = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),  # agent + state
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

        self.pointer = SingleHeadAttention(embedding_dim)

    def encode(self, tasks, agents):
        """
        编码阶段（只运行一次）

        Args:
            tasks: [batch, max_tasks, task_dim]
            agents: [batch, max_agents, agent_dim]

        Returns:
            task_encoded: [batch, max_tasks, embedding_dim]
            agent_encoded: [batch, max_agents, embedding_dim]
        """
        batch_size = tasks.shape[0]
        num_tasks = tasks.shape[1]
        num_agents = agents.shape[1]

        # 嵌入 + 位置编码
        task_embed = self.task_embedding(tasks)
        agent_embed = self.agent_embedding(agents)

        task_embed = task_embed + self.task_pos_encoding[:, :num_tasks, :]
        agent_embed = agent_embed + self.agent_pos_encoding[:, :num_agents, :]

        # 编码
        task_encoded = self.taskEncoder(task_embed, mask=None)
        agent_encoded = self.agentEncoder(agent_embed, mask=None)

        # 跨模态交互
        agent_with_task = self.agent_task_cross_attn(
            q=agent_encoded, h=task_encoded, mask=None
        )
        agent_encoded = agent_encoded + agent_with_task

        task_with_agent = self.task_agent_cross_attn(
            q=task_encoded, h=agent_encoded, mask=None
        )
        task_encoded = task_encoded + task_with_agent

        return task_encoded, agent_encoded

    def decode_step(self, agent_idx, task_encoded, agent_encoded, task_assigned_flags, task_mask):
        """
        解码一步（为单个智能体生成动作）

        Args:
            agent_idx: 当前智能体索引 [batch]
            task_encoded: [batch, max_tasks, embedding_dim]
            agent_encoded: [batch, max_agents, embedding_dim]
            task_assigned_flags: [batch, max_tasks] 任务是否已分配 (0/1)
            task_mask: [batch, max_tasks] 任务掩码 (True=不可选)

        Returns:
            probs: [batch, max_tasks]
            logps: [batch, max_tasks]
        """
        batch_size = task_encoded.shape[0]
        num_tasks = task_encoded.shape[1]

        # 提取当前智能体特征
        expanded_idx = agent_idx.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, 1, agent_encoded.shape[-1]
        )
        current_agent = torch.gather(agent_encoded, 1, expanded_idx).squeeze(1)

        # 任务状态嵌入（动态）
        task_with_state = torch.cat([
            task_encoded,
            task_assigned_flags.unsqueeze(-1)
        ], dim=-1)

        task_state_embed = self.state_embedding(task_with_state)

        # 生成查询
        avg_task_state = task_state_embed.mean(dim=1)
        query_input = torch.cat([current_agent, avg_task_state], dim=-1)
        query = self.query_generator(query_input).unsqueeze(1)

        # 计算注意力分数（应用掩码）
        probs, logps = self.pointer(
            q=query,
            h=task_state_embed,
            mask=task_mask.unsqueeze(1)
        )

        return probs.squeeze(1), logps.squeeze(1)