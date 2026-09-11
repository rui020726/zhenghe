# test_global_input.py
import torch
import numpy as np
from env.task_env import TaskEnv
from attention import CentralAttentionNet

# 创建环境
env = TaskEnv(
    per_species_range=(4, 6),
    species_range=(4, 6),
    tasks_range=(20, 30),
    prio_range=(0.3, 1.0),
    traits_dim=5,
    seed=42
)
env.init_state()

print("✅ Environment created")
print(f"  Agents: {env.agents_num}")
print(f"  Tasks: {env.tasks_num}")

# 获取全局状态
global_agents, global_tasks = env.get_global_state()
global_mask = env.get_global_mask()

print(f"\n✅ Global state extracted:")
print(f"  global_agents: {global_agents.shape}")  # (num_agents, 13)
print(f"  global_tasks: {global_tasks.shape}")    # (num_tasks+1, 18)
print(f"  global_mask: {global_mask.shape}")      # (num_tasks+1,)

# 创建网络
network = CentralAttentionNet(13, 18, 128)

# 模拟输入
agent_indices = torch.tensor([[0, 5, 10]], dtype=torch.long)

global_agents_tensor = torch.tensor(global_agents, dtype=torch.float32).unsqueeze(0)
global_tasks_tensor = torch.tensor(global_tasks, dtype=torch.float32).unsqueeze(0)
global_mask_tensor = torch.tensor(global_mask, dtype=torch.bool).unsqueeze(0)

# Padding
max_agents, max_tasks = 36, 31
if global_agents_tensor.shape[1] < max_agents:
    pad = torch.zeros(1, max_agents - global_agents_tensor.shape[1], 13)
    global_agents_tensor = torch.cat([global_agents_tensor, pad], dim=1)
if global_tasks_tensor.shape[1] < max_tasks:
    pad = torch.zeros(1, max_tasks - global_tasks_tensor.shape[1], 18)
    global_tasks_tensor = torch.cat([global_tasks_tensor, pad], dim=1)
    pad_mask = torch.ones(1, max_tasks - global_mask_tensor.shape[1], dtype=torch.bool)
    global_mask_tensor = torch.cat([global_mask_tensor, pad_mask], dim=1)

print(f"\n✅ Tensors prepared:")
print(f"  agents: {global_agents_tensor.shape}")
print(f"  tasks: {global_tasks_tensor.shape}")
print(f"  mask: {global_mask_tensor.shape}")
print(f"  indices: {agent_indices.shape}")

# 前向传播
with torch.no_grad():
    probs, logps = network(
        global_tasks_tensor,
        global_agents_tensor,
        global_mask_tensor,
        agent_indices
    )

print(f"\n✅ Forward pass successful!")
print(f"  probs: {probs.shape}")  # [1, 3, 31]
print(f"  probs sum: {probs.sum(dim=-1)}")  # 应该接近[1, 1, 1]

print("\n🎉 All tests passed!")