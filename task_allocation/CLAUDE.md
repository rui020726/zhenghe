# CTDE + PPO 异构多机器人任务分配 (Heterogeneous MRTA)

## 项目概要

研究方向：**CTDE（集中训练+分布式执行）+ PPO** 解决异构多机器人任务分配问题。
场景：多物种 agent（每物种 5 维 binary trait vector），多阶段任务（1-4 stage），潮汐式任务注入，Gossip 通信传播，deadline 软约束。

## 快速开始

```bash
# 训练
python old_driver.py

# 测试（用训练好的 checkpoint）
python old_test.py --checkpoint ./checkpoints_old/5/28/ctde_best.pth --tag exp_name

# 监控
tensorboard --logdir runs/
```

## 文件地图

### 核心系统（old_* — 唯一在用、功能完整的版本）

| 文件 | 行数 | 职责 |
|------|------|------|
| `old_task_env.py` | ~1000 | 环境模拟：多阶段任务、Gossip 通信、deadline 软约束、shaping/episode reward |
| `old_network.py` | ~240 | CTDE 网络：Actor (self→neighbor attention→task attention)，Critic (global attention→MLP) |
| `old_driver.py` | ~830 | PPO 训练：8 并行 env、GAE(λ)、5 阶段 curriculum、AMP/TF32 加速、断点续训 |
| `old_test.py` | ~490 | 模型评估：3 agent 规模 × 5 任务负载 = 15 组实验，DL-SR/expired/slack 指标 |

### 消融实验（Ab_* — 功能完整，独立运行）

| 文件 | 职责 |
|------|------|
| `Ab_task_env.py` | 带 `pre_positioning` 开关的 env（agent 可预知后续 stage 需求） |
| `Ab_old_network.py` | 维度可切换网络（15/18 baseline vs 20/23 pre-positioning） |
| `Ab_old_driver.py` | 消融训练 driver（一键切换 baseline/proposed） |
| `ab_old_test.py` | 消融测试（核心指标：stage gap time） |

### 简化重写版（driver.py + network.py + env/task_env.py + test.py）

从零重写的简化版（单阶段任务、无潮汐、MC returns），**功能不完整**，作为 old_* 的对照参考。

### 废弃/不完整

| 文件 | 问题 |
|------|------|
| `runner.py` + `worker.py` + `attention.py` + `central_controller.py` | Ray 分布式架构，worker 调用的 env 方法不存在 |
| `parameters.py` | Ray 架构的超参数，未被 old_* 使用 |

### 辅助工具

| 文件 | 职责 |
|------|------|
| `baselines.py` | 经典启发式 baseline（Hungarian, Greedy-*, CBBA） |
| `baseline_comparison.py` | 学习策略 vs baseline 对比 |
| `animate_run.py` / `render_demo.py` | 环境可视化 |
| `plot_train_curves.py` / `plot_multi_agst_results.py` | 训练/测试曲线绘图 |
| `generate_test_set.py` | 测试集生成器 |
| `fast_test.py` | 轻量快速测试 |

## 环境设计（old_task_env.py）

### 任务模型
- 1~4 个 stage，每 stage 有独立的 **binary requirement vector**（5 维，对应 5 种能力）
- stage 完成后自动切换，**释放无贡献的 agent**，通知新 stage 需求
- 三种来源：`initial`（初始生成）、`tidal`（潮汐爆发注入）、`discovered`（行进中动态发现）

### Agent 模型
- 每 agent 属于一个物种（species），物种决定其 binary ability vector
- **单任务专注**：每 agent 同时只能做一个任务的**一个能力维度**
- 状态：`idle` → `moving` → `working` → `idle`
- working timeout：连续工作超时衰减 + cooldown，防止死锁

### 通信模型
- 欧几里得距离二值通信图（comm_range 内一跳邻居）
- **Gossip 协议增量传播**：视距感知 + 一跳 exchange（增量交换 known_tasks/known_finished/task status + 时间戳覆盖）
- 信息不对称：潮汐任务仅通知 20% 前线 agent，其余靠通信传染
- 降频优化：每 `gossip_interval` 步（默认 2）传播一次

### Deadline 软约束
- 每个任务在创建时分配 absolute deadline：`deadline = max(estimated * tightness, estimated * 0.95)`
- `estimated = min_travel_time + total_work + 10.0`（buffer）
- **过期不删除、不释放成员、不标记完成**，任务仍可见可选
- 过期完成给 10% 奖励（vs 按时 100%），过期选择给小额正向引导激励

### 观测空间

**Local obs（20 维，per agent-task pair）：**
- dim 0-4: task current status（5）
- dim 5-9: future stage requirements or zeros（5）
- dim 10-11: relative location（2，归一化）
- dim 12: deadline urgency `tanh(remaining/50)`（1）
- dim 13: scene complexity（1）
- dim 14: num_members / 10（1）
- dim 15-19: info features — known_finished, known_assigned, dynamic_tasks, info_age, time_until_deadline（5）

**Global obs（23 维，per task）：**
- dim 0-4: status, dim 5-9: future req, dim 10-14: requirements, dim 15-16: location/1.414
- dim 17: feasible_assignment, dim 18: deadline_norm, dim 19: complexity
- dim 20: time_start, dim 21: time_finish, dim 22: num_members

**Self state（10 维）：** abilities(5) + location(2) + current_task + travel_dist + sum_waiting_time

## 网络架构（old_network.py）

```
CTDE_Network:
  Actor:
    self_encoder: Linear(10→128) → ReLU
    neighbor_encoder: Linear(10→128) → ReLU
    neighbor_attn: MultiHeadAttention(self→neighbors, with padding mask)
    task_encoder: Linear(20→128) → ReLU
    task_attn: MultiHeadAttention(Task=Q, Agent_context=K/V)  ← CTDE 核心设计
    task_count_modulation: FiLM-style conditioning
    sinusoidal_position_encoding: 可选
    output: Linear(256→1) per task → mask → softmax → action

  Critic:
    global_agent_encoder + global_task_encoder
    self_attention(agents+tasks concat)
    masked_mean_pooling → MLP → scalar value
```

关键设计决策：
- **Task attends to Agent**（不是 Agent attends to Task）——让每个任务通过 attention 找到最合适的 agent
- Actor KV 模式：`tokens`（self+neighbors 多 token，带 padding mask）
- Dropout 0.15 正则化

## 训练流程（old_driver.py）

### PPO 配置
- 8 并行环境，ThreadPoolExecutor
- UPDATE_STEP=8192，MINI_BATCH_SIZE=1024，K_EPOCHS=3
- GAE(λ=0.95) 计算 advantage，按 episode 分段
- 动态 padding batch → mini-batch PPO
- AMP bfloat16 + TF32 + inference_mode 加速

### 5 阶段 Curriculum

| Stage | Episode | 初始任务 | Duration | Deadline | Comm | Discovery |
|:-----:|---------|:--------:|:--------:|:--------:|:----:|:---------:|
| 1 热身 | 0-500 | 8-15 | ×3 | 1.5-2.5 | 0.50 | 0.01 |
| 2 起步 | 500-5K | 15-25 | ×5 | 1.2-2.0 | 0.40 | 0.03 |
| 3 进阶 | 5K-11K | 20-30 | ×7 | 1.0-1.5 | 0.35 | 0.05 |
| 4 主训 | 11K-15K | 25-35 | ×9 | 1.0-1.5 | 0.28 | 0.07 |
| 5 高压 | 15K+ | 30-40 | ×10 | 1.0-1.3 | 0.25 | 0.08 |

- 均匀任务池采样（全区间采样避免灾难性遗忘）
- Domain randomization：comm jitter、duration jitter、task count jitter
- OOD 抽查：20% 概率用 OOD 任务范围（35-50）
- LR 线性衰减：10000→15000（3e-4 → 5e-5）
- Entropy 线性衰减：0→8000（0.02 → 0.003）

### Reward 设计

**Shaping reward（每步，选任务时发放）：**
- 拥挤惩罚（-0.15）：去任务的人超过需求
- 孤独奖励（+0.10×urgency）：能填补空缺需求
- 距离软惩罚（-0.05×dist²）：远距离才显著
- 能力匹配（+0.05×urgency）：有能力贡献
- 冗余惩罚（-0.10）：无贡献
- 过期引导激励（+0.05×decay）：仍值得做但随时间衰减
- 总体系数：×0.25

**Episode reward（终末，spread 到 buffer）：**
| 项 | 系数 | 说明 |
|----|:----:|------|
| 完成奖励 | A=600 | `(n_on_time + 0.1×n_late) / n_total` |
| Speed 奖励 | λ=120 | 越早完成奖励越高 |
| Slack 奖励 | λ=120 | deadline - finish_time |
| 未完成惩罚 | η=40 | `n_unfinished / n_total` |
| 过期惩罚 | ω=400 | `n_expired / n_total` |
| 阶段完成 | γ=20 | 完成 stage 的比例 |
| Makespan | μ=15 | 最后完成时间 |
| Failed tasks | φ=1 | 失败的尝试次数 |

## 测试评估（old_test.py）

### 规模配置（方案B，2026-05-28）

| 规模 | species | per_species | 平均 agent |
|:----:|:-------:|:-----------:|:----------:|
| Small | 3-4 | 4-5 | 16 |
| Medium | 3-4 | 5-7 | 21 |
| Large | 4-5 | 5-7 | 22 |

### 任务负载（5 档）
[8-15], [15-25], [20-30], [30-40], [40-50]

### 核心指标
- **DL-SR**（Deadline Success Rate）：按时完成（不含过期）的任务比例
- **Task Completion**：总完成率（含过期完成）
- **Expired Rate**：过期任务比例
- **Avg Slack**：按时完成任务的平均 slack 时间
- **Resource Utilization**：agent 工作时间占比
- **Idle Rate**：agent 空闲时间占比

### 分来源统计
按 initial / tidal / discovered 三种来源分别统计 DL-SR 和完成率

## 当前性能（checkpoint 5/25，旧 reward）

| 规模 | DL-SR | 完成率 | Expired | 闲置率 |
|:----:|:-----:|:-----:|:-------:|:-----:|
| Small (7ag) | 4-19% | 53-87% | 75-90% | 9-45% |
| Medium (14ag) | 22-35% | 93%+ | 59-72% | 55-68% |
| Large (22ag) | 45-56% | 94%+ | 38-51% | 73-80% |

问题：Small 规模 agent 不足，Medium/Large 大多过期完成而非按时完成。

## 最近修改（2026-05-28）

在旧 checkpoint 基础上做了以下改动，**需要重新训练验证**：

1. **Reward 重构**（old_task_env.py）：
   - 过期任务：`-0.5` 惩罚 → `+0.05×decay` 引导式激励
   - 按时/过期梯度：5x → 10x（过期完成系数 0.2→0.1）
   - Episode reward 系数翻倍：A 300→600, lam 80→120, omega 200→400

2. **Agent 规模调整**（方案B）：Small 7→16, Medium 14→21, 训练时 10→18

3. **Deadline 放宽**：Stage 4/5 下限 0.95→1.0, 测试 (1.0,1.5)→(1.2,1.8)

4. **Gossip 降频参数化**：硬编码 → `gossip_interval=2` 可配置

5. **训练周期**：20000→15000, LR 衰减 10000→15000

6. **断点续训**：RESUME/RESUME_CKPT 支持自动查找最新 checkpoint

## 关键参数速查

```python
# 环境
traits_dim = 5          # 能力维度（binary）
max_task_size = 2       # 每任务最多 2 agent 同时工作
comm_range = 0.12~0.60  # 通信半径（curriculum 递减）

# 网络
hidden = 128            # 隐藏层维度
attention_heads = 4
dropout = 0.15
actor_kv_mode = "tokens"  # self+neighbors 多 token

# 训练
N_ENVS = 8
UPDATE_STEP = 8192
K_EPOCHS = 3
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPS_CLIP = 0.2

# Checkpoint
CHECKPOINT_DIR = "./checkpoints_old/5/28"
```

## 创新点与后续方向

### 已有创新
1. **Deadline 软约束**：过期不删除、不释放、仍可选、引导式激励（区别于 TARGETNET 的硬 -1 惩罚）
2. **潮汐式任务注入 + 信息不对称**：模拟真实物流/仓储的订单到达模式
3. **Gossip 增量通信**：时间戳驱动的增量交换，降频优化
4. **多阶段任务**：阶段切换释放无贡献 agent，中间奖励回填
5. **四分类 episode reward**：按时/过期完成/过期未完/未触碰，细粒度反馈

### 后续方向（来自记忆）
- 能量受限 agent（电池/充电约束）
- 对抗环境任务分配
- 灾难响应复杂场景（UAV-UGV 协同）
- 动态联盟形成
- 弹性制造/仓库重调度

### 已知问题
- 多物种能力覆盖不足时 generate_env 可能无限循环
- 大规模 agent 闲置率高（70-80%），需要更好的任务-agent 匹配
- 潮汐注入频率和数量可能需要自适应调节
