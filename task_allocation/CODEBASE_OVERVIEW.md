# CTDE-MRTA 项目完整说明文档

> 目标：让另一个 AI 或人类审查者无需逐行阅读代码即可理解项目全貌，并定位问题。

---

## 1. 项目是什么

**异构多机器人任务分配（Heterogeneous MRTA）系统**，用于搜救场景模拟。

- **范式**：CTDE（Centralized Training, Decentralized Execution）+ PPO 强化学习
- **场景**：多物种 agent（每种 5 维 binary trait vector）执行多阶段任务（1-4 阶段，每阶段不同需求向量）
- **通信**：Euclidean 距离通信图（comm_range 内一跳），Gossip 协议增量传播（视距感知 + 增量交换 + 时间戳覆盖方向）
- **任务注入**：潮汐式——定时批量注入新任务，仅通知 20% 前线 agent，其余靠通信传染
- **关键约束**：每个任务有 deadline，agent 必须在截止时间前完成

### 文件列表

| 文件 | 作用 |
|------|------|
| `old_task_env.py` | 时间驱动的多智能体环境模拟器 |
| `old_network.py` | Actor-Critic 神经网络（CTDE_Network 类） |
| `old_driver.py` | PPO 训练主程序 |
| `old_test.py` | 模型评估测试程序 |

---

## 2. 环境（old_task_env.py）—— TaskEnv 类

### 2.1 核心概念

- **时步驱动**：dt=1，每步依次执行 移动→到达→通信→潮汐→任务更新→Gossip传播→超时检测
- **Agent 状态机**：`idle → moving → working → idle`，有途重规划（en-route replan）
- **任务**：多阶段，每个阶段有独立的 5 维 binary 需求向量。所有阶段完成才算 finished
- **Deadline**：绝对时间值，超时标记 expired。当前为**软约束**：过期不删除任务，仍可被选择并完成，但会扣分
- **Gossip 传播频率**：每 2 步执行 1 次（`_step_counter % 2 == 0`）

### 2.2 关键参数及默认值

```
traits_dim = 5           # agent 能力 / 任务需求向量维度（binary）
max_task_size = 2        # 任务最大同时工作 agent 数（未使用）
duration_scale = 5-10    # 任务时长缩放（curriculum 逐阶段加大）
comm_range = 0.25-0.5    # 通信半径（curriculum 逐阶段缩小）
dt = 1.0                 # 时步增量
agent velocity = 0.2     # 移动速度（单位/步）
空间范围 = [0,1] × [0,1]  # 所有坐标在此范围内

# Shaping reward 权重
shaping_crowd_penalty = 0.15   # 拥挤惩罚（多人挤同一个任务）
shaping_lonely_bonus = 0.10    # 孤独奖励（填补空缺需求）
shaping_redundant_penalty = 0.30  # 能力不匹配惩罚
shaping_distance_penalty = 0.05   # 距离软惩罚（平方）
```

### 2.3 Agent 数据结构（agent_dic）

```python
{
    "abilities": np.array([0,1,1,0,0]),  # 5维binary能力向量
    "species": 0,                         # 物种ID
    "location": np.array([0.5, 0.3]),     # 当前位置
    "velocity": 0.2,                      # 移动速度
    "status": "idle"|"moving"|"working",  # 当前状态
    "current_task": -1 or task_id,        # 当前任务ID
    "target_pos": None or array,          # 移动目标
    "target_type": None|"task"|"depot",   # 目标类型
    "assigned": bool,                     # 是否已被分配
    "travel_dist": float,                 # 累计移动距离
    "sum_waiting_time": float,            # 累计等待时间
    "need_decision": bool,                # 是否需要决策
    "consecutive_timeouts": int,          # 连续超时次数
    "cooldown_tasks": {tid: end_time},    # 冷却中的任务
    "failed_tasks": [...],                # 失败任务记录
    "working_start_time": float,          # 开始工作时间
    "working_timeout_value": float,       # 超时阈值
}
```

### 2.4 Task 数据结构（task_dic）

```python
{
    "ID": str,                            # 唯一标识
    "location": np.array(shape=2),        # 位置
    "deadline": float,                    # 截止时间（绝对）
    "expired": bool,                      # 是否已过期
    "finished": bool,                     # 是否已完成
    "total_stages": 1-4,                  # 总阶段数
    "current_stage_index": int,           # 当前阶段索引
    "stages": [                           # 阶段列表
        {
            "stage_id": 0,
            "requirements": np.array(5),   # 本阶段需求向量（binary）
            "status": np.array(5),         # 剩余需求（初始=requirements）
            "time": float,                 # 阶段时长
            "time_left": float,            # 剩余时间
            "finished": bool,
        },
        ...
    ],
    "members": [agent_id, ...],           # 当前在组agent
    "feasible_assignment": bool,          # 当前是否存在可行分配
    "scene_complexity": 0.3-1.0,          # 场景复杂度
    "time_start": float,                  # 任务创建时间
    "time_finish": float|0,               # 任务完成/过期时间
}
```

### 2.5 Agent 知识结构（agent_knowledge）

每个 agent 维护一个本地知识字典，通过视距感知和 gossip 更新：

```python
{
    "known_tasks": set(),                 # 已知任务ID集合
    "known_finished": set(),              # 已知已完成任务ID集合
    "known_assigned": set(),              # 已知已分配任务ID集合
    "dynamic_tasks": set(),               # 动态注入的任务ID集合
    "task_locations": {tid: array},       # 已知任务位置
    "task_deadlines": {tid: float},       # 已知任务deadline
    "task_current_status": {tid: array},  # 已知任务当前剩余需求
    "task_requirements": {tid: array},    # 已知任务总需求
    "task_members": {tid: [agent_ids]},   # 已知任务组成员
    "task_last_update_time": {tid: float},# 信息最后更新时间
}
```

---

## 3. 网络（old_network.py）—— CTDE_Network 类

### 3.1 输入输出维度

```
Actor 输入:
  self_state:     [B, 10]   # agent自身特征
  neighbors:      [B, N_nb, 10]  # 邻居特征（≤20个，零填充）
  tasks:          [B, N_tasks, 20]  # 候选任务特征（可变长度，零填充）
  mask:           [B, N_tasks]  # True=illegal

Critic 输入:
  global_agents:  [B, N_agents, 13]  # 全局agent特征
  global_tasks:   [B, N_tasks, 23]   # 全局任务特征

Actor 输出: Categorical(logits=[B, N_tasks])
Critic 输出: scalar value [B, 1]
```

### 3.2 Self State 的 10 维组成

0-4:   agent abilities (5维 binary)
5-6:   agent location (2维)
7:     current_task (float)
8:     travel_dist (float)
9:     sum_waiting_time (float)

### 3.3 Neighbor 的 10 维组成（同 self，但来自邻居）

0-4:   neighbor abilities (5维)
5-6:   neighbor location (2维)
7:     neighbor current_task (float)
8:     neighbor assigned (float)
9:     neighbor travel_dist (float)

### 3.4 Task 观测的 20 维组成（Actor）

```
0-4:   task_current_status (5维)       # 当前剩余需求
5-9:   future_req (5维)               # 下一阶段需求
10-11: relative location (2维)         # (task_location - agent_location) / 1.414
12:    deadline_urgency (1维)          # tanh(time_until_deadline / 50.0)
13:    scene_complexity (1维)          # 0.3-1.0
14:    num_members (1维)               # len(members) / 10.0
15:    known_finished flag             # task是否在自己的known_finished中
16:    known_assigned flag             # task是否在自己的known_assigned中
17:    dynamic_tasks flag              # task是否为动态任务
18:    staleness (1维)                 # tanh((current_time - last_update) / 50.0)
19:    time_until_deadline_linear (1维) # min(time_until_deadline / 100.0, 1.0) ← 最近新增
```

### 3.5 Global Agent 的 13 维组成（Critic）
```
0-4:   abilities (5)
5-6:   location / 1.0 (2)
7:     current_task (float)
8:     assigned (float)
9-10:  预留位（填0）
11:    travel_dist
12:    sum_waiting_time
```

### 3.6 Global Task 的 23 维组成（Critic）
```
0-4:   task status (5)
5-9:   future_req (5)
10-14: task requirements (5)
15-16: location / 1.0 (2)
17:    feasible_assignment (float)
18:    deadline_norm (deadline / (current_time+100))
19:    scene_complexity
20:    time_start
21:    time_finish
22:    len(members)
```

### 3.7 Actor 架构（前向流程）

```
1. self_state → Linear(self_dim, hidden) → h_self [B,1,H]
2. neighbors → Linear(neighbor_dim, hidden) → MHA(h_self, h_nb, h_nb, key_padding_mask=nb_pad) → h_neighbor_agg [B,1,H]
3. h_agent = h_self + h_neighbor_agg
4. task_count_embed: Linear(1, H/4) → ReLU → Linear(H/4, H)  输入为 valid_counts/100
5. h_agent += count_feat * 0.1
6. tasks → Linear(task_dim, H) → + PosEncoding → h_tasks [B,N_t,H]
7. Task(Q) attend to Agent(K/V): h_task_attn = MHA(h_tasks, kv, kv, key_padding_mask)
   - kv_mode="summary": kv = h_agent [B,1,H]
   - kv_mode="tokens":  kv = cat([h_self, h_neighbors]) [B,1+N_nb,H]
8. h_fused = h_tasks + h_task_attn
9. logits = Linear(H, 1).squeeze(-1) → masked_fill(mask, -1e9)
10. return Categorical(logits=logits)
```

### 3.8 Critic 架构

```
1. agents → Linear(13, H), tasks → Linear(23, H)
2. all_entities = cat([h_agents, h_tasks]) [B, N_a+N_t, H]
3. key_padding_mask = cat([agent_pad, task_pad])
4. MHA(all_entities, all_entities, all_entities, key_padding_mask) → h_global
5. Masked Mean Pooling (排除padding) → h_pooled
6. Linear(H, 64) → ReLU → Dropout → Linear(64, 1) → scalar
```

### 3.9 关键超参数

```
hidden = 128
dropout = 0.15
num_heads = 4 (MHA)
use_pos_encoding = True (Sinusoidal)
actor_kv_mode = "tokens" (self + neighbors 作为 KV)
```

---

## 4. 训练（old_driver.py）

### 4.1 PPO 参数

```
LR = 3e-4 → 5e-5 (ep 14000-20000 线性衰减)
GAMMA = 0.99
GAE_LAMBDA = 0.95
K_EPOCHS = 3
EPS_CLIP = 0.2
MAX_EPISODES = 20000
N_ENVS = 8 (并行环境数)
UPDATE_STEP = 8192 (每次更新收集的 step 数)
MINI_BATCH_SIZE = 1024
ENTROPY_START = 0.02 → ENTROPY_END = 0.003 (ep 0→8000 线性衰减)
GRAD_NORM = 1.0
AMP = bfloat16
```

### 4.2 5 阶段 Curriculum

| 阶段 | Episode | 初始任务 | duration_scale | deadline tightness | comm_range |
|------|---------|---------|---------------|-------------------|------------|
| 1 | 0-500 | 8-15 | 3 | (1.0, 2.5) | 0.50 |
| 2 | 500-5000 | 15-25 | 5 | (0.7, 2.0) | 0.40 |
| 3 | 5000-11000 | 20-30 | 7 | (0.7, 1.5) | 0.35 |
| 4 | 11000-15000 | 25-35 | 9 | (0.5, 1.2) | 0.28 |
| 5 | 15000-20000 | 30-40 | 10 | (0.4, 1.0) | 0.25 |

每 episode 开始时随机采样（含 ±jitter），以 domain randomization 防止过拟合。

### 4.3 训练流程

```
1. 8个env并行step（ThreadPoolExecutor, 12线程）
2. 每个env内：收集每个需要决策的agent的state → batch前向actor → 采样action → apply → collect reward
3. 累积 UPDATE_STEP=8192 步 → GAE计算advantage → K_EPOCHS=3 mini-batch PPO更新
4. 每个episode结束 → get_episode_reward() → inject_terminal_reward(spread模式，均摊到episode各步)
5. 阶段完成奖励回填：task_update中发放 → driver通过agent_ids字段定位buffer位置回填
6. 每1000 ep保存checkpoint
```

### 4.4 并行与环境交互

```
ThreadPoolExecutor(max_workers=CPU_THREADS=12) 执行env.step()
每个step: 8个env并发step → 对每个env收集agent决策 → batch inference → apply_action
```

---

## 5. 奖励体系

### 5.1 Shaping Reward（即时，每 agent 选任务时）

在 `_compute_shaping_reward(agent_id, task_id)` 中计算：

```
reward = 0

# 拥挤惩罚
if remaining_needs <= 0:
    reward -= crowd_penalty * (total_others + 1)
elif total_others > 0:
    reward -= crowd_penalty * max(0, total_others - remaining_needs)

# 孤独奖励（填补空缺需求）
if remaining_needs > 0 and my_fill > 0:
    reward += lonely_bonus * my_fill * (1.5 - urgency)
elif still_needed.sum() <= 0 and total_others > 0:
    reward -= crowd_penalty * 0.5

# 距离惩罚（平方，近距离几乎无惩罚）
reward -= distance_penalty * (dist / 1.414)^2

# 能力匹配/不匹配
if my_contribution > 0:
    reward += 0.05 * my_contribution * (1.5 - urgency)
else:
    reward -= redundant_penalty * 0.33

# 过期任务额外惩罚（软约束）
if task["expired"]:
    reward -= 0.5

# urgency = tanh(time_until_deadline / 50.0)
# urgency=1.0 表示充裕，urgency≈0 表示紧迫
```

### 5.2 终端奖励（Episode 结束）

在 `get_episode_reward()` 中计算，7 个组成部分：

```
# 任务分类统计
n_finished_on_time:  finished=True AND expired=False
n_finished_late:     finished=True AND expired=True
n_expired_unfinished: not finished AND expired=True
n_unfinished:        not finished AND not expired

# 1. 完成奖励
completion = 200 * (n_finished_on_time + 0.5 * n_finished_late) / n_total

# 2. 速度奖励（仅准时完成）
speed = 40 * sum(max(0, 1 - finish_time / T_norm)) / n_total

# 3. Slack奖励（仅准时完成，早于deadline越多越好）
slack = 40 * sum(min(1, (deadline - finish_time) / T_norm)) / n_total

# 4. 未完成惩罚
unfinished_penalty = 40 * (n_unfinished / n_total)

# 5. 过期惩罚
expired_penalty = 80 * ((n_finished_late + n_expired_unfinished) / n_total)

# 6. 阶段完成奖励
stage_reward = 20 * (completed_stages / total_stages)

# 7. Failed tasks 惩罚（agent timeout后计入）
failed_penalty = 1.0 * (total_failed / n_agents)

# 8. Makespan 惩罚
makespan_penalty = 15 * (makespan / T_norm)

total = completion + speed + slack + stage - unfinished - expired - makespan - failed
```

四种状态净收益梯度：**按时+240 >> 过期后完成+20 > 未触碰-40 > 过期放弃-120**

### 5.3 阶段完成中间奖励

在 `task_update()` 中，阶段切换时发放。每个在组成员获得：
```
reward = 5.0 * urgency
```
其中 `urgency = 1.5 - tanh(max(0, deadline - current_time) / 50.0)`，越紧迫奖励越高。

---

## 6. deadline 机制（最近重构的核心部分）

### 6.1 过期检测（task_update）

```python
# 当前（软约束）
if current_time > task["deadline"] and not task.get("expired") and not task.get("finished"):
    task["expired"] = True
    # 不设 finished=True → 不释放成员 → 任务仍在观测中

# 之前在旧版（硬约束，已废弃）：
    # task["finished"] = True  ← 这导致过期=完成，骗取完成奖励
    # self._release_agent(aid)
    # task["members"] = []
```

### 6.2 deadline 计算公式

```python
travel_time = min_dist_to_any_depot / 0.2
total_work = sum(stage["time"] for all stages)
estimated = travel_time + total_work + 10.0  # 10 是 buffer
tightness = uniform(tightness_lo, tightness_hi)
deadline = base_time + max(estimated * tightness, estimated * 0.7)
```

注意：`estimated * 0.7` 是 2026-05-24 的修改。之前是 `total_work * 1.1`，后者不包含旅行时间，导致低 tightness 时 slack 不够覆盖移动距离。新公式的保底值包含 70% 的预估时间（含旅行时间），确保物理可行。

### 6.3 观测中的时间信号（双通道）

```
12维: deadline_urgency = tanh(time_until_deadline / 50.0)  # 非线性，50步内敏感
19维: time_linear = min(time_until_deadline / 100.0, 1.0)  # 线性，100步内精确倒计时
```

19 维之前是重复的 num_members（与 14 维重复），2026-05-17 替换。

---

## 7. 测试（old_test.py）

### 7.1 测试设置

```
3 种 agent 规模 × 5 种任务负载 = 15 组实验
每组 100 episodes, MAX_STEP=3000
DEADLINE_TIGHTNESS = (0.8, 1.5)  # 2026-05-24 从 0.5 提升到 0.8
CHECKPOINT_CANDIDATES = ["ctde_final.pth", "ctde_best.pth"]
EVAL_GREEDY = False  # 采样，非贪心
ACTOR_KV_MODE = "tokens"
```

### 7.2 Agent 规模

| Scale | species_range | per_species_range | 典型 agent 数 |
|-------|--------------|-------------------|-------------|
| small | (2, 3) | (3, 5) | ~7 |
| medium | (3, 4) | (4, 6) | ~14 |
| large | (4, 5) | (5, 7) | ~22 |

### 7.3 任务负载

(8,15), (15,25), (20,30), (30,40), (40,50) — 初始任务数，潮汐注入另加 ~170 个

### 7.4 测试指标

```
task_completion_rate:  所有任务中 finished=True 的比例
deadline_success_rate: finished=True AND expired=False 的比例（核心指标）
expired_rate:          expired=True 的比例
resource_utilization:  agent 处于非 idle 状态的时间占比
avg_slack:             按时完成的任务的平均 slack 时间
episode_reward:        get_episode_reward() 返回值
```

### 7.5 模型加载

```
strict=True 加载 state_dict
先尝试 tokens 模式，失败再尝试 summary 模式
```

---

## 8. 已知问题 & 已修复 Bug

### 已修复

| 问题 | 文件 | 修复时间 | 影响 |
|------|------|---------|------|
| ThreadPoolExecutor 未 import | old_driver.py | 5/11 | 运行时 NameError |
| CHECKPOINT_CANDIDATES 缺逗号 | old_test.py | 5/11 | 字符串拼接，永远找不到文件 |
| task_source 清空时机错误 | old_task_env.py | 5/11 | 初始任务标签丢失 |
| neighbor attention 缺 padding mask | old_network.py | 5/11 | 零填充参与注意力计算 |
| SAVE_EVERY_EP 缩进错误 | old_driver.py | 5/11 | 类定义解析异常 |
| Gossip 频率 2→3 步不一致 | old_task_env.py | 5/11 | 测试失真（已回退到 2） |
| **过期任务骗取完成奖励** | old_task_env.py | 5/17 | 训练学到"等过期"策略 |
| **观测缺线性时间信号** | old_task_env.py | 5/17 | deadline 感知不足 |
| **deadline 硬约束→软约束** | old_task_env.py | 5/17 | 过期即消失，无补救动力 |
| **deadline 保底公式不含旅行时间** | old_task_env.py | 5/24 | 低 tightness 物理不可行 |
| **测试 tightness 下限 0.5 太低** | old_test.py | 5/24 | 测试时产生不可行任务 |

### 仍然存在的问题

1. **Shaping reward 淹没终端信号**：episode reward 600-1000+，terminal reward 只占 3-14%。agent 日常收入远超 deadline 奖惩

2. **Agent 大量闲置**：Large 规模 idle rate 73-80%，22 个 agent 只干了 5-6 个的活。agent 宁可 idle 也不选过期/远距离任务

3. **终端奖励太稀疏**：3000 步才结算一次 deadline，没有中间催促信号

4. **多阶段信用分配**：同任务多 agent 平分阶段奖励，无法区分个体贡献差异

5. **Curriculum 中 Stage 4-5 的 tightness < 0.7**，所有任务 deadline 都落在 `estimated * 0.7` 保底值，随机性退化

6. **资源利用率极低**：即使是最好的 Large 规模，利用率也仅 20-27%

### 测试结果历史

| 日期 | DL SR | 完成率 | 过期率 | 资源利用 | 备注 |
|------|-------|--------|--------|---------|------|
| 5/17 修复前 | 10-22% | 94% | 71-85% | 15-17% | 旧 reward bug |
| 5/20 修复后(Large) | 31-39% | 94% | 55-64% | 20-27% | 软约束有效但不够 |
| 5/20 修复后(Small) | 2.8-14.5% | 52-88% | 79-92% | 54-92% | 修复后更差，暴露产能不足 |
| 5/24 Phase1后 | 待测试 | — | — | — | 刚改完公式 |

---

## 9. 待办改进方向

按优先级排列：

1. **Phase 1（已完成 5/24）**：deadline 公式改 `estimated*0.7` 保底 + tightness 下限提至 0.4
2. **Phase 2（待做）**：增强 deadline 信号——每步 urgency bonus、过程中阶段进度奖励、提升终端奖励权重
3. **Phase 3（待做）**：解决 agent 闲置——idle 惩罚、过期任务阶梯惩罚（非一刀切 -0.5）、放宽 mask
4. **Phase 4（待做）**：信用分配优化——按能力贡献比例分配阶段奖励

---

## 10. 文件路径

```
D:\code\task-03-修改为CTDE范式\
├── old_task_env.py      # 环境模拟器（~1000行）
├── old_network.py       # CTDE 网络（~240行）
├── old_driver.py        # PPO 训练（~800行）
├── old_test.py          # 评估测试（~500行）
├── checkpoints_old/5/20/# 模型保存目录
├── test_results/        # 测试结果 JSON
├── create_ppt.py        # PPT生成脚本
└── weekly_report.pptx   # 汇报PPT
```
