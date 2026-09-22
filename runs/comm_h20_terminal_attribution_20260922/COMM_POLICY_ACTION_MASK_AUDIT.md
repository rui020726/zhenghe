# COMM_POLICY_ACTION_MASK_AUDIT

日期：2026-09-22  
分支：`analysis/comm-h20-terminal-attribution`  
范围：只读诊断。**本轮不修训练器、不改协议、不跑训练。**

问题：预算可行性如何从 observation → action mask → sampling → log_prob → rollout →
optimizer → relay execution 传递？预算不足的 SEND 能否被采样、是否进入训练？
能否恢复“真正双动作自由”样本数？

## 一、结论摘要

| 问题 | 结论 |
|---|---|
| 预算不足 SEND 能否被采样？ | **能**。预算**不是** action mask。 |
| 是否进入训练 buffer / PPO loss？ | **是**（只要不是静默上界 forced 项）。 |
| log_prob 是否只覆盖合法动作？ | 只覆盖 **非 forced**；**不**按预算可行性过滤。 |
| rollout → optimizer 有无二次预算过滤？ | **无**。 |
| relay 拒绝时训练是否知情？ | **样本级不知情**；仅 episode 级 `refused_*` 计数。 |
| 能否从训练记录恢复双动作自由样本数？ | **不能** → **标缺失**。 |
| 离线可恢复？ | **能**（actionability 审计）：`both_actions_free=101` 点 / 8 流。 |

## 二、完整路径（file:line）

```text
observation
  integrated_sar/comm_policy.py:85-86   remaining_budget_bytes / budget_bytes_per_episode
  integrated_sar/comm.py:840-848        注入 observation
        ↓ 仅作特征
  integrated_sar/comm_learning.py:90-95,117-118
        budget_fraction / capacity_fraction（软特征，无 attempt_charge_bytes）
        ↓
action mask（唯一）
  runs/comm_budget_pilot_20260919/train_pilot.py:95-99
        forced(item) = age_steps ≥ MAX_WAIT（静默上界；urgent 时限更短）
        —— 无预算项
        ↓
sampling
  runs/comm_tight_budget_20260919/onpolicy.py:145-175
        must = forced(item) → 强制 SEND，不记 likelihood
        free 候选：multinomial(softmax(logits)) 可抽 SEND 或 HOLD
        —— 预算不足仍可抽到 SEND
        ↓
log_prob
  onpolicy.py:46-56,177,186-196
        joint_log_prob = Σ free 项；forced → usable=0
        tape: features/actions/free/log_prob/value/forced
        —— 无 send_executable / send_block_reason / payload_bytes
        ↓
rollout buffer
  runs/comm_tight_budget_20260919/train_onpolicy.py:113-141 flatten()
        原样打包 features/actions/free/log_prob/advantage
        ↓
optimizer
  train_onpolicy.py:145-216 update()
        usable 仅按 free（静默上界）加权；PPO ratio 用同一 log_prob
        refused_by_policy 只进汇总计数（:277 一带）
        ↓
relay execution（真正硬门）
  integrated_sar/comm.py:851-886 _emit()
        cost_bytes = payload_bytes × subscribers
        if not budget.can_send:  # comm_budget.py:68-81
            counters["budget_refused_sends"] += 1
            transcript "refused"
            return   # stays queued；不改 log_prob、不改样本
```

### 2.1 observation

`SendObservation.remaining_budget_bytes` / `budget_bytes_per_episode` / `capacity_remaining`
进入特征（`comm_learning.py:90-100,117-118`），是**比例/剩余量**，不是一次发送的
`attempt_charge_bytes`。策略看到“还剩多少”，但看不到“这一发要花多少、现在付不付得起”。

### 2.2 action mask

唯一硬 mask 是静默上界 `forced()`（`train_pilot.py:95-99`），在 `OnPolicyCollector.choose`
里变成 `must`（`onpolicy.py:145-147`）。forced 项被强制 SEND 且 `usable=0`（不进 actor 似然）。
**预算不参与 mask。** 文档与测试一致：budget 是 executor 约束，不是策略动作空间约束。

### 2.3 sampling

`onpolicy.py:175`：`torch.multinomial(probabilities[index], …)`。只要不是 `must`，
SEND/HOLD 都可被抽到，**与剩余预算无关**。

### 2.4 log_prob

`joint_log_prob`（`onpolicy.py:46-56`）只按 `usable`（= free，非 forced）加权。
预算非法的 SEND **仍计入** log_prob 与 entropy。

### 2.5 rollout → optimizer

`flatten` / `update`（`train_onpolicy.py:113-216`）对 free 决策全量进 surrogate。
`usable = target in free`，**不是** `send_executable`。无二次预算过滤。
`refused_by_policy` 仅汇总，不剔样本、不改 advantage。

### 2.6 relay execution

`comm.py:862-886`：`can_send` / `why_blocked`（`comm_budget.py:68-81`）在这里才拒绝。
拒绝写 counters + transcript，目标继续排队。训练侧无样本级回写。

## 三、明确回答：预算不足 SEND

1. **能被采样** —— 是。
2. **能进入训练** —— 是：进 tape → flatten → PPO surrogate（只要非 forced）。
3. **语义** —— 优化的是“策略想不想发”，不是“发得出去”。拒发的世界后果只通过
   episode 级 reward/GAE 间接回来，且 action 本身仍占 actor 似然。
4. 与已修的 PPO 概率不一致（`COMM_PPO_PROBABILITY_AUDIT.md`）**不是同一问题**：
   那是 μ≠π；本项是动作空间未含预算可行性。

## 四、真正双动作自由样本数

定义：同一候选点上 SEND 可执行 **且** HOLD 可执行（`both_actions_free`）。

### 4.1 训练记录：不能恢复 → **缺失**

| 字段 | tape 有？ | 双动作自由所需 |
|---|---|---|
| `free` / `forced` | 有 | 仅静默上界 ≠ 预算可行 |
| `features` | 有 | 含 `budget_fraction`，**无** `attempt_charge_bytes` |
| `send_executable` / `send_block_reason` | **无** | 必需 |
| `payload_bytes` × `subscribers` | **无** | 必需（才能重算 `can_send`） |
| 决策时刻 `budget_remaining` 原值 | **无**（仅比例） | 建议补 |
| 与 transcript `"refused"` 的逐步关联 | **无** | 建议补 |

训练侧可见的 `free_candidates` / `forced_sends`（如 `COMM_PPO_PROBABILITY_AUDIT.json`、
`TIGHT_SMOKE.json`）是**静默上界自由**，**不是**预算双动作自由。二者不可混用。

**结论：训练 buffer / replay 无法恢复真正双动作自由样本数 → 本轮标 `training_dual_action_free_samples: missing`。**

### 4.2 离线行动性审计：可恢复

`runs/comm_send_actionability_20260922/`（8 条 free 流，23612 候选）：

| 主类 | 计数 |
|---|---:|
| **both_actions_free** | **101**（8 流） |
| send_blocked_credit | 23511 |
| 其他阻塞 | 0 |

- post-first-emission 双动作可行：71 点 / 8 流。
- 主阻塞是**预算**（decision-wtd ~99.6%），不是容量或协议强制。
- 来源字段齐全：`send_executable` / `send_block_reason`（`audit_runner.py`、
  `send_actionability.py:107-108`）。

这是**审计表**上的双动作自由，不是训练 buffer 回放计数；可作后续训练修复后的验收对照。

## 五、缺陷定性（只诊断，不修）

| 项 | 状态 |
|---|---|
| 采样分布 = 记录 log_prob（PPO ratio 自洽） | **正确**（已在 `COMM_PPO_PROBABILITY_AUDIT` 修复并测过） |
| forced 项不进 actor 似然 | **正确** |
| 预算可行性 ∈ action mask | **未实现**（设计上放在 executor） |
| 非可执行 SEND 不进 actor loss | **未实现** |
| 双动作自由样本可审计 | **训练路径缺失**；离线审计已有 |
| relay 拒绝 → 样本级训练信号 | **缺失** |

**定性**：不是 log_prob/采样器错误；是 **动作空间 / rollout 过滤 / 可观测性缺口**——
预算不足的 SEND 仍占策略似然与 PPO 目标，且训练记录无法区分“想发”和“发得出”。
对“学习在预算下何时发”的估计对象，这构成 **训练正确性风险**（信用分配到不可执行动作）。

按预注册判定第 3 条：**若视其为预算动作 mask/rollout 正确性缺陷，下一步应先单独修复训练路径
（mask 或样本过滤 + 补 `send_executable`/`attempt_charge_bytes` 等字段），不继续通信效果实验。**
本轮只登记，不顺手改训练器。

## 六、修复方向（仅建议，不在本轮实施）

1. 在 `OnPolicyCollector.choose` 记录 `send_executable` / `send_block_reason` /
   `attempt_charge_bytes`（或 `payload_bytes`+`subscriber_count`）。
2. 明确估计对象：若学“可执行时机”，则对不可执行 SEND 的 actor 项置 `usable=0`（对齐 forced）；
   若学“偏好”，则保持现状但报告中必须分开计数。
3. rollout→optimizer 按 `send_executable` 过滤或降权；transcript `refused` 逐步对齐。
4. 验收：训练路径重算的 `both_actions_free` 应对齐 `COMM_SEND_ACTIONABILITY_ROWS` 同类定义。

## 七、判读边界

- 代码路径以仓库当前源码与 `runs/comm_tight_budget_20260919` / `comm_budget_pilot_20260919`
  训练脚本为准；未重跑训练。
- 101 双动作自由点来自 8 条开发 free 流，不作总体外推。
- 本审计不评价 CTDE 任务分配 mask（`old_task_env.get_hard_mask`），那是另一条动作路径。
