# ROUND32 归约数学：两臂到底改变了什么（任务 B，0 环境启动）

证据文件 `REDUCTION_MATH.json`（由 `r32_math.py` 生成），实现 `r32_trainer.py`。
本轮唯一主动改变量是 **actor surrogate 的归约集合与分母**；本文把"变了什么"和"没变什么"
都写成可核对的量，而不是声明。

## 1. 前提核对：冻结实现确实是"按记录行平均"

`v2_trainer.ppo_update` 里 actor 项是

```python
surrogate = torch.min(ratio * advantage, clamp(ratio, 1-eps, 1+eps) * advantage)   # 逐行张量
loss = (-surrogate.mean() + VALUE_COEFFICIENT*(value_task_loss + value_cost_loss)
        - ENTROPY_BONUS * entropy_t.mean())
```

`batch_samples` 一行 = 一步决策的 tape（不是一条候选消息），所以 `surrogate.mean()` 的分母
就是该 minibatch 的记录行数 `N`。前提成立。

**关键细节（本轮实测，不是推断）**：`onpolicy.joint_log_prob` 返回
`(log_probabilities[rows, picked] * usable).sum()`，`usable` 是采样时刻保存的自由集合。
一行若没有自由候选，它的 `new_log_prob` 是**恒等于 0 的常量**：

- 该行 **actor 梯度恒为 0**——在任何权重下都成立（在 3 组权重扰动上实测均为 `0.0`），
  不只是在 `ratio = 1` 处成立；
- 但该行 `s_i = min(ratio·A_i, clamp·A_i) = A_i` 的**数值不为 0**，仍进入 `surrogate.mean()`
  的分子与分母。

因此 `N/M` 关系只对**梯度**成立，对 loss **标量**不成立。两臂的 `policy_surrogate` 读数因此
不可直接互比，故每批同时记录与臂无关的 `policy_surrogate_all_rows_reference`。

## 2. 两臂定义

```text
N = 该 minibatch 的记录行数
M = 其中 actionable/free 非空的行数（active 行）

ALL_ROWS     actor 项 = -(1/N) * Σ_{i<N} s_i            （冻结表达式逐字保留）
ACTIVE_ROWS  actor 项 = -(1/max(M,1)) * Σ_{i<N} s_i · 1[i active]
```

`M = 0` 时 `(surrogate * row_active).sum()` 本身就是精确 0 且带 `grad_fn`，可安全反向；
不需要特殊分支，也不会产生 NaN。

active 身份取自 tape 保存的 `actionable`/`free`（顺序 mask 执行时刻写入）；PPO 重算不重新采样。
合法采样 HOLD 的行仍 active。一行有多个自由候选也只计一行权重。

## 3. 独立反例与接线测试（全部穿过实际 actor loss 路径）

合成批：12 行、9 行 active、3 行整行被 mask（`0 < M < N`，归约确实有差别）。

| 检查 | 结论 | 实测值 |
|---|---|---|
| T1 混合行梯度恒等式 | 通过 | `grad_ACTIVE = (N/M)·grad_ALL`，缩放后最大残差 1.9e-08（float32 求和舍入）；`l2` 比 1.333333439 vs 理论 4/3 |
| T1b 无效行零梯度与权重无关 | 通过 | 三组权重（0, +0.013, −0.021）下该行 actor 梯度最大绝对值均为 `0.0` |
| T2 全 active | 通过 | 用真实 all-active 布局（不篡改 `free`），两臂 loss 与梯度一致 |
| T3 全 masked | 通过 | actor 梯度 `l2 = 0.0`、loss 恰为 0 且有限；3 个 critic 样本全部保留 |
| T4 HOLD 计入 M | 通过 | 全 HOLD 批 `M = N = 12`；`M` 只取决于哪些候选自由，与选了哪个动作无关 |
| T5 真实顺序 mask 接线 | 通过 | ROUND30 old 臂 6 条真实 tape：`actionable == free`、三类候选闭合到 targets、masked 候选从不出现在 `send_probabilities`、每条都有整行被 mask 的行 |
| T6 补零行反例（仅合成） | 通过 | 追加 4 个零梯度行后 `ALL` 梯度按 `N/N' = 12/16 = 0.75` 缩放（实测 0.74999998），`ACTIVE` 梯度差 `0.0` |
| T7 其余损失不变 | 通过 | entropy/两 critic loss/`approx_kl`/`clip_fraction` 两臂逐位相等；`batch_samples` 与冻结件逐字段相等；总梯度差 = actor 项差（残差 1.2e-08） |
| T8 恢复契约 | 通过 | payload 与 `training_state` 记录 `reduction_mode`；跨归约恢复被拒；ROUND30 ep0 checkpoint 被拒；同模式往返权重哈希相同 |
| T9 ALL_ROWS ≡ 冻结训练器 | 通过 | 同一 fresh controller、同一合成批、Adam(3e-4)，两轮完整 minibatch 序列后**权重哈希逐次相等**；13 个冻结标量字段与 `module_gradient_l1_mean` 全等；权重确实动了 |
| T10 归约确实生效 | 通过 | minibatch 0 上"未被触碰的量"（entropy/两 critic/KL/clip/N/M/λ/ratio）两臂相等，而 `actor_surrogate_mean` 与**裁剪前总梯度范数**两臂已不同——后者正是被干预量，要求它相等反而是错的 |

## 4. 逐行差异登记（`source_diff`）

`difflib` 对 `v2_trainer.ppo_update` 与 `r32_trainer.ppo_update`（去注释行）分块分类：
`research_variable` 8 块、`passive_read` 4 块、**unexplained 0 块**。被删除的父件行只有 3 行，
且逐行归类：

1. `loss = (-surrogate.mean()` —— 研究变量本身；
2. 函数签名行 —— 新增 `reduction` 关键字参数（管线）；
3. `totals["surrogate"] += float(surrogate.mean().item())` —— 仅报告字段改为读臂自身的均值；
   对 `ALL_ROWS` 二者数值相同（T9 已验证 13 个标量字段全等）。

`r32_trainer.py` 刻意把 import 别名保持与父件一致（`common` 仍是 `v2_common`、`cm` 仍是
`cost_math`），因此副本里 `common.CLIP`、`common.ENTROPY_BONUS` 等逐字不变；
`identity_assertions` 进一步断言 `batch_samples / _bundle / likelihood_alignment / dual_update /
batch_g_hat / cm.gae / cm.actor_advantage` 与冻结件是**同一函数对象**，全部为真。

## 5. 作用范围（必须如实说明，不得扩大）

- 本消融**改变 actor 相对 entropy 与两个 critic 的更新权重**。这是研究变量的一部分，不是副作用；
- 它**不改变**：优势构造与归一化（两者都不做标准化）、GAE、奖励时间归属、两个 critic 的样本
  集合与归约、entropy 的归约与系数、λ 更新与其使用时序、PPO clip、KL、minibatch 构造与顺序、
  optimizer/学习率/全局梯度裁剪、网络、观测、mask、环境；
- 它也**不改变**总梯度的其它组成部分，因此不能把观察到的任何差异直接叫做"参数更新倍数"：
  `N/M` 只作用在 actor 项上，Adam 的非线性、critic 项与全局裁剪（`MAX_GRAD_NORM=0.5`，
  ROUND31 实测多数 minibatch 触发）都会改变最终步长。
  放大是否 surviving 到总梯度与裁剪之后，由任务 G 用批 0/15/31 的**更新前 checkpoint** 单独回答。

## 6. 资源计数

```text
环境启动 = 0                    新流 = 0
真实训练更新 = 0                合成 optimizer 更新 = 24 / 64（T9 16 + T10 8，单独分账）
被修改的历史产物 = 0            生产代码改动 = 0
```
