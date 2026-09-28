# ROUND23 GAE 与奖励时间契约

## 事件顺序

`Task03Adapter.step()` 先调用 `CommRelay.tick()`，再推进 `env.step(dt)`。relay tick 内，策略先选动作并写入 collector tape；成本 tap 在 tick 返回后读取发送计数增量；环境推进后，task tap 才记录该 tick 新完成的 on-time task。

```text
物理 tick d：  CommRelay.tick → 采样动作 a_d / 写 tape → 记录本 tick 成本
              Task03Adapter 推进 env.step → 产生 task reward → 记录 task reward
```

因此 tick `d` 的成本和任务奖励都发生在动作 `a_d` 之后。tick `d` 到下次动作前的事件属于 `a_d`；动作之前产生的前缀事件不能归给未来动作。

## 归属规则与四类案例

给定决策 tick `[7, 10]`、horizon 12：

| 事件 tick | 相对动作的位置 | 归属 |
|---:|---|---|
| 2 | 第一个动作之前 | 未分配前缀，单独记账 |
| 7 | 动作 7 采样后的同 tick | 决策 7 |
| 9 | 动作 7 与动作 10 之间 | 决策 7 |
| 12 | 最后动作 10 之后 | 决策 10 |

合成反例使用 task reward `{2:1, 7:2, 9:3, 12:4}`、cost bytes `{2:50, 7:100, 9:200, 12:300}`。正确分配为：task 前缀 1、决策 7 的 task reward 5、决策 10 的 task reward 4；cost 前缀 50 bytes、决策 7 计 300 bytes、决策 10 计 300 bytes。task/cost 全量账本仍闭合，但前缀不进入 actor 或 critic target。

零 value、`GAE_LAMBDA=0.95` 时决策 7 的 task return 是 `5 + 0.95×4 = 8.8`，决策 10 是 4。按 `B0=10000`，cost returns 分别为 0.0585 和 0.03。`test_sc_reward_timing.py` 同时检查四类事件和 checker 独立重算；旧“分配到后一个决策”实现给决策 7 的 return 是 9.55，并未通过反例。

总量闭合与时序归属是两项检查：task 的 `assigned + prefix = task_reward_total`；cost 的 `assigned_cost/B0 + prefix_cost/B0 = spent_bytes/B0`。闭合本身不能证明归属正确。

## GAE 时间单位和 next-value

一个 tape 行代表一个决策样本。此版 `gamma=1`、`GAE_LAMBDA=0.95`，λ 按 tape 中相邻决策行递推。若两次决策间隔多个物理 tick，当前登记协议仍按一个决策转移处理；不会因 gamma 为 1 就把 GAE λ 的时间单位改成物理 tick。本轮不调整超参。

每行的 `bootstrap[t]` 表示**该行转移的 next-state value**：非终止行取下一决策样本冻结 rollout 的 task/cost value；真实终止行取 0。若 rollout 因截断结束而非环境终止，最后一行传入截断后 observation 的 value。每个 episode 单独计算，GAE 不跨 episode 延续。

递推式为：

```text
delta_t = r_t + gamma * (1-terminal_t) * V_next_t - V_t
A_t     = delta_t + gamma * gae_lambda * (1-terminal_t) * A_(t+1)
return_t = V_t + A_t
```

非零 value 的闭式反例 `r=[0,1]`、`V=[0.4,0.6]`、`terminal=[false,true]`、`gamma=1`、`lambda=0.95`、`next_V=[0.6,0]` 得：

```text
delta      = [0.2, 0.4]
advantage  = [0.58, 0.4]
return     = [0.98, 1.0]
```

旧全零 next-value 在第一行得到 `A_0=-0.02`。对应直接闭式、截断 bootstrap 和 episode 隔离的合成检查位于 `test_sc_gae_next_value.py`；该测试也经过 `batch_samples` 核对行为 tape value。

## 历史重算边界

历史样本的只读重算记录在 `CORRECTED_TAPE_RECOMPUTE.json`。旧 per-tape advantage/return 没有落盘，文件把旧值标为按旧代码语义重建；修复后重算不是策略反事实成绩，也没有改写训练权重。
