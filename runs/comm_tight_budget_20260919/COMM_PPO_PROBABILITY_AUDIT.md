# PPO 行为概率一致性审计与修复

日期：2026-09-19。分支：`fix/comm-ppo-probability-budget`，起点
`feat/communication-training-stability@7834034b58fcb71dda2643f8216cbc8639189a69`。

对应总纲 v1.3 §5.5 与执行文档 `COMM_PPO_BUDGET_NEXT_ROUND.md` §2。本轮属于 S3：训练器正确性修复，
不启动任务分配重构，不改变通信协议、任务难度或分配模型。

## 一、缺陷是什么（源码事实）

`runs/comm_budget_pilot_20260919/train_pilot.py` 的 `Collector.choose` 在 161–173 行：

```python
if self.greedy:
    action = ...
elif random.random() < self.epsilon:      # epsilon = 0.10
    action = random.randint(0, 1)         # 均匀随机
else:
    action = int(torch.multinomial(probabilities[index]...))
actions[item.target_id] = action
log_prob += float(log_probabilities[index, action].item())   # 记录的是纯 pi
```

`train()` 以 `epsilon=EXPLORATION_EPSILON`（0.10）构造 rollout 采集器（459 行），`update()` 用同一
`log_prob` 作 `old_log_prob` 计算 PPO ratio（334 行）。因此实际优化的重要性是

```text
mu(a|s) = 0.9 * pi(a|s) + 0.1 * Uniform(a)      采样
log pi(a|s)                                      记录与更新使用
```

两者不是同一个分布。执行文档 §2 指定的最小修复方向是取消额外的 ε-greedy 覆盖，直接从有掩码的
策略分布采样，并记录对应的 `old_log_prob`；保留 entropy 正则与随机采样，不改成 greedy 训练。

不采用"把混合策略当作被优化策略、同时改采样与新旧似然"那条更复杂的路线；本轮也没有那样做。
只改 `old_log_prob` 而 numerator 仍用纯 π，不能恢复标准 on-policy PPO。

## 二、为什么三轮体检没有发现它

因为常规体检对它不敏感。`ratio` 在权重未更新时仍然等于 1（新旧都用同一个 π），参数照样更新，
学习曲线照样上升。执行文档 §2 明确写了这一点："旧错误实现也可能通过'ratio 初始为 1'，因此不能
只做这一项；必须核对真实采样分布。"

审计因此改测**采样频率与所记录分布的一致性**：对每条 rollout 记录，用记录里的特征重算 π，比较
实际抽到的动作频率，并逐决策算 `log mu(a|s) - log pi(a|s)` —— 这正是 PPO ratio 吞下的偏差量。

## 三、实测结果

主体：`train_pilot.Controller` 的种子 1 未训练初始化，以及第三轮 seed 1 在 400k 环境步的
checkpoint（`runs/comm_stability_20260919/train/seed1/stability_seed1_at400000.pt`）。
两条臂都在同样的 6 条训练流（700000–700005）、同样的隔离无损设置、同样的 87 900 B 额度上跑，
每集一个新采集器（`run_one` 用 `collector.records` 按步建表，步号跨集重复，共用采集器会让后一集
重复报告前一集的记录——这一点在本文件第一版里就错过，已修正）。

机器可读结果：`COMM_PPO_PROBABILITY_AUDIT.json`。

| 主体 | 采集器 | 决策数 | 自由候选 | 抽到的发送率 | 所记录 π 的发送率 | 差 | 差 / SE | mean(log μ−log π) | mean\|·\| | 最大 \|·\| |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 未训练初始化 | ε-greedy（旧） | 4 276 | 707 | 0.5728 | 0.5514 | +0.0214 | **1.15** | −0.0002 | 0.0016 | 0.075 |
| 未训练初始化 | 纯 π（新） | 4 167 | 808 | 0.5285 | 0.5399 | −0.0114 | −0.65 | — | — | — |
| 400k 已训练 | ε-greedy（旧） | 509 | 1 018 | **0.1778** | **0.1261** | **+0.0517** | **4.32** | **+0.0339** | **0.1502** | **1.394** |
| 400k 已训练 | 纯 π（新） | 505 | 1 028 | 0.1479 | 0.1621 | −0.0142 | −1.29 | — | — | — |

权重指纹：未训练 `7d7f7152f3104470`，已训练 `c8743282b00c3f84`；两者各自在同一主体的两条臂之间
完全相同，所以差异只可能来自采样器。

读法：

1. **未训练时看不出问题**。π 在初始化时接近均匀（0.55 附近），混合分布
   `0.9π + 0.05` 与 π 本身就差不了多少，逐决策的 log 偏差均值 −0.0002 nat，最大 0.075 nat，
   频率差 1.15 个标准误——统计上与噪声不可分。这就是三轮体检放它过去的机制。
2. **策略一旦变自信，偏差就显形**。本轮测的已训练策略（第三轮 seed 1 @400k）把发送率压到
   π≈0.126，旧采样器实际抽到 0.1778 —— **相对高估 41%**，4.32 个标准误；逐决策 `log μ−log π` 平均
   +0.034 nat（这是在 μ 抽样下对 KL(μ‖π) 的蒙特卡洛估计，非负），平均绝对值 0.150 nat，
   单条最坏 1.394 nat，即某个动作的真实重要性权重比被错报约 4 倍。
3. **新采集器在同一把尺子上回到噪声内**：两个主体的频率差分别是 −0.65 与 −1.29 个标准误，
   且逐决策 `joint_log_prob` 重算与记录值的最大绝对差为 0（未训练）／1e−6（已训练）。

因此这不是一处"数值上无所谓"的记录差异：它系统性地奖励了低概率动作，并把策略实际想发送的比例
抬到约 1.4 倍。旧结论必须限定为"该训练实现得到的策略未优于强规则"，而不是"规则已足够／该观测
动作没有规则外空间"——这也是总纲 §5.5 已经写下的降级。

## 四、改了什么

新增 `runs/comm_tight_budget_20260919/onpolicy.py`：

* `OnPolicyCollector.choose` —— 自由候选直接从 `softmax(logits)` 抽（`torch.multinomial`），
  没有 ε 覆盖；记录的 `log_prob` 就是这些抽到的动作在同一个 π 下的对数概率之和。评估仍走
  `greedy=True` 的部署口径，训练与评估的分工不变。
* `joint_log_prob` 与 `action_entropy` —— 采样侧与更新侧共用的唯一一份似然实现。`usable`
  掩码把被静默上界接管的候选乘成 0：既不进 likelihood，也不进 entropy，因此不给 actor 造梯度。
* `rng_state` / `restore_rng` —— Python、NumPy、Torch 三条随机流的保存与恢复。
* 记录里新增 `send_probabilities`（每个自由候选的 π(send)）与 `forced`（当步被上界接管的集合），
  即执行文档要求的"记录选择时的可行动作集合"与真实发送分布。

新增 `train_onpolicy.py`：同样的网络、奖励、clip 0.2、GAE λ0.95、γ0.99、value 0.5、
entropy 0.01、Adam 3e−4、grad clip 0.5、4 epoch、minibatch 256、4 episode/轮；`update()` 用上述
共用似然函数，并在每次更新前逐带重算 `joint_log_prob` 与记录值比较，超过 1e−4 nat 直接
`RuntimeError` 拒绝训练。检查点带 `trainer: "onpolicy"` 标记，`load()` 拒绝加载 ε-greedy 时代的
checkpoint —— 新实验从初始权重重训，不继承已学会完全等待的策略。

刻意没有改的东西：环境、C 臂分配、19 维 `comm-send-state/1` 观测、网络结构、任务流、奖励定义、
候选顺序、3/10 步共同静默上界、预算执行器、任何默认开关。没有引入 ACK，没有改通信范围、
deadline 或动作空间。最终测试 `800000..800399` 继续封存，本轮未生成、未查看。

旧的高预算 checkpoint 与两轮结果原样保留为历史对照；本提交不重跑、不改写它们的任何数字。

## 五、新增测试

`integrated_sar/tests/test_comm_ppo_probability.py`（12 项），逐条对上执行文档 §2 的清单：

| 文档要求 | 测试 |
|---|---|
| 固定可枚举概率的采样与日志概率一致 | `test_drawn_actions_match_the_distribution_that_is_logged`（p=0.75，4 000 抽，频率与均值 log 各按 4 SE 判） |
| （同一检查必须能失败） | `test_a_mixture_sampler_would_have_failed_this_same_check`：用 20 000 抽复现 ε=0.10 混合，实测偏离 π 达 8 SE，证明上面那条不是空判 |
| 联合动作 log_prob 等于自由动作项之和 | `test_the_joint_log_prob_is_the_sum_of_the_free_terms` |
| 权重未更新时重算 ratio 为 1 | `test_ratio_is_exactly_one_when_nothing_has_moved` |
| 人工改变概率后 ratio 符合已知解析值 | `test_the_ratio_follows_the_analytic_probability_after_a_change`（0.5→0.25：send 比 0.5、hold 比 1.5） |
| 被强制候选不给 actor 制造虚假梯度 | `test_a_bound_taken_over_candidate_carries_no_likelihood_and_no_entropy`、`test_a_forced_row_contributes_nothing_to_the_actor_but_the_value_head_still_learns`（对 logits 求梯度：强制行恰好为 0，critic 同 tick 仍有梯度） |
| 全部强制时 actor 无伪造动作更新 | `test_an_all_forced_step_has_zero_log_prob_and_zero_entropy` |
| checkpoint 保存并恢复 RNG、optimizer、数据 cursor | `test_the_three_random_streams_survive_a_round_trip`、`test_a_saved_run_carries_the_optimizer_moments_and_the_scenario_cursor` |
| 记录可行动作集合 | `test_the_actionable_set_is_recorded_with_the_decision` |

外加本轮 smoke（`TIGHT_SMOKE.json`，12 项全过、`failures: []`）里三条端到端检查：
真实 rollout 的逐带重算 gap 为 0；把一条带的动作翻转后训练**必须拒绝**（实测报
`max_abs_gap 0.531`，训练器抛错）；整步全强制时 actor 不动。续训等价性也在 smoke 里：
"连跑两轮" 与 "跑一轮、存盘、重载、再跑一轮" 的末次更新报告
（`at_steps/samples/optimizer_steps/approx_kl/clip_fraction/batch_on_time/batch_bytes/drawn_send_rate`）、
cursor 与末权指纹逐项相同。

验证：`python -m unittest discover -s integrated_sar/tests` → `Ran 610 tests OK (skipped=2)`
（此前 598 + 本轮 12）；`python -m integrated_sar` → `final_status=completed`。

## 六、这使哪些旧结论需要限定

| 旧表述 | 现在的状态 |
|---|---|
| 第三轮"三 seed 续训到 400k 后与 `hold_to_bound` 逐值相同、自主字节 0" | 事实保留（那是固定 checkpoint 的回放），但它是在错误行为概率下训练出的策略的性质，不能当成"学到等待"的机制证据 |
| "该场景与观测定义下规则已足够" | 按总纲 §5.5 降级为"当前这套训练实现得到的策略未优于强规则"。观察到平台既不证明优化充分，也不证明规则最优 |
| 首轮 pilot 的降本数字 | 仍只作历史对照；归因判定（收益属于共享静默上界）不依赖采样器，继续成立 |

## 七、复跑

```bash
# 审计（旧/新采集器 × 未训练初始化 / 400k checkpoint）
python runs/comm_tight_budget_20260919/probability_audit.py \
  --episodes 6 --threads 1 \
  --controller-path runs/comm_stability_20260919/train/seed1/stability_seed1_at400000.pt

# 正确性测试
python -m unittest integrated_sar.tests.test_comm_ppo_probability -v

# 本轮 smoke（12 项，含训练器拒绝错配带、续训等价）
python runs/comm_tight_budget_20260919/tight_preflight.py --threads 1
```
