# 下一轮：任务压力下的无模型通信 learnability calibration

日期：2026-09-22。对应总纲 v3.3。状态：**已执行完毕（2026-09-22 跑数据，2026-09-23 完成报告与推送）**；本文件不授权训练。

> **执行结果（详见 `runs/comm_task_pressure_learnability_20260922/COMM_TASK_PRESSURE_LEARNABILITY_RESULTS.md`）**：
> 按 6→8→10 逐层执行，每层 16 条全新流 × 9 臂 = 144 格，三层 432 格全部完整、0 训练。
> L0/L1 三层全过；**L2 只在 10 目标过带**（`send_once` 8.750 个/episode，on-time/detected = 0.900，
> 相对 `always_hold` +8.750 CI [+7.8125, +9.500]），6/8 目标因完成比 0.9271/0.9375 高于 0.90 天花板不过；
> **L3 三层都是 0/16 流**——最佳固定规则从未出现"同一步只能支付严格候选子集"；机制是每包 2 190–2 952 B、
> 极差只占额度 2.3 %–2.6 %，额度总共约 11 个包，所以失效形式是同一时刻成片拒付（每层 82–96 个臂×流）而非逐步选择。
> L4 的 envelope 增益为 +0.3125 / +0.000 / +0.250，CI 下界三层都是 0，且 8/10 目标的均值 Pareto 只剩 1 点。
> **结论：没有任何压力层通过 L0–L4 ⇒ 按 §9 停止；未计算 λ、未改 reward、未训练、未打开 `800000..800399`。**
> 12 目标层未运行：本轮自身缺陷（探针把无限额度转整数、录制登记展开 `GridPosition`）吃掉 20 次启动，
> 按"失败计入上限"后剩 144 次而该层需 160 次；继续须由使用者明确追加授权。
>
> **2026-09-23 后续（本文不改写，只追加指引）**：使用者按上面预留的口径追加授权 16 次启动（上限 640→656），
> 把预登记的第四层 `num_goals = 12` 补跑完。结果与判据见
> [runs/comm_task_pressure_learnability_20260922/COMM_TASK_PRESSURE_G12_EXTENSION_RESULTS.md](runs/comm_task_pressure_learnability_20260922/COMM_TASK_PRESSURE_G12_EXTENSION_RESULTS.md)
> 与扩展协议：g12 为 L0/L1/L2 过、**L3 3/16 流（门槛 8/16）不过**、L4 不过（envelope +0.0625，CI [0, 0.1875]），
> 于是这条"只加目标数"的轴四层全未通过，按 §9 正式停止场景扩展。原三层的判据与结果文件字节未变。

## 1. 起点与研究决策

从 `exp/comm-budget-masked-retrain@7e0a4233127a19fa9bf49672bbcf37218b587fff` 创建
`analysis/comm-task-pressure-learnability`。开始前核对远端 SHA，阅读 AGENTS、ARCHITECTURE、
PROJECT_STATUS、PROJECT_CHARTER v3.3，以及
`runs/comm_budget_masked_retrain_20260922/COMM_BUDGET_MASKED_RETRAIN_RESULTS.md/.json`。

修复动作空间后的 3 seed × 400k 已经排除“旧 mask 错误”和“绝对 actor 样本不足”两种解释；三臂均从
初值改善，但对 `send_once` 的主效果与非劣降本门槛均为 0/3。当前六目标开发设置中，`send_once`
平均完成 5.7917/6，最多只剩约 0.2083 个任务/episode 的效果空间。

本轮不再训练，也不追加步数。唯一问题是：

> 当只提高任务数量、保持预算和其余系统不变时，是否出现固定规则无法统一覆盖、但 sender-visible
> 任务压力能产生不同发送选择价值的场景；并且是否存在至少两个真实的成本—效果折中点？

这是一项**必要条件标定**，不是学习模型结果。通过只允许设计下一轮目标/奖励，不自动训练。

## 2. 为什么选择任务压力，而不是先加非平稳丢包

无 ACK 的 `/1` 发送端看不到包是否丢失；直接改变丢包率会改变结果，却不给当前模型可用的状态变量，
因此不能把“在不可观测信道上适应失败”解释为模型问题。任务数量则会改变候选竞争、截止时间压力与预算争用，
而 `/1` 已合法观察任务剩余时间、阶段、进度、需求变化、等待时间、剩余额度与 backlog。

同时，当前训练奖励只在任务按时完成时给分；通信字节是硬上限与事后指标，**不是 reward 项**。因此旧模型把
额度用到 96% 并不违反其优化目标。本轮先测出规则层面的成本—效果 Pareto 和经验斜率；只有真实前沿存在，
后续才讨论显式成本项或约束式目标。不得在本轮暗加字节惩罚。

## 3. 唯一自变量与冻结项

唯一自变量：`num_goals ∈ {6, 8, 10, 12}`。

冻结不变：

- PILOC 与任务分配 checkpoint、搜索机器人/救援机器人数量、地图、障碍分布、1,000 步 horizon；
- `policy_owned_timing`、29,300 B/episode、isolated 信息路径、投递概率 1.0、ACK 关闭；
- 成员一致性、binding、环境修复和预算顺序 mask；
- task deadline/work-duration 生成、CTDE/协调配置、消息格式与计费；
- 不修改 19 维观测、网络、奖励、生产默认值或任何规则参数。

跨压力层使用不同且不重叠的 episode seed，因此压力层之间只作趋势描述；每一层内部所有规则严格共享同一组流，
规则比较按 stream seed 配对。不得把不同目标数的非配对差写成因果效应。

## 4. 新的探索性数据块

在生成前对 Git、全部 manifest、盘上 stream/run artefact 和文档声明做四源冲突审计。候选块固定为：

| num_goals | episode seeds | 数量 |
|---:|---|---:|
| 6 | `1000000..1000015` | 16 |
| 8 | `1000100..1000115` | 16 |
| 10 | `1000200..1000215` | 16 |
| 12 | `1000300..1000315` | 16 |

任一 seed 已出现或存在多内容风险：**停止，先提交协议修订，不自动换号**。冲突审计与最终 seed/stream SHA256
manifest 必须先于矩阵结果存在。最多64条流是新的开发/标定数据，不是最终测试；`800000..800399` 继续不生成、
不读取，`900000..900199` 和 `700000..700071` 不进入本轮矩阵。

每个 stream 只录制一次搜索事件，然后由所有规则重放。录制失败、目标放置不足或未检测目标均如实保留，
技术失败计入上限，不补 seed。报告 placed/detected/registered targets，不能只用配置里的 num_goals 作分母。

## 5. 冻结规则库（先于任何新 stream 提交）

所有预算内规则都使用同一 29,300 B 执行器，不重新调参数：

1. `send_once`；
2. `event`；
3. `event_priority`；
4. `periodic_5`；
5. `periodic_20`；
6. `first_send_plus_bound_3_10`；
7. `bound_only_3_10`；
8. `always_hold`（通信地板，不参加“最强规则”选择）；
9. `event_unbudgeted`（无预算参考，不参加预算内竞争，也不得称严格 oracle）。

`random_send` 不进入主库，避免随机规则扩大逐流 hindsight envelope。不得看到结果后新增间隔、优先级权重或规则。

每层 16 流 × 9 臂，另加该层16次录制。按目标数逐层执行；一层通过 L0–L4 后立即停止，不生成更高层。
四层全部执行时总启动上限 **640**。所有失败都计入，达到上限即停止，不补跑。

## 6. 必须测量的量

### 搜索与任务压力

- placed / detected / registered / on-time / expired；
- detection time、assignment latency、deadline slack、并发开放任务数；
- 机器人利用率和因执行容量造成的不可完成任务，避免把分配瓶颈冒充通信瓶颈。

### 通信与预算竞争

- attempts、bytes、emissions、applied、duplicate/superseded、belief age；
- budget used fraction、first exhaustion step、budget/capacity refusal；
- first/update/retry 类型与发送时间；
- sender-visible 候选数、双动作可行点、在同一步/同一剩余额度下只能支付严格子集的
  `budget_competition_points`；
- 至少一个 budget competition point 的 stream 数。

### 规则比较

每个目标数分别计算：

- 每条规则的 on-time/episode、逐局率、bytes/episode 及配对 bootstrap CI；
- 样本均值 Pareto、逐流支配、带统计支持的比较，三者分开；
- 预算内**最佳固定规则**：均值按时完成最高，并列取字节更少，再并列按预登记规则顺序；
- **逐流 hindsight rule envelope**：每条流在七条预算内竞争规则里取按时完成最高、并列字节更少的结果；
  它只是不具部署性的经验上界，不是策略 oracle；
- envelope − 最佳固定规则的逐流配对差；
- 每条规则成为逐流唯一赢家/并列赢家的次数，以及至少一对规则是否出现双向胜负。

## 7. 压力层选择门禁

按 `6 → 8 → 10 → 12` 顺序逐层生成/运行/判定，选择**第一个**同时通过 L0–L4 的目标数；
通过后不得继续更高层，也不按差值最大者回头挑层。L2–L4 未过才进入下一层；L0/L1 未过按下文停止。

### L0 数据与接口可信

16/16 流有完整 final record，stream 摘要一致，九臂配置除 policy/预算参考外一致；无非法动作、账本残差、
混合 commit 或技术失败。预算臂从未超额。否则该层失败且整轮停止，不向更高压力偷偷前进。

### L1 不是搜索缺失造成的假压力

该层 placed targets 全部合法，且平均 detected/placed ≥ 0.75；同时报告最差流。未过则更高目标数不再运行，
因为压力已主要落在搜索覆盖而非通信调度。

### L2 通信确实有价值且未处于天花板

最佳预算内固定规则相对 `always_hold` 的按时完成差 CI 下界 > 0；并且最佳固定规则的
on-time/detected 平均比例位于 **[0.40, 0.90]**。低于0.40视为系统容量/搜索地板，高于0.90视为效果天花板。

### L3 预算产生真实选择

至少 8/16 流存在 `budget_competition_points > 0`，且最佳固定规则中至少 4/16 流预算使用率 ≥0.90。
只看到大量候选或大量拒付而没有“可支付严格子集”不算通过。

### L4 固定规则之间存在可利用的情境差

同时满足：

1. hindsight rule envelope − 最佳固定规则的按时完成差 ≥ **0.25 个/episode**，且95% CI下界 >0；
2. 至少两条预算内规则各自在 ≥3/16 流成为唯一赢家；
3. 至少一对非支配规则出现双向胜负（A在≥3流完成更多，B也在≥3流完成更多）；
4. 样本均值 Pareto 至少有两个点，二者完成差 ≥0.25个/episode、字节差 ≥2,930 B（预算10%）。

L4 证明“固定规则不能统一覆盖所测情境”的必要信号，但不证明神经策略一定能学到 envelope。

## 8. 奖励/目标校准输出（仍不训练）

若某层通过 L0–L4，利用该层的预算内 Pareto 计算相邻点的经验交换率：

`lambda_break_even = Δ(on_time targets) / Δ(bytes / 29,300)`。

只报告斜率和支持区间，不在本轮选择“最有利于学习模型”的 λ，也不修改 reward。下一轮协议只能从这些经验斜率
预登记最多三个 λ（含 λ=0 的纯完成对照），或选择约束式优化；不得凭主观系数直接训练。

还要核对通过层的 sender-visible 特征（deadline slack、backlog、stage/progress change、remaining budget）在不同
逐流赢家之间是否有支持差异。只作描述和可观测性审计，不拟合分类器、不看接收端 belief/真值作为策略输入。

## 9. 判定与停止

- **无压力层通过 L0–L4**：停止该学习路线的现有场景扩展；结论是“仅提高目标数未建立规则之外的可利用
  通信调度空间”，不训练、不改 reward、不试更多 goal 数。
- **某层通过**：交付该层、规则前沿、经验 envelope、预算竞争和 λ 斜率；只允许设计下一轮成本感知目标及
  训练协议，仍不自动训练、不开放封存块。
- 如果 unbudgeted 参考也很差，优先归因为搜索/分配/执行容量，不把它包装成通信学习机会。
- 不运行 H=20，不加 ACK/丢包，不使用旧学习 checkpoint，不修改生产默认值。

## 10. 交付与提交纪律

建议目录 `runs/comm_task_pressure_learnability_20260922/`：

- `COMM_TASK_PRESSURE_LEARNABILITY_PROTOCOL.md`（先于新 stream）；
- `SEED_COLLISION_AUDIT.json`、`STREAM_MANIFEST.json`（只登记实际生成的层，最多64流）；
- `RULE_MATRIX.json`、`PRESSURE_METRICS.json`；
- `COMM_TASK_PRESSURE_LEARNABILITY_RESULTS.md/.json`；
- `COMM_TASK_PRESSURE_LEARNABILITY_MANIFEST.json`；
- 只含实验 harness/分析/测试的脚本。

提交顺序：①冲突审计与协议；② harness/合成测试；③逐层 stream manifest；④矩阵与结果；⑤总纲/状态。
不提交 stream、逐 episode 日志、checkpoint 或用户未跟踪文件。执行中需要修改 harness 时，停止、隔离旧产物、
形成新 code-freeze；不得混合聚合。完整测试与 `python -m integrated_sar` 通过，推送并核对远端 SHA 后停止。
