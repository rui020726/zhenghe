# 下一轮执行指令：通信学习结论冻结与强规则独立工程确认

> 历史指令：已执行，交付为 `b5fb34e`。其后的复核轮（`COMM_RESEARCH_REVIEW_NEXT_ROUND.md`）也已执行完毕，
> 交付在 `runs/comm_evidence_review_20260921/`；当前入口是总纲 v2.2 §1。
> 本文中的 no-ACK/ACK 学习范围、成本比较对象与单步门禁解释，须按总纲 v2.1 复核更正；勿再次执行本矩阵。

日期：2026-09-21。对应总纲 v1.9，S3 收尾与工程确认。状态：执行方案，尚未运行。
> **状态：已执行完毕（历史指令）。** 其结论的引用口径已由 `COMM_RESEARCH_REVIEW_NEXT_ROUND.md` 那一轮复核并统一，见 `runs/comm_evidence_review_20260921/COMM_EVIDENCE_CORRECTIONS.md`；本文与本轮产物均保留原样。

## 1. 起点与本轮定位

从 `feat/comm-ack-feedback-gate@fbf96d8aa30fb2600fd710eab79a7b5f8121422a`
创建 `analysis/comm-rule-independent-confirmation`，只引入本次文档提交。先读 AGENTS、ARCHITECTURE、
PROJECT_STATUS、PROJECT_CHARTER，以及 `runs/comm_protocol_v2_20260920/` 和
`runs/comm_ack_value_20260920/` 的协议、结果、序列、门禁和 manifest。

本轮不再训练通信模型，目标分成两件互不混淆的事：

1. **冻结研究结论**：在已经测试的 no-ACK、policy-owned timing 和 costed ACK 协议中，学习策略
   没有建立强规则之外的价值；该结论限定于现有任务分布、观测、动作、奖励和训练资源。
2. **回答工程问题**：`retry_until_acked_2` 在有损训练流上以 +88.5% total bytes 换来
   +0.875 个按时任务/episode。这条成本—效果关系能否在全新数据和正常信息路径上复现？

ACK 门禁 3 未过不是“再训练一次”的许可：425 个反事实点中 96.5% 的 SEND/HOLD 对任务结局打平，
少数有差异的点由简单规则获得。不得追加训练、改奖励、加 RNN、调高丢包、收紧 deadline 或换场景
来追逐正结果。最终学习测试块 `800000..800399` 继续封存。

## 2. 先冻结协议和新数据，不看结果后改选择

创建 `COMM_RULE_CONFIRMATION_PROTOCOL.md`，在生成新流之前写定本文件的全部配置、臂、统计与停止条件。

候选独立工程块：`900000..900199`，共 200 个 seed。先把该块与全项目 manifests、streams 和所有
历史分析登记求交；只有交集为 0 才登记并生成。若有任一重叠，停止并报告，不自行换另一个数字段。
该块只用于规则工程确认，不能在未来称为学习模型的未见测试集；`800000..800399` 的用途不变。

在协议提交之后一次性生成/录制 200 条流，写生成配置、代码 blob、逐流 SHA256 和折叠摘要。
任何技术失败只能按预登记原因原 seed 重跑，不剔除困难 seed，不补抽“好看”的样本。

## 3. 冻结配置与比较臂

固定不变：PILOC、CTDE、修正环境、C 臂、任务生成、deadline、29,300 B 任务上行硬额度、
`policy_owned_timing`、ACK 合同、候选顺序、消息合并与执行安全边界。生产默认不变。

三个设置：

1. **isolated-lossless**：隔离视距/gossip，任务上行 delivery probability 1.0；ACK 臂反向信道同参数。
2. **isolated-lossy**：隔离视距/gossip，任务上行 delivery probability 0.7；ACK 臂反向信道同参数。
3. **normal-lossy**：开启项目原有视距与 gossip，任务上行及 ACK 反向信道 delivery probability 0.7。

延迟、subscriber 拓扑、episode 长度和 counter-based 随机键必须从第六轮的真实配置逐字段继承，
在协议中展开，不能只写“同上”。no-ACK 与 ACK 臂的前向丢包使用同一套预登记 keyed draws；
ACK 反向随机过程独立但可复现。无法一一对应的事件明确标为同分布比较。

五条臂在三个设置上全部运行，不根据中途结果删臂：

| 标签 | ACK | 发送规则 | 作用 |
|---|---|---|---|
| `noack_send_once` | 关 | `send_once` | 最省的 no-ACK 工程基线 |
| `ack_send_once` | 开 | `send_once` | 单独量化 ACK 协议开销 |
| `ack_retry_2` | 开 | `retry_until_acked_2` | 训练集完成数最强 ACK 规则 |
| `ack_periodic_20` | 开 | 已冻结的 `periodic_20` | ACK 规则前沿参照 |
| `ack_backoff_16` | 开 | 已冻结的 `backoff_16` | 退避型前沿参照/无损负控强臂 |

规则参数来自已完成的 48 条训练流，不在新 200 seed 上重新选择。旧学习 checkpoint 不参加本轮，
因为它们没有通过进入独立确认的门槛；把最好 seed 混入会造成选择偏差。

总规模为 200 seeds × 3 设置 × 5 臂 = 3,000 episode。先用 5 个非新块 seed 测吞吐、磁盘和单集
产物，再决定分片数；不得沿用旧耗时估计。正式任务独立 scratch/输出，分片只在一处应用。

## 4. 成本、指标与统计口径

主任务指标：按时完成任务数，单位 个/episode。同时报告逐局率、汇总率、expired、未发现目标和
逐目标新增失败。条件完成率与全部目标完成率分开。

通信指标必须分列：

- task uplink attempts/bytes；
- ACK reverse attempts/bytes；
- total attempts/bytes = uplink + ACK；
- delivered、applied、lost、duplicate、coalesced、refused；
- 信息年龄、发现到首次 belief/分配/工作/完成的延迟。

任何“降本”只看 total bytes。ACK 臂上行更少但反向字节更多，不得称省通信。

同 seed 配对百分位 bootstrap，B=10,000，统计 RNG 在协议中写死。分别比较：

1. `ack_retry_2 - noack_send_once`：协议整体的成本—效果差；
2. `ack_retry_2 - ack_send_once`：ACK 信息之上的重发规则收益；
3. `ack_periodic_20`、`ack_backoff_16` 与 `ack_retry_2`：规则前沿是否稳定。

每个比较给完成数差和 total-byte 差的均值、95% CI、胜/平/负；三个设置不合并样本。
计算“每增加一个按时任务所增加的字节”时，同时报告原始分子和分母；bootstrap 分母 ≤0 的样本
标记 undefined，不删除后再给出虚假稳定比值。

本轮不预设“一个任务值多少字节”的效用权重，因此不自动宣布 ACK 值得部署。结果应给出 break-even：
只有当使用者对每增加一个按时任务的价值高于实测增量字节时，ACK 重试才在该效用定义下合算。

## 5. 预登记的判读规则

### 无损负控

`ack_retry_2`、`ack_periodic_20` 或 `ack_backoff_16` 不应在 isolated-lossless 上稳定提高完成数；
若任一相对 `noack_send_once` 的完成数 CI 下界 > 0，先暂停并检查 ACK/规则是否改变了除反馈之外的行为。
ACK 臂 total bytes 增加是预期成本，不是通过。

### 有损隔离确认

若 `ack_retry_2 - noack_send_once` 的完成数 CI 下界 > 0，则第六轮“ACK 重试以更多成本换完成数”的
工程效应得到独立复现；同时必须报告 total bytes CI，不能用任务收益遮蔽成本。
若 CI 含 0，训练集上的 +0.875 不获得独立支持，停止部署讨论。

### 正常信息路径

只有 normal-lossy 上完成数 CI 下界 > 0，且没有执行正确性退化，才把 `ack_retry_2` 提议为
**可选高可靠模式候选**。即使满足，也不自动改默认值；是否接受字节成本由使用者决定。

若 isolated-lossy 有收益而 normal-lossy 没有，结论限定为“隔离上行实验有效，真实系统旁路使收益
不可分辨”，不部署 ACK。若正常路径任务结局相同但 total bytes 更高，明确判为工程上被支配。

### 规则前沿

若其他 ACK 规则在新数据上超过 `ack_retry_2`，不回头调参或重命名主规则；如实报告规则选择不稳定。
本轮只确认已锁定集合，不从 200 seeds 重新挑参数。

## 6. S3 研究结论冻结

生成 `COMMUNICATION_LEARNING_VERDICT.md` 和机器可读 JSON，按轮次列出：

- 首轮 pilot 与账本纠错；
- 强制上界归因与训练稳定性；
- PPO 行为概率修复与双预算 0/3；
- policy-owned timing 的主门槛 0/3、降本门槛 0/3；
- ACK 信息价值门禁 3 未过、未训练；
- 当前工程规则确认结果。

结论固定为：

> 在已经测试的任务分布、两类发送协议、no-ACK/ACK 反馈、观测、动作、奖励和有界训练资源下，
> 学习型发送控制器没有建立相对强规则的附加价值；简单规则形成了当前可复现的成本—效果前沿。

不得写成“强化学习永远无效”“规则全局最优”或“没有任何可学习空间”。同时明确：当前没有通过门槛的
学习候选，因此没有打开 `800000..800399`，也没有独立学习性能主张。

本轮完成后把 S3 标记为**范围准确的负结果，研究先导关闭**。如果用户未来仍要求学习贡献，必须另立
新假设和新阶段，例如具有现实依据的非平稳信道或任务压力；先做无模型 learnability calibration，
不能把本轮当作继续改场景直到模型获胜的授权。

## 7. 实现、验证与提交纪律

预计不需要修改生产通信代码，只新增冻结协议、流登记、评价/汇总脚本和结论文档。若发现会影响结果的
正确性缺陷，先最小复现、单独提交并使受影响产物失效；不得顺手重构任务分配或改默认配置。

聚合前逐文件校验：arm、ACK 开关、信息路径、delivery probability、预算、代码 blob、stream hash、
final record 和 metric schema 全一致。任何不一致 fail closed；隔离而不删除，再按原 seed 重跑。

交付至少包括：

- `COMM_RULE_CONFIRMATION_PROTOCOL.md`；
- `COMM_RULE_CONFIRMATION_RESULTS.md/.json`；
- `COMMUNICATION_LEARNING_VERDICT.md/.json`；
- `COMM_RULE_CONFIRMATION_MANIFEST.json`；
- 新 200 流的只读登记与 SHA 折叠；
- 必要的评价测试和复跑命令。

更新 PROJECT_CHARTER、PROJECT_STATUS 和 README 轮次索引。若修改任何生产代码，执行全量单元测试与
`python -m integrated_sar`；若仅新增分析脚本和文档，也至少运行脚本单测、协议 preflight、
`git diff --check`，并在提交中说明为什么没有重复生产测试。

checkpoint、原始流、逐 episode 日志不入 Git。显式 `git add`，不用 `-A`。非本轮创建的未跟踪
`architecture/` 和 `dev_summary.json` 不修改、不删除、不提交。推送后核对远端 SHA。

最终汇报只回答五个问题：

1. 第六轮 ACK 规则的任务收益在全新数据上是否复现？
2. 每多完成一个按时任务，增加多少上行、ACK 和总字节？
3. 正常信息路径下收益是否仍存在？
4. 哪些规则位于独立数据的 Pareto 前沿？
5. 当前 S3 学习结论和工程建议分别是什么？
