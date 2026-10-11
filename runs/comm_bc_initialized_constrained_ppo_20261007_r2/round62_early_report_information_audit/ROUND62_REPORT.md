# ROUND62：early 首报局部作用与合法决策信息审计

## 核验范围

从 ROUND61 交付头 `e14a7fe7ad24b888e81fcf4485f576b9268661c18` 建立本地分析分支。核对模型文件 SHA256 `573e4c26a4fee6f01405d4386ca73b518eeb1adb9cd8409bc83661e4113f6c5a`、参数 SHA256 `bce71a0e9810122c146950f12cf818a9a359406312d96dfc9c7a457ebb654e2a` 与冻结案例一致。模型为 R60 合规 seed33 ep128 greedy；本轮没有加载模型、执行 forward 或构造仿真，只解析既有结果、tape、生命周期、transcript、账本与源码。

L/L0 两臂和 R60 scene1700027 原结果的共同科学字段相同：5 个按时目标、4812 B、两条相同发射记录。R61 机器 gate 为 passed。8 对 FORCE 结果均从动作前相同的 tape 前缀到达同一目标、同一 tick、同一特征和同一 quote；SEND/SKIP 覆盖记录分别为 1/0。8/8 机会重现检查通过。

## 独立配对重算

任务 ID 集合由原 lifecycle 的 `final/completed` 行重算；成本取 lifecycle 最终通信 budget；发射序列从原 transcript 的 send 行重算。结果如下，费用差为 SEND−SKIP：

| 场景 / 目标 | 按时目标 SEND vs SKIP | 系统任务差 | gained / lost under SEND | 成本差 B |
|---|---:|---:|---|---:|
| 1700027 / target-5 | 1 vs 0 | +1 | gained target-5 / none | +2688 |
| 1700015 / target-0 | 1 vs 1 | 0 | none / none | +2496 |
| 1700012 / target-4 | 1 vs 1 | 0 | none / none | +2496 |
| 1700000 / target-2 | 1 vs 1 | 0 | none / none | +2688 |
| 1700009 / target-0 | 1 vs 1 | 0 | none / none | +2496 |
| 1700016 / target-0 | 1 vs 1 | 0 | none / none | +2682 |
| 1700018 / target-3 | 1 vs 1 | 0 | none / none | +378 |
| 1700005 / target-3 | 1 vs 1 | 0 | none / none | +2484 |

焦点 FORCE_SEND 在 step 134 发射 target-5，报价/实际计费 2688 B，6 个接收者尝试中 5 包投递、1 包丢失，5 个接收端记录 accepted；step 134 出现分配记录，target-5 于 step 156 完成。FORCE_SKIP 没有该发射，目标在 step 232 因 deadline_expired 失败。焦点两臂最终 ID 集合只差 target-5，没有目标交换。这个单场景随机实现中，关键 early 首报确实改变了任务结局。

7 个对照的目标和全系统最终集合均相同，6 个目标全部按时；SEND 侧有实际发送/投递，分配链也可能改变，但未改变最终目标集合。成本差是闭环全程成本，不等同于首报 quote；例如场景 1700018 的差为 378 B。accepted 不能当作收益，结果相同也不能定义为普遍安全跳过。

## 决策信息

动作前 A 层 19 维完整向量均保留在 [DECISION_INFORMATION_PANEL.json](DECISION_INFORMATION_PANEL.json)。8 个 float32 向量都不同，跨案例只有 index 14 `queue_backlog_targets` 和 index 18 `step_over_horizon` 有变化；index 13 `has_future_requirements` 八例均为 1。没有观察到“相同 19 维向量对应不同本次干预任务差”的碰撞。

B 层来自 `SendItem`/发送端时钟与预算。源码确认合法发送端持有 deadline、remaining_time、age、报价、可发送状态等。ROUND61 对 SKIP 行没有直接序列化原始 `SendItem`；报告中的 deadline 由同一目标 detection 记录的 deadline_offset 加检测时刻重建，remaining_time 再按 `current_time = decision_step - 1` 推得。此时钟关系由 R30 对 11,901 个候选行独立复核。8 个案例的 remaining_time 为 67.93–158.50 步，但 actor index 11 都是 `clip(remaining_time/50,-1,1)=1`，index 12 的 25 步 urgency 位均为 0。原始期限差异因此在旧 19 维期限编码中折叠。

对 R60 seed33 greedy 已保存的 32 场景作只读扩展，重算得到 146 个合法可付 early 首报候选，ready_step 为 1–134，baseline 146/146 都是 SKIP；index 11 为 1 的比例是 146/146。index 14 backlog 范围 1–6，index 18 时间列范围 0.001–0.134（30 个不同值）。这些场景没有 SEND/SKIP 干预标签，不能当成新增因果样本，也不能据此认定安全删除。

C 层为接收记录、assignment/progress/final、分配器及任务结局，仅用于事后解释；它不进入 A，也不用于案例排序或动作覆盖。完整逐臂发送、投递和分配生命周期已按案例保存在信息面板中。

## 历史期限编码结果

本轮查阅了仓库中的 R29/R30 原件。R29 已证明 102 个已有首报点的合法剩余期限在 `clip(remaining/50)` 列中全部饱和，并实现/验证了 `remaining/1000` 纯编码原型。R30 随后完成了从初值的单列编码训练消融：greedy 三 seed 的 48/48 场景任务数持平，成本均值分别为 -9、0、-334.875 B/episode、方向不一致；sampled 只有 seed22 与 seed33 各 1/16 场景出现任务增益。因此本轮没有把“再做一次相同单列编码训练”列为建议。出处：[ROUND29_REPORT.md](../../comm_deadline_encoding_audit_20260929/ROUND29_REPORT.md)、[ROUND30_REPORT.md](../../comm_deadline_encoding_ablation_20260930/ROUND30_REPORT.md) 与对应机器 verdict。

唯一建议的下一研究变量是 **动作时原始 remaining_time，在匹配 ready_step（现有 index 18）和 backlog（index 14）后，作为 SEND/SKIP 局部效应修饰量**。它可区分“首报较晚才出现导致救援窗口短”与“相同到达时机下 deadline slack 本身解释结局差”的解释；场景任务难度、目标位置和随机投递仍会混杂。本建议只说明下一项可检验变量，不构成新环境或训练授权。

## 消费更正与限制

ROUND61 账本有 19 个 slot：18 normal、1 reserve；18 项合格结果，slot 2 的 L0 首跑失败，slot 3 reserve 重试成功。builder 调用 19。`env_constructed` 事件只写入 18 次，但 slot 2 留存的 runner `run_config` 与 step-1 assignment、step-5 detection 证明该次也已构造环境并进入 runner，所以实际环境构造为 19。runner 返回 18 次，成功返回步数合计 18,000。

失败首跑的 ledger 没有 `run_called`：执行器只在 `_run_episode` 返回后写此事件。孤立 lifecycle 有 8 条记录，最晚时间戳 5，无最终 run 记录；由此可证明失败调用至少到达 step 5，但不能恢复精确执行步数。实际尝试步数应写为 **至少 18,005；精确总数未知**。每次配置 horizon 为 1000，按该上限计算的总数上界为 19,000。旧 claim 路径 `attempt_0001.json` 被同 logical 的 reserve 重试复用，slot 2 独立 claim 文件未留存；slot ledger 与部分 lifecycle 保留，未补造 claim。R61 报告所述 `shadow_gate` 命名故障没有找到原 traceback，故原因未能由 traceback 独立确认。

R61 实际启动/消费为 19/20，仍在本轮 R61 新授权上限内，余 1；18 个有效结果不等于 18 次实际消费。R60 的 608 次构造超过其 600 上限 8 次，是独立历史偏差，未改变。

本轮副作用：环境构造 0、仿真 0、新 claim 0、新 episode 0、真实/合成 optimizer 更新 0、PPO 重放 0、训练/RNG 恢复 0、checkpoint 写入 0、新评估 0、CPU forward 0。纯计算窄测试 5 项通过，exit code 0。两份受保护 ROUND46 文件 hash 与冻结值相同。

结论仅描述 8 个事后案例、一个冻结模型、每场景一次随机实现中的 SEND/SKIP 差异。改变消息数量后不声称后续逐包丢失经历相同；这些结果不证明消息普遍必要、安全跳过、总体非劣、泛化或部署价值。
