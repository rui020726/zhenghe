# 预算 mask 修复后的从初值重训：本轮协议（训练前冻结）

日期：2026-09-22。分支 `exp/comm-budget-masked-retrain`，起点 `fix/comm-budget-action-mask@e692607`。
对应总纲 v3.1 与执行指令 `COMM_BUDGET_MASKED_RETRAIN_NEXT_ROUND.md`。
**本文件在第一个训练 episode 之前单独提交（§6.7）；提交之后不得按结果修改判据。**

## 0. 唯一研究问题

在「预算不可执行的 SEND 已于采样前按真实执行顺序硬 mask、行为概率与 PPO 更新一致」的动作空间里，
固定主预算与原观测/奖励，**三个独立初始化的学习型发送策略能否在开发流上稳定超过锁定的强规则
`send_once`，或在任务效果非劣时降低字节？**

旧 400k checkpoint 是缺陷动作空间的产物：不加载权重、不恢复 optimizer、不作 warm start。

## 1. 冻结设置（任何一项不符即停，不进入训练）

| 项目 | 值 | 来源 |
|---|---|---|
| 上行预算 | **29,300 B/episode** | `stability.locked_budget("25pct")`，由 `retrain_common.setting()` 现场核对 |
| timing | `policy_owned_timing` | `protocol_common.POLICY_MODE` |
| 信息路径 | isolated（关现场刷新与 gossip） | `pilot_env.ISOLATED_GOSSIP_INTERVAL` |
| 信道 | 无损，投递概率 1.0 | 第五轮定义 |
| ACK | 关闭 | 生产默认，未改 |
| episode | 1,000 环境步 | `pilot_common.STREAM_STEPS`，逐 episode 实测核对 |
| 观测 | `comm-send-state/1`，19 维 | `comm_learning.FEATURE_NAMES` 长度现场断言 |
| 网络/奖励/PPO | 64-64 trunk、γ 0.99、λ 0.95、clip 0.2、v 0.5、熵 0.01、lr 3e-4、grad 0.5、4 epoch、minibatch 256 | 全部从 `train_pilot` 读取，不重述数值 |
| trainer | `onpolicy-v3-budget-mask` | `budget_masked.TRAINER_TAG` |
| mask / tape | `budget-sequential-mask/1` / `onpolicy-tape/2` | 同上 |
| 采样器 | `train_budget_masked.collector_for`（`mask=False, forced=nothing_forced, budget_mask=True`） | 上一轮通过 M0–M3 的实现，逐文件冻结 |
| PPO 循环 | `train_onpolicy.update` / `flatten` / `drawn_versus_logged` | 原样复用，不复制第二份 |

不同时跑 87,900 B，不加 ACK/重试，不改 deadline、任务分布、搜索/分配模型与生产默认值。

## 2. 数据（训练前重新核对内容摘要）

| 用途 | 流块 | 数量 | 状态 |
|---|---|---:|---|
| 训练 | 700000..700047 | 48 | 既有流循环重放，不生成新流 |
| 开发评价 | 700048..700071 | 24 | 已多轮使用，**不是独立测试** |
| 最终封存 | 800000..800399 | — | 盘上 artefact 必须仍为 0，不生成不读取 |
| 历史块 | 900000..900199 | — | 已消费，只作历史证据，本轮不进入 |

`retrain_common.verify_streams()` 逐条比对 72 条流的 sha256 与 round-2 manifest
（`runs/comm_budget_pilot_20260919/COMM_ROUND_MANIFEST.json seeds.stream_sha256`）；
编号相同而摘要不同即停。

## 3. 训练规模与确定性

- 训练 seed `1, 2, 3`，各从 `Controller(seed)` 随机初值开始，0 步保存权重指纹与 `_at0.pt`；
- 每 seed **400,000 环境步 = 400 episode**，合计上限 1,200,000；4 episode/update；单线程；
- checkpoint 0 / 100k / 200k / 300k / 400k，**400k 末轮是唯一主判据**；
- 三 seed 共用同一冻结的训练流访问序列（`TRAIN_SEEDS` 循环、cursor 进 checkpoint），
  模型初始化与策略抽样 RNG 按 training seed 区分；
- 不按中间开发集最好点选模型；不因某 seed 早赢而提前停，也不为没赢的 seed 追加步数。

checkpoint 必须携带并可核对：controller、optimizer、环境步、episode、stream cursor、
Python/NumPy/Torch 三条 RNG、trainer tag、mask/tape schema、feature layout、timing mode、
budget、步数上限、**code-freeze SHA**、配置 hash、初值指纹。缺任一项或值不符 → 拒绝 resume；
缺 RNG 同样拒绝。`--mode resume-check` 用真实训练流跑「连续两轮 vs 存盘恢复再跑一轮」，
要求场景顺序、动作 tape 摘要、loss、权重与 optimizer 状态逐项**完全相等**（不给浮点容差；
两边是同一 CPU 同一顺序，任何漂移都必须在报告里显形）。

## 4. 训练前门禁（§6 全过才允许启动三个 400k）

1. provenance 更正已交付，code-freeze SHA 已提交，执行路径相对 freeze 干净、无未跟踪模块；
2. 上轮 M0–M3 在当前树重跑通过（14 项契约套件 + 完整 integrated_sar 套件 + 上轮 17 项套件 +
   本轮套件 + `python -m integrated_sar`）；
3. `budget_mask=True`，费用/顺序/tape schema 与 `e692607` 语义一致（由 `budget_masked.identity()`
   与 `setting()` 现场断言）；
4. ≤12 episode 探针（每训练 seed 4 集，`optimizer.step=0`）中
   `requested == executed`、policy budget/capacity 拒付 = 0、log-prob 差 ≤ 1e−6、
   每集步数 = 1000、无代发；
5. 三个初值指纹互不相同，且构造路径不读任何 checkpoint（测试用 `torch.load` 探针证明）；
6. deterministic resume 回归通过；封存块盘上 artefact 仍为 0；
7. 本协议与主判据的提交时间早于第一个训练 episode。

额外一条本轮自己加的：**评价通路先跑一次**。0 optimizer.step 的 greedy + mask 评价在
一条**训练流**上执行一次（不花开发流），确认产生头号数字的代码在训练开始前已经跑过。

## 5. 样本口径（只报比例不合格）

`181/20,655 = 0.88 %` 只是上轮固定初值 smoke 的候选占比，不是停止门槛。逐 checkpoint 必须报：

- actionable 绝对总数、每 episode 的均值/中位/p25/p75/min/max；
- 含 ≥1 个 actionable 的 episode 数、零 actionable 的 episode 数、零决策 episode 数；
- 首报 / 更新 / 同视图重发三类消息数（来自 relay 自己的 `first_emissions`、`refresh_emissions`、
  `redundant_emissions`，闭合 `first + refresh == emissions`）与发送步位的 5/25/50/75/95 分位；
- 初始可执行、顺序后 actionable、被额度/容量 mask、请求、实际发射、执行端拒付；
- 每 update 的 actor 项（= actionable 候选数）、有效 minibatch 数、optimizer steps；
- stochastic 与 **greedy 重放**的每候选熵、SEND 率、首次额度 mask 步、零可执行步数
  （greedy 用同一批 tape 与同一 `CommBudget` 走一遍，不额外花 episode）；
- policy/value loss、explained variance、KL、clip fraction、gradient norm、NaN/Inf 计数；
- logged vs recomputed log-prob 最大差（阈值 1e−6，严于 `update` 自带的 1e−4）。

若某 seed 到 20k 步仍 actionable=0、有效 minibatch=0 或权重与初值完全相同，停该 seed 并归因为
训练接缝失败，不加步掩盖。

## 6. 评价计划

| 单元 | 模型 | 流 | episode |
|---|---|---|---:|
| 0 步 | 三 seed × `_at0` | 全部 24 条开发流 | 72 |
| 曲线 | 三 seed × 100k/200k/300k | 固定前 8 条（700048..700055） | 72 |
| 400k | 三 seed × `_at400000` | 全部 24 条开发流 | 72 |
| 规则 | `send_once` | 全部 24 条开发流 | 24 |

全部学习评价用 greedy 策略，且与训练共用同一预算 mask 与 `max_wait_steps=None` 的
policy-owned 参数。中间 checkpoint 只画曲线。

**规则重放一致性**：`send_once` 逐流与归档（round-5 `evaluation/25pct/rows_25pct_isolated_1.0.jsonl`
及其转录）比对任务结果、字节、`config_hash` **和发射序列**（步、目标、revision、cost_bytes、
订阅数）。出现无法解释的差异即停并归因，不换更弱的规则。

## 7. 统计与预登记判据

- 主效果：按时完成任务数，单位 **个/episode**；成本：上行尝试字节 **B/episode**（本轮无损）；
- 24 条流内 seed 级配对 percentile bootstrap，B=10 000，RNG=20260922；
  **每次调用只放 `{send_once, 该臂}` 两个臂**（pilot 的抽样按臂顺序消耗 RNG，共用行集会让区间
  依赖同时比较了谁）；
- 三个训练 seed 各给自己的区间；**不得把 3×24 当 72 个独立样本**；
- 报告逐流救回/新增失败、字节—完成 Pareto、模型间方差。

**主效果门槛（400k vs `send_once`）**：① ≥2/3 seed 完成差 CI 下界 > 0；② 其余 seed 点估计
≥ −0.05 个/episode；③ 正确性门禁 0 违规。

**非劣降本门槛（独立路径）**：① ≥2/3 seed 完成差 CI 下界 ≥ −0.05 **且**字节差 CI 上界 < 0；
② 其余 seed 完成差点估计 ≥ −0.05 且字节点估计不增加；③ 正确性门禁 0 违规。

学习 vs 自身 0 步、loss 下降、发送率变化都只作机制证据。

## 8. 启动上限与停止

| 类别 | episode |
|---|---:|
| 探针 | 12 |
| deterministic resume 回归 | 8 |
| greedy 评价通路 | 1 |
| 训练 | 1 200 |
| 学习评价 | 216 |
| 规则重放 | 24 |
| **合计** | **1 461 / 1 500** |

技术失败计入上限，不补流、不换 seed、不扩容；接近上限即停。

- 正确性失败、NaN/Inf、policy 拒付、log-prob 差 > 1e−6、resume 身份混合 → 停受影响运行并隔离，
  修复后该 seed 从 0 重开，不续接污染 checkpoint；若改动执行代码，旧探针作废并重新冻结。
- 两门槛都未过 → 结论限定为「**在修复动作空间、原 19 维观测/奖励与总 1.2M 训练步下，
  学习附加价值仍未建立**」；不自动加训练、不改特征、不换预算、不加 ACK、不恢复 H=20。
- 任一门槛通过 → 只说明开发集候选成立：提交独立测试协议草案并停止，**不自动打开**
  `800000..800399`。
- H=20 spillover 全程冻结。

## 9. 交付与提交顺序

目录 `runs/comm_budget_masked_retrain_20260922/`：
`COMM_BUDGET_MASK_PROVENANCE_CORRECTION.md/.json`、本协议、`PREFLIGHT.json`、`THROUGHPUT.json`、
`RESUME_CHECK.json`、`EVALUATION_COVERAGE.json`、`RULE_REPLAY_AGREEMENT.json`、
`TRAINING_CURVES.json`、`COMM_BUDGET_MASKED_RETRAIN_RESULTS.md/.json`、
`COMM_BUDGET_MASKED_RETRAIN_MANIFEST.json`、`CODE_FREEZE.json` 与可复跑 harness/测试。

提交：① provenance 更正；② 本协议；③ harness + 确定性恢复测试；④ code-freeze；
⑤ 训练与评价结果；⑥ 总纲/状态。freeze 后不得边跑边改；必须改则旧运行作废隔离并形成新 freeze。

checkpoint、录制流、逐 episode 原始日志、评价行文件一律不入库；他人未跟踪文件不改不删。
