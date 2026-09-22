# 下一轮：预算动作 mask 修复后的从初值有界重训

日期：2026-09-22。对应总纲 v3.1。**状态：已执行完毕（历史指令，勿重复执行）**——分支
`exp/comm-budget-masked-retrain`，结果见
`runs/comm_budget_masked_retrain_20260922/COMM_BUDGET_MASKED_RETRAIN_RESULTS.md/.json`：
正确性 0 违规、可学习样本按绝对量充分，主效果与非劣降本两条门槛均 0/3，因此结论限定为
"在修复动作空间、原 19 维观测/奖励与总 1.2M 训练步下，学习附加价值仍未建立"；
封存块 `800000..800399` 未打开，H=20、第二预算、ACK 与扩维未执行。

## 1. 决策、起点与唯一研究问题

从 `fix/comm-budget-action-mask@e692607212fcc67fc82d045a8a53d44c79bde4d7` 创建
`exp/comm-budget-masked-retrain`。开始前核对远端 SHA、工作区、AGENTS、ARCHITECTURE、
PROJECT_STATUS、PROJECT_CHARTER v3.1，以及
`runs/comm_budget_action_mask_fix_20260922/COMM_BUDGET_ACTION_MASK_RESULTS.md/.json`。

本轮正式授权从初值训练。唯一问题是：

> 在预算不可执行 SEND 已于采样前顺序 mask、行为概率与 PPO 更新一致的动作空间中，固定主预算和原观测/奖励，
> 三个独立初始化的学习型发送策略能否在开发流上稳定超过锁定的强规则 `send_once`，或在任务效果非劣时降低字节？

旧 400k checkpoint 是缺陷动作空间的产物，只可作为历史证据，**不得**加载权重、恢复 optimizer 或作为 warm start。
本轮每个学习臂必须从 `Controller(training_seed)` 的新随机初始化开始，并在 0 步保存权重指纹。

## 2. 先闭合上轮 provenance，不把账本瑕疵带进 checkpoint

上轮 manifest 的执行模块清单包含一个仓库中不存在、运行路径也未导入的
`integrated_sar/comm_transcript.py`，因而显示 `dirty_modules`；结果/manifest 的记录头是结果提交
`7f9929c`，而最终交付头 `e692607` 只追加了只读兼容测试。这不推翻 M0–M3，但正式训练前必须修正身份口径：

1. 写 `COMM_BUDGET_MASK_PROVENANCE_CORRECTION.md/.json`，说明该路径为何不在实际 import/执行图中；
2. 修复 manifest builder 的 executed-module allow-list，不重写旧 smoke 数字；
3. 协议、训练 harness、测试全部提交后形成一个 **code-freeze SHA**，在任何训练 episode 前登记；
4. manifest 对 freeze SHA 下的实际执行模块记录 Git blob/hash；运行前后必须
   `git diff --quiet <freeze> -- <executed paths>`；结果文档的后续提交不冒充执行代码 SHA；
5. 任一执行模块为未跟踪或相对 freeze 有改动时 fail closed，不启动/继续训练。

这是 provenance 修正，不新增 episode。旧结果文件保留，不就地伪装成在最终提交上运行。

## 3. 冻结的数据、场景与比较对象

### 数据用途

- 训练：既有 48 条流 `700000..700047`；可以循环重放，不生成新流；
- 开发评价：既有 24 条流 `700048..700071`；已经多轮使用，**不是独立测试**；
- 最终封存：`800000..800399` 继续保持盘上 artefact 为 0，不生成、不读取；
- `900000..900199` 已消费，只作历史证据，不进入本轮训练或评价。

训练前重新核对 stream 内容摘要与既有 manifest；编号相同但内容摘要不同必须停下。

### 唯一主设置

| 项目 | 冻结值 |
|---|---|
| timing | `policy_owned_timing` |
| 上行预算 | **29,300 B/episode** |
| ACK | 关闭 |
| 信息路径 | isolated（关闭现场抢先刷新与常规 gossip，沿用第五轮定义） |
| 信道 | 无损，delivery probability = 1.0 |
| episode | 1,000 环境步 |
| feature | `comm-send-state/1`，19 维，不增加费用特征 |
| 网络/奖励 | 与第五轮逐值相同，不改结构、不塑形 |
| trainer | `onpolicy-v3-budget-mask`，`budget-sequential-mask/1`，`onpolicy-tape/2` |

不同时跑 87,900 B，不加 ACK/重试，不改 deadline、任务分布、搜索/分配模型或生产默认值。

### 对照

主对照固定为第五轮在同一主预算、同一协议下训练前锁定的 `send_once`。在 24 条开发流上随本轮代码重放一次，
核对其发送序列、任务结果和字节；如果规则行为相对归档无解释差异，先停并归因，不换一个更弱规则。

同时评价三个学习 seed 的 0 步初值，用于回答“是否学到”；但学习 vs 自身初值不能替代学习 vs 强规则。
其他历史规则只作上下文引用，不重新搜索参数、不根据学习结果更换主对照。

## 4. 训练资源与确定性

### 训练规模

- training seed = `1, 2, 3`，三者都从随机初值开始；
- 每个 seed 固定 **400,000 环境步 = 400 episode**；总上限 **1,200,000 环境步**；
- 4 episode/rollout update；PPO 超参保持第五轮逐值相同；单线程；
- checkpoints：0 / 100k / 200k / 300k / 400k；**400k 末轮是唯一主判据**；
- 三个 seed 使用同一冻结的训练流访问序列，模型初始化与策略抽样 RNG 按 seed 区分；
- 不按中间开发集最好点选模型，不因某 seed 早赢而提前停，也不因没赢而给它追加步数。

训练前各 seed 运行 4 episode 吞吐/身份探针，共最多 12 episode、`optimizer.step=0`；据此报告实测时间，
不沿用旧耗时。探针不计入 1.2M 训练步，但计入本轮总启动上限。

### checkpoint 与可复现恢复

checkpoint 必须保存并核对：controller、optimizer、训练环境步、episode、scenario cursor、Python/NumPy/Torch RNG、
trainer tag、mask/tape schema、feature layout、code-freeze SHA、配置 hash 与初始权重 fingerprint。

训练前以训练流做一个小型“连续两轮 vs 保存后恢复再跑一轮”回归测试：下一批场景顺序、动作 tape、loss、
权重与 optimizer state 必须逐项一致或在预先声明的浮点容差内一致。身份不符和 RNG 缺失必须拒绝 resume。

## 5. 真实训练样本口径

`181/20,655 = 0.88%` 只描述上轮固定初值 smoke 的候选占比，**不是停止门槛**。其绝对量是
22.6 actionable/episode；按 400 episode 粗略外推约 9,000 个 actor 决策/seed，但真实数量会随策略消耗预算而变化。
因此本轮逐 checkpoint 必须同时报告：

- actionable 总数、每 episode 均值/中位数/分位数；
- 含至少一个 actionable 的 episode 数与零 actionable episode 数；
- first/update/retry 消息类型分布及 episode 内时间分布；
- initial executable、顺序后 actionable、masked by credit/capacity、请求、实际发射与拒付；
- 每 update 的 actor terms、有效 minibatch 数、optimizer steps；
- stochastic 与 greedy 的每候选熵、SEND 率、额度耗尽步；
- policy/value loss、explained variance、KL、clip fraction、gradient norm、NaN/Inf；
- logged vs recomputed log-prob 最大差。

只报 actionable 百分比或总候选数均不合格。若某 seed 到 20k 步仍累计 actionable=0、actor 有效 minibatch=0，
或权重从初值完全不变，则停止该 seed 并归因为训练接缝失败；不得用加步掩盖。

## 6. 训练前门禁

以下全部通过后才能启动三个 400k 运行：

1. provenance correction 闭合，code-freeze SHA 已提交且执行路径相对 freeze 干净；
2. 上轮 M0–M3 报告器可在当前树重跑通过；完整 `integrated_sar` 测试和 `python -m integrated_sar` 通过；
3. `budget_mask=True`，费用/顺序/tape schema 与 `e692607` 修复语义一致；
4. 12 episode 探针中 policy budget/capacity refusal=0、requested=executed、log-prob gap ≤1e−6；
5. 三个初始权重 fingerprint 不同，且均未加载旧 checkpoint；
6. deterministic resume 测试通过；封存块仍为 0 artefact；
7. 协议与主判据已单独提交，时间早于第一个训练 episode。

任一失败：不启动正式训练；只提交修复/失败归因。修代码后旧探针隔离，形成新的 freeze SHA，再从 0 开始。

## 7. 开发评价与统计

### 评价计划

- 0 步：三个 seed 各在全部 24 条开发流评价；
- 100k / 200k / 300k：各在固定的前 8 条开发流评价曲线；
- 400k：三个 seed 各在全部 24 条开发流评价；
- `send_once`：在全部 24 条开发流评价一次并按 seed 配对复用；
- 全部学习评价使用 greedy policy，但与 stochastic 训练共用相同预算 mask。

不从中间 checkpoint 选优。技术失败计入启动上限，不补流、不换 seed。总 episode 启动上限 **1,500**：
12 探针 + 1,200 训练 + 评价/规则余量；接近上限时停止而不是扩容。

### 指标

- 主效果：按时完成任务数，单位“个/episode”；同时列逐局率与汇总率，不混用；
- 成本：上行尝试字节，丢包仍计费（本轮无损）；
- 统计：24 个场景 seed 内配对 percentile bootstrap，B=10,000，RNG=20260922；
- 三个训练 seed 分别给出 CI，**不得把 3×24 当成 72 个独立样本**；
- 报告逐流新增失败/救回、任务级状态转移、字节—完成 Pareto 与模型间方差。

### 预登记判定

主效果门槛（400k vs `send_once`）：

1. 至少 2/3 seed 的按时完成差 95% CI 下界 **> 0 个/episode**；
2. 第三个 seed 的点估计 ≥ −0.05 个/episode；
3. 所有正确性门禁为 0 违规。

非劣降本门槛（独立路径）：

1. 至少 2/3 seed 的完成差 CI 下界 ≥ −0.05，且字节差 CI 上界 < 0；
2. 第三个 seed 的完成差点估计 ≥ −0.05，且字节点估计不增加；
3. 所有正确性门禁为 0 违规。

学习 vs 自身 0 步、训练 loss 下降、发送率改变都只作机制证据，不算通过上述门槛。

## 8. 停止与下一分支

- correctness failure、NaN/Inf、budget/capacity policy refusal、log-prob gap >1e−6、resume 身份混合：
  停受影响运行并隔离，修复后该 seed 从 0 重开；不续接污染 checkpoint。
- 三 seed 跑满但两条门槛均未过：结论写“在修复动作空间、原 19 维观测/奖励和 1.2M 总步下，
  学习附加价值仍未建立”；不自动加训练、不改特征、不换预算、不恢复 H=20。
- 任一门槛通过：只说明开发集候选成立；提交独立测试协议草案并停止，**不得自动打开** `800000..800399`。
- H=20 spillover 全程冻结，不能与本轮训练结果混为一个实验。

## 9. 交付、目录与提交纪律

目录建议 `runs/comm_budget_masked_retrain_20260922/`：

- `COMM_BUDGET_MASK_PROVENANCE_CORRECTION.md/.json`；
- `COMM_BUDGET_MASKED_RETRAIN_PROTOCOL.md`（训练前单独提交）；
- `PREFLIGHT.json`、`THROUGHPUT.json`、`TRAINING_CURVES.json`；
- `COMM_BUDGET_MASKED_RETRAIN_RESULTS.md/.json`；
- `COMM_BUDGET_MASKED_RETRAIN_MANIFEST.json`；
- 可复跑 harness、评价/统计脚本与对应测试。

提交顺序：① provenance correction；② 协议冻结；③ harness/确定性恢复测试；④ code-freeze；⑤ 训练结果与评价；
⑥ 总纲/状态。执行代码 freeze 后不得边跑边改；若必须修改，旧运行作废隔离并形成新 freeze。

不得提交 checkpoint、stream、逐 episode 原始日志或用户未跟踪文件。推送并核对远端 SHA 后停止。
本轮授权只有上述 3×400k 主预算训练；不授权第二预算、ACK 学习、H=20、最终封存测试或 observation 扩维。
