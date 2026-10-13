# ROUND71 执行依据（授权原文档案；不是第二份 spec）

- 任务：`ROUND71-APPROVED-CALIBRATION-FIXED-TRIAL`
- 分支：`exp/round71-critic-initialization-calibration`
- 轮次目录：`runs/comm_bc_initialized_constrained_ppo_20261007_r2/round71_critic_initialization_calibration/`
- 批准时间：2026-10-13（本地主会话，人类直接授权）
- 准备就绪交付（零消费）：`89fc83ef4ba45e3fde8168e3a3bab1cfa6fecb5d`（分支 `analysis/round71-critic-calibration-readiness`）

## 授权原文（逐字存档）

> 批准 ROUND71：582 正常＋18 储备，环境上限600；离线实际 trunk 输入上限72,000行；合成 optimizer step最多16次。以后同类有限实验由你决定并直接执行，不重复申请批准；不推送、不合并、不部署。

## 额度契约（硬性，失败不退额）

| 项目 | 上限 | 说明 |
| --- | ---: | --- |
| 正常计划启动 | 582 | 384 训练（3 seed × 128 集，首 4 集计入 128）+ 192 评估（3 seed × 2 模式 × 32 场景 1800000..1800031）+ 6 焦点（scene 1700027：NEW INIT/NEW ep128 × 3 seed × greedy） |
| 技术失败储备 | 18 | 任何失败不退额 |
| 环境构造硬上限 | 600 | 含失败、重试、隔离；不借用 R60/R66/R68 历史额度 |
| 离线 trunk 输入行 | 72,000 | 真实 (N,19) 行，N>0、单批 ≤1024，forward 前 debit，不 refund |
| 合成 optimizer step | 16 | 仅门禁用；真实训练更新另计 |

## 唯一科学干预

将每个 seed 的 R68-ep0-based NEW 模型的两个 critic 初始仿射头（task `inner.value_head`、cost `cost_head`）替换为
该 seed 自有 R68 128 集教师数据上的 per-seed dual-output ridge 校准；trunk、actor 头、初始策略不变。

教师数据 = R68 真实训练场景 1300000..1300047 的 384 个 tape（动作前 19 维输入 + 同时刻契约的 MC task/正成本回报），
经同一冻结 R68 ep0 trunk 提取真实 64 维 embedding，按 R66 顺序 mean-over-targets 汇聚到联合行；
每 seed 一次 ridge（1e-3、全行中心/尺度、常数列尺度 1、截距不惩罚、无 CV、无 seed 选择）；
精确逆变换 B=A/sigma、b=c-mu@B 写入真实 Linear(64,1) 头；随后重建 fresh optimizer + lambda=1 + R68 登记初始 RNG。

## 继承不变（以 R58/R60/R66/R68 已验收活体配置为准，不用导入默认值）

trunk + early/late actor 头（late 初始值 ×0.10，仅一次）、19 维观测、一次性 SEND/SKIP、真实顺序预算 mask、
正成本、lambda*A_task−A_cost、双 critic 与 critic→trunk 梯度隔离、奖励事件归属、GAE（γ=1、λ=1）、
ALL_ROWS 归约、熵、全局裁剪、对偶更新（eta=0.5）、搜索/分配 checkpoint、世界字段、6 目标、29,300B 预算、
ACK 关闭、三个训练 seed 及各自 128 集顺序、greedy 平局规则、sampled 评估 RNG 方案。

## 技术契约指针

私有准备与零消费验收（真实 R68 运行接线、来源清单、fail-closed 门、恢复来源校验、只读离线输入计数）
见 `analysis/round71-critic-calibration-readiness/`（`ROUND71_READINESS.json`、
`ROUND71_RUNTIME_BINDING_ACCEPTANCE.json`、`ROUND71_OFFLINE_INPUT_COUNT.json`）。
本轮不复制其治理历史；本文件只登记授权与额度。

## 如实登记

- 本轮新增离线教师数据暴露：不是与 R68 的同训练预算竞赛，报告必须如实说明。
- 评估块 1800000..1800031 是已使用开发块，不是未见确认集。
- 不推送、不合并、不部署。