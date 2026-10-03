# ROUND40 报告：first-report-only 通信策略训练消融

## 执行状态

- 分支：`exp/comm-first-report-training-ablation`。
- 实际环境启动：672 次；正常计划 672/672，技术储备 0/28，上限 700。
- 未构造环境的 preflight/worker 失败 6 次；不计入环境启动额度，逐次类型与恢复记录见启动前失败文件及 Git 外原件清单。
- 训练：NEW 三个 seed 各 128 集、32 个 PPO 批；实际 PPO 批数 96；训练 optimizer.step 次数 556；纯合成 optimizer 更新 0。
- OLD checkpoint 只读评估；源 checkpoint 字节哈希与启动前清单一致。
- 评估固定在已使用的 16 个开发场景，不作显著性、总体非劣或泛化声明。
- Git 外原始产物 4464 个，路径、大小和 SHA256见 `GIT_OUTSIDE_ARTIFACTS.json`。

## 预定问题的回答

### 1. 训练期 refresh 限制是否生效？

纯计算门禁及每个 seed 的首 4 集门禁均验证 `first_emission` 资格、固定 HOLD、候选类别闭合、free 集合、似然和 entropy 排除。训练行的 refresh 实际发射计数须为 0。
逐 seed 门禁数据见下表。

| Seed | 首 4 集 free 首报动作 | 固定 HOLD refresh 候选 | refresh 发射 | 似然最大误差 | 批 0 PPO steps |
|---:|---:|---:|---:|---:|---:|
| 11 | 46 | 457 | 0 | 5.96e-08 | 8 |
| 22 | 60 | 523 | 0 | 1.49e-07 | 8 |
| 33 | 45 | 468 | 0 | 1.19e-07 | 4 |

### 2–4. NEW 相对 OLD：首报、全 HOLD、解码模式与 seed

| Seed | 模式 | OLD 按时任务 | NEW 按时任务 | NEW−OLD 任务 | OLD 字节 | NEW 字节 | NEW−OLD 字节 | 任务改善/持平/退化场景 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | greedy | 95 | 88 | -7 | 234624 | 0 | -234624 | 0/11/5 |
| 11 | sampled | 96 | 96 | +0 | 239760 | 234924 | -4836 | 0/16/0 |
| 22 | greedy | 88 | 88 | +0 | 0 | 72960 | +72960 | 0/16/0 |
| 22 | sampled | 96 | 96 | +0 | 170940 | 240762 | +69822 | 0/16/0 |
| 33 | greedy | 96 | 96 | +0 | 239028 | 239028 | +0 | 0/16/0 |
| 33 | sampled | 96 | 96 | +0 | 240372 | 239376 | -996 | 0/16/0 |

NEW 获得和失去的按时目标 ID、逐场景任务差、最大损失和字节差保存在 `PAIRED_RESULTS.json`。按 seed 与模式分别阅读，不合并成独立样本。

### 5–6. 与 immediate、late147 和 always_hold 的绝对取舍

| Arm | 开发集按时任务 | 平均任务/场景 | 计量字节 | 平均字节/场景 | 首报发射 | refresh 发射 |
|---|---:|---:|---:|---:|---:|---:|
| immediate | 96 | 6.000 | 239028 | 14939.2 | — | — |
| late_discovery_gate_147 | 96 | 6.000 | 77412 | 4838.2 | 31 | 0 |
| always_hold | 88 | 5.500 | 0 | 0.0 | 0 | 0 |

NEW/OLD 对 immediate 与两个规则臂的逐 seed、逐模式比较已列入 `PAIRED_RESULTS.json`。这组开发复用数据只描述本轮固定场景；不据此选择新阈值。

## 训练和 checkpoint 身份

- PPO 批完成数：96；实际 optimizer.step：556。
- ROUND36 的 ep0、ep64、ep128 文件均在启动前按字节哈希核验，并只读载入计算参数摘要。
- NEW ep0 参数与相同 seed 的 ROUND36 ep0 逐参数摘要一致；NEW ep0/64/128 版本字段、episode、update、optimizer steps、lambda 和参数摘要见 `PAIRED_RESULTS.json`。
- 每个 PPO 批都有 immutable pre-update 与 post-update checkpoint、batch 参数前后链、optimizer/lambda/RNG 恢复状态和 commit receipt。OLD 未写回。
- 旧训练版本恢复到 NEW 被真实 `load_checkpoint` 入口拒绝；OLD 仅通过 read-only 入口评估。

## 限制与后续

ROUND39 的刷新抑制结果不能推出刷新普遍无用。本轮只检验训练候选动作资格变更，并且 OLD 也在相同的 first-report-only 外部运行限制下评估。开发场景此前已使用，结果不能替代独立确认。

建议由决策 agent 基于逐 seed 配对结果决定是否设计一轮独立确认；本轮不自动续训、调参、改默认值或推送。

## 补充：动作概率、更新读数与恢复/分析勘误

以下诊断由已保存 training/evaluation tapes 与批记录纯计算汇总，详表见 `TRAINING_ACTION_DIAGNOSTICS.json`。计数只包含 `free_first_reports`；预算/容量 mask、fixed-HOLD refresh 与 forced 候选不进入 actor 概率分母。代码中的 greedy 规则是 `P(SEND) >= P(HOLD)` 时 SEND（相等时 SEND）。

### 训练期 SEND/HOLD

| Seed | free 首报决策 | rollout 实际 SEND/HOLD | rollout SEND 比例 | tape 概率均值 | 按同一概率重建的 greedy SEND/HOLD | 无 free 首报候选行比例 |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 1,446 | 762 / 684 | 52.7% | 0.527 | 1,035 / 411 | 85.5% |
| 22 | 1,961 | 760 / 1,201 | 38.8% | 0.370 | 0 / 1,961 | 80.9% |
| 33 | 1,206 | 762 / 444 | 63.2% | 0.622 | 1,107 / 99 | 88.0% |

### ep128 最终模型开发集动作

| Seed | OLD greedy SEND/free | NEW greedy SEND/free | NEW greedy 平均 P(SEND) | NEW sampled SEND/free | NEW sampled 平均 P(SEND) |
|---:|---:|---:|---:|---:|---:|
| 11 | 94/322 | 0/6,416 | 0.357 | 93/264 | 0.354 |
| 22 | 0/6,416 | 26/6,279 | 0.480 | 96/203 | 0.448 |
| 33 | 96/96 | 96/96 | 0.728 | 96/147 | 0.730 |

96 个 OLD/NEW greedy 场景运行的 tapes 均与保存概率的 argmax 一致，free action mismatch 为 0。表中比例来自不同实际 rollout 状态，不是同一批逐消息候选的概率对照。NEW greedy 的主要变化是 s11 变为全 HOLD（对应 7 个任务退化），s22 只有 26 次自由首报 SEND 且任务不变、字节上升，s33 保持 96 次首报 SEND并与 immediate 同字节；没有跨 seed 的一致成本收益。sampled 三 seed 任务均与 OLD 持平，成本方向仍不一致。

### 批级优化读数

| Seed | PPO batches | optimizer.step | `g_hat` / lambda | 预裁剪全局梯度范数均值范围 | PPO ratio clip fraction 范围 | 最大 logged/recomputed log-prob 差 |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 32 | 188 | 0 / 1.0（全程） | 0.8606–5.7060 | 0–0.0035 | 2.39e-07 |
| 22 | 32 | 188 | 0 / 1.0（全程） | 1.4563–5.9249 | 0 | 2.39e-07 |
| 33 | 32 | 180 | 0 / 1.0（全程） | 0.8307–9.8049 | 0 | 1.79e-07 |

`gradient_norm_mean_preclip` 是冻结 trainer 的 `clip_grad_norm_` 返回的批内均值；`ppo_ratio_clip_fraction` 是 PPO 概率比裁剪率，不是全局梯度超阈次数。逐 minibatch 的全局梯度裁剪次数没有持久化，本报告不从均值推断。模块梯度 L1、每批概率分布、动作、mask 类别及 likelihood 差均在 `TRAINING_ACTION_DIAGNOSTICS.json` / `TRAINING_DIAGNOSTICS.json` 留存。

### 绝对任务/成本取舍

`late_discovery_gate_147` 为 96/96 个按时任务、77,412 B。保持 96 个任务的 NEW 结果是 s11 sampled（234,924 B，较 immediate 省 4,104 B）、s22 sampled（240,762 B）、s33 greedy（239,028 B）及 s33 sampled（239,376 B）；它们全部比 late147 多花 157,512、163,350、161,616、161,964 B。s11 greedy 和 s22 greedy 分别少 8 个任务；前者 0 B，后者 72,960 B。没有 NEW 模式同时达到 late147 的任务数与成本。

### 失败与恢复登记

- 运行环境：Python 3.10.19、PyTorch 2.9.0+cu128，CPU 执行；未下载权重或安装大型依赖。本地完成交付后停止，不推送。
- 构造环境前的 worker/preflight 错误共 6 条，全部没有在失败调用中构造环境或执行 optimizer step；每条原因及恢复记录保存在 `preconstruction_failures/`。成功环境账本仍为 672/672，`failed_technical` claim 为 0。
- 续跑/首批独立门禁期间 4 个运行器/审计适配错误逐条记录于 `training_progress/worker_failures/failure_01.json`–`failure_04.json`。第 1 条发生在 seed11 四个环境和 PPO batch 0 已完整提交后；随后 3 条没有启动环境或追加更新。修复只增加恢复元数据、将 structured cost stream 的 `airtime_bytes` 输入独立审计分桶器；真实 reward 路径、已提交更新与 checkpoint 未改。
- 初次汇总器因 checkpoint reconciliation 循环复用 seed33 的 summary 而误报 seed11 NEW ep0 身份不符（`failure_05.json`）。只读重载后，seed11 OLD 与 NEW ep0 参数摘要均为 `7eff0f40203fdae63437bb07a16204ab0289bf938e9c263f781b3adfc40d5eff`；冻结 analyzer 源文件保持未改，仅由版本化的 `r40_analysis_correction.py` 在内存绑定当前 seed 的 summary 后重建报告。其启动适配器首次调用错账本模块（`failure_06.json`），在分析/环境前退出；修正后纯分析成功。以上共 6 条续跑/后处理适配记录，不代表环境技术失败；储备使用 0。

**唯一建议：**不要把本变体晋级为有效通信节流策略；将此开发结果交给决策 agent 决定是否值得以新的、独立登记的场景块检验不同机制。本轮不追加环境或训练。
