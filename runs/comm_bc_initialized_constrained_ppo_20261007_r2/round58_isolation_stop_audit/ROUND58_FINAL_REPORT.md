# ROUND58 固定计划最终交付

**结论：原固定计划执行完成，但未建立跨 seed 一致的附加降本收益。** 本报告针对事前登记、此前未用于效果选择的 32 个前瞻性新场景；不宣称总体任务零损失、正式非劣或可部署。旧块 `development_go=false` 原件未改，本轮冻结方案没有给新场景块另设 go 判据。

## 执行、门禁与来源

- 训练：seed 11/22/33 各 128 集、32 批，合计 **384 集、96 个已提交 PPO 批**。最终 ep128 模型来自已批准的 V29/V32 源，三个种子的 checkpoint 字节哈希及末次提交见 `ROUND58_FINAL_ACCOUNTING.json`。原三个首批及首 12 集未重跑，历史 gen1/gen2 无效状态未用作训练初值。
- 评估：R53/R56/R58 × 3 seed × greedy/sampled × 32 场景 = **576 格**；immediate/late ×32 = **64 格**。场景固定 `1700000..1700031`。原始唯一 complete claim、row、lifecycle/transcript/tape 的适用收据和逐项 SHA 由 `r58_final_source_aware_stats.py` 只读核验，1024 个来源列于 `ROUND58_FINAL_SOURCE_INDEX.json`。原生规则只有三件真实产物，不伪造 actor tape。
- V42 真正 CLI 正向门禁 `ROUND58_V42C_ACTUAL_EXISTING_CELLS_GATE.json` 为 passed：复核既有 608 个评估格、批准身份、路径及冻结 native late 构造接缝，且在新 provider 前停止。V42 原受控执行会话 39613 最终退出码 **0**；`ROUND58_V42_CONTROLLED_STATE.json` 为 `fixed_plan_cli_complete`。七个子周期分别新增 4、5、5、5、5、4、4 格，**累计**完成 32 个 late 格；前六个子周期在已提交 cell 后、下一 claim 前到达预设安全边界，第七个仅完成剩余 4 格。历史 V41 late 构造失败发生在 claim/环境之前，原件保留。
- 真实 late 使用冻结 `v2_rules.py` 的 `cut_steps=147`。私有桥只将旧调用名 `cut=147.0` 精确映射到此参数；不修改规则判定、科学配置、canonical 源码或旧输出。训练仍为 β=1.00、critic→共享 trunk 梯度隔离、19 维一次性 SEND/SKIP、正成本与原冻结 PPO/GAE/对偶配置。修复/恢复身份及来源沿用 V42 freeze/authorization，未把修复后的新身份回贴旧 checkpoint。

## 原始账本与诊断

| 项目 | 最终实数 | 证据边界 |
|---|---:|---|
| 有效逻辑任务 | 1024/1024 | 384 训练 + 640 评估；0 缺失、0 在途 |
| gen3 normal 槽 | 1024/1024 | 原始 claim/slot |
| gen3 reserve 槽 | 5/24 | 五条技术失败均保留；剩余 19 |
| 历史 gen1/gen2 消费 | 16 | 原类别冲突不重标；原件核验 |
| 跨代累计尝试槽 | **1045/1064** | 16 + 1024 + 5；余量 19 |
| 跨代实际环境构造/运行 | **1043** | 两条零构造技术失败已占槽；不退额 |
| gen3 实际 optimizer hook | **556** | 原始跨根 call index 连续 1..556 |
| gen3 有效 PPO 批 | **96** | 93 条直接 PPO hook + 原首三批提交/记录；不补造直接 hook |
| 旧 gen1/gen2 optimizer 调用 | **48，另列** | 直接与重建证据混合，非有效 gen3 更新；来源/hash见账本 |
| checkpoint 写入 journal 事件 | **312** | 事件数，非不同文件数 |

五条失败中，seed11 episode4 的两次导入错误均零构造；seed11 episode43 外部中断曾构造/调用环境，claim 的 completed_steps=0 **不证明实际模拟步数为零**；模型格和 immediate 规则格各一次旧产物收据故障后完成了各自批准重试。所有旧失败 claim、row、journal、checkpoint 与停止记录均保留。历史 16 条原始 complete claim 分别记录 1000 完成步，gen3 原始 claim 至少记录 1,026,000 完成步，合计可核下界 1,042,000；episode43 中断的实际步数仍未知，不能把下界当精确总仿真步数。

被动逐 minibatch 原记录 **556 条**按冻结顺序、批索引与批提交闭合到 96 批；它不含逐行 training seed 字段。原 9 个梯度诊断点（每 seed 的批 0/15/31）均与当时更新前/后 checkpoint 的参数字节身份闭合；它们是诊断点，不是独立效果样本。全部 96 批的更新前/后 checkpoint 字节身份在 `ROUND58_FINAL_CHECKPOINT_INDEX.json` 闭合：其中 V18 源有各 9 个 before/after 文件在先前归档后不处于记录的原路径，但在唯一明确的归档目录里按 SHA 找到，未改写原件。详见 `ROUND58_FINAL_DIAGNOSTIC_COVERAGE.json`。完整原始跨根模型/claim/收据引用及 SHA 分见 `ROUND58_FINAL_SOURCE_INDEX.json`、`ROUND58_FINAL_ACCOUNTING.json`，权重、tape 与日志均未入 Git。

## 冻结统计与科学结果

从原始整数按场景配对，固定 `PCG64(20261008)`、共享 `10000×32` 重采样索引、`linear` 2.5%/97.5% 分位；候选−参照，seed 和 greedy/sampled 分列。表中任务数为 32 场景内按时完成总数，费用为 B/场景；所有未舍入差值、逐场景整数、目标 ID 交换、相对 R53/immediate/late 的区间在 **同一份** `ROUND58_FINAL_STATISTICS.json`。

| seed / mode | R56 按时任务 | R58 按时任务 | R56 B/场景 | R58 B/场景 | R58−R56 B/场景，95% CI |
|---|---:|---:|---:|---:|---|
| 11 greedy | 191 | 190 | 3900.375 | 3498 | −402.375 [−816, −78] |
| 11 sampled | 191 | 190 | 3828.375 | 3413.4375 | −414.9375 [−841.125, −78.375] |
| 22 greedy | 190 | 191 | 2788.125 | 3660.375 | +872.25 [246.375, 1667.7609375] |
| 22 sampled | 190 | 191 | 2859.375 | 3587.8125 | +728.4375 [234.1875, 1392.1171875] |
| 33 greedy | 190 | 191 | 3576.375 | 3822.375 | +246 [0, 654] |
| 33 sampled | 191 | 191 | 3660.375 | 3744.375 | +84 [0, 252] |

相对 R56，seed11 两模式在场景 `1700027` 丢失 `piloc-e1-target-5`；seed22 两模式救回同一目标；seed33 greedy 救回该目标、sampled 无目标集合变化。seed11 降本伴随任务损失，seed22 成本上升，seed33 成本上升且两个区间下界为零。故 **β=1.00 未产生跨 seed 一致的附加闭环降本**。这不是 β 梯度错误的证明，也不是调整 β、阈值或追加训练的授权。R58 对 immediate 的费用低，但这不消除对既定 R56 参照的混合结果和 seed11 任务损失。旧 4838.25 B 门槛不用于新块；CI 含 0 不称等效。此前未观察场景和真实部署的性能仍未建立。

## 本次验收与边界

- V42 冻结前受影响窄测试：native late 构造 **2/2，exit 0**；受控执行器 **1/1，exit 0**；实际 V42 CLI 零环境 gate **passed**。最终统计纯算术窄测试本次复验 **2/2，exit 0**。最终 source-aware 统计、accounting 和 verdict 脚本各 **exit 0**；这些是只读/纯计算，不能冒充新的模型更新。完整项目套件未重跑。
- V41 曾运行的两个过时历史测试存在硬编码早期 claim 数/待重试状态的失败；它们不构成当前 1024 原始任务、收据及冻结 V42 门禁的替代证据，也未被改写为成功。相关旧结果保留。
- 本次收尾分析新增环境、episode、claim、optimizer.step、RNG 恢复、checkpoint 写入、评估格均 **0**；V42 实际执行的 32 个 late 格已单列在上表总账，绝不把纯分析计数写成实验运行。保护的两个 ROUND46 脏文件维持既定 SHA256，未修改、未暂存。

**机器结论**：`ROUND58_FINAL_MACHINE_VERDICT.json`。执行完成与科学达标是两件事：前者成立，后者未成立。本轮本地收尾后停止；不推送、不合并、不部署、不自动开启下一轮。
