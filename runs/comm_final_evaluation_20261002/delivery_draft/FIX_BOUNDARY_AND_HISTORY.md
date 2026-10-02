# 历史问题与修复边界

状态：交付文档（ROUND37 §7 第 4 项）。本文件说明"修了什么、边界在哪里、什么仍未建立"，
不含本轮效果结论（数据出来前保持为空）。

## 1. 缺陷链（逐轮，按证据而非叙事）

| 轮次 | 事实 | 证据位置 |
|---|---|---|
| ROUND32 | 训练时任务奖励的采集晚一个 relay tick：包装 `env.step` 之后扫 registry，导致 `task_stream` 键整体偏后 | 该轮 rows 的 `task_stream` 键与后续逐事件核对的对照基线；修复前实现见 `git show df176bd…:runs/comm_cost_v2_rerun_20260928/v2_run.py` 的 `_install_taps` |
| ROUND33 | 旧 checkpoint 缺少采集版本标签 ⇒ 无法判断某权重是在哪种信用语义下训出来的 | `ROUND33_CHECKPOINT_MAPPING_ERRATUM.md` |
| ROUND34 | 确认时序错位并改为事件总线上同 tick 写入（`PublishedProgressRewardCollector`）；引入两个版本字段与恢复门；契约与合成回归固定 | `TASK_REWARD_TIMING_CONTRACT.md`、`TRAINING_VERSION_MANIFEST.json`、两套回归 |
| ROUND35 | 真实 runner 小规模核对：45 个完成事件逐条全对、写入键 = 归档键 − 1、A 级被动性逐字节、5/5 篡改副本在恢复前被拒、18 个旧 `.pt` 内容层补齐 | `ROUND35_SMOKE_REPORT.md` 与同目录 JSON |
| ROUND36 | 修复版冻结训练跑满：3 seed × 128 集、96 批、1284 优化器步、0 失败；V1-V6 与 R1-R6 全通过 | `ROUND36_TRAIN_REPORT.md`、`ROUND36_VERIFICATION.json`、`ROUND36_INDEPENDENT_RECHECK.json` |

## 2. 修复到底改了什么（可核对的边界）

只有一处干预：**任务奖励的采集点**从"`env.step` 之后扫 registry（下一拍才记）"搬到"同一拍
`ProgressEvent` 同步分发时读已提交的 registry"，附带 (a) tick 0 保留为未归属 prefix、
(b) 两个版本字段、(c) 恢复门。生产代码 `integrated_sar/`（世界、PILOC 搜索、CTDE 分配、通信
传输、任务登记、事件总线、生命周期指标）自 ROUND33 基线以来 0 次提交、0 字节变化。

函数级证据（ROUND36 G7）：`ppo_update`、`batch_samples`、`likelihood_alignment`、`dual_update`、
`batch_g_hat`、`_unassigned_prefix_totals`、`run_training_episode`、`action_seed`、
`_assert_transport`、`_booked`、`eval_action_seed` 与 ROUND32 时代逐字节相同；变化的只有
`_bundle`、`save_checkpoint`、`load_checkpoint`、`fresh_training_state`、`_install_taps`、
`_run_episode`、`run_eval` 与新增的三个恢复辅助函数。

## 3. 修复改变了什么、没改变什么（ROUND36 实测，口径限制在本轮之外同样适用）

| 没改变 | 改变 |
|---|---|
| 批 0 的 episode 级观测量与归档逐字节相同（12/12 格）；批 0 样本数与 `g_hat` 相同；每笔奖励的**总量**不变 | 每笔入账的 tick 早一拍，且 66 笔中 42 笔的**归属动作行**变化；批 0 的 surrogate 与更新后参数摘要随之不同 |
| 约束统计量的定义与配对方式 | 之后的参数轨迹、λ 轨迹、样本数（这是干预本身，不是效果） |

⇒ 任何"训练结果变好/变坏"的说法都不能由这个修复自身推出；它必须由本轮的独立终评来判，
而终评也只能给出观测差。

## 4. 仍未建立的主张（本轮不能顺手宣称）

1. **δ=0 的总体保证**：未观察到任务损失 ≠ 约束在总体上被满足；训练期三个 seed 的 `g_hat`
   32 批从未为负（ROUND36），这本身说明约束尚未被满足。
2. **训练算法的稳定性**：三个 seed 是同一配方下的三次拟合，不是独立场景样本；符号一致只算描述性。
3. **泛化**：本块 64 个场景是新登记块，但一次终评不足以支撑部署级泛化主张。
4. **优于强规则**：需要与 `immediate`、`late_discovery_gate_147`、`always_hold` 在同一批场景上
   的配对差与分母，才能谈"是否更省成本"；且必须同时报告任务侧。
5. **幂等/重复入账的实机验证**：真实数据里从未出现重复发布（ROUND35/36 计数为 0），
   重复分发不重复入账仍只有合成与结构证据。
6. **两类契约情形**（事件拍无动作行、首动作前 prefix）在本类场景块中不可达，仍只有合成覆盖。
7. **同一批丢包**：各臂发送次数不同 ⇒ 逐消息信道实现抽取次数不同，只可声称场景配对。

## 5. 已知缺陷与不阻塞本轮的遗留

| 项 | 状态 |
|---|---|
| `v2_recover` 一类重放路径的幂等未单独核查 | 本轮不使用（授权边界 §4 明确禁止） |
| ROUND32 的 legacy 采集结论 | 保留为 legacy 基线，不外推到修复版；本轮不引用其数字作效果结论 |
| ROUND34 的"本机无 PyTorch"豁免 | 对本机不成立，已在 ROUND35 实做补齐 |
| 采集器接线变更对旧 checkpoint 的兼容 | 只读可装载并带 `legacy_unversioned_capture_semantics` 标签，禁止续训（ROUND35 已在真实 `.pt` 上验） |
