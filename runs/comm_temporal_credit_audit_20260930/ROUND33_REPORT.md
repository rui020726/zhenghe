# ROUND33：任务奖励时间跨度与信用传递审计

本地分支：`analysis/comm-temporal-credit-audit`，从 ROUND32 交付头
`01e5a677cd18a7abfdb97585315ba05251d4e0de` 建立；原始执行提示词已独立保存并提交。

## 结论

发现一处违反冻结时间契约的任务奖励错位。ROUND32 的任务完成事件发生在 relay/tape tick `t`，但训练行里的任务奖励流在 `t+1` 才记账。源码调用顺序和全部 768 集归档记录互相印证：4,363 个按时完成事件全部在同一物理 tick 有 tape 行，4,363 个奖励单位全部出现在生命周期时间戳后一 tick；其中 2,821 个奖励被 `_bundle` 归到完成之后新采样的 tape 行。剩余 1,542 个事件在 `t+1` 没有新 tape 行，仍归到 `t` 行。

这不是 `gamma` 或 `lambda_GAE` 的选择问题。固定 critic 时，2,821 个被移到后一 tape 行的奖励，对完成 tick 行的直接 GAE 系数从 1 变为 `gamma × lambda_GAE = 0.95`；更重要的是，奖励还会落在任务已完成后的决策状态上。按 ROUND23 的停止规则，本轮只报告、不修训练器、不训练、不提出参数干预。

审计仍完成了授权的只读分析。18 个 `.pt` 文件的 SHA256 都与 checkpoint index 一致，但只有 6 个 batch 0 点的 checkpoint 参数和 lambda 身份与当前批记录一致；另 12 个 batch 15/31 点也与 `UPDATE_MECHANISM.json` 的样本数冲突。因此真实批次的 C/D/E 诊断只保留 6 个 batch 0 点，未替这 12 点选边或放宽容差。

## 契约违例证据

源码顺序如下：

1. `Task03Adapter.step` 先运行 `CommRelay.tick()`，这时策略采样并写出 step `t` 的 tape 行；随后执行任务环境步。
2. `v2_run._install_taps` 包装原始 `env.step`。原始环境步返回后，tap 立即扫描 `run.registry.snapshots()` 并写入任务奖励。
3. 新完成的 `ProgressEvent` 是 `Task03Adapter.step` 在环境步返回后才创建并返回的；`JointSAROrchestrator` 又是在 `Executor.step` 返回后才发布这些事件并更新 registry。
4. 因此 tap 在完成事件发布前读到的是旧 registry。下一 relay tick `t+1` 扫描时才看见完成状态并登记奖励。

相应位置：`runs/comm_cost_v2_rerun_20260928/v2_run.py:87–113`、`integrated_sar/adapters/task03.py:1187–1225`、`integrated_sar/orchestrator.py:94–100`。记录中的完成 timestamp 与奖励 tick 的逐集、逐事件匹配为：

| 核验 | 结果 |
|---|---:|
| 读取的训练集 / tape | 768 / 768，缺失 0，解析错误 0 |
| lifecycle 按时完成事件 / 对应奖励单位 | 4,363 / 4,363 |
| 完成事件时间戳与奖励 tick 相同 | 247 |
| 奖励 tick = 完成事件时间戳 + 1 | 4,363（100%） |
| `t+1` 新 tape 行接收奖励 | 2,821 |
| 奖励仍由 `t` 的 tape 行接收 | 1,542 |

完成事件所在行有 652 个 active、3,711 个 full-mask；训练奖励最终归属行为 599 个 active、3,764 个 full-mask。被移到 `t+1` 新行的 2,821 个单位中，424 个落在 active 行，2,397 个落在 full-mask 行。总奖励数仍与行记录和 lifecycle 对齐，说明总量闭合没有发现问题；错在事件时间与信用归属。

`v2_trainer._bundle` 本身确实选取 `step` 之前或同 tick 的最新决策（`v2_trainer.py:112–129`），所以 bundler 的实现符合其局部规则；但输入的任务奖励 tick 已晚一拍，组合起来仍违反 ROUND23 的事件时间契约。ROUND23 T4 明确要求区分账本总量和时序信用，并要求错误时停止、不修补，本轮据此停止参数解释。

## A：开发场景已有任务上限

对 16 个已使用开发场景只读核对显示：immediate 在 16/16 场景都按时完成 6 个任务，且每场景硬预算均为 29,300 B。192 行 learned/reference 比较全部满足同预算、场景/目标配置、冻结 checkpoint 和 discovery stream 对齐条件。

这只能登记为“这组已使用开发场景已有合法策略达到任务上限”。它不证明存在更低成本的零损失策略，也不代表模型能学到该策略。greedy 与 sampled 仍分开看；没有混合均值。

## B：训练时间轴与奖励/成本归属

- 768 个 episode 共 146,475 行：active 11,638（7.95%），full-mask 134,837（92.05%），no-candidate 0。全 mask 行保留在 GAE 递推中；没有候选的 tick 不生成 tape 行。
- 冻结参数为 `gamma=1.0`、`lambda_GAE=0.95`。递推单位是 tape/决策行，不是物理环境 tick；每跨一条 tape 行乘一次 lambda，包括 full-mask 行。`lambda_dual` 是另一个量。
- 训练记录中的 4,281 个非零任务奖励 tick，共 4,363 个单位；按已记录奖励 tick 到最近先前 active 行计算，行距离中位数 20（四分位 8–34，最大 313），环境 tick 距离中位数 21（四分位 8–38，最大 560）。全部 4,281 个 tick 都找到先前 active 行。这是归属时间接近度，不是因果证明；并且该 tick 本身晚于实际完成事件一拍。
- 通过 target ID 连接到首个自由决策的完成事件有 3,008 个，另 1,355 个完成目标没有可识别的先前自由决策行；3,008 条连接的首个自由决策至物理完成距离中位数为 22 行/22 环境 tick（四分位 15–28，最大 55）。身份连接不证明该动作造成了完成。
- 记录的 21,289,578 B 非零成本落在 active 行（5,489 个非零成本 tick），full-mask 行、前缀均为 0。成本 tap 在 relay tick 的动作发送后记账，检查未发现同类一拍偏移。
- 每集执行固定 1,000 个环境 tick，无早停。tape 没有 terminated/truncated 区分位；trainer 无条件把最后一条记录行设为 terminal，前序用下一 tape 行 value，末行 bootstrap 置零。因此最后一行究竟是吸收终止还是时间截断仍未确认，本轮不据此定性实现错误。

## C：任务优势分量（身份一致的六个 batch 0 点）

以下是每个 seed 的 active 行均值；`ALL_ROWS` 和 `ACTIVE_ROWS` 的 batch 0 在三个 seed 上数值相同。奖励分量来自已记录的延迟 task stream。闭合误差最大为 `5.0e-10`，与 `UPDATE_MECHANISM.json` 摘要闭合。

| Seed | Active 行数 | 显式任务奖励 | 当前 value 项 | next-value/bootstrap 项 | `A_task` | `A_cost` | `lambda_dual*A_task - A_cost` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 30 | 1.871 | +2.617 | −2.893 | 1.595 | 1.548 | 0.047 |
| 22 | 45 | 1.769 | +4.962 | −5.040 | 1.691 | 1.363 | 0.327 |
| 33 | 33 | 1.787 | +4.582 | −4.744 | 1.625 | 1.421 | 0.204 |

批 0 的 `lambda_dual=1`。current-value 与 next-value 项会大幅抵消，说明显式奖励不等于实际 `A_task`；这本身不是错误。b15/b31 的真实分解因为来源身份冲突而停止，不能外推这张表代表 18 点。

## D：延迟奖励直接传递

冻结公式在 critic 固定时给出 `(gamma × lambda_GAE)^d`，其中 `d` 是训练行距离，full-mask 行也计入 `d`。六个可分析 batch 0 点共有 1,168 个 prior-active-row/reward 对：距离中位数 25 行（四分位 15–38.25，最大 75），环境 tick 距离中位数 30（四分位 17–327，最大 497），直接奖励系数中位数 0.277（范围 0.0213–1）。系数只衡量显式奖励路径；固定 critic 时系数小不等于没有长期信息，critic/bootstrap 可能补偿。

纯计算合成测试 4/4 通过：非零 value 的双步手算、跨 full-mask 行的延迟奖励、非终止截断 bootstrap、真实终止边界阻断下一 episode 的反向泄漏。双步例子的独立结果为 `delta=[0.2,0.4]`、`A=[0.58,0.4]`、`return=[0.98,1.0]`；延迟奖励跨 3 个训练行和 6 个环境 tick 的直接系数为 `0.95^3=0.857375`。测试调用的是从归档 `cost_math.gae` 公式逐式翻译的 JavaScript 路径；当前 shell 无 Python/PyTorch，未能直接导入原生函数。

## E：critic 样本诊断

仅对身份一致的 batch 0 点，用实际存档奖励流、原 gamma/时间单位、终止与 bootstrap 向量计算样本 return。六点的 active 行 `value − return` 偏差为 −1.595 至 −1.691，MAE 为 1.595 至 1.691；full-mask 行偏差为 −0.465 至 −0.500，MAE 为 0.471 至 0.502；首个自由决策行偏差为 −1.354 至 −1.481，MAE 为 1.354 至 1.481。它们是当前采样轨迹的误差，不是期望价值真值或独立泛化证据；而且输入 task reward 已确认延迟一拍，不能拿这些数字为当前 critic 选参数。

checkpoint / 样本 provenance 门禁：

- 18/18 checkpoint 文件存在且 SHA256 与 `CHECKPOINT_INDEX.json` 相符；均未反序列化或 forward replay。
- 当前 batch record 与 tape 的行数、保存的 likelihood gate 样本数 18/18 相符。
- checkpoint 参数哈希与当前批记录、checkpoint lambda 与当前批 `lambda_at_rollout` 只有 6/18 相符（六个 b0）；12 个 b15/b31 两类身份都冲突。
- `UPDATE_MECHANISM.json` 的 A_task/recorded/recomputed 样本数在这 12 点也与当前 batch record/tape 不同。例：`ALL_ROWS_s22/b31` 当前 record/tape 为 1,518 行，而 mechanism 记录 324；未通过改容差处理。

## 范围与停止

- 输入来源：ROUND32 768 条训练 row、768 条原始 tape、lifecycle 日志、批记录、18 个 checkpoint 文件与 index、`UPDATE_MECHANISM.json`、冻结 trainer/GAE/ROUND23 修复资料；另只读核对 16 个旧开发场景。无材料缺失、没有成功/失败筛选，封存块未触碰。
- ROUND32 身份：配置哈希 `e9b2c067db5951f2`，冻结源码哈希 `b5615c1bed8b5686c478df4c92517e69558b986936dcbc026bd96dc82f40b6d2`，冻结头 `df176bdf8c3c265d8109494446073b23aee384a6`；搜索/分配 checkpoint SHA 与完整输入清单记录在 `TEMPORAL_CREDIT_AUDIT.json`。
- ROUND32 中 greedy 与 sampled 必须分开解释。大优势幅度不证明长期任务信用正确；相同的裁剪后梯度范数也不等于相同的 Adam 参数步长。ALL_ROWS/ACTIVE_ROWS 的负结果只削弱“单纯放大 actor surrogate 就足够”的解释。
- 新环境启动 0；新流 0；真实训练更新 0；合成 optimizer 更新 0；checkpoint 写回 0；网络尝试 0。仅运行 Node.js 数据校验/哈希/纯计算和 4 个合成 GAE 测试。
- 不改 gamma、GAE lambda、对偶 lambda、奖励或训练量。未修复源代码，未发布远端。
- **唯一下一步建议（未执行）**：修正 task reward tap，使其在产生 completion 的 relay tick 内记录该 tick 的 ProgressEvent/reward，并增加不构造环境的回归测试验证 `completion at t → task_stream[t]`；通过后再处理 12 个 batch provenance 冲突，再决定是否需要参数实验。

## 交付文件

- `EXECUTION_BASIS.md`：执行提示词原文。
- `TEMPORAL_CREDIT_AUDIT.json`：全训练时间轴、任务/成本归属、目标身份与输入覆盖。
- `GAE_COMPONENTS.json`：固定批分解、直接奖励系数与合成数学核验。
- `CRITIC_DIAGNOSTIC.json`：18 点 provenance 门禁及六个 b0 样本诊断。
- `ROUND33_VERDICT.json`：机读结论、停止原因与执行计数。
- `r33_temporal_credit_audit.js`、`r33_synthetic_tests.js`：只读审计和纯计算测试。
- `SYNTHETIC_GAE_TESTS.json`：手算预期及 4/4 测试结果。
