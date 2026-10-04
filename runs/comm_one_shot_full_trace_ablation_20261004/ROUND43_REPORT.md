# ROUND43 报告：一次性首报选择的完整 GAE trace 消融

## 结论

是否完成：是。三 seed 固定训练和固定评估均完成；总计 480 次正常启动、0 次技术储备，硬上限 600。

任务保持是否改善：否。三个 greedy seed 均保持 immediate 的开发块按时任务身份集合：否。

是否优于强规则：未观察到满足预登记联合改善条件的结果。条件为三个 greedy seed 各自保持 immediate 按时任务集合，且计费低于 late_discovery_gate。

跨 seed 是否一致：否。判定只针对已用于开发的16个场景，不代表正式非劣、总体零损失或泛化。

训练、评估和分析矩阵闭合后，原执行命令以退出码1停止在报告渲染阶段：报告器读取了不存在的顶层 `exception`，而保留的 preflight 记录把异常放在 `failure`，保留审计放在 `post_failure_read_only_audit`。这是报告接缝故障；没有失败科学格。以下收尾只读已有结果，不重进 `r43_execute.py`、不重建环境、不更新模型。原执行 progress 继续保留 `running`，本地交付完成态另记在 `ROUND43_STATUS.json`。

## 研究变量与边界

唯一研究变量为共享 `gae_lambda` 从 0.95 改为 1.0；`gamma=1.0`、`lambda_dual` 初值 1.0、`eta_lambda=0.05`、`delta=0`、`B0=10000 B` 均保持冻结。task 与 positive-cost GAE 共用新 trace 系数，组合仍为 `lambda_dual*A_task - A_positive_cost`。这是一项方法消融，不是修复旧 GAE 数学错误；GAE=1 同时改变 `A+V` critic target，且可能增大方差。本结果不能证明 ROUND42 的学习失败只由信用衰减导致。

## Greedy 主结果

- seed 11：目标集合与 immediate 完全一致=否；NEW 88 个按时任务 / 0 B，late 96 个 / 77,412 B；低于 late=是。
- seed 22：目标集合与 immediate 完全一致=是；NEW 96 个按时任务 / 239,028 B，late 96 个 / 77,412 B；低于 late=否。
- seed 33：目标集合与 immediate 完全一致=否；NEW 88 个按时任务 / 0 B，late 96 个 / 77,412 B；低于 late=是。

联合判定：`development_block_joint_improvement_not_observed`。

## Sampled 辅助结果

- seed 11：NEW 按时任务 94，计费 84,870 B，首报 SEND 34 / SKIP 62（自由选择 96）；OLD 按时任务 94，计费 94,710 B，首报 SEND 38 / SKIP 58（自由选择 96）。
- seed 22：NEW 按时任务 95，计费 154,428 B，首报 SEND 62 / SKIP 34（自由选择 96）；OLD 按时任务 93，计费 55,896 B，首报 SEND 22 / SKIP 74（自由选择 96）。
- seed 33：NEW 按时任务 93，计费 90,138 B，首报 SEND 36 / SKIP 60（自由选择 96）；OLD 按时任务 92，计费 50,250 B，首报 SEND 20 / SKIP 76（自由选择 96）。

完整逐 seed/mode 的 NEW−OLD、NEW−immediate、NEW−late、NEW−always_hold 配对结果和逐目标恢复/丢失列表见 `ROUND43_PAIRED_RESULTS.json`。绝对任务、载荷尝试字节、首报 SEND/SKIP、交付和丢包统计也在其中。

## 所有固定组的绝对结果

`Y` 是按时完成任务数，表中先给16场景总数，再给每集均值；`C` 是载荷发送尝试的计费字节（含丢包），先给总数，再给每集均值。首报列为发送次数/相应首报字节；SEND/SKIP 列为自由首报选择数。丢包仍按尝试字节计入 C。

| 组 | 按时 Y 总数 / 每集均值 | 计费 C 总数 / 每集均值 | 首报发射 / 首报字节 | 自由 SEND / SKIP | 丢包 |
|---|---:|---:|---:|---:|---:|
| immediate_reference_development | 96 / 6.000 | 239,028 / 14,939.25 B | 96 / 239,028 B | 0 / 0 | 175 |
| late_discovery_gate | 96 / 6.000 | 77,412 / 4,838.25 B | 31 / 77,412 B | 0 / 0 | 61 |
| always_hold | 88 / 5.500 | 0 / 0.00 B | 0 / 0 B | 0 / 0 | 0 |
| s11_greedy_OLD_round42_new_ep128 | 88 / 5.500 | 0 / 0.00 B | 0 / 0 B | 0 / 96 | 0 |
| s11_greedy_NEW_round43_ep128 | 88 / 5.500 | 0 / 0.00 B | 0 / 0 B | 0 / 96 | 0 |
| s11_sampled_OLD_round42_new_ep128 | 94 / 5.875 | 94,710 / 5,919.38 B | 38 / 94,710 B | 38 / 58 | 64 |
| s11_sampled_NEW_round43_ep128 | 94 / 5.875 | 84,870 / 5,304.38 B | 34 / 84,870 B | 34 / 62 | 56 |
| s22_greedy_OLD_round42_new_ep128 | 88 / 5.500 | 0 / 0.00 B | 0 / 0 B | 0 / 96 | 0 |
| s22_greedy_NEW_round43_ep128 | 96 / 6.000 | 239,028 / 14,939.25 B | 96 / 239,028 B | 96 / 0 | 175 |
| s22_sampled_OLD_round42_new_ep128 | 93 / 5.812 | 55,896 / 3,493.50 B | 22 / 55,896 B | 22 / 74 | 36 |
| s22_sampled_NEW_round43_ep128 | 95 / 5.938 | 154,428 / 9,651.75 B | 62 / 154,428 B | 62 / 34 | 107 |
| s33_greedy_OLD_round42_new_ep128 | 88 / 5.500 | 0 / 0.00 B | 0 / 0 B | 0 / 96 | 0 |
| s33_greedy_NEW_round43_ep128 | 88 / 5.500 | 0 / 0.00 B | 0 / 0 B | 0 / 96 | 0 |
| s33_sampled_OLD_round42_new_ep128 | 92 / 5.750 | 50,250 / 3,140.62 B | 20 / 50,250 B | 20 / 76 | 36 |
| s33_sampled_NEW_round43_ep128 | 93 / 5.812 | 90,138 / 5,633.62 B | 36 / 90,138 B | 36 / 60 | 65 |

## 按场景配对差和目标身份变化

方向统一为 `NEW − 对照`；ΔY、ΔC 同时列16场景总差与每集均值。救回/丢失集合按 `scene:task_id` 列出；`∅` 表示集合为空，不用净差替代目标身份核对。

| Seed | 模式 | 对照（NEW−对照） | ΔY 总数 / 每集 | ΔC 总数 / 每集 | NEW救回目标 | NEW丢失目标 |
|---:|---|---|---:|---:|---|---|
| 11 | greedy | OLD_round42_new_ep128 | +0 / +0.000 | +0 / +0.00 B | ∅ | ∅ |
| 11 | greedy | immediate | -8 / -0.500 | -239,028 / -14,939.25 B | ∅ | 1301001:piloc-e1-target-5, 1301002:piloc-e1-target-1, 1301002:piloc-e1-target-2, 1301002:piloc-e1-target-3, 1301002:piloc-e1-target-5, 1301003:piloc-e1-target-5, 1301013:piloc-e1-target-2, 1301015:piloc-e1-target-0 |
| 11 | greedy | late_discovery_gate | -8 / -0.500 | -77,412 / -4,838.25 B | ∅ | 1301001:piloc-e1-target-5, 1301002:piloc-e1-target-1, 1301002:piloc-e1-target-2, 1301002:piloc-e1-target-3, 1301002:piloc-e1-target-5, 1301003:piloc-e1-target-5, 1301013:piloc-e1-target-2, 1301015:piloc-e1-target-0 |
| 11 | greedy | always_hold | +0 / +0.000 | +0 / +0.00 B | ∅ | ∅ |
| 11 | sampled | OLD_round42_new_ep128 | +0 / +0.000 | -9,840 / -615.00 B | ∅ | ∅ |
| 11 | sampled | immediate | -2 / -0.125 | -154,158 / -9,634.88 B | ∅ | 1301001:piloc-e1-target-5, 1301013:piloc-e1-target-2 |
| 11 | sampled | late_discovery_gate | -2 / -0.125 | +7,458 / +466.12 B | ∅ | 1301001:piloc-e1-target-5, 1301013:piloc-e1-target-2 |
| 11 | sampled | always_hold | +6 / +0.375 | +84,870 / +5,304.38 B | 1301002:piloc-e1-target-1, 1301002:piloc-e1-target-2, 1301002:piloc-e1-target-3, 1301002:piloc-e1-target-5, 1301003:piloc-e1-target-5, 1301015:piloc-e1-target-0 | ∅ |
| 22 | greedy | OLD_round42_new_ep128 | +8 / +0.500 | +239,028 / +14,939.25 B | 1301001:piloc-e1-target-5, 1301002:piloc-e1-target-1, 1301002:piloc-e1-target-2, 1301002:piloc-e1-target-3, 1301002:piloc-e1-target-5, 1301003:piloc-e1-target-5, 1301013:piloc-e1-target-2, 1301015:piloc-e1-target-0 | ∅ |
| 22 | greedy | immediate | +0 / +0.000 | +0 / +0.00 B | ∅ | ∅ |
| 22 | greedy | late_discovery_gate | +0 / +0.000 | +161,616 / +10,101.00 B | ∅ | ∅ |
| 22 | greedy | always_hold | +8 / +0.500 | +239,028 / +14,939.25 B | 1301001:piloc-e1-target-5, 1301002:piloc-e1-target-1, 1301002:piloc-e1-target-2, 1301002:piloc-e1-target-3, 1301002:piloc-e1-target-5, 1301003:piloc-e1-target-5, 1301013:piloc-e1-target-2, 1301015:piloc-e1-target-0 | ∅ |
| 22 | sampled | OLD_round42_new_ep128 | +2 / +0.125 | +98,532 / +6,158.25 B | 1301001:piloc-e1-target-5, 1301015:piloc-e1-target-0 | ∅ |
| 22 | sampled | immediate | -1 / -0.062 | -84,600 / -5,287.50 B | ∅ | 1301002:piloc-e1-target-3 |
| 22 | sampled | late_discovery_gate | -1 / -0.062 | +77,016 / +4,813.50 B | ∅ | 1301002:piloc-e1-target-3 |
| 22 | sampled | always_hold | +7 / +0.438 | +154,428 / +9,651.75 B | 1301001:piloc-e1-target-5, 1301002:piloc-e1-target-1, 1301002:piloc-e1-target-2, 1301002:piloc-e1-target-5, 1301003:piloc-e1-target-5, 1301013:piloc-e1-target-2, 1301015:piloc-e1-target-0 | ∅ |
| 33 | greedy | OLD_round42_new_ep128 | +0 / +0.000 | +0 / +0.00 B | ∅ | ∅ |
| 33 | greedy | immediate | -8 / -0.500 | -239,028 / -14,939.25 B | ∅ | 1301001:piloc-e1-target-5, 1301002:piloc-e1-target-1, 1301002:piloc-e1-target-2, 1301002:piloc-e1-target-3, 1301002:piloc-e1-target-5, 1301003:piloc-e1-target-5, 1301013:piloc-e1-target-2, 1301015:piloc-e1-target-0 |
| 33 | greedy | late_discovery_gate | -8 / -0.500 | -77,412 / -4,838.25 B | ∅ | 1301001:piloc-e1-target-5, 1301002:piloc-e1-target-1, 1301002:piloc-e1-target-2, 1301002:piloc-e1-target-3, 1301002:piloc-e1-target-5, 1301003:piloc-e1-target-5, 1301013:piloc-e1-target-2, 1301015:piloc-e1-target-0 |
| 33 | greedy | always_hold | +0 / +0.000 | +0 / +0.00 B | ∅ | ∅ |
| 33 | sampled | OLD_round42_new_ep128 | +1 / +0.062 | +39,888 / +2,493.00 B | 1301001:piloc-e1-target-5 | ∅ |
| 33 | sampled | immediate | -3 / -0.188 | -148,890 / -9,305.62 B | ∅ | 1301003:piloc-e1-target-5, 1301013:piloc-e1-target-2, 1301015:piloc-e1-target-0 |
| 33 | sampled | late_discovery_gate | -3 / -0.188 | +12,726 / +795.38 B | ∅ | 1301003:piloc-e1-target-5, 1301013:piloc-e1-target-2, 1301015:piloc-e1-target-0 |
| 33 | sampled | always_hold | +5 / +0.312 | +90,138 / +5,633.62 B | 1301001:piloc-e1-target-5, 1301002:piloc-e1-target-1, 1301002:piloc-e1-target-2, 1301002:piloc-e1-target-3, 1301002:piloc-e1-target-5 | ∅ |

## λ_dual 完整轨迹

每行按32个 batch 的 `lambda_next` 顺序列出；起点均为 λ=1.0000，batch 更新规则为冻结的 `eta_lambda=0.05`。

| Seed | Batch λ_next（按 batch 0→31；共32次） |
|---:|---|
| 11 | `1.0125 → 1.0125 → 1.0375 → 1.0625 → 1.0625 → 1.0750 → 1.0750 → 1.0750 → 1.0750 → 1.1000 → 1.1250 → 1.1250 → 1.1375 → 1.1500 → 1.1500 → 1.1625 → 1.1625 → 1.1625 → 1.1625 → 1.1625 → 1.1625 → 1.1625 → 1.1750 → 1.1750 → 1.1750 → 1.2000 → 1.2000 → 1.2000 → 1.2125 → 1.2500 → 1.2625 → 1.2750` |
| 22 | `1.0000 → 1.0125 → 1.0250 → 1.0500 → 1.0500 → 1.0750 → 1.0750 → 1.0750 → 1.0750 → 1.1000 → 1.1250 → 1.1250 → 1.1375 → 1.1375 → 1.1375 → 1.1625 → 1.1750 → 1.1750 → 1.1750 → 1.1750 → 1.1875 → 1.2000 → 1.2125 → 1.2500 → 1.2500 → 1.2750 → 1.2750 → 1.2750 → 1.2875 → 1.3125 → 1.3250 → 1.3250` |
| 33 | `1.0000 → 1.0125 → 1.0125 → 1.0250 → 1.0250 → 1.0500 → 1.0500 → 1.0500 → 1.0500 → 1.0500 → 1.0500 → 1.0500 → 1.0500 → 1.0625 → 1.0875 → 1.0875 → 1.1000 → 1.1125 → 1.1125 → 1.1125 → 1.1250 → 1.1250 → 1.1375 → 1.1375 → 1.1375 → 1.1625 → 1.1625 → 1.1625 → 1.1625 → 1.1625 → 1.1625 → 1.1875` |

## 训练和路径完整性

- seed 11：episodes=128、batches=32、optimizer steps=352、λ_dual=1.275000、ep128 参数 SHA256=`40b8e0821b16b6376dff2ee2cb7dea6409490ddf0b6cb466bbd6a6730764553f`。
- seed 22：episodes=128、batches=32、optimizer steps=376、λ_dual=1.325000、ep128 参数 SHA256=`e683dcd0b0a9f3d1cd743f153821ee31a41deb6135d3017d9ac33ff22e16e2c1`。
- seed 33：episodes=128、batches=32、optimizer steps=288、λ_dual=1.187500、ep128 参数 SHA256=`429b2c0ac7bb80c335cedf39fd980a4205ca74804fe35eaa2be491a08cf02757`。

- 训练曲线和96个批记录：`ROUND43_TRAINING_CURVES.json`；每批含 `g_hat`、`lambda_dual` 更新、PPO指标、梯度摘要、场景和episode行身份。
- ep0/64/128 checkpoint 字节哈希、参数及模块摘要、optimizer/RNG摘要：`ROUND43_CHECKPOINT_INDEX.json`。
- 三个 seed 各32个批的参数链连续；每个 ep128 参数摘要与最后一批更新后状态核对。
- 首批真实 PPO 接线门禁：`ROUND43_FIRST_BATCH_GATE_SEED11.json`、`ROUND43_FIRST_BATCH_GATE_SEED22.json`、`ROUND43_FIRST_BATCH_GATE_SEED33.json`。
- 执行账本和逐 claim 计数：`runtime/ROUND43_EXECUTION_LEDGER.json`；完成逻辑任务=480，真实完成步=480000。

启动与更新对账：{"normal_used": 480, "reserve_used": 0, "total_used": 480, "environment_constructor_calls": 480, "joint_run_constructed_calls": 480, "run_invocation_calls": 480, "joint_run_method_calls": 480, "run_return_calls": 480, "run_failure_calls": 0, "completed_steps": 480000, "completed_logical_jobs": 480, "technical_failed_claims": 0, "retry_claims": 0, "ppo_update_calls": 96, "ppo_update_journal_entries": 96, "optimizer_step_calls": 1016, "optimizer_step_journal_entries": 1016, "checkpoint_write_calls": 297, "hard_cap": 600}。

## 恢复与计数勘误

首次冻结 `c7bc89d995c1a5e5c2b53303bebd036e85aa4c2d` 下，seed11 已完成 episode0–3 并提交 batch0。继续运行时在下一次 episode claim 前，冻结 ROUND40 训练入口因 `parameter_identity` 局部导入作用域错误抛出 `UnboundLocalError`。这次是运行器缺陷：新增 claim、环境构造、run 调用和优化更新均为0，不作为零任务/零成本 episode，也不计入480个正常任务。

恢复门禁复核了 after_b0 与 rolling 状态、完整模型与 Adam 状态、12个 Adam step、λ与历史、训练版本、游标、RNG 和剩余场景顺序。ROUND42 验证过的适配器只把 `parameter_identity` 导入移到训练入口作用域；训练语义版本保持 `constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/published-completion-event-at-relay-tick/2`。恢复从 episode4、update1、Adam step12、λ=1.0125 开始，首批没有重放。恢复冻结 `7ed1a03c445fa31fe90fd70a224a97b8366fa182` 使用执行代码身份 `round43-resume-adapter-and-ledger-reconciliation/1`。

第一次 recovery preflight 因旧累计计数断言而停止：它把历史 Adam 12次调用误判为本次门禁调用。该次无 claim、无环境构造；测试内的伪优化器只到达立即抛错的更新边界，真实模型、优化器与 checkpoint 未变。断言现改为比较门禁前后的累计计数；失败记录与产物哈希核验见 `failures/ROUND43_PREFLIGHT_COUNTER_ASSERTION_C8.json`。

停机时原始 ledger 的快照把 PPO/Adam 计数记为0、checkpoint 保存调用记为2。该旧快照作为原件保留；批提交、逐调用日志及 checkpoint 状态核验出的累计值分别为 PPO 1、Adam 12、checkpoint 保存调用4。checkpoint 保存调用依据调用日志，不由文件数量推算。精确证据和哈希见 `ROUND43_RESUME_RECONCILIATION.json`、`ROUND43_LEDGER_RECONCILIATION.json` 与 `ROUND43_SEED11_BATCH0_STATE_RECONCILIATION.json`。

一次 recovery preflight 因旧累计计数断言产生的历史失败记录也保留。真实 optimizer 更新=0，合成 optimizer 更新=0；测试曾记录1次立即失败的合成 step 尝试。当前报告按已保存的嵌套 `failure` 与 `post_failure_read_only_audit` 字段读取它，完整文案、观察计数和原件 SHA 见 `ROUND43_RESUME_RECONCILIATION.json` 与 `ROUND43_REPORT_FAILURE_CORRECTION.json`。字段兼容没有改写失败记录，也没有把它改成通过。

会话/usage 中断提示与实验进程故障分开记录。当前可核实的执行命令退出码1来自报告器 `KeyError: 'exception'`；不是仿真、训练或评估失败。报告终结期间新增 claim、环境构造、run、PPO、optimizer 更新和 checkpoint 写入均为0；execution ledger 与原 `running` progress 哈希前后保持不变。

## 初值与随机状态

NEW 均由同 seed fresh controller、空 Adam 和 `lambda_dual=1.0` 开始。每 seed 的模型参数哈希与 ROUND42 对应 ep0 相同；optimizer state、cursor=0、updates=0、optimizer_steps=0、lambda 初值相符。ROUND42 ep0 checkpoint 只读提供登记的 RNG，不提供模型权重或 optimizer 状态。

门禁发现当前进程 fresh-controller 构造后的 Python/NumPy 全局状态在部分 seed 上与旧进程捕获值不同；runner 在 controller 构造之后、首次 rollout 之前，通过实际 fresh-controller 接线恢复登记的 ep0 RNG。恢复后的 Python、NumPy、Torch 状态均逐项与相应 ep0 checkpoint 一致。环境每 episode 按原 `scenario_seed` 构造并 reseed 其自身状态；通信动作仍使用原逐集 SHA256 action seed；PPO minibatch 顺序仍用 `DATA_ORDER_SEED:update_index` 的独立 `random.Random`；场景顺序 SHA256 与 ROUND42 一致。这个核验支持相同的起始 RNG 状态，不代表三 seed 轨迹在参数更新后仍逐位配对。

细项见 `ROUND43_EP0_INITIALIZATION.json`、`ROUND43_PREFLIGHT.json` 和 `ROUND43_CONFIGURATION_DIFF.json`。

## 版本与旧结果勘误

ROUND43 登记新训练版本 `constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/published-completion-event-at-relay-tick/2`。实际恢复入口拒绝旧 ROUND42 已训练 checkpoint 作为 ROUND43 训练恢复源；旧模型仍允许带身份信息只读加载。ROUND42 的原始报告正文把 greedy NEW 全 SKIP 写成“两 seed”；逐行原件和配对表确认是三个 seed。勘误保存在 `ROUND42_REPORT_CORRECTION.json`，未改写 ROUND42 原报告、模型、tape 或日志。

## 基线复用与数据范围

本轮复用 64 个 immediate 行、32 个 late/always_hold 规则行，以及 ROUND42 NEW ep128 的96个 OLD评估行；三类均已校验原 manifest 哈希和对应身份。ROUND43 新增384个训练episode、96个 ep128 NEW评估。参考源、checkpoint及文件哈希见 `ROUND43_BASELINE_REUSE.json`、`ROUND43_SOURCE_INVENTORY.json` 与 `ROUND43_RAW_ARTIFACT_MANIFEST.json`。原始权重、tape、环境日志保留在本地 run 目录，没有写入报告仓库内容之外的文件。

## 预启动与复现身份

- Branch：`exp/comm-one-shot-full-trace-ablation`
- 预启动执行源码 HEAD：`7ed1a03c445fa31fe90fd70a224a97b8366fa182`
- 报告终结器 HEAD：`3fc1e5814cdb99d22caa53cb5dc8a8cb37c1841d`；`r43_report.py` SHA256：`9964f87695b00f5aebc19896ea9aab62a9411d3c8febdadf918cd4b96562f8a1`（与执行源码身份分列）
- Python：`D:\project_env\python.exe`；实验设备：CPU
- Runtime：Python 3.10.19、NumPy 2.2.6、PyTorch 2.9.0+cu128（CUDA 可用性=True，本轮运行固定 CPU）
- 执行依据 SHA256：`711e61495d9d7883cd9d636f8f1cf7b50e1336b0890245bc41a54ea370b7be89`
- 配置差异 SHA256：`6f149a2af31a816aef90b273d701b46ef9addd8abece2e3a3f2e40721e59e8d1`
- 原始产物清单：`ROUND43_RAW_ARTIFACT_MANIFEST.json`（3391个文件，SHA256 `2be5ddcc6a7c5282add650830b02a488df25766c58867eaea4d1e8eb2da4108e`）
- 无环境正确性门禁：通过，`draft_mode=false`；记录在 `ROUND43_PREFLIGHT.json`。
- 每个 seed 首四集使用真实 batch_samples/GAE/actor/恢复路径；通过后完成其余124集。
- 固定命令：`D:\project_env\python.exe runs/comm_one_shot_full_trace_ablation_20261004/r43_execute.py`。

## 终结阶段门禁

- 96 个 NEW 格按精确 `(seed, mode, scene)` 清单复核：每格1个 complete normal claim、1行、1条 tape、1份 lifecycle、1份 transcript；共480个 claim/row/linked-artifact 文件匹配原始 SHA 清单，且 ep128 参数身份逐 seed 闭合。
- 训练项384集、96批、三 seed 各128集/32批；ep0/64/128 checkpoint 身份与本地文件哈希复核，三份首批门禁均通过。
- 本次 report-only 阶段 checkpoint deserialization/forward=0。此前 analysis 已生成 `ROUND43_CHECKPOINT_INDEX.json`，含9个里程碑的只读身份核验。
- 账本累计：480正常启动、0 reserve、480构造、480 run、480000步、0运行失败；PPO 96、Adam step 1016、checkpoint 写入297。基线复用的64 immediate、32规则和96 OLD评估不计入这480次新启动。
- 完成态和报告故障更正见 `ROUND43_STATUS.json`、`ROUND43_REPORT_FAILURE_CORRECTION.json`。历史 `runtime/ROUND43_EXECUTION_PROGRESS.json` 保留 `running` 原值。

## 限制

结果来自已用于开发的固定16场景。没有独立确认实验、额外训练、调参、hazard decoder 或补充场景。通信成本仅指载荷尝试字节，丢包仍计费，不代表全系统通信或能耗。ROUND42 原始模型、tape、日志和报告均未覆盖。
