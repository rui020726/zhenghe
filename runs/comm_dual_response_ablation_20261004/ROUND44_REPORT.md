# ROUND44 报告：对偶响应速度单变量消融

## 结论

是否完成：是。384 集三 seed 训练、96 格固定评估全部完成；正常额度 480/480，技术储备 1/24，总 claim 481/504。
开发块联合判定：`development_block_joint_improvement_not_observed`。该判定要求三个 greedy seed 各自按场景保持 immediate 的按时任务身份集合，并且计费低于 late discovery gate。
本轮只改变每 batch 的 `eta_lambda`：0.05→0.5。结果只检验对偶响应速度假设；即使改善，也不能证明先前学习失败只有一个根因。
所有比较仍位于已使用的16场景开发块，不构成独立确认、总体零损失、正式非劣或泛化结论。

## 固定科研配置

- 训练版本：`constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/dual-eta-lambda-0p5/1/published-completion-event-at-relay-tick/2`。
- 共享 GAE trace=1.0、gamma=1.0、lambda_dual 初值=1.0、delta=0、B0=10,000 B 均保持冻结。
- 每四集 PPO batch 更新一次 dual：`lambda_next=max(0, lambda_before+0.5*g_hat)`；本批 actor 使用更新前 lambda。
- 384 个 NEW 训练 episode（seed 11/22/33 各128集），96 个 NEW ep128 固定评估格。
- 旧比较输入为 ROUND43 NEW ep128；64 immediate、32 late/always_hold 结果只读复用。
- 无新场景、新流、确认块、追加训练或第二个 eta 步长。成本为已计量载荷尝试字节，含丢包；不代表全系统通信或能耗。

## 绝对表现

| Group | On-time Y / mean per scene | Charged C / mean B per scene | First reports / bytes | Free SEND / SKIP |
|---|---:|---:|---:|---:|
| Immediate reference | 96 / 6.0000 | 239,028 / 14939.2500 B | 96 / 239,028 B | 0 / 0 |
| Late discovery gate | 96 / 6.0000 | 77,412 / 4838.2500 B | 31 / 77,412 B | 0 / 0 |
| Always hold | 88 / 5.5000 | 0 / 0.0000 B | 0 / 0 B | 0 / 0 |
| s11 greedy OLD R43 ep128 | 88 / 5.5000 | 0 / 0.0000 B | 0 / 0 B | 0 / 96 |
| s11 greedy NEW R44 ep128 | 88 / 5.5000 | 0 / 0.0000 B | 0 / 0 B | 0 / 96 |
| s11 sampled OLD R43 ep128 | 94 / 5.8750 | 84,870 / 5304.3750 B | 34 / 84,870 B | 34 / 62 |
| s11 sampled NEW R44 ep128 | 94 / 5.8750 | 109,836 / 6864.7500 B | 44 / 109,836 B | 44 / 52 |
| s22 greedy OLD R43 ep128 | 96 / 6.0000 | 239,028 / 14939.2500 B | 96 / 239,028 B | 96 / 0 |
| s22 greedy NEW R44 ep128 | 96 / 6.0000 | 239,028 / 14939.2500 B | 96 / 239,028 B | 96 / 0 |
| s22 sampled OLD R43 ep128 | 95 / 5.9375 | 154,428 / 9651.7500 B | 62 / 154,428 B | 62 / 34 |
| s22 sampled NEW R44 ep128 | 95 / 5.9375 | 164,562 / 10285.1250 B | 66 / 164,562 B | 66 / 30 |
| s33 greedy OLD R43 ep128 | 88 / 5.5000 | 0 / 0.0000 B | 0 / 0 B | 0 / 96 |
| s33 greedy NEW R44 ep128 | 88 / 5.5000 | 0 / 0.0000 B | 0 / 0 B | 0 / 96 |
| s33 sampled OLD R43 ep128 | 93 / 5.8125 | 90,138 / 5633.6250 B | 36 / 90,138 B | 36 / 60 |
| s33 sampled NEW R44 ep128 | 92 / 5.7500 | 60,282 / 3767.6250 B | 24 / 60,282 B | 24 / 72 |

Y 是按时完成任务数；C 是载荷发送尝试的计费字节。均值以16个开发场景计算，显示精度保留1/16分辨率。SEND/SKIP 仅计自由首报选择。

## 配对差与目标身份变化

| Seed | Mode | Comparison (right − left) | ΔY total / mean | ΔC total / mean B | Recovered / lost task IDs |
|---:|---|---|---:|---:|---:|
| 11 | greedy | NEW−OLD R43 | +0 / 0.0000 | +0 / 0.0000 B | 0 / 0 |
| 11 | sampled | NEW−OLD R43 | +0 / 0.0000 | +24,966 / 1560.3750 B | 0 / 0 |
| 22 | greedy | NEW−OLD R43 | +0 / 0.0000 | +0 / 0.0000 B | 0 / 0 |
| 22 | sampled | NEW−OLD R43 | +0 / 0.0000 | +10,134 / 633.3750 B | 0 / 0 |
| 33 | greedy | NEW−OLD R43 | +0 / 0.0000 | +0 / 0.0000 B | 0 / 0 |
| 33 | sampled | NEW−OLD R43 | -1 / -0.0625 | -29,856 / -1866.0000 B | 0 / 1 |
| 11 | greedy | NEW−immediate | -8 / -0.5000 | -239,028 / -14939.2500 B | 0 / 8 |
| 11 | sampled | NEW−immediate | -2 / -0.1250 | -129,192 / -8074.5000 B | 0 / 2 |
| 22 | greedy | NEW−immediate | +0 / 0.0000 | +0 / 0.0000 B | 0 / 0 |
| 22 | sampled | NEW−immediate | -1 / -0.0625 | -74,466 / -4654.1250 B | 0 / 1 |
| 33 | greedy | NEW−immediate | -8 / -0.5000 | -239,028 / -14939.2500 B | 0 / 8 |
| 33 | sampled | NEW−immediate | -4 / -0.2500 | -178,746 / -11171.6250 B | 0 / 4 |
| 11 | greedy | NEW−late_discovery_gate | -8 / -0.5000 | -77,412 / -4838.2500 B | 0 / 8 |
| 11 | greedy | NEW−always_hold | +0 / 0.0000 | +0 / 0.0000 B | 0 / 0 |
| 11 | sampled | NEW−late_discovery_gate | -2 / -0.1250 | +32,424 / 2026.5000 B | 0 / 2 |
| 11 | sampled | NEW−always_hold | +6 / 0.3750 | +109,836 / 6864.7500 B | 6 / 0 |
| 22 | greedy | NEW−late_discovery_gate | +0 / 0.0000 | +161,616 / 10101.0000 B | 0 / 0 |
| 22 | greedy | NEW−always_hold | +8 / 0.5000 | +239,028 / 14939.2500 B | 8 / 0 |
| 22 | sampled | NEW−late_discovery_gate | -1 / -0.0625 | +87,150 / 5446.8750 B | 0 / 1 |
| 22 | sampled | NEW−always_hold | +7 / 0.4375 | +164,562 / 10285.1250 B | 7 / 0 |
| 33 | greedy | NEW−late_discovery_gate | -8 / -0.5000 | -77,412 / -4838.2500 B | 0 / 8 |
| 33 | greedy | NEW−always_hold | +0 / 0.0000 | +0 / 0.0000 B | 0 / 0 |
| 33 | sampled | NEW−late_discovery_gate | -4 / -0.2500 | -17,130 / -1070.6250 B | 0 / 4 |
| 33 | sampled | NEW−always_hold | +4 / 0.2500 | +60,282 / 3767.6250 B | 4 / 0 |

完整逐场景 recovered/lost 任务 ID、OLD/NEW 原始汇总和规则配对保存在 `ROUND44_PAIRED_RESULTS.json`；表格和机器判定由同一分析对象生成。
没有预登记可用的置信区间，本轮不新增区间或等效性解释。

## 96格索引、attempt lineage 与执行完整性复核

- 收尾时只读重核 `audit_fixed_matrix`，结果与持久化审计一致：`ROUND44_EVAL_MATRIX_AUDIT.json` 状态为 passed，96 个唯一评估键；seed×mode 六组各16格。分析入口 `r44_analysis.py:160` 只从 `eval_audit["entries"]` 读取 NEW 行，未从 rows 目录 glob 取数。六组 NEW 汇总独立重算后与 `ROUND44_PAIRED_RESULTS.json` 一致。
- 重试键 `(11, greedy, 1301000)` 在科研索引中只有 attempt 2：row SHA256 `76d8e3fb4896ed9464cbc4d3b5608c1b25294ce3305c22b0cbbf4a481a04164c`，claim SHA256 `9acc52b91428fce0c9ccc7478b7df43a324b6664cac11918e14caf23ccb70eda`。失败的 attempt 1 行仍保留，SHA256 `409e8bd5d6a08a988c34486c1476c89104d341189c8064c988402387787417e5`，且不在96格索引内；科学比较只使用 attempt 2。完整逐场景任务 ID 差异仍见 paired results。
- 评估前基线采用 `runtime/ROUND44_EVAL_INPUT_IMMUTABLE_MANIFEST_R1.json`（SHA256 `b3bb6f7c2488b2acb03f467c799b6eb74554795cf9041eb30529031774a3ebeb`），按该清单中的实际路径逐项比对：204个训练 checkpoint 文件均存在，大小及 SHA256 全部匹配；缺失0、哈希/大小差异0。三份训练计数日志也与清单基线完全匹配：`optimizer_step_calls.jsonl` 1016条（SHA256 `7d47c9a208160ac0ef943663eb7b04f21bf1ca91072dc8ec619c7e5395771bd5`），`ppo_update_calls.jsonl` 96条（`6f653bcb1bdd55ef570322ea1ac7d719e103ff798f5483942abf464ed2df93a0`），`checkpoint_writes.jsonl` 297条（`7957ec093b345da3b0a1488e0fb09641ce437fd45d28d23ac2436ea793718bdd`）。这是评估前后的不可变性证据；评估后生成的 raw artifact manifest 未被用作评估前哈希基线。
- 执行身份链已修正并保留失败证据：队列游标修复提交为 `9c7fb0bc61da2ca925ff51ff8658c166cf5e2ba4`。首次 R3 身份提交 `a7e2f86392d31346a9bcfcc4cd8fe86b7068d066` 因绑定了换行规范化前的证据哈希，在任何 claim 前被 `verify_eval_identity` 拒绝；未启动环境。修订源提交 `c918259ee5d8c862f501fbdb30ca948fc05446d8` 绑定已提交 LF blob 哈希，修正身份提交 `09d758ccd16d3bbeaca525dee97155d391fb1404` 在评估前通过验证。该修正仅覆盖评估队列游标和执行身份证据。
- 执行完整性与科研判定分开：固定矩阵已完成（480个正常 claim、1个 reserve claim、96格评估），没有新增运行失败；原 attempt 1 技术失败仍计入账本。科研联合改善门槛未通过，结果仍为 `development_block_joint_improvement_not_observed`，不构成独立确认或泛化结论。

## 每批对偶轨迹与训练完整性

逐 seed 的32批 `g_hat`、dual 更新前后值、任务/成本、PPO 指标、likelihood 与参数链见 `ROUND44_TRAINING_CURVES.json`。
- seed 11: 128集/32批，Adam.step=348；dual 1.000000→3.500000，首批 g=+0.250000、eta=0.50，末批 g=+0.250000。
- seed 22: 128集/32批，Adam.step=372；dual 1.000000→4.125000，首批 g=+0.000000、eta=0.50，末批 g=+0.000000。
- seed 33: 128集/32批，Adam.step=296；dual 1.000000→2.875000，首批 g=+0.000000、eta=0.50，末批 g=+0.500000。

首批门禁：3/3 seed 通过实际 rollout→tape→batch_samples/GAE→actor/双 critic→PPO/dual→checkpoint 路径；逐批记录证明每 seed 恰有32次更新。R43/R44 首批 actor 与模型/optimizer 结果比较见 `ROUND44_FIRST_BATCH_COMPARISON.json`。
checkpoint ep0/64/128 的文件、参数、模块、optimizer、RNG 和状态游标见 `ROUND44_CHECKPOINT_INDEX.json`。旧 R43 ep128 只读使用。

## 额度对账

- normal claims：480；reserve claims：1；总计：481；硬上限：504。
- 构造调用：481；JointRun 构造：481；run 调用：481；完整步数：481000。
- PPO 更新：96；Adam.step：1016；checkpoint 保存调用：297。
- 评估独立索引复核：passed，96格，每个 seed×mode 16格。训练/执行失败与重试明细见 `ROUND44_EXECUTION_LEDGER_FINAL.json`。
- 评估索引纯计算测试：passed，0环境、0 optimizer.step；训练前 dual/版本/初值门禁：0环境、0实验 checkpoint 写入。
- 审计原始文件数：3452；原始 rows/logs/checkpoints 留在本机 Git 外运行目录。

## 身份与解释限制

科学执行冻结 HEAD：`09d758ccd16d3bbeaca525dee97155d391fb1404`。本报告代码在该冻结中已锁定；最终报告 SHA256 登记在 `ROUND44_STATUS_COMPLETE_EVAL_R1.json`。
判读按每 seed greedy 主结果、sampled 辅结果分列；不池化训练 seed、不挑 checkpoint/decoder。
全 SEND 最多说明任务恢复，若成本没有低于 late 仍不满足联合目标；全 SKIP 丢任务不算降本成功。
本轮结果仅作开发块探索性观察。本地提交后停止；未推送、部署、扩样、确认或续训。
