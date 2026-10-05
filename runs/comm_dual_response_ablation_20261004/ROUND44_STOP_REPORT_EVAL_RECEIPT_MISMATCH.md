# ROUND44 停止报告：首个固定评估的收据序列化不匹配

## 状态

ROUND44 的三 seed 训练已完成。固定评估在 NEW / seed 11 / greedy / 场景 1301000 的第一格结束后，收据校验报错并将该 claim 记为技术失败。流程依照冻结停止条件停止；没有热修、重跑或启动其他评估。原先的 [ROUND44_STOP_REPORT.md](ROUND44_STOP_REPORT.md)、失败文件、任务行和 tape 均保留。

- 科学冻结提交：`2353c4616746cf0a7c9fcb3a519a7042a28ba176`
- 本次执行身份 HEAD：`26b22680b8c3c5cb9bb91bc44799c76cec6f874a`（分支 `exp/comm-dual-response-ablation`）
- 最终配额账本：`runtime/ROUND44_EXECUTION_LEDGER.json`
- 本报告的逐项部分账本：`ROUND44_PARTIAL_EXECUTION_LEDGER.json`

## 训练进度

三个 seed 均完成 128 集、32 批；三个首批真实接线 gate 均通过。

| Seed | Episodes | PPO batches | Adam steps | Final λ | ep128 参数 SHA256 |
|---:|---:|---:|---:|---:|---|
| 11 | 128 | 32 | 348 | 3.5 | `a0e9e60a5f2ed040b9f40963a9c9129c84afcf71b5c39d3cad91ff78a6d2323c` |
| 22 | 128 | 32 | 372 | 4.125 | `ce45ad13794d6d39691dfe872f29fcec83a20b2fd5bfb032eac30aa22b6993fd` |
| 33 | 128 | 32 | 296 | 2.875 | `3af3f52dd0c293b2d71f9d5191d4b9d3f4641a8eb90cba4ee19b93744908c5a2` |
| **合计** | **384** | **96** | **1,016** | — | — |

R4 精确恢复 gate、三 seed 首批 gate/比较结果、96 次 PPO update、1,016 次 Adam step 与 297 条 checkpoint-write journal 记录见部分账本的 SHA256 清单。

## 失败格与持久化证据

执行器运行完成了 1 次评估环境构造和 1,000 步。Row 文件记录 `on_time=6`、`spent_bytes=0`、`task_reward_total=6`。该行的 artifact receipt 未创建，claim 状态为 `failed_technical`，所以该观测没有作为完成的固定评估结果或科学样本纳入结果矩阵。

- Claim：`failed_technical`；normal attempt `1`；环境构造 `1`；run 返回 `1`；完成步数 `1000`。
- 异常：`EvalIndexError: persisted row differs from the completed in-memory row: (11, 'greedy', 1301000)`
- Claim SHA256：`b81d51ea0d79dd4b276167b1708a9982314662eda8166d657f0d6e9d384b94d7`
- Row SHA256：`409e8bd5d6a08a988c34486c1476c89104d341189c8064c988402387787417e5`
- 失败捕获 SHA256：`9c23c46c95da06fec2b7d88f22dbdda7e8c580ecd252733b0923fa7b51a90140`
- Lifecycle / transcript / tape 文件哈希、大小见部分账本。

## 根因定位

当前代码的 JSON round-trip 相等性检查不兼容 Python tuple：

1. 参数摘要的 `heads` 字段由 `D:\code\整合\zhenghe-round40\runs\comm_cost_stage_b_20260927\sb_controller.py:61` 作为 tuple 返回。
2. Row writer 使用 `json.dumps(row, ...)` 写行；JSON 把 tuple 编码成 array。
3. [r44_eval_index.py](r44_eval_index.py#L280) 将文件读成 JSON 对象后，使用 `stored_row != row` 与原始内存对象直接比较。

因此，这格在模拟已经返回之后，于持久化身份收据阶段失败。归类为执行器的机械序列化/簿记错误；该格不进入科学评估矩阵。

## 账本与剩余额度

- Normal claim：`385/480`，其中 `384` 个 logical job complete、1 个 claim 技术失败。失败 claim 不退额。
- Technical reserve：`0/24`。
- 总 claim：`385/504`。
- 已构造环境：`385`；已完成模拟步：`385000`。
- 固定 NEW 评估矩阵：0/96 个完成 claim；第一格失败，95 格未启动。OLD/R43 与规则基线保持只读复用，本轮没有改写它们。
- 汇总分析、配对比较、最终科学判定均未生成。不能据此声称任务保持并节省通信成本，也不能声称实验已完成。

## 停止依据与下一步边界

[EXECUTION_BASIS.md](EXECUTION_BASIS.md#L42) 要求真实运行失败保留原 claim 并停止受影响工作，禁止热修冻结代码。当前执行身份不能安全地直接继续剩余 95 格。若要完成原评估矩阵，需要另行授权：建立独立、版本化的机械收据修复与门禁，并把该逻辑格的再次尝试作为技术储备 attempt 2；保留本 attempt 与所有产物，再运行尚未启动的 95 个固定格。训练权重无需重训。

**当前结果：训练完成，评估与结论未完成；执行已停止等待下一步决定。**
