# Stage B 判定：有界真实接线验证（不训练）

日期：2026-09-27。分支 `feat/comm-cost-stage-b`，起点 `5d0d71f4f779fb36982c468f34e7d801f1bc5de8`
（Stage A1 交付头），执行依据 `specs/ROUND21_COMM_COST_STAGE_B_INTEGRATION.md`，
协议冻结件 `STAGE_B_PROTOCOL.md` + `CONFIG_AND_MODEL_MANIFEST.json`。

## 结论：**接线验证通过（wiring verified）**，本轮停止

18/24 次启动全部成功（0 技术失败、0 储备消耗）；B1–B4 五组门禁在三 seed 上
**15/15 passed**（`GATE_RECHECK.json`）。这不是学习结果：**没有训练、没有权重更新、
没有学习收益或任务非劣结论**。C 阶段未启动，δ、降本门槛 ρ、归一化选择、训练量与
700/1600 上限**仍未批准**。

## 启动记账（`START_LEDGER.json`）

| 项 | 值 |
|---|---|
| 正常启动 | **18**（六路径 × 三 seed，逐进程台账求和） |
| 储备消耗 | **0** |
| 硬上限 | 24（未触顶） |
| 技术失败 | 0 |
| optimizer.step | **0**（无 optimizer 对象；逐臂前后参数哈希相等） |

## 六路径 × 三 seed 实测（`rows/sb_<path>_s<seed>_a1.json`）

| seed | R_send | I_send | R_hold | I_hold | P_init | P_repeat |
|---|---|---|---|---|---|---|
| 1100000 | 6 按时 / 15 132 B | 同左 | 6 按时 / **0 B** | 同左 | 6 按时 / 28 848 B | 同左 |
| 1100001 | 6 按时 / 14 766 B | 同左 | 5 按时 / **0 B** | 同左 | 5 按时 / 28 638 B | 同左 |
| 1100002 | 6 按时 / 14 526 B | 同左 | 6 按时 / **0 B** | 同左 | 6 按时 / 29 286 B | 同左 |

## 门禁明细

**B1 原生/注入行为等价（passed × 6 对）**：R_send↔I_send、R_hold↔I_hold 的
emission 序列（step/target/revision/cost_bytes/subscribers/first）、refusal 序列、
relay 全 counters、spent/airtime、任务结局、lifecycle 行为哈希 `behaviour_sha256`
逐项相等；差异只有预先列明的 artefact 路径与 run 标签两类元数据。两条 hold 路径
受控上行尝试与费用均为 0，ACK 各计数器为**结构性零**（`ack_enabled=false`，
不是缺字段填 0）。hold 路径仍经免费旁路完成多数任务（s1100001 为 5/6）——
这正是"下降本不一定降效果"的旁路事实，按协议不要求任务结果为 0。

**B2 新 actor 的真实动作与 mask（passed × 3）**：逐步 tape 与执行器转录闭合——
requested == emitted + refused（顺序一致）、每包计费 == relay 自己的
`attempt_charge_bytes`、shadow gate 信用walk 与 tape 记账一致；masked 候选零似然
零请求、forced 集为空（policy_owned_timing 且无 bound）；logged log_prob 与
相同固定初值权重从 tape features 重算的最大差 **1.19e-07**（≤ 既有 1e-6 门槛，
差距来自 float32 特征往返）；**策略零预算拒付**（`budget_refused_sends=0`），
即从未请求不可付的发送。三 seed 的决策步数 65 / 857 / 57（候选在视图里的持续
时间不同），masked 115 / 854 / 94，actionable 29 / 35 / 43。

**B3 真实费用与任务 return（passed × 9）**：每 episode Σ 逐步空气字节 ==
`spent_bytes` == 转录 Σ cost_bytes == relay `airtime_bytes`；归一化成本 == 字节/B0_test；
Σ 任务 reward == 同窗口 `window_metrics.on_time`；逐包 cost == payload × subscribers、
无重复结算。GAE、双 critic 读数与 `λ_test·A_task − A_cost` 的独立离线重算与行文件
最大差 ≤ 1e-6。配对 `g_hat = Y_ref − Y_π − δ_test`（Y_ref = 同 seed R_send）：
s1100000/1100002 为 0，**s1100001 为 1.0**（R_send 6 vs hold/P_init 5——
该 seed 上静默臂的配对损失可见）；hypothetical `λ_next = max(0, λ_test + η_test·ĝ)`
只入记录，**不写回、不影响任何后续决策**。

**B4 重复性与权重冻结（passed × 3）**：P_init/P_repeat 同初值、同采样 seed、同配置、
同 RNG ⇒ emission 序列、逐步成本流、任务流、任务结局、lifecycle 行为哈希逐位相同，
tape 的 features/actions/log_prob/value/cost_value 完全相同；actor 与双 critic
运行前后参数哈希相等且 P_init/P_repeat 之间相等；optimizer.step = 0。

## 接线常数（仅是算例，非批准）

B0_test = 10 000 B、λ_test = 1、δ_test = 0、η_test = 1、γ = 1、gae_lambda = 0.95、
actor 初值 seed = 20260927、采样 seed = 20260928。按 spec §3，这些**只用于可重算的
接线验证**，不批准为训练超参数或任务损失容忍度；逐集同时保存了原始字节/任务数
（换归一化无需重跑环境）。

## 不得据本轮声称

- 不得声称学习收益、成本学习有效、已满足 δ、胜过规则——本轮只验证接线一致；
- P_init 是**未训练**的固定初值策略：它烧掉 97–99.9% 信用（6 条 first + 若干 refresh），
  在 2/3 seed 上与 hold 路径任务数相同、在 s1100001 上同为 5——这是**固定初值的行为
  记录**，不是训练后的策略表现，不得读成"学习无用/有用"；
- 三个 seed 是第十六轮已claim开发块内的**开发接线复跑**，不得称作新独立测试集；
- 短轨迹上的约束统计不得外推为期望性能约束已满足（spec §5 B3 明确要求）。

## 未执行 / 未授权（如实登记）

- 未训练、未更新权重、未加载旧通信 checkpoint 恢复训练；未开封存集；未改生产默认值；
- 未跑 `integrated_sar` 全量测试套件与 AGENTS.md full smoke：46 项中 18 项导入 driver，
  无法预先确认为零 episode 启动，且预算不得外溢——本轮只运行自有零启动单测
  （Stage B 12 项 + Stage A 30 项，均全绿）；
- 容量 mask（`sends_per_step`）在本配置为 None，与 D1 A3 归档一致，本轮无触发点；
  真实 mask  exercised 的是字节预算 mask；
- C 阶段（δ、ρ、归一化选择、训练量、700/1600）未批准、未启动。

## 代码与产物身份

冻结件 `CONFIG_AND_MODEL_MANIFEST.json` 登记 harness、生产依赖（comm/policy/learning/
budget/joint_driver/task03/lifecycle_metrics/expired_attribution/task_registry）、
导入的第四/五/七/十八轮模块与两个 checkpoint 的 SHA-256；冻结后未改任何被依赖文件
（生产三棵树相对分支起点 `5d0d71f` 零 diff，`sb_freeze.py` 落盘核验）。
启动前复核发现的 `returns[0]` 口径问题（`gae_lambda=0.95` 时它是 GAE 混合量，
仅 `=1` 时等于无折扣总和）已改为"闭合校验奖励和与执行器账本"，并有合成测试锁定。

**本地 SHA**：分支 `feat/comm-cost-stage-b`；链
`5d0d71f → 1b6a45e（启动前文案复核）→ ea34775（冻结件）→ dbf94dc（Stage B 执行与判定）`。
**远端**：本轮授权普通推送 Stage A/A1 与 Stage B 分支、不 force、分叉则停。实际执行：
本会话中一次 `github.com:443` TCP 探测曾成功，但随后两次 `git ls-remote origin`
分别失败（连接被重置；连接超时）⇒ **未执行 push、未核验任何远端 ref，成功远端接触 0 次**
（本地交付不受影响；网络恢复后需另行核验推送）。
