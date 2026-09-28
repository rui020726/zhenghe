# ROUND25 协议：固定 seed33 greedy 候选的 64 新场景复验（执行依据为本轮用户指令）

- 日期：2026-09-28。状态：**启动前冻结完成，待执行**（本文随首启前冻结提交落盘）。
- run_id：`comm_cost_greedy_replication_20260928`；分支：`exp/comm-cost-greedy-replication`。
- **执行依据**：`runs/comm_cost_greedy_replication_20260928/EXECUTION_BASIS.md`（用户在本会话直接
  提供的完整指令，已登记 SHA256 `cada6bca546bb2efef778b75c9e2ccafab0e61615b9933b0301bc224f8d817f6`）。
  它取代旧 spec 文本与补充授权文件，作为本轮后续执行依据。
- **旧 spec 状态**：`ROUND25_COMM_COST_GREEDY_REPLICATION.md` 的离线 hash 校验维持 **MISMATCH**
  （`SPEC_VERIFICATION.json`）：不修改旧文件凑 hash，不声称旧件已核验；本协议的 hash 也不是旧 spec
  的期望 hash，只用于运行中防漂移。
- **远端状态**：`remote_verification = NOT_PERFORMED`，`publication_status = PENDING_NETWORK`；
  本会话不再网络重试；发布不是本轮继续运行的前置条件。网络恢复后再普通推送并核对远端，不 force，
  不把未发布或未核验状态写成成功。

## 1. 本轮唯一问题

> 开发集事后选定的 seed33 最终 greedy 模型，在新场景上能否比固定规则少花**已计量载荷尝试成本**，
> 同时未观察到按时任务数损失？

不检验训练算法整体成功率，不宣称学习已稳定胜过规则。候选是看过 ROUND24 16 个开发场景后**事后
选定的固定模型**；新场景评估检验此固定候选的泛化，不能消除模型筛选事实，也不能估计随机训练一次
的成功率。

## 2. 固定候选与三臂

| 臂 ID | 固定行为 | 用途 |
|---|---|---|
| `learned_s33_greedy` | ROUND24 seed33、ep128 最终 checkpoint、原 greedy 解码 | 唯一主候选 |
| `immediate` | 原 reference 实现（driver 原生，不注入） | 任务参照与成本参照 |
| `late_discovery_gate_147` | `LateDiscoveryGatePolicy(cut_steps=147.0)`，原阈值、原边界 | 唯一主成本比较对象 |

冻结项：checkpoint SHA256（`52f490ab96a2ddad…`，CANDIDATE_FREEZE.json）、特征/schema
（19-d actor 特征、`onpolicy-tape/2`、`budget-sequential-mask/1`）、mask 与执行顺序、argmax
平局归 SEND 的原版 greedy 解码；不改阈值或 temperature。

seed11/22、sampled、always_hold 及其他规则的既有结果留在历史表；本轮不为它们新增环境评估，
不根据新结果换候选。0 训练、0 蒸馏、0 调 λ/η/奖励/网络；**optimizer.step 必须为 0**（含合成
optimizer 更新）。

## 3. 配置与场景（逐字段继承已核验 ROUND24）

真实 PILOC + 冻结 CTDE；真实搜索、6 目标、1000 步；normal-lossy、投递成功率 0.7；29300 B 硬预算；
ACK 关；`policy_owned_timing`、无公共强制发送；保留真实视距感知与 gossip；生产默认值不变。
B0/λ 只记录来源，不作推理期旋钮。逐字段证据：`CONFIG_DIFF.json`（behaviour fields 与 ROUND24
冻结件逐项相同，`v2_freeze.configuration_hash` 重算值不变）。

费用口径仅称"已计量载荷尝试成本"：丢包照计、拒付计 0 并单列；不称全系统通信量或真实无线能耗。

场景：64 个此前未使用场景 `1400000..1400063`（碰撞审计 `SEED_COLLISION_AUDIT.json`：全项目
文件名与文档声称范围均无冲突；若冲突则按预定算法从 1400000..1499999 升序取前 64 个未占用 seed，
先冻结列表再生成——本块未冲突，未启用回退）。`800000..800399`、`900000..900199` 不触碰；
旧 16 个开发场景不与这 64 个合并统计。

三臂使用相同场景定义与原外生随机过程约定：环境/信道 RNG 来自场景 seed；学习臂动作流派生
`sha256('round24-eval:33:<scene>:greedy')`（逐字继承 ROUND24 final_greedy 块；greedy 不抽样）。
同 seed 不等于后续逐消息经历同一批丢包；不为改善配对结果改造信道 RNG。

## 4. 资源与启动前冻结

- 正常启动：64 场景 × 3 臂 = **192**；技术储备 **16**；独立硬上限 **208**；不挪用 ROUND24 剩余额度。
- 失败、重试、smoke、reference 全部计入；无额外录制、吞吐探测、仿真 smoke 或暗中 reset。
- 首两个场景（1400000、1400001）三臂共 **6 次**为技术门禁，包含在 192 内；门禁通过后自动完成
  余下 186 次。
- 记账：每次构造/启动仿真前先原子 claim（`ledger_claims/`，O_EXCL）+ 每进程账本；两者计数必须
  一致；分片按冻结 cell 列表确定性取模分配，resume 跳过已有 final record 的 cell，不重复分配。
- 冻结件（首启前本地提交）：本协议、EXECUTION_BASIS.md、执行头、ARM_FREEZE.json、
  CONFIG_DIFF.json、SCENE_REGISTRY.json、SEED_COLLISION_AUDIT.json、代码依赖身份、
  START_LEDGER.json（plan）、STATISTICS_PLAN.json、汇总器与判读脚本。
- 独立输出目录 `runs/comm_cost_greedy_replication_20260928/`，不写回历史目录；每条产物含
  场景/臂/模式/模型/代码身份/attempt。

## 5. 技术门禁与继续条件

零环境、零 optimizer 的推理路径预检（`RUN_PREFLIGHT.json`）：模型加载、greedy 解码、特征、
顺序预算 mask、费用汇总、唯一命名、缺格聚合失败关闭、统计件单读。首 6 次核验：初始场景与配置
可比（behaviour_config_sha256 与 transport 参数逐臂一致、冻结权重 digest 一致）、冻结权重/模型
buffer 每集前后不变、learned 走原 greedy 不混 sampled（逐 tape 行 argmax 复核）、动作/mask/发射/
拒付/预算逐步闭合、成本账本与任务生命周期及 final record 齐全、旧产物未覆盖。

技术门禁不要求任务或成本表现变好；性能退化不是技术失败，不删除、不隔离、不重跑择优。运行中只按
技术健康和配额停止；不按中间显著性调参、扩样或提前结束。发现影响行为或研究语义的代码缺陷则整轮
停止报告，不夹带修复后混跑同一矩阵。

## 6. 预登记统计

每场景 `dY = Y_候选 − Y_参照`（正更好）、`dC = C_候选 − C_参照`（负更省）。唯一主成本比较：
seed33 greedy 对 `late_discovery_gate_147`；任务比较同时对 `immediate` 和 late 报告；其余比较
预先标为次要描述。

以 64 场景为配对抽样单位：NumPy PCG64 `seed=20260928`，一次生成 10000×64 重采样索引，所有
任务/成本比较共用；95% percentile bootstrap，分位 `method='linear'`；表格、判读、verdict 只读
同一机读统计件 `PAIRED_COMPARISON.json`（其 matrix_sha256 与冻结方案一致才被接受）。

谓词（冻结）：任务改善 `dY>0`/相同 `dY==0`/退化 `dY<0`；成本更省 `dC<0`/相同 `dC==0`/更高
`dC>0`；联合占优 `dY>=0 ∧ dC<=0 ∧ 非全同`；权衡分两类 `(dY>0,dC>0)`、`(dY<0,dC<0)`；
单边变差 `(dY<0,dC==0)`、`(dY==0,dC>0)`；双输 `(dY<0,dC>0)`；全同单列。

零事件规则：64/64 无净退化时可附 `1−0.05^(1/64) ≈ 4.57%`（i.i.d. 场景假设下发生净退化场景
概率的单侧 95% 上限）；它不是期望任务损失上限，也不是部署安全保证；存在退化时只报实际比例。

## 7. 判定字段

- `primary_cost_evidence`：主成本 dC 的 95% CI 上界 < 0 才 `lower_cost_supported`，否则
  `lower_cost_not_established`（均值较低但区间含 0 不得写成已证明降本）。
- `observed_task_count_status`：对 immediate 和 late 均逐场景 dY>=0 才 `no_observed_count_loss`。
- `goal_identity_tradeoffs`：每场景失去/新增按时目标 ID 单列（含净数打平场景）。
- `constraint_guarantee`：始终 `not_established`。
- `training_seed_stability`：始终 `not_established`（ROUND24 三种子差异仅作历史背景）。

仅当主成本支持且无观察到净任务数损失时，才写"固定 seed33 greedy 候选的降本优势在这 64 个新场景
得到支持，未观察到净任务数损失"，并紧跟任务身份替换、稀有失败不可排除与事后候选选择的限定；
不写"学习方法普遍优于强规则"。出现任务权衡或成本证据不足则完整交付后停止。

## 8. 交付与停止

至少交付：执行依据与冻结件、来源/模型核验（REUSE_VERIFICATION.json，复用已核验件并登记 hash）、
场景碰撞审计、场景注册、启动账本、首 6 次门禁、完整性预检、绝对/配对/逐场景统计、目标身份权衡、
verdict、原始产物 hash 清单。本地分别提交协议/接线、结果、状态文档。

完成、正确性阻断或 208 次耗尽后停止：不续训、不换候选、不追加确认、不部署。网络不可用可交付
"本地执行完成，远端发布待补"；网络恢复后再核对远端并补普通推送。
