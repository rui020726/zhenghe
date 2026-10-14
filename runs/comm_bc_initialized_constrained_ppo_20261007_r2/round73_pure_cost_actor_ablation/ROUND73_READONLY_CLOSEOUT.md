# ROUND73 只读收口：纯成本 actor 消融的实际消费与科学链核验

分支 `exp/round73-pure-cost-actor-ablation`，HEAD `afdcb3100a6c78a392058201907bd5d79242ab5d`（收口时未变）。
本报告为零环境收口：不构造环境、不加载模型、不做前向/反向、不做 optimizer step、不重放 PPO、不写 checkpoint、不新建 claim/slot/permit、不跑测试、不重跑、不恢复剩余 6 格。所有数字要么直接读自耐久原件（标 ORIGINAL），要么由 stdlib 解析这些原件得出（标 DERIVED）。机器字段见同目录 `ROUND73_READONLY_CLOSEOUT.json`（`kind: round73-readonly-closeout/1`）。

## 一、实际消费与 600 上限

| 项目 | 数值 | 来源 |
|---|---|---|
| 已登记 launch slot / claim | 574，全部 normal，编号连续 1..574，每项 attempt=1 | ORIGINAL |
| 其中完成 | 573 | ORIGINAL |
| 未完成 | 1（`r73eval_model_new_ep128_s33_sampled_scene1800029`，slot 574） | ORIGINAL |
| 结果 row | 573 = train 384 + eval 189 | ORIGINAL |
| 已完成步数合计 | 573,000 | ORIGINAL |
| 许可 consumed 且 constructor_returned=true | 574 / 574（许可文件共 1148 份） | ORIGINAL |
| 硬上限 600 | 消耗 574，余量 26，未超限 | DERIVED |
| 常规上限 579 | 余量 5 | DERIVED |
| 技术储备 21 | 用 0，余 21 | DERIVED |
| 历史非计划 probe | 0 | ORIGINAL |
| 门禁阶段环境构造 | 0 | ORIGINAL |
| 两次技术失败尝试 | 各 0 构造 / 0 claim / 0 slot | ORIGINAL |

**574 claim 的构造覆盖与求和（逐字段）**：574 条 claim 全部具备 `environment_constructor_calls` 与 `joint_run_constructed_calls`，均为 int、均等于 1，和均为 574；`run_invocation_calls` 与 `joint_run_method_calls` 各 574；`run_return_calls` 为 573（1 条为 0）；`run_failure_calls` 与 `joint_run_build_failures` 全为 0；`completed_steps` 573 条为 1000、1 条为 0，合计 573,000。许可侧 1148 份文件中，574 份 `round68-construction-permit/1` 状态为 consumed，574 份 `constructor_returned=true`。booked 与 proven 在 574 上完全一致。

**claim ↔ row 一对一（本轮重新推导）**：574 claim、573 完成、573 row；"完成但无 row" 0 条，"有 row 但无完成 claim" 0 条。训练 row 三 seed 各 128 集且 0..127 连续；评估 row 为 s11/s22 两模式各 32 场、s33 greedy 32 场、s33 sampled 29 场（缺 1800029/1800030/1800031）。

> 更正：`_closeout_claims_audit.json` 的 `complete_claims_without_row = 573`，以及 `_closeout_claims_crosswalk.json` 的 `eval_coverage.row_scenes = 0`，都是标签归一化产物（claim 的 logical_job 名与 row 的 label 命名规则不同，eval row 只带 (seed, mode) 组标签），并非缺 row。原文件保留不改。

**中断格的真实消耗**：slot 574 的状态标签 `claimed_before_environment` 与事实不符——构造器已于 unix 1791961323.8945768 调用并返回，许可已 consumed，一次环境构造已入账且不退额。磁盘上留有 972 行 relay-policy 部分日志（同模式已完成格为 1355–1447 行），无 controller 日志、无 tape、无 row。`completed_steps = 0` 与 `simulation_started = false` 是 run 从未返回而未更新的标志，不能读成"零仿真工作"；确切步数无耐久记录，标 UNKNOWN。

## 二、scratch 前向计费证据与未知分列

- 耐久前向账本（ORIGINAL）：`raw/runtime/r73_forward_ledger.json` = 1030 rows / 130 calls / 0 refusals / 0 joint_rows，上限 6000。
- 重建收据逐条列支 124 笔、合计 1018 rows（110 笔单行前向 + 908 行按 `2 * n_records * EPOCHS_PER_BATCH` 的聚合预扣）。
- **未知分列**：耐久 1030/130 与列支 1018/124 相差 12 rows / 6 calls，无任何耐久文件列支这 6 次调用。收据自记 `rows_at_suite_start = 12`，可与之相符，但只是相符而非证明，归属标 UNKNOWN。
- 实测 trunk 行数/调用数：无耐久记录，只有区间 [110, 1030]。
- 合成 optimizer step 守卫（ORIGINAL）：`used = 8`、`cap = 8`。**这 8 是 PPO update 调用数，不是物理 optimizer.step 数**：守卫类与 licence 都声明以物理 `optimizer.step` 计（`round68_late_initial_confidence_ablation/r68_guards.py:95`），而计费点在 `test_r73_qualification_gates.py:347` 每次调用扣 1。每次 gate 调用内部做 4 个物理 Adam 步。
- 隔离归档 `ppo_update_calls.jsonl.synthetic.jsonl`：23596 字节 / 68 行，每行嵌套 `line.optimizer_steps = 4`，合计 **272 个物理 optimizer step**。其中 20818 字节 / 60 行前缀的 sha256 `950f3807…` 与真实运行收据自记值一致（只证明字节边界）。按固定标签序列推断为 9 次运行：1 次真实账本运行（8 调用 / 32 物理步）+ 8 次 scratch（60 调用 / 240 物理步）。
- **归属边界**：哪一次 scratch 运行扣了哪个账本未获证实（真实守卫的 8 个事件只能证明真实账本运行扣了它）；scratch 运行无墙钟时间戳；归档只记录被转移的行，不记录各次运行的扣费。
- **结论**：以声明单位衡量，合成 optimizer step 上限未被执行——真实账本运行一次即 32 物理步（超 24），全部归档运行合计 272。这不是对已有模型/结果的否决，但是资格预算合规性为 false 的根据。

## 三、三 seed 训练链资格

三个 seed 均通过同一组核验：INIT 为冻结的 R71 校准 INIT（init 文件 sha256 与 `ROUND73_FREEZE.json` 一致，仅加载模型）；batch 0 `parameters_before` 等于登记 INIT 参数摘要；32 条已提交 batch 记录的 before/after 参数链连续；batch 31 `parameters_after` 等于实际被评估的 ep128 模型参数摘要；`episodes_done_after_final = 128`；首批门禁 `actor_advantage_is_minus_a_cost` 与 `actor_objective = pure-cost/1` 通过；lambda 自 1.0 起更新并上报。

| seed | batch 记录 | 参数链 | ep128 参数摘要 | optimizer 步数 | lambda 1.0 → |
|---|---|---|---|---|---|
| s11 | 32 | 连续 0..31 | `186f35babf8a04b4…` | 296 | 2.625 |
| s22 | 32 | 连续 0..31 | `1a17020464d78cbd…` | 288 | 3.0 |
| s33 | 32 | 连续 0..31 | `c09c0b966de02a0f7…` | 308 | 3.125 |

三步数合计 892，与真实 PPO 日志 96 行 / 真实 optimizer 日志 892 行一致。ep128 文件 sha256：s11 `0f9319df29c1e8c0…`、s22 `f88d134dc8017c18…`、s33 `91a7925f6a86494d…`；磁盘上 ep128 == after_b31 == final。`raw/checkpoints` 下共 201 个 .pt（每 seed 67 个），synthetic .pt 仅 3 个且全在 `synthetic_test_fixtures` 下。

> 记账缺口：`raw/runtime/checkpoint_writes.jsonl` 为 0 字节，但 201 个 checkpoint 文件存在。写入事件由文件本身和参数链证明，不由该日志证明；这是记账缺口，不是"零写入"的证据。另：`after_b0` 与 `before_b1` 的文件字节不同（载荷带 batch 元数据），参数级连续性由 batch 记录证明；重新推导参数相等需要 torch 加载，本次收口不做。

**资格判定**：科学链三 seed 均已验证；资格预算合规性为 false（见第二节）。剩余 6 格不恢复，不补证据、不重放 PPO、不补跑。

## 四、greedy 96 格描述性配对（NEW - R71，同 seed 同 32 场景）

身份核验：两臂各 96/96 格文件齐全、无缺失；R73 三 seed 参数摘要互不相同，R71 亦不相同，说明比较的是三个不同模型。**早期路径未动**：两臂 early 首报实测均为 2 SEND / 132 SKIP，且 R71 与 R73 各自首批门禁记录的冻结模块摘要逐 seed 完全相同（s11 fe9c0082…、s22 b8a59378…、s33 568bea79…）；变化全部在 late：R71 49 SEND / 0 SKIP → R73 0 SEND / 49 SKIP。

| 读数 | R71 ep128 | R73 ep128 | 差 |
|---|---|---|---|
| 按时任务总数 | 183 | 170 | -13 |
| 计费字节/场景 | 4066.5 | 168.5625 | -3897.9375 |
| 场景成本 降/平/升 | — | — | 21 / 11 / 0 |

三个 seed 的逐场景结果在两臂中完全一致，因此在本开发块上不构成独立重复。13 个丢失目标在每个 seed 都是 R71 的 late SEND、R73 的 late SKIP（机制已核验）；丢失场景为 1800011/12/13/14/18/19/20/21/24/25/30 共 11 个场景，涉及 target-0/2/3/4/5；无救回目标。8 个场景（1800003/07/08/12/13/19/22/29）搜索模块只发现 5 个或 4 个目标，两臂上限都低于 6。

规则臂（只读复用同一 32 场景，无新构造）：IMMEDIATE 183 按时 / 14356.5 B每场景（459408 B，剔除 1 条重复行）；LATE_DISCOVERY_GATE 183 / 3897.9375 B每场景（124734 B）；ALWAYS_HOLD 170 / 0 B。R73 的按时任务数与 ALWAYS_HOLD 相同（170），但并非零计费（168.5625 B每场景）；R71 与 IMMEDIATE、LATE 规则臂任务相同（183），比 LATE 规则臂每场景多 168.5625 B、比 IMMEDIATE 少 10290 B。即 -13 是从 183 任务区间整体移到 170 任务区间，不是单一区间内的部分退化。

## 五、需要更正的数字（原件一律保留，只在此登记）

- `ROUND73_MACHINE_VERDICT.json`（根）：称训练/评估"未开始"、各项消费 0、结论 DEFERRED —— 实际 384 训练 + 189 评估已完成、573 row。
- `ROUND73_LEDGER.json`（根）：消费全 0、硬余量 600 —— 实际 574、硬余量 26。
- `raw/runtime/ROUND73_LEDGER.json`：记 384、硬余量 216 —— 落后 190，写于训练阶段结束、评估入账之前。
- `raw/runtime/ROUND73_EVAL_QUOTA_GUARD.json`：booked_total 384、硬余量 216 —— 实际 574 / 26。
- `ROUND73_FREEZE.json`：`phases_completed` 的 train/eval 为 false（身份字段本身正确未改）。
- `ROUND73_ACCOUNTING_AUDIT.json`：`run_attribution.scratch_totals.runs` 由 7 更正为 8（其自列 9 次运行中 8 次是 scratch，仅第 8 次是真实账本运行）；calls 60 与 steps 240 无误，已重新推导。
- 不可再引用的旧说法：硬余量 600、常规余量 579、储备余量 21、booked 384、环境构造 0、"8/8 合成 optimizer step 已守上限"。

## 六、可保留与必须暂缓的结论

**可保留**：纯成本 actor 目标（A = -A_cost）确曾在真实训练路径上生效于全部 384 集；三个 ep128 模型来自已验证链路；在该 32 场景开发块上，greedy 下 late 首报被全部转为 SKIP（按时 183→170、计费 4066.5→168.5625 B每场景，21 降 11 平 0 升）；任务损失有完整机制解释；早期路径由冻结摘要与实测首报双重支持为未改动。

**必须暂缓**：合成 optimizer step 上限得到遵守的说法；资格预算合规为 true 的说法；sampled 模式读数（96 格中 3 格无结果）；3 个 focal 1700027 格（从未 claim）；任何统计推断（stats 阶段未运行，无区间、无显著性、无非劣声明）；任何泛化、部署或投产结论；把 -13 单独归因于 actor 目标而非整个 R73 配置差。

## 七、零副作用与交付

本次收口计数器：环境构造 0、环境运行 0、仿真步 0、模型加载 0、CPU 前向 0、反向 0、真实 optimizer step 0、合成 optimizer step 0、PPO 重放 0、RNG 恢复 0、checkpoint 写入 0、新 claim/slot/permit 0、测试 0、实验进程 0；修改原件 0、删除原件 0。仅新增两个文件：`ROUND73_READONLY_CLOSEOUT.json` 与 `ROUND73_READONLY_CLOSEOUT.md`。两份 ROUND46 脏文件 sha256 维持 `cf3ad198…` 与 `93a2f69f…`，未修改、未暂存。本地显式点名提交，不提交权重与原始日志，不推送、不合并、不部署。