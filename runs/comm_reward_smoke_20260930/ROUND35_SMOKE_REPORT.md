# ROUND35 报告：修复版奖励接线的小规模真实 smoke 验证与恢复边界实机核对

分支 `fix/comm-task-reward-event-timing`；开工头 `c43ec5f1a12ef18238d4eed3aa90e3681e9ae58e`（基线
ROUND33 头 `c7d47d25e638cbe3728cf36e3a22b9cfc563a353` 为其祖先，已本地核验）；交付头 = 本仓库
`git log --oneline -1` 顶部那枚（本文件不预填哈希，避免猜）。产物 `runs/comm_reward_smoke_20260930/`，
执行依据原文存档 `EXECUTION_BASIS.md`。判定一律以 ROUND34 的
`TASK_REWARD_TIMING_CONTRACT.md` 为准。

```text
runner 进程 3 次（有效启动 2 次 + 首次尝试在 4 秒处失败）；episode 构造 15 / 上限 16，剩 1
optimizer.step 60 / 上限 100（启动内 36 + 验证重算 24）；单次最长 97.85 s（上限 2 小时）
新流 0；封存块 0；旧轮产物改写 0；网络尝试 0；推送 0；新 checkpoint 由验证脚本写回 0
环境：torch 2.9.0+cu128、numpy 2.2.6、Python 3.10.19（D:\project_env\python.exe）
     —— ROUND34 的"本机无 PyTorch"限制对本机不成立，故本轮无环境豁免项
```

**结论：go（可进入第 3 天冻结训练）**，附 3 条残余风险与 2 项未覆盖（见 §9、§10）。
smoke 的任何 loss/回报数字只用于证明链路走通，**不**作为学习效果证据；本轮验证的是机制与账本，
不是"学习效果变好"。

## 1. 唯一改动就是采集时点，且真实代码路径被原样执行

harness 只做两件事：把 `v2_common` 的五个输出路径全局与启动上限**重绑定到本轮目录**，
以及以只读方式取 ROUND24 的 64 份 immediate reference 作配对参照。rollout、PPO、对偶更新、
保存与恢复全部走真实模块（`v2_run.run_training_episode` → `_install_taps` →
`PublishedProgressRewardCollector`；`v2_trainer.ppo_update/batch_g_hat/dual_update/
save_checkpoint/load_checkpoint/load_checkpoint_read_only`；训练循环是 `v2_train.train_seed`）。
本轮未改任何生产/训练文件，也未改超参、网络、目标、奖励数值或成本路径。

启动前 12 项门禁全清（`ROUND35_PREFLIGHT.json`），其中三条值得单独说：

| 门禁 | 结果 |
|---|---|
| P5 四源初值相同 | `fresh_controller(11)` == 归档 ROUND32 批 0 更新前 == git 里启动前配对件 == ROUND24 ep0 里程碑经真实 loader 恢复，全等 `7eff0f40203fdae6…` ⇒ 被动性能做**逐字节**比对 |
| P6 参照身份 | 8 个 smoke 场景的 immediate reference 在两轮回记录里 behaviour_sha256 / 传输声明 / 发射序列 / relay 计数器全等；`config_hash` 跨轮不等**是预期**（它覆盖本轮自己的输出路径），因此不作比对键 |
| P9 ROUND34 回归重跑 | 两套测试在**有真实 torch** 的解释器下全部通过（上一轮是 torch stub 下通过的） |

## 2. 任务 1：同 tick 奖励的实机核对（45 个完成事件逐条）

启动①连续 8 集（冻结顺序的前 8 个场景），真实事件账本共 45 条完成事件，逐条登记在
`EVENT_TIMING_VERIFICATION.json` 的 `per_event_table`（含 event_id、observed_at、relay_tick、
clock_now、accepted、deadline、两个 expired 标记、on_time、credited）。

| 核对 | 结果 |
|---|---|
| C1 事件 relay_tick == int(event_time) | 45/45，无违例；9 个必填字段 0 缺失 |
| C2 task_stream 写入键 == 事件 tick | 8/8 集每键增量恰等于该 tick 的入账事件数 |
| C3 clock_now 时序 | 45/45 都是 `event_time - clock_now = 1.0`（契约里回调时钟仍是步前值，只作观察不作判据） |
| C4 归属用真实 `_bundle` 重算 | 每笔都落在"该 tick 当时或之前最近的动作行"；0 违例、0 意外 prefix |
| C7 与启动前登记的预测逐集相符 | e0 归档 `[9,26,28,34,48,194]` → 实测 `[8,25,27,33,47,193]`；8/8 集实测键恰为归档键 −1（这是可证伪项，未被证伪） |

六类情形对表（`C5`）：

| 契约情形 | 真实事件计数 | 说明 |
|---|---:|---|
| 同 tick 完成 | 45 | 全部完成事件都落在有动作行的 tick 上 |
| 跨 tick、下一 tick 有新动作 | 30 | 例 `e0:piloc-e1-target-3:8`（下一拍 9 也有动作行）——修复前这 30 笔会被下一行抢走 |
| 跨 tick、无新动作 | 0 | **本场景块结构性不出现**：归档 4281 笔里 0 笔如此；只有合成测试覆盖 |
| 末 tick / 最后动作行完成 | 6 | 例 `e0:piloc-e1-target-0:193`（同时是该集最后一个动作行） |
| 首动作之前 prefix | 0 | 该块 relay 从 tick 1 起且 tick 1 即有动作行；只有合成测试覆盖 |
| 完成但已过 deadline | 0 | 见 §4：本块的过期是以**非完成**状态出现的 |

## 3. 任务 3：账本与 registry 一致、无重复入账

- 资格五条（C6）：45/45 笔 credited 事件同时满足 完成 / registry 已接受 / 快照完成 / 双侧未过期 /
  `observed_at <= deadline`；0 笔违反；`credited_total = events = 45`。
- event_id 唯一：45 个 id 全不重复（C10）；同一目标 `credited` 次数 > 1 为 0。
- 每集闭合（C8）：`on_time == task_stream 总和 == task_reward_total`，且成本 tap == 账本字节；
  最后一次动作行之后的入账为 0；`unassigned_prefix` 为 0。
- **重复事件 ID 的幂等**在真实数据里未被触发（这 8 集没有重复发布），因此该点证据等级为
  "合成 + 结构"：`r35_failure_repro.py` 在真实总线 + 真实 registry + 真实 collector 上跑通的
  那一次记录了 `credited_now`/`ledger_rows_added` 的行为，加上 ROUND34 的重发布测试；本轮不把它
  说成实机验证过。

## 4. 任务 2：尾段捕获与 deadline 过期不入账

- 尾段：每个完成事件的写入键就是事件发布当拍（C2），所以末拍完成不需要后续 step；
  闭合 `on_time == 入账总数` 证明尾部没有漏捕（8/8 集成立）。
- deadline：本块里目标过期表现为**非完成**进度事件（`detail.expired=true`、
  `failure_reason=deadline_expired`）。真实数据里出现 2 集（`e1`、`e4`），事件时间分别在其目标
  deadline 之后一拍的同一时刻；这些事件**既不在账本里也不产生入账**，且行的 `expired` 计数与
  lifecycle 的过期事件数逐集相符（C9）。因此 collector 的第一道闸（状态必须是 COMPLETED）就把它
  挡住，deadline 守卫是第二层、本块真实数据走不到。

## 5. 任务 5：被动性证据与等级（`PASSIVITY_EVIDENCE.json`）

| 等级 | 范围 | 证据 |
|---|---|---|
| **A 直接逐字节** | 启动① e0..e3（批 0 的任何权重移动之前） | 与归档修复前记录比 17 个可观测字段 + 逐决策 tape 的 `features/actions/log_prob/actionable/targets/step` **全等**（行数 66/429/52/76 相同，`max_abs_log_prob_gap = 0.0`），`behaviour_sha256` 相同；唯一差异是 `task_stream` 键（早一拍）与新增的事件账本字段 |
| **B 间接** | e4..e7 | 权重已被修复后的更新移动（`log_prob` 差 0.0034–0.0080），但场景/horizon/传输声明逐项相同；此处分叉属干预效果，不算被动性反证 |
| **C 结构 + 计数** | 8/8 集 | 账本↔指标闭合、event_id 唯一、成本 tap↔账本闭合；collector 源码只读（不发布、不改 registry、不动时钟、不抽随机数）由 P9 的合成用例与 `LAUNCH2_ATTEMPT1_REPRO.json` 的四项不变量（rng/clock/registry/publish 计数）在同一批真实对象上复核 |

未做到的那一档：把只读仪表挂在**活的** rollout 上并落盘（仪表记录在内存里，进程因额度拒绝而未写出）。
这不改变 A 级证据的成立，但登记为未覆盖项（§10）。

## 6. 任务 4：真实恢复边界（`RESTORE_BOUNDARY.json` + `RESUME_CHAIN.json`）

| 项 | 结果 |
|---|---|
| 4a 新 checkpoint 真实恢复 | ep0 里程碑与 ep8 rolling 均经 `load_checkpoint` 成功恢复；两个版本字段齐全（`constrained-cost-v2/published-completion-event-at-relay-tick/2` 与 `published-completion-event-at-relay-tick/2`，payload 与 training_state 双写）；同一文件二次加载结果逐字段相同；恢复后进程 RNG 与文件内 rng 块一致 |
| 4b 篡改副本 | 5 种篡改（payload 的两个版本字段各改一次、两者删除、training_state 内两个字段各改一次）**5/5 被拒**；用计数桩观测到拒绝发生时 `_restore_controller` 0 次、`Adam` 构造 0 次、`restore_rng` 0 次，进程 RNG 未被移动 |
| 4c 真实旧 ROUND32 `.pt` | 取 ROUND32 自己的归档 manifest **重建** freeze 身份（9 字段，重建结果与文件自述相符）后：`load_checkpoint` 被**版本门**拒绝（信息含 "read-only under the current"）；换用 ROUND24 身份则更早的 freeze 门先拒（两层分开登记）；`load_checkpoint_read_only` 成功并给出 `legacy_unversioned_capture_semantics`，`optimizer_restored=false`、`rng_restored=false`（构造器与 `restore_rng` 被换成会抛错的桩仍未被触发） |
| 4d 续跑一致性 | 跨进程恢复真实发生（launch2 打印 `resumed seed 11 at episode 4 lambda=1.0125 updates=1`）；恢复后 e4、e5 两格对连续运行**逐字节相同**（20 字段 + 逐决策 tape），且行内登记的采样 λ 就是恢复态的 1.0125；保存点身份与两跑的批 0 后更新状态全等（参数哈希、四模块 digest、λ、updates=1、episodes_done=4） |
| 4d 的确定性补充（0 构造） | 从 ep0 里程碑恢复 + 用本轮自己的 e0..e3 行与 tape 重算批 0 更新 → 后更新参数哈希、λ、samples=623、12 步**与登记值逐字节相符** |

## 7. 任务 6：checkpoint 内容层补验（`CHECKPOINT_CONTENT_LAYER.json`）

ROUND34 把 18 个 `.pt` 的四层登记为"不可用/未运行"。本轮全部实做，且**不改写** ROUND34 的结论：

| 层 | 18/18 结果 |
|---|---|
| L1 文件字节哈希 | 本轮重算 == ROUND34 观测 == index 声明（三方全等）；处理前后哈希相同 ⇒ 旧文件未被改写 |
| L2 payload 元数据 | 真实反序列化读出 schema/trainer_tag/loss/gae/optimizer/reward_timing_schema、freeze、`episodes_done/updates/optimizer_steps/lambda`；λ 与 index 声明相符；旧文件确实**没有** `training_version`/`reward_timing_version`（这正是 4c 版本门拒绝的对象） |
| L3 whole-controller 参数摘要 | 从张量重算 `parameter_sha256`（参数 5700、shapes 相符）与 index 声明、批记录 `controller_parameters_before` **18/18 全等** |
| L4 四模块 digest | 用真实 `v2_train.module_digests` 重算，与 ROUND34 的声明 **18/18 全等**（不同命名空间的哈希未被互相替代） |
| L5 native forward + likelihood | 用各点自己的更新前权重对本批 tape 实跑前向，重算联合对数似然：18/18 在 1e-6 内（重算最大差 4.0605664253234863e-07，最小 5.960464477539063e-08），且每一点的重算最大差与归档侧车记录 `likelihood_alignment.max_abs_gap` 之差都是 **0.0**（18/18） |

本轮新产的 2 个 checkpoint 同样重算并通过（含一次真实 forward/likelihood：623 条 tape，重算 `max_abs_gap` = 2.384185791015625e-07，
与本轮批记录登记值完全相同）。

## 8. 修复在真实数据上改变了什么、没改变什么（`ATTRIBUTION_EFFECT.json`）

同一批格、真实 `_bundle` 重算：

| 未改变 | 改变 |
|---|---|
| episode 总量（on_time / emissions / 已花字节）、每批样本数 623 与 555、minibatch 数 12/12、optimizer 步数、**`g_hat`（b0 +0.25、b1 +0.25）与 λ（1.0125 / 1.025）**、未归属 prefix 0.0 | 每笔入账的 tick（早一拍）与**归属到哪一条动作行**：8 集 45 笔中 **30 笔**的归属行动作行改变；两批的 surrogate / 两个 critic loss / 梯度范数 / **更新后参数哈希**都不同 |

另在归档数据上量化过同一件事（0 启动、真实 `_bundle` 双跑）：768 格里 4281 笔入账中 2788 笔
（65%）在修复前被下一条动作行抢走，750/768 格的归集奖励发生变化。
含义：**约束统计量与 λ 轨迹不因该修复而变**，变的是逐决策的信用分配，因此不能把任何后续行为差异
说成"约束被满足得更好"。

## 9. go / no-go

**go：可以进入第 3 天冻结训练。** 判据是：契约要求的时序与归属在真实 runner 上逐事件成立（含可证伪
预测未被证伪）、被动性有 A 级逐字节证据、恢复边界在真实 `.pt` 上成立（含旧 checkpoint 不可热启动）、
checkpoint 内容层与 native likelihood 全部实做通过、账本与额度自证。**没有**任何一项结论依赖
smoke 的 loss 或回报水平。

冻结训练前仍需确认的残余风险（都不阻塞 go，但必须写进训练协议）：

1. **规模外推未知**：smoke 只有 8+6 次构造、60 步；修正归属后长训练的**轨迹**必然与归档不同
   （§8 已证它确实改变信用分配），因此旧运行不能作为新运行的预测基线。
2. **两类契约情形在本场景块不可达**（跨 tick 无新动作行、首动作前 prefix），只有合成用例覆盖；
   若训练块换场景或 relay 起点改变，需要重做可达性判断。
3. **幂等与"重复入账"未经实机触发**：真实数据没有重复发布，只有合成与结构性证据；训练期若引入
   任何重放/补算路径（如 `v2_recover`），必须重新检查这一点。
4. harness 的两处本轮缺陷需带走：per-directory 启动上限少算一次失败起始（保守方向）；
   collector 仪表记录未落盘。二者都在登记件里，且都不影响上述证据。

## 10. 停止与交付

```text
runner 进程 3（launch1 完整；launch2 第 1 次尝试 4 秒失败，消耗 1 次构造不退还；
                第 2 次为用户批准的那一次，e6 被真实 ledger 拒起）
episode 构造 15 / 16；optimizer.step 60 / 100；单次 wall-clock 最长 97.85 s / 2 h
新流 0；未触碰封存块；未改超参与目标与成本路径；未热启动任何旧 checkpoint
旧轮产物改写 0（18 个 .pt 与归档行/tape 处理前后字节哈希相同）
网络尝试 0、未推送、本分支无 upstream
```

产物：`EXECUTION_BASIS.md`、`ANCHORS_AND_ENVIRONMENT.json`、`PRE_SMOKE_CASE_COVERAGE.json`、
`ROUND35_PREFLIGHT.json`、`LAUNCH1_SUMMARY.json`、`EVENT_TIMING_VERIFICATION.json`、
`PASSIVITY_EVIDENCE.json`、`ATTRIBUTION_EFFECT.json`、`RESTORE_BOUNDARY.json`、`RESUME_CHAIN.json`、
`CHECKPOINT_CONTENT_LAYER.json`、`LAUNCH2_ATTEMPT1_FAILURE.json`、`LAUNCH2_ATTEMPT1_REPRO.json`、
`AUTHORISATION_EXTRA_INVOCATION.json`、`SMOKE_LAUNCH_BOOK.json`、`LAUNCH_ACCOUNTING.json`、
`r35_*.py`。原始 rows/tapes/lifecycles/checkpoints 留在 Git 外（`launch1/`、`launch2/`、
`tampered/` 子目录，.gitignore 白名单只放脚本与 JSON 交付件）。

本轮不续训、不调参、不追加启动额度、不开启下一轮实验、不部署、不推送。
