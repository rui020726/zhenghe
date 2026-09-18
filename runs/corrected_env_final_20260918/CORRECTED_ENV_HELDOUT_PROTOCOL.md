# 修正环境独立测试协议（CORRECTED_ENV_HELDOUT_PROTOCOL）

状态：**预注册**。本文件、`HELDOUT_SEEDS_V2.json`、`arms.py` 的真值表、`corrected_block.py`
与 `stats_v2.py` 一起先提交、后运行。运行开始后这些文件不再改动；需要改候选就中止本版验收，
保留全部产物，另定版本。

## 一、两臂真值表（冻结）

机器可读版本：`ARMS_CONFIG_TRUTH.json`（由 `python runs/corrected_env_final_20260918/arms.py`
生成，含每项配置、两个 checkpoint 的 SHA256、构造期望）。

| 配置项 | C | D |
| --- | --- | --- |
| `task_membership_consistency` | **True** | **True** |
| `use_coordinator` / `coordinator_policy` | True / `coalition_planner` | True / `coalition_planner` |
| `sticky_commitments`（→ `renew_commitments`） | True | True |
| `release_redundant_members` | **False** | **True** |
| `suppress_recommand_own_task` | **False** | **True** |
| `check_release_source_stage`（stage + 支援名单，批内逐条） | **False** | **True** |
| `residual_scale` / `allocation_interval` / `coordination_interval` / `command_ttl` | 0.25 / 5 / 5 / 10.0 | 同 |
| `dt` / `steps` | 1.0 / 1000 | 同 |
| `num_goals` / `grid_size` / `obstacle_density` / `detection_radius` / `search_agents` | 6 / 60 / 0.15 / 5 / 2 | 同 |
| `deadline_tightness` | (0.5, 2.0) | 同 |
| `rescue_species` / `rescue_per_species` / `actor_kv_mode` | 3 / 2 / tokens | 同 |
| 通信 | delivery 1.0，delay 0，subscribers all | 同 |
| `device` / 线程 | cpu；`OMP/MKL/OPENBLAS_NUM_THREADS=1`，`torch.set_num_threads(1)` 在 import 前 | 同 |
| `task_checkpoint` | `runs/joint_eval_20260916/ctde_final_7_1.pth`（回放不含 PILOC，见下） | 同 |
| trace schema | `allocation-decision-trace/3` | 同 |
| arm 标签 | `…+memfix` | `…+redrel+noself+srccheck+memfix` |

D 就是"当前已实现的全部保护"：覆盖保持的释放判据（毛需求 + 下一阶段）+ 重复命令抑制 +
源任务 stage 校验 + 源任务支援名单校验（批内逐条）。

`piloc_checkpoint` 在两个 arm 的配置里都**不出现**：配置校验会拒绝"回放流 + PILOC 权重"同时
存在（两条搜索策略不能并跑）。PILOC 只在录制阶段使用，其 SHA 冻结在真值表里。

### 启动时的构造断言（不看名字，看对象）

`arms.assert_built(run, arm)` 在每个 episode 构建后立刻执行，任何一条不满足就不产出任何行：

- 真正实例化出来的环境对象：类名 `TaskEnv`、模块文件 `old_task_env.py`、属性
  `membership_consistency` 与该臂期望一致；
- `run.gate` 确实是 `AssignmentGate`，且 `release_state_of` / `release_roster_of` 的
  接线（None 或 callable）与该臂期望一致（D 要求两个都在）；
- 展开 `ReplanCoordinator` 包装后的 `CoalitionPlannerCoordinator` 对象上，
  `release_redundant_members`、`renew_commitments` 与真值表一致；
  `suppress_recommand_own_task` 在真正承担它的那个对象（task adapter）上核对；
- run signature 的 arm 串必须含该臂应有后缀、不含不该有的后缀（例如 C 不得带 `+redrel`）；
- signature 记录的 `code_commit` 必须等于当前 HEAD；
- `arms.assert_frozen_tree()`：生产目录无未提交改动，且 `git diff <freeze> -- integrated_sar task_allocation` 为空。

## 二、seed：400 个从未用过的

`HELDOUT_SEEDS_V2.json`，规则事先写死：**从 500000 起升序取整，凡出现在任何历史来源里的都跳过，
取满 400 个即止**；不做抽样后剔除、不按难度筛、看到结果后不增删。
排除来源是机械扫出来的：`integrated_sar/data_splits.py` 的全部 split、`runs/` 下所有 JSON 里的
`seeds/seed/failure_seeds/control_seeds` 字段（扫了 2 975 个 manifest）、以及 `runs/` 下所有
产物与 stream 文件名里的 `s<数字>`。排除并集 **833 个 seed**，其 SHA256 摘要写进 manifest
（`c519a3f920ee…`），并且断言 `overlap_with_used == []`。

结果：`500000..500399`（400 个，一个都没被撞掉，因为历史全部落在 2xxxx–3xxxx 与 3 亿量级）。
旧环境那份 400（30000 段）不与本块合并、不重跑、不作本块的任何分母。

## 三、指标

**主指标**：逐 seed 的"按时完成率"配对差 `D − C`，率 = 该 episode 内按时完成的检出目标 / 检出目标。
区间：seed 级配对 percentile bootstrap，B=10 000，RNG 20260918，95%。只有主指标做推断。

**次指标**（描述性 + 名义区间，标注为探索性）：按时完成数、检出数、过期目标数、未解决目标数、
平均完成时刻、总行程、丢失工时、覆盖复位次数、空闲 robot-tick、最长空闲串、重规划决策数、
源任务校验拒绝数与恢复时间（`release_refusal_recovery_ticks_mean/_max`、
`release_refusal_unrecovered`）。

**正确性指标**（门槛，见第五节）：`false_credit_task_ticks`（真实幽灵工时）、
`dual_roster_robot_ticks`（重复能力贡献）、`executed_releases_that_uncovered_the_stage`
（错误释放，按边际贡献判定）、`closing_released_robot_assigned_elsewhere`（旧任务误释放他任务机器人）、
`duplicate_command_reset`（重复命令重置工时/名册）、`illegal_executed_assignment`（非法执行）、
`releases_without_future_information`。

**单位**：task-tick = 一个任务的一个阶段在一个环境 tick 上推进一次；robot-tick = 一台机器人一个 tick；
target = 一个被检出的救援目标（episode 内唯一）；event = 一次离散发生；
release attempts = 规划发布 + 规划期拒绝；accepted releases = 边界放行的跨任务改派命令数。

## 四、记录与重放

- 录制：`--record`，每个 seed 一次真实 PILOC 在线搜索、协调器关闭、
  **`task_membership_consistency=True`**，写 `streams/s<seed>.json`。两臂读同一份文件。
- 每个 episode 的 payload 记 `stream_sha256`，`stats_v2` 核对配对两臂的 stream 摘要必须相同。
- 环境变体进 run signature：`task_membership_consistency` 参与 `config_hash`，且 arm 串带
  `+memfix`（本轮新加，之前两环境共用一个标签）。
- 输出隔离：`block/C/` 与 `block/D/`；技术 smoke 走 `block_smoke/`（seed 600000+，
  与独立块的清单永不相交），smoke 的聚合不参与判定。
- 单主进程做 preflight（`--preflight`：构造断言 + stream 摘要）与汇总（`--aggregate`：
  清单闭合、final_status、commit、配对、stream 摘要、完整 final record）。
- 分片一次：任务清单 = manifest 的 400 个 seed，按下标取模分到 6 个 worker，一次成型；
  worker 只写自己那一份；父进程用 `Popen.wait()` 收真实退出码，非 0 即整批判失败并返回 1
  （不经过管道，不会被 tail 掩盖）。
- 原始流（jsonl / streams / 决策轨迹）不入 Git，但按 seed 定位、有 SHA、可用同一
  配置与 seed 复跑校验。

## 五、门槛与判读

1. **正确性优先**：两臂任何一次 `false_credit_task_ticks`、`dual_roster_robot_ticks`、
   `executed_releases_that_uncovered_the_stage`、`duplicate_command_reset`、
   `illegal_executed_assignment` 非零 → **本版验收失败**，逐事件定位，不用较高完成率抵消。
   （`releases_without_future_information` 单独报，不判失败：它是信息缺失的哨兵，不是有害事件。）
2. 主指标 CI 下界 > 0 → 可以在该测试分布上声明性能提升；CI 跨 0 → 报"证据不足"，
   **不等于两臂等效**；D 明确退化（配对差为负的 seed 显著）→ 逐个分析新增失败，不调参追测试集。
3. 报告所有 candidate 新增失败（`seeds_d_worse` 全列出），不因"直接原因是环境超时"剔除。
4. 零检测与技败处理按 `stats_v2` 顶部事先写死的规则：两臂都无检出的 episode 对剔除并公开计数；
   技败（未跑完 / commit 或 stream 不符 / 缺产物）剔除并按 seed 列明原因；
   两臂都满的 seed 差为 0，是真实测量，保留，并报其占比以说明可用信号量。
5. 只比较同一修正环境下的 C vs D。旧环境结果（A/B、上一轮 400、stress 块）作为历史诊断证据保留。

## 六、条件性的第八节

只有本块通过正确性门槛且没有未解决退化之后，才跑 60 个额外未使用 seed 的真实 PILOC
在线搜索（两臂同一修正环境、同一现有模型）。事先声明它检查什么：搜索与分配解耦的不变量成立时，
它检验的是**系统集成与真实搜索输入**，不重复声称证明任何双向反馈能力。
seed 取 500400..500459（同一排除规则，与本块 400 不重叠）。
