# 成员一致性收尾（MEMBERSHIP_FINAL_CHECK）

日期：2026-09-18 起点：`fix/task-membership-consistency @ d96b9deb9448f41859e3826e7e05d80b6aaa10e6`
（远端同名分支 HEAD 与本地一致，已 `git fetch` 核对。）

本轮只回答三件事：C 臂剩下那 1 个幽灵 tick 是什么；修正环境的三条正确性门槛是否真的过；
以及报告里那些数字各自的单位。结论：**那 1 个 tick 是测量窗口造成的误判（分类 C），
不是缺席机器人贡献能力；按工时更新时刻重测后，C/D 两臂三项门槛在 32 个 episode 上全部为 0。**

## 一、那 1 个幽灵 tick：定位与分类

`runs/corrected_env_final_20260918/ghost_localize.py` 重放 dev seed **20008**（A 与 C 两臂各报
1 tick，逐位相同，说明它既不是改派残留也不是释放机制造成），在同一个 tick 上读三次在场状态：
`task_update` 入口、`task_update` 出口、整个 step 之后。

| 字段 | 值 |
| --- | --- |
| seed / cell | 20008 / A 与 C 完全相同 |
| task | 内部 id 3，stage_index 1，需求 `[1,0,0,1,0]` |
| tick / 环境时刻 | 第 132 tick，t=132.0 |
| 工时 | `time_left` 8.924 → 7.924（本 tick 真实推进 1.0） |
| 推进时刻的 roster | `[2, 4]` |
| 两人的状态 | 均为 `status=working`、`current_task=3`、`distance_to_task=0.0000` |
| 环境推进工时所用能力 | `get_abilities([2,4]) = [1,1,1,2,0]` |
| 独立按真实在场计算 | 同一集合（两人都真在场），缺口 `[0,0,0,0,0]` |
| step 结束时的 roster | `[2]` —— 机器人 4 在 `_check_working_timeouts` 里随机超时离岗 |
| step 结束时的缺口 | 剩余供应 `[0,0,1,1,0]` → 缺口 `[1,0,0,0,0]` |

分类：**C（测量器在释放与日志顺序上误判）**，不是 B。旧探测器把"整个 step 之后"的在场状态
当成"工时更新时刻"的状态，而 `task_update` 之后还会执行 `_check_working_timeouts`
（`task_allocation/old_task_env.py:1112 → :826`）：那台真在现场的机器人是在工时推进*之后*
才被超时释放的。工时推进的整个窗口里，被计入能力的两名机器人**都真实在场**，
没有任何缺席或错误指派的机器人提供虚假能力。

A 类（瞬时名单不一致但未用于工时）在这个 episode 里也不成立——入口时刻名单与在场集合相同。

## 二、修正后的口径，以及它没有把任何指标"调"到 0

`runs/corrected_env_final_20260918/ghost_recheck.py` 把三项正确性判据搬到它们各自适用的时刻去读，
**同一次重放里同时记录旧口径**，因此新旧可以按 episode 对账，旧记录（`runs/membership_consistency_20260918/CELL_SUMMARY_*.json`
与 `cells/`）原样保留、不覆盖：

| 判据 | 读取时刻 | 单位 |
| --- | --- | --- |
| 虚假能力工时 `false_credit_task_ticks` | `task_update` 入口（工时真正计算所用的状态） | task-tick |
| 同一机器人同时为两任务供能力 `dual_roster_robot_ticks` | 每 tick 全部名册 | robot-tick |
| 旧任务结束/推进释放了 assignment 在别处的机器人 | `task_update` 入口 roster vs 出口 roster | 事件 |
| 释放确实降低真实覆盖 `executed_releases_that_uncovered_the_stage` | 离开发生的那一 tick，按边际贡献判定 | 事件 |

第四项也修了口径。原先用"上一 tick 覆盖完全、这一 tick 不完全"判定，会在**任务刚好推进到
更难阶段**时误报：seed 20083 tick 15（task 2，释放机器人 3，roster `[5,3]→[5]`，缺口 `[0,1,0,0,0]`）
就是这么被记上的。查真实数字：机器人 3 的能力是 `[1,0,0,0,0]`，机器人 5 是 `[1,0,0,1,1]`，
阶段需求 `[1,1,0,1,0]` —— 有 3 和没 3 的被覆盖单元都是 2 个（dim 0 由 5 重复供给，dim 1
从来没人能补）。改用"该成员在岗是否会*严格多覆盖*一些"后它不再计数，而 B 臂的真事件
（例如 task 1 只有机器人 1 在场，roster 还挂着 working-on-task-2 的 0 号与 moving-to-task-3 的 2 号）
照旧计为 141/183 tick。旧状态判据保留为 `departure_into_an_uncovered_stage` 单独一列，
并注明它含阶段推进。排除 seed、放宽幽灵定义都没有做，也不可能做：两口径同时产出。

### 修正口径后的四格（32 个 episode：15 dev + 17 stress，各 4 臂）

| cell | 环境 | 控制器 | 虚假能力 task-tick | 旧口径 task-tick | 双名册 robot-tick | 结束误释放在别处 | 执行后降低覆盖的释放 | 释放（发布/执行） | 按时/检出 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 旧 | baseline | 0 | 1 (dev) | 0 | 0 | 0 | 0 / 0 | 81/90 dev, 131/131 stress |
| B | 旧 | candidate | **141 dev / 183 stress** | 156 / 195 | **597 / 538** | **85 / 82** | **11 dev / 12 stress** | 104→51 / 85→42 | 90/90, 131/131 |
| C | 修正 | baseline | 0 | 1 (dev) | 0 | 0 | 0 | 0 / 0 | 81/90, 131/131 |
| D | 修正 | candidate+边界校验 | **0** | **0** | **0** | **0** | **0** | 105→52 / 118→44 | 90/90, 131/131 |

三条门槛（工时更新无虚假能力；同一机器人不同时为两任务供现场能力；旧任务结束不释放已在其他
任务工作的机器人）在 C 与 D 上全部满足，计数为 0；旧环境 B 上三条都不满足。
A 与 C 逐列相同，说明环境修复没有改动 baseline 的任何结局。

## 三、checkpoint 结论

> **修正环境下的开发回放已验证；独立泛化表现随后在同一修正环境、400 个未使用 seed 上完成测量
> （见 `CORRECTED_ENV_HELDOUT_VERDICT.md`）；在线搜索链路上的 60 seed 块对该难度没有可分辨信号
> （见 `CORRECTED_ENV_SYSTEM_VERDICT.md`）。**

本轮没有训练、没有改权重、没有动 checkpoint 文件；`shared_mappo_model_21_3100.pth` 与
`ctde_final_7_1.pth` 的 SHA256 记在 `ARMS_CONFIG_TRUTH.json` 与 `RELEASE_MANIFEST_V2.json`。
开发块覆盖 15 + 17 个 episode（含 5 个旧失败 seed 与 19 事件所在 seed）——那是定位与门槛验证，
不是泛化声明；泛化声明只属于那 400 个未使用 seed 的配对块。

## 四、本轮生产变更（全部默认关闭）

| 变更 | 位置 | 默认 |
| --- | --- | --- |
| 释放携带"当时相信的支援名单" | `contracts.ReleaseProvenance.supporting_members`、`coordinator._joint_redundant(supports=)` | 只写不改判 |
| 执行边界按批逐条校验支援是否还在 | `assignment_gate._inspect_release`，新拒绝原因 `stale_release_support` / `stale_release_agent_moved` | 仅在 `--check-release-source-stage` 下接线 |
| 下一阶段保护与三类"仍需要它"计数 | `coordinator`：`next_stage_needs_this_robot`、`stage_needs_this_robot`、`not_credibly_on_site`、`releases_without_future_information` | 记录，不改判 |
| 执行记录里可直接数出释放 | `lifecycle_log.log_assignment` 增 `release_from_task` / `release_stage_index`（无释放的行逐字节不变） | 记录 |
| 环境变体进入 arm 标签 | `allocation_trace.arm_name` 后缀 `+memfix` | 仅 `--task-membership-consistency` 时 |

测试：`integrated_sar/tests/test_release_execution_boundary.py`（13 项，六场景各一，含
"没有 roster 回调时行为与旧 gate 完全一致"），`test_redundant_release.py`、
`test_release_source_gate.py`、`test_coalition_reservation_metrics.py` 相应补钉；
环境侧新增两条方向相反的钉：修正环境下"已结束的任务不会把手指到别的任务上的机器人拽回来"
（`test_a_closing_task_does_not_release_a_robot_working_on_another`），以及旧记法下这个伤害
确实发生（`test_the_old_bookkeeping_let_a_closing_task_reach_back_for_the_robot`）。两条都
把等待长度控制在随机工作超时与新任务自身工时都够不着的范围内，免得断言变成对时钟的断言。
全量测试与 smoke 见"验证"一节末尾。

## 五、验证

* 全量测试 **551 通过（2 跳过）**（`python -m unittest discover -s integrated_sar/tests`）。
* smoke `python -m integrated_sar` → `final_status=completed`。
* 独立块与在线块的技术完整性：800 + 120 个 episode，0 技术失败，配对 stream 摘要逐 seed 一致，
  构造断言在每一次构建时执行（`--preflight` 输出记在 `PREFLIGHT_smoke.txt`）。
* 默认开关不变：`coordinator_policy`、`sticky_commitments`、`release_redundant_members`、
  `suppress_recommand_own_task`、`check_release_source_stage`、`task_membership_consistency`
  仍全部默认关闭；未合并、未重训、未删除任何旧产物。
