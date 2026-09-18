# 释放执行边界检查（RELEASE_EXECUTION_CHECK）

日期：2026-09-18 起点 HEAD：`d96b9deb9448f41859e3826e7e05d80b6aaa10e6`
对象：`--check-release-source-stage` 下，命令在进入执行器之前到底被校验了什么。

## 一、本轮之前，边界只校验了目的端 + 源任务的 stage

`AssignmentGate._inspect`（`integrated_sar/assignment_gate.py`）的顺序是：
TTL → 目的任务是否登记 → 目的 revision 是否相等 → **源任务 stage 是否仍等于命令携带的那个**。
源任务的**成员**、被释放机器人**自己现在的 assignment**，都没有进入任何判断。

规划侧的联合记账（`coordinator._joint_redundant`）确实保证了一批释放合起来不降低
*被相信的*阶段覆盖（逐个放行、放行后立刻从池里 pop），但那是规划时刻的信息，
边界只能看到世界当前的状态。两者的差就是本轮要堵的洞。

## 二、六场景，逐一给出"边界当时怎么做 / 现在怎么做"

`integrated_sar/tests/test_release_execution_boundary.py`（13 项）。

| # | 场景 | 本轮之前 | 现在 |
| --- | --- | --- | --- |
| 1 | 规划与执行之间源任务换了 stage | 拒绝 `stale_release_stage`，不执行任何命令，机器人继续原工作，进 `pending_replan` | 不变（回归钉住） |
| 2 | stage 没变，但另一名成员先离开 | **放行**（边界不看成员） | 拒绝 `stale_release_support`；拒绝不改变任何执行状态 |
| 3 | 同批两条改派，各自合法、合并抽空覆盖 | **两条都放行**（批内状态不滚动） | 按批逐条校验：先前被放行的释放会从"有效名册"里扣掉，第二条被拒；只执行第一条 |
| 4 | 机器人当前 assignment 已不是命令里的源任务 | **放行** | 拒绝 `stale_release_agent_moved` |
| 5 | 目的端校验失败（过期 / 未登记 / revision 陈旧） | 拒绝并给出原原因 | 顺序与强度不变；新的源任务校验只在目的端全过之后才跑 |
| 6 | 规划时 belief 缺 `future_requirements` | 静默放行，看不出来 | 仍放行（无法区分"最后一阶段"与"没信息"会误伤），但计入 `releases_without_future_information`，随 advice.metadata 发布 |

场景 2/3/4 的修复最小形状：命令多带一个字段 `release_support_members`（规划时刻被计入覆盖的
队友名单，就是 `_joint_redundant` 当时用的那个 on-site 池减去机器人自己）；边界多一个只读
回调 `release_roster_of(task_id) -> 注册表当前名册`。判据是
`支援名单 ⊆ 当前名册 − 本批已放行的离开者`；被释放机器人自己必须在当前名册上。
**不做** revision 的机械相等（注册表每 tick 都在 bump，用它拒绝等于拒绝机制本身——上一轮
实测拒了 979 次却没有换来任何安全）。名册读不到时退回 stage 校验，不因为一个账面字段读不出
就把机器人钉住。

### 权限与责任（显式记录，不隐含扩大架构假设）

- 新增真值只进边界：`release_roster_of` 由 `joint_driver._members_of(registry, ...)` 提供，
  只有 gate 读它。actor/planner/observation 的输入集合不变（不变量 9 的既有测试
  `test_no_registry_truth_is_needed_to_decide_a_release` 仍通过）。
- 反馈通道不变：拒绝只产生 `assignment_rejected` 记录 + `pending_replan` → advice 的
  `replan_agents`，与 TTL/revision 拒绝完全同一条路，没有把"最新真值"变成规划特征。
  若未来要把"源任务当前名册"回传给规划，那是**单独的接口变更**，本轮没有做。
- 命令不被静默升级：`task_revision`、`release_stage_index`、`release_support_members`
  都按规划时刻原样校验，陈旧就拒。

## 三、下一阶段保护：拦住了多少，是否造成长期占用

计数现在是分开的（本轮之前这三种"仍需要它"混在一起不可见）。32 个 episode（15 dev + 17 stress）：

| 计数（每臂合计） | D dev | D stress | 参照：B（旧环境）dev / stress |
| --- | --- | --- | --- |
| `stage_needs_this_robot`（当阶段就缺它） | 2 241 | 3 689 | 2 060 / 3 420 |
| `next_stage_needs_this_robot`（下一阶段需要它） | **230** | **490** | 221 / 462 |
| `absent_from_believed_roster`（名册与需求不同刻，保守不放） | 161 | 181 | 151 / 178 |
| `no_stage_requirement`（没有毛需求向量） | 0 | 0 | 0 / 0 |
| 释放：规划发布 / 边界执行 | 105 / 52 | 118 / 44 | 104 / 51 · 85 / 42 |
| 边界拒绝（`stale_release_*`） | 1（`stale_release_stage`） | 0 | 无源任务校验 |
| 执行后真的降低覆盖的释放 | **0** | **0** | 11 / 12 |
| `releases_without_future_information` | **0** | **0** | 0 / 0 |

（dev 精确值见 `GHOST_RECHECK_dev.json`；stress 见 `GHOST_RECHECK_stress.json`。）

是否长期占用：**没有观察到**。理由是三条一起看——(a) D 在两块的按时率与 C 完全相同
（dev 90/90 vs 81/90、stress 131/131 vs 131/131），(b) 释放机制仍然在跑（stress 44 次执行），
(c) `idle_robot_ticks` / `longest_idle_streak` 与 C 同级，被"下一阶段需要"留下的机器人是**继续在
原任务工作**，不是空等。但要说清楚这只覆盖 32 个 episode 与 `num_goals=6/9` 的负载；
独立块会把 `idle_robot_ticks`、恢复时间和拒绝数按 seed 配对再测一次。

`releases_without_future_information` 在两块的 32 个 episode 里都是 0：环境的
`_future_requirements` 总会投递一个完整维向量（最后一阶段是全零），所以真实运行里
"没有向量"没发生过；这条计数是为通信裁剪/降级配置留的哨兵，不是本轮的豁免。

**"下一阶段也冗余"不等于任意未来时刻安全。** 它覆盖的是 belief 里那一个向量；再往后的阶段
本轮没有任何机制能证明。可用的论证只有：每个决策周期都会重新判一次（`allocation_interval=5`），
并且命令落地时边界会重查 stage + 支援名单——这拦的是"判断已过期"，不是"未来某个阶段会需要"。
这条限制写在这里，不写成保证。

## 四、恢复时间

新记录字段：`release_refusal_recovered` / `release_refusal_unrecovered` /
`release_refusal_recovery_ticks_mean` / `_max`——被边界拒绝之后，那台机器人多久真正离开它
当时被留下的任务（下一次成功改派，或任务结束/释放）。独立块按 seed 汇总。开发块里
`stale_release_*` 只发生了 1 次（dev D），样本太小，恢复时间留到独立块报。

## 五、结论：执行安全是否满足，能不能启动独立测试

- 六场景全部有确定的、被测试钉住的行为；三种"判断已过期"的路径（换阶段、队友走了、
  机器人自己已经不是那条命令的对象）现在都在边界上被拒，且拒绝无副作用。
- 批量按实际执行后的状态逐条校验（`leaving` 累积），不再依赖"批内快照"。
- 不依赖每 tick 变化的 revision 做相等判断（有专项测试 `test_a_revision_move_alone_is_not_a_refusal`）。
- 32 个 episode 上，四项正确性判据（虚假能力工时、双名册供能力、旧任务误释放他任务机器人、
  执行后降低覆盖的释放）在 C/D 全为 0；旧环境 B 分别 141/183、597/538、85/82、11/12。

门槛满足，独立 C vs D 测试可以启动（协议见 `CORRECTED_ENV_HELDOUT_PROTOCOL.md`，
且在跑之前就提交）。
