# 搜索轨迹与身份核验（B1 被动性 + 历史复现）

机读件：[`SEARCH_TRACE_IDENTITY_REPORT.json`](SEARCH_TRACE_IDENTITY_REPORT.json)、
[`ROUND_VERIFICATION.json`](ROUND_VERIFICATION.json)。
判定件：[正确性判定](SEARCH_DETECTION_CORRECTNESS_VERDICT.md)、
[逐目标时间线](SEARCH_MISS_TIMELINES.md)。

```text
分支   analysis/search-trace-and-policy
起点   80af16b（第十七轮归因分支的末端提交）
本轮   0 训练、0 奖励改动、0 通信配置改动、0 场景难度改动
```

## 一、代码身份

自起点以来被改动的生产文件只有六个，全部是这一轮的记录旁路或它的接线：
`integrated_sar/search_probe.py`（新）、`integrated_sar/adapters/piloc.py`、
`integrated_sar/joint_driver.py`、`integrated_sar/info_probe.py`（只扩了身份字段表）、
`integrated_sar/tests/test_search_probe.py`（新）、`piloc/src/Map/timemap.py`。
`piloc/train.py`、`piloc/configs.py`、`piloc/src/DDQN/Agent.py` 与整个 `task_allocation/`
**一字未动**（`ROUND_VERIFICATION.json → checkpoint_identity` 逐条核过）。
两个 checkpoint 的 SHA256 记在同件里，与起点一致。
第十六轮与第十七轮的目录（协议、结果、脚本）自起点以来 0 处改动。

## 二、B1 被动性：通过（32 项检查全为真）

8 次启动＝两个候选 seed × 日志开/关 × 两次重复，每格 1000 步。开与关之间比较：

| 比较对象 | 结果 |
|---|---|
| 原有生命周期记录（`behaviour_view` 归一后逐记录） | **相同** |
| 检测序列（target, timestamp, discoverer） | **相同** |
| 发射序列（step, target, subscribers, cost, 序号） | **相同** |
| 扣费字节 / 任务结局（discovered, on_time, expired, completed） | **相同** |
| 配置行为字段（去掉路径类字段后的 config 指纹） | **相同** |
| **逐步** 1000 步：动作、奖励、done、`agents_pos`、`goals_pos`、`_py_rng` 状态摘要、`_np_rng` 状态摘要 | **逐步相同** |
| 同条件两次重复（attempt 1 vs 2） | **相同** |

逐步比较用的记录器（`sp_common.StepRecorder`）在两个条件里同样挂载在
`Environment.step` 上——它是判据的尺子，不是第三条臂；探针文件本身不参与逐字比较。

一处必须写明的更正：`run_cell` 早期存下的 `behaviour_sha` 是在
`info_probe.IDENTITY_FIELDS` 补齐之前算的，探针开的两行重复格因此各自带了自己的侧文件路径而
互不相等。身份报告现在**当场重算**两侧记录序列再判等（`recomputed_behaviour_sha` 已并列写出），
并把"尺子只有一份"这件事写进 `integrated_sar/info_probe.py` 的字段表注释里。
这个更正是归一化清单的修正，不是判据放宽：所有格子的逐步位置与 RNG 状态比较与哈希无关。

## 三、历史复现：14/14 格复现，0 格"未复现历史运行"

D1 的七个遗漏 seed、systemloop 的三个遗漏 seed、四个无遗漏控制 seed，各跑一次，逐格核对：

* 地图 `map_sha256`、机器人初始格、目标初始格 ⇒ **14/14 相同**；
* 配置行为字段 ⇒ 无差异（差异只出现在被替换的字段清单里，见下）；
* **完整检测序列**（同目标、同步、同发现者）⇒ **14/14 与归档逐条相同**；
* 任务结局（discovered / on_time / expired）⇒ 逐格与归档一致；
* 未发现目标数逐格为 D1：1,1,2,1,1,1,1（合计 8）；systemloop：1,1,2（合计 4）
  —— 与已交付的两批读数完全对上。

**替换清单**（不是缺陷，是配置版本演化，全部逐条登记在机读件里）：
systemloop 批的归档记录里 `comm_policy / gossip_interval / comm_range / comm_ack /
comm_budget_bytes / comm_max_wait_steps / …` 等字段当时写的是 `null`（字段还没出现或未被写），
本轮按今天数据类的默认值补，并列出补了什么；归档里有而今天不是配置字段的
`allocation_backend`、`search_backend` 之类派生字段被丢弃（它们今天由配置算出）。
补完之后，检测序列仍然逐条相同 ⇒ 这些替换没有改变被比较的行为。

## 四、三视图闭包

每格每步同时写下三件事：环境自己判定并删除的目标 ID、用环境自己的判据原语重数的
"谁能看到谁"、适配器实际发出的 DetectionEvent。

* 被动性 8 格 + 复现 14 格 + 冒烟 8 格：**闭包不一致步数 0，探针异常计数 0**；
* 一个目标被两个机器人同时看到的例子在真实运行里就出现（seed 1100051 第 1 步
  `piloc-e1-target-1` 被 agent_0 与 agent_1 同时看到，事件仍只有 1 条）⇒
  去重语义与"不要求对数等于事件数"这条纪律在同一段真实数据上成立。

## 五、口径与仍存在的问题

* 事后审计用仿真真值（目标隐藏位置、环境判定集合）时，**不得**称作策略当时可见的信息；
* "初始位置可见"不等于"第一次实际探测时仍可见"（`Environment.step` 先移动后判定）；
* 8 连通是合法观测的必要条件，不是四向机器人走得过去的充分证明；
* 仍未解决：本轮的闭包只覆盖 1000 步内环境自己的判定与适配器翻译这两环。
  注册表/中转之后的丢失属于第十七轮已量过的旁路（C1/C2/C3），不在本轮证据范围内；
* 仍未解决：B1 只在 `reward_mode` 为 v2/potential（默认）与 legacy 两条分支上都写了删除记账，
  本轮全部运行走的是 v2 分支，legacy 分支的探针路径只有单元测试覆盖，没有实测运行。
