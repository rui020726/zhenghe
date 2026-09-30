# ROUND34 执行报告：任务奖励时间归属与 checkpoint 身份协调

日期：2026-09-30

分支：`fix/comm-task-reward-event-timing`

执行范围：静态代码复核、既有记录核对、纯合成事件与恢复边界测试、本地版本化修复。未启动仿真、未训练、未联网、未改写旧产物。

## 执行依据与前置核验

- 在 `D:\code\整合\zhenghe` 核实仓库和 ROUND33 头 `c7d47d25e638cbe3728cf36e3a22b9cfc563a353`；ROUND32 头 `01e5a677cd18a7abfdb97585315ba05251d4e0de` 是其祖先。新分支从 ROUND33 头建立，未重置或覆盖已存在分支。
- ROUND33 文件交付和 checkpoint 原件均在本机。1040 个 row JSON 与 768 份训练 tape 的清单哈希均与 ROUND33 审计记录相同。
- 本机有 Python 3.13.5 和 NumPy 2.1.3，无可用项目虚拟环境；`import torch` 因缺少 PyTorch 失败。没有安装依赖或下载文件。因此不能反序列化 `.pt`、计算真实参数张量摘要，也不能运行模型 forward 或 native likelihood。
- 前置核验和时间轴记录见 `EXECUTION_BASIS.md`。ROUND32/33 既有文件、日志、tape、索引及旧报告均保留。

## 1. 真实事件顺序与时间错位判断

按 `Task03Adapter.step`、`JointSAROrchestrator.step`、`v2_run._install_taps`、`v2_trainer._bundle` 的实际代码，顺序为：

| 顺序 | 动作 | 时间与状态 |
|---:|---|---|
| 1 | `CommRelay.tick` 采样策略并推进 relay tick `t`；现有成本 tap 读取计数增量。 | 若有 tape 动作行，该行标记 `t`。 |
| 2 | adapter 调用 raw `env.step(dt)`。 | 环境推进到 tick `t` 的步末。 |
| 3 | `env.step` 返回后，adapter 构造 `ProgressEvent(observed_at=env.current_time)`。 | 事件时间是推进后的环境时间。 |
| 4 | orchestrator 同步发布事件；registry 订阅者先处理，后注册的奖励观察者随后运行。 | registry 已提交接受/拒绝后的状态；回调中的 `clock.now` 仍是步前值。 |
| 5 | orchestrator 推进 `clock.now`。 | 时钟到步后时间；观察者不修改时钟。 |
| 6 | 后续策略采样开始。 | 本 tick 的奖励已在此之前记入奖励流。 |

旧 `v2_run._install_taps` 在 raw `env.step` 返回时立即读取 registry。此时 adapter 尚未创建并发布本次 `ProgressEvent`，所以 registry 仍是旧状态；下个 tick 才看到完成状态，并以新的 relay tick 写奖励流。`_bundle` 的“奖励 tick 当时或之前最近的动作行”规则本身没有问题，问题出在奖励流写入过晚。

原始记录的一条可逐事件解释实例：`ACTIVE_ROWS`、seed 11、episode 0、scenario seed 1300027、目标 `piloc-e1-target-3`。生命周期记录为 `observed_at=8.0`、`expired=false`、事件 ID `task03-progress:e0:piloc-e1-target-3:8`；tape 有 step 8 和 step 9 两行；归档 episode row 将该单位记在 `task_stream[9]`。结合实际发布顺序，这是 tick 8 事件被旧 tap 在下一 tick 看见的具体实例。源字段、tape 与行记录支持 ROUND33 的时间错位结论。row/tape 清单哈希也与 ROUND33 一致；ROUND33 报告的 4363 个事件及其 2821/1542 分布可沿用为同一源与数据身份下的审计统计，但本轮未重扫全部事件。

结论：**时间错位经真实代码和原始记录确认**；这不是只由“两个数字相差 1”推断，也不需要全局减 1 时间戳。

## 2. 最小修复与回归结果

新增 `v2_reward_timing.py::PublishedProgressRewardCollector`。`v2_run._install_taps` 保留原 relay 成本计数采集，只取消包裹 raw `environment.step` 的任务奖励扫描；改为在现有 event bus 上、registry 订阅者之后观察已发布的 `ProgressEvent`。collector 按目标身份和 registry 完成状态转换做幂等判断，并检查事件/快照完成状态、接受状态、过期标记和 `observed_at <= deadline`，然后在当前 relay tick 写入奖励。它不发布事件、不改 registry、不推进时钟、不抽随机数；episode 结束时解除订阅。没有改环境状态转移、调度或发布次序。

`_bundle` 仅调整为保留 tick 0，使首动作前的 prefix 不会被丢弃；原有“最近的过去动作”分桶规则不变。成本路径、奖励数值、目标函数、观测、mask、网络、GAE、预算和训练量均未改。

修改前使用实际旧 tap、production event bus、registry 和 `_bundle` 的针对性合成测试，9 项用例全部按预期失败：同 tick 完成未捕获、下个动作行抢到前 tick 奖励、无新动作时源 tick 仍迟一拍、末 tick 漏捕获、多目标整体迟记、deadline 检查缺失及 prefix 倒灌等。修改后同组覆盖扩为 10 项并通过，其中含事件重复、部分阶段、超时、末 tick 和被测桶函数接线；手算期望表独立列在测试中，没有用 `_bundle` 生成期望答案。

passivity 用例检查 Python RNG 状态、event bus 历史、registry 接受/拒绝计数、fixture 消息和成本 counter delta。它们在合成夹具中通过；不能据此宣称真实仿真被动性已验证。

## 3. 训练版本边界与旧结果解释

新的训练版本为 `constrained-cost-v2/published-completion-event-at-relay-tick/2`，奖励采集版本为 `published-completion-event-at-relay-tick/2`。新 checkpoint payload 与 training state 均写入两个版本字段。真实恢复入口 `v2_trainer.load_checkpoint` 在 controller/optimizer/RNG 恢复前拒绝缺失或不匹配版本；评估入口使用显式的 `load_checkpoint_read_only`，只读加载 controller weights，不恢复 optimizer 或 RNG，并标注旧文件为 `legacy_unversioned_capture_semantics`。未来 freeze manifest 也记录版本身份。两项 mocked 边界测试通过，没有调用 `optimizer.step`。实际旧 `.pt` 读取仍未运行。

历史解释范围：旧运行中的任务数、字节、动作和模型行为仍是其已执行轨迹的观察值；旧训练奖励时间契约有一 tick 采集限制。ROUND32 归约比较是在旧采集语义下得到，不能直接外推到修复版本训练；该偏移本身也不能解释全部学习失败。旧 verdict 未改写为修复版本结论。

## 4. 18 个 checkpoint 的只读协调

全部 18 个 `.pt` 文件都计算了当前文件字节 SHA256，并与 index 比较；先核对 6 个 b0 初始化点，再核对 b15/b31。之后按 sidecar 的显式 `batch` 字段和文件名后缀关联 batch 记录，再核对 mechanism、episode/scenario row、tape 行数、lambda、更新计数及保存的参数摘要声明。

ROUND33 的 12 个 b15/b31 “冲突”由映射脚本把按文件名排序的数组误当成 batch 编号字典引起：数组位置 15 实为 b22，位置 31 实为 b9。按错误位置选择可重现 ROUND33 报告的 18 组 episode 列表；显式 batch 关联后，b0、b15、b31 共 18/18 的 index、batch sidecar、机制诊断和 tape 元数据声明一致。原 ROUND33 输出保持不动，修订解释记录在 `ROUND33_CHECKPOINT_MAPPING_ERRATUM.md`。

| 验证层 | 结果 | 解释 |
|---|---:|---|
| 文件字节 SHA256 | 18/18 通过 | `.pt` 字节与 index 声明相同。 |
| 元数据协调 | 18/18 通过 | 修正映射后，index、batch、mechanism 和 tape 的身份字段与计数一致。 |
| 参数内容 / 模块参数摘要 | 不可用 | index 中的摘要声明相互一致，但没有反序列化 tensor 来重算。 |
| payload 元数据 | 不可用 | PyTorch 不可用。 |
| forward / native likelihood | 未运行 | PyTorch 不可用；sidecar 记录的 18/18 likelihood gate 是归档记录，不是本轮重算。 |

每个文件的路径、字节哈希、arm、seed、batch、阶段、episode/optimizer 计数、lambda、参数摘要声明、训练版本声明、对应 sidecar/tape/mechanism 来源及每层状态见 `CHECKPOINT_IDENTITY_RECONCILIATION.json`。不同命名空间的 file SHA256、whole-controller 参数摘要、module digest 和 batch 身份字段没有互相替代比较。18 个文件均为该 batch 的 pre-update checkpoint；记录中的模型摘要属于待复算声明，不代表内容验证通过。

## 5. 测试与未执行项

- `test_reward_timing_regression.py`：10 项通过，包含实际采集与 `_bundle` 接线的合成事件用例。
- `test_checkpoint_version_boundary.py`：2 项通过，使用 mocked checkpoint payload 验证旧版本拒绝恢复和显式只读标签。
- 修改过的 runner、trainer、helper、freeze、测试及 reconciliation Python 文件：`py_compile` 通过。
- 修改前旧实现的定向反例：9 项均复现预期失败。
- 没有运行完整测试套件、SAR/JointRun 仿真、训练 smoke、真实 checkpoint 反序列化、真实 forward 或 native likelihood。

## 6. 交付状态及最终计数

ROUND34 的版本化实现、契约、回归记录、恢复版本清单、checkpoint reconciliation、ROUND33 勘误和总报告均按问题分主题提交在本地分支。执行依据单独提交；未推送到 GitHub。

| 活动 | 数量 |
|---|---:|
| 新环境启动 | 0 |
| 新流 | 0 |
| 真实训练更新 | 0 |
| 合成 optimizer 更新 | 0 |
| 旧 checkpoint 写回 | 0 |
| 网络尝试 / 推送 | 0 |

### 面向下一步决策的六项答复

1. **时间错位确认？** 是；真实发布顺序和一条原始生命周期/tape/reward 记录相互印证。ROUND33 汇总统计沿用时有同一数据清单哈希支撑，但未在本轮全量重扫。
2. **修复位置与执行行为？** 新 collector 位于奖励 runner tap，挂在现有事件发布之后、下次策略采样之前；不改事件产生/发布/调度行为，不改成本采集路径。
3. **测试范围？** 10 项合成奖励时序接线测试、2 项 checkpoint 版本边界测试及 py_compile 通过；没有真实仿真、真实 checkpoint load、forward/likelihood 或完整套件验证。
4. **18 个 checkpoint 状态？** 18/18 文件哈希通过；18/18 修正映射后的索引和归档元数据声明协调通过；18/18 参数内容、模块摘要、payload 内容、forward/likelihood 均未由本轮验证。
5. **旧结果限制？** 旧训练奖励归属有一 tick 采集限制；ROUND32 归约比较只代表旧语义；偏移不能单独解释全部学习失败，旧 verdict 保持原样。
6. **是否可进入小规模真实接线验证？** 代码和合成接线测试已具备候选基础，但本轮没有真实仿真被动性验证；按执行提示词，下一步 smoke 需另行批准，本轮不启动。
