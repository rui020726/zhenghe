# ROUND35 执行提示词：修复版奖励接线的小规模真实 smoke 验证与恢复边界实机核对

（本节为用户消息原文存档，逐字保留，独立提交。核验结果见 `ANCHORS_AND_ENVIRONMENT.json` 与本文件末尾。）

你是一名没有历史记忆的新执行 agent。本提示词自带全部必要背景。严格遵守授权边界，任何超出边界的动作先停止并报告。

## 一、背景与身份锚点

项目研究通信 SEND/HOLD 策略的保任务降成本训练。上一轮（ROUND34）确认并修复了一个正确性缺陷：旧版任务奖励采集在 raw env.step 返回时立即读 registry，导致完成事件晚一个 relay tick 才进奖励流。修复版改为在现有 ProgressEvent 事件总线上、registry 订阅者之后挂 `PublishedProgressRewardCollector`，在事件发布的当前 relay tick 写入 `task_stream[t]`。

- 执行仓库：`D:\code\整合\zhenghe`（Windows）。分支 `fix/comm-task-reward-event-timing`，本地头 `c43ec5f`，基线为 ROUND33 头 `c7d47d25e638cbe3728cf36e3a22b9cfc563a353`。
- 若该本地路径不存在，改用 GitHub 快照：`https://github.com/rui020726/zhenghe.git` 分支 `handoff/round34-task-reward-event-timing`，头 `f716b683f064222e09c02365649bae9d320cb339`。注意该快照基于 main `a557e80`，不是本地历史的精确推送；若只能用快照，在报告中显式登记这一来源差异。
- 开工前先核验：仓库路径、分支、头哈希；ROUND34 交付目录 `runs/comm_task_reward_event_timing_20260930/` 存在。核验结果记入 EXECUTION_BASIS 类文件。
- 同时核验执行机 Python 环境可用性（torch、numpy 版本），与 ROUND34 报告中"PyTorch 不可用"的旧记录区分开——该限制已解除，本轮所有依赖真实反序列化/forward 的核对项不再有环境豁免理由。

## 二、必读文件（开工前通读）

1. `runs/comm_task_reward_event_timing_20260930/ROUND34_REPORT.md` — 上一轮结论与未执行项。
2. `runs/comm_task_reward_event_timing_20260930/TASK_REWARD_TIMING_CONTRACT.md` — 运行时序表、奖励资格五条、归属规则、手算期望表。本轮全部判定以该契约为准。
3. `runs/comm_task_reward_event_timing_20260930/TRAINING_VERSION_MANIFEST.json` — 训练版本 `constrained-cost-v2/published-completion-event-at-relay-tick/2` 与恢复策略。
4. `runs/comm_cost_v2_rerun_20260928/v2_reward_timing.py`、`v2_run.py`、`v2_trainer.py`、`v2_train.py`、`v2_recover.py`。
5. `runs/comm_task_reward_event_timing_20260930/test_reward_timing_regression.py`、`test_checkpoint_version_boundary.py`。

## 三、授权边界（硬性）

- 环境：执行机已具备 PyTorch 与 NumPy，禁止新建虚拟环境或安装任何依赖。开工时先核验 `import torch` 可用并记录 torch/numpy 版本到执行依据文件；若实际不可用，立即停止并报告阻断，不得擅自安装。
- 真实 runner 启动：最多 2 次短程启动，每次 1 个 seed、episodes 数取能覆盖"同 tick 完成、跨 tick 完成、末 tick 完成、deadline 过期"四类情形的最小配置，单次 wall-clock 不超过 2 小时。两次启动的用途：第一次验证接线与账本，第二次验证 checkpoint 保存/恢复链路。
- optimizer.step：仅允许 smoke 所需的少量真实更新（总计不超过 100 次），唯一目的是让 checkpoint 保存/恢复链路可走通。smoke 产生的任何 loss/回报数字不得作为学习效果证据引用。
- 训练版本：必须使用 `constrained-cost-v2/published-completion-event-at-relay-tick/2` 与 `published-completion-event-at-relay-tick/2`，新 checkpoint 两字段齐全。
- 禁止：修改超参数/网络/目标/奖励数值/成本路径；网格搜索；挑选结果追加训练；热启动或静默续训任何旧 checkpoint；改写 ROUND32/33/34 旧产物；推送 GitHub；把 smoke 通过表述为"学习效果验证通过"。

## 四、任务清单

1. **同 tick 奖励实机核对**：从第一次 smoke 的真实 tape、episode row 和事件账本中，抽取全部完成事件，逐一核对：事件 relay tick == task_stream 写入 tick；`_bundle` 归属符合契约"当时或之前最近动作行"规则；与契约手算表六类情形逐一对上（同 tick、跨 tick 新动作、跨 tick 无新动作、末 tick、首动作前 prefix、deadline 过期）。给出逐事件核对表，不允许只给汇总计数。
2. **尾段与终止奖励**：确认最后一次执行 tick 上的完成事件被当场捕获（不需要后续 step）；deadline 过期事件无奖励且账本无入账。
3. **账本与 registry 一致性**：事件账本的 event id、observed_at、relay tick、registry 接受状态、deadline、credited 标记与 registry 提交记录一致；同一目标无重复入账；重复事件 ID 幂等。
4. **真实恢复边界**：
   a. 新 checkpoint 经 `v2_trainer.load_checkpoint` 真实恢复成功，版本字段匹配；
   b. 篡改 training_version 或 reward_timing_version 的副本被在 controller/optimizer/RNG 恢复之前拒绝；
   c. 真实旧 ROUND32 `.pt`（无版本字段）被 `load_checkpoint` 拒绝；经 `load_checkpoint_read_only` 以 `legacy_unversioned_capture_semantics` 标签只读加载成功，optimizer/RNG 不恢复；
   d. 恢复后继续的后续步与连续运行在同 seed 下可比对的部分保持一致，或如实报告不可比对的环节。
5. **真实被动性证据**：固定 seed 下，给出可观测证据说明 collector 不改变环境行为：动作序列、成本流、registry 接受/拒绝计数、事件发布序列与修复前契约预期一致；collector 自身不消耗 RNG、不发布事件、不改时钟。能直接对比就做对比，不能直接对比就报告间接证据并标注证据等级。
6. **真实 checkpoint 内容层补验（必做，无环境豁免）**：对 smoke 新产的 checkpoint，反序列化张量，重算 whole-controller 参数摘要与模块 digest，与 index 声明比对；旧 18 个 `.pt` 做只读反序列化摘要重算（不改动原文件），与 ROUND34 reconciliation 中"待复算声明"逐层比对，结果单独登记，不改写 ROUND34 原结论；forward / native likelihood 对新 checkpoint 实跑一次并记录，对旧 checkpoint 如侧车记录可比则比对、不可比如实标注。

## 五、交付物

- `runs/comm_reward_smoke_2026XXXX/ROUND35_SMOKE_REPORT.md`：含执行依据（仓库/分支/头/环境/torch 与 numpy 版本）、逐项核对表、被动性证据与等级、恢复边界结果、checkpoint 内容层补验结果、明确的 go/no-go 结论。
- 逐事件核对表、恢复边界记录、checkpoint 内容层比对记录为 JSON，随报告同目录。
- 按主题分提交到本地分支 `fix/comm-task-reward-event-timing`（或快照场景下新建分支并说明），报告最终头哈希。不推送远端。

## 六、退出条件

- 全部核对通过 → 报告"可进入第 3 天冻结训练"，并列出训练冻结前仍需确认的残余风险。
- 任一正确性核对失败 → 立即停止后续启动，给出最小复现、涉及文件与行号、失败事件样本；不带病进入训练。
- 环境阻断 → 报告阻断项与已尝试路径，不交付半成品结论。

---

## 附：开工核验结果（2026-09-30，本机）

| 核验项 | 结果 |
|---|---|
| 仓库路径 | `D:\code\整合\zhenghe` 存在，本轮在其下新建 `runs/comm_reward_smoke_20260930/` |
| 分支 | `fix/comm-task-reward-event-timing` |
| 本地头 | `c43ec5f1a12ef18238d4eed3aa90e3681e9ae58e`（与提示词的 `c43ec5f` 一致） |
| 基线 ROUND33 头 | `c7d47d25e638cbe3728cf36e3a22b9cfc563a353` 是 HEAD 祖先（`git merge-base --is-ancestor` 为真）；ROUND32 头 `01e5a677…` 同为祖先 |
| HEAD 之上四枚提交 | `38e7a3f`、`aa170c7`、`161bc6b`、`c43ec5f`（均为 round34 主题提交），无其它会话提交混入 |
| ROUND34 交付目录 | 存在，11 个文件的逐文件 sha256 记入 `ANCHORS_AND_ENVIRONMENT.json` |
| GitHub 快照 | **未使用**：本地路径与分支/头齐备，因此不存在需要登记来源差异的情形；本轮 0 次网络尝试 |
| Python 环境 | `D:\project_env\python.exe`，Python 3.10.19，torch `2.9.0+cu128`，numpy `2.2.6`，`cuda_available = True`，20 逻辑核 |
| 与 ROUND34 旧记录的区别 | ROUND34 记录"本机无可用 PyTorch（python 3.13.5 / numpy 2.1.3）"。该限制在本执行机上**不成立**：本解释器可 `import torch`。因此本轮所有依赖真实 `.pt` 反序列化、参数摘要重算、forward / native likelihood 的核对项**没有环境豁免理由**，必须实做 |
| 依赖安装 / 虚拟环境 | 0 次新建虚拟环境、0 次依赖安装（未执行 pip） |
| 工作区 | 跟踪文件 0 处未提交改动；15 份其它会话遗留未跟踪文件保持原样，不点名不提交 |
