# ROUND29 报告：发送端期限编码审计与最小观测原型

分支 `analysis/comm-deadline-encoding-audit`（起点 ROUND28 完整头
`21250b5eb093dd1a7f97cd06e9993ef1ac6abab0`，与本消息报告头 21250b5 一致）。
本轮：**0 环境启动、0 新流、0 训练、0 optimizer.step**；未运行任何可能构造仿真的生产
套件/smoke；未修改生产默认值、策略、预算、mask、ACK、奖励、阈值与权重；未把新编码接入
旧 checkpoint；发布状态 `PENDING_NETWORK`（本轮不要求网络重试）。

## 1. 场景归属更正（保留旧报告，仅新增更正）

只读核对三块场景与既有记录一致：

| 块 | 角色 | 核对结果 |
|---|---|---|
| 1300000..1300047 | 原训练块 | ROUND28 用前 16 个场景（1300000..1300015）；16 个 baseline 全部 6/6 按时 |
| 1301000..1301015 | ROUND22/24 **开发**块（"仅作开发历史"） | seed33 最终 greedy 逐场景按时数：11 个 6/6，5 个有失败（1301001=5、1301002=2、1301003=5、1301013=5、1301015=5），行已在 ROUND24 rows 内，无需新启动 |
| 1400000..1400063 | ROUND25 评估块 | 已报告失败场景 1400031（learned greedy 4 按时/2 过期）在此块内 |

更正：ROUND28 报告与判定 q4 把 1301000..1301015 称作"ROUND25 开发块……含按时损失记录"。
该归属有两处偏差：它是 ROUND22/24 开发块（ROUND25 文档明言"仅作开发历史"），且**不是**
包含 1400031 失败的场景块——1400031 在评估块 1400000..1400063。因此不能期待从该块获得
1400031 一类"可救回的失败标签"；它自己的失败是开发历史场景的失败实例（5/16，盘上已有）。
机器可读登记见 `SCENE_BLOCK_CORRECTION.json`；ROUND28 旧报告与判定原文保留未改。

同时登记的结论边界：满分基线没有提高按时数的空间但干预仍可造成退化；ROUND28 零任务变化
不证明首报时机无效；96 个目标不是 96 个独立场景；成本变化已经存在（7 更便宜/83 更贵/
6 持平），无须为制造任务收益标签挑失败场景。

## 2. 真实编码路径（源码均已逐字节核对）

调用链 `SendItem/SendObservation → 特征构造 → collector → 网络输入`：

1. **来源与单位**（`integrated_sar/comm.py:819-820`）：`deadline=float(state.deadline)`，
   `remaining_time=float(state.deadline) - now`。`state` 是 registry 快照
   （`joint_driver.py:1058 snapshot_of=registry.get`），`now` 是仿真时钟
   （`joint_driver.py:1072 now_of=lambda: clock.now`）。期限绝对量 =
   检测时刻 + deadline_offset（`comm.py:1146`），单位为**步**，动作前可用，是发送端合法
   可见量（relay 决策时从 registry 读的就是它）。
2. **特征构造**（`integrated_sar/comm_learning.py:74-121`，19 维 `comm-send-state/1`）：
   - 第 12 列（index 11）`remaining_time_over_horizon` = **`clip(remaining / 50, -1, 1)`**
     （:113，`URGENCY_SCALE_STEPS = 50.0`，:70）——分母是 50，不是 horizon；
   - 第 13 列 `urgency_now` = `1.0 if 0 <= remaining <= 25`（:114，另一条近截止阈值）；
   - 第 19 列 `step_over_horizon` = `min(1, step/1000)`（:120，`EPISODE_HORIZON_STEPS=1000.0`，
     :71）。
3. **collector**（`runs/comm_budget_action_mask_fix_20260922/budget_masked.py:177-179`，
   ROUND24–28 经 StageBCollector 子类复用）：`rows = [features(obs, item) ...]` →
   `np.float32` 矩阵 → `controller.forward`。**无二次缩放、无截断、无特征替换、无布局差异**；
   float32 只是数值精度转换。归档 collector（`runs/comm_tight_budget_20260919/onpolicy.py:
   157-159`）同形状。
4. **网络输入**（`runs/comm_budget_pilot_20260919/train_pilot.py:54-75`）：19 维直接进
   `Linear(19,64) → Tanh → Linear(64,64) → Tanh` + 动作头；**输入侧无归一化层**。
5. **horizon 是运行前已知配置**：`EPISODE_HORIZON_STEPS=1000.0` 是状态表示声明常量
   （:67-69 注释："declared constants of the state representation, not tuned per run"）；
   运行配置 `steps=1000`（`v2_common.py:110`、`r28_common.py:79`）。旧 env 包装器里的
   `horizon=float(summary.get("steps", 0))` 全部属于 `belief_age_accounting` 分析指标，
   **不在 actor 输入路径上**。
6. **与 ROUND28 启动身份的关系**：本轮读取的 18 个生产/导入模块与 ROUND28 `CODE_IDENTITY.json`
   记录**逐字节一致**（含 `comm_learning.py`、`comm.py`、`budget_masked.py`、`train_pilot.py` 等）；
   checkpoint `v2_train_s33_ep128.pt` sha256 亦与记录一致。规划端提示的历史提交
   `64cddcff…` 不在本地仓库，故一切结论以当前已核验字节为准。

**性质判定**（区分三类）：这是**命名不准确 + 有意采用的声明式有损编码**，不是实现违反契约。
依据：模块注释把 `URGENCY_SCALE_STEPS=50` 声明为状态表示的常量并配套 `urgency_now` 二值特征；
没有任何文档承诺期限列无损。列名叫 "over_horizon" 却以 50 为分母，是名称与计算不符。
仅因丢失信息就把旧实现宣布为正确性 bug 会过度断言；但"该列在首报点上不携带任何期限信息"
是可以证明的事实。

## 3. 信息损失量化（`ENCODING_AUDIT.json`，分来源不合并）

三个来源共 102 个首报决策点。原始期限由 registry probe 日志的 detection 记录
（`timestamp + deadline_offset`，发送端决策时实际读取的字段）与已实证的时钟规则
`now = tape step − 1` 重建；该规则先用 scene 1400010 全部 197 个候选行逐行复现编码值验证
（0 失配），再逐点交叉验证"存档特征 == clip(重建原始期限/50)"（三来源 102/102 一致）。

| 来源 | 样本 | 原始期限范围（步） | 原始不同值 | 旧编码列不同值 | 饱和占比 | 新列不同值 |
|---|---|---|---|---|---|---|
| A ROUND28 96 干预点 | 96 | 62.93–177.56 | 95 | **1**（全为 1.0） | 96/96 | 95 |
| B ROUND27 target-0/5 干预点 | 2 | 65.96–96.53 | 2 | **1** | 2/2 | 2 |
| C ROUND26 四改善场景首报记录 | 4 | 74.28–158.56 | 4 | **1** | 4/4 | 4 |

- 编码碰撞：A 源 95 个不同原始期限值（全精度；按 6 位小数舍入为 94 个）全部折叠为同一个
  1.0（含 62.93 与 177.56）。
- 从编码值反推原始期限：102/102 不可恢复（饱和到 +1 只给出"≥50 步"的下界）；可恢复的
  途径只有 registry 期限这一发送端合法通道（本轮即用它），不是从饱和值反推。
- 19 列逐列变化（A 源）：变化列为 `has_future_requirements`(2)、`queue_backlog_targets`(6)、
  `budget_remaining_fraction`(4)、`step_over_horizon`(42)；恒定 15 列。恒定原因分类：选择条件
  （first_emission/sent_count/since_last_emission_time/pending_event/changed_view/
  *_changed/progress_delta_abs/revision_delta 等，首报点无历史发送可比较）、编码截断
  （remaining_time_over_horizon=1.0、urgency_now=0.0）、证据不足（其余逐列记录在案）。
- 描述性并排（不作因果解释）：ROUND28 96 样本 ΔY 全 0（16 baseline 全 6/6 顶格）、ΔC 7 负/83
  正/6 零；四标签 96 发射、96 被接受、60 分配改变、0 任务改变。

## 4. 单列编码原型与纯计算测试

条件满足（截断存在且原始期限合法可见），故执行。

- 原型 `r29_encoding_prototype.py`：旧列 `clip(remaining/50,-1,1)`；候选列
  `remaining / configured_episode_horizon`（默认 1000）**不追加截断**。实现方式为"调用生产
  `features()` 后原位替换第 12 列"，其余 18 列与顺序由构造保证逐字节一致；保留 `urgency_now`；
  不替换生产编码器、不修改 collector/训练器、不从数据拟合尺度；horizon 非正/非有限与
  remaining 非有限均显式 `ValueError`；允许有限负值与超过 horizon 的值。
- 语义版本登记：`comm-send-state/1r29-deadline-over-horizon`（19 维同序，仅第 12 列定义不同），
  并登记兼容规则：旧 checkpoint/优化器状态/replay buffer **不得静默换用**于新语义（同维 ≠ 兼容）。
  本轮只登记接口要求，未修改任何生产恢复/评估入口，也未声称已完成真实恢复入口的验收。
- 测试 `r29_proto_tests.py`（从真实 `SendItem`/`SendObservation` 进入原型，预期答案手算）：
  **167 项检查全部通过，0 失败**。含 horizon=1000 时 remaining=60/100/500 → 旧列均 1.0、新列
  0.06/0.10/0.50；remaining=0/负/超 horizon 语义；非有限与非正显式失败；其余 18 列逐项一致；
  维数/顺序/版本声明；不读取执行侧与时钟字段；纯函数确定性；导入不引入 joint_driver/torch。
- 闭环 `r29_proto_on_data.py`：把 102 个真实原始期限值逐一送入原型代码，新列不同值数与原始
  不同值数**逐源相等（3/3）**，即现有数据中的期限差异被完整保留（新列为 remaining/1000 的
  单射，range 0.063–0.178）。

## 5. 三问结论

1. **是否证实当前编码压缩了合法可见的期限差异？——是。** 编码路径逐行核对：第 12 列实际为
   `clip(remaining/50,-1,1)`（命名不准确；属声明式有损编码，非契约违反）。三个冻结来源的
   102 个首报点上该列恒为 1.0，而原始期限 62.9–177.6 步、并集 101 个不同值（A 源 96 点 95 个
   不同值全部折叠成同一个 1.0）。差异在编码前合法可见，编码后完全消失。
2. **新编码是否在现有数据中保留了这些差异？——是（仅就可证的信息保持而言）。** 候选列
   `remaining/1000` 把 A 源 1 个编码值恢复为 95 个不同值，三来源 102 点经原型代码验证
   3/3 保留；语义版本与兼容规则已登记。
3. **是否证明这些差异能改善学习或任务表现？——本轮不能。** 本轮 0 环境启动、0 训练；原型
   是纯计算，未连接任何网络或评估；没有任何 Δ任务/Δ成本由新编码产生。相关性、个别案例可
   区分或合成测试通过都不构成学习可行性证明。

## 6. 下一轮建议（仅建议，启动额度另行批准）

若批准，建议做**从初值配对训练消融：原编码（`comm-send-state/1`） vs 单列新编码
（`comm-send-state/1r29-deadline-over-horizon`）**。保持网络结构、目标/损失、训练预算与其余
超参一致；唯一差异是第 12 列定义；从 `Controller(training_seed)` 初值起配，不 warm start
（并遵守已登记的兼容规则，旧 checkpoint 不得静默换用）；评估沿用既有流与判据。建议同步把
第 13 列 `urgency_now` 的保留/替换列为预登记的次要问题，不得在本轮证据下顺带改动；
本建议不构成训练授权，也不把任何分支自动转为训练许可。

## 7. 产物

- `EXECUTION_BASIS.md`（执行依据原文，sha256 873df4f8…）
- `SCENE_BLOCK_CORRECTION.json`（场景归属更正，旧报告保留）
- `ENCODING_AUDIT.json`（逐来源统计、字段出处、缺失情况、恒定原因）
- `r29_audit.py`（审计脚本）
- `r29_encoding_prototype.py` + `r29_proto_tests.py`（原型与 167 项纯计算测试）
- `r29_proto_on_data.py` + `ENCODING_PROTO_ON_DATA.json`（原型在真实数据上的保持性验证）
- `ROUND29_VERDICT.json`（执行状态与下一步建议）
