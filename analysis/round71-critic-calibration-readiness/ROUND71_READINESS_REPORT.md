# ROUND71 准备就绪报告（零消费；未获实验批准）

- 任务：`ROUND71-CRITIC-CALIBRATION-ZERO-USE-READINESS`
- 分支：`analysis/round71-critic-calibration-readiness`（建于已核验 `92571c62b27b0c38d755b501a6d0153b630a08ed`）
- 阶段：`ZERO_CONSUMPTION_PREPARATION_COMPLETE`
- 机器结果：`ROUND71_READINESS.json`；输入计数：`ROUND71_OFFLINE_INPUT_COUNT.json`
- 执行依据（提示词原文存档）：`ROUND71_EXECUTION_BASIS.md`

## 阶段状态（≤250 字）

仅完成私有准备与零消费验收，**未启动任何科研运行**：未收到人类新增实验批准，候选计划登记为
`APPROVAL_PENDING`。已交付校准初始化接缝（标准化 ridge 系数到原始 64 维 embedding 仿射头的精确闭式
变换，只许替换双 critic 头、就地替换被禁止）、资源/入口守卫（按真实 19 维输入行在 forward 前原子
debit、超额前拒绝、失败/重试/重复均计费、重启不清零、1024 分批、现存产物拒绝隐式重跑）、许可缺失
fail-closed、旧 checkpoint 非法恢复拒绝，以及只读离线输入计数。23/23 窄测试通过（纯算术/静态/spy）。
真实初始化等价、梯度、恢复门禁与任何真实拟合/前向/checkpoint **均未执行**。测试全绿不改变阶段。

## 候选离线输入量（只读计数，不重新提取 embedding）

| seed | 联合决策行 | 目标级 19 维输入行（权威） |
| --- | ---: | ---: |
| 11 | 11,009 | 20,013 |
| 22 | 9,762 | 18,691 |
| 33 | 11,275 | 20,224 |
| 合计 | 32,046 | **58,928** |

来源：R68 真实训练场景 1300000..1300047 的 96 个 batch 记录与 384 个 tape（非 180 评估块）。
按 1024 真实输入行分批需 58 批，最大真实输入 batch 1,024。候选离线上限 72,000 输入行，
余量 13,072；该上限只约束离线阶段，未来有许可的训练/评估消费另计。

## 当前准备覆盖

- **校准初始化接缝**：`standardized_to_raw_affine`（`B[j,:]=A[j,:]/sigma[j]`，`b=c-mu@B`）与
  `affine_to_critic_heads`（写成两个 `Linear(64,1)` 头）；`CriticHeadSet.replace_heads` 返回新对象，
  不就地改动；`ParameterContainer.install_critic_heads` 只替换两个 critic 头。设计依据是 R70 已核验的
  真实 critic 输入/仿射头路径，未改模型类。
- **资源/入口守卫**：`ForwardLedger` 在 forward **前** debit 真实 19 维输入行（含 target 展开、失败、
  重复），超额前拒绝，跨实例持久化；`build_batches` 按真实输入行分批，单步超限单独上报不静默拆分；
  `DuplicateOutputGuard` 拒绝隐式重跑；`LicenseGate` 对 8 类受控动作 fail-closed，`CalibrationInitializer`
  在第一道门前即抛错，任何协作者都不运行、不计费、不产出文件。
- **恢复合法性**：`validate_restore_source` 拒绝已训练 ep128、resume 用途、携带已训练 optimizer/lambda/
  训练 RNG 的来源，以及非 5 模块集合。
- **未宣称**：账本为进程内守护，**不宣称物理断电原子性**；spy/no-op 不是实际更新；未新增"回执必须仓库外"
  或"恰好若干 claim 文件"之类的条件。

## 尚未执行的真实门禁（需授权后）

真实初始化等价、真实梯度门（critic->trunk 隔离在校准头下仍成立）、真实恢复/重启门、真实离线 ridge 拟合
与系数变换、任何真实 embedding 提取/前向/claim/checkpoint 写入/episode、首批门禁、逐 seed 128 集训练与
固定 ep128 评估。这些均**未执行**，不能冒充验收已覆盖。

## 零副作用计数

环境构造 0；episode 0；claim 0；真实/合成 optimizer.step 0；PPO 重放 0；神经 forward 0（含 NumPy 等价）；
autograd 0；真实数据统计拟合 0；模型加载 0；打开 .pt 0；训练 RNG 恢复 0；checkpoint 写入 0。
允许并已使用：现有 JSON batch 记录与 tape 路径的只读计数/hash、内存纯算术系数变换、内存 spies、
可写 scratch 根下的测试用私有临时文件（用完即清）。

保护文件 `ROUND46_CURRENT_TASK.json`（`cf3ad198...`）、`ROUND46_R2_V6_STOP_STATE.json`（`93a2f69f...`）
字节未变且未暂存；未跟踪的 r59 日志未暂存。未推送、未合并、未部署。
## 更正轮次（append-only；不改写上方首轮内容）

首轮提交 `53f9eb7d2319f295c7c087ca2edd6606cc3979f6` 的守卫代码经委派源码审查发现 4 处真实缺口，
均在改写前于已提交 blob 中核实存在。本更正轮零消费修复并追加窄测试；首轮 23 项测试收据原样保留。

1. **许可门**：原 `licensed = bool(self.token)` 使任意字符串/字典通过。现要求经 verifier 接受的
   `RealLicense`（完整声明字段）；无 verifier 一律拒绝；test-only spy verifier 不能解锁真实路径；
   `require` 对缺失与 test-only 许可均拒绝。原始 token 在构造时即被拒。
2. **前向账本**：原 `saved.get("rows", 0)` 无身份/上限/校验，`_persist` 为普通写文件。现加载时
   全量校验（缺失/损坏/非整/负值/超上限/上限不符/异身份一律 `LedgerStateError`，绝不静默清零）；
   持久化改为同目录临时文件 + fsync + `os.replace`；变更由同目录锁串行化，残留锁在无明确记录的
   `recovery_basis` 时 fail closed，绝不自动删除；`debit` 先持久化新状态再提交内存，拒绝只记录不退款。
3. **真实 trunk 入口**：原 `run` 按**估计** target_counts 预扣全部批次，真实矩阵/调用数/批大小无守护。
  现唯一计费为 trunk 入口的真实行数：`guarded_trunk_forward` 校验 2 维 (N,19)、N>0、N<=1024，
   原子 debit 真实 N 后对同一矩阵恰好委托一次；估计量仅作 trace 元数据，永不作计费证据。
4. **恢复来源**：原信任调用方布尔标志且空模块列表可通行。现要求完整 identity 块（seed/role/
   structure/module_set/training_version/source_kind）；`r68_ep0_model_only` 为合法 INIT 来源，
   可携带冻结**初始** RNG（不被误判为已训练 RNG），不得携带已训练 optimizer/lambda/进阶训练 RNG；
   `new_full_restore` 仅可携带**本 run 自己**的已训练状态；外来 run、旧四模块、空模块集合、
   ep128、身份不全一律拒绝。

测试：47/47 通过，退出码 0（22 项首轮测试原样通过 + 25 项新增 spy 反例；
`test_legal_new_training_initialisation_is_accepted` 被
`test_superseded_incomplete_initialisation_is_now_refused` 显式取代并附更正说明）。
仍未宣称：测试中的 `ParameterContainer` 只是替身，非真实 R68 接线；默认 `trunk_delegate` 为恒等回显，
未接真实前向；不宣称物理断电原子性。更正轮消费：0 .pt 打开 / 0 模型加载 / 0 RNG 恢复 /
0 checkpoint 写入 / 0 真实或合成 optimizer 更新。阶段不变：候选计划仍 `APPROVAL_PENDING`，
测试全绿不构成批准。
## 最终账本更正（append-only；零消费）

委派审查指出 `ForwardLedger.debit` 的额度判定在锁外用 `self.rows`，锁内未重读持久状态：同一状态文件
构造两个实例（均 rows=0）时，双方都会通过各自过期副本的检查，后一次写会覆盖前一次消费（丢更新）。
本次最小修复，不新增恢复系统：

1. **同锁内取新鲜基值**：额度判定、拒绝、rows/calls/joint_rows 计算与原子保存全部在同一锁内完成，
   基值取经验证的最新**持久**状态（无持久文件时才回落内存态），新状态先落盘才发布内存；
   拒绝与持久化失败同样从新鲜基值计算，绝不覆写另一实例已消费的状态。
2. **不实现锁回收**：原 `_locked(recovery_basis=...)` 会 `os.unlink` 残留锁，可能删除活锁，且
   `last_recovery_basis` 仅内存、不构成持久恢复证据。现任何残留锁一律 fail closed，
   `debit` 不再接受 basis 参数（传参即 `TypeError`）；不做 pid 探测、不结束任何进程；
   正式回收路径需既有合法批准与独立只读的 owner-已结束证据，本轮零消费准备没有需要回收的真实运行锁。

新增反例（旧实现会失败）：同路径 cap=10，L1.debit(6) 后 L2.debit(6) 必须拒绝且持久 rows 仍为 6；
合法 4 使总数达 10 后再扣即拒；重启读回 10 不清零。拒绝不覆写他人 rows/calls/joint_rows；
持久化失败不回写他人消费。失败 spy 调用仍扣一次（原有反例保留），矩阵宽 19/批 1024 约束不变。

测试：50/50 通过，退出码 0（47 项收据中 `test_residual_lock_fails_closed_without_recovery_basis`
被 `test_residual_lock_always_fails_closed_no_basis_can_delete` 显式取代并附更正说明，另新增 3 项
丢更新/拒绝/持久化失败反例）。本轮消费仍全零。阶段不变：候选计划 `APPROVAL_PENDING`，
测试全绿不构成批准；声明层 restore 校验与默认 echo delegate 仍不构成真实 R68 接线。

## 真实运行接线轮（append-only；仍零消费；`APPROVAL_PENDING` 不变）

任务：`ROUND71-REAL-RUNTIME-BINDING-ZERO-EXEC`。本轮把未来校准 provider 绑到**真实** R68 运行源码，
不新增治理框架、不使用 mock 冒充接线，仍不启动任何科研运行。

新增文件（本地提交，未推送/合并/部署）：

- `r71_runtime_binding.py`：真实接线模块（导入轻量：torch/r68 只在受门控函数内惰性导入）。
- `r71_run_calibration.py`：真实 CLI（`--phase binding-acceptance|offline-fit|train|eval`）。
- `test_r71_runtime_binding.py`：22 项窄测试（静态/spy/纯算术）。
- `ROUND71_RUNTIME_SOURCE_MANIFEST.json`：只读来源清单（20 条来源 + 验证回执）。
- `ROUND71_RUNTIME_BINDING_ACCEPTANCE.json`：零消费验收回执（CLI 实跑生成）。

### 本轮接上的真实接缝（静态核验，非 mock）

1. **来源身份**：只读清单登记 canonical trainer/common/controller、冻结 PPO 原语、分头 controller 与
   frozen-early-path PPO 更新、critic→trunk detach、R68 身份/五模块 checkpoint、R68 executor
   （claim/slot/额度/construction permit）、R58 base-runner 接缝与评估接缝的实时 SHA256 与
   函数/行号解析；canonical 三件套 SHA 与 r66_frozen 的 `EXPECTED_*` 常量逐一相符（`0550b99e…`/
   `66d019a3…`/`9b2362ee…`）。6 个 R60 ep0/ep128 模型来源只按已登记期望身份记录，**从不打开或哈希 .pt**。
2. **运行stand-up**：`configure_r71_runtime` 经真实 `r68_execute.configure_and_inject` 起租，
   **只**重绑定额度（582/18/600，已登记未授权）、版本字面量与 checkpoint 存/读到 R71 契约；
   `trainer.ppo_update` 保持 R68 路由更新不变（critic detach + frozen-early-path 数学），
   顺序 mask、loss、对偶更新一律不动；`capture_live_config` 从活体 `common`/`r66_frozen`/`r68_frozen`
   读取 11 个冻结字段，任何漂移即拒。
3. **provider**：trunk 入口取 controller 自己的 `inner.trunk`（生产默认不再是恒等回显——此项更正上一轮
   "默认 trunk_delegate 为恒等回显" 的未覆盖声明；测试仅允许显式注入 spy）；(N,19) 校验、1024 守卫、
   `ForwardLedger` 在 forward **前** debit 真实行；mean-over-targets 汇聚按 R66 顺序（trunk 之后、
   逐联合行）；固定 dual-output ridge（1e-3、全行中心/尺度、常数列 1、截距不惩罚）；经
   `standardized_to_raw_affine` + `affine_to_critic_heads` 精确写回两个 `Linear(64,1)` critic 头；
   `copy_` 只装两个 critic 参数，安装前后逐模块参数 digest 证明 trunk/early/late 逐字节不变，不重缩放、
   不抽 RNG。
4. **载荷角色校验**（`validate_calibration_source`，纯描述、不开 .pt）：R68 ep0 model-only INIT
   （登记 file/parameter/cursor 身份、R68 版本字面量精确相等、禁 optimizer/lambda/RNG）、
   R68 ep128 只读对照（拒绝充当 INIT）、本 run 的 NEW full restore（R71 版本**精确**相等、
   `run_identity` 块逐字段等于活体身份、已训练状态必须齐全——仅声明 `source_kind` 或前缀版本一律拒绝）、
   规则只读对照。
5. **CLI 与门禁**：顺序固定为 licence gate → phase 要求 → model-read approval → configure；
   模型读入点是受控停止边界；`binding-acceptance` 无需 licence 实跑通过，其余 phase 在**任何**
   raw-root 副作用之前 fail closed（无 licence→`LicenseMissing`；test-only→`LicenseInvalid`；
   有 licence 无批准→`ModelReadBoundary`，raw-root 均未创建）。

### 验收与测试

- 验收回执：20 条来源、22 项检查、drift 0；活体模块（r68_execute/r68_frozen/r66_frozen/
  round53_trial_adapter）属性在场；R68 训练版本字面量与 `ROUND68_FREEZE.json` 相等；
  两条 fail-closed 实证均按预期拒绝。
- 测试：**72/72 通过，退出码 0**（原 50 项原样通过 + 新 22 项）。
- 本轮消费仍全零：0 环境构造、0 episode/claim、0 真实或合成 optimizer/autograd/PPO、
  0 神经前向（含 NumPy 等价）、0 真实数据拟合、0 模型加载、0 .pt 打开、0 RNG 恢复、0 checkpoint 写入。
  唯一文件读为已登记 JSON 与 .py 源哈希；活体模块导入仅作属性在场核验。

### 仍未执行 / 不宣称

真实初始化等价（校准头 vs R68 ep0 头）、真实梯度门禁、真实 restore/resume 门禁、真实离线 ridge 拟合
与 embedding 提取、首 batch 门禁、逐 seed 128 集训练与固定 ep128 评估**均未执行**；测试中的 numpy
结构替身仍非真实参数；测试全绿不构成批准。阶段不变：候选计划 `APPROVAL_PENDING`（582 正常 + 18 储备、
硬顶 600；离线 72,000 输入行 / 批 1,024；合成 optimizer 上限 16——均已登记，无一获授权）。