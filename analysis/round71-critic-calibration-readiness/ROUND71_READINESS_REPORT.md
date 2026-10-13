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