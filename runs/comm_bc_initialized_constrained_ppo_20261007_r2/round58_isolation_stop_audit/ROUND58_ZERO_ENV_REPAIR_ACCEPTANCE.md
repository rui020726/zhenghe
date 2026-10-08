# ROUND58 零环境修复验收（非科学计划完成）

## 范围与结论

本验收只覆盖隔离 launcher、真实运行时绑定及 R40 批提交接缝的窄范围零环境验证。它**不授权、不代表**ROUND58 的 Path C 四集替代、384 集训练或 640 格评估已经执行。科学计划仍未完成：本代次有效训练提交为 0，评估完成格为 0；此前历史任务均不能自动计入新的有效训练链。

验证时 HEAD 为 `3f7b6ee0a60f43ab6466f22d9bedb98ec8b2766a`，分支 `exp/round58-prior-strength-prospective-trial`。没有发现运行中的 Python/Git 进程。工作区仅两份用户保护文件处于既有 dirty 状态，未改、未暂存：

| 文件 | 接手/验收 SHA-256 |
|---|---|
| `runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_CURRENT_TASK.json` | `cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b` |
| `runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_R2_V6_STOP_STATE.json` | `93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18` |

## 历史证据与资格

只读核算保持三种概念分离：

- **历史环境调用**：gen1 4 次、gen2 12 次，共 16 次已知构造/运行调用；gen2 claim 均标 normal，但登记文本称曾使用 4 个 reserve，原账本仍仅列 normal=4/reserve=0，因此 normal/reserve 分类不能闭合。历史消费不退还，也不把矛盾改写成已确认 reserve。
- **历史 optimizer 实际调用**：证据支持总数 48，但证据粒度不完全相同：原 gen1 未落盘更新 8 次由故障 hook 的时间戳记录支持；gen1 错误恢复 16 次由两份不同 after-b0 状态各自 Adam step=8 及相异 hook 后缀间接支持；gen2 24 次有时间戳 hook 行支持，并由三份 after-b0 Adam step=8 状态交叉佐证。gen2 每批记录的“16”与 live PPO 的 265/256 样本、4 epochs、每 epoch 两个 minibatch矛盾，不能当作16次调用。不同证据来源不等价于48条完整、逐笔原始事件。
- **有效科学训练链**：0 个有效批、0 个有效 optimizer 更新、0 个有效训练游标。历史 R58 调用不能进入有效链，因为原代次绕过了 R53 critic-detach wrapper；旧提交/字段存在不改变这一判定。

原 gen1 seed11 e0–e3 的四条 complete claims 与各自 row/tape 哈希闭合，row 中 controller SHA 与已登记 BC 初始化参数 SHA 一致；这些是保留的历史候选输入，不是本次执行输入。它们没有被重放。四条 row 的每条都列出以下 12 类 substituted/defaulted 配置：

1. `map_path`：归档值 null，使用当前默认 None；
2. `comm_sends_per_tick`：归档值 null，使用当前默认 None；
3. `comm_budget_bytes`：归档值 null，使用当前默认 None；
4. `comm_max_wait_steps`：归档值 null，使用当前默认 None；
5. `comm_urgent_wait_steps`：归档值 null，使用当前默认 None；
6. `comm_ack_delay_ticks`：归档值 null，使用当前默认 None；
7. `comm_ack_delivery_probability`：归档值 null，使用当前默认 None；
8. `repository_root`：归档值 null，使用当前默认 None；
9. `replay_stream`：归档值 null，使用当前默认 None；
10. `search_backend`：归档字段被丢弃，非当前配置字段；
11. `allocation_backend`：归档字段被丢弃，非当前配置字段；
12. `Stage B frozen overrides applied` 聚合清单：`comm_ack`, `comm_budget_bytes`, `comm_delay_ticks`, `comm_delivery_probability`, `comm_max_wait_steps`, `comm_range`, `comm_sends_per_tick`, `comm_subscribers`, `comm_timing_mode`, `comm_urgent_wait_steps`, `decision_trace`, `device`, `gossip_interval`, `info_probe`, `mock`, `num_goals`, `quiet`, `record_stream`, `replay_stream`, `search_probe`, `steps`。

分类：第1–7及第12项中的通信/世界参数具有科学配置风险；`repository_root` 是路径/来源风险，`replay_stream` 是输入/记录来源风险；两个 backend 是可能影响实现选择的字段，当前 row 称其被丢弃。聚合 override 同时含可能影响科学语义的参数及设备、记录/探针等运行或被动观测控制。缺少完整有效值时不据此宣判每项已发生偏离，但也不能判为闭合。完整 minibatch 前 Python/NumPy/Torch RNG 状态、排列及独立 hash 封存的 canonical batch_samples/GAE 输入均未建立，所以原四集不获认证为可复现训练批。

## 本次隔离修复及证据

- 新增 `r58_private_launcher_repaired.py`。它导入未修改的 gen2 launcher，进入实际 R46 configure 路径并绑定真实 R45/R40/R53 模块对象；校验 trainer、实际 PPO globals/common 与修复 adapter 的对象身份。默认执行拒绝；恢复路径先执行 fail-closed 检查，再允许任何写入、恢复、RNG、环境或 optimizer 回调。当前无授权/freeze/quota，不存在可启动训练的路径。
- 新增 `test_r58_private_launcher.py`：通过实际 configure/bind 接缝验证路由；拒绝默认执行及旧 checkpoint，证明拒绝之前的恢复/写入/RNG/环境/更新回调数为0。
- 新增 `test_r58_actual_commit_seam.py`：穿过真实 R40 `train_seed`、PPO 调用及 `_write_batch_commit` 顺序，以内存 IO/保存 spy、stub episode provider 和 no-op optimizer step callback 注入诊断 gradient/hash/log 异常、核心保存异常和 receipt 异常。诊断错误不阻断核心提交；核心保存失败传播且不尝试成功回执；回执失败传播且不形成成功进度。未写实际文件。
- 复用旧 16 项窄测试，不重复全量仿真/项目套件；新 launcher 3 项、真实提交接缝 3 项，均退出码0。总计 22 项测试。两套新测试分开运行，避免活体模块缓存污染。
- 较早的16项测试：104个 fake spy optimizer callbacks，仅读梯度/计数；新提交接缝的5次 PPO 诊断路径各触发4个 no-op optimizer callback，共20次。两者合计124个 fake callback。真实 `torch.optim.Optimizer.step` 为0，参数/Adam 未更新。

测试不是生产训练：使用 stub episode，不构造环境、不读取旧 tape、不恢复训练 optimizer/RNG、不保存 checkpoint。磁盘 checkpoint/receipt 写入为0；事务顺序只经内存 spy 验证，不声称已证明真实文件系统崩溃原子性。

## 运行时身份与限制

通过实际导入路径读取的 canonical 模块身份：trainer SHA-256 `0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841`；common SHA-256 `66d019a344d9cfef3c67bd8b722f8cc27144b9f71ddc8e5496451929d12f746a`；controller SHA-256 `9b2362ee5dc436c04e5fd2d841646d637b5b3f371607bd13a759a60c4cb4cab1`。真实 PPO 的冻结运行值核对结果见 `LIVE_MODULE_IDENTITY.json`。修复接线保持 critic isolation wrapper，修正重复 minibatch 计数，并将被动诊断失败与核心 PPO/提交错误分离；未改研究数学。

这是一项**隔离修复验收**，不是 R58 科学执行门禁，也不冻结新的训练身份。未做 Path C 首四集替代；未核验/消费评估场景；没有新 claim、训练批、评估格或统计结论。因此报告提交并不意味着固定计划可以被称为完成。

## 操作计数

| 操作 | 本轮 |
|---|---:|
| 环境构造 / 环境启动 / 仿真 | 0 / 0 / 0 |
| 新 claim / 逻辑训练任务 / 评估格 | 0 / 0 / 0 |
| 真实 optimizer.step / 更新 | 0 / 0 |
| fake optimizer callbacks | 20（no-op） |
| checkpoint 或成功训练回执写入 | 0 / 0 |
| 旧训练 tape 重放 / 真实 RNG 恢复 | 0 / 0 |
| 全量项目或仿真套件 | 0 |

测试详情及结果在 `ZERO_ENV_REPAIR_ACCEPTANCE.json`。此前提交和报告仍保留；本次提交仅点名新增 launcher、两份测试和验收材料。没有推送、合并或部署。
