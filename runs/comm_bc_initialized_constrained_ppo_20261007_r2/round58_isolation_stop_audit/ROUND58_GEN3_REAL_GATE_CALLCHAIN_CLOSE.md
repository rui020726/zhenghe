# ROUND58 Gen3 实际门禁与调用链收口

状态：**只读核验完成；不续训、不追认后继续、不恢复任何状态。**

## 结论

- 修复后的身份 helper 已进入私有执行入口的实际首批身份判定；用 R42 实际 `first_batch_gate` 对 s11/s22/s33 各执行完整只读路径，三者均 `passed`。旧失败 gate 与 checkpoint 原样保留。
- 实际 prior wrapper 的 PPO 调用现由同一活体 trainer/R42 计数对象上的外层 bridge 计数。单次 no-op spy 证明一次调用增加一次、同一 live ledger 读到一次；不构造真实 PPO 更新。
- 三个既有 batch 的技术核验均达到 `qualified_for_later_resume_review`。这不是恢复授权，也不代表本轮恢复已验证或科学结果已采纳；本轮未恢复训练。
- 历史 critic 隔离证据等级为 **B**：执行身份/源码哈希与配置后 live binding、且 wrapper 在整个 compiled PPO delegate 外持有 detach context，共同支持历史路径；无逐 optimizer-step 的直接隔离遥测，因此不是 A。

## 身份门禁根因和修复边界

旧 gate 把 `payload.freeze` 与当前运行时 `round42_freeze` 整体相等比较。Gen3 serializer 按批准契约加入 `training_version` 和 `round58_repair_attestation`，故即使基础科学 freeze 未变也触发复合错误 `checkpoint identity does not match the frozen ROUND42 version`。本轮确认版本、动作语义、批准科学 freeze、attestation 来源/内容、payload/sidecar/批提交哈希链；仅对这两个登记 metadata 做结构化归一，归一后要求基础 freeze 精确相等。任意额外 freeze key 不会被忽略。三个 checkpoint 的历史执行身份分别核对，未拿本轮新源码 freeze 冒充历史期望值。

实际 gate 模块是 `runs/comm_one_shot_full_trace_ablation_20261004/r42_analysis.py`，SHA256 `3342367be6d8a3c732ac7bf541575d17ae8a168618bc3f9b1493781bf8345633`，函数 `first_batch_gate`。审计适配层只修正 gen3 metadata 与原科学 freeze 的比较接缝，之后实际 gate 其余检查均运行；只读 loader 保留 canonical loader 判据但在私有审计副本中抑制 RNG 恢复。

## 三批逐项结果

| seed | batch / episode | likelihood最大差 | reward/GAE样本最大差 | g_hat；λ | optimizer步（已记录） | 结果 |
|---|---|---:|---:|---|---:|---|
| 11 | batch 0；0–3 | 2.344791028e-13 | 0 | 0；1→1 | 8 | qualified_for_later_resume_review |
| 22 | batch 0；0–3 | 1.705302566e-12 | 0 | 0；1→1 | 8 | qualified_for_later_resume_review |
| 33 | batch 0；0–3 | 9.379164112e-13 | 0 | 0；1→1 | 8 | qualified_for_later_resume_review |

三者均核对 265 个样本、自由候选/mask闭合、正成本与奖励时序、GAE和组合约束、更新前后参数身份、10个 optimizer 参数、task/cost critic head 纳入优化器、batch/commit和前后 checkpoint 元数据及 sidecar SHA 绑定；gate 期间原件哈希均不变。保存记录显示每批 8 次 Adam 调用，批游标均为 episode4/update1。task/cost critic head 有梯度不等于 critic 对 trunk 有梯度；隔离结论来自下面的源码和活体调用链证据，而非梯度范数推断。

### Critic 隔离证据分级

- 冻结 gen3 attestation 所指 prior adapter、trainer/common、配置及 checkpoint-contract 内容哈希均通过；历史 source attestation 完整匹配。
- 历史执行接线 `round58_gen2/r58_execute.py` 调用 `install_on_verified_trainer`、`install_gate_prior(..., critic_input_detached, ...)`、将同一 trainer 绑定给 R40 并 `validate_live_binding`。prior adapter 的 `r58_ppo_update` 在调用编译后的 `patched_fn` **整个期间**持有 `controller_detach_context(controller)`；该调用链没有发现绕过该上下文的分支。
- 三批的运行身份与 batch/commit/checkpoint 身份链相符，故符合用户定义的 B 级证据。没有逐步 gradient journal，故不能声称 A 级直接运行遥测。

## PPO 计数和累计派生快照

漏计原因是 prior wrapper 曾直接调用编译的 PPO delegate，越过 R42 原 trainer counter wrapper。现在 bridge 包在实际 `trainer.ppo_update` 的最外层，与 R40 引用同一个 trainer；读取及 spy 确认 wrapper、counter 与 `r42_run` 是同一组活体对象。内存 no-op delegate 调用 1 次，counter `0→1`，实际 strict ledger snapshot 读到同一个值；真实 optimizer step 为 0。旧 journal 未改写，也未按 batch 数补造历史逐次调用。

实际配置的 `r42_run.ledger_snapshot(strict=True)` + `r40_common.ledger_snapshot` 只读汇总：12 normal claims、0 reserve、12 completed logical jobs、3 durable batch records、3 个由唯一提交 batch 派生的 PPO call 数、24 条原 optimizer hook journal 行；批数和 PPO 派生数闭合，optimizer rows 与批记录 8×3 闭合。PPO call 的历史逐调用 journal 曾被 wrapper 绕过，因此 3 是明确标记的派生重建值，不是原始调用事件。快照没有写入；派生快照失败不触碰核心提交。

## 副作用与证据限制

本次零：环境构造、episode provider、claim/journal/checkpoint写、optimizer.step、真实PPO更新、RNG restore、评估格、新claim、tape重放均为 0。只读 torch 反序列化 28 次（source attestation 1、pre-gate 6、实际gate 21；runtime setup 0）；模块 forward 调用 5,565（每 seed 1,855）；autograd.grad 0；内存 no-op PPO spy 1 次、memory writer spy 1 次。进程内 torch CPU RNG 状态变化 3 次（每个 gate 构造只读 controller 时发生），Python/NumPy RNG 未变；没有执行 RNG restore，审计进程已退出。不能把这描述成“审计期间 RNG 状态从未变化”。

窄测试：`test_r58_identity_ledger_close_v2` + `test_r58_private_launcher`，24 tests，退出码 0。实际完整 gate 审计脚本 `audit_r58_gen3_real_gate_callchain.py` 退出码 0，结果见同目录 `ROUND58_GEN3_REAL_GATE_CALLCHAIN_AUDIT.json`。未运行全量套件。

## 唯一后续路线

三个 batch 技术资格核验成立，可供**另行审批** episode4 精确恢复审查；当前不恢复。既有额度为累计28/1064、正常剩余1012、reserve剩余24（本轮无消费）。若后续批准恢复，仍须依本次合法 gen3 checkpoint 的精确恢复契约，另行明确授权；本报告不发放启动许可。

保护文件 SHA256 复验：`ROUND46_CURRENT_TASK.json` `cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b`；`ROUND46_R2_V6_STOP_STATE.json` `93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18`。两者不纳入本地提交。
