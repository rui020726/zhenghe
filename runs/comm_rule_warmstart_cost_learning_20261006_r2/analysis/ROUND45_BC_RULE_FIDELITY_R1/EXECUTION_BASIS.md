# ROUND45 BC 规则拟合与超额成本诊断：执行依据

## 本次授权范围

本轮只做已完成 ROUND45-R2 BC 控制的只读诊断：复核 teacher 标签时间基、从实际逐步 tape 恢复每目标第一次合法自由 SEND/SKIP、核对通信 transcript 与计费，并只读加载三份 BC 模型对真实保存输入做 CPU forward。输出限定为本目录内的分析脚本、机读摘要和报告，并在本地主题分支提交。

不启动环境或 `JointRun`，不运行 episode、训练、PPO、optimizer step，不拟合模型、不调阈值/解码器、不写 checkpoint/RNG、不增样、不修改旧 row、claim、冻结件、门槛或报告，也不推送远端。若真实合法首动作输入或 teacher 时间字段不可恢复，标记缺失，不用任务结果、接收者状态或事后数据补标签。

## 仓库和停止点

- 仓库：`D:\code\整合\zhenghe-round42`
- 诊断分支：`analysis/comm-bc-rule-fidelity`，从完整交付头 `273759c57ead7b729f5b5baee32b864bbc1e273d` 建立。
- 分支建立前工作区 clean；本地分支不存在后新建，没有覆盖或重置既有分支。
- 当前 R2 控制科学门槛是失败状态，PPO 与最终评估没有启动；本诊断不得改变此状态或恢复任何额度。

## 已核验输入

| 输入 | SHA256 / 状态 |
|---|---|
| `ROUND45_R2_CONTROL_RAW_ROW_RECOMPUTATION_R3_FINAL3.json` | 工作树字节 SHA256 `e7e954865fc65f8d8174f0ff68453cd7a698d0410373e2f6065adb83e8287f71` |
| `ROUND45_R2_BC_CONTROL_GATE_R3.json` | `ccad7597d9b75c46683cd1b49e252deb928b1ee3b183fa9c1c4580d106e5af05`，gate=`failed`，训练未获授权 |
| `ROUND45_R2_POST_BC_ENV_FREEZE_R3_FINAL3.json` | `1deba66aa01aeb96782ecfffaf05f4c24a02b05fe8fcde9bb747d47ef9aaec67` |
| `BC_DATA_AUDIT_R2.json` | `f3b85d73e676fa1788d5ac623d60eacdf6e6a146b6877b21db0da76b4dddc988` |
| `BC_DATASET.jsonl` | `93ffba45bfa484aa0f0aa5cbb334bd76c1284d0b21e7ca704b459ae6126f378d` |
| `ROUND45_R2_BC_TRAINING_SUMMARY.json` | `7e3168879d9b70c09ce8a772929ec4296f41ab3ac212c202759c4feaf971976b` |
| `ROUND45_R2_ADOPTED_CONTROL_R3_FINAL3.json` | `767510f295b5267a49c1a2bc0bd687f521942e078d9f9705f506d23ea756497b` |

首轮读回引用 `d54defefed1aea857805fb0d782b62dca3fc5f4fee74890946959a3f7b934e03` 与当前工作树 hash 不同。限定核对已确认：交付头 273759c 的 Git blob SHA256 为 `7c56f6f5a0d1a0cf83fab6d6f01d0efd21d7c99b14c82aeaa504b644b2c0631f`（blob OID `a1280b2c0f734f4ee0c4eb720dff55b4d0ce409e`），内容用 LF；Windows 工作树 SHA256 为 `e7e954865fc65f8d8174f0ff68453cd7a698d0410373e2f6065adb83e8287f71`，内容用 CRLF。两者 JSON 语义相同、48 个 row identity/结果字段相同。该文件是交付头新增文件，父提交没有该路径，因此旧 d54 字节无法从 Git 父版本恢复；不推测它的来源。当前可核验版中，首格 `failed_technical` claim 保持原状、adoption 单独引用一次，claim 状态为 47 complete + 1 failed technical，baseline 的 32 个原始 row manifest SHA 全匹配。分析以交付头所含语义和当前工作树原件为准。

R2 gate 的当前三 seed 数字：每 seed 16 场景、96 个按时目标；BC 字节合计 82,434 B，late 基线 77,412 B，差 +5,022 B / seed（+313.875 B/scene）。三 seed 的 48 个逐场景按时任务 ID 集合全部与 immediate 相同，但成本线三 seed 均未过。因此本次只诊断 BC 的模仿保真和成本来源；不得把结果说成 PPO 无效，因为 PPO 没有运行。

## 冻结 teacher 和时间口径

已冻结 teacher 为 `late_discovery_gate_147`，谓词 `first_emission && ready_step > 147.0`。来源审计中的字段为 `item.first_emission` 和 `item.ready_step`；`CommRelay` 年龄关系为 `age_steps=max(0, relay_tick-ready_step)`。在候选已发布且合法的首个自由动作行中，使用 tape 的 relay/tape observation tick 减保存的 exact integer float32 `age_steps` 复原整数 `ready_step`。`clock.now` 不参与标签。只有 first-emission 且两个策略动作仍合法、预算与容量允许、one-shot 状态尚未锁定的首行动作进入对照。

### 候选时间接线的源码核验

这次补充确认 `age_steps=0` 能否反推出 `ready_step=tick` 所需的前置条件。检查了冻结源码身份并做了静态顺序核验，没有导入运行入口或启动仿真：

1. `JointSAROrchestrator.step` 在一次策略/环境执行周期内先调用 search、同步发布 DetectionEvent，再同步更新 task sink 并调用 executor，最后才推进全局 `clock.now`。
2. 当前 R45 R3 控制入口为每格创建 `OneShotSendSkipCollector` 并把其 policy 传给 `r42_run._execute_cell`；`r42_run` 将构建委托到 canonical `v2_run`，`v2_run` 把 policy 挂到实际 `run.task_adapter.comm_relay`，再调用 `JointRun.run()`。`JointRun.run()` 对应逐步调用 `JointSAROrchestrator.run/step`。
3. `InMemoryEventBus.publish` 在调用返回前同步遍历订阅者。`JointSAROrchestrator` 构造时先订阅 registry；`build_joint_run` 随后订阅 `CommRelay.handle_detection`，所以 relay 处理器在 registry 接受并提交后执行。
4. `CommRelay.handle_detection` 对新接受事件写 `ready_step = 旧 relay tick + 1`。这一事件后，同一 orchestrator 周期经 `AssignmentGate.step` 委托给 `Task03Adapter.step`；Task03Adapter 先调用 `CommRelay.tick()`，而 tick 先递增计数，再 `_emit_selected()` 并构造 observation。
5. 因此，新检测在本次 observation 上的 ready step 等于当前 relay tick。较早排队的 ready step 不会超过当前 tick；无 event 的 view-change 路径以当前 tick 记时。`CommRelay` 中没有其他将 ready step 设为未来 tick 的写入路径。tape 用 `int(observation.step)`，且 `SendObservation.step` 直接等于 relay `_tick`。

据此，本批合法候选保证 `ready_step <= tape step`，`age_steps=max(0, tick-ready_step)` 不会在候选行截断未来时间；288 条首个自由动作行均为 age 0，故这些行的 ready step 可精确恢复为 tape tick。时间戳始终保留为不同概念；该关系不是把所有时间字段统一视为同一时钟。Teacher 仍只依据 `first_emission` 和 `ready_step`，不使用 `clock.now`、任务结果或接收端状态。

静态核验的 SHA256：

| 源文件 | SHA256 |
|---|---|
| `comm.py` | `9b7938ae1af6697a7f977026c7920a8e17bd71492cb605407fb02f7019828dbd` |
| `orchestrator.py` | `ad1380d4695d8fb90ea646d2b4ebd0d6d736879885feff79bef8067a1a956b74` |
| `task03.py` | `4dcd548f192690df383b9ddc5b254d54abf02891d6480f4aa244045acddbe7d1` |
| `event_bus.py` | `60700446e97a296099edcd9d4d8869f4fef55c00b7a625909f0d25ca10ca2fe4` |
| `joint_driver.py` | `2a682fecd5936cc4b2786477398eed55928c6f5781282bc29e4918cdc01c911b` |
| `assignment_gate.py` | `d9d7527c01c5c8d488c6b46745885bc51ef47fae76cbc0129630fadbcba4d473` |
| `r42_one_shot.py` | `3c342108148de42f395faeec25544976eed7efdfdd32a99685d73bd59101ab0d` |
| `r42_run.py` | `9c5feacb2ffcb2bb930d2b6022e2fca5210408765dc664859d400e6f2b5f8c2e` |
| `v2_run.py` | `90c6936c24c93999d03aac3f09b9b6506a9e4f8050f9115a0ef3fc839c9f16b7` |
| `r45_r2_execute_r3.py` | `f796ed07891bcc2457fe48e9ddd288a6c18c15a511fd8e505c5400a9e6819dd5` |

Teacher 与 BC 动作的分歧是保存输入上的局部模仿差异，不是两种策略的反事实效果。逐场景 transcript 只陈述实际记录的 target、tick、attempt、subscribers、loss 和收费，不声称运行间使用了相同的 channel RNG draws。对精确 19D 输入未见于 fit 仅称精确向量支持缺失，不单独称为广义分布偏移。

## 执行件

- `ROUND45_BC_RULE_FIDELITY_ANALYSIS_R1.py`：可重跑的只读分析脚本；核实 row/claim/adoption receipt、tape、transcript、baseline SHA 和冻结模型身份；重算训练 fit/validation confusion，并在真实第一自由动作输入上做只读 forward。
- `ROUND45_BC_RULE_FIDELITY_SUMMARY_R1.json`：逐 seed / scene、逐首次动作、emission/cost 闭合、模仿 confusion 与来源核验。
- `ROUND45_BC_RULE_FIDELITY_REPORT_R1.md`：简要人读报告。

运行仅应使用仓库既有 Python/PyTorch CPU 环境；本执行不调用环境/训练入口。脚本确认前后 RNG 状态及所有被读原件 hash，报告环境构造、run、PPO、optimizer 与 checkpoint 写入均为零增量。
