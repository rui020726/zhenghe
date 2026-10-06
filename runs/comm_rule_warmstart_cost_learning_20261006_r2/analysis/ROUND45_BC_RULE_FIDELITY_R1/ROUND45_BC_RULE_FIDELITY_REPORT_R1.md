# ROUND45 BC 规则拟合与超额成本只读诊断

- 输入冻结起点：`273759c57ead7b729f5b5baee32b864bbc1e273d`；分析分支：`analysis/comm-bc-rule-fidelity`。
- 机器摘要：`runs/comm_rule_warmstart_cost_learning_20261006_r2/analysis/ROUND45_BC_RULE_FIDELITY_R1/ROUND45_BC_RULE_FIDELITY_SUMMARY_R1.json`。
- R3 FINAL3 原始行重算 SHA256：`e7e954865fc65f8d8174f0ff68453cd7a698d0410373e2f6065adb83e8287f71`。
- Git/worktree 核对：提交 blob SHA256 `7c56f6f5a0d1a0cf83fab6d6f01d0efd21d7c99b14c82aeaa504b644b2c0631f`，worktree SHA256 `e7e954865fc65f8d8174f0ff68453cd7a698d0410373e2f6065adb83e8287f71`；字节相同=False，JSON 语义相同=True。
- 科学控制门槛仍失败；此诊断不重新打开 PPO 或评估额度。

## 三 seed 控制与成本复算

| Seed | BC bytes | late bytes | 差值 | BC 首报 SEND | teacher SEND | teacher 多发 | teacher 漏发 | task IDs = immediate |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 11 | 82,434 | 77,412 | +5,022 | 33 | 31 | 2 | 0 | 是 |
| 22 | 82,434 | 77,412 | +5,022 | 33 | 31 | 2 | 0 | 是 |
| 33 | 82,434 | 77,412 | +5,022 | 33 | 31 | 2 | 0 | 是 |

每 seed：16 场景，BC 与 immediate 的逐场景按时任务 ID 集合全部相同；成本总差为 +5,022 B。三 seed 的观察行为逐场景相同计数：自由动作 16/16，首发射目标与 tick 16/16，发射序列（含 revision/bytes/subscribers）16/16，19D 输入 16/16。
这些是原始转录里的逐项相等检查，不表示不同运行的信道随机数流相同。

原始行重算 SHA 的更正：ROUND45 交付头内 Git blob 使用 LF，blob SHA256 为 `7c56f6f5a0d1a0cf83fab6d6f01d0efd21d7c99b14c82aeaa504b644b2c0631f`；当前 Windows 工作树使用 CRLF，所以 SHA256 为 `e7e954865fc65f8d8174f0ff68453cd7a698d0410373e2f6065adb83e8287f71`，但 JSON 语义逐项相同。该文件在交付头中是新增文件，父提交没有可比较的旧版本；先前引用的 d54def…既不等于当前工作树也不等于 Git blob，不能从父提交恢复。已改用交付头实际包含的版本，其 48 行、首格 adoption、claims、baseline 来源校验、三个 seed 的 task/cost 字段均可读回核实。

## Teacher 与实际 checkpoint forward

Teacher 精确条件为 `first_emission && ready_step > 147`。当前 R45 R3 collector 经 `r42_run`/`v2_run` 接到真实 relay，再由 `JointRun.run` 逐步进入 orchestrator。Tape `step` 是 relay observation/tape tick；合法首个自由动作行的 `ready_step` 由 `step - age_steps` 复原，其中 `age_steps` 是保存的精确整数 float32 特征，对应冻结源中的 `age_steps=max(0, relay_tick-ready_step)`。真实调用链核验确认：orchestrator 同步发布 detection，registry 处理器先于 relay 处理器执行；relay 将新事件记为旧 tick+1，而同一周期经 AssignmentGate 到达的 Task03Adapter.step 先调用 relay.tick，tick 递增后才构造 observation。因此所有已提供候选均满足 ready_step <= observation.step，年龄的 max(0,...) 不会在候选行上截断未来 ready_step；288 条首选行 age_steps=0，故其 ready_step 可精确还原为 tape tick。Tape 的 step 来自 `int(observation.step)`；全局 `clock.now` 在之后才推进，没有用于 teacher 标签。相关源码 SHA 与顺序断言保存在机读摘要 `source_identities.candidate_time_source_chain`。未用任务结局或接收端状态打标签。

### 候选 ready tick 接线源码身份

- `comm.py` SHA256 `9b7938ae1af6697a7f977026c7920a8e17bd71492cb605407fb02f7019828dbd`
- `orchestrator.py` SHA256 `ad1380d4695d8fb90ea646d2b4ebd0d6d736879885feff79bef8067a1a956b74`
- `task03.py` SHA256 `4dcd548f192690df383b9ddc5b254d54abf02891d6480f4aa244045acddbe7d1`
- `event_bus.py` SHA256 `60700446e97a296099edcd9d4d8869f4fef55c00b7a625909f0d25ca10ca2fe4`
- `joint_driver.py` SHA256 `2a682fecd5936cc4b2786477398eed55928c6f5781282bc29e4918cdc01c911b`
- `assignment_gate.py` SHA256 `d9d7527c01c5c8d488c6b46745885bc51ef47fae76cbc0129630fadbcba4d473`
- `r42_one_shot.py` SHA256 `3c342108148de42f395faeec25544976eed7efdfdd32a99685d73bd59101ab0d`
- `r42_run.py` SHA256 `9c5feacb2ffcb2bb930d2b6022e2fca5210408765dc664859d400e6f2b5f8c2e`
- `v2_run.py` SHA256 `90c6936c24c93999d03aac3f09b9b6506a9e4f8050f9115a0ef3fc839c9f16b7`
- `r45_r2_execute_r3.py` SHA256 `f796ed07891bcc2457fe48e9ddd288a6c18c15a511fd8e505c5400a9e6819dd5`

| Seed | Split | Samples | Skip→Skip | Skip→Send | Send→Skip | Send→Send | Accuracy |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 11 | fit | 1548 | 1236 | 6 | 0 | 306 | 0.9961 |
| 11 | validation | 738 | 519 | 6 | 0 | 213 | 0.9919 |
| 22 | fit | 1548 | 1236 | 6 | 0 | 306 | 0.9961 |
| 22 | validation | 738 | 519 | 6 | 0 | 213 | 0.9919 |
| 33 | fit | 1548 | 1236 | 6 | 0 | 306 | 0.9961 |
| 33 | validation | 738 | 519 | 6 | 0 | 213 | 0.9919 |

## 两个额外 SEND 的逐项来源

每个 seed 都在相同两个保存输入上比 teacher 多发一次；每个场景其余首报 emission sequence 与 late 规则逐项相同。
| Scene | Target | Relay tick / ready_step | `p(SEND)` s11 / s22 / s33 | Charged bytes | attempts/subscribers | lost packets s11 / s22 / s33 |
|---:|:---|---:|:---|---:|---:|:---|
| 1301000 | piloc-e1-target-5 | 122 / 122 | 0.7307 / 0.7657 / 0.8014 | 2,508 | 6/6 | 2 / 2 / 2 |
| 1301014 | piloc-e1-target-5 | 140 / 140 | 0.9595 / 0.9544 / 0.9571 | 2,514 | 6/6 | 0 / 0 / 0 |

每 seed teacher 首选 31 SEND、BC 实际 33 SEND；额外两条首报分别收费 2,508 B 与 2,514 B，合计 `5,022 B`，正好等于每 seed 相对 late 的总成本差。ready_step 为 122 和 140，都远离 `>147` 边界；没有漏掉 teacher SEND。6 条重复观测对应 2 个独特 19D 输入，它们都不在冻结 fit/validation 的精确向量集合中。精确同值 teacher 标签冲突为 0；这支持局部未见输入上的模仿假阳性，不足以证明广义分布漂移或反事实策略效应。

三份 checkpoint 均按 FINAL3 SHA 和参数摘要只读加载；torch.load 读入 checkpoint payload 用于结构核对，但没有构造 optimizer、恢复 optimizer 状态或使用 checkpoint RNG。拟合/验证混淆矩阵由实际 float32 forward 重算，并与已保存训练指标逐项一致。控制行 forward 动作也逐条对照 tape。

## 归因边界

- 首个合法自由动作行 288 条；缺失 teacher label 0 条；19D 同值冲突（dataset）0 个，控制行 0 个。
- 首报 ready_step 在 146–149 的计数：`{'148': 3}`；对应 action/teacher 不同：`{}`。
- 288 条合法首选输入均未发现 pre-action legality 或 label/clock 重建异常；其中 age_steps 均为 0，所以这些具体行上 ready_step 数值等于 relay/tape tick。cutoff 邻域观测不构成多发原因。
- 三 seed 的 emission sequence 对 late 规则各有 14/16 场景逐项相同；不同的仅是 scene 1301000, 1301014 的额外 target-5 首报。
- 控制特征精确拟合向量支持：`{'exact_fit_vector': 144, 'not_exactly_seen_in_fit_or_validation': 138, 'validation_only_exact_vector': 6}`；其中不一致按支持类为：`{'not_exactly_seen_in_fit_or_validation': 6}`。精确向量未见只说明该向量未出现在拟合集，不能单独定性为广义分布偏移。
- Tape/feature/receipt/transcript/成本闭合问题以 JSON `anomalies` 字段记录；实际记录没有因同总字节而被假设行为相同。
- 本轮只定位 BC 对规则的动作保真和行为差异。不能把成本超线解释为 PPO 无效，因为 PPO 更新为 0。

## 范围与计数

本脚本只读取既有 JSON/JSONL、训练 dataset 与三个 BC checkpoint，执行 CPU 前向并写入本分析目录。新环境 0、run 调用 0、PPO 更新 0、optimizer step 0、checkpoint 写回 0。
