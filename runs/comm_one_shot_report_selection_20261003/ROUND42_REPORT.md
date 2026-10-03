# ROUND42 执行报告：一次性 SEND/SKIP 首报选择消融

**状态：按冻结故障纪律停止；没有形成任何策略效果结果。** 首个参考场景的 `JointRun` 构造后，运行器包装代码给只读的 `run` 属性赋值并抛出 `AttributeError`。账本记为 1 次正常失败启动；`run()` 未调用、仿真未开始。该错误属于本轮运行器接线缺陷，不能解释成模型或策略结果。

## 执行身份

- ROUND42 分支：`exp/comm-one-shot-report-selection`
- 首次环境尝试前完整 HEAD：`3c0b61cbdb702c3e4d8b1a4668030bda2db102ab`
- ROUND42 执行依据提交：`ffa87ceb1a9d48abe27fc124ef1a963470878604`
- ROUND40 完整交付 HEAD：`8f8245a665b15960e70194fc6e245221b70db90e`
- ROUND41 完整交付 HEAD：`cc28a28fc5fb13e31ab9ed0fc3456797aadcc115`
- 训练版本：`constrained-cost-v2/one-shot-send-skip/1/published-completion-event-at-relay-tick/2`
- 动作语义：`one-shot-send-skip/1`

ROUND42 执行代码及冻结身份在首次环境尝试前已本地提交。此次失败后没有改动代码、策略、数学或研究配置；只保留原运行记录，新增本报告、机器 verdict、artifact inventory 和审计副本，并更新当前状态。最终报告提交 HEAD 可从本报告所在分支的后续 Git 历史读取；上列 HEAD 是本次唯一实际执行的冻结 HEAD。

## 前置门禁

- ROUND40、ROUND41 指定交付头、Round40 输入源码身份、配置差异与 ROUND42 冻结哈希均由 preflight 核验。
- 8 个 ROUND42 Python 脚本通过 `py_compile`。
- 纯计算/接线测试 12/12 通过；测试 guard 记录 JointRun 构造 0、`optimizer.step` 0、checkpoint 写入 0。
- preflight 对 ROUND40 三个 seed 的 ep0 与 ep128 权重作只读内容核验与 CPU forward；六份权重的 logits 形状为 `[4, 2]`，双 critic 输出形状为 `[4]`，输出有限。
- 配置阶段的导入来源接线检查通过，模块路径均指向冻结的 ROUND40 或 ROUND42 文件；该检查不构造环境、不执行 optimizer 更新。
- 本次执行前，ROUND42 日志、rows、checkpoints、runtime 输出文件均为空。纯计算门禁和配置接线检查不属于仿真运行。

以上仅证明相应静态、纯计算、只读和导入接线检查通过；不替代真实运行门禁。ROUND42 首批训练门禁、episode/tape/奖励验证和策略效果分析均未执行。

## 首次运行及故障证据

尝试的逻辑任务为 `immediate_reference_scene1300000`，normal slot 1，场景 seed `1300000`。构造器返回 `JointRun` 后，ROUND42 的 `counted_build` 试图执行 `run.run = counted_run`。实际对象的 `run` 属性只读，抛出：

```text
AttributeError: 'JointRun' object attribute 'run' is read-only
```

异常发生在调用 `run.run()` 之前。账本分别记录了 `environment_constructor_calls=1`、`simulation_started=false`。结果行未产生；对应 JSONL 文件大小为 0 字节。运行器自动写入 `failed_technical` claim 和 stop 记录，没有自动重试。依据执行依据第九节“出现影响有效性的错误时保留原件和版本，停止受影响运行”，本轮在此停止，没有用 reserve 重跑，也没有在已冻结版本上热修后续跑。

## 启动与更新计数

| 项目 | 实际值 | 冻结计划/上限 |
|---|---:|---:|
| normal claims | 1（其中 1 次技术失败，不退额） | 672 |
| technical reserve | 0 | 最多 28 |
| 总 claim / 硬上限 | 1 / 700 | 700 |
| JointRun / 环境构造调用 | 1 | — |
| 仿真开始 | 0 | — |
| 成功环境结果 / episode | 0 | 672 格计划未完成 |
| PPO 批 / 真实 `optimizer.step` | 0 / 0 | — |
| ROUND42 checkpoint 写入 | 0 | — |
| 新流、重试 | 0、0 | 0、0 |

失败的 normal claim 已消耗一次启动额度。账本尚余 28 个 reserve，但额度存在并不授权绕过本轮明确的停止纪律。

## 模型与文件完整性

ROUND40 三个 seed 的 ep0/ep128 六份策略权重，在首次运行后重新计算的文件 SHA256 与 preflight 的 before/after SHA256 全部一致；两份冻结搜索/分配 checkpoint 也与 preflight SHA256 一致。完整路径、大小、哈希、参数摘要和只读身份见 `ROUND42_ARTIFACT_INVENTORY.json`。未在 ROUND42 写出训练 checkpoint，也未执行 NEW 训练；因此 NEW 只保留已核验的初值，不能报告训练后参数或学习行为。

运行账本、stop 记录、prelaunch identity 和失败异常的 Git 外原件保留在本轮输出目录的 `runtime/`、`failures/`；顶层同名审计副本已逐字节核对。执行器自动生成的 `ROUND42_STOP_REPORT.md`、`ROUND42_STOP_STATE.json` 也予以保留。空 JSONL 原件保留在 `logs/`，其大小与 SHA256 收录在 artifact inventory。原始模型和运行日志不纳入 Git。

## 结论与下一步边界

1. 一次性 SEND/SKIP 是否减少首报、保持任务表现或学会等待：**本轮无法回答**。只有一次构造失败，没有仿真样本。
2. ROUND42 策略效果、任务数、首报数、字节成本、配对比较：**均无可估计结果**；不能将预期方向或已有旧轮结果当作本轮结论。
3. 训练行为与学习：**没有发生训练**，真实 optimizer 更新为 0，不能声称模型学会或没有学会任何行为。
4. 可恢复性：本轮没有成功 episode 或已提交训练更新可供同版本恢复。执行依据要求保留原件并停止受影响运行；如果决策者要继续，需先另行批准并登记修复后的执行版本，重新完成启动前冻结与门禁，再决定是否消耗技术 reserve。此报告不自动授权重试。
5. 既有 ROUND40/ROUND41 结果保持原样，ROUND42 不改变其 verdict，也不推送、合并、部署或开启确认实验。
