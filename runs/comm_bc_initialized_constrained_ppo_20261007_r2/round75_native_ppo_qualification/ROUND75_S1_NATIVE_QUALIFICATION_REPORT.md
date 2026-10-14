# ROUND75 S1 真实原生 PPO 资格执行报告（一次 attempt，失败收口）

请求 ID：`R75_S1_NATIVE_QUALIFICATION_20261015`
执行分支：`exp/round75-native-ppo-qualification`
代码 HEAD：`af9b80e97bfeaf382108f0615ee317d9f0ea551d`（父 `cc0d3a29089bc361890e20520ad9800906184a24`，祖父 `7eff4a4218f275e2090ec2d9f42604b42bb6e7e9`）
执行时间：2026-10-15 01:56:54 - 01:57:01（+08:00），墙钟 7.571 s
结论：**唯一一次被许可的真实 attempt 已执行并失败；资格未通过；零付费资源消耗。**

## 1. 授权与执行边界

本执行仅由 `R75_S1_NATIVE_QUALIFICATION_20261015` 授权，取代 S0 的"尚未许可"边界，且只对这一个固定矩阵有效。它不是新效果实验、不是环境训练，不延长任何旧额度。授权明确只允许冻结 runner 的一次完整调用，且"若同 ID 已启动/完成，核验接续或交付，严禁再启动一次"。

启动前核验：HEAD、六个代码 SHA、三个数据 SHA、canonical ledger 九文件字节与冻结快照一致、12 events / charged 0、四 guard normal/reserve/refusal 全 0/0/0、无残留 lock、`raw/s1/` 不存在、请求 ID 在输出根内任何位置均未出现、无 python 进程在跑。全部通过后才进入 paid path，未重填任何旧许可。

## 2. 唯一 attempt 的实际结果

外层封装（纯 stdlib，`r75_s1_outer_exec.py`，SHA `b1c4fa4c3be0f2983fb3baf10e16c1e8d9c18a37c0b0736bbdc693c436c45115`）在启动任何付费 Python 之前，用 `O_CREAT|O_EXCL` 建立了 attempt 标记并写入 pre-launch manifest（含请求、时间、HEAD、全部 code/data SHA、argv、计划与零起点快照、封装自身 SHA）。随后只启动了一次 runner。

runner 内部纯预检正常执行一次并通过（68/68，`checks_failed: []`，stdin 无关），随后进入付费导入阶段并失败：

```
File ".../r75_s1_runner.py", line 1122, in run_s1
    np_mod, torch, r75_frozen, r75_guard = _load_paid_modules()
File ".../r75_s1_runner.py", line 301, in _load_paid_modules
    import r75_frozen
File ".../r75_frozen.py", line 61, in <module>
    import v2_trainer as trainer
ModuleNotFoundError: No module named 'v2_trainer'
```

真实退出码 1。这不是偶发环境噪声，而是冻结源码里的确定性接线缺陷：`v2_trainer.py`（连同 `v2_common`、`onpolicy`、`v2_controller`、`cost_math`）只存在于 `runs/comm_cost_v2_rerun_20260928/`，而 `r75_frozen.py` 只把 `R66_DIR` 插入 `sys.path`，从不调用 `r66_frozen.configure_source_imports()`；`r75_s1_runner.py` 也只插入自身的 `HERE`。因此在冻结命令（仓库根为 CWD、无 PYTHONPATH）下该导入必然失败，attempt 不可能成功。

按执行纪律：异常已保留失败与已耗原件，立即停止，未自动重试、未动用技术储备、未退款、未修改脚本或参数、未重跑一个 attempt 去挑通过。attempt 标记仍在，本请求 ID 禁止第二次启动。

## 3. 实际消费（与计划对照）

| 单位 | 计划 NORMAL | 实际 | 说明 |
| --- | --- | --- | --- |
| physical_optimizer_step | 9 | 0 | 未到达任何 optimizer.step |
| offline_trunk_target_row | 1680 | 0 | 未发生任何 trunk forward |
| joint_record | 2 | 0 | 未发生 batch read |
| environment_construction | 0 | 0 | 全程为 0 |
| ledger charged events | - | 0 | ledger 仍 12 events、0 charged，SHA 仍 `1f5929a4...` |
| torch 导入 | - | 1 次实际模块加载 | `r75_frozen` 与 `r66_frozen` 各有一条 import 语句，第二次命中模块缓存 |
| 模型 load / 构造 | - | 0 | 未到达 `build_gate_only_controller` |
| forward / backward | - | 0 / 0 | - |
| 真实或合成 optimizer step | - | 0 | - |
| PPO update / replay | - | 0 | - |
| gate controller / optimizer 构造 | 各最多 2 | 0 / 0 | - |
| checkpoint load / write | load 最多 3 | 0 / 0 | 无任何 `.pt` 产生或改写 |
| 训练恢复 / RNG 恢复 | - | 0 / 0 | - |
| attempt 数 | 1 | 1 | 已耗尽，禁止第二次 |

`state.count` 中"已记录完成调用"为 0，因为计数器位于 `_load_paid_modules()` 返回之后；实际导入 attempt 是已知下界（numpy、torch、r66_frozen 已导入，v2_trainer 失败），未把缺失计 0。失败发生在 try 之外，故 runner 未产出 rich failure receipt、attempt counters 或 trial record，这与已登记边界一致，并由外层封装的 stdout/stderr 全字节与真实退出码补齐。

## 4. 不能得出的结论

资格未通过，因此不能称原生 PPO 资格、真实 Adam、真实序列化恢复、likelihood_alignment ≤ 1e-6、A2/A3 一致性、双头等值复制或任何策略效果得到验证。本报告不宣称科学结论、不宣称真实 collector、不宣称完整训练恢复，也不宣称通信模块完成。失败原因与技术接线有关，与 R58 回放数据、模型参数或科学配置无关，但本轮无权据此改写任何冻结件。

## 5. 外层封装自身的缺陷（已登记，未修改）

`r75_s1_outer_wrapper_failure.json` 记录的是录制层缺陷：wrapper 的 `except BaseException` 把自己用于传递退出码的 `SystemExit` 当成外层崩溃记录下来。实际 runner 退出码是 1。原件按字节保留，澄清见 `raw/s1/r75_s1_outer_wrapper_failure_clarification.json`。修正方式已写明（让 `SystemExit` 先通过），但因本请求不允许第二次启动，修正只登记不应用，交付的 wrapper 与真实执行的那一份逐字节相同。

## 6. 保护与历史

`ROUND46_CURRENT_TASK.json`（`cf3ad198...b39b`）与 `ROUND46_R2_V6_STOP_STATE.json`（`93a2f69f...03e18`）未修改、未暂存，本轮重新哈希确认。R59 分析日志、全部旧 R74≥16 记录、原件缺失状态与未知消费保持原样。`raw/runtime` 九文件字节未变，旧 schema 零文件原样保留，未 bootstrap、未重建、未清零。未入模型权重、未入原始 runtime、未 push、未 merge、未部署。

## 7. 若要继续需要什么（不在本轮授权内）

需要决策端明确授权一次"最小私有路径修复 + 重新验收"：把 `runs/comm_cost_v2_rerun_20260928` 等规范目录以与 R45/R58 相同的方式置于 `sys.path`（或改由 runner 在付费导入前调用 `r66.configure_source_imports()`），并同步登记新的执行身份与 code SHA。这会改动冻结源码，因此必须作为新的显式授权处理，不能由本轮自动执行。