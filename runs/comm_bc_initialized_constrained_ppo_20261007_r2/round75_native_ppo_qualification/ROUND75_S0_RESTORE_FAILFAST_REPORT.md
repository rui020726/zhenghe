# ROUND75 S0 restore / fail-fast 收口

请求 ID：`R75_S0_RESTORE_FAILFAST_20261015`（接续 `R75_S0_FINAL_PREP_20261014` 与 `R75_S0_DRIVER_CLOSEOUT_20261015`）
分支：`exp/round75-native-ppo-qualification`
进入轮次 HEAD：`7eff4a4218f275e2090ec2d9f42604b42bb6e7e9`（父 `c16ce9362f93993192cf9a1ec3330cba90d25f30`）
输出根：`runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/`

本报告追加，不改写 `ROUND75_S0_REPORT.md`、`ROUND75_S0_ACCEPTANCE_REPORT.md`、`ROUND75_S0_FINAL_PREP_REPORT.md` 及任何历史原件。

## 1. 本轮范围

源码修订、stdlib、AST、假对象反例、既有 JSON 追加与本地收口。**未导入 Torch/NumPy 实际 NN 路径，未做模型加载/构造、forward/backward、optimizer/PPO、环境、采样、checkpoint 写入、训练恢复或新实验。S1 仍未放行，本报告不构成执行许可。**

接续状态文件：`R75_RESTORE_FAILFAST_TASK_STATE.json`（仅接续说明，不改账本、不作实验授权）。

## 2. 上一会话已落盘、本轮保留未重做的部分

- RNG 桥接 helper 组（`_payload_bytes` / `_rng_report_from_payloads` / `_rng_tensor_meta` / `_raw_equal` / `_first_raw_difference` / `_rng_payload_differences`）、`_bundle_to_saved` 中由已加载原始载荷重建 `saved["rng"]`、`verify_restore` 追加三类 RNG 原始载荷逐值比较。
- `ObservedStepEntry.step` 在许可数 != 1 时**先记录 `phase="refused"`（`delegated=False`）并 raise，绝不委派**；step 自身意外扣费同样 raise 并记 `refused_further_work`。
- `load_state_bundle` 显式 `torch.load(path, map_location="cpu", weights_only=False)`，并在 docstring 中写明 torch 2.9.0 的 `_default_to_weights_only` 默认解析与完整 bundle 含 NumPy MT19937 状态元组这一事实。
- imports 增加 `os` / `time` / `traceback`。

## 3. 本轮补齐的三项

### 3.1 fail-fast 门禁（原为“跑完 4 步再统一报失败”）

新增 `_gate_a1`、`_gate_loaded_bundle`、`_gate_a3_start` 三个门禁，全部在 `run_s1` 内问题发生处立即 raise：

- **A1 门禁**（在第一个付费 native update 之前）：actor/trunk 未移动实测、八个可训练对象映射一对一且唯一、数量等于冻结合同声明值（从 `PLAN["gate_only_controller"]["r71_research_config_unchanged"]` 解析，不硬编码）、两个 critic 模块的四个参数均有非空 Adam 状态。
- **bundle 门禁**（在 restore 被信任之前）：磁盘文件 sha256 必须等于本run保存时登记值；14 个 resume 字段必须齐全；`data_identity` / `configuration` / `code_identity` / `contract` 不得为空；`code_identity` 必须等于盘上活体来源。
- **A3 起点门禁**（在第二个付费 native update 之前）：`restore_report.restored_exactly` 必须为真；完整 A1 基线 vs A3 起点逐值无差异；三类 RNG 原始载荷与 A3 起点载荷无差异；**把即将更新的活体对象重新捕获一次**，要求与已验证起点零漂移；bundle cursor 与已验证 cursor 一致。门禁返回 `binding`，把 controller/optimizer/episode 对象 id、update_index 与 cursor 绑定到紧随其后的那次 native 调用。

### 3.2 异常持久回执（新增 `raw/s1/`，不碰 `raw/runtime/`）

`run_s1` 加窄的外层 `try / except BaseException / finally`：

- 每个付费阶段开始前先登记 `state.phase`（precheck、paid_module_import、A2_controller_construction、replay_data_load、A1_critic_only_control_step、A1_baseline_capture、A2_native_update、A3_controller_construction、A3_bundle_load、A3_restore、A3_start_gate、A3_native_update、closeout）。
- 失败时把 phase、原始异常类型与消息、完整 traceback、真实入口计数、观测到的 step 记录、每 controller meter 快照、durable counter/ledger 快照，以 **append + fsync** 写入 `raw/s1/r75_s1_failure_receipts.jsonl`；不掩盖原异常（原样 re-raise）、不重跑、不退款。
- 读不到的项记 `unknown` 并进 `unread` 列表，**绝不默认 0**；资源从未打开时也记 unknown。
- `finally` 中再把真实 attempt 计数 append+fsync 到 `raw/s1/r75_s1_attempt_counters.jsonl`，保证无论成败计数都落盘；成功路径置 `closed_out` 后不重复写。

### 3.3 真实入口计数（带单位与保守上界）

七个单位：`paid_module_imports`、`replay_data_reads`、`checkpoint_file_loads`、`gate_controller_constructions`、`optimizer_constructions`、`native_ppo_update_attempts`、`optimizer_step_attempts`。计数从真实调用点逐一记录，`optimizer_step_attempts` 直接由 `ObservedStepEntry.records` 现算（被拒绝的 step 也是真实发生的 attempt）。`_conservative_attempt_bounds()` 在运行前声明各单位上界（含 A2/A3 两次 controller 初始化、R58 ep0 文件 load、bundle load），与实计数并列写入 `record["attempt_accounting"]`，使“改标签缩小真实 attempt”不可能。记录本身不触发任何额外操作，也不改变 NN 调用图。

## 4. 本轮发现并修掉的两个真实缺陷

1. **观测器每次成功 step 写两条记录**（`phase="delegated"` 一条 + `phase="completed"` 一条）。`run_s1` 的 `total_real_steps = sum(len(entries) ...)` 因此会变成 18，而冻结合同是 9：一次完美运行也会误触“the real step count does not match the frozen plan”门禁。修为**每次 attempt 恰好一条终态记录**，委派事实由 `delegated` 布尔承载；`failed` 记录补 `delegated: True`。
2. **`_gate_a1` 若按字面写死 8 会与活体映射脱钩**：改为从冻结合同解析期望对象数，映射不唯一或不等于合同值时拒绝。

## 5. 测试

| 运行对象 | 结果 | 退出码 |
| --- | --- | --- |
| `test_r75_s0_driver_closeout.py`（40 例） | OK | 0 |
| `test_r75_s0_restore_failfast.py`（34 例，新增） | OK | 0 |
| `r75_s1_precheck.py` | 68/68 通过，`checks_failed: []` | 0 |
| `r75_s1_runner.py`（无参） | 只打印冻结计划摘要，不导入任何 paid 模块 | 0 |

**受影响的旧选择器（如实披露）**：`test_r75_s0_driver_closeout.py` 中 5 个 selector 编码了本轮要修的旧行为（无/多许可仍然委派），已按新契约改写为“拒绝即 raise、delegate 次数 0、已消费许可不退、一次许可仅一次 delegate 且仅一条记录”：`test_a_step_without_a_new_permit_is_a_failure_not_a_note`、`test_a_double_charged_permit_is_detected`、`test_a_sequence_of_steps_reconciles_one_permit_each`、`test_a_failed_step_is_recorded_and_never_refunded`、`test_the_observer_delegates_everything_except_step`。其余 35 例未改动、未重跑旧 7 例、历史 31 例与 20 例套件。

新增 `test_r75_s0_restore_failfast.py` 覆盖：手工拼装的完整 bundle 形态载荷经 `_bundle_to_saved -> restore_full_state -> verify_restore` 正反例；期望 RNG 报告由已加载载荷重建而非恢复后活体状态；活体 RNG 与载荷不一致被逐值捕获；深位单字节/长元组末位/前缀/dtype 差异均被定位而不被 repr 截断掩盖；无/多许可零委派且不退额、一许可一次 delegate 一条记录、被拒绝 step 仍计为 attempt、冻结 9 步来自“一步一记录”；三个门禁各自拒绝场景与干净场景通过；用行号证明 A1 门禁在第一次付费 update 之前、bundle 门禁在 restore 之前、A3 起点门禁在第二次付费 update 之前、`load_state_bundle` 在 `restore_full_state` 之前；失败回执的必要字段、append 不覆盖、读不到记 unknown 不记 0、单位与保守上界并列。

## 6. 开发期真实失败（均不耗额度，均如实登记）

测试自身缺陷，逐条定位后修正，未用于掩盖代码问题：

1. 临时目录用 `tempfile` 默认根，落在沙箱不可写路径，`PermissionError`；改为在 R75 输出根内建 scratch 目录并清理。
2. 深位单字节差异用例的构造使两侧在 255 位相等，`_raw_equal` 正确返回 True；改用 `bytes([7])*2047 + bytes([8])` 后定位到 `first_diff_byte = 2047`。
3. AST 顺序用例误用 `ast.walk` 的 BFS 顺序当作源码顺序；改为收集 `(lineno, name)` 后按行号判定。
4. 对含 `FakeTensor` 的 bundle 做 `json.dumps` 深拷贝导致 `TypeError`；改为直接用 `build_state_bundle` 产物。
5. 篡改 module 载荷的用例把基线与篡改后状态都按同一份载荷比较，因而必然相等；改为“restore 复现 bundle 为真，但 `compare_full_states(baseline, a3_start)` 报出 `module:trunk:trunk.weight`”，这与真实语义一致（bundle 是事实源，篡改由 bundle 门禁的 sha256/code identity 与基线比较捕获）。
6. 断言 critic 非空 Adam 的对象数写成 2，实际为 4（两个 critic 模块各两个可训练参数）。
7. `_installed` 误用于 `capture_full_state` 产物（键名是 `_numpy_state` 而非 `numpy_state`），改用手动安装。

## 7. 比较方式如实声明

- `compare_full_states` 走的是 **tensor 字节 digest + dtype + shape**，不是真实 torch Tensor 的逐值路径。
- 三类 RNG 是**原始载荷逐值**比较（tuple/list 递归、tensor 按全字节加 dtype/shape）。
- 分头前向等价**不表示**梯度范数、全局裁剪系数或 Adam 更新轨迹相同；上一轮已更正的表述保留，本轮未改写裁剪算法去制造等价。

## 8. 实际消费与失败

零收费。canonical `raw/runtime` 前后字节一致：12 事件、0 charged、四计数器全 0、无残留锁；`r75_total_ledger.jsonl` SHA 前后均为 `1f5929a4e29ed0a1ad4c4a638190969dd9a84eebd6fadd2b402c0e8ffa20df0b`。零 torch 导入、零模型加载/构造、零 forward/backward、零 optimizer step、零 PPO 更新/重放、零环境构造、零采样、零 checkpoint 写入、零训练恢复、零 RNG 恢复。`raw/` 下仍只有 `runtime/`；`raw/s1/` 未被本轮任何测试创建（测试把 `S1_DIR` 指向输出根内 scratch 并清理）。

技术失败即第 6 节 7 项，全部为测试自身缺陷，均不消耗额度。**代码侧真实缺陷 2 项，见第 4 节，已修。**

## 9. 已知边界（未解决风险）

1. `try` 从 `_load_paid_modules()` **之后**才开始：付费导入自身失败不会被回执（该失败零消耗，由解释器 traceback 呈现）。这样做是为了保持 `r75_s1_precheck.main` 与 `_load_paid_modules` 在 `run_s1` AST 中的既有先后断言不被破坏。
2. 回执写盘自身失败时，原异常仍原样抛出，不回执（已在 except 内窄捕获并打印提示）。
3. 三个门禁只覆盖本轮合同点名的前置失败；`run_s1` 末尾原有的 13 项失败门禁保持为最终退出码判定，与前置门禁并存（不删除、不合并）。
4. torch 2.9.0 下 `weights_only=False` 的兼容性只在文档与假对象层论证，**未在真实 torch 上验证**（本轮禁止导入）。

## 10. 保护与工作树

`ROUND46_CURRENT_TASK.json`（`cf3ad198…b39b`）与 `ROUND46_R2_V6_STOP_STATE.json`（`93a2f69f…03e18`）未修改、未暂存，哈希复核一致。`analysis/round59-communication-decision-density/r59_analysis_run.log` 保持未暂存。历史失败、模型、账本、raw 全部原样。

## 11. 未来命令（当前不得执行）

```text
D:\project_env\python.exe runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/r75_s1_runner.py --run-s1
```

执行前须先跑纯文件预检（零收费）：

```text
D:\project_env\python.exe runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/r75_s1_precheck.py
```

S1 计划上界不变：≤9 物理 optimizer 步（12 normal + 4 technical / hard 16）、≤1680 target 行（5000 + 1000 / hard 6000，单次 ≤1024）、140 次 trunk 调用、2 次 joint record、0 次环境构造。这些只是账本算术，不授权额外消费。