# ROUND75 S0 最终续行修订与完整收口

请求 ID：`R75_S0_DRIVER_CLOSEOUT_20261015`（接续 `R75_S0_FINAL_PREP_20261014`）
分支：`exp/round75-native-ppo-qualification`
进入轮次 HEAD：`c16ce9362f93993192cf9a1ec3330cba90d25f30`（父 `0f8574accae16676a75a248da180be9d49e9fa04`）
输出根：`runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/`

本报告追加，不改写 `ROUND75_S0_REPORT.md`、`ROUND75_S0_ACCEPTANCE_REPORT.md` 及任何历史原件。

## 1. 本轮范围

源码、stdlib、AST、既有 JSON、零收费预检与本地收口。**未导入 Torch/NumPy 实际 NN 路径，未做模型加载/构造、forward/backward、optimizer/PPO、环境、采样、checkpoint 写入、训练恢复或新实验。S1 仍未放行，本报告不构成执行许可。**

## 2. 已验修复的沿用（不重跑）

上一会话的两处最小 diff 本轮回填实跑证据，未重跑其 7 例反例，也未重跑历史 31/20 套件：

- KL 返回键：`r75_frozen.py:482` 现读 `totals["kl"]`（真实累加器写于 :455），字段名 `approx_kl` 保留。溯源 r66 原件 `r66_frozen.py:875` 读 `totals["kl"]`。
- 非零 `pre_meter` 差分：新增 `alignment_end` 累计快照（:293），`alignment_meter = _meter_delta(pre_meter, alignment_end)`（:294），`epochs_meter = _meter_delta(alignment_end, meter.snapshot())`（:468）。
- `r75_frozen.py` 本轮字节未变：SHA `a4b556a687c3aff9650404d11149cb387879a4b4009c85502fc970d7bbe27996`，536 行。
- `r75_guard.py` 字节未变：SHA `4d730f71fb34ad411043c65141b6bcbb3f7d2f3c4668a6b53054e6400e35d55f`。机制与源 pin 未弱化。
- `ROUND75_S1_RUN_PLAN.json` 字节未变：SHA `34024113349cdb2f774b9d2751ec0e1f46800ae2fb5720470254432b16ae26c6`。

## 3. 七个接缝缺口：盘上实况、修法与验证

逐项对照盘上 runner 源码确认，不把计划文字当实现。

### 缺口 1：saved 快照与活体对象共享可变储存（确认存在）

旧 runner 以 `module.state_dict()` 原引用构造 `_module_payloads`；`_adam_payloads` 虽 clone 了 `exp_avg`/`exp_avg_sq`，但 `step` 取的是 `state.get("step")` 原引用。A2 的 in-place step 自增与参数写入会改写这份“已保存”状态，A3 于是恢复到 A2 的更新后值而非 A1 值——A2/A3 确定性比较在结构上不可能发现任何问题。

修法：新增递归 `_clone_value`（tensor detach/clone、dict/list 递归、**保留容器类型**）、`_clone_module_state`，并把「报告层」与「载荷层」彻底分离。

### 缺口 2：把摘要当运行状态（确认存在）

`capture_full_state` 把 `state["step"]` 变成 `_tensor_digest(...)` 字典，`restore_full_state` 又把这个字典当作 Adam 的运行 step 赋回去。

修法：`_optimizer_payload` 构造不可变的 `optimizer.state_dict()` 形态载荷（state 按参数索引、param_groups 带真实超参与索引映射），恢复走真实 `optimizer.load_state_dict`；bootstrap 无 state 对象保持缺失，不补造。附带发现并修掉一个真实类型漂移：`_clone_value` 原会把 tuple 归一成 list，会使 param_groups 的 `betas` 在恢复后类型改变、进而在逐值比较里产生假差异。

### 缺口 3：bundle 不完整且 A3 未从磁盘恢复（确认存在）

`save_state_bundle` 只写模块/Adam 载荷与三类 RNG，漏 lambda/cursor/数据配置/结构身份；A3 的 `restore_full_state` 完全没读已写的 `.pt`。

修法：`build_state_bundle` 写全 14 个字段（五模块载荷、全部 Adam 值与八对象映射、optimizer 载荷含 param-group 超参、三类 RNG、lambda、配置、cursor、数据与代码身份、契约）；`load_state_bundle` 从实际序列化文件读回；`_bundle_to_saved` 适配；A3 只从磁盘文件恢复。

### 缺口 4：比较器漏掉科学字段（确认存在）

旧比较只比两个 native 返回，并硬编码 skip 掉 `trunk_entry_meter`、`batch_read`、`gate_records`、`gate_diag`；完全没有 A2/A3 终态全状态比较；`meter_additivity` 只记录、不影响退出码。

修法：冻结四个状态（A1 基线、A2 终态、A3 加载后起点、A3 终态）逐 tensor 值/dtype/shape 比较五模块、全部 Adam 状态与 param groups、lambda/RNG/cursor/数据与代码身份；`compare_scientific_returns` 只用 plan 预先冻结的排除清单；`reconcile_trunk_meter` 对 A3 第二控制器的绝对累计读数做“核对”而非“diff”；13 项失败门禁任一不符即非零退出。

### 缺口 5：固定非默认 RNG 三元组（部分存在）

RNG 有保存/恢复，但保存的是环境残留态，证明力不足；A3 恢复顺序正确纯属偶然。

修法：`_fixed_rng_triple` 在捕获前安装具名、非默认的三类 RNG 状态；恢复发生在 A3 构造之后、更新之前；`verify_restore` 逐值证明起点一致。`DATA_ORDER_SEED` 实际 shuffle 未改（`r75_native_ppo_update` 自建 `random.Random(DATA_ORDER_SEED:update_index)`，不读全局流）。

### 缺口 6：真实 step 入口观测（确认缺失）

没有任何 attempt/completed/failed 记录与逐许可对账；`trunk_moved` 是硬编码 `False`。

修法：`ObservedStepEntry` 被动代理，绑定同一 production counter——预扣发生在 step 调用之前，因此对账窗口取「上一步退出读数到本次进入读数」的增量，必须恰为 1（不缺许可、不双扣），且 step 自身不得计费；代理自身不 charge、不退额、不开新 root、不旁路。`run_a1` 改为从活体参数测量 trunk 不动与八对象覆盖，覆盖不足如实报告。

### 缺口 7：预检顺序、None 上限与字面配置（确认存在，且更严重）

- 预检在 `_load_paid_modules()` **之后**才调用，paid 模块已导入。
- `per_charge_cap` 从 raw counter JSON 读出 `None` 后直接比较，抛 `TypeError`。
- plan 内存的是仓库相对路径，按 CWD 解析，换目录即 `FileNotFoundError`。
- runner 把 GAE/eta/epochs/clip 等活体配置写成字面量。

**实测结论：该预检此前从未通过**（从仓库根跑崩在 `per_charge_cap`，从其他目录崩在相对路径），因此它实际上没有门禁过任何东西。

修法：预检移到 paid 模块之前；`verified_per_charge_cap` 从两个独立已验来源取 1024（guard 源属性 + 冻结契约），raw JSON 缺省为预期、若存在且矛盾即失败，**绝不与 None 比较**，不改写旧 counter、不 bootstrap；`resolve_path` 统一按仓库根解析（仓库根通过向上查找 `.git` 获得）；`_live_configuration` 从活体模块命名空间读全部配置值；适配器字节未改，CWD 独立性由调用方解析绝对路径提供。

## 4. 反例测试（新增，纯 stdlib + 假对象）

`test_r75_s0_driver_closeout.py`：40 例，7 类。无 Torch、无 NN、无环境、无采样、无 checkpoint 写入、无账本写入。

| 运行对象 | 结果 | 退出码 |
| --- | --- | --- |
| 旧 runner（`d68153da…`，本回合初盘上版本） | 4 failures + 30 errors | 1 |
| 新 runner（`d3f7e9a9…`） | 40 OK | 0 |

开发期我修过自己的测试缺陷（假对象 state 键控用 `id()` 而非参数对象、观测器许可窗口取错区间、`ast.unparse` 渲染形式、三个目录切换用例、非一对一映射用例的断言形式），全部如实记录，未用于掩盖代码问题；其中后续 2 failures 里的 etas tuple→list 失败经核实是驱动真实缺陷而非测试缺陷，已在第 9 节更正段单独登记，不影响本节其余缺陷清单。

预检实跑对比：

| 运行条件 | 修前 | 修后 |
| --- | --- | --- |
| 仓库根 | exit 1，`TypeError: '<=' not supported between instances of 'int' and 'NoneType'` | 68/68，exit 0 |
| R75 目录 | exit 1，`FileNotFoundError`（相对路径） | 68/68，exit 0 |
| `C:\` | exit 1，`FileNotFoundError` | 68/68，exit 0 |

`python r75_s1_runner.py`（无参）只打印冻结计划摘要，不导入任何 paid 模块，exit 0。

## 5. 未执行与未覆盖

- **S1 未执行、未授权**：`--run-s1` 一次都没有运行。`ROUND75_S1_RUN_PLAN.json` 状态仍为 `PREPARED_NOT_AUTHORIZED_NOT_EXECUTED`。
- 预检与 runner 仅编译与只读打印，未进入付费路径。
- 计划自列的未覆盖范围保持不变：torch 导入与 state-dict 映射结果、真实 `likelihood_alignment` 结果、任何真实或合成 optimizer step / PPO 更新 / 重放、真实 collector 与闭环仿真、任何评估块、多批训练与科研 checkpoint、任何通信策略效应/成本/任务结论。
- 新反例只覆盖本轮七项接缝缺口；未重跑旧 7 例、历史 31 例与 20 例套件。

## 6. 实际消费与失败

零收费。canonical `raw/runtime` 前后字节一致：12 事件、0 charged、四计数器全 0、无残留锁；`r75_total_ledger.jsonl` SHA 前后均为 `1f5929a4e29ed0a1ad4c4a638190969dd9a84eebd6fadd2b402c0e8ffa20df0b`。零 torch 导入、零模型加载/构造、零 forward/backward、零 optimizer step、零 PPO 更新/重放、零环境构造、零采样、零 checkpoint 写入、零训练恢复、零 RNG 恢复。

技术失败（都不消耗额度，均保留原因记录）：预检修前两类崩溃；新测试开发期 6 failures + 10 errors 与后续 2 failures——其中 1 项已核实为驱动真实缺陷（typed optimizer 恢复后 etas 由 tuple 变 list，即第 3 节缺口 2 附带发现并修掉的真实类型漂移，由 	est_r75_s0_driver_closeout.py 第 405/407/412 行回归断言捕获），不得归为测试自身缺陷；同 run 另一 failure 无逐条工具记录，按未知登记；6 failures + 10 errors 为测试自身或假对象缺陷（清单见第 4 节），逐条定位后修正；并行写文件互相覆盖导致两处补丁丢失，已改为顺序写入并复跑验证。

## 7. 保护与工作树

`ROUND46_CURRENT_TASK.json`（`cf3ad198…b39b`）与 `ROUND46_R2_V6_STOP_STATE.json`（`93a2f69f…03e18`）未修改、未暂存，哈希复核一致。R59 分析日志保持未暂存。历史失败、模型、账本、raw 全部原样。

一次性计划生成器 `_s1_plan_gen.py` 更名为 `r75_s1_plan_gen_archived.py` 明确归档，字节未变（SHA 前后均 `ba884cb85322b535fcf71aa5a090b8fb8fc283fc271c7d1cd8f51be68789bfb`）；`__pycache__` 构建产物已清除；本回合自用临时副本已删除。**更正（详见第 9 节）**：其修前内容并不能从 git HEAD 取回——修前 runner（SHA d68153da…）与修前 precheck 从未进入任何提交（git log --all 对两路径均为空，c16ce936 的 tree 不含二者），盘上递归哈希扫描亦无一致字节；旧原件登记为缺失，仅有当时 SHA 与工具记录作为独立证据层，不重造、不以重建副本冒充原件。

## 8. 未来命令（当前不得执行）

```text
D:\project_env\python.exe runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/r75_s1_runner.py --run-s1
```

执行前须先跑纯文件预检（零收费）：

```text
D:\project_env\python.exe runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/r75_s1_precheck.py
```

S1 计划上界：≤9 物理 optimizer 步（12 normal + 4 technical / hard 16）、≤1680 target 行（5000 + 1000 / hard 6000，单次 ≤1024）、140 次 trunk 调用、2 次 joint record、0 次环境构造；计划后 normal 余 3 步 / 3320 行，technical 保留 4 步 / 1000 行，hard 合计 7 / 4320。这些只是账本算术，不授权额外消费，也不补 R74 旧额度。
## 9. 本轮更正与登记（R75_S0_GIT_FINALIZE_20261015，零环境）

本段更正第 4、6、7 节的两处证据表述并补登身份，零环境：未导入 Torch/NumPy 实际 NN 路径，未做模型加载/构造、forward/backward、optimizer/PPO、环境、采样、checkpoint 写入、训练恢复、RNG 恢复，未重跑任何测试，未改 canonical `raw/runtime` 一个字节，未改任何已验证驱动/native/guard/adapter 运行逻辑。S1 仍未授权、未执行。

1. **失败归属更正（对应第 4、6 节）**：原写“后续 2 failures（均为测试自身或假对象缺陷）”不准确。已核实其中 `betas` 由 tuple 变 list 的失败（typed optimizer 恢复路径）是驱动 `_clone_value` 的真实缺陷，与第 3 节缺口 2“附带发现并修掉一个真实类型漂移”为同一事实，由 `test_r75_s0_driver_closeout.py` 第 405/407/412 行回归断言捕获；不得记为测试自身缺陷。同 run 另一 failure 无逐条工具记录，按未知登记，不推定归属。第 4 节所列假对象 state 键控、观测器许可窗口、`ast.unparse` 渲染、目录切换、非一对一断言形式等失败仍各自如实列为测试自身缺陷，本更正不扩大也不缩小该清单。
2. **旧原件可恢复性更正（对应第 7 节）**：原注“其修前内容在 git HEAD 中可完整取回”不成立。修前 runner（SHA `d68153dacb5b1023688ca67bb253a5410d0847e3396b13e76cda80ec8bdc3bd5`）与修前 precheck 均未被任何提交收录：`git log --all -- <两路径>` 均为空，`c16ce9362f93993192cf9a1ec3330cba90d25f30` 的 tree 只含 `r75_frozen.py`、`r75_guard.py`、`r75_replay_adapter.py` 与 S0 阶段文档/测试；对输出根及仓库根的只读递归哈希扫描未发现 `d68153da` 一致字节。据此登记：旧 runner 原件在盘上与 git 中均**缺失**；修前 precheck 原件同样缺失且其当时 SHA 无记录（未知）。二者仅有当时记录的 SHA 与上一会话工具调用记录作为独立证据层；不重造旧试验，不把任何重建副本冒充原件。
3. **身份登记**：当前 driver/precheck/native/guard/adapter/plan/两套测试/报告/verdict/归档生成器的完整 SHA、字节数与角色已追加登记于 `ROUND75_GIT_EXTERNAL_MANIFEST.json` 的 `committed_final_prep_revision` 段，`audited_code_head` 等旧字段保留并标注历史范围。`ROUND75_S1_RUN_PLAN.json` 经核为历史冻结的数据/预算/比较合同（字节未变，SHA `34024113…`），其 `source_identity` 只绑定上游 canonical 源与 r66 frozen，不含当前 driver 身份；当前 driver 绑定以上述 manifest 新段与本次本地提交为准，plan 本身不作改动（不改矩阵、配置、排除清单、预算）。最终 HEAD 以提交后 `git rev-parse` 实读为准，本文件不内嵌自称。
4. **消费与边界**：本更正段为零环境写入与显式本地提交。累计消费不变：canonical `raw/runtime` 仍为 12 事件、0 charged、四计数器 0/0、无残留锁；历史非计划消费 0；本回合新增 0。S1 继续等待决策端资格放行，本报告任何内容均不构成执行许可。