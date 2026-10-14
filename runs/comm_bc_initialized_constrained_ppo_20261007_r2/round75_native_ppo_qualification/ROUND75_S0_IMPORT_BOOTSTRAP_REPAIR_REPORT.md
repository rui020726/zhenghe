# ROUND75 S0 导入引导修复收口

请求 ID：`R75_S0_IMPORT_BOOTSTRAP_REPAIR_20261015`（接续 `R75_S0_DRIVER_CLOSEOUT_20261015`）
分支：`exp/round75-native-ppo-qualification`
进入轮次 HEAD：`464cb95076bb69cebde74e9a07a07a4670c128ed`（父 `af9b80e97bfeaf382108f0615ee317d9f0ea551d`）
输出根：`runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/`

本报告追加，不改写 `ROUND75_S0_FINAL_PREP_REPORT.md`、`ROUND75_S1_NATIVE_QUALIFICATION_REPORT.md` 及任何历史原件。
最终 HEAD 以提交后 `git rev-parse` 实读为准，本文件不内嵌自称。

## 1. 本轮范围

源码修复（付费 runner 之外的接线）、纯 stdlib/AST/假模块/假 subprocess 测试、零收费预检、冻结件更新与本地收口。
**未导入 Torch/NumPy 实际 NN 路径，未做模型加载/构造、forward/backward、optimizer/PPO、环境、采样、checkpoint 写入、训练恢复或新实验。S1 仍未放行，本报告不构成执行许可。**

## 2. 接手时盘上状态

上一回合已改未提交：`r75_frozen.py`、`r75_s1_precheck.py`、`ROUND75_S1_RUN_PLAN.json`；`r75_s1_outer_exec_v2.py` 在盘上但被 `.gitignore:548` 忽略、未 add。本轮不 reset、不丢弃这些改动，在其上继续。

## 3. 修复一：模块身份检查原是自比较

### 3.1 缺陷

修前的 `_verify_canonical_bindings` 对每个 canonical 绑定只做一件事：`module = CANONICAL_SOURCES.get(key)`，随后 `if module is not CANONICAL_SOURCES.get(key)`。同一个字典读两次，恒为同一对象，这个条件永远不成立——它不能证明"实际导入的是 pinned 文件"，只能证明字典能返回自己。同一处的 `__file__` 与 sha256 检查仍有效，但对象身份这一层是空的：一个同路径、同 digest 的冒名对象，或一个被缓存/优先级反转顶掉的 `sys.modules` 条目，都可以在这一层之上静默通过。

### 3.2 修法

`r75_frozen.py` 现在把"这个模块真正用到的绑定"显式命名，再三方比对：

- `CONTROLLER_MODULE = CANONICAL_SOURCES.get("controller_module")` 与 `COST_MATH_MODULE = sys.modules.get("cost_math")` 在 canonical 导入之后立即绑定；`trainer` / `common` 就是 canonical 导入绑定的模块级名字。
- 对 trainer / common / controller 三项：模块级使用的绑定必须**是** `configure_source_imports()` 返回的对象；`sys.modules` 里对应 canonical 导入名（`v2_trainer` / `v2_common` / `v2_controller`）缓存的对象也必须是同一对象；解析后的 `__file__` 必须落在 `r66.CANON` 之下；实际读到的文件必须哈希到 pin。任一层不符即 `RuntimeError`，并把每层的判定结果（`used_is_helper_return`、`sys_modules_is_helper_return`）写进 `CANONICAL_BINDINGS["bindings"]`。
- `onpolicy`：除既有 `TIGHT_ROUND/onpolicy.py` 路径与 digest pin 外，新增 `sys.modules["onpolicy"] is onpolicy`，以及 `onpolicy.action_entropy` / `onpolicy.joint_log_prob` 必须是本模块绑定的那两个对象。
- `cost_math`：`configure_source_imports()` 已经导入它，因此从 `sys.modules` 读回，**不在本模块再次 import**；校验其 `__file__` 等于已配置 `common.STAGE_A_DIR / "cost_math.py"`，且 digest 等于新 pin `EXPECTED_COST_MATH_SHA256 = bca99af0…9026`（从该活体文件读出，17384 B）。
- 活的 `GAE_LAMBDA=1.0` / `ETA_LAMBDA=0.5` / `GAMMA=1.0` 仍从已配置模块回读，不用默认值替代。

`CANONICAL_SOURCES = r66.configure_source_imports()` 依旧位于第一个 canonical 导入之前（AST 实测 helper 行号小于全部 canonical 导入行号），这一层接线未动。

### 3.3 关于"初始等价"的表述边界

- 分头后的**初始 forward 与 loss 等价**可以直接验证（两个头由同一原头深拷贝构造，深拷贝不取 RNG）。
- **未裁剪梯度**应按共享 trunk / 双 critic / 两个独立头 / 以及"两头合并回单头"的映射分别核验，不把某一层的相等当作全体相等。
- 不要求两个独立头各自的梯度等于旧共享头；不宣称分头后的全局范数、全局裁剪系数或 Adam 更新轨迹必然等同 OLD；不为制造等价而修改裁剪算法。

## 4. 修复二：修订封装的"历史即拒绝"与占位符

### 4.1 缺陷

`r75_s1_outer_exec_v2.py` 的 `verify_pre_charge` 有一句：`raw/s1` 存在且非空即记 problem。首次失败原件按合同必须永久保留，因此这条门禁会使**任何**合法后续请求必然失败，并形成"要么删原件、要么永远不能跑"的压力。同时 `EXPECT_SIBLING` 是一处从未被任何校验引用的乱码/`None` 占位（`"canonical_trainer": (r"D:\code\\u6574…", None)`），留着会让人误以为存在一项同级来源校验。

### 4.2 修法

- 删除 `EXPECT_SIBLING`，不补造替代校验；canonical trainer/common/controller 的身份由 `r75_frozen` 导入期 pin 与 plan `source_identity` 承担，此处不重复声明。
- 新增 `FIRST_ATTEMPT_DIR` / `FIRST_ATTEMPT_REQUEST_ID` / `EXPECT_FIRST_ATTEMPT_FILES`（九件原件的实读 digest，见第 8 节）。`first_attempt_state()` 每次运行都重新哈希这九件并登记字节数；原件缺失或改变才是 problem，**原件存在本身不是拒绝理由**。目录中未登记的文件只作 `unregistered_files` 记录，不构成拒绝。
- `duplicate_request_refusal()`：request id 至多运行一次，跨目录生效。`R75_S1_NATIVE_QUALIFICATION_20261015` 在任何输出目录都被拒绝；同一 id 在目标目录或历史目录已有 attempt marker 也被拒绝；一个不同的、经明确许可的新 id 写自己的独立目录，不会因为历史存在而被误拒。
- 返回码归属保持三分：`child_returncode`（真实子进程退出码）、`wrapper_return`（本进程返回值）、`wrapper_exception`（封装侧异常）永不合并；`__main__` 守卫先 `except SystemExit: raise` 再 `except BaseException`，SystemExit 不再被写成封装崩溃；`.bin` 为权威，`.txt` 为 `errors="replace"` 解码并有损登记。封装侧异常记录带上从 argv 解析出的 request id。
- 本任务只以 `--prepare` 模式运行，未 `--launch`，未启动生产 runner。

## 5. 修复三：输出路由接缝（真实 child 与外层同目录同身份）

### 5.1 缺陷

`r75_s1_runner.py:50` 为 `S1_DIR = HERE / "raw" / "s1"`，CLI 只收 `--run-s1` / `--print-plan`。修订封装只把**外层**输出改到 `raw/s1_revised`，真实 child 的 bundle、失败收据、trial record 与 durable counters 仍会落回历史 `raw/s1`。只凭外层独立目录不能宣称后续 attempt 隔离。

### 5.2 修法（仅路径/请求元数据接线）

- `r75_s1_runner.py` 保留 `S1_DIR = HERE / "raw" / "s1"` 作为默认值（历史原件留在原处），新增 `FIRST_ATTEMPT_REQUEST_ID` 与可绑定的 `EXECUTION_REQUEST_ID`。
- 新增 `_bind_execution_identity(request_id, attempt_dir)`：拒绝空 id、拒绝历史 id（任何目录）、拒绝历史 `raw/s1` 及其子目录、拒绝输出根之外的目录、拒绝目标目录中已存在同一 id 的 attempt marker；通过后绑定 `S1_DIR` 与 `EXECUTION_REQUEST_ID`。bundle（`r75_s1_A1_full_state.pt`）、失败收据、trial record、attempt counters 全部经由这个可绑定的 `S1_DIR`，因此与外层记录同目录、同身份。
- CLI 新增 `--request-id` / `--attempt-dir`；`main(argv=None)` 在 `--run-s1` 下**先于预检与任何付费导入**要求并绑定这两项，缺失或不合规则写 stderr 并返回 2。
- 失败收据与 durable counters 现在记录 `request_id = EXECUTION_REQUEST_ID or PLAN["request_id"]`、`plan_request_id = PLAN["request_id"]`、`attempt_dir = str(S1_DIR)`：新执行身份与历史数据/预算 plan 的旧 request/head 分开，不互相覆盖。
- 中央账本不动：`r75_guard.R75Resources.production()` 仍指向唯一 `raw/runtime`；新 attempt 目录不构成第二账本，不清零、不新建。
- 修订封装把身份与目录显式传给 child：`argv = [python, runner, "--run-s1", "--request-id", <id>, "--attempt-dir", <s1_dir>]`，并原样记入 attempt marker。
- 科学面未动：损失、数据、RNG 推进、模型/Adam 构造数量、PPO 矩阵、epochs/minibatch/LR/order/clip/dual 一概未改。
## 6. 窄测试与零环境核验（全部实跑）

新增 `test_r75_s0_import_bootstrap.py`（38192 B，纯 stdlib + 假模块 + 假 subprocess + 独立临时目录，临时目录建在本输出根内并清理）：**30 项，全部通过，exit 0**。

覆盖内容：

- **AST 接线**：helper 行号小于全部 canonical 导入行号；`_verify_canonical_bindings()` 模块级调用；`cost_math` 不出现在本模块的 import 语句中；身份检查里必须存在 `used = {` 与 `sys.modules.get(`，且不存在"把同一个字典读项再读一次"的行；`onpolicy` / `cost_math` 的 pin 与 plan 及盘上文件三方一致；helper 来源是 pinned `r66_frozen.py`。
- **真实校验函数对假模块执行**：从盘上 `r75_frozen.py` 用 AST 抽出常量与 `_sha256_of` / `_verify_canonical_bindings` 真源码，在含假 `r66` / 假 canonical 模块 / 假 `sys.modules` 的命名空间中执行（**不导入 r75_frozen、numpy、torch 或任何 canonical 模块**）。一致用例通过并回读 GAE/ETA/GAMMA；反例分别拒绝：同路径同 digest 但不同对象、`sys.modules` 缓存被顶换、`__file__` 指向别处、活体值被改成 0.95、`cost_math` 缺失、`cost_math` 来自另一棵树（digest 相同，仅路径可辨）、helper 未返回绑定、`onpolicy` 辅助函数未绑定。
- **修订封装对假 child**：prepare 只验证不起 child 不建 marker；子进程成功记 `child_returncode=0 / wrapper_return=0 / wrapper_exception=None`；子进程非零记 1/5/None 且不合并；同一 request id 第二次返回 4 且不启动第二个 child；历史 id 在另一个输出目录返回 4 且不建 marker；**新 id 不因历史存在被拒**（回归旧缺陷）；原件被改/缺失返回 3 且不起 child；subprocess 抛异常不被写成 child 结局；`_request_id_from_argv` 解析。
- **`__main__` 守卫**：SystemExit(3) 原样透传且不写封装崩溃记录；封装侧 RuntimeError 被记录一次且带上 argv 中的 request id。
- **路由接缝**：runner CLI 缺身份/缺目录返回 2 且 `S1_DIR` 未被改写、历史原件目录文件集不变；历史 id 在任何目录被 `_bind_execution_identity` 拒绝；历史 `raw/s1` 及其子目录被拒绝；输出根之外被拒绝；合法新 id 绑定目录与身份；目标目录已有同 id marker 被拒绝；外层把同一 id 与同一目录传给 child 并记入 marker；AST 核实 `_bind_execution_identity` 在 `main` 中先于 `run_s1`、记录带 `plan_request_id` 溯源、中央账本未被复制。

**真实 CLI 正向/反向实测**（纯 stdlib 路径，未触达付费导入）：

| 调用 | 结果 |
| --- | --- |
| `--run-s1`（无身份） | stderr 拒绝，exit 2 |
| `--run-s1 --request-id R75_S1_NATIVE_QUALIFICATION_20261015 --attempt-dir <临时目录>` | stderr 拒绝（历史 id 任何目录），exit 2 |
| `--run-s1 --request-id R75_CLI_PROBE --attempt-dir <raw/s1>` | stderr 拒绝（历史目录），exit 2 |
| `--run-s1 --request-id R75_CLI_PROBE --attempt-dir <输出根外>` | stderr 拒绝（根外），exit 2 |
| 无参数 / `--print-plan` | 打印冻结计划摘要，exit 0 |

五次调用后 `raw/s1` 九件原件逐字节不变（哈希复核一致），临时探测目录未被创建。

**零收费预检**：`r75_s1_precheck.py` 复跑 **74/74 通过，`checks_failed: []`，exit 0**（plan 新增 `canonical_cost_math` 条目被 `check_source_identity` 一并核对）。旧 40/34 套件证据登记复用，未重跑。

## 7. 纯开发失败与更正（不掩盖）

1. **自设 pin 挡住自己的 prepare**：首次以 prepare 模式运行修订封装时，`EXPECT_FIRST_ATTEMPT_FILES` 的第一项被误写为占位 digest（`3723fb40b71bbe...`），随后 plan 的 `r75_frozen` digest 也随修复而变化，预检因此拒绝（exit 3）。这是纯开发失败，未启动任何 child、未消耗任何额度；已按实读 digest 重写并复验，最终盘上值与 `raw/s1` 九件逐一相符。
2. **守卫测试的 spy 被模块头覆盖**：早期版本的守卫测试在执行模块头之前安装 `record_wrapper_failure` 间谍，被执行后的真实定义覆盖，于是一次假 child 的封装侧异常把一条记录写进了 `raw/s1_revised/r75_s1_outer_v2_wrapper_failure.json`（request id `R75_TEST_GUARD_ID`，纯测试桩触发，无真实子进程）。该文件作为本次纯开发记录保留，已在第 8 节登记；测试改为在执行模块头之后安装间谍。
3. **一次 PowerShell 批量替换把新函数写到调用之后**：修复 `_verify_canonical_bindings` 时，内存替换与写盘未在同一条命令内正确排序，导致旧函数体与新函数体并存、新定义位于 `CANONICAL_BINDINGS = _verify_canonical_bindings()` 之后。当场复检行号发现并重建文件，最终以编译通过与第 8 节 SHA 为准；该中间态从未提交、从未运行。

## 8. 身份与冻结

| 文件 | 字节 | SHA256 |
| --- | --- | --- |
| `r75_frozen.py` | 41689 | `83813d6e06971a1e1fc1e25d9e9f94b0fa72310e300e037c0fa73d539c540b80` |
| `r75_s1_runner.py` | 86613 | `0d773a54b201f08bb02e075d763dc720e87f8e652b02a6e3bf2b9079f69b893f` |
| `r75_s1_outer_exec_v2.py` | 25056 | `84b81cad3d4d22763a2c4bced2a0d81ba85ba5870ff0a2cd327699990d322fc6` |
| `r75_s1_precheck.py` | 23423 | `f44ce536abd5e0368b8bb99ed0a09167ff9ba699dcb507c9c05d0e60472c9f0c` |
| `ROUND75_S1_RUN_PLAN.json` | 20451 | `ab7120d5c469f2936ea7162023ad70b1d04dc3d50955c4e6535c46e6480ce735` |
| `test_r75_s0_import_bootstrap.py` | 38192 | `c36a2939e17c539f110912e5db2cca5c83893bed16176200c0dca15f5f3b6ba8` |

字节未变：`r75_guard.py` `4d730f71…d55f`、`r75_replay_adapter.py` `0034fa40…4dc2`、`r75_s1_outer_exec.py` `b1c4fa4c…45115`（首次真实 attempt 的封装，保持原样）。

新增 pin：`EXPECTED_COST_MATH_SHA256 = bca99af08479398a7aec2a2f085163d31edd532f109b8cfc2043c1bfcfb49026`，来源 `D:/code/整合/zhenghe/runs/comm_cost_controller_stage_a_20260927/cost_math.py`（即已配置 `v2_common.STAGE_A_DIR`）。**注意：`zhenghe-round42` checkout 下的同名文件 digest 完全相同**，因此区分来源靠的是解析后的 `__file__` 路径，不是 digest；这一点已在 plan 的 `canonical_cost_math.note` 中写明。plan 的 `source_identity` 另新增 `canonical_cost_math` 条目，`r75_frozen` 条目的 sha256 与 revision 说明同步更新；科学内容、预算、矩阵、排除清单未改。

`raw/s1` 九件首次失败原件逐字节不变（digest 见 verdict），`raw/runtime` 仍为 12 事件、0 charged、四计数器 0/0/0，`r75_total_ledger.jsonl` SHA `1f5929a4…df0b`；两份 ROUND46 保护文件未动未暂存。

## 9. 未解决边界

- 本轮全部证据都在源码 / AST / 假对象 / 假 child 层取得，**未对真实 Torch 验证**；`r75_frozen` 的导入期绑定检查只在假树上执行过。
- `--launch` 未执行，生产 runner 未启动；未来任何 attempt 仍需决策端明确放行，且必须带新的 request id 与独立 attempt 目录。
- 计划中的 A1（一次 critic-only）+ A2/A3（各一次完整 native update）：9 normal physical / 1680 target 行 / 140 trunk 调用 / 2 batch read / 0 环境，仍是计划，未授权执行；原 R75 额度 normal 12 + technical 4（hard 16）physical、normal 5000 + technical 1000（hard 6000）target、单批 1024 不扩、不清零、不借旧 R74。
## 10. 收口：目标核对、完成度与本地提交

### 10.1 内部 goal 与当前任务 ID 的区别

按接续指令先用 `get_goal` 实读内部目标状态，返回 `{"goal": null, "completionBudgetReport": null, "remainingTokens": null}`：**本会话不存在活动 goal**。因此不存在"旧 goal 自动接续覆盖新任务"的风险，也无需按完工审计关闭任何目标；同时按指令未创建新 goal、未建立替代聊天或实验、未以本指令暂停科研。

上一回合末尾转入的"只读核 `R75_S0_FINAL_PREP_20261014` 收件"是一次独立小核验：其对象是早已交付并本地提交（`7eff4a4`）的历史请求，与当前 `R75_S0_IMPORT_BOOTSTRAP_REPAIR_20261015` 不是同一任务。该核验结论（已收到、已执行、内容三项相符、当前 HEAD 实读为 `464cb95076bb69cebde74e9a07a07a4670c128ed`）已回传，不构成本轮交付，也未改动任何科研状态或额度。

当前任务 ID：`R75_S0_IMPORT_BOOTSTRAP_REPAIR_20261015`；内部 goal：空；进入轮次 HEAD：`464cb95076bb69cebde74e9a07a07a4670c128ed`（父 `af9b80e97bfeaf382108f0615ee317d9f0ea551d`）。

### 10.2 四项修复 + 父→child 输出路由完成度

| 项 | 状态 | 证据 |
| --- | --- | --- |
| 1. `_verify_canonical_bindings` 自比较 → 真实对象身份 / `__file__` / sha256 / 活体值 | 完成 | 第 3 节；窄测试 30/30 |
| 2. 修订封装"历史即拒绝"门禁 + 删除 `EXPECT_SIBLING` | 完成 | 第 4 节；`--prepare` exit 0 |
| 3. 窄反例测试（假模块 / 假 subprocess / 假 child） | 完成 | `test_r75_s0_import_bootstrap.py` 30/30，exit 0 |
| 4. 零收费预检复跑 + 报告 / verdict / manifest + 显式本地提交 | 本回合收口 | 预检 74/74，exit 0 |
| 附加. 父→child 输出路由与执行身份绑定（`S1_DIR` 可绑定 + `--request-id` / `--attempt-dir`） | 完成 | 第 5 节；5 次真实 CLI：4 次 exit 2 拒绝 + 1 次 exit 0 |

### 10.3 复用的既有证据（未重跑）

`test_r75_s0_import_bootstrap.py` 的 30/30 与 `r75_s1_precheck.py` 的 74/74 原输出/退出码在上一回合已取得。本回合只做**字节身份复核**（下表 SHA 与登记逐一相符）与 AST 解析复验，未重跑套件、未重跑付费路径；旧 40/34 套件与 68 项预检证据按登记复用。

| 文件 | 字节 | SHA256 |
| --- | --- | --- |
| `r75_frozen.py` | 41689 | `83813d6e06971a1e1fc1e25d9e9f94b0fa72310e300e037c0fa73d539c540b80` |
| `r75_s1_runner.py` | 86613 | `0d773a54b201f08bb02e075d763dc720e87f8e652b02a6e3bf2b9079f69b893f` |
| `r75_s1_outer_exec_v2.py` | 25056 | `84b81cad3d4d22763a2c4bced2a0d81ba85ba5870ff0a2cd327699990d322fc6` |
| `r75_s1_precheck.py` | 23423 | `f44ce536abd5e0368b8bb99ed0a09167ff9ba699dcb507c9c05d0e60472c9f0c` |
| `ROUND75_S1_RUN_PLAN.json` | 20451 | `ab7120d5c469f2936ea7162023ad70b1d04dc3d50955c4e6535c46e6480ce735`（CRLF 346 行，保持） |
| `test_r75_s0_import_bootstrap.py` | 38192 | `c36a2939e17c539f110912e5db2cca5c83893bed16176200c0dca15f5f3b6ba8` |

五个源码文件 AST 解析全部通过。字节未变复核：`r75_guard.py` `4d730f71fb34ad411043c65141b6bcbb3f7d2f3c4668a6b53054e6400e35d55f`（CRLF 458 行）、`r75_replay_adapter.py` `0034fa40fbb55362646c84e8ef987e1aa01d72c619bd1c82a244dc88a0224dc2`、`r75_s1_outer_exec.py` `b1c4fa4c3be0f2983fb3baf10e16c1e8d9c18a37c0b0736bbdc693c436c45115`（首次真实 attempt 的封装，保持原样）。

### 10.4 付费禁止与剩余边界

本回合及收口全程：0 torch 导入、0 NumPy 实际 NN 路径、0 模型加载/构造、0 forward/backward、0 真实或合成 optimizer step、0 PPO 更新/重放、0 环境构造、0 采样、0 checkpoint 写入、0 生产 runner 启动（无 `--launch`）、0 账本 charged 事件。中央账本未被复制、未被清零、未被扩额。原 S1 请求 `R75_S1_NATIVE_QUALIFICATION_20261015` 永久禁跑；未来 attempt 需新 request id + 独立 attempt 目录 + 决策端单独放行。

### 10.5 本地提交

显式点名 `git add -f` 后单次本地提交。最终 HEAD/parent 以提交后 `git rev-parse` 实读为准，本文件不内嵌自称。不 push、不 merge、不 deploy。

### 10.6 本收口回合自身的一处开发失败（如实登记）

写 verdict/task-state/manifest 时，`block()` 辅助函数只拼接了 `lines[1:]`，丢掉了首行键名，导致追加进去的 verdict 丢失 `"goal_audit": {` 开头行、`json.loads` 失败。该次 verdict 已被写出为损坏状态，随后按**修改前 SHA256 `b530dd4a12a363c85bdedb354901686292327e1190765953e7ceb4fe5f00d0c9` 逐字节恢复**，再以修正后的函数重写 verdict、task-state 与 manifest 并重新做 JSON 校验。损坏中间态从未提交、从未执行。同一回合还发生一次"文件尾部无换行符"导致的断言失败（`assert v.endswith("}\n")`），只读复核尾部字节后改为按实际尾部形态切片，未产生任何写入。

上述失败不消耗任何额度：0 torch、0 模型、0 optimizer/PPO、0 环境、0 checkpoint 写、0 账本事件、0 生产 runner 启动。
