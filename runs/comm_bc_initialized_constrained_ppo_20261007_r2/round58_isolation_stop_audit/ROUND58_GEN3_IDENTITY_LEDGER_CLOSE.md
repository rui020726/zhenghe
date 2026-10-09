# ROUND58 gen3 identity / ledger close

**结论：本轮只读审计与私有修复回归完成；不续训。三个已提交 batch 均为 `unverified`，不能作为本轮已通过门禁的续训源。** 原失败门禁、checkpoint、batch、claim、tape、journal 与旧快照均未更改。

## 1. 接手、操作边界

- 仓库：`D:\code\整合\zhenghe-round42`
- 分支：`exp/round58-prior-strength-prospective-trial`
- 起始 HEAD：`e644b26e8607f50369d59e9bb0bf588f9eb32b47`
- 开始与结束均未启动环境；本轮不运行 PPO gate 全流程、不恢复 RNG、不加载 checkpoint 进入训练。
- 本轮副作用计数：环境/仿真/新流/episode/claim/slot 写入/optimizer.step/PPO 重算/RNG恢复/checkpoint 写入/评估格均为 **0**。CPU forward 为0。仅只读反序列化检查旧 checkpoint 元数据；没有将 optimizer、模型或 RNG 状态恢复到训练运行时。
- 结束进程检查未发现 `python`/`pythonw` 执行进程。
- 两个保护文件结束 SHA256 与接手值一致，且未暂存：
  - `ROUND46_CURRENT_TASK.json`: `cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b`
  - `ROUND46_R2_V6_STOP_STATE.json`: `93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18`

## 2. 身份门禁：具体根因与字段证据

实际抛错位置为 `runs/comm_one_shot_gate_resume_20261003_v2/r42_analysis.py:342-345`。它把三个判断合成一条错误：payload `training_version`、`comm_action_semantics`、`freeze` 必须分别等于 frozen version、`SEMANTICS`、运行时 `round42_freeze`。

逐字段核对三个 seed 的 before/after 共六份 payload：

| 字段 | 期望 | 实际 | 结果 | 来源/说明 |
|---|---|---|---|---|
| `training_version` | gen3 freeze 中完整版本串（见机器审计 JSON） | 六个 payload、`training_state`、batch、commit、sidecar 都是同一版本串 | 匹配 | 期望来自 `ROUND58_GEN3_FREEZE.json`; 实际来自 CPU 只读 payload及相应 sidecar |
| `comm_action_semantics` | `one-shot-send-skip/1` | 六个 payload 与 `training_state` 均为 `one-shot-send-skip/1` | 匹配 | 期望为冻结动作合同及 R42 `SEMANTICS`; 实际为 payload |
| action eligibility | `first-report-only/1` | batch/commit/payload 均为该值 | 匹配 | R42 首报资格合同；不是复合断言中的失败项 |
| checkpoint schema | `constrained-cost-state/2` | 三个 post payload 均为该值 | 匹配 | 冻结 trainer schema 与实际 payload |
| freeze 身份整体相等 | R42 runtime 的未包装 `round42_freeze` | payload 的 freeze 已由私有 checkpoint contract 包装 | **不匹配** | `r58_checkpoint_contract_repaired.py` 的 `_attested()` 会把 gen3 `training_version` 和 `round58_repair_attestation` 注入 freeze；旧 gate 却逐对象要求 `payload.freeze == runtime["round42_freeze"]` |
| repair attestation | `ROUND58_GEN3_FREEZE.json:repair_identity` | payload top-level、`training_state`、freeze、sidecar 内均闭合一致 | 匹配 | 侧栏逐文件 SHA 与 checkpoint SHA 一致 |
| seed/游标 | before `[0,0,0]`; after `[4,1,8]`（episode/update/optimizer steps） | 三个 seed 均符合 | 匹配 | `training_state`；仅作为存档内容读取 |

因此不是 `training_version` 常量错：其字段和动作语义都匹配。也没有证据表明实际载入了错误 BC 模型或研究配置。真正拒绝来自**私有序列化合同增加的 freeze provenance 与旧 R42 完全相等比较不兼容**。错误文案过于笼统，但其被检查的 freeze 比较确实失败。

需要保留的证据限制：失败进程没有单独序列化其内存中的 `runtime["round42_freeze"]` 对象。因此可以证明 payload freeze 包含 serializer 添加的 attestation、旧比较要求未包装对象完全相等；不能恢复运行时对象每一个字节来报告所有潜在差异。本报告不把这项缺失伪装为完整字段级内存快照。完整字段、实际值、来源哈希与逐 seed payload 清单在 `ROUND58_GEN3_IDENTITY_LEDGER_AUDIT.json`。

门禁失败前已经有三个 seed 的 batch0，并非 seed11 失败后偷偷续跑：真实冻结执行顺序在 `r46_execute_r2.py:_run_first_four_and_all_gates` 中先按 seed 11/22/33 各运行首四集并持久化首批，再进入第二个 seed 循环逐个 gate。seed11 gate 首先失败，所以 seed22/33 gate 未到达，且没有 episode4+ 续训。这是执行器明确的两阶段顺序，不是事后从聊天推断。

## 3. 三个已提交 batch：只读资格判断

三个 batch 记录均指向各自 episode0–3 的4条 row/tape，claim、slot、batch record、commit、pre/post checkpoint 及 sidecar 的哈希可闭合。每批 265 samples、4 epochs、8 minibatches、8 optimizer steps；全局 optimizer hook 有24条 `step_invoked`。同权重似然差分别为 `2.34e-13`、`1.71e-12`、`9.38e-13`，均低于 `1e-6`。每批 `g_hat=0`、λ `1→1`。这些只证明存档记录所载内容和文件关系，不替代首批 gate。

| seed | BC 初始参数 SHA | post 参数 SHA | checkpoint 游标 | gate | 只读资格 |
|---|---|---|---|---|---|
| 11 | `37303c49…7f0330` | `acb74891…56d34d` | `4 / 1 / 8` | failed：freeze identity | `unverified` |
| 22 | `e8531e0a…2479b0` | `149c2798…f41e300f` | `4 / 1 / 8` | not reached | `unverified` |
| 33 | `10bae22f…e57a5` | `a1d91406…17a83` | `4 / 1 / 8` | not reached | `unverified` |

没有发现足以判 `invalid` 的明确训练数学偏差；也没有充分证据判为 `qualified_for_later_resume_review`。原因：seed11实际 gate 没通过；其余 gate 未到达；记录的 gradient diagnostics 是 actor/base-prior 分解，不是 critic→trunk 隔离的逐次运行观测；另有 PPO 调用计数接缝断开。R53隔离源码和 checkpoint attestation 可证明被冻结/声明的实现身份，不能单独证明运行时对象在每次 PPO 更新时绑定成功。

重要核验边界：失败 gate 在复合身份断言处停止，故它未完成后续 tape/reference/任务成本等检查。批记录的 likelihood/g_hat/λ不应扩大解释为完整门禁通过。

## 4. 4 与 12：快照为何落后，以及额度对账

原 `round58_gen3/raw/runtime/ROUND53_EXECUTION_LEDGER.json` 保留原样，记录4 normal slots。其最后写入为 `2026-10-09T01:50:44.951511Z`，恰在 seed11 首批 commit（`01:50:45.417Z`）之前，时间上对应首个 seed 的4次环境消费。之后有11条 `derived_snapshot_warnings`：batch record 数从1增长到3，但每次都因 `PPO update count mismatch: batch records=N, observed r42 trainer.ppo_update=0` 而 defer。

根因在私有 prior adapter：它动态编译 patched `ppo_update`，调用 `patched_fn`，然后替换 `trainer.ppo_update`；这绕过先前安装在 trainer method 上的 R42 PPO invocation counter。于是 optimizer 的24条实际 step hook 和三个 batch commit 都存在，但 R42 的 PPO counter 保持0。非致命 derived writer 保住了核心 claim，却没有刷新旧快照。故 4 是运行快照时间点的部分值，不是最后消费量。

本轮新增的私有 `derive_snapshot()` 将 claim/slot 消费、durable batch commits、PPO hook observed、optimizer hook observed 分栏；即便部分快照允许显示进度，只要 counter gap 就不报成功，final closure 失败关闭。另有 `bridge_ppo_update_counter()`，供新的冻结执行身份在 prior wrapper 外层绑定；**没有对旧 execution identity 热修或重跑**。

按12份原 claim 与12份 slot 独立对账：均为 `complete/normal`，无重复、缺失或类别不匹配。分类字段原样保留。累计账：

- 历史环境消费：16（历史类别冲突仍未重标）
- gen3 本代 normal：12 / 1024；reserve：0 / 24
- 累计环境消费：28 / 1064；hard headroom：1036
- normal 计划剩余：1012；reserve 余量：24
- optimizer：旧历史48（既有证据支持但不并入新轨迹）；gen3 24实际调用；累计有证据支持72
- gen3 committed batches：3；本轮审计承认的有效/通过门禁 batch：0
- 评估：0 / 640

已消耗12次正常环境任务的最终训练资格仍需逐批资格门禁决定；不能因待核验就删 claim 或退额。

## 5. 新私有修复与测试覆盖

新增 `r58_identity_ledger_close_v2.py`，身份 `round58-gen3-identity-close/1`：

- 生成与实际 `_attested()` 同构的唯一 gen3 expected freeze；对训练版本、动作语义、freeze、top-level/state attestation逐字段报错，不接受“一揽子新旧版本”。
- `call_first_batch_gate_with_attested_freeze()` 只包装 expected freeze，然后调用既有 gate 签名；不改 gate 其他断言，不改 canonical trainer。
- `bridge_ppo_update_counter()` 在新执行身份上保留 R42 invocation counter。
- claim/slot 多根归并保持 consumed attempts 唯一计数；batch commits 与 hook counts 分离；partial snapshot不能伪装 final success。

新增窄测试 **11项通过，退出码0**。覆盖旧相等比较的独立反例、正确gen3身份接受、版本/语义/freeze/attestation/旧代次拒绝、实际私有 `_attested` metadata构造与恢复拒绝、多根seed claim-slot对账、部分/最终状态以及 PPO counter bridge。测试中使用内存payload、假对象与spy，不调用真实或合成 optimizer.step，不写真实checkpoint。

实际第一批 gate 全流程没有重跑：其后段会进行大范围 tape/reference 读取，且本轮不能把改后比较结果冒充旧运行 gate 回执。修复 helper 的 spy 测试覆盖了其调用签名和 freeze 操作，但不等同于真实 runtime gate 已通过。

## 6. 文件与提交

新增/修改仅限本审计目录：

- `r58_identity_ledger_close_v2.py`
- `test_r58_identity_ledger_close_v2.py`
- `audit_r58_gen3_identity_ledger.py`
- `ROUND58_GEN3_IDENTITY_LEDGER_AUDIT.json`
- `ROUND58_GEN3_IDENTITY_LEDGER_TEST_RESULTS.json`
- 本报告 `ROUND58_GEN3_IDENTITY_LEDGER_CLOSE.md`

后续提交前会再次执行 `git diff --check` 和本次11项窄测试，并显式只暂存上述文件。任何权重、tape、日志、claim、checkpoint、旧快照及保护文件不入提交。

新代码仍是**审计/修复适配层**，尚未绑定为可执行训练许可；三批仍 `unverified`，当前科学 continuation 游标不能据此确认。若以后要审议从 episode4 续跑，至少需要负责人另行批准；基于精确 gen3 identity重新执行真实 gate（含 critic isolation、optimizer parameters、完整 tape/GAE/奖励/g_hat/λ闭合）；修复并验证 PPO counter桥和派生快照；解决运行时 expected freeze原件缺失造成的身份可审计性。不得自动恢复这三个 checkpoint。
