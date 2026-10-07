# ROUND48 状态更正（STATEMENT_CORRECTIONS.md）

任务：`ROUND48-READONLY-POLICY-DRIFT-PROBE`。本文件只更正措辞与登记口径，不改变任何历史判定、原件或 `development_go=false`。

## 更正 1：ROUND47 报告中 "s11、s33 成本显著上升（CI 排除下降）" 措辞不成立

- 原文（ROUND47_COST_GAP_REPORT.md 第二节）称 s11、s33 相对 BC 的成本上升"显著"。
- 实际未舍入配对 bootstrap CI（PCG64(20261007)，10000×16，linear）：
  - s11 PPO−BC：`ci_low = 0`（精确 0），`ci_high = +1406.625`；
  - s33 PPO−BC：`ci_low = 0`（精确 0），`ci_high = +1249.5`；
  - s22 PPO−BC：`ci_low = −928.434375`，`ci_high = 0`（精确 0）。
- 端点恰为 0 不是舍入：逐场景差中 13/16 为 0，自助重采样中约 3.5% 的均值恰为 0（全部抽中零差场景），2.5 分位落在 0。
- 更正为：**"s11、s33 均值成本上升，但 CI 下端恰为 0，未严格排除 0；s22 均值下降，CI 上端恰为 0，同样未严格排除 0。三个 seed 相对 BC 均无统计显著的成本变化。"**
- 该更正不涉及 V10/V9 的任何判定（它们从未做过此显著性声明）；`development_go=false` 不变。

## 更正 2：ROUND46-R2-V10 累计账本两处派生汇总漏计（已在 ROUND47 登记，此处汇总）

- `environment_constructor_calls`：账本 481 → 正确 **482**（attempt2 claim 位于 V7 根，公式硬编码 +2 漏计其 1 次构造）。
- `technical_failed_claims`：账本 2 → 正确 **4**（attempt1 在 386 基线内 + attempt2 + 1301001 + 1301002）。
- `completed_steps=482000` 经 96 笔逐笔重建确认**正确**；交接聊天中的 "389000" 为摘要算术口误，机器报告与账本原文均为 482000。
- 性质：纯派生汇总缺陷；核心 claim/row/receipt 原件完整无损。

## 更正 3：ep0 身份口径澄清

- ROUND46 训练初始化 = **R45 `BC_POST_MODEL_s{seed}.pt` 四模块策略**（V5 训练清单："ROUND45 BC_POST_MODEL four modules, model-only, same seed"），四模块参数哈希 = 注册的 37303c49/e8531e0a/10bae22f = BC 控制行 `bc_parameter_sha256` → **BC 控制行回放的就是 ROUND46 的 ep0 策略**。
- R42 目录下的 `v2_train_s{seed}_ep0.pt`（四模块哈希 7eff0f40/b74c10c8/0ddebe7d）是 R43 时代注册的 ep0，**不是** ROUND46 的训练初值，本轮探针未将其用作 ep0。
