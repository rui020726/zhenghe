# target-scoped 合并发送边界可行性协议（启动前冻结）

日期：2026-09-22。分支 `analysis/comm-target-scoped-feasibility`（`dfe120a` 起点）。
指令 `COMM_TARGET_SCOPED_FEASIBILITY_NEXT_ROUND.md`。对应总纲 v2.6。

## 0. 边界

- 固定四流：**700010, 700018, 700020, 700022**；每流 1 点（最早 post-first `both_actions_free` 的 `update`）。
- H ∈ **{1, 5, 20}**；分支 **EARLY_FLUSH / DEFER_FLUSH**。
- 最多 **4×3×2 = 24** 条边界分支；总启动（含失败/重跑）**≤ 28**。
- 每条只跑到 **窗口边界**，不继续到 episode 终局。
- 边界最新消息走真实 `_message`/`_emit`、真实预算与计费；**不预留、不赠送字节**。
- target 提前终态是吸收状态（`terminal_before_boundary`），不作处理后剔除。
- 不训练、不生成新流、不改预算/信道/deadline/协议/默认值；不跑完整任务效果对比。

## 1. 估计对象

| 分支 | t* | (t*, t*+H) | t*+H |
|---|---|---|---|
| `EARLY_FLUSH(H)` | 请求发送当前合法视图一次（真实 `_emit`） | 阻止该 **target** 全部任务上行 emission（不论 revision） | 请求发送**当时**最新 sender-visible 视图一次后停止 |
| `DEFER_FLUSH(H)` | 同样阻止（含 t* 当前视图） | 同上 | 同上 |

共同：其他 target / ACK / 感知 / gossip / 分配照常；ACK 不拦；边界 flush 付不起则真实拒绝并记账。

## 2. 选点（先于任何启动提交）

从 `runs/comm_send_actionability_20260922/COMM_SEND_ACTIONABILITY_ROWS.json` 按 seed 升序前四流，
每流取最早 `(step, target, revision)` 的 post-first、非终态、`both_actions_free` 且 `message_class=update`。

| seed | t* | target | rev | class | attempt_charge | budget_remaining（审计时刻） |
|---:|---:|---|---:|---|---:|---:|
| 700010 | 3 | piloc-e1-target-0 | 3 | update | 2616 | 18932 |
| 700018 | 3 | piloc-e1-target-2 | 3 | update | 2400 | 5048 |
| 700020 | 3 | piloc-e1-target-0 | 3 | update | 2820 | 12266 |
| 700022 | 3 | piloc-e1-target-2 | 3 | update | 2610 | 12170 |

无 outcome 字段。静态筛查只作规划信息，不替换预登记点。

## 3. 实现要求

- 可卸载 wrapper，不改 `integrated_sar` 生产行为。
- t* 前不拦消息；动作前摘要两分支逐字段相同。
- EARLY 在 t* 必须实际发射并扣费，否则 `start_not_actionable` 并停该点其余 H。
- 窗口内 target 的 emission 在 `_emit` 最终入口拦截；策略请求计数与执行拦截计数分列。
- 被拦请求不扣字节、不消耗 emission 次数；ledger ready/latest/coalesced 照常。
- 边界 flush 走 observation item → `_message` → `_emit`；无 item 时区分 terminal / missing_legal_view / 实现缺陷。
- 逐 seed/point/H/branch 唯一产物；`finally` 卸载 wrapper。
- 前向 sequential RNG 仅同分布；反向 ACK 保持 counter-based；记录 fork RNG 摘要与 unpaired keys。

## 4. 运行顺序

1. **冒烟**：前两点 × H=1 × 两分支（4 启动），验 t* 发/拦、合并、边界 flush/终态、账本。
2. 通过后跑满 4×3×2=24（冒烟计入，不重复）。
3. 失败可同点重跑并保留产物；总启动 ≤28。
4. 代码修复须新身份；成功结果不与新身份混合。

## 5. 门禁

- **T0 实现可信**：合成测试 + 冒烟（起点、全 revision 拦截、coalescing、终态、真实计费、恰够/差一字节、唯一产物、卸载、账本恒等式）。
- **T1 起点可执行**：四点 EARLY t* 发送均成功（不足则如实，不补点）。
- **T2 边界可解析**（**按 H 分别判**）：pair 可解析 = 两分支均为 `flushed_latest` 或 `terminal_before_boundary`。  
  某 H **≥3/4** pair 可解析且无实现错误 ⇒ 才可建议该 H 以后跑完整 episode（3=资源门槛）。  
  H=20 不过而短 H 过 ⇒ 只报较短延迟可执行。全不过 ⇒ 只能讨论 credit reservation 新协议。  
  **不得**免费 flush / 提高预算 / 改成窗口末不发送。

T2 通过也不证明任务收益、观测可预测优劣或学习优于规则。

## 6. 交付

`runs/comm_target_scoped_feasibility_20260922/`：本协议、`TARGET_POINTS_INITIAL.json`、静态筛查、
接口检查与合成测试、`COMM_TARGET_SCOPED_BOUNDARY_RESULTS.md/.json`、manifest、启动账本、脚本。
更新总纲/状态/README，本指令标历史。推送核对远端 SHA 后**停止**。
