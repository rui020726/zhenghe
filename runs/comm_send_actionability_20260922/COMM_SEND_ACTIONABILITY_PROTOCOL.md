# 通信 SEND/HOLD 动作可行性审计协议（启动前冻结）

日期：2026-09-22。分支 `analysis/comm-send-actionability-audit`。
前置：`analysis/comm-multistep-stage1` 远端 SHA = `5b1062620a348d2425201b2c9594df674b411e78`（已核对）。
对应指令 `COMM_SEND_ACTIONABILITY_AUDIT_NEXT_ROUND.md`。

## 0. 边界（强制）

- 不训练、不生成新流、不运行新的完整 S/H 结局分支。
- 不改预算、成功率、deadline、消息协议或生产默认值。
- 不触碰 `800000..800399`、`900000..900199`。
- 不提交 `architecture/`、`dev_summary.json`、`_DOC_PATCH_NOTES.md`、`_fix_diag_json.py`。
- 优先解析 Stage1 的 8 条 free；字段不足时最多重放同一 8 条流；**补跑 free + 短探针合计 ≤ 8 次启动**。
- 不换 seed、不补抽点。完成 A/B/C 门禁后停止，不自动开下一轮 S/H。

## 1. 为何需要重放（解析不足的证据）

Stage1 `free.json` 仅在**单个 t\*** 记录 observation 摘要；`ct_*.jsonl` 只含 `send`/`refused`/`event`/`deliver`/`ack`，**不含规则选择 HOLD 时的全部候选**。  
审计要求“不得只记录规则实际请求 SEND 的点”⇒ 必须在 free 轨迹上逐决策记录**全部**通信候选。  
因此对 8 条已选流各重放 1 次 free（**8/8 启动上限**），短前缀探针本轮不使用（除非 §5 触发）。

## 2. 审计对象与去重键

每条 free 流、每个通信子步、每个候选一条记录。去重键：

`(seed, step, communication_substep, target_id, semantic_revision, content_fingerprint, subscriber_set)`

同一步不同 target 仍是不同决策。`semantic_revision = SendItem.revision`；`content_fingerprint` 取自将要发射的 `DetectionMessage`（与 `comm_ack.content_fingerprint` 同源）；`subscriber_set` 为拓扑订阅者排序元组。

## 3. 字段（动作前）

合法性/计费：legal、terminal/expired、subscriber_count、payload_bytes、attempt_charge_bytes（= payload × |subscribers|，与 `_emit` 相同）。  
预算/容量：budget_remaining、budget_spent、sends_this_step、capacity_remaining、attempt_charge / budget_remaining。  
历史：sent_count、attempts_remaining、message_class ∈ {first, update, same_view_retry}。  
内容变化：changed_view、stage/requirements/members/progress 变化、age_steps、remaining_time、urgency。  
ACK：confirmed_fraction、ack_age、view_gap、unacked_broadcasts；无 ACK 为显式 -1。  
动作：rule_requested（策略是否请求 SEND）、send_executable、hold_executable、send_block_reason、forced、in_policy_interface。

## 4. SEND 可行谓词（与执行器共享）

**不得复制近似公式。** 谓词调用与 `CommRelay._emit` 相同的部件：

1. `message = relay._message(target, ledger, snapshot)`  
2. `attempt_charge = estimate_message_bytes(message) * len(relay._subscribers)`  
3. `budget.can_send(attempt_charge)` / `why_blocked(attempt_charge)` → `capacity_step` | `budget_exhausted` | `allowed`

终态/内容合法性：若 snapshot 终态，候选本不应进入 `_observation`（传输层抑制）；若仍出现，记 `content_or_terminal`。  
`CommBudget.can_send`/`why_blocked`/`estimate_message_bytes` 为生产代码本体，审计只调用不改写。  
合成回归：对同一 (cost, spent, sends_this_step) 断言谓词与 `can_send` 逐例一致；对固定 message 断言 charge 与 `_emit` 扣费公式一致。

## 5. 互斥主类（优先级 1→8，只入一类）

| 序 | 主类 | 判定 |
|---:|---|---|
| 1 | `both_actions_free` | HOLD 可执行 且 SEND 可执行 且无协议强制 |
| 2 | `send_blocked_credit` | 仅/首要阻塞 = `budget_exhausted` |
| 3 | `send_blocked_capacity_or_queue` | 首要阻塞 = `capacity_step` |
| 4 | `send_blocked_content_or_terminal` | 终态/非法内容 |
| 5 | `send_blocked_retry_exhausted` | 同视图重试且 attempts_remaining=0（执行器仍可能付费，但协议/规则层已无重试额度） |
| 6 | `hold_not_free_protocol_forced` | 协议/安全层强制发送或禁止 HOLD |
| 7 | `not_policy_decision` | 未进入策略自由选择接口 |
| 8 | `other` | 必须逐项解释 |

恒等式：`候选总数 = Σ 主类`；分组表同样闭合。

## 6. H=20 前瞻标签（非因果结果）

对每个 `both_actions_free` 且 post-first-emission（`message_class ∈ {update, same_view_retry}`）且非终态的候选，用**同一 free 轨迹**标注：

- 同视图有效步数、首次新 revision/终态步偏移；
- 固定规则（`event_retry_until_acked_5`）在 1 步 / 20 步内是否再次请求同视图 SEND；
- 20 步内是否存在可支付更新；view-scoped 窗口预计 `revision_truncated` 还是 `can_block_retry`。

禁止读取按时完成/超时/分支收益/旧反事实优劣。候选列表固定排序 `seed, step, target, revision`。

## 7. 短前缀探针（默认 0 次）

仅当纯校验与 Stage1 实际 `refused`/`send` 记录不能对齐时启用，≤4 个；点按固定排序取前四；请求一次 SEND 后立即终止。  
`both_actions_free` 若在探针中仍被拒 ⇒ **审计失败并停止**。探针不计入 S−H 性能。

## 8. 门禁

- **A 审计可信**：恒等式闭合；谓词共享/回归通过；无 outcome 字段；8 流完整（或缺额如实）。  
- **B 双动作可行**：≥4 条不同流各有 ≥1 个 post-first-emission、非终态、`both_actions_free` 且 SEND 可真正发射的点。  
- **C view-scoped 干预有效**：在通过 B 的点中，≥4 条不同流存在 20 步内会请求同视图重试、且窗口不会在请求前被新 revision/终态截断的点。

B、C 都过 ⇒ 下轮可预登记少量 S/H（本轮不执行）。B 过 C 不过 ⇒ 考虑 target-scoped 新估计对象。B 不过 ⇒ 不进入多步干预设计。

## 9. 交付与资源

目录 `runs/comm_send_actionability_20260922/`：本协议、`ROWS.json`、`RESULTS.md/.json`、`POINTS.json`、`MANIFEST.json`、合成测试。  
**本协议先于任何新增重放提交。** 新启动数记入 manifest，合计 ≤ 8。
