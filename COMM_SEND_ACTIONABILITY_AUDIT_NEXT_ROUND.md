# 下一轮：通信 SEND/HOLD 动作可行性审计

> 历史指令：已执行，交付为 `analysis/comm-send-actionability-audit@c6698f9`。
> 当前入口为 `COMM_TARGET_SCOPED_FEASIBILITY_NEXT_ROUND.md`；不要再次执行本审计。

> **状态：已执行完毕（历史指令）。** 分支 `analysis/comm-send-actionability-audit`，
> 产物 `runs/comm_send_actionability_20260922/`。8/8 free 重放、0 短探针；A/B 过、C 不过。
> 引用口径见 `COMM_SEND_ACTIONABILITY_RESULTS.md`。**请勿再次执行本审计矩阵。**

日期：2026-09-22。状态：待执行计划。本文只定义审计，不报告新结果。

## 1. 前置条件与定位

执行前必须先由持有本地提交的环境把 `analysis/comm-multistep-stage1` 推送到远端，并确认：

```text
remote HEAD = 5b1062620a348d2425201b2c9594df674b411e78
```

若远端仍不存在或 SHA 不一致，停止，不在旧文档分支上运行审计。网络不可达不是扩大本地未推送提交链的理由。

远端核对后，从该 SHA 创建 `analysis/comm-send-actionability-audit`。先读 AGENTS、ARCHITECTURE、
PROJECT_STATUS、PROJECT_CHARTER v2.4、Stage1 历史执行指令及
`runs/comm_multistep_stage1_20260922/` 的协议、POINTS、接口检查、结果、verdict 和 manifest。
保留未跟踪的 `architecture/`、`dev_summary.json`、`_DOC_PATCH_NOTES.md`、`_fix_diag_json.py`，
不修改、不删除、不提交。若后两项属于执行 agent 自己且确需入库，先说明用途并另作决定，不能混进审计提交。

Stage1 已证明的是：所选 8 个点上，强制 SEND 全部因 `insufficient_credit` 被拒，因而没有有效 S−H 对照；
它没有证明发送时机无效。下一步先测量**策略实际拥有多少双向选择**，不继续搜点跑结果。

本轮不训练、不生成新流、不运行新的完整 S/H 结局分支、不改预算/成功率/deadline/消息协议/
生产默认值。优先只解析 Stage1 的 8 条 free 基线；如果现有记录缺少审计字段，最多重放同一 8 条流、
每条 1 次 free，新增仿真启动上限 8。不得换 seed、补抽点或打开封存块。

## 2. 审计对象：动作前的通信决策状态

对每条 free 流的每个通信候选，在动作执行前记录一次规范化状态。不得只记录规则实际请求 SEND 的点，
否则会漏掉策略本可发送却选择 HOLD 的状态。按 `(seed, step, communication_substep, target_id,
semantic_revision, content_fingerprint, subscriber_set)` 去重；同一步不同候选仍是不同决策。

每条记录至少包含：

- 当前消息是否合法、是否终态/过期、订阅者数、payload 与按真实广播规则计算的本次尝试字节；
- 上行预算余额、当步容量/队列约束、已发次数、剩余重试次数、首发/更新/同视图重试类别；
- changed_view 与 stage/requirements/members/progress 变化，消息 age、belief 中的剩余时间和紧急度；
- ACK confirmed_fraction、ack age、view gap、未确认广播数；无 ACK 时保留显式 unknown；
- SEND、HOLD 各自是否可执行；不可执行的具体原因；规则请求动作、执行动作与拒绝原因分列；
- 是否由协议/安全层强制，是否真正进入了模型可自由选择的决策接口。

“SEND 可行”必须调用与实际通信执行器相同的纯检查或共享校验逻辑，覆盖内容合法性、终态抑制、
订阅者计费、剩余字节、当步容量与队列状态。不能复制一套近似公式后称为真值；不能用任务分配 mask。
若现有执行器没有无副作用的校验入口，先提取最小共享 predicate 并用回归测试证明实际执行与审计一致；
不得改变 SEND 的生产语义或默认配置。

## 3. 互斥分类与必须闭合的账

按优先级把每个候选分到一个且仅一个主类，次要原因另列：

1. `both_actions_free`：SEND 与 HOLD 都合法、可支付，且没有协议强制；
2. `send_blocked_credit`：唯一首要阻塞是剩余预算不足；
3. `send_blocked_capacity_or_queue`；
4. `send_blocked_content_or_terminal`；
5. `send_blocked_retry_exhausted`；
6. `hold_not_free_protocol_forced`；
7. `not_policy_decision`：候选未进入策略自由选择接口；
8. `other`：必须逐项解释，不能成为兜底大类。

验证恒等式：候选总数 = 八个主类之和；每个表格分组也要闭合。另分开统计：

- 首次发送、内容更新、同视图重试；
- 每条流与合计；时间 0–25%、25–50%、50–75%、75–100%；
- 剩余预算分箱，以及 `attempt_charge / budget_remaining`；
- 真正进入 actor 的自由决策数、硬掩码后仅剩单动作数、执行器拒绝数；
- `both_actions_free` 中 post-first-emission 的有意义更新数和同视图重试数。

必须同时报告 decision-weighted 与 stream-weighted 结果，防止一条候选特别多的流主导百分比。
本轮没有统计检验或方法性能主张，不能用 8 条开发流外推总体。

## 4. H=20 是否有机会形成有效干预

这是诊断，不用于按结果挑“好点”。对每个 `both_actions_free` 的 post-first-emission 候选，
利用同一 free 轨迹只标注后续机制：

- 20 步内同一视图是否仍有效；首次新 revision/终态发生在几步后；
- 固定规则是否在 1/20 步内再次请求发送同一视图；若从不请求，HOLD 窗口不会产生实际差异；
- 20 步内是否存在可支付的更新消息，以及 free 轨迹中的预算轨迹；
- view-scoped 窗口预计会被 revision 截断，还是有机会实际阻止重试。

这些是 prospective-mechanism 标签，不是因果结果。不得读取按时完成/超时、分支收益或旧反事实优劣来筛选。
如需列候选点，固定排序为 seed、step、target、revision，不按“持续时间最长”重新排名。

## 5. 最多四个短前缀一致性探针

只有当纯校验与 Stage1 实际拒绝记录不能证明审计 predicate 正确时，才允许最多 4 个短前缀探针：

- 从同一 free 配置确定性重放至预登记的 `both_actions_free` 点；
- 请求一次 SEND，记录是否真正发射、扣费与拒绝原因后立即终止，不跑任务结局；
- 点按固定排序取前四个，不能按最终任务结果挑选；
- 每个启动计入本轮 8 次总上限，因此补跑 free 与短探针合计不得超过 8；
- 探针只验证可行性分类，不作为 S−H 性能数据。

如果声称 `both_actions_free` 的点在短探针中仍被拒绝，审计失败并停止；先修共享校验，不能继续生成候选清单。

## 6. 决策门禁

交付后按以下顺序判读，不由 agent 自行改预算制造信号：

### A. 审计可信门禁

- 分类与分组恒等式全部闭合；
- 审计 predicate 与实际执行器共享或已由回归/短探针验证；
- 无结果字段参与记录、分类和候选排序；
- 8 条流记录完整，技术失败如实计入上限。

任一不满足：只交阻塞/修复报告，不设计下一轮 S−H。

### B. 双动作可行门禁

至少 4 条不同流各有 ≥1 个 post-first-emission、非终态、`both_actions_free` 的更新或重试点，
并且其 SEND 预计/实测能真正发射。这里的 4 是下一轮资源门槛，不是学习可行性的理论阈值。

若未过：结论限定为当前预算与协议下动作自由度不足。报告是预算耗尽、消息收费、容量还是协议强制导致；
停止并交用户决定是否研究预算/协议，不自动增大预算，也不退回只测首次发现上报来制造差异。

### C. view-scoped 干预有效性门禁

在通过 B 的点中，至少 4 条不同流存在 20 步内实际会请求同视图重试、且窗口不会在请求前被新 revision/
终态截断的点。该阈值同样只控制资源投入。

- B、C 都过：下一轮可预登记少量 view-scoped S/H 分支；本轮不执行。
- B 过、C 不过：说明 view-scoped 干预不适合该消息流。下一轮应讨论 target-scoped “延迟所有更新并在窗口末发送最新视图”这一**新估计对象**，不能暗改 Stage1 协议后声称延续同一实验。
- B 不过：不进入任何多步干预设计。

不管哪条分支，都不能由动作自由度直接推断 PPO 信用分配、观测充分性或学习优于规则。

## 7. 交付与停止

建议目录 `runs/comm_send_actionability_20260922/`，至少交付：

- `COMM_SEND_ACTIONABILITY_PROTOCOL.md`：代码身份、字段、互斥分类、资源上限，先于新增重放提交；
- `COMM_SEND_ACTIONABILITY_ROWS.json`：逐候选规范化记录；
- `COMM_SEND_ACTIONABILITY_RESULTS.md/.json`：闭合表、时间/预算/消息类型分布、A/B/C 门禁；
- `COMM_SEND_ACTIONABILITY_POINTS.json`：固定顺序的 prospective 列表，不含结局字段；
- `COMM_SEND_ACTIONABILITY_MANIFEST.json`：引用的 Stage1 artefact 摘要、新启动数、生产 diff、测试；
- 合成测试：八类互斥、广播扣费边界、恰好付得起、容量拒绝、重试耗尽、forced HOLD 非自由动作、
  审计字段不读取 outcome、同一执行器 predicate 与实际 SEND 一致。

修改分析/审计接缝时跑相关测试及 AGENTS 要求的集成测试和冒烟，并报告本轮实际结果。
不提交原始流、逐步日志、checkpoint；显式暂存文件。更新总纲、状态和 README，但不重写 Stage1 原始 verdict。

完成审计、给出 A/B/C 判定与一个建议后停止。**不自动启动下一轮 S/H、不训练、不改预算或协议。**
