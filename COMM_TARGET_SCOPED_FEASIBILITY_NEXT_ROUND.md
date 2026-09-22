# 下一轮：target-scoped 合并发送的边界可行性门禁

> **状态：已执行完毕（历史指令）。** 分支 `analysis/comm-target-scoped-feasibility`，
> 产物 `runs/comm_target_scoped_feasibility_20260922/`。24/28 启动；T0/T1 通过。
> T2：H=1 2/4 未过，H=5 0/4 未过，**H=20 3/4 通过**（可解析 pair 多为 `terminal_before_boundary`）。
> 按门禁可建议以后对 **H=20** 跑完整 target-scoped episode（本轮不执行）。**请勿再次执行本矩阵。**

日期：2026-09-22。对应总纲 v2.6（执行时）。状态：历史指令，已执行完毕（见文首标记）。

## 1. 起点与研究问题

从 `analysis/comm-send-actionability-audit@c6698f9935eae731dcf3f85153a665d3232435cf`
创建 `analysis/comm-target-scoped-feasibility`。先核对远端 SHA、工作区及审计产物摘要；阅读 AGENTS、
ARCHITECTURE、PROJECT_STATUS、PROJECT_CHARTER v2.6、Stage1 verdict 与动作可行性审计的协议/结果/manifest。

动作审计已经回答：8 条开发流中存在 71 个 post-first 双动作可行点，但同一视图几乎每步被新 revision
取代，view-scoped H=20 在 0/8 流上有效。下一项新问题是：

> 在 target 级别合并窗口中，“现在额外通知一次，窗口末再同步最新状态”与“窗口内不通知，只在末尾同步
> 最新状态”能否在现有硬预算下真正执行？

这是**新的估计对象**，不是 Stage1 的补跑，也不是学习模型。它利用 relay 已有的语义：未发送的变化保留在
ledger 中，后续 emission 读取发送端当时可见的最新 snapshot，旧变化被合并而不是逐条补发。

本轮只运行到窗口边界并停止，不跑 episode 最终结局、不训练、不生成新流、不改预算/信道/deadline/
任务分布/生产默认值。目标是先排除第二个实验不可行性：起点可发，但窗口末的同步可能已无预算。

## 2. 新估计对象：EARLY_FLUSH 与 DEFER_FLUSH

对预登记的 `(seed, t*, target)` 和 H∈{1,5,20}，两分支从动作生效前的相同前缀开始：

### `EARLY_FLUSH(H)`

1. 在 `t*` 请求发送该 target 当前合法视图一次；必须走真实 `_emit`、真实广播计费和硬约束。
2. 从该次发送之后到 `t*+H` 之前，阻止该 target 的所有任务消息 emission，不论 revision 如何变化；
   其他 target、ACK、现场感知、gossip、分配与执行照常演化。
3. `t*+H` 边界对仍未终态的 target 请求发送**该分支发送端此刻可见的最新视图**一次，然后立即停止运行。

### `DEFER_FLUSH(H)`

1. 从 `t*` 到 `t*+H` 之前阻止该 target 的所有任务消息 emission，包括 `t*` 当前视图。
2. `t*+H` 使用与上面相同的合法路径请求发送该分支此刻可见的最新视图一次，然后立即停止。

两臂共同的边界：

- 这是“**多一次早期通知的边际价值/成本**”门禁；EARLY 通常比 DEFER 多一次 emission，成本不匹配，
  后续若跑结局必须同时报告任务差与上行+ACK 总成本，不能称为同成本比较。
- 不预付、不保留、不赠送窗口末字节。边界 flush 若付不起就真实拒绝并记账；本轮正是测这个问题。
- 最新视图只能通过 relay 原有 snapshot lookup 和 `_message` 生成，不能读取未来、接收端 belief 或评估器真值。
- 两分支闭环演化后，边界最新内容可能不同，这是早期消息的总效应，不强行改成相同 payload。
- 若 target 在边界前终态，沿用生产终态抑制，不发送虚假任务消息；记 `terminal_before_boundary`，
  不能把这种 post-treatment 状态作为剔除该点的理由。
- target 范围只拦任务上行消息；已产生 ACK 继续真实到达。禁止拦截别的 target 或把 ACK 当免费同步。

## 3. 固定数据与选点

只用审计的 8 条 free 流与 `COMM_SEND_ACTIONABILITY_ROWS/POINTS.json`，不新增 seed，不读取 outcome 字段。
按 seed 升序取前四条流：700010、700018、700020、700022。每流选一个点：

1. `both_actions_free`；
2. post-first-emission、非终态；
3. 优先 message_class=`update`；按 step、target、revision 升序取第一条；
4. 若该流没有 update，取同顺序第一条 `same_view_retry` 并显式标注；没有则缺额，不换第五星。

在任何新模拟启动前生成并提交 `TARGET_POINTS_INITIAL.json`，包含源行摘要、attempt charge、预算余额与选择理由，
不含 on_time/expired/旧干预优劣。审计记录只用于选点；每个分支仍须在 t* 重新验证实际候选、内容与预算。

## 4. 先做静态边界筛查，但不以它代替分支门禁

利用原 free 轨迹为每点/H 报告：边界时刻原轨迹的剩余预算、最新合法消息估算成本、target 是否终态、
其他 target 的预算消耗与新 revision 数。这只是规划信息，因为干预会改变闭环状态和预算轨迹；
不得据此宣布可行，也不得按“预计最容易成功”替换预登记点。

## 5. 实现与正确性要求

使用分析目录中的可卸载 wrapper/hook，不改变 `integrated_sar` 生产行为与默认配置。允许复用 Stage1 的前缀重放、
状态摘要和独立产物机制，但 target-scoped 逻辑单独实现，不能悄悄修改历史 Stage1 wrapper 后覆盖旧解释。

必须满足：

- 在 t* 前不拦任何消息；动作前摘要两分支逐字段相同，覆盖 ledger、队列、预算、ACK、belief、时钟和 RNG 状态；
- EARLY 在 t* 的发送实际成功且真实扣费，否则该点/H 为 `start_not_actionable`，停止该点其余 H；
- 窗口内该 target 所有 revision 的 emission 在 `_emit` 最终入口被拦截，策略请求计数与执行拦截计数分列；
- 被窗口拦截的请求不扣字节、不消耗真实 emission/重试次数，但 ledger 的 ready/latest/coalesced 状态照常更新；
- 边界 flush 从当时 observation 中取得 target item，经原 `_message` 与 `_emit` 发送，真实扣上行预算并触发真实 ACK；
- 若边界没有 observation item，区分 terminal、没有合法 sender-visible view、实现缺陷；不得构造消息补齐；
- 每个 seed/point/H/branch 使用唯一目录、签名与 final boundary record；resume 必须逐字段匹配；
- wrapper 在 finally 中卸载，合成测试证明无干预配置逐字不变。

前向信道继续使用 sequential RNG：分支发射序列改变后，后续随机抽样只保证同分布，不保证逐包相同；
反向 ACK 保留 counter-based 方案。记录 fork 前 RNG 摘要、分支后抽样数和 ACK unpaired keys，不改随机机制。

## 6. 先两点冒烟，再跑最多 24 个边界分支

运行顺序：

1. 先对前两个有效点运行 H=1 的四条分支；验证 t* 发送/阻止、最新视图合并、边界 flush/终态与账本闭合。
2. 冒烟通过后，按固定顺序运行 4 点 × 3H × 2 分支 = 最多 24 个前缀分支。
3. 技术失败可同一点重跑并保留失败产物；**总模拟启动上限 28**，包括冒烟失败和重跑。
   冒烟成功结果直接计入 24 条，不重复运行。
4. 任一代码修复须提交新身份，已有成功结果不得与新身份混合；预检发现混合即隔离，不删除。

每条只运行到边界动作完成，记录实际联合步数与耗时；不继续到 episode 终局。不得因观察到早期 terminal 或 belief
差异临时延长运行。若 28 次用尽，按实际缺额交付。

## 7. 预注册记录与门禁

逐点/H/分支记录：

- t* 当前 revision/fingerprint、起点 SEND 请求/成功/拒绝、真实字节；
- 窗口内 target 的策略请求数、最终入口拦截数、产生/合并的 revision 数、最新 sender-visible view；
- 其他 target emission 与预算消耗；
- 边界状态：`flushed_latest`、`terminal_before_boundary`、`refused_credit`、`refused_capacity`、
  `missing_legal_view` 或 `implementation_error`；
- 边界 message revision/fingerprint/payload/attempt bytes、flush 前后余额、ACK 计费；
- 首次接收 belief 分歧与边界时是否一致，仅作机制记录，不读作最终任务收益；
- RNG 位移、运行签名、final boundary record。

验收分三层：

### T0 实现可信

合成测试与两点冒烟证明起点、target 全 revision 拦截、最新视图 coalescing、终态抑制、边界真实计费、
预算恰好够/差一字节、唯一产物名、卸载恢复和账本恒等式。任何实现错误未闭合则停止。

### T1 起点可执行

四个预登记点的 EARLY t* 发送均实际成功；不足四个时如实报告，不补点。该门禁失败说明审计与实际执行仍不一致，
不得进入完整结局实验。

### T2 边界可解析

对每个 H 分开判定。一个 pair 可解析，当两分支在边界分别是：

- `flushed_latest`，或
- target 已在该分支 `terminal_before_boundary`（吸收状态，不能因可能受处理影响而剔除）。

非终态分支若因预算/容量拒绝或缺少合法 view，则该 pair 对“立即 vs 延迟后同步”不可执行。
某个 H 至少 3/4 pair 可解析且无实现错误，才可建议下一轮对该 H 跑完整 episode；3 是资源门槛，不是理论阈值。

判读：

- H=20 过 T2：可设计小型完整 target-scoped 结局实验；本轮不执行。
- H=20 不过、H=5 或 H=1 通过：只说明较短延迟可执行。是否仍回答研究问题交用户决定，不自动缩短窗口。
- 所有 H 都不过：当前全局硬预算不支持“窗口末同步最新状态”。下一步若继续，必须把 per-target credit reservation
  作为新的协议设计单独讨论；不能赠送 flush、偷偷提高预算或改成“最终不发”。

即使 T2 通过，也不证明多步发送影响任务结果、合法观测可预测优劣或学习模型能够超过规则。

## 8. 交付与停止

目录 `runs/comm_target_scoped_feasibility_20260922/`：

- `COMM_TARGET_SCOPED_PROTOCOL.md`、`TARGET_POINTS_INITIAL.json`、静态筛查；
- `COMM_TARGET_SCOPED_INTERFACE_CHECK.json` 与合成测试；
- `COMM_TARGET_SCOPED_BOUNDARY_RESULTS.md/.json`，逐点/H 的 T0/T1/T2；
- `COMM_TARGET_SCOPED_MANIFEST.json` 与启动账本、源码/config/stream 摘要、生产 diff、真实测试结果；
- 可复跑脚本和必要的分析代码。

运行相关测试以及 AGENTS 规定的集成测试/冒烟；不沿用上一轮测试数充当本轮结果。
不提交原始流、逐步 transcript、checkpoint；显式暂存文件，不触碰用户未跟踪文件。
更新总纲、状态与 README，并把本指令标为历史已执行；推送并核对远端 SHA。

交付 T0/T1/T2 后停止。**不自动运行完整 episode、不训练、不改预算、不实现信用预留。**
