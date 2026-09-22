# 多步发送诊断第一阶段结果

日期：2026-09-22。基线规则 `event_retry_until_acked_5`（`max_attempts=5`）。
总仿真启动 **40 / 40**（含 4 次接口失败后隔离的 attempt1）。

## 规模

- validated 点：8；四分支齐全：7；缺额：1
- 缺额原因：启动上限（s700033 的 4 分支未跑），不补抽、不换 seed。
- 首轮 4 分支因视图锁定过早失败，产物保留于各 `H*/failed_attempt1/`，修复后重跑并通过接口门禁。

## 逐点成对（S − H）

| 点 | H | ΔonT | S total | H total | Δtotal | S 强发 | H 窗口拦 | 窗口结束 | 同视图补发 | 新 revision |
|---|---:|---:|---:|---:|---:|---|---:|---|---|---|
| s700010 t33 | 1 | 0 | 44529 | 44529 | 0 | refused_insufficient_credit | 1 | window_expired_then_revision | None | 49 |
| s700010 t33 | 20 | 0 | 44529 | 44529 | 0 | refused_insufficient_credit | 16 | window_truncated_by_revision | None | 49 |
| s700020 t620 | 1 | 0 | 44196 | 44196 | 0 | refused_insufficient_credit | 1 | window_expired_then_revision | None | 635 |
| s700020 t620 | 20 | 0 | 44196 | 44196 | 0 | refused_insufficient_credit | 15 | window_truncated_by_revision | None | 635 |
| s700022 t29 | 1 | 0 | 39620 | 39620 | 0 | refused_insufficient_credit | 1 | window_expired_then_revision | None | 30 |
| s700022 t29 | 20 | 0 | 39620 | 39620 | 0 | refused_insufficient_credit | 1 | window_truncated_by_revision | None | 30 |
| s700042 t46 | 1 | 0 | 40564 | 40564 | 0 | refused_insufficient_credit | 1 | window_expired_then_revision | None | 78 |
| s700042 t46 | 20 | 0 | 40564 | 40564 | 0 | refused_insufficient_credit | 20 | window_expired_then_revision | None | 78 |
| s700046 t423 | 1 | 0 | 41190 | 41190 | 0 | refused_insufficient_credit | 1 | window_expired_then_revision | None | 427 |
| s700046 t423 | 20 | 0 | 41190 | 41190 | 0 | refused_insufficient_credit | 4 | window_truncated_by_revision | None | 427 |
| s700018 t22 | 1 | 0 | 43744 | 43744 | 0 | refused_insufficient_credit | 1 | window_expired_then_revision | None | 23 |
| s700018 t22 | 20 | 0 | 43744 | 43744 | 0 | refused_insufficient_credit | 1 | window_truncated_by_revision | None | 23 |
| s700031 t19 | 1 | 0 | 41405 | 41405 | 0 | refused_insufficient_credit | 1 | window_expired_then_revision | None | 20 |
| s700031 t19 | 20 | 0 | 41405 | 41405 | 0 | refused_insufficient_credit | 1 | window_truncated_by_revision | None | 20 |
| s700033 t12 | 1 | — | — | — | — | — | — | incomplete | — | — |
| s700033 t12 | 20 | — | — | — | — | — | — | incomplete | — | — |

## 机制观察（逐点描述，不作方法主张）

1. **S 强发**：见上表 `S 强发` 列。`refused_insufficient_credit` 表示干预层请求了发送，但通信执行器因预算拒绝——按协议这**不是**成功的 S 分支，不能与 H 做 S−H 因果对照。
2. **H 窗口拦截**：多数点在 H=20 上有非零 `window_blocked_requests`；H=1 通常只拦 t* 一步。
3. **同视图补发**：`never`/空 表示窗口内及之后未再发射锁定视图；替代发射出现在新 revision 列。
4. **窗口结束**：本批多数点在窗口内或紧随其后出现新 revision（`window_truncated_by_revision` / `window_expired_then_revision`）——长窗口假设在这些点上检验时间很短，**不能**据此说发送时机无价值。
5. **任务结局**：在已跑点上 S/H/free 的 `on_time` 多数相同；在 S 未真正强发时，这只说明“未能强发 + 窗口拦截”相对 free 未改变完成数，**不**等于“发送时机无跨步影响”。

## 判读边界

- 不以 7 点 p 值或行级 bootstrap 作方法主张；H=1 与 H=20 是不同估计对象。
- 前向 RNG 位移、预算拒绝与窗口截断必须与任何任务差异并列。
- 本实验是 ACK 规则（`event_retry_until_acked_5`）下的局部机制诊断，与 no-ACK 学习曲线分开解释。
- **不支持**“PPO 信用分配是根因”“发送端观测可辨优劣”“学习可赢规则”。
- 结论允许不确定；是否扩样或改点由使用者决定。

