# ROUND45-R2 预启动账本勘误

旧报告 `ROUND45_R2_REPORT_R3.md` 保留，SHA256 `e849f537c7c1a1f69864f8d609420a9e0a05710771b797443e34a00686eea89b`；其中环境累计数写为 0。
本版只读强制入口的 snapshot/write 读回已冻结到独立版本化 ledger 快照，确认该累计数应为 1。

## 更正后的累计计数

- normal/reserve/total：1/0/1。
- 环境构造 / run / 返回 / 完成步：1 / 1 / 1 / 1000。
- claim 原始完成数：0；adoption：1；有效唯一任务：1。
- PPO / optimizer / checkpoint：0 / 0 / 0。
- 原 claim 仍为 `failed_technical`；原 claim 与 row 哈希不变；三次 snapshot、两次 write 均无计数增量。
- 本次勘误没有启动新环境或创建 claim。

读回收据：`ROUND45_R2_ADOPTED_LEDGER_READBACK_R3_FINAL3.json`（`d393a34ee01d3d52f1578d096bb219a6fee09a1346cc7a514ef8b2ade8d4b588`）。
