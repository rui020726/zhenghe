# ROUND37/ROUND38 文案勘误登记

本文件保留并更正旧报告的两处表述；旧报告和运行产物不作改写。

## 1. seed11 与 seed33 的发射/刷新合计

更正读数：ROUND37 的 64 场景评估中，seed11 与 seed33 合计有 **1,406 次广播发射**，其中 **1,018 次是 refresh 发射**。依据为 ROUND38 的 `COMMUNICATION_USAGE.json`：seed11 为 705 次发射、515 次 refresh；seed33 为 701 次发射、503 次 refresh。这里的“广播发射”是一条 sender-side emission；它可能对应多个收件人传输尝试。丢包仍按实际尝试字节计费。

旧报告路径：

- `runs/comm_research_recovery_20261002/COMMUNICATION_USAGE.json`
- `runs/comm_research_recovery_20261002/RESEARCH_RESTART.md`

## 2. greedy 的精确定义

旧报告中“全部自由候选满足 `p[SEND] >= p[HOLD]`”的说法应读作解码器遵循 greedy argmax 规则，不能读成所有候选的 `p(SEND) >= 0.5`，也不能说所有合法候选都选择 SEND。代码规则是：对每个通过既有 mask 的候选，`SEND` 当且仅当 `p[SEND] >= p[HOLD]`；否则选择 `HOLD`，相等时选 `SEND`。因此只有实际选了 SEND 的候选才必然满足 `p[SEND] >= p[HOLD]`。

依据：

- `runs/comm_budget_action_mask_fix_20260922/budget_masked.py` 中冻结的 greedy 分支；
- `runs/comm_research_recovery_20261002/COMMUNICATION_USAGE.json` 的逐决策 argmax 一致性字段；
- `runs/comm_final_evaluation_20261002/ROUND37_EVALUATION_REPORT.md` 的 T4 行及交付 runbook 中的解码说明。

本勘误只修正文案含义，不改变 checkpoint、策略输出、评估行或 ROUND37/ROUND38 verdict。
