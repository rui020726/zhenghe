# COMM_H20_TERMINAL_ATTRIBUTION

日期：2026-09-22  
分支：`analysis/comm-h20-terminal-attribution`  
基线：`analysis/comm-target-scoped-feasibility@9001a85cc643f6eccf6da5dc98078e3b95a6b49a`

本轮 **0 新 episode / 0 训练 / 0 新流 / 0 参数修改**。只解析既有 target-scoped 24 条边界分支、
actionability free 基线与已有训练记录。机器可读：`COMM_H20_TERMINAL_ATTRIBUTION.json`。
解析脚本：`analyze_h20_terminal.py`（只读）。

## 一、分类图例（互斥）

| 类 | 含义 |
|---|---|
| `both_terminal_identical` | 双方均在边界前终态且终态快照完全一致 |
| `both_terminal_same_time_outcome_diff` | 双方同刻终态，但 belief / assignment / work / progress 有分歧 |
| `terminal_time_diff` | 双方均终态但终态时间不同 |
| `one_terminal_only` | 仅一侧终态 |
| `boundary_refused` | 边界 flush 双方 `refused_credit`（非有效 flush，也非终态吸收） |

**不把 `terminal_before_boundary` 自动解释成有效 flush。** 成功 EARLY 投递须 `start_send_ok=true`
且 `start_revision ∈ revisions_seen`（应用可核）。

## 二、逐 seed / branch

四对均为 t*=3、H=20、target-scoped。EARLY 在 t* 强发；DEFER 窗口末 flush。

| seed | target | 分类 | EARLY 终态时刻 | DEFER 终态时刻 | EARLY 投递/应用 | 边界状态 E/D |
|---:|---|---|---:|---:|---|---|
| 700010 | piloc-e1-target-0 | `both_terminal_same_time_outcome_diff` | 23 | 23 | ok / rev3 已见 | terminal_before_boundary |
| 700018 | piloc-e1-target-2 | `both_terminal_same_time_outcome_diff` | 23 | 23 | ok / rev3 已见 | terminal_before_boundary |
| 700020 | piloc-e1-target-0 | `boundary_refused` | — | — | ok / rev3 已见 | refused_credit |
| 700022 | piloc-e1-target-2 | `both_terminal_same_time_outcome_diff` | 23 | 23 | ok / rev3 已见 | terminal_before_boundary |

计数：`both_terminal_identical` **0**；`both_terminal_same_time_outcome_diff` **3**；
`terminal_time_diff` **0**；`one_terminal_only` **0**；`boundary_refused` **1**。

### 2.1 s700010（t3, H20）— 同刻终态，结局分歧

- **终态类型/时间**：双方 `terminal_at=23`（同刻）；EARLY 侧 target-0 `completed@19`。
- **EARLY 投递/应用**：`start_send_ok=true`，`start_send_bytes=2616`，`start_revision=3`，
  `revisions_seen=[1,2,3]`（rev3 应用可核）。DEFER 侧 `revisions_seen=[1,2]`（无 rev3）。
- **belief / assignment / work / progress 分歧**（EARLY vs DEFER）：
  - `piloc-e1-target-0`：EARLY `completed` 干净收尾、`assigned=[]`；
    DEFER `completed` 但 `failure_reason=stale_task_revision`、残留 `assigned=["5"]`。
  - `piloc-e1-target-2`：EARLY `in_progress` progress **0.201848** rev5 assigned `["5"]`；
    DEFER `detected` progress **0.0** rev2 assigned `[]`。
  - `piloc-e1-target-5`：EARLY `detected` progress 0.0 rev2；
    DEFER `in_progress` progress **0.088468** rev4 assigned `["5"]`。
- **free 对照**：双方相对 actionability free 全集均有差异（窗口截断 + 干预改变轨迹）；
  free 侧多目标已 `completed`。此处 free 只作对照，不作因果归因。

### 2.2 s700018（t3, H20）— 同刻终态，结局分歧

- **终态类型/时间**：双方 `terminal_at=23`。
- **EARLY 投递/应用**：`start_send_ok=true`，`start_send_bytes=2400`，rev3 已见。DEFER 无 rev3。
- **分歧**：
  - `piloc-e1-target-0`：assigned EARLY `["1","3"]` vs DEFER `["3"]`。
  - `piloc-e1-target-1`：EARLY `detected` rev2；DEFER `assigned` rev3 `["1"]`，剩余需求不同。
  - `piloc-e1-target-5`：DEFER 残留 `assigned=["3"]` 且 `failure_reason=stale_task_revision`；EARLY 干净。
- **free 对照**：双方相对 free 均有 progress/status/assignment 差异（截断 + 干预）。

### 2.3 s700020（t3, H20）— `boundary_refused`

- **终态类型/时间**：双方均未在窗口前终态；边界 flush 双方 `refused_credit`。
- **EARLY 投递/应用**：t* 发送成功（`start_send_ok=true`，rev3 已见），但**边界 flush 付不起**。
- **分歧（边界时刻，非终态）**：
  - `piloc-e1-target-0`：assigned EARLY `["4"]` vs DEFER `["4","2"]`。
  - `piloc-e1-target-2`：rev EARLY 4 vs DEFER 3；assigned `["5","2"]` vs `["5"]`。
- **free 对照**：有差异；free 侧含 `deadline_expired` / Timeout 终态，H20 窗口分支被边界截断。
- **判读**：本对不计入“有效 EARLY flush → 终态分歧”信号；它是预算拒付，不是终态吸收。

### 2.4 s700022（t3, H20）— 同刻终态，结局分歧

- **终态类型/时间**：双方 `terminal_at=23`。
- **EARLY 投递/应用**：`start_send_ok=true`，`start_send_bytes=2610`，rev3 已见。
- **分歧**：
  - `piloc-e1-target-4`：EARLY `in_progress` progress **0.847985** rev13 assigned `["0","5"]`；
    DEFER `assigned` progress **0.0** rev3 assigned `["0"]`。
- **free 对照**：双方相对 free 有差异；free 侧多目标 `completed`。

## 三、成功 EARLY 投递伴随的分歧汇总

| 条件 | 结果 |
|---|---|
| 成功 EARLY 投递且 rev 应用可核 | **4/4**（700010/700018/700020/700022） |
| 其中出现结局/工作进度/assignment 分歧 | **3** 个不同 seed（700010、700018、700022） |
| 终态时间分歧 | **0**（有终态的三对均为 t=23） |
| 三对完全相同提前终态且无分歧 | **否**（`both_terminal_identical=0`） |

分歧形态以 **assignment 归属、task_revision、progress / remaining_requirements** 为主，
不是单纯终态时刻漂移。DEFER 侧多次出现 `stale_task_revision` 残留指派，EARLY 侧更常干净收尾
或推进到更高 revision 的工作状态。

**限制**：这些是 H=20 **边界窗口分支**（在边界/终态停），不是完整 episode；不能外推任务收益、
学习优于规则或完整 spillover 效应。`terminal_before_boundary` 只说明资源门在该 H 吸收了窗口，
不证明“延迟 flush 本身可支付”。

## 四、判定（按预注册条件）

1. **至少两个不同 seed 出现由成功 EARLY 投递伴随的终态时间、工作进度或结局分歧**  
   → **成立**（700010、700018、700022）。  
   ⇒ **可再讨论**小型完整 H=20 spillover 实验（本轮**不**运行、不设计细节）。
2. 三对均为相同提前终态且无有效分歧 → **不成立**（不触发“禁止完整 H=20”支）。
3. 训练正确性 / 动作 mask 缺陷 → 见 `COMM_POLICY_ACTION_MASK_AUDIT.md`（独立诊断）。
4. 若最终既无训练缺陷也无 H=20 信号才停 target-scoped → 本条不触发。

综合：H=20 侧**有**可讨论信号；是否立刻开 spillover，取决于动作路径诊断是否被定为须先修的缺陷。
