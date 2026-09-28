# ROUND25 §2 诊断：sampled/greedy 解码、观测、顺序 mask、概率与费用路径（0 新环境启动）

日期：2026-09-28。只读 ROUND24 落盘件（ep128 checkpoint × 3、eval tape × 96、transcript、
rows）与冻结代码身份 `9cfc0650…`。**未修改任何阈值、未训练、未启动环境、未改动 ROUND24
任何文件**。机读件 `DECODER_DIAGNOSTIC.json`，脚本 `decoder_diagnostic.py`。

## 结论：无身份/观测/解码/mask/计费错误（passed = true，problems = []）

判定错误的标准全部通过：概率重算、log_prob 重算、greedy argmax、free/masked 集合、
顺序 shadow 重放、masked 不泄漏似然、schema、逐条计费、账本闭合全部零违例。**可解释的
是策略行为，不是程序缺陷。**

## 1. 同一 checkpoint 与同一条动作路径（三 seed × 两模式）

| seed×模式 | 自由候选数 | p(SEND) 重算最大差 | log_prob 重算最大差 | greedy argmax 违例 | mask 集合违例 | shadow 重放违例 |
|---|---:|---:|---:|---:|---:|---:|
| 11 final_greedy | 178 | 0.0e+00 | 6.0e-08 | 0 | 0 | 0 |
| 11 final_sampled | 273 | 0.0e+00 | 1.8e-07 | 0 | 0 | 0 |
| 22 final_greedy | 6 416 | 0.0e+00 | 1.2e-07 | 0 | 0 | 0 |
| 22 final_sampled | 2 000 | 0.0e+00 | 3.0e-07 | 0 | 0 | 0 |
| 33 final_greedy | 2 085 | 0.0e+00 | 2.4e-07 | 0 | 0 | 0 |
| 33 final_sampled | 544 | 0.0e+00 | 2.7e-07 | 0 | 0 | 0 |

- 两模式用**同一** ep128 里程碑（文件 SHA256 与 `CHECKPOINT_INDEX.json` 一致），特征构造、
  顺序预算 mask、候选执行顺序完全同路径；greedy 行动作逐条等于重算概率的 argmax
  （旧冻结分支 `SEND iff p(SEND)>=p(HOLD)`，平局归 SEND，0 违例）；sampled 行动作按
  `torch.multinomial` 与登记动作 RNG 派生，其似然由 log_prob 重算确认（≤3.0e-07）。
- `free` = 顺序 shadow 认可的集合（`honours_this_candidate`），masked 只带生效理由；
  observation 侧初始标志单独登记。被 mask 候选不进入 likelihood/entropy 统计（0 泄漏）。
- 顺序 mask 重放：按 tape 目标顺序、从 `shadow_budget_before` 起只对 sampled 出的 SEND 扣费，
  复现每个候选的 `budget_after`，0 违例。capacity mask 全程 0 例（该配置容量为 None）。
- 计费路径：每条 send `cost_bytes == payload_bytes × subscribers`（逐条 0 违例），
  Σ == row `spent_bytes`，拒付计 0，shadow 预算不为负。
- masked 原因交叉表：seed 11 greedy/sampled 各有 6216/5413 例 budget 类 masked——全部是
  观测时执行器已判预算耗尽（例：余 1112 B < 单价 2898 B）的合法候选，非缺陷。

## 2. 同输入前缀对照（seed33，16 场景）

每场景第一条 tape（动作前状态确实相同：特征逐位相同、候选集相同、同一 forward logits）
**16/16 完全相同**；其中 **7/16 场景第一个决策的动作就不同**（greedy 选 HOLD、sampled
抽中 SEND，或反之）⇒ 之后闭环状态分叉，后续行不做逐行对照（按 spec 只对照真实同输入前缀）。
两模式的整集分布分开描述：greedy 每集 0–3 次发射、0–7770 B；sampled 每集约 10–11 次、
约 28 k B。

## 3. 自由候选的分层统计（解释分歧，不作为任何阈值/候选判断）

| seed×模式 | 自由候选 | 平均 p(SEND) | 平均二元熵(bit) | argmax=SEND 占比 | margin≤0.01/0.05/0.10 占比 | 首发 p | 刷新 p |
|---|---:|---:|---:|---:|---|---|---|
| 11 greedy | 178 | 0.639 | 0.937 | 0.994 | 0.6%/6.2%/21.9% | 0.571 | 0.664 |
| 11 sampled | 273 | 0.627 | 0.944 | 0.967 | 2.2%/12.5%/29.3% | 0.554 | 0.661 |
| 22 greedy | 6 416 | 0.181 | 0.624 | 0.000 | 0/0/0 | 0.181 | — |
| 22 sampled | 2 000 | 0.052 | 0.286 | 0.000 | 0/0/0 | 0.043 | 0.067 |
| 33 greedy | 2 085 | 0.407 | 0.965 | 0.007 | 3.1%/21.9%/66.0% | 0.424 | 0.301 |
| 33 sampled | 544 | 0.331 | 0.912 | 0.000 | 0/0.9%/5.5% | 0.374 | 0.312 |

- **seed33 的机制**：p(SEND) 中心在 0.5 以下 ⇒ argmax 几乎处处是 HOLD（0.7% 的自由候选
  p≥0.5），但 entropy 高（≈0.9–0.97 bit，策略远离确定性），sampling 按 p≈0.3–0.4 逐候选
  抽 ⇒ 每集成百次自由候选下必然产生 ~10 次 SEND。这是**结构性解码差异**（多数低概率
  Bernoulli 叠加），不是采样/掩码/计费错误，也不据此推断欠训练或改阈值。
- **seed11**：p(SEND)≈0.63、argmax 占比≈0.97 ⇒ greedy≈sampled≈高发送（~28 k B）。
- **seed22**：p(SEND)≈0.05–0.18、argmax 占比 0 ⇒ greedy 全 HOLD（0 B），但 sampling 仍按
  p 零星发送（边界附近概率 + 可付刷新是直接线索）；margin 带全 0（远离边界）。
- 首发与刷新的 p(SEND) 差异方向一致（刷新略低/相近）；retry 类 0 例（无预算拒付后重试，
  被 mask 的候选不产生 requested）。离线概率之和不是闭环真实发射数的估计，已按 spec 不作
  因果归因。

## 4. 回读 ROUND24 逐场景原始表（seed33 greedy，16 场景）

- 对 **immediate**：任务 16/16 平（dY 全 0，无退化场景），字节均差 −12 540.8 B/场景。
- 对 **late_discovery_gate**：任务 **16/16 平**（dY 全 0），字节均差 −2 439.8 B/场景。
- ROUND24 报告里“对 late 的两场景更差”（1301003、1301004）：**仅是字节指标更差**
  （+5 148 B、+7 770 B），**任务数完全相同**（各 6/6）⇒ 不是任务退化。这两个场景 greedy
  各发 3 次（7 644 B、7 770 B）而 late 只发 4 838 B 均值。
- 联合占优（任务不降且字节更省）：对 immediate 16/16，对 late 14/16、1 平、1 更差
  （前述 1301004）+1 更差（1301003）。

## 5. 处置

无正确性阻断，不触发“0 新环境启动下停止并报告”的正确性条款。但按用户补充指令四，
**评估启动还需原 spec 内容核验通过**——该项当前 MISMATCH（见
`SPEC_VERIFICATION.json`），故三臂 × 64 场景评估**未启动**；本诊断不改任何阈值、不挑
checkpoint、不否决候选，只作为 §3 冻结与后续评估的前置证据。
