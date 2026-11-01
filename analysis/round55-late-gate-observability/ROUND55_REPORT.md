# ROUND55 报告 — late 门控可观测性与关键统计表述收口

分支 `analysis/round55-late-gate-observability`（自 R54 交付 d0b91b94… 建立）。
零操作：环境构造 0｜仿真 0｜新流 0｜训练 0｜optimizer 更新 0｜checkpoint 写入 0｜新 episode 行 0｜CPU forward 0。
测试：`r55_zero_tests.py` 7/7 通过（exit 0）；R54 的 27 项套件与 528 启动账本按指示未重复。

## 一、窄范围统计表述核对（STATEMENT_RECONCILIATION.json）

审计对象："s11 greedy 相对 late 显著更贵（+157.1\*）"。

- 冻结机器格（ROUND53_STATISTICS.json）：point +157.125，CI95 = **[0.0, 471.375]**，无 significant 字段。
- 用冻结索引（PCG64(20261008)，10000 次）独立重算：**逐位一致**；实际场景差向量 = 15 个 0 + 一个 +2514（scene 1301014）。
- 解析核对：P(16 次重抽全部错过该格) = (15/16)^16 = 0.35607 → 2.5 分位 = 0；P(k≤3) = 0.98487 → 97.5 分位 = 3×157.125 = 471.375。
- **判定**：
  - "某个目标解释全部净缺口" — **成立**（唯一决策格 1301014/target-5）；
  - "只有一个非零场景差" — **成立**（且这正是 35.6% 重抽样错过它、CI 含 0 的原因）；
  - "显著更贵（95%）" — **不成立**（下界恰为 0，含 0）。
- **更正登记**（旧报告保留不改）：R53_REPORT 与 R54_REPORT 中"显著更差/显著"措辞为**待更正**，正确表述为上；R54_CORRECTIONS 当时只更正了舍入数字，未触及该措辞。development_go=false 与成本均值均不受影响。

## 二、late 真实判定式（SOURCE_INPUT_MAP.json / GATE_OBSERVABILITY.json）

源码：`zhenghe/runs/comm_cost_v2_rerun_20260928/v2_rules.py`（LateDiscoveryGatePolicy，LATE_CUT_STEPS=147.0，由 r42_run.run_rule 经 v2_rules.build_rule 构造并与其登记参数互检）。

- **判定式**：对每个候选 `SEND iff item.first_emission AND float(item.ready_step) > 147.0`（**严格大于**）。
- **字段来源**：`SendItem.ready_step`（int，relay tick 计的目标首个未发送变更簿记步）与 `SendItem.first_emission`（bool），均为**动作前合法发送端原始观测**（SendObservation.items）的直接字段。
- **单位与时钟**：relay tick，dt=1.0，整数刻度；比较在整数 tick 上做。
- **决策时机**：目标的一次性首报决策点（tape 中 `one_shot_state_before == "undecided"`）。
- **策略历史依赖**：无——纯函数，不读策略自身历史；预算拒付后下一步重试（first_emission 保持真）。
- **"147"不是 tape step**：它是 ready_step 的比较阈值；已核验的两个焦点格决策步 140/148 恰好跨越它，但那是巧合性的数据位置，不是定义。

## 三、两张映射的判定

- **A（动作前合法原始状态可取得）**：成立。规则策略的 `decide()` 直接读动作前 SendObservation 的 `first_emission/ready_step`。
- **B（仅从 19 维 float32 向量+冻结常量可恢复）**：**成立**。恢复函数：
  `first_emission = (f[5]==1.0)；step = round(f[18]×1000)；ready_step = step − round(f[0])；late_sends = first_emission ∧ ready_step > 147`。
  依据：slot0 = `age_steps = max(0, tick − ready_step)`（integrated_sar/comm.py），slot18 = `min(1, step/1000)`（comm_learning.features，EPISODE_HORIZON_STEPS=1000），slot5 = first_emission。不使用 scene/target ID、结局、接收方知识或拟合阈值。

## 四、现有记录验证（GATE_OBSERVABILITY.json → validation）

- 覆盖：48 条候选 greedy tape 全量解析，**288 个首报决策**（48 场景 × 6 目标，每目标恰一次，无重复）。
- **48/48 场景预测 late 动作集合与实际 late 行完全一致（0 mismatch）**。
- **0 个字节相同的 float32 向量对应相反门控标签**（无可分例碰撞）。
- 焦点格：1301014 t5 → ready_step=140 ≤ 147 → late 跳过 ✓（三 seed 一致，age=0）；1301010 t5 → ready_step=148 > 147 → late 发送 ✓。
- 边界纯计算例：float32 step/1000 对 0..1000 全部精确回整；147→SKIP / 148→SEND 严格边界；age 偏移（age=1、step=148 → ready=147 → SKIP）全部通过。
- 推论更正（针对 R54 第五答）：**"当前输入不能解释 late 的跳过"不再成立**——门控量在 19 维中可恢复；信息不缺失，是两个学习模型在该状态上未利用它（候选在 ready=140 发送、ready=148 跳过，与门控方向相反；BC 在两处均接近确定发送——BC 训练分布中 ready≈121 的目标会在 ≈121 步被立即决策，step-140 才决策属离路径状态）。结论属**类别 1：门控量已在 19 维中可恢复**——下一方法应研究如何利用已有信息（如对 ready_step 的显式利用/门控一致性目标），而不是改观测 schema。
- 接收端状态（消息价值/闭环因果）按约定保留为独立开放问题，不混入本结论。

## 五、唯一下一步建议

在类别 1 证据下：**不做表示消融、不补录、不改 schema**。建议的唯一后续是"利用已有信息"的单一变量方法实验：在训练目标中显式加入门控一致性利用（例如以恢复出的 ready_step>147 作为策略输入的辅助结构或正则项，唯一变量），复用同一 528 计划结构评估。**授权需求**：涉及训练（环境构造+optimizer 更新），需项目负责人按既有流程单独授权；本轮未执行、未新增数值标准、未追加额度。

## 交付物

`STATEMENT_RECONCILIATION.json`、`GATE_OBSERVABILITY.json`、`SOURCE_INPUT_MAP.json`、`r55_validate.py`、`r55_zero_tests.py`、本报告。旧源码、报告、verdict、权重、日志均未覆盖。
