# 证据修订清单：十项核对（原文 → 修订）

日期：2026-09-21。分支 `analysis/comm-evidence-review`，总纲 v2.1，执行指令 `COMM_RESEARCH_REVIEW_NEXT_ROUND.md` §3。机器可读：`COMM_EVIDENCE_CORRECTIONS.json`。

**本轮不产生新数据**：新 episode 0、新流 0、训练 0。归档件（`runs/comm_rule_confirm_20260921/COMM_RULE_CONFIRMATION_RESULTS.md`、`runs/comm_rule_confirm_20260921/COMMUNICATION_LEARNING_VERDICT.md` 及其判定）原样保留、可追溯；修订件是本目录内的 `_REVISED` 文档，由脚本从机读对象渲染。所有核对结论只区分两类：**只改解释/措辞**，或**统一到已有的一套数值口径**（均值不变），没有任何预注册判据被后验修改。

## 汇总表

| # | 项 | 是否改变数值 | 修订落点 |
|---|---|---|---|
| 1 | 单位与信道语义 | 只改解释与措辞；配置未动 | COMM_RULE_CONFIRMATION_RESULTS_REVISED.md 〇/一、COMMUNICATION_LEARNING_VERDICT_REVISED.md 四 |
| 2 | 成本方向不得渲染为'更好' | 只改措辞；底层计数与均值未变 | REVISED.md 二/三（`_cost_counts`） |
| 3 | 显著性、等效与'前沿'的定义 | 只改解释；无预注册判据被改 | REVISED.md 三（`frontier_definitions` 三口径） |
| 4 | 比值区间必须是条件区间 | 只改标注与呈现；未改任何抽样实现 | REVISED.md 二/六（`ratios.*.interval_kind`、`primary_ratio_check`） |
| 5 | 重复数字的根因：同一量被抽了两次样 | 数值（分位边界）统一到已有的一套；未新跑任何 episode，均值不变 | REVISED.md 三/七、REVISED.json `old_output_diff`、verdict 修订 §六 |
| 6 | ACK 比较对象的开关 | 只改解释；第六轮数字原样保留 | REVISED.md 四（比较对象表）、verdict 修订 §四 |
| 7 | 因果范围：重试差 ≠ 反馈净价值 | 只改解释 | verdict 修订 §四、REVISED.md 三（`causal_scope`） |
| 8 | 学习结论的三条事实 | 只改解释 | verdict 修订 §一/二 |
| 9 | 训练充分性的表述 | 只改解释；未追加训练 | verdict 修订 §二、COMM_LEARNING_DIAGNOSTIC.md |
| 10 | 门禁 3 的含义边界 | 只改解释；门禁判定与停止决定未改 | verdict 修订 §四（门禁 3 含义边界） |

## 逐项

### 1. 单位与信道语义

- **原表述**：`COMM_RULE_CONFIRMATION_RESULTS.md` 每个设置标题写'上行与反向丢包均 1.0 / 0.7'，并把三种成本（B/ep、B/新增任务、个/ep）在同一页里混用；verdict 与总纲沿用同一措辞。
- **证据路径/字段**：`integrated_sar/comm.py:894`（`if self._rng.random() > self._delivery_probability: drop`）与 `integrated_sar/comm_ack.py:212`（`stream.random() < delivery_probability` 判送达）：字段 `delivery_probability` 是**成功率**。1.0 ⇒ 丢包率 0.0，0.7 ⇒ 丢包率 0.3。
- **更正内容**：设置标题改为'上行与反向投递成功率均 1.0（丢包率 0.0）/ 0.7（丢包率 0.3）'；报告加'〇、单位与口径'图例：个任务/episode（完成数差）、B/episode（每局字节差）、B/新增按时任务（比值）、完成率（[0,1] 比例，不是百分点）。四者不可互换：+0.835 是个/ep，+26 354 是 B/ep，31 562 是 B/新增任务，不是 B/ep。。31 561.7 一律带单位写作 B/新增按时任务，26,354.1 带单位 B/episode。
- **是否改变数值或只改解释**：只改解释与措辞；配置未动
- **修订落在**：COMM_RULE_CONFIRMATION_RESULTS_REVISED.md 〇/一、COMMUNICATION_LEARNING_VERDICT_REVISED.md 四

### 2. 成本方向不得渲染为'更好'

- **原表述**：归档报告 total-byte 列的逐流计数写 '200 更好 / 0 相等 / 0 更差'，判读段同样用'更好'描述 +26 354 B 的字节增量。
- **证据路径/字段**：`runs/comm_rule_confirm_20260921/render_results.py:_counts` 把完成数与字节的seeds_better/seeds_worse 复用同一套'更好/更差'词；字节轴上 seeds_better 计的是差值为正的流。
- **更正内容**：字节轴改用'更高/相等/更低'（`revise_results._cost_counts`），只有 on_time 轴保留'更好/相等/更差'；判读段对 +26 354.1 B/ep 明确写'是更贵，不是更好'。
- **是否改变数值或只改解释**：只改措辞；底层计数与均值未变
- **修订落在**：REVISED.md 二/三（`_cost_counts`）

### 3. 显著性、等效与'前沿'的定义

- **原表述**：归档报告把 periodic 相对锁定规则的结果写成'规则前沿：稳定 ⇒ 未超过锁定规则'，并在正文用'打平'描述（总纲 §5.12 同）。
- **证据路径/字段**：修订 results：isolated-lossy `ack_periodic_20 − ack_retry_2` 完成数 -0.0050 个/ep，CI [-0.1700, 0.1600] 含 0，下界 < −0.05；total-byte 差 -3,569.0 B/ep，CI [-4,278.5, -2,875.4]。
- **更正内容**：CI 含 0 只写'未检出差异'，不写效果相同，也不写通过 −0.05 非劣界（本例下界 -0.170 连非劣界都没过）；成本侧更省单独成立照单报。'前沿'拆成三个定义分别给结论：样本均值 Pareto、逐流支配、有统计支持的支配（`frontier_definitions`），不再用'稳定/不稳定'替代定义。
- **是否改变数值或只改解释**：只改解释；无预注册判据被改
- **修订落在**：REVISED.md 三（`frontier_definitions` 三口径）

### 4. 比值区间必须是条件区间

- **原表述**：归档报告在丢掉非正分母抽样后仍把结果列标题写作 '95 % CI'，未区分无条件/条件区间；对分母极小的 periodic 比较也照样给出比值数字。
- **证据路径/字段**：修订 results：无损 periodic 比较 4116/10 000 个抽样分母非正；正常路径 periodic 比较 1222/10 000；主比值（retry/noack，isolated-lossy）未定义抽样 0。
- **更正内容**：新增'区间类型'与'可稳健解释'两列：丢掉非正分母后得到的区间标 `conditional`，不是无条件 95 % CI；点估计分母非正者记 undefined 且不删样本求稳。主比值单独核对：冻结 RNG 区间 [26,502.7, 38,459.3] vs 独立种子 20260921 区间 [26,521.8, 38,508.4] ⇒ 对抽样不敏感，可引用，并同时给分子分母。
- **是否改变数值或只改解释**：只改标注与呈现；未改任何抽样实现
- **修订落在**：REVISED.md 二/六（`ratios.*.interval_kind`、`primary_ratio_check`）

### 5. 重复数字的根因：同一量被抽了两次样

- **原表述**：主表 CI [+0.685, +0.995] vs verdict/判读段 [+0.685, +0.990]；成本 CI [26 002, 26 688] vs [26 018, 26 687]；periodic 两处 CI 也不一致。
- **证据路径/字段**：`confirm_matrix.judge()` 用只含两条臂的行集重算 `_difference`，`report()` 用全部五条臂；`evaluate_pilot.paired` 每次调用重新播种但按臂序逐块消耗抽样 ⇒ 同一统计量落在随机流不同位置。修订 results 的 `old_output_diff` 核对 16 对，均值改变 = False，仅分位边界不同的 9 对已逐条列出。
- **更正内容**：所有统计量只算一次（每 (设置, 轴, 基线) 一次全臂集调用），判读段与 verdict 只从该对象切片；§三 标题改为'切片，不再计算'。统一后的口径是主表那一套。`aggregate_invariance` 逐项核对 3 000 行输入的摘要未变（585 个字段全部相同）。
- **是否改变数值或只改解释**：数值（分位边界）统一到已有的一套；未新跑任何 episode，均值不变
- **修订落在**：REVISED.md 三/七、REVISED.json `old_output_diff`、verdict 修订 §六

### 6. ACK 比较对象的开关

- **原表述**：多处把第六轮的 `send_once` 称作'最强 no-ACK 规则'，与本轮 +26 354 B 并列引用。
- **证据路径/字段**：`ack_matrix.run_arm` 对全部 22 臂固定 `ack=True`；gate JSON `2_information_pays` 的基线 `send_once` total = 22,110.3 B，其中 ACK 7,374 B ⇒ 该基线反向信道是开的，只是策略不消费确认状态。
- **更正内容**：报告新增 §四表：逐对象列出 ACK 开关/是否消费反馈/重试能力/额度。三个成本差是三个不同比较：+26 354.1（基线无信道）、+19 100.7（基线有信道不消费）、+19 562（第六轮同类基线、48 条训练流）；不再作为同一基线的变化并置。
- **是否改变数值或只改解释**：只改解释；第六轮数字原样保留
- **修订落在**：REVISED.md 四（比较对象表）、verdict 修订 §四

### 7. 因果范围：重试差 ≠ 反馈净价值

- **原表述**：'ACK 信息真实且有后果（一条简单规则拿到全部增益）'与'不是多花字节买分'。
- **证据路径/字段**：ack_retry_2 − noack_send_once 同时改变反向信道、是否消费确认、是否有第二次尝试；逐目标新增失败 23 个、救回 190 个（净差 167 = 200 × 0.835）。
- **更正内容**：该差值写成三者联合效应；无损打平只支持'该重试收益与所测丢包条件有关'，不支持'不是花钱买分'。分离反馈价值需要'相近重试机会/调参资源的无反馈重试'对照，本轮未新增任何运行（列为下一步）。
- **是否改变数值或只改解释**：只改解释
- **修订落在**：verdict 修订 §四、REVISED.md 三（`causal_scope`）

### 8. 学习结论的三条事实

- **原表述**：归档 verdict 把六轮并陈为一段理由，读者易读成'ACK 模型也训练失败'。
- **证据路径/字段**：S3-1…S3-4 训练的是 no-ACK 观测；S3-5 只有规则与门禁（gate JSON `decision = stop and deliver the ACK gate negative result`）；本轮 3 000 episode 五条臂全是规则。
- **更正内容**：verdict 修订版 §一 用三行分开：no-ACK 未建立稳定优势 / ACK 从未训练模型 / 本轮无学习臂；并把'ACK 模型训练失败'列入 FORBIDDEN_CLAIMS 的禁用清单。
- **是否改变数值或只改解释**：只改解释
- **修订落在**：verdict 修订 §一/二

### 9. 训练充分性的表述

- **原表述**：归档 verdict 理由段：'六轮中每一轮的失败都有具体归因——欠训练（已用扩训排除）…'。
- **证据路径/字段**：`COMM_LEARNING_DIAGNOSTIC.json`：曲线非单调（25 % seed1 @100k=@200k=2.9583 后跳升；75 % seed1 5.4583→4.7917→5.5417→5.7500），value loss 未持久化，optimizer.step 仅 2 792–5 676 次。
- **更正内容**：删除'已用扩训排除'的绝对表述，改为'扩训排除的是完全学不动，不排除仍欠训练、时序信用分配过稀或表征不足'；停止决定原样保留。四种解释的证据表见诊断文档。
- **是否改变数值或只改解释**：只改解释；未追加训练
- **修订落在**：verdict 修订 §二、COMM_LEARNING_DIAGNOSTIC.md

### 10. 门禁 3 的含义边界

- **原表述**：'425 个受控反事实点里 96.5 % 发送/沉默对任务结局无差别 ⇒ ACK 状态没有形成可学习的逐决策取舍'，并被总纲转写为'96.5 % 的决策点上…'。
- **证据路径/字段**：gate JSON `3_real_choice`：425 点、一致性 425/425，门槛 20 点 / 8 流；`ack_counterfactual.POINTS_PER_STREAM` 注释与门禁文档 §三记录了每流 2 点（93 点）→ 修 orders_selection 顺序缺陷 → 每流 10 点（425 点）的扩样顺序，打平率 87 %→96.5 %。
- **更正内容**：表述限定为：这是特定采样分布、单步干预与固定后续策略下的观测；各 ≥20 点/≥8 流是资源门槛，不是强化学习可行性的必要条件；不写'96.5 % 的所有决策没有学习空间'。补样顺序如实披露。跨步取舍是否存在的问题交给多步干预协议（本轮只设计）。
- **是否改变数值或只改解释**：只改解释；门禁判定与停止决定未改
- **修订落在**：verdict 修订 §四（门禁 3 含义边界）

## 未纳入本轮的东西

- 不为修文字重跑仿真；现有矩阵不足以复核的项在诊断文档里标'证据不足/待测'，不用推测补齐。
- 不改生产模块与默认值（`comm_ack=False`、`comm_timing_mode=legacy_shared_bound`）。
- 不触碰封存测试块 `800000..800399`；`900000..900199` 已消费，只作既有证据使用。
- 不修改非本轮创建的 `architecture/` 与 `runs/coordinator_baselines_20260917/dev_summary.json`。

## 复跑

```bash
cd runs/comm_evidence_review_20260921
python revise_results.py && python revise_verdict.py && python evidence_corrections.py
```

