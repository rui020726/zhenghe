# ROUND61：冻结模型 early 首报价值配对干预（18 项固定计划已完成）

分支 `exp/round61-early-report-value-intervention`。对冻结的 R60 合规 seed33 ep128 greedy 模型，做单因素 early 首报 SEND/SKIP 配对干预。只读加载五模块模型参数（param `bce71a0e…`，file SHA `573e4c26…`，load_mode read_only，未恢复 optimizer/λ/训练 RNG/游标）。不训练、不改 β/147/观测/世界配置。机器结论 `ROUND61_MACHINE_VERDICT.json`，逐案例配对 `raw/ROUND61_PAIRED_ANALYSIS.json`，账本 `raw/ROUND60_GUARD_SNAPSHOT.json` + `raw/guard_ledger.jsonl`，案例冻结 `ROUND61_CASE_FREEZE.json`。

## 资源守卫（先修，零环境验收）

`r61_guard.py`：唯一启动入口，调用真实 builder 前原子领取并消耗 slot（O_CREAT|O_EXCL），构造失败/中断/隔离/重试均不退额；分别记录 slot 消费、builder 调用、环境构造、run 调用/返回、实际步数、核心结果资格。`test_r61_guard.py` 16 项纯合成+spy 测试全过：19 消费后仅可再领 1、20 后拒第 21；构造失败/结果不合格不退额；重启/换子目录/隔离不清零；备份非新执行、同输出不去重；并发抢最后 slot 仅一胜；未注册路径在环境构造前被拒；R61 与 R60 账本独立、R60 历史 608 不被当作 R61 余量；R61 为新授权。R60 超限 8 为历史偏差，不退额、不事后批准、不借用。

## 8 案例冻结（从既有输入，不跑环境选例）

候选= R60 seed33 greedy 已存 32 场景 eval tape 中真实已发现、baseline 首报为 free_one_shot_choice（自由/合法/可付）、first_emission、ready_step≤147 的目标。共 **146 个 early 候选，全部 baseline 动作为 SKIP（0 个 SEND）**——该冻结模型 SKIP 掉每一个 early 首报。距离仅用合法输入列 13/14/15/18 按范围缩放平方和，恒定列记 0；选择顺序：焦点必选 → SEND 候选最多 3 个不同场景（本例 0）→ SKIP 按距离补足 8 → 不足再从其余合格候选补；并列按 scene、target。动作分层不足按规则处理，不强行造标签。结果：焦点 scene1700027/piloc-e1-target-5（ready=134，SKIP）+ 7 个 SKIP 对照（1700015/1700012/1700000/1700009/1700016/1700018/1700005，dist 0.20–0.90）。

## 干预与门禁

干预器 `r61_intervene.py`：冻结 `OneShotSendSkipCollector._choose_masked` 源码注入唯一一处——在选定目标首次合法可付自由决策处、真实顺序 mask 之后、模型自身动作抽取之后、一次性 latch/发射/shadow-gate 计费之前，将动作覆盖为 FORCE_SEND(SEND)/FORCE_SKIP(HOLD)；latch、requested、decision_class、gate 计费均用覆盖后动作，故 one-shot 关闭状态正确更新、SEND 真实计费、同 _emit 路径、不额外发射、不重试；记录原模型概率与原 greedy 动作；每 episode 仅覆盖一次，之后保持冻结模型原 greedy。L（原 greedy 路径）与 L0（装干预器但禁用覆盖）门禁 **passed**：L==L0==R60 焦点原件（on_time=5、bytes=4812、emission 序列一致），模型文件与参数不变，训练副作用 0；路径/版本/输出根 metadata 差异显式分列。门禁过才执行 16 格。

## 五点收口

1. **关键目标在 SEND/SKIP 下是否改变结局**：是。焦点 1700027/target-5：FORCE_SEND → 目标按时（Y_SEND=1），全系统 6；FORCE_SKIP → 目标丢失（Y_SKIP=0，deadline_expired），全系统 5。目标按时差 Y_SEND−Y_SKIP=**+1**，成本差 **+2688B**（SEND 更贵但救回目标，无其他目标得失）。即该 early 首报是 consequential 的。
2. **对照 early 首报的收益变化**：7 个对照全部 Y_SEND=Y_SKIP、全系统按时 6→6 不变、无新增/丢失目标；仅 FORCE_SEND 更贵（+378..+2688B）。即这些 early 首报对该实现的任务结局 non-consequential（SKIP 是“安全”的），SEND 只是多付费。
3. **合法输入的区分线索与仍缺证据**：8 例动作前 19 维向量中，焦点与对照在列 13/14/15/18 上相近但不同；**0 个相同 float32 向量出现不同干预收益**（未观测到状态混叠）。但这是 8 个事后选定案例、单一冻结策略、每场景一次随机实现的闭环干预：相近向量不能证明不可分，未发现冲突也不能证明 19 维充分；原始 deadline/remaining_time 等合法 SendItem 字段只提供候选线索（焦点 target-5 deadline≈231.85、ready=134），本轮不加特征、不拟合规则、不训练分类器、不建立部署策略。
4. **消费/失败/重试与 20 上限闭合**：**19/20 slot 消耗**（18 normal + 1 reserve），headroom 1；18 项合格结果（L + L0 + 16 FORCE）；1 次技术失败（L0 首跑因 `shadow_gate` 命名缺陷中断，保留原件不退额，修复后零环境重验收、以 reserve 重试成功）；builder 调用 19、记录环境构造 18（L0 首跑环境已构造但其 post-run 记录因崩溃缺失，实际=19= builder 调用数）、run 返回 18、步数 18000。有效结果 18 ≠ 实际消费 19（差即 L0 首跑）。未超 20 上限。
5. **训练副作用与 HEAD**：真实/合成 optimizer 更新、PPO 重放、训练恢复、checkpoint 写入全部 **0**（`runtime/checkpoint_writes.jsonl` 0 行、无 v2_train_*.pt、模型文件 SHA 不变、只读加载未恢复 optimizer/RNG）；只读 CPU forward 仅用于模型身份核验，未额外计数。HEAD 见本次收口提交。

## 结论边界

这 8 例是事后选定案例、单一冻结策略、每场景一次随机实现的闭环干预。能描述该实现中焦点 early 首报的 SEND/SKIP 差异（SEND 救回 target-5、SKIP 丢失），不能证明消息普遍必要、安全跳过、总体非劣或泛化；不做小样本显著性包装，不把 L/L0 算作额外独立样本；不将“结局相同”标成安全删除真值，不将 accepted 当成任务收益；不宣称 19 维充分或不可分。两份 ROUND46 脏文件 SHA256 维持 `cf3ad198…`、`93a2f69f…`，未修改、未暂存。本地显式点名提交报告/verdict/代码/测试/冻结/账目；不提交权重与原始仿真日志；不推送/合并/部署。完成后停止，不自动开启训练或下一实验。
