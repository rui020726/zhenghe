# ROUND30 报告：原期限编码 vs 新期限编码的从初值配对训练消融

分支 `exp/comm-deadline-encoding-ablation`（起点 ROUND29 完整头
`99eb746baaa073550f9062fec5c87a4b4711827b`，继承 ROUND28 `21250b5`），产物
`runs/comm_deadline_encoding_ablation_20260930/`。执行依据 `EXECUTION_BASIS.md`（sha256
`d80a7055…`），协议 `ROUND30_PROTOCOL.md`，冻结件 `CONFIG_AND_MODEL_MANIFEST.json`
（freeze_sha256 `b05f07bb…`，configuration_hash `e9b2c067db5951f2` 与 ROUND24 逐字节相同）。

**启动 1040/1040 正常、0 失败、0 重试、储备 40 未动**（64 immediate reference + 768 训练
［6 cell × 128 集］+ 192 评估 + 16 late 规则）。合成 optimizer 更新 44 次（≤64 上限，
与真实更新分账）。发布状态 `PENDING_NETWORK`，本会话网络重试 0。

## 1. 两处证据冲突的零启动核验（§3）

### 3.1 开发块任务数冲突——ROUND29 的文件选择错误，ROUND24 判定成立

从 96 行 eval 行 + 3 个 ep128 里程碑建立身份表（轮次/v1-v2、seed、里程碑、模式、
checkpoint SHA256、场景、配置身份、路径、按时数；`CONFLICT_RECHECK_3_1.json`）：

- ROUND29 曾以"seed33 final greedy 5/16 失败"登记的数值，与 **seed-11** final_greedy 行
  16/16 逐值一致（on_time/spent/emissions），与 seed22/33 均 0/16；
- 盘上 seed33 final greedy 实为 **16/16 全按时**，与 ROUND24 判定及 COMPARISON.json 一致；
- 全部 96 行的 controller 参数哈希与各自 seed 的 ep128 里程碑相符——模式、里程碑、版本、
  配置均无差异，冲突只来自文件选择；
- 裁定：ROUND24 报告成立；ROUND29 的开发块任务数字（5/16）更正为 seed11 行的误挂，
  其块归属核心（1301000..1301015 是 ROUND22/24 开发块、1400031 在评估块）不受影响；
  旧报告保留，本报告新增更正，未追加任何仿真。

### 3.2 时钟关系 `clock.now = tape step − 1`——独立证实

- 源码级：`orchestrator.step()` 内 `executor.step()` 驱动 `relay.tick()` 之后才
  `clock.advance_to(now+dt)`；relay `_tick` 从 1 计、tape step 即该值，而
  `now_of=lambda: clock.now` 读推进前时钟 ⇒ now = step − 1
  （`CLOCK_RECHECK_3_2.json` E1）。
- 数据级：81 个 tape、11 901 个候选行中 **756 行未饱和**，`now=step−1` 下 0 失配、
  `now=step` 下 811 失配；首候选步恒等式（remaining = deadline_offset + 1 且首候选步 =
  检测时刻）478/478 成立。ROUND29 的重建规则从未只靠饱和行证明。
- 本轮 NEW 编码直接消费动作前 `SendItem.remaining_time`，不由分析日志拼装。

## 2. 两臂身份（§4/§6）

- OLD = 生产 `features()`（`comm-send-state/1`）；NEW = ROUND29 登记原型，第 12 列
  `remaining/1000` 不截断（`comm-send-state/1r29-deadline-over-horizon`）。
- 作用域 collector：父类 `_choose_masked` 逐字复制，difflib 登记**恰好一行不同**且就是
  特征行构造（`rows = [self._encoder(observation, item) …]`）；OLD 臂与父类 tape 逐字段
  一致（预检 4）。训练用 StageB MRO，cost critic 读同一臂 tape。
- 初始化配对：三个 seed 两臂参数/逐模块摘要/optimizer 状态/λ **全部逐字节相等**
  （`INIT_PAIRING.json`）；首批门禁实测"更新前权重 = 登记初值"两臂均真。
- 预检 132 项 0 失败：tape↔网络输入（float32 精确）、PPO 样本同臂语义、optimizer 四模块
  精确集合、归档反例套件、mask/likelihood 语义、保存/恢复与跨臂/跨轮拒绝、恢复后确定性
  下一步等于不中断路径。
- 参考核验：64 个新跑 immediate reference 与 ROUND24 归档行 **behaviour_sha256 逐字节
  一致 64/64**（配置与代码复现归档行为）。
- 冻结后两处登记变更（`CODE_IDENTITY.json`）：`r30_blocks.py` 入口分片参数（11 次启动后，
  非行为代码）、`r30_check.py` 门禁 G3 改为用 batch-0 更新前权重重算（72 次启动后，
  验证工具修正；原失败因用更新后权重比对更新前 tape，正确门禁差 2.4e-07/1.8e-07 ≤ 1e-6）。

## 3. 训练与首报诊断（§9）

| 臂 × seed | λ 1.0 → | updates | optimizer.steps | 首报点 | 原始期限不同值 | 期限列编码不同值 | 首报动作 |
|---|---:|---:|---:|---:|---:|---:|---|
| old s11 | 1.5000 | 32 | 464 | 762 | 272 | **1**（全 1.0） | HOLD 495 / SEND 267 |
| old s22 | 1.2375 | 32 | 340 | 762 | 272 | **1** | HOLD 675 / SEND 87 |
| old s33 | 1.4750 | 32 | 464 | 762 | 272 | **1** | HOLD 525 / SEND 237 |
| new s11 | 1.5000 | 32 | 464 | 762 | 272 | **272** | HOLD 476 / SEND 286 |
| new s22 | 1.2375 | 32 | 340 | 762 | 272 | **272** | HOLD 676 / SEND 86 |
| new s33 | 1.4875 | 32 | 464 | 762 | 272 | **272** | HOLD 533 / SEND 229 |

- 训练内部复现 ROUND29 审计：OLD 臂 272 个不同原始期限（61.1–180.2 步）全部折叠为 1 个
  编码值；NEW 臂 272 个原始值保留为 272 个编码值（0.061–0.180）。
- s11/s22 两臂 λ 轨迹与 optimizer.steps 完全相同；s33 出现微小分叉（λ 1.4875 vs 1.4750，
  samples 987=987）。逐批 task/cost/g_hat/λ、双动作与 masked 样本数、两 critic 损失与逐
  模块梯度读数见 `DIAGNOSTICS.json`；未见缺失模块或伪样本。
- accepted 比例未用作任务收益；未拟合任何诊断分类器或阈值。

## 4. 配对比较（§8；`PAIRED_COMPARISON.json`）

**greedy（主执行表）：ΔY = 0，48/48 场景持平，无目标集合交换。**

| seed | ΔY mean（CI95） | ΔY 正/零/负 | ΔC mean B/ep（CI95） | ΔC 便/等/贵 |
|---|---|---|---|---|
| 11 | 0.0（[0, 0]） | 0/16/0 | −9.0（[−27.0, 0.0]） | 1/15/0 |
| 22 | 0.0（[0, 0]） | 0/16/0 | 0.0（[0, 0]） | 0/16/0 |
| 33 | 0.0（[0, 0]） | 0/16/0 | −334.875（[−847.5, 0.0]） | 2/14/0 |

**sampled（单独报告，约束诊断用）：**

| seed | ΔY mean（CI95） | ΔC mean B/ep（CI95） | 说明 |
|---|---|---|---|
| 11 | 0.0（[0, 0]） | −57.0（[−359.6, 240.0]） | 6 便 6 等 4 贵 |
| 22 | +0.0625（[0, 0.1875]） | −4197.75（[−6122.3, −2593.5]） | 1 场景（1301003）both_better：+1 目标且省 14 328 B |
| 33 | +0.1875（[0, 0.5625]） | +58.5（[−55.9, 165.8]） | 1 场景（1301002）+3 目标且更贵 +552 B；其余 15 场景任务持平（3 更便宜、10 小幅更贵、2 持平） |

- 无任何"省字节但丢任务"场景（cheaper_but_lost_task = 0 全 6 比较）。
- 相对既有强规则（16 场景均值）：immediate 6.0 按时/14 939.3 B；late_discovery_gate_147
  6.0/4 838.3 B。新编码 greedy：s11 5.5/28 198.9 B（old 28 207.9）、s22 5.5/0 B（两臂均
  全 HOLD）、s33 **6.0/2 063.6 B**（old 2 398.5 B，同按时下更便宜 334.9 B）。
- 区间为描述性：16 场景配对抽样、PCG64 20260930、10 000 次、linear 分位、共用
  10000×16 索引、手算反例通过；不覆盖训练 seed 不确定性；sampled 每场景一次。
- 三个 seed 分别展示、未池化、未挑最佳；开发块已反复使用，无确认性显著性、无 δ=0 保证、
  无独立泛化主张。

## 5. 五问回答

1. **两臂是否仅有登记的编码差异？——是。** 单行源码差异登记 + OLD 与父类 tape 逐字段一致
   + 生产/导入模块身份与 ROUND24 逐字节相同 + configuration_hash 与 ROUND24 相同
   （`e9b2c067db5951f2`）+ 初始化逐字节配对；其余全部为共享冻结代码。
2. **新编码相对原编码，任务和成本各有什么变化？**
   - 任务：greedy 48/48 持平、零退化零提升、无目标集合交换；sampled 零退化，s22/s33 各
     1 场景有任务增益（+1、+3 个按时目标）。
   - 成本：greedy 平均 −9.0（s11）/0（s22）/−334.9 B/ep（s33），方向在 seed 间不一致；
     sampled s22 −4 197.8（CI 不含 0）、s33 +58.5（CI 含 0）、s11 −57.0（CI 含 0）。
3. **效果是否跨三个训练 seed 一致？——任务侧一致（三 seed ΔY 全 0，greedy），成本侧不一致
   （−9 / 0 / −335 B/ep，含零与反向）。** 一致性只读作逐 seed 均值符号，不包装为稳定性
   或非劣证明。
4. **greedy 与 sampled 是否给出不同结论？——是。** greedy 零任务变化；sampled 在两个 seed
   各出现 1 个场景的任务增益。sampled 观察训练策略本身，其约束估计不转用于 greedy。
5. **哪些结论仍未建立？** 新编码提升（或损害）学习表现的证据不足：greedy 任务零变化、
   成本变化小且方向不一致、sampled 增益各只有 1/16 场景；无确认性显著性/非劣效/泛化
   主张；候选布局的可行性、 effort 与部署价值未建立；不能把本结果解读为"消除饱和即收益"
   ——即使出现收益，也只能归因于整项编码替换（信息保留与数值尺度同时改变）。

## 6. 产物索引

- 证据核验：`CONFLICT_RECHECK_3_1.json`、`CLOCK_RECHECK_3_2.json`
- 冻结与门禁：`CONFIG_AND_MODEL_MANIFEST.json`、`ROUND30_PROTOCOL.md`、
  `RUN_PREFLIGHT.json`、`INIT_PAIRING.json(+6)`、`FIRST4_GATE_old_s11.json`、
  `FIRST4_GATE_new_s11.json`
- 运行记录：`START_LEDGER.json`（1040/1040，0 失败）、`CHECKPOINT_INDEX.json`（6 个
  ep128 里程碑哈希）、`TRAINING_CURVES.json`、`DIAGNOSTICS.json`、`ARTEFACT_MANIFEST.json`
  （3 062 日志/1 040 行/216 checkpoint，路径+大小+sha256）
- 统计：`PAIRED_COMPARISON.json`（含 hand_check）、`ROUND30_VERDICT.json`、`CODE_IDENTITY.json`
- 原始日志/权重/tape/checkpoint 按既有规则留在 Git 外。

## 7. 停止

本地交付后停止：不追加训练、不调参、不换候选、不启动确认实验、不部署、不自动推送。
下一轮方向（仅建议，非授权）：在"整项编码替换"内部解耦信息保留与数值尺度（例如把新列
再做一次无截断的尺度归一），或把观测设计讨论推进到 ACK/接收端状态；二者均需另行批准。
