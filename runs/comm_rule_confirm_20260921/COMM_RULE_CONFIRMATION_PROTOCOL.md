# 规则独立工程确认协议（预注册：先于任何新数据写定）

日期：2026-09-21。分支 `analysis/comm-rule-independent-confirmation`，起点
`feat/comm-ack-feedback-gate@fbf96d8aa30fb2600fd710eab79a7b5f8121422a`，只引入文档提交
`2e341b1b321feef6c270b82cf81dd1f9d5c3b3bd`。对应总纲 v1.9 §5.11/§9 与执行指令
`COMMUNICATION_S3_CLOSURE_NEXT_ROUND.md`。产物目录 `runs/comm_rule_confirm_20260921/`。

**状态：协议冻结，尚未录制任何新流、尚未跑任何正式 episode。** 本文在任何新数据产生之前写定
全部配置、臂、统计与停止条件；结果文档只能报告本文写明的量，不得在看结果后增删判据。

## 1. 本轮问的问题（两件互不混淆的事）

1. **冻结研究结论**：在已经测试的 no-ACK、policy-owned timing 和 costed ACK 协议中，学习策略
   没有建立强规则之外的价值；该结论限定于现有任务分布、观测、动作、奖励和训练资源。
2. **回答工程问题**：第六轮中 `retry_until_acked_2` 在 48 条有损训练流上以 +19 562 B
   total bytes（+88.5 %）换来 +0.875 个按时任务/episode（CI [+0.5625, +1.1875]）。
   这条成本—效果关系能否在**全新数据**和**正常信息路径**上复现？

本轮不训练任何模型，不延长旧 checkpoint，不确认单个最好 seed，不改变奖励、网络深度、任务分布、
deadline 或生产默认配置。ACK 门禁 3 未过不是"再训练一次"的许可。学习最终测试块
`800000..800399` 继续封存。

## 2. 新工程块 `900000..900199` 与零重叠审计

**候选块：`900000..900199`（含两端，共 200 个 seed）。** 它只用于规则工程确认，
**不得在未来称为学习模型的未见测试集**；`800000..800399` 的用途不变。

审计在生成任何流之前执行，取三源证据，**artefact 优先**（已执行过的 episode 不能被"没看过"）：

1. **盘上 artefact**：`runs/**` 下所有 episode 产物文件名（`rp_`/`tr_`/`ct_` 前缀、
   `_s<seed>.jsonl` 结尾）与已录制流文件（`s<seed>.json`）——这是全项目实际执行过的并集，
   不论有没有文档提过；
2. **各轮脚本声明的 seed 区间**：`runs/*/**/*.py` 里 `range(...)` 形式、下界 ≥1000 的声明
   （小区间是循环计数器，不算场景块）；
3. **同号流内容比对**：同名 `s<N>.json` 若内容不同，是命名冲突而非数据泄漏，必须展开报告。

**通过条件：候选块与三源的交集均为 0。** 若发现任一重叠，停止并报告，**不自行更换 seed 段**。
审计脚本 `block_audit.py` 输出 `COMM_RULE_BLOCK_AUDIT.json`，含交集清单与判定。

### 2.1 流的录制与登记

审计通过后**一次性**录制 200 条流，录制配置逐字段取冻结块自己的常量
（`runs/corrected_env_final_20260918/corrected_block.py: RECORD_CONFIGURATION`：live PILOC
搜索、coordinator off、membership-consistent 环境、2 搜索 + 3×2 救援、6 目标、60×60、
障碍密度 0.15、检测半径 5、deadline tightness (0.5, 2.0)、dt 1.0、1000 步）。每条流写
生成配置、当时代码 blob、逐流 SHA256 与折叠摘要到 `COMM_RULE_STREAMS.json`。

**技术失败处理**：只能按预登记原因（进程被杀、磁盘满、checkpoint 加载失败）用**原 seed**
重跑；不剔除困难 seed，不补抽"好看"的样本。任何重跑都登记原因与次数。

## 3. 冻结配置（逐字段继承第六轮，不写"同上"）

以下字段全部来自第六轮 `runs/comm_ack_value_20260920/` 的真实运行配置，逐字段展开：

| 字段 | 值 | 来源 |
|---|---|---|
| 发送时机协议 | `policy_owned_timing` | 第五轮新增、第六轮沿用；传输层不代发 |
| 静默上界 | `max_wait_steps=None`、`urgent_wait_steps=None` | 第六轮 `ack_matrix.run_arm`；上界不由传输层执行，五条臂也都不引用 3/10 上界，因此五条臂的有效静默约束同样为"无" |
| 任务上行硬额度 | 29 300 B/episode | 锁定标定 `COMM_BUDGET_LOCKED.json` 的 25 % 档（`protocol_common.BUDGETS_BYTES["25pct"]`） |
| 每步发射容量 | `sends_per_tick=None`（不设） | 第六轮同 |
| 上行延迟 | `delay_ticks=0` | 第六轮同 |
| 上行丢包 | 按设置：1.0 / 0.7 / 0.7 | 见 §4 |
| ACK 开关 | 按臂：`noack_send_once` 关，其余四条开 | brief §3 |
| ACK 反向延迟 | `ack_delay_ticks=0` | 第六轮 `ACK_DELAY_TICKS` |
| ACK 反向丢包 | 与所在设置的上行丢包**同值**（1.0 / 0.7 / 0.7） | brief §3"ACK 臂反向信道同参数"；显式传入，不依赖默认 |
| subscriber 拓扑 | `comm_subscribers="all"` → 全部 6 台救援机器人（3 species × 2） | C 臂冻结配置；广播一次按 6 个接收者计费 |
| episode 长度 | 1000 联合步，dt = 1.0 | 冻结块 `steps=1000` |
| 场景 | 60×60、障碍 0.15、6 目标、deadline tightness [0.5, 2.0]、检测半径 5 | 冻结块 |
| 分配侧 | C 臂：coalition_planner + sticky_commitments + task_membership_consistency，CTDE/PILOC checkpoint 按 `ARMS_CONFIG_TRUTH.json` 哈希核对 | 冻结块 |
| 观测/动作/奖励 | 不动；ACK 臂的 `/2` 四维特征只供规则读取，本轮无学习策略 | — |

### 3.1 随机数：两套过程，两种配对强度

- **前向（上行）丢包**：每 episode 一个 `np.random.default_rng(episode_seed)` 序列流，
  episode 开始时重新播种，每次发射按 subscriber 顺序每接收者一抽。这是第六轮的既有机制
  （`comm.py:_emit`）。由于各臂发射序列不同，**物理包无法一一对应**：no-ACK 与 ACK 臂的
  前向丢包是**同分布比较**（同 seed、同机制，不是同一串事件），报告必须这样标注。
- **反向（ACK）丢包**：counter-based，key = `(episode_seed, "uplink_ack", subscriber,
  target_id, ordinal)`，交付 tick 进入 seed 序列（`comm.py:_deliver_acks`）。两个 ACK 臂
  面对同一 key 抽到同一个数（共同随机数）；该过程独立于前向流，ACK 开启不消耗前向流。
- ACK 关闭时不构造任何反馈状态、无反向 RNG、无额外 transcript——`noack_send_once` 的
  前向路径与历史 no-ACK 路径逐字节相同。

### 3.2 运行器不是漂移

本轮自带 episode runner（`confirm_env.py`），它是第六轮 `protocol_env.run_episode` 的副本，
只多一个"显式指定流路径"的参数（新块流不在旧轮目录里）。`block_audit.py`/preflight 必须包含
**同流一致性检查**：同 seed、同配置下，本轮 runner 与 `protocol_env.run_episode` 的
任务结局、全部传输计数、belief 统计、`config_hash` 与 `arm_label` 逐字段相同，否则本轮
runner 是漂移，不得开跑。

## 4. 三个设置与五条臂

**三个设置**（不合并样本）：

| 设置 | 信息路径 | 上行 delivery | ACK 反向 delivery |
|---|---|---|---|
| `isolated-lossless` | 隔离（`gossip_interval=100000`，视距刷新与一跳 gossip 关闭） | 1.0 | 1.0 |
| `isolated-lossy` | 隔离 | 0.7 | 0.7 |
| `normal-lossy` | 正常（`gossip_interval=2`，保留视距刷新与 gossip） | 0.7 | 0.7 |

**五条臂**（在三个设置上全部运行，不根据中途结果删臂）：

| 标签 | ACK | 发送规则（policy/params） | 作用 |
|---|---|---|---|
| `noack_send_once` | 关 | `send_once` | 最省的 no-ACK 工程基线 |
| `ack_send_once` | 开 | `send_once` | 单独量化 ACK 协议开销 |
| `ack_retry_2` | 开 | `retry_until_acked`，`max_attempts=2` | 训练集完成数最强 ACK 规则 |
| `ack_periodic_20` | 开 | `periodic`，`interval=20` | ACK 规则前沿参照 |
| `ack_backoff_16` | 开 | `exponential_backoff`，`base_interval=16,max_interval=32` | 退避型前沿参照／无损负控强臂 |

规则参数来自已完成的 48 条训练流（第六轮矩阵），**不在新 200 seed 上重新选择**。旧学习
checkpoint 不参加本轮（它们没有通过进入独立确认的门槛；混入最好 seed 会造成选择偏差）。

## 5. 规模、吞吐探测与分片

总规模 **200 seeds × 3 设置 × 5 臂 = 3 000 episode**。

1. **先测吞吐**：用 5 个**非新块** seed（从 48 条训练流取）跑全部 15 个臂×设置组合的样本，
   实测单集耗时、单集磁盘与产物完整性，再决定分片数。**不得沿用旧耗时估计**（第六轮均值
   3.81 s/episode 只作上限参考，不作为计划依据）。
2. 正式任务写独立 scratch/输出（`runs/comm_rule_confirm_20260921/evaluation/<设置>/...`），
   checkpoint、原始流、逐 episode 日志不入 Git。
3. **分片只在一处应用**（一个 `shard_of` 函数）；resume 扫描在分片之外，不得二次分片。
4. 任何技术失败按 §2.1 处理。

## 6. 指标与统计口径

**主任务指标**：按时完成任务数，单位 个/episode，冻结定义
`final_progress >= 1.0 且 final_failure_reason != "deadline_expired"`，按 `task_id` 去重。
同时报告逐局率、汇总率、expired、未发现目标、逐目标新增失败；条件完成率（已发现中按时）
与全部目标完成率分开。

**通信指标必须分列**：

- task uplink attempts / bytes；
- ACK reverse attempts / bytes（丢失也计费）；
- **total attempts / bytes = uplink + ACK**；
- delivered、applied、lost、duplicate、coalesced、refused；
- 信息年龄、发现到首次 belief / 分配 / 工作 / 完成的延迟。

任何"降本"结论**只看 total bytes**。ACK 臂上行更少但反向字节更多，不得称省通信。

**统计**：同 seed 配对百分位 bootstrap，**B = 10 000，RNG seed = 20260919**（沿用
`evaluate_pilot.paired` 的冻结实现，seed 级配对，2.5/97.5 分位）。三个比较：

1. `ack_retry_2 − noack_send_once`：协议整体的成本—效果差；
2. `ack_retry_2 − ack_send_once`：ACK 信息之上的重发规则收益；
3. `ack_periodic_20`、`ack_backoff_16` 与 `ack_retry_2`：规则前沿是否稳定。

每个比较给完成数差与 total-byte 差的均值、95 % CI、胜/平/负（逐 seed 计数）。
**三个设置不合并样本。**

**每增加一个按时任务的字节增量**：点估计 = 均值分子 / 均值分母，同时报告原始分子与分母；
bootstrap 逐 replicate 计算比值，**分母 ≤ 0 的 replicate 标记 undefined 并报告其数量**，
分位数只在 defined replicate 上计算；点估计分母 ≤ 0 时整个比值记 undefined，不删除样本后
给出虚假稳定比值。三个分子分别报告：total bytes、uplink bytes、ACK bytes。

**break-even**：本轮**不预设**"一个任务值多少字节"的效用权重，因此不自动宣布 ACK 值得部署。
结论只给出实测增量字节；只有当使用者对每增加一个按时任务的价值高于该增量时，ACK 重试才在
该效用定义下合算。

## 7. 预登记判读规则（先于结果）

### 7.1 无损负控

`ack_retry_2`、`ack_periodic_20` 或 `ack_backoff_16` **不应**在 `isolated-lossless` 上稳定
提高完成数；若任一相对 `noack_send_once` 的完成数 CI 下界 > 0，**先暂停并检查 ACK/规则是否
改变了除反馈之外的行为**。ACK 臂 total bytes 增加是预期成本，不是通过。

### 7.2 有损隔离确认

若 `ack_retry_2 − noack_send_once` 的完成数 CI 下界 > 0，则第六轮"ACK 重试以更多成本换
完成数"的工程效应得到**独立复现**；同时必须报告 total bytes CI，不能用任务收益遮蔽成本。
若 CI 含 0，训练集上的 +0.875 **不获得独立支持，停止部署讨论**。

### 7.3 正常信息路径

只有 `normal-lossy` 上完成数 CI 下界 > 0，且没有执行正确性退化，才把 `ack_retry_2` 提议为
**可选高可靠模式候选**。即使满足，也不自动改默认值；是否接受字节成本由使用者决定。

若 `isolated-lossy` 有收益而 `normal-lossy` 没有，结论限定为"隔离上行实验有效，真实系统旁路
使收益不可分辨"，**不部署 ACK**。若正常路径任务结局相同但 total bytes 更高，明确判为
**工程上被支配**。

### 7.4 规则前沿

若其他 ACK 规则在新数据上超过 `ack_retry_2`，**不回头调参或重命名主规则**；如实报告规则
选择不稳定。本轮只确认已锁定集合，不从 200 seeds 重新挑参数。

## 8. 停止条件

- 本轮不训练、不延长旧 checkpoint、不确认单个最好 seed、不改奖励/网络/任务分布/deadline/
  生产默认值；不得通过提高丢包率、收紧 deadline 或改变场景制造学习信号。
- `800000..800399` 不读、不生成；`900000..900199` 不得在结果出现后改作学习测试。
- 聚合前逐文件校验：**arm、ACK 开关、信息路径、delivery probability、预算、代码 blob、
  stream hash、final record 和 metric schema 全一致**；任何不一致 fail closed——
  隔离而不删除，再按原 seed 重跑。
- 正确性指标：虚假能力推进、重复能力贡献、错误释放、非法实际执行、持续无进展，任一非 0
  即报告并停止判读（按第六轮口径，这些是执行安全边界）。

## 9. 交付物

- `COMM_RULE_CONFIRMATION_PROTOCOL.md`（本文）；
- `COMM_RULE_BLOCK_AUDIT.json` / `COMM_RULE_STREAMS.json`（零重叠审计与 200 流登记）；
- `COMM_RULE_CONFIRMATION_RESULTS.md` / `.json`；
- `COMMUNICATION_LEARNING_VERDICT.md` / `.json`；
- `COMM_RULE_CONFIRMATION_MANIFEST.json`；
- 必要的评价测试与复跑命令。

## 10. 复跑

```bash
cd runs/comm_rule_confirm_20260921
python block_audit.py                    # 900000..900199 零重叠审计（在任何录制之前）
python record_streams.py --shards 4 --shard 0   # 录制 200 条新流（shard 0..3）
python confirm_common.py                 # 自检：runner 与第六轮无漂移
python confirm_preflight.py              # 前置检查（含吞吐探测、成本口径、封存块）
python confirm_matrix.py --mode matrix --shards 6 --shard 0   # 正式 3000 episode（shard 0..5）
python confirm_matrix.py --mode report   # 汇总 + 配对 bootstrap + 判读
python confirm_manifest.py               # manifest（git blob 身份、产物折叠、封存检查）
```
