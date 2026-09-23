# 完整系统瓶颈盘点：同一目标 ID 串起八个阶段（只读，0 episode、0 训练）

日期：2026-09-23。分支 `analysis/comm-research-closure`，起点
`analysis/comm-recipient-topology-feasibility@9db4cdf073d2…`（远端完整 SHA 逐字符核对）。
机读件 [`JOINT_SYSTEM_BOTTLENECK_AUDIT.json`](JOINT_SYSTEM_BOTTLENECK_AUDIT.json)，
逐格链重建 `CHAIN_RECONSTRUCTION.json`，阶段可归因性 `STAGE_CAPABILITY.json`，
输入冻结 `CLOSURE_INPUT_MANIFEST.json`，边界核验 `CLOSURE_VERIFICATION.json`。

## 〇、这一份是什么，凭什么信

* **只做一件事**：把已经存在的运行数据按同一个 `target_id` 串起来，看每个阶段还剩多少目标、
  各阶段之间花了多少步、目标死在哪一步。**0 新 episode、0 新流、0 `optimizer.step`、
  0 checkpoint 加载、0 启动、未改任何生产参数**（机械核实见 `CLOSURE_VERIFICATION.json`，
  `passed=true`：41 个只读输入 SHA256 不变、8 个日志族文件数与字节数不变、
  生产目录与起点 `git diff` 为空、账本 `starts` 恒 666、封存块盘上 artefact 0、
  五个环境入口 tripwire 调用日志为空）。
* **40 项与已发布结果的核对全部一致**（`reconciliation_summary`）。这张表是本文件的可信度来源：
  它把本轮自己算出来的数和归档值逐个对上——包括
  `SYSTEM_LOOP.json` 的 360/356/350/356、`expired_attribution.json` 的
  `never_assigned=18` 与 `completed/expired`、四层 `GATES_g*.json` 的
  `best_on_time_rate_of_detected`（0.927 08 / 0.937 5 / 0.9 / 0.781 72）、
  第十三轮登记的"g12 各臂从未报出的目标数"（`send_once` 8、`periodic_20` 43、
  `bound_only_3_10` 48、`periodic_5` 66、`event` 73）、
  以及第八轮统一后的 `+0.835 个/ep` 与 `+26 354.1 B/ep`。
  核对过程中它抓到过两次我自己的错（把 16 条流的比率之和当成了比率、把归档值抄成 26 354.0），
  都已在脚本里改正而不是改期望值。
* **口径纪律**：一个阶段"量不出来"时写 `not attributable`，**不折成 0**。
  把"部署路径没有 per-recipient 知晓记录"写成"部署路径信息损失为 0"是这份表最容易被伪造出来的一句话。

## 一、八个阶段各自能不能归因

| # | 阶段 | 日志里有没有 | 依据 |
|---:|---|---|---|
| 1 | 放置 | **没有记录**，只能取 `run_config.detail.configuration.num_goals` | 未被发现的目标不产生任何记录 ⇒ "未发现"的分母是配置数，不是日志 |
| 2 | 搜索发现 | 有：`detection` 事件 | `lifecycle_log.py:165`，带 `timestamp`/`discovered_by`/`grid_position`/`total_stages`/`deadline_offset` |
| 3 | 消息交付 | **只有开了 `--comm-transcript` 的运行有** | `comm.py:968` `send`、`:467` `deliver`；恒等式 `attempts = packets + packets_lost` 在抽查里逐条成立 |
| 4 | 机器人获知 | **没有自己的事件类型**，只能从 `deliver.accepted=true` 反推 | 生命周期事件全集恰为 6 种（`run_config/detection/progress/assignment/assignment_rejected/final`），脚本用"字面量不存在"反过来证明这条断言 |
| 5 | 分配 | 有：`assignment` / `assignment_rejected` + `progress status=en_route` | `lifecycle_log.py:257/290`；但 joint_eval 那批 `assignment` 记录 96.8 % 挂在 `__run__` 上（1 090/1 126），链只能走状态行 |
| 6 | 到场 | **代理量**：`progress status=assigned` | `task03.py:1328`；真实到场时刻只在 decision trace 的 `on_site`/`group_arrived_at` 里（19 316 个文件、42.4 GB，本轮**未读**） |
| 7 | 开始工作 | **代理量**：`progress status=in_progress` | `task03.py:1326` |
| 8 | 按时完成 / 过期 | 有：`final` + `progress`；"按时"是**离线算的** | `lifecycle_metrics.py:134` `done <= deadline`；`old_task_env.py:1134` 用严格 `>`，且 `hard_expire=False`（过期只是标志，工作继续） |

**代理量的真实代价，在本文件里看得见**：g12 `send_once` 的 `dispatched_to_working` 均值是
**−18.0 步**（n=46）。负号不是 bug，是"指纹不变就不写行"这条采样规则的结果——
`en_route` 可以在 `in_progress` 之后才被观测到。所以第 6/7 阶段的时间戳**只能当上下界用，不能当量数值用**。

## 二、正常在线路径（部署配置：视距刷新 + 一跳 gossip 都开着）

### P1 在线块，真实 PILOC 搜索，60 seeds × 2 臂（`coalition_validation_20260918/systemloop`）

| 臂 | 放置 | 发现 | **未发现** | 按时完成 | 过期 | 唯一可归因损失 |
|---|---:|---:|---:|---:|---:|---|
| `baseline` | 360 | 356 | 4 (1.11 %) | 350 | 6 | 分配/执行（6 个全部"到场并工作过但没做完"） |
| `candidate` | 360 | 356 | 4 (1.11 %) | 356 | 0 | **只有搜索那 4 个** |

* 阶段延迟（配对同 seed，中位/均值，单位步）：`detected→working` 2/8.2（baseline）→ 1/2.4（candidate）；
  `detected→completed` 21/24.2 → 17/17.4；`detected→deadline` 中位 **115.6**（两臂同一批目标）。
* 这一行独立复现了归档结论"**瓶颈是搜索覆盖率而不是分配**"
  （`corrected_env_final_20260918/CORRECTED_ENV_SYSTEM_VERDICT.md:39`，那一块是 360/355/355）。
  两块 60-seed 在线块都给出同一形状：**凡被发现的目标几乎都按时完成，损失集中在"根本没找到"**。

### P2 回放块，冻结检测流，30 seeds × 3 臂（`joint_eval_20260916`）

| 臂 | 放置 | 发现 | 未发现 | 按时 | 过期 | 从未派出 | 派出/工作但没做完 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nocoord` | 180 | 171 | 9 (5 %) | 136 | 35 | **18** | 17 |
| `rs0.25` | 180 | 171 | 9 | 152 | 19 | 0 | 19 |
| `rs0.5` | 180 | 171 | 9 | 149 | 22 | 0 | 22 |

* "从未派出 18"与"17 = 16 finished_after_deadline + 1 stopped_mid_work"与归档
  `expired_attribution.json` 的七类计数**逐类相等**（这是 `never_assigned` 与
  `finished_after_deadline+stopped_mid_work` 两个口径第一次被独立方法复现）。
* 信息侧在这一块**不可归因**：投递概率 1.0、延迟 0、且没有转录 ⇒ "检测到"与"知晓"被配置成同一件事。
* `detected→dispatched` 中位从 17 步（无协调器）降到 1 步（`rs0.25`）——
  协调器买到的是**派出延迟**，而这一块没有任何一步是通信造成的。

**两条部署路径共同的说法**：在归档的在线数据里，可归因的损失依次是
①搜索未发现（1.1 %–5 %），②分配/执行（0 %–20 %，随协调器配置大幅变化），
③信息未到达 = **量不出来**（既没有非平凡信道，也没有 per-recipient 记录）。

## 三、隔离通信路径（29 300 B 硬额度、`policy_owned_timing`、无损零延迟，576 格）

| 臂（四层各 16 条流） | 放置 | 发现 | 从未送达 | 送达但工作未完 | 按时 | 字节/ep（四层均值） |
|---|---:|---:|---:|---:|---:|---:|
| `always_hold` | 576 | 555 | **555** | 0 | 0 | 0 |
| `send_once` | 576 | 555 | 8 | 11 / 8 / 16 / 33 | 478 | 21 249.8 |
| `periodic_20` | 576 | 555 | 95 | 5 / 4 / 18 / 28 | 403 | 27 340.8 |
| `event` | 576 | 555 | 216 | 5 / 2 / 8 / 13 | 311 | 28 319.8 |
| `event_unbudgeted`（无额度天花板） | 576 | 555 | 0 | 0 / 1 / 4 / 1 | **549** | 171 278.2 |

* **一次报告都没收到的目标数（`send_once`）只出现在 12 目标层：8 个**，
  与第十三轮独立登记的数字一致。
* **额度造成的其余损失不是"没人说"，而是"只说了一次"**：`send_once` 每一层里
  "送达但工作未完"的那些目标，收到的发射数**恰好等于目标数**、刷新数**恰好为 0**
  （g6 11/11/0、g8 8/8/0、g10 16/16/0、g12 33/33/0）。
* **信道上没有任何时机可赚**：全部 576 格里 `detected→首次发射` 与 `首次发射→首次被知晓`
  的中位与均值都是 **0 步**（零延迟、投递概率 1.0，`deliver` 行只出现在包真的到达时）。
  ⇒ 隔离设置下"何时说"的全部自由度就是"说不说"，这正是
  `no_temporal_allocation_signal` 的机制面表述。
* 配对天花板差（同 16 条流、同 seed、信道确定 ⇒ 精确配对）：
  `event_unbudgeted − send_once` 完成 **−12 / −7 / −12 / −40**（按 g6/g8/g10/g12，
  逐流：0/9/7、0/12/4、1/9/6、0/4/12 分别 better/equal/worse），
  `send_once` 一侧省下 1 574 160 / 2 054 676 / 2 812 716 / 3 160 272 B。
  ⇒ g12 上"把额度抬到不 binding"要多花 **197 517 B/ep 换 2.5 个/ep = 79 007 B/个**，
  是第七轮 ACK 路线 break-even（31 562 B/个）的 **2.5 倍**。
  **注意混淆项**：天花板臂同时改了额度与刷新策略，所以这 40 个不能全部记到"缺刷新"头上；
  而"在同一额度里多花钱买刷新"（`periodic_20`、`event`）实测**更差**（403、311 < 478）。

## 四、同一份发射账本，只切换环境自带传感：这才是信息侧价值最干净的读法

`comm_rule_confirm_20260921` 的 200 条全新流 × 5 臂 × 3 设置里，
有三条臂在 `normal-lossy` 与 `isolated-lossy` 之间的**前向信道账本逐字段相同**
（`transmission_attempts` / `packets_delivered` / `packets_lost` / `emissions` / `first_emissions` 全部相等）
⇒ 它们之间的差**不是**"谁发得多"，而是"同样的丢包落在有没有自带传感的系统里"。

| 臂 | 尝试/ep | 到达/ep | 丢失/ep | **正常路径 on-time/ep** | **隔离路径 on-time/ep** | 差 | 过期/ep 正常 vs 隔离 |
|---|---:|---:|---:|---:|---:|---:|---|
| `noack_send_once` | 35.16 | 24.605 | 10.555 | **5.810 0** | **4.480 0** | **−1.330 0** | 0.045 vs 1.370 |
| `ack_send_once` | 35.16 | 24.605 | 10.555 | 5.810 0 | 4.480 0 | −1.330 0 | 0.045 vs 1.370 |
| `ack_retry_2` | 65.82 | 46.315 | 19.505 | 5.810 0 | 5.315 0 | −0.495 0 | 0.045 vs 0.540 |

* **同一批丢掉的包（10.555 个/ep），在部署配置下几乎不掉任务（过期 0.045/ep），
  在隔离配置下掉 1.33 个/ep。** 这是"环境自带传感替上行 absorb 了多少"的最大、也最干净的一次量化。
* 与第十三轮的信息源归属相互印证（本轮从 `TRAFFIC_PROBE.json` 48 行原始数据重算）：
  部署 + 正常上行 首次知晓 **408/408 来自上行**；部署 + 静默上行 **390/408 = 95.6 %** 仍然发生
  （视距刷新 388 + 一跳 gossip 2）；隔离 + 静默上行 **0 次 belief 写、0 次首次知晓**。
* 信道自身的代价（同一隔离设置，无损 → 投递 0.7）：`noack_send_once` **−1.035 0 个/ep**。
* 转录抽查（每设置 20 格、共 60 格）：20/20 逐格自洽，实测丢失份额
  `isolated-lossless` 0.0、`isolated-lossy` 0.257 5、`normal-lossy` 0.251 5 ⇒ 归档账本不是笔糊涂账。

## 五、结论：当前损失最大的环节，以及它随配置怎么变

1. **部署（在线）配置下，可归因的最大损失在"分配/执行"与"搜索未发现"两侧，信息侧量不出来。**
   `candidate` 臂上唯一损失是搜索的 4/360；无协调器时执行侧吃掉 35/171。
   证据：P1 + P2（含与两份归档件的逐类核对）。
2. **隔离配置下，信息侧才是主损失**：`always_hold` 555/555、`event` 216、`periodic_20` 95、
   `send_once` 8；另有 68 个目标"只被告知一次然后没做完"。
   证据：P3 的 576 格链，与两个已发布数字对上。
3. **同一个信道损失在两种配置下价格差 30 倍**（1.33 个/ep vs 0.045 个/ep 过期）：
   这不是通信策略的功劳或过失，是**架构把上行旁路掉了**。
   证据：P4 的三条"发射账本逐字段相同"的臂。
4. **搜索未发现是唯一在所有配置里都稳定存在的损失**（1.1 % / 5 % / 3.65 %）。
   任何"少通信但任务不掉"的主张都必须先承认：它掉不了的那部分，本来就不是通信负责的。

## 六、配对有效性（哪些差可以算，哪些不能）

| 差 | 分母 | 是否配对 | 理由 |
|---|---|---|---|
| P1 baseline vs candidate | 60 seeds | **是** | 同 seed 同场景，信道无损零延迟 ⇒ 不引入随机性 |
| P2 三臂之间 | 30 seeds | **是** | 同 seed + 同一份冻结检测流 |
| P3 九臂之间 | 16 条流 | **是（精确）** | 同 stream seed，信道确定性 |
| P4 normal-lossy vs isolated-lossy | 200 seeds | **仅三条臂是** | 前向信道账本逐字段相同；另外两条 ACK 臂的发射序列随设置改变 ⇒ 只同分布，不共随机数 |
| P4 同一设置内跨臂 | 200 seeds | **只同分布** | 前向丢包是逐 episode 序列流，不同策略消耗不同随机数 |
| 任何跨 P1..P4 的比 | — | **否** | 场景、额度、信息路径都不同，只能并列陈述 |

## 七、还缺哪些可归因日志（因此才有第八节那份协议）

1. **per-(目标, 机器人) 的知晓时刻与 belief age**：没有它，"信息未到达"与"信息过期"
   在部署路径上永远合不成一个可审计的量。
2. **per-recipient 的丢包记录**：现在只有每条发射一个整数计数。
3. **真实的到场时刻**：只在未读的 42.4 GB decision trace 里；生命周期侧只有代理状态行，
   连"先到场还是先开工"都能颠倒。
4. **"放置"没有记录**：未发现的 target id 不出现在任何日志中，因此
   阶段 1→2 的流失只能对着配置数说，不能对着日志说。
5. **没有任何一个归档运行同时具备**：真实在线搜索 + 非平凡信道 + 逐目标执行归因。
   P1/P2 无损，P4 回放检测，P3 关掉自带传感 ⇒ 这条链在现有数据里是**断成三段的**。

## 八、下一步（决定权在使用者，本轮不执行）

现有数据**足以**给出上面第 1–4 条结论，但**不足以**把"信息未到达"在部署路径上变成一个
可审计的损失量——而那个量恰好是判定通信学习路线在真实系统里有没有位置的前提
（收束件 §三 的 N6）。因此本轮按指令只交付一份**最小配对验证协议**：
[`MINIMAL_PAIRED_VALIDATION_PROTOCOL.md`](MINIMAL_PAIRED_VALIDATION_PROTOCOL.md)。
**协议未执行**：不跑 episode、不加日志之外的代码、不改默认值。

## 九、本文不声称的事

* 每个桶都是**描述性**的，不是干预收益；凡"量不出来"的地方写的是 `not attributable`，不是 0。
* 不声称"部署路径信息损失为 0"，只声称它在现有日志下不可归因。
* 不声称任何学习型控制器的收益或无效；也不据本文件提议修改生产默认策略。
* 不把 `send_once` 与无额度天花板之间的差解释成"刷新的因果效应"（混淆项已在 §三标明）。
* 不重跑、不改写任何一轮的原判定；本轮只读，写入仅限本目录（沙箱 + 一条测试证明）。

## 十、产物与复跑

```text
closure_common.py      边界：tripwire、写盘沙箱、流式读取、分位数
closure_inputs.py      CLOSURE_INPUT_MANIFEST.json     冻结 41 个只读输入 + 8 个日志族 + 治理文档起始态
closure_stages.py      STAGE_CAPABILITY.json           八阶段可归因性：源码锚点 + 逐件探针
closure_chain.py       CHAIN_RECONSTRUCTION.json       逐目标八阶段链（P1 120、P2 90、P3 576 格）
closure_bottleneck.py  JOINT_SYSTEM_BOTTLENECK_AUDIT.json  分桶、排序、40 项核对、配对分级
closure_verify.py      CLOSURE_VERIFICATION.json       边界机械核验（passed=true）
test_closure_harness.py 21 项，0 episode
```

复跑顺序即上表顺序；`closure_chain.py` 读 430 MB 逐集日志，其余为秒级。
