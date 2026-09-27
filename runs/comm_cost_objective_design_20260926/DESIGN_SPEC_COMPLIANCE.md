# 与原 spec 的逐项差异（**挂起待补**：两份原件未获得）

## 一、状态：本文件目前**不能**给出与原 spec 的逐项差异

要求的文档提交 `72c4c3f71e07339bf6533fbd771aef3b2682103c`
（含 `ROUND19_COMM_MAINLINE_RETURN.md`、`ROUND19_CONSTRAINED_COMM_CONTROLLER_DESIGN.md`）
**未获得**，因此本文件**只登记两件事**：
① 拿不到原件的实测原因与请求方式；② 依据**使用者 2026-09-26 指令**（它复述了 spec 的要求）
可以当场核对的条目。**不得**把本文件读成"已符合原 spec"的证明；原件到达后本文件按
"旧件不改、修订另写独立补充件"的纪律**另出一份差异表**（或在本文件追加 §四，旧段落保留）。

### 拿不到原件的实测证据

`github.com:443` 在 TCP 层不可达：`git fetch` 2 次、`git ls-remote` 5 次（含 HTTP/1.1 变体）全部失败，
`git push` 0 次成功；同一时刻 `pypi.org` 返回 200、`api.github.com` 可达但仓库私有 ⇒ 未认证 404；
DNS 解析正常（`20.205.243.166`）、系统未配代理、IP:443 直连失败。
逐条记录见 [`REMOTE_RETRY_LOG.md`](REMOTE_RETRY_LOG.md)。**成功的远端接触数 = 0**。

### 请求：请用以下任一方式把两份文件交给我

1. 直接把两份 `.md` 的**内容粘贴**到对话里（最简单）；或
2. 把文件放到本机任意路径并告诉我路径（例如 `D:\code\整合\zhenghe\_inbox\`），我读本地文件；或
3. 在能连 GitHub 的机器上执行
   `git bundle create plan.bundle refs/heads/docs/comm-cost-objective-plan`，
   把 `plan.bundle` 交回本机 ⇒ 我 `git fetch plan.bundle` 后可核对**完整 SHA**、
   验祖先关系，再按原指令 cherry-pick（这条最符合"核对完整 SHA + 先验祖先关系"的要求）。

在原件到达之前，我**不会**再凭摘要推定 spec 的内容，也**不会**把本轮修订说成"符合 spec"。

## 二、可以当场核对的条目（依据 = 使用者 2026-09-26 指令，非原 spec）

| 指令条目 | 本轮修订的落点 | 状态 |
|---|---|---|
| 一、获取并完整读取两份文档 | 未获得；已登记实测原因并给出三种传送方式 | **未完成（阻塞）** |
| 一、保留当前本地成果，不回退、不重写已有提交 | 旧件一字未改；`2afc0f6`/`8a6c63b`/`2174760` 三枚提交保留；修订**独立提交** | 完成 |
| 二.1 区分整策略差异与单消息因果 | `COMM_COST_OBJECTIVE_DESIGN_REVISED.md` §二.1 | 完成 |
| 二.2 "承重/严格冗余"改为描述性标签 | 同件 §二.2（三个新标签 + 旧名停用 + 机读件 `label_semantics`） | 完成 |
| 二.3 核对 352 的分桶整数、百分比分母、字节汇总 | 同件 §二.3 + `BUCKET_ARITHMETIC_RECHECK.json`（重算与旧件逐值相同、四项闭合全真、并查出**字节份额被写在目标数旁**的引用错误） | 完成 |
| 二.4 说明 AUC 的标签/特征/划分/拟合与选阈值过程；无独立验证则标为开发集探索性 | 同件 §二.4（十项出处 + "无划分、无拟合、阈值与标签同批" + 明确标 **开发集探索性**） | 完成 |
| 二.5 `late_only` 定义不得混用；阈值由本批选出则本批不得作独立确认 | 同件 §二.5（(a) 晚发现才发送 vs (b) 延迟已有信息；本轮算的是 (a)；改名 `late_discovery_gate(ĉ)`；污染登记） | 完成 |
| 二.6 费用称"当前模型的载荷尝试成本"；未计量项单列 | 同件 §二.6（含 7 类未计量项与"不得称为真实空口/全系统成本"） | 完成 |
| 三、恢复 N4 主线：最小化期望已计量成本，约束相对固定参考策略的按时损失 ≤ δ | 同件 §三.1（形式化）、§三.3（对偶变量实现） | 完成 |
| 三、预算是硬约束不替代成本目标 | 同件 §三.2 | 完成 |
| 三、说明成本如何进入优化 | §三.3（精确归因 + cost critic + λ 更新） | 完成 |
| 三、说明任务约束如何估计与更新 | §三.4（参考策略固定、配对估计、λ 更新式、判定用 CI） | 完成 |
| 三、actor/critic、mask、实际计费如何衔接 | §三.3 表 + §三.5 对照表（计费仍单点在 `comm.py:953/974`；mask 沿用顺序预算 mask） | 完成 |
| 三、不新增非法观测、不改搜索/分配权重 | §三.3（critic 也不加价格维度 ⇒ **无 schema 变更**）、§三.6 清单 | 完成 |
| 三、与旧"只奖励完成数"训练的区别 | §三.5 六行对照表 | 完成 |
| 三、预算扫描只能作备选并解释外层机制 | §三.8（外层选档机制 + 四条缺点 + 回退需再批准） | 完成 |
| 三、删除"≥3 个 binding 内点"硬前置 | §三.7（删除，并给出本实现特有的理由：成本水平是训练输出而非输入） | 完成 |
| 四、拆成 A/B/C 三阶段而不是一次申请 1 188 次 | `COMM_PILOT_STAGED_APPROVAL.md`（A = 0 启动；B ≤ 24；C ≤ 488，硬上限 520） | 完成 |
| 四、每阶段确切资源上限、继续条件、停止条件；不按中间显著性自动扩样 | 同件三张上限表 + 继续/停止条件 + C 阶段"不按中间显著性自动扩样"明条 | 完成 |
| 四、δ=0.10、降本门槛、训练量保持"建议，待批准" | 同件抬头与三处 **[建议，待批准]** 标注 | 完成 |
| 四、`late_only` 只是新候选对照，不代表学习可行性已建立 | 同件抬头 + A 阶段"本阶段不实现（未获批准）" + C 阶段"只在单独批准后加入" | 完成 |
| 五、四份交付件 | `DESIGN_SPEC_COMPLIANCE.md`（本件，**挂起待补**）、`COMM_COST_OBJECTIVE_DESIGN_REVISED.md`、`COMM_PILOT_STAGED_APPROVAL.md`、`DECISION_SUMMARY.md` | 3 完成 + 1 挂起 |
| 五、旧结果保留、修订独立提交 | 旧件未改；修订在**新提交**里 | 完成 |
| 五、网络恢复后推送并核验，否则报告本地完整 SHA | 本地完整 SHA 见 §三；推送仍不可达 | 部分完成（推送阻塞） |
| 五、完成即停止，不自动实现或训练 | 阶段 A/B/C **一个都未执行**；0 启动、0 训练、0 生产代码改动 | 完成 |

## 三、本地完整 SHA（推送不可达时按要求报告）

```text
分支   docs/comm-cost-objective-design
起点   4c71fead0047298101869200a8518059afc42ccc（第十八轮头，本地逐字符核对）
提交链 2afc0f6 → 8a6c63b → 2174760 → 53b1527（本修订，完整 SHA
       53b1527b0195504e8334212c1fa128cbae4e02b7）
远端   未核验；最后实际核验过的远端 ref 仍是
       refs/remotes/origin/analysis/real-search-undetected-attribution = 80af16ba4b90ce0c2d5456dd8b8e57068b4513e4
```

## 四、逐项差异（**原件已于 2026-09-27 离线取得并核对后填写**）

原件获取方式：用户传送 `D:\code\整合\specs\comm-cost-original-specs-72c4c3f.bundle`
（SHA256 `016c41f5c2b719e8192ff57b3a08ef1cf9c26cb4ffd58635a50ca8e467fe6714`，与指令给的哈希一致）。
`git bundle verify` 通过、前置 `80af16b…` 本地存在、
`git fetch <bundle>` 后 `git rev-parse FETCH_HEAD` = `72c4c3f71e07339bf6533fbd771aef3b2682103c`
（与要求的完整 SHA 逐字符相同），父提交即第十七轮头、只新增两份文件。
在新建分支 `feat/comm-cost-stage-a`（起点 `efca2a5…`，即第十九轮修订头）上 cherry-pick 为
`c539711`，两文件内容与远端逐字节相同（sha256 已登记）。
原 spec 的**有效要求只读取、不重做**已完成的归档与分析；下文"差异"均指本端交付物与它的偏差。

### 4.1 遵守项

| 原 spec 条目 | 本端落点 | 判定 |
|---|---|---|
| 总 spec §0：先核验 4c71fea、从该头建分支、只挑一个文档提交、不回退不覆盖 | 起点核验 + `c539711` 只导入两份文档，工作区 clean | 遵守 |
| 总 spec §1 授权：只读分析 + 合成测试，0 仿真/0 新流/0 训练/0 optimizer.step/0 权重/0 生产默认值 | Stage A 边界全部实测为 0，生产树 diff 为空 | 遵守 |
| 总 spec §2：读 AGENTS/总纲/状态/架构 + 第十八轮协议/结果/manifest/账本 + 通信收束件 | 第十九轮已读；本轮补读两份原 spec 与 Stage A 指令 | 遵守 |
| 总 spec §3：整数闭合 placed→detected→on_time→expired，查清差出的每个目标 | `SPEC_ITEM_VERIFICATION.json`：A 351/9/350/1、B 291/69/290/0，差出的 2 个目标逐一点名 | 遵守 |
| 总 spec §3：35/20/5 由逐 seed 数据验证，禁止猜顺序 | 直方图 `{−4:3, −3:4, −2:15, −1:13, 0:20, +1:3, +2:2}` ⇒ 35/20/5，`matches_round18_labels=true` | 遵守 |
| 总 spec §3：诊断集与新 60-seed 比较不混池；原 S1 判定原样保留 | 第十九轮文档与机读件分列；本轮未改任何旧判定 | 遵守 |
| 总 spec §4：成本术语 `C_uplink`/`C_ack`/`C_measured`/unmetered | `SPEC_ITEM_VERIFICATION.json:cost_terms_from_archive` 按此口径落值 | 遵守 |
| 设计 spec §1：成本是**已计量**成本，与"固定字节预算下只最大化按时"不同 | 第十九轮修订件 §二.6 即此口径；Stage A 的 `c` 只用已计量尝试字节 | 遵守 |
| 设计 spec §2：`minimize E[C]` s.t. `E[Y−Y_ref] ≥ −δ`；B0 预锁定；不用每臂成功数作分母 | Stage A `cost_math`：`B0` 固定常数 + 两条符号测试 | 遵守 |
| 设计 spec §2：参考策略从公平基线表固定，不从测试结果挑 | 建议参考 = `immediate`（生产默认），非按结果挑 | 遵守 |
| 设计 spec §3：SEND/HOLD、复用顺序 shadow-budget mask、真实报价、执行器顺序 | Stage A 只做纯函数版本并明确"不得成为第二套价格定义"；接线留给 B | 遵守 |
| 设计 spec §3：不可执行项 likelihood/entropy 权重为零、critic 样本保留 | `masked_log_likelihood` + 三个测试 | 遵守 |
| 设计 spec §3：不加 ACK、不改信道/压缩/接收者/共享上界 | Stage A 零生产改动，未触碰 | 遵守 |
| 设计 spec §4：只选**一个**优化器方案，不同时试多种 | 选拉格朗日/原初—对偶；固定成本权重列为备选且需先说明单位与标定 | 遵守 |
| 设计 spec §4：说明乘子更新/范围/频率/初值/稀疏奖励归因 | `dual_step` + `η_λ`/`λ_max` 待批准；λ 轨迹逐批次记录 | 遵守 |
| 设计 spec §6：正确性测试七项（计费一致、符号、约束统计同源、全 HOLD 不补偿、mask/采样可重算、默认行为不变、种子与 RNG 可核验） | 前五项已在合成层覆盖（见 §4.3）；后两项属 Stage B 真实接线 | 部分遵守（B 阶段） |
| 设计 spec §7：不重开"通信 vs 静默"、不人为提高失败率、不加订阅者常数、不要求训练前证明必胜 | Stage A 与 B/C 计划均未触碰这些 | 遵守 |
| Stage A 指令 §1：只做 Stage A，不启动环境/episode/封存/真实 checkpoint/真实 rollout/optimizer.step/pilot/B/C，不改生产路径或默认值 | 全部实测为 0（`STAGE_A_TEST_RESULTS.json`） | 遵守 |
| Stage A 指令 §2：验 bundle → 导入对象 → 核对完整提交 → 从修订头建 `feat/comm-cost-stage-a` → cherry-pick 两文件提交；文件已存在则逐字比较，不覆盖不明差异 | 按序执行，SHA 与字节双重核对 | 遵守 |
| Stage A 指令 §4：合成测试清单逐条覆盖，报告实测数量，不跑全量 smoke | 27 项全绿；全量 smoke 列为未授权 | 遵守 |
| Stage A 指令 §6：manifest 明确 0 启动/0 流/0 step/零 diff/无 checkpoint/未读封存，用可复查命令证明 | 见 §4.4 的 manifest 条目 | 遵守 |

### 4.2 偏差与缺口（据实登记）

| # | 偏差 | 性质 | 处置 |
|---|---|---|---|
| 1 | 总 spec §3 要求的 `ROUND18_RECONCILIATION_ADDENDUM.md/.json` 未按该**文件名**交付；等价内容在第十九轮的 `ROUND18_ARCHIVE_CLOSURE.md/.json` | 命名偏差 | 已在第十九轮交付物中覆盖；本轮 `SPEC_ITEM_VERIFICATION.json` 补齐其中三项未核对项 |
| 2 | 总 spec §4 要求的 `DEPLOYED_COMM_BASELINE_INVENTORY.md/.json` 未按该文件名交付；等价内容在 `COMM_COST_COVERAGE.md/.json`（含四臂配置身份/数据源/seed 集合/整数与字节口径） | 命名偏差 | 本轮 `cost_terms_from_archive` 按 `C_uplink`/`C_ack`/`C_measured`/unmetered 补齐术语口径；基线仍是"已归档读数搬运，未新跑矩阵" |
| 3 | 总 spec §5 要求的 `COMM_COST_OBJECTIVE_DESIGN.md` / `COMM_COST_PILOT_DRAFT.md` / `COMM_ROUND19_DECISION.md` 命名与本端 `COMM_COST_OBJECTIVE_DESIGN_REVISED.md` / `COMM_PILOT_STAGED_APPROVAL.md` / `DECISION_SUMMARY.md` 不同 | 命名偏差 | 三份内容等价并已被第十九轮修订件取代；Stage A 指令 §5 的命名优先级更高 |
| 4 | 第十九轮"启动上限 488/520"与"3 seed × 400k 步"不可同时闭合 | **账本缺口** | 已在 `STAGE_BC_RESOURCE_AND_STATISTICS.md` 给出两个可闭合选项（推荐 672+28=700），并明确不缩短 horizon 修补账本 |
| 5 | 第十九轮的"δ=0.10 + n=120 可判定"论证不充分（正态近似、无真实配对分布与功效假设） | **统计论证缺口** | 同件 §五修订判定口径为"配对差分布 + 配对 bootstrap CI + 事前功效模拟；不足则标探索性，不给确定性验收承诺" |
| 6 | `late_discovery_gate` 的实现未获批准；原 spec 的"规则参数允许在同一开发/验证范围内公平选择"与它"由本批开发数据选出 ⇒ 本批不得再作独立确认"并存 | 待批准 | 保持未实现；ĉ 若将来选定，必须在跑新 seed 段前冻结 |
| 7 | 正确性测试第 6/7 项（默认行为不变、种子与 RNG 可核验）需要真实接线 | Stage B 范围 | 已在 B 阶段继续条件中列明，Stage A 只做合成层 |

### 4.3 与两份原 spec 无冲突的关键点

原 spec §2 的目标形式（`minimize E[C_measured]` s.t. `E[Y_ref − Y_policy] ≥ −δ`）与
本端第十九轮修订件 §三.1 **形式一致**；本端把 `C_measured` 明确拆成
`C_uplink + C_ack`（ACK 未开启是结构性零、缺日志不是零），这比原 spec 的表述更严，不构成冲突。
原 spec §3.2 的"若设计认为必须新增特征，逐字段列合法来源并作为独立待批准变更"——
本端 Stage A **未新增任何 actor 特征**（19 维不变，`cost_math.build_actor_features` 用白名单
结构性拒绝结局派生字段），因此不触发该条款。

### 4.4 原件与哈希登记

```text
bundle          D:\code\整合\specs\comm-cost-original-specs-72c4c3f.bundle
bundle sha256   016c41f5c2b719e8192ff57b3a08ef1cf9c26cb4ffd58635a50ca8e467fe6714
文档提交        72c4c3f71e07339bf6533fbd771aef3b2682103c（父 80af16ba…，仅 +2 文件）
本端导入提交   c539711（cherry-pick 到 feat/comm-cost-stage-a@efca2a5 之上）
两份 spec sha256（本端工作区）
  ROUND19_COMM_MAINLINE_RETURN.md              112391a3333be23ec729bb5f0df934dd0dc5b971aa5f5c3f9c897205c6fc94cd
  ROUND19_CONSTRAINED_COMM_CONTROLLER_DESIGN.md 82d952d6e0b5bcb80bb96d80fb1ba801c782788d6d3111e367176e6fae452bfd
```

**在 §4 补完之后，第十九轮修订件的推荐方案不再是"未经原 spec 核对"**：
它现在有原件依据；但 δ、降本门槛、训练量与规则臂实现**仍是建议、待批准**，
本端**不**把 Stage A 通过当作学习可行性已建立的证据。

## 五、本地完整 SHA（推送不可达时按要求报告）

```text
分支   feat/comm-cost-stage-a
起点   efca2a50c8d06f6fa7062c204d94c98b137206a4（第十九轮修订头，本地逐字符核对）
提交链 2afc0f6 → 8a6c63b → 2174760 → 53b1527 → efca2a5 → c539711（导入原 spec）→ <Stage A 提交>
远端   未核验；本会话成功的远端接触为 0（见 REMOTE_RETRY_LOG.md 的逐条记录）
```

## 六、本轮到此停止

Stage A 已通过并提交；B/C 未启动，也未自动进入。

