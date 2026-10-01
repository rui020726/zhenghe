# 交付目录骨架（任务 3d）

状态：骨架 + 机器登记件。`ARTEFACT_HASH_REGISTRY.json` 由 `r36_delivery.py` 在三次启动全部结束后
生成，本文件解释它的结构与"为什么 Git 里看不到原件"。

## 1. 本轮交付目录

```
runs/comm_final_train_20261001/
├── EXECUTION_BASIS.md                 提示词原文 + 开工核验 + 归档基准 + 额度表
├── FROZEN_CONFIG_MANIFEST.json        任务 1：与 ROUND32 冻结 Stage C 的逐项比对结果（含两个版本字段）
├── ROUND36_PREFLIGHT.json             G1-G12 启动前门禁
├── ROUND36_VERIFICATION.json          V1-V5 训练后复核（版本字段/账本/批 0 逐字节/训练量/旧件未改写）
├── LEDGER_RECONCILIATION.json         任务 4：三 seed 账本对账表（全部由落盘件重算）
├── TRAIN_LAUNCH_BOOK.json             启动账本（进程数、构造数、优化器步数、失败登记、储备使用）
├── ROUND36_TRAIN_REPORT.md            交付报告（go/no-go）
├── r36_common.py                      路径重绑定、额度守卫、账本、归档基准、代码身份
├── r36_preflight.py / _worker.py / _launch.py / _sanity.py / _verify.py / _reconcile.py / _delivery.py
├── delivery_draft/
│   ├── ARCHITECTURE_DRAFT.md          3a 事件顺序与数据流
│   ├── REPRODUCTION_DRAFT.md          3b 环境/依赖/命令/配置路径/seed
│   ├── RESULTS_TABLE_TEMPLATE.md      3c 结果表模板（留空待第 4 天）
│   ├── DELIVERY_SKELETON.md           本文件
│   └── ARTEFACT_HASH_REGISTRY.json    代码身份 + 模型清单 + Git 外原件哈希表
└── seed{11,22,33}/                    每 seed 一个产物子目录
    ├── FREEZE_MANIFEST.json           该次启动的两个版本字段 + 继承的 freeze token + 当时代码身份
    ├── LAUNCH_RECORD.json             退出码、wall-clock、上限、ledger 摘要、console 路径
    ├── TRAINING_RECORD.json           真实 train_seed 的返回（集数/更新数/步数/λ/参数哈希/历史）
    ├── SANITY.json                    版本字段齐全 + 真实恢复 + 一次前向 + 训练量核对
    ├── CHECKPOINT_HASHES.json         四个 .pt 的字节 SHA256 与状态字段 + 96 份批记录哈希
    ├── LAMBDA_TRAJECTORY.json         批级 λ/g_hat/样本数/步数轨迹
    ├── logs/    （Git 外）lifecycle `*_rp.jsonl`、transcript `*_ct.jsonl`、tape `*_tape.jsonl`、
    │            START_LEDGER_p<pid>.json、launch_console.log 在上一层同名目录内
    ├── rows/    （Git 外）每集一行 JSON，含逐事件账本 `task_reward_events`
    └── checkpoints/（Git 外）`v2_train_s<seed>.pt`（rolling）+ `_ep0/_ep64/_ep128.pt`（不可变里程碑）
                     + `v2_batch_s<seed>_b0..b31.json`（批记录）
```

`.gitignore` 为本轮开了白名单：`*.py` / `*.md` / `*.json`（含 `seed*/*.json` 与
`delivery_draft/*.{md,json}`）进版本库；`seed*/{logs,rows,checkpoints,rejected}/` 与
`launch_console.log` 不进。

## 2. 代码身份（可复核，不靠描述）

| 块 | 内容 | 出处 |
|---|---|---|
| `code_identity.production` | 8 个 `integrated_sar/` 生产件哈希 | 与 ROUND32 manifest 的 `code_identity.production` 逐字节相同（G6） |
| `code_identity.imported_training_modules` | Stage B 采集器/控制器、pilot、mask-fix、tight-round、round18 共 13 件 | 与 ROUND32 manifest 相同（G6） |
| `code_identity.stage_c_harness` | 10 个 `v2_*.py`（含 ROUND34 新增的 `v2_reward_timing.py`） | 本轮 import 的真实训练路径 |
| `code_identity.round36_scripts` | 本轮 8 个 `r36_*.py` | 交付件自身 |
| `code_identity.frozen_checkpoints` | PILOC 与 CTDE 两个 `.pth` | 与 ROUND32 manifest 相同 ⇒ 搜索/分配冻结现状未动 |
| `freeze_identity` | ROUND24 manifest 的四字段 token（继承件，附 provenance 说明） | checkpoint payload 里的 `freeze` 必须与它逐字段相等才能恢复 |
| 两个版本字段 | `constrained-cost-v2/published-completion-event-at-relay-tick/2`、`published-completion-event-at-relay-tick/2` | 每个 `.pt` 的 payload 与 `training_state` 双写，每行 row 也写 |

## 3. 模型清单（登记在 `ARTEFACT_HASH_REGISTRY.json` 的 `model_inventory`）

每个 seed 四个 `.pt`（ep0 / ep64 / ep128 / rolling），逐件登记：

| 字段 | 含义 |
|---|---|
| `bytes_sha256` | 文件字节哈希（原件在 Git 外，靠它校验） |
| `training_version` / `reward_timing_version` | 两个版本字段（payload 与 `training_state` 都要齐） |
| `freeze_token_matches_this_round` | payload 的 `freeze` 是否等于本轮使用的 token |
| `episodes_done` / `updates` / `optimizer_steps` / `lambda` | 该点的训练状态 |
| `parameter_sha256` | whole-controller 参数内容摘要（只读恢复后重算） |
| `module_digests` | trunk / action_head / task_value_head / cost_head 四模块 digest |
| `optimizer_restored` / `rng_restored` | 恒为 `false`：登记过程不建 optimizer、不动 RNG |

评估（第 4 天）只认两个时间点：**登记初值 ep0** 与 **ep128 不可变里程碑**；rolling 不是评估对象。

## 4. Git 外原件登记表结构

`raw_artefacts_kept_outside_git.seed<NN>`：

| 键 | 内容 |
|---|---|
| `files_with_sha256` | 该 seed 目录下每一个 `rows/*.json`、`logs/*.jsonl`、`checkpoints/*.pt`、`checkpoints/v2_batch_*.json` 的相对路径 → SHA256 |
| `bytes` / `total_bytes` / `file_count` | 体量与件数 |
| `episode_start_ledgers` | 每 PID 账本：`starts`（含失败照计）、`results` 条数、账本自身哈希 |

审阅者据此可做三件事：确认某个 row/tape 未被改动；确认本轮没写过任何旧轮的字节；确认 Git 里
的结论与 Git 外的原件是同一批数据。

## 5. 与旧轮的关系（谁是谁的基线）

| 轮 | 本轮如何使用 | 明确不做什么 |
|---|---|---|
| ROUND24 | 只读参照缓存（64 份 immediate reference）；训练代码所在目录 | 不写它的目录；不加载它的 `.pt` 作热启动 |
| ROUND32 | 归档训练量与逐集耗时（wall-clock 基准）；批 0 逐字节被动性的对照对象 | 不改写它的行/批记录/checkpoint；不把它的 legacy 采集结论外推到本轮 |
| ROUND33/34 | 时序契约与版本门定义；两套回归在真实 torch 下重跑（G12） | 不改写 ROUND34 的"待复算声明"结论 |
| ROUND35 | 残余风险 1-4 转写为本轮训练协议；旧件哈希对账基线 | 不把 smoke 数字当学习效果；不重跑它的额度 |

## 6. 交付状态

| 交付物 | 状态 |
|---|---|
| `ROUND36_TRAIN_REPORT.md` | 待三次启动与复核完成后写 |
| `FROZEN_CONFIG_MANIFEST.json` | 已生成（G4-G7 全清） |
| `LEDGER_RECONCILIATION.json` | 待填（脚本已就位） |
| `delivery_draft/` 3a/3b/3c/3d | 草稿完成；`ARTEFACT_HASH_REGISTRY.json` 待启动结束后生成 |
| 本地提交（按主题）与最终头哈希 | 报告中给出，不推送远端 |
