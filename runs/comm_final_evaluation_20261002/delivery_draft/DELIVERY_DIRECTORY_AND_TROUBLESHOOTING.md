# 交付目录与故障排查

状态：交付文档（ROUND37 §7 第 6 项）。运行手册里的命令与本页的目录结构配套使用。

## 1. 本轮交付目录

```
runs/comm_final_evaluation_20261002/
├── EXECUTION_BASIS.md                      授权边界原文 + 开工核验（0 启动）
├── SCENE_USAGE_AUDIT.json                  全项目场景占用审计 + 选块依据（不看结局）
├── EVALUATION_IDENTITY.json                起点身份、历史通过项引用、环境、模型身份登记
├── FROZEN_EVALUATION_PLAN.json             九臂 / 64 场景 / 配置 / 统计方案 / 额度（先冻结）
├── ROUND37_GATE.json                       首场景九臂技术门禁（T1-T8，计入 576）
├── EVALUATION_LAUNCH_BOOK.json             本轮自记启动账本（失败照计）
├── EVALUATION_LEDGER_RECONCILIATION.json   claim / 行 / 账本 / 真实 ledger 四方闭合
├── ROUND37_STATISTICS.json                 唯一统计结果件（所有表与文字引用它）
├── RESULTS_TABLES.md                       由统计件机械渲染的表格
├── ROUND37_EVALUATION_REPORT.md            最终报告与五项明确区分
├── r37_common.py                           冻结对象、额度、原子 claim、身份、统计常量
├── r37_scene_audit.py / _freeze.py / _run.py / _gate.py / _stats.py / _tables.py
├── r37_reconcile.py / _delivery.py         对账与登记
├── delivery_draft/                         §7 六项交付文档 + Git 外原件哈希登记表
├── rows/        （Git 外）576 份评估行，含逐事件账本与 checkpoint 装载身份
├── logs/        （Git 外）每格 3 份 jsonl（tape / lifecycle / transcript）+ 每 PID 启动账本
├── claims/      （Git 外）576 份原子领取记录（含 pid 与时刻）
├── shards/      （Git 外）四个分片的运行摘要与控制台日志
└── checkpoints/ （Git 外）本轮不训练 ⇒ 无新权重；目录保留为空
```

`.gitignore` 为本轮开白名单：脚本、`*.md`、`*.json`（含 `delivery_draft/`）进版本库；
`rows/`、`logs/`、`claims/`、`shards/`、`checkpoints/` 不进，但**逐件路径与 SHA256**登记在
`delivery_draft/ARTEFACT_HASH_REGISTRY.json`。

## 2. 交付冻结后要能回答的五个问题（§8 的五项区分）

| 问题 | 允许的回答形式 |
|---|---|
| 1 系统是否可运行、可复现 | 二元事实：576 格全部到齐、四方闭合、命令与哈希齐全 ⇒ 可/不可 |
| 2 学习模型是否观察到任务损失 | 每 seed、每模式的配对差 + 场景级改善/持平/退化计数 + 按时集合是否交换 |
| 3 是否比强规则更省成本 | 相对 `immediate` 与相对 `late_discovery_gate_147` 的字节差，与任务差**同时**给出 |
| 4 三个训练 seed 是否一致 | 三 seed 差值符号是否一致（描述性），不得升格为"算法稳定" |
| 5 哪些主张仍未建立 | 逐条列出（δ=0 总体保证、泛化、算法稳定性、幂等实机、两类不可达情形、采样不确定性） |

## 3. 故障排查

| 症状 | 定位顺序 | 处置 |
|---|---|---|
| 分片跑完但行数不足 576 | `EVALUATION_LEDGER_RECONCILIATION.json` 的 `missing_scenarios_per_arm` → `failed_book_events` → 对应 `shards/log_shard_i.txt` 的 `TECHNICAL FAILURE` 行 | 属技术失败才用失败储备：`r37_run.py --retry-failed`；性能差不重跑 |
| `claim` 已存在但无行 | 该格在被领取后异常退出 | 走 `--retry-failed`（写 `job_<i>.retry.claim`），并在报告里登记该格的重跑原因与耗时 |
| 统计脚本报矩阵不完整 | 缺格 | 禁止用缺格数据出表；先补技术失败格 |
| 报告与表格数字不一致 | 报告手抄了数字 | 一律重新渲染 `RESULTS_TABLES.md` 并从其引用（同源约束） |
| 想调阈值 / 换 ep64 / 挑 seed | —— | 授权边界禁止；属于新一轮，且需新的额度 |
| 怀疑旧轮被写入 | `R5`（对账脚本与哈希登记表）+ 旧轮目录 mtime 扫描 | 立即停止并报告，不自行回滚他人文件 |

## 4. 交付后的冻结纪律

- 评估完成后：代码、模型、报告数字三者冻结；打包与演示只引用本轮已交付件。
- 不因结果好坏触发重训、续训、调参或新增实验（授权边界 §8 明确）。
- 不部署、不自动推送；学习模型与规则基线之间的最终部署选择留给用户决定。
- 任何后续修改（哪怕只是文档）都必须是新提交，且不得改写本轮已交付的数字。
