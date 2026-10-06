# ROUND45-R2 FINAL3 执行与停止报告

## 结果判定

BC-only 控制 48/48 格均完成，技术执行有效性门禁通过。进入 PPO 的科学控制门禁未通过：三个 seed 均在全部 16 个场景保持与 immediate 完全相同的按时任务 ID 集合，但各自平均计费字节都高于固定 late 规则基线。因此停止在 BC 控制阶段；384 个 PPO 训练格和 96 个最终评估格均未启动。

这将“执行接线有效”与“学习贡献/成本收益”分开：前者通过，后者在进入 PPO 前即未达到冻结门槛。16 个已使用开发场景不是独立确认集。

## 三 seed 原始行重算

门槛为每 seed 精确匹配 immediate 的 16 个逐场景按时任务 ID 集合，并且平均计费不超过 late 规则的 **4838.250 B/场景**。下表来自 48 个控制原始 row 和 16 个 immediate 原始 row 的独立重算；on-time IDs 由每行 `task_reward_events` 中 `credited=true` 且 `on_time=true` 的唯一 task ID 直接提取，字节由每行 `spent_bytes` 求和。

| Seed | scene_count | scene_identity_set_match_all | on_time_total | spent_bytes_total | mean_spent_bytes | 与 4838.25 的差 | passes_training_gate |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 16 | 是 | 96 | 82,434 | 5152.125 | +313.875 | 否 |
| 22 | 16 | 是 | 96 | 82,434 | 5152.125 | +313.875 | 否 |
| 33 | 16 | 是 | 96 | 82,434 | 5152.125 | +313.875 | 否 |

每 seed BC 合计 **82,434 B**，对应 late 合计 **77,412 B**，高出 **5,022 B/seed**。任务侧条件在三 seed 全部通过；成本侧条件在三 seed 全部失败。未池化成一个合并 seed 指标，也未选择单个 seed 继续。

机读门禁：`ROUND45_R2_BC_CONTROL_GATE_R3.json`，SHA256 `ccad7597d9b75c46683cd1b49e252deb928b1ee3b183fa9c1c4580d106e5af05`。原始行独立重算和 claim/row 对应核验：`ROUND45_R2_CONTROL_RAW_ROW_RECOMPUTATION_R3_FINAL3.json`，SHA256 `e7e954865fc65f8d8174f0ff68453cd7a698d0410373e2f6065adb83e8287f71`。核验确认 48 个唯一 seed-scene 控制行，47 个 claim 为 complete、原首格 1 个 claim 保持 `failed_technical` 并由独立 adoption 采用一次；采用 row SHA 与原 receipt 相符。48 行逐场景 task ID 与 immediate 的集合比较全部通过；32 个 immediate/late baseline row SHA 均同时匹配 ROUND42 raw manifest 与冻结 gate。

## 累计账本

计数源：真实 live R3 FINAL3 runtime ledger `runtime/ROUND45_R2_EXECUTION_LEDGER_R3_FINAL3.json`，SHA256 `c6c7884c9dad7fb2f823b3dce5e7a7ffa7e46c6babd7481dccaf12c3ab3f08fa`。

| 项目 | 累计值 |
|---|---:|
| normal claims / 上限 | 48 / 528 |
| 技术 reserve / 上限 | 0 / 24 |
| 总 claim 使用 / 硬上限 | 48 / 552 |
| 环境构造 | 48 |
| `run()` 调用 / 返回 | 48 / 48 |
| 完成步数 | 48,000 |
| raw 完成 claim jobs | 47 |
| adopted 首格 | 1 |
| 有效唯一控制 job（含 adoption） | 48 |
| 历史 technical failed claim / 本阶段新增 | 1 / 0 |
| retry | 0 |
| PPO update / 控制阶段 optimizer step / 控制阶段 checkpoint 写入 | 0 / 0 / 0 |

48 个环境对应 47 个 raw completed jobs + 1 个采用的原首格。首格原 claim 仍为技术失败，原 claim/row 没有改写或退费。BC 本次 3,000 次真实更新、家族累计 4,000 次；没有重跑 BC。前序合成 optimizer 更新为 2，本控制阶段新增 optimizer/PPO/checkpoint 写入均为 0。

固定计划剩余 **480 个 normal key**（384 PPO + 96 终评）未执行，技术 reserve 24 未使用；这些额度未转作其他用途。

## FINAL3 前置复核、修正与历史停止

- `load_runtime(enforce_prelaunch=True)` 实际读回通过：3 次 ledger snapshot、2 次 ledger write。prelaunch 时 raw completed=0、adopted=1、effective unique=1；所有环境/claim/run/step/PPO/optimizer/checkpoint 增量均为 0。
- 原失败 claim SHA 前后均为 `b06ce74fa6967c955e7496a8a5e91ff6c2aeb2d1302711dae98ee5c109c61c1f`；原控制 row SHA 前后均为 `9fd29cdf30ccdca598ec083d89f93515d1c3772aaa29f6d120c8441d459d5b16`。读回收据：`ROUND45_R2_ADOPTED_LEDGER_READBACK_R3_FINAL3.json`（`d393a34ee01d3d52f1578d096bb219a6fee09a1346cc7a514ef8b2ade8d4b588`）；冻结 prelaunch ledger snapshot：`runtime/ROUND45_R2_PRELAUNCH_LEDGER_SNAPSHOT_R3_FINAL3.json`（`742e2ceab77209ae56bff4bd602404ba29ca921027600bee2bb8a2b0a614032b`）。
- 原 `ROUND45_R2_REPORT_R3.md` 保留不覆盖，SHA `e849f537c7c1a1f69864f8d609420a9e0a05710771b797443e34a00686eea89b`。其中 prelaunch 环境数 0 的错误已由 `ROUND45_R2_PRELAUNCH_LEDGER_CORRECTION_R3_FINAL3.md` 版本化勘误为累计 1；本报告再从 FINAL3 实时账本报告控制完成后的累计 48。
- 生成 FINAL3 freeze 前有一次容器回归因缺少 freeze 被前置条件拒绝；未读取 checkpoint、未创建 claim、环境及训练增量 0。完成 freeze 后四模块只读 eval 回归通过，未恢复 optimizer/RNG；之后 prelaunch 实际强制读回通过。
- 旧 `CURRENT_TASK.stop_reason` 的 tracked-path allowlist 错误保留在历史字段和 failure evidence，FINAL3 status-prefix regression 已修复并通过；它不是本次终止原因。

## 冻结和停止

- 分支：`exp/comm-rule-warmstart-cost-learning-r2`。
- 本次执行前 HEAD：`48a257f98ff03f4fcbd3eb2214f4b1ed91011872`；R3 runner 源码 commit：`51cbf982970de71b6d8bfae203c9f931f1120ed1`；prelaunch freeze commit：`11bf3b5b30f77500532e9aad807ae4bfccdcb4f0`。
- 最终停止机读状态：`ROUND45_R2_STOP_STATE_R3_FINAL3.json`，SHA256 `c6f8b9755631de8ee538987321f0b57fca0490ae9701ae9efec51fef63a6a467`。
- 当前停止原因仅为固定 BC 控制科学成本门槛失败。`actor_type:discrete` 是运行日志输出，不是本次失败原因。
- 模型权重、tape、逐行原始日志继续留在 Git 外；本地提交结果摘要与证据。不推送、不部署、不改门槛、不续训。
