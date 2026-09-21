# 强规则独立工程确认：五条锁定规则 × 三个设置 × 200 条全新流

日期：2026-09-21。分支 `analysis/comm-rule-independent-confirmation`，起点 `feat/comm-ack-feedback-gate@fbf96d8`（只引入文档提交 `2e341b1`），协议冻结提交 `5ddd7f5`，流登记提交 `fbeb0b8`。对应总纲 v1.9 与执行指令 `COMMUNICATION_S3_CLOSURE_NEXT_ROUND.md`。

机器可读：`COMM_RULE_CONFIRMATION_RESULTS.json`（全部逐流配对统计、比值、判读与 Pareto 前沿）。本文只报告，不改判据；判据在运行前写定于 `COMM_RULE_CONFIRMATION_PROTOCOL.md`。

**本轮不训练任何模型**，不延长旧 checkpoint，不确认单个最好 seed，不改奖励、网络、任务分布、deadline 或生产默认配置。学习最终测试块 `800000..800399` 继续封存：未生成、未读取、盘上无 artefact。

## 〇、本轮跑了什么

| 项 | 值 |
|---|---|
| 新工程块 | `900000..900199`（200 seed），与全项目四源零重叠 |
| 流 | 200 条、53,630,398 B，fold `6768e72359c3100c…`，检测数合计 1172 |
| 规模 | 200 seed × 3 设置 × 5 臂 = **3,000 / 3,000 episode**，未合并设置样本 |
| 吞吐 | 实测 2.53 s/episode、2.79 MB/episode（探测 75 集，5 个非新块 seed） |
| 统计 | 同 seed 配对百分位 bootstrap，B = 10,000，RNG 20260919，逐流胜/平/负 |
| 成本口径 | total = 任务上行尝试字节 + ACK 反向尝试字节，两项分列，ACK 丢失也计费 |
| 训练 | **未执行**（0 个 seed、0 环境步） |

> 全部 15 个 (设置, 臂) 单元均为 200/200 集，无缺格、无重跑剔除。

## 一、三个设置上的任务效果与成本

### isolated-lossless（隔离信息路径，上行与反向丢包均 1.0）

*negative control: no packet loss on either channel*

| 臂 | onT/ep | 上行 B | ACK B | **total B** | ACK 占比 | 额度使用 | 信息年龄 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `noack_send_once` | 5.5150 | 14,605 | 0 | **14,605** | 0.0 % | 0.498 | 95.5 步 |
| `ack_send_once` | 5.5150 | 14,605 | 10,364 | **24,969** | 41.5 % | 0.498 | 95.5 步 |
| `ack_retry_2` | 5.5150 | 14,605 | 10,364 | **24,969** | 41.5 % | 0.498 | 95.5 步 |
| `ack_periodic_20` | 5.5300 | 24,198 | 16,388 | **40,586** | 40.5 % | 0.826 | 39.7 步 |
| `ack_backoff_16` | 5.5150 | 14,605 | 10,364 | **24,969** | 41.5 % | 0.498 | 95.5 步 |

| 臂 | 条件完成率 | 全部目标完成率 | expired/ep | unresolved/ep | 发现/ep |
|---|---:|---:|---:|---:|---:|
| `noack_send_once` | 0.9421 | 0.9192 | 0.3400 | 0.0050 | 5.8600 |
| `ack_send_once` | 0.9421 | 0.9192 | 0.3400 | 0.0050 | 5.8600 |
| `ack_retry_2` | 0.9421 | 0.9192 | 0.3400 | 0.0050 | 5.8600 |
| `ack_periodic_20` | 0.9445 | 0.9217 | 0.3250 | 0.0050 | 5.8600 |
| `ack_backoff_16` | 0.9421 | 0.9192 | 0.3400 | 0.0050 | 5.8600 |

| 臂 | 发现→首次分配 | 发现→首次工作 | 派遣→首次工作 | 发现→按时完成 |
|---|---:|---:|---:|---:|
| `noack_send_once` | 4.06 步 | 9.08 步 | 5.01 步 | 26.63 步 |
| `ack_send_once` | 4.06 步 | 9.08 步 | 5.01 步 | 26.63 步 |
| `ack_retry_2` | 4.06 步 | 9.08 步 | 5.01 步 | 26.63 步 |
| `ack_periodic_20` | 15.48 步 | 21.18 步 | 5.71 步 | 41.33 步 |
| `ack_backoff_16` | 4.06 步 | 9.08 步 | 5.01 步 | 26.63 步 |

**通信明细（每 episode 均值）**

| 臂 | 发射 | 首发 | 刷新 | 重复发射 | 尝试 | 丢失 | 交付应用 | 接收判重 | 拒付 | ACK 尝试 | ACK 丢失 | ACK 推进 | ACK 重申 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `noack_send_once` | 5.86 | 5.86 | 0.00 | 0.00 | 35.16 | 0.00 | 35.16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `ack_send_once` | 5.86 | 5.86 | 0.00 | 0.00 | 35.16 | 0.00 | 35.16 | 0.00 | 0.00 | 35.16 | 0.00 | 35.16 | 0.00 |
| `ack_retry_2` | 5.86 | 5.86 | 0.00 | 0.00 | 35.16 | 0.00 | 35.16 | 0.00 | 0.00 | 35.16 | 0.00 | 35.16 | 0.00 |
| `ack_periodic_20` | 9.40 | 5.68 | 3.73 | 0.28 | 56.43 | 0.00 | 54.35 | 1.57 | 11.79 | 54.35 | 0.00 | 54.35 | 0.00 |
| `ack_backoff_16` | 5.86 | 5.86 | 0.00 | 0.00 | 35.16 | 0.00 | 35.16 | 0.00 | 0.00 | 35.16 | 0.00 | 35.16 | 0.00 |

### isolated-lossy（隔离信息路径，上行与反向丢包均 0.7）

*the round-6 primary setting, on fresh streams*

| 臂 | onT/ep | 上行 B | ACK B | **total B** | ACK 占比 | 额度使用 | 信息年龄 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `noack_send_once` | 4.4800 | 14,605 | 0 | **14,605** | 0.0 % | 0.498 | 285.4 步 |
| `ack_send_once` | 4.4800 | 14,605 | 7,253 | **21,858** | 33.1 % | 0.498 | 285.4 步 |
| `ack_retry_2` | 5.3150 | 27,312 | 13,647 | **40,959** | 33.3 % | 0.932 | 128.7 步 |
| `ack_periodic_20` | 5.3100 | 25,352 | 12,038 | **37,390** | 32.3 % | 0.865 | 50.4 步 |
| `ack_backoff_16` | 5.1300 | 26,345 | 12,790 | **39,135** | 32.7 % | 0.899 | 88.6 步 |

| 臂 | 条件完成率 | 全部目标完成率 | expired/ep | unresolved/ep | 发现/ep |
|---|---:|---:|---:|---:|---:|
| `noack_send_once` | 0.7645 | 0.7467 | 1.3700 | 0.0100 | 5.8600 |
| `ack_send_once` | 0.7645 | 0.7467 | 1.3700 | 0.0100 | 5.8600 |
| `ack_retry_2` | 0.9077 | 0.8858 | 0.5400 | 0.0050 | 5.8600 |
| `ack_periodic_20` | 0.9075 | 0.8850 | 0.5400 | 0.0100 | 5.8600 |
| `ack_backoff_16` | 0.8765 | 0.8550 | 0.7200 | 0.0100 | 5.8600 |

| 臂 | 发现→首次分配 | 发现→首次工作 | 派遣→首次工作 | 发现→按时完成 |
|---|---:|---:|---:|---:|
| `noack_send_once` | 4.85 步 | 14.85 步 | 10.40 步 | 30.19 步 |
| `ack_send_once` | 4.85 步 | 14.85 步 | 10.40 步 | 30.19 步 |
| `ack_retry_2` | 3.86 步 | 10.13 步 | 6.28 步 | 26.85 步 |
| `ack_periodic_20` | 16.73 步 | 25.40 步 | 8.65 步 | 44.89 步 |
| `ack_backoff_16` | 4.95 步 | 14.65 步 | 9.74 步 | 32.14 步 |

**通信明细（每 episode 均值）**

| 臂 | 发射 | 首发 | 刷新 | 重复发射 | 尝试 | 丢失 | 交付应用 | 接收判重 | 拒付 | ACK 尝试 | ACK 丢失 | ACK 推进 | ACK 重申 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `noack_send_once` | 5.86 | 5.86 | 0.00 | 0.00 | 35.16 | 10.55 | 24.61 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `ack_send_once` | 5.86 | 5.86 | 0.00 | 0.00 | 35.16 | 10.55 | 24.61 | 0.00 | 0.00 | 24.61 | 7.47 | 17.13 | 0.00 |
| `ack_retry_2` | 10.97 | 5.86 | 5.11 | 5.11 | 65.82 | 19.50 | 46.31 | 0.00 | 112.14 | 46.31 | 14.12 | 32.19 | 0.00 |
| `ack_periodic_20` | 9.88 | 5.64 | 4.24 | 0.57 | 59.28 | 17.55 | 39.98 | 1.42 | 15.39 | 39.98 | 12.02 | 27.96 | 0.00 |
| `ack_backoff_16` | 10.28 | 5.55 | 4.72 | 0.81 | 61.65 | 18.23 | 42.47 | 0.81 | 409.38 | 42.47 | 12.78 | 29.69 | 0.00 |

### normal-lossy（正常信息路径，上行与反向丢包均 0.7）

*the real information path: sight refresh and gossip stay on*

| 臂 | onT/ep | 上行 B | ACK B | **total B** | ACK 占比 | 额度使用 | 信息年龄 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `noack_send_once` | 5.8100 | 14,605 | 0 | **14,605** | 0.0 % | 0.498 | 15.8 步 |
| `ack_send_once` | 5.8100 | 14,605 | 7,253 | **21,858** | 33.1 % | 0.498 | 15.8 步 |
| `ack_retry_2` | 5.8100 | 27,312 | 13,647 | **40,959** | 33.3 % | 0.932 | 15.3 步 |
| `ack_periodic_20` | 5.8400 | 16,773 | 6,037 | **22,810** | 27.2 % | 0.573 | 7.8 步 |
| `ack_backoff_16` | 5.7450 | 26,009 | 11,037 | **37,045** | 29.8 % | 0.888 | 12.8 步 |

| 臂 | 条件完成率 | 全部目标完成率 | expired/ep | unresolved/ep | 发现/ep |
|---|---:|---:|---:|---:|---:|
| `noack_send_once` | 0.9915 | 0.9683 | 0.0450 | 0.0050 | 5.8600 |
| `ack_send_once` | 0.9915 | 0.9683 | 0.0450 | 0.0050 | 5.8600 |
| `ack_retry_2` | 0.9915 | 0.9683 | 0.0450 | 0.0050 | 5.8600 |
| `ack_periodic_20` | 0.9967 | 0.9733 | 0.0150 | 0.0050 | 5.8600 |
| `ack_backoff_16` | 0.9807 | 0.9575 | 0.1100 | 0.0050 | 5.8600 |

| 臂 | 发现→首次分配 | 发现→首次工作 | 派遣→首次工作 | 发现→按时完成 |
|---|---:|---:|---:|---:|
| `noack_send_once` | 3.72 步 | 8.22 步 | 4.50 步 | 25.51 步 |
| `ack_send_once` | 3.72 步 | 8.22 步 | 4.50 步 | 25.51 步 |
| `ack_retry_2` | 3.71 步 | 8.20 步 | 4.49 步 | 25.49 步 |
| `ack_periodic_20` | 4.30 步 | 8.38 步 | 4.08 步 | 25.78 步 |
| `ack_backoff_16` | 3.78 步 | 8.25 步 | 4.47 步 | 25.52 步 |

**通信明细（每 episode 均值）**

| 臂 | 发射 | 首发 | 刷新 | 重复发射 | 尝试 | 丢失 | 交付应用 | 接收判重 | 拒付 | ACK 尝试 | ACK 丢失 | ACK 推进 | ACK 重申 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `noack_send_once` | 5.86 | 5.86 | 0.00 | 0.00 | 35.16 | 10.55 | 24.61 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `ack_send_once` | 5.86 | 5.86 | 0.00 | 0.00 | 35.16 | 10.55 | 24.61 | 0.00 | 0.00 | 24.61 | 7.47 | 17.13 | 0.00 |
| `ack_retry_2` | 10.97 | 5.86 | 5.11 | 5.11 | 65.82 | 19.50 | 46.31 | 0.00 | 16.04 | 46.31 | 14.12 | 32.19 | 0.00 |
| `ack_periodic_20` | 6.29 | 4.46 | 1.84 | 0.27 | 37.77 | 11.29 | 19.11 | 0.00 | 0.67 | 19.11 | 5.76 | 13.35 | 0.00 |
| `ack_backoff_16` | 10.09 | 5.61 | 4.48 | 0.59 | 60.54 | 17.99 | 36.41 | 0.00 | 62.33 | 36.41 | 11.02 | 25.39 | 0.00 |

## 二、四个预注册比较

每个比较同时给出**完成数差**与**total-byte 差**的均值、95 % CI 与逐流胜/平/负。任何“降本”只能引用 total 列；上行与 ACK 分列只用于说明成本在哪一侧。

### isolated-lossless

| 比较 | 完成数差 (个/ep) | 95 % CI | 逐流 | total-byte 差 (B/ep) | 95 % CI | 逐流 |
|---|---:|---|---:|---|---:|---|
| `ack_backoff_16 - ack_retry_2` | 0.0000 | [0.0000, 0.0000] | 0 更好 / 200 相等 / 0 更差 | 0.0 | [0.0, 0.0] | 0 更好 / 200 相等 / 0 更差 |
| `ack_periodic_20 - ack_retry_2` | 0.0150 | [-0.1000, 0.1250] | 34 更好 / 140 相等 / 26 更差 | 15616.8 | [14797.8, 16422.0] | 200 更好 / 0 相等 / 0 更差 |
| `ack_retry_2 - ack_send_once` | 0.0000 | [0.0000, 0.0000] | 0 更好 / 200 相等 / 0 更差 | 0.0 | [0.0, 0.0] | 0 更好 / 200 相等 / 0 更差 |
| `ack_retry_2 - noack_send_once` | 0.0000 | [0.0000, 0.0000] | 0 更好 / 200 相等 / 0 更差 | 10364.3 | [10240.1, 10470.6] | 200 更好 / 0 相等 / 0 更差 |

| 比较 | ΔonT/ep (分母) | 每任务上行 B | 每任务 ACK B | 每任务 total B | total CI | 未定义 replicate |
|---|---:|---:|---:|---:|---|---:|
| `ack_backoff_16 - ack_retry_2` | 0.00000 | — | — | **undefined** | — | — |
| `ack_periodic_20 - ack_retry_2` | 0.01500 | 639,560.0 | 401,561.7 | **1,041,121.7** | [111,363.2, 3,157,053.8] | 4116 |
| `ack_retry_2 - ack_send_once` | 0.00000 | — | — | **undefined** | — | — |
| `ack_retry_2 - noack_send_once` | 0.00000 | — | — | **undefined** | — | — |

| 比较 | 新增失败目标 | 救回目标 | 明细 |
|---|---:|---:|---|
| `ack_backoff_16 - ack_retry_2` | 0 | 0 | — |
| `ack_periodic_20 - ack_retry_2` | 49 | 52 | s900001/piloc-e1-target-1(expired), s900011/piloc-e1-target-4(expired), s900012/piloc-e1-target-1(expired), s900012/piloc-e1-target-2(expired), s900012/piloc-e1-target-5(expired), s900015/piloc-e1-target-0(expired), s900015/piloc-e1-target-4(expired), s900018/piloc-e1-target-5(expired) …共 49 个 |
| `ack_retry_2 - ack_send_once` | 0 | 0 | — |
| `ack_retry_2 - noack_send_once` | 0 | 0 | — |

### isolated-lossy

| 比较 | 完成数差 (个/ep) | 95 % CI | 逐流 | total-byte 差 (B/ep) | 95 % CI | 逐流 |
|---|---:|---|---:|---|---:|---|
| `ack_backoff_16 - ack_retry_2` | -0.1850 | [-0.3500, -0.0300] | 42 更好 / 107 相等 / 51 更差 | -1823.4 | [-2460.6, -1202.2] | 109 更好 / 0 相等 / 91 更差 |
| `ack_periodic_20 - ack_retry_2` | -0.0050 | [-0.1700, 0.1600] | 52 更好 / 110 相等 / 38 更差 | -3569.0 | [-4278.5, -2875.4] | 63 更好 / 0 相等 / 137 更差 |
| `ack_retry_2 - ack_send_once` | 0.8350 | [0.6850, 0.9950] | 118 更好 / 72 相等 / 10 更差 | 19100.7 | [18841.0, 19353.8] | 200 更好 / 0 相等 / 0 更差 |
| `ack_retry_2 - noack_send_once` | 0.8350 | [0.6850, 0.9950] | 118 更好 / 72 相等 / 10 更差 | 26354.1 | [26002.3, 26687.8] | 200 更好 / 0 相等 / 0 更差 |

| 比较 | ΔonT/ep (分母) | 每任务上行 B | 每任务 ACK B | 每任务 total B | total CI | 未定义 replicate |
|---|---:|---:|---:|---:|---|---:|
| `ack_backoff_16 - ack_retry_2` | -0.18500 | — | — | **undefined** | — | — |
| `ack_periodic_20 - ack_retry_2` | -0.00500 | — | — | **undefined** | — | — |
| `ack_retry_2 - ack_send_once` | 0.83500 | 15,218.4 | 7,656.7 | **22,875.1** | [19,229.5, 27,826.8] | 0 |
| `ack_retry_2 - noack_send_once` | 0.83500 | 15,218.4 | 16,343.3 | **31,561.7** | [26,502.7, 38,459.3] | 0 |

| 比较 | 新增失败目标 | 救回目标 | 明细 |
|---|---:|---:|---|
| `ack_backoff_16 - ack_retry_2` | 99 | 62 | s900001/piloc-e1-target-0(expired), s900001/piloc-e1-target-1(expired), s900001/piloc-e1-target-2(expired), s900001/piloc-e1-target-3(expired), s900001/piloc-e1-target-4(expired), s900001/piloc-e1-target-5(expired), s900002/piloc-e1-target-3(expired), s900003/piloc-e1-target-0(expired) …共 99 个 |
| `ack_periodic_20 - ack_retry_2` | 81 | 80 | s900001/piloc-e1-target-0(expired), s900001/piloc-e1-target-1(expired), s900001/piloc-e1-target-2(expired), s900001/piloc-e1-target-4(expired), s900001/piloc-e1-target-5(expired), s900004/piloc-e1-target-3(expired), s900005/piloc-e1-target-0(expired), s900008/piloc-e1-target-2(expired) …共 81 个 |
| `ack_retry_2 - ack_send_once` | 23 | 190 | s900000/piloc-e1-target-3(expired), s900007/piloc-e1-target-4(expired), s900029/piloc-e1-target-1(expired), s900057/piloc-e1-target-1(expired), s900064/piloc-e1-target-3(expired), s900067/piloc-e1-target-1(expired), s900072/piloc-e1-target-4(expired), s900086/piloc-e1-target-4(expired) …共 23 个 |
| `ack_retry_2 - noack_send_once` | 23 | 190 | s900000/piloc-e1-target-3(expired), s900007/piloc-e1-target-4(expired), s900029/piloc-e1-target-1(expired), s900057/piloc-e1-target-1(expired), s900064/piloc-e1-target-3(expired), s900067/piloc-e1-target-1(expired), s900072/piloc-e1-target-4(expired), s900086/piloc-e1-target-4(expired) …共 23 个 |

### normal-lossy

| 比较 | 完成数差 (个/ep) | 95 % CI | 逐流 | total-byte 差 (B/ep) | 95 % CI | 逐流 |
|---|---:|---|---:|---|---:|---|
| `ack_backoff_16 - ack_retry_2` | -0.0650 | [-0.1000, -0.0350] | 0 更好 / 187 相等 / 13 更差 | -3913.6 | [-4549.3, -3302.9] | 31 更好 / 0 相等 / 169 更差 |
| `ack_periodic_20 - ack_retry_2` | 0.0300 | [-0.0150, 0.0800] | 5 更好 / 193 相等 / 2 更差 | -18148.4 | [-19206.2, -17146.1] | 0 更好 / 0 相等 / 200 更差 |
| `ack_retry_2 - ack_send_once` | 0.0000 | [0.0000, 0.0000] | 0 更好 / 200 相等 / 0 更差 | 19100.7 | [18841.0, 19353.8] | 200 更好 / 0 相等 / 0 更差 |
| `ack_retry_2 - noack_send_once` | 0.0000 | [0.0000, 0.0000] | 0 更好 / 200 相等 / 0 更差 | 26354.1 | [26002.3, 26687.8] | 200 更好 / 0 相等 / 0 更差 |

| 比较 | ΔonT/ep (分母) | 每任务上行 B | 每任务 ACK B | 每任务 total B | total CI | 未定义 replicate |
|---|---:|---:|---:|---:|---|---:|
| `ack_backoff_16 - ack_retry_2` | -0.06500 | — | — | **undefined** | — | — |
| `ack_periodic_20 - ack_retry_2` | 0.03000 | -351,296.0 | -253,649.5 | **-604,945.5** | [-3,663,304.9, -216,960.1] | 1222 |
| `ack_retry_2 - ack_send_once` | 0.00000 | — | — | **undefined** | — | — |
| `ack_retry_2 - noack_send_once` | 0.00000 | — | — | **undefined** | — | — |

| 比较 | 新增失败目标 | 救回目标 | 明细 |
|---|---:|---:|---|
| `ack_backoff_16 - ack_retry_2` | 13 | 0 | s900002/piloc-e1-target-3(expired), s900011/piloc-e1-target-4(expired), s900045/piloc-e1-target-2(expired), s900049/piloc-e1-target-2(expired), s900051/piloc-e1-target-5(expired), s900067/piloc-e1-target-1(expired), s900089/piloc-e1-target-4(expired), s900090/piloc-e1-target-3(expired) …共 13 个 |
| `ack_periodic_20 - ack_retry_2` | 3 | 9 | s900012/piloc-e1-target-2(expired), s900167/piloc-e1-target-1(expired), s900167/piloc-e1-target-5(expired) |
| `ack_retry_2 - ack_send_once` | 0 | 0 | — |
| `ack_retry_2 - noack_send_once` | 0 | 0 | — |

## 三、预注册判读

### isolated-lossless

**无损负控：通过。** proceed: the negative control behaved as the protocol expects。

| 臂 | 完成数差 | 95 % CI | 判定 | total-byte 差 |
|---|---:|---|---|---:|
| `ack_backoff_16` | 0.0000 | [0.0000, 0.0000] | 未稳定提高 | 10364.3 B |
| `ack_periodic_20` | 0.0150 | [-0.1000, 0.1300] | 未稳定提高 | 25981.2 B |
| `ack_retry_2` | 0.0000 | [0.0000, 0.0000] | 未稳定提高 | 10364.3 B |

> the ACK arms' higher total bytes on this setting are the protocol's cost, not a pass and not a failure。

**规则前沿：稳定。** the locked rule stays the strongest completion rule of the set on this data。

- `ack_backoff_16 − ack_retry_2`：0.0000 个/ep，CI [0.0000, 0.0000]，逐流 0 更好 / 200 相等 / 0 更差 ⇒ 未超过锁定规则；
- `ack_periodic_20 − ack_retry_2`：0.0150 个/ep，CI [-0.1000, 0.1300]，逐流 34 更好 / 140 相等 / 26 更差 ⇒ 未超过锁定规则；

**Pareto 前沿（逐流双轴）：`ack_periodic_20`, `noack_send_once`。** 被支配：`ack_backoff_16`, `ack_retry_2`, `ack_send_once`。

### isolated-lossy

**有损隔离确认：复现。** round 6's cost-effect effect gains independent support on fresh streams; report the byte increase beside the task gain。

- 完成数差 **0.8350 个/ep**，CI [0.6850, 0.9900]，逐流 118 更好 / 72 相等 / 10 更差；
- total-byte 差 **26354.1 B/ep**，CI [26017.6, 26687.4]，逐流 200 更好 / 0 相等 / 0 更差。

**规则前沿：稳定。** the locked rule stays the strongest completion rule of the set on this data。

- `ack_backoff_16 − ack_retry_2`：-0.1850 个/ep，CI [-0.3500, -0.0300]，逐流 42 更好 / 107 相等 / 51 更差 ⇒ 未超过锁定规则；
- `ack_periodic_20 − ack_retry_2`：-0.0050 个/ep，CI [-0.1750, 0.1550]，逐流 52 更好 / 110 相等 / 38 更差 ⇒ 未超过锁定规则；

**Pareto 前沿（逐流双轴）：`ack_backoff_16`, `ack_periodic_20`, `ack_retry_2`, `noack_send_once`。** 被支配：`ack_send_once`。

### normal-lossy

**正常信息路径：不可作为部署候选。** do not propose deployment: the real information path gives no completion gain to justify the bytes。

- 完成数差 **0.0000 个/ep**，CI [0.0000, 0.0000]，逐流 0 更好 / 200 相等 / 0 更差；
- total-byte 差 **26354.1 B/ep**，CI [26017.6, 26687.4]，逐流 200 更好 / 0 相等 / 0 更差；
- 逐目标：新增失败 **0** 个、救回 **0** 个；结局相同但更贵：**是**。

**规则前沿：稳定。** the locked rule stays the strongest completion rule of the set on this data。

- `ack_backoff_16 − ack_retry_2`：-0.0650 个/ep，CI [-0.1000, -0.0350]，逐流 0 更好 / 187 相等 / 13 更差 ⇒ 未超过锁定规则；
- `ack_periodic_20 − ack_retry_2`：0.0300 个/ep，CI [-0.0150, 0.0800]，逐流 5 更好 / 193 相等 / 2 更差 ⇒ 未超过锁定规则；

**Pareto 前沿（逐流双轴）：`ack_periodic_20`, `noack_send_once`。** 被支配：`ack_backoff_16`, `ack_retry_2`, `ack_send_once`。

## 四、随机数与配对强度

- **前向丢包**：per-episode np.random.default_rng(scenario_seed) stream, one draw per (subscriber, emission): same mechanism and seed for every arm, so a same-distribution comparison, not the same events。
- **反向 ACK 丢包**：counter-based (episode_seed, 'uplink_ack', subscriber, target, ordinal) plus the delivery tick: common random numbers between ACK-on arms。
- 因此：no-ACK 与 ACK 臂之间是**同分布比较**（同 seed、同机制，不是同一串事件）；ACK 臂之间在相同 (episode, subscriber, target, ordinal) 键上是**共同随机数**。

## 五、复跑

```bash
cd runs/comm_rule_confirm_20260921
python block_audit.py
python record_streams.py --shards 4 --shard 0
python record_streams.py --mode merge
python confirm_preflight.py --threads 1
python confirm_matrix.py --mode probe
python confirm_matrix.py --mode matrix --shards 8 --shard 0
python confirm_matrix.py --mode report
python render_results.py
python confirm_manifest.py --threads 1
```

