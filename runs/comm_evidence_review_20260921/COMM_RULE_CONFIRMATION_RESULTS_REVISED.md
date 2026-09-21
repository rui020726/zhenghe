# 规则独立工程确认：修订版报告（同一统计对象、统一措辞）

日期：2026-09-21。分支 `analysis/comm-evidence-review`，对应总纲 v2.1 与执行指令 `COMM_RESEARCH_REVIEW_NEXT_ROUND.md` §2/§3。修订对象：`../comm_rule_confirm_20260921/COMM_RULE_CONFIRMATION_RESULTS.md`（该文件与其判定原样保留、可追溯）。

**本文不产生任何新数据**：episode 0、新流 0、训练 0；输入是归档的 3,000 行（8 个分片，逐文件 sha256 见附录），统计量仍为 seed 级配对百分位 bootstrap（B = 10,000，RNG 20260919），三个设置不合并样本。原停止判定与预注册判据未改；改的只有一处统计组织方式（每个量只抽一次样）与措辞口径。

## 〇、单位与口径

- 单位只用这四种，不互换：个任务/episode（完成数差）、B/episode（每局字节差）、B/新增按时任务（比值）、完成率（[0,1] 比例，不是百分点）。四者不可互换：+0.835 是个/ep，+26 354 是 B/ep，31 562 是 B/新增任务，不是 B/ep。。
- 信道字段是 `delivery_probability`（**成功率**）：投递成功率：comm.py 用 rng.random() > p 判丢包，comm_ack.py 用 random() < p 判送达。成功率 1.0 ⇒ 丢包率 0.0；成功率 0.7 ⇒ 丢包率 0.3。
- 字节差只说更高/相等/更低；只有完成数才说更好/相等/更差。
- 比值一律配分子（B/ep）与分母（个/ep）一起给；丢掉非正分母后得到的区间标 `conditional`。

## 一、三个设置上的任务效果与成本

### isolated-lossless（隔离信息路径，上行与反向投递成功率均 1.0（丢包率 0.0））

*negative control: no packet loss on either channel*

| 臂 | onT (个/ep) | 上行 B/ep | ACK B/ep | **total B/ep** | ACK 占比 | 额度使用 | 信息年龄(步) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `noack_send_once` | 5.5150 | 14,605 | 0 | **14,605** | 0.0 % | 0.498 | 95.5 |
| `ack_send_once` | 5.5150 | 14,605 | 10,364 | **24,969** | 41.5 % | 0.498 | 95.5 |
| `ack_retry_2` | 5.5150 | 14,605 | 10,364 | **24,969** | 41.5 % | 0.498 | 95.5 |
| `ack_periodic_20` | 5.5300 | 24,198 | 16,388 | **40,586** | 40.5 % | 0.826 | 39.7 |
| `ack_backoff_16` | 5.5150 | 14,605 | 10,364 | **24,969** | 41.5 % | 0.498 | 95.5 |

| 臂 | 条件完成率 | 全部目标完成率 | expired (个/ep) | unresolved (个/ep) | 发现 (个/ep) |
|---|---:|---:|---:|---:|---:|
| `noack_send_once` | 0.9421 | 0.9192 | 0.3400 | 0.0050 | 5.8600 |
| `ack_send_once` | 0.9421 | 0.9192 | 0.3400 | 0.0050 | 5.8600 |
| `ack_retry_2` | 0.9421 | 0.9192 | 0.3400 | 0.0050 | 5.8600 |
| `ack_periodic_20` | 0.9445 | 0.9217 | 0.3250 | 0.0050 | 5.8600 |
| `ack_backoff_16` | 0.9421 | 0.9192 | 0.3400 | 0.0050 | 5.8600 |

### isolated-lossy（隔离信息路径，上行与反向投递成功率均 0.7（丢包率 0.3））

*the round-6 primary setting, on fresh streams*

| 臂 | onT (个/ep) | 上行 B/ep | ACK B/ep | **total B/ep** | ACK 占比 | 额度使用 | 信息年龄(步) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `noack_send_once` | 4.4800 | 14,605 | 0 | **14,605** | 0.0 % | 0.498 | 285.4 |
| `ack_send_once` | 4.4800 | 14,605 | 7,253 | **21,858** | 33.1 % | 0.498 | 285.4 |
| `ack_retry_2` | 5.3150 | 27,312 | 13,647 | **40,959** | 33.3 % | 0.932 | 128.7 |
| `ack_periodic_20` | 5.3100 | 25,352 | 12,038 | **37,390** | 32.3 % | 0.865 | 50.4 |
| `ack_backoff_16` | 5.1300 | 26,345 | 12,790 | **39,135** | 32.7 % | 0.899 | 88.6 |

| 臂 | 条件完成率 | 全部目标完成率 | expired (个/ep) | unresolved (个/ep) | 发现 (个/ep) |
|---|---:|---:|---:|---:|---:|
| `noack_send_once` | 0.7645 | 0.7467 | 1.3700 | 0.0100 | 5.8600 |
| `ack_send_once` | 0.7645 | 0.7467 | 1.3700 | 0.0100 | 5.8600 |
| `ack_retry_2` | 0.9077 | 0.8858 | 0.5400 | 0.0050 | 5.8600 |
| `ack_periodic_20` | 0.9075 | 0.8850 | 0.5400 | 0.0100 | 5.8600 |
| `ack_backoff_16` | 0.8765 | 0.8550 | 0.7200 | 0.0100 | 5.8600 |

### normal-lossy（正常信息路径，上行与反向投递成功率均 0.7（丢包率 0.3））

*the real information path: sight refresh and gossip stay on*

| 臂 | onT (个/ep) | 上行 B/ep | ACK B/ep | **total B/ep** | ACK 占比 | 额度使用 | 信息年龄(步) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `noack_send_once` | 5.8100 | 14,605 | 0 | **14,605** | 0.0 % | 0.498 | 15.8 |
| `ack_send_once` | 5.8100 | 14,605 | 7,253 | **21,858** | 33.1 % | 0.498 | 15.8 |
| `ack_retry_2` | 5.8100 | 27,312 | 13,647 | **40,959** | 33.3 % | 0.932 | 15.3 |
| `ack_periodic_20` | 5.8400 | 16,773 | 6,037 | **22,810** | 27.2 % | 0.573 | 7.8 |
| `ack_backoff_16` | 5.7450 | 26,009 | 11,037 | **37,045** | 29.8 % | 0.888 | 12.8 |

| 臂 | 条件完成率 | 全部目标完成率 | expired (个/ep) | unresolved (个/ep) | 发现 (个/ep) |
|---|---:|---:|---:|---:|---:|
| `noack_send_once` | 0.9915 | 0.9683 | 0.0450 | 0.0050 | 5.8600 |
| `ack_send_once` | 0.9915 | 0.9683 | 0.0450 | 0.0050 | 5.8600 |
| `ack_retry_2` | 0.9915 | 0.9683 | 0.0450 | 0.0050 | 5.8600 |
| `ack_periodic_20` | 0.9967 | 0.9733 | 0.0150 | 0.0050 | 5.8600 |
| `ack_backoff_16` | 0.9807 | 0.9575 | 0.1100 | 0.0050 | 5.8600 |

## 二、四个预注册比较（每个量只抽样一次）

完成数列用'更好/相等/更差'，total-byte 列用'更高/相等/更低'——字节差的正值一律是'更贵'，不是优势。上行与 ACK 分列只用于说明成本落在哪一侧；任何降本主张只看 total。

### isolated-lossless（上行与反向投递成功率均 1.0（丢包率 0.0））

| 比较 | 完成数差 (个/ep) | 95 % CI | 逐流 | total-byte 差 (B/ep) | 95 % CI | 逐流 |
|---|---:|---|---:|---:|---|---|
| `ack_retry_2` − `noack_send_once` | 0.0000 | [0.0000, 0.0000] | 0 更好 / 200 相等 / 0 更差 | 10364.3 | [10240.1, 10470.6] | 200 更高 / 0 相等 / 0 更低 |
| `ack_retry_2` − `ack_send_once` | 0.0000 | [0.0000, 0.0000] | 0 更好 / 200 相等 / 0 更差 | 0.0 | [0.0, 0.0] | 0 更高 / 200 相等 / 0 更低 |
| `ack_periodic_20` − `ack_retry_2` | 0.0150 | [-0.1000, 0.1250] | 34 更好 / 140 相等 / 26 更差 | 15616.8 | [14797.8, 16422.0] | 200 更高 / 0 相等 / 0 更低 |
| `ack_backoff_16` − `ack_retry_2` | 0.0000 | [0.0000, 0.0000] | 0 更好 / 200 相等 / 0 更差 | 0.0 | [0.0, 0.0] | 0 更高 / 200 相等 / 0 更低 |

| 比较 | 分母 ΔonT (个/ep) | 每任务上行 B | 每任务 ACK B | 每任务 total B | total 区间 | 区间类型 | 未定义 replicate | 可稳健解释 |
|---|---:|---:|---:|---:|---|---|---:|---|
| `ack_retry_2 − noack_send_once` | 0.00000 | — | — | **undefined** | n/a（未计算） | not_applicable | 0 | 否 |
| `ack_retry_2 − ack_send_once` | 0.00000 | — | — | **undefined** | n/a（未计算） | not_applicable | 0 | 否 |
| `ack_periodic_20 − ack_retry_2` | 0.01500 | 639,560.0 | 401,561.7 | **1,041,121.7** | [111,363.2, 3,157,053.8] | conditional | 4116 | 否 |
| `ack_backoff_16 − ack_retry_2` | 0.00000 | — | — | **undefined** | n/a（未计算） | not_applicable | 0 | 否 |

> `ack_retry_2` − `noack_send_once` 比值：点估计分母非正 ⇒ undefined；成本差与效果差本身照常报告，不删除样本后给出比值。
> `ack_retry_2` − `ack_send_once` 比值：点估计分母非正 ⇒ undefined；成本差与效果差本身照常报告，不删除样本后给出比值。
> `ack_periodic_20` − `ack_retry_2` 比值：分位数只在 5884/10000 个 defined replicate 上计算：条件区间，不是无条件 95 % CI；分母（完成数差）的 CI 含 0：比值符号依赖少量分母的抽样，不可稳健解释，应引用原始的成本差与效果差。
> `ack_backoff_16` − `ack_retry_2` 比值：点估计分母非正 ⇒ undefined；成本差与效果差本身照常报告，不删除样本后给出比值。

| 比较 | 新增失败 (个目标) | 救回 (个目标) |
|---|---:|---:|
| `ack_retry_2` − `noack_send_once` | 0 | 0 |
| `ack_retry_2` − `ack_send_once` | 0 | 0 |
| `ack_periodic_20` − `ack_retry_2` | 49 | 52 |
| `ack_backoff_16` − `ack_retry_2` | 0 | 0 |

### isolated-lossy（上行与反向投递成功率均 0.7（丢包率 0.3））

| 比较 | 完成数差 (个/ep) | 95 % CI | 逐流 | total-byte 差 (B/ep) | 95 % CI | 逐流 |
|---|---:|---|---:|---:|---|---|
| `ack_retry_2` − `noack_send_once` | 0.8350 | [0.6850, 0.9950] | 118 更好 / 72 相等 / 10 更差 | 26354.1 | [26002.3, 26687.8] | 200 更高 / 0 相等 / 0 更低 |
| `ack_retry_2` − `ack_send_once` | 0.8350 | [0.6850, 0.9950] | 118 更好 / 72 相等 / 10 更差 | 19100.7 | [18841.0, 19353.8] | 200 更高 / 0 相等 / 0 更低 |
| `ack_periodic_20` − `ack_retry_2` | -0.0050 | [-0.1700, 0.1600] | 52 更好 / 110 相等 / 38 更差 | -3569.0 | [-4278.5, -2875.4] | 63 更高 / 0 相等 / 137 更低 |
| `ack_backoff_16` − `ack_retry_2` | -0.1850 | [-0.3500, -0.0300] | 42 更好 / 107 相等 / 51 更差 | -1823.4 | [-2460.6, -1202.2] | 109 更高 / 0 相等 / 91 更低 |

| 比较 | 分母 ΔonT (个/ep) | 每任务上行 B | 每任务 ACK B | 每任务 total B | total 区间 | 区间类型 | 未定义 replicate | 可稳健解释 |
|---|---:|---:|---:|---:|---|---|---:|---|
| `ack_retry_2 − noack_send_once` | 0.83500 | 15,218.4 | 16,343.3 | **31,561.7** | [26,502.7, 38,459.3] | unconditional | 0 | 是 |
| `ack_retry_2 − ack_send_once` | 0.83500 | 15,218.4 | 7,656.7 | **22,875.1** | [19,229.5, 27,826.8] | unconditional | 0 | 是 |
| `ack_periodic_20 − ack_retry_2` | -0.00500 | — | — | **undefined** | n/a（未计算） | not_applicable | 0 | 否 |
| `ack_backoff_16 − ack_retry_2` | -0.18500 | — | — | **undefined** | n/a（未计算） | not_applicable | 0 | 否 |

> `ack_periodic_20` − `ack_retry_2` 比值：点估计分母非正 ⇒ undefined；成本差与效果差本身照常报告，不删除样本后给出比值。
> `ack_backoff_16` − `ack_retry_2` 比值：点估计分母非正 ⇒ undefined；成本差与效果差本身照常报告，不删除样本后给出比值。

| 比较 | 新增失败 (个目标) | 救回 (个目标) |
|---|---:|---:|
| `ack_retry_2` − `noack_send_once` | 23 | 190 |
| `ack_retry_2` − `ack_send_once` | 23 | 190 |
| `ack_periodic_20` − `ack_retry_2` | 81 | 80 |
| `ack_backoff_16` − `ack_retry_2` | 99 | 62 |

### normal-lossy（上行与反向投递成功率均 0.7（丢包率 0.3））

| 比较 | 完成数差 (个/ep) | 95 % CI | 逐流 | total-byte 差 (B/ep) | 95 % CI | 逐流 |
|---|---:|---|---:|---:|---|---|
| `ack_retry_2` − `noack_send_once` | 0.0000 | [0.0000, 0.0000] | 0 更好 / 200 相等 / 0 更差 | 26354.1 | [26002.3, 26687.8] | 200 更高 / 0 相等 / 0 更低 |
| `ack_retry_2` − `ack_send_once` | 0.0000 | [0.0000, 0.0000] | 0 更好 / 200 相等 / 0 更差 | 19100.7 | [18841.0, 19353.8] | 200 更高 / 0 相等 / 0 更低 |
| `ack_periodic_20` − `ack_retry_2` | 0.0300 | [-0.0150, 0.0800] | 5 更好 / 193 相等 / 2 更差 | -18148.4 | [-19206.2, -17146.1] | 0 更高 / 0 相等 / 200 更低 |
| `ack_backoff_16` − `ack_retry_2` | -0.0650 | [-0.1000, -0.0350] | 0 更好 / 187 相等 / 13 更差 | -3913.6 | [-4549.3, -3302.9] | 31 更高 / 0 相等 / 169 更低 |

| 比较 | 分母 ΔonT (个/ep) | 每任务上行 B | 每任务 ACK B | 每任务 total B | total 区间 | 区间类型 | 未定义 replicate | 可稳健解释 |
|---|---:|---:|---:|---:|---|---|---:|---|
| `ack_retry_2 − noack_send_once` | 0.00000 | — | — | **undefined** | n/a（未计算） | not_applicable | 0 | 否 |
| `ack_retry_2 − ack_send_once` | 0.00000 | — | — | **undefined** | n/a（未计算） | not_applicable | 0 | 否 |
| `ack_periodic_20 − ack_retry_2` | 0.03000 | -351,296.0 | -253,649.5 | **-604,945.5** | [-3,663,304.9, -216,960.1] | conditional | 1222 | 否 |
| `ack_backoff_16 − ack_retry_2` | -0.06500 | — | — | **undefined** | n/a（未计算） | not_applicable | 0 | 否 |

> `ack_retry_2` − `noack_send_once` 比值：点估计分母非正 ⇒ undefined；成本差与效果差本身照常报告，不删除样本后给出比值。
> `ack_retry_2` − `ack_send_once` 比值：点估计分母非正 ⇒ undefined；成本差与效果差本身照常报告，不删除样本后给出比值。
> `ack_periodic_20` − `ack_retry_2` 比值：分位数只在 8778/10000 个 defined replicate 上计算：条件区间，不是无条件 95 % CI；分母（完成数差）的 CI 含 0：比值符号依赖少量分母的抽样，不可稳健解释，应引用原始的成本差与效果差。
> `ack_backoff_16` − `ack_retry_2` 比值：点估计分母非正 ⇒ undefined；成本差与效果差本身照常报告，不删除样本后给出比值。

| 比较 | 新增失败 (个目标) | 救回 (个目标) |
|---|---:|---:|
| `ack_retry_2` − `noack_send_once` | 0 | 0 |
| `ack_retry_2` − `ack_send_once` | 0 | 0 |
| `ack_periodic_20` − `ack_retry_2` | 3 | 9 |
| `ack_backoff_16` − `ack_retry_2` | 13 | 0 |

## 三、预注册判读（切片，不再计算）

### isolated-lossless（上行与反向投递成功率均 1.0（丢包率 0.0））

**无损负控：通过。** proceed: the negative control behaved as the protocol expects。

| 臂 | 完成数差 (个/ep) | 95 % CI | 判定 | total-byte 差 (B/ep) | 方向 |
|---|---:|---|---|---:|---|
| `ack_retry_2` | 0.0000 | [0.0000, 0.0000] | 未稳定提高 | 10364.3 | 更贵（CI 下界 > 0） |
| `ack_periodic_20` | 0.0150 | [-0.1000, 0.1250] | 未稳定提高 | 25981.2 | 更贵（CI 下界 > 0） |
| `ack_backoff_16` | 0.0000 | [0.0000, 0.0000] | 未稳定提高 | 10364.3 | 更贵（CI 下界 > 0） |

> the ACK arms' higher total bytes on this setting are the protocol's cost, not a pass and not a failure。

**规则比较（预注册动作：不回头调参、不重命名主规则）**

- `ack_periodic_20` 相对锁定的 `ack_retry_2`：完成数未检出差异（CI 含 0）：只能说没测出差别，不能称效果相同；未通过 −0.05 个/ep 的非劣界（CI 下界 -0.100 < −0.05）；total bytes 更贵（CI 下界 > 0），两轴结论各自独立。
- `ack_backoff_16` 相对锁定的 `ack_retry_2`：完成数未检出差异（CI 含 0）：只能说没测出差别，不能称效果相同；通过了 −0.05 个/ep 的非劣界（CI 下界 +0.000 ≥ −0.05），但该界只在'不显著更差'的意义上成立，不建立等效；total bytes 未检出差异（CI 含 0），两轴结论各自独立。

**'前沿'必须写明是哪一个定义**（三个定义、三种结论）

1. 样本均值 Pareto 前沿：`ack_periodic_20`, `noack_send_once`；被均值支配：`ack_backoff_16`, `ack_retry_2`, `ack_send_once`。
2. 逐流支配（每流双轴不劣且至少一流严格更优）：前沿 `ack_periodic_20`, `noack_send_once`；被支配 `ack_backoff_16`, `ack_retry_2`, `ack_send_once`。
3. 有统计支持的支配（完成数 CI 下界 > 0 且 total-byte CI 上界 < 0）：无此配对。

> 本定义下未被支持的配对不是'等效'，只是未检出差异；预注册动作：no challenger beats the locked rule with statistical support on this data; the locked primary rule stays, and this says nothing about the other two frontier definitions。

### isolated-lossy（上行与反向投递成功率均 0.7（丢包率 0.3））

**有损隔离确认：复现。** round 6's cost-effect effect gains independent support on fresh streams; report the byte increase beside the task gain。

- 完成数差 **+0.8350 个/ep**，CI [0.6850, 0.9950]，118 更好 / 72 相等 / 10 更差（更高（CI 下界 > 0））；
- total-byte 差 **+26,354.1 B/ep**，CI [26002.3, 26687.8]，200 更高 / 0 相等 / 0 更低（更贵（CI 下界 > 0））——是更贵，不是更好；
- 因果范围：ack_retry_2 相对 noack_send_once 同时改变了三件事：打开反向信道、开始消费确认状态、以及允许第二次尝试。差值是三者的联合效应，不是'ACK 信息本身'的净价值。

**规则比较（预注册动作：不回头调参、不重命名主规则）**

- `ack_periodic_20` 相对锁定的 `ack_retry_2`：完成数未检出差异（CI 含 0）：只能说没测出差别，不能称效果相同；未通过 −0.05 个/ep 的非劣界（CI 下界 -0.170 < −0.05）；total bytes 更省（CI 上界 < 0），两轴结论各自独立。
- `ack_backoff_16` 相对锁定的 `ack_retry_2`：完成数更低（CI 上界 < 0）；total bytes 更省（CI 上界 < 0），两轴结论各自独立。

**'前沿'必须写明是哪一个定义**（三个定义、三种结论）

1. 样本均值 Pareto 前沿：`ack_periodic_20`, `ack_retry_2`, `noack_send_once`；被均值支配：`ack_backoff_16`, `ack_send_once`。
2. 逐流支配（每流双轴不劣且至少一流严格更优）：前沿 `ack_backoff_16`, `ack_periodic_20`, `ack_retry_2`, `noack_send_once`；被支配 `ack_send_once`。
3. 有统计支持的支配（完成数 CI 下界 > 0 且 total-byte CI 上界 < 0）：无此配对。

> 本定义下未被支持的配对不是'等效'，只是未检出差异；预注册动作：no challenger beats the locked rule with statistical support on this data; the locked primary rule stays, and this says nothing about the other two frontier definitions。

### normal-lossy（上行与反向投递成功率均 0.7（丢包率 0.3））

**正常信息路径：不可作为部署候选。** do not propose deployment: the real information path gives no completion gain to justify the bytes。

- 完成数差 **+0.0000 个/ep**，CI [0.0000, 0.0000]，0 更好 / 200 相等 / 0 更差；
- total-byte 差 **+26,354.1 B/ep**，CI [26002.3, 26687.8]，200 更高 / 0 相等 / 0 更低；
- 逐目标：新增失败 **0** 个、救回 **0** 个；结局相同但更贵（工程上被支配）：**是**。

**规则比较（预注册动作：不回头调参、不重命名主规则）**

- `ack_periodic_20` 相对锁定的 `ack_retry_2`：完成数未检出差异（CI 含 0）：只能说没测出差别，不能称效果相同；通过了 −0.05 个/ep 的非劣界（CI 下界 -0.015 ≥ −0.05），但该界只在'不显著更差'的意义上成立，不建立等效；total bytes 更省（CI 上界 < 0），两轴结论各自独立。
- `ack_backoff_16` 相对锁定的 `ack_retry_2`：完成数更低（CI 上界 < 0）；total bytes 更省（CI 上界 < 0），两轴结论各自独立。

**'前沿'必须写明是哪一个定义**（三个定义、三种结论）

1. 样本均值 Pareto 前沿：`ack_periodic_20`, `noack_send_once`；被均值支配：`ack_backoff_16`, `ack_retry_2`, `ack_send_once`。
2. 逐流支配（每流双轴不劣且至少一流严格更优）：前沿 `ack_periodic_20`, `noack_send_once`；被支配 `ack_backoff_16`, `ack_retry_2`, `ack_send_once`。
3. 有统计支持的支配（完成数 CI 下界 > 0 且 total-byte CI 上界 < 0）：无此配对。

> 本定义下未被支持的配对不是'等效'，只是未检出差异；预注册动作：no challenger beats the locked rule with statistical support on this data; the locked primary rule stays, and this says nothing about the other two frontier definitions。

## 四、ACK 比较对象：开关、消费、重试能力与预算

第六轮的参照 `send_once` 是**反向信道打开、但策略不消费确认状态**（每视图一次上报，没有重发能力）：它的 total 里已含 ACK 字节。本轮把它拆成两条臂显式对照——`noack_send_once`（信道关）与 `ack_send_once`（信道开、不消费）。因此下面三个成本差是**三个不同的比较**，不能读成同一条基线发生了变化。

| 比较对象 | 所在轮次 | ACK 开关 | 是否消费 ACK | 重试能力 | 上行额度 |
|---|---|---|---|---|---|
| `send_once（第六轮矩阵臂）` | S3-5（48 条训练流） | 开 | 否（每条视图只发一次，无从消费） | 无 | 29 300 B/episode |
| `noack_send_once（本轮基线）` | S3-6（200 条新流） | 关 | 否（无反馈信道） | 无 | 29 300 B/episode |
| `ack_send_once（本轮）` | S3-6（200 条新流） | 开 | 否（与第六轮 send_once 同行为，只是显式计费） | 无 | 29 300 B/episode |
| `retry_until_acked_2 / ack_retry_2` | S3-5 与 S3-6 | 开 | 是（按未确认订阅者重发，max_attempts=2） | 有（每视图最多 2 次尝试） | 29 300 B/episode |

- +26 354.1 B/ep = `ack_retry_2` − `noack_send_once`（本轮 200 新流，基线**无 ACK 信道**）：协议整体（开信道 + 消费反馈 + 允许二次尝试）的成本差；
- +19 100.7 B/ep = `ack_retry_2` − `ack_send_once`（本轮 200 新流，基线**有 ACK 信道、不消费**）：重发规则本身的成本增量；
- +19 562 B/ep = 第六轮 `retry_until_acked_2` − `send_once`（48 条**训练**流，基线同为有 ACK、不消费）：与上一行同类，但数据、轮次不同，不是'同一基线的成本变化'。

## 五、随机数与配对强度

- **前向（上行）丢包**：per-episode np.random.default_rng(scenario_seed) stream, one draw per (subscriber, emission): same mechanism and seed for every arm, so a same-distribution comparison, not the same events。
- **反向 ACK 丢包**：counter-based (episode_seed, 'uplink_ack', subscriber, target, ordinal) plus the delivery tick: common random numbers between ACK-on arms。
- 因此：no-ACK 与 ACK 臂之间是**同分布比较**（同 seed、同机制，不是同一串事件）；ACK 臂之间在相同 (episode, subscriber, target, ordinal) 键上是**共同随机数**。跨这两类比较的数字不能互相代入解释。

## 六、主比值单独核对

- B/新增按时任务（total），isolated-lossy，ack_retry_2 − noack_send_once：**31,561.7 B per additional on-time target（不是 B/episode）**（分子 26,354.1 B/ep、分母 0.8350 个/ep）。
- 冻结区间（RNG 20260919，未定义 replicate 0）：[26,502.7, 38,459.3]；独立复核区间（RNG 20260921，未定义 0）：[26,521.8, 38,508.4]。
- 两条区间来自同一对逐 seed 差向量、不同抽样种子：区间宽度一致即说明该比值对抽样不敏感，可以引用；不一致则只引用分子分母两个原始量。
- 经济解释只给量纲正确的条件：设新增按时任务价值 V（效用/任务）、字节成本 c（效用/B），此样本均值下净效用改善需 `V × 0.835 > c × 26354.1`。未指定 V、c 前不宣布值得部署；total 仍只含任务上行与 ACK，不含未计入的 gossip/系统通信。

## 七、与归档输出的差异登记

- 原因：归档报告对同一统计量调用了两次 bootstrap：比较表用全部五条臂的行集，判读段用只含两条臂的行集。`evaluate_pilot.paired` 每次调用重新播种，但按臂序逐块消耗抽样，行集大小不同 ⇒ 同一统计量落在随机流的不同位置 ⇒ 均值相同、分位数边界略有差别。本轮把所有统计量收敛到一个对象，判读段只切片、不再计算。
- 核对 16 对旧/新数值：均值是否改变 = 否（仅分位边界）；差异条目 9 项。
- 旧 COMMUNICATION_LEARNING_VERDICT 与 RESULTS.md §三 引用的 CI（如 [+0.685,+0.990]、[26 018, 26 687]）属于两臂抽样；修订版统一为比较表那一套（[+0.685,+0.995]、[26 002, 26 688]）。原始 3,000 行输入的摘要未变，改判据未发生。

| 设置 | 段落 | 量 | 旧 CI | 新 CI | 均值 |
|---|---|---|---|---|---|
| isolated-lossless | frontier | ack_periodic_20 - ack_retry_2 (on_time) | [-0.1000, 0.1300] | [-0.1000, 0.1250] | 0.0150 = 0.0150 |
| isolated-lossless | negative_control | ack_backoff_16 - noack_send_once (total_bytes) | [10244.5140, 10472.2223] | [10243.4752, 10471.4730] | 10364.3400 = 10364.3400 |
| isolated-lossless | negative_control | ack_periodic_20 - noack_send_once (on_time) | [-0.1000, 0.1300] | [-0.1000, 0.1250] | 0.0150 = 0.0150 |
| isolated-lossless | negative_control | ack_periodic_20 - noack_send_once (total_bytes) | [25137.0323, 26801.2925] | [25126.4450, 26831.2151] | 25981.1650 = 25981.1650 |
| isolated-lossless | negative_control | ack_retry_2 - noack_send_once (total_bytes) | [10244.5140, 10472.2223] | [10240.1077, 10470.6308] | 10364.3400 = 10364.3400 |
| isolated-lossy | lossy_confirmation | ack_retry_2 - noack_send_once (completion) | [0.6850, 0.9900] | [0.6850, 0.9950] | 0.8350 = 0.8350 |
| isolated-lossy | lossy_confirmation | ack_retry_2 - noack_send_once (total_bytes) | [26017.6484, 26687.4026] | [26002.2656, 26687.7902] | 26354.0600 = 26354.0600 |
| isolated-lossy | frontier | ack_periodic_20 - ack_retry_2 (on_time) | [-0.1750, 0.1550] | [-0.1700, 0.1600] | -0.0050 = -0.0050 |
| normal-lossy | normal_path | ack_retry_2 - noack_send_once (total_bytes) | [26017.6484, 26687.4026] | [26002.2656, 26687.7902] | 26354.0600 = 26354.0600 |

- **3,000 行输入摘要不变性核对**：585 个 (设置, 臂, 字段) 均值逐项与归档 results 比对，全部相同。

## 八、复跑与卫生

```bash
cd runs/comm_evidence_review_20260921
python revise_results.py                      # 读 3,000 行 → 修订 results（唯一统计对象）
python revise_verdict.py                     # 由修订 results 渲染 verdict
python evidence_corrections.py               # 十项核对与原文→修订映射
python learning_diagnostic.py                # 仅用已有训练记录做诊断（不跑 episode）
python -m unittest discover -s . -p 'test_*.py' -v
```

本轮实测：0 episode、0 新流、0 训练。`800000..800399` 未生成、未读取、未改动；`900000..900199` 已消费，只作既有证据使用，不作为下一候选的未见测试。生产默认值未改动（`comm_ack=False`、`comm_timing_mode=legacy_shared_bound`）。

### 附录：输入分片

| 文件 | 字节 | sha256 |
|---|---:|---|
| `matrix_shard0.json` | 1,014,331 | `1ad390468c2220ec…` |
| `matrix_shard1.json` | 1,015,445 | `7bc4c85bd8575d5e…` |
| `matrix_shard2.json` | 1,017,191 | `a1aea6eda52cad47…` |
| `matrix_shard3.json` | 1,018,786 | `f69a99cf053fd0d8…` |
| `matrix_shard4.json` | 1,018,113 | `5392f0e2cb8448aa…` |
| `matrix_shard5.json` | 1,016,073 | `cc051f6c27970976…` |
| `matrix_shard6.json` | 1,015,688 | `4538628145a88745…` |
| `matrix_shard7.json` | 1,016,633 | `4e603eb952b234fb…` |

