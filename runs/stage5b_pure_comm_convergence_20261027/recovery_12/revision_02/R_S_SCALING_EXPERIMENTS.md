# R / S 跨规模评估：搜救运力 R 与搜索智能体 S 两条正交轴

零新训练：全部权重都是封板检查点；对比臂六个、场景三个锚定场景（2301901–2301903）、信道 0.7 丢包无 ACK。98 个 pin 文件未改动。

## 口径

| 项 | R 轴 | S 轴 |
|---|---|---|
| 扫描 | `R ∈ {3,6,9,12,15,18}` × `G ∈ {24,36,48}` | `S ∈ {1,2,3,4,6}` × `R ∈ {6,12}` × `G ∈ {24,48}` |
| 扇出 | N=3 跑满六档 R；N=6 只跑 `R≥6`（5 档），故 11 个 (R,N) 组合 | N=6 固定 |
| 世界/点 | 3 档 G × 组合 × 3 场景 | 4 格 × 3 场景 |
| 新增/臂 | **63**（`R∈{3,9,15,18}`；`R=6,12` 复用归档） | **48**（`S∈{1,3,4,6}`；`S=2` 复用归档） |
| 页预算 | `v4t`/`v4s` = 同格 immediate 实测页数 − 1；`slack_matched_rate` = 同格 v4t 实测页数 | 同左，且预算按 **S=2** 测量后在各 S 上保持不变 |

**白名单怎么登记**：`scale_program.RESCUE_EXTENDED = (3,6,9,12,15,18)` 与 `SEARCH_AGENTS_EXTENDED = (1,2,3,4,6)`，各自与训练集合并成 `RESCUE_ACCEPTED` / `SEARCH_AGENTS_ACCEPTED`；`world.verify_resolved_configuration` 经 `cell_expectation_overrides` 放行 `rescue_robots` 与 `search_agents`，`SCALE_EXPECTATION_OVERRIDE_REASONS` 增加 `rescue_robots_extended` / `search_agents_extended` 两条理由。**训练采样网格 `CELLS` 仍是 15 格、`cell_for` 仍只从 15 格里抽**；S 不进入 cell 元组（它作为独立参数随世界传递），所以既有归档的目录名一个都没变。特征 #16 继续按实际 `(R/6)/(G/24)` 计算，R=18 时读出 3.0。

## 表 R1　R 轴：发送量与字节削减（3 个锚定场景，聚合全部 11 个组合）

| R | immediate 发送/世界 | immediate 字节削减 | slack_cap 发送/世界 | slack_cap 字节削减 | slack_matched 发送/世界 | slack_matched 字节削减 | v4f 发送/世界 | v4f 字节削减 | v4s 发送/世界 | v4s 字节削减 | v4t 发送/世界 | v4t 字节削减 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 35.7 | — | 34.7 | +2.8 % | 11.7 | +67.4 % | 11.9 | +68.8 % | 10.3 | +73.4 % | 11.9 | +68.8 % |
| 6 | 35.7 | — | 35.0 | +1.8 % | 17.0 | +51.4 % | 17.6 | +50.9 % | 19.5 | +46.0 % | 17.5 | +51.2 % |
| 9 | 35.7 | — | 34.7 | +2.8 % | 24.7 | +28.7 % | 24.9 | +28.6 % | 31.1 | +9.8 % | 24.6 | +29.8 % |
| 12 | 35.7 | — | 35.0 | +1.8 % | 30.2 | +13.4 % | 30.7 | +12.1 % | 34.8 | +2.2 % | 30.1 | +13.8 % |
| 15 | 35.7 | — | 34.7 | +2.8 % | 34.7 | +2.8 % | 35.7 | +0.0 % | 34.7 | +2.8 % | 34.7 | +2.8 % |
| 18 | 35.7 | — | 34.7 | +2.8 % | 34.7 | +2.8 % | 35.7 | +0.0 % | 34.7 | +2.8 % | 34.7 | +2.8 % |

## 表 R2　按时完成率与相对 immediate 的比值

| R | immediate 准时 | slack_cap 准时 | slack_cap / imm | slack_matched 准时 | slack_matched / imm | v4f 准时 | v4f / imm | v4s 准时 | v4s / imm | v4t 准时 | v4t / imm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 54/324 = 0.1667 | 53/324 = 0.1636 | 0.981 | 20/324 = 0.0617 | 0.370 | 92/324 = 0.2840 | 1.704 | 82/324 = 0.2531 | 1.519 | 92/324 = 0.2840 | 1.704 |
| 6 | 288/648 = 0.4444 | 289/648 = 0.4460 | 1.003 | 137/648 = 0.2114 | 0.476 | 310/648 = 0.4784 | 1.076 | 298/648 = 0.4599 | 1.035 | 309/648 = 0.4769 | 1.073 |
| 9 | 434/648 = 0.6698 | 502/648 = 0.7747 | 1.157 | 370/648 = 0.5710 | 0.853 | 403/648 = 0.6219 | 0.929 | 426/648 = 0.6574 | 0.982 | 396/648 = 0.6111 | 0.912 |
| 12 | 567/648 = 0.8750 | 586/648 = 0.9043 | 1.034 | 511/648 = 0.7886 | 0.901 | 519/648 = 0.8009 | 0.915 | 549/648 = 0.8472 | 0.968 | 510/648 = 0.7870 | 0.899 |
| 15 | 594/648 = 0.9167 | 599/648 = 0.9244 | 1.008 | 599/648 = 0.9244 | 1.008 | 612/648 = 0.9444 | 1.030 | 594/648 = 0.9167 | 1.000 | 594/648 = 0.9167 | 1.000 |
| 18 | 614/648 = 0.9475 | 606/648 = 0.9352 | 0.987 | 606/648 = 0.9352 | 0.987 | 621/648 = 0.9583 | 1.011 | 603/648 = 0.9306 | 0.982 | 603/648 = 0.9306 | 0.982 |

## 表 R3　每完成任务消息数与广播超时浪费率

| R | immediate msg/on | immediate waste | slack_cap msg/on | slack_cap waste | slack_matched msg/on | slack_matched waste | v4f msg/on | v4f waste | v4s msg/on | v4s waste | v4t msg/on | v4t waste |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 5.944 | 0.832 | 5.887 | 0.830 | 5.250 | 0.810 | 1.163 | 0.140 | 1.134 | 0.118 | 1.163 | 0.140 |
| 6 | 2.229 | 0.551 | 2.180 | 0.541 | 2.234 | 0.552 | 1.019 | 0.019 | 1.178 | 0.151 | 1.019 | 0.019 |
| 9 | 1.479 | 0.324 | 1.243 | 0.196 | 1.200 | 0.167 | 1.114 | 0.102 | 1.312 | 0.238 | 1.116 | 0.104 |
| 12 | 1.132 | 0.117 | 1.075 | 0.070 | 1.063 | 0.059 | 1.066 | 0.061 | 1.140 | 0.123 | 1.063 | 0.059 |
| 15 | 1.081 | 0.075 | 1.042 | 0.040 | 1.042 | 0.040 | 1.049 | 0.047 | 1.051 | 0.048 | 1.051 | 0.048 |
| 18 | 1.046 | 0.044 | 1.030 | 0.029 | 1.030 | 0.029 | 1.034 | 0.033 | 1.035 | 0.034 | 1.035 | 0.034 |

## 表 S1　S 轴：发送量与字节削减（3 个锚定场景，4 格）

| S | immediate 发送/世界 | immediate 字节削减 | slack_cap 发送/世界 | slack_cap 字节削减 | slack_matched 发送/世界 | slack_matched 字节削减 | v4f 发送/世界 | v4f 字节削减 | v4s 发送/世界 | v4s 字节削减 | v4t 发送/世界 | v4t 字节削减 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32.0 | — | 32.0 | -0.1 % | 27.8 | +13.0 % | 28.2 | +12.5 % | 26.9 | +16.6 % | 28.2 | +12.5 % |
| 2 | 35.7 | — | 35.0 | +1.8 % | 25.0 | +30.0 % | 25.4 | +29.7 % | 28.1 | +21.9 % | 25.0 | +30.8 % |
| 3 | 36.0 | — | 34.5 | +4.1 % | 22.0 | +39.0 % | 22.4 | +39.0 % | 26.6 | +27.0 % | 21.8 | +40.7 % |
| 4 | 36.0 | — | 34.5 | +3.9 % | 24.0 | +33.5 % | 24.5 | +33.2 % | 26.9 | +26.0 % | 23.8 | +35.2 % |
| 6 | 36.0 | — | 34.5 | +4.0 % | 22.2 | +37.7 % | 23.0 | +37.5 % | 27.2 | +25.0 % | 22.3 | +39.3 % |

## 表 S2　按时完成率与相对 immediate 的比值

| S | immediate 准时 | slack_cap 准时 | slack_cap / imm | slack_matched 准时 | slack_matched / imm | v4f 准时 | v4f / imm | v4s 准时 | v4s / imm | v4t 准时 | v4t / imm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 305/432 = 0.7060 | 303/432 = 0.7014 | 0.993 | 272/432 = 0.6296 | 0.892 | 296/432 = 0.6852 | 0.970 | 285/432 = 0.6597 | 0.934 | 296/432 = 0.6852 | 0.970 |
| 2 | 285/432 = 0.6597 | 301/432 = 0.6968 | 1.056 | 220/432 = 0.5093 | 0.772 | 295/432 = 0.6829 | 1.035 | 285/432 = 0.6597 | 1.000 | 290/432 = 0.6713 | 1.018 |
| 3 | 245/432 = 0.5671 | 249/432 = 0.5764 | 1.016 | 217/432 = 0.5023 | 0.886 | 250/432 = 0.5787 | 1.020 | 239/432 = 0.5532 | 0.976 | 244/432 = 0.5648 | 0.996 |
| 4 | 218/432 = 0.5046 | 217/432 = 0.5023 | 0.995 | 207/432 = 0.4792 | 0.950 | 242/432 = 0.5602 | 1.110 | 218/432 = 0.5046 | 1.000 | 233/432 = 0.5394 | 1.069 |
| 6 | 160/432 = 0.3704 | 210/432 = 0.4861 | 1.312 | 172/432 = 0.3981 | 1.075 | 191/432 = 0.4421 | 1.194 | 175/432 = 0.4051 | 1.094 | 189/432 = 0.4375 | 1.181 |

## 表 S3　每完成任务消息数与广播超时浪费率

| S | immediate msg/on | immediate waste | slack_cap msg/on | slack_cap waste | slack_matched msg/on | slack_matched waste | v4f msg/on | v4f waste | v4s msg/on | v4s waste | v4t msg/on | v4t waste |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.259 | 0.206 | 1.267 | 0.211 | 1.224 | 0.183 | 1.142 | 0.124 | 1.133 | 0.118 | 1.142 | 0.124 |
| 2 | 1.502 | 0.334 | 1.395 | 0.283 | 1.364 | 0.267 | 1.034 | 0.033 | 1.182 | 0.154 | 1.034 | 0.033 |
| 3 | 1.763 | 0.433 | 1.663 | 0.399 | 1.217 | 0.178 | 1.076 | 0.071 | 1.335 | 0.251 | 1.074 | 0.069 |
| 4 | 1.982 | 0.495 | 1.908 | 0.476 | 1.391 | 0.281 | 1.215 | 0.177 | 1.482 | 0.325 | 1.223 | 0.182 |
| 6 | 2.700 | 0.630 | 1.971 | 0.493 | 1.552 | 0.356 | 1.445 | 0.308 | 1.869 | 0.465 | 1.418 | 0.295 |

## R 轴的相变：字节削减与按时率如何随运力移动

| R | v4t 发送/世界 | v4t 字节比 | v4t 准时/immediate | v4t 准时分母 | v4f 字节比 | slack_cap 字节比 |
|---|---|---|---|---|---|---|
| 3 | 11.9 | 0.312 | 1.704 | 92/324 | 0.312 | 0.972 |
| 6 | 17.5 | 0.488 | 1.073 | 309/648 | 0.491 | 0.982 |
| 9 | 24.6 | 0.702 | 0.912 | 396/648 | 0.714 | 0.972 |
| 12 | 30.1 | 0.862 | 0.899 | 510/648 | 0.879 | 0.982 |
| 15 | 34.7 | 0.972 | 1.000 | 594/648 | 1.000 | 0.972 |
| 18 | 34.7 | 0.972 | 0.982 | 603/648 | 1.000 | 0.972 |

| R | N=3 字节比 | N=3 准时比 | N=6 字节比 | N=6 准时比 |
|---|---|---|---|---|
| 3 | 0.312 | 1.704 | n/a | n/a |
| 6 | 0.435 | 0.965 | 0.515 | 1.181 |
| 9 | 0.612 | 0.755 | 0.747 | 1.088 |
| 12 | 0.776 | 0.866 | 0.904 | 0.930 |
| 15 | 0.972 | 0.977 | 0.972 | 1.024 |
| 18 | 0.972 | 0.977 | 0.972 | 0.987 |

## S 轴：搜索智能体增多后，谁先被积压压垮

| S | immediate 发送/世界 | immediate 准时 | v4t 发送/世界 | v4t 准时 | v4t 准时比 | v4t 浪费率 | slack_matched 浪费率 |
|---|---|---|---|---|---|---|---|
| 1 | 32.0 | 305/432 | 28.2 | 296/432 | 0.970 | 0.124 | 0.183 |
| 2 | 35.7 | 285/432 | 25.0 | 290/432 | 1.018 | 0.033 | 0.267 |
| 3 | 36.0 | 245/432 | 21.8 | 244/432 | 0.996 | 0.069 | 0.178 |
| 4 | 36.0 | 218/432 | 23.8 | 233/432 | 1.069 | 0.182 | 0.281 |
| 6 | 36.0 | 160/432 | 22.3 | 189/432 | 1.181 | 0.295 | 0.356 |

## R 轴读出来的相变

| R | 运力画像 | 字节削减 | v4t/immediate 准时 | v4t 浪费率 | immediate 浪费率 |
|---|---|---|---|---|---|
| 3 | 极度紧缺（3 台） | 68.8 % | 1.704 | 0.140 | 0.832 |
| 6 | 训练下限 | 51.2 % | 1.073 | 0.019 | 0.551 |
| 9 | 中低 | 29.8 % | 0.912 | 0.104 | 0.324 |
| 12 | 训练上限 | 13.8 % | 0.899 | 0.059 | 0.117 |
| 15 | 中高 | 2.8 % | 1.000 | 0.048 | 0.075 |
| 18 | 极度充裕 | 2.8 % | 0.982 | 0.034 | 0.044 |

**字节削减随 R 单调不增**：68.8 % → 51.2 % → 29.8 % → 13.8 % → 2.8 % → 2.8 %。运力越紧，v4t 剪得越深（R=3 时 68.8 %，R=18 时 2.8 %）；这与「配额 = immediate−1 页」的设定一起说明，深度节流不是策略在紧缩时误判，而是它在紧缩时唯一能保住准时的做法。

**「少发多救」（v4t 按时数严格多于 immediate）只出现在 R=3（1.704）、R=6（1.073）**；从 `R=9` 起比值降到 1 或以下（R=15 恰为 1.000），即运力一旦够用，v4t 的节流就退化为「同水平略降」而不是反超。这就是 R 轴上的相变位置。

## S 轴读出来的积压抑制

| S | immediate 发送/世界 | immediate 浪费率 | v4t 发送/世界 | v4t 浪费率 | 浪费率差（v4t − immediate） |
|---|---|---|---|---|---|
| 1 | 32.0 | 0.206 | 28.2 | 0.124 | -0.081 |
| 2 | 35.7 | 0.334 | 25.0 | 0.033 | -0.301 |
| 3 | 36.0 | 0.433 | 21.8 | 0.069 | -0.364 |
| 4 | 36.0 | 0.495 | 23.8 | 0.182 | -0.313 |
| 6 | 36.0 | 0.630 | 22.3 | 0.295 | -0.335 |

**搜索智能体越多，immediate 的浪费涨得比 v4t 快**：S 从 1 增到 6 时，immediate 的广播超时浪费率从 0.206 涨到 0.630（+0.424），v4t 从 0.124 涨到 0.295（+0.171）。两者之差在每个 S 上都是负的（v4t 更低），且差距从 -0.081 扩大到 -0.335——这就是「同样预算下，谁把积压挡在了超时之外」的直接读数。注意 v4t 的发送量并不随 S 增长（28.2 → 22.3 条/世界），它与 immediate 的发送量差（3.8 → 13.7）说明它挡住的正是那部分新增广播。

**但这条读法有一处必须写明**：`slack_matched_rate` 在同样的 S 上浪费率也低于 immediate（见上表 S3），所以「浪费率更低」不是 v4t 独有的；v4t 相对它多出来的是**准时数**（表 S2 的 v4t/immediate 比在 S=4、S=6 是1.069 与 1.181，而 slack_matched 是 0.950 与 1.075）。换句话说：抑制浪费是「少发」的普遍后果，而 v4t 的增量价值是**在不发的前提下仍然完成更多**。

## 明确不能说的话

* 两个轴都只有 **3 个锚定场景、无种子重复**；差一个目标的差异不应解读。
* `R=3` 只有 N=3 一个扇出、`R=6..18` 才有 N=6，所以 R 轴在 R=3 处的合并数与其余档位不是同一批 (R,N) 组合；逐档的 N=3 / N=6 分列已经给出，合并列只用于看趋势。
* S 轴上页预算按 **S=2 测量后固定**，所以「v4t 发送量不随 S 变化」是设计的一部分，不是发现；本轴能说的是**同样预算下准时数与浪费率如何随 S 变化**。
* `S=1` 时 `search_agents=1`，PILOC 的观测半径与检测率随之改变，因此 S 轴的差异是「发现过程 + 任务生成」的联合效应，不能单独归因于「积压抑制」。
* 没有任何一档改变了训练分布；`S≠2`、`R∉{6,12}`、`G∉{12,24,48}` 都是零样本外推。

## 复现

```
D:/project_env/python.exe analysis/rs_scaling.py --axis r --stage immediate --workers 8
D:/project_env/python.exe analysis/rs_scaling.py --axis r --stage rules --workers 8
D:/project_env/python.exe analysis/rs_scaling.py --axis r --stage v4f --workers 8
D:/project_env/python.exe analysis/rs_scaling.py --axis r --stage v4s --workers 8
D:/project_env/python.exe analysis/rs_scaling.py --axis r --stage v4t --workers 8
D:/project_env/python.exe analysis/rs_scaling.py --axis r --stage slack_matched --workers 8
D:/project_env/python.exe analysis/rs_scaling.py --axis s --stage immediate --workers 8
D:/project_env/python.exe analysis/rs_scaling.py --axis s --stage rules --workers 8
D:/project_env/python.exe analysis/rs_scaling.py --axis s --stage v4f --workers 8
D:/project_env/python.exe analysis/rs_scaling.py --axis s --stage v4s --workers 8
D:/project_env/python.exe analysis/rs_scaling.py --axis s --stage v4t --workers 8
D:/project_env/python.exe analysis/rs_scaling.py --axis s --stage slack_matched --workers 8
D:/project_env/python.exe analysis/rs_waste.py --axis r --workers 8
D:/project_env/python.exe analysis/rs_waste.py --axis s --workers 8
D:/project_env/python.exe analysis/rs_scaling_report.py
D:/project_env/python.exe analysis/plot_rs_figures.py
```

| 产物 | 路径 |
|---|---|
| R / immediate 世界读数 | `execution/r_s_scaling/r/normal_immediate/RESULTS.json` |
| R / slack_cap 世界读数 | `execution/r_s_scaling/r/slack_heuristic_cap/RESULTS.json` |
| R / slack_matched 世界读数 | `execution/r_s_scaling/r/slack_matched_rate/RESULTS.json` |
| R / v4f 世界读数 | `execution/r_s_scaling/r/v4f_200/RESULTS.json` |
| R / v4s 世界读数 | `execution/r_s_scaling/r/v4s_300/RESULTS.json` |
| R / v4t 世界读数 | `execution/r_s_scaling/r/v4t_100/RESULTS.json` |
| R 轴广播浪费率 | `execution/r_s_scaling/r/WASTE.json` |
| S / immediate 世界读数 | `execution/r_s_scaling/s/normal_immediate/RESULTS.json` |
| S / slack_cap 世界读数 | `execution/r_s_scaling/s/slack_heuristic_cap/RESULTS.json` |
| S / slack_matched 世界读数 | `execution/r_s_scaling/s/slack_matched_rate/RESULTS.json` |
| S / v4f 世界读数 | `execution/r_s_scaling/s/v4f_200/RESULTS.json` |
| S / v4s 世界读数 | `execution/r_s_scaling/s/v4s_300/RESULTS.json` |
| S / v4t 世界读数 | `execution/r_s_scaling/s/v4t_100/RESULTS.json` |
| S 轴广播浪费率 | `execution/r_s_scaling/s/WASTE.json` |
| 本报告的数 | `execution/r_s_scaling/R_S_SCALING.json` |
| 图 5 / 图 6 | `execution/figures/fig5_r_scaling_curve.*`、`fig6_s_scaling_curve.*` |

| 输入 | sha256（前 16 位） |
|---|---|
| `execution/baselines/grid_v2/BASELINES.json` | `95064c8753881114` |
| `execution/g_scaling/normal_immediate/RESULTS.json` | `08a11046335bc75f` |
| `execution/baselines/paper_throttles_v1/BASELINES.json` | `cf7e7187f30a0bc2` |
| `execution/g_scaling/slack_heuristic_cap/RESULTS.json` | `bd9c680f52132e49` |
| `execution/baselines/paper_slack_matched/BASELINES.json` | `3c612890edd56bb1` |
| `execution/g_scaling/slack_matched_rate/RESULTS.json` | `1b45f4522ed69a50` |
| `execution/train_v4f_anchor/eval_200_s8/EVAL.json` | `5dbbbef428a3d8de` |
| `execution/g_scaling/v4f_200/RESULTS.json` | `1b12a1771572a74e` |
| `execution/train_v4s_cap300/eval_300_s8/EVAL.json` | `999fdc7e76909198` |
| `execution/g_scaling/v4s_300/RESULTS.json` | `a131f2cc497458a6` |
| `execution/train_v4t_cap_frozen/eval_100_s8/EVAL.json` | `2e1e8fe3deba51e1` |
| `execution/g_scaling/v4t_100/RESULTS.json` | `2af92701d29616ed` |
| `execution/r_s_scaling/r/normal_immediate/RESULTS.json` | `9abdb75ccb885381` |
| `execution/r_s_scaling/r/slack_heuristic_cap/RESULTS.json` | `789551ef3b916849` |
| `execution/r_s_scaling/r/slack_matched_rate/RESULTS.json` | `00ec3bd316cc528d` |
| `execution/r_s_scaling/r/v4f_200/RESULTS.json` | `8a87ba5efc3cb951` |
| `execution/r_s_scaling/r/v4s_300/RESULTS.json` | `9cd73ccbb71be956` |
| `execution/r_s_scaling/r/v4t_100/RESULTS.json` | `208a874b63d30a34` |
| `execution/r_s_scaling/s/normal_immediate/RESULTS.json` | `74def8c35683cfc4` |
| `execution/r_s_scaling/s/slack_heuristic_cap/RESULTS.json` | `f3d9046803077e5c` |
| `execution/r_s_scaling/s/slack_matched_rate/RESULTS.json` | `588b488c6926b93e` |
| `execution/r_s_scaling/s/v4f_200/RESULTS.json` | `70ec088130fdba8c` |
| `execution/r_s_scaling/s/v4s_300/RESULTS.json` | `7ffe0b5a263481f2` |
| `execution/r_s_scaling/s/v4t_100/RESULTS.json` | `930cabea24a8ad61` |

## 图片（`analysis/plot_rs_figures.py` 生成，零重跑）

IEEE 风格与图 1–5 一致：西文 Times New Roman、中文 SimSun、数学符号 STIX（字形已嵌入，`pdf.fonttype = 42`）；四边闭合 0.8 pt 黑框、刻度朝内、浅灰虚线网格；无图内标题/气泡/点值标注，子图编号写在横轴标签内；图例为细黑框、置于图上方；Ours-Frozen 为红色实心点，Pure-Policy 为更大的空心蓝环且 zorder 更高，两者重合时读作「红点套蓝环」。R 轴的横轴是**连续数值轴**（步长恒为 3，不需要人为等距），S 轴的横轴是**等距类别轴**（1/2/3/4/6 不等距，线性轴会把最后一档拉宽一倍）。

| 图 | 文件 | 像素 | sha256（前 16 位） |
|---|---|---|---|
| fig5_r_scaling_curve | `execution/figures/fig5_r_scaling_curve.png` | 2193×1806 | `a92007fb5c8d40ef` |
| fig5_r_scaling_curve | `execution/figures/fig5_r_scaling_curve.pdf` |  | `61a7fa5d6e432377` |
| fig6_s_scaling_curve | `execution/figures/fig6_s_scaling_curve.png` | 2193×1806 | `cf0747278533b744` |
| fig6_s_scaling_curve | `execution/figures/fig6_s_scaling_curve.pdf` |  | `7765ed2f8a00ece0` |

