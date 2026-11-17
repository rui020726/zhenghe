# 论文交付清单（PAPER_READY_MANIFEST）

**范围**：论文的图表与正文只展示一个学习模型——`v4f@200`（同一份权重，`cap=None`），统一记作 **本文方法 (Ours)**；十三个基线照旧（其中四个是具名经典算法，LinUCB-Bandit 是唯一的轻量学习基线，作为对照而非第二个“本文模型”）。`v4t / v4s / v4g / v4u / v4v` 只存在于审计文档（`PAPER_EXPERIMENTS.md`、`EXTENDED_MECHANISM_EXPERIMENTS.md`、`PROTOCOL_METRICS_REPORT.md`），论文不引用这些代号。

## 1　LaTeX 表

| 文件 | 内容 | sha256（前 16 位） |
|---|---|---|
| `execution/latex/table1_main_comparison.tex` | 表 1：13 基线（含 4 个具名经典算法）+ 本文方法，七项指标 | `88f27c5d7be20be2` |
| `execution/latex/table3_orthogonal_comparison.tex` | 表 3：特征集 × 决策机制的正交对照 | `8f0a4ac856dcc26d` |
| `execution/latex/table2_ablation_mechanism.tex` | 表 2：本文方法自身的特征消融（上）与 #16 容量反事实（下） | `cc5c58fa2eebe2da` |
| `execution/latex/table4_scale_classical.tex` | 表 4：G ≥ 48 两套规模网格上经典四臂 vs 本文方法（工作点冻结迁移） | `9a56d1ea398a896d` |

表 1 的判据说明（与文件内脚注一致）：头条“按时达标格数”按**放宽判据**（逐格 `on_time ≥ ⌈0.95×Immediate⌉` 且全网字节比 `< 0.85`、浪费率 `< 0.10`）统计，本文方法 12/15；脚注同时给出**严格双条件**（另需每格字节更少）的通过数 3/15 与十三个基线的读数。

## 2　论文图

| 图 | 文件（PNG 300 dpi / PDF 矢量） | PNG sha256 |
|---|---|---|
| 图 1：G 扩展四联（字节/按时/消息/浪费） | `execution/figures/fig1_g_scaling_efficiency.png`<br>`execution/figures/fig1_g_scaling_efficiency.pdf` | `7345f919b3f345b8` |
| 图 2：R=6 相变与 R6/R12 对照 | `execution/figures/fig2_r6_phase_transition.png`<br>`execution/figures/fig2_r6_phase_transition.pdf` | `8a7633ae5ca6667e` |
| 图 3：三格位下的发送量与按时率 | `execution/figures/fig3_throttling_attribution.png`<br>`execution/figures/fig3_throttling_attribution.pdf` | `0fb143a26a9e8d0a` |
| 图 4：效率—按时权衡（G=72 汇总） | `execution/figures/fig4_efficiency_pareto_all8.png`<br>`execution/figures/fig4_efficiency_pareto_all8.pdf` | `4b722d408788652c` |
| 图 5：绝对按时完成率随 G | `execution/figures/fig5_absolute_on_time.png`<br>`execution/figures/fig5_absolute_on_time.pdf` | `6c891fd78d05fdec` |
| 图 15：投递率 8 档鲁棒性 | `execution/figures/fig15_delivery_prob_sweep.png`<br>`execution/figures/fig15_delivery_prob_sweep.pdf` | `f2ab4bc7cdca46eb` |
| 图 16：外加配额约束下的排序器对比 | `execution/figures/fig16_budget_spectrum_pareto.png`<br>`execution/figures/fig16_budget_spectrum_pareto.pdf` | `f6d116c49a5d75c6` |
| 图 18：下游响应时延与有效吞吐 | `execution/figures/fig18_queueing_delay_and_goodput.png`<br>`execution/figures/fig18_queueing_delay_and_goodput.pdf` | `afdca8333c161974` |
| 图 19：G ≥ 48 两套规模网格上 RED/Token/AHP vs 本文方法（按时与字节四联；LinUCB 只入表与文档） | `execution/figures/fig19_scale_classical_comparison.png`<br>`execution/figures/fig19_scale_classical_comparison.pdf` | `68f05efffa20a7f4` |
| 图 20：G = 24…108 成功率与信道使用次数/世界（单轴拼接、口径切换处标注，五臂；LinUCB 只入表与文档） | `execution/figures/fig20_success_and_channel_uses.png`<br>`execution/figures/fig20_success_and_channel_uses.pdf` | `7f5dc811ed07432e` |

**不进入论文的图（内部审计）**：

| 图 | 说明 |
|---|---|
| `execution/figures/fig11_master_all_experiments` | 三套实验分列总图（内部审计） |
| `execution/figures/fig12_master_unified_g` | 15 档同轴总图（内部审计） |
| `execution/figures/fig7_g_rs_grid_r` | 深档 G×R 网格（内部审计） |
| `execution/figures/fig8_g_rs_grid_s` | 深档 G×S 网格（内部审计） |
| `execution/figures/fig9_g_rs_compare_r` | 深档六臂对照 R（内部审计） |
| `execution/figures/fig10_g_rs_compare_s` | 深档六臂对照 S（内部审计） |
| `execution/figures/fig13_absolute_totals` | 绝对量总图（内部审计） |
| `execution/figures/fig14_absolute_per_task` | 每任务绝对量（内部审计） |

## 3　核心统计结果

配对检验：本文方法 对 八个对照，4 指标 × 3 场景组 = 96 项，分两族（已登记基线 / 具名经典算法）；Wilcoxon 为主检验，Holm 在每族每组（16 项）内校正。下表取 `all(8)` 组：

| 对照 | 指标 | 均值差 | 95% CI | Wilcoxon p | Holm p |
|---|---|---|---|---|---|
| normal_immediate | 按时完成率 | -0.0250 | [-0.0524, +0.0024] | 1.40e-01 | 5.62e-01 |
| normal_immediate | 字节比 | -0.1834 | [-0.2324, -0.1345] | 5.24e-08 | 7.86e-07 |
| normal_immediate | 每完成任务消息数 | -0.4558 | [-0.6503, -0.2614] | 5.53e-08 | 7.86e-07 |
| normal_immediate | 广播超时浪费率 | -0.1493 | [-0.1965, -0.1022] | 1.55e-08 | 2.48e-07 |
| slack_heuristic_cap | 按时完成率 | -0.0245 | [-0.0555, +0.0065] | 9.76e-01 | 9.76e-01 |
| slack_heuristic_cap | 字节比 | -0.1407 | [-0.1923, -0.0890] | 2.47e-01 | 7.40e-01 |
| slack_heuristic_cap | 每完成任务消息数 | -0.3544 | [-0.5572, -0.1516] | 7.93e-05 | 8.72e-04 |
| slack_heuristic_cap | 广播超时浪费率 | -0.1105 | [-0.1591, -0.0618] | 3.81e-05 | 4.57e-04 |
| slack_matched_rate | 按时完成率 | +0.0594 | [+0.0323, +0.0864] | 2.61e-07 | 3.39e-06 |
| slack_matched_rate | 字节比 | +0.0285 | [+0.0076, +0.0494] | 1.80e-04 | 1.80e-03 |
| slack_matched_rate | 每完成任务消息数 | -0.2173 | [-0.3644, -0.0702] | 2.40e-03 | 1.44e-02 |
| slack_matched_rate | 广播超时浪费率 | -0.0814 | [-0.1277, -0.0350] | 1.20e-03 | 9.60e-03 |
| random_matched_rate | 按时完成率 | +0.0418 | [+0.0178, +0.0659] | 3.60e-04 | 3.24e-03 |
| random_matched_rate | 字节比 | +0.0135 | [-0.0110, +0.0381] | 2.98e-01 | 7.40e-01 |
| random_matched_rate | 每完成任务消息数 | -0.1053 | [-0.2054, -0.0052] | 3.54e-03 | 1.77e-02 |
| random_matched_rate | 广播超时浪费率 | -0.0477 | [-0.0804, -0.0150] | 1.82e-03 | 1.28e-02 |
| red_aqm | 按时完成率 | +0.0545 | [+0.0237, +0.0853] | 1.74e-04 | 1.91e-03 |
| red_aqm | 字节比 | -0.0423 | [-0.0872, +0.0027] | 8.41e-02 | 5.04e-01 |
| red_aqm | 每完成任务消息数 | -0.3913 | [-0.5638, -0.2188] | 8.98e-07 | 1.17e-05 |
| red_aqm | 广播超时浪费率 | -0.1187 | [-0.1663, -0.0712] | 1.32e-06 | 1.59e-05 |
| token_bucket | 按时完成率 | +0.3887 | [+0.3546, +0.4228] | 6.23e-21 | 9.35e-20 |
| token_bucket | 字节比 | +0.0258 | [-0.0258, +0.0774] | 7.99e-01 | 1.00e+00 |
| token_bucket | 每完成任务消息数 | -2.0301 | [-2.9630, -1.0973] | 1.25e-19 | 1.75e-18 |
| token_bucket | 广播超时浪费率 | -0.4582 | [-0.4973, -0.4191] | 2.27e-21 | 3.63e-20 |
| ahp_utility | 按时完成率 | +0.0318 | [+0.0101, +0.0534] | 2.70e-03 | 2.16e-02 |
| ahp_utility | 字节比 | -0.0063 | [-0.0319, +0.0194] | 9.76e-01 | 1.00e+00 |
| ahp_utility | 每完成任务消息数 | -0.1247 | [-0.2249, -0.0244] | 3.73e-04 | 3.36e-03 |
| ahp_utility | 广播超时浪费率 | -0.0552 | [-0.0870, -0.0234] | 2.63e-04 | 2.63e-03 |
| linucb_bandit | 按时完成率 | -0.0099 | [-0.0306, +0.0108] | 1.50e-01 | 7.52e-01 |
| linucb_bandit | 字节比 | -0.0197 | [-0.0427, +0.0032] | 4.50e-02 | 3.15e-01 |
| linucb_bandit | 每完成任务消息数 | -0.0134 | [-0.0658, +0.0390] | 3.82e-01 | 1.00e+00 |
| linucb_bandit | 广播超时浪费率 | -0.0112 | [-0.0335, +0.0111] | 3.25e-01 | 1.00e+00 |

**读表注意（字节比 vs Slack-Cap）**：均值差 −0.141，但符号秩 p=0.247（Holm 0.740）。原因是差值按格聚成两簇：运力宽松的格子里本文方法字节比≈1.0（比 Slack-Cap 高约 0.03），拥塞格子里≈0.35（低约 0.6），符号在格之间翻转，而检验把 120 个世界当作独立样本。该差异应按逐格表读，不能按单一 p 值读。

高负载（R=6，G≥36，71 世界）下游响应时延 `D_resp` 的配对 Wilcoxon：

| 对照 | 世界 | 均值差（步） | 95% 分位区间 | p |
|---|---|---|---|---|
| Slack-Matched − Immediate | 70 | -17.93 | [-48.50, +19.88] | 1.96e-09 |
| 本文方法 − Immediate | 71 | -40.37 | [-69.87, -14.02] | 2.43e-13 |
| 本文方法 − Slack-Matched | 70 | -22.32 | [-47.53, +5.36] | 3.82e-12 |
| v4t − Immediate（审计） | 71 | -40.37 | [-69.87, -14.02] | 2.43e-13 |
| v4t − Slack-Matched（审计） | 70 | -22.32 | [-47.53, +5.36] | 3.82e-12 |

协议指标核验：`PDR_link` 的八档实测值与设定 p 的偏差 ≤ 0.013（见 `PROTOCOL_METRICS_REPORT.md` §5）；`E_task` 取 `α_tx=1.0, α_rx=0.5`。

## 4　封板状态

- 门禁：`python -B -m unittest discover -s tests` → 112 项通过；
- 冻结件：`analysis/verify_pins.py` → 98/98，drift 0；
- 论文数据来源：`execution/PAPER_TABLE.md`（机读主表）、`execution/PROTOCOL_METRICS.json`、`execution/PAIRED_TESTS.json`、`execution/extended/EXTENDED.json`。

