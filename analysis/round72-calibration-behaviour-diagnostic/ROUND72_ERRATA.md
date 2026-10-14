# ROUND72 更正记录（Errata）— 最终验收合并收口

起点 6d28f40afcac8e2818fa7c1bc6f3f6f5c5bf8519；保留该交付与全部历史原件，仅新增本更正。零环境：0 构造、0 模型加载、0 前向、0 采样/恢复、0 autograd/optimizer/PPO、0 checkpoint、0 新 claim。机器验收包见 ROUND72_ERRATA.json。

## 更正 1：PRIMARY 分母按 mode 拆分

1098 个自由首报决策是两种 mode 合计，不是“greedy 1098”。greedy 与 sampled 各 96 格、各 549 个自由决策（early 402、late 147）。

- greedy：96 格，steps/features/actions 96/96 全同，argmax 翻转 0、实际 action 翻转 0；late 147/147 概率变化、early 逐比特相同。greedy 的任务/成本在它自己的 96 格上与 R68 一致。
- sampled：96 格，features_equal 83/96、steps_equal 90/96、实际 action 翻转 14、argmax 翻转 0。
- sampled 的特征并非全等，因此 sampled 的概率差不能全部视为“同一状态下的纯参数变化”，也不能据此定量排除采样方差；保留“概率确实变化、实际行为确实不同”的描述证据。192 是总评估矩阵，sampled 确实不同。

## 更正 2：预算审计改为实测归因

budget_bytes=29300 现由真实 episode row 带内证实。从真实 row 归约 spent_bytes、budget_used_fraction、refusal_sequence、cost_stream 的 budget_refused_sends/capacity_refused_sends、collector_summary 的 mask 计数；executor 拒付与 collector 预 mask 两个通道都已核查。三个 split：dev-eval 学习臂 192 行、dev-eval immediate 33 行、训练 384 行；三者 budget_refused_sends=0、capacity_refused_sends=0、refusal_sequence 空、masked_budget_or_capacity=0。最大 used_fraction：学习 0.354、immediate 0.542、训练 0.341。原“spent<29300 推 NOT binding”升级为“实测零拒付、零 mask”；训练块（1300000..1300047）原件确含这些字段，非缺失。以上为实测零，不是以 0 代替缺失字段。

## 更正 3：teacher 与计费口径

70693 是 R71 离线拟合的前向计费总量（含门禁/诊断前向），不是 teacher/fit 数据本身。teacher 原规模 32046 joint / 58928 target（委办提供，未从定位字段独立复算）。11765 历史 per-call 归属缺口原样保留，不补造旧日志。R70 拟合来源 1300000..1300047 照旧。

## 更正 4：“边界余量”措辞限定

“机制是边界余量，不是饱和”过强。更正为：本次实际更新未跨 p=0.5，这是一个边界余量的描述；单凭此不能排除饱和、梯度噪声或观测问题；late 概率最高接近 0.99。beta=1.00 先验已证实只在训练损失中，推理路径无附加先验 logit。

## R71 逐 seed 主 task/cost（NEW_EP128 减 R68_TRAINED_ALPHA）

- greedy：s11/s22/s33 任务差 0、成本差 0、无目标丢失（各自 96 greedy 格与 R68 一致）。
- sampled：s11 成本 -155.8B/场景（CI 含 0）、s22 +331.1B/场景（CI 不含 0）、s33 -642.2B/场景且丢失 1 目标；三 seed 方向不一，与逐决策 late 概率位移相符，不全归因采样方差。

## 消费与 HEAD

本轮实际消费全为 0。证据 HEAD fe53d476efb56baaae194d68d7f749a546b14389；交付 HEAD 6d28f40afcac8e2818fa7c1bc6f3f6f5c5bf8519；本 erratum 为其上新本地提交。

## 追加更正 2b：预算/拒付字段的逐行覆盖收据（coverage receipt）

新增 ROUND72_COVERAGE_RECEIPT.json（由 r72_coverage.py 生成，零环境只读）。逐行核验 609 个真实 episode row 的字段“存在性 + 类型”，只在字段确实存在处累计数值：

- executor 拒付通道（refusal_sequence、cost_stream.*.budget_refused_sends / capacity_refused_sends）：三个 split 的每一行都以 list/dict 存在，且全部实测为 0（学习 192、immediate 33、训练 384；分别覆盖 192000/33000/384000 步，缺字段步数均为 0）。这是“实测零且字段在场”，不是以 0 代替缺失字段。
- collector 预 mask 通道（one_shot_counts.masked_budget_or_capacity、eligibility_counts.{budget,capacity}_masked_first_reports）：学习 192 行与训练 384 行运行了 learned collector（collector_summary 为 dict），字段在场且实测 0；immediate 33 行是规则臂，collector_summary 为 None，从未运行 learned collector，故该通道记为 N/A，不是“实测 0”。更正 2 中 budget_agg 对 immediate 报告的两个 mask 计数 0 是 .get 默认值，由本收据重新定性为 N/A；immediate 的 executor 拒付 0 则确为实测（字段在场）。
- immediate 行组成更正：33 行 = 32 个开发块场景（1800000..1800031）各一行 + 1 个 1800000 的重复重跑（run immediate_a2，spent 15876，与 immediate_a1 逐字节相同）。该 glob 中不存在 1700027 焦点行（focal_scene_1700027_rows=0），早前“焦点行”的假定据此更正；重复重跑不得并入按场景均值。

以上不改变任何科学结论，仅把“实测零、非缺失”的措辞收紧到确有字段覆盖的通道；机器锁定见 test_r72_coverage_checks.py。