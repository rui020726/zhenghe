# 最终总结：任务约束下的通信降本（stage5b · recovery_12/revision_02）

**一句话**：在**预先注册**的严苛判据下，带页配额表的 `v4t@100` 最好（9/15）；在本轮事后定义的**合理判据**
（逐格保准时 + 全局降字节与降浪费）下，**无配额表的纯策略网络 `v4f@200` 最好（12/15）**，且它不需要任何
配额表、任何新增训练。两个结论同时成立，差别只在“字节条款放在哪一层”。

- 数据：全部来自已封板权重与已归档世界；本轮**零新训练、零新世界仿真**。
- 冻结：98/98 个 pin 文件 0 drift；门禁 106 项通过。
- 复现入口：`analysis/dual_gate_report.py`、`analysis/plot_dual_gate_figure.py`、`analysis/export_latex_tables.py`。

---

## 1. 两个判据（必须分开读）

| | 旧判据（预先注册） | 新判据（本轮事后定义，**探索性**） |
|---|---|---|
| 逐格准时 | `on_time ≥ ⌈0.95 × immediate⌉` | 同左 |
| 字节 | **每格** `bytes < immediate` | **全局** 总字节比 < 0.85 |
| 浪费 | — | **全局** 广播浪费率 < 0.10 |

新判据是本轮才定义的（阈值在看到数据之后选定），因此**不是**预先注册的判定；论文里只能写成
“探索性的判据复核”，不能写成“通过了注册判据”。预先注册的结论不变：v4f 3/15、v4t 9/15。

## 2. 主表（15 格 × 3 锚定场景 = 45 世界/臂）

| 方法 | 旧判据 | 准时条款 | 新判据 | 按时/imm | 字节/imm | 条/任务 | 浪费率 |
|---|---|---|---|---|---|---|---|
| Immediate | 0/15 | 15/15 | 0/15 | 1.000 | 1.000 | 1.396 | 0.284 |
| Slack-Cap | 11/15 | 11/15 | 0/15 | 1.063 | 0.976 | 1.282 | 0.220 |
| Slack-Matched | 6/15 | 6/15 | 0/15 | 0.869 | 0.806 | 1.177 | 0.150 |
| v4v_pure（价格衰减重训） | 0/15 | 12/15 | 0/15 | 1.016 | 1.000 | 1.374 | 0.272 |
| v4t@100（网络 + 配额表） | 9/15 | 9/15 | 9/15 | 0.972 | **0.800** | 1.052 | **0.049** |
| **v4f@200（纯网络）** | 3/15 | 12/15 | **12/15** | **0.984** | 0.820 | **1.064** | 0.060 |

“旧判据”为预先注册；“新判据”为本轮事后定义。分母：每臂 45 个世界、1 260 个目标；
比值以同一批世界里的 immediate 为分母。

**为什么纯网络能赢**：它有 9 个格子被判“不过”，原因**只是字节恰好等于 immediate**——在这些轻载格
（G12 全部 5 格、G24_R12 三格、G48_R12_N12）里，“把每条都发出去”本来就是正确做法，旧判据却把它
当成失败。把成本条款移到全局之后，这 9 格全部转为通过；它真正过不了的只有 3 格：
`G24_R6_N3`（33/47）、`G48_R12_N3`（72/106）、`G48_R12_N6`（108/114）。

## 3. 消融链

| 步骤 | 配置 | 新判据 | 字节/imm | 浪费 | 读数 |
|---|---|---|---|---|---|
| (a) | 纯网络、无配额（v4f@200） | **12/15** | 0.820 | 0.060 | 基准 |
| (b) | + 硬配额表（v4t@100） | 9/15 | 0.800 | 0.049 | 用 3 格换 2.4 % 字节；G=12 上剪 1 页就少 1–2 个目标 |
| (c) | + 强制发送下限（v4u@100） | 0/15 | 0.968 | 0.279 | 补发后期消息只花钱：+16.8 % 字节、5.7× 浪费，一格没买到 |
| (d) | 价格衰减重训（v4v_pure） | 0/15 | 1.000 | 0.272 | 去掉成本项 → 退回全量广播（1 245 条 = immediate） |
| (e1) | 屏蔽 #16（J1） | — | — | 0.119 | 按时 −96、发送 −37：容量列承载决策 |
| (e2) | 屏蔽 #2 / #11 退回 50（J1） | — | — | — | 与完整网络**逐世界相同**，两列都不影响决策 |
| (e3) | #16 覆写 2.0（J2） | — | — | — | 六个 R=12 格：用量 52–83 % → 99–100 %，按时 5/6 格上升 |

(a)→(d) 的结论是一致的：**“砍多少”靠执行器，“砍哪条”才靠网络**。把成本项关掉，网络不会变聪明，
只会变回 immediate；把页预算加回来，能省 2.4 %，但离散粒度会在 G=12 反噬。

## 4. 五个机理探针（零新训练）

| 探针 | 规模 | 关键读数 |
|---|---|---|
| I 信道鲁棒性 | 1 008 世界 | v4t 在 p ≥ 0.5 保持 0.94–1.02× 按时、2.7–3.0 kB/按时任务；p=0.3 首跌破 0.95（0.939）。event_maxwait 按时最高（1.15×）但 35 kB/任务 |
| J1 特征屏蔽 | 225 世界 | #16 −96 按时、#14 +71 但字节 +11.7 %、#17 +42 且 −1.7 %；#2 与 #11 零差异（附 50/200 安装探针） |
| J2 容量反事实 | 36 世界 | #16→2.0 后发送量全部推回页预算（+8…+34 条/世界），证实「高 G 自发节流」由 #16 控制 |
| K 配额谱系 | 576 世界 | R=6 上 40 % 配额即达 1.021×（字节 0.355），三个规则选择器同配额只有 0.667/0.809/0.844；满配额时规则反超 |
| L 微观画像 | 0 世界（纯日志） | v4t 广播 2 408 个目标 94.1 % 按时，未广播 937 个**零**按时；发送延迟恒 0（发现即决定） |
| M 实时性 | 6 世界 | G 12→144：0.32→0.72 ms/决策（P99 ≤ 1.01 ms），5 635 参数 / 22 kB |

## 5. 能说 / 不能说

**能说**

1. 旧判据把“该全发的格”判成失败：v4f 有 9 格仅因 `bytes == immediate` 被否，而按时全部达标。
2. 在“逐格保准时 + 全局降成本”的判据下，无配额表的纯网络 12/15，优于配额表 9/15、Slack-Cap 11/15。
3. 配额表买到的是 2.4 % 字节与更低的浪费（0.049 vs 0.060），代价是 3 个通过格。
4. 纯网络的单任务开销已经是全场最低之一（1.064 条/任务、2.9 kB/任务），且浪费率只有 immediate 的 21 %。
5. 关掉成本项会让策略退回全量广播；特征 #2 / #11 在本平台上不影响任何决策。

**不能说**

1. **不能说新判据是“注册判据”**：阈值是在看到数据之后定的，必须标为探索性；预先注册的结论仍是 v4f 3/15、v4t 9/15。
2. **不能说纯网络在按时数上全场最好**：Slack-Cap 1.063× 更高，event_maxwait 1.15× 更高；纯网络的 12/15 来自“不误伤轻载格”，不是“完成得更多”。
3. **不能说配额没有必要**：若部署需要一个硬字节上限，v4t 仍是唯一同时满足 9/15 与 0.800 的臂；反过来，纯网络需要你接受 0.820 而不是 0.800。
4. **不能说 v4v 证明了“无约束更好”**：它只证明了“去掉成本项会退化为广播全量”，这是负结果，不是替代方案。
5. **不能把 12/15 与 9/15 当成同一条曲线上的两点**：一个来自新判据、一个在新判据下仍是 9/15，但两者的通过集合不同（v4f 的 3 个失败格与 v4t 的 6 个不同）。

## 6. v4f-only 图集（唯一学习臂 = v4f@200）

按“只保留 v4f、隐藏其他 v4 检查点”的要求重绘，输出在 `execution/figures/v4f_only/`；
已发布的 fig1–17 与其哈希清单保持不动。每张图都只含 `Immediate / Slack-Cap / Slack-Matched / v4f`。

| 图 | 覆盖的实验 | 四个子图 | 关键读数 |
|---|---|---|---|
| `v4f_g_scaling` | G 十档（12–72）× 5 组 (R,N) | 字节削减 / 相对按时 / 每任务字节 / 浪费 | G=72 省 45.9 %、按时 1.056；G=54 是低点（0.831） |
| `v4f_dual_gate` | 15 格 × 3 锚定场景 | 逐格通过数 / 相对按时 / 字节比 / 浪费 | **12/15**、0.984、0.820、0.060（Slack-Cap 11/15、6/15 见 0.869） |
| `v4f_budget_pareto` | 2 格 × 9 档配额 × 4 选择器 | R=6 与 R=12 的字节—按时轨迹 | R=6 上 40 % 配额即 1.021×（0.355 字节）；50 % 以上 v4f 不再多花 |
| `v4f_g_rs_grid` | 深档 G 72–144 × R × S | R 轴准时/字节、S 轴准时/字节 | R=6 1.985×（0.346 字节）；R=18 0.816；S 1→4：0.940→1.224 |
| `v4f_master_g` | A/B/C 三套网格同一条 G 轴 | 相对按时 / 字节比 | 12→144 全档单调省字节（1.00→0.40），按时在 0.87–1.23 之间 |
| `v4f_r_s_scaling` | R 3–18、S 1–6 两条正交轴 | R 轴准时/字节、S 轴准时/字节 | R=3 1.70×、R≥9 ≤1.0；S=1 0.94 → S=6 1.19 |

**未纳入本图集的一项**：投递率扫描（实验 I）当时只跑了 immediate / dedup / event_maxwait /
slack_matched / v4s / v4t，**没有 v4f 的世界**，所以无法在不重跑的前提下加入；该实验的图仍是
`fig15_delivery_prob_sweep`（含 v4s/v4t），仅在 `EXTENDED_MECHANISM_EXPERIMENTS.md` 中引用。

| 文件 | PNG sha256 |
|---|---|
| `v4f_g_scaling.png` / `.pdf` | `303a1a9463f4e148` / `5b8af774077379fb` |
| `v4f_dual_gate.png` / `.pdf` | `07b8a233c6185d51` / `f0d5c9661898c137` |
| `v4f_budget_pareto.png` / `.pdf` | `9aaa8b854efb1a52` / `870693c2ca3897a4` |
| `v4f_g_rs_grid.png` / `.pdf` | `a3184ef5a91357cc` / `2be852fc65d4d45d` |
| `v4f_master_g.png` / `.pdf` | `fda92c6bf979c102` / `b37030a00deb519e` |
| `v4f_r_s_scaling.png` / `.pdf` | `d1b24d5cfda0483e` / `fccc05b5547b35bc` |

复现：`D:/project_env/python.exe analysis/plot_v4f_only_figures.py`

## 7. 交付物

| 类别 | 文件 |
|---|---|
| 图 | `execution/figures/fig17_dual_gate_ablation.{png,pdf}`（300 dpi + 矢量） |
| LaTeX | `execution/latex/table1_main_comparison.tex`、`table2_ablation_mechanism.tex`（booktabs 三线表） |
| 数据 | `execution/relaxed_gate/{DUAL_GATE.md,DUAL_GATE.json,RELAXED_GATE.json}` |
| 报告 | `EXTENDED_MECHANISM_EXPERIMENTS.md`（I/J/K/L/M）、`PAPER_EXPERIMENTS.md`（14 臂主表）、`G_SCALING_EXPERIMENTS.md`、`R_S_SCALING_EXPERIMENTS.md`、`G_RS_GRID_EXPERIMENTS.md`、`MASTER_TABLE.md`、`OVERVIEW_EXPERIMENTS.md` |
| 代码 | `analysis/{dual_gate_report.py,relaxed_gate.py,plot_dual_gate_figure.py,export_latex_tables.py,extended_*.py,micro_profile.py}` |

## 8. 复现

```bash
D:/project_env/python.exe analysis/dual_gate_report.py      # 双口径全表 + JSON
D:/project_env/python.exe analysis/plot_dual_gate_figure.py # fig17
D:/project_env/python.exe analysis/export_latex_tables.py   # 两张 tex
D:/project_env/python.exe -B -m unittest discover -s tests  # 106 项门禁
D:/project_env/python.exe analysis/verify_pins.py           # 98/98, 0 drift
```

## 9. 证据哈希（前 16 位）

| 文件 | sha256 |
|---|---|
| `execution/relaxed_gate/DUAL_GATE.md` | `6cace30a7f21e3e6` |
| `execution/relaxed_gate/DUAL_GATE.json` | `d1f7736dadcd80d3` |
| `execution/relaxed_gate/RELAXED_GATE.json` | `f107f7947b45967b` |
| `execution/figures/fig17_dual_gate_ablation.png` | `c31e109d7fc96ab4` |
| `execution/figures/fig17_dual_gate_ablation.pdf` | `026f7a48ca843eaa` |
| `execution/latex/table1_main_comparison.tex` | `304efcaeb12e47f8` |
| `execution/latex/table2_ablation_mechanism.tex` | `dd2ce8eb93803dd6` |
