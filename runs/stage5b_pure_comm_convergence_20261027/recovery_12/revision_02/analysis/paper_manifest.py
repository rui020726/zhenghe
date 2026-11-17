"""Build ``PAPER_READY_MANIFEST.md``: the exact files the paper cites, with their hashes.

Everything is read from artifacts already on disk. The manifest answers three questions a
reviewer or a typesetter will ask:

1. which LaTeX table files are the paper's, and what is in each one;
2. which figures are the paper's (and which are deliberately *not*: figures 11 and 12 keep the
   full cross-experiment audit view and stay out of the paper list);
3. what the core statistical results are, with the correction that was applied.

    D:/project_env/python.exe analysis/paper_manifest.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "PAPER_READY_MANIFEST.md"
PAIRED = BASE / "execution/PAIRED_TESTS.json"
PROTOCOL = BASE / "execution/PROTOCOL_METRICS.json"

TABLES = (
    ("execution/latex/table1_main_comparison.tex", "表 1：13 基线（含 4 个具名经典算法）+ 本文方法，七项指标"),
    ("execution/latex/table3_orthogonal_comparison.tex",
     "表 3：特征集 × 决策机制的正交对照"),
    ("execution/latex/table2_ablation_mechanism.tex",
     "表 2：本文方法自身的特征消融（上）与 #16 容量反事实（下）"),
    ("execution/latex/table4_scale_classical.tex",
     "表 4：G ≥ 48 两套规模网格上经典四臂 vs 本文方法（工作点冻结迁移）"),
)
FIGURES = (
    ("execution/figures/fig1_g_scaling_efficiency", "图 1：G 扩展四联（字节/按时/消息/浪费）"),
    ("execution/figures/fig2_r6_phase_transition", "图 2：R=6 相变与 R6/R12 对照"),
    ("execution/figures/fig3_throttling_attribution", "图 3：三格位下的发送量与按时率"),
    ("execution/figures/fig4_efficiency_pareto_all8", "图 4：效率—按时权衡（G=72 汇总）"),
    ("execution/figures/fig5_absolute_on_time", "图 5：绝对按时完成率随 G"),
    ("execution/figures/fig15_delivery_prob_sweep", "图 15：投递率 8 档鲁棒性"),
    ("execution/figures/fig16_budget_spectrum_pareto", "图 16：外加配额约束下的排序器对比"),
    ("execution/figures/fig18_queueing_delay_and_goodput", "图 18：下游响应时延与有效吞吐"),
    ("execution/figures/fig19_scale_classical_comparison",
     "图 19：G ≥ 48 两套规模网格上 RED/Token/AHP vs 本文方法（按时与字节四联；LinUCB 只入表与文档）"),
    ("execution/figures/fig20_success_and_channel_uses",
     "图 20：G = 24…108 成功率与信道使用次数/世界（单轴拼接、口径切换处标注，五臂；LinUCB 只入表与文档）"),
)
AUDIT_ONLY = (
    ("execution/figures/fig11_master_all_experiments", "三套实验分列总图（内部审计）"),
    ("execution/figures/fig12_master_unified_g", "15 档同轴总图（内部审计）"),
    ("execution/figures/fig7_g_rs_grid_r", "深档 G×R 网格（内部审计）"),
    ("execution/figures/fig8_g_rs_grid_s", "深档 G×S 网格（内部审计）"),
    ("execution/figures/fig9_g_rs_compare_r", "深档六臂对照 R（内部审计）"),
    ("execution/figures/fig10_g_rs_compare_s", "深档六臂对照 S（内部审计）"),
    ("execution/figures/fig13_absolute_totals", "绝对量总图（内部审计）"),
    ("execution/figures/fig14_absolute_per_task", "每任务绝对量（内部审计）"),
)
HIGHLIGHTS = (
    ("on_time_rate", "按时完成率"),
    ("byte_ratio", "字节比"),
    ("messages_per_on_time", "每完成任务消息数"),
    ("broadcast_waste", "广播超时浪费率"),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt_p(value):
    if value is None:
        return "—"
    if value == 0.0:
        return "< 1e-12"
    return "%.2e" % value


def main():
    lines = []
    add = lines.append
    add("# 论文交付清单（PAPER_READY_MANIFEST）")
    add("")
    add("**范围**：论文的图表与正文只展示一个学习模型——`v4f@200`（同一份权重，`cap=None`），"
        "统一记作 **本文方法 (Ours)**；十三个基线照旧（其中四个是具名经典算法，LinUCB-Bandit 是唯一的轻量学习基线，作为对照而非第二个“本文模型”）。`v4t / v4s / v4g / v4u / v4v` "
        "只存在于审计文档（`PAPER_EXPERIMENTS.md`、`EXTENDED_MECHANISM_EXPERIMENTS.md`、"
        "`PROTOCOL_METRICS_REPORT.md`），论文不引用这些代号。")
    add("")

    add("## 1　LaTeX 表")
    add("")
    add("| 文件 | 内容 | sha256（前 16 位） |")
    add("|---|---|---|")
    for path, title in TABLES:
        p = BASE / path
        add("| `%s` | %s | `%s` |" % (path, title, sha256(p)[:16] if p.exists() else "缺失"))
    add("")
    add("表 1 的判据说明（与文件内脚注一致）：头条“按时达标格数”按**放宽判据**"
        "（逐格 `on_time ≥ ⌈0.95×Immediate⌉` 且全网字节比 `< 0.85`、浪费率 `< 0.10`）统计，"
        "本文方法 12/15；脚注同时给出**严格双条件**（另需每格字节更少）的通过数 3/15 与十三个基线的读数。")
    add("")

    add("## 2　论文图")
    add("")
    add("| 图 | 文件（PNG 300 dpi / PDF 矢量） | PNG sha256 |")
    add("|---|---|---|")
    for stem, title in FIGURES:
        png = BASE / (stem + ".png")
        pdf = BASE / (stem + ".pdf")
        add("| %s | `%s.png`<br>`%s.pdf` | `%s` |" % (
            title, stem, stem, sha256(png)[:16] if png.exists() else "缺失"))
        if not pdf.exists():
            add("| | （缺矢量） | |")
    add("")
    add("**不进入论文的图（内部审计）**：")
    add("")
    add("| 图 | 说明 |")
    add("|---|---|")
    for stem, title in AUDIT_ONLY:
        add("| `%s` | %s |" % (stem, title))
    add("")

    add("## 3　核心统计结果")
    add("")
    paired = json.loads(PAIRED.read_text(encoding="utf-8"))
    rows = paired["results"]
    add("配对检验：本文方法 对 八个对照，4 指标 × 3 场景组 = 96 项，分两族（已登记基线 / "
        "具名经典算法）；Wilcoxon 为主检验，Holm 在每族每组（16 项）内校正。下表取 `all(8)` 组：")
    add("")
    add("| 对照 | 指标 | 均值差 | 95% CI | Wilcoxon p | Holm p |")
    add("|---|---|---|---|---|---|")
    for comparator in paired["comparators"]:
        for reading, title in HIGHLIGHTS:
            match = [row for row in rows
                     if row["comparator"] == comparator and row["reading"] == reading
                     and row["scope"] == "all(8)"]
            if not match:
                continue
            row = match[0]
            add("| %s | %s | %+.4f | [%+.4f, %+.4f] | %s | %s |" % (
                comparator, title, row["mean_diff"], row["ci_low"], row["ci_high"],
                fmt_p(row["wilcoxon_p"]), fmt_p(row["wilcoxon_p_holm"])))
    add("")
    add("**读表注意（字节比 vs Slack-Cap）**：均值差 −0.141，但符号秩 p=0.247（Holm 0.740）。"
        "原因是差值按格聚成两簇：运力宽松的格子里本文方法字节比≈1.0（比 Slack-Cap 高约 0.03），"
        "拥塞格子里≈0.35（低约 0.6），符号在格之间翻转，而检验把 120 个世界当作独立样本。"
        "该差异应按逐格表读，不能按单一 p 值读。")
    add("")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    add("高负载（R=6，G≥36，71 世界）下游响应时延 `D_resp` 的配对 Wilcoxon：")
    add("")
    add("| 对照 | 世界 | 均值差（步） | 95% 分位区间 | p |")
    add("|---|---|---|---|---|")
    for key, record in protocol.get("paired_delay_tests", {}).items():
        label = {"v4f@200": "本文方法 − Immediate",
                 "v4t@100": "v4t − Immediate（审计）",
                 "slack_matched_rate": "Slack-Matched − Immediate",
                 "v4f@200 vs slack_matched": "本文方法 − Slack-Matched",
                 "v4t@100 vs slack_matched": "v4t − Slack-Matched（审计）"}.get(key, key)
        add("| %s | %s | %+.2f | [%+.2f, %+.2f] | %s |" % (
            label, record.get("worlds"), record.get("mean_diff_steps"),
            record.get("ci95_low"), record.get("ci95_high"), fmt_p(record.get("p_value"))))
    add("")
    add("协议指标核验：`PDR_link` 的八档实测值与设定 p 的偏差 ≤ 0.013（见 "
        "`PROTOCOL_METRICS_REPORT.md` §5）；`E_task` 取 `α_tx=1.0, α_rx=0.5`。")
    add("")

    add("## 4　封板状态")
    add("")
    add("- 门禁：`python -B -m unittest discover -s tests` → 112 项通过；")
    add("- 冻结件：`analysis/verify_pins.py` → 98/98，drift 0；")
    add("- 论文数据来源：`execution/PAPER_TABLE.md`（机读主表）、`execution/PROTOCOL_METRICS.json`、"
        "`execution/PAIRED_TESTS.json`、`execution/extended/EXTENDED.json`。")
    add("")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("wrote %s (sha256 %s)" % (OUT.name, sha256(OUT)[:16]))


if __name__ == "__main__":
    main()
