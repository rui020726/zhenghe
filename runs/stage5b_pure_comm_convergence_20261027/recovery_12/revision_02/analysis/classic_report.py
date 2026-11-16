"""Summarise the four classical arms against the learned model on the 120-world benchmark.

Reads only finished artefacts:

* ``execution/baselines/classic_v1/BASELINES.json``    the four arms on 15 cells x 8 DEV scenes
* ``execution/baselines/grid_v2/BASELINES.json``       the same worlds under ``immediate``
* ``execution/train_v4f_anchor/eval_200_s8/EVAL.json`` Ours on the same worlds
* ``execution/COMM_EFFICIENCY.json``                   messages per on-time task, waste
* ``execution/PROTOCOL_METRICS.json``                  ``E_task`` and ``D_resp``
* ``execution/PAIRED_TESTS.json``                      Ours vs each arm, Wilcoxon + Holm
* ``execution/classic/calibration/*/RESULTS.json``     the operating-point selection

Writes ``CLASSIC_ALGORITHM_COMPARISON.md`` and ``execution/classic/CLASSIC_RESULTS.json``.

    D:/project_env/python.exe analysis/classic_report.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "analysis"))

import extended_report as R  # noqa: E402

ANCHOR = (2301901, 2301902, 2301903)
ALL_SCENES = tuple(range(2301901, 2301909))
CELLS = R.TRAIN_CELLS
CLASSIC = ("red_aqm", "token_bucket", "ahp_utility", "linucb_bandit")
LABEL = {"red_aqm": "RED-AQM", "token_bucket": "Token-Bucket",
         "ahp_utility": "AHP-Utility", "linucb_bandit": "LinUCB-Bandit",
         "slack_heuristic_cap": "EDF-Slack", "slack_matched_rate": "EDF-Matched",
         "normal_immediate": "Flooding", "v4f@200": "本文方法 (Ours)"}
OUT_MD = BASE / "CLASSIC_ALGORITHM_COMPARISON.md"
OUT_JSON = BASE / "execution/classic/CLASSIC_RESULTS.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path, arm=None):
    payload = json.loads((BASE / path).read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("results") or []:
        if row.get("status") != "ok":
            continue
        if arm is not None and row.get("arm") not in (None, arm):
            continue
        rows.append(row)
    return rows


def score(rows, base_rows, scenes):
    own = R.group([row for row in rows if int(row["scene"]) in scenes])
    ref = R.group([row for row in base_rows if int(row["scene"]) in scenes])
    strict = clause = 0
    for cell in CELLS:
        key = R.label(cell)
        mine = R.group([row for row in rows if R.label(row["cell"]) == key
                        and int(row["scene"]) in scenes])
        theirs = R.group([row for row in base_rows if R.label(row["cell"]) == key
                          and int(row["scene"]) in scenes])
        if not mine["worlds"] or not theirs["worlds"]:
            continue
        required = (95 * theirs["on_time"] + 99) // 100
        ok_time = mine["on_time"] >= required
        clause += int(ok_time)
        strict += int(ok_time and mine["spent_bytes"] < theirs["spent_bytes"])
    return own, ref, strict, clause


def main():
    immediate = load_rows("execution/baselines/grid_v2/BASELINES.json", "normal_immediate")
    classic = load_rows("execution/baselines/classic_v1/BASELINES.json")
    ours = load_rows("execution/train_v4f_anchor/eval_200_s8/EVAL.json")
    efficiency = json.loads((BASE / "execution/COMM_EFFICIENCY.json")
                            .read_text(encoding="utf-8"))["per_arm"]
    protocol = json.loads((BASE / "execution/PROTOCOL_METRICS.json")
                          .read_text(encoding="utf-8"))["main"]["all(8)"]
    tests = json.loads((BASE / "execution/PAIRED_TESTS.json").read_text(encoding="utf-8"))
    chosen = json.loads((BASE / "execution/classic/CHOSEN.json").read_text(encoding="utf-8"))
    model = json.loads((BASE / "execution/classic/model/linucb.json").read_text(encoding="utf-8"))

    table = {}
    for arm, rows in ([(arm, [r for r in classic if r["arm"] == arm]) for arm in CLASSIC]
                      + [("v4f@200", ours)]):
        own, ref, strict, clause = score(rows, immediate, ALL_SCENES)
        anchor_own, anchor_ref, _s, _c = score(rows, immediate, ANCHOR)
        group = ((efficiency.get(arm) or {}).get("groups") or {}).get("all(8)") or {}
        waste = group.get("broadcast_waste_ratio")
        relaxed = clause if (own["spent_bytes"] / ref["spent_bytes"] < 0.85
                             and waste is not None and waste < 0.10) else 0
        table[arm] = {
            "worlds": own["worlds"], "on_time": own["on_time"], "goals": own["goals"],
            "q": own["q"], "on_time_ratio": own["on_time"] / ref["on_time"],
            "anchor_ratio": anchor_own["on_time"] / anchor_ref["on_time"],
            "spent_bytes": own["spent_bytes"],
            "byte_ratio": own["spent_bytes"] / ref["spent_bytes"],
            "messages": group.get("messages_per_on_time"),
            "waste": waste, "strict": strict, "on_time_cells": clause, "relaxed": relaxed,
            "e_task": (protocol.get(arm) or {}).get("e_task"),
            "d_resp": ((protocol.get(arm) or {}).get("d_resp") or {}).get("mean"),
        }

    # Per-cell byte ratio for the four arms: RED's load signal is cell-selective by construction.
    per_cell = {}
    for arm in CLASSIC + ("v4f@200",):
        rows = [r for r in classic if r["arm"] == arm] if arm in CLASSIC else ours
        for cell in CELLS:
            key = R.label(cell)
            mine = R.group([r for r in rows if R.label(r["cell"]) == key])
            base = R.group([r for r in immediate if R.label(r["cell"]) == key])
            if not mine["worlds"] or not base["worlds"]:
                continue
            entry = per_cell.setdefault(key, {})
            entry[arm] = {"byte_ratio": mine["spent_bytes"] / base["spent_bytes"],
                          "on_time_ratio": mine["on_time"] / base["on_time"],
                          "q": mine["q"]}

    # Calibration points actually run (byte ratio of each candidate, 2 training scenes).
    reference_path = BASE / "execution/classic/calibration/normal_immediate/reference/RESULTS.json"
    reference = []
    if reference_path.exists():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))["results"]
    base_bytes = {(tuple(row["cell"]), int(row["scene"])): int(row["spent_bytes"])
                  for row in reference}
    calibration = {}
    for arm in CLASSIC:
        calibration[arm] = []
        directory = BASE / "execution/classic/calibration" / arm
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*/RESULTS.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = [row for row in payload["results"] if row.get("status") == "ok"]
            spent = sum(int(row["spent_bytes"]) for row in rows)
            denominator = sum(base_bytes[(tuple(row["cell"]), int(row["scene"]))]
                              for row in rows
                              if (tuple(row["cell"]), int(row["scene"])) in base_bytes)
            on_time = sum(int(row["on_time"]) for row in rows)
            goals = sum(int(row["generated_goals"]) for row in rows)
            calibration[arm].append({
                "point": payload["point"], "byte_ratio": spent / denominator,
                "on_time_rate": on_time / goals if goals else None, "worlds": len(rows)})

    lines = []
    lines.append("# 四个具名经典算法与本文方法的同基准对比")
    lines.append("")
    lines.append("基准：15 个 (G, R, N) 格 × 8 个 DEV 场景 = 120 个确定性世界，"
                 "所有臂逐世界配对（同一 `scale_program.eval_seed`）。")
    lines.append("四个新算法都只包在 `immediate` 外层：候选集合与全量广播完全相同，"
                 "差别只在哪些候选被准入/整形。未改动环境、中继、冻结网络或任何 pin 文件。")
    lines.append("")
    lines.append("## 1. 算法与固定参数")
    lines.append("")
    weights = model["theta"]
    lines.append("| 算法 | 机制 | 状态特征 | 固定参数 |")
    lines.append("|---|---|---|---|")
    lines.append("| RED-AQM | 队列负载概率准入 | #14 backlog/G、#16 capacity_ratio；"
                 "负载 x=(backlog/G)/cap_ratio | min_th=%.2f, max_th=%.2f, max_p=%.2f |"
                 % (chosen["chosen"]["red_aqm"]["point"]["min_th"],
                    chosen["chosen"]["red_aqm"]["point"]["max_th"],
                    chosen["chosen"]["red_aqm"]["point"]["max_p"]))
    bucket = chosen["chosen"].get("token_bucket", {}).get("point", {})
    depth = bucket.get("depth", 3.0)
    depth_text = ("min(12, 0.6 x 0.82 x immediate pages)" if depth == "auto"
                  else "%.0f" % float(depth))
    lines.append("| Token-Bucket | 固定速率令牌整形 | 速率/桶深（不读逐目标状态） | "
                 "桶深 %s（每格），速率=(0.82×immediate 页面 − 桶深)/1000 步 |"
                 % depth_text)
    lines.append("| AHP-Utility | 五指标加权阈值 | #11,#10,#14,#16,#17 | 权重 (%.4f, %.4f, "
                 "%.4f, %.4f, %.4f)，CR=%.4f，阈值 %.2f |"
                 % (*[w for w in _ahp_weights()], _ahp_cr(),
                    chosen["chosen"]["ahp_utility"]["point"]["threshold"]))
    lines.append("| LinUCB-Bandit | 线性 UCB（一步近视） | 19 维无量纲状态 | α=%.2f，"
                 "阈值 %.4f，训练 AUC=%.4f（45 世界、%d 样本、正例率 %.3f） |"
                 % (model["alpha"], chosen["chosen"]["linucb_bandit"]["point"]["threshold"],
                    model["training"]["auc_in_sample"], model["training"]["samples"],
                    model["training"]["base_rate"]))
    lines.append("")
    lines.append("AHP 判断矩阵（机制权重来自主特征向量）：")
    lines.append("")
    lines.append("```")
    for row in _ahp_matrix():
        lines.append(" ".join("%6.3f" % value for value in row))
    lines.append("```")
    lines.append("")
    lines.append("## 2. 工作点标定（训练场景；RED 另做一步仅按成本的 DEV 校准）")
    lines.append("")
    lines.append("| 算法 | 候选点 | 字节比 | 按时率 | 选中 |")
    lines.append("|---|---|---|---|---|")
    for arm in CLASSIC:
        selected = chosen["chosen"][arm]["point"]
        for entry in calibration.get(arm, []):
            if arm == "token_bucket" and set(entry["point"]) == {"fraction"}:
                # The first depth-3 pass predates the depth field; the depth-3 row above is
                # the same run and carries it.
                continue
            mark = "✓" if entry["point"] == selected else ""
            lines.append("| %s | `%s` | %.3f | %.4f | %s |"
                         % (LABEL[arm], json.dumps(entry["point"], sort_keys=True),
                            entry["byte_ratio"], entry["on_time_rate"], mark))
    lines.append("")
    lines.append("RED 的例外：训练场景上 max_p=0.20 的字节比是 0.838，但同样参数在 120 个 DEV 世界上"
                 "测得 0.877（高于 0.85 成本条款）。因此只按“字节比”再做一步 0.20→0.25，"
                 "DEV 字节比降到 0.841；这一步没有读取任何按时率。其余三个算法的工作点全部只由"
                 "训练场景决定，没有 DEV 校准。")
    lines.append("")
    lines.append("## 3. 主结果（8 场景合计）")
    lines.append("")
    lines.append("| 方法 | 按时 | 相对按时率 (锚定) | 字节比 | 消息/任务 | 浪费率 | "
                 "放宽达标格 | 严格达标格 | E_task | D_resp |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    order = CLASSIC + ("v4f@200",)
    for arm in order:
        record = table[arm]
        lines.append("| %s | %d/%d = %.4f | %.3f (%.3f) | %.3f | %s | %s | %d/15 | %d/15 | %s | %s |"
                     % (LABEL[arm], record["on_time"], record["goals"], record["q"],
                        record["on_time_ratio"], record["anchor_ratio"], record["byte_ratio"],
                        _f(record["messages"]), _f(record["waste"]), record["relaxed"],
                        record["strict"], _f(record["e_task"], "%.0f"),
                        _f(record["d_resp"], "%.1f")))
    lines.append("")
    lines.append("两个达标口径与主表一致：放宽 = 每格 on-time ≥ ⌈0.95×Immediate⌉ 且全网字节比 <0.85、"
                 "浪费率 <0.10；严格 = 逐格按时且每格字节更少。")
    lines.append("")
    lines.append("## 4. 逐格字节比（RED 的负载信号按格选择性生效）")
    lines.append("")
    header = "| 格 | " + " | ".join(LABEL[arm] for arm in order) + " |"
    lines.append(header)
    lines.append("|---" * (len(order) + 1) + "|")
    for cell in CELLS:
        key = R.label(cell)
        entry = per_cell.get(key, {})
        values = []
        for arm in order:
            item = entry.get(arm)
            values.append("—" if item is None else "%.3f / %.3f"
                          % (item["byte_ratio"], item["on_time_ratio"]))
        lines.append("| %s | %s |" % (key, " | ".join(values)))
    lines.append("")
    lines.append("每格为“字节比 / 相对按时率”，分母是同格同场景的 Immediate。")
    lines.append("")
    lines.append("## 5. 配对检验：本文方法 vs 四个经典算法（8 场景，Wilcoxon + Holm）")
    lines.append("")
    lines.append("| 对照 | 指标 | n | 差值均值 | Wilcoxon p | Holm p |")
    lines.append("|---|---|---|---|---|---|")
    for entry in tests["results"]:
        if entry["comparator"] not in CLASSIC or entry["scope"] != "all(8)":
            continue
        lines.append("| %s | %s | %d | %s | %s | %s |"
                     % (LABEL[entry["comparator"]], entry["reading"], entry["n"],
                        _f(entry.get("median_diff")), _f(entry.get("wilcoxon_p")),
                        _f(entry.get("wilcoxon_p_holm"))))
    lines.append("")
    lines.append("Holm 家族 = 4 个经典对照 × 4 个指标 = 每场景组 16 项；"
                 "差值均值只作描述，推断看 Wilcoxon 与 Holm 列。")
    lines.append("")
    lines.append("## 6. 能说与不能说")
    lines.append("")
    ours = table["v4f@200"]
    best = min(CLASSIC, key=lambda arm: table[arm]["byte_ratio"])
    lines.append("- 能说：在同一 120 世界上，本文方法放宽达标 **%d/15**、严格 **%d/15**，"
                 "字节比 **%.3f**；四个经典算法分别为 %s。"
                 % (ours["relaxed"], ours["strict"], ours["byte_ratio"],
                    "、".join("%s %d/%d" % (LABEL[arm], table[arm]["relaxed"],
                                            table[arm]["strict"]) for arm in CLASSIC)))
    lines.append("- 能说：四个经典算法均已在训练场景上标定到同一 0.82 字节工作点附近，"
                 "因此差异来自“选哪些目标”，不是“花多少钱”（选中工作点字节比 %s）。"
                 % "、".join("%s %.3f" % (LABEL[arm], chosen["chosen"][arm]["byte_ratio"])
                             for arm in CLASSIC))
    lines.append("- 不能说：本文方法在“绝对按时数”上全面优于经典算法。"
                 "经典算法中 %s 的字节比更低（%.3f），代价是按时数下降；"
                 "逐项比较以第 5 节的配对检验为准。"
                 % (LABEL[best], table[best]["byte_ratio"]))
    lin = table["linucb_bandit"]
    lin_test = [entry for entry in tests["results"]
                if entry["comparator"] == "linucb_bandit" and entry["reading"] == "on_time_rate"
                and entry["scope"] == "all(8)"]
    if lin["relaxed"] == ours["relaxed"]:
        p_text = _f(lin_test[0].get("wilcoxon_p")) if lin_test else "—"
        lines.append("- **能说（对本文不利，必须写）**：LinUCB-Bandit 与本文方法在放宽判据上同为 "
                     "**%d/15**、严格判据同为 **%d/15**；其相对按时率 %.3f（本文 %.3f），"
                     "字节比 %.3f（本文 %.3f）。逐世界配对的按时率差中位 %s，Wilcoxon p=%s。"
                     "在这套基准上，一步近视的 19 维线性 UCB 已经复现了本文方法的主要结果；"
                     "本文方法只在字节比上低 %.1f %%（%.3f vs %.3f），"
                     "不能说它“显著更好”。"
                     % (lin["relaxed"], lin["strict"], lin["on_time_ratio"],
                        ours["on_time_ratio"], lin["byte_ratio"], ours["byte_ratio"],
                        _f(lin_test[0].get("median_diff")) if lin_test else "—", p_text,
                        100.0 * (lin["byte_ratio"] - ours["byte_ratio"]) / lin["byte_ratio"],
                        ours["byte_ratio"], lin["byte_ratio"]))
    lines.append("- 不能说：这四个实现是文献算法的完整复现。它们是**准入/整形语义**的最小实现，"
                 "参数由本实验的工作点标定；RED 的队列仅取待发积压，不含路由器缓冲区的平均队列。")
    lines.append("- 不能说：LinUCB 学到的策略比启发式更强。它的 AUC 是训练集内读数（%.4f），"
                 "训练样本全部来自 immediate 的发送，没有“不发送”的反事实。"
                 % model["training"]["auc_in_sample"])
    lines.append("")
    lines.append("## 7. 证据")
    lines.append("")
    for name, path in (
            ("BASELINES.json", "execution/baselines/classic_v1/BASELINES.json"),
            ("CHOSEN.json", "execution/classic/CHOSEN.json"),
            ("linucb.json", "execution/classic/model/linucb.json"),
            ("COMM_EFFICIENCY.json", "execution/COMM_EFFICIENCY.json"),
            ("PROTOCOL_METRICS.json", "execution/PROTOCOL_METRICS.json"),
            ("PAIRED_TESTS.json", "execution/PAIRED_TESTS.json")):
        path = BASE / path
        if path.exists():
            lines.append("- `%s` — sha256 `%s`" % (name, sha256(path)))
    lines.append("")
    lines.append("## 8. 审查清单")
    lines.append("")
    lines.append("1. 四个算法只在评估策略层新增（`stage5b_comm/classic_baselines.py`），"
                 "`integrated_sar/` 与 98 个 pin 文件未改。")
    lines.append("2. 每个候选参数点都在训练场景（2201904–2201905）上跑完 30 世界；"
                 "DEV 场景只用于最终 120 世界，唯一例外是 RED 的一步“仅按字节比”的校准"
                 "（0.20→0.25），已在本文件 §2 单独记录。")
    lines.append("3. 所有臂逐世界使用 `scale_program.eval_seed`，与 `immediate`/Ours 配对。")
    lines.append("4. AHP 的一致性比率已断言 <0.1；判断矩阵与特征向量写入本文件。")
    lines.append("5. 每个世界的中继计数、字节与按时数由归档行直接汇总，"
                 "`COMM_EFFICIENCY` 的提取已断言与归档行一致。")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    payload = {"schema": "stage5b-classic-results/1", "table": table,
               "per_cell": per_cell, "calibration": calibration,
               "chosen": chosen, "model_training": model["training"],
               "sources": {
                   "classic": sha256(BASE / "execution/baselines/classic_v1/BASELINES.json")}}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    print("wrote %s" % OUT_MD)
    print("wrote %s" % OUT_JSON)
    for arm in order:
        record = table[arm]
        print("%-14s %d/%d  rel %.3f  bytes %.3f  relaxed %d strict %d"
              % (LABEL[arm], record["on_time"], record["goals"], record["on_time_ratio"],
                 record["byte_ratio"], record["relaxed"], record["strict"]))


def _f(value, pattern="%.4f"):
    return "—" if value is None else pattern % value


def _ahp_matrix():
    from stage5b_comm import classic_baselines
    return classic_baselines.AHP_PAIRWISE


def _ahp_weights():
    from stage5b_comm import classic_baselines
    return classic_baselines._AHP_WEIGHTS


def _ahp_cr():
    from stage5b_comm import classic_baselines
    return classic_baselines._AHP_CR


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
