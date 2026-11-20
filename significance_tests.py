import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import random

# --------- 配置 ---------
PILOC_JSON     = "raw_metrics/piloc.json"
PILOC_PH_JSON  = "raw_metrics/piloc_ph.json"
OUT_DIR        = "stats_out"
BOOTSTRAP_B    = 10000
RANDOM_SEED    = 42
ALPHA          = 0.05
# ------------------------

def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_alignment(data_a, data_b):
    nA = data_a["num_episodes"]
    nB = data_b["num_episodes"]
    if nA != nB:
        raise ValueError(f"Episode count mismatch: {nA} vs {nB}")
    keys = ["success_flags", "steps", "targets_found", "final_coverages", "coverage_aucs"]
    for k in keys:
        if len(data_a[k]) != nA or len(data_b[k]) != nB:
            raise ValueError(f"Length mismatch for key={k}")
    return nA

def cohen_d_paired(diff):
    return np.mean(diff) / np.std(diff, ddof=1)

def cliffs_delta_pairs(x, y):
    more = sum(xi > yi for xi, yi in zip(x, y))
    less = sum(xi < yi for xi, yi in zip(x, y))
    n = len(x)
    return (more - less) / n

def bootstrap_ci_mean(diff, B=10000, seed=42):
    rng = random.Random(seed)
    diff = np.asarray(diff)
    n = len(diff)
    samples = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        samples.append(np.mean(diff[idx]))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)

def mcnemar_from_flags(success_a, success_b, exact_threshold=25):
    # 构造 2x2
    a = sum((sa == 1) and (sb == 1) for sa, sb in zip(success_a, success_b))
    b = sum((sa == 1) and (sb == 0) for sa, sb in zip(success_a, success_b))
    c = sum((sa == 0) and (sb == 1) for sa, sb in zip(success_a, success_b))
    d = sum((sa == 0) and (sb == 0) for sa, sb in zip(success_a, success_b))
    n_discordant = b + c

    if n_discordant == 0:
        return {"a": a, "b": b, "c": c, "d": d,
                "statistic": 0.0, "p_value": 1.0,
                "method": "degenerate (no discordant pairs)"}

    # 小样本用精确二项检验，大样本用带连续性校正的卡方
    if n_discordant < exact_threshold:
        k = min(b, c)
        res = stats.binomtest(k=k, n=n_discordant, p=0.5, alternative="two-sided")
        p = res.pvalue
        stat = np.nan
        method = "McNemar exact (binomial)"
    else:
        stat = (abs(b - c) - 1) ** 2 / n_discordant
        p = stats.chi2.sf(stat, df=1)
        method = "McNemar chi-square CC"

    return {"a": a, "b": b, "c": c, "d": d,
            "statistic": float(stat), "p_value": float(p), "method": method}

def paired_test(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    if len(diff) < 3:
        return {"test": "N/A", "p": np.nan, "mean_diff": float(np.mean(diff)), "normal": False, "diff": diff}

    # Shapiro 正态性检验（对差值）
    try:
        sw_p = stats.shapiro(diff)[1]
    except Exception:
        sw_p = 0.0
    normal = bool(sw_p > ALPHA)

    if normal:
        t_stat, p_val = stats.ttest_rel(x, y)
        test_name = "paired t-test"
    else:
        try:
            w_stat, p_val = stats.wilcoxon(x, y, zero_method='wilcox', correction=False)
        except ValueError:
            p_val = 1.0
        test_name = "Wilcoxon signed-rank"
    return {"test": test_name, "p": float(p_val), "mean_diff": float(np.mean(diff)), "normal": normal, "diff": diff}

def summarize_effects(diff, normal):
    diff = np.asarray(diff, dtype=float)
    mean_diff = float(np.mean(diff))
    ci_lo, ci_hi = bootstrap_ci_mean(diff, B=BOOTSTRAP_B, seed=RANDOM_SEED)
    out = {"mean_diff": mean_diff, "ci_lo": ci_lo, "ci_hi": ci_hi}
    if normal and np.std(diff, ddof=1) > 0:
        out["cohen_d"] = float(cohen_d_paired(diff))
    else:
        # 以 0 作为基准估计 Cliff's delta（正差 vs 负差）
        out["cliffs_delta"] = float(cliffs_delta_pairs(diff, np.zeros_like(diff)))
    return out

def format_p(p):
    if p is None or np.isnan(p):
        return "NA"
    if p < 1e-6: return "<1e-6"
    if p < 1e-3: return f"{p:.2e}"
    return f"{p:.4f}"

def main():
    Path(OUT_DIR).mkdir(exist_ok=True, parents=True)
    data_piloc    = load_results(PILOC_JSON)
    data_piloc_ph = load_results(PILOC_PH_JSON)

    n = check_alignment(data_piloc, data_piloc_ph)
    print(f"[INFO] Loaded paired data for {n} episodes")

    success_a = data_piloc["success_flags"]
    success_b = data_piloc_ph["success_flags"]
    steps_a   = data_piloc["steps"]
    steps_b   = data_piloc_ph["steps"]
    targets_a = data_piloc["targets_found"]
    targets_b = data_piloc_ph["targets_found"]
    covfin_a  = data_piloc["final_coverages"]
    covfin_b  = data_piloc_ph["final_coverages"]
    auc_a     = data_piloc["coverage_aucs"]
    auc_b     = data_piloc_ph["coverage_aucs"]

    # SR: McNemar
    mc = mcnemar_from_flags(success_a, success_b)
    delta_sr_pp = (np.mean(success_a) - np.mean(success_b)) * 100.0

    # Steps
    steps_res = paired_test(steps_a, steps_b)
    steps_eff = summarize_effects(steps_res["diff"], steps_res["normal"])

    # Targets
    targets_res = paired_test(targets_a, targets_b)
    targets_eff = summarize_effects(targets_res["diff"], targets_res["normal"])

    # Final coverage
    covfin_res = paired_test(covfin_a, covfin_b)
    covfin_eff = summarize_effects(covfin_res["diff"], covfin_res["normal"])

    # Coverage AUC
    auc_res = paired_test(auc_a, auc_b)
    auc_eff = summarize_effects(auc_res["diff"], auc_res["normal"])

    # 汇总表
    rows = []
    rows.append({
        "Metric": "SR",
        "PILOC_mean": f"{np.mean(success_a):.3f}",
        "PILOC-ph_mean": f"{np.mean(success_b):.3f}",
        "Delta(pp)": f"{delta_sr_pp:.2f}",
        "Test": mc["method"],
        "Stat": "NA" if np.isnan(mc["statistic"]) else f"{mc['statistic']:.3f}",
        "p": format_p(mc["p_value"]),
        "Effect": f"b={mc['b']}, c={mc['c']}"
    })

    def build_row(name, a, b, res, eff):
        d = {
            "Metric": name,
            "PILOC_mean": f"{np.mean(a):.4f}",
            "PILOC-ph_mean": f"{np.mean(b):.4f}",
            "Delta": f"{eff['mean_diff']:.4f}",
            "Test": res["test"],
            "p": format_p(res["p"]),
            "Normal": res["normal"],
            "CI": f"[{eff['ci_lo']:.4f}, {eff['ci_hi']:.4f}]"
        }
        if "cohen_d" in eff:
            d["EffectSize"] = f"d={eff['cohen_d']:.3f}"
        elif "cliffs_delta" in eff:
            d["EffectSize"] = f"delta={eff['cliffs_delta']:.3f}"
        else:
            d["EffectSize"] = ""
        return d

    rows.append(build_row("Steps", steps_a, steps_b, steps_res, steps_eff))
    rows.append(build_row("Targets", targets_a, targets_b, targets_res, targets_eff))
    rows.append(build_row("FinalCoverage", covfin_a, covfin_b, covfin_res, covfin_eff))
    rows.append(build_row("CoverageAUC", auc_a, auc_b, auc_res, auc_eff))

    df = pd.DataFrame(rows)

    # 保存 CSV（带 BOM 便于 Excel）
    out_csv = Path(OUT_DIR) / "significance_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] Summary -> {out_csv}")

    # 控制台打印表格：优先 Markdown（若安装了 tabulate），否则回退为字符串
    try:
        from tabulate import tabulate
        md_table = tabulate(df, headers='keys', tablefmt='github', showindex=False)
        print(md_table)
        Path(OUT_DIR, "significance_summary.md").write_text(md_table, encoding="utf-8")
    except Exception:
        print(df.to_string(index=False))

    # 生成报告文本（避免使用特殊字符，确保 Windows 下无编码问题）
    report_lines = []
    report_lines.append(
        f"PILOC vs PILOC-ph success rate difference: Delta={delta_sr_pp:.2f} pp "
        f"({mc['method']}, p={format_p(mc['p_value'])})."
    )

    def add_report(metric_name, res, eff):
        es = f"Cohen d={eff['cohen_d']:.3f}" if 'cohen_d' in eff else f"Cliff delta={eff['cliffs_delta']:.3f}"
        report_lines.append(
            f"For {metric_name}, mean difference Delta={eff['mean_diff']:.4f} "
            f"({res['test']} p={format_p(res['p'])}, 95% CI {eff['ci_lo']:.4f}-{eff['ci_hi']:.4f}; {es})."
        )

    add_report("steps", steps_res, steps_eff)
    add_report("targets recovered", targets_res, targets_eff)
    add_report("final coverage", covfin_res, covfin_eff)
    add_report("coverage AUC", auc_res, auc_eff)

    report_txt = "\n".join(report_lines)

    # 写 UTF-8 报告
    Path(OUT_DIR, "report_utf8.txt").write_text(report_txt, encoding="utf-8")
    # 写 ASCII 安全报告（可选）
    report_ascii = report_txt.replace("Δ", "Delta").replace("–", "-")
    Path(OUT_DIR, "report_ascii.txt").write_text(report_ascii, encoding="ascii", errors="ignore")

    print(f"[SAVE] Reports -> {Path(OUT_DIR, 'report_utf8.txt')} and report_ascii.txt")

if __name__ == "__main__":
    main()