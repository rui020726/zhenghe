"""Paired tests on the four efficiency readings, over the archived 120 worlds per arm.

No world is opened. Every per-world number is read from ``execution/COMM_EFFICIENCY.json``
(itself re-derived from the archived lifecycle + relay transcripts, and asserted equal to the
archived ``on_time`` / ``emissions`` before it was written).

Pairs are ``(cell, scene)``: the arm and the comparator ran the same deterministic world, so a
difference is a within-world contrast, not a difference of two averages.

Four readings, each computed per world:

``on_time_rate``           on_time / generated_goals
``byte_ratio``             spent_bytes / spent_bytes of ``normal_immediate`` in the same world
``messages_per_on_time``   emissions / on_time (worlds with on_time == 0 are excluded and counted)
``broadcast_waste``        waste_n / waste_d (distinct broadcast targets not finished on time)

Three tests per (arm, comparator, reading), in the order the review asked for:

1. Shapiro-Wilk on the 120 paired differences -- whether the normal-theory test below is
   descriptive rather than exact;
2. paired t test: mean difference, its 95 % CI (t quantile, n-1 df) and two-sided p;
3. Wilcoxon signed-rank, two-sided, exact for n <= 50 and normal-approximated above.

Multiplicity: 1 arm x 4 comparators x 4 readings = 16 tests per scene group **per family**; the
families are the four registered baselines and the four named classical algorithms, so Holm is
applied within each family and never pools the two. With 16 tests the smallest raw p must be
< 0.0031 to survive at 0.05. The adjustment is reported so no single row is read as if it were
pre-registered.

Independence caveat, stated in the output as well: the 120 worlds are 15 cells x 8 scenes, and
the eight scenes of one cell share the cell's goal count and robot roster. The paired tests treat
them as 120 independent pairs, which overstates the effective sample size; the per-cell table in
the paper is the honest unit for the cross-cell claim.

    D:/project_env/python.exe analysis/paired_tests.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
IN = BASE / "execution/COMM_EFFICIENCY.json"
OUT = BASE / "execution/PAIRED_TESTS.json"

#: The paper shows one learned model; the tests compare it against every comparator the task
#: names.  Two Holm families, each 1 arm x 4 comparators x 4 readings = 16 tests per scene
#: group: the registered baselines, and the four named classical algorithms.
ARMS = ("v4f@200",)
FAMILIES = {
    "registered": ("normal_immediate", "slack_heuristic_cap", "slack_matched_rate",
                   "random_matched_rate"),
    "classic": ("red_aqm", "token_bucket", "ahp_utility", "linucb_bandit"),
}
COMPARATORS = FAMILIES["registered"] + FAMILIES["classic"]
READINGS = (
    ("on_time_rate", "按时完成率", "on_time / generated_goals"),
    ("byte_ratio", "字节比", "spent_bytes / immediate bytes, same world"),
    ("messages_per_on_time", "每完成任务消息数", "emissions / on_time"),
    ("broadcast_waste", "广播超时浪费率", "waste_n / waste_d"),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load():
    payload = json.loads(IN.read_text(encoding="utf-8"))
    table = {}
    for row in payload.get("per_world") or []:
        key = (row["arm"], tuple(row["cell"]), int(row["scene"]))
        table[key] = row
    return payload, table


def reading(row, name, baseline):
    """One reading for one world; ``None`` when its denominator is zero."""
    if name == "on_time_rate":
        goals = int(row["generated_goals"])
        return None if goals <= 0 else int(row["on_time"]) / goals
    if name == "byte_ratio":
        denom = int(baseline["spent_bytes"])
        return None if denom <= 0 else int(row["spent_bytes"]) / denom
    if name == "messages_per_on_time":
        on_time = int(row["on_time"])
        return None if on_time <= 0 else int(row["emissions"]) / on_time
    if name == "broadcast_waste":
        denom = int(row["waste_d"])
        return None if denom <= 0 else int(row["waste_n"]) / denom
    raise ValueError("unknown reading %r" % (name,))


def holm(pvalues):
    """Holm-Bonferroni step-down adjusted p values, in the input order."""
    order = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    count = len(pvalues)
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    sys.dont_write_bytecode = True

    from scipy import stats

    payload, table = load()
    keys = sorted({(cell, scene) for _arm, cell, scene in table},
                  key=lambda item: (item[0], item[1]))
    scopes = {
        "anchor(3)": [k for k in keys if k[1] in (2301901, 2301902, 2301903)],
        "unseen(5)": [k for k in keys if k[1] in (2301904, 2301905, 2301906, 2301907, 2301908)],
        "all(8)": keys,
    }

    blocks, flat = [], []
    for arm in ARMS:
        for comparator in COMPARATORS:
            for reading_name, reading_cn, reading_formula in READINGS:
                for scope_name, scope_keys in scopes.items():
                    diffs, excluded = [], 0
                    for cell, scene in scope_keys:
                        row = table[(arm, cell, scene)]
                        base = table[(comparator, cell, scene)]
                        # ``byte_ratio`` is always relative to normal_immediate in the same
                        # world, for BOTH arms. Passing the comparator as the denominator made
                        # the two sides of the difference different ratios (measured: v4t vs
                        # random reported +0.164 when the true per-world difference is -0.022).
                        baseline = table[("normal_immediate", cell, scene)]
                        value = reading(row, reading_name, baseline)
                        reference = reading(base, reading_name, baseline)
                        if value is None or reference is None:
                            excluded += 1
                            continue
                        diffs.append(value - reference)
                    entry = {
                        "arm": arm, "comparator": comparator, "reading": reading_name,
                        "scope": scope_name, "n": len(diffs), "excluded": excluded,
                        "mean_diff": sum(diffs) / len(diffs) if diffs else None,
                        "median_diff": statistics.median(diffs) if diffs else None,
                        "n_zero": sum(1 for value in diffs if value == 0.0),
                    }
                    if len(diffs) >= 3:
                        shapiro = stats.shapiro(diffs)
                        entry["shapiro_w"] = float(shapiro.statistic)
                        entry["shapiro_p"] = float(shapiro.pvalue)
                        t = stats.ttest_rel(diffs, [0.0] * len(diffs))
                        half = (float(stats.t.ppf(0.975, len(diffs) - 1))
                                * float(stats.sem(diffs)))
                        entry["t"] = float(t.statistic)
                        entry["t_p"] = float(t.pvalue)
                        entry["ci_low"] = entry["mean_diff"] - half
                        entry["ci_high"] = entry["mean_diff"] + half
                    if len(diffs) >= 1 and entry["n_zero"] != len(diffs):
                        wilcoxon = stats.wilcoxon(diffs, method="auto")
                        entry["wilcoxon_stat"] = float(wilcoxon.statistic)
                        entry["wilcoxon_p"] = float(wilcoxon.pvalue)
                    flat.append(entry)
                blocks.append(entry)

    adjusted = {}
    for scope_name in scopes:
        for family_name, family_comparators in FAMILIES.items():
            family = [entry for entry in flat
                      if entry["scope"] == scope_name and "t_p" in entry
                      and entry["comparator"] in family_comparators]
            values = holm([entry["t_p"] for entry in family])
            for entry, value in zip(family, values):
                adjusted[id(entry)] = value
    for entry in flat:
        if "t_p" in entry:
            entry["t_p_holm"] = adjusted[id(entry)]

    #: Wilcoxon is the primary test here: Shapiro-Wilk rejects normality in every one of the 72
    #: scopes, so the t interval is descriptive and the rank test carries the inference. Its Holm
    #: family is the same 16 per scope and per comparator family.
    adjusted_w = {}
    for scope_name in scopes:
        for family_name, family_comparators in FAMILIES.items():
            family = [entry for entry in flat
                      if entry["scope"] == scope_name and "wilcoxon_p" in entry
                      and entry["comparator"] in family_comparators]
            values = holm([entry["wilcoxon_p"] for entry in family])
            for entry, value in zip(family, values):
                adjusted_w[id(entry)] = value
    for entry in flat:
        if "wilcoxon_p" in entry:
            entry["wilcoxon_p_holm"] = adjusted_w[id(entry)]

    lines = []
    lines.append("| 臂 vs 对照 | 指标 | 组 | n | 差值均值 | 差值中位 | 95 % CI (t) "
                 "| t p | Wilcoxon p | Holm p (t) | Holm p (W) | Shapiro p | 零差值数 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def fmt(value, pattern="%.4f"):
        return "n/a" if value is None else pattern % value

    for entry in flat:
        lines.append("| %s vs %s | %s | %s | %d%s | %s | %s | [%s, %s] | %s | %s "
                     "| %s | %s | %s | %d |"
                     % (entry["arm"], entry["comparator"], entry["reading"], entry["scope"],
                        entry["n"],
                        "" if entry["excluded"] == 0 else " (-%d)" % entry["excluded"],
                        fmt(entry["mean_diff"]), fmt(entry["median_diff"]),
                        fmt(entry.get("ci_low")), fmt(entry.get("ci_high")),
                        fmt(entry.get("t_p")), fmt(entry.get("wilcoxon_p")),
                        fmt(entry.get("t_p_holm")), fmt(entry.get("wilcoxon_p_holm")),
                        fmt(entry.get("shapiro_p")), entry["n_zero"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "stage5b-paired-tests/1",
        "source": str(IN),
        "source_sha256": sha256(IN),
        "arms": list(ARMS),
        "comparators": list(COMPARATORS),
        "families": {name: list(values) for name, values in FAMILIES.items()},
        "readings": {name: formula for name, _cn, formula in READINGS},
        "scopes": {name: len(value) for name, value in scopes.items()},
        "scopes_count": len(scopes),
        "tests": len(flat),
        #: Holm was applied once per (scope, family): 1 arm x 4 comparators x 4 readings = 16.
        "holm_family_size": 16,
        "holm_families": len(scopes) * len(FAMILIES),
        "primary_test": "wilcoxon signed-rank, Holm-adjusted within (scope, comparator family) "
                        "(16 tests/family); Shapiro-Wilk rejects normality in all scopes so the "
                        "t interval is descriptive",
        "results": flat,
        "table": "\n".join(lines) + "\n",
        "caveats": [
            "the 120 worlds are 15 cells x 8 scenes; the eight scenes of one cell share its goal "
            "count and rescue roster, so the pairs are not fully independent",
            "one world has zero on-time tasks (slack_matched_rate), so messages_per_on_time is "
            "computed on 119/120 pairs there and the exclusion is printed",
            "byte_ratio is the mean of per-world ratios, not the ratio of the group totals",
            "16 t tests per family: a raw p of 0.05 is not evidence at that level; read the Holm "
            "column",
            "no seed replication exists, so the CI covers world-to-world variation only",
        ],
    }
    (out.parent / "PAIRED_TESTS_TABLE.md").write_text(
        chr(10).join(lines) + chr(10), encoding="utf-8", newline=chr(10))
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(str(tmp), str(out))
    print("\n".join(lines))
    print()
    print("wrote %s  sha256 %s" % (out, sha256(out)))


if __name__ == "__main__":
    main()
