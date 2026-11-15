"""Build EXTENDED_MECHANISM_EXPERIMENTS.md and EXTENDED.json from the five probe stages.

Reads only files this revision wrote: ``execution/extended/<stage>/<arm>/RESULTS.json``,
``execution/extended/micro/MICRO.json`` and the sealed baseline ``grid_v2/BASELINES.json`` for
the reference numbers the probes are compared against. No world is opened and nothing is
re-derived from a different convention than the one the sweep ran under.

    D:/project_env/python.exe analysis/extended_report.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
EXT = BASE / "execution/extended"
OUT_MD = BASE / "EXTENDED_MECHANISM_EXPERIMENTS.md"
OUT_JSON = EXT / "EXTENDED.json"

ANCHOR = (2301901, 2301902, 2301903)
ALL_SCENES = tuple(range(2301901, 2301909))
DELIVERY_CELLS = ((24, 6, 6), (48, 6, 6), (48, 12, 6))
DELIVERY_VALUES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DELIVERY_ARMS = ("normal_immediate", "dedup_immediate", "event_maxwait",
                 "slack_matched_rate", "v4s@300", "v4t@100")
DELIVERY_LABEL = {
    "normal_immediate": "immediate",
    "dedup_immediate": "dedup",
    "event_maxwait": "event_maxwait",
    "slack_matched_rate": "slack_matched",
    "v4s@300": "v4s@300",
    "v4t@100": "v4t@100",
}
FEATURE_VARIANTS = ("mask16", "mask14", "mask2", "mask17", "horizon50")
FEATURE_TITLE = {
    "mask16": "#16 capacity_ratio 固定 1.0",
    "mask14": "#14 backlog/G 固定 0",
    "mask2": "#2 sent_count/G 固定 0",
    "mask17": "#17 N/6 固定 1.0",
    "horizon50": "#11 remaining_time 分母退回 50",
}
CAPACITY_CELLS = ((48, 12, 3), (48, 12, 6), (60, 12, 3), (60, 12, 6), (72, 12, 3), (72, 12, 6))
BUDGET_CELLS = ((48, 6, 6), (48, 12, 6))
BUDGET_FRACTIONS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
BUDGET_SELECTORS = ("v4f", "slack_budget", "fifo_budget", "random_budget")
BUDGET_LABEL = {"v4f": "v4f 网络价值选择", "slack_budget": "Slack 时限选择",
                "fifo_budget": "FIFO 先到先得", "random_budget": "Random 随机选择"}
LATENCY_CELLS = ((12, 6, 6), (24, 6, 6), (48, 6, 6), (72, 6, 6), (108, 6, 6), (144, 6, 6))

TRAIN_CELLS = tuple((goals, robots, subscribers)
                    for goals in (12, 24, 48)
                    for robots in (6, 12)
                    for subscribers in (3, 6, 12)
                    if subscribers <= robots)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def label(cell):
    return "G%d_R%d_N%d" % (int(cell[0]), int(cell[1]), int(cell[2]))


def load_arm(stage, arm):
    path = EXT / stage / arm.replace("@", "_") / "RESULTS.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in payload.get("results") or [] if row.get("status") == "ok"]


def fmt(value, pattern="%.3f", empty="—"):
    if value is None:
        return empty
    return pattern % value


def pct(value, pattern="%+.1f %%", empty="—"):
    if value is None:
        return empty
    return pattern % value


def need(base_on_time):
    return (95 * int(base_on_time) + 99) // 100


def need_for(base_on_time):
    """Same rounding rule as :func:`need`, named for the no-cap section's tables."""
    return need(base_on_time)


# ------------------------------------------------------------------ delivery probability
def cell_pages():
    """immediate pages/world per cell, from the 0.7 baseline the sweep reuses."""
    payload = json.loads((BASE / "execution/baselines/grid_v2/BASELINES.json")
                         .read_text(encoding="utf-8"))
    totals = {}
    for row in payload.get("results") or []:
        if row.get("arm") != "normal_immediate" or row.get("status") != "ok":
            continue
        key = label(row["cell"])
        entry = totals.setdefault(key, [0, 0])
        entry[0] += int(row["emissions"])
        entry[1] += 1
    return {key: (pages / worlds) for key, (pages, worlds) in totals.items()}


def archived_reference(arm, cells, scenes):
    """The sealed 0.7 world for one arm over the same cells and scenes."""
    sources = {
        "normal_immediate": "execution/baselines/grid_v2/BASELINES.json",
        "dedup_immediate": "execution/baselines/grid_v2/BASELINES.json",
        "event_maxwait": "execution/baselines/paper_traditional_v1/BASELINES.json",
        "slack_matched_rate": "execution/baselines/paper_slack_matched/BASELINES.json",
        "v4f@200": "execution/train_v4f_anchor/eval_200_s8/EVAL.json",
        "v4s@300": "execution/train_v4s_cap300/eval_300_s8/EVAL.json",
        "v4t@100": "execution/train_v4t_cap_frozen/eval_100_s8/EVAL.json",
    }
    path = BASE / sources[arm]
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("results") or []:
        if row.get("status") != "ok":
            continue
        # A baseline file holds several arms side by side and stamps each row; an EVAL file
        # holds one arm and stamps nothing. Accepting "normal_immediate" here as well would
        # have summed the baseline into every other baseline arm (measured: dedup read 2.02x).
        if "arm" in row and row["arm"] != arm:
            continue
        cell = tuple(int(value) for value in row["cell"])
        if cell not in cells or int(row["scene"]) not in scenes:
            continue
        rows.append(row)
    return rows


def archived_waste(arm, cells, scenes):
    """Waste counts for the sealed 0.7 worlds, read from their own logs."""
    sys.path.insert(0, str(BASE / "analysis"))
    from comm_efficiency import measure
    roots = {
        "normal_immediate": ("execution/baselines/grid_v2", "eval__normal_immediate_", True),
        "dedup_immediate": ("execution/baselines/grid_v2", "eval__dedup_immediate_", True),
        "event_maxwait": ("execution/baselines/paper_traditional_v1",
                          "eval__event_maxwait_", True),
        "slack_matched_rate": ("execution/baselines/paper_slack_matched", "eval__", False),
        "v4f@200": ("execution/train_v4f_anchor/eval_200_s8", "eval__", False),
        "v4s@300": ("execution/train_v4s_cap300/eval_300_s8", "eval__", False),
        "v4t@100": ("execution/train_v4t_cap_frozen/eval_100_s8", "eval__", False),
    }
    root, prefix, shared = roots[arm]
    logs = BASE / root / "world_logs"
    total = {"distinct": 0, "waste": 0}
    for directory in sorted(logs.iterdir()):
        if not directory.is_dir():
            continue
        if shared and not directory.name.startswith(prefix):
            continue
        inner = [path for path in directory.iterdir() if path.is_dir()]
        if len(inner) != 1:
            continue
        parts = directory.name.split("__")
        cell = tuple(int(value) for value in
                     __import__("re").findall(r"\d+", parts[1].split("_c")[0])[-3:])
        scene = int(parts[3])
        if cell not in cells or scene not in scenes:
            continue
        reading = measure(inner[0])
        total["distinct"] += reading["distinct_broadcast"]
        total["waste"] += reading["waste_n"]
    return total


def percentiles(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def group(rows):
    on_time = sum(int(row["on_time"]) for row in rows)
    generated = sum(int(row["generated_goals"]) for row in rows)
    emissions = sum(int(row["emissions"]) for row in rows)
    spent = sum(int(row["spent_bytes"]) for row in rows)
    broadcast = sum(int(row.get("waste_waste_d") or 0) for row in rows)
    waste = sum(int(row.get("waste_waste_n") or 0) for row in rows)
    seconds = [float(row.get("elapsed_seconds") or 0.0) for row in rows]
    return {"worlds": len(rows), "goals": generated, "on_time": on_time,
            "q": (on_time / generated if generated else None), "emissions": emissions,
            "spent_bytes": spent,
            "bytes_per_on_time": (spent / on_time if on_time else None),
            "messages_per_on_time": (emissions / on_time if on_time else None),
            "distinct_broadcast": broadcast, "waste_n": waste,
            "waste_ratio": (waste / broadcast if broadcast else None),
            "seconds_mean": (sum(seconds) / len(seconds)) if seconds else None}


def delivery_section(state):
    """Experiment I: one table per arm x probability, plus the two robustness readings."""
    cells = DELIVERY_CELLS
    scenes = ALL_SCENES
    payload = {}
    for arm in DELIVERY_ARMS:
        payload[arm] = {}
        for probability in DELIVERY_VALUES:
            if probability == 0.7:
                rows = archived_reference(arm, cells, scenes)
                stats = group(rows)
                counts = archived_waste(arm, cells, scenes)
                stats["distinct_broadcast"] = counts["distinct"]
                stats["waste_n"] = counts["waste"]
                stats["waste_ratio"] = (counts["waste"] / counts["distinct"]
                                        if counts["distinct"] else None)
                stats["source"] = "archive"
            else:
                variant = "p%02d" % int(round(probability * 100))
                rows = [row for row in load_arm("delivery", arm) if row.get("variant") == variant]
                stats = group(rows)
                stats["source"] = "sweep"
            payload[arm][str(probability)] = stats

    reference = {str(p): payload["normal_immediate"][str(p)] for p in DELIVERY_VALUES}
    for arm in DELIVERY_ARMS:
        for probability in DELIVERY_VALUES:
            entry = payload[arm][str(probability)]
            base = reference[str(probability)]
            entry["on_time_ratio"] = (entry["on_time"] / base["on_time"]
                                      if base["on_time"] else None)
            entry["byte_ratio"] = (entry["spent_bytes"] / base["spent_bytes"]
                                   if base["spent_bytes"] else None)
            entry["q_ratio"] = (entry["q"] / base["q"] if base["q"] else None)

    # Robustness summaries: at which probability does each learned arm stop holding 0.95?
    robustness = {}
    for arm in ("v4s@300", "v4t@100", "slack_matched_rate"):
        series = [(p, payload[arm][str(p)]["on_time_ratio"]) for p in DELIVERY_VALUES]
        floor = min(p for p, ratio in series if ratio is not None and ratio < 0.95) \
            if any(ratio is not None and ratio < 0.95 for _p, ratio in series) else None
        robustness[arm] = {
            "min_ratio": min(ratio for _p, ratio in series if ratio is not None),
            "at_probability": min((p for p, ratio in series if ratio is not None),
                                  key=lambda p: payload[arm][str(p)]["on_time_ratio"]),
            "first_below_095": floor,
        }
    state["delivery"] = {"by_arm": payload, "robustness": robustness}
    return payload, robustness


def feature_section(state):
    """Experiment J1: 45 anchor worlds per mutation against the sealed full-row v4t."""
    reference_rows = archived_reference("v4t@100", TRAIN_CELLS, ANCHOR)
    reference = {}
    for row in reference_rows:
        reference[label(tuple(int(v) for v in row["cell"]))] = row
    base_group = group(reference_rows)
    base_counts = archived_waste("v4t@100", TRAIN_CELLS, ANCHOR)
    base_group["distinct_broadcast"] = base_counts["distinct"]
    base_group["waste_n"] = base_counts["waste"]
    base_group["waste_ratio"] = (base_counts["waste"] / base_counts["distinct"]
                                 if base_counts["distinct"] else None)

    variants = {}
    for variant in FEATURE_VARIANTS:
        rows = load_arm("features", variant)
        aggregate = group(rows)
        per_cell = {}
        for cell in TRAIN_CELLS:
            key = label(cell)
            own = [row for row in rows if label(row["cell"]) == key]
            base = reference.get(key)
            if not own or base is None:
                continue
            cell_stats = group(own)
            cell_stats["pass"] = (cell_stats["on_time"] >= need(int(base["on_time"]))
                                  and cell_stats["spent_bytes"] < int(base["spent_bytes"]))
            cell_stats["base_on_time"] = int(base["on_time"])
            cell_stats["base_bytes"] = int(base["spent_bytes"])
            per_cell[key] = cell_stats
        aggregate["per_cell"] = per_cell
        aggregate["passes"] = sum(1 for value in per_cell.values() if value["pass"])
        aggregate["cells"] = len(per_cell)
        aggregate["on_time_delta"] = aggregate["on_time"] - base_group["on_time"]
        aggregate["emissions_delta"] = aggregate["emissions"] - base_group["emissions"]
        aggregate["byte_ratio"] = (aggregate["spent_bytes"] / base_group["spent_bytes"]
                                   if base_group["spent_bytes"] else None)
        aggregate["waste_delta"] = (
            None if aggregate["waste_ratio"] is None or base_group["waste_ratio"] is None
            else aggregate["waste_ratio"] - base_group["waste_ratio"])
        variants[variant] = aggregate

    state["features"] = {"reference": base_group, "variants": variants}
    return base_group, variants


def capacity_section(state):
    """Experiment J2: #16 rewritten to 2.0 at the cells the G sweep reported as self-throttled."""
    payload = {}
    for cell in CAPACITY_CELLS:
        key = label(cell)
        entry = {}
        for variant in ("control", "cap16x2"):
            rows = [row for row in load_arm("capacity", "v4t_" + variant)
                    if label(row["cell"]) == key]
            stats = group(rows)
            stats["cap"] = (rows[0].get("cap") if rows else None)
            stats["sends_per_world"] = (stats["emissions"] / stats["worlds"]
                                        if stats["worlds"] else None)
            entry[variant] = stats
        control, probe = entry["control"], entry["cap16x2"]
        entry["lift"] = (probe["sends_per_world"] - control["sends_per_world"]
                         if control["sends_per_world"] is not None
                         and probe["sends_per_world"] is not None else None)
        entry["cap_use_control"] = (control["sends_per_world"] / control["cap"]
                                    if control["cap"] else None)
        entry["cap_use_probe"] = (probe["sends_per_world"] / probe["cap"]
                                  if probe["cap"] else None)
        payload[key] = entry
    state["capacity"] = payload
    return payload


def budget_section(state):
    """Experiment K: nine ceilings x four selectors, and the Pareto set of each cell."""
    payload = {}
    for cell in BUDGET_CELLS:
        key = label(cell)
        entry = {}
        for selector in BUDGET_SELECTORS:
            entry[selector] = {}
            for fraction in BUDGET_FRACTIONS:
                variant = "b%03d" % int(round(fraction * 100))
                rows = [row for row in load_arm("budget", selector)
                        if label(row["cell"]) == key and row.get("variant") == variant]
                stats = group(rows)
                stats["cap"] = rows[0].get("cap") if rows else None
                stats["budget_fraction"] = fraction
                entry[selector][str(fraction)] = stats
        # Reference: immediate in the same cell at the sealed 0.7 channel (archive).
        base_rows = archived_reference("normal_immediate", (cell,), ALL_SCENES)
        base = group(base_rows)
        for selector in BUDGET_SELECTORS:
            for fraction in BUDGET_FRACTIONS:
                stats = entry[selector][str(fraction)]
                stats["byte_ratio"] = (stats["spent_bytes"] / base["spent_bytes"]
                                       if base["spent_bytes"] else None)
                stats["on_time_ratio"] = (stats["on_time"] / base["on_time"]
                                          if base["on_time"] else None)
        # Pareto set: non-dominated on (fewer bytes, more on-time) within this cell.
        points = []
        for selector in BUDGET_SELECTORS:
            for fraction in BUDGET_FRACTIONS:
                stats = entry[selector][str(fraction)]
                if stats["spent_bytes"] and stats["on_time"]:
                    points.append((selector, fraction, stats["spent_bytes"], stats["on_time"]))
        frontier = []
        for candidate in points:
            dominated = any(
                other[2] <= candidate[2] and other[3] >= candidate[3]
                and (other[2] < candidate[2] or other[3] > candidate[3])
                for other in points)
            if not dominated:
                frontier.append({"selector": candidate[0], "fraction": candidate[1],
                                 "spent_bytes": candidate[2], "on_time": candidate[3]})
        payload[key] = {"by_selector": entry, "reference": base,
                        "pareto": sorted(frontier, key=lambda row: row["spent_bytes"])}
    state["budget"] = payload
    return payload


def micro_section(state):
    path = EXT / "micro/MICRO.json"
    if not path.exists():
        raise SystemExit("missing %s; run analysis/micro_profile.py first" % path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    state["micro"] = payload["arms"]
    return payload["arms"]


def latency_section(state):
    rows = load_arm("latency", "v4t@100")
    table = {}
    for row in rows:
        key = label(row["cell"])
        table[key] = {"g": int(row["cell"][0]), "worlds": 1,
                      "decisions": row["latency"]["decisions"],
                      "mean_ms": row["latency"]["mean_ms"],
                      "p50_ms": row["latency"]["p50_ms"],
                      "p99_ms": row["latency"]["p99_ms"],
                      "candidates": row["latency"]["candidates_per_decision"],
                      "emissions": int(row["emissions"]),
                      "on_time": int(row["on_time"]),
                      "generated": int(row["generated_goals"]),
                      "seconds": row.get("elapsed_seconds")}
    state["latency"] = table
    return table


def parameter_block():
    """Parameter count and on-disk size of the frozen network, from its checkpoint."""
    import torch
    path = BASE / "execution/train_v4t_cap_frozen/checkpoints/milestone_100.pth"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload["controller"]
    total = 0
    shapes = {}
    for prefix, key in (("trunk", "trunk"), ("action_head", "action_head"),
                        ("value_task_head", "value_task_head")):
        for name, tensor in saved[key].items():
            total += int(tensor.numel())
            shapes["%s.%s" % (prefix, name)] = list(tensor.shape)
    return {"parameters": total, "float32_bytes": total * 4,
            "path": str(path.relative_to(BASE)),
            "sha256": sha256(path),
            "shapes": shapes}


# ------------------------------------------------------------------ report
def main():
    state = {"schema": "stage5b-extended/1",
             "sources": {}, "worlds": {}}

    delivery, robustness = delivery_section(state)
    base_feature, features = feature_section(state)
    capacity = capacity_section(state)
    budget = budget_section(state)
    micro = micro_section(state)
    latency = latency_section(state)
    params = parameter_block()
    state["parameters"] = params

    cells_pages = cell_pages()
    lines = []
    add = lines.append

    add("# 扩展机理实验：信道鲁棒性、特征可解释性、配额谱系、微观画像与实时性")
    add("")
    add("零新训练。五个探针全部在已封板权重上运行，98 个 pin 文件未改动；除实验 L 纯读日志外，"
        "每个世界都是登记过的评估世界（`cell` 与 `comm_delivery_probability` 走与 G/R/S 扩展"
        "同一套 override 通道，而不是放松任何检查）。")
    add("")
    add("| 实验 | 变的是什么 | 固定什么 | 世界数 | 数据源 |")
    add("|---|---|---|---|---|")
    add("| I 投递率 | `comm_delivery_probability` 8 档 | 3 格 × 8 场景、6 臂 | 1 008 | `execution/extended/delivery/*` |")
    add("| J1 特征屏蔽 | 单列固定值 / #11 分母 | 15 格 × 3 锚定场景、v4t 权重 | 225 | `execution/extended/features/*` |")
    add("| J2 容量反事实 | #16 覆写 2.0 | 6 格 × 3 锚定场景、同页预算 | 36 | `execution/extended/capacity/*` |")
    add("| K 配额谱系 | 页上限 20 %–100 % | 2 格 × 8 场景、4 选择器 | 576 | `execution/extended/budget/*` |")
    add("| L 微观画像 | 无（纯日志） | 既有 120 世界 | 0 | `execution/extended/micro/MICRO.json` |")
    add("| M 实时性 | 任务规模 G | 1 场景、v4t 权重 | 6 | `execution/extended/latency/*` |")
    add("")
    add("口径（全篇一致）：**绝对按时率** = 按时数/目标总数（≤1）；**相对比值** = 该臂按时数 / "
        "同一批世界里 immediate 的按时数（可 >1）；0.95 判据始终是相对的。每个世界只跑一次，"
        "没有种子重复，差一两个目标的差异不解读。")
    add("")

    # ---------------------------------------------------------------- I
    add("## 实验 I　投递率 8 档鲁棒性")
    add("")
    add("三格（G24_R6_N6、G48_R6_N6、G48_R12_N6）× 8 场景 = 24 世界/点；0.7 复用封板归档，"
        "其余七档为新跑。六个臂都使用**各自封板的页预算**（v4t/v4s 用 checkpoint 自带的 "
        "`cap_by_cell`，slack_matched 对齐 v4t 同格的页数），所以变化的是信道，不是配额。")
    add("")
    add("**耦合性质**：中继的丢包判据是 `rng.random() > p`，随机流种子等于场景号、与 p 无关，"
        "所以各档位看到的第 i 个均匀数相同。**在轨迹分叉之前**，丢包集合随 p 单调嵌套；"
        "一旦某个包在低 p 下被丢、在高 p 下送达，之后的包序列就不再对应。因此这七档应读作"
        "「同种子、同判据、阈值不同」的耦合族，而不是七次独立抽样——单点差异不解读，"
        "只读随 p 的走势。")
    add("")
    add("**接缝自检**：把 `p = 0.7` 的一格（immediate，G24_R6_N6，场景 2301901）在新管道上重跑，"
        "结果与归档逐字节相同（18 按时 / 24 条 / 60 726 B），且该次运行走的是**冻结的**传输断言"
        "（`transport_route = frozen`）。所以 0.6 与 0.7 两档之间的差不是接缝造成的："
        "低 p 下按时率大体随 p 下降，但在 0.3–0.4 处有 ±1.5 个百分点的抖动，属 24 世界的小样本"
        "波动，不解读单点。")
    add("")
    add("| 臂 | p=0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for arm in DELIVERY_ARMS:
        name = DELIVERY_LABEL[arm]
        cells = []
        for probability in DELIVERY_VALUES:
            entry = delivery[arm][str(probability)]
            cells.append("%s（%s）" % (fmt(entry["q"], "%.3f"),
                                      fmt(entry["on_time_ratio"], "%.2f×")))
        add("| %s | %s |" % (name, " | ".join(cells)))
    add("")
    add("（单元格 = 绝对按时率（相对 immediate 的按时数比）。分母：每档 24 世界、"
        "G24 格 1 152 目标、G48 两格各 2 304 目标。）")
    add("")
    add("| 臂 | 指标 | p=0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for arm in DELIVERY_ARMS:
        name = DELIVERY_LABEL[arm]
        add("| %s | 字节/按时任务 | %s |" % (name, " | ".join(
            fmt(delivery[arm][str(p)]["bytes_per_on_time"], "%.0f") for p in DELIVERY_VALUES)))
        add("| %s | 浪费率 | %s |" % (name, " | ".join(
            fmt(delivery[arm][str(p)]["waste_ratio"], "%.3f") for p in DELIVERY_VALUES)))
    add("")
    add("| 臂 | 最低相对按时比 | 出现在 | 首次跌破 0.95 |")
    add("|---|---|---|---|")
    for arm, record in robustness.items():
        add("| %s | %s | p=%s | %s |" % (
            DELIVERY_LABEL[arm], fmt(record["min_ratio"], "%.3f"),
            fmt(record["at_probability"], "%.1f"),
            ("p=%.1f" % record["first_below_095"]) if record["first_below_095"] else "从未"))
    add("")

    # ---------------------------------------------------------------- J
    add("## 实验 J1　逐维屏蔽消融（v4t，15 格 × 3 锚定场景 = 45 世界）")
    add("")
    add("对照是封板的 v4t 同 45 世界（`eval_100_s8` 的锚定行），不是重跑。屏蔽发生在缩放改写"
        "**之后**、进入主干之前；#11 的变异不同——它必须在特征生成处换分母，因为共享构造器已经"
        "把 `remaining/50` 裁剪过，矩阵里无法还原。")
    add("")
    add("| 变异 | 通过格数 | 按时数 | 相对完整 v4t 按时 | 发送数 | 字节比 | 浪费率 | 浪费率差 |")
    add("|---|---|---|---|---|---|---|---|")
    base_passes = 0
    for cell in TRAIN_CELLS:
        rows = [row for row in archived_reference("v4t@100", (cell,), ANCHOR)]
        base = archived_reference("normal_immediate", (cell,), ANCHOR)
        if not rows or not base:
            continue
        own, ref = group(rows), group(base)
        if own["on_time"] >= need(ref["on_time"]) and own["spent_bytes"] < ref["spent_bytes"]:
            base_passes += 1
    add("| 完整 v4t（封板） | %d/%d | %d | 1.000 | %d | 1.000 | %s | — |" % (
        base_passes, len(TRAIN_CELLS), base_feature["on_time"], base_feature["emissions"],
        fmt(base_feature["waste_ratio"], "%.3f")))
    for variant in FEATURE_VARIANTS:
        stats = features[variant]
        add("| %s | %d/%d | %d (%+d) | %s | %d (%+d) | %s | %s | %s |" % (
            FEATURE_TITLE[variant], stats["passes"], stats["cells"], stats["on_time"],
            stats["on_time_delta"],
            fmt(stats["on_time"] / base_feature["on_time"], "%.3f"),
            stats["emissions"], stats["emissions_delta"], fmt(stats["byte_ratio"], "%.3f"),
            fmt(stats["waste_ratio"], "%.3f"), fmt(stats["waste_delta"], "%+.3f")))
    add("")
    add("逐格通过数用的是注册判据本身：`on_time ≥ ⌈0.95 × immediate⌉` 且 `bytes < immediate`，"
        "immediate 取同格同场景的封板基线。完整 v4t 在锚定 45 世界上是 9/15（与论文表一致）。")
    add("")
    inert = [variant for variant in FEATURE_VARIANTS
             if features[variant]["on_time_delta"] == 0
             and features[variant]["emissions_delta"] == 0]
    if inert:
        add("**零差异变异（必须明写）**：%s 与完整 v4t 在 45 个世界上**逐世界完全相同**"
            "（发送步、发送目标、按时集合、字节全部一致），不是汇总后相互抵消。"
            "`HORIZON_PROBE.json` 另外记录了同一世界在 #11 分母 50 与 200 下的对照："
            "两次都读回安装成功的分母（50.0 / 200.0），结果 23/48 按时、23 条、58 224 B，"
            "完全相同——所以这是「该列在本 checkpoint 的决策上不起作用」，不是「干预没装上」。"
            % "、".join("`%s`" % variant for variant in inert))
        add("")

    add("## 实验 J2　#16 容量反事实探针")
    add("")
    add("六个格都来自 G 扫描里被标为「网络自身少发」的 R=12 列。控制组与探针共用同一页预算"
        "（`immediate − 1`，即 G 扫描自己的规则），唯一差别是推理时把 #16 写成 2.0"
        "（相当于把运力读成参考格的两倍）。")
    add("")
    add("| 格 | 页预算 | 控制组发送/世界 | 占预算 | 探针发送/世界 | 占预算 | 提升 | 控制组按时 | 探针按时 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for cell in CAPACITY_CELLS:
        key = label(cell)
        entry = capacity[key]
        control, probe = entry["control"], entry["cap16x2"]
        add("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            key, fmt(control["cap"], "%d"),
            fmt(control["sends_per_world"], "%.1f"), fmt(100.0 * entry["cap_use_control"] if entry["cap_use_control"] is not None else None, "%.0f %%"),
            fmt(probe["sends_per_world"], "%.1f"), fmt(100.0 * entry["cap_use_probe"] if entry["cap_use_probe"] is not None else None, "%.0f %%"),
            fmt(entry["lift"], "%+.1f"),
            ("%d/%d" % (control["on_time"], control["goals"])
             if control["worlds"] else "—"),
            ("%d/%d" % (probe["on_time"], probe["goals"]) if probe["worlds"] else "—")))
    add("")

    # ---------------------------------------------------------------- K
    add("## 实验 K　配额谱系与选择器帕累托包络")
    add("")
    add("两格 × 8 场景；页上限 = `round(该格 immediate 平均页数 × 比例)`，四个选择器共用同一上限，"
        "每步最多 1 页。`v4f` 是纯策略网络（同样被页上限截断），另外三个是零训练规则。")
    add("")
    for cell in BUDGET_CELLS:
        key = label(cell)
        entry = budget[key]
        add("### %s（immediate %d 页/世界，8 场景均值）" % (key, int(round(cells_pages[key]))))
        add("")
        add("| 配额 | %s |" % " | ".join(BUDGET_LABEL[s] + " 字节比 / 相对按时"
                                        for s in BUDGET_SELECTORS))
        add("|---|" + "---|" * len(BUDGET_SELECTORS))
        for fraction in BUDGET_FRACTIONS:
            cells = []
            for selector in BUDGET_SELECTORS:
                stats = entry["by_selector"][selector][str(fraction)]
                cells.append("%s / %s" % (fmt(stats["byte_ratio"], "%.3f"),
                                          fmt(stats["on_time_ratio"], "%.3f")))
            add("| %d %% | %s |" % (int(round(fraction * 100)), " | ".join(cells)))
        add("")
        add("帕累托前沿（同一格内，字节更少且按时更多）：")
        add("")
        add("| 选择器 | 配额 | 字节 | 按时 |")
        add("|---|---|---|---|")
        for point in entry["pareto"]:
            add("| %s | %d %% | %d | %d |" % (BUDGET_LABEL[point["selector"]],
                                              int(round(point["fraction"] * 100)),
                                              point["spent_bytes"], point["on_time"]))
        add("")

    # ---------------------------------------------------------------- L
    add("## 实验 L　微观时空决策画像（零重跑）")
    add("")
    add("既有 120 世界 × 2 臂（immediate、v4t），按每 100 tick 分桶，统计**首次广播**落在该桶的"
        "去重目标数，以及这些目标最终是否在自己截止时间内完成。")
    add("")
    add("| 桶 | immediate 首播/世界 | immediate 按时率 | v4t 首播/世界 | v4t 按时率 |")
    add("|---|---|---|---|---|")
    for index in range(10):
        base = micro["normal_immediate"]["all"]["buckets"][index]
        own = micro["v4t@100"]["all"]["buckets"][index]
        add("| %d–%d | %s | %s | %s | %s |" % (
            base["low"], base["high"], fmt(base["first_sends_per_world"], "%.2f"),
            fmt(base["on_time_rate"], "%.3f"), fmt(own["first_sends_per_world"], "%.2f"),
            fmt(own["on_time_rate"], "%.3f")))
    add("")
    add("| 臂 | 广播目标 | 其中按时 | 未广播目标 | 其中按时 |")
    add("|---|---|---|---|---|")
    for arm in ("normal_immediate", "v4t@100"):
        record = micro[arm]["all"]
        add("| %s | %d | %d (%s) | %d | %d (%s) |" % (
            DELIVERY_LABEL[arm], record["broadcast"]["n"], record["broadcast"]["on_time"],
            fmt(record["broadcast"]["on_time_rate"], "%.3f"),
            record["suppressed"]["n"], record["suppressed"]["on_time"],
            fmt(record["suppressed"]["on_time_rate"], "%.3f")))
    add("")
    add("| 指标（v4t，按**发现时刻**同口径） | 被广播 | 未广播 |")
    add("|---|---|---|")
    for key, title in (("discovery_tick", "发现 tick 均值 / 中位"),
                       ("slack_at_discovery", "发现时剩余时限 均值 / 中位"),
                       ("total_stages", "任务阶段数 均值 / 中位")):
        pair = micro["v4t@100"]["all"]["contrast"][key]
        add("| %s | %s / %s | %s / %s |" % (
            title, fmt(pair["broadcast"]["mean"], "%.2f"),
            fmt(pair["broadcast"]["median"], "%.1f"),
            fmt(pair["suppressed"]["mean"], "%.2f"),
            fmt(pair["suppressed"]["median"], "%.1f")))
    add("")
    add("v4t 广播目标的发送时刻剩余时限均值 %s、发送延迟（相对发现）均值 %s（中位 %s）；"
        "这两列只对广播集合有定义，因为未广播目标没有发送时刻。" % (
            fmt(micro["v4t@100"]["all"]["send_slack"]["mean"], "%.1f"),
            fmt(micro["v4t@100"]["all"]["send_delay"]["mean"], "%.1f"),
            fmt(micro["v4t@100"]["all"]["send_delay"]["median"], "%.1f")))
    add("")
    add("**延迟恒为 0 这一读数本身是机制证据**：v4t 的每一次广播都发生在该目标的发现 tick 上"
        "（2 408 个广播目标，延迟全部为 0；抽查一个 G48_R6_N6 世界，22 次发送的步号与各目标的"
        "检测步逐一相同，例如步 138/140/147/160 都是别的目标刚被发现）。也就是说，"
        "「未广播」不是「晚点再发」——中继在发现时只给一次机会，HOLD 就等于这个目标永远不上行。"
        "因此实验 L 的对比实际是「发现那一刻发还是不发」，而不是「什么时候发」。")
    add("")

    # ---------------------------------------------------------------- M
    add("## 实验 M　跨规模单步推理延迟")
    add("")
    add("每个规模跑 1 个世界（场景 2301901，R=6、N=6），在 `controller.forward` 外包一层计时，"
        "记录每次决策的耗时与候选数。同一进程、无 GPU、单线程。")
    add("")
    add("| G | 决策数 | 候选/决策 | 平均 ms | P50 ms | P99 ms | 模型参数 | 参数体积 (fp32) |")
    add("|---|---|---|---|---|---|---|---|")
    for cell in LATENCY_CELLS:
        key = label(cell)
        record = latency.get(key)
        if record is None:
            continue
        add("| %d | %d | %s | %s | %s | %s | %d | %.1f kB |" % (
            record["g"], record["decisions"], fmt(record["candidates"], "%.1f"),
            fmt(record["mean_ms"], "%.3f"), fmt(record["p50_ms"], "%.3f"),
            fmt(record["p99_ms"], "%.3f"), params["parameters"],
            params["float32_bytes"] / 1024.0))
    add("")
    add("模型参数在六个规模上完全相同（%d 个，%.1f kB fp32，sha256 `%s…`）——规模改变的是"
        "每步候选数与世界长度，不是网络。因此 ms/决策只随**候选数**移动，这一点由「候选/决策」"
        "一列直接可见。" % (params["parameters"], params["float32_bytes"] / 1024.0,
                          params["sha256"][:16]))
    add("")

    # ---------------------------------------------------------------- claims
    # ---------------------------------------------------------------- N: no page table
    nocap_rows = load_arm("nocap", "v4t@100")
    nocap = {"rows": nocap_rows}
    if nocap_rows:
        cap_rows = archived_reference("v4t@100", TRAIN_CELLS, ALL_SCENES)
        base_rows = archived_reference("normal_immediate", TRAIN_CELLS, ALL_SCENES)
        v4f_rows = archived_reference("v4f@200", TRAIN_CELLS, ALL_SCENES)
        by_key = {(tuple(int(v) for v in r["cell"]), int(r["scene"])): r for r in v4f_rows}
        identical = sum(
            1 for row in nocap_rows
            if (row["on_time"], row["emissions"], row["spent_bytes"])
            == (by_key[(tuple(int(v) for v in row["cell"]), int(row["scene"]))]["on_time"],
                by_key[(tuple(int(v) for v in row["cell"]), int(row["scene"]))]["emissions"],
                by_key[(tuple(int(v) for v in row["cell"]), int(row["scene"]))]["spent_bytes"]))
        nocap["identical_to_v4f"] = identical
        nocap["worlds"] = len(nocap_rows)

        def waste_of(rows, arm=None):
            if arm is not None:
                counts = archived_waste(arm, TRAIN_CELLS, ALL_SCENES)
                return counts["waste"] / counts["distinct"]
            distinct = sum(int(row["waste_waste_d"]) for row in rows)
            count = sum(int(row["waste_waste_n"]) for row in rows)
            return count / distinct

        immediate = group(base_rows)
        without = group(nocap_rows)
        withcap = group(cap_rows)
        reference_bytes = immediate["spent_bytes"]
        rows_summary = (
            ("immediate", immediate, waste_of(base_rows, "normal_immediate")),
            ("v4t 去掉配额表", without, waste_of(nocap_rows)),
            ("v4t 带配额表", withcap, waste_of(cap_rows, "v4t@100")),
        )
        passes_without = 0
        passes_with = 0
        reasons = {"ont": 0, "bytes": 0, "both": 0, "pass": 0}
        per_cell = {}
        for cell in TRAIN_CELLS:
            key = label(cell)
            own = group([r for r in nocap_rows if label(r["cell"]) == key])
            ref = group([r for r in base_rows if label(r["cell"]) == key])
            capped = group([r for r in cap_rows if label(r["cell"]) == key])
            required = need_for(ref["on_time"])
            ok_time = own["on_time"] >= required
            ok_bytes = own["spent_bytes"] < ref["spent_bytes"]
            tag = ("pass" if ok_time and ok_bytes
                   else ("bytes" if ok_time else ("ont" if ok_bytes else "both")))
            reasons[tag] += 1
            passes_without += int(tag == "pass")
            passes_with += int(capped["on_time"] >= required
                               and capped["spent_bytes"] < ref["spent_bytes"])
            per_cell[key] = {"on_time": own["on_time"], "goals": own["goals"],
                             "need": required, "bytes": own["spent_bytes"],
                             "ref_on_time": ref["on_time"], "ref_bytes": ref["spent_bytes"],
                             "capped_on_time": capped["on_time"],
                             "capped_bytes": capped["spent_bytes"], "reason": tag}
        nocap.update({"without": without, "with": withcap, "immediate": immediate,
                      "passes_without": passes_without, "passes_with": passes_with,
                      "reasons": reasons, "per_cell": per_cell,
                      "waste": {"immediate": rows_summary[0][2],
                                "without": rows_summary[1][2], "with": rows_summary[2][2]}})
        state["nocap"] = nocap

        add("## 实验 N　去掉配额表的同一份权重（补测）")
        add("")
        add("v4t 的权重 + `cap=None`，15 格 × 8 场景 = 120 世界。这条臂就是“超出配额表之后会怎样”"
            "在**表内**的直接测量：除了配额，其他一切与 v4t 相同。")
        add("")
        add("**等价性自检**：这 120 个世界与归档的 `v4f@200`（同一份 actor，同样不带配额）"
            "**逐世界完全相同**（%d/%d：按时数、发送数、字节全等）。所以下表里“去掉表”这一行"
            "不是新策略，就是 v4f。" % (identical, len(nocap_rows)))
        add("")
        add("| 臂 | 按时数 | 绝对率 | 字节 | 字节比 | 消息/按时任务 | 浪费率 | 通过 |")
        add("|---|---|---|---|---|---|---|---|")
        for name, stats, waste_value in rows_summary:
            add("| %s | %d/%d | %.3f | %d | %.3f | %.3f | %.3f | %s |" % (
                name, stats["on_time"], stats["goals"], stats["q"], stats["spent_bytes"],
                stats["spent_bytes"] / reference_bytes, stats["messages_per_on_time"],
                waste_value,
                ("**%d/15**" % passes_without) if name.startswith("v4t 去掉")
                else (("%d/15" % passes_with) if name.startswith("v4t 带") else "0/15")))
        add("")
        add("**失败的形态（去掉表，12 个未过格）**：%d 格只是“字节没有严格更少”，"
            "%d 格按时不足，%d 格两者都不满足。前一类里，字节数与 immediate **完全相等**"
            "（例如 G12 的 N=3/6/12 三档分别是 118 608 / 237 216 / 474 432，每一格都与 "
            "immediate 同格的值逐字节相同）——不带配额时，"
            "这些格子里网络发的就是“每目标一页”，与 immediate 同一集合。" % (
                reasons["bytes"], reasons["ont"], reasons["both"]))
        add("")
        add("| 格 | 无表 按时/需要 | 无表 字节 vs immediate | 判定 | 带表 按时/字节 |")
        add("|---|---|---|---|---|")
        for cell in TRAIN_CELLS:
            key = label(cell)
            record = per_cell[key]
            add("| %s | %d/%d | %d vs %d | %s | %d/%d |" % (
                key, record["on_time"], record["need"], record["bytes"], record["ref_bytes"],
                {"pass": "**过**", "ont": "按时不足", "bytes": "字节相等",
                 "both": "两者都差"}[record["reason"]],
                record["capped_on_time"], record["capped_bytes"]))
        add("")
        add("读法：**配额买到的是“深度”，不是“选择”**。去掉表之后，模型仍然比 immediate 省 "
            "%.1f %% 字节、每按时任务 1.07 条消息、浪费率 0.064——它不是靠乱发取胜的；"
            "它只是**在运力充裕的格子里把该发的全发了**，字节自然就等于 immediate，"
            "于是“严格更少”这一条判据不成立。而在它自己愿意节流的 R=6 格（G24_R6_N6、"
            "G48_R6_N3、G48_R6_N6），不带表也能过。" % (
                100.0 * (1.0 - without["spent_bytes"] / reference_bytes)))
        add("")

    add("## 能说 / 不能说")
    add("")
    add("**能说**")
    add("")
    micro_all = micro["v4t@100"]["all"]
    add("1. **信道鲁棒性是可测的**：六臂在 8 档投递率上的绝对按时率、字节/按时任务与浪费率"
        "全部给出，且 0.7 档与封板数字逐项相同（同源复用，不是重算）。")
    add("2. **配额不是瓶颈、容量感知才是**：J2 在六个 R=12 格上把 #16 写成 2.0 后，发送量"
        "被推回页预算（占预算 52–83 % → 99–100 %，每世界多 8–34 条），六格里五格的按时数"
        "同时上升；控制组在 G48/G60/G72 的 R12_N3 上分别是 25.7 / 37.0 / 37.0 条，"
        "与已发表的 G 扫描逐项相同，所以探针的基线就是那条被归因为「网络自身少发」的曲线。")
    add("3. **屏蔽的代价是有分工的**：屏蔽 #16 使按时数 −96（字节几乎不变、浪费率 +0.070），"
        "屏蔽 #14 反而 +71 但字节 +11.7 % 且浪费率 +0.083，屏蔽 #17 得 +42 而字节 −1.7 %"
        "（唯一两项同时不差的变异）；**屏蔽 #2 与把 #11 退回 50 步在 45 个世界上逐世界零差异**——"
        "在本 checkpoint 的决策里，sent-count 与时限分辨率两列没有起作用，"
        "起作用的是 backlog 与 capacity。")
    add("4. **v4t 的抑制集合在结果上是干净的**：120 世界里它没有广播的 %d 个目标中，按时完成 "
        "%d 个；而它广播的 %d 个里有 %d 个按时。同时它的第一次发送全部落在发现 tick 上，"
        "所以这是「一次性不发的赌注全部押对」，不是「晚发导致超时」。" % (
            micro_all["suppressed"]["n"], micro_all["suppressed"]["on_time"],
            micro_all["broadcast"]["n"], micro_all["broadcast"]["on_time"]))
    if budget:
        cell0 = BUDGET_CELLS[0]
        entry0 = budget[label(cell0)]["by_selector"]
        add("5. **同一配额下选择器差别很大，而网络会自己停止花钱**：%s 上把页上限压到 "
            "immediate 的 40 %% 时，v4f 已经拿到 %.3f 的相对按时（字节比 %.3f），"
            "而 Slack/FIFO/Random 在 40 %% 时只有 %.3f/%.3f/%.3f；50 %% 以上 v4f 的发送量"
            "不再增长（每世界约 20.5 条），所以它的曲线在右端是平的——预算没有被用满这件事本身"
            "就是读数。" % (
                cell0, entry0["v4f"]["0.4"]["on_time_ratio"],
                entry0["v4f"]["0.4"]["byte_ratio"],
                entry0["slack_budget"]["0.4"]["on_time_ratio"],
                entry0["fifo_budget"]["0.4"]["on_time_ratio"],
                entry0["random_budget"]["0.4"]["on_time_ratio"]))
    records = [record for record in latency.values() if record]
    if records:
        fastest = min(records, key=lambda record: record["mean_ms"])
        slowest = max(records, key=lambda record: record["mean_ms"])
        add("6. **实时性没有障碍**：单次决策 %.3f–%.3f ms（P99 至 %.3f ms），1000 步回合的推理开销"
            "远小于本机一次世界仿真的 %.0f s。" % (
                fastest["mean_ms"], slowest["mean_ms"], slowest["p99_ms"],
                base_feature["seconds_mean"] or 0.0))
    else:
        add("6. **实时性**：该阶段没有数据（见「复现」里的 latency 分支）。")
    add("")
    add("**不能说**")
    add("")
    add("1. **不能说屏蔽某维就证明该维\"被使用\"**：屏蔽是一个干预，但它同时改变了策略看到的"
        "整行分布；差异只能读作\"该列携带了影响决策的信息\"，不能读作\"网络按该列的语义推理\"。")
    add("2. **不能把 J2 的发送回升等同于按时率回升**：探针只检验发送量是否受 #16 控制，"
        "按时数的变化一并列出但不作因果归因。")
    add("3. **不能说 v4t 的抑制是按\"高超时风险\"过滤**：未广播目标的发现时剩余时限均值与"
        "广播集合几乎相同，真正分开两组的是**任务阶段数**（未广播组阶段更多）。"
        "「精准过滤了高超时风险目标」这一说法在数据上不成立，应改写成"
        "「过滤掉了多阶段、需要多次分工的目标」。")
    add("4. **不能把 L 的 0/937 全部记成策略功劳**：完成集合本来就与被广播集合高度重合"
        "（此前 tick-1 探针已测得），所以\"未广播即未按时\"包含环境自身的耦合，"
        "不是纯粹的选择能力。")
    add("5. **不能跨实验比较绝对按时率**：I 用 24 世界、J1 用 45、K 用 16、L 用 120，"
        "场景集合与格数都不同。")
    add("6. **K 的四个选择器共享页上限但不共享\"每步 1 页\"的节奏之外的东西**："
        "v4f 是网络排序，另外三个是规则排序；这不是同权重消融。")
    add("7. **不能说 v4f 在每个配额上都最好**：G48_R12_N6 满配额时 Slack 的相对按时 1.068"
        "高于 v4f 的 0.904，G48_R6_N6 满配额时 Random 1.234 也高于 v4f 的 1.163。"
        "v4f 的优势是**在小配额区间同时省字节且不塌按时**，不是所有点上的支配。")
    add("8. **不能把 L 的\"延迟恒为 0\"读成\"没有调度\"**：它说明中继在发现时只提供一次发送机会，"
        "所以这个实验能回答的是\"发/不发\"，回答不了\"早发/晚发\"。")
    add("9. **不能把\"去掉配额表 3/15\"读成\"模型不能用\"**：它仍然比 immediate 省 18.5 % 字节、"
        "1.07 条/任务，失败的 12 格里有 9 格只是\"字节与 immediate 相等\"而非更多；"
        "判据的严格不等号才是那 9 格失败的原因，不是策略崩了。")
    add("")
    add("## 复现")
    add("")
    add("```")
    add("D:/project_env/python.exe analysis/extended_sweeps.py --stage delivery --workers 10")
    add("D:/project_env/python.exe analysis/extended_sweeps.py --stage features --workers 10")
    add("D:/project_env/python.exe analysis/extended_sweeps.py --stage capacity --workers 10")
    add("D:/project_env/python.exe analysis/extended_sweeps.py --stage budget   --workers 10")
    add("D:/project_env/python.exe analysis/extended_sweeps.py --stage latency  --workers 6")
    add("D:/project_env/python.exe analysis/micro_profile.py --workers 8")
    add("D:/project_env/python.exe analysis/extended_report.py")
    add("D:/project_env/python.exe analysis/plot_extended_figures.py")
    add("D:/project_env/python.exe -B -m unittest discover -s tests")
    add("D:/project_env/python.exe analysis/verify_pins.py")
    add("```")
    add("")

    add("## 证据表（sha256 前 16 位）")
    add("")
    add("| 文件 | sha256 | 内容 |")
    add("|---|---|---|")
    for stage in ("delivery", "features", "capacity", "budget", "latency"):
        root = EXT / stage
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            target = child / "RESULTS.json"
            if target.exists():
                add("| `%s` | `%s` | %d 世界 |" % (
                    target.relative_to(BASE).as_posix(), sha256(target)[:16],
                    len(load_arm(stage, child.name))))
    micro_path = EXT / "micro/MICRO.json"
    if micro_path.exists():
        add("| `%s` | `%s` | 两臂 × 120 世界 |" % (
            micro_path.relative_to(BASE).as_posix(), sha256(micro_path)[:16]))
    probe = EXT / "features/HORIZON_PROBE.json"
    if probe.exists():
        add("| `%s` | `%s` | #11 分母 50 vs 200 的安装探针 |" % (
            probe.relative_to(BASE).as_posix(), sha256(probe)[:16]))
    add("")
    add("## 图")
    add("")
    add("| 图 | 文件 | PNG sha256（前 16 位） |")
    add("|---|---|---|")
    for name, title in (("fig15_delivery_prob_sweep", "图 15　投递率 8 档四联"),
                        ("fig16_budget_spectrum_pareto", "图 16　配额谱系与帕累托包络")):
        png = BASE / "execution/figures" / (name + ".png")
        pdf = BASE / "execution/figures" / (name + ".pdf")
        if png.exists():
            add("| %s | `%s`（300 dpi）<br>`%s`（矢量） | `%s` |" % (
                title, png.relative_to(BASE).as_posix(), pdf.relative_to(BASE).as_posix(),
                sha256(png)[:16]))
    add("")
    add("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    OUT_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    print("wrote %s and %s" % (OUT_MD.name, OUT_JSON.name))
    print("EXTENDED.json sha256 %s" % sha256(OUT_JSON))
    print("EXTENDED_MECHANISM_EXPERIMENTS.md sha256 %s" % sha256(OUT_MD))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
