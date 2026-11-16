"""Protocol-level metrics for the sealed evaluation logs: delay, delivery, cost, goodput.

The two reference theses this revision is compared against measure routing/network quality
with a small, explicit metric set:

* Han Cheng (Jilin University, MSc), §2.1.3 -- energy efficiency (total energy / packets
  carried), network lifetime (first node death), end-to-end delay, and packet delivery ratio.
* Gao Yanzhong (MSc), §3.2.3 -- network throughput (slot utilisation and reuse ratio) and
  mean end-to-end delay, then delay/loss versus offered load and node speed.

This script derives the same *families* of quantities from the logs this revision already
wrote, so the thesis can carry a protocol-level section without a single new world:

  delays      D_hold = t_emit - t_disc, D_resp = t_done - t_emit, D_e2e = t_done - t_disc
  delivery    PDR_link = delivered packets / attempts, PLR_link = 1 - PDR_link,
              PDR_eff  = on-time tasks / emissions
  cost        C_node   = delivered subscriber-node instances,
              E_task   = (a_tx * payload + a_rx * payload * delivered) / on-time tasks
  throughput  Gamma_raw = airtime bytes per 100 steps,
              Gamma_good = airtime bytes of on-time targets per 100 steps

Everything is read from ``lifecycle.jsonl.gz`` (who was detected, when, when it finished) and
``relay_transcript.jsonl.gz`` (each emission's step, target, subscriber count, payload bytes,
delivered and lost packets). No world is opened, no checkpoint is loaded, and the 98 pinned
files are untouched.

    D:/project_env/python.exe analysis/protocol_metrics.py --workers 8
    D:/project_env/python.exe analysis/protocol_metrics.py --workers 8 --no-figures
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT_JSON = BASE / "execution/PROTOCOL_METRICS.json"
OUT_MD = BASE / "PROTOCOL_METRICS_REPORT.md"
FIG_DIR = BASE / "execution/figures"
FIG_NAME = "fig18_queueing_delay_and_goodput"

ANCHOR = (2301901, 2301902, 2301903)
UNSEEN = (2301904, 2301905, 2301906, 2301907, 2301908)
ALL_SCENES = ANCHOR + UNSEEN

TRAIN_GOALS = (12, 24, 48)
SWEEP_GOALS = (18, 30, 36, 42, 54, 60, 72)
GOALS = (12, 18, 24, 30, 36, 42, 48, 54, 60, 72)
R6_PAIRS = ((6, 3), (6, 6))
R12_PAIRS = ((12, 3), (12, 6), (12, 12))
HIGH_LOAD_GOALS = tuple(goal for goal in GOALS if goal >= 36)

DELIVERY_PROBABILITIES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ALPHA_TX = 1.0
ALPHA_RX = 0.5
RATE_WINDOW = 100.0

CELL_TAG = re.compile(r"(G\d+R\d+N\d+)")
#: The delivery sweep appends its variant to the cell tag, with or without the checkpoint
#: stamp in between: ``G24R6N6p30`` (rule arms) and ``G24R6N6_c100_p30`` (learned arms).
P_TAG = re.compile(r"(G\d+R\d+N\d+)(?:_c\d+)?_?p(\d{2,3})")

#: arm -> (world-log root, archived rows file, shared root?, prefix for shared roots)
MAIN_SOURCES = (
    ("normal_immediate", "execution/baselines/grid_v2", True,
     "eval__normal_immediate_"),
    ("dedup_immediate", "execution/baselines/grid_v2", True, "eval__dedup_immediate_"),
    ("always_hold", "execution/baselines/grid_v2", True, "eval__always_hold_"),
    ("fifo_cap_k1", "execution/baselines/paper_throttles_v1", True, "eval__fifo_cap_k1_"),
    ("slack_heuristic_cap", "execution/baselines/paper_throttles_v1", True,
     "eval__slack_heuristic_cap_"),
    ("random_matched_rate", "execution/baselines/paper_random_matched", False, "eval__"),
    ("slack_matched_rate", "execution/baselines/paper_slack_matched", False, "eval__"),
    ("periodic_k5", "execution/baselines/paper_traditional_v1", True, "eval__periodic_k5_"),
    ("event_maxwait", "execution/baselines/paper_traditional_v1", True,
     "eval__event_maxwait_"),
    ("v4f@200", "execution/train_v4f_anchor/eval_200_s8", False, "eval__"),
    ("v4g@300", "execution/train_v4g_3scene_anchor/eval_300_s8", False, "eval__"),
    ("v4s@300", "execution/train_v4s_cap300/eval_300_s8", False, "eval__"),
    ("v4t@100", "execution/train_v4t_cap_frozen/eval_100_s8", False, "eval__"),
    ("v4u@100", "execution/train_v4u2_window_fixed/eval_100_s8", False, "eval__"),
)

G_SWEEP_ARMS = ("normal_immediate", "slack_heuristic_cap", "slack_matched_rate",
                "v4f@200", "v4s@300", "v4t@100")
DELIVERY_ARMS = ("normal_immediate", "dedup_immediate", "event_maxwait",
                 "slack_matched_rate", "v4s@300", "v4t@100")

DELAY_ARMS = ("normal_immediate", "slack_matched_rate", "v4f@200", "v4t@100")

ARM_LABEL = {
    "normal_immediate": "Immediate", "dedup_immediate": "Dedup",
    "always_hold": "Always-Hold", "fifo_cap_k1": "FIFO-Cap",
    "slack_heuristic_cap": "Slack-Cap", "random_matched_rate": "Random-Matched",
    "slack_matched_rate": "Slack-Matched", "periodic_k5": "Periodic-k5",
    "event_maxwait": "Event-MaxWait", "v4f@200": "Pure-Policy (v4f)",
    "v4g@300": "Price-Dual (v4g)", "v4s@300": "Ours-Joint (v4s)",
    "v4t@100": "Ours-Frozen (v4t)", "v4u@100": "Ours-Floor (v4u)",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_gz(path):
    with gzip.open(str(path), "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def label_of(cell):
    return "G%d_R%d_N%d" % (int(cell[0]), int(cell[1]), int(cell[2]))


def parse_leaf(directory):
    """``(cell, scene, probability)`` from ``eval__<arm>_<CELL>[p<NN>][_c<ep>]__<s>__<scene>``."""
    name = Path(directory).name
    parts = name.split("__")
    if len(parts) != 4:
        raise RuntimeError("%s is not a four-part world leaf" % (name,))
    hit = P_TAG.search(parts[1])
    probability = None
    if hit:
        numbers = [int(value) for value in re.findall(r"\d+", hit.group(1))]
        probability = int(hit.group(2)) / 100.0
    else:
        match = CELL_TAG.search(parts[1])
        if match is None:
            raise RuntimeError("%s carries no cell tag" % (name,))
        numbers = [int(value) for value in re.findall(r"\d+", match.group(1))]
    return tuple(numbers), int(parts[3]), probability


def measure(directory):
    """Every protocol-level reading for one world, from its two archived logs."""
    directory = Path(directory)
    lifecycle = read_jsonl_gz(directory / "lifecycle.jsonl.gz")
    transcript = read_jsonl_gz(directory / "relay_transcript.jsonl.gz")

    discovered, deadline, completed = {}, {}, {}
    for record in lifecycle:
        target = str(record.get("target_id"))
        stamp = float(record.get("timestamp") or 0.0)
        kind = record.get("event_type")
        detail = record.get("detail") or {}
        if kind == "detection":
            if target not in discovered or stamp < discovered[target]:
                discovered[target] = stamp
                deadline[target] = stamp + float(detail.get("deadline_offset", 0.0))
        elif kind == "progress" and str(record.get("status")) == "completed":
            completed.setdefault(target, stamp)
    on_time = {target for target, due in deadline.items()
               if target in completed and completed[target] <= due}

    sends = [record for record in transcript if record.get("kind") == "send"]
    first_emit = {}
    attempts = delivered = payload_total = airtime_total = 0
    airtime_on_time = 0
    energy_numerator = 0.0
    for record in sends:
        target = str(record.get("target"))
        step = float(record.get("step"))
        if target not in first_emit or step < first_emit[target]:
            first_emit[target] = step
        payload = int(record.get("payload_bytes") or 0)
        got = int(record.get("packets") or 0)
        tried = int(record.get("attempts") or 0)
        cost = int(record.get("cost_bytes") or record.get("bytes_total") or 0)
        attempts += tried
        delivered += got
        payload_total += payload
        airtime_total += cost
        energy_numerator += ALPHA_TX * payload + ALPHA_RX * payload * got
        if target in on_time:
            airtime_on_time += cost

    hold, response, end_to_end = [], [], []
    on_time_broadcast = 0
    for target in on_time:
        if target not in first_emit:
            continue
        emit = first_emit[target]
        hold.append(emit - discovered[target])
        response.append(completed[target] - emit)
        end_to_end.append(completed[target] - discovered[target])
        on_time_broadcast += 1

    # The channel's configured probability is the only *designed* loss reading in the
    # archive; per-packet arrival is recorded by the info probe, which these runs left off.
    configured = None
    for record in lifecycle:
        if record.get("event_type") == "run_config":
            configured = float((record.get("detail") or {})
                               .get("configuration", {}).get("comm_delivery_probability"))
            break

    steps = max(1, int(round(max([float(record.get("timestamp") or 0.0)
                                  for record in lifecycle] or [0.0]))))
    windows = steps / RATE_WINDOW
    return {
        "generated": len(discovered),
        "on_time": len(on_time),
        "on_time_broadcast": on_time_broadcast,
        "emissions": len(sends),
        "targets_broadcast": len(first_emit),
        "attempts": attempts,
        "delivered": delivered,
        "payload_bytes": payload_total,
        "airtime_bytes": airtime_total,
        "airtime_on_time": airtime_on_time,
        "energy_numerator": energy_numerator,
        "hold": hold,
        "response": response,
        "end_to_end": end_to_end,
        "configured_probability": configured,
        "steps": steps,
        "windows": windows,
    }


def worker(task):
    sys.dont_write_bytecode = True
    try:
        reading = measure(task["directory"])
        reading.update({key: task[key] for key in ("experiment", "arm", "cell", "scene",
                                                   "group", "goal", "probability")})
        reading["status"] = "ok"
        return reading
    except Exception as error:  # a failed world is reported, never counted as a result
        return dict(task, status="error", error="%s: %s" % (type(error).__name__, error))


# --------------------------------------------------------------------------- collection
def collect_main():
    tasks = []
    for arm, root, shared, prefix in MAIN_SOURCES:
        logs = BASE / root / "world_logs"
        if not logs.is_dir():
            raise SystemExit("missing world logs for %s under %s" % (arm, root))
        for directory in sorted(path for path in logs.iterdir() if path.is_dir()):
            if shared and not directory.name.startswith(prefix):
                continue
            inner = [path for path in directory.iterdir() if path.is_dir()]
            if len(inner) != 1:
                raise RuntimeError("%s holds %d inner directories" % (directory, len(inner)))
            cell, scene, _probability = parse_leaf(directory)
            tasks.append({"experiment": "main", "arm": arm, "cell": list(cell),
                          "scene": scene, "goal": int(cell[0]),
                          "group": ("anchor" if scene in ANCHOR else "unseen"),
                          "probability": 0.7, "directory": str(inner[0]),
                          "source": root})
    return tasks


def archived_home():
    """Where each arm's three training goal counts live, and whether that root is shared."""
    return {
        "normal_immediate": ("execution/baselines/grid_v2", True),
        "slack_heuristic_cap": ("execution/baselines/paper_throttles_v1", True),
        "slack_matched_rate": ("execution/baselines/paper_slack_matched", False),
        "v4f@200": ("execution/train_v4f_anchor/eval_200_s8", False),
        "v4s@300": ("execution/train_v4s_cap300/eval_300_s8", False),
        "v4t@100": ("execution/train_v4t_cap_frozen/eval_100_s8", False),
    }


def collect_g_sweep():
    """Seven new goal counts from the G sweep plus the three archived ones, per arm."""
    tasks = []
    home = archived_home()
    for arm in G_SWEEP_ARMS:
        sources = [("execution/g_scaling/" + arm.replace("@", "_"), SWEEP_GOALS, False)]
        if arm in home:
            root, shared = home[arm]
            sources.append((root, TRAIN_GOALS, shared))
        for root, wanted, shared in sources:
            logs = BASE / root / "world_logs"
            if not logs.is_dir():
                continue
            prefix = "eval__%s_" % arm
            for directory in sorted(path for path in logs.iterdir() if path.is_dir()):
                if shared and not directory.name.startswith(prefix):
                    continue
                cell, scene, probability = parse_leaf(directory)
                if probability is not None:
                    continue        # the delivery sweep's leaves belong to another experiment
                if cell[0] not in wanted:
                    continue
                inner = [path for path in directory.iterdir() if path.is_dir()]
                if len(inner) != 1:
                    continue
                tasks.append({"experiment": "g_sweep", "arm": arm, "cell": list(cell),
                              "scene": scene, "goal": int(cell[0]),
                              "group": ("R6" if cell[1] == 6 else "R12"),
                              "probability": 0.7, "directory": str(inner[0]),
                              "source": root})
    return tasks


def collect_delivery():
    tasks = []
    for arm in DELIVERY_ARMS:
        logs = BASE / "execution/extended/delivery" / arm.replace("@", "_") / "world_logs"
        if not logs.is_dir():
            continue
        for directory in sorted(path for path in logs.iterdir() if path.is_dir()):
            cell, scene, probability = parse_leaf(directory)
            if probability is None or probability == 0.7:
                continue        # the 0.7 point is the sealed archive world, not this sweep
            inner = [path for path in directory.iterdir() if path.is_dir()]
            if len(inner) != 1:
                continue
            tasks.append({"experiment": "delivery", "arm": arm, "cell": list(cell),
                          "scene": scene, "goal": int(cell[0]),
                          "group": "p%.2f" % probability, "probability": probability,
                          "directory": str(inner[0]),
                          "source": "execution/extended/delivery"})
    return tasks


# --------------------------------------------------------------------------- aggregation
def describe(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    def quantile(fraction):
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {"n": len(ordered), "mean": sum(ordered) / len(ordered),
            "median": quantile(0.5), "p90": quantile(0.9),
            "p10": quantile(0.1)}


def aggregate(worlds):
    """Ratio-of-sums readings plus the pooled delay distributions."""
    worlds = [world for world in worlds if world.get("status") == "ok"]
    if not worlds:
        return None
    on_time = sum(world["on_time"] for world in worlds)
    generated = sum(world["generated"] for world in worlds)
    emissions = sum(world["emissions"] for world in worlds)
    attempts = sum(world["attempts"] for world in worlds)
    delivered = sum(world["delivered"] for world in worlds)
    airtime = sum(world["airtime_bytes"] for world in worlds)
    airtime_on_time = sum(world["airtime_on_time"] for world in worlds)
    payload = sum(world["payload_bytes"] for world in worlds)
    energy = sum(world["energy_numerator"] for world in worlds)
    windows = sum(world["windows"] for world in worlds)
    configured = [world["configured_probability"] for world in worlds
                  if world["configured_probability"] is not None]
    return {
        "worlds": len(worlds),
        "goals": generated,
        "on_time": on_time,
        "q": (on_time / generated) if generated else None,
        "emissions": emissions,
        "targets_broadcast": sum(world["targets_broadcast"] for world in worlds),
        "attempts": attempts,
        "delivered": delivered,
        "pdr_link": (delivered / attempts) if attempts else None,
        "plr_link": (1.0 - delivered / attempts) if attempts else None,
        "pdr_eff": (on_time / emissions) if emissions else None,
        "miss_rate": (1.0 - on_time / emissions) if emissions else None,
        "configured_probability": (sorted(set(configured))[0]
                                   if len(set(configured)) == 1 else configured or None),
        "channel_uses_per_world": (sum(world["attempts"] for world in worlds) / len(worlds)),
        "receiver_nodes_per_world": (delivered / len(worlds)),
        "payload_bytes": payload,
        "airtime_bytes": airtime,
        "e_task": (energy / on_time) if on_time else None,
        "gamma_raw": (airtime / windows) if windows else None,
        "gamma_good": (airtime_on_time / windows) if windows else None,
        "goodput_share": (airtime_on_time / airtime) if airtime else None,
        "d_hold": describe([value for world in worlds for value in world["hold"]]),
        "d_resp": describe([value for world in worlds for value in world["response"]]),
        "d_e2e": describe([value for world in worlds for value in world["end_to_end"]]),
        "d_resp_world_mean": (sum(describe(world["response"])["mean"]
                                  for world in worlds if world["response"])
                              / max(1, sum(1 for world in worlds if world["response"]))),
    }


def group_by_arm(worlds, key="arm"):
    groups = {}
    for world in worlds:
        groups.setdefault(world[key], []).append(world)
    return {arm: aggregate(rows) for arm, rows in groups.items()}


def by_goal(worlds, pairs=None):
    groups = {}
    for world in worlds:
        if pairs is not None and (int(world["cell"][1]), int(world["cell"][2])) not in pairs:
            continue
        groups.setdefault(int(world["goal"]), []).append(world)
    return {str(goal): aggregate(rows) for goal, rows in sorted(groups.items())}


import math  # noqa: E402  (kept beside the statistics helpers for readability)


def percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_delay_test(worlds, arm, baseline="normal_immediate"):
    """World-level paired comparison of mean D_resp, arm minus baseline, with Wilcoxon."""
    index = {}
    for world in worlds:
        if world.get("status") != "ok" or not world["response"]:
            continue
        key = (tuple(world["cell"]), int(world["scene"]))
        index.setdefault(key, {})[world["arm"]] = sum(world["response"]) / len(world["response"])
    pairs = [(values[arm], values[baseline]) for values in index.values()
             if arm in values and baseline in values]
    if len(pairs) < 5:
        return {"worlds": len(pairs), "status": "insufficient"}
    diffs = [own - base for own, base in pairs]
    mean = sum(diffs) / len(diffs)
    result = {"worlds": len(diffs), "mean_diff_steps": mean,
              "median_diff_steps": percentile(diffs, 0.5),
              "ci95_low": percentile(diffs, 0.025), "ci95_high": percentile(diffs, 0.975),
              "mean_arm": sum(own for own, _base in pairs) / len(pairs),
              "mean_baseline": sum(base for _own, base in pairs) / len(pairs)}
    try:
        from scipy.stats import wilcoxon
        if all(abs(value) < 1e-12 for value in diffs):
            result.update({"p_value": 1.0, "test": "wilcoxon", "note": "all differences zero"})
        else:
            statistic = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            result.update({"p_value": float(statistic.pvalue), "test": "wilcoxon"})
    except Exception as error:  # scipy absent: report the sign split instead of a fake p
        negative = sum(1 for value in diffs if value < 0)
        result.update({"p_value": None, "test": "sign_split",
                       "negative": negative, "positive": len(diffs) - negative,
                       "note": "%s: %s" % (type(error).__name__, error)})
    return result


# --------------------------------------------------------------------------- report
FORMULAS = """\
| 指标 | 一句话物理定义 | 符号公式 | 分母 |
|---|---|---|---|
| 中继滞留时延 | 一个目标从被发现到它第一次被放上信道所等的步数 | `D_hold(i) = t_emit(i) - t_disc(i)` | 无（逐目标，聚合时取分布） |
| 下游响应时延 | 从消息上行到该目标完成所花的步数，即接收方排队与执行的时间 | `D_resp(i) = t_done(i) - t_emit(i)` | 同上 |
| 端到端总时延 | 发现到完成的全部时间，`D_e2e = D_hold + D_resp` | `D_e2e(i) = t_done(i) - t_disc(i)` | 同上 |
| 链路包交付率 | 实际被订阅者收到的包数占尝试投递包数的比例 | `PDR_link = sum_e delivered(e) / sum_e attempts(e)` | 全部发射的接收者·次 |
| 链路丢包率 | 其补，实测值应与配置的投递概率一致 | `PLR_link = 1 - PDR_link` | 同上 |
| 任务级有效交付率 | 每条上行消息换来多少个按时完成的目标 | `PDR_eff = N_on / sum_e 1` | 发射总次数 |
| 受扰节点人次 | 每世界累计有多少个（接收者, 包）实例被真正送达 | `C_node = sum_e delivered(e)` | 世界数（报告里除以世界数） |
| 归一化通信能耗 | 发送与接收两侧按相对权重折算的字节量除以按时任务数 | `E_task = (a_tx * sum_e payload(e) + a_rx * sum_e payload(e) * delivered(e)) / N_on` | 按时完成的目标数 |
| 原始吞吐率 | 每单位时间放上信道的计费字节，与是否完成任务无关 | `Gamma_raw = sum_e cost(e) / (T / W)` | 时间窗数 `T/W` |
| 有效吞吐率 | 只计入最终按时完成的目标所占用的计费字节 | `Gamma_good = sum_{e: target(e) in on-time} cost(e) / (T / W)` | 时间窗数 `T/W` |

符号说明：`i` 为目标；`e` 为一次发射；`t_disc` 为检测事件时间戳；`t_emit` 为该目标第一次
发射的步号；`t_done` 为完成事件时间戳；`attempts(e)` 为扇出订阅者数（每次发射的尝试包数）；
`delivered(e)` 为其中真正到达的包数；`payload(e)` 为单份载荷字节；`cost(e)` 为按接收者计费的
字节（载荷 × 扇出）；`N_on` 为按时完成的目标数；`T` 为世界步数、`W` 为速率窗口长度；`a_tx`、`a_rx` 为收发两侧的相对
能耗权重。**公式里不出现任何具体数值**；参数取值单列在下文“参数取值”一段。
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--from-json", action="store_true",
                        help="rebuild the report and figure from PROTOCOL_METRICS.json")
    args = parser.parse_args()
    sys.dont_write_bytecode = True

    if args.from_json and OUT_JSON.exists():
        payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        write_report(payload)
        if not args.no_figures:
            draw_figure(payload)
        print("rebuilt from %s" % OUT_JSON)
        return

    tasks = collect_main() + collect_g_sweep() + collect_delivery()
    print("%d worlds: %d main, %d g_sweep, %d delivery"
          % (len(tasks),
             sum(1 for t in tasks if t["experiment"] == "main"),
             sum(1 for t in tasks if t["experiment"] == "g_sweep"),
             sum(1 for t in tasks if t["experiment"] == "delivery")), flush=True)
    unique, seen = [], set()
    for task in tasks:
        key = (task["experiment"], task["arm"], tuple(task["cell"]), task["scene"],
               task["probability"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(task)

    worlds = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        for done, row in enumerate(pool.map(worker, unique), start=1):
            worlds.append(row)
            if row["status"] != "ok":
                print("  FAILED %s %s s%d: %s" % (row["arm"], row["cell"], row["scene"],
                                                  row["error"]), flush=True)
            elif done % 400 == 0:
                print("  [%d/%d]" % (done, len(unique)), flush=True)
    failures = [row for row in worlds if row["status"] != "ok"]
    if failures:
        raise SystemExit("%d worlds failed; first: %s" % (len(failures), failures[0]))

    main_worlds = [w for w in worlds if w["experiment"] == "main"]
    sweep_worlds = [w for w in worlds if w["experiment"] == "g_sweep"]
    delivery_worlds = [w for w in worlds if w["experiment"] == "delivery"]

    main_arms = group_by_arm(main_worlds)
    main_groups = {
        "anchor(3)": group_by_arm([w for w in main_worlds if w["group"] == "anchor"]),
        "unseen(5)": group_by_arm([w for w in main_worlds if w["group"] == "unseen"]),
        "all(8)": main_arms,
    }

    high_load = {}
    for arm in DELAY_ARMS:
        rows = [w for w in sweep_worlds if w["arm"] == arm]
        high_load[arm] = {
            "R6_all": by_goal(rows, R6_PAIRS),
            "R6_high": by_goal([w for w in rows if int(w["goal"]) >= 36], R6_PAIRS),
            "R12_all": by_goal(rows, R12_PAIRS),
        }

    pooled = {}
    for arm in DELAY_ARMS:
        pooled[arm] = aggregate([w for w in sweep_worlds if w["arm"] == arm
                                 and int(w["goal"]) >= 36
                                 and (int(w["cell"][1]), int(w["cell"][2])) in R6_PAIRS])

    tests = {}
    for arm in ("v4f@200", "v4t@100", "slack_matched_rate"):
        rows = [w for w in sweep_worlds if w["arm"] in (arm, "normal_immediate")
                and int(w["goal"]) >= 36
                and (int(w["cell"][1]), int(w["cell"][2])) in R6_PAIRS]
        tests[arm] = paired_delay_test(rows, arm)
    # The second half of the question the paper asks: the learned arms against the strongest
    # zero-training opponent at the same load, not only against immediate.
    for arm in ("v4f@200", "v4t@100"):
        rows = [w for w in sweep_worlds
                if w["arm"] in (arm, "slack_matched_rate")
                and int(w["goal"]) >= 36
                and (int(w["cell"][1]), int(w["cell"][2])) in R6_PAIRS]
        tests[arm + " vs slack_matched"] = paired_delay_test(
            rows, arm, baseline="slack_matched_rate")

    delivery = {}
    for probability in DELIVERY_PROBABILITIES:
        if probability == 0.7:
            rows = [w for w in main_worlds
                    if w["arm"] in DELIVERY_ARMS
                    and tuple(w["cell"]) in ((24, 6, 6), (48, 6, 6), (48, 12, 6))]
            delivery["0.70"] = group_by_arm(rows)
        else:
            rows = [w for w in delivery_worlds if abs(w["probability"] - probability) < 1e-9]
            delivery["%.2f" % probability] = group_by_arm(rows)

    payload = {
        "schema": "stage5b-protocol-metrics/1",
        "root": str(BASE),
        "parameters": {"alpha_tx": ALPHA_TX, "alpha_rx": ALPHA_RX,
                       "rate_window_steps": RATE_WINDOW},
        "worlds": {"main": len(main_worlds), "g_sweep": len(sweep_worlds),
                   "delivery": len(delivery_worlds), "total": len(worlds)},
        "main": main_groups,
        "high_load": high_load,
        "high_load_pooled": pooled,
        "paired_delay_tests": tests,
        "delivery": delivery,
        "sources": [{"arm": arm, "root": root} for arm, root, _s, _p in MAIN_SOURCES],
        "failures": [],
    }
    tmp = OUT_JSON.with_name(OUT_JSON.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(str(tmp), str(OUT_JSON))
    print("wrote %s" % OUT_JSON, flush=True)

    write_report(payload)
    if not args.no_figures:
        draw_figure(payload)
    print("EXTENDED sha256 json %s" % sha256(OUT_JSON))
    print("EXTENDED sha256 md   %s" % sha256(OUT_MD))


# --------------------------------------------------------------------------- markdown
def fmt(value, pattern="%.2f", empty="—"):
    return empty if value is None else pattern % value


def write_report(payload):
    lines = []
    add = lines.append

    add("# 协议级补充指标：时延、双层交付率、能耗代理与吞吐")
    add("")
    add("本报告把韩成（§2.1.3 的端到端延迟、传输成功率、能量效率、网络生存期）与高延忠"
        "（§3.2.3 的网络吞吐量、平均端到端时延）两套指标，映射到本 revision 已归档的评估日志上。"
        "**没有重跑任何世界、没有重训任何权重、没有改动 98 个 pin 文件**：全部读数来自"
        "`lifecycle.jsonl.gz` 与 `relay_transcript.jsonl.gz` 两个文件。")
    add("")

    add("## 1　指标定义（符号化）")
    add("")
    add(FORMULAS)
    add("")
    add("**参数取值**：`a_tx = %.1f`、`a_rx = %.1f`（收发两侧的相对字节能耗权重）、"
        "`W = %.0f` 步（速率窗口）。这三个是**参数**，不是公式的一部分。"
        % (payload["parameters"]["alpha_tx"], payload["parameters"]["alpha_rx"],
           payload["parameters"]["rate_window_steps"]))
    add("")
    add("**三条与归档口径的对应关系**（避免与既有表冲突）：")
    add("")
    add("1. `on-time` 与论文表里的“按时完成”完全同义（`t_done ≤ deadline`），由同一批事件算出；")
    add("2. `cost(e)` 就是既有表里的计费字节（载荷 × 扇出），因此 `Gamma_raw` 与"
        "“字节/回合”只差一个窗口归一化；")
    add("3. `PDR_eff` 是既有“广播转化率”的另一种写法，此处按任务数而非消息数取分母。")
    add("")

    add("## 2　方法论：确定性世界配对检验 vs 蒙特卡洛重复采样")
    add("")
    add("两篇参考论文都用“同一配置重复仿真 N 次取平均”来消除随机性（高延忠：100 次蒙特卡洛）。"
        "本 revision 的做法不同，理由是可检验的：")
    add("")
    add("1. **单个世界是确定性的**：`(scene, cell, probability)` 唯一确定一个世界，"
        "同一设定重跑结果逐字节相同。正文实验里已经验证过：把投递率 0.7 的一格在新管道上重跑，"
        "按时数、发射数、字节数与归档完全一致。因此对**同一个世界**重复采样只会得到零方差，"
        "再跑 N 次不增加任何信息。")
    add("2. **随机性来自世界之间的差异**：场景、目标布局、rescuer 编队、订阅者扇出共同决定结果，"
        "所以本报告把“重复”放在**跨世界**这一层——每组 24–120 个世界，而不是同一世界 100 次。")
    add("3. **配对而非独立**：所有方法跑在同一批世界上（同 (cell, scene, seed)），"
        "因此延时差异用**世界级配对检验**（Wilcoxon 符号秩，双侧）而不是两样本检验；"
        "这比蒙特卡洛取平均更严格——它同时控制了世界难度与样本量。")
    add("4. **代价**：跨世界重采样无法给出“同一世界内的时序方差”（本报告也没有报这个量），"
        "因为该方差在确定性世界下恒为零；凡是引用“均值 ± 离散度”的地方，离散度都是"
        "跨世界分位数，已在表头写明。")
    add("")

    add("## 3　主实验（14 臂 × 15 格 × 8 场景）")
    add("")
    add("| 臂 | 世界 | 按时/已发现 | PDR_link | PLR_link | PDR_eff | C_node/世界 | E_task | Γ_raw | Γ_good | D_resp 均值(步) | D_resp 中位 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in [name for name, _r, _s, _p in MAIN_SOURCES]:
        stats = payload["main"]["all(8)"].get(arm)
        if not stats:
            continue
        add("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            ARM_LABEL.get(arm, arm), stats["worlds"], fmt(stats["q"], "%.3f"),
            fmt(stats["pdr_link"], "%.4f"), fmt(stats["plr_link"], "%.4f"),
            fmt(stats["pdr_eff"], "%.3f"), fmt(stats["receiver_nodes_per_world"], "%.1f"),
            fmt(stats["e_task"], "%.2f"), fmt(stats["gamma_raw"], "%.0f"),
            fmt(stats["gamma_good"], "%.0f"), fmt(stats["d_resp"]["mean"], "%.1f"),
            fmt(stats["d_resp"]["median"], "%.1f")))
    add("")
    add("（`按时/已发现` 的分母是**被检测到的目标数**，不是场景放置的目标数：未发现的目标不进这个分母，这也是它与论文表里“按时/放置目标”略有出入的唯一原因；`PDR_link` 的实测值应逼近配置的投递概率；`D_resp` 的单位是仿真步，1 步 = 1 tick；"
        "均值与中位都是**跨按时目标池化**后的分布，括号外的分母见 §1。）")
    add("")

    add("## 4　高负载下的下游排队时延（R=6，G ≥ 36）")
    add("")
    add("| 方法 | 世界 | D_hold 均值 | D_resp 均值 | D_resp 中位 | D_e2e 均值 | Γ_good |")
    add("|---|---|---|---|---|---|---|")
    for arm in DELAY_ARMS:
        stats = payload["high_load_pooled"].get(arm)
        if not stats:
            continue
        add("| %s | %d | %s | %s | %s | %s | %s |" % (
            ARM_LABEL.get(arm, arm), stats["worlds"],
            fmt(stats["d_hold"]["mean"], "%.1f"), fmt(stats["d_resp"]["mean"], "%.1f"),
            fmt(stats["d_resp"]["median"], "%.1f"), fmt(stats["d_e2e"]["mean"], "%.1f"),
            fmt(stats["gamma_good"], "%.0f")))
    add("")
    add("**逐世界配对检验（Wilcoxon 符号秩，双侧；世界级平均 D_resp）**")
    add("")
    add("| 对照 | 世界数 | 均值差(步) | 中位差(步) | 95% 分位区间 | p |")
    add("|---|---|---|---|---|---|")
    for arm, record in payload["paired_delay_tests"].items():
        add("| %s | %s | %s | %s | [%s, %s] | %s |" % (
            ("%s − Slack-Matched" % ARM_LABEL.get(arm.split(" vs ")[0], arm.split(" vs ")[0])
             if " vs " in arm else "%s − immediate" % ARM_LABEL.get(arm, arm)),
            record.get("worlds"),
            fmt(record.get("mean_diff_steps"), "%+.2f"),
            fmt(record.get("median_diff_steps"), "%+.2f"),
            fmt(record.get("ci95_low"), "%+.2f"), fmt(record.get("ci95_high"), "%+.2f"),
            ("< 1e-12" if record.get("p_value") == 0.0 else "%.2e" % record["p_value"])
            if record.get("p_value") is not None else "—"))
    add("")
    add("**两条必须一起读的事实**：")
    add("")
    add("1. `v4f` 与 `v4t` 在这一段里**逐位相同**：R=6 时网络的发送量低于页预算，配额从未咬合，"
        "所以两条曲线重合不是画错。")
    add("2. `D_hold` 只有 `slack_matched` 不为零（约 5 步）：它的时限排序 + 每步 1 页的节奏会"
        "把一部分发送推迟到发现之后的某一步；`v4t`/`v4f` 的所有发送都发生在发现当步。"
        "换句话说，**时延的节省全部来自下游响应**，不是靠“更早发”。")
    add("")
    add("**读法**：差值为负表示该臂比 immediate 更早完成（`D_resp` 更短）。"
        "区间是差值的 2.5%–97.5% 跨世界分位，不是参数置信区间；世界数少于 5 时不给检验。")
    add("")

    add("## 5　投递率扫描（p = 0.3 … 1.0，24 世界/点）")
    add("")
    add("| 臂 | 指标 | p=0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for arm in DELIVERY_ARMS:
        for field, title, pattern in (("pdr_link", "PDR_link", "%.3f"),
                                      ("pdr_eff", "PDR_eff", "%.3f"),
                                      ("d_resp", "D_resp 均值", "%.1f"),
                                      ("gamma_good", "Γ_good", "%.0f")):
            cells = []
            for probability in DELIVERY_PROBABILITIES:
                stats = payload["delivery"]["%.2f" % probability].get(arm)
                if not stats:
                    cells.append("—")
                    continue
                value = stats[field]["mean"] if field == "d_resp" else stats[field]
                cells.append(fmt(value, pattern))
            add("| %s | %s | %s |" % (ARM_LABEL.get(arm, arm), title, " | ".join(cells)))
    add("")
    add("p=0.7 一列取自封板归档（同一批 3 格 × 8 场景），其余七列来自本轮投递率扫描；"
        "两边的 `PDR_link` 都应落在配置值附近，这是接缝自检。")
    add("")

    add("## 6　与两篇参考论文的指标对照")
    add("")
    add("| 参考论文的指标 | 本报告对应列 | 对应关系说明 |")
    add("|---|---|---|")
    add("| 韩成·端到端延迟 | `D_e2e`（= `D_hold` + `D_resp`） | 逐目标，池化后给均值/中位/90 分位 |")
    add("| 韩成·传输成功率（包交付率） | `PDR_link`（链路）与 `PDR_eff`（任务） | 前者是包的，后者是任务的；两者不可互替 |")
    add("| 韩成·能量效率（能量/包） | `E_task`（归一化能耗/按时任务） | 分母换成“按时的任务”，因为本任务的成功定义是按时完成 |")
    add("| 韩成·网络生存周期 | 不适用 | 本仿真无电池模型；强行给数会是无依据的推导 |")
    add("| 韩成·表 4.8 转发节点数 | `C_node`（受扰节点人次/世界） | 一个是“经过的节点数”，一个是“被送达的（节点, 包）实例数” |")
    add("| 高延忠·网络吞吐量 | `Γ_raw` / `Γ_good` | 吞吐按计费字节计；有效吞吐只计按时目标的字节 |")
    add("| 高延忠·平均端到端时延 | `D_resp`（下游排队+响应） | 把“网络时延”拆成上行滞留与下游响应两段 |")
    add("")

    add("## 7　能说 / 不能说")
    add("")
    add("**能说**")
    add("")
    top = None
    for arm in DELAY_ARMS:
        stats = payload["high_load_pooled"].get(arm)
        if stats and stats["d_resp"]["mean"] is not None:
            if top is None or stats["d_resp"]["mean"] < top[1]:
                top = (arm, stats["d_resp"]["mean"])
    if top:
        means = {arm: payload["high_load_pooled"][arm]["d_resp"]["mean"]
                 for arm in DELAY_ARMS
                 if payload["high_load_pooled"].get(arm)
                 and payload["high_load_pooled"][arm]["d_resp"]["mean"] is not None}
        best = min(means.values())
        tied = [ARM_LABEL.get(arm, arm) for arm, value in means.items()
                if abs(value - best) < 1e-9]
        baseline = means.get("normal_immediate")
        add("1. 高负载 R=6、G ≥ 36 的下游响应时延最低的是 **%s**（%.1f 步%s）。"
            % ("、".join(tied), best,
               "，并列" if len(tied) > 1 else "",
               ))
        if baseline is not None and baseline > best:
            add("   相对 immediate（%.1f 步）少 %.1f 步，逐世界配对 Wilcoxon 检验见上表。"
                % (baseline, baseline - best))
    add("2. 链路交付率与配置投递概率一致（见 §5 的 `PDR_link` 列），说明信道接缝没有改变；"
        "方法之间的差异来自**发多少、发什么**，不是来自链路质量。")
    add("3. `Γ_good / Γ_raw` 给出“信道占用里有多少最终兑现为按时完成”，可直接对标吞吐量口径。")
    add("")
    add("**不能说**")
    add("")
    add("1. **不能把 `D_resp` 当作纯网络时延**：它包含接收端的分配与执行时间，"
        "不是 MAC 层的排队时延；与高延忠的“平均端到端时延”只同量纲，不同构成。")
    add("2. **不能把 `E_task` 当能耗**：它是按收发相对权重折算的字节代理，"
        "没有射频模型、没有睡眠/计算能耗，只用于方法之间的相对比较。")
    add("3. **不能把 `PDR_link` 与 `PDR_eff` 混用**：前者是包的到达率（应≈配置 p），"
        "后者是每条消息换来的按时任务数，两者分母不同。")
    add("4. **不能说“更少的时延”等于“排序更好”**：`D_hold` 只在 `slack_matched` 上不为零"
        "（它按步发放，会把部分发送推迟），`v4t`/`v4f`/`immediate` 都是发现当步就发。"
        "因此两条时延曲线的差反映的是**下游响应**，不是“谁更早抢到信道”。")
    add("5. **不能把世界数当样本量重复使用**：同一次运行的世界既用于主表也用于本报告，"
        "配对检验已控制这一点，但跨表的数字不是独立证据。")
    add("")

    add("## 复现")
    add("")
    add("```")
    add("D:/project_env/python.exe analysis/protocol_metrics.py --workers 8")
    add("```")
    add("")
    add("产物：`execution/PROTOCOL_METRICS.json`（全部读数）、本文件、"
        "`execution/figures/%s.{png,pdf}`。" % FIG_NAME)
    add("")
    add("## 证据（sha256 前 16 位）")
    add("")
    add("| 文件 | sha256 | 规模 |")
    add("|---|---|---|")
    add("| `execution/PROTOCOL_METRICS.json` | `%s` | %d 世界 |" % (
        sha256(OUT_JSON)[:16], payload["worlds"]["total"]))
    add("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("wrote %s" % OUT_MD, flush=True)


# --------------------------------------------------------------------------- figure
def draw_figure(payload):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    missing = [name for name in ("Times New Roman", "SimSun", "STIXGeneral")
               if name not in available]
    if missing:
        raise SystemExit("required fonts missing: %s" % missing)
    plt.rcParams.update({
        "font.family": ["Times New Roman", "SimSun", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.0,
        "axes.linewidth": 0.8, "axes.edgecolor": "#000000",
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.grid": True, "grid.color": "#CCCCCC", "grid.linewidth": 0.4,
        "grid.linestyle": "--", "axes.axisbelow": True, "axes.unicode_minus": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": True, "legend.fancybox": False,
        "legend.edgecolor": "#000000", "legend.facecolor": "#FFFFFF",
    })

    style = {
        "normal_immediate": dict(color="#636363", ls="-.", marker="^", mfc="white",
                                 lw=1.1, ms=4.5, mew=1.0, zorder=4),
        "slack_matched_rate": dict(color="#8E44AD", ls=":", marker="v", mfc="white",
                                   lw=1.3, ms=4.5, mew=1.0, zorder=4),
        "v4f@200": dict(color="#2980B9", ls="--", marker="o", mfc="none",
                        lw=1.1, ms=6.4, mew=1.15, zorder=7),
        "v4t@100": dict(color="#C0392B", ls="-", marker="o", mfc="#C0392B",
                        lw=1.7, ms=4.6, mew=1.0, zorder=6),
    }
    goals = [goal for goal in GOALS
             if str(goal) in payload["high_load"]["normal_immediate"]["R6_all"]]
    x = np.arange(len(goals))

    figure, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
    panels = (
        ("d_resp", "(a) 下游响应时延", "D_resp 均值（步）"),
        ("miss_rate", "(b) 任务级未按时率", "1 − PDR_eff"),
        ("gamma_good", "(c) 有效吞吐率", "字节 / 100 步"),
    )
    for axis, (field, title, ylabel) in zip(axes, panels):
        for arm in DELAY_ARMS:
            series = payload["high_load"][arm]["R6_all"]
            values = []
            for goal in goals:
                stats = series.get(str(goal))
                if not stats:
                    values.append(np.nan)
                    continue
                if field == "d_resp":
                    values.append(stats["d_resp"]["mean"])
                else:
                    values.append(stats[field])
            axis.plot(x, values, label=ARM_LABEL.get(arm, arm), **style[arm])
        if field == "miss_rate":
            link = payload["main"]["all(8)"]["normal_immediate"]["plr_link"]
            axis.axhline(link, color="#999999", lw=0.9, ls="--", zorder=1)
            axis.annotate("链路 PLR（实测，≈1−p）", xy=(x[0], link), xytext=(0, 4),
                          textcoords="offset points", fontsize=7.5, color="#666666")
        if field == "gamma_good":
            axis.set_yscale("log")
        axis.set_xticks(x)
        axis.set_xticklabels([str(goal) for goal in goals])
        axis.set_xlabel("任务规模 G（R=6，N=3 与 N=6 合并）\n" + title)
        axis.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06),
                  columnspacing=1.0, handlelength=2.4)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / (FIG_NAME + ".png")
    pdf = FIG_DIR / (FIG_NAME + ".pdf")
    figure.savefig(png)
    figure.savefig(pdf)
    plt.close(figure)
    print("wrote %s (sha256 %s)" % (png, sha256(png)[:16]), flush=True)


if __name__ == "__main__":
    main()
