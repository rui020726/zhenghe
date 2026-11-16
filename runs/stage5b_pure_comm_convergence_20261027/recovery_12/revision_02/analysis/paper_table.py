"""The paper's main comparison table: learned arms vs no-training baselines, same cells, same scenes.

Sources, all read-only:

* ``execution/baselines/grid_v2/BASELINES.json`` -- ``normal_immediate``, ``dedup_immediate``,
  ``always_hold`` on all 8 DEV scenes.
* ``execution/baselines/paper_throttles_v1/BASELINES.json`` -- the throttles that wrap
  ``immediate`` and need no network (``fifo_cap_k1``, ``slack_heuristic_cap``), same 8 scenes.
* ``execution/baselines/paper_random_matched/BASELINES.json`` /
  ``.../paper_slack_matched/BASELINES.json`` -- the two arms matched to the learned arm's own
  page depth.
* ``execution/baselines/paper_traditional_v1/BASELINES.json`` -- the two traditional triggers
  from the frozen catalogue (``periodic_k5`` time-triggered, ``event_maxwait`` event-triggered).
* ``execution/<tag>/eval_<ep>_s8/EVAL.json`` -- the learned arms on the same 8 scenes.
* ``execution/COMM_EFFICIENCY.json`` -- the four efficiency readings, re-derived from the
  archived lifecycle and relay transcripts of exactly those worlds (see ``analysis/comm_efficiency.py``;
  the extraction asserts its ``emissions`` and ``on_time`` against the same archived rows used here).

Every row is reported under three column groups: the three anchor scenes, the five unseen scenes,
and the eight-scene union. Within a group the per-cell verdict is the registered rule applied to
that group's own denominators: on-time at least 95 % of ``normal_immediate`` **in the same cell and
the same scenes**, and strictly fewer bytes. A cell is never compared to another cell.

    D:/project_env/python.exe analysis/paper_table.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ANCHOR = (2301901, 2301902, 2301903)
UNSEEN = (2301904, 2301905, 2301906, 2301907, 2301908)
GROUPS = (("anchor(3)", ANCHOR), ("unseen(5)", UNSEEN), ("all(8)", ANCHOR + UNSEEN))

LEARNED = (
    ("v4f@200 (network)", "train_v4f_anchor", 200, "v4f@200"),
    ("v4g@300 (price)", "train_v4g_3scene_anchor", 300, "v4g@300"),
    ("v4s@300 (cap/drift)", "train_v4s_cap300", 300, "v4s@300"),
    ("v4t@100 (cap+frozen)", "train_v4t_cap_frozen", 100, "v4t@100"),
    ("v4u@100 (cap + flat floor 2)", "train_v4u2_window_fixed", 100, "v4u@100"),
)
RULES = ("normal_immediate", "dedup_immediate", "always_hold")
THROTTLES = ("fifo_cap_k1", "random_matched_rate", "slack_heuristic_cap",
             "slack_matched_rate")
TRADITIONAL = ("periodic_k5", "event_maxwait")
CLASSIC = ("red_aqm", "token_bucket", "ahp_utility", "linucb_bandit")
#: Order in the printed table, and the efficiency key each label reads.
#: The paper's table: thirteen non-learning baselines (five of them named classical algorithms)
#: plus the one learned model. The other learned checkpoints (v4g/v4s/v4t/v4u) stay in the audit
#: record but are not in this table.
ARM_ORDER = (
    ("normal_immediate", "normal_immediate", "Flooding (全量广播)"),
    ("dedup_immediate", "dedup_immediate", "Dedup-Flooding (仅去重)"),
    ("always_hold", "always_hold", "Always-Hold (从不发送)"),
    ("periodic_k5", "periodic_k5", "TT-Periodic (定时触发)"),
    ("event_maxwait", "event_maxwait", "ET-MaxWait (事件+超时)"),
    ("fifo_cap_k1", "fifo_cap_k1", "FCFS-DropTail (fifo_cap)"),
    ("slack_heuristic_cap", "slack_heuristic_cap", "EDF-Slack (slack_cap)"),
    ("random_matched_rate", "random_matched_rate", "Random-Drop (同字节随机)"),
    ("slack_matched_rate", "slack_matched_rate", "EDF-Matched (slack_matched)"),
    ("red_aqm", "red_aqm", "RED-AQM"),
    ("token_bucket", "token_bucket", "Token-Bucket"),
    ("ahp_utility", "ahp_utility", "AHP-Utility"),
    ("linucb_bandit", "linucb_bandit", "LinUCB-Bandit"),
    ("v4f@200", "v4f@200", "**本文方法 (Ours)**"),
)
#: Short names used as per-cell column headers in table B.
LABELS = {key: key for key, _eff, _label in ARM_ORDER}


def cell_label(cell):
    goals, robots, subscribers = (int(value) for value in cell)
    return "G%d_R%d_N%d" % (goals, robots, subscribers)


def load_grid():
    payload = json.loads((BASE / "execution/baselines/grid_v2/BASELINES.json")
                         .read_text(encoding="utf-8"))
    table = {}
    for row in payload.get("results") or []:
        if row.get("status") != "ok":
            continue
        key = (row["arm"], tuple(row["cell"]), int(row["scene"]))
        table[key] = (int(row["on_time"]), int(row.get("generated_goals") or 0),
                      int(row["spent_bytes"]), int(row["emissions"]))
    return table


def load_throttles():
    """All wrapping arms, later files winning.

    ``paper_throttles_v1`` holds the first pass of the random arm, which was measured to spend
    exactly ``immediate``'s bytes (1245/1245 pages) because the coin was re-drawn per decision
    while the relay re-offers unsent targets. ``paper_random_matched`` is the corrected run --
    one flip per target per episode, keep rate read off v4t -- and supersedes that arm only.
    """
    payload = {}
    for name in ("paper_throttles_v1", "paper_random_matched", "paper_slack_matched",
                 "paper_traditional_v1"):
        path = BASE / "execution/baselines" / name / "BASELINES.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")).get("results") or []:
            if row.get("status") != "ok":
                continue
            payload[(row["arm"], tuple(row["cell"]), int(row["scene"]))] = row
    table = {}
    for key, row in payload.items():
        table[key] = (int(row["on_time"]), int(row.get("generated_goals") or 0),
                      int(row["spent_bytes"]), int(row["emissions"]))
    return table, {key[0] for key in payload}


def load_learned(tag, episode):
    path = BASE / "execution" / tag / ("eval_%d_s8" % episode) / "EVAL.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = {}
    for row in payload.get("results") or []:
        if row.get("status") != "ok":
            continue
        key = (tuple(row["cell"]), int(row["scene"]))
        table[key] = (int(row["on_time"]), int(row.get("generated_goals") or 0),
                      int(row["spent_bytes"]), int(row["emissions"]))
    return table


def load_classic(arm):
    """The four named classical arms on the same 15 cells x 8 scenes."""
    path = BASE / "execution/baselines/classic_v1/BASELINES.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = {}
    for row in payload.get("results") or []:
        if row.get("status") != "ok" or row.get("arm") != arm:
            continue
        key = (tuple(row["cell"]), int(row["scene"]))
        table[key] = (int(row["on_time"]), int(row.get("generated_goals") or 0),
                      int(row["spent_bytes"]), int(row["emissions"]))
    return table


def load_efficiency():
    path = BASE / "execution/COMM_EFFICIENCY.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("per_arm") or {})


def summarise(getter, scenes, cells):
    """Aggregate one scene group under the registered per-cell rule."""
    on = goals = by = ref_on = ref_bytes = 0
    passes = 0
    for cell in cells:
        cell_on = cell_goals = cell_by = ref_cell_on = ref_cell_by = 0
        for scene in scenes:
            value = getter(cell, scene)
            ref = REF[(cell, scene)]
            if value is None or ref is None:
                return None
            cell_on += value[0]
            cell_goals += value[1]
            cell_by += value[2]
            ref_cell_on += ref[0]
            ref_cell_by += ref[2]
        passes += int(cell_on >= math.ceil(0.95 * ref_cell_on) and cell_by < ref_cell_by)
        on += cell_on
        goals += cell_goals
        by += cell_by
        ref_on += ref_cell_on
        ref_bytes += ref_cell_by
    return {"on_time": on, "goals": goals, "q": on / goals if goals else 0.0,
            "ratio": on / ref_on if ref_on else 0.0,
            "bytes": by, "byte_ratio": by / ref_bytes if ref_bytes else 0.0,
            "passes": passes, "cells": len(cells)}


REF = {}


def fmt(value, pattern):
    return "n/a" if value is None else pattern % value


def main():
    grid = load_grid()
    throttle, throttle_arms = load_throttles()
    efficiency = load_efficiency()
    for (arm, cell, scene), value in grid.items():
        if arm == "normal_immediate":
            REF[(cell, scene)] = value
    cells = sorted({cell for cell, _scene in REF})
    print("cells: %d   scenes: %d anchor + %d unseen"
          % (len(cells), len(ANCHOR), len(UNSEEN)))

    learned_tables = {}
    for label, tag, episode, key in LEARNED:
        learned_tables[key] = load_learned(tag, episode)

    def getter_for(key):
        if key in RULES:
            return lambda cell, scene, k=key: grid.get((k, cell, scene))
        if key in throttle_arms:
            return lambda cell, scene, k=key: throttle.get((k, cell, scene))
        if key in CLASSIC:
            table = load_classic(key)
            return None if table is None else (lambda cell, scene, t=table: t.get((cell, scene)))
        table = learned_tables.get(key)
        return None if table is None else (lambda cell, scene, t=table: t.get((cell, scene)))

    missing = []
    print()
    print("| arm | scenes | on-time | vs immediate | bytes | byte ratio | passes "
          "| channel uses/world | B per on-time task | msg per on-time task "
          "| on-time per msg | broadcast waste |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, _eff_key, label in ARM_ORDER:
        getter = getter_for(key)
        if getter is None:
            missing.append(label)
            print("| %s | all | (missing) | | | | | | | | | |" % label)
            continue
        for group_name, scenes in GROUPS:
            stats = summarise(getter, scenes, cells)
            if stats is None:
                print("| %s | %s | (incomplete) | | | | | | | | | |"
                      % (label, group_name))
                continue
            efficiency_key = _eff_key
            group = ((efficiency.get(efficiency_key) or {}).get("groups") or {}).get(
                group_name) or {}
            needed = ("channel_uses_per_world", "bytes_per_on_time", "messages_per_on_time",
                      "conversion", "broadcast_waste_ratio")
            if not all(name in group for name in needed):
                print("| %s | %s | %d/%d = %.4f | %.3f | %d | %.3f | %d/%d "
                      "| (no efficiency reading) | | | | |"
                      % (label, group_name, stats["on_time"], stats["goals"], stats["q"],
                         stats["ratio"], stats["bytes"], stats["byte_ratio"],
                         stats["passes"], stats["cells"]))
                continue
            print("| %s | %s | %d/%d = %.4f | %.3f | %d | %.3f | %d/%d | %s | %s "
                  "| %s | %s | %s |"
                  % (label, group_name, stats["on_time"], stats["goals"], stats["q"],
                     stats["ratio"], stats["bytes"], stats["byte_ratio"],
                     stats["passes"], stats["cells"],
                     fmt(group["channel_uses_per_world"], "%.1f"),
                     fmt(group["bytes_per_on_time"], "%.1f"),
                     fmt(group["messages_per_on_time"], "%.4f"),
                     fmt(group["conversion"], "%.4f"),
                     fmt(group["broadcast_waste_ratio"], "%.4f")))
    print()
    print("message columns: msg per on-time task = emissions / on-time task, and on-time per msg")
    print("is its reciprocal (the broadcast conversion rate). channel uses = emissions x N,")
    print("summed over the group's worlds and divided by them. broadcast waste = the share of")
    print("distinct broadcast targets that did not finish inside their own deadline.")
    if missing:
        print()
        print("MISSING: %s" % ", ".join(missing))

    detail_arms = ("normal_immediate", "event_maxwait", "slack_heuristic_cap",
                   "slack_matched_rate", "v4f@200")
    print()
    print()
    print("## table B  per-cell detail, three anchor scenes: sends per world, its ratio to")
    print("## normal_immediate, on-time, messages per on-time task, broadcast waste")
    print()
    print("| cell | immediate sends | immediate on-time | event_maxwait sends (x imm) "
          "| event_maxwait on-time | slack_heuristic_cap sends (x imm) | slack_heuristic_cap on-time "
          "| slack_matched_rate sends (x imm) | slack_matched_rate on-time "
          "| Ours sends (x imm) | Ours on-time | Ours msg/task | Ours waste |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cell in cells:
        line = "| %s |" % cell_label(cell)
        base_sends = None
        for key in detail_arms:
            getter = getter_for(key)
            stats = summarise(getter, ANCHOR, [cell]) if getter else None
            per_cell = ((efficiency.get(key) or {}).get("cells_anchor") or {}).get(
                cell_label(cell))
            if stats is None or not per_cell:
                line += " n/a |" + (" n/a | n/a |" if key == "v4f@200" else " n/a |")
                continue
            sends = per_cell["emissions"] / per_cell["worlds"]
            if key == "normal_immediate":
                base_sends = sends
                line += " %.1f | %d/%d |" % (sends, stats["on_time"], stats["goals"])
                continue
            ratio = "%.2fx" % (sends / base_sends) if base_sends else "n/a"
            line += " %.1f (%s) | %d/%d |" % (sends, ratio, stats["on_time"], stats["goals"])
            if key == "v4f@200":
                line += " %.3f | %.3f |" % (per_cell["messages_per_on_time"],
                                            per_cell["broadcast_waste_ratio"])
        print(line)
    print()
    print("sends = emissions per world over the three anchor scenes; the bracket is the ratio to")
    print("normal_immediate in the same cell and the same scenes (below 1.00x is fewer sends).")
    print("msg/task and waste are the table-1 readings restricted to this cell.")


if __name__ == "__main__":
    sys.exit(main())
