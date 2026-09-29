"""ROUND29 §5：用三个冻结来源量化发送端期限编码的信息损失。

来源（分来源报告，不混成一个总体样本）：
  A. ROUND28 的 96 个首报干预点（训练场景 1300000..1300015，baseline 单元）；
  B. ROUND27 的 target-0 / target-5 干预点（固定场景 1400031，L 路径基线）；
  C. ROUND26 四个改善场景（1400010/1400014/1400017/1400020）已保存的首报发送记录。

每个样本只用可追溯的动作前发送端信息：
  - 编码后特征：来自各轮 tape/干预点存档（``remaining_time_over_horizon`` 是第 12 列，0-based index 11）；
  - 原始期限：来自 registry probe 日志的 detection 记录（``timestamp + deadline_offset``），
    这是发送端在决策时实际读取的 registry 快照字段（``snapshot_of=registry.get``），
    动作前可用；决策时刻时钟规则 ``now = tape step - 1`` 已用 scene 1400010 全量 197 行
    逐行复现编码值验证（0 失配）。

禁止事项：不从饱和值反推期限、不引入接收端真值或终态、不拟合任何分类器/阈值/新策略。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

URGENCY_SCALE_STEPS = 50.0          # integrated_sar/comm_learning.py:70
EPISODE_HORIZON_STEPS = 1000.0      # integrated_sar/comm_learning.py:71
DEADLINE_COLUMN = 11                # FEATURE_NAMES index of remaining_time_over_horizon
FEATURE_NAMES: tuple[str, ...] = (
    "age_steps",
    "since_last_emission_time",
    "sent_count",
    "pending_event",
    "changed_view",
    "first_emission",
    "stage_changed",
    "requirements_changed",
    "members_changed",
    "progress_delta_abs",
    "revision_delta",
    "remaining_time_over_horizon",
    "urgency_now",
    "has_future_requirements",
    "queue_backlog_targets",
    "budget_remaining_fraction",
    "capacity_remaining_fraction",
    "subscribers",
    "step_over_horizon",
)

R28_DIR = ROOT / "runs/comm_cost_firstreport_diagnosis_20260929"
R27_DIR = ROOT / "runs/comm_cost_firstreport_intervention_20260929"
R26_DIR = ROOT / "runs/round26_analysis_20260929"
R25_DIR = ROOT / "runs/comm_cost_greedy_replication_20260928"


# ------------------------------------------------------------------ io helpers
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def deadlines_of(rp_path: Path) -> dict[str, float]:
    """目标 -> 绝对期限（步）。取该目标第一条 detection 记录的 timestamp + deadline_offset。

    该量与 relay 决策时从 registry 快照读到的 ``SendItem.deadline`` 同源
    （comm.py:819-820: ``deadline=float(state.deadline)``，
    ``_message_from_event``: ``deadline=detected_at + deadline_offset``），是发送端合法可见量。
    """
    out: dict[str, float] = {}
    for row in read_jsonl(rp_path):
        if row.get("event_type") != "detection":
            continue
        target = row.get("target_id")
        detail = row.get("detail") or {}
        if target in out or "deadline_offset" not in detail:
            continue
        out[target] = float(row["timestamp"]) + float(detail["deadline_offset"])
    return out


def old_encode(remaining: float) -> float:
    """生产编码（comm_learning.py:113）：clip(remaining / 50, -1, 1)。"""
    return max(-1.0, min(1.0, remaining / URGENCY_SCALE_STEPS))


def new_encode(remaining: float) -> float:
    """候选单列编码：remaining / configured_episode_horizon，无追加截断。"""
    return remaining / EPISODE_HORIZON_STEPS


def distinct(values: list[float]) -> tuple[int, list[float]]:
    seen = sorted(set(float(v) for v in values))
    return len(seen), seen


# ------------------------------------------------------------------ source A
def source_a() -> list[dict[str, Any]]:
    """ROUND28 的全部 96 个首报干预点：存档的编码特征 + 从 baseline rp 日志重建的原始期限。"""
    points: list[dict[str, Any]] = []
    doc = json.loads((R28_DIR / "INTERVENTION_POINTS.json").read_text(encoding="utf-8"))
    cache: dict[int, dict[str, float]] = {}
    for scene_entry in doc["scenes"]:
        scene = int(scene_entry["scene"])
        if scene not in cache:
            rp = R28_DIR / "logs" / f"r28_baseline_s{scene}_a1_rp.jsonl"
            cache[scene] = deadlines_of(rp)
        deadlines = cache[scene]
        for point in scene_entry["intervention_points"]:
            target = point["target"]
            step = int(point["step"])
            deadline = deadlines.get(target)
            raw = None if deadline is None else deadline - (step - 1)
            points.append({
                "source": "ROUND28-96",
                "scene": scene,
                "target": target,
                "step": step,
                "features": [float(v) for v in point["features"]],
                "raw_remaining": raw,
                "original_action": point["original_action"],
                "send_probability": point["send_probability"],
                "deadline": deadline,
            })
    return points


# ------------------------------------------------------------------ source B
def source_b() -> list[dict[str, Any]]:
    """ROUND27 的 target-0、target-5 干预点：步骤取自 ROUND27 判定的 override_steps，
    特征取自 L 路径（无拦截器基线）tape 同行。"""
    verdict = json.loads((R27_DIR / "ROUND27_VERDICT.json").read_text(encoding="utf-8"))
    tape = {}
    for row in read_jsonl(R27_DIR / "logs/r27_L_s1400031_a1_tape.jsonl"):
        tape[int(row["step"])] = row
    deadlines = deadlines_of(R27_DIR / "logs/r27_L_s1400031_a1_rp.jsonl")
    wanted = {
        "piloc-e1-target-0": int(verdict["interventions"]["E0"]["piloc-e1-target-0"]["override_steps"][0]),
        "piloc-e1-target-5": int(verdict["interventions"]["E5"]["piloc-e1-target-5"]["override_steps"][0]),
    }
    points = []
    for target, step in sorted(wanted.items()):
        row = tape.get(step)
        if row is None or target not in row["targets"]:
            continue
        index = row["targets"].index(target)
        points.append({
            "source": "ROUND27-targets",
            "scene": 1400031,
            "target": target,
            "step": step,
            "features": [float(v) for v in row["features"][index]],
            "raw_remaining": deadlines[target] - (step - 1),
            "original_action": "SEND" if int(row["actions"][target]) == 1 else "HOLD",
            "send_probability": float(row["send_probabilities"][target]),
            "deadline": deadlines[target],
        })
    return points


# ------------------------------------------------------------------ source C
def source_c() -> list[dict[str, Any]]:
    """ROUND26 四个改善场景已保存的首报发送记录（learned 臂 first=true），
    特征行取自 ROUND26 当时所读的同一 ROUND25 learned-greedy tape。"""
    improved = json.loads((R26_DIR / "IMPROVED_SCENES.json").read_text(encoding="utf-8"))
    points = []
    for scene, entry in sorted(improved["scenes"].items()):
        scene = int(scene)
        tape = {}
        for row in read_jsonl(R25_DIR / f"logs/r25_learned_s33_greedy_s{scene}_a1_tape.jsonl"):
            tape[int(row["step"])] = row
        deadlines = deadlines_of(R25_DIR / f"logs/r25_learned_s33_greedy_s{scene}_a1_rp.jsonl")
        for target, record in entry.items():
            if not target.startswith("piloc-"):
                continue
            learned = record.get("learned_s33_greedy") or {}
            firsts = [s for s in (learned.get("messages") or {}).get("sends", []) if s.get("first")]
            if not firsts:
                continue
            step = int(firsts[0]["step"])
            row = tape.get(step)
            if row is None or target not in row["targets"]:
                continue
            index = row["targets"].index(target)
            points.append({
                "source": "ROUND26-improved-4",
                "scene": scene,
                "target": target,
                "step": step,
                "features": [float(v) for v in row["features"][index]],
                "raw_remaining": deadlines[target] - (step - 1),
                "original_action": "SEND" if int(row["actions"][target]) == 1 else "HOLD",
                "send_probability": float(row["send_probabilities"][target]),
                "deadline": deadlines[target],
            })
    return points


# ------------------------------------------------------------------ per-source stats
def source_stats(points: list[dict[str, Any]]) -> dict[str, Any]:
    raw_values = [p["raw_remaining"] for p in points if p["raw_remaining"] is not None]
    old_values = [old_encode(r) for r in raw_values]
    new_values = [new_encode(r) for r in raw_values]

    old_distinct, old_list = distinct(old_values)
    new_distinct, new_list = distinct(new_values)
    raw_distinct, raw_list = distinct(raw_values)

    saturated_high = sum(1 for v in old_values if v == 1.0)
    saturated_low = sum(1 for v in old_values if v == -1.0)

    # 逐列特征统计：不同值数量与常量判定
    columns = []
    for index, name in enumerate(FEATURE_NAMES):
        values = [p["features"][index] for p in points]
        d, vals = distinct(values)
        columns.append({"index": index, "name": name, "distinct": d, "values": vals[:8] + (["..."] if d > 8 else [])})

    # 从编码值反推不可行的记录：饱和到 ±1 的记录
    unrecoverable = [
        {
            "scene": p["scene"], "target": p["target"], "step": p["step"],
            "reason": "saturated_at_+1_raw_at_least_50" if p["raw_remaining"] >= URGENCY_SCALE_STEPS
            else "saturated_at_-1_raw_at_most_-50",
            "raw_remaining": p["raw_remaining"],
        }
        for p in points
        if p["raw_remaining"] is not None
        and (p["raw_remaining"] >= URGENCY_SCALE_STEPS or p["raw_remaining"] <= -URGENCY_SCALE_STEPS)
    ]
    recoverable = len(points) - len(unrecoverable)

    # 编码碰撞：同一编码值对应的不同原始值组
    groups: dict[float, set[float]] = {}
    for p in points:
        if p["raw_remaining"] is None:
            continue
        groups.setdefault(old_encode(p["raw_remaining"]), set()).add(round(p["raw_remaining"], 6))
    collisions = [
        {"encoded_value": k, "distinct_raw_count": len(v), "raw_values": sorted(v)[:8]}
        for k, v in sorted(groups.items()) if len(v) >= 2
    ]

    # 存档编码值与从原始值重算的编码值必须一致（交叉验证）
    mismatch = [
        {"scene": p["scene"], "target": p["target"], "step": p["step"],
         "stored": p["features"][DEADLINE_COLUMN], "recomputed": old_encode(p["raw_remaining"])}
        for p in points
        if p["raw_remaining"] is not None
        and abs(p["features"][DEADLINE_COLUMN] - old_encode(p["raw_remaining"])) > 1e-9
    ]

    return {
        "sample_count": len(points),
        "raw_remaining": {
            "min": min(raw_values) if raw_values else None,
            "max": max(raw_values) if raw_values else None,
            "distinct": raw_distinct,
            "values": raw_list,
        },
        "old_encoded_column": {
            "distinct": old_distinct,
            "values": old_list,
            "saturated_high_count": saturated_high,
            "saturated_low_count": saturated_low,
            "saturated_share": (saturated_high + saturated_low) / len(old_values) if old_values else None,
        },
        "new_encoded_column": {
            "distinct": new_distinct,
            "min": min(new_values) if new_values else None,
            "max": max(new_values) if new_values else None,
            "values": new_list,
        },
        "columns": columns,
        "constant_columns": [c["name"] for c in columns if c["distinct"] == 1],
        "raw_unrecoverable_from_encoded_value": {
            "count": len(unrecoverable),
            "records": unrecoverable,
        },
        "raw_recoverable_from_encoded_value": {
            "count": recoverable,
            "rule": "remaining = encoded_value * 50 exactly when -1 < encoded < 1; saturated records only "
                    "give a bound (remaining >= 50 or <= -50), not the value",
        },
        "encoded_collisions": collisions,
        "stored_vs_recomputed_mismatch": mismatch,
    }


# ------------------------------------------------------------------ constancy causes
def constancy_causes(stats: dict[str, Any]) -> dict[str, str]:
    """常量列原因分类：选择条件 / 编码截断 / 原始数据恒定 / 证据不足。"""
    causes: dict[str, str] = {}
    selection_caused = {
        "first_emission": "首报选择条件：首报点上尚无历史发送，first_emission 恒为 1",
        "sent_count": "首报选择条件：尚无发送，sent_count 恒为 0",
        "since_last_emission_time": "首报选择条件：无上次发送，恒为 0",
        "pending_event": "relay 仅在有待报变更/检测时生成 item，首报点恒为 1",
        "changed_view": "item 生成条件即视图有变更，首报点恒为 1",
        "requirements_changed": "首报点无已发视图可比较，语义恒为 0/False",
        "members_changed": "首报点无已发视图可比较，语义恒为 0/False",
        "stage_changed": "首报点无已发视图可比较，语义恒为 0/False",
        "progress_delta_abs": "首报点无已发进度，差值恒为 0",
        "revision_delta": "首报点无已发 revision 可减，恒等于当前 revision",
    }
    encoding_caused = {
        "remaining_time_over_horizon": "编码截断：原始期限全部 >= 50 步，clip(remaining/50,-1,1) 恒为 1",
        "urgency_now": "近截止阈值：原始期限全部 > 25 步，urgency_now 恒为 0",
    }
    for name in stats["constant_columns"]:
        if name in selection_caused:
            causes[name] = selection_caused[name]
        elif name in encoding_caused:
            causes[name] = encoding_caused[name]
        else:
            causes[name] = "证据不足：本周证据无法判定该列恒定的机制原因"
    return causes


def main() -> int:
    sources = {
        "A_round28_96_points": source_a(),
        "B_round27_two_interventions": source_b(),
        "C_round26_four_improved_scenes": source_c(),
    }
    payload: dict[str, Any] = {
        "round": "29",
        "section": "deadline encoding audit: quantifying information loss from frozen artefacts",
        "encoding_constants": {
            "URGENCY_SCALE_STEPS": URGENCY_SCALE_STEPS,
            "EPISODE_HORIZON_STEPS": EPISODE_HORIZON_STEPS,
            "deadline_column_index": DEADLINE_COLUMN,
            "source_code": "integrated_sar/comm_learning.py:113 (clip), :114 (urgency_now), :120 (step_over_horizon)",
        },
        "field_provenance": {
            "encoded_feature_columns": "tape rows and archived intervention points stored by ROUND25/ROUND26/ROUND27/ROUND28 (same production features() output, stored verbatim)",
            "raw_deadline": "registry probe (_rp.jsonl) detection records: timestamp + detail.deadline_offset; this is the sender-visible registry snapshot the relay reads at decision time (comm.py:819-820, joint_driver.py:1058)",
            "decision_time": "the tape row's step under the verified clock rule now = tape step - 1 (evidence below)",
            "not_used": "receiver beliefs, delivery/accepted outcomes, final task state, send probabilities of other rounds, and any post-hoc terminal field are not inputs to any number in this audit",
        },
        "missing_and_gaps": {
            "ROUND26_own_records_lack_features": "runs/round26_analysis_20260929/IMPROVED_SCENES.json stores per-target first-send steps, charges and chain records but no actor feature rows; source C's feature rows were joined from the ROUND25 learned-greedy tapes ROUND26 itself read, and every point is cross-validated by the stored==recomputed check",
            "round28_point_count_is_not_scene_count": "A source's 96 points live on 16 scenes; per-source statistics are descriptive and are not 96 independent scenarios",
            "first_report_points_only": "the 102 points are first-report decision points; this round makes no claim about refresh decision points or other opportunity classes",
            "nonfinite_inputs": "no raw deadline in these records was non-finite; the prototype's explicit-failure behaviour is covered by the synthetic tests, not by data",
        },
        "clock_rule": {
            "rule": "now = tape step - 1",
            "evidence": "scene 1400010 learned tape: 197/197 candidate rows reproduce the stored "
                        "remaining_time_over_horizon exactly from deadline - (step - 1); 0 mismatch",
        },
        "sources": {},
    }
    for name, points in sources.items():
        stats = source_stats(points)
        stats["constancy_causes"] = constancy_causes(stats)
        payload["sources"][name] = stats
    totals = {name: len(points) for name, points in sources.items()}
    payload["totals"] = {"per_source": totals, "sum_of_samples": sum(totals.values()),
                         "note": "三来源选择机制不同，不合并为单一总体；除标注的描述性并排外不做跨源统计"}
    out = Path(__file__).resolve().parent / "ENCODING_AUDIT.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    for name, stats in payload["sources"].items():
        print(name, "samples:", stats["sample_count"],
              "| raw distinct:", stats["raw_remaining"]["distinct"],
              "| old distinct:", stats["old_encoded_column"]["distinct"],
              "| saturated share:", round(stats["old_encoded_column"]["saturated_share"], 4),
              "| new distinct:", stats["new_encoded_column"]["distinct"],
              "| unrecoverable:", stats["raw_unrecoverable_from_encoded_value"]["count"],
              "| mismatches:", len(stats["stored_vs_recomputed_mismatch"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
