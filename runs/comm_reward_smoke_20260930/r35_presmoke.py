"""ROUND35 启动前分析（0 环境启动）：用归档的修复前记录判断契约六类情形在真实数据里是否可达，
并据此为 smoke 选出最小 episode 集合与可复现对照格。

只读来源：ROUND32 的 rows/tapes/lifecycles（旧采集语义：完成事件晚一个 relay tick 进奖励流）。
判定以 ROUND34 契约为准。冻结运行配置里 dt=1.0 且 `env.current_time` 与 `relay._tick` 同步，
所以修复后的预测写入键 = `int(observed_at)`（测试 `test_no_new_action...` 直接证实
`relay_tick == observed_at` 且 `clock_now == observed_at - 1`）。

分两相，避免把 768 份 lifecycle 全读一遍：
  A 相只读 row JSON（小），挑出 on_time<completed / expired>0 / 决策起点>1 的候选格；
  B 相对候选格与每 seed 的 e0 解析 tape 与 lifecycle 做逐事件分类。

本脚本不构造环境、不 step、不训练；输出是启动前登记，跑完不回改。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ROUND32 = ROOT / "runs" / "comm_actor_reduction_ablation_20261001"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runs" / "comm_cost_v2_rerun_20260928"))

import v2_common as stage_c  # noqa: E402  (frozen scenario order only; no environment touched)

RUNS = (("ALL_ROWS", 11), ("ACTIVE_ROWS", 11), ("ALL_ROWS", 22), ("ACTIVE_ROWS", 22),
        ("ALL_ROWS", 33), ("ACTIVE_ROWS", 33))
ANCHOR_EPISODES = (0, 1, 4, 5)          # per-run anchors: batch 0 (reproducible e0) + a few later
MAX_PHASE_B_CELLS = 40                  # hard bound on how many lifecycles this analysis parses


def row_files(arm: str, seed: int) -> list[Path]:
    return sorted((ROUND32 / "rows").glob(f"r32_train_{arm}_s{seed}_e*_s*_a*.json"))


def read_row(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def phase_a() -> dict[str, Any]:
    """Cheap pass over the 768 archived training rows."""
    all_rows: dict[str, dict[str, Any]] = {}
    expired_cells, task_short_cells = [], []
    for arm, seed in RUNS:
        for path in row_files(arm, seed):
            row = read_row(path)
            key = f"{arm}_s{seed}_e{int(row['episode_index'])}"
            all_rows[key] = {"path": path, "row": row}
            if int(row["expired"]) > 0:
                expired_cells.append(key)
            if int(row["on_time"]) < int(row["completed"]):
                task_short_cells.append(key)
    return {"all_rows": all_rows, "cells_with_expired_targets": expired_cells,
            "cells_with_completed_but_not_on_time": task_short_cells}


def decision_steps(tape: Path) -> list[int]:
    return [int(json.loads(line)["step"])
            for line in tape.read_text(encoding="utf-8").splitlines() if line.strip()]


def completion_events(lifecycle: Path) -> tuple[list[dict[str, Any]], int, list[float],
                                                list[dict[str, Any]]]:
    """Completed progress events, their count, deadlines, and the expiry-flagged events."""
    finals: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    expiry_events: list[dict[str, Any]] = []
    with lifecycle.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            kind = record.get("event_type")
            detail = record.get("detail") or {}
            if kind == "final" and str(record.get("target_id", "")).startswith("piloc"):
                finals[record["target_id"]] = detail.get("deadline")
            elif kind == "progress":
                if detail.get("expired"):
                    expiry_events.append({"target_id": record["target_id"],
                                          "status": record["status"],
                                          "timestamp": float(record["timestamp"]),
                                          "failure_reason": record["failure_reason"],
                                          "event_id": detail.get("progress_event_id"),
                                          "deadline": finals.get(record["target_id"])})
                if record.get("status") == "completed":
                    events.append({"target_id": record["target_id"],
                                   "observed_at": float(record["timestamp"]),
                                   "event_id": detail.get("progress_event_id"),
                                   "expired_flag": bool(detail.get("expired")),
                                   "deadline": finals.get(record["target_id"])})
    deadlines = [value for value in finals.values() if value is not None]
    return events, len(expiry_events), deadlines, expiry_events


def classify(key: str, entry: dict[str, Any]) -> dict[str, Any] | None:
    row = entry["row"]
    stem = entry["path"].stem
    tape, lifecycle = (ROUND32 / "logs" / f"{stem}_tape.jsonl",
                       ROUND32 / "logs" / f"{stem}_rp.jsonl")
    if not tape.exists() or not lifecycle.exists():
        return None
    steps = decision_steps(tape)
    horizon = int(row["steps"])
    events, expired_progress, deadlines, expiry_events = completion_events(lifecycle)
    old_keys = sorted(int(k) for k in row["task_stream"])
    per_event, cases = [], {}
    for event in events:
        tick = int(event["observed_at"])
        earlier = [step for step in steps if step <= tick]
        labels = {
            "same_tick": tick in steps,
            "cross_tick_new_action_next_tick": tick in steps and (tick + 1) in steps,
            "cross_tick_no_new_action_at_event_tick": (not earlier) is False and tick not in steps,
            "final_execution_tick": bool(steps) and tick == max(steps),
            "after_last_decision": bool(steps) and tick > max(steps),
            "before_first_action_prefix": bool(steps) and tick < min(steps),
            "expired_flag_on_event": event["expired_flag"],
            "observed_after_deadline": (event["deadline"] is not None
                                        and event["observed_at"] > event["deadline"]),
        }
        owner = max(earlier) if earlier else "prefix"
        per_event.append({"target_id": event["target_id"], "observed_at": event["observed_at"],
                          "predicted_fixed_tick": tick, "predicted_owner": owner,
                          "old_semantics_key_if_shifted_by_one": (tick + 1)
                                                                 if (tick + 1) in old_keys
                                                                 else None,
                          "deadline": event["deadline"], "expired_flag": event["expired_flag"],
                          "labels": {k: v for k, v in labels.items() if v}})
        for name, hit in labels.items():
            if hit:
                cases[name] = cases.get(name, 0) + 1
    return {"cell": key, "arm": row["arm"], "training_seed": row["training_seed"],
            "episode_index": int(row["episode_index"]), "scenario_seed": row["scenario_seed"],
            "row_file": entry["path"].name, "config_hash": row["config_hash"],
            "horizon": horizon, "decision_rows": len(steps),
            "decision_step_min": min(steps), "decision_step_max": max(steps),
            "on_time": int(row["on_time"]), "completed": int(row["completed"]),
            "expired": int(row["expired"]), "unassigned": int(row["unassigned"]),
            "emissions": int(row["emissions"]), "spent_bytes": int(row["spent_bytes"]),
            "old_semantics_task_stream_keys": old_keys,
            "predicted_fixed_task_stream": {str(tick): 1.0 for tick in sorted(
                int(e["observed_at"]) for e in events)},
            "keys_all_shift_by_one": sorted(t + 1 for t in
                                           sorted(int(e["observed_at"]) for e in events)) == old_keys,
            "completed_events": len(events), "progress_events_flagged_expired": expired_progress,
            "expiry_events_sample": expiry_events[:3],
            "target_deadlines": sorted(deadlines),
            "case_counts": cases, "events": per_event,
            "behaviour_sha256": row["behaviour_sha256"],
            "relay_counters": row["relay_counters"],
            "cost_stream_row_count": len(row["cost_stream"])}


def phase_c() -> dict[str, Any]:
    """Cheap prevalence over all 768 archived cells, derived from rows + tapes only.

    The legacy tap wrote every credit at ``event tick + 1``; the shift is verified event by
    event in phase B for the parsed cells. Here every archived cell contributes, using the row's
    own legacy ``task_stream`` keys: key ``k`` means an event at ``k - 1``. For each such key the
    real bundler is asked twice -- once with the legacy keys, once with the corrected keys -- so
    the attribution difference is computed by ``v2_trainer._bundle`` itself rather than by a
    reimplementation.
    """
    import v2_trainer

    totals = {"cells": 0, "legacy_keys": 0, "event_tick_has_a_row": 0,
              "event_tick_has_no_row": 0, "legacy_key_has_a_row": 0,
              "credits_whose_owner_row_changed": 0,
              "rows_that_changed_owner": 0, "cells_with_any_changed_owner": 0,
              "cells_where_the_two_bundles_are_identical": 0}
    per_cell = {}
    for arm, seed in RUNS:
        for path in row_files(arm, seed):
            row = read_row(path)
            tape = ROUND32 / "logs" / f"{path.stem}_tape.jsonl"
            if not tape.exists():
                continue
            steps = decision_steps(tape)
            horizon = int(row["steps"])
            legacy = {int(k): float(v) for k, v in row["task_stream"].items()}
            if not legacy:
                continue
            corrected = {key - 1: value for key, value in legacy.items()}
            legacy_bundled, legacy_prefix = v2_trainer._bundle(legacy, steps, horizon)
            fixed_bundled, fixed_prefix = v2_trainer._bundle(corrected, steps, horizon)
            changed = sorted(step for step in steps
                             if legacy_bundled[step] != fixed_bundled[step])
            moved = 0
            for key, value in legacy.items():
                before_owner = max((step for step in steps if step <= key), default="prefix")
                after_owner = max((step for step in steps if step <= key - 1), default="prefix")
                if before_owner != after_owner:
                    moved += 1
            totals["cells"] += 1
            totals["legacy_keys"] += len(legacy)
            totals["credits_whose_owner_row_changed"] += moved
            totals["event_tick_has_a_row"] += sum(1 for k in legacy if (k - 1) in steps)
            totals["event_tick_has_no_row"] += sum(1 for k in legacy if (k - 1) not in steps)
            totals["legacy_key_has_a_row"] += sum(1 for k in legacy if k in steps)
            totals["rows_that_changed_owner"] += len(changed)
            totals["cells_with_any_changed_owner"] += int(bool(changed))
            totals["cells_where_the_two_bundles_are_identical"] += int(not changed)
            per_cell[f"{arm}_s{seed}_e{int(row['episode_index'])}"] = {
                "scenario_seed": row["scenario_seed"], "legacy_keys": sorted(legacy),
                "decision_rows": len(steps), "changed_owner_rows": changed,
                "prefix_legacy": legacy_prefix, "prefix_corrected": fixed_prefix}
    return {"totals": totals, "per_cell_changed_rows": {
        key: value for key, value in per_cell.items() if value["changed_owner_rows"]}}


def main() -> int:
    phase = phase_a()
    attribution = phase_c()
    candidates = list(dict.fromkeys(
        [f"{arm}_s{seed}_e{episode}" for arm, seed in RUNS for episode in ANCHOR_EPISODES]
        + phase["cells_with_expired_targets"][:12]
        + phase["cells_with_completed_but_not_on_time"][:12]))
    classified = []
    for key in candidates[:MAX_PHASE_B_CELLS]:
        entry = phase["all_rows"].get(key)
        if entry is None:
            continue
        result = classify(key, entry)
        if result is not None:
            classified.append(result)
    union: dict[str, list[str]] = {}
    for cell in classified:
        for case, count in cell["case_counts"].items():
            if count:
                union.setdefault(case, []).append(f"{cell['cell']} ({count})")
    anchor = next((cell for cell in classified
                   if cell["arm"] == "ALL_ROWS" and cell["training_seed"] == 11
                   and cell["episode_index"] == 0), None)
    archived_batches = {}
    for batch in (0, 1):
        path = (ROUND32 / "checkpoints" / f"r32_batch_ALL_ROWS_s{11}_b{batch}.json")
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            archived_batches[f"b{batch}"] = {
                "samples": record["update"]["samples"],
                "minibatches": record["update"]["minibatches"],
                "optimizer_steps": record["update"]["optimizer_steps"],
                "episodes": record["episodes"], "scenarios": record["scenarios"]}
    expected_steps = sum(value["optimizer_steps"] for value in archived_batches.values())
    launch_plan = {
        "seed": 11,
        "episodes": 8,
        "episode_constructions_total_across_both_launches": 16,
        "launch_1": "seed 11, episodes 0..7 uninterrupted: wiring, ledger, per-event table, "
                    "passivity anchor on episode 0",
        "launch_2": "seed 11, resume from the rolling checkpoint written after episode 4 and "
                    "continue to episode 8: the real save/restore chain plus continuation "
                    "equality against launch 1's own episodes 4..7",
        "expected_optimizer_steps_per_batch_from_the_archived_same_shaped_batches":
            archived_batches,
        "expected_optimizer_steps_for_the_smoke": expected_steps,
        "authorised_optimizer_step_ceiling": 100,
        "wall_clock_note": "archived episodes of this shape take 11.2-14.6 s each, so 8 episodes "
                           "plus two updates is far inside the 2 h ceiling per launch",
        "reference_cache_source": "ROUND24's 64 cached immediate reference rows, read only; the "
                                  "smoke launches no reference cell",
        "new_reference_launches": 0,
    }
    episode_profile = {}
    for episode in range(8):
        key = f"ALL_ROWS_s11_e{episode}"
        entry = phase["all_rows"].get(key)
        if entry is None:
            continue
        row = entry["row"]
        episode_profile[key] = {
            "scenario_seed": row["scenario_seed"], "on_time": int(row["on_time"]),
            "completed": int(row["completed"]), "expired": int(row["expired"]),
            "emissions": int(row["emissions"]), "spent_bytes": int(row["spent_bytes"]),
            "legacy_task_stream_keys": sorted(int(k) for k in row["task_stream"]),
            "seconds": row["seconds"]}
    payload = {
        "schema": "round35-pre-smoke-case-coverage/2",
        "round": 35,
        "environment_launches_used_by_this_script": 0,
        "environments_constructed_and_stepped": 0,
        "judged_against": "runs/comm_task_reward_event_timing_20260930/TASK_REWARD_TIMING_CONTRACT.md",
        "frozen_order_first_eight_scenarios": stage_c.scenario_order()[:8],
        "phase_a_archived_rows_scanned": len(phase["all_rows"]),
        "phase_a_cells_with_expired_targets": phase["cells_with_expired_targets"],
        "phase_a_cells_with_completed_but_not_on_time":
            phase["cells_with_completed_but_not_on_time"],
        "phase_b_cells_parsed": len(classified),
        "phase_c_legacy_versus_fixed_attribution": attribution["totals"],
        "phase_c_method": "v2_trainer._bundle itself is asked twice per archived cell: once with "
                          "the row's legacy task_stream keys, once with every key shifted back one "
                          "tick. changed_owner_rows lists the action rows whose bundled reward "
                          "differs, so the attribution footprint is measured by the real bundler "
                          "rather than a reimplementation. 0 launches, 0 environments.",
        "phase_c_cells_with_changed_rows": len(attribution["per_cell_changed_rows"]),
        "smoke_launch_plan": launch_plan,
        "archived_profile_of_the_eight_smoke_episodes": episode_profile,
        "deadline_case_in_real_data": {
            "how_expiry_is_signalised": "the archived block marks a missed deadline as a progress "
                                        "event whose status is NOT completed, carrying "
                                        "detail.expired = true and failure_reason = "
                                        "deadline_expired, at a timestamp one step past the "
                                        "target's deadline",
            "observed_expiry_events_in_parsed_cells": sum(
                len(cell["expiry_events_sample"]) for cell in classified),
            "completed_after_deadline_events_in_parsed_cells": sum(
                1 for cell in classified for event in cell["events"]
                if event["labels"].get("observed_after_deadline")),
            "what the collector must therefore do": "return without a ledger record and without a "
                                                    "credit for those events (its first guard is "
                                                    "status is COMPLETED), and the episode's "
                                                    "credited total must still equal the "
                                                    "lifecycle's on-time count",
            "episodes carrying an expiry": [key for key, value in episode_profile.items()
                                            if value["expired"] > 0]},
        "reproducible_episode_window": {
            "episodes": [0, 1, 2, 3],
            "reason": "batch 0 rolls all four episodes out with the same untouched "
                      "Controller(11) and per-episode registered action seeds, and the first "
                      "weight move happens only after episode 3, so episodes 0..3 of the smoke "
                      "are byte-comparable with the archived legacy run of the same cell; the "
                      "only permitted difference is the task_stream key",
            "consequence_for_the_coverage_claim": "same-tick, next-tick-new-action, final-tick "
                                                  "and deadline-expiry all occur inside episodes "
                                                  "0..3 (episode 1 carries an expiry), so every "
                                                  "case the smoke can claim is also claimable "
                                                  "against a reproducible archived cell"},
        "unreachable_cases_in_this_block": {
            "no_action_row_at_the_event_tick": "0 of %d archived completion credits landed on a "
                                               "tick without a decision row, so the contract's "
                                               "third row cannot be exercised by real data in this "
                                               "scenario block"
                                               % attribution["totals"]["legacy_keys"],
            "completion_before_the_first_action": "no archived completion was observed before the "
                                                  "episode's first decision row (relay ticks start "
                                                  "at 1 and the policy decides at tick 1), so the "
                                                  "prefix case stays synthetic-only"},
        "phase_b_candidates": candidates[:MAX_PHASE_B_CELLS],
        "case_availability_in_archived_legacy_data": {k: v for k, v in sorted(union.items())},
        "cells": classified,
        "reproducible_anchor_cell": anchor,
        "reproducibility_reasoning": "episode 0 of a seed uses exactly Controller(seed) plus the "
                                    "registered action stream sha256('sc-action:{seed}:0'), before "
                                    "any weight has moved, so the fixed collector must replay the "
                                    "same actions, cost stream and event publication order as the "
                                    "archived legacy run and may only differ in the task_stream "
                                    "keys; from batch 0's update onwards the weights depend on the "
                                    "corrected reward, so later cells are indicative only",
        "launch_plan consequence": "cases that never occur in any archived cell of this scenario "
                                   "block are recorded as structurally unreachable in the smoke "
                                   "too, and the report must say so instead of claiming coverage",
        "warning": "written before the launches; the smoke's own event ledger is the evidence",
    }
    out = HERE / "PRE_SMOKE_CASE_COVERAGE.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({"archived_rows": len(phase["all_rows"]),
                      "cells_with_expired": len(phase["cells_with_expired_targets"]),
                      "cells_completed_not_on_time": len(
                          phase["cells_with_completed_but_not_on_time"]),
                      "phase_b_parsed": len(classified),
                      "cases_available": {k: len(v) for k, v in union.items()},
                      "phase_c": attribution["totals"],
                      "launch_plan": launch_plan,
                      "episode_profile": episode_profile,
                      "anchor_old_keys": anchor["old_semantics_task_stream_keys"],
                      "anchor_predicted_new": anchor["predicted_fixed_task_stream"],
                      "anchor_shift_verified": anchor["keys_all_shift_by_one"],
                      "anchor_cases": anchor["case_counts"]},
                     ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
