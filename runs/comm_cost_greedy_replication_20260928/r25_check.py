"""ROUND25 汇总与门禁：绝对指标、逐场景差、目标身份权衡、首 6 次技术门禁。

门禁只做技术核验（执行依据 §五）：初始场景/配置可比、冻结权重与 buffer 每集前后不变、
learned 走原 greedy 不混 sampled、动作/mask/发射/拒付/预算逐步闭合、成本账本与任务生命周期
与 final record 齐全、旧产物未覆盖。门禁不要求任务或成本表现变好。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import r25_common as common
import r25_run as runner
import r25_stats as stats

SEND, HOLD = 1, 0


# ------------------------------------------------------------------ 绝对指标 -------------
def absolute_metrics(table: dict[str, dict[int, dict[str, Any]]] | None = None) -> dict[str, Any]:
    table = stats.load_rows() if table is None else table
    stats.require_complete(table)
    scenes = [int(scene) for scene in common.NEW_SCENARIOS]
    arms: dict[str, Any] = {}
    for arm in common.ARM_ORDER:
        rows = [table[arm][scene] for scene in scenes]
        on_time = [int(row["on_time"]) for row in rows]
        completed = [int(row["completed"]) for row in rows]
        expired = [int(row["expired"]) for row in rows]
        discovered = [int(row["discovered"]) for row in rows]
        spent = [int(row["spent_bytes"]) for row in rows]
        emissions = [int(row["emissions"]) for row in rows]
        budget_refusals = [int(row["budget_refused_sends"]) for row in rows]
        capacity_refusals = [int(row["capacity_refused_sends"]) for row in rows]
        usage = [float(row["budget_used_fraction"]) for row in rows]
        arms[arm] = {
            "role": common.ARMS[arm]["role"],
            "scenes": len(rows),
            "on_time_total": int(sum(on_time)),
            "on_time_mean_per_scene": round(float(np.mean(on_time)), 6),
            "on_time_median": float(np.median(on_time)),
            "on_time_min": int(min(on_time)), "on_time_max": int(max(on_time)),
            "completed_total": int(sum(completed)),
            "on_time_over_completed": round(sum(on_time) / max(1, sum(completed)), 6),
            "expired_total": int(sum(expired)),
            "discovered_total": int(sum(discovered)),
            "spent_bytes_total": int(sum(spent)),
            "spent_bytes_mean_per_scene": round(float(np.mean(spent)), 3),
            "spent_bytes_median": float(np.median(spent)),
            "emissions_total": int(sum(emissions)),
            "emissions_mean_per_scene": round(float(np.mean(emissions)), 3),
            "budget_refused_sends_total": int(sum(budget_refusals)),
            "capacity_refused_sends_total": int(sum(capacity_refusals)),
            "refusal_bytes_billed": 0,
            "budget_used_fraction_mean": round(float(np.mean(usage)), 6),
            "budget_used_fraction_max": round(float(np.max(usage)), 6),
            "per_scene_on_time": on_time,
            "per_scene_spent_bytes": spent,
            "per_scene_emissions": emissions,
        }
    return {"round": "25", "section": "absolute metrics of the three fixed arms",
            "scenes": scenes, "arms": arms,
            "cost_unit": "measured payload-attempt cost in bytes: dropped packets billed in full, "
                         "refusals billed zero and listed separately; not whole-system traffic",
            "pooling": "no ROUND24 development scene is pooled into these numbers"}


def write_absolute_metrics() -> Path:
    payload = absolute_metrics()
    path = common.OUTPUT_DIR / "ABSOLUTE_METRICS.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


# ------------------------------------------------------------------ 逐场景与目标身份 -------------
def per_scene_differences() -> dict[str, Any]:
    paired = stats.write_statistics()
    payload = json.loads(paired.read_text(encoding="utf-8"))
    table = stats.load_rows()
    scenes = [int(scene) for scene in common.NEW_SCENARIOS]
    candidate = common.CANDIDATE_ARM
    differences = {}
    for comparator in common.TASK_COMPARATORS:
        differences[comparator] = payload["comparisons"][comparator]["per_scene"]
    identity: dict[str, Any] = {}
    for comparator in common.TASK_COMPARATORS:
        rows = []
        for scene in scenes:
            cand_ids = set(table[candidate][scene]["goal_identity"]["on_time_goal_ids"])
            comp_ids = set(table[comparator][scene]["goal_identity"]["on_time_goal_ids"])
            dy = len(cand_ids) - len(comp_ids)
            rows.append({
                "scene": scene,
                "candidate_on_time_goal_ids": sorted(cand_ids),
                "comparator_on_time_goal_ids": sorted(comp_ids),
                "lost_on_time_goal_ids": sorted(comp_ids - cand_ids),
                "gained_on_time_goal_ids": sorted(cand_ids - comp_ids),
                "dY": dy, "net_tie": dy == 0,
                "identity_swap_despite_net_tie": bool(dy == 0 and (comp_ids - cand_ids)),
            })
        identity[comparator] = rows
    return {"round": "25", "section": "per-scene differences and goal identity tradeoffs",
            "candidate_arm": candidate, "comparators": list(common.TASK_COMPARATORS),
            "per_scene": differences, "goal_identity": identity,
            "note": "every scene is listed, including net-tie scenes; a net tie does not prove "
                    "each target is safe"}


def write_per_scene() -> Path:
    payload = per_scene_differences()
    path = common.OUTPUT_DIR / "PER_SCENE_DIFFERENCES.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def write_task_identity() -> Path:
    differences = per_scene_differences()["goal_identity"]
    body = {"round": "25", "section": "goal identity tradeoffs (on-time goal IDs)",
            "candidate_arm": common.CANDIDATE_ARM, "comparators": differences}
    path = common.OUTPUT_DIR / "TASK_IDENTITY_TRADEOFFS.json"
    path.write_text(json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


# ------------------------------------------------------------------ 首 6 次技术门禁 -------------
def _load_row(arm: str, scene: int, attempt: int = 1) -> dict[str, Any]:
    path = common.row_path(arm, scene, attempt)
    if not path.exists():
        raise RuntimeError(f"missing final record {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_tape(row: dict[str, Any], controller: Any) -> dict[str, Any]:
    """Re-derive every tape row of one learned episode from the frozen checkpoint."""
    tape_path = Path(row["artefacts"]["tape"])
    if not tape_path.exists():
        raise RuntimeError(f"{row['arm']} s{row['scenario_seed']}: no tape artefact")
    tapes = [json.loads(line) for line in tape_path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    transcript_sends = {(int(step), str(target)) for step, target, *_ in
                        row["emission_sequence"]}
    max_prob_gap = 0.0
    violations: list[str] = []
    rows_checked = 0
    for tape in tapes:
        if tape.get("action_mask_schema") != "budget-sequential-mask/1":
            violations.append(f"step {tape['step']}: mask schema {tape.get('action_mask_schema')}")
        if tape.get("schema") != "onpolicy-tape/2":
            violations.append(f"step {tape['step']}: tape schema {tape.get('schema')}")
        targets = list(tape["targets"])
        if set(tape["free"]) != set(tape["actionable"]):
            violations.append(f"step {tape['step']}: free != actionable")
        if set(tape["masked"]) & set(tape["actionable"]):
            violations.append(f"step {tape['step']}: masked leaks into actionable")
        if sorted(tape["forced"] + tape["masked"] + tape["actionable"]) != sorted(targets):
            violations.append(f"step {tape['step']}: candidate classes do not cover all targets")
        if not tape["features"]:
            continue
        matrix = torch.as_tensor(np.asarray(tape["features"], dtype=np.float32))
        with torch.no_grad():
            logits, _values = controller.forward(matrix)
            probabilities = torch.softmax(logits, dim=-1)
        for index, target in enumerate(targets):
            send_p = float(probabilities[index, SEND].item())
            recorded = float(tape["send_probabilities"].get(target, float("nan")))
            if target in tape["actionable"]:
                max_prob_gap = max(max_prob_gap, abs(send_p - recorded))
                action = int(tape["actions"][target])
                greedy = SEND if probabilities[index, SEND] >= probabilities[index, HOLD] else HOLD
                if action != greedy:
                    violations.append(
                        f"step {tape['step']} {target}: action {action} != greedy {greedy}")
                if action == SEND and (int(tape["step"]), target) not in transcript_sends:
                    violations.append(f"step {tape['step']} {target}: SEND not in the transcript")
            else:
                if int(tape["actions"][target]) != HOLD:
                    violations.append(
                        f"step {tape['step']} {target}: non-actionable action is not HOLD")
        shadow = tape["candidate_shadow"]
        expected_before = float(tape["shadow_budget_before"])
        for target in targets:
            entry = shadow[target]
            if abs(float(entry["budget_before"]) - expected_before) > 1e-9:
                violations.append(f"step {tape['step']} {target}: shadow bookkeeping broke")
            spent_here = (int(tape["attempt_charge_bytes"][target])
                          if int(tape["actions"][target]) == SEND else 0)
            expected_after = expected_before - spent_here
            if abs(float(entry["budget_after"]) - expected_after) > 1e-9:
                violations.append(f"step {tape['step']} {target}: shadow charge mismatch")
            expected_before = float(entry["budget_after"])
        if abs(expected_before - float(tape["shadow_budget_after"])) > 1e-9:
            violations.append(f"step {tape['step']}: episode shadow does not close")
        rows_checked += 1
    for step, target, *_ in row["emission_sequence"]:
        if (int(step), str(target)) not in {
                (int(tape["step"]), t)
                for tape in tapes for t, action in tape["actions"].items() if action == SEND}:
            violations.append(f"step {step} {target}: transcript send not produced by the tape")
    return {"tape_rows": rows_checked, "max_probability_gap": max_prob_gap,
            "violations": violations, "passed": not violations and max_prob_gap <= 1e-6}


def first6_gate() -> dict[str, Any]:
    """The technical gate over the first two scenes' three arms (6 launches, inside the 192)."""
    table = stats.load_rows()
    controller, candidate_identity = runner.load_candidate_controller()
    frozen_weights = {"piloc": common.sha256_file(common.CHECKPOINT_FILES[0]),
                      "ctde": common.sha256_file(common.CHECKPOINT_FILES[1])}
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            failures.append(name)

    # 1. all six cells exist with a complete final record
    cells = []
    for scene in common.GATE_SCENARIOS:
        for arm in common.ARM_ORDER:
            row = _load_row(arm, scene)
            cells.append(row)
            if not row["goal_identity"]["log_complete"]:
                record(f"final_record_{arm}_s{scene}", False, "lifecycle log has no final record")
            if int(row["steps"]) != common.STEPS_PER_EPISODE:
                record(f"steps_{arm}_s{scene}", False, row["steps"])
            if int(row["optimizer_steps"]) != 0:
                record(f"optimizer_{arm}_s{scene}", False, row["optimizer_steps"])
    record("six_cells_present", len(cells) == 6, [f"{r['arm']}_s{r['scenario_seed']}" for r in cells])

    # 2. initial scene and configuration comparability across arms
    for scene in common.GATE_SCENARIOS:
        digests = {row["arm"]: row["behaviour_config_sha256"] for row in cells
                   if int(row["scenario_seed"]) == scene}
        record(f"behaviour_config_s{scene}", len(set(digests.values())) == 1, digests)
        expected_fields, _ = common.stage_b_inherited_config(scene)
        expected = runner._behaviour_config_sha256(expected_fields)
        record(f"behaviour_config_matches_inheritance_s{scene}",
               set(digests.values()) == {expected}, expected)
        for row in cells:
            if int(row["scenario_seed"]) != scene:
                continue
            declared = row["transport_parameters"]
            ok = (declared["policy"] == row["policy"]
                  and declared["timing_mode"] == "policy_owned_timing"
                  and declared["ack_enabled"] is False
                  and float(declared["delivery_probability"]) == 0.7
                  and int(row["budget_bytes"]) == common.COMM_BUDGET_BYTES
                  and row["run_signature_checkpoints"]["piloc"] == frozen_weights["piloc"]
                  and row["run_signature_checkpoints"]["ctde"] == frozen_weights["ctde"])
            record(f"transport_and_weights_{row['arm']}_s{scene}", ok, declared)

    # 3. frozen weights and model buffer unchanged around and across episodes
    for row in cells:
        if row["arm"] != common.CANDIDATE_ARM:
            continue
        before = row["controller_parameters_before"]
        after = row["controller_parameters_after"]
        same_episode = before is not None and before == after
        record(f"parameters_unchanged_in_episode_s{row['scenario_seed']}", same_episode,
               {"before": before, "after": after})
    learned_rows = [row for row in cells if row["arm"] == common.CANDIDATE_ARM]
    summaries = {json.dumps(row["controller_parameters_after"], sort_keys=True) for row in learned_rows}
    record("parameters_identical_across_gate_episodes", len(summaries) == 1, len(summaries))
    record("candidate_identity_matches_freeze",
           all(row["model_identity"]["checkpoint_sha256"] == common.sha256_file(
               common.LEARNED_CHECKPOINT) for row in learned_rows),
           candidate_identity["checkpoint_sha256"])

    # 4. the learned arm is the original greedy decode, not a mixed sampled path
    for row in learned_rows:
        summary = (row.get("collector_summary") or {}).get("timing") or {}
        tape_result = _verify_tape(row, controller)
        record(f"greedy_decode_and_mask_s{row['scenario_seed']}", tape_result["passed"],
               {"max_probability_gap": tape_result["max_probability_gap"],
                "tape_rows": tape_result["tape_rows"],
                "violations": tape_result["violations"][:6]})
        record(f"collector_schemas_s{row['scenario_seed']}",
               summary.get("action_mask_schema") == "budget-sequential-mask/1"
               and summary.get("tape_schema") == "onpolicy-tape/2"
               and summary.get("trainer_tag") == "onpolicy-v3-budget-mask",
               summary)

    # 5. cost ledger, task lifecycle and refusal bookkeeping close
    for row in cells:
        cost_sum = sum(int(entry["airtime_bytes"]) for entry in row["cost_stream"].values())
        record(f"cost_ledger_closes_{row['arm']}_s{row['scenario_seed']}",
               cost_sum == int(row["spent_bytes"]) == int(row["airtime_tap_total"]),
               {"tap": cost_sum, "spent": row["spent_bytes"]})
        credited = sum(float(value) for value in row["task_stream"].values())
        record(f"task_lifecycle_closes_{row['arm']}_s{row['scenario_seed']}",
               credited == int(row["on_time"]) == len(row["goal_identity"]["on_time_goal_ids"])
               == int(row["task_reward_total"]),
               {"tap": credited, "on_time": row["on_time"]})
        record(f"refusals_split_{row['arm']}_s{row['scenario_seed']}",
               int(row["budget_refused_sends"]) + int(row["capacity_refused_sends"])
               == len(row["refusal_sequence"]) and int(row["refusal_bytes_billed"]) == 0,
               {"budget": row["budget_refused_sends"], "capacity": row["capacity_refused_sends"]})
        record(f"ack_zero_{row['arm']}_s{row['scenario_seed']}",
               not any(row["ack_ledger"].values()), row["ack_ledger"])

    # 6. old artefacts untouched
    round24_rows = len(list((common.ROUND24_DIR / "rows").glob("v2_*.json")))
    record("round24_rows_untouched", round24_rows == 672, round24_rows)
    own_paths = {str(Path(row["artefacts"]["row"])) for row in cells}
    record("artefacts_inside_round_directory",
           all(str(common.OUTPUT_DIR) in path for path in own_paths), sorted(own_paths))

    payload = {
        "round": "25",
        "section": "first-6 technical gate (2 scenes x 3 arms, inside the 192 launches)",
        "gate_scenes": [int(scene) for scene in common.GATE_SCENARIOS],
        "cells": [{"arm": row["arm"], "scene": int(row["scenario_seed"]),
                   "on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
                   "emissions": int(row["emissions"])} for row in cells],
        "checks": checks,
        "failures": failures,
        "passed": not failures,
        "technical_only": "the gate never requires a task or cost improvement; a performance "
                          "regression is not a technical failure and is not deleted, isolated "
                          "or rerun for a better draw",
        "launches_used": common.launch_count(),
    }
    common.write_json("FIRST6_GATE.json", payload)
    return payload


def write_gate() -> Path:
    payload = first6_gate()
    return common.OUTPUT_DIR / "FIRST6_GATE.json"


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="ROUND25 aggregation and gate")
    parser.add_argument("mode", choices=("gate", "absolute", "perscene", "identity", "all"))
    args = parser.parse_args(argv)
    if args.mode == "gate":
        payload = first6_gate()
        print(json.dumps({"passed": payload["passed"], "failures": payload["failures"]},
                         ensure_ascii=False))
        return 0 if payload["passed"] else 1
    if args.mode == "absolute":
        write_absolute_metrics()
        print("absolute metrics written")
        return 0
    if args.mode == "perscene":
        write_per_scene()
        print("per-scene differences written")
        return 0
    if args.mode == "identity":
        write_task_identity()
        print("task identity tradeoffs written")
        return 0
    write_absolute_metrics()
    write_per_scene()
    write_task_identity()
    print("aggregates written")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
