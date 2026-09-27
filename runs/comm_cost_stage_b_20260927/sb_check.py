"""Stage B 门禁检查器：只读落盘 artefact 的独立重算（B1-B4）。

设计原则：本模块**不调用 runner、不构造 `JointRun`、不启动环境**；每条判据都从
`rows/*.json`、`logs/*_tape.jsonl`、`logs/*_ct.jsonl`、`logs/*_rp.jsonl` 重新推导，
与运行时的行文件相互独立。相同权重重算 = 用预登记固定初始化 seed 重建
`StageBController`（ACTOR_INIT_SEED 已冻结在协议里）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sb_common as common
from sb_controller import StageBController

import cost_math as cm

from integrated_sar.comm_policy import transport_trim_key
from integrated_sar.lifecycle_metrics import load_records, window_metrics


# ------------------------------------------------------------------ 读取 -------------
def load_row(path_name: str, seed: int, attempt: int = 1) -> dict[str, Any]:
    path = common.cell_paths(path_name, seed, attempt)["row"]
    if not path.exists():
        raise FileNotFoundError(f"missing row artefact: {path}")
    return common.read_json(path)


def load_tape(path_name: str, seed: int, attempt: int = 1) -> list[dict[str, Any]]:
    path = common.cell_paths(path_name, seed, attempt)["tape"]
    if not path.exists():
        return []
    return list(common.iter_jsonl(path))


def load_transcript(path_name: str, seed: int, attempt: int = 1) -> list[dict[str, Any]]:
    path = common.cell_paths(path_name, seed, attempt)["transcript"]
    return list(common.iter_jsonl(path))


def load_lifecycle(path_name: str, seed: int, attempt: int = 1) -> list[dict[str, Any]]:
    path = common.cell_paths(path_name, seed, attempt)["lifecycle"]
    return load_records(Path(path))


# ------------------------------------------------------------------ B1 等价 -------------
def gate_b1_equivalence(seed: int) -> dict[str, Any]:
    """Native vs injected, on the packets themselves: same messages, steps, order, money."""
    checks = []
    for reference, injected, label in (("R_send", "I_send", "native immediate vs injected"),
                                       ("R_hold", "I_hold", "native always_hold vs injected")):
        left = load_row(reference, seed)
        right = load_row(injected, seed)
        compared = {
            "emission_sequence": (left["emission_sequence"], right["emission_sequence"]),
            "refusal_sequence": (left["refusal_sequence"], right["refusal_sequence"]),
            "relay_counters": (left["relay_counters"], right["relay_counters"]),
            "spent_bytes": (left["spent_bytes"], right["spent_bytes"]),
            "airtime_tap_total": (left["airtime_tap_total"], right["airtime_tap_total"]),
            "on_time": (left["on_time"], right["on_time"]),
            "completed": (left["completed"], right["completed"]),
            "expired": (left["expired"], right["expired"]),
            "discovered": (left["discovered"], right["discovered"]),
            "behaviour_sha256": (left["behaviour_sha256"], right["behaviour_sha256"]),
            "ack_ledger": (left["ack_ledger"], right["ack_ledger"]),
        }
        differences = {key: {"reference": a, "injected": b}
                       for key, (a, b) in compared.items() if a != b}
        # The two hold paths must have spent nothing at all; the send paths must have spent
        # the same money as each other (already implied by the field comparisons).
        hold_zero = (left["spent_bytes"] == 0 and left["airtime_tap_total"] == 0
                     and not any(left["relay_counters"].get(key) for key in
                                 ("emissions", "transmission_attempts")))
        checks.append({
            "pair": [reference, injected], "label": label,
            "match": not differences, "differences": differences,
            "hold_paths_spent_zero": hold_zero if reference == "R_hold" else None,
            "allowed_differences": list(common.ALLOWED_METADATA_DIFFERENCES),
            "note": "artefact paths / run labels are the only pre-listed differences and are "
                    "not part of any compared field",
        })
    return {"seed": seed, "gate": "B1", "checks": checks,
            "passed": all(check["match"] for check in checks)}


# ------------------------------------------------------------------ B2 动作与 mask -------------
def gate_b2_actions(seed: int) -> dict[str, Any]:
    """The tape's requests must be exactly what the executor honoured, in order, at the price quoted."""
    row = load_row("P_init", seed)
    tape = load_tape("P_init", seed)
    transcript = load_transcript("P_init", seed)
    sends = [record for record in transcript if record["kind"] == "send"]
    refusals = [record for record in transcript if record["kind"] == "refused"]
    sends_by_step: dict[int, list[dict[str, Any]]] = {}
    for record in sends:
        sends_by_step.setdefault(int(record["step"]), []).append(record)
    refusals_by_step: dict[int, list[dict[str, Any]]] = {}
    for record in refusals:
        refusals_by_step.setdefault(int(record["step"]), []).append(record)

    mismatches: list[dict[str, Any]] = []
    for entry in tape:
        step = int(entry["step"])
        requested = [target for target in entry["targets"] if entry["requested"][target]]
        executed = [record for record in sends_by_step.get(step, [])]
        refused = [record for record in refusals_by_step.get(step, [])]
        honoured = [str(record["target"]) for record in executed]
        turned_away = [str(record["target"]) for record in refused]
        if requested != honoured + turned_away:
            mismatches.append({"step": step, "kind": "requested_vs_executor",
                               "requested": requested, "executed": honoured,
                               "refused": turned_away})
            continue
        # The price the executor charged must be the relay's own quote for that candidate.
        quoted = {target: int(charge) for target, charge in entry["attempt_charge_bytes"].items()}
        for record in executed + refused:
            target = str(record["target"])
            if target in quoted and int(record["cost_bytes"]) != quoted[target]:
                mismatches.append({"step": step, "kind": "quote_mismatch", "target": target,
                                   "transcript_cost": int(record["cost_bytes"]),
                                   "relay_quote": quoted[target]})
        # A masked candidate must never have been requested.
        for target in entry["masked"]:
            if entry["requested"].get(target):
                mismatches.append({"step": step, "kind": "masked_was_requested", "target": target})
        # Consistency of the shadow walk: recompute the credit from the tape's own numbers.
        if entry["sender_state"]["budget_bytes_per_episode"] is not None:
            shadow_after = float(entry["shadow_budget_after"])
            expected_after = (float(entry["sender_state"]["remaining_budget_bytes"])
                              - sum(quoted[target] for target in requested))
            if abs(shadow_after - expected_after) > 1e-6:
                mismatches.append({"step": step, "kind": "shadow_walk_mismatch",
                                   "shadow_after": shadow_after, "expected_after": expected_after})

    controller = StageBController(common.ACTOR_INIT_SEED)
    rebuilt_hash = controller.parameter_summary()["parameter_sha256"]
    after = row.get("controller_parameters_after") or {}
    weight_identity = rebuilt_hash == after.get("parameter_sha256")

    log_prob_gaps: list[float] = []
    dimension_ok = True
    for entry in tape:
        matrix = torch.as_tensor(np.asarray(entry["features"], dtype=np.float32))
        if matrix.shape[1] != cm.ACTOR_FEATURE_DIM:
            dimension_ok = False
        logits, _ = controller.forward(matrix)
        actions = torch.tensor([entry["actions"][t] for t in entry["targets"]], dtype=torch.long)
        allowed = torch.tensor([[True, True] if t in entry["actionable"] else [True, False]
                                for t in entry["targets"]], dtype=torch.bool)
        ll, _ = cm.masked_log_likelihood(logits, actions, allowed)
        log_prob_gaps.append(abs(float(ll) - float(entry["log_prob"])))

    forced_total = sum(len(entry["forced"]) for entry in tape)
    return {
        "seed": seed, "gate": "B2",
        "decision_steps": len(tape),
        "request_execution_mismatches": mismatches,
        "max_log_prob_gap": max(log_prob_gaps) if log_prob_gaps else 0.0,
        "log_prob_tolerance": common.LOG_PROB_TOLERANCE,
        "weight_identity_rebuilt_matches_run": weight_identity,
        "actor_feature_dim_ok": dimension_ok,
        "forced_total": forced_total,
        "optimizer_steps": row.get("optimizer_steps"),
        "passed": (not mismatches and (not log_prob_gaps or max(log_prob_gaps) <= common.LOG_PROB_TOLERANCE)
                   and weight_identity and dimension_ok and forced_total == 0
                   and row.get("optimizer_steps") == 0),
    }


# ------------------------------------------------------------------ B3 费用与任务闭合 -------------
def gate_b3_closure(seed: int) -> dict[str, Any]:
    """Money and tasks must close onto the executor's own ledger and the lifecycle metric."""
    checks = []
    reference = load_row("R_send", seed)
    for path_name in ("I_hold", "P_init", "P_repeat"):
        row = load_row(path_name, seed)
        transcript = load_transcript(path_name, seed)
        sends = [record for record in transcript if record["kind"] == "send"]
        transcript_bytes = sum(int(record["cost_bytes"]) for record in sends)
        stream_bytes = sum(int(entry["airtime_bytes"])
                           for entry in row["cost_stream"].values())
        records = load_lifecycle(path_name, seed)
        metrics = window_metrics(records, row["steps"])
        task_total = sum(float(value) for value in row["task_stream"].values())
        compared = {
            "spent_bytes": (row["spent_bytes"], int(row["relay_counters"]["airtime_bytes"])),
            "transcript_bytes": (transcript_bytes, row["spent_bytes"]),
            "tap_stream_bytes": (stream_bytes, row["spent_bytes"]),
            "emissions": (len(sends), int(row["relay_counters"]["emissions"])),
            "task_stream_total": (task_total, row["on_time"]),
            "lifecycle_on_time": (metrics.on_time, row["on_time"]),
            "airtime_counter": (int(row["relay_counters"]["airtime_bytes"]),
                                sum(int(r["cost_bytes"]) for r in sends)),
            "normalised_cost": (round(row["spent_bytes"] / common.B0_TEST, 9),
                                round(transcript_bytes / common.B0_TEST, 9)),
        }
        differences = {key: {"row": a, "check": b} for key, (a, b) in compared.items() if a != b}
        # Per-packet pricing: every emission is billed payload x addressed subscribers, once.
        pricing_bad = [record for record in sends
                       if int(record["cost_bytes"])
                       != int(record["payload_bytes"]) * int(record["subscribers"])]
        keys = [(int(record["step"]), str(record["target"])) for record in sends]
        double_billed = len(keys) != len(set(keys))
        g_hat = (int(reference["on_time"]) - int(row["on_time"])) - common.DELTA_TEST
        checks.append({
            "path": path_name, "match": not differences, "differences": differences,
            "send_pricing_consistent": not pricing_bad,
            "double_billed_emissions": double_billed,
            "g_hat": g_hat, "y_reference_on_time": reference["on_time"],
            "y_policy_on_time": row["on_time"],
            "hypothetical_lambda_next": cm.project_nonnegative(
                common.LAMBDA_TEST + common.ETA_TEST * g_hat),
            "dual_note": "hypothetical only: computed for the record, never written back and "
                         "never read by any later decision",
        })
    return {"seed": seed, "gate": "B3", "checks": checks,
            "passed": all(check["match"] and check["send_pricing_consistent"]
                          and not check["double_billed_emissions"] for check in checks)}


def gate_b3_math(seed: int) -> dict[str, Any]:
    """GAE, critic readings and the combined advantage, recomputed from the tape and streams."""
    row = load_row("P_init", seed)
    tape = load_tape("P_init", seed)
    decision_steps = sorted(int(entry["step"]) for entry in tape)
    cost_stream = {int(step): float(entry["airtime_bytes"])
                   for step, entry in row["cost_stream"].items()}
    task_stream = {int(step): float(value) for step, value in row["task_stream"].items()}
    import sb_collector as sbc

    bundled_cost = sbc.bundle_rewards(cost_stream, decision_steps, int(row["steps"]))
    bundled_task = sbc.bundle_rewards(task_stream, decision_steps, int(row["steps"]))
    rewards_cost = torch.tensor([-bundled_cost[s] / common.B0_TEST for s in decision_steps],
                                dtype=torch.float64)
    rewards_task = torch.tensor([bundled_task[s] for s in decision_steps], dtype=torch.float64)
    values_task = torch.tensor([float(entry["value"]) for entry in tape], dtype=torch.float64)
    values_cost = torch.tensor([float(entry["cost_value"]) for entry in tape], dtype=torch.float64)
    terminal = torch.zeros(len(decision_steps), dtype=torch.bool)
    terminal[-1] = True
    advantages_task, _ = cm.gae(rewards_task, values_task, gamma=common.GAMMA,
                                gae_lambda=common.GAE_LAMBDA, terminal=terminal,
                                bootstrap=torch.zeros(len(decision_steps), dtype=torch.float64))
    advantages_cost, _ = cm.gae(rewards_cost, values_cost, gamma=common.GAMMA,
                                gae_lambda=common.GAE_LAMBDA, terminal=terminal,
                                bootstrap=torch.zeros(len(decision_steps), dtype=torch.float64))
    combined = cm.actor_advantage(advantages_task, advantages_cost, common.LAMBDA_TEST)
    logged = row.get("episode_math") or {}
    gaps = {
        "advantages_task": _max_gap(advantages_task.tolist(),
                                    logged.get("advantages_task", [])),
        "advantages_cost": _max_gap(advantages_cost.tolist(),
                                    logged.get("advantages_cost", [])),
        "combined_advantage": _max_gap(combined.tolist(),
                                       logged.get("combined_advantage", [])),
        "sum_rewards_task": abs(float(rewards_task.sum())
                                - float(logged.get("sum_rewards_task", 0.0))),
        "sum_rewards_cost": abs(float(rewards_cost.sum())
                                - float(logged.get("sum_rewards_cost", 0.0))),
    }
    return {
        "seed": seed, "gate": "B3-math", "max_gaps": gaps,
        "g_hat_check": {
            "y_reference": load_row("R_send", seed)["on_time"],
            "y_policy": row["on_time"], "delta_test": common.DELTA_TEST,
            "g_hat": int(load_row("R_send", seed)["on_time"]) - int(row["on_time"]) - common.DELTA_TEST,
        },
        "passed": max(gaps.values()) <= 1e-6,
    }


def _max_gap(recomputed: list[float], logged: list[float]) -> float:
    if len(recomputed) != len(logged):
        return float("inf")
    return max((abs(float(a) - float(b)) for a, b in zip(recomputed, logged)), default=0.0)


# ------------------------------------------------------------------ B4 重复性与权重 -------------
def gate_b4_repeatability(seed: int) -> dict[str, Any]:
    """Same seed, parameters, configuration and RNG: identical money, tape and task outcome."""
    first = load_row("P_init", seed)
    second = load_row("P_repeat", seed)
    tape_first = load_tape("P_init", seed)
    tape_second = load_tape("P_repeat", seed)
    compared = {
        "emission_sequence": (first["emission_sequence"], second["emission_sequence"]),
        "refusal_sequence": (first["refusal_sequence"], second["refusal_sequence"]),
        "cost_stream": (first["cost_stream"], second["cost_stream"]),
        "task_stream": (first["task_stream"], second["task_stream"]),
        "on_time": (first["on_time"], second["on_time"]),
        "expired": (first["expired"], second["expired"]),
        "spent_bytes": (first["spent_bytes"], second["spent_bytes"]),
        "behaviour_sha256": (first["behaviour_sha256"], second["behaviour_sha256"]),
    }
    differences = {key: {"P_init": a, "P_repeat": b} for key, (a, b) in compared.items() if a != b}
    tape_fields = ("features", "actions", "log_prob", "masked", "actionable", "forced",
                   "value", "cost_value", "attempt_charge_bytes")
    tape_equal = len(tape_first) == len(tape_second) and all(
        all(entry_a.get(field) == entry_b.get(field) for field in tape_fields)
        for entry_a, entry_b in zip(tape_first, tape_second))
    weights = {
        "P_init_before_after_equal": (
            (first.get("policy_identity") or {}).get("controller_parameters_before", {}).get(
                "parameter_sha256")
            == (first.get("controller_parameters_after") or {}).get("parameter_sha256")),
        "P_repeat_before_after_equal": (
            (second.get("policy_identity") or {}).get("controller_parameters_before", {}).get(
                "parameter_sha256")
            == (second.get("controller_parameters_after") or {}).get("parameter_sha256")),
        "P_init_P_repeat_after_equal": (
            (first.get("controller_parameters_after") or {}).get("parameter_sha256")
            == (second.get("controller_parameters_after") or {}).get("parameter_sha256")),
    }
    return {
        "seed": seed, "gate": "B4",
        "field_differences": differences, "tape_identical": tape_equal, "weights": weights,
        "optimizer_steps": [first.get("optimizer_steps"), second.get("optimizer_steps")],
        "passed": (not differences and tape_equal and all(weights.values())
                   and first.get("optimizer_steps") == 0 and second.get("optimizer_steps") == 0),
    }


def run_seed_gates(seed: int) -> dict[str, Any]:
    gates = {
        "B1": gate_b1_equivalence(seed),
        "B2": gate_b2_actions(seed),
        "B3": gate_b3_closure(seed),
        "B3_math": gate_b3_math(seed),
        "B4": gate_b4_repeatability(seed),
    }
    return {"seed": seed, "gates": gates,
            "all_passed": all(gate["passed"] for gate in gates.values())}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Stage B gates, recomputed from artefacts")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    seeds = [args.seed] if args.seed else list(common.SEEDS)
    payload = {"seeds": [run_seed_gates(seed) for seed in seeds]}
    payload["all_passed"] = all(entry["all_passed"] for entry in payload["seeds"])
    common.write_json("GATE_RECHECK.json", payload)
    print(json.dumps({"seeds": [entry["seed"] for entry in payload["seeds"]],
                      "all_passed": payload["all_passed"],
                      "per_seed": {entry["seed"]: {name: gate["passed"] for name, gate in
                                                   entry["gates"].items()}
                                   for entry in payload["seeds"]}}, ensure_ascii=False))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
