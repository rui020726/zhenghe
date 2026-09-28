"""ROUND25 cell 运行器：三臂各一格，先记账再构造 driver，逐步成本/任务 tap 与闭合断言。

与 ROUND24 v2_run 的关系：_run_episode 的接缝（transport 断言、tap、闭合校验、artefact 字段）
逐项继承同一结构，仅公共件换成 r25_common、命名换成 r25_ 前缀；学习臂只跑原 greedy 解码
（BudgetMaskedCollector(greedy=True, budget_mask=True)），不混 sampled 路径。
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import r25_common as common
import r25_rules as rules
from integrated_sar.comm_learning import LearnedSendPolicy  # noqa: F401  (policy seam type)
from integrated_sar.contracts import TaskStatus
from integrated_sar.joint_driver import RunConfiguration, build_joint_run
from integrated_sar.lifecycle_metrics import load_records, window_metrics

EXPECTED_ACK_ZERO_KEYS = (
    "ack_emitted", "ack_attempts", "ack_attempt_bytes", "ack_delivered", "ack_lost",
    "ack_frames_advanced", "ack_frames_reaffirmed",
)
COST_KEYS = ("airtime_bytes", "transmission_attempts", "emissions", "first_emissions",
             "refresh_emissions", "packets_lost", "packets_delivered", "deliveries_applied",
             "budget_refused_sends", "capacity_refused_sends", "emissions_suppressed_terminal")

_CODE_IDENTITY: dict[str, Any] | None = None


def code_identity() -> dict[str, Any]:
    global _CODE_IDENTITY
    if _CODE_IDENTITY is None:
        _CODE_IDENTITY = common.code_identity()
    return _CODE_IDENTITY


def eval_action_seed(training_seed: int, scenario_seed: int, mode: str) -> int:
    """The per-evaluation action RNG seed: stable sha256 of the inherited ROUND24 label.

    Greedy blocks draw no samples, but the stream is seeded identically to the ROUND24
    final_greedy block so nothing else can drift; it never touches the environment or channel
    RNG (those come from the scenario seed inside the driver).
    """
    return int(hashlib.sha256(
        f"{common.EVAL_RNG_LABEL}:{int(training_seed)}:{int(scenario_seed)}:{mode}".encode(
            "utf-8")).hexdigest()[:16], 16)


def load_candidate_controller() -> tuple[Any, dict[str, Any]]:
    """The fixed candidate: ROUND24 seed33 ep128 milestone under its ROUND24 freeze identity."""
    import v2_controller
    import v2_trainer as trainer

    freeze = common.round24_freeze_payload()
    path = common.LEARNED_CHECKPOINT
    controller, _optimizer, state = trainer.load_checkpoint(path, freeze=freeze)
    if int(state["training_seed"]) != common.CANDIDATE_TRAINING_SEED:
        raise RuntimeError(f"checkpoint holds training seed {state['training_seed']}, expected 33")
    if int(state["episodes_done"]) != common.CANDIDATE_EPISODES:
        raise RuntimeError(f"checkpoint holds {state['episodes_done']} episodes, expected 128")
    if int(state["optimizer_steps"]) <= 0:
        raise RuntimeError("the milestone carries no training history; refusing a fresh-weight arm")
    identity = {
        "arm": common.CANDIDATE_ARM,
        "checkpoint": common._rel(path),
        "checkpoint_sha256": common.sha256_file(path),
        "training_seed": int(state["training_seed"]),
        "episodes_done": int(state["episodes_done"]),
        "lambda_final": float(state["lambda"]),
        "mode": common.ARMS[common.CANDIDATE_ARM]["mode"],
        "decode": "SEND iff p(SEND) >= p(HOLD), ties to SEND (original frozen greedy decode)",
        "controller_parameter_summary": v2_controller.parameter_identity(controller),
    }
    return controller, identity


def _assert_transport(run: Any, label: str, expected_policy: str) -> dict[str, Any]:
    relay = run.task_adapter.comm_relay
    declared = relay.transport_parameters()
    if declared["policy"] != expected_policy:
        raise RuntimeError(f"{label}: relay policy {declared['policy']!r} != {expected_policy!r}")
    if declared["timing_mode"] != "policy_owned_timing":
        raise RuntimeError(f"{label}: timing mode {declared['timing_mode']!r}")
    if declared["bound_enforced_by_transport"] is not False:
        raise RuntimeError(f"{label}: the transport is enforcing a silence bound")
    if declared["ack_enabled"]:
        raise RuntimeError(f"{label}: the reverse channel must be structurally off")
    if float(declared["delivery_probability"]) != 0.7 or int(declared["delay_ticks"]) != 0:
        raise RuntimeError(f"{label}: the channel is not the frozen normal-lossy one")
    usage = relay.budget.usage()
    if int(usage["bytes_per_episode"] or 0) != common.STAGE_B_INHERITED_OVERRIDES["comm_budget_bytes"]:
        raise RuntimeError(f"{label}: the byte credit never reached the executor")
    if usage["sends_per_step"] is not None:
        raise RuntimeError(f"{label}: per-step capacity must stay None as archived")
    return declared


def _install_taps(run: Any) -> dict[str, Any]:
    """Per-step cost and task streams, keyed on the relay tick (Stage B convention)."""
    relay = run.task_adapter.comm_relay
    cost_stream: dict[int, dict[str, int]] = {}
    task_stream: dict[int, float] = {}
    credited: set[str] = set()

    original_tick = relay.tick

    def tapped_tick() -> tuple[str, ...]:
        before = {key: int(relay.counters[key]) for key in COST_KEYS}
        result = original_tick()
        after = {key: int(relay.counters[key]) for key in COST_KEYS}
        cost_stream[int(relay._tick)] = {key: after[key] - before[key] for key in COST_KEYS}
        return result

    relay.tick = tapped_tick
    environment = run.task_adapter.env
    original_step = environment.step

    def tapped_step(*args: Any, **kwargs: Any) -> Any:
        result = original_step(*args, **kwargs)
        gained = 0.0
        for snapshot in run.registry.snapshots():
            if (snapshot.status is TaskStatus.COMPLETED and not snapshot.expired
                    and snapshot.task_id not in credited):
                credited.add(snapshot.task_id)
                gained += 1.0
        if gained:
            step = int(relay._tick)
            task_stream[step] = task_stream.get(step, 0.0) + gained
        return result

    environment.step = tapped_step
    return {"relay": relay, "cost_stream": cost_stream, "task_stream": task_stream,
            "original_tick": original_tick, "original_step": original_step}


def _behaviour_sha256(records: list[dict[str, Any]]) -> str:
    from integrated_sar.info_probe import behaviour_view

    return hashlib.sha256(
        json.dumps([behaviour_view(record) for record in records], sort_keys=True,
                   default=str).encode("utf-8")).hexdigest()


def _on_time_goal_ids(records: list[dict[str, Any]], steps: int) -> dict[str, Any]:
    """Per-goal on-time identity from the lifecycle log, same rule as window_metrics.

    A target counts as on-time when it was discovered, has a completed progress record and
    finished at or before its deadline (detected_at + deadline_offset). IDs are returned so a
    net task tie cannot hide a target-identity swap.
    """
    discovered: dict[str, tuple[float, float]] = {}
    completed_at: dict[str, float] = {}
    log_complete = False
    for record in records:
        event_type = record.get("event_type")
        timestamp = float(record.get("timestamp") or 0.0)
        target = str(record.get("target_id"))
        if event_type == "detection" and timestamp <= steps:
            detail = record.get("detail") or {}
            discovered[target] = (timestamp, timestamp + float(detail.get("deadline_offset", 0.0)))
        elif event_type == "progress" and record.get("status") == "completed" and timestamp <= steps:
            completed_at.setdefault(target, timestamp)
        elif event_type == "final" and target == "__run__":
            log_complete = True
    on_time, expired, completed = [], [], []
    for target, (_detected_at, deadline) in discovered.items():
        done = completed_at.get(target)
        if done is None:
            if deadline <= steps:
                expired.append(target)
        else:
            completed.append(target)
            if done <= deadline:
                on_time.append(target)
    return {"on_time_goal_ids": sorted(on_time), "completed_goal_ids": sorted(completed),
            "expired_goal_ids": sorted(expired), "log_complete": bool(log_complete),
            "steps": int(steps)}


def _behaviour_config_sha256(fields: dict[str, Any]) -> str:
    """A path- and arm-independent digest of the research configuration of one episode.

    ``run_signature.config_hash`` includes the per-arm transcript path, so it cannot be
    compared across arms; this digest keeps only the behaviour fields (identity keys excluded)
    and is what the gate compares to prove the three arms ran the same world on the same seed.
    """
    identity = {"output", "comm_transcript", "info_probe", "search_probe", "decision_trace",
                "record_stream", "replay_stream", "seed", "comm_policy", "comm_policy_params"}
    payload = {key: value for key, value in sorted(fields.items()) if key not in identity}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     default=str).encode("utf-8")).hexdigest()


def _run_episode(label: str, arm: str, scenario_seed: int, *, policy: str,
                 attach: Any = None, attempt: int = 1, controller: Any = None,
                 collector: Any = None, action_rng_seed: int | None = None,
                 model_identity: dict[str, Any] | None = None,
                 rule_identity: dict[str, Any] | None = None,
                 parameters_before: dict[str, Any] | None = None) -> dict[str, Any]:
    fields, substituted = common.stage_b_inherited_config(scenario_seed)
    fields["comm_policy"] = policy
    paths = common.cell_paths(arm, scenario_seed, attempt)
    fields["output"] = str(paths["lifecycle"])
    fields["comm_transcript"] = str(paths["transcript"])
    configuration = RunConfiguration(**fields)
    configuration.validate()
    run = build_joint_run(configuration)
    declared = _assert_transport(run, f"{label} s{scenario_seed}", policy)
    if attach is not None:
        run.task_adapter.comm_relay.attach_policy(attach)

    taps = _install_taps(run)
    started = time.time()
    if action_rng_seed is not None:
        # The communication policy's own sampling stream, reseeded after construction and
        # before the first step; the environment and channel RNG come from the scenario seed
        # inside the driver, so this never touches them.
        torch.manual_seed(int(action_rng_seed))
    try:
        summary = run.run()
    finally:
        taps["relay"].tick = taps["original_tick"]
        run.task_adapter.env.step = taps["original_step"]

    relay = run.task_adapter.comm_relay
    records = load_records(Path(str(fields["output"])))
    metrics = window_metrics(records, configuration.steps)
    credited_total = int(sum(taps["task_stream"].values()))
    if metrics.on_time != credited_total:
        raise RuntimeError(
            f"{label} s{scenario_seed}: the tap credited {credited_total} on-time targets, the "
            f"lifecycle log says {metrics.on_time}; reward and metric have diverged")
    cost_total_tap = sum(row["airtime_bytes"] for row in taps["cost_stream"].values())
    spent = int(relay.budget.spent_bytes)
    if cost_total_tap != spent:
        raise RuntimeError(f"{label} s{scenario_seed}: tap {cost_total_tap} != ledger {spent}")

    transcript = list(common.iter_jsonl(Path(str(fields["comm_transcript"]))))
    sends = [record for record in transcript if record["kind"] == "send"]
    refusals = [record for record in transcript if record["kind"] == "refused"]
    counted = {
        "emissions": len(sends),
        "budget_refused_sends": sum(1 for r in refusals if r.get("reason") == "budget_exhausted"),
        "capacity_refused_sends": sum(1 for r in refusals if r.get("reason") == "capacity_step"),
    }
    for key, value in counted.items():
        if int(relay.counters[key]) != value:
            raise RuntimeError(
                f"{label} s{scenario_seed}: relay counted {relay.counters[key]} {key}, the "
                f"transcript holds {value}; two ledgers of one episode have diverged")
    ack_summary = {key: int(relay.counters.get(key, 0)) for key in EXPECTED_ACK_ZERO_KEYS}
    if any(ack_summary.values()):
        raise RuntimeError(f"{label} s{scenario_seed}: ACK is off but the reverse ledger moved")
    if int(summary.get("steps", 0)) != int(configuration.steps):
        raise RuntimeError(f"{label} s{scenario_seed}: ran {summary.get('steps')} steps")
    if spent < 0 or spent > int(fields["comm_budget_bytes"]):
        raise RuntimeError(f"{label} s{scenario_seed}: spent {spent} outside the credit")

    tape_summary = None
    parameters_after = None
    if collector is not None:
        tape_summary = (collector.tape_summary() if hasattr(collector, "tape_summary")
                        else {"timing": collector.timing(),
                              "note": "greedy eval collector: the base collector's timing, "
                                      "not the Stage B tape summary"})
        parameters_after = collector.controller.parameter_summary()
        paths["tape"].parent.mkdir(parents=True, exist_ok=True)
        with paths["tape"].open("w", encoding="utf-8", newline="\n") as handle:
            for tape in collector.records:
                handle.write(json.dumps(tape, ensure_ascii=False, sort_keys=True) + "\n")

    identity = code_identity()
    goal_identity = _on_time_goal_ids(records, configuration.steps)
    if len(goal_identity["on_time_goal_ids"]) != int(metrics.on_time):
        raise RuntimeError(
            f"{label} s{scenario_seed}: {len(goal_identity['on_time_goal_ids'])} on-time goal ids "
            f"vs window_metrics {metrics.on_time}; the identity extractor and the metric diverged")
    if not goal_identity["log_complete"]:
        raise RuntimeError(f"{label} s{scenario_seed}: the lifecycle log has no final record")
    return {
        "round": "25",
        "arm": arm,
        "mode": "greedy" if arm == common.CANDIDATE_ARM else (
            "native" if common.ARMS[arm]["kind"] == "reference" else "rule"),
        "scenario_seed": int(scenario_seed),
        "attempt": int(attempt),
        "policy": policy,
        "model_identity": model_identity,
        "rule_identity": rule_identity,
        "code_identity_freeze_sha256": identity["freeze_sha256"],
        "spent_bytes": spent, "budget_bytes": int(fields["comm_budget_bytes"]),
        "budget_used_fraction": round(spent / int(fields["comm_budget_bytes"]), 6),
        "airtime_tap_total": cost_total_tap,
        "emissions": len(sends),
        "budget_refused_sends": counted["budget_refused_sends"],
        "capacity_refused_sends": counted["capacity_refused_sends"],
        "refusal_bytes_billed": 0,
        "on_time": int(metrics.on_time), "completed": int(metrics.completed),
        "expired": int(metrics.expired), "discovered": int(metrics.discovered),
        "active": int(metrics.active), "unassigned": int(metrics.unassigned),
        "task_reward_total": credited_total,
        "b0_bytes": common.B0_BYTES,
        "normalised_cost_total": round(spent / common.B0_BYTES, 9),
        "goal_identity": goal_identity,
        "emission_sequence": [[int(r["step"]), str(r["target"]), int(r["revision"]),
                               int(r["cost_bytes"]), int(r["subscribers"]), bool(r["first"])]
                              for r in sends],
        "refusal_sequence": [[int(r["step"]), str(r["target"]), str(r["reason"])]
                             for r in refusals],
        "cost_stream": {str(step): row for step, row in sorted(taps["cost_stream"].items())},
        "task_stream": {str(step): value for step, value in sorted(taps["task_stream"].items())},
        "ack_ledger": ack_summary,
        "transport_parameters": declared,
        "relay_counters": {key: int(value) for key, value in relay.counters.items()},
        "behaviour_sha256": _behaviour_sha256(records),
        "config_hash": run.run_signature["config_hash"],
        "behaviour_config_sha256": _behaviour_config_sha256(fields),
        "run_signature_arm": run.run_signature["arm"],
        "run_signature_checkpoints": run.run_signature["checkpoints"],
        "run_signature_code_commit": run.run_signature["code_commit"],
        "steps": int(summary.get("steps", 0)),
        "seconds": round(time.time() - started, 2),
        "collector_summary": tape_summary,
        "controller_parameters_before": parameters_before,
        "controller_parameters_after": parameters_after,
        "optimizer_steps": 0,
        "substituted_fields": substituted,
        "artefacts": {"lifecycle": str(paths["lifecycle"]), "transcript": str(paths["transcript"]),
                      "tape": str(paths["tape"]), "row": str(paths["row"])},
    }


def _booked(label: str, arm: str, row: dict[str, Any], ledger: Any) -> None:
    ledger.record(label=label, seed=int(row["scenario_seed"]), attempt=int(row["attempt"]),
                  ok=True, detail=f"{arm}: on_time={row['on_time']} bytes={row['spent_bytes']}")
    common.ROWS.mkdir(parents=True, exist_ok=True)
    Path(row["artefacts"]["row"]).write_text(
        json.dumps(row, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def run_arm(arm: str, scenario_seed: int, *, attempt: int = 1, controller: Any = None,
            controller_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """One cell of the fixed three-arm panel; every launch is booked before the driver exists."""
    if arm not in common.ARMS:
        raise ValueError(f"unknown arm {arm!r}; registered: {sorted(common.ARMS)}")
    spec = common.ARMS[arm]
    ledger = common.LaunchLedger()
    label = f"{arm}_s{scenario_seed}"
    ledger.reserve("normal", arm, scenario_seed, attempt, label)
    try:
        if spec["kind"] == "reference":
            row = _run_episode(label, arm, scenario_seed, policy=str(spec["policy"]),
                               attempt=attempt,
                               rule_identity=rules.arm_identity(arm))
        elif spec["kind"] == "rule":
            policy = rules.build_arm_policy(arm, scenario_seed)
            row = _run_episode(label, arm, scenario_seed, policy="learned", attach=policy,
                               attempt=attempt,
                               rule_identity=rules.arm_identity(arm, policy))
        else:
            from budget_masked import BudgetMaskedCollector

            if controller is None:
                raise RuntimeError("the learned arm needs its frozen candidate controller")
            collector = BudgetMaskedCollector(controller, forced=lambda item: False, mask=False,
                                              greedy=(spec["mode"] == "greedy"), budget_mask=True)
            row = _run_episode(label, arm, scenario_seed, policy="learned",
                               attach=collector.policy(), attempt=attempt, controller=controller,
                               collector=collector,
                               action_rng_seed=eval_action_seed(int(spec["training_seed"]),
                                                                scenario_seed, str(spec["mode"])),
                               model_identity=None, parameters_before=controller_parameters)
            row["model_identity"] = {
                "checkpoint": common._rel(common.LEARNED_CHECKPOINT),
                "checkpoint_sha256": common.sha256_file(common.LEARNED_CHECKPOINT),
                "training_seed": int(spec["training_seed"]),
                "episodes_done": common.CANDIDATE_EPISODES,
                "mode": str(spec["mode"]),
                "decode": "SEND iff p(SEND) >= p(HOLD), ties to SEND (original frozen greedy)",
            }
            if (controller_parameters is not None and row["controller_parameters_after"] is not None
                    and row["controller_parameters_after"] != controller_parameters):
                raise RuntimeError(
                    f"{label}: controller parameters changed across the episode; the frozen "
                    "weights are not frozen")
    except Exception as error:
        ledger.record(label=label, seed=scenario_seed, attempt=attempt, ok=False,
                      detail=f"{type(error).__name__}: {error}")
        raise
    _booked(label, arm, row, ledger)
    return row
