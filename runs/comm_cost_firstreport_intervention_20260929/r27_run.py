"""ROUND27 cell 运行器：五条路径各一格，先记账再构造 driver，逐步成本/任务 tap 与闭合断言。

与 r25_run 的关系：接缝逐项继承同一结构（transport 断言、tap、闭合校验、artefact 字段），
仅公共件换成 r27_common、命名换成 r27_ 前缀、臂换成五条路径；干预记录来自 collector 的
first_report_overrides 记账与真实 transcript 的交叉核对。本 runner 不构造任何 optimizer，
学习路径 optimizer.step 恒为 0。
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

import r27_common as common
from integrated_sar.contracts import TaskStatus
from integrated_sar.joint_driver import RunConfiguration, build_joint_run
from integrated_sar.lifecycle_metrics import load_records, window_metrics
from r27_intervention import FirstReportAdvancer

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

    Identical label and formula to ROUND25/r25_run so the L path's decision stream is the same
    one the ROUND25 learned arm ran on. Greedy draws nothing anyway; the stream never touches
    the environment or channel RNG.
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
        "checkpoint": common._rel(path),
        "checkpoint_sha256": common.sha256_file(path),
        "training_seed": int(state["training_seed"]),
        "episodes_done": int(state["episodes_done"]),
        "lambda_final": float(state["lambda"]),
        "mode": "greedy",
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
    """Per-goal on-time identity from the lifecycle log, same rule as window_metrics."""
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
    """A path-independent digest of the research configuration of one episode."""
    identity = {"output", "comm_transcript", "info_probe", "search_probe", "decision_trace",
                "record_stream", "replay_stream", "seed", "comm_policy", "comm_policy_params"}
    payload = {key: value for key, value in sorted(fields.items()) if key not in identity}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     default=str).encode("utf-8")).hexdigest()


def _intervention_evidence(path_name: str, collector: Any, transcript: list[dict[str, Any]],
                           targets: tuple[str, ...]) -> dict[str, Any]:
    """Join the collector's opportunity bookkeeping with what the relay really put on the air.

    The tape says what the decision path offered and what was overridden; the transcript says
    what the transport actually emitted, lost and refused. Neither alone is the intervention.
    """
    opportunity_records: list[dict[str, Any]] = []
    for record in getattr(collector, "records", []) or []:
        block = record.get("first_report_overrides")
        if not block:
            continue
        for note in block.get("opportunities", []):
            if note.get("target") in targets:
                opportunity_records.append(note)
    sends = [r for r in transcript if r["kind"] == "send"]
    refusals = [r for r in transcript if r["kind"] == "refused"]
    per_target: dict[str, Any] = {}
    for target in targets:
        mine = [n for n in opportunity_records if n["target"] == target]
        fired = [n for n in mine if n.get("override_applied")]
        target_sends = [r for r in sends if r.get("target") == target]
        target_refusals = [r for r in refusals if r.get("target") == target]
        per_target[target] = {
            "opportunities_recorded": len(mine),
            "first_opportunity_step": min((n["step"] for n in mine), default=None),
            "override_applied": bool(fired),
            "override_steps": [n["step"] for n in fired],
            "override_details": fired,
            "non_override_outcomes": sorted({n.get("outcome") for n in mine
                                             if not n.get("override_applied")}),
            "real_sends": [{"step": int(r["step"]), "revision": int(r["revision"]),
                            "cost_bytes": int(r["cost_bytes"]), "packets": int(r["packets"]),
                            "packets_lost": int(r["packets_lost"]),
                            "subscribers": int(r["subscribers"]), "first": bool(r["first"]),
                            "send_reason": r.get("send_reason"),
                            "ready_step": int(r.get("ready_step", -1))}
                           for r in target_sends],
            "real_refusals": [{"step": int(r["step"]), "reason": r.get("reason"),
                               "cost_bytes": int(r.get("cost_bytes", 0)),
                               "send_reason": r.get("send_reason")}
                              for r in target_refusals],
            "deliveries": [{"step": int(r["step"]), "agent": r.get("agent"),
                            "accepted": bool(r.get("accepted")), "reason": r.get("reason"),
                            "sent_step": int(r.get("sent_step", -1))}
                           for r in transcript
                           if r["kind"] == "deliver" and r.get("target") == target],
        }
    return {
        "schema": "round27-intervention-evidence/1",
        "path": path_name,
        "targets": list(targets),
        "collector_opportunities_recorded": len(opportunity_records),
        "per_target": per_target,
        "interceptor_identity": (collector.intervention_identity()
                                 if hasattr(collector, "intervention_identity") else None),
    }


def _target_outcomes(records: list[dict[str, Any]], steps: int) -> dict[str, Any]:
    """Both named targets' whole-scene lifecycle outcome, one entry each."""
    discovered: dict[str, float] = {}
    deadlines: dict[str, float] = {}
    completed_at: dict[str, float] = {}
    final: dict[str, dict[str, Any]] = {}
    for record in records:
        target = str(record.get("target_id") or "")
        if not target.startswith("piloc-"):
            continue
        event_type = record.get("event_type")
        timestamp = float(record.get("timestamp") or 0.0)
        if event_type == "detection":
            detail = record.get("detail") or {}
            discovered[target] = timestamp
            deadlines[target] = timestamp + float(detail.get("deadline_offset", 0.0))
        elif event_type == "progress" and record.get("status") == "completed":
            completed_at.setdefault(target, timestamp)
        elif event_type == "final" and target != "__run__":
            final[target] = {"status": record.get("status"), "detail": record.get("detail")}
    out: dict[str, Any] = {}
    for target in sorted(set(discovered) | set(final)):
        done = completed_at.get(target)
        deadline = deadlines.get(target)
        out[target] = {
            "discovered_at": discovered.get(target),
            "deadline": deadline,
            "completed_at": done,
            "on_time": bool(done is not None and deadline is not None and done <= deadline),
            "expired": bool(done is None and deadline is not None and deadline <= steps),
            "final_record": final.get(target),
        }
    return out


def _run_episode(label: str, path_name: str, scenario_seed: int, *, policy: str,
                 attach: Any = None, attempt: int = 1, controller: Any = None,
                 collector: Any = None, action_rng_seed: int | None = None,
                 model_identity: dict[str, Any] | None = None,
                 parameters_before: dict[str, Any] | None = None,
                 override_targets: tuple[str, ...] = ()) -> dict[str, Any]:
    fields, substituted = common.stage_b_inherited_config(scenario_seed)
    fields["comm_policy"] = policy
    paths = common.cell_paths(path_name, attempt)
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
        "round": "27",
        "path": path_name,
        "mode": "greedy",
        "scenario_seed": int(scenario_seed),
        "attempt": int(attempt),
        "policy": policy,
        "model_identity": model_identity,
        "override_targets": list(override_targets),
        "intervention_identity": (collector.intervention_identity()
                                  if hasattr(collector, "intervention_identity") else
                                  {"note": "no interceptor installed on this path"}),
        "intervention_evidence": _intervention_evidence(path_name, collector, transcript,
                                                        override_targets),
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
        "target_outcomes": _target_outcomes(records, configuration.steps),
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


def _booked(label: str, row: dict[str, Any], ledger: Any) -> None:
    ledger.record(label=label, attempt=int(row["attempt"]), ok=True,
                  detail=f"{row['path']}: on_time={row['on_time']} bytes={row['spent_bytes']}")
    common.ROWS.mkdir(parents=True, exist_ok=True)
    Path(row["artefacts"]["row"]).write_text(
        json.dumps(row, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def run_path(path_name: str, *, attempt: int = 1, controller: Any = None,
             controller_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """One path run of the fixed panel; every launch is booked before the driver exists."""
    if path_name not in common.PATHS:
        raise ValueError(f"unknown path {path_name!r}; registered: {sorted(common.PATHS)}")
    spec = common.PATHS[path_name]
    ledger = common.LaunchLedger()
    label = f"{path_name}_s{common.SCENE}"
    ledger.reserve("normal", path_name, attempt, label)
    try:
        override_targets = tuple(spec.get("override_targets") or ())
        if spec.get("interceptor"):
            from budget_masked import BudgetMaskedCollector  # noqa: F401  (parent seam)

            if controller is None:
                raise RuntimeError("an interceptor path needs its frozen candidate controller")
            collector = FirstReportAdvancer(
                controller, targets=override_targets,
                enabled=bool(spec.get("override_enabled", True)))
            row = _run_episode(
                label, path_name, common.SCENE, policy="learned", attach=collector.policy(),
                attempt=attempt, controller=controller, collector=collector,
                action_rng_seed=eval_action_seed(common.CANDIDATE_TRAINING_SEED, common.SCENE,
                                                 "greedy"),
                model_identity=None, parameters_before=controller_parameters,
                override_targets=override_targets)
        else:
            if controller is None:
                raise RuntimeError("the learned path needs its frozen candidate controller")
            from budget_masked import BudgetMaskedCollector

            collector = BudgetMaskedCollector(controller, forced=lambda item: False, mask=False,
                                              greedy=True, budget_mask=True)
            row = _run_episode(
                label, path_name, common.SCENE, policy="learned", attach=collector.policy(),
                attempt=attempt, controller=controller, collector=collector,
                action_rng_seed=eval_action_seed(common.CANDIDATE_TRAINING_SEED, common.SCENE,
                                                 "greedy"),
                model_identity=None, parameters_before=controller_parameters,
                override_targets=override_targets)
        row["model_identity"] = {
            "checkpoint": common._rel(common.LEARNED_CHECKPOINT),
            "checkpoint_sha256": common.sha256_file(common.LEARNED_CHECKPOINT),
            "training_seed": common.CANDIDATE_TRAINING_SEED,
            "episodes_done": common.CANDIDATE_EPISODES,
            "mode": "greedy",
            "decode": "SEND iff p(SEND) >= p(HOLD), ties to SEND (original frozen greedy)",
        }
        if (controller_parameters is not None and row["controller_parameters_after"] is not None
                and row["controller_parameters_after"] != controller_parameters):
            raise RuntimeError(
                f"{label}: controller parameters changed across the episode; the frozen "
                "weights are not frozen")
    except Exception as error:
        ledger.record(label=label, attempt=attempt, ok=False,
                      detail=f"{type(error).__name__}: {error}")
        raise
    _booked(label, row, ledger)
    return row


def main() -> None:
    """CLI: run one path of the fixed panel, loading the frozen candidate once."""
    import sys

    if len(sys.argv) != 2 or sys.argv[1] not in common.PATH_ORDER:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} "
                         f"<{'|'.join(common.PATH_ORDER)}>")
    path_name = sys.argv[1]
    controller, identity = load_candidate_controller()
    parameters = controller.parameter_summary()
    row = run_path(path_name, controller=controller, controller_parameters=parameters)
    evidence = row["intervention_evidence"]
    print(json.dumps({
        "path": path_name, "scene": row["scenario_seed"], "attempt": row["attempt"],
        "on_time": row["on_time"], "expired": row["expired"], "spent_bytes": row["spent_bytes"],
        "emissions": row["emissions"], "seconds": row["seconds"],
        "goal_identity": row["goal_identity"],
        "target_outcomes": {target: {key: value for key, value in value.items()
                                     if key != "final_record"}
                            for target, value in row["target_outcomes"].items()},
        "intervention": {
            "identity": row["intervention_identity"],
            "per_target": {target: {
                "opportunities_recorded": entry["opportunities_recorded"],
                "override_applied": entry["override_applied"],
                "override_steps": entry["override_steps"],
                "real_sends": entry["real_sends"],
            } for target, entry in evidence["per_target"].items()},
        },
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
