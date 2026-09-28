"""Stage C 运行器：immediate reference、五条规则、训练 rollout、冻结评估。

每个 cell 先记账再构造 driver；配置逐字段继承 D1 A3 的 normal-lossy 世界 + Stage B 冻结
覆盖（v2_common.stage_b_inherited_config），各臂只差 comm_policy 一个键。共用 relay 的
真实报价/预算/容量/合并/终态处理；逐步成本/任务 tap 与闭合断言与 Stage B 同一套结构。

训练 rollout 只推理+记账，权重更新全部发生在 episode 之间的 v2_trainer.ppo_update；
评估用 greedy（argmax）或 sampled（与训练同一 masked 随机策略）前向，冻结权重与 lambda。
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

import v2_common as common
import v2_rules as rules
from sb_collector import StageBCollector

from integrated_sar.comm_learning import LearnedSendPolicy
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


def action_seed(training_seed: int, episode_index: int) -> int:
    """The per-episode action RNG seed, derived from (training seed, episode index).

    Registered verbatim from the Stage C protocol (spec §1): each communication training seed
    gets an independent stream (the three seeds differ), the stream is per-episode reproducible,
    and the derivation is a plain sha256 so it never touches the environment or channel RNG
    (those come from the scenario seed).
    """
    return int(hashlib.sha256(f"sc-action:{int(training_seed)}:{int(episode_index)}".encode(
        "utf-8")).hexdigest()[:16], 16)


def eval_action_seed(training_seed: int, scenario_seed: int, mode: str) -> int:
    """The per-evaluation action RNG seed: stable sha256 of the ROUND24 label (spec §5).

    Fixed before the first launch and registered in the protocol: one seed per
    (training seed, scenario, mode), never Python's process-random ``hash``, so the stream is
    reproducible across processes and never disturbs the environment or channel RNG. Greedy
    blocks draw no samples, but the stream is seeded identically so nothing else can drift.
    """
    return int(hashlib.sha256(
        f"{common.EVAL_RNG_LABEL}:{int(training_seed)}:{int(scenario_seed)}:{mode}".encode(
            "utf-8")).hexdigest()[:16], 16)


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


def _run_episode(label: str, kind: str, scenario_seed: int, *, policy: str,
                 attach: Any = None, attempt: int = 1, controller: Any = None,
                 collector: Any = None, action_rng_seed: int | None = None,
                 tag: str = "") -> dict[str, Any]:
    fields, substituted = common.stage_b_inherited_config(scenario_seed)
    fields["comm_policy"] = policy
    paths = common.cell_paths(kind, scenario_seed, attempt, tag=tag)
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
        # before the first step: the environment and channel RNG come from the scenario seed
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

    return {
        "kind": kind, "label": label, "scenario_seed": int(scenario_seed),
        "attempt": int(attempt), "policy": policy,
        "spent_bytes": spent, "budget_bytes": int(fields["comm_budget_bytes"]),
        "budget_used_fraction": round(spent / int(fields["comm_budget_bytes"]), 6),
        "airtime_tap_total": cost_total_tap,
        "emissions": len(sends),
        "on_time": int(metrics.on_time), "completed": int(metrics.completed),
        "expired": int(metrics.expired), "discovered": int(metrics.discovered),
        "active": int(metrics.active), "unassigned": int(metrics.unassigned),
        "task_reward_total": credited_total,
        "b0_bytes": common.B0_BYTES,
        "normalised_cost_total": round(spent / common.B0_BYTES, 9),
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
        "steps": int(summary.get("steps", 0)),
        "seconds": round(time.time() - started, 2),
        "collector_summary": tape_summary,
        "controller_parameters_after": parameters_after,
        "optimizer_steps": 0,
        "substituted_fields": substituted,
        "artefacts": {"lifecycle": str(paths["lifecycle"]), "transcript": str(paths["transcript"]),
                      "tape": str(paths["tape"]), "row": str(paths["row"])},
    }


def _booked(label: str, kind: str, row: dict[str, Any], ledger: Any) -> None:
    ledger.record(label=label, seed=int(row["scenario_seed"]), attempt=int(row["attempt"]),
                  ok=True, detail=f"{label}: on_time={row['on_time']} bytes={row['spent_bytes']}")
    common.ROWS.mkdir(parents=True, exist_ok=True)
    Path(row["artefacts"]["row"]).write_text(
        json.dumps(row, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def run_reference(scenario_seed: int, *, attempt: int = 1, kind: str = "reference") -> dict[str, Any]:
    """One cached immediate reference run (native policy; counted in the 672)."""
    ledger = common.LaunchLedger()
    label = f"ref_s{scenario_seed}"
    ledger.reserve("normal", label, scenario_seed, attempt)
    try:
        row = _run_episode(label, kind, scenario_seed, policy="immediate")
    except Exception as error:
        ledger.record(label=label, seed=scenario_seed, attempt=attempt, ok=False,
                      detail=f"{type(error).__name__}: {error}")
        raise
    _booked(label, kind, row, ledger)
    return row


def run_rule(arm: str, scenario_seed: int, *, attempt: int = 1) -> dict[str, Any]:
    """One rule arm on one scenario (native or injected rule policy; counted in the 672)."""
    ledger = common.LaunchLedger()
    label = f"rule_{arm}_s{scenario_seed}"
    ledger.reserve("normal", label, scenario_seed, attempt)
    try:
        if arm == common.REFERENCE_ARM:
            row = _run_episode(label, "rule", scenario_seed, policy="immediate", tag=arm)
        else:
            policy = rules.build_rule(arm, scenario_seed)
            row = _run_episode(label, "rule", scenario_seed, policy="learned", attach=policy,
                               tag=arm)
            row["rule_identity"] = rules.rule_identity(arm)
            row["rule_identity"]["built_parameters"] = dict(policy.parameters())
    except Exception as error:
        ledger.record(label=label, seed=scenario_seed, attempt=attempt, ok=False,
                      detail=f"{type(error).__name__}: {error}")
        raise
    _booked(label, "rule", row, ledger)
    return row


def run_training_episode(controller: Any, *, training_seed: int, episode_index: int,
                         scenario_seed: int, lam: float) -> dict[str, Any]:
    """One training rollout: sampling collector, per-episode action RNG, tap streams, tape.

    No weight is touched here; the PPO update happens between episodes (v2_trainer).
    """
    from budget_masked import BudgetMaskedCollector

    ledger = common.LaunchLedger()
    label = f"train_s{training_seed}_e{episode_index}"
    ledger.reserve("normal", label, scenario_seed, episode_index)
    try:
        collector = StageBCollector(controller)
        row = _run_episode(label, "train", scenario_seed, policy="learned",
                           attach=collector.policy(), attempt=episode_index,
                           controller=controller, collector=collector,
                           action_rng_seed=action_seed(training_seed, episode_index),
                           tag=f"train_s{training_seed}_e{episode_index}")
    except Exception as error:
        ledger.record(label=label, seed=scenario_seed, attempt=episode_index, ok=False,
                      detail=f"{type(error).__name__}: {error}")
        raise
    row["training_seed"] = int(training_seed)
    row["episode_index"] = int(episode_index)
    row["lambda_at_rollout"] = float(lam)
    row["lambda"] = float(lam)   # the trainer's contract key (frozen for the whole batch)
    _booked(label, "train", row, ledger)
    # The live objects are attached after the JSON row is written: they feed the PPO update
    # and must never be serialised into the artefact.
    row["collector"] = collector
    row["controller"] = controller
    return row


def run_eval(controller: Any, *, training_seed: int, timepoint: str, mode: str,
             scenario_seed: int) -> dict[str, Any]:
    """One dev-evaluation rollout in a registered mode: greedy (argmax) or sampled.

    ``sampled`` is the training masked random policy itself (same collector class and
    parameters as a training rollout); ``greedy`` is only another explicitly named decision
    rule. Weights are frozen, no training happens here, and the action stream comes from the
    registered evaluation seed (spec §5), once per scene per mode.
    """
    from budget_masked import BudgetMaskedCollector

    if mode not in ("greedy", "sampled"):
        raise ValueError(f"unknown evaluation mode {mode!r}")
    ledger = common.LaunchLedger()
    block = f"{timepoint}_{mode}"
    label = f"eval_s{training_seed}_{block}_s{scenario_seed}"
    ledger.reserve("normal", label, scenario_seed, 0)
    try:
        collector = BudgetMaskedCollector(controller, forced=lambda item: False, mask=False,
                                          greedy=(mode == "greedy"), budget_mask=True)
        row = _run_episode(label, "eval", scenario_seed, policy="learned",
                           attach=collector.policy(), attempt=0,
                           controller=controller, collector=collector,
                           action_rng_seed=eval_action_seed(training_seed, scenario_seed, mode),
                           tag=f"eval_s{training_seed}_{block}")
    except Exception as error:
        ledger.record(label=label, seed=scenario_seed, attempt=0, ok=False,
                      detail=f"{type(error).__name__}: {error}")
        raise
    row["training_seed"] = int(training_seed)
    row["timepoint"] = str(timepoint)
    row["eval_mode"] = str(mode)
    row["eval_block"] = block
    _booked(label, "eval", row, ledger)
    return row
