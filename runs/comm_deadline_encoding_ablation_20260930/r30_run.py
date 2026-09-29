"""ROUND30 §4.4/§7 运行器：immediate reference、训练 rollout、冻结评估、late 规则臂。

与 ROUND24 v2_run 的关系（逐项登记）：
- `_run_episode` 是 `v2_run._run_episode` 的逐字复制，仅两处 seam 不同：产物路径走本轮
  `r30_common.cell_paths`（不写入 ROUND24 目录）、记账走本轮并行安全台账；世界配置继承
  （`stage_b_inherited_config`）、transport 断言、taps、闭合断言、tape 落盘全部原样。
- 配置继承、规则臂定义（`v2_rules.build_rule`）、checkpoint 文件哈希核验都复用 ROUND24
  已核验件，不重新定义。
- 训练 rollout 与评估都经 `r30_encoding` 的作用域 collector 选择编码臂；训练/采样/PPO 重算/
  评估因此必然读到同一臂的编码语义。

每个 cell 先记账再构造 driver；失败计入账本、不返还额度。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import r30_common as common
import r30_encoding as enc
import v2_rules as rules
from integrated_sar.comm_learning import LearnedSendPolicy
from integrated_sar.contracts import TaskStatus
from integrated_sar.joint_driver import RunConfiguration, build_joint_run
from integrated_sar.lifecycle_metrics import load_records, window_metrics
from v2_run import (
    COST_KEYS,
    EXPECTED_ACK_ZERO_KEYS,
    _assert_transport,
    _behaviour_sha256,
    _install_taps,
    eval_action_seed,
    action_seed,
)


def run_episode(label: str, kind: str, scenario_seed: int, *, policy: str,
                attach: Any = None, attempt: int = 1, controller: Any = None,
                collector: Any = None, action_rng_seed: int | None = None,
                tag: str = "") -> dict[str, Any]:
    """One episode: the archived ``v2_run._run_episode`` body with this round's paths/ledger.

    Registered seam diff: ``common.cell_paths`` (this round's directories) and the ledger
    reserve/record calls go through this round's parallel-safe ledger; every other line is the
    archived implementation, including the transport assertion, the per-step taps, the
    reward/ledger closure assertions and the tape write.
    """
    fields, substituted = common.v2c.stage_b_inherited_config(scenario_seed)
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


def run_reference(scenario_seed: int, *, attempt: int = 1) -> dict[str, Any]:
    """One cached immediate reference run (native policy; counted in the 1040)."""
    ledger = common.LaunchLedger()
    label = f"ref_s{scenario_seed}"
    ledger.reserve("normal", label, scenario_seed, attempt)
    try:
        row = run_episode(label, "reference", scenario_seed, policy="immediate")
    except Exception as error:
        ledger.record(label=label, seed=scenario_seed, attempt=attempt, ok=False,
                      detail=f"{type(error).__name__}: {error}")
        raise
    _booked(label, "reference", row, ledger)
    return row


def run_training_episode(controller: Any, *, arm: str, training_seed: int, episode_index: int,
                         scenario_seed: int, lam: float) -> dict[str, Any]:
    """One training rollout on the bound encoder arm: sampling collector, per-episode action RNG.

    No weight is touched here; the PPO update happens between episodes (r30_train reuses the
    frozen v2_trainer update).
    """
    ledger = common.LaunchLedger()
    label = f"train_{arm}_s{training_seed}_e{episode_index}"
    ledger.reserve("normal", label, scenario_seed, episode_index)
    try:
        collector = enc.scoped_training_collector(controller, arm)
        row = run_episode(label, "train", scenario_seed, policy="learned",
                          attach=collector.policy(), attempt=episode_index,
                          controller=controller, collector=collector,
                          action_rng_seed=action_seed(training_seed, episode_index),
                          tag=f"train_{arm}_s{training_seed}_e{episode_index}")
    except Exception as error:
        ledger.record(label=label, seed=scenario_seed, attempt=episode_index, ok=False,
                      detail=f"{type(error).__name__}: {error}")
        raise
    row["training_seed"] = int(training_seed)
    row["episode_index"] = int(episode_index)
    row["arm"] = str(arm)
    row["feature_layout"] = enc.ENCODERS[arm]["layout"]
    row["lambda_at_rollout"] = float(lam)
    row["lambda"] = float(lam)   # the trainer's contract key (frozen for the whole batch)
    _booked(label, "train", row, ledger)
    # The live objects are attached after the JSON row is written: they feed the PPO update
    # and must never be serialised into the artefact.
    row["collector"] = collector
    row["controller"] = controller
    return row


def run_eval(controller: Any, *, arm: str, training_seed: int, timepoint: str, mode: str,
             scenario_seed: int) -> dict[str, Any]:
    """One dev-evaluation rollout in a registered mode: greedy (argmax) or sampled.

    ``sampled`` is the training masked random policy itself (same collector class and
    parameters as a training rollout); ``greedy`` is only another explicitly named decision
    rule. Weights are frozen, no training happens here, and the action stream comes from the
    registered evaluation seed, once per scene per mode.
    """
    if mode not in ("greedy", "sampled"):
        raise ValueError(f"unknown evaluation mode {mode!r}")
    ledger = common.LaunchLedger()
    block = f"{timepoint}_{mode}"
    label = f"eval_{arm}_s{training_seed}_{block}_s{scenario_seed}"
    ledger.reserve("normal", label, scenario_seed, 0)
    try:
        collector = enc.scoped_eval_collector(controller, arm, greedy=(mode == "greedy"))
        row = run_episode(label, "eval", scenario_seed, policy="learned",
                          attach=collector.policy(), attempt=0,
                          controller=controller, collector=collector,
                          action_rng_seed=eval_action_seed(training_seed, scenario_seed, mode),
                          tag=f"eval_{arm}_s{training_seed}_{block}")
    except Exception as error:
        ledger.record(label=label, seed=scenario_seed, attempt=0, ok=False,
                      detail=f"{type(error).__name__}: {error}")
        raise
    row["training_seed"] = int(training_seed)
    row["timepoint"] = str(timepoint)
    row["eval_mode"] = str(mode)
    row["eval_block"] = block
    row["arm"] = str(arm)
    row["feature_layout"] = enc.ENCODERS[arm]["layout"]
    _booked(label, "eval", row, ledger)
    return row


def run_rule(arm_name: str, scenario_seed: int, *, attempt: int = 1) -> dict[str, Any]:
    """One rule arm on one scenario (native or injected rule policy; counted in the 1040)."""
    ledger = common.LaunchLedger()
    label = f"rule_{arm_name}_s{scenario_seed}"
    ledger.reserve("normal", label, scenario_seed, attempt)
    try:
        if arm_name == common.REFERENCE_ARM:
            row = run_episode(label, "rule", scenario_seed, policy="immediate", tag=arm_name)
        else:
            policy = rules.build_rule(arm_name, scenario_seed)
            row = run_episode(label, "rule", scenario_seed, policy="learned", attach=policy,
                              tag=arm_name)
            row["rule_identity"] = rules.rule_identity(arm_name)
            row["rule_identity"]["built_parameters"] = dict(policy.parameters())
    except Exception as error:
        ledger.record(label=label, seed=scenario_seed, attempt=attempt, ok=False,
                      detail=f"{type(error).__name__}: {error}")
        raise
    _booked(label, "rule", row, ledger)
    return row
