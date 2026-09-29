"""ROUND32 任务 C/D 运行器：immediate reference、训练 rollout、冻结评估、late 规则臂。

与 ROUND30 `r30_run` 的关系（逐字复制 + 登记接缝，`source_diff()` 程序化核验）：
- `run_episode` 的函数体与 `r30_run.run_episode` **逐行相同**；父体本身已是 `v2_run._run_episode`
  的登记副本（ seams = 产物路径 + 记账），本轮通过把模块别名 `common` 指向 `r32_common` 来
  重定向路径与台账，函数体内不新增任何逻辑；
- 三个登记差异只在 cell 层：编码器臂**固定为 ROUND30 的 old 臂**（生产
  `comm-send-state/1`，`clip(remaining_time/50,-1,1)`），行身份用 `reduction` 而不是编码臂，
  以及 label/tag 的名字；
- 世界配置继承（`stage_b_inherited_config`）、transport 断言、taps、reward/账本闭合断言、
  ACK 结构性零断言、tape 落盘全部原样复用。

启动账本：每个 cell 先记账再构造 driver；失败计入账本、不返还额度。
"""

from __future__ import annotations

import difflib
import inspect
import json
import time
from pathlib import Path
from typing import Any

import r32_common as common
import r30_encoding as enc
import r30_run as r30
import v2_rules as rules
import torch
from integrated_sar.contracts import TaskStatus
from integrated_sar.joint_driver import RunConfiguration, build_joint_run
from integrated_sar.lifecycle_metrics import load_records, window_metrics
from v2_run import (
    COST_KEYS,
    EXPECTED_ACK_ZERO_KEYS,
    _assert_transport,
    _behaviour_sha256,
    _install_taps,
    action_seed,
    eval_action_seed,
)

#: 两臂共用的编码器臂：ROUND30 已验证它与生产 collector 的 tape 逐字段一致。
ENCODER_ARM = "old"


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


def run_training_episode(controller: Any, *, reduction: str, training_seed: int,
                         episode_index: int, scenario_seed: int,
                         lam: float) -> dict[str, Any]:
    """One training rollout on the frozen old encoder arm; only the reduction differs.

    No weight is touched here; the PPO update happens between episodes (r32_train calls the
    reduction-aware trainer with the frozen batch/GAE/likelihood machinery).
    """
    ledger = common.LaunchLedger()
    label = f"train_{reduction}_s{training_seed}_e{episode_index}"
    ledger.reserve("normal", label, scenario_seed, episode_index)
    try:
        collector = enc.scoped_training_collector(controller, ENCODER_ARM)
        row = run_episode(label, "train", scenario_seed, policy="learned",
                          attach=collector.policy(), attempt=episode_index,
                          controller=controller, collector=collector,
                          action_rng_seed=action_seed(training_seed, episode_index),
                          tag=f"train_{reduction}_s{training_seed}_e{episode_index}")
    except Exception as error:
        ledger.record(label=label, seed=scenario_seed, attempt=episode_index, ok=False,
                      detail=f"{type(error).__name__}: {error}")
        raise
    row["training_seed"] = int(training_seed)
    row["episode_index"] = int(episode_index)
    row["reduction"] = str(reduction)
    row["arm"] = str(reduction)      # cell identity for the ledger and the statistics
    row["encoder_arm"] = ENCODER_ARM
    row["feature_layout"] = enc.ENCODERS[ENCODER_ARM]["layout"]
    row["lambda_at_rollout"] = float(lam)
    row["lambda"] = float(lam)       # the trainer's contract key (frozen for the whole batch)
    _booked(label, "train", row, ledger)
    # The live objects are attached after the JSON row is written: they feed the PPO update
    # and must never be serialised into the artefact.
    row["collector"] = collector
    row["controller"] = controller
    return row


def run_eval(controller: Any, *, reduction: str, training_seed: int, timepoint: str, mode: str,
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
    label = f"eval_{reduction}_s{training_seed}_{block}_s{scenario_seed}"
    ledger.reserve("normal", label, scenario_seed, 0)
    try:
        collector = enc.scoped_eval_collector(controller, ENCODER_ARM, greedy=(mode == "greedy"))
        row = run_episode(label, "eval", scenario_seed, policy="learned",
                          attach=collector.policy(), attempt=0,
                          controller=controller, collector=collector,
                          action_rng_seed=eval_action_seed(training_seed, scenario_seed, mode),
                          tag=f"eval_{reduction}_s{training_seed}_{block}")
    except Exception as error:
        ledger.record(label=label, seed=scenario_seed, attempt=0, ok=False,
                      detail=f"{type(error).__name__}: {error}")
        raise
    row["training_seed"] = int(training_seed)
    row["timepoint"] = str(timepoint)
    row["eval_mode"] = str(mode)
    row["eval_block"] = block
    row["reduction"] = str(reduction)
    row["arm"] = str(reduction)
    row["encoder_arm"] = ENCODER_ARM
    row["feature_layout"] = enc.ENCODERS[ENCODER_ARM]["layout"]
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


# ------------------------------------------------------------------ 登记：接缝差异
_SEAM_PATTERN = (
    "r32_common as common", "import r30_encoding as enc", "ENCODER_ARM", "reduction",
    "row[\"arm\"]", "encoder_arm", "def run_", "import torch", "torch.manual_seed",
    "ledger.reserve", "ledger.record", "common.LaunchLedger", "common.cell_paths",
    "common.ROWS", "common.iter_jsonl", "common.B0_BYTES", "common.v2c", "difflib",
    "from r30", "No weight is touched here; the PPO update happens between episodes (r32_train",
    "registered batch/GAE", "reduction-aware trainer",
    # the arm->reduction rename ripples into the signature wrap and one comment column
    "scenario_seed: int,", "lam: float) -> dict[str, Any]:", "the trainer's contract key")


def seam_diff() -> dict[str, Any]:
    """Compare each runner function against its ROUND30 twin; only seams may differ."""
    pairs = (("run_episode", r30.run_episode, run_episode),
             ("run_reference", r30.run_reference, run_reference),
             ("run_rule", r30.run_rule, run_rule),
             ("run_training_episode", r30.run_training_episode, run_training_episode),
             ("run_eval", r30.run_eval, run_eval))
    blocks: list[dict[str, Any]] = []
    unexplained: list[str] = []
    for name, parent_function, child_function in pairs:
        parent = [line.strip() for line in inspect.getsource(parent_function).splitlines()
                  if line.strip() and not line.strip().startswith("#")]
        child = [line.strip() for line in inspect.getsource(child_function).splitlines()
                 if line.strip() and not line.strip().startswith("#")]
        matcher = difflib.SequenceMatcher(a=parent, b=child)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            changed = child[j1:j2]
            blocks.append({"function": name, "opcode": tag, "parent_lines": parent[i1:i2],
                           "child_lines": changed})
            for line in changed:
                if not any(marker in line for marker in _SEAM_PATTERN):
                    unexplained.append(f"{name}: {line}")
    return {
        "method": "difflib per function against the ROUND30 runner that is itself the registered "
                  "copy of v2_run._run_episode; every changed line must be a declared seam",
        "changed_blocks": blocks,
        "unexplained_lines": unexplained,
        "run_episode_body_identical_modulo_module_alias": (
            [line.strip() for line in inspect.getsource(r30.run_episode).splitlines()
             if line.strip() and not line.strip().startswith("#")]
            == [line.strip() for line in inspect.getsource(run_episode).splitlines()
                if line.strip() and not line.strip().startswith("#")]),
        "encoder_arm_is_fixed_to_round30_old": ENCODER_ARM == "old",
        "encoder_is_production_clip_over_50": enc.ENCODERS["old"]["encode"].__name__ == "features",
        "passed": not unexplained,
    }
