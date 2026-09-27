"""Stage B episode runner：六条路径的真实接线、逐步成本/任务 tap 与闭合断言。

对应 STAGE_B_PROTOCOL.md §3/§4/§5。每个 cell 先记账再启动；技术失败也记账并保留产物。
本模块只做接线与记账，**不更新任何权重**：没有 optimizer，`.step()` 计数恒 0。
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

import sb_common as common
import sb_collector as sbc
from sb_controller import StageBController

from integrated_sar.contracts import TaskStatus
from integrated_sar.joint_driver import RunConfiguration, build_joint_run
from integrated_sar.lifecycle_metrics import load_records, window_metrics

PATH_POLICY = {"R_send": "immediate", "I_send": "learned", "R_hold": "always_hold",
               "I_hold": "learned", "P_init": "learned", "P_repeat": "learned"}
LEARNING_PATHS = ("P_init", "P_repeat")
EXPECTED_ACK_ZERO_KEYS = (
    "ack_emitted", "ack_attempts", "ack_attempt_bytes", "ack_delivered", "ack_lost",
    "ack_frames_advanced", "ack_frames_reaffirmed",
)


def _behaviour_sha256(records: list[dict[str, Any]]) -> str:
    """The established behaviour identity: paths and run_config stripped, then hashed."""
    from integrated_sar.info_probe import behaviour_view

    return hashlib.sha256(
        json.dumps([behaviour_view(record) for record in records], sort_keys=True,
                   default=str).encode("utf-8")
    ).hexdigest()


def _assert_transport(run: Any, path_name: str) -> dict[str, Any]:
    """The relay must be running this round's frozen transport, checked before the first step."""
    relay = run.task_adapter.comm_relay
    declared = relay.transport_parameters()
    expected_policy = PATH_POLICY[path_name]
    if declared["policy"] != expected_policy:
        raise RuntimeError(f"{path_name}: relay policy {declared['policy']!r} != {expected_policy!r}")
    if declared["timing_mode"] != "policy_owned_timing":
        raise RuntimeError(f"{path_name}: timing mode {declared['timing_mode']!r}")
    if declared["bound_enforced_by_transport"] is not False:
        raise RuntimeError(
            f"{path_name}: the transport is enforcing a silence bound it was told to leave "
            "to the arm")
    if declared["ack_enabled"]:
        raise RuntimeError(f"{path_name}: the reverse channel must be structurally off")
    if float(declared["delivery_probability"]) != 0.7:
        raise RuntimeError(f"{path_name}: delivery probability is not the frozen 0.7")
    if int(declared["delay_ticks"]) != 0:
        raise RuntimeError(f"{path_name}: delay is not the frozen 0")
    budget_usage = relay.budget.usage()
    if int(budget_usage["bytes_per_episode"] or 0) != common.STAGE_B_OVERRIDES["comm_budget_bytes"]:
        raise RuntimeError(f"{path_name}: the byte credit never reached the executor")
    if budget_usage["sends_per_step"] is not None:
        raise RuntimeError(f"{path_name}: per-step capacity must stay None as archived")
    return declared


def _attach_policy(run: Any, path_name: str) -> dict[str, Any]:
    """Install the path's decision-maker through the public seam, before the first tick."""
    relay = run.task_adapter.comm_relay
    identity: dict[str, Any] = {"path": path_name, "policy": PATH_POLICY[path_name],
                                "collector": None}
    from integrated_sar.comm_learning import LearnedSendPolicy

    if path_name == "I_send":
        relay.attach_policy(LearnedSendPolicy(sbc.immediate_equivalent_fn))
        identity["injected"] = {"decide": "immediate_equivalent_fn",
                                "predicate": "pending_event, ordered by transport_trim_key"}
    elif path_name == "I_hold":
        relay.attach_policy(LearnedSendPolicy(sbc.hold_equivalent_fn))
        identity["injected"] = {"decide": "hold_equivalent_fn", "predicate": "always empty"}
    elif path_name in LEARNING_PATHS:
        controller = StageBController()
        identity["controller_parameters_before"] = controller.parameter_summary()
        identity["collector"] = sbc.StageBCollector(controller)
        relay.attach_policy(identity["collector"].policy())
    return identity


def _install_taps(run: Any) -> dict[str, Any]:
    """Per-step cost and task reward streams, keyed on the relay's own tick number.

    The relay ticks once per joint step, before the world advances (task03.py:1193), so the
    tick number is the one time base the decision steps, the cost deltas and the booked
    completions all share.
    """
    relay = run.task_adapter.comm_relay
    cost_keys = ("airtime_bytes", "transmission_attempts", "emissions", "first_emissions",
                 "refresh_emissions", "packets_lost", "packets_delivered", "deliveries_applied",
                 "budget_refused_sends", "capacity_refused_sends", "emissions_suppressed_terminal")
    cost_stream: dict[int, dict[str, int]] = {}
    task_stream: dict[int, float] = {}
    credited: set[str] = set()

    original_tick = relay.tick

    def tapped_tick() -> tuple[str, ...]:
        before = {key: int(relay.counters[key]) for key in cost_keys}
        result = original_tick()
        after = {key: int(relay.counters[key]) for key in cost_keys}
        cost_stream[int(relay._tick)] = {key: after[key] - before[key] for key in cost_keys}
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
            "credited": credited, "original_tick": original_tick,
            "original_step": original_step}


def _bundle(stream: dict[int, float], decision_steps: list[int], horizon: int) -> dict[int, float]:
    return sbc.bundle_rewards(stream, decision_steps, horizon)


def _episode_math(collector: Any, taps: dict[str, Any], horizon: int) -> dict[str, Any]:
    """Form task/cost returns, critic predictions, GAE and the combined advantage.

    gamma=1 keeps the per-step reward stream undiscounted (episode-total semantics); the
    GAE return then mixes the residuals with ``gae_lambda``, so the closure that is checked
    against the executor's ledger is the *reward sum* per episode, not ``returns[0]``.
    """
    import cost_math as cm

    tapes = collector.records
    decision_steps = sorted(int(tape["step"]) for tape in tapes)
    if not decision_steps:
        return {"decision_steps": [],
                "note": "no decision step: the policy was never offered a candidate"}
    cost_stream = {int(step): float(row["airtime_bytes"])
                   for step, row in taps["cost_stream"].items()}
    task_stream = {int(step): float(value) for step, value in taps["task_stream"].items()}
    bundled_cost = _bundle(cost_stream, decision_steps, horizon)
    bundled_task = _bundle(task_stream, decision_steps, horizon)
    rewards_cost = torch.tensor([-bundled_cost[step] / common.B0_TEST for step in decision_steps],
                                dtype=torch.float64)
    rewards_task = torch.tensor([bundled_task[step] for step in decision_steps],
                                dtype=torch.float64)
    values_task = torch.tensor([float(tape["value"]) for tape in tapes], dtype=torch.float64)
    values_cost = torch.tensor([float(tape["cost_value"]) for tape in tapes], dtype=torch.float64)
    terminal = torch.zeros(len(decision_steps), dtype=torch.bool)
    terminal[-1] = True   # the configured horizon is the episode boundary (registered)
    bootstrap = torch.zeros(len(decision_steps), dtype=torch.float64)
    advantages_task, returns_task = cm.gae(rewards_task, values_task, gamma=common.GAMMA,
                                           gae_lambda=common.GAE_LAMBDA, terminal=terminal,
                                           bootstrap=bootstrap)
    advantages_cost, returns_cost = cm.gae(rewards_cost, values_cost, gamma=common.GAMMA,
                                           gae_lambda=common.GAE_LAMBDA, terminal=terminal,
                                           bootstrap=bootstrap)
    combined = cm.actor_advantage(advantages_task, advantages_cost, common.LAMBDA_TEST)
    return {
        "decision_steps": decision_steps,
        "rewards_cost_per_decision_step": [round(float(v), 9) for v in rewards_cost],
        "rewards_task_per_decision_step": [round(float(v), 9) for v in rewards_task],
        "values_task": [round(float(v), 9) for v in values_task],
        "values_cost": [round(float(v), 9) for v in values_cost],
        "advantages_task": [round(float(v), 9) for v in advantages_task],
        "advantages_cost": [round(float(v), 9) for v in advantages_cost],
        "combined_advantage": [round(float(v), 9) for v in combined],
        "returns_task_first": round(float(returns_task[0]), 9),
        "returns_cost_first": round(float(returns_cost[0]), 9),
        "sum_rewards_task": round(float(rewards_task.sum()), 9),
        "sum_rewards_cost": round(float(rewards_cost.sum()), 9),
        "gamma": common.GAMMA, "gae_lambda": common.GAE_LAMBDA,
        "b0_test_bytes": common.B0_TEST,
        "cost_normalisation": cm.CostNormalization(b0_bytes=common.B0_TEST).as_dict(),
        "convention": "rewards between decision steps bundle into the later decision step; "
                      "terminal at the configured horizon; gamma=1 keeps the reward stream "
                      "undiscounted, gae_lambda=0.95 only weights the advantage estimate -- "
                      "the episode closure is on the reward sums, not on returns[0]",
    }


def _run_episode(path_name: str, seed: int, attempt: int) -> dict[str, Any]:
    fields, substituted = common.stage_b_fields(path_name, seed, attempt)
    paths = common.cell_paths(path_name, seed, attempt)
    configuration = RunConfiguration(**fields)
    configuration.validate()
    run = build_joint_run(configuration)
    declared = _assert_transport(run, path_name)
    identity = _attach_policy(run, path_name)

    taps = _install_taps(run)
    started = time.time()
    # The sampling stream is reseeded after construction and before the first step, so the
    # two repeats of the learned path draw the same actions on the same world (protocol §4).
    torch.manual_seed(common.SAMPLING_SEED)
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
            f"{path_name} s{seed}: the tap credited {credited_total} on-time targets, the "
            f"lifecycle log says {metrics.on_time}; reward and metric have diverged"
        )

    cost_total_tap = sum(row["airtime_bytes"] for row in taps["cost_stream"].values())
    spent = int(relay.budget.spent_bytes)
    if cost_total_tap != spent:
        raise RuntimeError(
            f"{path_name} s{seed}: the per-step tap sums to {cost_total_tap} bytes, the "
            f"executor's ledger says {spent}"
        )

    collector = identity.get("collector")
    math_payload: dict[str, Any] = {}
    collector_summary: dict[str, Any] | None = None
    parameters_after: dict[str, Any] | None = None
    if collector is not None:
        math_payload = _episode_math(collector, taps, int(configuration.steps))
        collector_summary = collector.tape_summary()
        parameters_after = collector.controller.parameter_summary()
        if (parameters_after["parameter_sha256"]
                != identity["controller_parameters_before"]["parameter_sha256"]):
            raise RuntimeError(
                f"{path_name} s{seed}: controller parameters changed during inference")
        paths["tape"].parent.mkdir(parents=True, exist_ok=True)
        with paths["tape"].open("w", encoding="utf-8", newline="\n") as handle:
            for tape in collector.records:
                handle.write(json.dumps(tape, ensure_ascii=False, sort_keys=True) + "\n")
        identity["collector"] = None   # the object itself never enters the row

    transcript_path = Path(str(fields["comm_transcript"]))
    transcript = list(common.iter_jsonl(transcript_path))
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
                f"{path_name} s{seed}: the relay counted {relay.counters[key]} {key}, the "
                f"transcript holds {value}; two ledgers of one episode have diverged"
            )
    ack_summary = {key: int(relay.counters.get(key, 0)) for key in EXPECTED_ACK_ZERO_KEYS}
    if declared["ack_enabled"] is False and any(ack_summary.values()):
        raise RuntimeError(f"{path_name} s{seed}: ACK is off but the reverse ledger moved")

    if int(summary.get("steps", 0)) != int(configuration.steps):
        raise RuntimeError(
            f"{path_name} s{seed}: the episode ran {summary.get('steps')} steps, not "
            f"{configuration.steps}"
        )

    return {
        "round": "21", "phase": "Stage B integration, no training",
        "path": path_name, "policy": PATH_POLICY[path_name], "seed": int(seed),
        "attempt": int(attempt),
        "policy_identity": identity,
        "collector_summary": collector_summary,
        "controller_parameters_after": parameters_after,
        "optimizer_steps": 0,
        "setting_overrides": dict(common.STAGE_B_OVERRIDES),
        "substituted_fields": substituted,
        "spent_bytes": spent,
        "budget_bytes": int(common.STAGE_B_OVERRIDES["comm_budget_bytes"]),
        "budget_used_fraction": round(spent / common.STAGE_B_OVERRIDES["comm_budget_bytes"], 6),
        "airtime_tap_total": cost_total_tap,
        "emissions": len(sends),
        "on_time": int(metrics.on_time),
        "completed": int(metrics.completed),
        "expired": int(metrics.expired),
        "discovered": int(metrics.discovered),
        "active": int(metrics.active),
        "unassigned": int(metrics.unassigned),
        "task_reward_total": credited_total,
        "b0_test_bytes": common.B0_TEST,
        "normalised_cost_total": round(spent / common.B0_TEST, 9),
        "emission_sequence": [[int(r["step"]), str(r["target"]), int(r["revision"]),
                               int(r["cost_bytes"]), int(r["subscribers"]), bool(r["first"])]
                              for r in sends],
        "refusal_sequence": [[int(r["step"]), str(r["target"]), str(r["reason"])]
                             for r in refusals],
        "cost_stream": {str(step): row_ for step, row_ in sorted(taps["cost_stream"].items())},
        "task_stream": {str(step): value for step, value in sorted(taps["task_stream"].items())},
        "ack_ledger": ack_summary,
        "transport_parameters": declared,
        "relay_counters": {key: int(value) for key, value in relay.counters.items()},
        "behaviour_sha256": _behaviour_sha256(records),
        "config_hash": run.run_signature["config_hash"],
        "steps": int(summary.get("steps", 0)),
        "seconds": round(time.time() - started, 2),
        "episode_math": math_payload,
        "artefacts": {
            "lifecycle_log": str(paths["lifecycle"]),
            "transcript": str(paths["transcript"]),
            "tape": str(paths["tape"]),
            "row": str(paths["row"]),
        },
    }


def run_cell(path_name: str, seed: int, attempt: int = 1, *, kind: str = "normal") -> dict[str, Any]:
    """One Stage B cell: book the launch, run it, record the outcome (success or failure)."""
    if path_name not in PATH_POLICY:
        raise ValueError(f"unknown path {path_name!r}")
    ledger = common.LaunchLedger()
    ledger.reserve(kind, path_name, seed, attempt)
    try:
        row = _run_episode(path_name, seed, attempt)
    except Exception as error:  # a technical failure still spends the launch
        ledger.record(path_name=path_name, seed=seed, attempt=attempt, ok=False,
                      detail=f"{path_name}: {type(error).__name__}: {error}")
        raise
    ledger.record(path_name=path_name, seed=seed, attempt=attempt, ok=True,
                  detail=f"{path_name}: on_time={row['on_time']} bytes={row['spent_bytes']}")
    common.ROWS.mkdir(parents=True, exist_ok=True)
    Path(row["artefacts"]["row"]).write_text(
        json.dumps(row, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return row


def main(argv: list[str]) -> int:
    """Run the six paths for one seed: ``python sb_run.py --seed 1100000 [--path P_init]``."""
    import argparse

    parser = argparse.ArgumentParser(description="one Stage B cell (books its launch first)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--path", choices=list(PATH_POLICY), default=None)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--kind", choices=("normal", "reserve"), default="normal")
    args = parser.parse_args(argv)
    paths = [args.path] if args.path else list(common.PATHS)
    for path_name in paths:
        row = run_cell(path_name, args.seed, args.attempt, kind=args.kind)
        print(json.dumps({"path": row["path"], "seed": row["seed"],
                          "on_time": row["on_time"], "spent_bytes": row["spent_bytes"],
                          "emissions": row["emissions"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
