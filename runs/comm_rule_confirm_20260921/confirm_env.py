"""This round's episode runner: round 6's, with an explicit stream path.

``protocol_env.run_episode`` is the runner every archived round-6 number was produced by, and
it resolves its stream through the pilot's directory. This round's 200 engineering streams
live in this round's own directory, so the resolver differs -- and a copy that differs must
prove it is not a drift. ``runner_agreement`` below runs the same training-stream episode
through both runners and compares the task outcome, every transport counter, the belief
statistics, the configuration hash and the arm label. If those agree, the only differences
here are the stream path and the declared reverse-channel probability, which is what the
protocol says they should be.

    python confirm_env.py    # prints the agreement verdict for one training stream
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable

import confirm_common as common

from integrated_sar.comm_policy import build_comm_policy
from integrated_sar.contracts import TaskStatus
from integrated_sar.expired_attribution import build_timelines
from integrated_sar.joint_driver import RunConfiguration, build_joint_run
from integrated_sar.lifecycle_metrics import load_records


def on_time_count(timelines: dict[str, Any]) -> int:
    """The frozen definition, unchanged since the round that fixed the environment."""
    return sum(
        1
        for timeline in timelines.values()
        if timeline.final_progress is not None
        and float(timeline.final_progress) >= 1.0
        and timeline.final_failure_reason != "deadline_expired"
    )


def run_episode(
    seed: int,
    *,
    directory: Path,
    label: str,
    policy: str = "immediate",
    policy_params: dict[str, Any] | None = None,
    budget_bytes: int | None = None,
    sends_per_step: int | None = None,
    information: str = "isolated",
    delivery_probability: float = 1.0,
    delay_ticks: int = 0,
    max_wait_steps: int | None = None,
    urgent_wait_steps: int | None = None,
    timing_mode: str = common.POLICY_MODE,
    ack: bool = False,
    ack_delay_ticks: int | None = None,
    ack_delivery_probability: float | None = None,
    stream: Path | None = None,
    controller: Any | None = None,
    policy_hook: Callable[[Any], None] | None = None,
    on_step: Callable[[int, float], None] | None = None,
    read_timelines: bool = True,
) -> dict[str, Any]:
    """One episode, its task outcome and the whole transport ledger, on a stated setting.

    ``max_wait_steps=None`` builds a relay with no bound declared at all: under
    ``policy_owned_timing`` the transport enforces no silence bound, and none of this round's
    five arms quotes one, so every arm's effective silence constraint is identically "none".

    ``ack`` turns the reverse channel on. Its delay and loss are passed explicitly and equal
    the setting's own uplink values, which is the protocol's rule; the ACK ledger is separate
    from the task credit and is reported beside it, never merged into it.
    """
    parameters = dict(policy_params or {})
    values = common.configuration_for(
        seed,
        directory=directory,
        label=label,
        stream=stream,
        comm_policy=policy,
        comm_policy_params=",".join(f"{key}={value}" for key, value in sorted(parameters.items())),
        comm_budget_bytes=budget_bytes,
        comm_sends_per_tick=sends_per_step,
        comm_delivery_probability=delivery_probability,
        comm_delay_ticks=delay_ticks,
        comm_max_wait_steps=max_wait_steps,
        comm_urgent_wait_steps=urgent_wait_steps,
        comm_timing_mode=timing_mode,
        comm_ack=bool(ack),
        comm_ack_delay_ticks=ack_delay_ticks,
        comm_ack_delivery_probability=ack_delivery_probability,
        gossip_interval=(
            common.protocol_common.ISOLATED_GOSSIP_INTERVAL
            if information == "isolated"
            else 2
        ),
        quiet=True,
    )
    started = time.time()
    Path(directory).mkdir(parents=True, exist_ok=True)
    run = build_joint_run(RunConfiguration(**values))
    relay = run.task_adapter.comm_relay
    if controller is not None:
        relay.attach_policy(controller)
    else:
        expected = build_comm_policy(policy, parameters)
        if relay.policy.name != expected.name or dict(relay.policy.parameters()) != dict(
            expected.parameters()
        ):
            raise RuntimeError(
                f"{label} s{seed}: built policy {relay.policy.name!r} {relay.policy.parameters()}, "
                f"asked for {expected.name!r} {expected.parameters()}"
            )
    if relay.budget.bytes_per_episode != budget_bytes:
        raise RuntimeError(f"{label} s{seed}: the byte credit never reached the executor")
    declared = relay.transport_parameters()
    if declared["timing_mode"] != timing_mode:
        raise RuntimeError(
            f"{label} s{seed}: the relay reports timing mode {declared['timing_mode']!r}, not "
            f"{timing_mode!r}; a protocol that cannot state which one it ran cannot be reported"
        )
    if bool(declared["ack_enabled"]) != bool(ack):
        raise RuntimeError(
            f"{label} s{seed}: the relay reports ack_enabled={declared['ack_enabled']!r}, not "
            f"{ack!r}; a run that cannot state whether it had a reverse channel cannot be reported"
        )
    if ack and declared.get("ack_delivery_probability") != float(ack_delivery_probability):
        raise RuntimeError(
            f"{label} s{seed}: the relay reports ack_delivery_probability="
            f"{declared.get('ack_delivery_probability')!r}, not {ack_delivery_probability!r}"
        )
    if declared["delivery_probability"] != float(delivery_probability):
        raise RuntimeError(
            f"{label} s{seed}: the relay reports delivery_probability="
            f"{declared['delivery_probability']!r}, not {delivery_probability!r}"
        )

    environment = run.task_adapter.env
    original_step = environment.step
    rewards_by_step: dict[int, float] = {}
    credited: set[str] = set()

    def tapped_step(*args: Any, **kwargs: Any) -> Any:
        result = original_step(*args, **kwargs)
        gained = 0.0
        for snapshot in run.registry.snapshots():
            if (
                snapshot.status is TaskStatus.COMPLETED
                and not snapshot.expired
                and snapshot.task_id not in credited
            ):
                credited.add(snapshot.task_id)
                gained += 1.0
        if gained:
            step = int(run.orchestrator.step_index) + 1
            rewards_by_step[step] = rewards_by_step.get(step, 0.0) + gained
            if on_step is not None:
                on_step(step, gained)
        return result

    environment.step = tapped_step
    if policy_hook is not None:
        policy_hook(relay)
    try:
        summary = run.run()
    finally:
        environment.step = original_step

    spent = relay.budget.spent_bytes
    if budget_bytes is not None and spent > budget_bytes:
        raise RuntimeError(
            f"{label} s{seed}: budget violated, spent {spent} against a credit of {budget_bytes}"
        )
    transcript = common.pilot.metrics.load_transcript(values["comm_transcript"])
    records = load_records(Path(values["output"])) if read_timelines else []
    timelines = build_timelines(records, arm="C", seed=seed) if read_timelines else {}
    credited_total = int(sum(rewards_by_step.values()))
    completed = on_time_count(timelines) if read_timelines else credited_total
    if read_timelines and completed != credited_total:
        raise RuntimeError(
            f"{label} s{seed}: the tap credited {credited_total} on-time targets, the lifecycle "
            f"log says {completed} -- reward and metric have diverged"
        )
    final_records = [record for record in records if record.get("event_type") == "final"]
    if read_timelines and not final_records:
        raise RuntimeError(
            f"{label} s{seed}: the lifecycle log has no final record; the episode did not "
            "finish and its metric cannot be trusted"
        )
    # Per-target outcomes and latencies: the first make "new per-target failures" countable
    # against a baseline, the second are the protocol's timeliness columns. Both are read
    # from the same timelines the metric is, so they cannot disagree with it.
    outcomes: dict[str, str] = {}
    for target_id, timeline in timelines.items():
        if timeline.final_failure_reason == "deadline_expired":
            outcomes[str(target_id)] = "expired"
        elif (
            timeline.final_progress is not None
            and float(timeline.final_progress) >= 1.0
            and timeline.final_failure_reason != "deadline_expired"
        ):
            outcomes[str(target_id)] = "on_time"
        else:
            outcomes[str(target_id)] = "unresolved"

    def _mean_latency(attribute: str) -> float | None:
        values = [
            float(getattr(timeline, attribute))
            for timeline in timelines.values()
            if getattr(timeline, attribute) is not None
        ]
        return round(sum(values) / len(values), 4) if values else None

    completion_latencies = [
        float(timeline.last_work - timeline.detected_at)
        for timeline in timelines.values()
        if timeline.last_work is not None and outcomes.get(str(timeline.target_id)) == "on_time"
    ]
    latencies = {
        "detection_to_assignment": _mean_latency("assignment_latency"),
        "detection_to_work": _mean_latency("delivery_latency"),
        "dispatch_to_work": _mean_latency("dispatch_to_work"),
        "detection_to_completion": (
            round(sum(completion_latencies) / len(completion_latencies), 4)
            if completion_latencies
            else None
        ),
        "targets_with_latency": sum(
            1
            for timeline in timelines.values()
            if timeline.assignment_latency is not None
        ),
    }
    return {
        "seed": int(seed),
        "label": label,
        "policy": policy,
        "policy_parameters": parameters,
        "information": information,
        "timing_mode": timing_mode,
        "bound_enforced_by_transport": declared["bound_enforced_by_transport"],
        "ack_enabled": bool(declared["ack_enabled"]),
        "ack_parameters": {
            key: declared[key]
            for key in (
                "ack_delay_ticks",
                "ack_delivery_probability",
                "ack_rng_scheme",
                "ack_billing",
            )
        },
        # The relay's own reverse ledger read *after* the episode: the pre-run snapshot above
        # is the declared protocol, this one is the traffic. Two independent counts of the
        # same ACK bytes that must agree before any cost is quoted.
        "ack_ledger": relay.transport_parameters().get("ack_usage"),
        "budget_bytes": budget_bytes,
        "sends_per_step": sends_per_step,
        "delivery_probability": delivery_probability,
        "delay_ticks": delay_ticks,
        "max_wait_steps": max_wait_steps,
        "urgent_wait_steps": urgent_wait_steps,
        "spent_bytes": spent,
        "budget_used_fraction": round(spent / budget_bytes, 5) if budget_bytes else None,
        "return_on_time": credited_total,
        "rewards_by_step": {str(key): value for key, value in rewards_by_step.items()},
        "ledger": {
            **common.pilot.metrics.transport_accounting(transcript),
            **{key: int(value) for key, value in relay.counters.items()},
        },
        "task": {
            "targets": len(timelines),
            "on_time": completed,
            "expired": sum(
                1
                for timeline in timelines.values()
                if timeline.final_failure_reason == "deadline_expired"
            ),
            "unresolved": sum(
                1
                for timeline in timelines.values()
                if not (
                    timeline.final_progress is not None and float(timeline.final_progress) >= 1.0
                )
                and timeline.final_failure_reason != "deadline_expired"
            ),
        },
        "belief": (
            common.pilot.metrics.belief_age_accounting(
                transcript,
                horizon=float(summary.get("steps", 0)),
                robots=[str(agent_id) for agent_id in environment.agent_dic],
                truth=common.pilot.metrics.truth_timeline(records),
                active_until=common.pilot.metrics.terminal_times(records),
            )
            if read_timelines
            else {}
        ),
        "seconds": round(time.time() - started, 2),
        "arm_label": run.run_signature["arm"],
        "config_hash": run.run_signature["config_hash"],
        "stream_sha256": common.sha256(Path(values["replay_stream"])),
        "latencies": latencies,
        "targets_outcome": outcomes,
        "artefacts": {
            "lifecycle_log": values["output"],
            "transcript": values["comm_transcript"],
            "decision_trace": values["decision_trace"],
        },
    }


def runner_agreement(seeds: tuple[int, ...], *, directory: Path) -> dict[str, Any]:
    """The copy must not be a drift: same outcome, ledger, identity and configuration hash.

    Compared against round 6's runner rather than against a remembered number. Both runners
    get the same label and directory and replay the *same* stream file (the pilot's recorded
    training stream, which round 6's resolver also picks), so any difference is the runner
    and not the data. Each run parses its own artefacts before returning, so the second only
    overwrites files that were already read.
    """
    disagreements: dict[str, Any] = {}
    for seed in seeds:
        stream = common.pilot.stream_path(seed)
        if not stream.exists():
            raise FileNotFoundError(f"no pilot training stream for seed {seed}: {stream}")
        here = run_episode(
            seed,
            directory=directory,
            label="agree",
            policy="event",
            budget_bytes=common.BUDGET_BYTES,
            timing_mode=common.POLICY_MODE,
            max_wait_steps=None,
            urgent_wait_steps=None,
            delivery_probability=common.ack_common.PRIMARY_DELIVERY_PROBABILITY,
            ack=True,
            ack_delay_ticks=common.ACK_DELAY_TICKS,
            ack_delivery_probability=common.ack_common.PRIMARY_DELIVERY_PROBABILITY,
            stream=stream,
        )
        there = common.protocol_env.run_episode(
            seed,
            directory=directory,
            label="agree",
            policy="event",
            budget_bytes=common.BUDGET_BYTES,
            timing_mode=common.POLICY_MODE,
            max_wait_steps=None,
            urgent_wait_steps=None,
            delivery_probability=common.ack_common.PRIMARY_DELIVERY_PROBABILITY,
            ack=True,
            ack_delay_ticks=common.ACK_DELAY_TICKS,
            ack_delivery_probability=common.ack_common.PRIMARY_DELIVERY_PROBABILITY,
        )
        fields = ("on_time", "targets", "expired", "unresolved")
        left = {
            "task": {key: here["task"][key] for key in fields},
            "spent": here["spent_bytes"],
            "ledger": here["ledger"],
            "belief": here["belief"],
            "config_hash": here["config_hash"],
            "arm_label": here["arm_label"],
            "stream_sha256": here["stream_sha256"],
        }
        right = {
            "task": {key: there["task"][key] for key in fields},
            "spent": there["spent_bytes"],
            "ledger": there["ledger"],
            "belief": there["belief"],
            "config_hash": there["config_hash"],
            "arm_label": there["arm_label"],
            "stream_sha256": common.sha256(stream),
        }
        if left != right:
            differences = sorted(key for key in left if left[key] != right[key])
            ledger_keys = sorted(
                key for key in left["ledger"] if left["ledger"][key] != right["ledger"][key]
            )
            disagreements[str(seed)] = {
                "fields": differences,
                "ledger": {key: [left["ledger"][key], right["ledger"][key]] for key in ledger_keys},
            }
    return {
        "seeds": list(seeds),
        "policy": "event",
        "budget_bytes": common.BUDGET_BYTES,
        "mismatches": disagreements,
        "agree": not disagreements,
    }


def main(argv: list[str]) -> int:
    seeds = tuple(common.pilot.TRAIN_SEEDS[:2])
    directory = common.HERE / "preflight"
    directory.mkdir(parents=True, exist_ok=True)
    result = runner_agreement(seeds, directory=directory)
    import json

    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result["agree"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
