"""One episode of the communication pilot, and what it earned.

Every arm -- rule or learned -- goes through this same call: the frozen allocation side, the
same recorded detection stream, the same budget executor, the same transport rules. The only
difference between arms is which object decides what goes on the air, which is what makes a
paired comparison mean anything.

The reward is the task metric, not a hand-tuned blend: one credit per target completed before
its deadline, booked on the step it happened. Airtime is neither rewarded nor punished -- the
budget is a hard constraint the executor enforces -- so a controller is never paid for holding
information it was not asked to hold, and never paid for saving bytes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable

import pilot_common as common

from integrated_sar.comm_policy import build_comm_policy  # noqa: E402
from integrated_sar.contracts import TaskStatus  # noqa: E402
from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

ISOLATED_GOSSIP_INTERVAL = 100000

#: The silence bound every pilot arm obeys -- rule and learned alike -- and the urgency window
#: that tightens it. Declared once here so the transport, the collector's action mask and the
#: report cannot drift apart.
MAX_WAIT_STEPS = 10
MAX_WAIT_WHEN_URGENT_STEPS = 3
URGENCY_STEPS = 25.0


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
    policy: str = "event",
    policy_params: dict[str, Any] | None = None,
    budget_bytes: int | None = None,
    sends_per_step: int | None = None,
    information: str = "isolated",
    delivery_probability: float = 1.0,
    delay_ticks: int = 0,
    max_wait_steps: int | None = MAX_WAIT_STEPS,
    urgent_wait_steps: int | None = MAX_WAIT_WHEN_URGENT_STEPS,
    stream: Path | None = None,
    controller: Any | None = None,
    on_step: Callable[[int, float], None] | None = None,
    read_timelines: bool = True,
) -> dict[str, Any]:
    """Run one episode and return its task outcome plus the full transport ledger."""
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
        gossip_interval=(
            ISOLATED_GOSSIP_INTERVAL if information == "isolated" else 2
        ),
        quiet=True,
    )
    started = time.time()
    Path(directory).mkdir(parents=True, exist_ok=True)
    run = build_joint_run(RunConfiguration(**values))
    relay = run.task_adapter.comm_relay
    expected = build_comm_policy(policy, parameters)
    if controller is not None:
        relay.attach_policy(controller)
    elif relay.policy.name != expected.name or dict(relay.policy.parameters()) != dict(
        expected.parameters()
    ):
        raise RuntimeError(
            f"{label} s{seed}: built policy {relay.policy.name!r} {relay.policy.parameters()}, "
            f"asked for {expected.name!r} {expected.parameters()}"
        )
    if relay.budget.bytes_per_episode != budget_bytes:
        raise RuntimeError(f"{label} s{seed}: the byte credit never reached the executor")

    # The reward tap books a completion when it happens, against the same definition used to
    # score the episode afterwards; the two are asserted equal below so a reward that drifts
    # away from the reported metric fails the run instead of quietly outscoring it.
    environment = run.task_adapter.env
    original_step = environment.step
    rewards_by_step: dict[int, float] = {}
    credited: set[str] = set()

    def tapped_step(*args: Any, **kwargs: Any) -> Any:
        result = original_step(*args, **kwargs)
        gained = 0.0
        for snapshot in run.registry.snapshots():
            # The same completion the offline metric counts: finished, and not by having
            # missed its deadline first.
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
    try:
        summary = run.run()
    finally:
        environment.step = original_step

    spent = relay.budget.spent_bytes
    if budget_bytes is not None and spent > budget_bytes:
        raise RuntimeError(
            f"{label} s{seed}: budget violated, spent {spent} against a credit of {budget_bytes}"
        )
    transcript = common.metrics.load_transcript(values["comm_transcript"])
    records = load_records(Path(values["output"])) if read_timelines else []
    timelines = build_timelines(records, arm="C", seed=seed) if read_timelines else {}
    credited_total = int(sum(rewards_by_step.values()))
    completed = on_time_count(timelines) if read_timelines else credited_total
    if read_timelines and completed != credited_total:
        raise RuntimeError(
            f"{label} s{seed}: the tap credited {credited_total} on-time targets, the lifecycle "
            f"log says {completed} -- reward and metric have diverged"
        )
    return {
        "seed": int(seed),
        "label": label,
        "policy": policy,
        "policy_parameters": parameters,
        "information": information,
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
            **common.metrics.transport_accounting(transcript),
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
                    timeline.final_progress is not None
                    and float(timeline.final_progress) >= 1.0
                )
                and timeline.final_failure_reason != "deadline_expired"
            ),
        },
        "belief": (
            common.metrics.belief_age_accounting(
                transcript,
                horizon=float(summary.get("steps", 0)),
                robots=[str(agent_id) for agent_id in environment.agent_dic],
                truth=common.metrics.truth_timeline(records),
                active_until=common.metrics.terminal_times(records),
            )
            if read_timelines
            else {}
        ),
        "seconds": round(time.time() - started, 2),
        "arm_label": run.run_signature["arm"],
        "config_hash": run.run_signature["config_hash"],
        "artefacts": {
            "lifecycle_log": values["output"],
            "transcript": values["comm_transcript"],
        },
    }


def main(argv: list[str]) -> int:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=common.TRAIN_SEEDS[0])
    parser.add_argument("--policy", default="event")
    parser.add_argument("--budget-bytes", type=int, default=None)
    args = parser.parse_args(argv)
    row = run_episode(
        args.seed,
        directory=common.HERE / "selftest",
        label="selftest",
        policy=args.policy,
        budget_bytes=args.budget_bytes,
    )
    print(
        f"seed={row['seed']} policy={row['policy']} on_time={row['task']['on_time']}/"
        f"{row['task']['targets']} spent={row['spent_bytes']} "
        f"attempts={row['ledger']['transmission_attempts']} "
        f"applied={row['ledger']['deliveries_applied']} arm={row['arm_label']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
