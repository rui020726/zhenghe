"""This round's episode runner: the frozen pilot's, with the timing protocol as an argument.

``pilot_env.run_episode`` builds the configuration and the reward tap for every round since the
pilot, and it does not know about ``comm_timing_mode``. Editing it would move an executed module
out from under the archived rounds that were produced by it, so this round carries its own copy
instead -- and a check that the copy is not a drift: ``legacy_runner_agreement`` below runs the
same episode through both and compares the task outcome, every transport counter and the run
identity. If that holds, everything new here is the protocol dial and the policy hook.

The hook exists because the mask audit needs to see the candidate list the relay assembled, which
is only observable from the policy seam. It is called after the relay is built and before the first
step, in the window where ``CommRelay.attach_policy`` still accepts a policy at all.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable

import protocol_common as common

from integrated_sar.comm_policy import build_comm_policy
from integrated_sar.contracts import TaskStatus
from integrated_sar.expired_attribution import build_timelines
from integrated_sar.joint_driver import RunConfiguration, build_joint_run
from integrated_sar.lifecycle_metrics import load_records

LEGACY = common.LEGACY_MODE
POLICY_OWNED = common.POLICY_MODE


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
    max_wait_steps: int | None = common.pilot_env.MAX_WAIT_STEPS,
    urgent_wait_steps: int | None = common.pilot_env.MAX_WAIT_WHEN_URGENT_STEPS,
    timing_mode: str = LEGACY,
    controller: Any | None = None,
    policy_hook: Callable[[Any], None] | None = None,
    on_step: Callable[[int, float], None] | None = None,
    read_timelines: bool = True,
) -> dict[str, Any]:
    """One episode, its task outcome and the whole transport ledger, on a stated protocol.

    ``max_wait_steps=None`` builds a relay with no bound declared at all. Under
    ``policy_owned_timing`` the declared numbers stay on the relay as constants and produce no
    emission by themselves; passing them is what lets the truth table say "declared, not enforced".
    """
    parameters = dict(policy_params or {})
    values = common.pilot.configuration_for(
        seed,
        directory=directory,
        label=label,
        stream=None,
        comm_policy=policy,
        comm_policy_params=",".join(f"{key}={value}" for key, value in sorted(parameters.items())),
        comm_budget_bytes=budget_bytes,
        comm_sends_per_tick=sends_per_step,
        comm_delivery_probability=delivery_probability,
        comm_delay_ticks=delay_ticks,
        comm_max_wait_steps=max_wait_steps,
        comm_urgent_wait_steps=urgent_wait_steps,
        comm_timing_mode=timing_mode,
        gossip_interval=(common.ISOLATED_GOSSIP_INTERVAL if information == "isolated" else 2),
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
        "timing_mode": timing_mode,
        "bound_enforced_by_transport": declared["bound_enforced_by_transport"],
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
                    timeline.final_progress is not None and float(timeline.final_progress) >= 1.0
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


def rollout_episode(
    controller: Any,
    seed: int,
    *,
    collector: Any,
    budget_bytes: int,
    directory: Path,
    label: str,
    timing_mode: str = POLICY_OWNED,
    information: str = "isolated",
    delivery_probability: float = 1.0,
) -> dict[str, Any]:
    """One training rollout: the collector is the only sender, and the tape sums to the score.

    The reward-to-decision alignment is the pilot's, copied rather than imported because
    ``train_pilot.run_one`` reaches for ``pilot_env`` and so cannot carry a timing mode. Each
    reward belongs to the next decision that could still act on it; rewards after the last
    decision join that one, so the credits along the tape add up to the episode score exactly.
    """
    rewards: dict[int, float] = {}

    def on_step(step: int, gained: float) -> None:
        rewards[step] = rewards.get(step, 0.0) + gained

    row = run_episode(
        seed,
        directory=directory,
        label=label,
        policy="learned",
        budget_bytes=budget_bytes,
        controller=collector.policy(),
        on_step=on_step,
        information=information,
        delivery_probability=delivery_probability,
        timing_mode=timing_mode,
        max_wait_steps=common.pilot_env.MAX_WAIT_STEPS if timing_mode == LEGACY else None,
        urgent_wait_steps=(
            common.pilot_env.MAX_WAIT_WHEN_URGENT_STEPS if timing_mode == LEGACY else None
        ),
    )
    by_step = {int(record["step"]): record for record in collector.records}
    decisions = sorted(by_step)
    total = float(sum(rewards.values()))
    aligned: list[float] = []
    if decisions:
        previous = 0
        for step in decisions:
            aligned.append(sum(value for key, value in rewards.items() if previous < key <= step))
            previous = step
        aligned[-1] += total - sum(aligned)
    if abs(sum(aligned) - total) > 1e-9:
        raise RuntimeError(
            f"{label} s{seed}: the tape's rewards sum to {sum(aligned)}, the episode scored {total}"
        )
    return {
        "row": row,
        "decisions": [by_step[step] for step in decisions],
        "rewards": aligned,
        "steps": decisions,
    }


def legacy_runner_agreement(
    seeds: tuple[int, ...], *, directory: Path, budget_bytes: int
) -> dict[str, Any]:
    """The copy must not be a drift: same outcome, ledger, identity and configuration hash.

    Compared against the pilot's runner rather than against a remembered number, so the claim
    stays checkable after this file changes. Both runners get the *same* label and directory: the
    transcript path sits inside the hashed configuration, so naming the two runs differently would
    make ``config_hash`` differ for a reason that has nothing to do with the experiment. Each run
    parses its own artefacts before it returns, so the second one only overwrites files that have
    already been read.
    """
    import pilot_env

    fields = ("on_time", "targets", "expired", "unresolved")
    disagreements: dict[str, Any] = {}
    for seed in seeds:
        here = run_episode(
            seed,
            directory=directory,
            label="agree",
            policy="event",
            budget_bytes=budget_bytes,
            timing_mode=LEGACY,
        )
        there = pilot_env.run_episode(
            seed,
            directory=directory,
            label="agree",
            policy="event",
            budget_bytes=budget_bytes,
        )
        left = {
            "task": {key: here["task"][key] for key in fields},
            "spent": here["spent_bytes"],
            "ledger": here["ledger"],
            "belief": here["belief"],
            "config_hash": here["config_hash"],
            "arm_label": here["arm_label"],
        }
        right = {
            "task": {key: there["task"][key] for key in fields},
            "spent": there["spent_bytes"],
            "ledger": there["ledger"],
            "belief": there["belief"],
            "config_hash": there["config_hash"],
            "arm_label": there["arm_label"],
        }
        if left != right:
            differences = sorted(
                key for key in left if left[key] != right[key]
            )
            ledger_keys = sorted(
                key
                for key in left["ledger"]
                if left["ledger"][key] != right["ledger"][key]
            )
            disagreements[str(seed)] = {
                "fields": differences,
                "ledger": {key: [left["ledger"][key], right["ledger"][key]] for key in ledger_keys},
            }
    return {
        "seeds": list(seeds),
        "policy": "event",
        "budget_bytes": budget_bytes,
        "mismatches": disagreements,
        "agree": not disagreements,
    }


__all__ = [
    "LEGACY",
    "POLICY_OWNED",
    "run_episode",
    "rollout_episode",
    "legacy_runner_agreement",
    "on_time_count",
]

if __name__ == "__main__":  # pragma: no cover
    sys.stdout.reconfigure(encoding="utf-8")
    print(
        "usage: imported by this round's scripts; "
        "python protocol_v2_preflight.py runs the agreement check"
    )
