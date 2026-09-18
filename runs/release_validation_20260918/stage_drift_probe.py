"""Is the release reachable on a stale stage belief? (release brief section 2.5)

The belief a release verdict reads is the robot's own, and a belief is only as fresh as the
last message that reached it. The question this probe answers is narrow: when a robot
believes its task's *current* stage needs nothing (the stage is satisfied, so the believed
remaining requirement is all zeros) while the same belief carries the next stage's
requirement, does the release path still take the robot away?

Runs no experiment: it calls the coordinator on constructed states and prints the verdict.

    D:/project_env/python.exe runs/release_validation_20260918/stage_drift_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from integrated_sar.coordinator import CoalitionPlannerCoordinator
from integrated_sar.contracts import (
    AgentSnapshot,
    AgentTaskBelief,
    CoordinatorInput,
    WorldPosition,
)

TRAITS = 5
NOW = 10.0


def vec(*dims: int) -> tuple[float, ...]:
    out = [0.0] * TRAITS
    for dim in dims:
        out[dim] = 1.0
    return tuple(out)


def task_belief(task_id, requirement, *, future=(), stage=0, deadline=100.0, position=(0.0, 0.0)):
    return AgentTaskBelief(
        task_id=task_id,
        position=WorldPosition(*position),
        remaining_requirements=tuple(float(v) for v in requirement),
        deadline=deadline,
        stage_index=stage,
        progress=0.0,
        severity=0.5,
        future_requirements=tuple(float(v) for v in future),
        members=(),
        updated_at=0.0,
        finished=False,
        dynamic=False,
        confidence=1.0,
        revision=1,
    )


def on_site(agent_id, dims, current_task, beliefs):
    return AgentSnapshot(
        agent_id=agent_id,
        position=WorldPosition(0.1, 0.1),
        abilities=vec(*dims),
        status="working",
        current_task=current_task,
        needs_decision=False,
        known_tasks=tuple(belief.task_id for belief in beliefs),
        task_beliefs=tuple(beliefs),
    )


def verdict(agents, *, release):
    coordinator = CoalitionPlannerCoordinator(renew_commitments=True, release_redundant_members=release)
    advice = coordinator.advise(CoordinatorInput(current_time=NOW, tasks=(), agents=tuple(agents)))
    return {
        "releases": int(advice.metadata.get("redundant_releases", 0)),
        "replan_agents": sorted(advice.replan_agents),
        "reservations": {agent_id: res.task_id for (task_id, agent_id), res in advice.reservations.items()},
    }


def scenario_stale_stage():
    """Both robots believe T's current stage is done, and both believe the next stage needs two of them."""
    beliefs_a = [
        task_belief("T", (0, 0, 0, 0, 0), future=(2, 0, 0, 0, 0), stage=1),
        task_belief("U", (1, 0, 0, 0, 0)),
        task_belief("V", (1, 0, 0, 0, 0)),
    ]
    beliefs_b = [
        task_belief("T", (0, 0, 0, 0, 0), future=(2, 0, 0, 0, 0), stage=1),
        task_belief("U", (1, 0, 0, 0, 0)),
        task_belief("V", (1, 0, 0, 0, 0)),
    ]
    return [
        on_site("a", (0,), "T", beliefs_a),
        on_site("b", (0,), "T", beliefs_b),
    ]


def scenario_stale_stage_no_future():
    """Same, but the belief carries no next-stage information at all."""
    beliefs = [
        task_belief("T", (0, 0, 0, 0, 0), stage=1),
        task_belief("U", (1, 0, 0, 0, 0)),
        task_belief("V", (1, 0, 0, 0, 0)),
    ]
    return [on_site("a", (0,), "T", list(beliefs)), on_site("b", (0,), "T", list(beliefs))]


def scenario_live_gap():
    """Control: the believed current stage still needs both, so neither is redundant."""
    beliefs = [
        task_belief("T", (2, 0, 0, 0, 0), future=(1, 0, 0, 0, 0), stage=1),
        task_belief("U", (1, 0, 0, 0, 0)),
        task_belief("V", (1, 0, 0, 0, 0)),
    ]
    return [on_site("a", (0,), "T", list(beliefs)), on_site("b", (0,), "T", list(beliefs))]


def main() -> int:
    for name, builder in (
        ("stale stage belief, next stage needs both", scenario_stale_stage),
        ("stale stage belief, no next-stage information", scenario_stale_stage_no_future),
        ("control: believed current stage still needs both", scenario_live_gap),
    ):
        off = verdict(builder(), release=False)
        on = verdict(builder(), release=True)
        print(f"{name}")
        print(f"    flag off: {off}")
        print(f"    flag on : {on}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
