"""One executed release opened a coverage gap in cell D. What disagreed with what?

The corrected recheck found a single event across the two development blocks: seed 20008's
counterpart 20083, tick 15, task 2, robot 3 released, and the stage left uncovered on one
unit after it went. Everything upstream said the release was safe -- the planner's joint
verdict, the source-stage check, the support check. So one of the readings the release was
judged against did not match the world the command landed in.

This replays that episode and writes the whole chain for every release the coordinator
publishes: the belief the verdict used (stage, gross requirement, next-stage vector, roster,
revision), the truth at the same moment (stage, requirement, roster, abilities), what the
execution boundary read when the command arrived (live stage, live roster), and whether the
command was accepted. The harmful tick is then readable as a specific disagreement rather
than as an argument.

    D:/project_env/python.exe runs/corrected_env_final_20260918/release_event_20083.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
try:
    import torch

    torch.set_num_threads(1)
except Exception:
    pass

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (REPO, HERE, REPO / "runs" / "membership_consistency_20260918",
             REPO / "runs" / "release_validation_20260918"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from integrated_sar.coordinator import CoalitionPlannerCoordinator  # noqa: E402
from integrated_sar.joint_driver import build_joint_run  # noqa: E402

import membership_measurement as mm  # noqa: E402

OUT = HERE / "release_event"
SEED = 20083
CELL = "D"
EPS = 1e-9


def main() -> int:
    directory = OUT
    directory.mkdir(parents=True, exist_ok=True)
    stream = mm.DEV_STREAMS / f"s{SEED}.json"
    configuration, arm = mm.configuration_for(CELL, SEED, stream, "dev", directory)
    run = build_joint_run(configuration)
    env = run.task_adapter.env
    adapter = run.task_adapter
    inner_advise = CoalitionPlannerCoordinator.advise
    apply_commands = run.gate.apply_assignments
    ledger: list[dict] = []
    pending: dict[str, dict] = {}

    def truth_for(external_id: str) -> dict | None:
        internal = adapter.external_to_internal.get(str(external_id))
        task = env.task_dic.get(int(internal)) if internal is not None else None
        if task is None:
            return None
        stage = mm.stage_of(task)
        return {
            "internal_id": int(internal),
            "stage_index": int(task["current_stage_index"]),
            "requirement": [int(v) for v in stage["requirements"]],
            "status_after_last_recompute": [int(v) for v in stage["status"]],
            "roster": [int(v) for v in task["members"]],
            "clock_left": round(float(stage["time_left"]), 3),
            "member_abilities": {
                str(int(m)): [int(v) for v in env.agent_dic[int(m)]["abilities"]]
                for m in task["members"] if int(m) in env.agent_dic
            },
        }

    def advise_with_record(self, state):
        advice = inner_advise(self, state)
        for agent_id, provenance in advice.releases.items():
            agent = next(item for item in state.agents if item.agent_id == agent_id)
            belief = agent.belief_for(provenance.task_id)
            record = {
                "planned_at_env_time": round(float(env.current_time), 3),
                "agent_id": str(agent_id),
                "agent_status": str(agent.status),
                "source_task": str(provenance.task_id),
                "provenance": {
                    "stage_index": int(provenance.stage_index),
                    "revision": int(provenance.revision),
                    "supporting_members": list(provenance.supporting_members),
                },
                "belief_used_by_the_verdict": None if belief is None else {
                    "stage_index": int(belief.stage_index),
                    "stage_requirements": [int(v) for v in belief.stage_requirements],
                    "remaining_requirements": [int(v) for v in belief.remaining_requirements],
                    "future_requirements": [int(v) for v in belief.future_requirements],
                    "members": [str(v) for v in belief.members],
                    "revision": int(belief.revision),
                    "deadline": round(float(belief.deadline), 3),
                    "finished": bool(belief.finished),
                },
                "truth_when_planning": truth_for(provenance.task_id),
                "own_abilities": [int(v) for v in agent.abilities],
            }
            pending[agent_id] = record
            ledger.append(record)
        return advice

    def apply_with_record(commands):
        known = {row.command_id for row in run.gate.rejections}
        apply_commands(commands)
        refused = {row.command_id for row in run.gate.rejections} - known
        for issued in commands:
            metadata = issued.metadata or {}
            source = metadata.get("release_from_task")
            if not source:
                continue
            record = pending.pop(str(issued.agent_id), None)
            if record is None:
                continue
            record["command"] = {
                "issued_at": round(float(issued.issued_at), 3),
                "destination": str(issued.task_id),
                "destination_revision": issued.task_revision,
                "release_stage_index": int(metadata.get("release_stage_index", -1)),
                "release_support_members": [
                    str(v) for v in metadata.get("release_support_members", ())
                ],
            }
            record["at_the_boundary"] = {
                "env_time": round(float(env.current_time), 3),
                "live_stage_and_revision": (
                    None if run.gate.release_state_of is None
                    else run.gate.release_state_of(str(source))
                ),
                "live_roster": (
                    None if run.gate.release_roster_of is None
                    else [str(v) for v in run.gate.release_roster_of(str(source))]
                ),
                "refused": issued.command_id in refused,
            }
            record["truth_when_the_command_landed"] = truth_for(source)
        return None

    CoalitionPlannerCoordinator.advise = advise_with_record
    run.gate.apply_assignments = apply_with_record
    try:
        summary = run.run()
    finally:
        CoalitionPlannerCoordinator.advise = inner_advise
        if run.decision_trace is not None:
            run.decision_trace.close()
    payload = {
        "cell": CELL,
        "seed": SEED,
        "arm": arm,
        "releases": ledger,
        "status_counts": summary.get("status_counts"),
    }
    destination = directory / f"release_chain_{CELL}_s{SEED}.json"
    part = destination.with_suffix(".json.part")
    part.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(part, destination)
    print(f"releases planned: {len(ledger)}")
    for record in ledger:
        if record["planned_at_env_time"] > 20.0:
            continue
        print(json.dumps(record, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
