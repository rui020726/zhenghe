"""Where the "illegal execution" counts come from, event by event.

The block's detector counts a command as illegal when the environment already says the
destination task is finished at the moment the command is applied. It fired three times in arm
C -- the arm with no release mechanism at all -- and three times in arm D, so whatever it is,
it is not something the candidate introduced. Whether it is nevertheless a real violation
depends on what the boundary was supposed to know: the gate screens a command against the
registry, and the registry only learns a task has closed when the adapter's next progress
event is accepted. A belief that lagged behind a just-closed task sends a robot travelling
toward it, and the architecture's own answer is the arrival check, not the command gate.

This replays the affected seeds and writes both readings for every such command: the
environment truth, the registry's status and terminal flag and revision at that moment, the
revision the gate compared against, and what the robot believed.

    D:/project_env/python.exe runs/corrected_env_final_20260918/illegal_probe.py --arm C --seed 500111
"""

from __future__ import annotations

import argparse
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
for path in (REPO, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import arms  # noqa: E402
import corrected_block as block  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402

OUT = HERE / "illegal_probe"


def probe(label: str, seed: int) -> dict:
    directory = OUT / label
    directory.mkdir(parents=True, exist_ok=True)
    arm = arms.ARMS[label]
    configuration_stream = block.stream_path(seed)
    run = build_joint_run(
        RunConfiguration(
            **{
                **arm.configuration,
                "seed": seed,
                "replay_stream": str(configuration_stream),
                "output": str(directory / f"probe_{label}_s{seed}.jsonl"),
                "decision_trace": str(directory / f"probe_tr_{label}_s{seed}.jsonl"),
            }
        )
    )
    env = run.task_adapter.env
    registry = run.registry
    inner_apply_action = env.apply_action
    events: list[dict] = []

    def apply_action(agent_id, action_index, action_map):
        kind, tid = (
            action_map[action_index] if 0 <= action_index < len(action_map) else ("depot", None)
        )
        if kind == "task" and tid is not None:
            task = env.task_dic.get(int(tid))
            if task is not None and task.get("finished"):
                external = run.task_adapter.internal_to_external.get(int(tid), str(tid))
                snapshot = registry.get(str(external))
                # The last command the gate was handed for this robot carries the revision it
                # compared; the adapter keeps the mapping from the issued command.
                belief = env.agent_knowledge[int(agent_id)]
                events.append(
                    {
                        "env_time": round(float(env.current_time), 3),
                        "agent_id": str(agent_id),
                        "destination_internal": int(tid),
                        "destination_external": str(external),
                        "env_truth": {
                            "finished": True,
                            "expired": bool(task.get("expired", False)),
                            "roster": [int(v) for v in task["members"]],
                            "progress": round(float(env.task_progress_of(task)), 3),
                        },
                        "registry_at_that_moment": None if snapshot is None else {
                            "status": str(snapshot.status.value),
                            "terminal": bool(snapshot.status.terminal),
                            "revision": int(snapshot.revision),
                            "progress": round(float(snapshot.progress), 3),
                            "expired": bool(snapshot.expired),
                        },
                        "robot_belief": {
                            "knows_the_task": int(tid) in belief.get("known_tasks", set()),
                            "believes_finished": int(tid) in belief.get("known_finished", set()),
                            "believed_status": [
                                int(v) for v in (
                                    belief.get("task_current_status", {}).get(int(tid))
                                    if belief.get("task_current_status", {}).get(int(tid)) is not None
                                    else ()
                                )
                            ],
                            "believed_progress": round(
                                float(belief.get("task_progress", {}).get(int(tid), -1.0)), 3
                            ),
                        },
                    }
                )
        return inner_apply_action(agent_id, action_index, action_map)

    env.apply_action = apply_action
    try:
        summary = run.run()
    finally:
        env.apply_action = inner_apply_action
        if run.decision_trace is not None:
            run.decision_trace.close()
    payload = {
        "arm": label,
        "seed": seed,
        "commands_aimed_at_a_task_already_closed_in_truth": len(events),
        "of_those_terminal_in_the_registry": sum(
            1 for event in events
            if event["registry_at_that_moment"] is not None
            and event["registry_at_that_moment"]["terminal"]
        ),
        "events": events,
    }
    destination = directory / f"probe_{label}_s{seed}.json"
    part = destination.with_suffix(".json.part")
    part.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(part, destination)
    print(json.dumps(payload, ensure_ascii=False, indent=1)[:2400])
    return payload


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=tuple(arms.ARMS))
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args(argv)
    probe(args.arm, args.seed)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
