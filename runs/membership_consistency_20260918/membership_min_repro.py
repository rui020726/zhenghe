"""Minimal reproduction: does a reassigned robot keep serving the task it left?

Two tasks, two robots, no coordinator, no beliefs, no randomness beyond the scenario seed.
Robot 0 is the only robot that can serve either task's first stage. It is sent to A, arrives,
covers A, and is then re-tasked to B -- which is what an accepted cross-task assignment does
inside the real loop. The question is mechanical: after it leaves, does A still count as
covered, does A's work clock keep running down, is the robot carried on two rosters at once,
and what happens when A finally closes?

The scripted scenario is played twice: under the environment's existing bookkeeping, and
under ``membership_consistency``. No environment logic is patched here -- stage requirements
and positions are set, then the same ``apply_action`` and ``step`` the adapter calls.

    D:/project_env/python.exe runs/membership_consistency_20260918/membership_min_repro.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from integrated_sar.task03_runtime import build_task_env, load_task_env_module

TRAITS = 5
A, B = 0, 1
A_NEED = (0, 0, 0, 0, 1)   # only robot 0 carries dim 4
B_NEED = (1, 0, 0, 0, 0)   # only robot 0 carries dim 0
STAGE_TIME = 60.0
EPS = 1e-9


def single_stage(env, tid, location, need):
    env._spawn_new_task(tid, location=np.array(location, dtype=float))
    task = env.task_dic[tid]
    requirement = np.array(need, dtype=int)
    task["stages"] = [
        {
            "stage_id": 0,
            "requirements": requirement.copy(),
            "status": requirement.copy(),
            "time": STAGE_TIME,
            "time_left": STAGE_TIME,
            "finished": False,
        }
    ]
    task["total_stages"] = 1
    task["current_stage_index"] = 0
    task["requirements"] = requirement.copy()
    task["status"] = requirement.copy()
    task["time"] = STAGE_TIME
    task["members"] = []
    task["deadline"] = 400.0
    return task


def stage(task):
    return task["stages"][min(int(task["current_stage_index"]), len(task["stages"]) - 1)]


def on_site(env, task, tid):
    return [
        member for member in task["members"]
        if env.agent_dic[member].get("status") == "working"
        and int(env.agent_dic[member].get("current_task", -1)) == tid
    ]


def describe(env, tid, label):
    task = env.task_dic[tid]
    present = on_site(env, task, tid)
    gap_present_only = np.maximum(
        stage(task)["requirements"]
        - (env.get_abilities(present) if present else np.zeros(TRAITS)),
        0,
    )
    return (
        f"{label:>18} | roster={task['members']} on-site={present} "
        f"time_left={round(float(stage(task)['time_left']), 2):>5} "
        f"computed_status={list(np.asarray(task['status'], dtype=int))} "
        f"gap_if_only_on_site_counted={list(np.asarray(gap_present_only, dtype=int))}"
    )


def play(membership_consistency: bool) -> dict:
    env = build_task_env(
        load_task_env_module(),
        seed=11,
        species=2,
        per_species=1,
        dt=1.0,
        depot_positions=[(0.5, 0.5), (0.5, 0.5)],
        membership_consistency=membership_consistency,
    )
    env.reset(seed=11)
    single_stage(env, A, (0.60, 0.50), A_NEED)
    single_stage(env, B, (0.60, 0.90), B_NEED)
    env.agent_dic[0]["location"] = np.array([0.55, 0.50])
    env.agent_dic[1]["location"] = np.array([0.55, 0.50])
    action_map = [("depot", None), ("task", A), ("task", B)]

    def send(agent_id, tid):
        index = next(i for i, (kind, value) in enumerate(action_map) if value == tid)
        env.apply_action(agent_id, index, action_map)

    print(f"  roster consistency = {membership_consistency}")
    send(0, A)
    for _ in range(6):
        env.step()
    print("   arrived at A      ", describe(env, A, ""))

    send(0, B)
    for _ in range(8):
        env.step()
    roster_after_leaving = 0 in env.task_dic[A]["members"]
    print("   re-tasked to B    ", describe(env, A, ""))
    for _ in range(8):
        env.step()
    print("   robot 0 works at B", describe(env, B, ""), "|", describe(env, A, "A meanwhile"))

    # Ticks where A's clock ran down although nobody standing on A covered its requirement.
    ghost = 0
    for _ in range(20):
        before = float(stage(env.task_dic[A])["time_left"])
        env.step()
        task = env.task_dic[A]
        if task.get("finished"):
            break
        present = on_site(env, task, A)
        gap = np.maximum(
            stage(task)["requirements"]
            - (env.get_abilities(present) if present else np.zeros(TRAITS)),
            0,
        )
        if float(stage(task)["time_left"]) < before - EPS and float(gap.sum()) > EPS:
            ghost += 1

    dual = {
        str(member): [tid for tid, task in env.task_dic.items() if member in task["members"]]
        for member in {m for task in env.task_dic.values() for m in task["members"]}
    }
    dual = {member: tasks for member, tasks in dual.items() if len(tasks) > 1}

    finished_without_audience = False
    stolen_from_b = False
    for _ in range(80):
        if env.task_dic[A]["finished"]:
            break
        was_working_on_b = (
            env.agent_dic[0].get("status") == "working"
            and int(env.agent_dic[0].get("current_task", -1)) == B
        )
        env.step()
        if env.task_dic[A]["finished"]:
            finished_without_audience = not on_site(env, env.task_dic[A], A)
            # A's closing releases everyone on A's roster -- including a robot that has
            # since moved to B, which is only on A's roster because of the stale entry.
            stolen_from_b = was_working_on_b and int(
                env.agent_dic[0].get("current_task", -1)
            ) != B
            break
    agent = env.agent_dic[0]
    print("   after A closed    ", describe(env, B, ""), "|", describe(env, A, "A"))
    print(f"   robot 0 now: status={agent['status']} current_task={agent['current_task']}")
    print(
        f"   A ghost ticks while absent={ghost}  dual-roster robots={dual or 'none'}  "
        f"A closed with nobody on site={finished_without_audience}  "
        f"A's closing pulled robot 0 off B={stolen_from_b}\n"
    )
    return {
        "a_roster_still_holds_the_departed_robot": roster_after_leaving,
        "a_ghost_ticks_while_absent": ghost,
        "dual_roster_robots": dual,
        "a_finished": bool(env.task_dic[A]["finished"]),
        "a_finished_with_nobody_on_site": finished_without_audience,
        "a_closing_pulled_the_robot_off_b": stolen_from_b,
        "b_roster": list(env.task_dic[B]["members"]),
        "b_on_site": on_site(env, env.task_dic[B], B),
        "b_time_left": round(float(stage(env.task_dic[B])["time_left"]), 2),
    }


def main() -> int:
    print("robot0 abilities (1,0,0,0,1), robot1 (0,1,1,1,0)")
    print(f"A needs {A_NEED} at (0.60,0.50), B needs {B_NEED} at (0.60,0.90), stage time {STAGE_TIME}")
    print("Only robot 0 can serve either stage. It arrives at A, then is re-tasked to B.\n")
    print("== environment as it has always behaved ==")
    before = play(False)
    print("== with roster consistency enforced ==")
    after = play(True)
    print("== what the fix changes ==")
    for key in before:
        marker = "same" if before[key] == after[key] else "CHANGED"
        print(f"  {key:<44} as-is={str(before[key]):<34} fixed={str(after[key]):<34} {marker}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
