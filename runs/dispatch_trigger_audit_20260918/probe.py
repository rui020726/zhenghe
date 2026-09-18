"""Instrumentation-only probe for the dispatch-trigger audit.

Replays the binding baseline arm on the failure seeds and records, per tick, the
facts the committed traces do not carry: every robot's working/idle state and
eligibility, every task's stage gap as the environment holds it, what the
coordinator was shown and returned, and which agents the allocator skipped.

The probe only *reads* state after the fact (wrapped calls delegate to the
originals and then record); it changes no decision, which ``--verify`` checks by
comparing the produced decision trace against the committed one.

    D:/project_env/python.exe runs/dispatch_trigger_audit_20260918/probe.py [--verify] [seeds...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402

SEEDS = [20003, 20008, 20052, 20070, 20083]
COMMITTED = REPO / "runs" / "coordinator_baselines_20260917" / "heldout" / "arms" / "coalplan0.25sticky"
DECISION_FIELDS = (
    "step",
    "agent_id",
    "chosen_task",
    "actual_assignment",
    "needs_decision",
    "raw_choice",
    "post_mask_choice",
    "chosen_index",
)


def _snap(agent: dict) -> dict:
    return {
        "status": agent.get("status"),
        "nd": bool(agent.get("need_decision", False)),
        "task": agent.get("current_task"),
        "ws": agent.get("working_start_time"),
        "wt": agent.get("working_timeout_value"),
        "cd": {int(k): round(float(v), 2) for k, v in (agent.get("cooldown_tasks") or {}).items()},
        "pos": [round(float(v), 4) for v in agent.get("location", ())],
        "ab": [float(v) for v in agent.get("abilities", ())],
        "im": bool(agent.get("immediate_replan", False)),
    }


def _task_snap(env, external_of) -> dict:
    out = {}
    for tid, task in env.task_dic.items():
        index = min(int(task["current_stage_index"]), len(task["stages"]) - 1)
        stage = task["stages"][index]
        out[external_of.get(tid, str(tid))] = {
            "stage": int(task["current_stage_index"]),
            "members": sorted(int(m) for m in task["members"]),
            "req": [float(v) for v in stage["requirements"]],
            "gap": [float(v) for v in stage["status"]],
            "time_left": round(float(stage["time_left"]), 3),
            "deadline": round(float(task["deadline"]), 3),
            "expired": bool(task.get("expired")),
            "finished": bool(task.get("finished")),
        }
    return out


def _belief_snap(snapshot) -> dict:
    beliefs = {}
    for belief in snapshot.task_beliefs or ():
        beliefs[belief.task_id] = {
            "stage": int(belief.stage_index),
            "req": [int(v) for v in belief.remaining_requirements],
            "prog": round(float(belief.progress), 4),
            "rev": int(belief.revision),
            "fin": bool(belief.finished),
            "dl": round(float(belief.deadline), 3),
            "pos": [round(float(belief.position.x), 4), round(float(belief.position.y), 4)]
            if belief.position is not None
            else None,
        }
    return beliefs


def probe_seed(seed: int) -> Path:
    out = HERE / f"probe_s{seed}.jsonl"
    trace = HERE / f"probe_tr_s{seed}.jsonl"
    configuration = RunConfiguration(
        seed=seed,
        steps=1000,
        device="cpu",
        output=str(HERE / f"probe_rp_s{seed}.jsonl"),
        decision_trace=str(trace),
        task_checkpoint=str(REPO / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth"),
        mock=False,
        coordinator_policy="coalition_planner",
        sticky_commitments=True,
        use_coordinator=True,
        residual_scale=0.25,
        allocation_interval=5,
        coordination_interval=5,
        command_ttl=10.0,
        dt=1.0,
        detection_radius=5,
        search_agents=2,
        num_goals=6,
        grid_size=60,
        obstacle_density=0.15,
        rescue_species=3,
        rescue_per_species=2,
        actor_kv_mode="tokens",
        comm_delivery_probability=1.0,
        comm_delay_ticks=0,
        comm_subscribers="all",
        quiet=True,
        replay_stream=str(
            REPO / "runs" / "coordinator_baselines_20260917" / "heldout" / "streams" / f"s{seed}.json"
        ),
    )
    run = build_joint_run(configuration)
    env = run.task_adapter.env
    external_of = {v: k for k, v in run.task_adapter.internal_to_external.items()}

    coordinator_state: dict = {}
    allocator_state: dict = {}
    original_advise = run.orchestrator.coordinator.advise
    original_allocate = run.orchestrator.allocator.allocate

    def advise(state):
        shown = {}
        for snapshot in state.agents:
            shown[str(snapshot.agent_id)] = {
                "status": snapshot.status,
                "task": snapshot.current_task,
                "beliefs": _belief_snap(snapshot),
            }
        advice = original_advise(state)
        coordinator_state.update(
            {
                "shown": shown,
                "revision": int(advice.revision),
                "replan": sorted(advice.replan_agents),
                "reservations": [
                    {
                        "id": r.commitment_id,
                        "task": r.task_id,
                        "agent": r.agent_id,
                        "slot": r.ability_slot,
                        "dim": r.slot_dimension,
                        "until": round(float(r.valid_until), 3),
                    }
                    for r in advice.reservations.values()
                ],
            }
        )
        return advice

    def allocate(tasks, agents, advice, current_time):
        allocator_state["agents"] = {
            str(a.agent_id): {"status": a.status, "nd": bool(a.needs_decision), "task": a.current_task}
            for a in agents
        }
        allocator_state["replan"] = sorted(advice.replan_agents)
        commands = original_allocate(tasks, agents, advice, current_time)
        allocator_state["commands"] = [
            {"agent": c.agent_id, "task": c.task_id, "cmd": c.command_id} for c in commands
        ]
        return commands

    run.orchestrator.coordinator.advise = advise
    run.orchestrator.allocator.allocate = allocate

    original_step = run.orchestrator.step

    def step():
        report = original_step()
        row = {
            "step": report.step,
            "t": round(float(report.current_time), 3),
            "agents": {aid: _snap(agent) for aid, agent in env.agent_dic.items()},
            "tasks": _task_snap(env, external_of),
            "coordinator": dict(coordinator_state),
            "allocator": dict(allocator_state),
        }
        coordinator_state.clear()
        allocator_state.clear()
        out.open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
        return report

    run.orchestrator.step = step
    summary = run.run()
    print(
        f"seed {seed}: completed={summary.get('status_counts', {}).get('completed')} "
        f"expired={summary.get('status_counts', {}).get('expired')} -> {out.name}",
        flush=True,
    )
    return trace


def verify_seed(seed: int, probe_trace: Path) -> dict:
    """The probe run must reproduce the committed run's decisions exactly."""

    def decisions(path: Path) -> list[tuple]:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("event_type") != "decision":
                continue
            rows.append(tuple(row.get(field) for field in DECISION_FIELDS))
        return rows

    committed = decisions(COMMITTED / f"tr_coalplan0.25sticky_s{seed}.jsonl")
    produced = decisions(probe_trace)
    same = committed == produced
    committed_final = {
        row["target_id"]: (row["status"], row["failure_reason"], row["progress"])
        for line in (COMMITTED / f"rp_coalplan0.25sticky_s{seed}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if (row := json.loads(line)).get("event_type") == "final"
    }
    produced_final = {
        row["target_id"]: (row["status"], row["failure_reason"], row["progress"])
        for line in (HERE / f"probe_rp_s{seed}.jsonl").read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line)).get("event_type") == "final"
    }
    finals_same = committed_final == produced_final
    return {
        "seed": seed,
        "decision_rows_committed": len(committed),
        "decision_rows_produced": len(produced),
        "decisions_identical": same,
        "finals_identical": finals_same,
        "first_difference": None
        if same
        else next(
            (
                [i, a, b]
                for i, (a, b) in enumerate(zip(committed, produced))
                if a != b
            ),
            ["length", len(committed), len(produced)],
        ),
    }


def main(argv: list[str]) -> int:
    verify_only = "--verify" in argv
    argv = [a for a in argv if a != "--verify"]
    seeds = [int(value) for value in argv] or SEEDS
    results = []
    for seed in seeds:
        trace = HERE / f"probe_tr_s{seed}.jsonl"
        if not verify_only:
            probe_seed(seed)
        results.append(verify_seed(seed, trace))
    for row in results:
        print(json.dumps(row))
    ok = all(r["decisions_identical"] and r["finals_identical"] for r in results)
    print(f"instrumentation-only: {'VERIFIED' if ok else 'TRAJECTORY CHANGED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
