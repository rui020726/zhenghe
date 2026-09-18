"""The corrected-environment independent block: record, replay, measure, judge.

Two arms on one environment. Both read the same recorded detection stream per seed, so the
only difference between the paired numbers is the controller. Every replay carries its own
correctness instrumentation, because on this project a performance number without the
membership ledger next to it has already proved meaningless once.

    D:/project_env/python.exe runs/corrected_env_final_20260918/corrected_block.py --preflight
    ... corrected_block.py --record --worker 0 --workers 6
    ... corrected_block.py --replay D --worker 0 --workers 6
    ... corrected_block.py --aggregate

Raw streams and lifecycle logs stay out of git; ``BLOCK_ROWS.json`` keeps one row per episode
with the digest of the stream it replayed and the artefacts it wrote.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any
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
for path in (REPO, HERE, REPO / "runs" / "coalition_validation_20260918"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import arms  # noqa: E402
from integrated_sar.coordinator import CoalitionPlannerCoordinator  # noqa: E402
from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

EPS = 1e-9
SEEDS_FILE = HERE / "HELDOUT_SEEDS_V2.json"
# Where episodes are written. The technical smoke never shares a directory with the block, so
# a smoke row cannot be counted as an episode of the independent test.
RUN_DIR = arms.OUT
RECORD_CONFIGURATION = {
    "steps": arms.STEPS,
    "device": "cpu",
    "piloc_checkpoint": str(arms.PIL_CHECKPOINT),
    "task_checkpoint": str(arms.CTDE_CHECKPOINT),
    "mock": False,
    "use_coordinator": False,
    "search_agents": 2,
    "num_goals": 6,
    "grid_size": 60,
    "obstacle_density": 0.15,
    "detection_radius": 5,
    "deadline_tightness": (0.5, 2.0),
    "dt": 1.0,
    "rescue_species": 3,
    "rescue_per_species": 2,
    "actor_kv_mode": "tokens",
    "comm_delivery_probability": 1.0,
    "comm_delay_ticks": 0,
    "comm_subscribers": "all",
    "quiet": True,
    # The block's own environment: a stream recorded under the old bookkeeping would carry
    # the old bookkeeping's arrivals into both arms.
    "task_membership_consistency": True,
}


def git_head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()


def seeds() -> list[int]:
    return [int(seed) for seed in json.loads(SEEDS_FILE.read_text(encoding="utf-8"))["seeds"]]


def stream_path(seed: int) -> Path:
    return arms.STREAMS / f"s{seed}.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(part, path)


# ---------------------------------------------------------------- recording
def record_one(seed: int) -> dict:
    arms.STREAMS.mkdir(parents=True, exist_ok=True)
    (HERE / "record_logs").mkdir(parents=True, exist_ok=True)
    stream = stream_path(seed)
    if stream.exists():
        return {"seed": seed, "skipped": True, "sha256": sha256(stream)}
    configuration = RunConfiguration(
        seed=seed,
        output=str(HERE / "record_logs" / f"rec_s{seed}.jsonl"),
        decision_trace=str(HERE / "record_logs" / f"rec_tr_s{seed}.jsonl"),
        record_stream=str(stream),
        **RECORD_CONFIGURATION,
    )
    started = time.time()
    summary = build_joint_run(configuration).run()
    if not stream.exists():
        raise AssertionError(f"seed {seed}: the recorder produced no stream")
    payload_stream = json.loads(stream.read_text(encoding="utf-8"))
    payload = {
        "seed": seed,
        "sha256": sha256(stream),
        "frames": int(len(payload_stream.get("frames", []))),
        "detections": int(
            sum(len(frame.get("detections", [])) for frame in payload_stream.get("frames", []))
        ),
        "steps_recorded": int(payload_stream.get("steps_recorded", 0)),
        # What says an episode ran to its horizon: the summary carries the steps taken and the
        # environment clock, and nothing else does. "final_status" is not a field of it.
        "run_steps": int(summary.get("steps", 0)),
        "run_time": round(float(summary.get("current_time", 0.0)), 3),
        "targets_discovered_by_registry": int(summary.get("targets_discovered", 0)),
        "seconds": round(time.time() - started, 1),
    }
    write_json(HERE / "records" / f"record_s{seed}.json", payload)
    print(
        f"[record s{seed}] frames={payload['frames']} detections={payload['detections']} "
        f"sha={payload['sha256'][:10]}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------- replay + measurement
def measure(run: object, arm_label: str, seed: int, summary: dict, hooks: dict) -> dict:
    configuration = run.configuration
    directory = Path(configuration.output).parent
    records = load_records(Path(configuration.output))
    timelines = build_timelines(records, arm=arm_label, seed=seed)
    counts = summary.get("status_counts", {}) or {}
    on_time = [
        timeline
        for timeline in timelines.values()
        if timeline.final_progress >= 1.0 and timeline.final_failure_reason != "deadline_expired"
    ]
    completion_times = [
        float(point.timestamp)
        for timeline in on_time
        for point in timeline.progress
        if point.progress is not None and float(point.progress) >= 1.0
    ]
    rejections: dict[str, int] = {}
    for row in records:
        if row.get("event_type") == "assignment_rejected":
            reason = str(row.get("failure_reason"))
            rejections[reason] = rejections.get(reason, 0) + 1
    env = run.task_adapter.env
    decisions = [
        json.loads(line)
        for line in Path(configuration.decision_trace).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    replanned = [row for row in decisions if row.get("event_type") == "decision" and row.get("replanned")]
    return {
        "seed": seed,
        "arm": arm_label,
        "stream_sha256": sha256(stream_path(seed)),
        "signature_arm": run.run_signature["arm"],
        "code_commit": run.run_signature["code_commit"],
        "config_hash": run.run_signature["config_hash"],
        # What says an episode ran to its horizon: the summary carries the steps taken and the
        # environment clock, and nothing else does. "final_status" is not a field of it.
        "run_steps": int(summary.get("steps", 0)),
        "run_time": round(float(summary.get("current_time", 0.0)), 3),
        "targets_discovered_by_registry": int(summary.get("targets_discovered", 0)),
        # ---- performance, per episode
        "detections": len(timelines),
        "on_time": len(on_time),
        "on_time_rate": round(len(on_time) / max(1, len(timelines)), 6),
        "expired": int(counts.get("expired", 0)),
        "deadline_expired_targets": sum(
            1
            for t in timelines.values()
            if t.final_failure_reason == "deadline_expired"
        ),
        "unresolved": sum(
            1
            for t in timelines.values()
            if not (t.final_progress is not None and float(t.final_progress) >= 1.0)
            and t.final_failure_reason != "deadline_expired"
        ),
        "completed_env_tasks": int(counts.get("completed", 0)),
        "mean_completion_time": round(float(np.mean(completion_times)), 3) if completion_times else None,
        "travel_dist_total": round(
            float(sum(float(agent["travel_dist"]) for agent in env.agent_dic.values())), 3
        ),
        "environment_ticks": hooks["environment_ticks"],
        "robot_ticks": hooks["robot_ticks"],
        "idle_robot_ticks": hooks["idle_robot_ticks"],
        "longest_idle_streak": round(float(summary.get("longest_idle_streak", 0.0)), 3),
        "coverage_resets": int(sum(t.progress_setbacks for t in timelines.values())),
        "lost_work": round(float(sum(t.discarded_work_steps for t in timelines.values())), 3),
        "replan_decisions": len(replanned),
        # ---- correctness, per episode
        "false_credit_task_ticks": hooks["false_credit_task_ticks"],
        "false_credit_capability_units": round(hooks["false_credit_units"], 3),
        "legacy_ghost_task_ticks": hooks["legacy_ghost_task_ticks"],
        "dual_roster_robot_ticks": hooks["dual_roster_robot_ticks"],
        "closing_released_robot_assigned_elsewhere": hooks[
            "closing_released_robot_assigned_elsewhere"
        ],
        "executed_releases_that_uncovered_the_stage": hooks[
            "executed_releases_that_uncovered_the_stage"
        ],
        "duplicate_command_reset": hooks["duplicate_command_reset"],
        "assignment_to_task_already_closed_in_truth": hooks[
            "assignment_to_task_already_closed_in_truth"
        ],
        "state_written_on_a_closed_task": hooks["state_written_on_a_closed_task"],
        "rejections_by_reason": dict(sorted(rejections.items())),
        "releases_published": hooks["releases_published"],
        "releases_executed": hooks["releases_executed"],
        "releases_without_future_information": hooks["releases_without_future_information"],
        "release_refusals_by_planner": dict(sorted(hooks["planner_refusals"].items())),
        "release_refusal_recovered": len(hooks["release_refusal_recovery_ticks"]),
        "release_refusal_unrecovered": hooks["release_refusals_unrecovered"],
        "release_refusal_recovery_ticks_mean": (
            round(float(np.mean(hooks["release_refusal_recovery_ticks"])), 3)
            if hooks["release_refusal_recovery_ticks"]
            else None
        ),
        "release_refusal_recovery_ticks_max": (
            round(float(np.max(hooks["release_refusal_recovery_ticks"])), 3)
            if hooks["release_refusal_recovery_ticks"]
            else None
        ),
        "report_rows": len(records),
    }


def replay_one(arm_label: str, seed: int) -> dict:
    arm = arms.ARMS[arm_label]
    directory = RUN_DIR / arm_label
    configuration_stream = stream_path(seed)
    run = arm.build(seed, configuration_stream, directory)
    env = run.task_adapter.env
    inner_task_update = env.task_update
    inner_step = run.orchestrator.step
    inner_advise = CoalitionPlannerCoordinator.advise
    apply_commands = run.gate.apply_assignments
    inner_apply_action = env.apply_action
    hooks = {
        "environment_ticks": 0,
        "robot_ticks": 0,
        "idle_robot_ticks": 0,
        "false_credit_task_ticks": 0,
        "false_credit_units": 0.0,
        "legacy_ghost_task_ticks": 0,
        "dual_roster_robot_ticks": 0,
        "closing_released_robot_assigned_elsewhere": 0,
        "executed_releases_that_uncovered_the_stage": 0,
        "duplicate_command_reset": 0,
        "assignment_to_task_already_closed_in_truth": 0,
        "state_written_on_a_closed_task": 0,
        "releases_published": 0,
        "releases_executed": 0,
        "releases_without_future_information": 0,
        "planner_refusals": defaultdict(int),
        "release_refusal_recovery_ticks": [],
        "release_refusals_unrecovered": 0,
    }
    accepted_releases: set[tuple[int, str]] = set()
    # task id -> (roster size, clock) the first time it was seen finished
    closed_state: dict[int, tuple[int, float]] = {}
    # agent -> (source internal id, time the boundary refused to take it off that task)
    awaiting_recovery: dict[str, tuple[int, float]] = {}
    # task id -> (roster at the previous call, stage then, member -> assignment then)
    previous_state: dict[int, tuple[list[int], int, dict[int, int]]] = {}

    def rows_of(tid, task):
        rows = []
        for member in [int(value) for value in task["members"]]:
            agent = env.agent_dic.get(member)
            if agent is None:
                rows.append({"member": member, "status": "missing-agent", "on_site": False})
                continue
            rows.append(
                {
                    "member": member,
                    "on_site": bool(
                        agent.get("status") == "working"
                        and int(agent.get("current_task", -1)) == tid
                    ),
                    "abilities": [int(v) for v in agent["abilities"]],
                }
            )
        return rows

    def task_update():
        for tid, task in env.task_dic.items():
            if task.get("finished"):
                after = closed_state.get(int(tid))
                roster = [int(value) for value in task["members"]]
                clock = float(arms_module_stage(task)["time_left"])
                if after is not None and (len(roster) > after[0] or clock < after[1]):
                    # A finished task must not gain a member or spend work: this is the one
                    # shape that would be an illegal execution rather than a wasted trip.
                    hooks["state_written_on_a_closed_task"] += 1
                continue
            roster = [int(value) for value in task["members"]]
            roster_at_entry = list(roster)
            requirement = np.asarray(arms_module_stage(task)["requirements"], dtype=float)
            stage_now = int(task["current_stage_index"])
            previous = previous_state.get(int(tid))
            if previous is not None:
                for member in [item for item in previous[0] if item not in roster]:
                    remaining = [item for item in roster if item != member]
                    with_member = env.get_abilities(remaining + [member])
                    without = (
                        env.get_abilities(remaining)
                        if remaining
                        else np.zeros_like(requirement)
                    )
                    lowered = float(
                        np.sum(np.minimum(requirement, with_member))
                        - np.sum(np.minimum(requirement, without))
                    )
                    if lowered <= EPS:
                        continue
                    if (int(tid), str(member)) in accepted_releases:
                        hooks["executed_releases_that_uncovered_the_stage"] += 1
                        accepted_releases.discard((int(tid), str(member)))
                for member in [item for item in previous[0] if item not in roster]:
                    assignment = previous[2].get(member)
                    if assignment is not None and int(assignment) != int(tid):
                        # A task that closed or moved on let go of a robot whose assignment
                        # was another task: it was only ever listed there as residue.
                        hooks["closing_released_robot_assigned_elsewhere"] += 1
            assignments = {
                int(member): int(env.agent_dic[int(member)].get("current_task", -1))
                for member in roster_at_entry
                if int(member) in env.agent_dic
            }
            previous_state[int(tid)] = (roster, stage_now, assignments)
        before = {
            tid: {
                "clock": float(arms_module_stage(task)["time_left"]),
                "stage": int(task["current_stage_index"]),
                "requirement": np.asarray(arms_module_stage(task)["requirements"], dtype=float),
                "rows": rows_of(tid, task),
            }
            for tid, task in env.task_dic.items()
            if not task.get("finished")
        }
        result = inner_task_update()
        for tid, snapshot in before.items():
            task = env.task_dic.get(tid)
            if task is None or task.get("finished"):
                continue
            now_stage = int(task["current_stage_index"])
            now_clock = float(arms_module_stage(task)["time_left"])
            if now_stage != snapshot["stage"] or now_clock >= snapshot["clock"] - EPS:
                continue
            present = [row["abilities"] for row in snapshot["rows"] if row["on_site"]]
            supply = (
                np.sum(np.array(present, dtype=float), axis=0)
                if present
                else np.zeros_like(snapshot["requirement"])
            )
            gap = np.maximum(snapshot["requirement"] - supply, 0.0)
            if float(gap.sum()) > EPS:
                hooks["false_credit_task_ticks"] += 1
                hooks["false_credit_units"] += float(gap.sum())
        return result

    def step():
        for tid, task in env.task_dic.items():
            if task.get("finished") and int(tid) not in closed_state:
                closed_state[int(tid)] = (
                    len(task["members"]),
                    float(arms_module_stage(task)["time_left"]),
                )
        clocks = {
            tid: float(arms_module_stage(task)["time_left"])
            for tid, task in env.task_dic.items()
            if not task.get("finished")
        }
        report = inner_step()
        hooks["environment_ticks"] += 1
        hooks["robot_ticks"] += len(env.agent_dic)
        hooks["idle_robot_ticks"] += sum(
            1 for agent in env.agent_dic.values() if agent.get("status") == "idle"
        )
        for tid, clock in clocks.items():
            task = env.task_dic.get(tid)
            if task is None or task.get("finished"):
                continue
            if float(arms_module_stage(task)["time_left"]) >= clock - EPS:
                continue
            rows = rows_of(tid, task)
            requirement = np.asarray(arms_module_stage(task)["requirements"], dtype=float)
            present = [row["abilities"] for row in rows if row["on_site"]]
            supply = np.sum(np.array(present, dtype=float), axis=0) if present else np.zeros_like(requirement)
            if float(np.maximum(requirement - supply, 0.0).sum()) > EPS:
                hooks["legacy_ghost_task_ticks"] += 1
        seen: dict[int, int] = {}
        for task in env.task_dic.values():
            for member in task["members"]:
                seen[int(member)] = seen.get(int(member), 0) + 1
        hooks["dual_roster_robot_ticks"] += sum(1 for count in seen.values() if count > 1)
        for agent_id, (source, refused_at) in list(awaiting_recovery.items()):
            agent = env.agent_dic.get(int(agent_id))
            if agent is not None and int(agent.get("current_task", -1)) != source:
                hooks["release_refusal_recovery_ticks"].append(
                    round(float(env.current_time) - refused_at, 3)
                )
                del awaiting_recovery[agent_id]
        return report

    def apply_action(agent_id, action_index, action_map):
        agent = env.agent_dic.get(agent_id)
        kind, tid = action_map[action_index] if 0 <= action_index < len(action_map) else ("depot", None)
        if agent is not None and kind == "task" and tid is not None:
            task = env.task_dic.get(int(tid))
            if task is not None and task.get("finished"):
                # Not a corrupted write: the arrival check is what the architecture uses to
                # turn down a task that closed while the robot was on its way. Counted as the
                # wasted trip it is, and separated from the test below, which looks for an
                # actual write against a closed task.
                hooks["assignment_to_task_already_closed_in_truth"] += 1
            working_elsewhere = (
                agent.get("status") == "working" and int(agent.get("current_task", -1)) == int(tid)
            )
            if working_elsewhere:
                clock_before = (
                    float(arms_module_stage(task)["time_left"]) if task is not None else None
                )
                roster_before = list(task["members"]) if task is not None else []
                result = apply_one(agent_id, action_index, action_map)
                if task is not None and (
                    float(arms_module_stage(task)["time_left"]) != clock_before
                    or list(task["members"]) != roster_before
                ):
                    # A command sending a robot back onto the task it is already working must
                    # not restart the clock or shuffle the roster. The corrected environment
                    # makes it a true no-op; this counts any deviation.
                    hooks["duplicate_command_reset"] += 1
                return result
        return apply_one(agent_id, action_index, action_map)

    def advise_with_counters(self, state):
        advice = inner_advise(self, state)
        metadata = advice.metadata or {}
        hooks["releases_published"] += int(metadata.get("redundant_releases", 0))
        hooks["releases_without_future_information"] += int(
            metadata.get("releases_without_future_information", 0)
        )
        for reason, value in (metadata.get("release_refusals") or {}).items():
            hooks["planner_refusals"][str(reason)] += int(value)
        return advice

    def apply_with_ledger(commands):
        known = {row.command_id for row in run.gate.rejections}
        apply_commands(commands)
        refused = {row.command_id for row in run.gate.rejections} - known
        for issued in commands:
            metadata = issued.metadata or {}
            source = metadata.get("release_from_task")
            if not source or issued.command_id in refused:
                continue
            hooks["releases_executed"] += 1
            internal = run.task_adapter.external_to_internal.get(str(source))
            if internal is not None:
                accepted_releases.add((int(internal), str(issued.agent_id)))
        for row in run.gate.rejections:
            if row.command_id in known or not str(row.reason).startswith("stale_release"):
                continue
            internal = run.task_adapter.external_to_internal.get(str(row.release_from_task))
            if internal is None:
                continue
            # How long the refused robot stayed where the refused command left it. The
            # refusal is only conservative if the robot is not then forgotten about.
            awaiting_recovery.setdefault(str(row.agent_id), (int(internal), float(row.rejected_at)))

    def arms_module_stage(task):
        return task["stages"][min(int(task["current_stage_index"]), len(task["stages"]) - 1)]

    apply_one = inner_apply_action
    env.apply_action = apply_action
    env.task_update = task_update
    run.orchestrator.step = step
    run.gate.apply_assignments = apply_with_ledger
    CoalitionPlannerCoordinator.advise = advise_with_counters
    try:
        started = time.time()
        summary = run.run()
    finally:
        # No hook survives the episode: a worker replays several seeds in one process, and a
        # second wrapper stacked on the first would count the same tick twice.
        CoalitionPlannerCoordinator.advise = inner_advise
        env.apply_action = inner_apply_action
        env.task_update = inner_task_update
        run.orchestrator.step = inner_step
        run.gate.apply_assignments = apply_commands
        if run.decision_trace is not None:
            run.decision_trace.close()
    hooks["release_refusals_unrecovered"] = len(awaiting_recovery)
    payload = measure(run, arm_label, seed, summary, hooks)
    payload["seconds"] = round(time.time() - started, 1)
    write_json(directory / f"episode_{arm_label}_s{seed}.json", payload)
    print(
        f"[{arm_label} s{seed}] on_time={payload['on_time']}/{payload['detections']} "
        f"expired={payload['expired']} false_credit={payload['false_credit_task_ticks']} "
        f"dual_roster={payload['dual_roster_robot_ticks']} bad_release="
        f"{payload['executed_releases_that_uncovered_the_stage']} "
        f"dup_reset={payload['duplicate_command_reset']} closed_trip={payload['assignment_to_task_already_closed_in_truth']} closed_write={payload['state_written_on_a_closed_task']} "
        f"releases={payload['releases_executed']}/{payload['releases_published']}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------- preflight
def preflight(seeds_subset: list[int]) -> int:
    head = arms.assert_frozen_tree()
    print(f"freeze {head} clean, HEAD {git_head()[:12]}")
    for label, arm in arms.ARMS.items():
        seed = seeds_subset[0]
        run = arm.build(seed, stream_path(seed), HERE / "preflight")
        signature = run.run_signature
        print(
            f"  {label}: built arm={signature['arm']} env membership_consistency="
            f"{getattr(run.task_adapter.env, 'membership_consistency', None)!r} "
            f"gate stage={run.gate.release_state_of is not None} "
            f"roster={run.gate.release_roster_of is not None} "
            f"config_hash={str(signature['config_hash'])[:12]}"
        )
    for seed in seeds_subset:
        stream = stream_path(seed)
        if not stream.exists():
            print(f"  missing stream for {seed}")
            return 1
    digests = {sha256(stream_path(seed)) for seed in seeds_subset}
    print(f"  streams for {len(seeds_subset)} seeds, {len(digests)} distinct digests")
    return 0


# ---------------------------------------------------------------- aggregation
HORIZON_FIELDS = ("run_steps", "run_time")
# The gate reads GATED_FIELDS. The wasted trip toward a task the environment already closed
# is reported in the same table but does not gate: it happens in the arm with no release
# mechanism too, the arrival check turns it down before any state is written, and the state
# write test below is what "illegal" would mean here.
GATED_FIELDS = (
    "false_credit_task_ticks",
    "dual_roster_robot_ticks",
    "closing_released_robot_assigned_elsewhere",
    "executed_releases_that_uncovered_the_stage",
    "duplicate_command_reset",
    "state_written_on_a_closed_task",
)
REPORTED_ONLY_FIELDS = ("legacy_ghost_task_ticks", "assignment_to_task_already_closed_in_truth")
CORRECTNESS_FIELDS = GATED_FIELDS + REPORTED_ONLY_FIELDS + (
    "releases_without_future_information",
)


def aggregate(expected_seeds: list[int] | None = None) -> int:
    rows: dict[str, list[dict]] = defaultdict(list)
    for label in arms.ARMS:
        for path in sorted((RUN_DIR / label).glob(f"episode_{label}_s*.json")):
            rows[label].append(json.loads(path.read_text(encoding="utf-8")))
    by_seed = {
        label: {row["seed"]: row for row in values} for label, values in rows.items()
    }
    paired = sorted(set(by_seed.get("C", {})) & set(by_seed.get("D", {})))
    missing = {
        label: sorted(set(values) - set(paired)) for label, values in by_seed.items()
    }
    problems: list[str] = []
    for label, values in by_seed.items():
        for seed, row in values.items():
            if int(row.get("run_steps", 0)) != arms.STEPS or float(row.get("run_time", 0.0)) < arms.STEPS - 0.5:
                problems.append(
                    f"{label} s{seed}: horizon not reached (steps={row.get('run_steps')}, "
                    f"time={row.get('run_time')})"
                )
            if not arms.same_production(str(row.get("code_commit"))):  # same rule as stats_v2
                problems.append(f"{label} s{seed}: commit {row.get('code_commit')}")
    if missing.get("C") or missing.get("D"):
        problems.append(f"unpaired seeds: { {k: v for k, v in missing.items() if v} }")
    if expected_seeds is not None:
        # The manifest is the task list: nothing missing, nothing extra, no substitution.
        want = sorted(int(seed) for seed in expected_seeds)
        for label, values in by_seed.items():
            got = sorted(values)
            if got != want:
                problems.append(
                    f"{label}: {len(got)} episodes vs {len(want)} scheduled "
                    f"(missing {sorted(set(want) - set(got))[:5]}, "
                    f"extra {sorted(set(got) - set(want))[:5]})"
                )
    stream_pairs = {
        (row["seed"], row["stream_sha256"])
        for values in by_seed.values()
        for row in values.values()
    }
    if len({seed for seed, _ in stream_pairs}) != len(stream_pairs):
        problems.append("an arm replayed a different stream than its pair for some seed")

    summary: dict[str, Any] = {
        "episodes": {label: len(values) for label, values in rows.items()},
        "paired_seeds": len(paired),
        "technical_failures": problems[:40],
        "technical_failure_count": len(problems),
    }
    for label, values in rows.items():
        summary[label] = {
            field: int(sum(row[field] for row in values)) for field in CORRECTNESS_FIELDS
        }
        summary[label]["episodes"] = len(values)
        summary[label]["detections"] = int(sum(row["detections"] for row in values))
        summary[label]["on_time"] = int(sum(row["on_time"] for row in values))
        summary[label]["expired"] = int(sum(row["expired"] for row in values))
        summary[label]["unresolved"] = int(sum(row["unresolved"] for row in values))
        summary[label]["rejections_by_reason"] = {
            reason: sum(row["rejections_by_reason"].get(reason, 0) for row in values)
            for reason in sorted({r for row in values for r in row["rejections_by_reason"]})
        }
        summary[label]["release_refusal_unrecovered"] = int(
            sum(row["release_refusal_unrecovered"] for row in values)
        )
        recovered = [
            row["release_refusal_recovery_ticks_mean"]
            for row in values
            if row["release_refusal_recovery_ticks_mean"] is not None
        ]
        summary[label]["release_refusal_recovery_ticks_mean"] = (
            round(float(np.mean(recovered)), 3) if recovered else None
        )
        summary[label]["release_refusals_by_planner"] = {
            reason: sum(row["release_refusals_by_planner"].get(reason, 0) for row in values)
            for reason in sorted(
                {r for row in values for r in row["release_refusals_by_planner"]}
            )
        }
    write_json(HERE / "BLOCK_SUMMARY.json", summary)
    print(json.dumps(summary, indent=1))
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--replay", choices=tuple(arms.ARMS))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--workers", type=int, default=arms.WORKERS)
    args = parser.parse_args(argv)
    global RUN_DIR
    pool = seeds()
    if args.smoke:
        # Technical smoke only: never from the independent block's own seed list.
        RUN_DIR = HERE / "block_smoke"
        pool = [600000 + index for index in range(args.smoke)]
    if args.preflight:
        return preflight(pool)
    if args.aggregate:
        return aggregate(None if args.smoke else seeds())
    if args.record or args.replay:
        label = args.replay
        jobs = [(label, seed) for seed in pool]
        if args.worker is not None:
            for position, (label, seed) in enumerate(jobs):
                if position % args.workers != args.worker:
                    continue
                if jobs[0][0] is None:
                    if stream_path(seed).exists():
                        continue
                    record_one(seed)
                else:
                    if (RUN_DIR / str(label) / f"episode_{label}_s{seed}.json").exists():
                        continue
                    replay_one(str(label), seed)
            return 0
        processes = [
            subprocess.Popen(
                [
                    sys.executable, str(Path(__file__).resolve()),
                    *(["--replay", str(label)] if label else ["--record"]),
                    *(["--smoke", str(args.smoke)] if args.smoke else []),
                    "--worker", str(index), "--workers", str(args.workers),
                ],
                cwd=str(REPO),
            )
            for index in range(args.workers)
        ]
        codes = [process.wait() for process in processes]
        if any(codes):
            print(f"worker failures: {codes}", file=sys.stderr)
            return 1
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
