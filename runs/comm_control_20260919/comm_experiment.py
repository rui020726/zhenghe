"""Run the pre-registered communication conditions over the development seeds.

One row per (condition, seed): what the uplink did, what the robots came to believe, and what
the tasks became. Conditions come from ``conditions.py`` and are not edited against results;
the frozen allocation arm is the same in every row, so a difference between rows is a
difference in information timing and nothing else.

The runner wraps three collaborators to read costs that no decision depends on -- the
allocator's and coordinator's wall-clock time per call, and how many robot-steps were spent
idle -- and removes the wrappers in ``finally``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import comm_metrics
import common
import conditions

from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

HERE = common.HERE
ROWS = HERE / "rows"


def on_time_count(timelines: dict[str, Any]) -> int:
    """The frozen round's definition: finished, and not by missing its deadline."""
    return sum(
        1
        for timeline in timelines.values()
        if timeline.final_progress is not None
        and float(timeline.final_progress) >= 1.0
        and timeline.final_failure_reason != "deadline_expired"
    )


class RunProbe:
    """Wall-clock, activity and receiver-belief accounting around one episode.

    The belief sample is taken from the robots' own knowledge after each environment step and
    judged against the registry's live snapshot, so it counts *every* channel that informed
    them -- the uplink, the environment's line-of-sight refresh, gossip, arrival verification.
    The transcript fold then says how much of that the uplink alone would have carried.
    """

    def __init__(self, run: Any) -> None:
        self.run = run
        self.originals: dict[str, Any] = {}
        self.calls: dict[str, int] = {"allocate": 0, "advise": 0, "env_step": 0}
        self.seconds: dict[str, float] = {"allocate": 0.0, "advise": 0.0}
        self.idle_robot_steps = 0
        self.robot_steps = 0
        self.ages: list[float] = []
        self.stage_error_steps = 0
        self.requirement_error_steps = 0
        self.first_knowledge: dict[tuple[str, str], float] = {}
        self.unowned_active_steps = 0
        self.active_task_steps = 0
        self.closed_belief_steps = 0

    def __enter__(self) -> "RunProbe":
        allocator = self.run.task_adapter
        coordinator = self.run.orchestrator.coordinator
        environment = self.run.task_adapter.env
        self.originals = {
            "allocate": allocator.allocate,
            "advise": coordinator.advise,
            "env_step": environment.step,
        }
        probe = self

        def allocate(tasks, agents, advice, current_time):
            started = time.perf_counter()
            result = probe.originals["allocate"](tasks, agents, advice, current_time)
            probe.seconds["allocate"] += time.perf_counter() - started
            probe.calls["allocate"] += 1
            return result

        def advise(inputs):
            started = time.perf_counter()
            result = probe.originals["advise"](inputs)
            probe.seconds["advise"] += time.perf_counter() - started
            probe.calls["advise"] += 1
            return result

        def step(*args, **kwargs):
            result = probe.originals["env_step"](*args, **kwargs)
            probe.calls["env_step"] += 1
            statuses = [agent.get("status") for agent in environment.agent_dic.values()]
            probe.robot_steps += len(statuses)
            probe.idle_robot_steps += sum(1 for status in statuses if status == "idle")
            probe._sample_beliefs()
            return result

        allocator.allocate = allocate
        coordinator.advise = advise
        environment.step = step
        return self

    def _sample_beliefs(self) -> None:
        run = self.run
        environment = run.task_adapter.env
        adapter = run.task_adapter
        now = float(environment.current_time)
        active = {
            snapshot.task_id
            for snapshot in run.registry.active_snapshots()
        }
        informed: set[str] = set()
        for agent_id, knowledge in environment.agent_knowledge.items():
            for internal_id in knowledge["known_tasks"]:
                external = adapter.internal_to_external.get(int(internal_id))
                if external is None:
                    continue
                informed.add(external)
                key = (str(agent_id), external)
                if key not in self.first_knowledge:
                    self.first_knowledge[key] = now
                update_time = float(knowledge["task_last_update_time"].get(internal_id, 0.0))
                snapshot = run.registry.get(external)
                if snapshot is None or snapshot.status.terminal:
                    # A closed task the robot knows about is not stale information: it knows
                    # it is closed. Those steps are counted separately and excluded from age.
                    self.closed_belief_steps += 1
                    continue
                self.ages.append(now - update_time)
                self.stage_error_steps += int(
                    int(knowledge["task_stage_index"].get(internal_id, 0)) != snapshot.current_stage
                )
                believed = tuple(
                    int(value) for value in knowledge["task_current_status"].get(internal_id, ())
                )
                actual = tuple(int(value) for value in snapshot.remaining_requirements)
                self.requirement_error_steps += int(believed != actual and any(actual))
        self.active_task_steps += len(active)
        self.unowned_active_steps += len(active - informed)

    def __exit__(self, *exception: Any) -> bool:
        allocator = self.run.task_adapter
        allocator.allocate = self.originals["allocate"]
        self.run.orchestrator.coordinator.advise = self.originals["advise"]
        allocator.env.step = self.originals["env_step"]
        return False

    def belief_row(self) -> dict[str, Any]:
        arrivals = sorted(self.first_knowledge.values())
        return {
            "robot_target_beliefs": len(self.ages),
            "known_robot_target_pairs": len(self.first_knowledge),
            "robots_informed": len({robot for robot, _ in self.first_knowledge}),
            "belief_age_steps_mean": round(sum(self.ages) / len(self.ages), 3) if self.ages else None,
            "belief_age_steps_max": round(max(self.ages), 3) if self.ages else None,
            "time_to_first_knowledge_mean": round(sum(arrivals) / len(arrivals), 3) if arrivals else None,
            "time_to_first_knowledge_max": round(max(arrivals), 3) if arrivals else None,
            "stage_error_steps": self.stage_error_steps,
            "requirement_error_steps": self.requirement_error_steps,
            "active_task_steps": self.active_task_steps,
            "closed_belief_steps_skipped": self.closed_belief_steps,
            "unknown_to_everyone_steps": self.unowned_active_steps,
        }

    def as_row(self) -> dict[str, Any]:
        return {
            "allocation_calls": self.calls["allocate"],
            "coordinator_calls": self.calls["advise"],
            "environment_steps": self.calls["env_step"],
            "allocation_seconds": round(self.seconds["allocate"], 4),
            "coordinator_seconds": round(self.seconds["advise"], 4),
            "allocation_milliseconds_per_call": (
                round(1000.0 * self.seconds["allocate"] / self.calls["allocate"], 4)
                if self.calls["allocate"]
                else None
            ),
            "robot_steps": self.robot_steps,
            "idle_robot_steps": self.idle_robot_steps,
            "idle_share": round(self.idle_robot_steps / max(1, self.robot_steps), 5),
        }


def read_rows(directory: Path) -> list[dict[str, Any]]:
    """Every row already written, across all shards."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob("rows_shard*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def run_one(condition: conditions.Condition, seed: int, *, directory: Path) -> dict[str, Any]:
    configuration = common.base_configuration(
        seed,
        directory=directory,
        arm="C",
        comm_transcript=str(directory / f"ct_C_{condition.name}_s{seed}.jsonl"),
        **condition.configuration_overrides(),
    )
    started = time.time()
    run = build_joint_run(RunConfiguration(**configuration))
    relay = run.task_adapter.comm_relay
    # The construction assertion: the object that was built must be the one the condition
    # named, not just the configuration that asked for it.
    if relay.policy.name != condition.policy:
        raise RuntimeError(
            f"{condition.name}: built a {relay.policy.name!r} relay, asked for {condition.policy!r}"
        )
    if dict(relay.policy.parameters()) != dict(condition.params):
        raise RuntimeError(
            f"{condition.name}: built parameters {relay.policy.parameters()}, "
            f"asked for {condition.params}"
        )
    if relay.transport_parameters()["sends_per_tick"] != condition.sends_per_tick:
        raise RuntimeError(f"{condition.name}: the emission budget did not reach the relay")
    if "+memfix" not in run.run_signature["arm"]:
        raise RuntimeError(f"{condition.name}: the run is not on the corrected environment")

    with RunProbe(run) as probe:
        summary = run.run()
    elapsed = round(time.time() - started, 1)

    records = load_records(Path(configuration["output"]))
    timelines = build_timelines(records, arm="C", seed=seed)
    transcript = comm_metrics.load_transcript(configuration["comm_transcript"])
    on_time = on_time_count(timelines)
    decisions = [
        json.loads(line)
        for line in Path(configuration["decision_trace"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    replanned = [row for row in decisions if row.get("event_type") == "decision" and row.get("replanned")]
    environment = run.task_adapter.env
    row = {
        "condition": condition.name,
        "role": condition.role,
        "information": condition.information,
        "seed": seed,
        "arm": "C",
        "signature_arm": run.run_signature["arm"],
        "config_hash": run.run_signature["config_hash"],
        "code_commit": run.run_signature["code_commit"],
        "stream_sha256": run.run_signature["stream"]["sha256"],
        "seconds": elapsed,
        "run_steps": int(summary.get("steps", 0)),
        "run_time": round(float(summary.get("current_time", 0.0)), 3),
        "transport": {
            **comm_metrics.transport_accounting(transcript),
            **{key: value for key, value in summary["communication"].items() if key != "subscribers"},
        },
        # What the robots actually believed, counting every channel that reached them.
        "belief_all_channels": probe.belief_row(),
        # The same picture from the uplink's own records: the part a send policy controls.
        "belief_via_uplink": comm_metrics.belief_age_accounting(
            transcript,
            horizon=float(summary.get("steps", 0)),
            robots=[str(agent_id) for agent_id in environment.agent_dic],
            truth=comm_metrics.truth_timeline(records),
            active_until=comm_metrics.terminal_times(records),
        ),
        "performance": {
            "targets": len(timelines),
            "on_time": on_time,
            "on_time_rate": round(on_time / max(1, len(timelines)), 6),
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
            **comm_metrics.task_outcomes(timelines),
            "travel_dist_total": round(
                float(sum(float(agent["travel_dist"]) for agent in environment.agent_dic.values())), 3
            ),
        },
        "decisions": {
            **probe.as_row(),
            "replan_decisions": len(replanned),
            "assignment_commands": int(summary.get("assignment_commands", 0)),
            "rejected_assignments": int(summary.get("rejected_assignments", 0)),
            "resource_utilization": summary.get("resource_utilization"),
        },
        "artefacts": {
            "lifecycle_log": str(configuration["output"]),
            "transcript": str(configuration["comm_transcript"]),
            "decision_trace": str(configuration["decision_trace"]),
        },
    }
    return row


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="*", default=[condition.name for condition in conditions.CONDITIONS])
    parser.add_argument("--seeds", type=int, nargs="*", default=list(common.DEV_SEEDS[:12]))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--rows", default=str(ROWS / f"rows_shard{{shard}}.jsonl"))
    args = parser.parse_args(argv)

    rows_path = Path(args.rows.format(shard=args.shard))
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    directory = ROWS / f"shard{args.shard}"
    directory.mkdir(parents=True, exist_ok=True)
    # The resume view is global. Each shard writes its own file, so asking only this file what
    # is done would have every shard re-run the cells another shard had already taken.
    done = {(row["condition"], int(row["seed"])) for row in read_rows(ROWS)}
    order = {name: index for index, name in enumerate(conditions.BY_NAME)}
    cells = sorted(
        (
            (name, int(seed))
            for name in args.conditions
            for seed in args.seeds
            if (name, int(seed)) not in done
        ),
        key=lambda cell: (order.get(cell[0], len(order)), cell[1]),
    )
    cells = [
        cell
        for index, cell in enumerate(cells)
        if index % max(1, args.shards) == args.shard % max(1, args.shards)
    ]
    by_name = conditions.BY_NAME
    failures = 0
    with rows_path.open("a", encoding="utf-8") as handle:
        for name, seed in cells:
            condition = by_name[name]
            try:
                row = run_one(condition, seed, directory=directory)
            except Exception as error:  # a technical failure is a row, not a lost batch
                failures += 1
                row = {
                    "condition": name,
                    "seed": seed,
                    "technical_failure": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(limit=6),
                }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            performance = row.get("performance", {})
            transport = row.get("transport", {})
            print(
                f"[{name} s{seed}] on_time={performance.get('on_time')}/{performance.get('targets')} "
                f"emissions={transport.get('emissions')} packets={transport.get('packets_sent')} "
                f"bytes={transport.get('payload_bytes_total')} "
                f"age={row.get('belief_all_channels', {}).get('belief_age_steps_mean')} "
                f"{row.get('seconds')}s"
                + (f" FAILED: {row.get('technical_failure')}" if "technical_failure" in row else ""),
                flush=True,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
