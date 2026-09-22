"""This round's episode runner: the frozen runner's ledger, plus a read-only candidate probe.

Two things are new relative to round 7's copy, and each has to prove it changed nothing else.

1. **A stream from this round's own layer.** The replay configuration is the frozen C arm's with
   ``num_goals`` and ``replay_stream`` set, so ``assert_bundle_matches_configuration`` refuses any
   arm whose goal count differs from the recorded world. A missing stream is an error, never a
   silent fallback to another round's data.
2. **A probe on the policy seam.** The budget-competition measurement needs the candidate list the
   relay assembled -- charges, payability, remaining credit -- which is only observable there. The
   probe reads ``SendObservation`` and returns the rule's own answer object untouched (round 6 lost
   a whole counterfactual run to a probe that rebuilt a tuple as a set), and it never sends anything
   on its own.

The two proofs are the same run: an arm replayed through this runner *with* the probe must reproduce
the archived round-5 row for the same arm and stream, which were produced with no probe at all. If
both hold, the probe is transparent and the runner is not a drift; if only the task outcome matches,
neither claim is made.

    python pressure_env.py     # prints the agreement verdict for archived streams
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pressure_common as common

from integrated_sar.comm_policy import build_comm_policy, parse_parameters
from integrated_sar.contracts import TaskStatus
from integrated_sar.expired_attribution import build_timelines
from integrated_sar.joint_driver import RunConfiguration, build_joint_run
from integrated_sar.lifecycle_metrics import load_records

#: The three configuration fields an arm may legitimately differ on (protocol §7 L0).
ALLOWED_DIFFERENCE_KEYS = ("comm_policy", "comm_policy_params", "comm_budget_bytes")

#: How many competition steps keep their full candidate detail in a row. The aggregate counts are
#: complete; this is the readable evidence sample, and the cap keeps a row a few kilobytes.
COMPETITION_SAMPLE_LIMIT = 20


class CandidateProbe:
    """Wrap a rule, record what it was offered, return its answer verbatim.

    Only ``decide``, ``name``, ``parameters()`` and ``orders_selection`` are reachable from the
    transport (``comm.py`` uses nothing else), so those are the only attributes delegated. The
    return value is the wrapped rule's object, not a copy: ``orders_selection`` rules hand back an
    ordered tuple whose order decides who gets paid when the credit cannot cover everyone, and a
    probe that reorders it would silently change the episode it is supposed to be measuring.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.counts: dict[str, int] = {
            "steps_offered": 0,
            "visible_total": 0,
            "payable_total": 0,
            "competition_points": 0,
            "competition_points_credit_cause": 0,
            "competition_points_capacity_only": 0,
            "joint_only_points": 0,
            "multi_payable_points": 0,
            "credit_blocked_steps": 0,
            "capacity_blocked_steps": 0,
            "max_visible": 0,
            "max_payable": 0,
        }
        self.first_credit_block_step: int | None = None
        self.block_reasons: dict[str, int] = {}
        self.sums: dict[str, float] = {
            "remaining_at_competition": 0.0,
            "visible_at_competition": 0.0,
            "payable_at_competition": 0.0,
            "mean_slack_at_competition": 0.0,
            "backlog_at_competition": 0.0,
        }
        self.sums_n = 0
        self.competition_sample: list[dict[str, Any]] = []
        self.slack_observations: list[float] = []
        self.backlog_by_step: list[int] = []

    # -- delegation -------------------------------------------------------
    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def orders_selection(self) -> bool:
        return bool(getattr(self._inner, "orders_selection", False))

    def parameters(self) -> dict[str, Any]:
        return dict(self._inner.parameters())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CandidateProbe({self._inner!r})"

    # -- measurement ------------------------------------------------------
    def decide(self, observation: Any) -> Any:
        chosen = self._inner.decide(observation)
        self._observe(observation)
        return chosen

    def _observe(self, observation: Any) -> None:  # noqa: C901 - one pass over one step
        items = list(observation.items)
        if not items:
            return
        remaining = observation.remaining_budget_bytes
        charges = [
            None if item.attempt_charge_bytes is None else int(item.attempt_charge_bytes)
            for item in items
        ]
        payable_flags = [bool(item.send_executable) for item in items]
        payable = sum(1 for flag in payable_flags if flag)
        unknown_charge = sum(1 for charge in charges if charge is None)
        visible = len(items)
        total_charge = None if unknown_charge else int(sum(charges or []))

        counts = self.counts
        counts["steps_offered"] += 1
        counts["visible_total"] += visible
        counts["payable_total"] += payable
        counts["max_visible"] = max(counts["max_visible"], visible)
        counts["max_payable"] = max(counts["max_payable"], payable)

        slack = [float(item.remaining_time) for item in items]
        self.slack_observations.extend(slack)
        self.backlog_by_step.append(int(observation.backlog_targets))

        # Primary definition, fixed in the protocol: at least two candidates in view and a
        # non-empty strict subset of them payable under the credit left at this step.
        competition = visible >= 2 and 1 <= payable < visible
        joint_only = (
            visible >= 2
            and payable == visible
            and total_charge is not None
            and remaining is not None
            and total_charge > int(remaining)
        )
        credit_blocked = sum(
            1 for item_ in items if str(item_.send_block_reason) == "budget_exhausted"
        )
        if competition:
            counts["competition_points"] += 1
            if credit_blocked:
                counts["competition_points_credit_cause"] += 1
            else:
                counts["competition_points_capacity_only"] += 1
            self.sums_n += 1
            self.sums["remaining_at_competition"] += float(remaining or 0)
            self.sums["visible_at_competition"] += visible
            self.sums["payable_at_competition"] += payable
            self.sums["mean_slack_at_competition"] += sum(slack) / len(slack)
            self.sums["backlog_at_competition"] += int(observation.backlog_targets)
            if len(self.competition_sample) < COMPETITION_SAMPLE_LIMIT:
                self.competition_sample.append(
                    {
                        "step": int(observation.step),
                        "remaining_bytes": None if remaining is None else int(remaining),
                        "capacity_remaining": observation.capacity_remaining,
                        "visible": visible,
                        "payable": payable,
                        "candidates": [
                            [
                                str(item.target_id),
                                None if charge is None else int(charge),
                                bool(flag),
                                str(item.send_block_reason),
                                int(item.first_emission),
                                round(float(item.remaining_time), 3),
                            ]
                            for item, charge, flag in zip(items, charges, payable_flags)
                        ],
                    }
                )
        if joint_only:
            counts["joint_only_points"] += 1
        if visible >= 2 and payable >= 2:
            counts["multi_payable_points"] += 1
        reasons = {str(item.send_block_reason) for item in items}
        for reason in sorted(reasons):
            self.block_reasons[reason] = self.block_reasons.get(reason, 0) + 1
        if payable == 0:
            if "budget_exhausted" in reasons:
                counts["credit_blocked_steps"] += 1
                if self.first_credit_block_step is None:
                    self.first_credit_block_step = int(observation.step)
            elif "capacity_step" in reasons:
                counts["capacity_blocked_steps"] += 1

    # -- reporting --------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        n = self.sums_n
        steps = [int(entry["step"]) for entry in self.competition_sample]
        return {
            **self.counts,
            "first_credit_exhaustion_step": self.first_credit_block_step,
            "block_reasons": dict(sorted(self.block_reasons.items())),
            "mean_remaining_bytes_at_competition": (
                round(self.sums["remaining_at_competition"] / n, 1) if n else None
            ),
            "mean_visible_at_competition": round(self.sums["visible_at_competition"] / n, 3) if n else None,
            "mean_payable_at_competition": round(self.sums["payable_at_competition"] / n, 3) if n else None,
            "mean_slack_at_competition": round(self.sums["mean_slack_at_competition"] / n, 3) if n else None,
            "mean_backlog_at_competition": round(self.sums["backlog_at_competition"] / n, 3) if n else None,
            "competition_steps_sample": steps,
            "competition_sample": self.competition_sample,
            "max_backlog_observed": max(self.backlog_by_step, default=0),
        }


def on_time_count(timelines: dict[str, Any]) -> int:
    """The frozen definition, unchanged since the round that fixed the environment."""
    return sum(
        1
        for timeline in timelines.values()
        if timeline.final_progress is not None
        and float(timeline.final_progress) >= 1.0
        and timeline.final_failure_reason != "deadline_expired"
    )


def _per_target_facts(timelines: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Per-target outcome plus the search/capacity facts, read from the same timelines the metric is."""
    outcomes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for target_id, timeline in timelines.items():
        completed = (
            timeline.final_progress is not None and float(timeline.final_progress) >= 1.0
        ) and timeline.final_failure_reason != "deadline_expired"
        if timeline.final_failure_reason == "deadline_expired":
            outcome = "expired"
        elif completed:
            outcome = "on_time"
        else:
            outcome = "unresolved"
        outcomes[str(target_id)] = outcome
        rows.append(
            {
                "target": str(target_id),
                "outcome": outcome,
                "detected_at": round(float(timeline.detected_at), 3),
                "deadline": round(float(timeline.deadline), 3),
                "work_needed": round(float(timeline.work_needed), 3),
                "first_assignment": (
                    None if timeline.first_assignment is None else round(float(timeline.first_assignment), 3)
                ),
                "assignment_latency": (
                    None
                    if timeline.assignment_latency is None
                    else round(float(timeline.assignment_latency), 3)
                ),
                "first_work": None if timeline.first_work is None else round(float(timeline.first_work), 3),
                "last_work": None if timeline.last_work is None else round(float(timeline.last_work), 3),
                "deadline_slack_before_work": (
                    None
                    if timeline.first_work is None
                    else round(float(timeline.deadline) - float(timeline.first_work), 3)
                ),
                "final_progress": (
                    None if timeline.final_progress is None else round(float(timeline.final_progress), 4)
                ),
                "assigned_agents": list(timeline.assigned_agents),
                "rejected": int(timeline.rejected),
            }
        )
    return outcomes, rows


def _occupancy_by_step(rows: list[dict[str, Any]], steps: int) -> dict[str, Any]:
    """Concurrency and robot use, assembled from the per-target intervals.

    A task is open from its detection until its last work (or the horizon); a robot is occupied
    between its own first and last work on that task. Both are step-counted, so the numbers are
    intervals the lifecycle log actually shows and not an assumption about travel.
    """
    open_counts = [0] * (steps + 1)
    busy: dict[int, set[str]] = {}
    never_assigned = 0
    assigned_without_work = 0
    for row in rows:
        start = int(row["detected_at"])
        end = int(row["last_work"]) if row["last_work"] is not None else steps
        end = max(start, min(steps, end))
        for step in range(start, end + 1):
            open_counts[step] += 1
        if row["first_assignment"] is None:
            never_assigned += 1
            continue
        if row["first_work"] is None:
            assigned_without_work += 1
            continue
        work_end = int(row["last_work"]) if row["last_work"] is not None else steps
        for agent in row["assigned_agents"]:
            for step in range(int(row["first_work"]), max(int(row["first_work"]), work_end) + 1):
                busy.setdefault(step, set()).add(agent)
    robots = common.RECORD_CONFIGURATION["rescue_species"] * common.RECORD_CONFIGURATION["rescue_per_species"]
    used = [len(busy.get(step, set())) for step in range(steps + 1)]
    ordered = sorted(open_counts)
    return {
        "mean_concurrent_open": round(sum(open_counts) / len(open_counts), 4),
        "p95_concurrent_open": ordered[int(0.95 * (len(ordered) - 1))],
        "max_concurrent_open": max(open_counts, default=0),
        "mean_robot_utilisation": round(sum(used) / (len(used) * robots), 5),
        "never_assigned_targets": never_assigned,
        "assigned_without_work_targets": assigned_without_work,
        "work_demand_vs_capacity": round(
            sum(float(row["work_needed"]) for row in rows) / float(robots * steps), 4
        ),
    }


def run_episode(
    goals: int,
    seed: int,
    *,
    arm: str,
    directory: Path,
    probe: bool = True,
    stream: Path | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> dict[str, Any]:
    """One arm on one stream of one pressure layer, with the frozen setting of §1."""
    spec = common.arm_spec(arm)
    parameters = parse_parameters(spec.get("params", "")) or {}
    stream = common.stream_path(goals, seed) if stream is None else Path(stream)
    values = common.replay_configuration(
        goals, seed, directory=directory, label=arm, stream=stream
    )
    budget_bytes = None if spec["unbudgeted"] else common.BUDGET_BYTES
    values.update(
        comm_policy=spec["policy"],
        comm_policy_params=",".join(f"{key}={value}" for key, value in sorted(parameters.items())),
        comm_budget_bytes=budget_bytes,
        comm_delivery_probability=common.DELIVERY_PROBABILITY,
        comm_delay_ticks=0,
        comm_max_wait_steps=common.BOUND_MAX_WAIT_STEPS if spec["bounded"] else None,
        comm_urgent_wait_steps=common.BOUND_URGENT_WAIT_STEPS if spec["bounded"] else None,
        comm_timing_mode=common.TIMING_MODE,
        comm_ack=common.ACK_ENABLED,
        gossip_interval=common.protocol_common.ISOLATED_GOSSIP_INTERVAL,
        quiet=True,
    )
    started = time.time()
    Path(directory).mkdir(parents=True, exist_ok=True)
    run = build_joint_run(RunConfiguration(**values))
    relay = run.task_adapter.comm_relay
    expected = build_comm_policy(spec["policy"], parameters)
    if relay.policy.name != expected.name or dict(relay.policy.parameters()) != dict(
        expected.parameters()
    ):
        raise RuntimeError(
            f"{arm} g{goals} s{seed}: built policy {relay.policy.name!r} "
            f"{relay.policy.parameters()}, asked for {expected.name!r} {expected.parameters()}"
        )
    declared = relay.transport_parameters()
    if declared["timing_mode"] != common.TIMING_MODE:
        raise RuntimeError(f"{arm} g{goals} s{seed}: relay reports {declared['timing_mode']!r}")
    if bool(declared["ack_enabled"]) != common.ACK_ENABLED:
        raise RuntimeError(f"{arm} g{goals} s{seed}: the reverse channel state is not the frozen one")
    if float(declared["delivery_probability"]) != common.DELIVERY_PROBABILITY:
        raise RuntimeError(f"{arm} g{goals} s{seed}: the channel is not the lossless one this round froze")
    if relay.budget.bytes_per_episode != budget_bytes:
        raise RuntimeError(f"{arm} g{goals} s{seed}: the byte credit never reached the executor")
    if spec["unbudgeted"] and relay.budget.bytes_per_episode is not None:
        raise RuntimeError(f"{arm} g{goals} s{seed}: a ceiling reference must carry no credit")
    if declared["bound_enforced_by_transport"] is not False:
        raise RuntimeError(
            f"{arm} g{goals} s{seed}: the transport is enforcing a silence bound it was told to "
            "leave to the arm; the rule library of §3 would not be the one being measured"
        )

    wrapped: CandidateProbe | None = None
    if probe:
        wrapped = CandidateProbe(relay.policy)
        relay._policy = wrapped

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
    try:
        summary = run.run()
    finally:
        environment.step = original_step

    spent = relay.budget.spent_bytes
    if budget_bytes is not None and spent > budget_bytes:
        raise RuntimeError(f"{arm} g{goals} s{seed}: budget violated, {spent} > {budget_bytes}")
    if int(summary.get("steps", 0)) != common.STREAM_STEPS:
        raise RuntimeError(
            f"{arm} g{goals} s{seed}: the episode ran {summary.get('steps')} steps, not the "
            f"frozen {common.STREAM_STEPS}"
        )

    transcript = common.pilot.metrics.load_transcript(values["comm_transcript"])
    records = load_records(Path(values["output"]))
    timelines = build_timelines(records, arm="C", seed=seed)
    outcomes, target_rows = _per_target_facts(timelines)
    credited_total = int(sum(rewards_by_step.values()))
    completed = on_time_count(timelines)
    if completed != credited_total:
        raise RuntimeError(
            f"{arm} g{goals} s{seed}: the tap credited {credited_total} on-time targets, the "
            f"lifecycle log says {completed}; reward and metric have diverged"
        )
    final_records = [record for record in records if record.get("event_type") == "final"]
    if not final_records:
        raise RuntimeError(
            f"{arm} g{goals} s{seed}: the lifecycle log has no final record; L0 cannot be judged "
            "on an episode that did not finish"
        )

    payload = json.loads(stream.read_text(encoding="utf-8"))
    placed = int(payload.get("scenario", {}).get("num_goals_placed", -1))
    detected = len(
        {
            str(detection.get("target_id", detection.get("target")))
            for frame in payload.get("frames", [])
            for detection in frame.get("detections", [])
        }
    )
    if placed != goals:
        raise RuntimeError(
            f"g{goals} s{seed}: the recorded world placed {placed} targets, not {goals}; the "
            "layer's own scenario is not the one this round registered"
        )

    sends = [record for record in transcript if record["kind"] == "send"]
    refusals = [record for record in transcript if record["kind"] == "refused"]
    emission_types = {
        "first": sum(1 for record in sends if record.get("first")),
        "refresh": sum(1 for record in sends if not record.get("first")),
    }
    if emission_types["first"] + emission_types["refresh"] != len(sends):
        raise RuntimeError(f"{arm} g{goals} s{seed}: the send-type split does not close")
    counted_from_transcript = {
        "emissions": len(sends),
        "first_emissions": emission_types["first"],
        "refresh_emissions": emission_types["refresh"],
        "budget_refused_sends": sum(
            1 for record in refusals if record.get("reason") == "budget_exhausted"
        ),
        "capacity_refused_sends": sum(
            1 for record in refusals if record.get("reason") == "capacity_step"
        ),
    }
    for key, value in counted_from_transcript.items():
        if int(relay.counters[key]) != value:
            raise RuntimeError(
                f"{arm} g{goals} s{seed}: the relay counted {relay.counters[key]} {key}, the "
                f"transcript holds {value}; two ledgers of one episode have diverged"
            )

    # Everything that must be identical across the nine arms of a layer, with the three fields an
    # arm is allowed to differ on -- and the per-cell paths, which carry the arm's own label --
    # removed. One hash of that is the layer's setting; two different hashes is a mixed matrix.
    path_keys = ("output", "decision_trace", "comm_transcript", "replay_stream")
    setting = {
        key: value
        for key, value in values.items()
        if key not in ALLOWED_DIFFERENCE_KEYS and key not in path_keys
    }
    signature = hashlib.sha256(
        json.dumps(setting, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]

    return {
        "arm": arm,
        "setting_signature": signature,
        "role": spec["role"],
        "goals": int(goals),
        "seed": int(seed),
        "probed": bool(probe),
        "policy": spec["policy"],
        "policy_parameters": parameters,
        "budget_bytes": budget_bytes,
        "spent_bytes": int(spent),
        "budget_used_fraction": (round(spent / budget_bytes, 5) if budget_bytes else None),
        "on_time": completed,
        "expired": sum(1 for row in target_rows if row["outcome"] == "expired"),
        "unresolved": sum(1 for row in target_rows if row["outcome"] == "unresolved"),
        "registered": len(target_rows),
        "detected": detected,
        "placed": placed,
        "on_time_rate_of_detected": (
            round(completed / detected, 5) if detected else None
        ),
        "emissions": len(sends),
        "emission_types": emission_types,
        # Packet-level identity, used by the agreement gate and by any later claim that a rule
        # here is the same rule that produced an archived row: same packets, same steps, same
        # order, same money. "Same mean" is not this claim (round 10's lesson).
        "emission_sequence": [
            [
                int(record["step"]),
                str(record["target"]),
                int(record["revision"]),
                int(record["cost_bytes"]),
                int(record["subscribers"]),
                bool(record["first"]),
            ]
            for record in sends
        ],
        "refusal_sequence": [
            [int(record["step"]), str(record["target"]), str(record["reason"]),
             str(record.get("send_reason"))]
            for record in refusals
        ],
        "send_steps": [int(record["step"]) for record in sends],
        "refusals": {
            "total": len(refusals),
            "budget_exhausted": sum(
                1 for record in refusals if record.get("reason") == "budget_exhausted"
            ),
            "capacity_step": sum(
                1 for record in refusals if record.get("reason") == "capacity_step"
            ),
            "first_budget_refusal_step": min(
                (int(r["step"]) for r in refusals if r.get("reason") == "budget_exhausted"),
                default=None,
            ),
            "reasons": sorted({str(r.get("reason")) for r in refusals}),
        },
        "ledger": {
            **common.pilot.metrics.transport_accounting(transcript),
            **{key: int(value) for key, value in relay.counters.items()},
        },
        "belief": common.pilot.metrics.belief_age_accounting(
            transcript,
            horizon=float(summary.get("steps", 0)),
            robots=[str(agent_id) for agent_id in environment.agent_dic],
            truth=common.pilot.metrics.truth_timeline(records),
            active_until=common.pilot.metrics.terminal_times(records),
        ),
        "occupancy": _occupancy_by_step(target_rows, common.STREAM_STEPS),
        "targets": target_rows,
        "targets_outcome": outcomes,
        "probe": wrapped.summary() if wrapped is not None else None,
        "stream_sha256": common.sha256(stream),
        "config_hash": run.run_signature["config_hash"],
        "arm_label": run.run_signature["arm"],
        "timing_mode": declared["timing_mode"],
        "bound_enforced_by_transport": declared["bound_enforced_by_transport"],
        "ack_enabled": bool(declared["ack_enabled"]),
        "delivery_probability": float(declared["delivery_probability"]),
        "steps": int(summary.get("steps", 0)),
        "seconds": round(time.time() - started, 2),
        "artefacts": {
            "lifecycle_log": values["output"],
            "transcript": values["comm_transcript"],
            "decision_trace": values["decision_trace"],
        },
    }


def run_arm(goals: int, seed: int, arm: str, *, directory: Path | None = None) -> dict[str, Any]:
    """One matrix cell, with the ledger entry that pays for it."""
    directory = Path(directory) if directory is not None else common.layer_runs(goals)
    ledger = common.LaunchLedger(f"LAUNCH_LEDGER_g{goals}_p{os.getpid()}.json")
    ledger.reserve("matrix", goals, seed, arm=arm)
    try:
        row = run_episode(goals, seed, arm=arm, directory=directory)
    except Exception as error:  # noqa: BLE001 - a failure still spends the launch
        ledger.record("matrix", goals, seed, False, f"{arm}: {type(error).__name__}: {error}"[:400], arm=arm)
        raise
    ledger.record(
        "matrix",
        goals,
        seed,
        True,
        f"{arm}: on_time={row['on_time']} bytes={row['spent_bytes']}",
        arm=arm,
    )
    return row


def agreement_with_archived_rows(seeds: tuple[int, ...], arms: tuple[str, ...]) -> dict[str, Any]:
    """This runner + probe must reproduce the archived round-5 rule rows, field by field.

    The archived rows were produced without any probe, so agreement proves two things at once:
    the runner is not a drift from the one that made them, and the probe changed nothing.
    """
    path = (
        common.PROTOCOL_ROUND / "evaluation" / "25pct" / "rows_25pct_isolated_1.0.jsonl"
    )
    archived: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            archived[(str(record["arm"]), int(record["seed"]))] = record
    directory = common.HERE / "agreement"
    checks: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in arms:
            reference = archived.get((arm, int(seed)))
            if reference is None:
                raise FileNotFoundError(
                    f"no archived round-5 row for {arm} s{seed}; the agreement check cannot be "
                    "judged against a remembered number"
                )
            ledger = common.LaunchLedger("LAUNCH_LEDGER_agreement.json")
            ledger.reserve("gate", 6, seed, arm=arm)
            row = run_episode(
                6, seed, arm=arm, directory=directory, stream=common.pilot.stream_path(seed)
            )
            ledger.record("gate", 6, seed, True, f"agreement {arm}", arm=arm)
            archived_transcript = Path(str(reference.get("transcript", "")))
            reference_sequence: list[Any] | None = None
            if archived_transcript.exists():
                reference_rows = common.pilot.metrics.load_transcript(archived_transcript)
                reference_sequence = [
                    [
                        int(record["step"]),
                        str(record["target"]),
                        int(record["revision"]),
                        int(record["cost_bytes"]),
                        int(record["subscribers"]),
                        bool(record["first"]),
                    ]
                    for record in reference_rows
                    if record["kind"] == "send"
                ]
            explained: dict[str, str] = {}
            if (
                compared_belief_missing(row, reference)
            ):
                explained["belief_age"] = (
                    "the rule emitted nothing, so no robot-target pair was ever informed: this "
                    "round keeps that as an explicit absence while round 5's renderer wrote "
                    "`... or 0.0`. Same episode, two renderings of the same empty set."
                )
            compared = {
                "on_time": (row["on_time"], int(reference["on_time_targets_per_episode"])),
                "expired": (row["expired"], int(reference["expired_targets_per_episode"])),
                "detected": (row["detected"], int(reference["detected_targets_this_episode"])),
                "bytes": (row["spent_bytes"], int(reference["airtime_bytes"])),
                "emissions": (row["emissions"], int(reference["emissions"])),
                "credit_fraction": (
                    row["budget_used_fraction"],
                    float(reference["credit_used_fraction"]),
                ),
                "budget_refusals": (
                    row["refusals"]["budget_exhausted"],
                    int(reference["budget_refusals"]),
                ),
                "deliveries_applied": (
                    row["ledger"]["deliveries_applied"],
                    int(reference["deliveries_applied"]),
                ),
                "belief_age": (
                    row["belief"].get("belief_age_steps_mean"),
                    reference["belief_age_steps"],
                ),
            }
            differences = {
                key: value
                for key, value in compared.items()
                if _differs(value[0], value[1]) and key not in explained
            }
            checks.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "match": not differences,
                    "differences": differences,
                    "reference_config_hash": reference["config_hash"],
                    "this_config_hash": row["config_hash"],
                    "reference_transcript": str(reference.get("transcript", "")),
                    "reference_transcript_present": bool(archived_transcript.exists()),
                    "emission_sequence_equal": (
                        None if reference_sequence is None
                        else reference_sequence == row["emission_sequence"]
                    ),
                    "emission_sequence_length": [
                        len(reference_sequence or []), len(row["emission_sequence"])
                    ],
                    "explained_differences": explained,
                }
            )
    return {
        "archived_rows": str(path.relative_to(common.ROOT)),
        "checks": checks,
        "agree": all(check["match"] for check in checks),
        "note": (
            "config_hash is not part of the agreement: it folds the transcript path into the hash, "
            "so a different directory must change it (round 10 established this). The compared "
            "quantities are task outcome, money, emissions, refusals, deliveries and belief age."
        ),
    }


def _differs(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left != right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) > 1e-9
    return left != right


def compared_belief_missing(row: dict[str, Any], reference: dict[str, Any]) -> bool:
    """True only for the one case where an empty belief set was rendered two ways.

    A rule that never emitted has no robot-target pair to age, so its mean belief age does not
    exist. This round reports that as ``None``; round 5 coerced it to ``0.0``. The check demands
    the emission counts agree at zero, so a missing belief age on a rule that did speak is still a
    mismatch.
    """
    if row["emissions"] or int(reference["emissions"]):
        return False
    return row["belief"].get("belief_age_steps_mean") is None and float(
        reference["belief_age_steps"]
    ) == 0.0


def explain_sequence_mismatch(
    seeds: tuple[int, ...], arms: tuple[str, ...]
) -> dict[str, Any]:
    """Say which packet moved, using only artefacts already on disk.

    A gate that reports ``False`` without saying what differs is not usable: the sequences are
    re-read from the archived transcript and from this round's own transcript, so the explanation
    costs no new episode.
    """
    rows = load_archived_rows()
    report: dict[str, Any] = {}
    for seed in seeds:
        for arm in arms:
            reference = rows.get((arm, int(seed)))
            if reference is None:
                continue
            archived = Path(str(reference["transcript"]))
            mine = sorted(common.HERE.glob(f"agreement/ct_{arm}_g*_s{seed}.jsonl"))
            if not archived.exists() or not mine:
                report[f"{arm} s{seed}"] = {
                    "available": {"archived": archived.exists(), "fresh": [m.name for m in mine]}
                }
                continue
            left = _send_tuples(common.pilot.metrics.load_transcript(archived))
            right = _send_tuples(common.pilot.metrics.load_transcript(mine[0]))
            archived_records = common.pilot.metrics.load_transcript(archived)
            fresh_records = common.pilot.metrics.load_transcript(mine[0])
            provenance = {
                "archived_transcript_send_bytes": sum(
                    int(record["cost_bytes"])
                    for record in archived_records
                    if record["kind"] == "send"
                ),
                "archived_row_airtime_bytes": int(reference["airtime_bytes"]),
                "archived_transcript_emissions": sum(
                    1 for record in archived_records if record["kind"] == "send"
                ),
                "archived_row_emissions": int(reference["emissions"]),
                "self_consistent": sum(
                    int(record["cost_bytes"])
                    for record in archived_records
                    if record["kind"] == "send"
                ) == int(reference["airtime_bytes"]),
                "fresh_transcript_send_bytes": sum(
                    int(record["cost_bytes"])
                    for record in fresh_records
                    if record["kind"] == "send"
                ),
            }
            if left == right:
                report[f"{arm} s{seed}"] = {
                    "equal": True,
                    "packets": len(left),
                    "transcript_provenance": provenance,
                }
                continue
            fields = _record_field_diffs(
                [r for r in common.pilot.metrics.load_transcript(archived) if r["kind"] == "send"],
                [r for r in common.pilot.metrics.load_transcript(mine[0]) if r["kind"] == "send"],
            )
            report[f"{arm} s{seed}"] = {
                "equal": False,
                "transcript_provenance": provenance,
                "field_diffs": fields,
                "packets": [len(left), len(right)],
                "same_multiset": sorted(map(str, left)) == sorted(map(str, right)),
                "first_difference_index": next(
                    (index for index, (a, b) in enumerate(zip(left, right)) if a != b), None
                ),
                "side_by_side": [
                    {"archived": a, "fresh": b}
                    for a, b in zip(left, right)
                    if a != b
                ][:6],
                "only_in_archived": [a for a in left if a not in right][:6],
                "only_in_fresh": [b for b in right if b not in left][:6],
            }
    return report


FIELD_ORDER = (
    "step", "target", "revision", "payload_bytes", "cost_bytes", "subscribers", "first",
    "send_reason", "ready_step", "due_step", "now", "progress", "stage_index", "remaining",
    "members", "attempts", "packets", "packets_lost", "spent_after_bytes", "backlog_steps",
    "bytes_total", "deadline",
)


def _record_field_diffs(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    """Which fields of the packets differ, and whether the two runs share any packet at all.

    Matching is done on ``(step, target)`` because the question is "what changed about this
    emission", not "in what order did the two files list them".
    """
    key = lambda record: (int(record["step"]), str(record["target"]))
    left_by_key = {key(record): record for record in left}
    right_by_key = {key(record): record for record in right}
    shared = sorted(set(left_by_key) & set(right_by_key))
    per_field: dict[str, list[Any]] = {}
    for step, target in shared:
        a, b = left_by_key[(step, target)], right_by_key[(step, target)]
        for field in FIELD_ORDER:
            if a.get(field) != b.get(field):
                per_field.setdefault(field, []).append([a.get(field), b.get(field), step, target])
    return {
        "packets_matched_by_step_and_target": len(shared),
        "only_in_archived_steps": sorted(k[0] for k in set(left_by_key) - set(right_by_key))[:12],
        "only_in_fresh_steps": sorted(k[0] for k in set(right_by_key) - set(left_by_key))[:12],
        "fields_that_differ": {
            field: values[:4] for field, values in sorted(per_field.items())
        },
        "field_diff_counts": {field: len(values) for field, values in sorted(per_field.items())},
    }


def _send_tuples(records: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            int(record["step"]),
            str(record["target"]),
            int(record["revision"]),
            int(record["cost_bytes"]),
            int(record["subscribers"]),
            bool(record["first"]),
        ]
        for record in records
        if record["kind"] == "send"
    ]


def load_archived_rows() -> dict[tuple[str, int], dict[str, Any]]:
    path = common.PROTOCOL_ROUND / "evaluation" / "25pct" / "rows_25pct_isolated_1.0.jsonl"
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rows[(str(record["arm"]), int(record["seed"]))] = record
    return rows


def main(argv: list[str]) -> int:
    mode = argv[0] if argv else "agreement"
    if mode == "explain":
        report = explain_sequence_mismatch(
            (700048,),
            ("first_send_plus_bound_3_10", "event_priority", "send_once", "always_hold"),
        )
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0
    #: One archived stream, four arms chosen so the risky seams are all in the sample:
    #: ``event_priority`` orders its own selection (round 6's probe hazard),
    #: ``first_send_plus_bound_3_10`` and ``event_priority`` both hit the credit hard enough to be
    #: refused on this stream, and ``always_hold`` is the zero-emission floor.
    seeds = (700048,)
    arms = ("send_once", "event_priority", "first_send_plus_bound_3_10", "always_hold")
    result = agreement_with_archived_rows(seeds, arms)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result["agree"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
