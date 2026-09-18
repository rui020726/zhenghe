"""Per-decision trace of the allocation step: what the allocator saw, and what it chose.

The joint ablation leaves one failure mode -- tasks whose deadline passes before the
executors finish -- and the attribution pass already ruled out three explanations for
it: no expired target was infeasible at detection, none was blocked by missing
abilities, and the batch published with a lossless, zero-delay uplink. What is left is
a delivery defect: the first step of delivered work lands a median of 86 steps after
detection (rs0.25) while a straight-line trip takes at most 7.1 steps.

Three competing explanations for those 86 steps need three different fixes:

* the CTDE actor prefers the depot action even when a legal task is on offer -- a
  distribution-shift problem, fixed by adapting CTDE on the real detection stream;
* the allocator keeps re-tasking a robot before it arrives -- a churn problem, fixed by
  a minimum commitment time, which is also the strong baseline a learned coordinator
  must beat;
* robots are sent one at a time, so the group that satisfies a stage requirement never
  assembles on site -- a grouping problem, fixed by a coordinator that convenes
  compatible robots together.

The lifecycle log cannot tell these apart: it records the assignment that was *issued*,
never the alternatives that were rejected or the logits behind the choice. This module
adds a switchable record of the decision itself. For every (robot, decision instant) it
keeps

* the action space that robot's own belief produced, the hard mask, and how many of the
  task actions were legal;
* the CTDE logits, the coordinator residual and the final logits, one entry per action;
* the action that won, and whether the robot was idle or already committed;
* where the robot was, how far its current navigation target still is, and the implied
  ETA at ``velocity = 0.2``;
* for statistics only, the ground truth about the chosen and the current target: who is
  on site, who is en route, whether the assembled group already covers the stage
  requirement, and how far the deadline still is.

Those truth fields are written into the trace and never read back into a decision: the
trace writer is a sink attached next to the adapter, the decision path is identical
whether it is attached or not, and invariant 9 is untouched. They are needed because
the questions above are about the world rather than about one robot's belief -- "is a
sufficient group already there" cannot be answered from a belief that lags the scene.

The trace is off by default and enabled by ``--decision-trace PATH``; run this module
for a CLI report over ``tr_<arm>_s<seed>.jsonl`` files.
"""

from __future__ import annotations

import argparse
import glob as globmodule
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

from .run_signature import TRACE_SCHEMA as TRACE_FORMAT

# The schema version a trace declares and the one a run signature records are one constant,
# because a reader that accepts one version and a writer that stamps another would silently
# stop checking anything.
DEPOT_INDEX = 0
# ``old_task_env`` moves each robot by velocity * dt per step and snaps it onto its
# target once within EPS_ARRIVE, so these two constants turn a distance into the ETA
# the allocator is effectively choosing between.
ROBOT_VELOCITY = 0.2
EPS_ARRIVE = 1e-6

# The per-cycle counters a reserving coordinator publishes in ``advice.metadata``. Named
# here so the planner and the reader of its traces cannot drift apart silently.
PLAN_COUNTERS = (
    "planned_reservations",
    "redundant_candidates",
    "expired_reservations",
    "superseded_reservations",
    "cancelled_reservations",
)


def arm_name(
    *,
    use_coordinator: bool,
    residual_scale: float,
    task_logit_offset: float = 0.0,
    coordinator_policy: str = "heuristic",
    binding_commitments: bool = False,
    release_redundant_members: bool = False,
    suppress_recommand_own_task: bool = False,
) -> str:
    """The arm label shared with the ``rp_<arm>_s<seed>.jsonl`` lifecycle logs.

    Scalar calibration and the residual policy are part of the arm's identity: a
    trace that cannot say "this was the uncalibrated actor" or "this was the
    coalition policy" cannot be pooled with others without silently averaging two
    different policies. The defaults keep the pre-existing labels exactly as they
    were, so the frozen 20260916 traces keep reading as ``nocoord``/``rs0.25``.
    """
    if use_coordinator:
        # The policy is part of the arm's identity: two residual policies must never
        # share a label, or their traces would pool silently. Names that are not listed
        # keep the historical "coal" prefix, so the frozen 20260916 traces still read
        # as ``rs0.25``/``coal0.25``.
        prefix = {
            "heuristic": "rs",
            "coalition_planner": "coalplan",
            "coalition_assembly_scheduler": "coalasm",
            "coalition_redundant_release": "coalredrel",
            "deadline_aware": "daw",
        }.get(coordinator_policy, "coal")
        label = f"{prefix}{residual_scale:g}"
    else:
        label = "nocoord"
    if task_logit_offset:
        label = f"{label}+cal{task_logit_offset:g}"
    if binding_commitments:
        label = f"{label}sticky"
    if release_redundant_members:
        # An arm label is the only place two policies are told apart in a pooled file:
        # a run that re-decides redundant members must never share a label with one
        # that does not, exactly like the binding suffix above.
        label = f"{label}+redrel"
    if suppress_recommand_own_task:
        label = f"{label}+noself"
    return label


def trace_meta(
    configuration: Any, signature: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Everything a trace file needs to identify itself without the lifecycle log.

    ``signature`` is the run signature the lifecycle log also carries. It is what makes the
    trace checkable rather than merely named: the arm in the file's name is a claim, and
    the signature is the run's own statement of which arm it was, from which commit, over
    which stream. A trace without one is a trace nothing may be aggregated from.
    """
    meta = {
        "arm": arm_name(
            use_coordinator=bool(configuration.use_coordinator),
            residual_scale=float(configuration.residual_scale),
            task_logit_offset=float(
                getattr(configuration, "ctde_task_logit_offset", 0.0)
            ),
            coordinator_policy=str(
                getattr(configuration, "coordinator_policy", "heuristic")
            ),
            binding_commitments=bool(
                getattr(configuration, "sticky_commitments", False)
            ),
            release_redundant_members=bool(
                getattr(configuration, "release_redundant_members", False)
                or getattr(configuration, "coordinator_policy", "") == "coalition_redundant_release"
            ),
            suppress_recommand_own_task=bool(
                getattr(configuration, "suppress_recommand_own_task", False)
                or getattr(configuration, "coordinator_policy", "") == "coalition_redundant_release"
            ),
        ),
        "coordinator_policy": str(
            getattr(configuration, "coordinator_policy", "heuristic")
        ),
        "seed": int(configuration.seed),
        "steps": int(configuration.steps),
        "allocation_backend": configuration.allocation_backend,
        "search_backend": configuration.search_backend,
        "use_coordinator": bool(configuration.use_coordinator),
        "sticky_commitments": bool(
            getattr(configuration, "sticky_commitments", False)
        ),
        "release_redundant_members": bool(
            getattr(configuration, "release_redundant_members", False)
        ),
        "suppress_recommand_own_task": bool(
            getattr(configuration, "suppress_recommand_own_task", False)
        ),
        "residual_scale": float(configuration.residual_scale),
        "ctde_task_logit_offset": float(
            getattr(configuration, "ctde_task_logit_offset", 0.0)
        ),
        "command_ttl": float(configuration.command_ttl),
        "allocation_interval": int(configuration.allocation_interval),
        "coordination_interval": int(configuration.coordination_interval),
        "task_checkpoint": configuration.task_checkpoint,
        "replay_stream": configuration.replay_stream,
        "record_stream": configuration.record_stream,
        "comm_delivery_probability": float(configuration.comm_delivery_probability),
        "comm_delay_ticks": int(configuration.comm_delay_ticks),
    }
    if signature is not None:
        meta["run_signature"] = dict(signature)
    return meta


@dataclass(frozen=True, slots=True)
class DecisionFacts:
    """One allocation decision, as gathered by the adapter at the moment it happens.

    The vectors are aligned with ``action_task_ids`` and index 0 is the depot action,
    matching ``TaskEnv.get_local_obs``'s ``action_map``. A masked entry carries ``None``
    rather than the ``-1e9`` sentinel the network writes, so a reader can tell "illegal"
    from "very unlikely" without knowing the sentinel.
    """

    step: int
    timestamp: float
    agent_id: str
    status: str
    abilities: tuple[float, ...]
    position: tuple[float, float]
    travel_dist: float
    needs_decision: bool
    replanned: bool
    belief_task_count: int
    action_task_ids: tuple[str | None, ...]
    mask: tuple[bool, ...]
    chosen_index: int
    ctde_logits: tuple[float | None, ...] | None = None
    bias: tuple[float | None, ...] | None = None
    final_logits: tuple[float | None, ...] | None = None
    # The scalar calibration applied to legal task logits in this decision, if any.
    # Recorded next to the raw CTDE logits so ``final`` can be re-derived:
    # final[task] == ctde[task] + task_logit_offset + bias[task] for a legal task.
    task_logit_offset: float | None = None
    current_kind: str | None = None
    current_task: str | None = None
    current_task_age: float | None = None
    current_target_distance: float | None = None
    current_target_eta: float | None = None
    current_target_taken: bool | None = None
    truth: Mapping[str, Any] = field(default_factory=dict)
    # The advice this decision was made under. A reservation is a stated intention --
    # this robot covers a named slot of a named task until a stated time -- which is a
    # different object from the bounded per-pair bias every policy publishes, so the two
    # are recorded separately and can be compared rather than pooled.
    advice_revision: int | None = None
    # Task ids the advice held a commitment for, team-wide. This is what makes "how long
    # from detection to a full reservation" measurable at all: a plan can be complete for
    # a target without the robot that decides next being one of the reserved ones.
    advice_reservations: tuple[str, ...] = ()
    # The commitment this robot itself held for a task on its own menu, if any.
    held_reservation: Mapping[str, Any] | None = None
    # The issuing planner's own cycle counters (planned/redundant/expired/superseded/
    # cancelled). Counted once per advice revision, since they describe the transition
    # into that revision rather than a property of one decision.
    advice_plan: Mapping[str, Any] = field(default_factory=dict)
    # Commitment continuity. ``commitment_present`` says the advice held a commitment for
    # a task on this robot's menu; ``commitment_held`` says it survived the validity
    # conditions; ``commitment_release_reason`` is the first condition that failed, so a
    # released commitment is never mistaken for an ignored one. ``commitment_overridden``
    # is the allocator's own verdict differing from a commitment that still stands -- the
    # number the continuity fix exists to drive to zero, recorded whether or not the
    # commitment was binding.
    commitment_present: bool = False
    commitment_held: bool = False
    commitment_release_reason: str | None = None
    commitment_task: str | None = None
    commitment_slot: int | None = None
    commitment_slot_dimension: int | None = None
    commitment_belief_status: str | None = None
    commitment_belief_revision: int | None = None
    commitment_mask_legal: bool | None = None
    commitment_overridden: bool = False
    commitment_suppressed_command: bool = False
    # The three choices one decision contains, all read at the instant it was made:
    # ``raw_choice`` is the allocator's own preference, ``forced_choice`` is the action a
    # standing commitment imposed instead, and ``chosen_index`` (written as
    # ``post_mask_choice``) is what survived the hard mask and was executed. Recording all
    # three is what makes "the model wanted elsewhere and the commitment held it" tellable
    # from "the model's own choice was the committed task".
    raw_choice: int | None = None
    forced_choice: int | None = None
    # The task the issued command named, and its id -- the values the lifecycle log also
    # received, rather than a second derivation of them. ``actual_assignment`` is None for
    # a depot action and for a decision that issued no command at all.
    actual_assignment: str | None = None
    command_id: str | None = None
    # The identity of the commitment this robot held, stable across renewals, so a reader
    # can follow one commitment through many advisories.
    reservation_id: str | None = None
    # The conditions a held commitment was tested against, in the order they were checked,
    # up to and including the first that failed. The check stops at the first failure, so a
    # condition absent here was never evaluated, not evaluated and false.
    commitment_validity: Mapping[str, bool] = field(default_factory=dict)
    # Whether the decision ends with the robot on the task it was committed to. A
    # commitment can be held and not executed only when nothing binds the allocator, which
    # is exactly the measurement the binding arm exists to make.
    reservation_executed: bool | None = None

    @property
    def chosen_kind(self) -> str:
        return "depot" if int(self.chosen_index) == DEPOT_INDEX else "task"

    @property
    def chosen_task(self) -> str | None:
        index = int(self.chosen_index)
        if index == DEPOT_INDEX or index >= len(self.action_task_ids):
            return None
        return self.action_task_ids[index]

    @property
    def legal_task_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index in range(1, len(self.action_task_ids))
            if index < len(self.mask) and not self.mask[index]
            and self.action_task_ids[index] is not None
        )

    def as_json(self, meta: Mapping[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event_type": "decision",
            "arm": meta.get("arm"),
            "seed": meta.get("seed"),
            "backend": meta.get("allocation_backend"),
            "step": int(self.step),
            "timestamp": _round(self.timestamp),
            "agent_id": str(self.agent_id),
            "status": str(self.status),
            "abilities": [_round(value) for value in self.abilities],
            "position": [_round(value) for value in self.position],
            "travel_dist": _round(self.travel_dist),
            "needs_decision": bool(self.needs_decision),
            "replanned": bool(self.replanned),
            "belief_task_count": int(self.belief_task_count),
            "action_task_ids": list(self.action_task_ids),
            "mask": [bool(value) for value in self.mask],
            "chosen_index": int(self.chosen_index),
            # The same number the continuity contract names ``post_mask_choice``: the value
            # the analysis reads and the name the decision record promises are one fact.
            "post_mask_choice": int(self.chosen_index),
            "raw_choice": None if self.raw_choice is None else int(self.raw_choice),
            "forced_choice": (
                None if self.forced_choice is None else int(self.forced_choice)
            ),
            "actual_assignment": self.actual_assignment,
            "command_id": self.command_id,
            "chosen_kind": self.chosen_kind,
            "chosen_task": self.chosen_task,
            "current_kind": self.current_kind,
            "current_task": self.current_task,
            "truth": _jsonable(self.truth),
        }
        for key, values in (
            ("ctde_logits", self.ctde_logits),
            ("bias", self.bias),
            ("final_logits", self.final_logits),
        ):
            payload[key] = (
                None
                if values is None
                else [None if value is None else _round(value) for value in values]
            )
        for key, value in (
            ("current_task_age", self.current_task_age),
            ("current_target_distance", self.current_target_distance),
            ("current_target_eta", self.current_target_eta),
        ):
            payload[key] = None if value is None else _round(value)
        payload["task_logit_offset"] = (
            None if self.task_logit_offset is None else _round(self.task_logit_offset)
        )
        payload["current_target_taken"] = (
            None if self.current_target_taken is None else bool(self.current_target_taken)
        )
        payload["advice_revision"] = (
            None if self.advice_revision is None else int(self.advice_revision)
        )
        payload["advice_reservations"] = list(self.advice_reservations)
        payload["held_reservation"] = (
            None
            if self.held_reservation is None
            else _jsonable(self.held_reservation)
        )
        payload["advice_plan"] = _jsonable(self.advice_plan)
        payload["commitment_present"] = bool(self.commitment_present)
        payload["commitment_held"] = bool(self.commitment_held)
        payload["commitment_release_reason"] = self.commitment_release_reason
        payload["commitment_task"] = self.commitment_task
        payload["commitment_slot"] = (
            None if self.commitment_slot is None else int(self.commitment_slot)
        )
        payload["commitment_slot_dimension"] = (
            None
            if self.commitment_slot_dimension is None
            else int(self.commitment_slot_dimension)
        )
        payload["commitment_belief_status"] = self.commitment_belief_status
        payload["commitment_belief_revision"] = (
            None
            if self.commitment_belief_revision is None
            else int(self.commitment_belief_revision)
        )
        payload["commitment_mask_legal"] = (
            None if self.commitment_mask_legal is None else bool(self.commitment_mask_legal)
        )
        payload["commitment_overridden"] = bool(self.commitment_overridden)
        payload["commitment_suppressed_command"] = bool(self.commitment_suppressed_command)
        payload["reservation_id"] = self.reservation_id
        payload["commitment_validity"] = {
            str(key): bool(value) for key, value in self.commitment_validity.items()
        }
        payload["reservation_executed"] = (
            None if self.reservation_executed is None else bool(self.reservation_executed)
        )
        return payload


def _round(value: Any) -> Any:
    return round(float(value), 6)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


class DecisionTraceWriter:
    """JSONL sink for :class:`DecisionFacts`, carrying the run identity on every line.

    The writer is attached to the adapter (``adapter.decision_trace = writer``) and is
    the only thing a trace-enabled run adds to the decision path: the adapter builds the
    facts after it has already decided, so nothing recorded can influence a choice.
    """

    def __init__(self, sink: str | os.PathLike[str] | TextIO, *, meta: Mapping[str, Any]) -> None:
        self.meta = dict(meta)
        if isinstance(sink, (str, os.PathLike)):
            self._owns_stream = str(sink) != "-"
            self._stream: TextIO = (
                sys.stdout
                if not self._owns_stream
                else open(os.fspath(sink), "w", encoding="utf-8")
            )
        else:
            self._owns_stream = False
            self._stream = sink
        self._closed = False
        self.records_written = 0
        self._write({"event_type": "trace_config", "format": TRACE_FORMAT, "meta": self.meta})

    def record(self, facts: DecisionFacts) -> None:
        if self._closed:
            raise RuntimeError("decision trace is already closed")
        self._write(facts.as_json(self.meta))

    def finish(self, summary: Mapping[str, Any] | None = None) -> None:
        self._write(
            {
                "event_type": "trace_final",
                "format": TRACE_FORMAT,
                "meta": self.meta,
                "records": self.records_written,
                "summary": _jsonable(dict(summary)) if summary else None,
            }
        )

    def reset(self) -> None:
        """A new episode keeps the same sink; the header is not repeated."""
        self.records_written = 0

    def _write(self, payload: Mapping[str, Any]) -> None:
        self._stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._stream.flush()
        # Only decisions count: the header and the final line are not decisions, and the
        # count is what the closing line reports back.
        if payload.get("event_type") == "decision":
            self.records_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_stream and self._stream is not sys.stdout:
            self._stream.close()

    def __enter__(self) -> "DecisionTraceWriter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def read_trace(path: str | os.PathLike[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(meta, decisions)`` for one trace file."""
    meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("event_type")
            if kind == "trace_config":
                meta = dict(record.get("meta") or {})
            elif kind == "decision":
                rows.append(record)
    return meta, rows


def legal_task_indices(decision: Mapping[str, Any]) -> list[int]:
    mask = decision.get("mask") or []
    ids = decision.get("action_task_ids") or []
    return [
        index
        for index in range(1, len(ids))
        if index < len(mask) and not mask[index] and ids[index] is not None
    ]


def ctde_margin(decision: Mapping[str, Any]) -> float | None:
    """Best legal task logit minus the depot logit, or None when it is not defined.

    The depot action is index 0 and ``get_hard_mask`` never masks it, so the margin is
    the allocator's own preference between "go work" and "go home" before the
    coordinator residual is added.
    """
    logits = decision.get("ctde_logits")
    if not logits:
        return None
    depot = logits[DEPOT_INDEX] if DEPOT_INDEX < len(logits) else None
    if depot is None:
        return None
    legal = [logits[index] for index in legal_task_indices(decision) if logits[index] is not None]
    if not legal:
        return None
    return max(legal) - depot


@dataclass(slots=True)
class FileMetrics:
    """Everything one trace file contributes to the arm-level report."""

    arm: str
    seed: int
    decisions: int = 0
    legal_decisions: int = 0
    no_task_on_menu: int = 0
    all_tasks_masked: int = 0
    busy_decisions: int = 0
    depot_with_legal: int = 0
    depot_with_legal_idle: int = 0
    depot_with_legal_committed: int = 0
    depot_with_legal_replan: int = 0
    depot_despite_margin: int = 0
    task_despite_margin: int = 0
    margin_positive: int = 0
    margins: list[float] = field(default_factory=list)
    task_commitments: int = 0
    arrived_commitments: int = 0
    switched_before_arrival: int = 0
    switch_to_task: int = 0
    switch_to_depot: int = 0
    durations: list[float] = field(default_factory=list)
    targets: dict[str, dict[str, float]] = field(default_factory=dict)
    # Coalition commitments. ``reserved_decisions`` counts decisions made by a robot
    # holding a commitment for a task on its own menu. Of those, ``executed_reservations``
    # ends on the committed task and ``raw_choice_taken_reservations`` is the subset where
    # the allocator's own preference was already that task. The two differ exactly where a
    # commitment forced a choice the residual did not want, so they are counted separately
    # rather than pooled into one "followed the plan" number.
    reserved_decisions: int = 0
    executed_reservations: int = 0
    raw_choice_taken_reservations: int = 0
    # The planner's own per-cycle counters, keyed by advice revision so a cycle that
    # appears on several decision rows is counted once.
    advice_plans: dict[int, dict[str, Any]] = field(default_factory=dict)
    # First timestamp at which an advice held a commitment for each target. Kept apart
    # from ``targets`` so a target that was reserved but never dispatched cannot leak
    # into the per-target tables the other metrics are pooled from.
    reservation_times: dict[str, float] = field(default_factory=dict)
    # Timings measured from detection, and the capability gap left at the last look.
    dispatched_targets: int = 0
    never_reserved: int = 0
    reservation_gaps: list[float] = field(default_factory=list)
    arrival_gaps: list[float] = field(default_factory=list)
    group_arrival_gaps: list[float] = field(default_factory=list)
    progress_gaps: list[float] = field(default_factory=list)
    open_dimensions: list[int] = field(default_factory=list)
    oversupplied_targets: int = 0
    excess_units: list[float] = field(default_factory=list)

    def plan_totals(self) -> dict[str, int]:
        """Sum the planner's cycle counters, counting each advice revision once.

        The counters describe the transition into a revision -- what expired, what was
        superseded, what the new plan cancelled -- so a revision seen on twenty decision
        rows is still one transition. Summing per row would multiply it by twenty.
        """
        totals = {key: 0 for key in PLAN_COUNTERS}
        for plan in self.advice_plans.values():
            for key in PLAN_COUNTERS:
                value = plan.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    totals[key] += value
        return totals

    def as_row(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "decisions": self.decisions,
            "legal_decisions": self.legal_decisions,
            "no_task_on_menu": self.no_task_on_menu,
            "all_tasks_masked": self.all_tasks_masked,
            "busy_decisions": self.busy_decisions,
            "depot_with_legal": self.depot_with_legal,
            "depot_with_legal_share": _ratio(self.depot_with_legal, self.legal_decisions),
            "depot_with_legal_idle": self.depot_with_legal_idle,
            "depot_with_legal_committed": self.depot_with_legal_committed,
            "depot_with_legal_replan": self.depot_with_legal_replan,
            "depot_despite_margin": self.depot_despite_margin,
            "task_despite_margin": self.task_despite_margin,
            "margin_positive": self.margin_positive,
            "margin_median": _median(self.margins),
            "task_commitments": self.task_commitments,
            "arrived_commitments": self.arrived_commitments,
            "switched_before_arrival": self.switched_before_arrival,
            "switch_to_task": self.switch_to_task,
            "switch_to_depot": self.switch_to_depot,
            "duration_median": _median(self.durations),
            "reserved_decisions": self.reserved_decisions,
            "executed_reservations": self.executed_reservations,
            "raw_choice_taken_reservations": self.raw_choice_taken_reservations,
            "dispatched_targets": self.dispatched_targets,
            "never_reserved": self.never_reserved,
            "never_reserved_share": _ratio(self.never_reserved, self.dispatched_targets),
            "reservation_gap_median": _median(self.reservation_gaps),
            "arrival_gap_median": _median(self.arrival_gaps),
            "group_arrival_gap_median": _median(self.group_arrival_gaps),
            "progress_gap_median": _median(self.progress_gaps),
            "open_dimensions": list(self.open_dimensions),
            "oversupplied_targets": self.oversupplied_targets,
            "excess_units_median": _median(self.excess_units),
            "plan_totals": self.plan_totals(),
            "targets": {
                target: dict(row) for target, row in sorted(self.targets.items())
            },
        }


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if not denominator else numerator / denominator


def _median(values: Sequence[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[int(fraction * (len(ordered) - 1))], 3)


def analyse(rows: Sequence[Mapping[str, Any]], *, arm: str, seed: int) -> FileMetrics:
    """Per-file metrics: decision outcomes, churn and group formation."""
    metrics = FileMetrics(arm=arm, seed=seed)
    per_agent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for decision in rows:
        metrics.decisions += 1
        legal = legal_task_indices(decision)
        margin = ctde_margin(decision)
        # "Chose the depot" means very different things depending on the menu: an empty
        # action space is not a preference, and a fully masked one is a hard constraint.
        if len(decision.get("action_task_ids") or ()) <= 1:
            metrics.no_task_on_menu += 1
        elif not legal:
            metrics.all_tasks_masked += 1
        if decision.get("current_kind") is not None or str(decision.get("status")) != "idle":
            metrics.busy_decisions += 1
        if margin is not None:
            metrics.margins.append(margin)
            metrics.margin_positive += int(margin > 0.0)
        if legal:
            metrics.legal_decisions += 1
            if _chosen_is_depot(decision):
                metrics.depot_with_legal += 1
                if decision.get("current_kind") is None:
                    metrics.depot_with_legal_idle += 1
                else:
                    metrics.depot_with_legal_committed += 1
            if margin is not None:
                if margin > 0.0 and _chosen_is_depot(decision):
                    metrics.depot_despite_margin += 1
                if margin <= 0.0 and not _chosen_is_depot(decision):
                    metrics.task_despite_margin += 1
        if decision.get("replanned") and _chosen_is_depot(decision) and legal:
            metrics.depot_with_legal_replan += 1
        commitment = decision.get("held_reservation")
        if isinstance(commitment, Mapping):
            metrics.reserved_decisions += 1
            # Schema /1 records carried a single ``honoured`` flag, meaning the raw choice
            # took the committed task -- which under an advisory allocator is the same event
            # as executing it. Falling back to it keeps the frozen artefacts analysable
            # under the schema they were written with; /2 records carry both facts.
            executed = commitment.get("executed", commitment.get("honoured"))
            raw_taken = commitment.get("raw_choice_taken", commitment.get("honoured"))
            if executed:
                metrics.executed_reservations += 1
            if raw_taken:
                metrics.raw_choice_taken_reservations += 1
        revision = decision.get("advice_revision")
        plan = decision.get("advice_plan")
        if revision is not None and isinstance(plan, Mapping):
            if any(key in plan for key in PLAN_COUNTERS):
                metrics.advice_plans[int(revision)] = dict(plan)
        per_agent[str(decision.get("agent_id"))].append(decision)
        _observe_target(metrics, decision)

    for agent_rows in per_agent.values():
        _walk_commitments(metrics, agent_rows)
    _measure_targets(metrics)
    return metrics


def _measure_targets(metrics: FileMetrics) -> None:
    """Per-file timings from detection, and the capability gap left at the last look.

    Measured here rather than in :func:`summarise` because the pooled view keeps the
    element-wise minimum of every float and would therefore read a vector by comparing
    it lexicographically: "the requirement still open at the last look" has to be read
    per file, before pooling, or it means nothing.
    """
    dimensions = len(metrics.open_dimensions)
    for row in metrics.targets.values():
        if row.get("first_dispatch") is None:
            continue
        metrics.dispatched_targets += 1
        detected = row.get("detected_at")
        if detected is not None:
            for key, bucket in (
                ("reserved_at", metrics.reservation_gaps),
                ("arrived_at", metrics.arrival_gaps),
                ("group_arrived_at", metrics.group_arrival_gaps),
                ("worked_at", metrics.progress_gaps),
            ):
                if key in row:
                    bucket.append(max(0.0, float(row[key]) - float(detected)))
        if "reserved_at" not in row:
            metrics.never_reserved += 1
        remaining = list(row.get("remaining_last") or ())
        if remaining:
            if not dimensions:
                metrics.open_dimensions = [0] * len(remaining)
                dimensions = len(remaining)
            for index in range(min(dimensions, len(remaining))):
                metrics.open_dimensions[index] += int(remaining[index] > 0)
        excess = float(row.get("capacity_excess") or 0.0)
        if excess > 0.0:
            metrics.oversupplied_targets += 1
            metrics.excess_units.append(excess)


def _chosen_is_depot(decision: Mapping[str, Any]) -> bool:
    return int(decision.get("chosen_index", DEPOT_INDEX)) == DEPOT_INDEX


def _observe_target(metrics: FileMetrics, decision: Mapping[str, Any]) -> None:
    """Record, per target, when it was first dispatched to and when its group formed.

    Also records the three timings the coalition experiment is judged on -- detection to
    a reservation, to the group being on site, and to the first step of work -- plus the
    requirement still open at the last look, which is that target's missing-capability
    vector. All of them are read off the ground-truth block of a decision, so they exist
    for every arm; an arm with no reserving policy simply never fills ``reserved_at``.
    """
    targets = (decision.get("truth") or {}).get("targets") or {}
    chosen = decision.get("chosen_task")
    timestamp = float(decision.get("timestamp") or 0.0)
    # Commitments are scanned first and kept in their own table. A plan can hold a
    # commitment for a target long before any decision names it -- that is the whole
    # point of planning ahead -- and the ground-truth block only ever describes the
    # target a decision is about, so a reservation read from that block would appear to
    # arrive at the same instant as the dispatch and the lead time would vanish.
    for target_id in decision.get("advice_reservations") or ():
        if not target_id:
            continue
        held = metrics.reservation_times.get(target_id)
        if held is None or timestamp < held:
            metrics.reservation_times[target_id] = timestamp
    for target_id, entry in targets.items():
        if not target_id:
            continue
        row = metrics.targets.setdefault(target_id, {})
        row.setdefault("deadline", float(entry.get("deadline") or 0.0))
        if entry.get("detected_at") is not None:
            # A detection stamped at t=0 is a real timestamp, so "absent" and "zero" must
            # stay distinguishable: the timings below are skipped when the key is missing
            # rather than when its value is falsy.
            row.setdefault("detected_at", float(entry["detected_at"]))
        row.setdefault("first_considered", timestamp)
        if target_id == chosen:
            row["first_dispatch"] = min(row.get("first_dispatch", timestamp), timestamp)
        reserved_at = metrics.reservation_times.get(target_id)
        if reserved_at is not None:
            row["reserved_at"] = reserved_at
        if entry.get("covered"):
            row["covered_at"] = min(row.get("covered_at", timestamp), timestamp)
        if entry.get("en_route_covers"):
            row["filled_at"] = min(row.get("filled_at", timestamp), timestamp)
        on_site = list(entry.get("on_site") or ())
        en_route = list(entry.get("en_route") or ())
        if on_site:
            row["arrived_at"] = min(row.get("arrived_at", timestamp), timestamp)
            if not en_route:
                row["group_arrived_at"] = min(
                    row.get("group_arrived_at", timestamp), timestamp
                )
        requirements = entry.get("requirements")
        remaining = entry.get("remaining")
        stage_index = entry.get("stage_index")
        if requirements is not None:
            row.setdefault("requirements", list(requirements))
            row.setdefault("dispatch_stage", stage_index)
        if requirements is not None and remaining is not None:
            if stage_index == row.get("dispatch_stage"):
                # Only the stage this target was dispatched in is tracked. A target that
                # advanced reports the *next* stage's requirement, which is open work by
                # definition and would otherwise read as a capability that never arrived.
                row["remaining_last"] = list(remaining)
            if list(remaining) != list(requirements):
                row["worked_at"] = min(row.get("worked_at", timestamp), timestamp)
        capacity = entry.get("capacity")
        if capacity is not None and requirements is not None:
            excess = sum(
                max(0.0, float(got) - float(need))
                for got, need in zip(capacity, requirements)
            )
            row["capacity_excess"] = max(row.get("capacity_excess", 0.0), excess)


def _walk_commitments(metrics: FileMetrics, rows: Sequence[Mapping[str, Any]]) -> None:
    """Turn one robot's decisions into task commitments and how each one ended.

    A commitment is an interval in which the robot's navigation target stayed the same
    task. It ends when a later decision shows a different target, which is the only way
    a target can change; a commitment that never shows ``status == "working"`` never
    reached the scene and therefore never delivered a step of work.
    """
    ordered = sorted(rows, key=lambda row: float(row.get("timestamp") or 0.0))
    open_key: tuple[str | None, str | None] | None = None
    started = 0.0
    last_seen = 0.0
    worked = False
    arrived = False

    def close(end_reason: str, ended: float) -> None:
        if open_key is None or open_key[0] != "task":
            return
        metrics.task_commitments += 1
        metrics.durations.append(max(0.0, ended - started))
        if worked or arrived:
            metrics.arrived_commitments += 1
        elif end_reason != "run_end":
            metrics.switched_before_arrival += 1
            if end_reason == "switched_to_task":
                metrics.switch_to_task += 1
            else:
                metrics.switch_to_depot += 1

    for decision in ordered:
        key = (decision.get("current_kind"), decision.get("current_task"))
        timestamp = float(decision.get("timestamp") or 0.0)
        if open_key is None or key != open_key:
            if open_key is not None:
                kind, _ = key
                close("switched_to_task" if kind == "task" else "switched_to_depot", timestamp)
            open_key, started, worked, arrived = key, timestamp, False, False
        if str(decision.get("status")) == "working":
            worked = True
        if decision.get("current_target_taken"):
            arrived = True
        last_seen = timestamp
    close("run_end", last_seen)


def _ci95(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), 0.0
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def summarise(rows: Sequence[FileMetrics]) -> dict[str, dict[str, Any]]:
    """Pool the per-seed metrics into one block per arm."""
    by_arm: dict[str, list[FileMetrics]] = defaultdict(list)
    for row in rows:
        by_arm[row.arm].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for arm, files in sorted(by_arm.items()):
        def total(field_name: str) -> int:
            return sum(getattr(item, field_name) for item in files)

        seeds = sorted(item.seed for item in files)
        margins = [value for item in files for value in item.margins]
        durations = [value for item in files for value in item.durations]
        shares = [
            share
            for share in (item.as_row()["depot_with_legal_share"] for item in files)
            if share is not None
        ]
        share_mean, share_ci = _ci95(shares)

        targets: dict[str, dict[str, float]] = {}
        for item in files:
            for target_id, row in item.targets.items():
                pooled = targets.setdefault(str(item.seed) + ":" + target_id, dict(row))
                for key, value in row.items():
                    pooled[key] = min(pooled.get(key, value), value)
        dispatch_gaps: list[float] = []
        covered_gaps: list[float] = []
        filled_gaps: list[float] = []
        reservation_gaps = [value for item in files for value in item.reservation_gaps]
        arrival_gaps = [value for item in files for value in item.arrival_gaps]
        group_arrival_gaps = [value for item in files for value in item.group_arrival_gaps]
        progress_gaps = [value for item in files for value in item.progress_gaps]
        excess_units = [value for item in files for value in item.excess_units]
        plan_totals = {
            key: sum(item.plan_totals()[key] for item in files) for key in PLAN_COUNTERS
        }
        dispatched = total("dispatched_targets")
        open_dimensions = [0] * max(
            (len(item.open_dimensions) for item in files), default=0
        )
        for item in files:
            for index, value in enumerate(item.open_dimensions):
                open_dimensions[index] += value
        never_reserved_shares = [
            share
            for share in (item.as_row()["never_reserved_share"] for item in files)
            if share is not None
        ]
        reserved_mean, reserved_ci = _ci95(never_reserved_shares)
        never_covered = 0
        never_filled = 0
        for row in targets.values():
            first = row.get("first_dispatch")
            if first is None:
                continue
            if "covered_at" in row:
                covered_gaps.append(row["covered_at"] - first)
            else:
                never_covered += 1
            if "filled_at" in row:
                filled_gaps.append(row["filled_at"] - first)
            else:
                never_filled += 1
            dispatch_gaps.append(first)

        summary[arm] = {
            "seeds": seeds,
            "files": len(files),
            "decisions": total("decisions"),
            "legal_decisions": total("legal_decisions"),
            "no_task_on_menu": total("no_task_on_menu"),
            "no_task_on_menu_share": _ratio(total("no_task_on_menu"), total("decisions")),
            "all_tasks_masked": total("all_tasks_masked"),
            "busy_decisions": total("busy_decisions"),
            "depot_with_legal": total("depot_with_legal"),
            "depot_with_legal_share": _ratio(total("depot_with_legal"), total("legal_decisions")),
            "depot_with_legal_share_mean_over_seeds": round(share_mean, 4),
            "depot_with_legal_share_ci95": round(share_ci, 4),
            "depot_with_legal_idle": total("depot_with_legal_idle"),
            "depot_with_legal_committed": total("depot_with_legal_committed"),
            "depot_with_legal_replan": total("depot_with_legal_replan"),
            "depot_despite_margin": total("depot_despite_margin"),
            "task_despite_margin": total("task_despite_margin"),
            "margin_positive_share": _ratio(total("margin_positive"), len(margins)),
            "margin_median": _median(margins),
            "margin_mean": round(statistics.mean(margins), 4) if margins else None,
            "margin_p90": _percentile(margins, 0.9),
            "task_commitments": total("task_commitments"),
            "arrived_commitments": total("arrived_commitments"),
            "switched_before_arrival": total("switched_before_arrival"),
            "switched_before_arrival_share": _ratio(
                total("switched_before_arrival"), total("task_commitments")
            ),
            "switch_to_task": total("switch_to_task"),
            "switch_to_depot": total("switch_to_depot"),
            "duration_median": _median(durations),
            "duration_p90": _percentile(durations, 0.9),
            "targets_dispatched": len(targets),
            "group_covered_share": _ratio(len(covered_gaps), len(targets)),
            "group_covered_gap_median": _median(covered_gaps),
            "group_covered_gap_p90": _percentile(covered_gaps, 0.9),
            "group_filled_share": _ratio(len(filled_gaps), len(targets)),
            "group_filled_gap_median": _median(filled_gaps),
            "never_covered": never_covered,
            "never_filled": never_filled,
            # Coalition commitments: what the plan promised, whether the decision ended on
            # the committed task, and the ways a commitment ends (the plan dropped it, or
            # the clock did). ``executed_share`` is the fraction of decisions made under a
            # commitment that ended on the committed task; ``raw_choice_share`` is the
            # subset where the allocator's own preference was that task anyway, so the two
            # together say how much of the fidelity a binding commitment had to impose.
            "reserved_decisions": total("reserved_decisions"),
            "executed_reservations": total("executed_reservations"),
            "raw_choice_taken_reservations": total("raw_choice_taken_reservations"),
            "executed_share": _ratio(
                total("executed_reservations"), total("reserved_decisions")
            ),
            "raw_choice_share": _ratio(
                total("raw_choice_taken_reservations"), total("reserved_decisions")
            ),
            "plan_totals": plan_totals,
            # How far ahead of the allocator the plan runs: detection to the first
            # advice that holds a commitment for the target, to the group standing on
            # site, and to the first step of work. An arm with no reserving policy fills
            # only the last two, and its ``never_reserved`` is every dispatched target.
            "dispatched_targets": dispatched,
            "never_reserved": total("never_reserved"),
            "never_reserved_share": _ratio(total("never_reserved"), dispatched),
            "never_reserved_share_mean_over_seeds": round(reserved_mean, 4)
            if never_reserved_shares
            else None,
            "never_reserved_share_ci95": round(reserved_ci, 4)
            if never_reserved_shares
            else None,
            "reservation_gap_median": _median(reservation_gaps),
            "reservation_gap_p90": _percentile(reservation_gaps, 0.9),
            "arrival_gap_median": _median(arrival_gaps),
            "group_arrival_gap_median": _median(group_arrival_gaps),
            "progress_gap_median": _median(progress_gaps),
            # Which ability dimensions were still open the last time each dispatched
            # target was looked at, as a share of dispatched targets. This is the
            # missing-capability vector: it says *what* never arrived, not just how many
            # targets were missed.
            "missing_capability_target_share": [
                round(value / dispatched, 4) if dispatched else None
                for value in open_dimensions
            ],
            "missing_capability_targets": open_dimensions,
            # Over-dispatch, measured at the target rather than at the plan: the first
            # time the abilities on site plus en route exceeded the stage requirement.
            "oversupplied_targets": total("oversupplied_targets"),
            "oversupplied_share": _ratio(total("oversupplied_targets"), dispatched),
            "excess_units_median": _median(excess_units),
        }
    return summary


def report(summary: Mapping[str, Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for arm in sorted(summary):
        block = summary[arm]
        lines.append(
            f"== {arm} | seeds {len(block['seeds'])} | decisions {block['decisions']} | "
            f"with a legal task {block['legal_decisions']} =="
        )
        lines.append(
            f"  action space: nothing but the depot  {_fmt(block['no_task_on_menu'])} "
            f"({_pct(block['no_task_on_menu_share'])} of decisions)   "
            f"tasks all masked {_fmt(block['all_tasks_masked'])}   "
            f"robot not idle {_fmt(block['busy_decisions'])}"
        )
        lines.append(
            f"  chose depot with a legal task      {_fmt(block['depot_with_legal'])} "
            f"({_pct(block['depot_with_legal_share'])}) "
            f"per-seed {_pct(block['depot_with_legal_share_mean_over_seeds'])}"
            f"±{block['depot_with_legal_share_ci95']:.3f}"
        )
        lines.append(
            f"    robot idle                       {_fmt(block['depot_with_legal_idle'])}   "
            f"already committed {_fmt(block['depot_with_legal_committed'])}   "
            f"replan-triggered {_fmt(block['depot_with_legal_replan'])}"
        )
        lines.append(
            f"  CTDE wanted a task (margin > 0)    {_pct(block['margin_positive_share'])} "
            f"; median margin {_num(block['margin_median'])} "
            f"(p90 {_num(block['margin_p90'])})"
        )
        lines.append(
            f"    wanted a task, depot was chosen  {_fmt(block['depot_despite_margin'])}   "
            f"wanted depot, a task was chosen {_fmt(block['task_despite_margin'])}"
        )
        lines.append(
            f"  task commitments                   {_fmt(block['task_commitments'])} "
            f"; reached the scene {_pct(_ratio(block['arrived_commitments'], block['task_commitments']))}"
        )
        lines.append(
            f"    switched before arriving          {_fmt(block['switched_before_arrival'])} "
            f"({_pct(block['switched_before_arrival_share'])}) "
            f"-> another task {_fmt(block['switch_to_task'])}, "
            f"-> depot {_fmt(block['switch_to_depot'])}"
        )
        lines.append(
            f"  commitment duration (steps)        median {_num(block['duration_median'])} "
            f"p90 {_num(block['duration_p90'])}"
        )
        lines.append(
            f"  group on site covers requirement   "
            f"{_pct(block['group_covered_share'])} of {block['targets_dispatched']} dispatched "
            f"targets, gap after first dispatch median {_num(block['group_covered_gap_median'])} "
            f"p90 {_num(block['group_covered_gap_p90'])} (never {block['never_covered']})"
        )
        lines.append(
            f"  with en-route robots added         "
            f"{_pct(block['group_filled_share'])} of targets, gap median "
            f"{_num(block['group_filled_gap_median'])} (never {block['never_filled']})"
        )
        plan = block["plan_totals"]
        lines.append(
            f"  coalition commitments              planned {_fmt(plan['planned_reservations'])} "
            f"; decisions under one {_fmt(block['reserved_decisions'])} "
            f"(ended on the committed task {_pct(block['executed_share'])}; "
            f"of those the raw choice already agreed "
            f"{_pct(block['raw_choice_share'])})"
        )
        lines.append(
            f"    ended by plan                    superseded "
            f"{_fmt(plan['superseded_reservations'])} cancelled "
            f"{_fmt(plan['cancelled_reservations'])} ; ended by clock expired "
            f"{_fmt(plan['expired_reservations'])} ; redundant candidates declined "
            f"{_fmt(plan['redundant_candidates'])}"
        )
        lines.append(
            f"  detection -> reservation           median "
            f"{_num(block['reservation_gap_median'])} steps (p90 "
            f"{_num(block['reservation_gap_p90'])}); never "
            f"{_fmt(block['never_reserved'])} of {_fmt(block['dispatched_targets'])} = "
            f"{_pct(block['never_reserved_share'])}"
        )
        lines.append(
            f"  detection -> group on site / work  median "
            f"{_num(block['arrival_gap_median'])} / {_num(block['progress_gap_median'])} "
            f"steps (all on site {_num(block['group_arrival_gap_median'])})"
        )
        vector = block["missing_capability_target_share"]
        lines.append(
            "  missing capability vector          "
            + (" ".join(_pct(value) for value in vector) if vector else "?")
            + f"  (share of {_fmt(block['dispatched_targets'])} dispatched targets still open"
            " at the last look, by ability dimension)"
        )
        lines.append(
            f"  over-dispatch                      {_pct(block['oversupplied_share'])} of "
            f"targets ever had more ability on site+en route than the stage needed "
            f"(median excess {_num(block['excess_units_median'])} units)"
        )
        lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return f"{int(value):>7}" if isinstance(value, (int, float)) else "      ?"


def _num(value: Any) -> str:
    return "?" if value is None else f"{float(value):.2f}"


def _pct(value: Any) -> str:
    return "?" if value is None else f"{float(value):.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="integrated_sar.allocation_trace",
        description="Report allocation decisions: depot bias, churn and group formation.",
    )
    parser.add_argument("patterns", nargs="+", help="trace globs, e.g. tr_*_s*.jsonl")
    parser.add_argument("--json", type=str, default=None, help="write the per-seed rows here")
    args = parser.parse_args(argv)

    paths: list[str] = []
    for pattern in args.patterns:
        paths.extend(sorted(globmodule.glob(pattern)))
    if not paths:
        print("no allocation traces matched", args.patterns)
        return 1

    rows: list[FileMetrics] = []
    for path in paths:
        meta, decisions = read_trace(path)
        if not decisions:
            continue
        rows.append(
            analyse(
                decisions,
                arm=str(meta.get("arm") or "unknown"),
                seed=int(meta.get("seed") or 0),
            )
        )
    summary = summarise(rows)
    print(report(summary))

    if args.json:
        payload = {
            "format": TRACE_FORMAT,
            "files": len(paths),
            "arms": summary,
            "per_seed": [row.as_row() for row in rows],
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"per-seed decision metrics written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())