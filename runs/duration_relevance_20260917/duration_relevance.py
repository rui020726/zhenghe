"""Phase A of the stage-duration ablation: is the duration the message carries relevant?

The commitment-continuity round left six legacy targets expiring under a binding
commitment with no explanation but "the candidates were ordered wrong". The coalition
planner's own docstring names the suspect: ``AgentTaskBelief`` keeps a task's
requirements and deadline but drops the work duration the ``DetectionMessage``
carried, so every budget check online is optimistic by exactly the work time.

This script decides whether that dropped number could have changed anything, before
any contract is touched. It does two things and nothing else:

1. Replays the affected seeds on the recorded ``coalplan0.25sticky`` configuration,
   recording every ``DetectionMessage`` every robot actually received. The replay is
   accepted only when its decision sequence is identical to the committed trace, so
   what follows is a measurement of the run that produced those numbers, not of a
   look-alike.
2. Reads the durations back out of the messages those robots were handed, and asks the
   question the online coordinator cannot: would the group it chose have arrived and
   worked inside the deadline once the work time is subtracted -- and would a
   different legal candidate have made it?

Nothing here is imported by ``integrated_sar``, no belief field is added, and no arm is
registered. If the gate fails, this directory is the whole result.

Run from the repository root::

    D:/project_env/python.exe runs/duration_relevance_20260917/duration_relevance.py record
    D:/project_env/python.exe runs/duration_relevance_20260917/duration_relevance.py analyse
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from integrated_sar.adapters.task03 import Task03Adapter  # noqa: E402
from integrated_sar.coordinator import (  # noqa: E402
    CoalitionPlannerCoordinator,
    _travel_steps,
)
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402

ARMS_DIR = REPO / "runs" / "coordinator_baselines_20260917" / "heldout" / "arms"
ARM = "coalplan0.25sticky"
CONTINUITY = REPO / "runs" / "learnability_20260917" / "commitment_continuity.json"

# The six targets the binding arm still could not explain: the residual ``candidate_ordering_error``
# class after the execution breaks were fixed. Hard-coded rather than recomputed so this file stays
# readable next to the evidence it is judging, and cross-checked against the artefact below.
ORDERING_FAILURES = [
    (20003, "piloc-e1-target-1"),
    (20003, "piloc-e1-target-0"),
    (20008, "piloc-e1-target-2"),
    (20052, "piloc-e1-target-3"),
    (20070, "piloc-e1-target-2"),
    (20083, "piloc-e1-target-4"),
]

_FIELDS = {field.name for field in dataclasses.fields(RunConfiguration)}


def committed_rows(seed: int) -> list[dict]:
    """The decision records of the run being analysed, in order."""
    path = ARMS_DIR / ARM / f"tr_{ARM}_s{seed}.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("event_type") == "decision"
    ]


def recorded_targets() -> list[int]:
    """Distinct seeds the six failures fall on, plus what the artefact says about them."""
    payload = json.loads(CONTINUITY.read_text(encoding="utf-8"))
    rows = payload["sticky"]["analysis_legacy"]["binding_expired_rows"]
    ordering = [row for row in rows if row["class_reason"] == "candidate_ordering_error"]
    keyed = {(int(row["seed"]), str(row["target"])): row for row in ordering}
    expected = {(seed, target) for seed, target in ORDERING_FAILURES}
    if set(keyed) != expected:
        raise SystemExit(
            f"the ordering failures moved: artefact says {sorted(keyed)}, this file expects {sorted(expected)}"
        )
    return sorted({seed for seed, _ in ORDERING_FAILURES})


def arm_configuration(seed: int) -> RunConfiguration:
    """Rebuild the recorded arm's configuration, pointed at scratch outputs."""
    path = ARMS_DIR / ARM / f"rp_{ARM}_s{seed}.jsonl"
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    stored = dict(first["detail"]["configuration"])
    stored["output"] = str(HERE / f"replay_rp_s{seed}.jsonl")
    stored["decision_trace"] = str(HERE / f"replay_tr_s{seed}.jsonl")
    stored["quiet"] = True
    # ``as_dict`` appends the two derived backends alongside the fields; they are
    # properties of the configuration, not arguments to it.
    derived = {key: stored.pop(key) for key in ("search_backend", "allocation_backend") if key in stored}
    unknown = set(stored) - _FIELDS
    if unknown:
        raise SystemExit(f"recorded configuration carries unknown fields: {sorted(unknown)}")
    configuration = RunConfiguration(**stored)
    if (configuration.search_backend, configuration.allocation_backend) != (
        derived.get("search_backend"),
        derived.get("allocation_backend"),
    ):
        raise SystemExit(
            f"seed {seed} replays on a different backend than the recorded run: "
            f"{derived} vs {configuration.search_backend}/{configuration.allocation_backend}"
        )
    return configuration


def _message_dict(message) -> dict:
    return {
        "event_id": message.event_id,
        "target_id": message.target_id,
        "sent_at": float(message.sent_at),
        "position": [float(message.position.x), float(message.position.y)],
        "stages": [
            {"requirements": list(stage.requirements), "duration": float(stage.duration)}
            for stage in message.stages
        ],
        "current_stage": int(message.current_stage),
        "remaining_requirements": list(message.remaining_requirements),
        "deadline": float(message.deadline),
        "severity": float(message.severity),
        "confidence": float(message.confidence),
        "progress": float(message.progress),
        "members": list(message.members),
        "revision": int(message.revision),
    }


def _belief_dict(belief) -> dict:
    return {
        "task_id": belief.task_id,
        "position": [float(belief.position.x), float(belief.position.y)],
        "remaining_requirements": list(belief.remaining_requirements),
        "deadline": float(belief.deadline),
        "stage_index": int(belief.stage_index),
        "progress": float(belief.progress),
        "severity": float(belief.severity),
        "future_requirements": list(belief.future_requirements),
        "members": list(belief.members),
        "updated_at": float(belief.updated_at),
        "finished": bool(belief.finished),
        "dynamic": bool(belief.dynamic),
        "confidence": float(belief.confidence),
        "revision": int(belief.revision),
    }


def _snapshot_agents(agents) -> list[dict]:
    return [
        {
            "agent_id": str(agent.agent_id),
            "position": [float(agent.position.x), float(agent.position.y)],
            "abilities": list(agent.abilities),
            "status": str(agent.status),
            "current_task": agent.current_task,
            "beliefs": [_belief_dict(belief) for belief in agent.task_beliefs],
        }
        for agent in agents
    ]


def record_seed(seed: int) -> dict:
    """Replay one seed, capturing deliveries, and check the decisions still match."""
    run = build_joint_run(arm_configuration(seed))
    deliveries: list[dict] = []
    advisories: list[dict] = []
    original = run.task_adapter.deliver_detection

    def watching(message, agent_id):
        accepted = original(message, agent_id)
        deliveries.append(
            {
                "at": float(run.orchestrator.current_time),
                "agent_id": str(agent_id),
                "accepted": bool(accepted),
                "message": _message_dict(message),
            }
        )
        return accepted

    run.task_adapter.deliver_detection = watching

    # The relay carries the detections, but a robot's belief also moves through the
    # executor's own sensing and gossip, so the ordering has to be rebuilt from the
    # beliefs the coordinator was actually handed rather than from the letters alone.
    coordinator = run.orchestrator.coordinator
    original_advise = coordinator.advise

    def watching_advice(typed_input):
        advice = original_advise(typed_input)
        advisories.append(
            {
                "at": float(typed_input.current_time),
                "agents": _snapshot_agents(typed_input.agents),
                "revision": int(advice.revision),
                "advice_valid_until": float(advice.valid_until),
                "reservations": [
                    {
                        "agent_id": str(agent_id),
                        "task_id": str(task_id),
                        "ability_slot": int(reservation.ability_slot),
                        "slot_dimension": int(reservation.slot_dimension),
                        "valid_until": float(reservation.valid_until),
                        "score": float(reservation.score),
                    }
                    for (agent_id, task_id), reservation in sorted(
                        advice.reservations.items(),
                        key=lambda item: (str(item[0][0]), str(item[0][1])),
                    )
                ],
            }
        )
        return advice

    coordinator.advise = watching_advice
    run.run()

    expected = committed_rows(seed)
    produced = [
        json.loads(line)
        for line in (HERE / f"replay_tr_s{seed}.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("event_type") == "decision"
    ]
    if len(produced) != len(expected):
        raise SystemExit(f"seed {seed}: replay wrote {len(produced)} decisions, the run wrote {len(expected)}")
    mismatches = []
    for before, after in zip(expected, produced):
        differs = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
        if differs:
            mismatches.append(
                {
                    "step": before.get("step"),
                    "agent_id": before.get("agent_id"),
                    "keys": differs,
                    "committed": {key: before.get(key) for key in differs},
                    "replayed": {key: after.get(key) for key in differs},
                }
            )
    payload = {
        "seed": seed,
        "arm": ARM,
        "faithful": not mismatches,
        "decisions": len(produced),
        "deliveries": len(deliveries),
        "accepted_deliveries": sum(1 for row in deliveries if row["accepted"]),
        "advice_observations": len(advisories),
        "mismatches": mismatches[:5],
        "messages": deliveries,
        "advisories": advisories,
    }
    (HERE / f"deliveries_s{seed}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return payload


def do_record() -> None:
    seeds = recorded_targets()
    print(f"replaying {len(seeds)} seeds of {ARM}: {seeds}", flush=True)
    for seed in seeds:
        payload = record_seed(seed)
        print(
            f"seed {seed}: {payload['decisions']} decisions, "
            f"{payload['accepted_deliveries']}/{payload['deliveries']} deliveries accepted, "
            f"faithful={payload['faithful']}",
            flush=True,
        )
        if not payload["faithful"]:
            print(json.dumps(payload["mismatches"], indent=1), flush=True)
            raise SystemExit(f"seed {seed} replay is not the recorded run; stop")
    # The recording proved the replay reproduces the run, so the next step is allowed to
    # reason over these decisions instead of the committed ones.
    (HERE / "record.done").write_text("ok", encoding="utf-8")


def robot_velocity() -> float:
    """The step-per-second the executor actually moves robots by.

    Read from the same constant the offline attribution and the oracle both use, so the
    ETA in this file is the ETA the allocator is choosing between rather than a third
    model of travel.
    """
    from integrated_sar.expired_attribution import ROBOT_VELOCITY

    return float(ROBOT_VELOCITY)


def stage_work(durations: list[float], stage_index: int, progress: float) -> tuple[float, float]:
    """(remaining time on the current stage, time still owed by later stages).

    ``AgentTaskBelief.progress`` is whole-task: ``(stage_index + stage_fraction) / total``,
    and the fraction only moves while the stage is fully covered -- which is the same rule
    the simulator uses when it refills the stage clock the moment cover breaks. So
    ``duration * (1 - fraction)`` is the covered work genuinely still owed, not a
    wall-clock guess, and both numbers come from what the robot was told.
    """
    total = max(1, len(durations))
    index = min(max(int(stage_index), 0), total - 1)
    fraction = min(1.0, max(0.0, float(progress) * total - index))
    remaining = float(durations[index]) * (1.0 - fraction)
    return remaining, sum(float(value) for value in durations[index + 1:])


def _durations_by_target(payload: dict) -> dict[str, list[float]]:
    """Stage durations exactly as the detection letters carried them."""
    table: dict[str, list[float]] = {}
    for row in payload["messages"]:
        message = row["message"]
        durations = [float(stage["duration"]) for stage in message["stages"]]
        previous = table.get(message["target_id"])
        if previous is not None and previous != durations:
            raise SystemExit(
                f"{message['target_id']}: the letters disagree about the stage durations"
            )
        table[message["target_id"]] = durations
    return table


def _latest_advisory(payload: dict, moment: float) -> dict | None:
    due = [row for row in payload["advisories"] if row["at"] <= moment]
    return due[-1] if due else None


def decision_facts(payload: dict, velocity: float) -> list[dict]:
    """Every allocation decision, re-scored with the durations the letters carried.

    Nothing here reads the trace's ``truth`` block or the registry: the candidate list,
    the mask and the ranking come from the decision record, and every task number comes
    from what the deciding robot had been told.
    """
    durations = _durations_by_target(payload)
    rows: list[dict] = []
    trace = (HERE / f"replay_tr_s{payload['seed']}.jsonl").read_text(encoding="utf-8")
    for line in trace.splitlines():
        decision = json.loads(line)
        if decision.get("event_type") != "decision":
            continue
        moment = float(decision["timestamp"])
        advisory = _latest_advisory(payload, moment)
        if advisory is None:
            continue
        robot = next(
            (row for row in advisory["agents"] if row["agent_id"] == str(decision["agent_id"])),
            None,
        )
        if robot is None:
            continue
        beliefs = {belief["task_id"]: belief for belief in robot["beliefs"]}
        candidates = []
        for index, task_id in enumerate(decision["action_task_ids"]):
            # The mask is the simulator's: True marks a suppressed action.
            if task_id is None or decision["mask"][index]:
                continue
            belief = beliefs.get(task_id)
            if belief is None or belief["finished"]:
                continue
            specs = durations.get(task_id)
            if not specs:
                continue
            eta = math.hypot(
                robot["position"][0] - belief["position"][0],
                robot["position"][1] - belief["position"][1],
            ) / velocity
            remaining, future = stage_work(specs, belief["stage_index"], belief["progress"])
            slack = belief["deadline"] - moment
            candidates.append(
                {
                    "index": index,
                    "task_id": task_id,
                    "eta": round(eta, 3),
                    "remaining_work": round(remaining, 3),
                    "future_work": round(future, 3),
                    "slack": round(slack, 3),
                    "margin": round(slack - eta - remaining, 3),
                    "stage_index": belief["stage_index"],
                    "stages": len(specs),
                    "final_logit": (decision["final_logits"] or [])[index],
                }
            )
        if not candidates:
            continue
        # The ranking the run used, and the ranking the same candidates get once the
        # work time is subtracted from the budget: a candidate that cannot fit its own
        # remaining work before its deadline falls behind every candidate that can, and
        # nothing else moves.
        original = sorted(candidates, key=lambda row: -(row["final_logit"] or 0.0))
        aware = sorted(
            candidates,
            key=lambda row: (row["margin"] < 0.0, -(row["final_logit"] or 0.0)),
        )
        rows.append(
            {
                "seed": payload["seed"],
                "step": decision["step"],
                "timestamp": moment,
                "agent_id": robot["agent_id"],
                "abilities": robot["abilities"],
                "chosen_task": decision["chosen_task"],
                "commitment_held": decision.get("commitment_held"),
                "original_order": [row["task_id"] for row in original],
                "duration_aware_order": [row["task_id"] for row in aware],
                "candidates": candidates,
            }
        )
    return rows


def target_report(payload: dict, rows: list[dict], target: str) -> dict:
    """What the durations would have changed for one failing target."""
    involved = [row for row in rows if target in row["original_order"]]
    promotions = []
    for row in involved:
        before = row["original_order"].index(target)
        after = row["duration_aware_order"].index(target)
        facts = next(item for item in row["candidates"] if item["task_id"] == target)
        if after >= before:
            continue
        displaced = row["original_order"][after]
        loser = next(item for item in row["candidates"] if item["task_id"] == displaced)
        promotions.append(
            {
                "step": row["step"],
                "timestamp": row["timestamp"],
                "agent_id": row["agent_id"],
                "chose": row["chosen_task"],
                "target_rank": [before, after],
                "promoted_over": displaced,
                "target_facts": facts,
                "displaced_facts": loser,
            }
        )
    best = max((max(item["margin"] for item in row["candidates"] if item["task_id"] == target)
                for row in involved), default=None)
    return {
        "target": target,
        "decisions_seeing_it": len(involved),
        "best_feasibility_margin": None if best is None else round(best, 3),
        "ever_feasible": bool(best is not None and best >= 0.0),
        "promotions": promotions,
        "promotion_count": len(promotions),
    }


def do_analyse() -> dict:
    velocity = robot_velocity()
    report: dict[str, object] = {"robot_velocity": velocity, "seeds": {}}
    for seed in recorded_targets():
        payload = json.loads((HERE / f"deliveries_s{seed}.json").read_text(encoding="utf-8"))
        if not payload["faithful"]:
            raise SystemExit(f"seed {seed} was never shown to replay faithfully")
        rows = decision_facts(payload, velocity)
        targets = [target for s, target in ORDERING_FAILURES if s == seed]
        report["seeds"][str(seed)] = {
            "decisions_scored": len(rows),
            "targets": {target: target_report(payload, rows, target) for target in targets},
        }
    (HERE / "duration_facts.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    for seed, block in report["seeds"].items():
        for target, facts in block["targets"].items():
            print(
                f"seed {seed} {target}: {facts['decisions_seeing_it']} decisions, "
                f"best margin {facts['best_feasibility_margin']}, "
                f"feasible {facts['ever_feasible']}, promotions {facts['promotion_count']}",
                flush=True,
            )
    return report


def _knowledge_of(payload: dict) -> dict[str, dict[str, dict]]:
    """What each robot was told, and when it first heard it.

    Built per robot from that robot's own accepted deliveries, so a duration here is a
    duration that robot actually received -- not a table of truth the analysis happens
    to hold. ``first_received`` is what gates every use of it below.
    """
    knowledge: dict[str, dict[str, dict]] = {}
    for row in payload["messages"]:
        if not row["accepted"]:
            continue
        message = row["message"]
        per_robot = knowledge.setdefault(row["agent_id"], {})
        entry = per_robot.setdefault(
            message["target_id"],
            {
                "durations": [float(stage["duration"]) for stage in message["stages"]],
                "first_received": float(row["at"]),
            },
        )
        entry["first_received"] = min(entry["first_received"], float(row["at"]))
    return knowledge


class DurationAwarePlanner(CoalitionPlannerCoordinator):
    """The binding coalition planner, allowed to read the duration its belief dropped.

    A prototype for this gate only: it lives in the analysis directory, is not
    registered as an arm, and changes one thing -- the pairs the planner is willing to
    spend a free robot on. Two modes, because the question is whether duration
    discriminates at all rather than which formula wins:

    ``prune``  a pair whose believed slack cannot cover its own travel plus the covered
               work still owed is dropped from the plan entirely. The estimate is
               deliberately optimistic (this robot alone, this stage only, no queue for
               the other slots), so what is dropped could not have finished even under
               the most favourable reading of its own belief. This is the same rule the
               planner already applies to a task past its believed deadline, with the
               work time added.
    ``rank``   nothing is dropped; the same margin enters ``_row_value`` as one bounded
               term, so a pair with budget left outranks one without.
    """

    def __init__(
        self,
        knowledge: dict[str, dict[str, dict]],
        mode: str = "prune",
        robot_velocity: float = 0.2,
        horizon: float = 50.0,
        weight: float = 0.35,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if mode not in {"prune", "rank"}:
            raise ValueError(f"unknown duration mode {mode!r}")
        self.knowledge = knowledge
        self.mode = mode
        self.robot_velocity = float(robot_velocity)
        self.horizon = max(float(horizon), 1e-6)
        self.weight = float(weight)
        self.pruned_pairs = 0
        self.used_pairs = 0

    def margin(self, agent, belief, now: float) -> float | None:
        """Believed budget left after this robot arrives and this stage is worked.

        ``None`` means the robot was never told the work time -- it has no opinion, and
        the pair is judged exactly as the recorded arm judged it.
        """
        told = self.knowledge.get(str(agent.agent_id), {}).get(belief.task_id)
        if told is None or told["first_received"] > now:
            return None
        remaining, _owed_later = stage_work(
            told["durations"], int(belief.stage_index), float(belief.progress)
        )
        eta = _travel_steps(agent.position, belief.position, self.robot_velocity)
        return float(belief.deadline) - now - eta - remaining

    def _usable(self, agent, belief, now: float) -> bool:
        margin = self.margin(agent, belief, now)
        if margin is None:
            return True
        self.used_pairs += 1
        return margin >= 0.0

    def _plan(self, agents, candidates, committed, now):
        if self.mode != "prune":
            return super()._plan(agents, candidates, committed, now)
        narrowed = []
        for order, agent_id, abilities, beliefs in candidates:
            agent = agents[agent_id]
            kept = tuple(belief for belief in beliefs if self._usable(agent, belief, now))
            self.pruned_pairs += len(beliefs) - len(kept)
            narrowed.append((order, agent_id, abilities, kept))
        return super()._plan(agents, narrowed, committed, now)

    def _row_value(self, agent, belief, fill, now):
        base = super()._row_value(agent, belief, fill, now)
        if self.mode != "rank":
            return base
        margin = self.margin(agent, belief, now)
        if margin is None:
            return base
        self.used_pairs += 1
        return base + self.weight * max(-1.0, min(1.0, margin / self.horizon))


def outcomes(path: Path) -> dict[str, str]:
    """Final believed status of every target a lifecycle log recorded."""
    table: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        target = record.get("target_id")
        status = record.get("status")
        if target and target != "__run__" and status:
            table[str(target)] = str(status)
    return table


def counterfactual(seed: int, mode: str) -> dict:
    """Re-run one recorded seed with the planner told the work durations."""
    import integrated_sar.joint_driver as joint_driver

    payload = json.loads((HERE / f"deliveries_s{seed}.json").read_text(encoding="utf-8"))
    knowledge = _knowledge_of(payload)
    variant_dir = HERE / "counterfactual" / mode
    variant_dir.mkdir(parents=True, exist_ok=True)

    configuration = arm_configuration(seed)
    configuration = dataclasses.replace(
        configuration,
        output=str(variant_dir / f"rp_duration_{mode}_s{seed}.jsonl"),
        decision_trace=str(variant_dir / f"tr_duration_{mode}_s{seed}.jsonl"),
    )

    original_build = joint_driver._build_coordinator
    built: list[DurationAwarePlanner] = []

    def build(stored_configuration):
        if stored_configuration.coordinator_policy != "coalition_planner":
            raise SystemExit(
                f"seed {seed} is not a coalition planner arm, the patch would not be a like-for-like"
            )
        planner = DurationAwarePlanner(
            knowledge,
            mode=mode,
            renew_commitments=stored_configuration.sticky_commitments,
        )
        built.append(planner)
        return planner

    joint_driver._build_coordinator = build
    try:
        run = build_joint_run(configuration)
        run.run()
    finally:
        joint_driver._build_coordinator = original_build

    planner = built[-1]
    return {
        "seed": seed,
        "mode": mode,
        "outcomes": outcomes(Path(configuration.output)),
        "baseline": outcomes(ARMS_DIR / ARM / f"rp_{ARM}_s{seed}.jsonl"),
        "pruned_pairs": planner.pruned_pairs,
        "duration_informed_pairs": planner.used_pairs,
    }


def do_counterfactual(mode: str) -> dict:
    report: dict[str, object] = {"mode": mode, "seeds": {}}
    for seed in recorded_targets():
        result = counterfactual(seed, mode)
        changed = {
            target: [before, result["outcomes"].get(target)]
            for target, before in sorted(result["baseline"].items())
            if result["outcomes"].get(target) != before
        }
        report["seeds"][str(seed)] = {**result, "changed": changed}
        print(
            f"seed {seed} [{mode}]: {result['duration_informed_pairs']} pairs informed, "
            f"{result['pruned_pairs']} pruned",
            flush=True,
        )
        for target, (before, after) in changed.items():
            print(f"    {target}: {before} -> {after}", flush=True)
    (HERE / f"counterfactual_{mode}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return report


def _lifecycle(seed: int, target: str) -> dict:
    """What the committed run did with one target, read from that run's own log."""
    path = ARMS_DIR / ARM / f"rp_{ARM}_s{seed}.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("target_id") == target
    ]
    if not rows:
        raise SystemExit(f"seed {seed}: {target} has no lifecycle records in the committed arm")
    times = [float(row["timestamp"]) for row in rows]
    covered = [row for row in rows if row["status"] == "in_progress"]
    expired = [t for t, row in zip(times, rows) if row["status"] == "expired"]
    return {
        "detected_at": min(times),
        "first_cover_at": min((float(r["timestamp"]) for r in covered), default=None),
        "expired_at": min(expired, default=None),
        "last_progress": covered[-1]["progress"] if covered else 0.0,
        "last_stage": covered[-1]["stage_index"] if covered else None,
        "total_stages": rows[0]["detail"]["total_stages"],
        "deadline_offset": rows[0]["detail"]["deadline_offset"],
        "never_worked": not covered,
    }


def do_gate() -> dict:
    """The Phase A verdict: could the dropped duration have changed any of the six."""
    velocity = robot_velocity()
    counterfactuals = {
        mode: json.loads((HERE / f"counterfactual_{mode}.json").read_text(encoding="utf-8"))["seeds"]
        for mode in ("prune", "rank")
    }
    verdict: dict[str, object] = {
        "question": "do the stage durations the DetectionMessage carries, and drops on entry "
                    "to AgentTaskBelief, change the outcome of the six ordering failures?",
        "gate": "at least 3 of 6 rescuable by duration-aware ordering, belief-only",
        "targets": {},
    }
    rescued = {"prune": 0, "rank": 0}
    regressed = {"prune": 0, "rank": 0}
    for seed in recorded_targets():
        payload = json.loads((HERE / f"deliveries_s{seed}.json").read_text(encoding="utf-8"))
        rows = decision_facts(payload, velocity)
        durations = _durations_by_target(payload)
        trace = [json.loads(line) for line in
                 (HERE / f"replay_tr_s{seed}.jsonl").read_text(encoding="utf-8").splitlines()
                 if json.loads(line).get("event_type") == "decision"]
        for target in [t for s, t in ORDERING_FAILURES if s == seed]:
            involved = [row for row in rows if target in row["original_order"]]
            in_budget = [
                row for row in involved
                if next(c for c in row["candidates"] if c["task_id"] == target)["margin"] >= 0.0
            ]
            seen = sum(1 for d in trace if target in d["action_task_ids"])
            legal = sum(
                1
                for d in trace
                if any(t == target and not d["mask"][i] for i, t in enumerate(d["action_task_ids"]))
            )
            life = _lifecycle(seed, target)
            owed, later = (
                stage_work(durations[target], int(life["last_stage"] or 0), float(life["last_progress"]))
                if not life["never_worked"] else (None, None)
            )
            entry = {
                "decisions_naming_it": seen,
                "decisions_where_it_was_legal": legal,
                "decisions_re_scored": len(involved),
                "decisions_still_in_budget": len(in_budget),
                "chosen_while_in_budget": sum(1 for row in in_budget if row["chosen_task"] == target),
                "best_feasibility_margin": max(
                    (c["margin"] for r in involved for c in r["candidates"] if c["task_id"] == target),
                    default=None,
                ),
                "promotions": sum(
                    1 for row in involved
                    if row["original_order"].index(target) > row["duration_aware_order"].index(target)
                ),
                "demotions": sum(
                    1 for row in involved
                    if row["original_order"].index(target) < row["duration_aware_order"].index(target)
                ),
                "lifecycle": life,
                "work_owed_when_last_seen": None if owed is None else round(owed, 3),
                "later_stages_owed": None if later is None else round(later, 3),
                "stage_durations": durations.get(target),
                "counterfactual": {
                    mode: counterfactuals[mode][str(seed)]["changed"].get(target)
                    for mode in ("prune", "rank")
                },
            }
            for mode in ("prune", "rank"):
                change = counterfactuals[mode][str(seed)]["changed"].get(target)
                if change and change[1] == "completed":
                    rescued[mode] += 1
                if change and change[1] not in ("completed", None):
                    regressed[mode] += 1
            verdict["targets"][f"{seed}/{target}"] = entry
            print(
                f"{seed}/{target}: legal {legal}/{seen}, in budget "
                f"{entry['decisions_still_in_budget']} chosen {entry['chosen_while_in_budget']}, "
                f"never worked {life['never_worked']}, cover from {life['first_cover_at']} "
                f"(detected {life['detected_at']}, deadline "
                f"{life['detected_at'] + life['deadline_offset']:.1f}), "
                f"owed {entry['work_owed_when_last_seen']}, "
                f"promotions {entry['promotions']}, demotions {entry['demotions']}",
                flush=True,
            )
    verdict["rescued"] = rescued
    verdict["regressed"] = regressed
    whole_run = {}
    for mode in ("prune", "rank"):
        changes = {seed: blk["changed"] for seed, blk in counterfactuals[mode].items() if blk["changed"]}
        whole_run[mode] = {
            "changed": changes,
            "completed": sum(1 for b in changes.values() for t in b.values() if t[1] == "completed"),
            "lost": sum(1 for b in changes.values() for t in b.values() if t[0] == "completed"),
        }
    verdict["whole_run"] = whole_run
    verdict["passed"] = max(rescued.values()) >= 3
    verdict["verdict"] = (
        "PASS" if verdict["passed"] else
        f"FAIL: duration-aware ordering rescued {max(rescued.values())} of 6"
    )
    (HERE / "DURATION_GATE.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(verdict["verdict"], "| rescued", rescued, "| regressed", regressed, flush=True)
    return verdict


def main(argv: list[str]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    command = argv[1] if len(argv) > 1 else "record"
    if command == "record":
        do_record()
    elif command == "analyse":
        do_analyse()
    elif command == "counterfactual":
        do_counterfactual(argv[2] if len(argv) > 2 else "prune")
    elif command == "gate":
        do_gate()
    else:
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main(sys.argv)
