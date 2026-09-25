"""B0: a passive information-path probe that records four things the lifecycle log cannot.

The archived rounds could answer "was it discovered / assigned / finished", but not
"did this robot ever receive it, when did it come to believe it, was that belief out of
date at the moment it chose an action, and when did it physically get there". Those five
questions are what the deployed-path information loss has always been measured *against*,
so until they are recorded the loss is not zero — it is unmeasurable.

The probe is built only when a path is configured, and it is deliberately limited to
reading:

* it never consumes a random number, so the channel's packet draws and the environment's
  scenario draws are untouched;
* it never writes into an environment, registry or belief structure, so no robot can act
  on what the probe recorded;
* it is not an actor observation: no observation vector reads from it, so an experiment
  that turns it on cannot change the policy's input;
* it writes one side file, leaving the lifecycle JSONL byte-identical to a probe-less run —
  which is what makes the paired-replay acceptance check (§B0 of the D1 revision) a
  comparison of files rather than of interpretations.

Identity discipline: goals are named by the registry's external ``target_id``, robots by
``str(agent_id)`` — the same strings the subscriber tuple uses. Internal integer task ids
are translated through the adapter's own map, and a translation that stops being one-to-one
is reported rather than guessed at.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

PROBE_SCHEMA = "sar-info-probe/1"

#: Configuration and artefact-name fields. Two runs may differ in these without differing
#: in behaviour, so a behavioural comparison has to remove them first — and say which ones.
IDENTITY_FIELDS = ("output", "info_probe", "decision_trace", "record_stream", "comm_transcript",
                  "search_probe")

#: The run summary's self-description block. It mixes the knobs the transport was configured
#: with and the counters it accumulated, so a caller cannot tell an echo from an observation
#: from inside it. The counters are compared field by field by whoever knows which of them a
#: given contrast is allowed to move; the declarations are compared in the configuration
#: record, which is what they are actually a copy of.
TRANSPORT_SELF_DESCRIPTION = "communication"


def behaviour_view(record: dict[str, Any]) -> dict[str, Any]:
    """A record with declarations and paths stripped, for comparing runs that wrote elsewhere.

    The configuration record is dropped entirely and the run-level metrics lose the transport
    self-description block: a relay that reports "I was told to run policy_owned_timing" is
    restating its input, not recording something that happened.
    """
    if record.get("event_type") == "run_config":
        return {}
    body = {key: value for key, value in record.items() if key != "detail"}
    detail = {key: value for key, value in (record.get("detail") or {}).items()
              if key != "configuration"}
    metrics = detail.get("metrics")
    if isinstance(metrics, dict):
        detail["metrics"] = {key: value for key, value in metrics.items()
                             if key not in IDENTITY_FIELDS
                             and key != TRANSPORT_SELF_DESCRIPTION}
    body["detail"] = detail
    return body


class InfoProbe:
    """Collects placed / delivery / knowledge / arrival records for one episode."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.placed: list[dict[str, Any]] = []
        self.deliveries: list[dict[str, Any]] = []
        self.knowledge: list[dict[str, Any]] = []
        self.arrivals: list[dict[str, Any]] = []
        self.staleness: dict[tuple[str, str], dict[str, Any]] = {}
        self._internal_to_external: dict[int, str] = {}
        self._known: set[tuple[str, str]] = set()
        self._arrived: set[tuple[str, str]] = set()
        self._arrived_pairs: set[tuple[str, str]] = set()
        self._planned_targets: set[str] = set()
        self.anomalies: list[str] = []

    # ------------------------------------------------------------- placement --------------
    def note_placed(self, goals: Iterable[tuple[str, Any]]) -> None:
        """Record the goals that exist at reset, whether or not anything ever finds them."""
        if self.placed:
            return
        for index, (target_id, position) in enumerate(goals):
            self.placed.append({
                "kind": "placed", "goal_index": index, "target_id": str(target_id),
                "grid_position": [int(position[0]), int(position[1])],
            })
            self._planned_targets.add(str(target_id))

    # ------------------------------------------------------------- delivery ---------------
    def note_delivery(self, *, step: int, target_id: str, robot_id: str, arrived: bool,
                      revision: int | None = None, payload_bytes: int | None = None,
                      ordinal: int | None = None, packet_index: int | None = None) -> None:
        """One packet, one named recipient: the fact ``packets_lost`` could not carry."""
        self.deliveries.append({
            "kind": "delivery", "step": int(step), "target_id": str(target_id),
            "robot_id": str(robot_id), "arrived": bool(arrived),
            "revision": revision, "payload_bytes": payload_bytes,
            "emission": ordinal, "packet_index": packet_index,
        })
        if arrived:
            self._arrived_pairs.add((str(target_id), str(robot_id)))

    # ---------------------------------------------------------- per-step sample -----------
    def sample(self, *, step: int, now: float, env: Any,
               external_to_internal: dict[str, int]) -> None:
        """Read beliefs and rosters once and record what changed since the last sample."""
        internal_to_external = self._map_ids(external_to_internal)
        truth = getattr(env, "task_dic", {}) or {}
        knowledge_by_agent = getattr(env, "agent_knowledge", {}) or {}
        for agent_id, knowledge in sorted(knowledge_by_agent.items()):
            robot_id = str(agent_id)
            for internal_id in sorted(knowledge.get("known_tasks", ()) or ()):
                target_id = internal_to_external.get(int(internal_id))
                if target_id is None:
                    self.anomalies.append(f"unknown_internal_task_id:{internal_id}@{step}")
                    continue
                pair = (target_id, robot_id)
                if pair not in self._known:
                    self._known.add(pair)
                    # What is knowable here is only "a packet of this goal's news reached this
                    # robot". Whether the belief gate applied it is the transcript's
                    # deliver.accepted, joined at analysis time.
                    channel = ("uplink_arrived" if pair in self._arrived_pairs
                               else ("preexisting" if step <= 1 else "non_uplink"))
                    self.knowledge.append({
                        "kind": "knowledge", "step": int(step), "target_id": target_id,
                        "robot_id": robot_id, "channel": channel,
                    })
                self._sample_staleness(pair, step=step, now=now, knowledge=knowledge,
                                       internal_id=int(internal_id), truth=truth)
        for internal_id, task in sorted(truth.items()):
            target_id = internal_to_external.get(int(internal_id))
            if target_id is None:
                continue
            agents = getattr(env, "agent_dic", {}) or {}
            for member in [str(m) for m in (task.get("members", ()) or ())]:
                pair = (target_id, member)
                if pair in self._arrived:
                    continue
                self._arrived.add(pair)
                agent = agents.get(_as_int(member))
                self.arrivals.append({
                    "kind": "arrival", "step": int(step), "target_id": target_id,
                    "robot_id": member,
                    "distance": _distance(agent, task) if agent is not None else None,
                })

    def _map_ids(self, external_to_internal: dict[str, int]) -> dict[int, str]:
        reverse: dict[int, str] = {}
        for external, internal in external_to_internal.items():
            if int(internal) in reverse and reverse[int(internal)] != str(external):
                self.anomalies.append(
                    f"internal_id_reused:{internal}:{reverse[int(internal)]}:{external}")
                continue
            reverse[int(internal)] = str(external)
        self._internal_to_external = reverse
        return reverse

    def _sample_staleness(self, pair: tuple[str, str], *, step: int, now: float,
                          knowledge: dict[str, Any], internal_id: int,
                          truth: dict[Any, Any]) -> None:
        task = truth.get(internal_id) or truth.get(str(internal_id))
        if task is None:
            return
        entry = self.staleness.setdefault(pair, {
            "kind": "staleness", "target_id": pair[0], "robot_id": pair[1], "checks": 0,
            "mismatch_checks": 0, "stage_mismatch_checks": 0, "members_mismatch_checks": 0,
            "progress_mismatch_checks": 0, "max_age_steps": 0.0, "first_mismatch_step": None,
        })
        believed_stage = knowledge.get("task_stage_index", {}).get(internal_id)
        true_stage = min(int(task.get("current_stage_index", 0)), len(task.get("stages", [0])) - 1)
        believed_members = [str(m) for m in (knowledge.get("task_members", {}).get(internal_id)
                                             or ())]
        true_members = sorted(str(m) for m in (task.get("members", ()) or ()))
        believed_progress = knowledge.get("task_progress", {}).get(internal_id)
        true_progress = _true_progress(task)
        last_update = knowledge.get("task_last_update_time", {}).get(internal_id)
        age = max(0.0, float(now) - float(last_update)) if last_update is not None else None
        stage_bad = believed_stage is not None and int(believed_stage) != int(true_stage)
        members_bad = sorted(believed_members) != true_members
        progress_bad = (believed_progress is not None and true_progress is not None
                        and abs(float(believed_progress) - float(true_progress)) > 1e-9)
        mismatch = bool(stage_bad or members_bad or progress_bad)
        entry["checks"] += 1
        entry["stage_mismatch_checks"] += int(stage_bad)
        entry["members_mismatch_checks"] += int(members_bad)
        entry["progress_mismatch_checks"] += int(progress_bad)
        if age is not None:
            entry["max_age_steps"] = max(float(entry["max_age_steps"]), float(age))
        if mismatch:
            entry["mismatch_checks"] += 1
            if entry["first_mismatch_step"] is None:
                entry["first_mismatch_step"] = int(step)

    # ------------------------------------------------------------------ output ------------
    def records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = list(self.placed)
        out.extend(self.deliveries)
        out.extend(self.knowledge)
        out.extend(self.arrivals)
        out.extend(sorted(self.staleness.values(), key=lambda row: (row["target_id"],
                                                                   row["robot_id"])))
        known_targets = {row["target_id"] for row in self.knowledge}
        channels: dict[str, int] = {}
        for row in self.knowledge:
            channels[row["channel"]] = channels.get(row["channel"], 0) + 1
        out.append({
            "kind": "probe_summary", "schema": PROBE_SCHEMA,
            "placed": len(self.placed), "deliveries": len(self.deliveries),
            "packets_arrived": sum(1 for row in self.deliveries if row["arrived"]),
            "packets_lost": sum(1 for row in self.deliveries if not row["arrived"]),
            "knowledge_rows": len(self.knowledge), "arrival_rows": len(self.arrivals),
            "knowledge_channels": channels,
            "staleness_pairs": len(self.staleness),
            "targets_known_by_someone": len(known_targets),
            "targets_never_known": len({row["target_id"] for row in self.placed} - known_targets),
            "arrived_recipient_pairs": len(self._arrived_pairs),
            "anomalies": sorted(set(self.anomalies))[:20],
            "anomaly_count": len(self.anomalies),
        })
        return out

    def write(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for row in self.records():
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return self.path

    def reset(self) -> None:
        """Drop one episode's worth of records; the caller writes the file before resetting."""
        self.__init__(self.path)


def _as_int(value: Any) -> Any:
    text = str(value)
    return int(text) if text.lstrip("-").isdigit() else value


def _distance(agent: dict[str, Any], task: dict[str, Any]) -> float | None:
    try:
        ax, ay = float(agent["location"][0]), float(agent["location"][1])
        tx, ty = float(task["location"][0]), float(task["location"][1])
    except Exception:  # a missing location must not abort a run mid-episode
        return None
    return round(math.hypot(ax - tx, ay - ty), 6)


def _true_progress(task: dict[str, Any]) -> float | None:
    if task.get("finished"):
        return 1.0
    stages = task.get("stages") or []
    index = int(task.get("current_stage_index", 0) or 0)
    if not stages or index >= len(stages):
        return None
    stage = stages[index]
    requirement = stage.get("requirements")
    status = stage.get("status")
    if requirement is None or status is None:
        return None
    total = float(sum(abs(float(value)) for value in requirement))
    if total <= 0.0:
        return 1.0
    remaining = float(sum(max(0.0, float(value)) for value in status))
    return round(1.0 - remaining / total, 6)
