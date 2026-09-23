"""Chain each goal through the eight stages, per information path, from stored logs only.

Four archived paths are read. They differ in exactly the ways that matter for
attribution, so they are reported separately and never pooled:

* whether search ran live or detections were replayed from a frozen stream;
* whether the uplink was the only way to learn (isolated) or the environment's own
  line-of-sight refresh and gossip were on (deployed);
* whether a per-message transcript exists at all, and whether the channel could drop.

Outcome counts and the on-time test come from the project's own
``integrated_sar.lifecycle_metrics.window_metrics`` rather than a re-derivation, so a
disagreement with the published tables would show up as a bug in this file and not as
a new result. The per-goal classes below refine those counts; the file reports whether
the two agree.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import closure_common as cc

from integrated_sar.lifecycle_metrics import window_metrics  # analysis only, no environment

REPO = cc.REPO_ROOT
WINDOW = 1000
RUN_MARKER = "__run__"

STAGES = ["placed", "discovered", "reported", "known", "dispatched", "on_site",
          "working", "outcome"]

ISOLATED_ARMS = ["send_once", "always_hold", "event_unbudgeted", "periodic_5", "periodic_20",
                 "event", "event_priority", "bound_only_3_10", "first_send_plus_bound_3_10"]
CONFIRM_ARMS = ["noack_send_once", "ack_send_once", "ack_retry_2", "ack_periodic_20",
                "ack_backoff_16"]

#: (start key, end key) pairs. Names beginning with ``comm_`` read the transcript.
HOPS: dict[str, tuple[str, str]] = {
    "detected_to_first_transmit": ("detected_at", "comm_first_send_step"),
    "detected_to_first_known": ("detected_at", "comm_first_known_step"),
    "first_transmit_to_first_known": ("comm_first_send_step", "comm_first_known_step"),
    "detected_to_dispatched": ("detected_at", "first_en_route"),
    "detected_to_on_site": ("detected_at", "first_assigned"),
    "detected_to_working": ("detected_at", "first_in_progress"),
    "detected_to_completed": ("detected_at", "completed_at"),
    "dispatched_to_working": ("first_en_route", "first_in_progress"),
    "known_to_deadline": ("comm_first_known_step", "deadline"),
    "detected_to_deadline": ("detected_at", "deadline"),
}


def path_specs() -> dict[str, dict[str, Any]]:
    def layer_seeds(layer: str) -> range:
        start = {"g6": 1000000, "g8": 1000100, "g10": 1000200, "g12": 1000300}[layer]
        return range(start, start + 16)

    return {
        "P1_deployed_live_search": {
            "groups": {"systemloop": {
                "dir": "runs/coalition_validation_20260918/systemloop",
                "arms": ["baseline", "candidate"],
                "lifecycle": "{arm}_rp_s{seed}.jsonl",
                "seeds": list(range(31000, 31060))}},
            "channel": {"delivery_probability": 1.0, "delay_ticks": 0, "transcript": False,
                        "line_of_sight_refresh": True, "gossip": True},
            "search_backend": "live PILOC weights, 60 seeds, 2 arms",
            "pairing": ("same seed and same scenario for both arms; the channel is lossless with "
                        "zero delay, so it contributes no randomness to the pair"),
        },
        "P2_deployed_replayed_search": {
            "groups": {"joint_eval": {
                "dir": "runs/joint_eval_20260916",
                "arms": ["nocoord", "rs0.25", "rs0.5"],
                "lifecycle": "rp_{arm}_s{seed}.jsonl",
                "seeds": list(range(10000, 10030))}},
            "channel": {"delivery_probability": 1.0, "delay_ticks": 0, "transcript": False,
                        "line_of_sight_refresh": True, "gossip": True},
            "search_backend": "detections replayed from a frozen stream: the search stage is "
                              "inherited from the recording run, not measured here",
            "pairing": "arms share the seed and the frozen detection stream; channel adds nothing",
        },
        "P3_isolated_budgeted_comm": {
            "groups": {layer: {
                "dir": f"runs/comm_task_pressure_learnability_20260922/runs/{layer}",
                "arms": ISOLATED_ARMS,
                "lifecycle": f"rp_{{arm}}_{layer}_s{{seed}}.jsonl",
                "transcript": f"ct_{{arm}}_{layer}_s{{seed}}.jsonl",
                "seeds": list(layer_seeds(layer))}
                for layer in ("g6", "g8", "g10", "g12")},
            "channel": {"delivery_probability": 1.0, "delay_ticks": 0, "transcript": True,
                        "line_of_sight_refresh": False, "gossip": False},
            "search_backend": "replayed detections, isolated (gossip_interval 100000), "
                              "29 300 B budget, policy_owned_timing",
            "pairing": "arms share the seed; the channel is deterministic, so a per-stream pair "
                       "is an exact match on everything except the send decisions",
        },
        "P4_lossy_confirmation": {
            "mode": "published_aggregate",
            "aggregate": "runs/comm_rule_confirm_20260921/COMM_RULE_CONFIRMATION_RESULTS.json",
            "groups": {setting: {
                "dir": f"runs/comm_rule_confirm_20260921/evaluation/{setting}",
                "arms": CONFIRM_ARMS,
                "transcript": f"ct_{setting}_{{arm}}_s{{seed}}.jsonl",
                "seeds": list(range(900000, 900200))}
                for setting in ("normal-lossy", "isolated-lossy", "isolated-lossless")},
            "channel": {"delivery_probability": 0.7, "delay_ticks": 0, "transcript": True,
                        "line_of_sight_refresh": None, "gossip": None},
            "search_backend": "replayed detections on 200 fresh streams; only normal-lossy keeps "
                              "line-of-sight refresh and gossip on",
            "why_not_rescanned": (
                "the round's own aggregate already carries per-arm discovered / on-time / "
                "expired / unresolved counts and the detection-to-assignment, -work and "
                "-completion latencies over 1 000 episodes each; re-deriving 1.65 GB of "
                "lifecycle logs would add cost and a second definition of the same quantity, "
                "so this round reads the aggregate and spot-checks the transcripts instead"
            ),
            "pairing": ("the forward-loss stream is consumed per episode, so two arms that send "
                        "different things draw different channel outcomes: same scenario, "
                        "not common random numbers"),
        },
    }


# ---------------------------------------------------------------------------- per file ------
def new_goal() -> dict[str, Any]:
    return {"detected_at": None, "deadline": None, "deadline_offset": None,
            "first_en_route": None, "first_assigned": None, "first_in_progress": None,
            "completed_at": None, "expired_seen": False, "max_progress": 0.0,
            "goal_level_assignments": 0, "rejections": 0, "final_status": None}


def chain_lifecycle(path: Path) -> dict[str, Any]:
    placed: int | None = None
    goals: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
            kind = record.get("event_type")
            target = str(record.get("target_id"))
            if kind == "run_config":
                placed = (record.get("detail", {}).get("configuration") or {}).get("num_goals")
                continue
            if target == RUN_MARKER:
                continue
            goal = goals.setdefault(target, new_goal())
            stamp = float(record.get("timestamp") or 0.0)
            detail = record.get("detail") or {}
            if kind == "detection":
                if goal["detected_at"] is None or stamp < goal["detected_at"]:
                    goal["detected_at"] = stamp
                if detail.get("deadline_offset") is not None:
                    goal["deadline_offset"] = float(detail["deadline_offset"])
            elif kind == "progress":
                status = str(record.get("status"))
                field = {"en_route": "first_en_route", "assigned": "first_assigned",
                         "in_progress": "first_in_progress", "completed": "completed_at"}.get(status)
                if field and goal[field] is None:
                    goal[field] = stamp
                if status == "expired":
                    goal["expired_seen"] = True
                if record.get("progress") is not None:
                    goal["max_progress"] = max(goal["max_progress"], float(record["progress"]))
            elif kind == "assignment":
                goal["goal_level_assignments"] += 1
            elif kind == "assignment_rejected":
                goal["rejections"] += 1
            elif kind == "final":
                goal["final_status"] = str(record.get("status"))
                if detail.get("deadline") is not None:
                    goal["deadline"] = float(detail["deadline"])
    for goal in goals.values():
        if goal["deadline"] is None and goal["detected_at"] is not None \
                and goal["deadline_offset"] is not None:
            goal["deadline"] = goal["detected_at"] + goal["deadline_offset"]
        goal.setdefault("deadline", None)
    return {"placed": placed, "goals": goals, "metrics": window_metrics(records, WINDOW).as_row()}


def new_comm_goal() -> dict[str, Any]:
    return {"event_step": None, "comm_first_send_step": None, "comm_sends": 0,
            "comm_refreshes": 0, "comm_bytes": 0, "comm_packets_arrived": 0,
            "comm_packets_lost": 0, "comm_first_known_step": None, "comm_robots_informed": 0,
            "comm_refusals": 0, "comm_first_refusal_step": None}


def chain_transcript(path: Path) -> dict[str, dict[str, Any]]:
    goals: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            kind = str(record.get("kind"))
            target = str(record.get("target") or "")
            if not target:
                continue
            goal = goals.setdefault(target, new_comm_goal())
            step = float(record.get("step") or 0.0)
            if kind == "event":
                if goal["event_step"] is None:
                    goal["event_step"] = step
            elif kind == "send":
                goal["comm_sends"] += 1
                if record.get("first"):
                    current = goal["comm_first_send_step"]
                    goal["comm_first_send_step"] = step if current is None else min(current, step)
                else:
                    goal["comm_refreshes"] += 1
                goal["comm_bytes"] += int(record.get("cost_bytes") or 0)
                goal["comm_packets_arrived"] += int(record.get("packets") or 0)
                goal["comm_packets_lost"] += int(record.get("packets_lost") or 0)
            elif kind == "deliver" and record.get("accepted"):
                goal["comm_robots_informed"] += 1
                current = goal["comm_first_known_step"]
                goal["comm_first_known_step"] = step if current is None else min(current, step)
            elif kind == "refused":
                goal["comm_refusals"] += 1
                if goal["comm_first_refusal_step"] is None:
                    goal["comm_first_refusal_step"] = step
    return goals


# --------------------------------------------------------------------------- classes --------
def classify(goal: dict[str, Any], comm: dict[str, Any] | None,
             information_isolates: bool) -> str:
    """One exclusive class per goal, walked in stage order.

    ``information_isolates`` is true only where the uplink is the sole way a remote
    detection can become belief, which is the only setting in which an
    "information did not arrive" class can be asserted rather than assumed.
    """
    deadline = goal.get("deadline")
    if information_isolates:
        known = (comm or {}).get("comm_first_known_step")
        if known is None:
            return "c2_info_never_arrived"
        if deadline is not None and known > deadline:
            return "c2_info_arrived_after_deadline"
    done = goal.get("completed_at")
    if done is not None:
        return "c4_complete_on_time" if deadline is not None and done <= deadline \
            else "c4_complete_late"
    if goal.get("expired_seen") or (deadline is not None and deadline <= WINDOW):
        if goal.get("first_en_route") is None and goal.get("first_assigned") is None \
                and goal.get("first_in_progress") is None:
            return "c3_known_never_dispatched"
        if goal.get("first_in_progress") is None:
            return "c3_dispatched_never_worked"
        return "c3_worked_but_incomplete"
    return "c5_unresolved_at_window_end"


class Arm:
    def __init__(self) -> None:
        self.episodes = 0
        self.placed = 0
        self.discovered = 0
        self.on_time = 0
        self.metrics_totals = Counter()
        self.classes = Counter()
        self.reached = Counter()
        self.hops: dict[str, list[float]] = {name: [] for name in HOPS}
        self.comm = Counter()
        self.bytes = 0
        self.per_seed_on_time: dict[int, int] = {}
        self.per_seed_bytes: dict[int, int] = {}
        #: For each outcome class, how much message traffic that goal actually got.
        self.class_facts: dict[str, Counter] = {}

    def add(self, row: dict[str, Any], spec: dict[str, Any]) -> None:
        transcript_on = bool(spec["channel"].get("transcript"))
        self.episodes += 1
        self.placed += int(row["placed"] or 0)
        metrics = row["metrics"]
        self.discovered += int(metrics["discovered"])
        self.on_time += int(metrics["on_time"])
        self.per_seed_on_time[int(row["seed"])] = int(metrics["on_time"])
        for field in ("completed", "expired", "active", "unassigned"):
            self.metrics_totals[field] += int(metrics[field])
        for target, goal in row["goals"].items():
            comm = row["comm"].get(target) if transcript_on else None
            information_isolates = transcript_on and not spec["channel"].get(
                "line_of_sight_refresh", False)
            outcome = classify(goal, comm, information_isolates)
            self.classes[outcome] += 1
            facts = self.class_facts.setdefault(outcome, Counter())
            facts["goals"] += 1
            if comm is not None:
                facts["sends"] += comm["comm_sends"]
                facts["refreshes"] += comm["comm_refreshes"]
                facts["informed_rows"] += comm["comm_robots_informed"]
                facts["with_first_report"] += 1 if comm["comm_first_send_step"] is not None else 0
                facts["packets_lost"] += comm["comm_packets_lost"]
                facts["refusals"] += comm["comm_refusals"]
            if goal["completed_at"] is not None and goal["deadline"] is not None:
                facts.setdefault("slacks", []).append(goal["deadline"] - goal["completed_at"])
            if goal["max_progress"] < 1.0:
                facts["goals_with_incomplete_coverage"] += 1
        self.reached["2_discovered"] += int(metrics["discovered"])
        self.reached["8_outcome"] += int(metrics["completed"])
        for target, goal in row["goals"].items():
            for field, key in (("first_en_route", "5_dispatched"),
                               ("first_assigned", "6_on_site"),
                               ("first_in_progress", "7_working")):
                if goal[field] is not None:
                    self.reached[key] += 1
            comm = row["comm"].get(target) if transcript_on else None
            for name, (start_key, end_key) in HOPS.items():
                start = row_value(goal, comm, start_key)
                end = row_value(goal, comm, end_key)
                if start is None or end is None:
                    continue
                self.hops[name].append(end - start)
            if transcript_on:
                if comm is None:
                    self.comm["goals_with_no_message_record"] += 1
                    continue
                self.comm["sends"] += comm["comm_sends"]
                self.comm["first_reports"] += 1 if comm["comm_first_send_step"] is not None else 0
                self.comm["refreshes"] += comm["comm_refreshes"]
                self.comm["packets_arrived"] += comm["comm_packets_arrived"]
                self.comm["packets_lost"] += comm["comm_packets_lost"]
                self.comm["refusals"] += comm["comm_refusals"]
                self.comm["goals_known_by_nobody"] += 1 if comm["comm_first_known_step"] is None \
                    else 0
                self.bytes += comm["comm_bytes"]
                self.per_seed_bytes[int(row["seed"])] = \
                    self.per_seed_bytes.get(int(row["seed"]), 0) + comm["comm_bytes"]

    def render(self) -> dict[str, Any]:
        episodes = max(1, self.episodes)
        facts = {}
        for outcome, counter in sorted(self.class_facts.items()):
            entry = {key: value for key, value in sorted(counter.items())
                     if key != "slacks" and not isinstance(value, list)}
            slacks = counter.get("slacks") or []
            entry["slack_at_completion_steps"] = cc.quantiles([float(s) for s in slacks]) \
                if slacks else None
            facts[outcome] = entry
        return {
            "episodes": self.episodes,
            "goals_placed": self.placed,
            "goals_discovered": self.discovered,
            "not_discovered": self.placed - self.discovered,
            "on_time_total": self.on_time,
            "on_time_per_episode": round(self.on_time / episodes, 4),
            "registry_class_counts": dict(sorted(self.metrics_totals.items())),
            "per_goal_classes": dict(sorted(self.classes.items())),
            "per_goal_class_total": sum(self.classes.values()),
            "per_goal_class_comm_facts": facts,
            "first_transition_observed": dict(sorted(self.reached.items())),
            "hop_latency_steps": {name: cc.quantiles(values)
                                  for name, values in sorted(self.hops.items()) if values},
            "comm": {key: value for key, value in sorted(self.comm.items())},
            "comm_bytes_sum": self.bytes,
            "per_seed_on_time": {str(key): value for key, value in sorted(
                self.per_seed_on_time.items())},
        }


def row_value(goal: dict[str, Any], comm: dict[str, Any] | None, key: str) -> float | None:
    if key.startswith("comm_"):
        return None if comm is None else comm.get(key)
    return goal.get(key)


def scan_group(spec: dict[str, Any], path_spec: dict[str, Any]) -> dict[str, Any]:
    base = REPO / spec["dir"]
    transcript_on = bool(path_spec["channel"].get("transcript"))
    arms: dict[str, Arm] = {}
    missing: list[str] = []
    for arm_name in spec["arms"]:
        arm = arms.setdefault(arm_name, Arm())
        for seed in spec["seeds"]:
            lifecycle = base / spec["lifecycle"].format(arm=arm_name, seed=seed)
            if not lifecycle.exists():
                missing.append(lifecycle.name)
                continue
            row = chain_lifecycle(lifecycle)
            row["seed"] = seed
            row["comm"] = {}
            if transcript_on and spec.get("transcript"):
                tpath = base / spec["transcript"].format(arm=arm_name, seed=seed)
                if tpath.exists():
                    row["comm"] = chain_transcript(tpath)
                else:
                    missing.append(tpath.name)
            arm.add(row, path_spec)
    rendered = {name: arm.render() for name, arm in sorted(arms.items())}
    return {"episodes_found": sum(a["episodes"] for a in rendered.values()),
            "missing_files": len(missing), "missing_examples": missing[:5],
            "arms": rendered, "seeds_per_arm": len(spec["seeds"])}


def paired_contrasts(groups: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-seed paired differences, only where the shared seed really means a shared scenario.

    These are designed contrasts inside one archived matrix, not post-hoc comparisons of
    different runs: every pair below is the same stream seed, the same scenario and the same
    frozen channel, and the pairing grade of each path is stated in the path configuration.
    """
    out: list[dict[str, Any]] = []
    for group, data in sorted(groups.items()):
        names = sorted(data["arms"])
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                left = data["arms"][first]["per_seed_on_time"]
                right = data["arms"][second]["per_seed_on_time"]
                shared = sorted(set(left) & set(right))
                if not shared:
                    continue
                deltas = [right[key] - left[key] for key in shared]
                entry: dict[str, Any] = {
                    "group": group, "pair": [first, second],
                    "streams_paired": len(shared),
                    "on_time_difference_second_minus_first": {
                        "sum": sum(deltas),
                        "per_stream": cc.quantiles([float(delta) for delta in deltas]),
                        "streams_where_second_better": sum(1 for delta in deltas if delta > 0),
                        "streams_equal": sum(1 for delta in deltas if delta == 0),
                        "streams_where_second_worse": sum(1 for delta in deltas if delta < 0),
                    },
                }
                left_bytes = data["arms"][first].get("comm_bytes_sum", 0)
                right_bytes = data["arms"][second].get("comm_bytes_sum", 0)
                if left_bytes or right_bytes:
                    entry["comm_bytes_difference_total"] = right_bytes - left_bytes
                    entry["bytes_per_extra_on_time_goal"] = cc.ratio(
                        abs(right_bytes - left_bytes), abs(sum(deltas)))
                out.append(entry)
    return out


def spot_check_transcripts(spec: dict[str, Any], group_spec: dict[str, Any],
                           files_per_arm: int = 4) -> dict[str, Any]:
    """Re-derive a handful of transcript cells to confirm the aggregate's bookkeeping."""
    base = REPO / group_spec["dir"]
    checked = 0
    agreement = 0
    loss_seen = {"packets_arrived": 0, "packets_lost": 0, "attempts": 0}
    for arm_name in group_spec["arms"]:
        for seed in group_spec["seeds"][:files_per_arm]:
            path = base / group_spec["transcript"].format(arm=arm_name, seed=seed)
            if not path.exists():
                continue
            goals = chain_transcript(path)
            arrived = sum(g["comm_packets_arrived"] for g in goals.values())
            lost = sum(g["comm_packets_lost"] for g in goals.values())
            informed = sum(1 for g in goals.values() if g["comm_first_known_step"] is not None)
            loss_seen["packets_arrived"] += arrived
            loss_seen["packets_lost"] += lost
            loss_seen["attempts"] += arrived + lost
            checked += 1
            agreement += 1 if informed <= len(goals) else 0
    return {"cells_checked": checked, "cells_consistent": agreement, **loss_seen,
            "lost_share_of_attempts": cc.ratio(loss_seen["packets_lost"], loss_seen["attempts"])}


def read_aggregate(path: Path, settings: list[str]) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    keep = ("episodes", "discovered_per_episode", "on_time_per_episode", "expired_per_episode",
            "unresolved_per_episode", "emissions", "first_emissions", "refresh_emissions",
            "transmission_attempts", "packets_delivered", "packets_lost", "uplink_bytes",
            "ack_bytes", "total_bytes", "budget_used_fraction", "budget_refused_sends",
            "detection_to_assignment", "detection_to_work", "detection_to_completion",
            "dispatch_to_work", "belief_age_steps", "completion_rate_all_goals")
    out: dict[str, Any] = {}
    for setting in settings:
        arms = document["settings"][setting]
        out[setting] = {arm: {key: values[key] for key in keep if key in values}
                        for arm, values in sorted(arms.items())}
    out["_source_block"] = document.get("block")
    out["_episodes_expected"] = document.get("episodes_expected")
    out["_incomplete_cells"] = document.get("incomplete_cells")
    return out


def run() -> int:
    cc.install_env_tripwire()
    payload: dict[str, Any] = {"round": cc.ROUND, "stages": STAGES,
                              "window_steps": WINDOW, "paths": {}}
    for name, spec in path_specs().items():
        if spec.get("mode") == "published_aggregate":
            payload["paths"][name] = {
                "configuration": {key: value for key, value in spec.items()
                                  if key != "groups"},
                "published_aggregate": read_aggregate(REPO / spec["aggregate"],
                                                      list(spec["groups"])),
                "transcript_spot_checks": {
                    group: spot_check_transcripts(spec, group_spec)
                    for group, group_spec in sorted(spec["groups"].items())},
            }
            print(f"[{name}] read aggregate, no episode logs parsed", flush=True)
            continue
        groups = {group: scan_group(group_spec, spec)
                  for group, group_spec in sorted(spec["groups"].items())}
        payload["paths"][name] = {
            "configuration": {key: value for key, value in spec.items() if key != "groups"},
            "groups": groups,
            "paired_contrasts": paired_contrasts(groups),
        }
        for group, data in sorted(groups.items()):
            print(f"[{name}/{group}] episodes={data['episodes_found']} "
                  f"missing={data['missing_files']} arms={list(data['arms'])}", flush=True)
    payload["env_tripwire"] = cc.tripwire_report()
    cc.write_json("CHAIN_RECONSTRUCTION.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
