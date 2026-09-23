"""Which of the eight goal-lifecycle stages the existing logs can actually attribute.

Two independent kinds of evidence, both required:

* the *producer* of each stage, resolved by locating a named anchor in the current
  source tree (a stale anchor raises, so the table cannot quietly rot);
* a *probe* over the archived JSONL families, which reports what fields those
  specific files really contain. Code that could log a stage is not the same thing
  as a run that did log it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import closure_common as cc

REPO = cc.REPO_ROOT
RUN_MARKER = "__run__"
# ------------------------------------------------------------------ producer anchors --------
#: stage -> (anchor label, file, literal that must appear in that file)
ANCHORS: dict[str, list[tuple[str, str, str]]] = {
    "placed": [("scenario_goal_count", "integrated_sar/joint_driver.py", "num_goals: int = 6")],
    "search_discovered": [
        ("detection_record", "integrated_sar/lifecycle_log.py", "def handle_detection"),
        ("registry_accepts_detection", "integrated_sar/task_registry.py",
         "def handle_detection(self, event: DetectionEvent)"),
    ],
    "message_delivered": [
        ("send_record", "integrated_sar/comm.py", "def _emit"),
        ("deliver_record", "integrated_sar/comm.py",
         "def _deliver(self, message: DetectionMessage, agent_id: str)"),
        ("charge_is_attempt_based", "integrated_sar/comm.py", "attempt_charge_bytes"),
    ],
    "robot_knew": [
        ("belief_write_gate", "integrated_sar/adapters/task03.py", "def deliver_detection"),
        ("known_tasks_update", "task_allocation/old_task_env.py", 'knowledge["known_tasks"].add'),
    ],
    "allocated": [
        ("assignment_record", "integrated_sar/lifecycle_log.py", "def log_assignment"),
        ("en_route_status", "integrated_sar/adapters/task03.py", "TaskStatus.EN_ROUTE"),
    ],
    "arrived": [
        ("assigned_status", "integrated_sar/adapters/task03.py", "TaskStatus.ASSIGNED"),
        ("on_site_in_trace", "integrated_sar/allocation_trace.py", "group_arrived_at"),
    ],
    "started_work": [
        ("in_progress_status", "integrated_sar/adapters/task03.py", "TaskStatus.IN_PROGRESS"),
        ("stage_advance", "task_allocation/old_task_env.py", "time_left"),
    ],
    "outcome": [
        ("deadline_test", "task_allocation/old_task_env.py", 'if self.current_time > task["deadline"]'),
        ("expiry_is_a_flag_not_a_stop", "task_allocation/old_task_env.py", "hard_expire=False,"),
        ("status_mapping", "integrated_sar/adapters/task03.py", "TaskStatus.EXPIRED if"),
        ("on_time_computed_offline", "integrated_sar/lifecycle_metrics.py", "on_time +="),
        ("final_record", "integrated_sar/lifecycle_log.py", "def log_final"),
    ],
}

#: What each stage is called in the logs, and how direct it is.
STAGE_SPEC: list[dict[str, Any]] = [
    {
        "stage": 1,
        "key": "placed",
        "what": "a goal exists in the scenario, whether or not anything ever finds it",
        "evidence_kind": "derived_from_config",
        "fields": ["run_config detail.configuration.num_goals (per episode)"],
        "why_not_direct": (
            "no record is emitted for a goal that nothing detects, so the denominator of "
            "'never discovered' is only the configured count, and duplicate/merged detections "
            "would make placed != distinct target ids without any log saying so"
        ),
    },
    {
        "stage": 2,
        "key": "search_discovered",
        "what": "a search robot detected the goal and the registry accepted it",
        "evidence_kind": "direct",
        "fields": [
            "detection: timestamp, discovered_by, source_event_id, detail.grid_position",
            "detail.confidence, detail.total_stages, detail.deadline_offset",
        ],
        "why_not_direct": None,
    },
    {
        "stage": 3,
        "key": "message_delivered",
        "what": "a message carrying the goal was transmitted and reached a receiver",
        "evidence_kind": "direct_only_where_transcript_enabled",
        "fields": [
            "send: step, target, first, cost_bytes, attempts (billed), packets (arrived),"
            " packets_lost, subscribers (a COUNT)",
            "deliver: step, target, agent, revision, accepted, reason",
            "refused: step, target, reason, cost_bytes, remaining_bytes",
        ],
        "why_not_direct": (
            "per send the ledger closes as attempts == packets + packets_lost (checked in the"
            " probe), but neither side names *which* subscriber lost the packet, so 'this robot"
            " never received it' cannot be read back; and the deployed-path runs before"
            " comm_rule_confirm carry no transcript at all"
        ),
    },
    {
        "stage": 4,
        "key": "robot_knew",
        "what": "this specific rescue robot's belief now contains the goal",
        "evidence_kind": "inferred",
        "fields": ["deliver accepted=true is the only per-(goal, robot) knowledge record"],
        "why_not_direct": (
            "belief updates are never logged as their own event, and they also happen through "
            "three non-message channels (line-of-sight refresh, one-hop gossip, on-arrival "
            "verify), so 'knew' cannot be separated from 'was told' without the probe counts"
        ),
    },
    {
        "stage": 5,
        "key": "allocated",
        "what": "an assignment command naming this goal was issued and accepted",
        "evidence_kind": "direct_with_caveat",
        "fields": ["assignment: timestamp, assigned_agents, detail.command_id/reason/valid_until",
                   "progress status 'en_route'"],
        "why_not_direct": (
            "assignment records are written with target_id '__run__' whenever the command "
            "carries no target id, so goal linkage is per-run, not guaranteed; the practical "
            "chain key is the first 'en_route' progress record"
        ),
    },
    {
        "stage": 6,
        "key": "arrived",
        "what": "an assigned robot is physically at the goal",
        "evidence_kind": "proxy",
        "fields": ["progress status 'assigned' (members non-empty, work not yet feasible)",
                   "decision trace truth.targets[*].on_site"],
        "why_not_direct": (
            "progress records are emitted only on a fingerprint change, so a robot that arrives "
            "and starts working between two polls produces no 'assigned' row at all; arrival "
            "time is therefore an upper bound, not a measurement"
        ),
    },
    {
        "stage": 7,
        "key": "started_work",
        "what": "the stage requirement is covered and the clock is running",
        "evidence_kind": "proxy",
        "fields": ["progress status 'in_progress', stage_index, progress, remaining_requirements"],
        "why_not_direct": (
            "the same fingerprint rule applies; additionally a coverage break resets time_left "
            "to the full stage duration, so 'working' is not a monotone quantity and elapsed "
            "work cannot be read off the progress fraction"
        ),
    },
    {
        "stage": 8,
        "key": "outcome",
        "what": "completed on time, completed late, expired, or unresolved",
        "evidence_kind": "direct",
        "fields": ["final: status, detail.deadline, detail.discovered_at, detail.updated_at",
                   "progress status 'completed' / 'expired'"],
        "why_not_direct": None,
    },
]

# ------------------------------------------------------------------------ data probes -------
#: family name -> (glob dir, representative file names, what to look for)
FAMILIES: dict[str, dict[str, Any]] = {
    "deployed_live_search": {
        "dir": "runs/coalition_validation_20260918/systemloop",
        "sample": ["baseline_rp_s31000.jsonl", "candidate_rp_s31000.jsonl"],
        "path": "deployed (gossip_interval default 2), live PILOC search, 60 seeds x 2 arms",
    },
    "deployed_replayed_search": {
        "dir": "runs/joint_eval_20260916",
        "sample": ["rp_nocoord_s10000.jsonl", "rp_rs0.25_s10000.jsonl"],
        "path": "deployed, detections replayed from a frozen stream, delivery 1.0 / delay 0",
    },
    "isolated_budgeted_comm": {
        "dir": "runs/comm_task_pressure_learnability_20260922/runs/g12",
        "sample": ["rp_send_once_g12_s1000300.jsonl", "ct_send_once_g12_s1000300.jsonl",
                   "tr_send_once_g12_s1000300.jsonl"],
        "path": "isolated (gossip_interval 100000), 29300 B budget, delivery 1.0 / delay 0",
    },
    "deployed_lossy_comm": {
        "dir": "runs/comm_rule_confirm_20260921/evaluation/normal-lossy",
        "sample": ["ct_normal-lossy_noack_send_once_s900000.jsonl",
                   "rp_normal-lossy_noack_send_once_s900000.jsonl",
                   "ct_normal-lossy_ack_retry_2_s900000.jsonl"],
        "path": "deployed line-of-sight + gossip, delivery probability 0.7, 200 fresh seeds",
    },
}


def family_of(filename: str) -> str:
    for token in ("rp", "ct", "tr"):
        if filename.startswith(f"{token}_") or f"_{token}_" in filename:
            return token
    raise ValueError(f"cannot tell which log family {filename!r} belongs to")


def resolve_anchors() -> dict[str, Any]:
    out: dict[str, Any] = {}
    missing: list[str] = []
    for stage, items in ANCHORS.items():
        located: list[dict[str, Any]] = []
        for label, rel, needle in items:
            lines = (REPO / rel).read_text(encoding="utf-8").splitlines()
            hits = [index + 1 for index, text in enumerate(lines) if needle in text]
            if not hits:
                missing.append(f"{stage}/{label}: {needle!r} not in {rel}")
                continue
            located.append({"label": label, "file": rel, "line": hits[0], "all_hits": hits[:6]})
        out[stage] = located
    if missing:
        raise RuntimeError("stage anchors drifted; refusing to publish a stale table: "
                           + "; ".join(missing))
    return out


def probe_lifecycle(path: Path) -> dict[str, Any]:
    events: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    assignment_linkage: Counter[str] = Counter()
    targets: set[str] = set()
    final_status: Counter[str] = Counter()
    deadline_fields: Counter[str] = Counter()
    for record in cc.iter_jsonl(path):
        kind = str(record.get("event_type"))
        events[kind] += 1
        target = str(record.get("target_id"))
        if kind != "run_config":
            targets.add(target)
        if kind == "progress":
            statuses[str(record.get("status"))] += 1
        if kind == "final" and target != RUN_MARKER:
            final_status[str(record.get("status"))] += 1
            if isinstance(record.get("detail"), dict):
                for field in ("deadline", "discovered_at", "updated_at"):
                    if record["detail"].get(field) is not None:
                        deadline_fields[field] += 1
        if kind == "assignment":
            assignment_linkage["run_level" if target == RUN_MARKER else "goal_level"] += 1
    return {
        "records": sum(events.values()),
        "event_types": dict(events),
        "progress_statuses": dict(statuses),
        "assignment_target_linkage": dict(assignment_linkage),
        "final_statuses": dict(final_status),
        "final_deadline_fields": dict(deadline_fields),
        "distinct_target_ids": len(targets),
    }


def probe_transcript(path: Path) -> dict[str, Any]:
    kinds: Counter[str] = Counter()
    deliver_accepted: Counter[str] = Counter()
    deliver_reasons: Counter[str] = Counter()
    send_first: Counter[str] = Counter()
    lossy = {"sends_with_lost_packets": 0, "attempts_billed": 0, "packets_arrived": 0,
             "packets_lost": 0, "sends_where_attempts_ne_arrived_plus_lost": 0}
    recipient_field_present = 0
    refuse_reasons: Counter[str] = Counter()
    for record in cc.iter_jsonl(path):
        kind = str(record.get("kind"))
        kinds[kind] += 1
        if kind == "deliver":
            deliver_accepted[str(record.get("accepted"))] += 1
            deliver_reasons[str(record.get("reason"))] += 1
        if kind == "send":
            send_first[str(record.get("first"))] += 1
            attempts = int(record.get("attempts") or 0)
            arrived = int(record.get("packets") or 0)
            lost = int(record.get("packets_lost") or 0)
            lossy["sends_with_lost_packets"] += 1 if lost else 0
            lossy["attempts_billed"] += attempts
            lossy["packets_arrived"] += arrived
            lossy["packets_lost"] += lost
            if attempts != arrived + lost:
                lossy["sends_where_attempts_ne_arrived_plus_lost"] += 1
            if "recipients" in record or "subscriber_ids" in record:
                recipient_field_present += 1
        if kind == "refused":
            refuse_reasons[str(record.get("reason"))] += 1
    return {
        "records": sum(kinds.values()),
        "kinds": dict(kinds),
        "deliver_accepted": dict(deliver_accepted),
        "deliver_reasons": dict(deliver_reasons),
        "send_first_flag": dict(send_first),
        "refusal_reasons": dict(refuse_reasons),
        "packet_loss": lossy,
        "per_recipient_loss_field_present": recipient_field_present,
    }


def probe_trace(path: Path) -> dict[str, Any]:
    kinds: Counter[str] = Counter()
    on_site_rows = 0
    decision_rows = 0
    for record in cc.iter_jsonl(path):
        kind = str(record.get("event_type"))
        kinds[kind] += 1
        if kind != "decision":
            continue
        decision_rows += 1
        truth = record.get("truth") or {}
        for facts in (truth.get("targets") or {}).values():
            if facts.get("on_site"):
                on_site_rows += 1
                break
    return {"records": sum(kinds.values()), "kinds": dict(kinds),
            "decision_rows": decision_rows, "rows_with_someone_on_site": on_site_rows}


RUN_MARKER = "__run__"

#: Literals whose *absence* from the integrated_sar package is the evidence that a
#: stage has no log of its own. If any of these ever appears, the corresponding
#: "missing" claim has to be re-examined, so the absence is checked, not assumed.
ABSENT_UNTIL_NOW: dict[str, list[str]] = {
    "no_delivery_or_belief_lifecycle_event": [
        'event_type": "deliver', 'event_type="deliver', 'event_type": "believ',
        'event_type="believ',
    ],
    "no_arrival_or_work_start_lifecycle_event": [
        'event_type": "arriv', 'event_type="arriv', 'event_type="started',
        'event_type": "started',
    ],
    "no_placed_goal_record": ['event_type="goal_placed', 'event_type": "goal_placed'],
}

#: Files that would be scanned for the above.
SCAN_PACKAGE = "integrated_sar"


def check_absent_records() -> dict[str, Any]:
    hits: dict[str, dict[str, list[str]]] = {}
    root = REPO / SCAN_PACKAGE
    sources = sorted(root.rglob("*.py"))
    text = {path.relative_to(REPO).as_posix(): path.read_text(encoding="utf-8") for path in sources}
    for claim, needles in ABSENT_UNTIL_NOW.items():
        found: dict[str, list[str]] = {}
        for name, body in text.items():
            present = [needle for needle in needles if needle in body]
            if present:
                found[name] = present
        hits[claim] = {"scanned_files": len(text), "found": found, "claim_holds": not found}
    return hits


def enumerated_lifecycle_event_types() -> list[str]:
    """The closed set of ``event_type`` values the package can write."""
    literals: set[str] = set()
    for path in sorted((REPO / SCAN_PACKAGE).rglob("*.py")):
        for match in re.finditer(r'"(run_config|detection|progress|assignment|assignment_rejected|final)"',
                                 path.read_text(encoding="utf-8")):
            literals.add(match.group(1))
    return sorted(literals)


def run() -> int:
    cc.install_env_tripwire()
    anchors = resolve_anchors()
    families: dict[str, Any] = {}
    for name, spec in FAMILIES.items():
        base = REPO / spec["dir"]
        entries: dict[str, Any] = {}
        for filename in spec["sample"]:
            path = base / filename
            if not path.exists():
                entries[filename] = {"missing_file": True}
                continue
            if family_of(filename) == "rp":
                entries[filename] = probe_lifecycle(path)
            elif family_of(filename) == "ct":
                entries[filename] = probe_transcript(path)
            else:
                entries[filename] = probe_trace(path)
        families[name] = {"configuration": spec["path"], "files": str(base.relative_to(REPO)),
                          "probes": entries}
    payload = {
        "round": cc.ROUND,
        "stages": STAGE_SPEC,
        "producer_anchors": anchors,
        "probes": families,
        "lifecycle_event_type_set": enumerated_lifecycle_event_types(),
        "absent_record_checks": check_absent_records(),
        "env_tripwire": cc.tripwire_report(),
        "headline": {
            "stages_directly_attributable_in_a_lifecycle_log": [2, 5, 8],
            "stages_only_inferrable": [4, 6, 7],
            "stages_with_no_record_at_all": [1, 3],
            "note": (
                "stage 3 is direct *only* in runs that turned on --comm-transcript, and even "
                "there the losing recipient of a dropped packet is not recorded; stage 1 has no "
                "record anywhere, so 'never discovered' is measured against a configured count"
            ),
        },
    }
    written = cc.write_json("STAGE_CAPABILITY.json", payload)
    print(f"wrote {written.name}")
    print(json.dumps(payload["headline"], ensure_ascii=False, indent=1))
    for name, spec in families.items():
        print(f"-- {name}: {spec['configuration']}")
        for filename, probe in spec["probes"].items():
            print(f"   {filename}: {json.dumps(probe, ensure_ascii=False)[:400]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
