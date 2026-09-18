"""The known-but-not-dispatched ledger, computed from probe replays.

For every failure target and its matched controls this walks the probe rows tick
by tick and, for every tick a task's stage gap is open, classifies the first
layer that blocked progress:

  no_belief           no robot both able and holding the task in belief
  capable_all_locked  able, believing robots exist but all are working (nd=False)
  no_plan             a free able robot existed and the planner planned nothing
  no_command          a reservation existed but the allocator did not command
  in_transit          the command was issued; the wait is travel

It also records, per robot, the spans it spent working while its own believed
requirements for the current stage had no overlap with its abilities --
``contribution_zero`` -- which is the fact any wake-up rule must be built on.

    D:/project_env/python.exe runs/dispatch_trigger_audit_20260918/ledger.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent / "coalition_formation_latency_20260918") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "coalition_formation_latency_20260918"))

import stage_timeline as st  # noqa: E402

EPS = 1e-9


def load_probe(seed: int) -> list[dict]:
    return [json.loads(line) for line in (HERE / f"probe_s{seed}.jsonl").read_text(encoding="utf-8").splitlines()]


def internal_of(run, external_id) -> str | None:
    for internal, external in run.items():
        if external == external_id:
            return str(internal)
    return None


def ability_vector(values) -> np.ndarray:
    return np.array(values, dtype=float)


def analyse_target(seed: int, target: str, rows: list[dict], mapping: dict) -> dict:
    """One target's blocking ledger, from detection to final cover or expiry.

    Beliefs and eligibility are *states*, not per-tick events: the coordinator and
    allocator run on a cadence, so every tick carries forward the last known belief
    and eligibility until something replaces them. Classifying only on advise ticks
    would misread "the coordinator was not consulted" as "nobody knew".
    """
    internal = internal_of(mapping, target)
    detected_at = None
    spans: list[dict] = []
    current: dict | None = None
    believed_stage_first: dict[str, float | None] = {}
    beliefs: dict[str, dict] = {}
    agent_status: dict[str, str] = {}
    agent_nd: dict[str, bool] = {}
    last_replan: set[str] = set()
    last_reservations: list[dict] = []

    for row in rows:
        t = row["t"]
        task = row["tasks"].get(internal)
        agents = row["agents"]
        shown = row.get("coordinator", {}).get("shown") or {}
        alloc_view = row.get("allocator", {}).get("agents") or {}

        if detected_at is None and task is not None:
            detected_at = t

        for aid, view in shown.items():
            belief = (view.get("beliefs") or {}).get(target)
            if belief is not None:
                beliefs[aid] = belief
                if task is not None and belief["stage"] == task["stage"] and aid not in believed_stage_first:
                    believed_stage_first[aid] = t
        for aid, view in alloc_view.items():
            if view.get("status"):
                agent_status[aid] = view["status"]
            agent_nd[aid] = bool(view.get("nd"))
        if row.get("allocator") is not None:
            last_replan = set(row["allocator"].get("replan") or [])
        if row.get("coordinator", {}).get("revision") is not None:
            last_reservations = [
                r for r in (row["coordinator"].get("reservations") or []) if r["task"] == target
            ]

        if task is None or task["finished"]:
            if current is not None:
                spans.append(current)
                current = None
            continue

        gap = np.array(task["gap"], dtype=float)
        key = (task["stage"], tuple(task["members"]))
        if current is None or current["stage"] != key[0] or tuple(current["members"]) != key[1]:
            if current is not None:
                spans.append(current)
            current = {
                "stage": task["stage"],
                "members": list(task["members"]),
                "requirement": task["req"],
                "from": t,
                "to": t,
                "first_cover_at": None,
                "block_ticks": {},
                "who_could": {},
            }
        current["to"] = t
        if float(gap.sum()) <= EPS and current["first_cover_at"] is None:
            current["first_cover_at"] = t

        if float(gap.sum()) > EPS:
            believing, free, eligible = [], [], []
            for aid, agent in agents.items():
                abilities = ability_vector(agent["ab"])
                if float(np.minimum(gap, abilities).sum()) <= EPS:
                    continue
                belief = beliefs.get(aid)
                if belief is not None and not belief.get("fin", False):
                    believing.append(aid)
                    if agent_status.get(aid) == "idle":
                        free.append(aid)
                    if agent_nd.get(aid) or aid in last_replan:
                        eligible.append(aid)
            commands = [
                c for c in (row.get("allocator", {}).get("commands") or []) if c["task"] == target
            ]
            in_transit = [
                aid for aid, agent in agents.items()
                if agent["status"] == "moving" and agent["task"] == int(internal or -1)
            ]
            reserved = [r for r in last_reservations if float(r["until"]) >= t]
            if task.get("expired"):
                # Past the deadline no dispatch can change the outcome; counting
                # these ticks as "blocked" would flatter the fix.
                layer = "post_deadline"
            elif not believing:
                layer = "no_belief"
            elif in_transit:
                layer = "in_transit"
            elif not free and not eligible:
                layer = "capable_all_locked"
            elif not reserved and not commands:
                layer = "no_plan"
            else:
                layer = "no_command"
            current["block_ticks"][layer] = current["block_ticks"].get(layer, 0) + 1
            merged = set(current["who_could"].get(layer, [])) | set(believing)
            current["who_could"][layer] = sorted(merged)

    if current is not None:
        spans.append(current)

    return {
        "seed": seed,
        "target": target,
        "detected_at": detected_at,
        "believed_stage_first": believed_stage_first,
        "stage_spans": spans,
    }


def agent_locked_spans(seed: int, rows: list[dict]) -> list[dict]:
    """Per robot: spans working while adding nothing the others don't already add.

    A member's own ability is already inside the env's gap, so redundancy has to be
    judged against the requirement minus the *other* members' abilities: removing
    this robot must not open any believed gap for it to be dead weight. That is the
    condition a belief-only wake-up rule can test (members from the belief, abilities
    from the snapshots).
    """
    out: list[dict] = []
    open_span: dict[str, dict] = {}
    for row in rows:
        tasks = row["tasks"]
        for aid, agent in row["agents"].items():
            span = open_span.get(aid)
            if agent["status"] == "working" and agent["task"] is not None:
                task = tasks.get(str(agent["task"]))
                if task:
                    requirement = np.array(task["req"], dtype=float)
                    others = [
                        np.array(row["agents"][str(m)]["ab"], dtype=float)
                        for m in task["members"]
                        if str(m) != aid and str(m) in row["agents"]
                    ]
                    covered_without = (
                        np.sum(others, axis=0) if others else np.zeros(len(requirement))
                    )
                    gap_without = np.maximum(requirement - covered_without, 0.0)
                    abilities = ability_vector(agent["ab"])
                    zero = float(np.minimum(gap_without, abilities).sum()) <= EPS
                else:
                    zero = False
                if span is None or span["task"] != agent["task"]:
                    if span is not None:
                        out.append(span)
                    span = {
                        "agent": aid,
                        "task": agent["task"],
                        "from": row["t"],
                        "to": row["t"],
                        "contribution_zero_ticks": 0,
                        "ticks": 0,
                    }
                    open_span[aid] = span
                span["to"] = row["t"]
                span["ticks"] += 1
                if zero:
                    span["contribution_zero_ticks"] += 1
            else:
                if span is not None:
                    out.append(span)
                    open_span.pop(aid, None)
    out.extend(open_span.values())
    return [s for s in out if s["ticks"] >= 5]


def internal_mapping(seed: int, rows: list[dict], timelines: dict) -> dict[str, str]:
    """internal id -> external target id, matched by the task's unique deadline."""
    by_deadline = {
        round(float(entry.deadline), 3): entry.target
        for entry in timelines.values()
        if entry.seed == seed
    }
    mapping: dict[str, str] = {}
    for row in rows:
        for internal, task in row["tasks"].items():
            if internal in mapping:
                continue
            external = by_deadline.get(round(float(task["deadline"]), 3))
            if external is not None:
                mapping[internal] = external
    return mapping


def main() -> int:
    import case_reports as cr  # noqa: E402  (reuse the matched controls)

    timelines = st.build_all(st.FAILURE_SEEDS)
    payloads = {seed: load_probe(seed) for seed in st.FAILURE_SEEDS}
    mappings = {
        seed: internal_mapping(seed, payloads[seed], timelines) for seed in st.FAILURE_SEEDS
    }
    for seed, mapping in mappings.items():
        print(f"seed {seed}: mapped {len(mapping)} internal tasks")

    failures_by_key = {f"{t.seed}/{t.target}": t for t in timelines.values()}
    failure_keys = [f"{s}/{t}" for s, t in st.FAILURES]
    pool = [t for key, t in timelines.items() if key not in failure_keys]
    match_keys = ("delivery_budget", "stage_count", "mean_duration", "requirement_count",
                  "detected_at", "active_tasks", "capable_total")

    ledger = {"failures": {}, "controls": {}, "locked_spans": {}}
    for key in failure_keys:
        failure = failures_by_key[key]
        ledger["failures"][key] = analyse_target(
            failure.seed, failure.target, payloads[failure.seed], mappings[failure.seed]
        )
        for control in cr.matched_controls(failure, pool, match_keys, k=5):
            control_seed, control_target = control["target"].split("/")
            ckey = control["target"]
            if ckey in ledger["controls"]:
                continue
            ledger["controls"][ckey] = analyse_target(
                int(control_seed), control_target,
                payloads[int(control_seed)], mappings[int(control_seed)],
            )

    for seed in st.FAILURE_SEEDS:
        ledger["locked_spans"][seed] = agent_locked_spans(seed, payloads[seed])

    (HERE / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"failures: {len(ledger['failures'])}, controls: {len(ledger['controls'])} -> ledger.json")
    for key, entry in ledger["failures"].items():
        print(f"\n== {key} ==")
        for span in entry["stage_spans"]:
            blocks = ", ".join(f"{k}:{v}" for k, v in sorted(span["block_ticks"].items()))
            print(
                f"  stage {span['stage']} [{span['from']:.0f}->{span['to']:.0f}] "
                f"members={span['members']} first_cover={span['first_cover_at']} blocks: {blocks or '-'}"
            )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
