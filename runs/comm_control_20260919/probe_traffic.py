"""Where does a rescue robot's task knowledge actually come from today?

The measurement this round has to rest on. For each development seed it replays the frozen
C-arm pipeline and attributes *every* write into a robot's task belief to the code path that
made it, so the transport under study (the search-to-rescue uplink) can be compared with the
paths that bypass it (TaskEnv's line-of-sight refresh, one-hop gossip, arrival verification).

It then replays the same seeds with the uplink muted -- the counterfactual that answers
"if sending stops, does the delayed information still arrive?" -- and reports how much
belief still forms and how the task outcomes move.

Instrumentation only: every hook wraps a function, records, and calls through, and the bus
handler is swapped back in ``finally``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import common

from integrated_sar.contracts import DetectionEvent  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402

CALLER_TAGS = {
    "deliver_detection": "uplink",
    "_share_task_belief": "gossip",
    "propagate_task_completion": "los_refresh",
    "verify_task_on_arrival": "arrival",
}
# Resolved outermost-first: refresh_task_belief is shared by the line-of-sight path and by
# arrival verification, so the frame that *called* it is what distinguishes the two channels.
TAG_ORDER = ("gossip", "uplink", "arrival", "los_refresh")


def _attributor(env: Any, records: list[dict], orchestrator: Any) -> Callable[[int, dict], Any]:
    """Wrap ``learn_task_view`` so each belief write is recorded with its real caller."""

    def frame_tag() -> str:
        frame = sys._getframe(1)
        seen: set[str] = set()
        for _ in range(14):
            seen.add(CALLER_TAGS.get(frame.f_code.co_name, ""))
            if frame.f_back is None:
                break
            frame = frame.f_back
        for tag in TAG_ORDER:
            if tag in seen:
                return tag
        return "other"

    original = env.learn_task_view

    def wrapper(agent_id: int, view: dict) -> Any:
        records.append(
            {
                "step": int(orchestrator.step_index),
                "now": float(orchestrator.current_time),
                "agent": int(agent_id),
                "task": int(view["task_id"]),
                "source": frame_tag(),
                "update_time": float(view.get("update_time", 0.0)),
                "revision": int(view.get("revision", 0) or 0),
            }
        )
        return original(agent_id, view)

    return wrapper


def _gossip_wrapper(
    env: Any, records: list[dict], orchestrator: Any, original: Callable[..., Any]
) -> Callable[..., Any]:
    """Wrap ``_share_task_belief``, which writes a neighbour's knowledge without any message.

    Gossip does not pass through ``learn_task_view``: it fills the receiver's own
    dictionaries. Without this hook a task a robot first heard from a neighbour would be
    attributed to whichever later write happened to be visible.
    """
    by_identity = {id(knowledge): agent_id for agent_id, knowledge in env.agent_knowledge.items()}

    def wrapper(sender: dict, tid: int, receiver: dict) -> Any:
        result = original(sender, tid, receiver)
        records.append(
            {
                "step": int(orchestrator.step_index),
                "now": float(orchestrator.current_time),
                "agent": int(by_identity.get(id(receiver), -1)),
                "task": int(tid),
                "source": "gossip",
                "update_time": float(receiver["task_last_update_time"].get(tid, 0.0)),
                "revision": int(receiver.get("task_revision", {}).get(tid, 0) or 0),
            }
        )
        return result

    return wrapper


def probe(seed: int, *, mute_uplink: bool, directory: Path, isolate: bool = False) -> dict[str, Any]:
    configuration = common.base_configuration(seed, directory=directory)
    if isolate:
        # One switch gates both the line-of-sight refresh and the one-hop gossip, because both
        # live inside propagate_task_completion: this is the counterfactual "the uplink is the
        # only message path", leaving arrival verification as the only non-message channel.
        configuration["gossip_interval"] = 100000
    run = build_joint_run(RunConfiguration(**configuration))
    env = run.task_adapter.env
    relay = run.task_adapter.comm_relay
    bus = run.orchestrator.event_bus
    handlers = bus._subscriptions[DetectionEvent]
    relay_index = next(
        index
        for index, handler in enumerate(handlers)
        if getattr(handler, "__self__", None) is relay
    )

    belief_writes: list[dict] = []
    relay_events: list[dict] = []
    original_learn = env.learn_task_view
    original_share = env._share_task_belief
    original_tick = relay.tick
    original_deliver = run.task_adapter.deliver_detection
    original_handler = handlers[relay_index]

    env.learn_task_view = _attributor(env, belief_writes, run.orchestrator)
    env._share_task_belief = _gossip_wrapper(env, belief_writes, run.orchestrator, original_share)

    def counting_deliver(message: Any, agent_id: str) -> bool:
        accepted = original_deliver(message, agent_id)
        relay_events.append(
            {
                "kind": "deliver",
                "step": int(relay._tick),
                "target": str(message.target_id),
                "agent": str(agent_id),
                "revision": int(getattr(message, "revision", 0)),
                "sent_at": float(message.sent_at),
                "accepted": bool(accepted),
            }
        )
        return accepted

    def counting_tick() -> Any:
        queue_age = [relay._tick + 1 - item[0] for item in relay._pending]
        relay_events.append(
            {
                "kind": "tick",
                "step": int(relay._tick) + 1,
                "in_flight": len(relay._pending),
                "max_ready_age": max(queue_age, default=0),
            }
        )
        return original_tick()

    def producing_handler(event: DetectionEvent) -> None:
        before = len(relay._pending)
        original_handler(event)
        relay_events.append(
            {
                "kind": "produce",
                "step": int(relay._tick),
                "target": str(event.target_id),
                "event_id": str(event.event_id),
                "queued": len(relay._pending) - before,
            }
        )

    def muted_handler(event: DetectionEvent) -> None:
        relay_events.append(
            {
                "kind": "muted",
                "step": int(relay._tick),
                "target": str(event.target_id),
                "event_id": str(event.event_id),
                "queued": 0,
            }
        )

    run.task_adapter.deliver_detection = counting_deliver
    relay.tick = counting_tick
    handlers[relay_index] = muted_handler if mute_uplink else producing_handler
    try:
        summary = run.run()
    finally:
        handlers[relay_index] = original_handler
        relay.tick = original_tick
        run.task_adapter.deliver_detection = original_deliver
        env._share_task_belief = original_share
        env.learn_task_view = original_learn

    first_source: dict[tuple[int, int], str] = {}
    for record in sorted(belief_writes, key=lambda item: (item["step"], item["agent"])):
        first_source.setdefault((record["agent"], record["task"]), record["source"])

    kinds = Counter(item["kind"] for item in relay_events)
    status_counts = dict(summary.get("status_counts", {}))
    return {
        "seed": seed,
        "muted": bool(mute_uplink),
        "isolated": bool(isolate),
        "targets_discovered": summary.get("targets_discovered"),
        "detection_events": summary.get("detection_events"),
        "accepted_event_count": summary.get("accepted_event_count"),
        "belief_writes_by_source": dict(Counter(r["source"] for r in belief_writes)),
        "first_belief_source_by_pair": dict(Counter(first_source.values())),
        "distinct_targets_known": len({record["task"] for record in belief_writes}),
        "robots": len(env.agent_dic),
        "uplink": {
            "produced": kinds["produce"] + kinds["muted"],
            "muted": kinds["muted"],
            "deliveries": kinds["deliver"],
            "deliveries_accepted": sum(1 for i in relay_events if i["kind"] == "deliver" and i["accepted"]),
            "delivered_targets": len({i["target"] for i in relay_events if i["kind"] == "deliver"}),
        },
        "status_counts": status_counts,
        "assignment_commands": summary.get("assignment_commands"),
        "rejected_assignments": summary.get("rejected_assignments"),
        "resource_utilization": summary.get("resource_utilization"),
        "steps": summary.get("steps"),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(common.DEV_SEEDS[:4]))
    parser.add_argument("--out", default=str(common.HERE / "TRAFFIC_PROBE.json"))
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="also run each seed with TaskEnv's sensing refresh and gossip switched off, so the "
        "uplink is the only message path: the validity gate for the isolated conditions",
    )
    args = parser.parse_args(argv)

    directory = common.HERE / "probe"
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    shapes = [(False, False)] if not args.isolate else [(False, False), (True, False), (False, True), (True, True)]
    for seed in args.seeds:
        for mute, isolate in shapes:
            row = probe(seed, mute_uplink=mute, directory=directory, isolate=isolate)
            rows.append(row)
            print(
                f"seed={seed} muted={int(mute)} isolated={int(isolate)} "
                f"uplink={row['uplink']} "
                f"writes={row['belief_writes_by_source']} "
                f"first={row['first_belief_source_by_pair']} "
                f"known={row['distinct_targets_known']} "
                f"status={row['status_counts']}",
                flush=True,
            )
    common.write_json(Path(args.out), {"rows": rows})
    totals: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        key = ("isolated" if row["isolated"] else "deployed") + ("_muted" if row["muted"] else "_live")
        totals[key].update(row["first_belief_source_by_pair"])
    print(json.dumps({key: dict(value) for key, value in totals.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
