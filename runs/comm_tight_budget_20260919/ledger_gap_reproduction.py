"""§3: reproduce the ledger's remaining gap, and find out which mechanism causes it.

Round 3's attribution reported a negative residual in the generation identity -- fewer generated
items than non-redundant emissions -- and named the cause as poll-side coalescing: a second change
seen by polling the registry while an older one is still queued. This file builds the two candidate
mechanisms separately and measures each one's effect on the identity, because they call for
different remedies and only one of them can produce the sign that was measured:

* **A, poll-side coalescing.** Two registry changes arrive before one emission. The second one is
  counted as neither a generated item nor a coalesce. But it does not create a second emission
  either, so both sides of the identity are missing the same unit and the residual stays at zero.
* **B, a revision that moves without a view moving.** ``generated_items`` counts transitions of
  ``_View`` -- what a robot could act on -- while the transcript's ``redundant`` test compares the
  registry ``revision``. A packet can then be "not redundant" by revision while answering no view
  change at all. That drives the left side of the identity below the right side: the sign round 3
  measured.

"Redundant" is therefore reported under both readings, and the real episodes are run through the
same two readings, so the diagnosis is a measurement rather than an argument.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import tight_common as common

from integrated_sar.comm import CommRelay, CommTranscript
from integrated_sar.comm_budget import CommBudget
from integrated_sar.comm_policy import build_comm_policy
from integrated_sar.contracts import (
    DetectionEvent,
    GridPosition,
    ProgressEvent,
    TargetProfile,
    TaskStageSpec,
    TaskStatus,
    WorldPosition,
)
from integrated_sar.task_registry import TaskRegistry


class RecordingWriter:
    """The receiver side: remember what arrived, apply the real freshness rule."""

    def __init__(self) -> None:
        self.received: list[tuple[str, str, int, float]] = []
        self.believed: dict[tuple[str, str], tuple[float, int]] = {}

    def deliver_detection(self, message: Any, agent_id: str) -> bool:
        key = (str(agent_id), str(message.target_id))
        known = self.believed.get(key)
        if known is not None and (float(message.sent_at), int(message.revision)) <= known:
            return False
        self.believed[key] = (float(message.sent_at), int(message.revision))
        self.received.append((str(agent_id), message.target_id, int(message.revision), float(message.progress)))
        return True

    def belief_freshness(self, agent_id: str, target_id: str):
        return self.believed.get((str(agent_id), target_id))


def detection(target_id: str = "target-1", *, detected_at: float = 1.0) -> DetectionEvent:
    return DetectionEvent(
        event_id=f"detection:{target_id}",
        target_id=target_id,
        grid_position=GridPosition(59, 59),
        world_position=WorldPosition(1.0, 1.0),
        detected_at=detected_at,
        discovered_by="scout_0",
        confidence=1.0,
        profile=TargetProfile(stages=(TaskStageSpec((1, 0, 0, 0, 0), 3.0),), deadline_offset=500.0),
    )


def progress(
    target_id: str,
    *,
    event_id: str,
    observed_at: float,
    stage_index: int = 0,
    remaining: tuple[int, ...] = (1, 0, 0, 0, 0),
    value: float = 0.0,
    members: tuple[str, ...] = (),
    status: TaskStatus = TaskStatus.IN_PROGRESS,
) -> ProgressEvent:
    return ProgressEvent(
        event_id=event_id,
        task_id=target_id,
        observed_at=observed_at,
        status=status,
        stage_index=stage_index,
        remaining_requirements=remaining,
        progress=value,
        members=members,
    )


class Run:
    """A registry, a relay with a transcript, and a clock, wired the way the driver wires them."""

    def __init__(self, *, policy: str = "periodic", params: dict[str, Any] | None = None) -> None:
        self.registry = TaskRegistry()
        self.writer = RecordingWriter()
        self.transcript = CommTranscript()
        self.relay = CommRelay(
            subscribers=("0", "1"),
            writer=self.writer,
            snapshot_of=self.registry.get,
            accepted_of=self.registry.was_accepted,
            seed=7,
            policy=build_comm_policy(policy, params or {}),
            budget=CommBudget(),
            transcript=self.transcript,
        )
        self.tick = 0

    def publish(self, event: DetectionEvent) -> None:
        self.registry.handle_detection(event)
        self.relay.handle_detection(event)

    def report(self, event: ProgressEvent) -> None:
        self.registry.handle_progress(event)

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            self.tick += 1
            self.relay.tick()

    def identity(self) -> dict[str, Any]:
        return identity(self.relay.counters, self.transcript.records)


def identity(counters: dict[str, int], transcript: list[dict[str, Any]]) -> dict[str, Any]:
    """The generation identity under three readings of "redundant".

    ``generated_items`` counts state changes the sender witnessed; ``coalesced_changes`` counts the
    subset of those that joined a change already waiting, so the two are not additive. The three
    readings differ only in what makes a packet a repeat: the registry revision, the content as the
    relay itself judges it, and the content rebuilt from the transcript after the fact.
    """
    sends = [record for record in transcript if record["kind"] == "send"]
    last_revision: dict[str, int] = {}
    last_content: dict[str, tuple] = {}
    redundant_by_revision = redundant_by_content = 0
    for send in sends:
        target = str(send["target"])
        content = (
            int(send["stage_index"]),
            round(float(send["progress"]), 6),
            tuple(int(value) for value in send["remaining"]),
            tuple(str(member) for member in send["members"]),
        )
        redundant_by_revision += int(last_revision.get(target) == int(send["revision"]))
        redundant_by_content += int(target in last_content and last_content[target] == content)
        last_revision[target] = int(send["revision"])
        last_content[target] = content
    emissions = int(counters["emissions"])
    generated = int(counters["generated_items"])
    coalesced = int(counters["coalesced_changes"])
    suppressed = int(counters["emissions_suppressed_terminal"])
    live = counters.get("redundant_emissions")
    return {
        "emissions": emissions,
        "generated_items": generated,
        "coalesced_changes": coalesced,
        "suppressed_terminal": suppressed,
        "redundant_by_revision": redundant_by_revision,
        "redundant_by_content_offline": redundant_by_content,
        "redundant_counted_live": None if live is None else int(live),
        "balance_reading_redundant_by_revision": generated
        - (emissions - redundant_by_revision)
        - coalesced
        - suppressed,
        "balance_reading_redundant_by_content": generated
        - (emissions - redundant_by_content)
        - coalesced
        - suppressed,
        "balance_using_the_live_counter": (
            None if live is None
            else generated - (emissions - int(live)) - coalesced - suppressed
        ),
    }


def case_poll_side_coalescing() -> dict[str, Any]:
    """A: one detection, then two registry changes, all folded into one emission."""
    run = Run(policy="periodic", params={"interval": 20})
    run.publish(detection())
    run.step(2)
    run.report(progress("target-1", event_id="p1", observed_at=3.0, value=0.3, members=("0",)))
    run.step(2)
    run.report(progress("target-1", event_id="p2", observed_at=5.0, value=0.6, members=("0", "1")))
    run.step(16)
    result = run.identity()
    result["case"] = "A poll-side coalescing: 3 changes, 1 packet"
    result["note"] = (
        "post-fix, each merge books one generated item and one coalesce, and the single letter is "
        "one distinct emission, so all three readings of the identity close at zero; pre-fix this "
        "same sequence reported 1 generated / 0 coalesced and also closed at zero, which is why "
        "the mechanism could not be the source of a negative residual"
    )
    return result


def case_revision_without_view() -> dict[str, Any]:
    """B: a registry update that moves the revision but nothing a robot reads."""
    run = Run(policy="periodic", params={"interval": 20})
    run.publish(detection())
    run.step(20)
    before = run.identity()
    run.report(
        progress(
            "target-1", event_id="status-only", observed_at=30.0, value=0.0,
            remaining=(1, 0, 0, 0, 0), members=(),
        )
    )
    run.step(20)
    after = run.identity()
    after["case"] = "B revision without a view change: 2 packets, 1 change"
    after["emissions_before_second_case"] = before["emissions"]
    after["note"] = (
        "this is the reading that produced round 3's negative residual: the registry revision moved "
        "while every field a robot reads stayed put, so the revision reading calls the second packet "
        "new (balance -1) while the content reading and the relay's own redundant_emissions both "
        "call it a repeat (balance 0). generated_items and the redundancy test have to be defined "
        "over the same notion of 'the same thing', or the identity compares two counting objects"
    )
    return after


def case_real_episode(seed: int, budget_bytes: int) -> dict[str, Any]:
    """C: both readings on a real episode, where the mechanism has to show up or not."""
    import pilot_env

    row = pilot_env.run_episode(
        seed,
        directory=common.HERE / "ledger_reproduction",
        label="ledger_reproduction",
        policy="event",
        budget_bytes=budget_bytes,
        information="isolated",
        delivery_probability=1.0,
    )
    records = common.metrics.load_transcript(row["artefacts"]["transcript"])
    return {
        "case": f"C real episode s{seed} policy=event",
        "spent_bytes": int(row["spent_bytes"]),
        **identity(row["ledger"], records),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(common.TRAIN_SEEDS[:4]))
    parser.add_argument("--budget-bytes", type=int, default=common.BUDGETS_BYTES["75pct"])
    args = parser.parse_args(argv)
    report = {
        "candidates": [case_poll_side_coalescing(), case_revision_without_view()],
        "real_episodes": [case_real_episode(seed, args.budget_bytes) for seed in args.seeds],
    }
    for group in ("candidates", "real_episodes"):
        for entry in report[group]:
            print(json.dumps(entry, ensure_ascii=False, sort_keys=True))
    common.write_json(common.json_path("COMM_LEDGER_GAP_REPRODUCTION.json"), report)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
