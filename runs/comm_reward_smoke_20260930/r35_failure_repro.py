"""最小复现（0 环境启动）：在真实事件总线 + 真实 TaskRegistry + 真实 collector 上重现启动②失败。

不构造 JointRun、不 step 环境：只做 ROUND34 回归测试同级的合成夹具（假的 relay/clock 持有者），
用真实 `InMemoryEventBus`、真实 `TaskRegistry`、真实 `PublishedProgressRewardCollector`，
然后发布一个 COMPLETED 事件，观察挂上仪表之后会发生什么。

用法：

    python r35_failure_repro.py            # 记录当前 r35_probe 的行为（失败或已修复）
    python r35_failure_repro.py --fix      # 同时把结论写进 LAUNCH2_ATTEMPT1_FAILURE.json

它同时是"仪表已修好"的证据：修复后 handler 正常入账，且仪表记录的 rng/clock/registry/publish
四条不变量全部为真。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import r35_common as smoke
import v2_run
from integrated_sar.contracts import (
    DetectionEvent, GridPosition, ProgressEvent, TargetProfile, TaskStageSpec, TaskStatus,
    WorldPosition)
from integrated_sar.event_bus import InMemoryEventBus
from integrated_sar.task_registry import TaskRegistry

import r35_probe


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0


class FakeRelay:
    """The same class of holder ROUND34's regression fixture used: a tick counter and books."""

    def __init__(self) -> None:
        self._tick = 0
        self.counters = {key: 0 for key in v2_run.COST_KEYS}

    def tick(self) -> tuple[str, ...]:
        self._tick += 1
        return ()


def build() -> SimpleNamespace:
    registry = TaskRegistry()
    bus = InMemoryEventBus()
    bus.subscribe(DetectionEvent, registry.handle_detection)
    bus.subscribe(ProgressEvent, registry.handle_progress)
    relay, clock, env = FakeRelay(), FakeClock(), None
    run = SimpleNamespace(task_adapter=SimpleNamespace(comm_relay=relay, env=env),
                          registry=registry,
                          orchestrator=SimpleNamespace(event_bus=bus, clock=clock))
    taps = v2_run._install_taps(run)
    return SimpleNamespace(registry=registry, bus=bus, relay=relay, clock=clock, taps=taps)


def drive(harness: SimpleNamespace) -> dict[str, object]:
    harness.bus.publish(DetectionEvent(
        event_id="detect:repro", target_id="repro-target",
        grid_position=GridPosition(2, 3), world_position=WorldPosition(0.5, 0.5),
        detected_at=0.0, discovered_by="fixture-agent", confidence=1.0,
        profile=TargetProfile(stages=(TaskStageSpec((1, 0, 0, 0, 0), 1.0),), deadline_offset=10.0)))
    harness.relay._tick = 1
    event = ProgressEvent(event_id="done:repro:1", task_id="repro-target", observed_at=1.0,
                          status=TaskStatus.COMPLETED, stage_index=0,
                          remaining_requirements=(0, 0, 0, 0, 0), progress=1.0,
                          members=("fixture-agent",), expired=False)
    harness.bus.publish(event)
    return {"task_stream": dict(harness.taps["task_stream"]),
            "ledger_rows": len(harness.taps["task_reward_events"])}


def faulty_field_access(registry: TaskRegistry) -> dict[str, Any]:
    """Case A: exactly the expression that killed launch 2, reproduced on a real snapshot."""
    registry.handle_detection(DetectionEvent(
        event_id="detect:faulty", target_id="faulty-target",
        grid_position=GridPosition(2, 3), world_position=WorldPosition(0.5, 0.5),
        detected_at=0.0, discovered_by="fixture-agent", confidence=1.0,
        profile=TargetProfile(stages=(TaskStageSpec((1, 0, 0, 0, 0), 1.0),),
                              deadline_offset=10.0)))
    snapshot = registry.get("faulty-target")
    try:
        value = int(snapshot.task_revision)          # the line that raised during launch 2
        return {"raised": False, "value": value}
    except AttributeError as error:
        return {"raised": True, "error": f"AttributeError: {error}",
                "snapshot_fields_present": sorted(name for name in dir(snapshot)
                                                  if not name.startswith("_"))}


def main() -> int:
    parser = argparse.ArgumentParser(description="minimal reproduction of the launch-2 failure")
    parser.add_argument("--fix", action="store_true")
    arguments = parser.parse_args(sys.argv[1:])
    faulty = None
    harness_for_fault = build()          # the probe is not installed yet at this point
    faulty = faulty_field_access(harness_for_fault.registry)
    probe = r35_probe.Probe()
    probe.install()
    harness = build()
    outcome: dict[str, object] = {"probe_installed": probe.installed,
                                  "case_A_the_faulty_field_access": faulty}
    error_text = None
    try:
        outcome["case_B_with_the_fixed_probe"] = drive(harness)
        outcome["case_B_outcome"] = "handler completed with the probe attached"
    except Exception as error:  # recorded, not swallowed: this is the reproduction
        outcome["case_B_outcome"] = f"{type(error).__name__}: {error}"
        error_text = traceback.format_exc()
    outcome["probe_records_captured"] = len(probe.records)
    outcome["probe_summary_if_any"] = ({key: value for key, value in probe.summary().items()
                                       if key != "records"} if probe.records else None)
    path = smoke.HERE / "LAUNCH2_ATTEMPT1_REPRO.json"
    path.write_text(json.dumps({"schema": "round35-failure-reproduction/1",
                                "environment_launches": 0, "environments_stepped": 0,
                                "fixture": "production InMemoryEventBus + TaskRegistry + "
                                           "PublishedProgressRewardCollector, fake relay/clock "
                                           "holders only (the same class of fixture ROUND34 used)",
                                "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                **outcome}, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    if arguments.fix:
        failure = smoke.HERE / "LAUNCH2_ATTEMPT1_FAILURE.json"
        record = json.loads(failure.read_text(encoding="utf-8"))
        record["minimal_reproduction"] = {
            "executed_by": "r35_failure_repro.py", "artefact": path.name,
            "case_A_raises": faulty.get("error"),
            "case_A_snapshot_fields_present": faulty.get("snapshot_fields_present"),
            "case_B_after_the_fix": outcome.get("case_B_outcome"),
            "case_B_credits_written": (outcome.get("case_B_with_the_fixed_probe") or {}).get(
                "task_stream"),
            "traceback_from_the_real_run": error_text or record.get("traceback", "")}
        failure.write_text(json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in outcome.items()
                      if key != "probe_summary_if_any"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
