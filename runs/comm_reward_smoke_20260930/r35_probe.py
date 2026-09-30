"""ROUND35 被动性直接仪表（0 额外启动）：把读数挂在真实 collector 与真实事件总线上。

只在启动②启用。它**只读**：包裹 `PublishedProgressRewardCollector.handle_progress` 与
`InMemoryEventBus.publish`，在调用真实实现前后取快照并计数，不改任何状态、不吞异常。

它自身是否扰动运行，由一个独立的守卫来证明：启动②的 episodes 0..3 必须与未挂仪表的启动①
逐字节相同（同一登记初值、同一动作流），否则仪表本身就是一处干预，必须停止并报告。

每次 handler 调用记录：
  rng_python / rng_numpy / rng_torch_cpu 三个状态哈希在调用前后是否相同；
  clock.now 前后是否相同（collector 不得推进时钟）；
  registry 的 accepted/rejected 计数与目标快照的前后是否相同（collector 不得改真值）；
  publish 计数增量（collector 不得发布事件）；
  task_stream 增量与 credited 结果（这就是它唯一允许的写入）。
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy
import torch

import v2_run
from integrated_sar.contracts import TaskStatus
from integrated_sar.event_bus import InMemoryEventBus


def _digest(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:24]


def _rng_snapshot() -> dict[str, str]:
    state = {"python": _digest(random.getstate()), "numpy": _digest(
        numpy.random.get_state()[1].tobytes())}
    state["torch_cpu"] = _digest(bytes(torch.get_rng_state().tolist()))
    return state


def _registry_view(registry: Any, task_id: str) -> dict[str, Any]:
    """What the registry has committed for this target, read only and field-tolerant.

    The launch-2 failure was this function asking for a field the snapshot class does not have;
    every read now goes through getattr with an explicit "absent" marker so an unexpected schema
    shows up as a missing field in the record instead of killing a real run.
    """
    snapshot = registry.get(task_id)
    absent = object()

    def field(name: str, cast):
        value = getattr(snapshot, name, absent)
        return "absent" if value is absent else cast(value)

    return {"accepted_count": int(getattr(registry, "accepted_event_count", -1)),
            "rejected_count": int(getattr(registry, "rejected_event_count", -1)),
            "status": field("status", str),
            "expired": field("expired", bool),
            "deadline": field("deadline", float),
            "stage_index": field("stage_index", lambda value: value),
            "progress": field("progress", lambda value: value)}


class Probe:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.publishes_during_handler = 0
        self._inside = False
        self._bus_counter = 0
        self.installed = False

    def install(self) -> None:
        if self.installed:
            return
        original_handler = v2_run.PublishedProgressRewardCollector.handle_progress
        original_publish = InMemoryEventBus.publish
        probe = self

        def patched_publish(self_bus: Any, event: Any) -> Any:
            probe._bus_counter += 1
            if probe._inside:
                probe.publishes_during_handler += 1
            return original_publish(self_bus, event)

        def patched_handler(self_collector: Any, event: Any) -> Any:
            rng_before = _rng_snapshot()
            clock_before = float(self_collector.clock.now)
            registry_before = _registry_view(self_collector.registry, event.task_id)
            stream_before = dict(self_collector.task_stream)
            credited_before = set(self_collector.credited_task_ids)
            records_before = len(self_collector.records)
            publishes_before = probe._bus_counter
            probe._inside = True
            try:
                result = original_handler(self_collector, event)
            finally:
                probe._inside = False
            rng_after = _rng_snapshot()
            stream_after = dict(self_collector.task_stream)
            probe.records.append({
                "event_id": str(event.event_id), "task_id": str(event.task_id),
                "status": str(event.status),
                "rng_unchanged": rng_before == rng_after,
                "rng_python_before": rng_before["python"], "rng_python_after": rng_after["python"],
                "rng_numpy_before": rng_before["numpy"], "rng_numpy_after": rng_after["numpy"],
                "rng_torch_before": rng_before["torch_cpu"],
                "rng_torch_after": rng_after["torch_cpu"],
                "clock_before": clock_before, "clock_after": float(self_collector.clock.now),
                "clock_unchanged": clock_before == float(self_collector.clock.now),
                "registry_before": registry_before,
                "registry_after": _registry_view(self_collector.registry, event.task_id),
                "registry_view_unchanged": registry_before == _registry_view(
                    self_collector.registry, event.task_id),
                "publications_caused_by_the_handler": probe._bus_counter - publishes_before,
                "task_stream_delta": {key: stream_after.get(key, 0.0) - stream_before.get(key, 0.0)
                                      for key in set(stream_before) | set(stream_after)},
                "credited_now": len(self_collector.credited_task_ids) - len(credited_before),
                "ledger_rows_added": len(self_collector.records) - records_before,
                "early_return_without_a_row": (event.status is not TaskStatus.COMPLETED
                                               and len(self_collector.records) == records_before),
            })
            return result

        v2_run.PublishedProgressRewardCollector.handle_progress = patched_handler
        InMemoryEventBus.publish = patched_publish
        self.installed = True

    def summary(self) -> dict[str, Any]:
        rows = self.records
        return {"handler_calls": len(rows),
                "calls_caught_by_the_probe": len(rows),
                "completed_events": sum(1 for row in rows if not row["early_return_without_a_row"]),
                "non_completed_events_seen": sum(1 for row in rows
                                                 if row["early_return_without_a_row"]),
                "rng_never_moved": all(row["rng_unchanged"] for row in rows),
                "clock_never_moved": all(row["clock_unchanged"] for row in rows),
                "registry_never_mutated": all(row["registry_view_unchanged"] for row in rows),
                "events_never_published_by_the_handler": all(
                    row["publications_caused_by_the_handler"] == 0 for row in rows),
                "credits_written": sum(sum(row["task_stream_delta"].values()) for row in rows),
                "offenders": [row for row in rows if not (row["rng_unchanged"]
                                                           and row["clock_unchanged"]
                                                           and row["registry_view_unchanged"]
                                                           and row["publications_caused_by_the_handler"]
                                                           == 0)],
                "publishes_during_handler_total": self.publishes_during_handler,
                "records": rows}


def dump(probe: Probe, path: Path, extra: dict[str, Any]) -> Path:
    payload = {"schema": "round35-collector-probe/1", **extra, **probe.summary()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path
