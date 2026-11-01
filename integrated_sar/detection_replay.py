"""Record the PILOC detection stream once, replay it for every allocator arm.

The joint ablation compares coordinator settings, so every arm must face exactly
the same search events. Running the frozen PILOC policy inside each arm *should*
be deterministic, but it is not asserted anywhere, and a single silent truncation
(one arm crashing mid-episode) is enough to make the arms incomparable. This
module makes the shared stream explicit:

* :func:`record_detection_stream` runs the search side alone for ``steps`` ticks
  and serialises every :class:`~integrated_sar.contracts.DetectionEvent` together
  with a hash of the scenario map;
* :class:`ReplaySearchModule` implements the ``SearchModule`` port by releasing
  the recorded events at the steps they were recorded at;
* :func:`assert_bundle_matches_configuration` fails loudly when a replay would
  run on a different map, seed, radius, clock or horizon than the recording.

The bundle is plain JSON so it can live next to the lifecycle logs and be diffed
by hand.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .contracts import (
    DetectionEvent,
    GridPosition,
    SearchStepResult,
    TargetProfile,
    TaskStageSpec,
    WorldPosition,
)
from .errors import ConfigurationError
from .scenarios import SearchScenario

BUNDLE_FORMAT = "integrated_sar.detection_stream/1"


def scenario_map_sha256(scenario: SearchScenario) -> str:
    """Hash the obstacle grid plus the placement the scenario was built with."""
    digest = hashlib.sha256()
    obstacles = scenario.obstacles
    digest.update(f"{obstacles.shape[0]}x{obstacles.shape[1]}:".encode("utf-8"))
    digest.update(obstacles.tobytes())
    for label, cells in (
        ("agents", scenario.agent_positions),
        ("goals", scenario.goal_positions),
    ):
        for cell in cells:
            digest.update(f"|{label}:{cell.row},{cell.col}".encode("utf-8"))
    return digest.hexdigest()


def _profile_to_json(profile: TargetProfile) -> dict[str, Any]:
    return {
        "stages": [{"requirements": list(s.requirements), "duration": s.duration} for s in profile.stages],
        "deadline_offset": profile.deadline_offset,
        "severity": profile.severity,
        "metadata": dict(profile.metadata),
    }


def _profile_from_json(payload: dict[str, Any]) -> TargetProfile:
    return TargetProfile(
        stages=tuple(
            TaskStageSpec(tuple(int(v) for v in s["requirements"]), float(s["duration"]))
            for s in payload["stages"]
        ),
        deadline_offset=float(payload["deadline_offset"]),
        severity=float(payload["severity"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _event_to_json(event: DetectionEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "target_id": event.target_id,
        "grid_position": [event.grid_position.row, event.grid_position.col],
        "world_position": [event.world_position.x, event.world_position.y],
        "detected_at": event.detected_at,
        "discovered_by": event.discovered_by,
        "confidence": event.confidence,
        "map_version": event.map_version,
        "metadata": dict(event.metadata),
        "profile": _profile_to_json(event.profile),
    }


def _event_from_json(payload: dict[str, Any]) -> DetectionEvent:
    row, col = payload["grid_position"]
    x, y = payload["world_position"]
    return DetectionEvent(
        event_id=payload["event_id"],
        target_id=payload["target_id"],
        grid_position=GridPosition(int(row), int(col)),
        world_position=WorldPosition(float(x), float(y)),
        detected_at=float(payload["detected_at"]),
        discovered_by=str(payload["discovered_by"]),
        confidence=float(payload["confidence"]),
        profile=_profile_from_json(payload["profile"]),
        map_version=int(payload["map_version"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _json_scalar(value: Any) -> Any:
    """Reduce numpy scalars and tuples to plain JSON types."""
    if isinstance(value, (tuple, list)):
        return [_json_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, bool) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):  # numpy scalar
        return _json_scalar(item())
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _metrics_to_json(metrics: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_scalar(value) for key, value in metrics.items()}


class DetectionStream:
    """One recorded search episode, indexed by step."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        if bundle.get("format") != BUNDLE_FORMAT:
            raise ConfigurationError(
                f"{bundle.get('path', '?')} is not a detection stream bundle "
                f"(format={bundle.get('format')!r}, expected {BUNDLE_FORMAT!r})"
            )
        self.format = str(bundle["format"])
        self.path = str(bundle.get("path", ""))
        self.seed = int(bundle["seed"])
        self.grid_size = int(bundle["grid_size"])
        self.detection_radius = int(bundle["detection_radius"])
        self.dt = float(bundle["dt"])
        self.steps_recorded = int(bundle["steps_recorded"])
        self.episode_horizon = int(bundle["episode_horizon"])
        self.map_sha256 = str(bundle["map_sha256"])
        self.scenario = dict(bundle.get("scenario", {}))
        self._by_step: dict[int, tuple[DetectionEvent, ...]] = {}
        self._metrics_by_step: dict[int, dict[str, Any]] = {}
        for frame in bundle["frames"]:
            step = int(frame["step"])
            self._by_step[step] = tuple(_event_from_json(e) for e in frame["detections"])
            self._metrics_by_step[step] = dict(frame.get("metrics", {}))

    @property
    def events(self) -> list[DetectionEvent]:
        return [event for step in sorted(self._by_step) for event in self._by_step[step]]

    def detections_at(self, step: int) -> tuple[DetectionEvent, ...]:
        return self._by_step.get(int(step), ())

    def metrics_at(self, step: int) -> dict[str, Any]:
        return dict(self._metrics_by_step.get(int(step), {}))

    def fingerprint(self) -> list[tuple[float, str, int, int]]:
        """(detected_at, target_id, row, col) per event -- the equality assertion."""
        return [
            (e.detected_at, e.target_id, e.grid_position.row, e.grid_position.col)
            for e in self.events
        ]

    def write(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.as_bundle(), handle, ensure_ascii=False, indent=1, sort_keys=True)
            handle.write("\n")
        return target

    def as_bundle(self) -> dict[str, Any]:
        frames = [
            {
                "step": step,
                "detections": [_event_to_json(e) for e in self._by_step[step]],
                "metrics": _metrics_to_json(self._metrics_by_step[step]),
            }
            for step in sorted(self._by_step)
        ]
        return {
            "format": self.format,
            "seed": self.seed,
            "grid_size": self.grid_size,
            "detection_radius": self.detection_radius,
            "dt": self.dt,
            "steps_recorded": self.steps_recorded,
            "episode_horizon": self.episode_horizon,
            "map_sha256": self.map_sha256,
            "scenario": self.scenario,
            "frames": frames,
        }


class RecordingSearchModule:
    """Wrap a live search module and capture the detections it actually emits.

    Recording happens inside the joint run rather than as a separate pass, so the
    stream written at the end is exactly the event sequence the arms will replay --
    there is no second PILOC run that could drift from the first.
    """

    def __init__(self, inner: Any, *, scenario: SearchScenario, configuration: Any) -> None:
        self.inner = inner
        self.scenario = scenario
        self.configuration = configuration
        self._frames: dict[int, tuple[DetectionEvent, ...]] = {}
        self._metrics: dict[int, dict[str, Any]] = {}
        self._step = 0

    def reset(self) -> None:
        reset = getattr(self.inner, "reset", None)
        if callable(reset):
            reset()
        self._step = 0

    def step(self, current_time: float, dt: float) -> SearchStepResult:
        result = self.inner.step(current_time, dt)
        self._frames[self._step] = tuple(result.detections)
        self._metrics[self._step] = dict(result.metrics)
        self._step += 1
        return result

    def __getattr__(self, name: str) -> Any:
        # Everything the driver reads off the search adapter (env, horizon, radius)
        # keeps coming from the wrapped live module.
        return getattr(self.inner, name)

    @property
    def detections_recorded(self) -> int:
        return sum(len(events) for events in self._frames.values())

    def to_stream(self) -> DetectionStream:
        return DetectionStream(
            {
                "format": BUNDLE_FORMAT,
                "path": "",
                "seed": int(self.configuration.seed),
                "grid_size": int(self.configuration.grid_size),
                "detection_radius": int(self.configuration.detection_radius),
                "dt": float(self.configuration.dt),
                "steps_recorded": self._step,
                "episode_horizon": int(getattr(self.inner, "episode_horizon", self._step) or self._step),
                "map_sha256": scenario_map_sha256(self.scenario),
                "scenario": self.scenario.as_dict(),
                "frames": [
                    {
                        "step": step,
                        "detections": [_event_to_json(e) for e in self._frames[step]],
                        "metrics": _metrics_to_json(self._metrics[step]),
                    }
                    for step in sorted(self._frames)
                ],
            }
        )


def load_detection_stream(path: str | os.PathLike[str]) -> DetectionStream:
    resolved = Path(path)
    if not resolved.is_file():
        raise ConfigurationError(f"detection stream bundle not found: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"detection stream bundle {resolved} is not valid JSON: {exc}") from exc
    payload["path"] = str(resolved)
    return DetectionStream(payload)


def record_detection_stream(
    *,
    search_adapter: Any,
    scenario: SearchScenario,
    configuration: Any,
    steps: int,
) -> DetectionStream:
    """Step the search side alone and capture every detection it produces.

    Prefer :class:`RecordingSearchModule` inside a joint run when the recording must
    be the very episode the arms replay; this standalone pass exists so a stream can
    also be produced without building the task side at all.

    ``search_adapter`` is a :class:`~integrated_sar.adapters.piloc.PilocSearchAdapter`
    (or anything exposing ``step(current_time, dt) -> SearchStepResult``).
    """
    search_adapter.reset()
    frames: dict[int, tuple[DetectionEvent, ...]] = {}
    metrics: dict[int, dict[str, Any]] = {}
    now = 0.0
    for index in range(int(steps)):
        result = search_adapter.step(now, configuration.dt)
        frames[index] = tuple(result.detections)
        metrics[index] = dict(result.metrics)
        now += float(configuration.dt)
    bundle = DetectionStream(
        {
            "format": BUNDLE_FORMAT,
            "path": "",
            "seed": int(configuration.seed),
            "grid_size": int(configuration.grid_size),
            "detection_radius": int(configuration.detection_radius),
            "dt": float(configuration.dt),
            "steps_recorded": int(steps),
            "episode_horizon": max(int(steps), 1),
            "map_sha256": scenario_map_sha256(scenario),
            "scenario": scenario.as_dict(),
            "frames": [
                {
                    "step": step,
                    "detections": [_event_to_json(e) for e in frames[step]],
                    "metrics": _metrics_to_json(metrics[step]),
                }
                for step in sorted(frames)
            ],
        }
    )
    return bundle


class _ReplayEnvironmentView:
    """The ``env`` surface :class:`JointRun.summarize` reads off the search adapter."""

    def __init__(self, stream: DetectionStream) -> None:
        self._stream = stream
        self._step = 0
        self.map_size = (stream.grid_size, stream.grid_size)

    def set_step(self, step: int) -> None:
        self._step = int(step)

    @property
    def goals_pos(self) -> list[tuple[int, int]]:
        metrics = self._stream.metrics_at(self._step - 1) if self._step > 0 else {}
        remaining = metrics.get("remaining_targets")
        if remaining is None:
            remaining = self._stream.scenario.get("num_goals_placed", 0)
        return [(0, 0)] * int(remaining)


class ReplaySearchModule:
    """A ``SearchModule`` that replays a recorded detection stream.

    It never touches PILOC, so the arms differ only in coordination and
    allocation. Every released event is the recorded object, so two arms built
    from the same bundle cannot drift apart.
    """

    def __init__(self, stream: DetectionStream) -> None:
        self.stream = stream
        self.env = _ReplayEnvironmentView(stream)
        self.episode_horizon = stream.episode_horizon
        self.detection_radius = stream.detection_radius
        self._step = 0
        self._clock_now = 0.0

    def reset(self) -> None:
        self._step = 0
        self._clock_now = 0.0
        self.env.set_step(0)

    def step(self, current_time: float, dt: float) -> SearchStepResult:
        if self._step >= self.stream.steps_recorded:
            raise ConfigurationError(
                f"the recorded detection stream covers {self.stream.steps_recorded} steps; "
                f"step {self._step + 1} was requested. Re-record with more --steps."
            )
        expected_now = float(self._step) * self.stream.dt
        if abs(float(current_time) - expected_now) > 1e-9:
            raise ConfigurationError(
                f"replay clock drift at step {self._step + 1}: the orchestrator passed "
                f"current_time={current_time!r} but the recording was made at "
                f"{expected_now!r} (dt={self.stream.dt})"
            )
        detections = self.stream.detections_at(self._step)
        metrics = self.stream.metrics_at(self._step)
        self._step += 1
        self._clock_now = float(current_time) + float(dt)
        self.env.set_step(self._step)
        return SearchStepResult(detections=detections, metrics=metrics)

    @property
    def detections_released(self) -> int:
        return sum(len(self.stream.detections_at(step)) for step in range(self._step))


def assert_bundle_matches_configuration(
    stream: DetectionStream, configuration: Any, steps: int
) -> None:
    """Fail before the first tick when the replay would not be the recorded world."""
    mismatches: list[str] = []
    if int(configuration.seed) != stream.seed:
        mismatches.append(f"seed {configuration.seed} != recorded {stream.seed}")
    if int(configuration.grid_size) != stream.grid_size:
        mismatches.append(f"grid_size {configuration.grid_size} != recorded {stream.grid_size}")
    if int(configuration.detection_radius) != stream.detection_radius:
        mismatches.append(
            f"detection_radius {configuration.detection_radius} != recorded {stream.detection_radius}"
        )
    if abs(float(configuration.dt) - stream.dt) > 1e-12:
        mismatches.append(f"dt {configuration.dt} != recorded {stream.dt}")
    if int(steps) > stream.steps_recorded:
        mismatches.append(f"--steps {steps} exceeds the recorded {stream.steps_recorded}")
    if int(configuration.num_goals) != int(stream.scenario.get("num_goals_placed", -1)):
        mismatches.append(
            f"num_goals {configuration.num_goals} != recorded "
            f"{stream.scenario.get('num_goals_placed')}"
        )
    if mismatches:
        raise ConfigurationError(
            f"detection stream {stream.path or '<memory>'} does not match this run: "
            + "; ".join(mismatches)
        )


def assert_streams_equal(reference: DetectionStream, other: DetectionStream) -> None:
    """Compare two bundles event by event (used to prove replay fidelity)."""
    left, right = reference.fingerprint(), other.fingerprint()
    if reference.map_sha256 != other.map_sha256:
        raise AssertionError(
            f"map hash differs: {reference.map_sha256} vs {other.map_sha256}"
        )
    if left != right:
        detail = next(
            (f"index {i}: {a} vs {b}" for i, (a, b) in enumerate(zip(left, right)) if a != b),
            f"length {len(left)} vs {len(right)}",
        )
        raise AssertionError(f"detection streams differ ({detail})")


__all__: Iterable[str] = (
    "BUNDLE_FORMAT",
    "DetectionStream",
    "ReplaySearchModule",
    "assert_bundle_matches_configuration",
    "assert_streams_equal",
    "load_detection_stream",
    "record_detection_stream",
    "scenario_map_sha256",
)
