"""Zero-environment regression for the production ROUND42 counted-builder seam."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from r42_run import _make_counted_builder, _new_run_progress


class _StepCounter:
    def __init__(self) -> None:
        self.step_index = 0


class _ReadOnlyRun:
    __slots__ = ("orchestrator", "calls", "last_args", "last_kwargs", "answer", "value")

    def __init__(self, *, answer: Any = None) -> None:
        self.orchestrator = _StepCounter()
        self.calls = 0
        self.last_args: tuple[Any, ...] = ()
        self.last_kwargs: dict[str, Any] = {}
        self.answer = answer
        self.value = "before"

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.last_args = args
        self.last_kwargs = dict(kwargs)
        self.orchestrator.step_index += int(kwargs.pop("steps", 0))
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def _harness(builder):
    progress = _new_run_progress()
    claim = Path("synthetic-claim.json")
    current_claim = {"value": claim}
    claim_record: dict[str, Any] = {}
    writes: list[dict[str, Any]] = []
    builds: list[Any] = []

    def counted_builder(configuration):
        run = builder(configuration)
        builds.append(run)
        return run

    def update_claim(path: Path, **updates: Any) -> None:
        assert path == claim
        claim_record.update(updates)

    def write_progress(path: Path, value: dict[str, int]) -> None:
        update_claim(path, **value)
        claim_record["simulation_started"] = int(value.get("completed_steps", 0)) > 0
        writes.append(dict(claim_record))

    def dispatch(call_target, _collector):
        return call_target()

    seam = _make_counted_builder(
        counted_builder,
        active_claim=lambda: current_claim["value"],
        active_progress=lambda: progress,
        update_claim=update_claim,
        collector=lambda: None,
        dispatch_run=dispatch,
        write_progress=write_progress,
    )
    return seam, progress, builds, claim_record, writes


def _assert(condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(detail)


def main() -> int:
    # The exact old implementation must fail on this slots-backed, read-only method.
    raw = _ReadOnlyRun()
    try:
        raw.run = lambda: None  # type: ignore[method-assign]
    except AttributeError:
        old_assignment_rejected = True
    else:
        old_assignment_rejected = False
    _assert(old_assignment_rejected, "regression fixture did not reject old run assignment")

    answer = object()
    made: list[_ReadOnlyRun] = []
    seen_configurations: list[Any] = []

    def build(configuration):
        seen_configurations.append(configuration)
        run = _ReadOnlyRun(answer=answer)
        made.append(run)
        return run

    seam, progress, builds, claim, _writes = _harness(build)
    config = object()
    proxy = seam(config)
    _assert(len(made) == len(builds) == 1, "builder did not construct exactly once")
    _assert(len(seen_configurations) == 1 and seen_configurations[0] is config,
            "builder configuration object changed across the counting seam")
    _assert(made[0].calls == 0, "wrapping invoked run")
    _assert(proxy.orchestrator is made[0].orchestrator, "proxy copied the orchestrator")
    _assert(proxy.value == "before" and made[0].value == "before",
            "proxy construction changed delegated state")
    _assert(progress["environment_constructor_calls"] == 1
            and progress["joint_run_constructed_calls"] == 1
            and progress["run_invocation_calls"] == 0,
            f"construction counters are wrong: {progress}")

    marker = object()
    _assert(proxy.run(marker, steps=7, token="sentinel") is answer,
            "proxy changed the underlying return object")
    _assert(made[0].calls == 1, "underlying run did not execute exactly once")
    _assert(made[0].last_args == (marker,), "run positional arguments changed")
    _assert(made[0].last_kwargs == {"steps": 7, "token": "sentinel"},
            "run keyword arguments changed")
    _assert(progress["run_invocation_calls"] == 1
            and progress["joint_run_method_calls"] == 1
            and progress["run_return_calls"] == 1
            and progress["run_failure_calls"] == 0,
            f"successful run counters are wrong: {progress}")
    _assert(progress["completed_steps"] == 7 and claim["simulation_started"] is True,
            f"completed steps were not counted from the underlying orchestrator: {claim}")
    proxy.value = "delegated-write"
    _assert(made[0].value == "delegated-write" and proxy.value == "delegated-write",
            "proxy attribute write did not delegate to the same object")

    original_error = ValueError("sentinel run failure")
    failing = _ReadOnlyRun(answer=original_error)
    fail_seam, fail_progress, fail_builds, fail_claim, _ = _harness(lambda _cfg: failing)
    fail_proxy = fail_seam(config)
    try:
        fail_proxy.run("arg")
    except ValueError as caught:
        _assert(caught is original_error, "run failure was wrapped or replaced")
    else:
        raise AssertionError("run failure was swallowed")
    _assert(len(fail_builds) == 1 and failing.calls == 1,
            "failure path repeated construction or run")
    _assert(fail_progress["run_invocation_calls"] == 1
            and fail_progress["joint_run_method_calls"] == 1
            and fail_progress["run_failure_calls"] == 1
            and fail_progress["run_return_calls"] == 0,
            f"run failure classification is wrong: {fail_progress}")
    _assert(fail_progress["completed_steps"] == 0
            and fail_claim["simulation_started"] is False,
            "failed zero-step run was reported as a simulation start")

    build_error = RuntimeError("sentinel builder failure")

    def failing_builder(_configuration):
        raise build_error

    build_seam, build_progress, build_list, build_claim, _ = _harness(failing_builder)
    try:
        build_seam(config)
    except RuntimeError as caught:
        _assert(caught is build_error, "builder exception was wrapped or replaced")
    else:
        raise AssertionError("builder failure was swallowed")
    _assert(build_progress["environment_constructor_calls"] == 1
            and build_progress["joint_run_build_failures"] == 1
            and build_progress["joint_run_constructed_calls"] == 0
            and build_progress["run_invocation_calls"] == 0,
            f"builder failure was misclassified: {build_progress}")
    _assert(not build_list and build_claim.get("simulation_started") is False,
            "builder failure touched a nonexistent object or marked simulation started")

    forbidden_activity = {
        "real_joint_run_constructors": 0,
        "sar_environment_constructors": 0,
        "optimizer_step_calls": 0,
        "checkpoint_write_calls": 0,
    }
    _assert(all(value == 0 for value in forbidden_activity.values()),
            f"synthetic seam test crossed a forbidden boundary: {forbidden_activity}")
    print("ROUND42_RUNNER_COUNTING_GUARD " + json.dumps({
        "passed": True,
        "test_count": 8,
        "covered": [
            "old_read_only_run_assignment_rejected",
            "builder_once_and_wrapper_does_not_run",
            "same_collaborators_and_delegated_attributes",
            "one_underlying_run_with_identical_args_and_return_object",
            "run_exception_identity_and_failure_counters",
            "builder_exception_identity_and_no_secondary_object_access",
            "completed_steps_drive_simulation_started",
            "no_duplicate_construction_or_run",
        ],
        **forbidden_activity,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
