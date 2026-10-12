"""Private deadline overlay and single-use live factory/run guards for ROUND64."""
from __future__ import annotations

from dataclasses import replace

from r64_guard import R64GuardError


def install_deadline_overlay(provider, attempt, target_id):
    if getattr(provider, "_r64_overlay_installed", False):
        raise R64GuardError("deadline overlay installed more than once")
    original = provider.profile_for

    def profile_for(target, position, detected_at):
        profile = original(target, position, detected_at)
        changed = str(target) == str(target_id)
        actual_offset = float(profile.deadline_offset) + (attempt.delta if changed else 0)
        if actual_offset <= 0:
            raise R64GuardError("deadline offset adjustment is non-positive")
        attempt.profile_records.append({
            "target": str(target), "detected_at": float(detected_at),
            "position": [int(position.row), int(position.col)],
            "original_deadline_offset": float(profile.deadline_offset),
            "delta": attempt.delta if changed else 0,
            "actual_deadline_offset": actual_offset,
            "original_rng_generation_call_completed": True,
        })
        if not changed or attempt.delta == 0:
            return profile
        return replace(profile, deadline_offset=actual_offset)

    provider.profile_for = profile_for
    provider._r64_overlay_installed = True


def _record_message(attempt, target, deadline, source):
    if str(target) == "piloc-e1-target-5":
        attempt.message_records.append({"source": source, "target": str(target),
                                        "deadline": float(deadline)})


def instrument_run(run, attempt, target_id="piloc-e1-target-5", overlay_enabled=True):
    if overlay_enabled:
        install_deadline_overlay(run.search_adapter.profile_provider, attempt, target_id)

    registry = run.registry

    # The orchestrator already subscribed the original bound registry method.
    # Observe after that subscription instead of replacing the instance method:
    # EventBus stores bound method objects, so later reassignment cannot rebind it.
    from integrated_sar.contracts import DetectionEvent

    def observe_detection(event):
        if not isinstance(event, DetectionEvent):
            return
        if event.target_id == target_id:
            snapshot = registry.get(target_id)
            attempt.registry_records.append({
                "event_id": event.event_id,
                "event_deadline": float(event.detected_at + event.profile.deadline_offset),
                "event_offset": float(event.profile.deadline_offset),
                "registry_deadline": None if snapshot is None else float(snapshot.deadline),
                "registry_accepted": bool(registry.was_accepted(event.event_id)),
            })

    run.orchestrator.event_bus.subscribe(DetectionEvent, observe_detection)
    attempt.registry_observer_identity = {
        "mechanism": "append-only passive EventBus subscription",
        "registry_handler_reinvoked": False,
        "observer_runs_after_existing_subscribers": True,
    }

    task_adapter = run.task_adapter
    original_sync = task_adapter.sync_tasks

    def sync_tasks(tasks):
        result = original_sync(tasks)
        for task in tasks:
            if task.task_id != target_id:
                continue
            internal = task_adapter.external_to_internal.get(target_id)
            env_task = task_adapter.env.task_dic.get(internal) if internal is not None else None
            deadline = None if env_task is None else float(env_task.get("deadline"))
            attempt.task_env_records.append({"registry_deadline": float(task.deadline),
                                             "task_env_deadline": deadline})
        return result

    task_adapter.sync_tasks = sync_tasks

    relay = task_adapter.comm_relay
    original_from_event = relay._message_from_event
    original_from_snapshot = relay._message_from_snapshot

    def message_from_event(event):
        message = original_from_event(event)
        _record_message(attempt, event.target_id, message.deadline, "detection_event")
        return message

    def message_from_snapshot(target, event_id, snapshot):
        message = original_from_snapshot(target, event_id, snapshot)
        _record_message(attempt, target, message.deadline, "registry_snapshot")
        return message

    relay._message_from_event = message_from_event
    relay._message_from_snapshot = message_from_snapshot
    attempt.live_run = run


def _counted_builder_delegate(builder):
    code = getattr(builder, "__code__", None)
    cells = getattr(builder, "__closure__", None) or ()
    if code is None or len(code.co_freevars) != len(cells):
        return None
    closure = dict(zip(code.co_freevars, (cell.cell_contents for cell in cells), strict=True))
    return closure.get("original_builder") if code.co_name == "counted_build" else None


def install_counting_hooks(runner, joint_driver, joint_run_class, guard, *,
                           run_proxy_class=None, target_id="piloc-e1-target-5"):
    """Bind the actual runner, builder and run method; reject unregistered/double entry."""
    if getattr(runner, "_r64_counting_hooks", False):
        raise R64GuardError("R64 live counting hooks already installed")

    original_episode = runner._run_episode
    original_builder = runner.build_joint_run
    original_cli_builder = joint_driver.build_joint_run
    original_run = joint_run_class.run
    if original_builder is original_cli_builder:
        builder_binding = "direct-joint-driver-builder"
        if run_proxy_class is not None:
            raise R64GuardError("unexpected counted-run proxy class for a direct builder")
    elif (_counted_builder_delegate(original_builder) is original_cli_builder and
          run_proxy_class is not None):
        builder_binding = "r42-counted-builder-delegates-to-cli-joint-driver"
    else:
        raise R64GuardError("runner builder does not resolve to the CLI JointRun factory")

    def guarded_episode(*args, **kwargs):
        attempt = guard.require_active()
        if getattr(attempt, "episode_started", False):
            raise R64GuardError("one slot cannot invoke two episodes")
        attempt.episode_started = True
        return original_episode(*args, **kwargs)

    def guarded_builder(configuration):
        attempt = guard.require_active()
        if not guard.allow_build:
            raise R64GuardError("preflight-bound runtime refuses all real builders")
        attempt.builder_called()
        run = original_builder(configuration)
        if builder_binding == "r42-counted-builder-delegates-to-cli-joint-driver":
            if not isinstance(run, run_proxy_class):
                raise R64GuardError("counted builder returned an unexpected run proxy type")
            target_run = object.__getattribute__(run, "_target")
        else:
            if not isinstance(run, joint_run_class):
                raise R64GuardError("CLI builder returned an unexpected JointRun type")
            target_run = run
        attempt.env_constructed_now()
        attempt.configuration = configuration.as_dict()
        instrument_run(run, attempt, target_id, overlay_enabled=attempt.arm != "L")
        attempt.live_run = target_run
        return run

    def guarded_joint_run(run, steps=None):
        attempt = guard.require_active()
        if attempt.live_run is not run:
            raise R64GuardError("run object was not built by the active R64 attempt")
        attempt.run_called()
        try:
            result = original_run(run, steps)
        except BaseException as exc:
            actual_steps = int(round(float(run.orchestrator.current_time)))
            attempt.run_failed(actual_steps, type(exc).__name__ + ": " + str(exc))
            raise
        attempt.run_returned(int(result.get("steps", -1)))
        return result

    runner._run_episode = guarded_episode
    runner.build_joint_run = guarded_builder
    joint_driver.build_joint_run = guarded_builder
    joint_run_class.run = guarded_joint_run
    runner._r64_counting_hooks = True
    return {
        "runner_module": runner.__name__,
        "joint_driver_module": joint_driver.__name__,
        "builder_binding": builder_binding,
        "counted_builder_delegate_verified": _counted_builder_delegate(original_builder) is original_cli_builder
        if builder_binding == "r42-counted-builder-delegates-to-cli-joint-driver" else None,
        "run_proxy_class": None if run_proxy_class is None else
            f"{run_proxy_class.__module__}.{run_proxy_class.__qualname__}",
        "runner_builder_wrapper": getattr(original_builder, "__qualname__", type(original_builder).__name__),
        "joint_run_method": f"{joint_run_class.__module__}.{joint_run_class.__qualname__}.run",
        "cli_builder_source": original_cli_builder.__code__.co_filename,
        "builder_hooked_in_runner_and_cli": True,
        "run_method_hooked": True,
    }


def authority_chain(attempt, lifecycle_detection):
    """Return and validate the single-deadline path from profile to environment/send item."""
    focus_profiles = [row for row in attempt.profile_records if row["target"] == "piloc-e1-target-5"]
    if len(focus_profiles) != 1:
        raise R64GuardError(f"expected one target profile generation, got {len(focus_profiles)}")
    profile = focus_profiles[0]
    expected_deadline = profile["detected_at"] + profile["actual_deadline_offset"]
    registry = attempt.registry_records
    task_env = attempt.task_env_records
    messages = attempt.message_records
    if (len(registry) != 1 or not registry[0].get("registry_accepted") or
            registry[0]["registry_deadline"] is None or
            abs(registry[0]["registry_deadline"] - expected_deadline) > 1e-9):
        raise R64GuardError("TaskRegistry deadline differs from the adjusted DetectionEvent")
    if not task_env or any(x["task_env_deadline"] is None or
                           abs(x["task_env_deadline"] - expected_deadline) > 1e-9 for x in task_env):
        raise R64GuardError("Task03 environment deadline differs from TaskRegistry")
    if not messages or any(abs(x["deadline"] - expected_deadline) > 1e-9 for x in messages):
        raise R64GuardError("CommRelay message deadline differs from the canonical deadline")
    detail = lifecycle_detection.get("detail", {})
    logged_deadline = float(lifecycle_detection["timestamp"]) + float(detail["deadline_offset"])
    if abs(logged_deadline - expected_deadline) > 1e-9:
        raise R64GuardError("lifecycle deadline differs from the canonical deadline")
    return {
        "profile": profile,
        "absolute_deadline": expected_deadline,
        "registry": registry,
        "task_env_samples": task_env,
        "messages": messages,
        "lifecycle_deadline": logged_deadline,
        "all_authority_consumers_equal": True,
    }
