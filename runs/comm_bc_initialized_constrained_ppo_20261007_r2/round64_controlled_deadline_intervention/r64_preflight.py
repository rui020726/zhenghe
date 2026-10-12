"""Zero-environment evidence for the frozen-input R64 live runtime and feature path."""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from r64_execute import (EXPECTED_FILE_SHA, EXPECTED_PARAM_SHA, R60_DIR, R64_EXECUTION_VERSION,
                         SCENE, TARGET, read_json, read_jsonl, setup_runtime, sha256)

HERE = Path(__file__).resolve().parent
R60_TAPE = R60_DIR / "raw" / "logs" / "v2_r60_eval_new_s33_greedy_s1700027_a1_tape.jsonl"
R60_LIFE = R60_DIR / "raw" / "logs" / "v2_r60_eval_new_s33_greedy_s1700027_a1_rp.jsonl"
R60_ROW = R60_DIR / "raw" / "rows" / "v2_r60_eval_new_s33_greedy_s1700027_a1.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"non-serializable frozen configuration value: {type(value).__name__}")


def saved_focus_features_by_deadline_delta():
    from integrated_sar.comm_learning import features
    from integrated_sar.comm_policy import SendItem, SendObservation

    tape = next(row for row in read_jsonl(R60_TAPE)
                if row["step"] == 134 and TARGET in row.get("targets", []))
    i = tape["targets"].index(TARGET)
    base = [float(x) for x in tape["features"][i]]
    life = read_jsonl(R60_LIFE)
    event = next(row for row in life if row.get("event_type") == "detection" and
                 row.get("target_id") == TARGET)
    detected_at = float(event["timestamp"])
    original_offset = float(event["detail"]["deadline_offset"])
    deadline = detected_at + original_offset
    step = int(tape["step"])
    now = float(step - 1)
    state = tape["sender_state"]
    quote = int(tape["attempt_charge_bytes"][TARGET])
    item = SendItem(
        target_id=TARGET, ready_step=step, age_steps=int(base[0]),
        pending_event=bool(base[3]), changed_view=bool(base[4]),
        first_emission=bool(base[5]), revision=1, revision_delta=int(base[10]),
        stage_index=0, stage_changed=bool(base[6]), progress=0.0,
        progress_delta=float(base[9]), requirements_changed=bool(base[7]),
        members_changed=bool(base[8]), deadline=deadline, remaining_time=deadline - now,
        sent_count=int(base[2]), last_sent_step=None,
        since_last_emission_time=float(base[1]), has_future_requirements=bool(base[13]),
        attempt_charge_bytes=quote, send_executable=True, send_block_reason="allowed",
    )
    observation = SendObservation(
        step=step, current_time=now, items=(item,),
        remaining_budget_bytes=float(state["remaining_budget_bytes"]),
        budget_bytes_per_episode=int(state["budget_bytes_per_episode"]),
        capacity_remaining=state["capacity_remaining"],
        backlog_targets=int(state["backlog_targets"]), subscribers=int(state["subscribers"]),
    )
    recomputed_base = [float(x) for x in features(observation, item)]
    if recomputed_base != base:
        raise RuntimeError("reconstructed saved focus SendObservation does not reproduce stored 19D input")
    variants = []
    for delta in (-30, 0, 30, 60):
        shifted_deadline = deadline + delta
        shifted = replace(item, deadline=shifted_deadline,
                          remaining_time=shifted_deadline - now)
        vector = [float(x) for x in features(observation, shifted)]
        variants.append({"delta": delta, "deadline": shifted_deadline,
                         "remaining_time_at_saved_clock": shifted.remaining_time,
                         "features_19d": vector,
                         "changed_indices_from_saved": [j for j, (a, b) in enumerate(zip(base, vector))
                                                        if a != b],
                         "quote_bytes_fixture": quote,
                         "quote_note": "held at saved value only for pure feature recomputation; live R64 reprices honestly"})
    return {
        "source_tape": str(R60_TAPE), "source_tape_sha256": sha(R60_TAPE),
        "source_lifecycle": str(R60_LIFE), "source_lifecycle_sha256": sha(R60_LIFE),
        "source_row_sha256": sha(R60_ROW), "target": TARGET, "decision_step": step,
        "clock_now": now, "original_deadline_offset": original_offset,
        "original_deadline": deadline, "saved_features_19d": base,
        "recomputed_saved_features_19d": recomputed_base,
        "base_vector_exact_match": True, "variants": variants,
        "interpretation": "A typed SendObservation was reconstructed from saved actual R60 tape/lifecycle fields; no environment was constructed."
    }


def verify_live_event_bus_receipt_observer():
    from integrated_sar.contracts import (
        DetectionEvent, GridPosition, TargetProfile, TaskStageSpec, WorldPosition,
    )
    from integrated_sar.event_bus import InMemoryEventBus
    from integrated_sar.orchestrator import JointSAROrchestrator
    from integrated_sar.task_registry import TaskRegistry
    from r64_runtime import instrument_run

    class CountingRegistry(TaskRegistry):
        def __init__(self):
            super().__init__()
            self.detection_calls = 0

        def handle_detection(self, event):
            self.detection_calls += 1
            super().handle_detection(event)

    registry = CountingRegistry()
    bus = InMemoryEventBus()
    subscribed_handler = registry.handle_detection
    bus.subscribe(DetectionEvent, subscribed_handler)
    run = SimpleNamespace(
        search_adapter=SimpleNamespace(profile_provider=SimpleNamespace()),
        registry=registry,
        orchestrator=SimpleNamespace(event_bus=bus),
        task_adapter=SimpleNamespace(
            sync_tasks=lambda tasks: None, external_to_internal={},
            env=SimpleNamespace(task_dic={}),
            comm_relay=SimpleNamespace(
                _message_from_event=lambda event: event,
                _message_from_snapshot=lambda target, event_id, snapshot: snapshot,
            ),
        ),
    )
    attempt = SimpleNamespace(
        delta=0, arm="L0", registry_records=[], profile_records=[],
        task_env_records=[], message_records=[],
    )
    instrument_run(run, attempt, overlay_enabled=False)
    event = DetectionEvent(
        event_id="r64-live-bus-spy", target_id=TARGET,
        grid_position=GridPosition(20, 30),
        world_position=WorldPosition(30 / 59, 20 / 59),
        detected_at=134.0, discovered_by="agent_0", confidence=1.0,
        profile=TargetProfile(
            stages=(TaskStageSpec((1, 0, 0, 0, 0), 3.0),), deadline_offset=100.0,
        ),
    )
    bus.publish(event)
    method_source = inspect.getsource(JointSAROrchestrator.__init__)
    expected_subscription = "self.event_bus.subscribe(DetectionEvent, self.registry.handle_detection)"
    if expected_subscription not in method_source:
        raise RuntimeError("live orchestrator no longer subscribes the registry's bound handler")
    if registry.handle_detection.__func__ is not subscribed_handler.__func__:
        raise RuntimeError("observer modified the live registry handler binding")
    if registry.detection_calls != 1 or registry.accepted_event_count != 1:
        raise RuntimeError("live bus did not invoke exactly one accepted registry transition")
    if len(attempt.registry_records) != 1 or not attempt.registry_records[0]["registry_accepted"]:
        raise RuntimeError("R64 passive observer missed the live accepted registry receipt")
    return {
        "status": "passed", "environment_constructed": False,
        "registry_handler_calls": registry.detection_calls,
        "accepted_events": registry.accepted_event_count,
        "observer_receipts": len(attempt.registry_records),
        "handler_object_unchanged": True,
        "active_module_paths": {
            "contracts": str(Path(inspect.getfile(DetectionEvent)).resolve()),
            "event_bus": str(Path(inspect.getfile(InMemoryEventBus)).resolve()),
            "orchestrator": str(Path(inspect.getfile(JointSAROrchestrator)).resolve()),
            "task_registry": str(Path(inspect.getfile(TaskRegistry)).resolve()),
        },
        "orchestrator_bound_subscription_source_verified": True,
    }


def run_zero_env_preflight(output_root):
    cfg = setup_runtime(Path(output_root) / "raw", preflight=True)
    runner = cfg["runner"]
    guard = cfg["r64_guard"]
    try:
        runner._run_episode("unregistered-preflight", "eval", SCENE, policy="learned")
    except Exception as exc:
        from r64_guard import R64GuardError
        if not isinstance(exc, R64GuardError):
            raise
        unregistered_refused = True
    else:
        raise RuntimeError("unregistered actual runner path was not refused")
    fields, substituted = cfg["common"].stage_b_inherited_config(SCENE)
    from integrated_sar.joint_driver import RunConfiguration

    fields["comm_policy"] = "learned"
    paths = cfg["common"].cell_paths("eval", SCENE, 1, tag="r64_preflight")
    fields["output"] = str(paths["lifecycle"])
    fields["comm_transcript"] = str(paths["transcript"])
    live_configuration = RunConfiguration(**fields)
    live_configuration.validate()
    live_configuration = _json_safe(live_configuration.as_dict())
    baseline_lifecycle = read_jsonl(R60_LIFE)
    baseline_run_config = next(row["detail"]["configuration"] for row in baseline_lifecycle
                               if row.get("event_type") == "run_config")
    normalized_live = dict(live_configuration)
    normalized_baseline = dict(baseline_run_config)
    path_differences = {
        "output": [normalized_baseline.pop("output"), normalized_live.pop("output")],
        "comm_transcript": [normalized_baseline.pop("comm_transcript"),
                            normalized_live.pop("comm_transcript")],
    }
    if normalized_live != normalized_baseline:
        raise RuntimeError("R64 live RunConfiguration differs from the frozen R60 baseline")
    expected_fields = {
        "steps": 1000, "num_goals": 6, "comm_budget_bytes": 29300,
        "comm_ack": False, "seed": SCENE,
    }
    live_fields = {key: fields.get(key) for key in expected_fields}
    if live_fields != expected_fields:
        raise RuntimeError(f"frozen live config mismatch: {live_fields}")
    model_path = cfg["model_file_path"]
    if sha(model_path) != EXPECTED_FILE_SHA:
        raise RuntimeError("read-only model file identity changed in preflight")
    vectors = saved_focus_features_by_deadline_delta()
    return {
        "kind": "round64-zero-environment-preflight/1",
        "execution_version": R64_EXECUTION_VERSION,
        "environment_builds": 0, "environment_runs": 0, "environment_steps": 0,
        "claims": 0, "slots": guard.snapshot()["slots_consumed"],
        "optimizer_updates": 0, "ppo_replays": 0, "rng_restores": 0,
        "checkpoint_writes": 0, "model_forwards": 0,
        "read_only_model_loads": 1,
        "model": cfg["model_identity"], "model_file_sha_after": sha(model_path),
        "live_binding": cfg["identity"], "count_hook_binding": cfg["r64_hook_identity"],
        "live_event_bus_receipt_spy": verify_live_event_bus_receipt_observer(),
        "unregistered_runner_refused_before_builder": unregistered_refused,
        "actual_live_config_subset": live_fields,
        "actual_live_configuration": live_configuration,
        "configuration_comparison": {
            "matches_R60_after_only_output_path_rebinding": True,
            "explicit_path_differences": path_differences,
        },
        "substituted_fields": substituted,
        "feature_recomputation": vectors,
        "synthetic_failures_preserved": [{
            "kind": "scratch-path-permission",
            "detail": "system TemporaryDirectory raw-root creation denied before configuration; no builder, environment, claim, or slot call. The temp path was not cleaned or elevated.",
        }],
        "status": "passed",
    }


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=HERE / "preflight")
    args = parser.parse_args()
    result = run_zero_env_preflight(args.output_root)
    path = HERE / "ROUND64_PREFLIGHT.json"
    suffix = 1
    while path.exists():
        suffix += 1
        path = HERE / f"ROUND64_PREFLIGHT_{suffix:02d}.json"
    payload = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "live_binding": result["live_binding"],
                      "feature_changes": {str(v["delta"]): v["changed_indices_from_saved"]
                                          for v in result["feature_recomputation"]["variants"]}},
                     ensure_ascii=False))
