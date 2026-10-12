"""Fixed ROUND64 executor. Import is inert; only guarded `run_cell` reaches a builder."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2 = HERE.parent
R60_DIR = R2 / "round60_partitioned_actor_head_ablation"
R61_DIR = R2 / "round61_early_report_value_intervention"
RAW = HERE / "raw"
SCENE = 1700027
TARGET = "piloc-e1-target-5"
SEED = 33
MODE = "greedy"
DELTAS = (-30, 0, 30, 60)
EXPECTED_FILE_SHA = "573e4c26a4fee6f01405d4386ca73b518eeb1adb9cd8409bc83661e4113f6c5a"
EXPECTED_PARAM_SHA = "bce71a0e9810122c146950f12cf818a9a359406312d96dfc9c7a457ebb654e2a"
R64_EXECUTION_VERSION = "round64-controlled-deadline-intervention/6"
PLAN = [
    {"logical": "L", "delta": 0, "arm": "L", "action": None},
    {"logical": "L0", "delta": 0, "arm": "L0", "action": None},
    *[{"logical": f"d{delta:+d}_{arm}", "delta": delta, "arm": arm,
       "action": arm} for delta in DELTAS for arm in ("FORCE_SEND", "FORCE_SKIP")],
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = __import__("os").open(path, __import__("os").O_CREAT | __import__("os").O_EXCL |
                              __import__("os").O_WRONLY)
    with __import__("os").fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        __import__("os").fsync(stream.fileno())


def write_or_verify(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"existing immutable R64 record conflicts: {path}")
        return
    write_exclusive(path, value)


def persist_episode_row(raw_root, row):
    path = Path(row["artefacts"]["row"]).resolve()
    if not path.is_relative_to(Path(raw_root).resolve()):
        raise RuntimeError(f"R64 row output escaped the frozen raw root: {path}")
    write_exclusive(path, row)
    return path


def _slug(value):
    return re.sub(r"[^0-9A-Za-z_.+-]+", "_", str(value))


def setup_runtime(raw=RAW, *, preflight=False):
    """Configure actual R60 modules in the private R64 root, then load weights read-only."""
    sys.path.insert(0, str(R60_DIR))
    sys.path.insert(0, str(R61_DIR))
    import r61_execute
    import r60_execute
    import r60_partitioned as r60
    from r64_guard import R64Guard
    from r64_intervention import make_intervene_collector
    from r64_runtime import install_counting_hooks

    raw = Path(raw).resolve()
    if raw != RAW.resolve() and not preflight:
        raise RuntimeError("live R64 execution is bound to its single frozen raw root")
    cfg = r61_execute.setup(raw)
    model_path = R60_DIR / "raw" / "checkpoints" / "v2_train_s33_ep128.pt"
    if sha256(model_path) != EXPECTED_FILE_SHA:
        raise RuntimeError("R60 ep128 model file hash differs from frozen R64 identity")
    controller, state, identity = r60.load_checkpoint_read_only(model_path,
                                                               freeze=r60_execute.R60_FREEZE)
    parameter_sha = controller.parameter_summary()["parameter_sha256"]
    if parameter_sha != EXPECTED_PARAM_SHA:
        raise RuntimeError("R60 ep128 parameter identity differs from frozen R64 identity")
    if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
        raise RuntimeError("R64 model-only read path restored optimizer or RNG")
    cfg["controller"] = controller
    cfg["r64_cls"] = make_intervene_collector(cfg["plain_cls"])
    cfg["model_identity"] = {
        "model_path": str(model_path), "file_sha256": EXPECTED_FILE_SHA,
        "parameter_sha256": parameter_sha, "structure": identity.get("structure"),
        "parameter_identity": {"parameter_sha256": parameter_sha},
        "execution_version": R64_EXECUTION_VERSION,
        "load_mode": "read_only", "load_scope": "model_only", "optimizer_restored": False,
        "rng_restored": False, "training_cursor_restored": False,
    }
    if Path(cfg["identity"]["raw_root"]).resolve() != raw:
        raise RuntimeError("R60 output-owning modules are not bound to R64 raw root")
    joint_driver = sys.modules.get("integrated_sar.joint_driver")
    if joint_driver is None:
        raise RuntimeError("actual integrated_sar.joint_driver module is absent")
    runner = cfg["runner"]
    r42_run = cfg["r42_run"]
    guard = R64Guard(raw, allow_build=not preflight)
    hook_identity = install_counting_hooks(
        runner, joint_driver, joint_driver.JointRun, guard,
        run_proxy_class=r42_run._CountedJointRunProxy,
    )
    cfg["r64_guard"] = guard
    cfg["r64_hook_identity"] = hook_identity
    cfg["model_file_sha_before"] = sha256(model_path)
    cfg["model_file_path"] = model_path
    cfg["read_only_load_state"] = state
    cfg["read_only_load_identity"] = identity
    return cfg


def _on_time_ids(records, window=1000):
    detections = {}
    completed = {}
    for event in records:
        target = event.get("target_id")
        if not target or target == "__run__":
            continue
        if event.get("event_type") == "detection" and float(event.get("timestamp", 0)) <= window:
            detail = event.get("detail") or {}
            detections[target] = (float(event["timestamp"]),
                                  float(event["timestamp"]) + float(detail["deadline_offset"]))
        elif event.get("event_type") == "progress" and event.get("status") == "completed":
            completed.setdefault(target, float(event["timestamp"]))
    return sorted(target for target, (detected, deadline) in detections.items()
                  if target in completed and completed[target] <= deadline and completed[target] <= window)


def _focus_detection(records):
    found = [r for r in records if r.get("event_type") == "detection" and r.get("target_id") == TARGET]
    if len(found) != 1:
        raise RuntimeError(f"focus detection count must be one, found {len(found)}")
    return found[0]


def _focus_decision(tape):
    found = []
    for row in tape:
        targets = row.get("targets", [])
        if TARGET not in targets or TARGET not in row.get("free_first_reports", []):
            continue
        index = targets.index(TARGET)
        found.append({"step": int(row["step"]), "features_19d": row["features"][index],
                      "original_action": int(row["actions"][TARGET]),
                      "quote_bytes": row.get("attempt_charge_bytes", {}).get(TARGET),
                      "audit": row.get("candidate_audit", {}).get(TARGET),
                      "send_probabilities": row.get("send_probabilities", {}).get(TARGET),
                      "row": row})
    if len(found) != 1:
        raise RuntimeError(f"focus free first-report decision count must be one, found {len(found)}")
    return found[0]


def _read_cell_evidence(row, collector, attempt, arm, delta):
    from integrated_sar.lifecycle_metrics import load_records, window_metrics

    life_path = Path(row["artefacts"]["lifecycle"])
    tape_path = Path(row["artefacts"]["tape"])
    transcript_path = Path(row["artefacts"]["transcript"])
    lifecycle = load_records(life_path)
    tape = read_jsonl(tape_path)
    transcript = read_jsonl(transcript_path)
    detection = _focus_detection(lifecycle)
    decision = _focus_decision(tape)
    metric = window_metrics(lifecycle, 1000)
    ids = _on_time_ids(lifecycle)
    actual_offset = float(detection["detail"]["deadline_offset"])
    deadline = float(detection["timestamp"]) + actual_offset
    base_profile = [x for x in attempt.profile_records if x["target"] == TARGET]
    if arm == "L":
        original_offset = actual_offset
        if delta != 0:
            raise RuntimeError("L baseline must not carry a deadline shift")
    else:
        if len(base_profile) != 1:
            raise RuntimeError("R64 target profile generation was not recorded exactly once")
        original_offset = base_profile[0]["original_deadline_offset"]
        if abs(actual_offset - (original_offset + delta)) > 1e-9:
            raise RuntimeError("logged target offset differs from the frozen delta")
    if int(row.get("steps", -1)) != 1000 or metric.on_time != int(row["on_time"]):
        raise RuntimeError("episode steps or independent lifecycle on-time metric mismatch")
    if int(row.get("optimizer_steps", -1)) != 0:
        raise RuntimeError("unexpected optimizer step in frozen evaluation")
    load_identity = row.get("checkpoint_load_identity", {})
    if load_identity.get("parameter_identity", {}).get("parameter_sha256") != EXPECTED_PARAM_SHA:
        raise RuntimeError("episode row is not bound to the frozen R60 model parameters")
    if (load_identity.get("load_mode") != "read_only" or
            load_identity.get("optimizer_restored") is not False or
            load_identity.get("rng_restored") is not False):
        raise RuntimeError("episode restored training state or used a non-read-only model load")
    ack_ledger = row.get("ack_ledger")
    if (row.get("budget_bytes") != 29300 or not isinstance(ack_ledger, dict) or
            any(int(value) != 0 for value in ack_ledger.values())):
        raise RuntimeError("frozen budget or ACK-off check failed")
    if row.get("transport_parameters", {}).get("ack_enabled") is not False:
        raise RuntimeError("live CommRelay ACK channel is not disabled")
    config = attempt.configuration
    config_checks = {"steps": 1000, "num_goals": 6, "comm_budget_bytes": 29300,
                     "comm_ack": False, "seed": SCENE}
    if any(config.get(key) != value for key, value in config_checks.items()):
        raise RuntimeError(f"actual JointRun configuration differs from R64 freeze: {config_checks}")
    if decision["step"] != 134:
        raise RuntimeError(f"focus opportunity moved from ready 134 to {decision['step']}")
    if float(detection["timestamp"]) != 134.0:
        raise RuntimeError(f"focus detection time moved from frozen step 134 to {detection['timestamp']}")

    registry_record = [x for x in attempt.registry_records if x["event_id"] == detection["detail"]["detection_event_id"]]
    task_env_records = attempt.task_env_records
    message_records = attempt.message_records
    consumer_audit = None
    if len(registry_record) != 1 or not registry_record[0].get("registry_accepted"):
        raise RuntimeError("passive EventBus observer did not see an accepted registry receipt")
    if arm == "L":
        snapshot = attempt.live_run.registry.get(TARGET)
        internal = attempt.live_run.task_adapter.external_to_internal.get(TARGET)
        env_task = attempt.live_run.task_adapter.env.task_dic.get(internal) if internal is not None else None
        if snapshot is None or env_task is None:
            raise RuntimeError("baseline canonical registry/Task03 task missing")
        registry_deadline = float(snapshot.deadline)
        task_env_deadline = float(env_task["deadline"])
        messages = message_records
    else:
        if not task_env_records or not message_records:
            raise RuntimeError("a deadline authority consumer did not expose a target record")
        registry_deadline = float(registry_record[0]["registry_deadline"])
        task_env_deadline = float(task_env_records[-1]["task_env_deadline"])
        messages = message_records
    if abs(registry_deadline - deadline) > 1e-9 or abs(task_env_deadline - deadline) > 1e-9:
        raise RuntimeError("canonical registry and Task03 task deadlines diverged")
    if messages and any(abs(float(m["deadline"]) - deadline) > 1e-9 for m in messages):
        raise RuntimeError("CommRelay message deadline diverged from canonical deadline")

    iv = getattr(collector, "r64_record", None)
    if arm.startswith("FORCE_"):
        if not iv or iv["target"] != TARGET or not iv["sequentially_affordable"]:
            raise RuntimeError("force action failed to occur at the legal payable opportunity")
        if int(iv["ready_step"]) != decision["step"] or iv["features_19d"] != decision["features_19d"]:
            raise RuntimeError("interposer input differs from the persisted live actor tape")
        if abs(float(iv["deadline"]) - deadline) > 1e-9:
            raise RuntimeError("SendItem deadline differs from the canonical deadline")
        if abs(float(iv["remaining_time"]) - (deadline - float(iv["current_time"]))) > 1e-6:
            raise RuntimeError("SendItem remaining time does not derive from canonical deadline/clock")
        if bool(iv["legal_initial"]) is not True or iv["initial_block_reason"] != "allowed":
            raise RuntimeError("force interposer crossed a sequential budget mask")
        consumer_audit = {"registry_deadline": registry_deadline,
                          "task_env_deadline": task_env_deadline,
                          "message_deadlines": messages,
                          "send_item_deadline": iv["deadline"],
                          "send_item_remaining_time": iv["remaining_time"],
                          "send_item_clock": iv["current_time"],
                          "all_authority_consumers_equal": True}
    else:
        consumer_audit = {"registry_deadline": registry_deadline,
                          "task_env_deadline": task_env_deadline,
                          "message_deadlines": messages,
                          "all_authority_consumers_equal": True}

    target_chain = [x for x in lifecycle if x.get("target_id") == TARGET]
    target_events = [x for x in transcript if x.get("target") == TARGET]
    return {
        "logical": attempt.logical, "slot": attempt.slot, "delta": delta, "arm": arm,
        "scenario_seed": SCENE, "model_identity": cfg_model_identity(attempt),
        "metrics": {"steps": int(row["steps"]), "on_time": int(metric.on_time),
                    "on_time_ids": ids, "spent_bytes": int(row["spent_bytes"]),
                    "emission_sequence": row["emission_sequence"]},
        "focus": {
            "target": TARGET, "detection_step": float(detection["timestamp"]),
            "original_deadline_offset": original_offset, "actual_deadline_offset": actual_offset,
            "absolute_deadline": deadline,
            "ready_step": decision["step"], "baseline_model_action": decision["original_action"],
            "features_19d": decision["features_19d"], "quote_bytes": decision["quote_bytes"],
            "send_probabilities": decision["send_probabilities"],
            "intervention": iv, "target_chain": target_chain,
            "target_transcript": target_events,
        },
        "profile_generation_records": attempt.profile_records,
        "registry_records": attempt.registry_records,
        "task_env_deadline_records": task_env_records,
        "message_deadline_records": message_records,
        "consumer_consistency": consumer_audit,
        "configuration": attempt.configuration,
        "run_signature": attempt.live_run.run_signature,
        "artifacts": {name: {"path": str(Path(path).resolve()), "sha256": sha256(path)}
                      for name, path in row["artefacts"].items()},
        "row": {"path": str(Path(row["artefacts"]["row"]).resolve()),
                "sha256": sha256(row["artefacts"]["row"])},
    }


def cfg_model_identity(attempt):
    return attempt.model_identity


def run_cell(cfg, spec, *, retry=False):
    from r64_guard import R64GuardError
    guard = cfg["r64_guard"]
    if guard.root != RAW.resolve() or not guard.allow_build:
        raise R64GuardError("cell execution is not bound to the unique live R64 root")
    logical, delta, arm = spec["logical"], int(spec["delta"]), spec["arm"]
    previous = guard.terminal(logical)
    if previous in ("result_qualified", "result_disqualified"):
        raise R64GuardError(f"refusing to repeat terminal cell {logical}")
    use_reserve = previous == "interrupted"
    if use_reserve and not retry:
        raise R64GuardError(f"{logical} interrupted; retry requires explicit --retry-failed")
    if retry and not use_reserve:
        raise R64GuardError("reserve may only retry an interrupted logical")
    category = "reserve" if use_reserve else "normal"
    guard.snapshot()  # Verify the cumulative hard cap at the cell boundary.
    slug = _slug(logical)
    existing = [json.loads(p.read_text(encoding="utf-8")) for p in guard.slots.glob("*.slot")
                if json.loads(p.read_text(encoding="utf-8")).get("logical") == logical]
    attempt_number = len(existing) + 1
    cell_path = cfg["cells_dir"] / f"{slug}.json"
    with guard.attempt(logical, category=category, retry=use_reserve,
                       delta=delta, arm=arm) as attempt:
        attempt.model_identity = cfg["model_identity"]
        claim_path = cfg["raw_root"] / "launch_claims" / slug / f"attempt_{attempt_number:02d}.json"
        write_exclusive(claim_path, {"task": "ROUND64", "logical": logical, "slot": attempt.slot,
                                     "category": category, "retry": use_reserve,
                                     "delta": delta, "arm": arm,
                                     "model_parameter_sha256": EXPECTED_PARAM_SHA})
        attempt.row_path = cfg["raw_root"] / "rows" / f"{slug}_a{attempt_number:02d}.json"
        r42_run = cfg["r42_run"]
        r42_run._ACTIVE_CLAIM = claim_path
        r42_run._ACTIVE_PROGRESS = r42_run._new_run_progress()
        if arm == "L":
            collector = cfg["plain_cls"](cfg["controller"], greedy=True)
        else:
            from budget_masked import HOLD, SEND
            forced = SEND if arm == "FORCE_SEND" else HOLD
            collector = cfg["r64_cls"](cfg["controller"], greedy=True, target=TARGET,
                                        forced_action=forced, reason=f"round64:{delta}:{arm}",
                                        override_enabled=arm.startswith("FORCE_"))
        try:
            row = cfg["runner"]._run_episode(
                slug, "eval", SCENE, policy="learned", attach=collector.policy(),
                controller=cfg["controller"], collector=collector,
                action_rng_seed=cfg["runner"].eval_action_seed(SEED, SCENE, MODE),
                attempt=attempt_number, tag=f"{slug}_a{attempt_number:02d}",
                checkpoint_load_identity=cfg["model_identity"],
            )
        finally:
            progress = dict(r42_run._ACTIVE_PROGRESS or {})
            r42_run._ACTIVE_CLAIM = None
            r42_run._ACTIVE_PROGRESS = None
        attempt.configuration = getattr(attempt, "configuration", None)
        if attempt.configuration is None:
            raise RuntimeError("builder did not capture actual live configuration")
        persist_episode_row(cfg["raw_root"], row)
        if (int(progress.get("completed_steps", -1)) != 1000 or int(row.get("steps", -1)) != 1000):
            raise RuntimeError("fixed 1000-step episode did not complete")
        for name, artifact_path in row.get("artefacts", {}).items():
            resolved = Path(artifact_path).resolve()
            if not resolved.is_relative_to(cfg["raw_root"].resolve()):
                raise RuntimeError(f"R64 {name} artifact escaped its frozen raw root: {resolved}")
        evidence = _read_cell_evidence(row, collector, attempt, arm, delta)
        evidence["progress"] = progress
        evidence["row_summary"] = row
        evidence["collector_timing"] = collector.timing()
        evidence["collector_tape_summary"] = collector.tape_summary()
        evidence["execution_identity"] = cfg["identity"]
        evidence["counter_hook_identity"] = cfg["r64_hook_identity"]
        if sha256(cfg["model_file_path"]) != EXPECTED_FILE_SHA:
            raise RuntimeError("frozen model file changed during an R64 cell")
        after = cfg["controller"].parameter_summary()["parameter_sha256"]
        if after != EXPECTED_PARAM_SHA:
            raise RuntimeError("frozen model parameters changed during R64 cell")
        evidence["model_parameter_sha_after"] = after
        write_exclusive(cell_path, evidence)
        attempt.qualify(True, f"steps=1000 target_ready={evidence['focus']['ready_step']}")
        attempt.finish()
        attempt.evidence = evidence
        return evidence


def _normalized_lifecycle(path):
    rows = [row for row in read_jsonl(path) if row.get("event_type") != "run_config"]
    for row in rows:
        if row.get("event_type") == "final" and row.get("target_id") == "__run__":
            row.get("detail", {}).get("metrics", {}).pop("comm_transcript", None)
    return rows


def _final_transcript_path(path):
    finals = [row for row in read_jsonl(path) if row.get("event_type") == "final" and
              row.get("target_id") == "__run__"]
    if len(finals) != 1:
        raise RuntimeError("lifecycle must contain exactly one final event")
    return finals[0].get("detail", {}).get("metrics", {}).get("comm_transcript")


def gate_compare(L, L0):
    behavior_fields = ("steps", "on_time", "on_time_ids", "spent_bytes", "emission_sequence")
    same_metrics = all(L["metrics"][k] == L0["metrics"][k] for k in behavior_fields)
    l_art = L["artifacts"]
    l0_art = L0["artifacts"]
    tape_equal = read_jsonl(l_art["tape"]["path"]) == read_jsonl(l0_art["tape"]["path"])
    transcript_equal = read_jsonl(l_art["transcript"]["path"]) == read_jsonl(l0_art["transcript"]["path"])
    lifecycle_equal = (_normalized_lifecycle(l_art["lifecycle"]["path"]) ==
                       _normalized_lifecycle(l0_art["lifecycle"]["path"]))
    return {
        "L_equals_L0": bool(same_metrics and tape_equal and transcript_equal and lifecycle_equal),
        "science_fields": {k: {"L": L["metrics"][k], "L0": L0["metrics"][k]}
                           for k in behavior_fields},
        "bytewise_tape_equal": tape_equal, "bytewise_transcript_equal": transcript_equal,
        "lifecycle_equal_after_explicit_metadata_exclusions": lifecycle_equal,
        "lifecycle_metadata_normalization": {
            "field": "final.detail.metrics.comm_transcript",
            "L": _final_transcript_path(l_art["lifecycle"]["path"]),
            "L0": _final_transcript_path(l0_art["lifecycle"]["path"]),
        },
        "metadata_differences": {
            "logical": [L["logical"], L0["logical"]],
            "artifact_paths": {k: [L["artifacts"][k]["path"], L0["artifacts"][k]["path"]]
                               for k in L["artifacts"]},
            "run_signature": [L["run_signature"], L0["run_signature"]],
        },
    }


def build_gate_reconciliation(raw_root, model_identity):
    raw_root = Path(raw_root).resolve()
    old_path = raw_root / "ROUND64_GATE.json"
    old_gate = read_json(old_path)
    if old_gate.get("status") != "failed":
        raise RuntimeError("path-metadata reconciliation requires the preserved failed gate")
    cells_root = raw_root / "cells"
    L, L0 = read_json(cells_root / "L.json"), read_json(cells_root / "L0.json")
    gate = gate_compare(L, L0)
    r60row_path = R60_DIR / "raw" / "rows" / "v2_r60_eval_new_s33_greedy_s1700027_a1.json"
    r60row = read_json(r60row_path)
    r60life = r60row["artefacts"]["lifecycle"]
    baseline = {
        "row_path": str(r60row_path), "row_sha256": sha256(r60row_path),
        "on_time": r60row["on_time"], "on_time_ids": _on_time_ids(read_jsonl(r60life)),
        "spent_bytes": r60row["spent_bytes"], "emission_sequence": r60row["emission_sequence"],
        "L_tape_equal_R60": read_jsonl(L["artifacts"]["tape"]["path"]) ==
                            read_jsonl(r60row["artefacts"]["tape"]),
        "L_transcript_equal_R60": read_jsonl(L["artifacts"]["transcript"]["path"]) ==
                                  read_jsonl(r60row["artefacts"]["transcript"]),
        "L_lifecycle_equal_R60_after_exact_path_normalization":
            _normalized_lifecycle(L["artifacts"]["lifecycle"]["path"]) ==
            _normalized_lifecycle(r60life),
        "R60_final_transcript_path": _final_transcript_path(r60life),
    }
    gate["matches_R60_focus"] = all((
        baseline["L_tape_equal_R60"], baseline["L_transcript_equal_R60"],
        baseline["L_lifecycle_equal_R60_after_exact_path_normalization"],
        L["metrics"]["on_time"] == baseline["on_time"],
        L["metrics"]["on_time_ids"] == baseline["on_time_ids"],
        L["metrics"]["spent_bytes"] == baseline["spent_bytes"],
        L["metrics"]["emission_sequence"] == baseline["emission_sequence"],
    ))
    gate["status"] = "passed" if gate["L_equals_L0"] and gate["matches_R60_focus"] else "failed"
    gate["R60_focus"] = baseline
    gate["model_identity"] = model_identity
    return {
        "kind": "round64-neutral-gate-path-metadata-reconciliation/1",
        "status": gate["status"],
        "preserved_original_gate": {"status": old_gate["status"], "path": str(old_path),
                                    "sha256": sha256(old_path)},
        "explicitly_normalized_field": "final.detail.metrics.comm_transcript",
        "metadata_path_values": {
            "L": _final_transcript_path(L["artifacts"]["lifecycle"]["path"]),
            "L0": _final_transcript_path(L0["artifacts"]["lifecycle"]["path"]),
            "R60": baseline["R60_final_transcript_path"],
        },
        "strict_original_reason": "final lifecycle metrics embed each run's own transcript output path",
        "reconciled_gate": gate,
    }


def _decision_prefix(cell):
    tape = read_jsonl(cell["artifacts"]["tape"]["path"])
    target = cell["focus"]["target"]
    decision_step = cell["focus"]["ready_step"]
    prior = [row for row in tape if int(row["step"]) < decision_step]
    same = next(row for row in tape if int(row["step"]) == decision_step and target in row["targets"])
    index = same["targets"].index(target)
    prior_same_tick = {"targets": same["targets"][:index], "features": same["features"][:index],
                       "actions": {k: v for k, v in same["actions"].items()
                                   if k in same["targets"][:index]},
                       "audit": {k: v for k, v in same["candidate_audit"].items()
                                 if k in same["targets"][:index]}}
    life = read_jsonl(cell["artifacts"]["lifecycle"]["path"])
    pre_life = [row for row in life if row.get("event_type") != "run_config" and
                (float(row.get("timestamp", 0)) < decision_step or
                 (float(row.get("timestamp", 0)) == decision_step and
                  row.get("event_type") == "detection"))]
    return {"prior_tape_rows": prior, "prior_same_tick_candidates": prior_same_tick,
            "pre_action_lifecycle": pre_life}


def compare_force_pairs(cells):
    pairs = {}
    for delta in DELTAS:
        send = cells[f"d{delta:+d}_FORCE_SEND"]
        skip = cells[f"d{delta:+d}_FORCE_SKIP"]
        equal_prefix = _decision_prefix(send) == _decision_prefix(skip)
        pairs[str(delta)] = {"send_logical": send["logical"], "skip_logical": skip["logical"],
                             "common_prefix_equal": equal_prefix}
        if not equal_prefix:
            raise RuntimeError(f"SEND/SKIP pre-action prefix diverged for delta {delta}")
    return pairs


def _cell_cursor(cfg, spec, retry_logical=None):
    from r64_guard import R64GuardError

    guard = cfg["r64_guard"]
    logical = spec["logical"]
    path = cfg["cells_dir"] / f"{_slug(logical)}.json"
    terminal = guard.terminal(logical)
    slot_rows = [value for value in guard._slot_payloads() if value.get("logical") == logical]
    if terminal == "result_qualified":
        if not path.is_file():
            raise R64GuardError(f"qualified cell lacks its immutable evidence file: {logical}")
        result = read_json(path)
        if result.get("logical") != logical:
            raise R64GuardError(f"qualified cell evidence identity mismatch: {logical}")
        return result
    if terminal == "result_disqualified":
        raise R64GuardError(f"cell was disqualified and cannot be rerun: {logical}")
    if terminal == "interrupted":
        if retry_logical != logical:
            raise R64GuardError(f"{logical} interrupted; explicit reserve retry required")
        return run_cell(cfg, spec, retry=True)
    if slot_rows:
        raise R64GuardError(f"{logical} has consumed a slot without a terminal receipt; refusing replay")
    if path.exists():
        raise R64GuardError(f"unclaimed cell evidence exists without a slot receipt: {logical}")
    if retry_logical == logical:
        raise R64GuardError(f"reserve retry requested for an unconsumed cell: {logical}")
    return run_cell(cfg, spec)


def run_phase(cfg, phase, retry_logical=None):
    guard = cfg["r64_guard"]
    root = cfg["raw_root"]
    gate_file = root / "ROUND64_GATE.json"
    cells = {}
    if phase in ("gate", "all"):
        for spec in PLAN[:2]:
            cells[spec["logical"]] = _cell_cursor(cfg, spec, retry_logical)
        gate = gate_compare(cells["L"], cells["L0"])
        r60row_path = R60_DIR / "raw" / "rows" / "v2_r60_eval_new_s33_greedy_s1700027_a1.json"
        r60row = read_json(r60row_path)
        r60life = r60row["artefacts"]["lifecycle"]
        r60tape = r60row["artefacts"]["tape"]
        r60ct = r60row["artefacts"]["transcript"]
        r60_ids = _on_time_ids(read_jsonl(r60life))
        baseline = {
            "row_path": str(r60row_path), "row_sha256": sha256(r60row_path),
            "on_time": r60row["on_time"], "on_time_ids": r60_ids,
            "spent_bytes": r60row["spent_bytes"],
            "emission_sequence": r60row["emission_sequence"],
            "L_tape_equal_R60": read_jsonl(cells["L"]["artifacts"]["tape"]["path"]) == read_jsonl(r60tape),
            "L_transcript_equal_R60": read_jsonl(cells["L"]["artifacts"]["transcript"]["path"]) == read_jsonl(r60ct),
            "L_lifecycle_equal_R60_excluding_run_config":
                _normalized_lifecycle(cells["L"]["artifacts"]["lifecycle"]["path"]) ==
                _normalized_lifecycle(r60life),
        }
        gate["matches_R60_focus"] = all((baseline["L_tape_equal_R60"],
                                           baseline["L_transcript_equal_R60"],
                                           baseline["L_lifecycle_equal_R60_excluding_run_config"],
                                           cells["L"]["metrics"]["on_time"] == baseline["on_time"],
                                           cells["L"]["metrics"]["on_time_ids"] == baseline["on_time_ids"],
                                           cells["L"]["metrics"]["spent_bytes"] == baseline["spent_bytes"],
                                           cells["L"]["metrics"]["emission_sequence"] == baseline["emission_sequence"]))
        gate["status"] = "passed" if gate["L_equals_L0"] and gate["matches_R60_focus"] else "failed"
        gate["R60_focus"] = baseline
        gate["model_identity"] = cfg["model_identity"]
        write_or_verify(gate_file, gate)
        if gate["status"] != "passed":
            raise RuntimeError("L/L0/R60 baseline gate failed; matrix remains blocked")
    if phase in ("matrix", "all"):
        reconciliation = build_gate_reconciliation(root, cfg["model_identity"])
        if reconciliation.get("status") != "passed":
            raise RuntimeError("fixed intervention matrix blocked: explicit gate reconciliation failed")
        write_or_verify(root / "ROUND64_GATE_RECONCILIATION.json", reconciliation)
        for spec in PLAN[2:]:
            cells[spec["logical"]] = _cell_cursor(cfg, spec, retry_logical)
        pair_check = compare_force_pairs(cells)
        write_or_verify(root / "ROUND64_PAIR_PREFIX_GATE.json", pair_check)
    return {"phase": phase, "cells_returned": sorted(cells), "guard": guard.snapshot()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("gate", "matrix", "all"), required=True)
    parser.add_argument("--retry-failed", default=None)
    args = parser.parse_args(argv)
    cfg = setup_runtime(RAW)
    output = run_phase(cfg, args.phase, retry_logical=args.retry_failed)
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
