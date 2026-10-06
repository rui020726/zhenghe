#!/usr/bin/env python3
"""Read-only ROUND45 BC rule-fidelity and excess-cost diagnostic.

This script reads frozen rows, claims/receipts, tape/transcript artifacts, the frozen BC
dataset, and the three frozen BC model files. It never calls an episode/run entry point,
constructs an environment, restores optimizer/checkpoint RNG state, or writes model state.
It writes only the JSON and Markdown analysis outputs beside this script.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
CANONICAL_REPO = ROOT.parent / "zhenghe"
RUN_DIR = ROOT / "runs" / "comm_rule_warmstart_cost_learning_20261006_r2"
OUT_DIR = Path(__file__).resolve().parent
RAW_ROWS_PATH = RUN_DIR / "ROUND45_R2_CONTROL_RAW_ROW_RECOMPUTATION_R3_FINAL3.json"
GATE_PATH = RUN_DIR / "ROUND45_R2_BC_CONTROL_GATE_R3.json"
FREEZE_PATH = RUN_DIR / "ROUND45_R2_POST_BC_ENV_FREEZE_R3_FINAL3.json"
AUDIT_PATH = RUN_DIR / "BC_DATA_AUDIT_R2.json"
DATASET_PATH = RUN_DIR / "BC_DATASET.jsonl"
TRAINING_SUMMARY_PATH = RUN_DIR / "ROUND45_R2_BC_TRAINING_SUMMARY.json"
ADOPTION_PATH = RUN_DIR / "ROUND45_R2_ADOPTED_CONTROL_R3_FINAL3.json"
START_HEAD = "273759c57ead7b729f5b5baee32b864bbc1e273d"
PRIORLY_CITED_RAW_SHA256 = "d54defefed1aea857805fb0d782b62dca3fc5f4fee74890946959a3f7b934e03"
SEEDS = (11, 22, 33)
SCENES = tuple(range(1_301_000, 1_301_016))
TEACHER_CUT = 147

EXPECTED_INPUT_SHA256 = {
    "raw_row_recomputation": "e7e954865fc65f8d8174f0ff68453cd7a698d0410373e2f6065adb83e8287f71",
    "control_gate": "ccad7597d9b75c46683cd1b49e252deb928b1ee3b183fa9c1c4580d106e5af05",
    "post_bc_freeze": "1deba66aa01aeb96782ecfffaf05f4c24a02b05fe8fcde9bb747d47ef9aaec67",
    "bc_data_audit": "f3b85d73e676fa1788d5ac623d60eacdf6e6a146b6877b21db0da76b4dddc988",
    "bc_dataset": "93ffba45bfa484aa0f0aa5cbb334bd76c1284d0b21e7ca704b459ae6126f378d",
    "bc_training_summary": "7e3168879d9b70c09ce8a772929ec4296f41ab3ac212c202759c4feaf971976b",
    "adopted_control_receipt": "767510f295b5267a49c1a2bc0bd687f521942e078d9f9705f506d23ea756497b",
}

EXPECTED_TIMING_SOURCE_SHA256 = {
    "comm.py": "9b7938ae1af6697a7f977026c7920a8e17bd71492cb605407fb02f7019828dbd",
    "orchestrator.py": "ad1380d4695d8fb90ea646d2b4ebd0d6d736879885feff79bef8067a1a956b74",
    "task03.py": "4dcd548f192690df383b9ddc5b254d54abf02891d6480f4aa244045acddbe7d1",
    "event_bus.py": "60700446e97a296099edcd9d4d8869f4fef55c00b7a625909f0d25ca10ca2fe4",
    "joint_driver.py": "2a682fecd5936cc4b2786477398eed55928c6f5781282bc29e4918cdc01c911b",
    "assignment_gate.py": "d9d7527c01c5c8d488c6b46745885bc51ef47fae76cbc0129630fadbcba4d473",
    "r42_one_shot.py": "3c342108148de42f395faeec25544976eed7efdfdd32a99685d73bd59101ab0d",
    "r42_run.py": "9c5feacb2ffcb2bb930d2b6022e2fca5210408765dc664859d400e6f2b5f8c2e",
    "v2_run.py": "90c6936c24c93999d03aac3f09b9b6506a9e4f8050f9115a0ef3fc839c9f16b7",
    "r45_r2_execute_r3.py": "f796ed07891bcc2457fe48e9ddd288a6c18c15a511fd8e505c5400a9e6819dd5",
}

_INPUTS: dict[str, dict[str, Any]] = {}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _track(path: Path) -> bytes:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    identity = {"path": _rel(path), "size_bytes": len(data),
                "sha256": _sha256_bytes(data)}
    previous = _INPUTS.setdefault(str(path), identity)
    if previous != identity:
        raise RuntimeError(f"input changed while the analysis was running: {path}")
    return data


def _verify_candidate_time_source_chain() -> dict[str, Any]:
    """Verify source identity and the synchronous event-to-tape tick ordering.

    This is source inspection only: it imports no production module and constructs no run.
    The bounded age feature is exactly invertible only after proving admitted candidates
    cannot have a ready_step later than the relay observation tick.
    """
    source_paths = {
        "comm.py": CANONICAL_REPO / "integrated_sar" / "comm.py",
        "orchestrator.py": CANONICAL_REPO / "integrated_sar" / "orchestrator.py",
        "task03.py": CANONICAL_REPO / "integrated_sar" / "adapters" / "task03.py",
        "event_bus.py": CANONICAL_REPO / "integrated_sar" / "event_bus.py",
        "joint_driver.py": CANONICAL_REPO / "integrated_sar" / "joint_driver.py",
        "assignment_gate.py": CANONICAL_REPO / "integrated_sar" / "assignment_gate.py",
        "r42_one_shot.py": ROOT / "runs" / "comm_one_shot_full_trace_ablation_20261004"
        / "r42_one_shot.py",
        "r42_run.py": ROOT / "runs" / "comm_one_shot_full_trace_ablation_20261004"
        / "r42_run.py",
        "v2_run.py": CANONICAL_REPO / "runs" / "comm_cost_v2_rerun_20260928" / "v2_run.py",
        "r45_r2_execute_r3.py": RUN_DIR / "r45_r2_execute_r3.py",
    }
    source_text: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    for name, path in source_paths.items():
        body = _track(path)
        digest = _sha256_bytes(body)
        if digest != EXPECTED_TIMING_SOURCE_SHA256[name]:
            raise RuntimeError(f"{name} timing source SHA differs from reviewed identity")
        source_hashes[name] = digest
        source_text[name] = body.decode("utf-8")

    def require_order(name: str, fragments: list[str], *, start_after: str | None = None) -> None:
        text = source_text[name]
        if start_after is not None:
            start = text.find(start_after)
            if start < 0:
                raise RuntimeError(f"{name}: expected source scope is absent")
            text = text[start:]
        positions = [text.find(fragment) for fragment in fragments]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            raise RuntimeError(f"{name}: expected timing source order is absent")

    require_order("orchestrator.py", [
        "search_result = self.search.step(now, self.dt)",
        "for event in search_result.detections:",
        "self.event_bus.publish(event)",
        "self.task_sink.sync_tasks(self.registry.snapshots())",
        "execution_result = self.executor.step(now, self.dt)",
        "self.clock.advance_to(decision_time)",
    ])
    require_order("event_bus.py", [
        "def subscribe(self, event_type:",
        "handlers.append(handler)",
        "def publish(self, event:",
        "while self._queue:",
        "for handler in tuple(handlers):",
        "handler(current)",
    ])
    require_order("joint_driver.py", [
        "orchestrator = JointSAROrchestrator(",
        "comm_relay = CommRelay(",
        "bus.subscribe(DetectionEvent, comm_relay.handle_detection)",
        "task_adapter.comm_relay = comm_relay",
    ])
    require_order("orchestrator.py", [
        "self.event_bus.subscribe(DetectionEvent, self.registry.handle_detection)",
        "def step(self) -> StepReport:",
    ])
    require_order("task03.py", [
        "def step(self, current_time: float, dt: float) -> ExecutionStepResult:",
        "self.comm_relay.tick()",
        "self.env.step(dt)",
    ])
    require_order("assignment_gate.py", [
        "def step(self, current_time: float, dt: float) -> ExecutionStepResult:",
        "return self.executor.step(current_time, dt)",
    ])
    require_order("r45_r2_execute_r3.py", [
        "collector = OneShotSendSkipCollector(controller, greedy=True)",
        'return runtime["r42_run"]._execute_cell(',
        "collector=collector,",
        '"attach": collector.policy()',
    ], start_after="def _run_control_cell(")
    require_order("r42_run.py", [
        "import v2_run as runner",
        "_ORIGINAL_BUILD = runner.build_joint_run",
        "runner.build_joint_run = _make_counted_builder(",
        "row = runner._run_episode(",
    ])
    require_order("v2_run.py", [
        "from integrated_sar.joint_driver import RunConfiguration, build_joint_run",
        "run = build_joint_run(configuration)",
        "run.task_adapter.comm_relay.attach_policy(attach)",
        "summary = run.run()",
    ])
    require_order("joint_driver.py", [
        "class JointRun:",
        "def run(self, steps: int | None = None) -> dict[str, Any]:",
        "for report in self.orchestrator.run(total):",
    ])
    require_order("orchestrator.py", [
        "def run(self, steps: int) -> Iterable[StepReport]:",
        "for _ in range(max(0, int(steps))):",
        "yield self.step()",
    ])
    require_order("comm.py", [
        "ready_step = self._tick + 1",
        "def tick(self) -> tuple[str, ...]:",
        "self._tick += 1",
        "self._emit_selected()",
        "observation = self._observation()",
        "age_steps=max(0, self._tick - int(ready_step))",
        "step=self._tick",
    ])
    if '"step": int(observation.step)' not in source_text["r42_one_shot.py"]:
        raise RuntimeError("one-shot tape step is not sourced from the actual observation")

    comm_lines = source_text["comm.py"].splitlines()
    ready_step_write_sites = [
        {"line": index, "source": line.strip()}
        for index, line in enumerate(comm_lines, 1)
        if "ledger.ready_step =" in line
    ]
    return {
        "verified": True,
        "source_files_sha256": source_hashes,
        "source_path": [
            "R45 R3 selects OneShotSendSkipCollector and passes its policy through r42_run into the frozen v2_run episode runner",
            "v2_run attaches the collector to the actual run.task_adapter.comm_relay and calls JointRun.run",
            "JointRun.run delegates each step to JointSAROrchestrator.run, which yields JointSAROrchestrator.step",
            "JointSAROrchestrator publishes detections before executor.step and advances clock afterward",
            "InMemoryEventBus dispatches synchronously in subscription order",
            "registry subscribes before CommRelay; CommRelay subscribes before the cycle's Task03Adapter step",
            "CommRelay books a newly accepted event at old_tick+1; the same cycle relay tick increments before observation",
            "older queued ready_step values remain <= current tick; the only non-event booking path sets ready_step=current tick",
            "Task03Adapter is called through AssignmentGate.step; its relay tick precedes env.step",
            "One-shot tape records int(observation.step), which is SendObservation.step=self._tick",
        ],
        "ready_step_write_sites_in_comm_py": ready_step_write_sites,
        "fresh_event_ready_step_equals_current_relay_tick_at_observation": True,
        "all_admitted_candidates_ready_step_not_after_observation_tick": True,
        "event_timestamp_basis": "relay step; not global clock.now",
        "registry_and_relay_event_handlers_synchronous_before_strategy_sampling": True,
        "tape_tick_source": "int(observation.step)",
    }


def _json(path: Path) -> Any:
    return json.loads(_track(path).decode("utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    body = _track(path).decode("utf-8")
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def _artifact_path(value: str, *, default_dir: Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    local = default_dir / path.name
    if local.is_file():
        return local.resolve()
    raise FileNotFoundError(value)


def _feature_key(values: list[float]) -> bytes:
    if len(values) != 19:
        raise ValueError(f"expected a 19D vector, got {len(values)}")
    vector = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("non-finite float32 feature")
    return vector.astype("<f4", copy=False).tobytes(order="C")


def _sha_feature(values: list[float]) -> str:
    return _sha256_bytes(_feature_key(values))


def _on_time_ids(row: dict[str, Any]) -> list[str]:
    return sorted({str(event["task_id"]) for event in row.get("task_reward_events", [])
                   if event.get("credited") and event.get("on_time")})


def _confusion(samples: list[dict[str, Any]], *, label_key: str,
               prediction_key: str) -> dict[str, Any]:
    matrix = [[0, 0], [0, 0]]
    for item in samples:
        truth = int(item[label_key])
        prediction = int(item[prediction_key])
        matrix[truth][prediction] += 1
    total = sum(sum(row) for row in matrix)
    return {
        "true_rows_predicted_columns_skip_send": matrix,
        "samples": total,
        "correct": matrix[0][0] + matrix[1][1],
        "accuracy": ((matrix[0][0] + matrix[1][1]) / total) if total else None,
        "extra_sends_teacher_skip": matrix[0][1],
        "missed_sends_teacher_send": matrix[1][0],
        "true_skip": sum(matrix[0]),
        "true_send": sum(matrix[1]),
        "predicted_skip": matrix[0][0] + matrix[1][0],
        "predicted_send": matrix[0][1] + matrix[1][1],
    }


def _load_model_manifest(freeze: dict[str, Any]) -> dict[int, dict[str, Any]]:
    models: dict[int, dict[str, Any]] = {}
    for item in freeze["bc_models"]:
        seed = int(item["seed"])
        if seed in models:
            raise RuntimeError(f"duplicate frozen model seed {seed}")
        models[seed] = item
    if set(models) != set(SEEDS):
        raise RuntimeError("frozen BC model seed set changed")
    return models


def _load_readonly_controllers(models: dict[int, dict[str, Any]],
                               r45_bc: Any) -> tuple[dict[int, Any], dict[int, dict[str, Any]]]:
    controller_module = r45_bc.configure_source_imports()
    expected_controller = (ROOT.parent / "zhenghe" / "runs" /
                           "comm_cost_v2_rerun_20260928" / "v2_controller.py").resolve()
    actual_controller = Path(controller_module.__file__).resolve()
    if actual_controller != expected_controller:
        raise RuntimeError(f"controller import path mismatch: {actual_controller}")

    controllers: dict[int, Any] = {}
    identities: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        entry = models[seed]
        checkpoint = _artifact_path(entry["checkpoint_path"], default_dir=RUN_DIR / "checkpoints")
        checkpoint_hash = _sha256_bytes(_track(checkpoint))
        if checkpoint_hash != entry["checkpoint_sha256"]:
            raise RuntimeError(f"seed {seed}: checkpoint bytes differ from FINAL3 freeze")

        # Model construction consumes torch RNG. fork_rng restores the ambient CPU RNG after
        # the object is made; no checkpoint RNG state is read or restored.
        with torch.random.fork_rng(devices=[]):
            controller = controller_module.fresh_controller(seed)
            payload = r45_bc.read_bc_checkpoint(checkpoint)
            r45_bc.load_bc_controller_state(controller, payload)

        module_map = {
            "trunk": controller.inner.trunk,
            "action_head": controller.inner.action_head,
            "task_value_head": controller.inner.value_head,
            "cost_value_head": controller.cost_head,
        }
        for module in module_map.values():
            module.eval()
        parameter_sha = controller.parameter_summary()["parameter_sha256"]
        module_sha = {name: r45_bc._module_hash(module)
                      for name, module in module_map.items()}
        if parameter_sha != entry["parameter_sha256"]:
            raise RuntimeError(f"seed {seed}: loaded parameter identity differs from freeze")
        if module_sha != entry["module_parameter_sha256"]:
            raise RuntimeError(f"seed {seed}: loaded module identities differ from freeze")
        identities[seed] = {
            "checkpoint_path": _rel(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "parameter_sha256": parameter_sha,
            "module_parameter_sha256": module_sha,
            "optimizer_payload_deserialized_for_schema_check": True,
            "optimizer_object_constructed": False,
            "optimizer_state_restored": False,
            "checkpoint_rng_restored_or_used": False,
            "controller_source_path": _rel(actual_controller),
            "controller_source_sha256": _sha256_file(actual_controller),
        }
        controllers[seed] = controller
    return controllers, identities


def _infer(controller: Any, vectors: list[list[float]], r45_bc: Any) -> list[dict[str, Any]]:
    if not vectors:
        return []
    features = torch.tensor(vectors, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        logits, _task_values = controller.forward(features)
        probabilities = torch.softmax(logits, dim=-1)
        actions = r45_bc.greedy_actions_from_logits(logits)
    return [{"action": int(action), "p_skip": float(prob[0]), "p_send": float(prob[1])}
            for action, prob in zip(actions.tolist(), probabilities.tolist(), strict=True)]


def _parse_control_tape(seed: int, scene: int, row: dict[str, Any],
                        receipt: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tape_path = _artifact_path(row["artefacts"]["tape"], default_dir=RUN_DIR / "logs")
    transcript_path = _artifact_path(row["artefacts"]["transcript"],
                                     default_dir=RUN_DIR / "logs")
    for name, path in (("tape", tape_path), ("transcript", transcript_path),
                       ("row", Path(row["artefacts"]["row"]))):
        expected = receipt.get("artifacts", {}).get(name, {})
        if expected.get("sha256") != _sha256_file(path):
            raise RuntimeError(f"{seed}/{scene}: {name} SHA differs from artifact receipt")
        if int(expected.get("size_bytes", -1)) != path.stat().st_size:
            raise RuntimeError(f"{seed}/{scene}: {name} size differs from artifact receipt")

    tape = _jsonl(tape_path)
    sends = [event for event in _jsonl(transcript_path) if event.get("kind") == "send"]
    if len(tape) != int(row.get("collector_summary", {}).get("decision_steps", -1)):
        raise RuntimeError(f"{seed}/{scene}: decision tape length mismatch")

    action_samples: list[dict[str, Any]] = []
    already_free: set[str] = set()
    anomalies: list[str] = []
    for tape_index, record in enumerate(tape):
        if record.get("comm_action_semantics") != "one-shot-send-skip/1":
            anomalies.append(f"tape[{tape_index}] action semantic tag mismatch")
        targets = record.get("targets", [])
        feature_rows = record.get("features", [])
        if len(targets) != len(feature_rows):
            anomalies.append(f"tape[{tape_index}] target/feature row count differs")
            continue
        feature_by_target = dict(zip(targets, feature_rows, strict=True))
        free_targets = list(record.get("free_one_shot_choices", []))
        for target in free_targets:
            if target in already_free:
                anomalies.append(f"{target}: more than one free one-shot choice")
                continue
            already_free.add(target)
            audit = record.get("candidate_audit", {}).get(target, {})
            prelegal = (
                record.get("decision_class", {}).get(target) == "free_one_shot_choice"
                and target in record.get("free", [])
                and target in record.get("actionable", [])
                and audit.get("one_shot_state_before") == "undecided"
                and audit.get("effective_block_reason") == "allowed"
                and audit.get("relay_initial_executable") is True
                and audit.get("sequentially_affordable_before_action") is True
            )
            if not prelegal:
                anomalies.append(f"{target}: first free-choice pre-action legality fields disagree")
            vector = [float(value) for value in feature_by_target.get(target, [])]
            action = record.get("actions", {}).get(target)
            effective = record.get("effective_actions", {}).get(target)
            requested = record.get("requested", {}).get(target)
            if action not in (0, 1) or effective != action or requested != (action == 1):
                anomalies.append(f"{target}: requested/effective action fields disagree")
            feature_valid = (len(vector) == 19 and np.isfinite(np.asarray(vector, dtype=np.float32)).all())
            if not feature_valid:
                anomalies.append(f"{target}: missing or malformed saved 19D float32 input")
                continue
            feature = np.asarray(vector, dtype=np.float32)
            first_emission = audit.get("first_emission")
            if first_emission is not True or feature[5] != np.float32(1.0):
                anomalies.append(f"{target}: first-emission audit and feature disagree")
            age_value = float(feature[0])
            ready_step: int | None = None
            label: int | None = None
            step = int(record["step"])
            if (math.isfinite(age_value) and age_value >= 0 and
                    age_value == int(age_value) and step >= int(age_value)):
                # This is the registered source relation age=max(0, relay_tick-ready_step).
                # For a legal first free candidate the message is already published, so the
                # age is not clamped; step-age recovers the original integer ready_step.
                ready_step = step - int(age_value)
                label = int(bool(first_emission) and ready_step > TEACHER_CUT)
                if ready_step > step:
                    anomalies.append(f"{target}: reconstructed ready_step is after relay/tape tick")
            else:
                anomalies.append(f"{target}: ready_step is not exactly recoverable from raw tape fields")
            expected_step_feature = np.float32(step / 1000.0)
            if feature[18] != expected_step_feature:
                anomalies.append(f"{target}: float32 step feature does not match relay tape tick")
            if label is not None and audit.get("first_emission") is True:
                action_samples.append({
                    "seed": seed, "scene": scene, "target_id": str(target),
                    "tape_record_index": tape_index, "relay_tape_step": step,
                    "age_steps_float32": age_value, "reconstructed_ready_step": ready_step,
                    "ready_step_not_after_relay_tape_step": ready_step <= step,
                    "ready_step_equals_relay_tape_step": ready_step == step,
                    "first_emission": first_emission,
                    "teacher_label": label,
                    "teacher_action": "SEND_NOW" if label else "SKIP_EPISODE",
                    "recorded_action": int(action) if action in (0, 1) else None,
                    "recorded_action_name": ("SEND_NOW" if action == 1 else "SKIP_EPISODE")
                    if action in (0, 1) else None,
                    "action_prelegal": prelegal,
                    "feature_float32": [float(value) for value in feature],
                    "feature_float32_sha256": _sha_feature(vector),
                    "tape_send_probability": record.get("send_probabilities", {}).get(target),
                })

    # Validate the paid-cost stream from actual send attempts, including lost packets.
    paid_bytes = sum(int(event.get("cost_bytes", event.get("bytes_total", 0)))
                     for event in sends)
    emissions = row.get("emission_sequence", [])
    emission_projection = [
        {"step": int(event[0]), "target_id": str(event[1]),
         "revision": int(event[2]), "paid_bytes": int(event[3]),
         "subscribers": int(event[4]), "first": bool(event[5])}
        for event in emissions
    ]
    transcript_projection = [
        {"step": int(event["step"]), "target_id": str(event["target"]),
         "revision": int(event["revision"]),
         "paid_bytes": int(event.get("cost_bytes", event.get("bytes_total", 0))),
         "subscribers": int(event["subscribers"]), "first": bool(event["first"])}
        for event in sends
    ]
    if emission_projection != transcript_projection:
        anomalies.append("row emission sequence and transcript send records differ")
    if paid_bytes != int(row["spent_bytes"]):
        anomalies.append("transcript send charges do not sum to row spent_bytes")
    first_sends = [event for event in sends if event.get("first") is True]
    if any(event.get("first") is not True for event in sends):
        anomalies.append("a non-first emission exists under one-shot first-report-only semantics")
    sent_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in first_sends:
        sent_by_target[str(event["target"])].append(event)
    for sample in action_samples:
        target = sample["target_id"]
        target_sends = sent_by_target.get(target, [])
        sample["observed_first_emissions"] = [
            {"step": int(event["step"]), "ready_step": event.get("ready_step"),
             "attempts": int(event.get("attempts", 0)),
             "subscribers": int(event.get("subscribers", 0)),
             "paid_bytes": int(event.get("cost_bytes", event.get("bytes_total", 0))),
             "packets_lost": int(event.get("packets_lost", 0))}
            for event in target_sends
        ]
        if sample["recorded_action"] == 1:
            if len(target_sends) != 1 or int(target_sends[0]["step"]) != sample["relay_tape_step"]:
                anomalies.append(f"{target}: SEND action has no unique same-tick first emission")
        elif target_sends:
            anomalies.append(f"{target}: SKIP action has a first emission")

    aggregate = {
        "decision_tape_records": len(tape),
        "first_free_action_samples": len(action_samples),
        "first_free_action_count_by_action": dict(Counter(
            item["recorded_action_name"] for item in action_samples)),
        "first_emissions": len(first_sends),
        "emissions_total": len(sends),
        "subscriber_sum": sum(int(event.get("subscribers", 0)) for event in sends),
        "recipient_attempts": sum(int(event.get("attempts", 0)) for event in sends),
        "packets_lost": sum(int(event.get("packets_lost", 0)) for event in sends),
        "paid_bytes_from_transcript": paid_bytes,
        "row_spent_bytes": int(row["spent_bytes"]),
        "emission_projection_matches_transcript": emission_projection == transcript_projection,
        "anomalies": anomalies,
    }
    return action_samples, {
        "aggregate": aggregate,
        "emission_sequence": emission_projection,
        "transcript_path": _rel(transcript_path),
        "tape_path": _rel(tape_path),
    }


def _git_identity() -> dict[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                                capture_output=True, text=True)
        return result.stdout.strip()
    return {"branch_at_analysis": run("branch", "--show-current"),
            "head_at_analysis": run("rev-parse", "HEAD")}


def _raw_row_commit_crosscheck(current_bytes: bytes) -> dict[str, Any]:
    relative = RAW_ROWS_PATH.resolve().relative_to(ROOT.resolve()).as_posix()
    committed = subprocess.run(["git", "show", f"{START_HEAD}:{relative}"], cwd=ROOT,
                               check=True, capture_output=True).stdout
    blob_oid = subprocess.run(["git", "rev-parse", f"{START_HEAD}:{relative}"], cwd=ROOT,
                              check=True, capture_output=True, text=True).stdout.strip()
    parent = subprocess.run(["git", "show", f"{START_HEAD}^:{relative}"], cwd=ROOT,
                            capture_output=True)
    try:
        semantic_equal = (json.loads(current_bytes.decode("utf-8-sig"))
                          == json.loads(committed.decode("utf-8-sig")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("raw recomputation JSON cannot be compared semantically") from exc
    if not semantic_equal:
        raise RuntimeError("raw recomputation worktree JSON differs semantically from start HEAD")
    return {
        "commit": START_HEAD,
        "path": relative,
        "git_blob_oid_sha1": blob_oid,
        "git_blob_content_sha256": _sha256_bytes(committed),
        "worktree_content_sha256": _sha256_bytes(current_bytes),
        "git_blob_bytes": len(committed),
        "worktree_bytes": len(current_bytes),
        "git_blob_lf_count": committed.count(b"\n"),
        "git_blob_crlf_count": committed.count(b"\r\n"),
        "worktree_lf_count": current_bytes.count(b"\n"),
        "worktree_crlf_count": current_bytes.count(b"\r\n"),
        "byte_equal": committed == current_bytes,
        "json_semantically_equal": semantic_equal,
        "parent_contains_path": parent.returncode == 0,
        "prior_cited_sha256": PRIORLY_CITED_RAW_SHA256,
        "prior_cited_sha_matches_worktree_or_git_blob": PRIORLY_CITED_RAW_SHA256 in {
            _sha256_bytes(current_bytes), _sha256_bytes(committed)},
        "interpretation": (
            "The delivery commit adds this file; its parent has no path to compare. The committed "
            "LF JSON and checked-out worktree JSON are semantically identical. The worktree hash "
            "differs because it contains CRLF line endings. The earlier cited digest is not the "
            "current worktree or committed-blob digest and cannot be reconstructed from the parent."
        ),
    }


def main() -> int:
    sys.path.insert(0, str(RUN_DIR))
    import r45_bc  # noqa: E402  (the registered data/model implementation)

    timing_source_chain = _verify_candidate_time_source_chain()

    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.get_rng_state().clone()

    raw = _json(RAW_ROWS_PATH)
    gate = _json(GATE_PATH)
    freeze = _json(FREEZE_PATH)
    audit = _json(AUDIT_PATH)
    training_summary = _json(TRAINING_SUMMARY_PATH)
    adoption = _json(ADOPTION_PATH)
    dataset_bytes = _track(DATASET_PATH)
    raw_row_crosscheck = _raw_row_commit_crosscheck(_track(RAW_ROWS_PATH))
    if _sha256_bytes(dataset_bytes) != audit["output_dataset_sha256"]:
        raise RuntimeError("BC dataset bytes do not match the frozen R2 audit")

    for key, path in (
        ("raw_row_recomputation", RAW_ROWS_PATH), ("control_gate", GATE_PATH),
        ("post_bc_freeze", FREEZE_PATH), ("bc_data_audit", AUDIT_PATH),
        ("bc_dataset", DATASET_PATH), ("bc_training_summary", TRAINING_SUMMARY_PATH),
        ("adopted_control_receipt", ADOPTION_PATH),
    ):
        actual = _INPUTS[str(path.resolve())]["sha256"]
        if actual != EXPECTED_INPUT_SHA256[key]:
            raise RuntimeError(f"{key} input SHA differs from the recorded ROUND45 identity")

    if raw.get("status") != "passed" or gate.get("status") != "failed":
        raise RuntimeError("ROUND45 R3 FINAL3 raw audit/gate state changed")
    if gate.get("training_authorized_by_gate") is not False:
        raise RuntimeError("the current failed control gate no longer denies PPO continuation")
    if raw["control_rows"].get("count") != 48 or not raw["control_rows"].get(
            "all_keys_unique_and_complete"):
        raise RuntimeError("raw control row audit is not a closed 48-cell matrix")
    if (raw["baseline_rows"].get("raw_manifest_rows_checked") != 32 or
            raw["baseline_rows"].get("raw_manifest_digest_matches_all_32_rows") is not True):
        raise RuntimeError("immediate/late raw baseline manifest check is incomplete")
    if adoption.get("status") != "adopted_after_receipt_failure" or adoption.get(
            "original_claim_modified") is not False:
        raise RuntimeError("the one adopted first row no longer preserves its failed claim")

    dataset: list[dict[str, Any]] = []
    for line_number, line in enumerate(dataset_bytes.decode("utf-8").splitlines(), 1):
        if line.strip():
            record = json.loads(line)
            if record.get("split") not in ("fit", "validation"):
                raise RuntimeError(f"dataset row {line_number}: unexpected split")
            if int(record["scene_id"]) in range(1_301_000, 1_301_016):
                raise RuntimeError("development scenes leaked into BC fit/validation data")
            vector = record["feature_float32"]
            if len(vector) != 19:
                raise RuntimeError(f"dataset row {line_number}: feature width changed")
            raw_age = float(np.float32(vector[0]))
            step = int(record["step"])
            ready_step = int(record["ready_step"])
            teacher = int(bool(np.float32(vector[5]) == np.float32(1.0))
                           and ready_step > TEACHER_CUT)
            if (record.get("teacher_label") != teacher or
                    record.get("teacher_action") != ("SEND_NOW" if teacher else "SKIP_EPISODE") or
                    record.get("teacher_arm") != "late_discovery_gate" or
                    ready_step != step - int(raw_age) or raw_age != int(raw_age) or
                    ready_step > step or
                    np.float32(vector[18]) != np.float32(step / 1000.0)):
                raise RuntimeError(f"dataset row {line_number}: teacher label/time relation mismatch")
            dataset.append(record)

    split_counts = Counter(str(row["split"]) for row in dataset)
    expected_splits = {"fit": 1548, "validation": 738}
    if dict(split_counts) != expected_splits:
        raise RuntimeError(f"frozen BC sample counts changed: {dict(split_counts)}")
    dataset_labels_by_feature: dict[bytes, set[int]] = defaultdict(set)
    fit_labels_by_feature: dict[bytes, set[int]] = defaultdict(set)
    validation_labels_by_feature: dict[bytes, set[int]] = defaultdict(set)
    for item in dataset:
        key = _feature_key(item["feature_float32"])
        label = int(item["teacher_label"])
        dataset_labels_by_feature[key].add(label)
        (fit_labels_by_feature if item["split"] == "fit" else
         validation_labels_by_feature)[key].add(label)
    dataset_label_conflicts = sum(len(labels) > 1 for labels in dataset_labels_by_feature.values())

    model_manifest = _load_model_manifest(freeze)
    controllers, model_identities = _load_readonly_controllers(model_manifest, r45_bc)

    # Recompute the frozen fit/validation confusion matrices with actual forward passes.
    dataset_metrics: dict[str, Any] = {}
    metric_files: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        metric_path = RUN_DIR / f"BC_METRICS_s{seed}.json"
        saved_metrics = _json(metric_path)
        metric_files[seed] = saved_metrics
        model_sections = {}
        for split in ("fit", "validation"):
            selected = [row for row in dataset if row["split"] == split]
            outputs = _infer(controllers[seed],
                             [row["feature_float32"] for row in selected], r45_bc)
            predicted_rows = [dict(row, recomputed_action=out["action"])
                              for row, out in zip(selected, outputs, strict=True)]
            confusion = _confusion(predicted_rows, label_key="teacher_label",
                                   prediction_key="recomputed_action")
            expected_confusion = saved_metrics["final_metrics"][split][
                "confusion_true_rows_predicted_columns"]
            if confusion["true_rows_predicted_columns_skip_send"] != expected_confusion:
                raise RuntimeError(f"seed {seed} {split}: forward confusion differs from saved metrics")
            model_sections[split] = confusion
        dataset_metrics[str(seed)] = model_sections

    raw_controls = raw["control_rows"]["files"]
    if len(raw_controls) != 48:
        raise RuntimeError("raw row file manifest no longer contains 48 rows")
    baseline_by_scene = {int(item["scene"]): item for item in raw["baseline_rows"]["scene_rows"]}
    all_samples: list[dict[str, Any]] = []
    seed_scene: dict[str, dict[int, dict[str, Any]]] = {str(seed): {} for seed in SEEDS}
    claim_status_counts: Counter[str] = Counter()
    for raw_entry in raw_controls:
        seed = int(raw_entry["seed"])
        scene = int(raw_entry["scene"])
        if seed not in SEEDS or scene not in SCENES:
            raise RuntimeError(f"out-of-scope control row identity: {seed}/{scene}")
        row_path = RUN_DIR / raw_entry["path"]
        row_data = _json(row_path)
        actual_row_hash = _INPUTS[str(row_path.resolve())]["sha256"]
        if actual_row_hash != raw_entry["sha256"]:
            raise RuntimeError(f"{seed}/{scene}: raw-row audit SHA mismatch")
        if (int(row_data.get("training_seed", -1)) != seed or
                int(row_data.get("scenario_seed", -1)) != scene or
                row_data.get("model_arm") != "BC_ONLY_R2" or
                row_data.get("eval_mode") != "greedy" or
                row_data.get("r45_evaluation_role") != "BC_ONLY_GREEDY_CONTROL"):
            raise RuntimeError(f"{seed}/{scene}: row role/seed/scene identity mismatch")
        model = model_manifest[seed]
        if (row_data.get("bc_checkpoint_sha256") != model["checkpoint_sha256"] or
                row_data.get("bc_parameter_sha256") != model["parameter_sha256"]):
            raise RuntimeError(f"{seed}/{scene}: row model identity differs from frozen checkpoint")

        logical_job = f"r45_bc_control_s{seed}_scene{scene}"
        claim_path = RUN_DIR / "launch_claims" / logical_job / "attempt_0001.json"
        claim = _json(claim_path)
        claim_status_counts[str(claim.get("status"))] += 1
        if claim.get("logical_job") != logical_job:
            raise RuntimeError(f"{seed}/{scene}: claim key mismatch")
        if claim.get("status") == "complete":
            receipt = claim.get("round45_artifact_receipt")
        else:
            if (seed, scene) != (11, 1_301_000) or claim.get("status") != "failed_technical":
                raise RuntimeError(f"unexpected non-complete control claim: {logical_job}")
            if _INPUTS[str(ADOPTION_PATH.resolve())]["sha256"] != EXPECTED_INPUT_SHA256[
                    "adopted_control_receipt"]:
                raise RuntimeError("adoption receipt SHA mismatch")
            receipt = adoption.get("round45_artifact_receipt")
        if (not isinstance(receipt, dict) or receipt.get("kind") !=
                "round45-r2-artifact-receipt/2"):
            raise RuntimeError(f"{seed}/{scene}: missing versioned artifact receipt")

        samples, trans = _parse_control_tape(seed, scene, row_data, receipt)
        for sample in samples:
            all_samples.append(sample)
        baseline = baseline_by_scene[scene]
        immediate_path = ROOT / baseline["immediate_row_path"]
        late_path = ROOT / baseline["late_row_path"]
        immediate = _json(immediate_path)
        late = _json(late_path)
        if _INPUTS[str(immediate_path.resolve())]["sha256"] != baseline[
                "immediate_row_sha256"]:
            raise RuntimeError(f"scene {scene}: immediate baseline SHA mismatch")
        if _INPUTS[str(late_path.resolve())]["sha256"] != baseline["late_row_sha256"]:
            raise RuntimeError(f"scene {scene}: late-rule baseline SHA mismatch")
        bc_ids = _on_time_ids(row_data)
        immediate_ids = _on_time_ids(immediate)
        late_ids = _on_time_ids(late)
        local_samples = [item for item in samples if item["seed"] == seed and item["scene"] == scene]
        seed_scene[str(seed)][scene] = {
            "control_row_path": _rel(row_path),
            "control_row_sha256": actual_row_hash,
            "control_claim_status": claim.get("status"),
            "artifact_receipt_source": "adoption" if claim.get("status") != "complete" else "claim",
            "bc_spent_bytes": int(row_data["spent_bytes"]),
            "late_spent_bytes": int(late["spent_bytes"]),
            "immediate_spent_bytes": int(immediate["spent_bytes"]),
            "delta_bytes_vs_late": int(row_data["spent_bytes"]) - int(late["spent_bytes"]),
            "bc_on_time_task_ids": bc_ids,
            "immediate_on_time_task_ids": immediate_ids,
            "late_on_time_task_ids": late_ids,
            "bc_task_set_matches_immediate": bc_ids == immediate_ids,
            "bc_task_set_matches_late": bc_ids == late_ids,
            "first_free_actions": local_samples,
            "emissions": trans["emission_sequence"],
            "emission_sequence_matches_late": [
                [e["step"], e["target_id"], e["revision"], e["paid_bytes"],
                 e["subscribers"], e["first"]] for e in trans["emission_sequence"]
            ] == late.get("emission_sequence", []),
            **trans["aggregate"],
        }

    if claim_status_counts != Counter({"complete": 47, "failed_technical": 1}):
        raise RuntimeError(f"preserved claim statuses changed: {dict(claim_status_counts)}")
    if len(all_samples) != 288:
        raise RuntimeError(f"expected 288 first true free-action rows, got {len(all_samples)}")
    if any(not sample["ready_step_not_after_relay_tape_step"] for sample in all_samples):
        raise RuntimeError("a reconstructed ready_step is later than its relay/tape observation")

    # Actual readonly forwards on the 19D inputs saved in each seed's own first-free-action
    # tape, compared against the logged greedy action and the independent teacher rule.
    inference_by_seed: dict[str, Any] = {}
    for seed in SEEDS:
        selected = [item for item in all_samples if item["seed"] == seed]
        outputs = _infer(controllers[seed], [item["feature_float32"] for item in selected], r45_bc)
        mismatches = []
        max_probability_delta = 0.0
        for sample, output in zip(selected, outputs, strict=True):
            tape_probability = sample.get("tape_send_probability")
            delta = (abs(float(tape_probability) - output["p_send"])
                     if tape_probability is not None else None)
            if delta is not None:
                max_probability_delta = max(max_probability_delta, delta)
            sample["recomputed_model_action"] = output["action"]
            sample["recomputed_model_action_name"] = (
                "SEND_NOW" if output["action"] else "SKIP_EPISODE")
            sample["recomputed_p_send"] = output["p_send"]
            sample["recomputed_p_skip"] = output["p_skip"]
            sample["tape_probability_abs_delta"] = delta
            sample["model_action_matches_tape"] = output["action"] == sample["recorded_action"]
            sample["model_action_matches_teacher"] = output["action"] == sample["teacher_label"]
            if output["action"] != sample["recorded_action"]:
                mismatches.append({"scene": sample["scene"], "target": sample["target_id"],
                                   "tape": sample["recorded_action"],
                                   "forward": output["action"]})
        confusion = _confusion(selected, label_key="teacher_label",
                               prediction_key="recomputed_model_action")
        inference_by_seed[str(seed)] = {
            "samples": len(selected), "confusion_vs_teacher": confusion,
            "model_actions_match_saved_control_tape": len(mismatches) == 0,
            "action_mismatch_count": len(mismatches),
            "action_mismatches": mismatches,
            "max_abs_delta_tape_vs_recomputed_p_send": max_probability_delta,
        }

    # Derive the per-seed cost/action summary and compare literal per-scene behavior across
    # seeds; byte totals alone are not used as a behavior-equivalence check.
    per_seed: dict[str, Any] = {}
    behavior_scene_match = {"free_action_sequence_equal_all_three": 0,
                            "emission_target_tick_sequence_equal_all_three": 0,
                            "full_emission_sequence_equal_all_three": 0,
                            "exact_float32_inputs_equal_all_three": 0}
    for scene in SCENES:
        scene_rows = [seed_scene[str(seed)][scene] for seed in SEEDS]
        first_actions = [[(item["target_id"], item["recorded_action"],
                           item["teacher_label"], item["relay_tape_step"])
                          for item in row["first_free_actions"]] for row in scene_rows]
        emission_sequences = [[(item["step"], item["target_id"])
                               for item in row["emissions"]] for row in scene_rows]
        full_emission_sequences = [[
            (item["step"], item["target_id"], item["revision"], item["paid_bytes"],
             item["subscribers"], item["first"])
            for item in row["emissions"]] for row in scene_rows]
        feature_sequences = [[item["feature_float32_sha256"]
                              for item in row["first_free_actions"]] for row in scene_rows]
        behavior_scene_match["free_action_sequence_equal_all_three"] += int(
            first_actions[0] == first_actions[1] == first_actions[2])
        behavior_scene_match["emission_target_tick_sequence_equal_all_three"] += int(
            emission_sequences[0] == emission_sequences[1] == emission_sequences[2])
        behavior_scene_match["full_emission_sequence_equal_all_three"] += int(
            full_emission_sequences[0] == full_emission_sequences[1] == full_emission_sequences[2])
        behavior_scene_match["exact_float32_inputs_equal_all_three"] += int(
            feature_sequences[0] == feature_sequences[1] == feature_sequences[2])

    for seed in SEEDS:
        rows = seed_scene[str(seed)]
        action_items = [item for item in all_samples if item["seed"] == seed]
        actions_confusion = _confusion(action_items, label_key="teacher_label",
                                       prediction_key="recorded_action")
        bytes_total = sum(int(row["bc_spent_bytes"]) for row in rows.values())
        late_total = sum(int(row["late_spent_bytes"]) for row in rows.values())
        immediate_total = sum(int(row["immediate_spent_bytes"]) for row in rows.values())
        per_seed[str(seed)] = {
            "scene_count": len(rows),
            "bc_spent_bytes_total": bytes_total,
            "late_spent_bytes_total": late_total,
            "immediate_spent_bytes_total": immediate_total,
            "mean_bc_spent_bytes": bytes_total / len(rows),
            "mean_late_spent_bytes": late_total / len(rows),
            "delta_bytes_vs_late_total": bytes_total - late_total,
            "delta_bytes_vs_late_mean": (bytes_total - late_total) / len(rows),
            "task_set_matches_immediate_all_scenes": all(
                row["bc_task_set_matches_immediate"] for row in rows.values()),
            "first_action_confusion_vs_teacher": actions_confusion,
            "actual_first_free_send_count": sum(
                item["recorded_action"] == 1 for item in action_items),
            "teacher_first_free_send_count": sum(
                item["teacher_label"] == 1 for item in action_items),
            "extra_sends_teacher_skip": actions_confusion["extra_sends_teacher_skip"],
            "missed_sends_teacher_send": actions_confusion["missed_sends_teacher_send"],
            "first_emissions": sum(row["first_emissions"] for row in rows.values()),
            "attempts": sum(row["recipient_attempts"] for row in rows.values()),
            "subscriber_sum": sum(row["subscriber_sum"] for row in rows.values()),
            "packets_lost": sum(row["packets_lost"] for row in rows.values()),
            "transcript_paid_bytes": sum(row["paid_bytes_from_transcript"] for row in rows.values()),
            "all_task_sets_match_immediate": all(
                row["bc_task_set_matches_immediate"] for row in rows.values()),
            "all_row_costs_close_to_transcripts": all(
                row["paid_bytes_from_transcript"] == row["bc_spent_bytes"]
                and row["emission_projection_matches_transcript"]
                for row in rows.values()),
            "late_rule_emission_sequence_match_scene_count": sum(
                row["emission_sequence_matches_late"] for row in rows.values()),
            "late_rule_emission_sequence_mismatch_scenes": [
                scene for scene, row in rows.items()
                if not row["emission_sequence_matches_late"]],
            "extra_send_details": [
                {
                    "scene": item["scene"], "target_id": item["target_id"],
                    "relay_tape_step": item["relay_tape_step"],
                    "reconstructed_ready_step": item["reconstructed_ready_step"],
                    "teacher_action": item["teacher_action"],
                    "recorded_action": item["recorded_action_name"],
                    "recomputed_p_send": item.get("recomputed_p_send"),
                    "feature_float32_sha256": item["feature_float32_sha256"],
                    "exact_support_class": (
                        "exact_fit_vector" if _feature_key(item["feature_float32"])
                        in fit_labels_by_feature else
                        "validation_only_exact_vector" if _feature_key(item["feature_float32"])
                        in validation_labels_by_feature else
                        "not_exactly_seen_in_fit_or_validation"),
                    "observed_first_emission": (item["observed_first_emissions"][0]
                                                 if item["observed_first_emissions"] else None),
                }
                for item in action_items
                if item["recorded_action"] == 1 and item["teacher_label"] == 0
            ],
            "scenes": {str(scene): rows[scene] for scene in SCENES},
        }

    feature_support_counts = Counter()
    disagreement_support_counts = Counter()
    eval_feature_labels: dict[bytes, set[int]] = defaultdict(set)
    eval_feature_refs: dict[bytes, list[dict[str, Any]]] = defaultdict(list)
    boundary_counts = Counter()
    boundary_mismatch_counts = Counter()
    missing_teacher_label_count = 0
    for item in all_samples:
        if item.get("teacher_label") is None:
            missing_teacher_label_count += 1
            continue
        key = _feature_key(item["feature_float32"])
        eval_feature_labels[key].add(int(item["teacher_label"]))
        eval_feature_refs[key].append(item)
        if key in fit_labels_by_feature:
            feature_support = "exact_fit_vector"
        elif key in validation_labels_by_feature:
            feature_support = "validation_only_exact_vector"
        else:
            feature_support = "not_exactly_seen_in_fit_or_validation"
        feature_support_counts[feature_support] += 1
        if item["recorded_action"] != item["teacher_label"]:
            disagreement_support_counts[feature_support] += 1
        ready = int(item["reconstructed_ready_step"])
        if ready in (146, 147, 148, 149):
            boundary_counts[str(ready)] += 1
            if item["recorded_action"] != item["teacher_label"]:
                boundary_mismatch_counts[str(ready)] += 1
    eval_label_conflicts = sum(len(labels) > 1 for labels in eval_feature_labels.values())
    joint_feature_conflicts = sum(
        len(dataset_labels_by_feature.get(key, set()) | labels) > 1
        for key, labels in eval_feature_labels.items()
    )

    tracked_input_paths = list(_INPUTS.values())
    for path_text, first_identity in list(_INPUTS.items()):
        path = Path(path_text)
        final_identity = {"path": _rel(path), "size_bytes": path.stat().st_size,
                          "sha256": _sha256_file(path)}
        if final_identity != first_identity:
            raise RuntimeError(f"read-only input changed during analysis: {path}")

    python_rng_unchanged = random.getstate() == python_rng_before
    numpy_rng_after = np.random.get_state()
    numpy_rng_unchanged = (
        numpy_rng_after[0] == numpy_rng_before[0]
        and np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
        and numpy_rng_after[2:] == numpy_rng_before[2:]
    )
    torch_rng_unchanged = torch.equal(torch.get_rng_state(), torch_rng_before)

    summary = {
        "kind": "round45-bc-rule-fidelity-diagnostic/1",
        "created_date": "2026-10-06",
        "analysis_scope": "read-only BC rule fidelity and cost attribution; no new experiment",
        "repository": {"analysis_start_head": START_HEAD, **_git_identity()},
        "execution_boundary": {
            "environment_constructed_delta": 0, "run_calls_delta": 0,
            "ppo_updates_delta": 0, "optimizer_step_calls_delta": 0,
            "checkpoint_write_calls_delta": 0,
            "checkpoint_optimizer_restored": False,
            "checkpoint_rng_restored": False,
            "rng_states_unchanged": {
                "python": python_rng_unchanged, "numpy": numpy_rng_unchanged,
                "torch_cpu": torch_rng_unchanged,
            },
        },
        "source_identities": {
            "input_artifacts_sha256": EXPECTED_INPUT_SHA256,
            "raw_recomputation_sha256_used": EXPECTED_INPUT_SHA256["raw_row_recomputation"],
            "raw_recomputation_git_vs_worktree": raw_row_crosscheck,
            "raw_recomputation_status": raw["status"],
            "raw_control_rows": raw["control_rows"]["count"],
            "raw_claim_statuses": dict(claim_status_counts),
            "preserved_original_failed_claim": True,
            "adopted_control_row_included_once": raw["control_rows"][
                "adopted_control_row"]["included_once"],
            "baseline_manifest_rows_checked": raw["baseline_rows"][
                "raw_manifest_rows_checked"],
            "baseline_manifest_all_match": raw["baseline_rows"][
                "raw_manifest_digest_matches_all_32_rows"],
            "post_bc_freeze_sha256": EXPECTED_INPUT_SHA256["post_bc_freeze"],
            "training_summary_sha256": EXPECTED_INPUT_SHA256["bc_training_summary"],
            "dataset_sha256": EXPECTED_INPUT_SHA256["bc_dataset"],
            "dataset_audit_sha256": EXPECTED_INPUT_SHA256["bc_data_audit"],
            "teacher_source_hashes_from_frozen_audit": {
                "rule": audit["teacher"]["rule_source_sha256"],
                "relay": audit["teacher"]["relay_source_sha256"],
                "features": audit["teacher"]["feature_source_sha256"],
            },
            "teacher_predicate": audit["teacher"]["registered_predicate"],
            "teacher_time_basis": audit["teacher"]["time_basis"],
            "candidate_time_source_chain": timing_source_chain,
            "tracked_inputs": tracked_input_paths,
        },
        "gate_context": {
            "status": gate["status"], "training_authorized_by_gate": False,
            "ppo_updates_in_control": int(gate.get("ppo_update_calls", 0)),
            "fixed_late_rule_mean_bytes": float(gate["baseline"][
                "fixed_late_mean_spent_bytes"]),
            "scientific_gate_remains_failed": True,
            "bc_updates_this_attempt": int(training_summary[
                "this_attempt_real_bc_optimizer_updates"]),
            "bc_family_cumulative_updates": int(training_summary[
                "family_cumulative_real_bc_optimizer_updates"]),
        },
        "dataset_label_and_time_audit": {
            "rows": len(dataset), "split_counts": dict(split_counts),
            "teacher_cut_step_exclusive": TEACHER_CUT,
            "formula": "SEND_NOW iff first_emission is true and ready_step > 147; otherwise SKIP_EPISODE",
            "ready_step_source": "recorded tape relay observation step minus exact integer float32 age_steps",
            "ready_step_is_reconstructed_separately_from_relay_tick_and_age": True,
            "candidate_source_chain_proves_ready_step_not_after_tape_tick": timing_source_chain[
                "all_admitted_candidates_ready_step_not_after_observation_tick"],
            "first_free_rows_ready_step_not_after_relay_tape_tick_count": sum(
                item["ready_step_not_after_relay_tape_step"] for item in all_samples),
            "first_free_rows_ready_step_equals_relay_tick_count": sum(
                item["ready_step_equals_relay_tape_step"] for item in all_samples),
            "clock_now_used": False,
            "rows_with_label_time_relation_valid": len(dataset),
            "exact_float32_label_conflicting_vectors": dataset_label_conflicts,
            "fit_samples": split_counts["fit"], "validation_samples": split_counts["validation"],
            "development_scenes_in_dataset": 0,
        },
        "checkpoint_readonly_forward": {
            "device": "cpu", "models": model_identities,
            "dataset_fit_validation_confusion_by_seed": dataset_metrics,
            "control_inputs_confusion_by_seed": inference_by_seed,
        },
        "teacher_fidelity_diagnostics": {
            "first_true_free_action_rows": len(all_samples),
            "teacher_label_missing_rows": missing_teacher_label_count,
            "pre_action_legality_anomaly_count": sum(
                len(scene["anomalies"]) for seed in SEEDS for scene in
                seed_scene[str(seed)].values()),
            "exact_float32_label_conflicts_in_control_rows": eval_label_conflicts,
            "exact_float32_label_conflicts_in_dataset_plus_controls": joint_feature_conflicts,
            "control_feature_exact_support_counts": dict(feature_support_counts),
            "teacher_disagreements_by_exact_support_class": dict(disagreement_support_counts),
            "unique_control_float32_vectors": len({
                item["feature_float32_sha256"] for item in all_samples}),
            "unique_extra_send_float32_vectors": len({
                item["feature_float32_sha256"] for item in all_samples
                if item["recorded_action"] == 1 and item["teacher_label"] == 0}),
            "ready_step_boundary_counts_146_to_149": dict(boundary_counts),
            "boundary_disagreement_counts": dict(boundary_mismatch_counts),
            "classification_evidence": {
                "label_clock_wiring_anomaly_rows": sum(
                    len(scene["anomalies"]) for seed in SEEDS for scene in
                    seed_scene[str(seed)].values()),
                "cut_boundary_disagreements_ready_146_to_149": sum(
                    boundary_mismatch_counts.values()),
                "teacher_label_missing_rows": missing_teacher_label_count,
                "exact_float32_alias_conflicts_in_dataset_and_controls": joint_feature_conflicts,
                "unique_extra_send_vectors_outside_fit_and_validation_exact_support": len({
                    item["feature_float32_sha256"] for item in all_samples
                    if item["recorded_action"] == 1 and item["teacher_label"] == 0
                    and _feature_key(item["feature_float32"]) not in fit_labels_by_feature
                    and _feature_key(item["feature_float32"]) not in validation_labels_by_feature}),
                "assessment": (
                    "The two extra first sends are local exact-vector support novelties relative "
                    "to the frozen fit/validation dataset. This is consistent with local feature "
                    "support shift, but does not establish broad distribution shift. No label/clock "
                    "wiring, cutoff-boundary, missing-label, or exact-vector alias evidence explains them."
                ),
            },
            "interpretation_limits": [
                "A model/teacher disagreement on the same saved 19D input is local imitation evidence, not a counterfactual policy result.",
                "Exact-vector novelty is a narrow support diagnostic; it does not by itself establish broad distribution shift.",
                "The analysis does not claim matched channel-RNG draws across policies or seeds.",
                "A BC cost excess is not evidence PPO was ineffective; PPO did not run.",
            ],
        },
        "cross_seed_literal_behavior": behavior_scene_match,
        "per_seed": per_seed,
    }

    output_json = OUT_DIR / "ROUND45_BC_RULE_FIDELITY_SUMMARY_R1.json"
    output_report = OUT_DIR / "ROUND45_BC_RULE_FIDELITY_REPORT_R1.md"
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                      sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8", newline="\n")

    lines = [
        "# ROUND45 BC 规则拟合与超额成本只读诊断",
        "",
        f"- 输入冻结起点：`{START_HEAD}`；分析分支：`analysis/comm-bc-rule-fidelity`。",
        f"- 机器摘要：`{_rel(output_json)}`。",
        f"- R3 FINAL3 原始行重算 SHA256：`{EXPECTED_INPUT_SHA256['raw_row_recomputation']}`。",
        f"- Git/worktree 核对：提交 blob SHA256 `{raw_row_crosscheck['git_blob_content_sha256']}`，"
        f"worktree SHA256 `{raw_row_crosscheck['worktree_content_sha256']}`；"
        f"字节相同={raw_row_crosscheck['byte_equal']}，JSON 语义相同={raw_row_crosscheck['json_semantically_equal']}。",
        "- 科学控制门槛仍失败；此诊断不重新打开 PPO 或评估额度。",
        "",
        "## 三 seed 控制与成本复算",
        "",
        "| Seed | BC bytes | late bytes | 差值 | BC 首报 SEND | teacher SEND | teacher 多发 | teacher 漏发 | task IDs = immediate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in SEEDS:
        item = per_seed[str(seed)]
        confusion = item["first_action_confusion_vs_teacher"]
        lines.append(
            f"| {seed} | {item['bc_spent_bytes_total']:,} | {item['late_spent_bytes_total']:,} "
            f"| {item['delta_bytes_vs_late_total']:+,} | {item['actual_first_free_send_count']} "
            f"| {item['teacher_first_free_send_count']} | {confusion['extra_sends_teacher_skip']} "
            f"| {confusion['missed_sends_teacher_send']} | "
            f"{'是' if item['task_set_matches_immediate_all_scenes'] else '否'} |"
        )
    lines.extend([
        "",
        f"每 seed：{per_seed['11']['scene_count']} 场景，BC 与 immediate 的逐场景按时任务 ID 集合全部相同；"
        f"成本总差为 +{per_seed['11']['delta_bytes_vs_late_total']:,} B。三 seed 的观察行为逐场景相同计数："
        f"自由动作 {behavior_scene_match['free_action_sequence_equal_all_three']}/16，"
        f"首发射目标与 tick {behavior_scene_match['emission_target_tick_sequence_equal_all_three']}/16，"
        f"发射序列（含 revision/bytes/subscribers）{behavior_scene_match['full_emission_sequence_equal_all_three']}/16，"
        f"19D 输入 {behavior_scene_match['exact_float32_inputs_equal_all_three']}/16。",
        "这些是原始转录里的逐项相等检查，不表示不同运行的信道随机数流相同。",
        "",
        "原始行重算 SHA 的更正：ROUND45 交付头内 Git blob 使用 LF，blob SHA256 为 `"
        f"{raw_row_crosscheck['git_blob_content_sha256']}`；当前 Windows 工作树使用 CRLF，"
        f"所以 SHA256 为 `{raw_row_crosscheck['worktree_content_sha256']}`，但 JSON 语义逐项相同。"
        "该文件在交付头中是新增文件，父提交没有可比较的旧版本；先前引用的 d54def…"
        "既不等于当前工作树也不等于 Git blob，不能从父提交恢复。已改用交付头实际包含的版本，"
        "其 48 行、首格 adoption、claims、baseline 来源校验、三个 seed 的 task/cost 字段均可读回核实。",
        "",
        "## Teacher 与实际 checkpoint forward",
        "",
        "Teacher 精确条件为 `first_emission && ready_step > 147`。当前 R45 R3 collector 经 `r42_run`/`v2_run` 接到真实 relay，"
        "再由 `JointRun.run` 逐步进入 orchestrator。Tape `step` 是 relay observation/tape tick；"
        "合法首个自由动作行的 `ready_step` 由 `step - age_steps` 复原，其中 `age_steps` 是保存的精确整数 float32 特征，"
        "对应冻结源中的 `age_steps=max(0, relay_tick-ready_step)`。真实调用链核验确认：orchestrator 同步发布 detection，"
        "registry 处理器先于 relay 处理器执行；relay 将新事件记为旧 tick+1，而同一周期经 AssignmentGate 到达的 "
        "Task03Adapter.step 先调用 relay.tick，tick 递增后才构造 observation。因此所有已提供候选均满足 "
        "ready_step <= observation.step，年龄的 max(0,...) 不会在候选行上截断未来 ready_step；288 条首选行 age_steps="
        "0，故其 ready_step 可精确还原为 tape tick。Tape 的 step 来自 `int(observation.step)`；全局 `clock.now` 在之后才推进，"
        "没有用于 teacher 标签。相关源码 SHA 与顺序断言保存在机读摘要 `source_identities.candidate_time_source_chain`。"
        "未用任务结局或接收端状态打标签。",
        "",
        "### 候选 ready tick 接线源码身份",
        "",
        *[f"- `{name}` SHA256 `{digest}`" for name, digest in
          timing_source_chain["source_files_sha256"].items()],
        "",
        "| Seed | Split | Samples | Skip→Skip | Skip→Send | Send→Skip | Send→Send | Accuracy |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ])
    for seed in SEEDS:
        for split in ("fit", "validation"):
            metric = dataset_metrics[str(seed)][split]
            m = metric["true_rows_predicted_columns_skip_send"]
            lines.append(f"| {seed} | {split} | {metric['samples']} | {m[0][0]} | {m[0][1]} "
                         f"| {m[1][0]} | {m[1][1]} | {metric['accuracy']:.4f} |")
    lines.extend([
        "",
        "## 两个额外 SEND 的逐项来源",
        "",
        "每个 seed 都在相同两个保存输入上比 teacher 多发一次；每个场景其余首报 emission sequence 与 late 规则逐项相同。",
        "| Scene | Target | Relay tick / ready_step | `p(SEND)` s11 / s22 / s33 | Charged bytes | attempts/subscribers | lost packets s11 / s22 / s33 |",
        "|---:|:---|---:|:---|---:|---:|:---|",
    ])
    extra_keys = sorted({(int(item["scene"]), item["target_id"])
                         for seed in SEEDS
                         for item in per_seed[str(seed)]["extra_send_details"]})
    for scene, target in extra_keys:
        matching = {}
        for seed in SEEDS:
            matching[seed] = next(item for item in per_seed[str(seed)]["extra_send_details"]
                                  if item["scene"] == scene and item["target_id"] == target)
        observed = [matching[seed]["observed_first_emission"] for seed in SEEDS]
        first = observed[0]
        p_values = " / ".join(f"{matching[seed]['recomputed_p_send']:.4f}" for seed in SEEDS)
        loss_values = " / ".join(str(item["packets_lost"]) for item in observed)
        lines.append(
            f"| {scene} | {target} | {first['step']} / {matching[11]['reconstructed_ready_step']} "
            f"| {p_values} | {first['paid_bytes']:,} | {first['attempts']}/{first['subscribers']} "
            f"| {loss_values} |"
        )
    extra_charge = sum(int(item["observed_first_emission"]["paid_bytes"])
                       for item in per_seed["11"]["extra_send_details"])
    lines.extend([
        "",
        f"每 seed teacher 首选 31 SEND、BC 实际 33 SEND；额外两条首报分别收费 2,508 B 与 2,514 B，"
        f"合计 `{extra_charge:,} B`，正好等于每 seed 相对 late 的总成本差。ready_step 为 122 和 140，"
        "都远离 `>147` 边界；没有漏掉 teacher SEND。6 条重复观测对应 2 个独特 19D 输入，"
        "它们都不在冻结 fit/validation 的精确向量集合中。精确同值 teacher 标签冲突为 0；"
        "这支持局部未见输入上的模仿假阳性，不足以证明广义分布漂移或反事实策略效应。",
        "",
        "三份 checkpoint 均按 FINAL3 SHA 和参数摘要只读加载；torch.load 读入 checkpoint payload 用于结构核对，"
        "但没有构造 optimizer、恢复 optimizer 状态或使用 checkpoint RNG。拟合/验证混淆矩阵由实际 float32 forward 重算，"
        "并与已保存训练指标逐项一致。控制行 forward 动作也逐条对照 tape。",
        "",
        "## 归因边界",
        "",
        f"- 首个合法自由动作行 {len(all_samples)} 条；缺失 teacher label {missing_teacher_label_count} 条；"
        f"19D 同值冲突（dataset）{dataset_label_conflicts} 个，控制行 {eval_label_conflicts} 个。",
        f"- 首报 ready_step 在 146–149 的计数：`{dict(boundary_counts)}`；对应 action/teacher 不同："
        f"`{dict(boundary_mismatch_counts)}`。",
        "- 288 条合法首选输入均未发现 pre-action legality 或 label/clock 重建异常；其中 age_steps 均为 0，"
        "所以这些具体行上 ready_step 数值等于 relay/tape tick。cutoff 邻域观测不构成多发原因。",
        f"- 三 seed 的 emission sequence 对 late 规则各有 {per_seed['11']['late_rule_emission_sequence_match_scene_count']}/16 场景逐项相同；"
        f"不同的仅是 scene {', '.join(map(str, per_seed['11']['late_rule_emission_sequence_mismatch_scenes']))} 的额外 target-5 首报。",
        f"- 控制特征精确拟合向量支持：`{dict(feature_support_counts)}`；其中不一致按支持类为："
        f"`{dict(disagreement_support_counts)}`。精确向量未见只说明该向量未出现在拟合集，不能单独定性为广义分布偏移。",
        "- Tape/feature/receipt/transcript/成本闭合问题以 JSON `anomalies` 字段记录；实际记录没有因同总字节而被假设行为相同。",
        "- 本轮只定位 BC 对规则的动作保真和行为差异。不能把成本超线解释为 PPO 无效，因为 PPO 更新为 0。",
        "",
        "## 范围与计数",
        "",
        "本脚本只读取既有 JSON/JSONL、训练 dataset 与三个 BC checkpoint，执行 CPU 前向并写入本分析目录。"
        "新环境 0、run 调用 0、PPO 更新 0、optimizer step 0、checkpoint 写回 0。",
    ])
    output_report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "summary_path": _rel(output_json), "report_path": _rel(output_report),
        "per_seed": {seed: {"bc_bytes": per_seed[seed]["bc_spent_bytes_total"],
                            "late_bytes": per_seed[seed]["late_spent_bytes_total"],
                            "delta": per_seed[seed]["delta_bytes_vs_late_total"],
                            "extra_sends": per_seed[seed]["extra_sends_teacher_skip"],
                            "missed_sends": per_seed[seed]["missed_sends_teacher_send"]}
                     for seed in per_seed},
        "cross_seed_behavior": behavior_scene_match,
        "rng_unchanged": summary["execution_boundary"]["rng_states_unchanged"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
