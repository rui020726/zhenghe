"""ROUND75 qualification replay adapter: frozen R58 row+tape -> ``batch_samples`` episode.

Pure Python (no torch import). It rebuilds the episode object ``v2_trainer.batch_samples``
expects from the two frozen artefacts of one recorded R58 rollout episode:

- the tape (``*_tape.jsonl``): per-decision-step records with the SAME fields the real collector
  produced (``step``, ``features`` as per-candidate 19-d rows, ``targets``, ``actions``,
  ``actionable``, ``log_prob``, ``value``, ``cost_value``);
- the row (``*_a1.json``): the episode wrapper (``steps``, ``cost_stream`` with ``airtime_bytes``
  per tick, ``task_stream``, ``lambda_at_rollout``, ``episode_index``).

Nothing is truncated, resampled, rewritten or backfilled: the recorded behaviour log-probs are
used exactly as recorded, and ``likelihood_alignment`` stays enabled. The adapter only validates
and re-keys. This is a QUALIFICATION data path over R58-lineage rollouts; it is not an R71
trajectory and not research training data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

TAPE_SCHEMA = "round42-one-shot-send-skip-tape/1"
FEATURE_DIM = 19
QUALIFICATION_SAMPLE_CAP = 256          # one minibatch per epoch at MINIBATCH_STEPS=256
REQUIRED_TAPE_KEYS = ("step", "features", "targets", "actions", "actionable",
                      "log_prob", "value", "cost_value")
REQUIRED_ROW_KEYS = ("steps", "cost_stream", "task_stream", "lambda_at_rollout",
                     "episode_index", "scenario_seed", "training_seed", "behaviour_sha256")


class AdapterError(RuntimeError):
    pass


class _Collector:
    """Minimal stand-in exposing ``.records`` exactly like the real collector."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tape(tape_path) -> list[dict[str, Any]]:
    records = []
    with open(tape_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def episode_from_row_tape(row: dict[str, Any], tape_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the ``batch_samples`` episode contract. Re-keying only; no value is altered."""
    cost_stream = {}
    for step, entry in row["cost_stream"].items():
        cost_stream[int(step)] = {"airtime_bytes": float(entry["airtime_bytes"])}
    task_stream = {int(step): float(value) for step, value in row["task_stream"].items()}
    return {"collector": _Collector([dict(record) for record in tape_records]),
            "steps": int(row["steps"]),
            "cost_stream": cost_stream,
            "task_stream": task_stream,
            "lambda": float(row["lambda_at_rollout"]),
            "episode_index": int(row["episode_index"])}


def validate_episode(episode: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on any schema/sample-budget problem; return the episode facts."""
    records = episode["collector"].records
    if not records:
        raise AdapterError("empty tape: no decision step")
    for index, record in enumerate(records):
        missing = [key for key in REQUIRED_TAPE_KEYS if key not in record]
        if missing:
            raise AdapterError(f"tape record {index} missing keys {missing}")
        if record.get("schema") != TAPE_SCHEMA:
            raise AdapterError(f"tape record {index} schema {record.get('schema')!r}")
        for feature_row in record["features"]:
            if len(feature_row) != FEATURE_DIM:
                raise AdapterError(
                    f"tape record {index}: feature row dim {len(feature_row)} != {FEATURE_DIM}")
    steps = [int(record["step"]) for record in records]
    if steps != sorted(steps):
        raise AdapterError("tape steps are not monotonic")
    if len(records) > QUALIFICATION_SAMPLE_CAP:
        raise AdapterError(
            f"batch has {len(records)} samples > qualification cap {QUALIFICATION_SAMPLE_CAP}")
    if int(episode["steps"]) <= 0:
        raise AdapterError("episode horizon must be positive")
    if not episode["cost_stream"] or not episode["task_stream"]:
        raise AdapterError("cost/task stream missing")
    early = late = gate_eligible = 0
    for record in records:
        for feature_row, target in zip(record["features"], record["targets"]):
            if target not in record["actionable"]:
                continue
            ready = int(round(float(feature_row[18]) * 1000.0) - round(float(feature_row[0])))
            if ready <= 147:
                early += 1
                if float(feature_row[5]) == 1.0:
                    gate_eligible += 1
            else:
                late += 1
    return {"records": len(records), "candidate_rows": sum(len(r["features"]) for r in records),
            "first_step": steps[0], "last_step": steps[-1],
            "free_early_rows": early, "free_late_rows": late,
            "gate_eligible_early_rows": gate_eligible,
            "horizon": int(episode["steps"]), "lambda": float(episode["lambda"]),
            "episode_index": int(episode["episode_index"])}


def load_qualification_batch(row_path, tape_path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load + validate one frozen recorded episode; return (episode, provenance facts)."""
    row_path, tape_path = Path(row_path), Path(tape_path)
    row = json.loads(row_path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_ROW_KEYS if key not in row]
    if missing:
        raise AdapterError(f"row missing keys {missing}")
    records = load_tape(tape_path)
    episode = episode_from_row_tape(row, records)
    facts = validate_episode(episode)
    facts.update({
        "row_path": str(row_path), "row_sha256": sha256_file(row_path),
        "tape_path": str(tape_path), "tape_sha256": sha256_file(tape_path),
        "scenario_seed": int(row["scenario_seed"]), "training_seed": int(row["training_seed"]),
        "behaviour_sha256": row["behaviour_sha256"],
        "behaviour_parameter_sha256": row["controller_parameters_after"]["parameter_sha256"],
        "behaviour_parameter_count": row["controller_parameters_after"]["parameter_count"],
        "behaviour_heads": row["controller_parameters_after"]["heads"],
        "training_version": row["training_version"],
        "policy": row["policy"], "decoder": row["collector_summary"]["timing"]["decoder"],
        "substituted_fields": row.get("substituted_fields", []),
        "budget_bytes": int(row["budget_bytes"]), "b0_bytes": float(row["b0_bytes"]),
        "on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
    })
    return episode, facts