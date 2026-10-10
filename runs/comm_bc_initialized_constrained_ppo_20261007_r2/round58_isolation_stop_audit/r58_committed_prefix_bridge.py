"""Hash-closed, zero-environment handoff for a completed, uncommitted R58 batch prefix.

This is a private recovery adapter, not a rollout or PPO implementation.  It
only reads already complete episode artifacts after an exact batch commit.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import r58_uncommitted_prefix_probe as prior_probe


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def episode(claim_root: Path, source_root: Path, seed: int, ep: int) -> tuple[dict, list, dict]:
    """Select one complete attempt, preserving every failed predecessor."""
    folder = Path(claim_root) / "launch_claims" / f"new_training_s{seed}_e{ep}"
    attempts = sorted(folder.glob("attempt_*.json"))
    if not attempts:
        raise FileNotFoundError(folder)
    entries = [read(path) for path in attempts]
    complete = [entry for entry in entries if entry.get("status") == "complete"]
    if (len(complete) != 1 or any(entry is not complete[0] and
                                   entry.get("status") != "failed_technical" for entry in entries)):
        raise RuntimeError(f"episode {ep}: attempt history is not uniquely complete")
    claim = complete[0]
    if (claim.get("logical_job") != f"new_training_s{seed}_e{ep}"
            or int(claim.get("run_return_calls", -1)) != 1
            or int(claim.get("completed_steps", -1)) != 1000):
        raise RuntimeError(f"episode {ep}: completed claim counters differ")
    receipt = claim.get("round45_artifact_receipt", {})
    paths = claim.get("round45_artifact_paths", {})
    for key in ("row", "tape", "lifecycle", "transcript"):
        path = Path(paths[key]).resolve()
        parent = (Path(source_root) / ("rows" if key == "row" else "logs")).resolve()
        ref = receipt.get("artifacts", {}).get(key, {})
        if (path.parent != parent or ref.get("path") != str(path)
                or ref.get("sha256") != digest(path) or ref.get("size_bytes") != path.stat().st_size):
            raise RuntimeError(f"episode {ep}: {key} source bytes are not receipt-bound")
    row = read(Path(paths["row"]))
    tape = [json.loads(line) for line in Path(paths["tape"]).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if (int(row.get("training_seed", -1)) != seed or int(row.get("episode_index", -1)) != ep
            or int(row.get("scenario_seed", -1)) != int(claim["metadata"]["scenario_seed"])
            or int(row.get("steps", -1)) != 1000 or len(tape) != receipt["decision_tape_records"]
            or any(int(a["step"]) >= int(b["step"]) for a, b in zip(tape, tape[1:]))
            or float(sum(float(v["airtime_bytes"]) for v in row["cost_stream"].values()))
               != float(row["spent_bytes"])
            or float(sum(float(v) for v in row["task_stream"].values()))
               != float(row["task_reward_total"])):
        raise RuntimeError(f"episode {ep}: row/tape input mismatch")
    return row, tape, claim


def materialize(controller: Any, *, seed: int, claim_root: Path,
                source_root: Path, source_checkpoint: Path,
                validate_current_source: Any) -> dict[str, Any]:
    """Validate a committed checkpoint and its contiguous complete prefix.

    The runtime passes its actual selector result.  A local commit and record,
    not just a checkpoint sidecar, must close the source before any prefix is
    accepted.  No RNG is left changed by this qualification step.
    """
    if seed != 11:
        return {"rows": {}, "last_episode": None, "source": None}
    source_root = Path(source_root).resolve()
    selected = Path(source_checkpoint).resolve()
    if selected.parent != (source_root / "checkpoints").resolve():
        raise RuntimeError("current prefix source is outside the resume checkpoint root")
    source = validate_current_source(selected, seed)
    payload = torch.load(selected, map_location="cpu", weights_only=False)
    state = payload["training_state"]
    cursor = int(state["episodes_done"])
    updates = int(state["updates"])
    if (source.get("sha256") != digest(selected) or cursor != 4 * updates
            or source.get("batch") != updates - 1):
        raise RuntimeError("prefix source differs from the committed cursor")
    commit_path = source_root / "checkpoints" / f"r40_batch_commit_s{seed}_b{updates - 1}.json"
    if not commit_path.is_file() or source.get("commit_sha256") != digest(commit_path):
        raise RuntimeError("prefix source has no exact local durable batch commit")
    commit = read(commit_path)
    record = source_root / "checkpoints" / commit["batch_record"]
    if (digest(record) != commit["batch_record_sha256"]
            or int(commit["episodes_done"]) != cursor
            or int(commit["optimizer_steps"]) != int(state["optimizer_steps"])
            or controller.parameter_summary()["parameter_sha256"]
               != commit["controller_parameters"]["parameter_sha256"]):
        raise RuntimeError("prefix source controller/record/commit do not close")
    rows: dict[int, dict] = {}
    tapes: dict[int, list] = {}
    for ep in range(cursor, cursor + 4):
        folder = Path(claim_root) / "launch_claims" / f"new_training_s{seed}_e{ep}"
        if not folder.exists():
            break
        row, tape, _ = episode(claim_root, source_root, seed, ep)
        if (float(row["lambda"]) != float(state["lambda"])
                or row["controller_parameters_after"]["parameter_sha256"]
                   != commit["controller_parameters"]["parameter_sha256"]):
            raise RuntimeError(f"prefix episode {ep} weights/lambda differ from source")
        rows[ep], tapes[ep] = row, tape
    if not rows:
        return {"rows": {}, "last_episode": None, "source": source}
    if list(rows) != list(range(cursor, cursor + len(rows))):
        raise RuntimeError("persisted prefix has a gap")
    trainer = prior_probe._frozen_trainer()
    original = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    common_before = {key: getattr(trainer.common, key)
                     for key in ("GAE_LAMBDA", "GAMMA", "B0_BYTES")}
    try:
        for setting, key in (("GAE_LAMBDA", "gae_lambda"), ("GAMMA", "gamma"),
                             ("B0_BYTES", "b0_bytes")):
            if float(getattr(trainer.common, setting)) != float(state[key]):
                raise RuntimeError(f"live {setting} differs from committed checkpoint")
        previous_checkpoint = (source_root / "checkpoints"
                               / f"v2_train_s{seed}_after_b{updates - 2}.pt")
        previous_commit_path = (source_root / "checkpoints"
                                / f"r40_batch_commit_s{seed}_b{updates - 2}.json")
        if not previous_checkpoint.is_file() or not previous_commit_path.is_file():
            raise RuntimeError("independent pre-commit RNG control source is missing")
        previous_commit = read(previous_commit_path)
        if digest(previous_checkpoint) != previous_commit["postupdate_checkpoint_sha256"]:
            raise RuntimeError("independent pre-commit RNG control bytes changed")
        previous_payload = torch.load(previous_checkpoint, map_location="cpu", weights_only=False)
        previous_controller = trainer._restore_controller(
            previous_payload, previous_payload["training_state"])
        control_row, control_tape, _ = episode(claim_root, source_root, seed, cursor - 1)
        control = prior_probe._replay_episode_rng(
            control_row, control_tape, previous_controller)
        if not prior_probe._same_state(control["rng"], payload["rng"]):
            raise RuntimeError("independent previous-episode RNG control differs from commit")
        replay = {ep: prior_probe._replay_episode_rng(rows[ep], tapes[ep], controller)
                  for ep in rows}
        samples = trainer.batch_samples([
            {**rows[ep], "collector": SimpleNamespace(records=tapes[ep])} for ep in rows])
        if len(samples) != sum(len(tapes[ep]) for ep in rows):
            raise RuntimeError("frozen batch_samples did not close prefix")
        if any(not all(math.isfinite(float(item[key])) for key in
                       ("advantage", "return_task", "return_cost", "log_prob"))
               for item in samples):
            raise RuntimeError("nonfinite prefix sample")
        return {"rows": {ep: {**rows[ep], "collector": SimpleNamespace(records=tapes[ep])}
                         for ep in rows},
                "tapes": tapes, "last_episode": max(rows), "source": source,
                "proof": {"cursor": cursor, "episodes": list(rows),
                          "previous_episode_rng_equals_commit": True,
                          "decision_records": {str(ep): len(tapes[ep]) for ep in rows},
                          "samples": len(samples),
                          "source_commit_sha256": source["commit_sha256"],
                          "replay": {str(ep): {k: v for k, v in replay[ep].items() if k != "rng"}
                                     for ep in rows}}}
    finally:
        random.setstate(original[0])
        np.random.set_state(original[1])
        torch.set_rng_state(original[2])
        for key, value in common_before.items():
            setattr(trainer.common, key, value)


def restore_post_episode_rng(row: dict, tape: list, controller: Any) -> dict[str, Any]:
    """Advance only the already-proven last episode's process RNG streams."""
    return prior_probe._replay_episode_rng(row, tape, controller)
