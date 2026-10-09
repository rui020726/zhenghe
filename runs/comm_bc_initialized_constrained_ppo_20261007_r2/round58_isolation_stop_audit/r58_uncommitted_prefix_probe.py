"""Read-only evidence for the interrupted seed-11 batch-10 rollout prefix.

This module deliberately has no training or environment entry point.  It uses
the frozen batch_samples function and in-memory RNG draws only.  In particular,
it does *not* make the existing resume launcher eligible for execution.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
R2 = HERE.parent
RAW = R2 / "round58_gen3_resume_v18" / "raw"
CLAIMS = R2 / "round58_gen3" / "raw" / "launch_claims"
CP = RAW / "checkpoints"
QUARANTINED_B9 = (HERE / "round58_resume_v18_superseded_identity_quarantine_20261010T0030Z"
                  / "checkpoints" / "v2_train_s11.pt")
B9_ORIGINAL_SHA = "92ad17ca9dfd08e4e1d5a606e76e202aa043ac0d2e8a550cc27e303f7d6533e2"
FROZEN_TRAINER_SHA = "0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frozen_trainer() -> Any:
    canon = REPO.parent / "zhenghe" / "runs"
    for directory in ("comm_budget_pilot_20260919", "comm_cost_stage_b_20260927",
                      "comm_cost_v2_rerun_20260928"):
        source = canon / directory
        if not source.is_dir():
            raise FileNotFoundError(source)
        sys.path.insert(0, str(source))
    import v2_trainer  # imported only after the exact canonical roots are present
    if (Path(v2_trainer.__file__).resolve() != (canon / "comm_cost_v2_rerun_20260928"
                                               / "v2_trainer.py").resolve()
            or sha(Path(v2_trainer.__file__)) != FROZEN_TRAINER_SHA):
        raise RuntimeError("frozen trainer import resolved to a different file")
    return v2_trainer


def _same_state(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and left.shape == right.shape
                and np.array_equal(left, right))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(_same_state(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
                and len(left) == len(right)
                and all(_same_state(a, b) for a, b in zip(left, right, strict=True)))
    return left == right


def _check_chain(name: str, commit_sha: str) -> str:
    manifest = read_json(HERE / "ROUND58_GEN3_CHECKPOINT_ATTESTATION_MIGRATION.json")
    chain = [item for item in manifest["migrated"] if Path(item["file"]).name == name]
    current = commit_sha
    if not chain:
        raise RuntimeError(f"no attestation migration recorded for {name}")
    for item in chain:
        if item["sha_before"] != current:
            raise RuntimeError(f"migration chain gap for {name}")
        current = item["sha_after"]
    if sha(CP / name) != current:
        raise RuntimeError(f"migrated checkpoint bytes differ for {name}")
    sidecar = read_json(Path(str(CP / name) + ".r58.json"))
    if sidecar["checkpoint_sha256"] != current:
        raise RuntimeError(f"sidecar does not bind migrated {name}")
    return current


def _episode(ep: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    claim = read_json(CLAIMS / f"new_training_s11_e{ep}" / "attempt_0001.json")
    if (claim.get("status") != "complete" or claim.get("logical_job") != f"new_training_s11_e{ep}"
            or int(claim.get("run_return_calls", -1)) != 1
            or int(claim.get("completed_steps", -1)) != 1000):
        raise RuntimeError(f"episode {ep}: completed claim does not close")
    receipt = claim.get("round45_artifact_receipt", {})
    paths = claim.get("round45_artifact_paths", {})
    for key in ("row", "tape", "lifecycle", "transcript"):
        path = Path(paths[key]).resolve()
        approved_parent = (RAW / ("rows" if key == "row" else "logs")).resolve()
        if path.parent != approved_parent or sha(path) != receipt["artifacts"][key]["sha256"]:
            raise RuntimeError(f"episode {ep}: {key} is outside its hash-bound source")
    row = read_json(Path(paths["row"]))
    tape = [json.loads(line) for line in Path(paths["tape"]).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if (int(row["training_seed"]) != 11 or int(row["episode_index"]) != ep
            or int(row["scenario_seed"]) != int(claim["metadata"]["scenario_seed"])
            or int(row["steps"]) != 1000
            or len(tape) != receipt["decision_tape_records"]
            or not tape or any(int(a["step"]) >= int(b["step"])
                               for a, b in zip(tape, tape[1:]))
            or float(row["lambda"]) != 1.0
            or float(sum(float(x["airtime_bytes"]) for x in row["cost_stream"].values()))
               != float(row["spent_bytes"])
            or float(sum(float(x) for x in row["task_stream"].values()))
               != float(row["task_reward_total"])):
        raise RuntimeError(f"episode {ep}: row/tape math input mismatch")
    return row, tape, claim


def _normalised_rng() -> dict[str, Any]:
    py = list(random.getstate())
    py[1] = list(py[1])
    np_state = list(np.random.get_state())
    np_state[1] = np_state[1].tolist()
    return {"python": py, "numpy": np_state, "torch": torch.get_rng_state().tolist()}


def _replay_episode_rng(row: dict[str, Any], tape: list[dict[str, Any]], controller: Any,
                        *, check_values: bool = True) -> dict[str, Any]:
    # timemap.Environment.__init__ is the sole global Python/NumPy draw after
    # piloc_runtime reseeds them.  Its reset and the world step use local RNGs.
    seed = int(row["scenario_seed"])
    random.seed(seed)
    np.random.seed(seed)
    random.randrange(2 ** 63)
    np.random.randint(2 ** 31 - 1)
    # r40._run_episode reseeds Torch after construction, before the first step.
    torch.manual_seed(int(row["action_rng_seed"]))
    mismatched_actions = 0
    max_logprob_error = 0.0
    max_value_error = 0.0
    max_cost_value_error = 0.0
    draws = 0
    with torch.no_grad():
        for record in tape:
            features = np.asarray(record["features"], dtype=np.float32)
            if features.shape != (len(record["targets"]), 19):
                raise RuntimeError("decision tape has non-19D or nonaligned features")
            matrix = torch.as_tensor(features)
            logits, values = controller.forward(matrix)
            cost_values = controller.cost_values(matrix)
            probabilities = torch.softmax(logits, dim=-1)
            log_probabilities = torch.log_softmax(logits, dim=-1)
            free = record["free_one_shot_choices"]
            if free != record["actionable"]:
                raise RuntimeError("free/actionable decision ordering differs")
            predicted_log_prob = 0.0
            for target in free:
                index = record["targets"].index(target)
                actual_action = int(record["actions"][target])
                predicted_log_prob += float(log_probabilities[index, actual_action])
                replayed_action = int(torch.multinomial(probabilities[index], 1).item())
                mismatched_actions += replayed_action != actual_action
                draws += 1
            max_logprob_error = max(max_logprob_error, abs(predicted_log_prob -
                                                        float(record["log_prob"])))
            max_value_error = max(max_value_error, abs(float(values.mean()) -
                                                    float(record["value"])))
            max_cost_value_error = max(max_cost_value_error,
                                       abs(float(cost_values.mean()) - float(record["cost_value"])))
    if (mismatched_actions or max_logprob_error > 1e-6
            or (check_values and (max_value_error > 1e-6 or max_cost_value_error > 1e-6))):
        raise RuntimeError("same-weight action/likelihood/value replay disagrees with tape")
    return {"records": len(tape), "free_draws": draws,
            "action_mismatches": mismatched_actions,
            "max_logprob_error": max_logprob_error,
            "max_task_value_error": max_value_error,
            "max_cost_value_error": max_cost_value_error,
            "rng": _normalised_rng()}


def _verify() -> dict[str, Any]:
    trainer = _frozen_trainer()
    commit9 = read_json(CP / "r40_batch_commit_s11_b9.json")
    commit8 = read_json(CP / "r40_batch_commit_s11_b8.json")
    if (commit9["rolling_checkpoint_sha256"] != B9_ORIGINAL_SHA
            or sha(QUARANTINED_B9) != B9_ORIGINAL_SHA
            or commit9["episodes_done"] != 40 or commit9["updates"] != 10
            or commit9["optimizer_steps"] != 60):
        raise RuntimeError("batch9 committed pre-prefix state is not exact")
    current8 = _check_chain("v2_train_s11_after_b8.pt", commit8["postupdate_checkpoint_sha256"])
    current9 = _check_chain("v2_train_s11.pt", B9_ORIGINAL_SHA)
    payload8 = torch.load(CP / "v2_train_s11_after_b8.pt", map_location="cpu", weights_only=False)
    payload9 = torch.load(QUARANTINED_B9, map_location="cpu", weights_only=False)
    migrated9 = torch.load(CP / "v2_train_s11.pt", map_location="cpu", weights_only=False)
    if not all(_same_state(payload9[field], migrated9[field])
               for field in ("controller", "optimizer", "rng")):
        raise RuntimeError("batch9 attestation migration changed model, Adam or RNG")
    original_state = {k: v for k, v in payload9["training_state"].items()
                      if k != "round58_repair_attestation"}
    migrated_state = {k: v for k, v in migrated9["training_state"].items()
                      if k != "round58_repair_attestation"}
    if not _same_state(original_state, migrated_state):
        raise RuntimeError("batch9 attestation migration changed training state")
    controller8 = trainer._restore_controller(payload8, payload8["training_state"])
    controller9 = trainer._restore_controller(payload9, payload9["training_state"])
    row39, tape39, _ = _episode(39)
    control = _replay_episode_rng(row39, tape39, controller8)
    if control["rng"] != payload9["rng"]:
        raise RuntimeError("independent episode39 RNG reconstruction does not match committed batch9")
    state = payload9["training_state"]
    for name, expected in (("gae_lambda", 1.0), ("lambda", 1.0)):
        if float(state[name]) != expected:
            raise RuntimeError(f"batch9 frozen {name} differs")
    trainer.common.GAE_LAMBDA = float(state["gae_lambda"])
    trainer.common.GAMMA = float(state["gamma"])
    trainer.common.B0_BYTES = float(state["b0_bytes"])
    episodes = []
    per_episode = {}
    for ep in (40, 41, 42):
        row, tape, _ = _episode(ep)
        if row["controller_parameters_after"]["parameter_sha256"] != \
                commit9["controller_parameters"]["parameter_sha256"]:
            raise RuntimeError(f"episode {ep}: controller identity differs from batch9")
        replay = _replay_episode_rng(row, tape, controller9)
        episodes.append({**row, "collector": SimpleNamespace(records=tape)})
        per_episode[str(ep)] = {k: v for k, v in replay.items() if k != "rng"}
    samples = trainer.batch_samples(episodes)
    if len(samples) != sum(x["records"] for x in per_episode.values()):
        raise RuntimeError("frozen batch_samples omitted or duplicated a decision")
    if any(not all(math.isfinite(float(sample[name])) for name in
                   ("advantage", "return_task", "return_cost", "log_prob")) for sample in samples):
        raise RuntimeError("recovered batch prefix contains non-finite PPO input")
    rng42 = _normalised_rng()
    return {"kind": "round58-seed11-uncommitted-prefix-readonly-proof/1",
            "status": "prefix_inputs_and_rng_reconstructable_not_committed",
            "b9_original_checkpoint_sha256": B9_ORIGINAL_SHA,
            "b8_migrated_checkpoint_sha256": current8,
            "b9_migrated_checkpoint_sha256": current9,
            "b9_migration_numeric_state_equal": True,
            "control_episode39_rng_equals_committed_b9": True,
            "episodes": per_episode, "prefix_decision_samples": len(samples),
            "episode42_rng_component_sha256": {
                key: hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()
                for key, value in rng42.items()},
            "scope": "pure in-memory forward, RNG replay and frozen batch_samples; zero environment/optimizer/checkpoint writes"}


def verify() -> dict[str, Any]:
    """Check the prefix while leaving all three caller RNG streams untouched."""
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    trainer = _frozen_trainer()
    common_before = {key: getattr(trainer.common, key)
                     for key in ("GAE_LAMBDA", "GAMMA", "B0_BYTES")}
    try:
        return _verify()
    finally:
        for key, value in common_before.items():
            setattr(trainer.common, key, value)
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


def materialize_for_live_batch(controller: Any) -> dict[int, dict[str, Any]]:
    """Return exactly the three completed, uncommitted episode inputs.

    The caller must have restored the exact committed batch-9 controller.  This
    function never samples, claims, constructs an environment or updates weights.
    The caller may use ``_replay_episode_rng`` after episode 42 to install the
    independently validated post-rollout RNG state before episode 43 begins.
    """
    trainer = _frozen_trainer()
    source_state = torch.load(QUARANTINED_B9, map_location="cpu", weights_only=False)[
        "training_state"]
    for configured, source in (("GAE_LAMBDA", "gae_lambda"), ("GAMMA", "gamma"),
                               ("B0_BYTES", "b0_bytes")):
        if float(getattr(trainer.common, configured)) != float(source_state[source]):
            raise RuntimeError(f"live frozen {configured} differs from batch9 source")
    proof = verify()
    commit = read_json(CP / "r40_batch_commit_s11_b9.json")
    if (proof["status"] != "prefix_inputs_and_rng_reconstructable_not_committed"
            or controller.parameter_summary()["parameter_sha256"]
            != commit["controller_parameters"]["parameter_sha256"]):
        raise RuntimeError("live controller differs from the batch9 prefix source")
    result = {}
    for ep in (40, 41, 42):
        row, tape, _ = _episode(ep)
        result[ep] = {**row, "collector": SimpleNamespace(records=tape),
                      "controller": controller}
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2, sort_keys=True))
