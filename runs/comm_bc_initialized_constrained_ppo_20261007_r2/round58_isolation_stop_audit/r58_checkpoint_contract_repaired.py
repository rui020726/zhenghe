"""Version/attestation guard on the live R40 trainer save/load seam."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


# The controller intentionally imports this frozen pilot module lazily when a
# StageCController is reconstructed.  Bind only the reviewed source directory;
# never search arbitrary paths or accept a same-named third-party module.
TRAIN_PILOT_RELATIVE_PATH = Path("runs/comm_budget_pilot_20260919/train_pilot.py")
TRAIN_PILOT_SHA256 = "87dcf2e5251c99fc533ba52480a43da89902a3a0c371f5b9cfec6240d202ae81"
TRAIN_PILOT_BINDING_VERSION = "round58-frozen-train-pilot-restore-import/1"


def train_pilot_source_identity(repo_root: Path) -> dict[str, str]:
    # The approved R58 runtime imports the frozen dependency tree from the
    # sibling `zhenghe` root (the same root that supplies v2_trainer).  The
    # branch-local ROUND42 copy is not the live training dependency.
    canonical_root = Path(repo_root).resolve().parent / "zhenghe"
    source = (canonical_root / TRAIN_PILOT_RELATIVE_PATH).resolve()
    if not source.is_file():
        raise RuntimeError(f"frozen train_pilot source is missing: {source}")
    actual = _file_sha256(source)
    if actual != TRAIN_PILOT_SHA256:
        raise RuntimeError(f"frozen train_pilot source SHA-256 mismatch: {actual}")
    return {"path": str(source), "sha256": actual,
            "frozen_manifest_sha256": TRAIN_PILOT_SHA256,
            "binding_version": TRAIN_PILOT_BINDING_VERSION}


@contextmanager
def _verified_train_pilot_import_scope(repo_root: Path):
    identity = train_pilot_source_identity(repo_root)
    source = Path(identity["path"])
    expected_dir = str(source.parent)
    loaded = sys.modules.get("train_pilot")
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
        if loaded_path != source or _file_sha256(loaded_path) != identity["sha256"]:
            raise RuntimeError(f"拒绝错误来源的 train_pilot 模块: {loaded_path}")
        yield identity
        return

    previous_path = list(sys.path)
    while expected_dir in sys.path:
        sys.path.remove(expected_dir)
    sys.path.insert(0, expected_dir)
    importlib.invalidate_caches()
    try:
        yield identity
    finally:
        sys.path[:] = previous_path


def _exact_state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) \
            and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) \
            and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) \
            and left.keys() == right.keys() \
            and all(_exact_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) \
            and len(left) == len(right) \
            and all(_exact_state_equal(a, b) for a, b in zip(left, right, strict=True))
    try:
        import numpy as np
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) \
                and left.dtype == right.dtype and left.shape == right.shape \
                and np.array_equal(left, right)
    except ImportError:
        pass
    try:
        result = left == right
        return bool(result) if not isinstance(result, torch.Tensor) else bool(result.all())
    except Exception:
        return False


def _attested(value: dict[str, Any], identity: dict[str, Any], *, version: str) -> dict[str, Any]:
    result = deepcopy(value)
    result["training_version"] = version
    result["round58_repair_attestation"] = deepcopy(identity)
    return result


def require_payload(payload: Any, identity: dict[str, Any], *, version: str) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError("R58 resume refused: checkpoint payload is not a mapping")
    state, freeze = payload.get("training_state"), payload.get("freeze")
    if (payload.get("training_version") != version
            or payload.get("round58_repair_attestation") != identity
            or not isinstance(state, dict)
            or state.get("training_version") != version
            or state.get("round58_repair_attestation") != identity
            or not isinstance(freeze, dict)
            or freeze.get("training_version") != version
            or freeze.get("round58_repair_attestation") != identity):
        raise RuntimeError("R58 resume refused before restore: missing/mismatched repaired-version attestation")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(trainer: Any, r40_train: Any, *, identity: dict[str, Any],
            training_version: str, expected_checkpoint_root: Path,
            versions: Any | None = None, r42_run: Any | None = None,
            approved_sources: dict[int, dict[str, Any]] | None = None,
            resume_map_sha256: str | None = None,
             committed_continuation: dict[str, Any] | None = None,
             additional_continuations: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Wrap actual v2 trainer serialization and R40 candidate enumeration in memory."""
    if getattr(r40_train, "trainer", None) is not trainer:
        raise RuntimeError("cannot install checkpoint contract: R40 trainer object identity mismatch")
    names = ("save_checkpoint", "load_checkpoint", "load_checkpoint_read_only",
             "fresh_training_state", "TRAINING_VERSION")
    trainer_before = {name: getattr(trainer, name) for name in names}
    versions_before = (None if versions is None else
                       {"module": versions, "TRAINING_VERSION": getattr(versions, "TRAINING_VERSION", None)})
    r40_names = ("train_seed", "_latest_resume_source", "_recover_final_batch_commit")
    r40_before = {name: getattr(r40_train, name) for name in r40_names}
    optional_r40_names = ("_preserving_checkpoint_path", "_write_immutable_snapshot",
                          "_save_milestone")
    optional_r40_before = {name: getattr(r40_train, name)
                           for name in optional_r40_names if hasattr(r40_train, name)}
    run_torch_load_cpu = (None if r42_run is None else
                          getattr(r42_run, "torch_load_cpu", None))
    save0 = trainer.save_checkpoint
    load0 = trainer.load_checkpoint
    load_ro0 = trainer.load_checkpoint_read_only
    fresh0 = trainer.fresh_training_state
    latest0 = r40_train._latest_resume_source
    train0 = r40_train.train_seed
    preserve0 = getattr(r40_train, "_preserving_checkpoint_path", None)
    immutable0 = getattr(r40_train, "_write_immutable_snapshot", None)
    milestone0 = getattr(r40_train, "_save_milestone", None)
    root = Path(expected_checkpoint_root).resolve()
    approved_sources = {int(k): v for k, v in (approved_sources or {}).items()}
    historical_path_seed = {Path(v["path"]).resolve(): int(k)
                            for k, v in approved_sources.items()}
    continuation = committed_continuation or {}
    continuation_path = (Path(continuation["path"]).resolve() if continuation else None)
    continuation_by_seed = ({11: continuation} if continuation else {})
    continuation_by_seed.update({int(k): v for k, v in
                                 (additional_continuations or {}).items()})
    continuation_by_path = {Path(v["path"]).resolve(): (seed, v)
                            for seed, v in continuation_by_seed.items()}
    if len(continuation_by_path) != len(continuation_by_seed):
        raise RuntimeError("R58 committed continuation paths are not unique")
    repo_root = Path(__file__).resolve().parents[3]
    train_pilot_identity = train_pilot_source_identity(repo_root)
    configured_pilot_dir = Path(getattr(getattr(r40_train, "common", None),
                                        "PILOT_ROUND", "")).resolve()
    if configured_pilot_dir != Path(train_pilot_identity["path"]).parent:
        raise RuntimeError("configured v2 common PILOT_ROUND does not match frozen train_pilot source")

    def load_controller_state(loader, *args, **kwargs):
        # The private configure bridge restores sys.path after configure. The
        # StageCController imports train_pilot lazily, so bind only this exact
        # verified source around actual controller reconstruction.
        with _verified_train_pilot_import_scope(repo_root):
            result = loader(*args, **kwargs)
            module = sys.modules.get("train_pilot")
            if module is None:
                raise RuntimeError("controller restore did not load frozen train_pilot")
            loaded_path = Path(getattr(module, "__file__", "")).resolve()
            loaded_sha = _file_sha256(loaded_path)
            if loaded_path != Path(train_pilot_identity["path"]) \
                    or loaded_sha != train_pilot_identity["sha256"]:
                raise RuntimeError("controller restore loaded train_pilot outside frozen identity")
            trainer._round58_train_pilot_binding = dict(train_pilot_identity)
            return result

    def checked_path(path: Path) -> Path:
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("R58 checkpoint path escaped the frozen generation root") from exc
        return resolved

    def sidecar_path(path: Path) -> Path:
        return Path(path).with_name(Path(path).name + ".r58.json")

    def require_sidecar(path: Path) -> dict[str, Any]:
        checked = checked_path(path)
        receipt_path = sidecar_path(checked)
        if not receipt_path.is_file():
            raise RuntimeError("R58 resume refused before checkpoint deserialization: repaired sidecar missing")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("kind") != "round58-checkpoint-sidecar/1"
                or receipt.get("checkpoint_name") != checked.name
                or receipt.get("training_version") != training_version
                or receipt.get("round58_repair_attestation") != identity
                or receipt.get("checkpoint_sha256") != _file_sha256(checked)):
            raise RuntimeError("R58 resume refused before checkpoint deserialization: sidecar mismatch")
        return receipt

    def write_sidecar(path: Path) -> None:
        checked = checked_path(path)
        receipt = {"kind": "round58-checkpoint-sidecar/1",
                   "checkpoint_name": checked.name,
                   "checkpoint_sha256": _file_sha256(checked),
                   "training_version": training_version,
                   "round58_repair_attestation": deepcopy(identity)}
        target = sidecar_path(checked)
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        if temporary.exists():
            raise RuntimeError("stale R58 checkpoint sidecar temporary requires inspection")
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)

    def finish_renamed_checkpoint(path: Path, temporary_kind: str) -> None:
        final = checked_path(path)
        target_sidecar = sidecar_path(final)
        if target_sidecar.is_file():
            require_sidecar(final)
        else:
            write_sidecar(final)
        temporary_checkpoint = final.with_name(
            final.name + f".{os.getpid()}.{temporary_kind}-tmp")
        temporary_sidecar = sidecar_path(temporary_checkpoint)
        if temporary_sidecar.is_file():
            receipt = json.loads(temporary_sidecar.read_text(encoding="utf-8"))
            if (receipt.get("round58_repair_attestation") != identity
                    or receipt.get("checkpoint_sha256") != _file_sha256(final)):
                raise RuntimeError("renamed R58 checkpoint sidecar does not match final bytes")
            temporary_sidecar.unlink()

    def save(controller, optimizer, state, path, *, freeze):
        checked_path(path)
        state2 = _attested(state, identity, version=training_version)
        freeze2 = _attested(freeze, identity, version=training_version)
        # The actual ROUND42/R40 serializer remains the writer; inject top-level provenance
        # exactly at its torch.save boundary, while the canonical schema code stays untouched.
        save_torch = torch.save
        def attested_torch_save(payload, file, *args, **kwargs):
            payload2 = dict(payload)
            payload2["round58_repair_attestation"] = deepcopy(identity)
            return save_torch(payload2, file, *args, **kwargs)
        torch.save = attested_torch_save
        try:
            result = save0(controller, optimizer, state2, path, freeze=freeze2)
            write_sidecar(path)
            return result
        finally:
            torch.save = save_torch

    def load(path, *, freeze):
        resolved = Path(path).resolve()
        if resolved in continuation_by_path:
            continuation_seed, entry = continuation_by_path[resolved]
            if _file_sha256(resolved) != entry["checkpoint_sha256"]:
                raise RuntimeError("R58 exact batch9 continuation bytes changed")
            commit = Path(entry["commit_path"])
            if _file_sha256(commit) != entry["commit_sha256"]:
                raise RuntimeError("R58 exact batch9 commit changed")
            payload = torch.load(resolved, map_location="cpu", weights_only=False)
            source_freeze = deepcopy(freeze)
            source_freeze.pop("round58_repair_attestation", None)
            original_freeze = payload.get("freeze")
            if (not isinstance(original_freeze, dict)
                    or {k: v for k, v in original_freeze.items()
                        if k != "round58_repair_attestation"} != source_freeze
                    or original_freeze.get("round58_repair_attestation")
                        != entry["original_identity"]
                    or payload.get("round58_repair_attestation")
                        != entry["original_identity"]):
                raise RuntimeError("R58 batch9 freeze/producer identity differs from approved source")
            controller, optimizer, state = load_controller_state(
                load0, resolved, freeze=original_freeze)
            approved_cursor = entry["cursor"]
            if (not _exact_state_equal(controller.state_dict(), payload.get("controller"))
                    or not _exact_state_equal(optimizer.state_dict(), payload.get("optimizer"))
                    or not _exact_state_equal(trainer.rng_state(), payload.get("rng"))
                    or (int(state.get("episodes_done", -1)), int(state.get("updates", -1)),
                        int(state.get("optimizer_steps", -1))) !=
                       (int(approved_cursor["episodes_done"]), int(approved_cursor["updates"]),
                        int(approved_cursor["optimizer_steps"]))):
                raise RuntimeError("R58 committed source model/Adam/RNG/cursor did not restore exactly")
            state = deepcopy(state)
            state["round58_resume_provenance"] = {
                "kind": "round58-exact-committed-continuation-source/1",
                "source_checkpoint_sha256": entry["checkpoint_sha256"],
                "source_commit_sha256": entry["commit_sha256"],
                "training_seed": continuation_seed,
                "cursor": deepcopy(approved_cursor),
            }
            state["round58_repair_attestation"] = deepcopy(identity)
            return controller, optimizer, state
        if resolved in historical_path_seed:
            seed = historical_path_seed[resolved]
            source = approved_sources[seed]
            if _file_sha256(resolved) != source["checkpoint_sha256"]:
                raise RuntimeError("R58 historical source bytes changed before restore")
            sidecar = Path(str(resolved) + ".r58.json")
            if _file_sha256(sidecar) != source["sidecar_sha256"]:
                raise RuntimeError("R58 historical source sidecar changed before restore")
            payload = torch.load(resolved, map_location="cpu", weights_only=False)
            source_freeze = deepcopy(freeze)
            source_freeze.pop("round58_repair_attestation", None)
            from r58_resume_seam import validate_historical_payload
            validate_historical_payload(payload, source, base_freeze=source_freeze)
            controller, optimizer, state = load_controller_state(
                load0, resolved,
                freeze=_attested(source_freeze, source["original_identity"],
                                 version=training_version))
            if not _exact_state_equal(controller.state_dict(), payload.get("controller")):
                raise RuntimeError("R58 resumed model weights differ from the approved checkpoint")
            if not _exact_state_equal(optimizer.state_dict(), payload.get("optimizer")):
                raise RuntimeError("R58 resumed Adam state/moments differ from the approved checkpoint")
            if not _exact_state_equal(trainer.rng_state(), payload.get("rng")):
                raise RuntimeError("R58 restored Python/NumPy/Torch RNG differs from the approved checkpoint")
            if (int(state.get("episodes_done", -1)), int(state.get("updates", -1)),
                    int(state.get("optimizer_steps", -1))) != (4, 1, 8):
                raise RuntimeError("R58 historical restore returned a cursor other than 4/1/8")
            state = deepcopy(state)
            state["round58_resume_provenance"] = {
                "kind": "round58-qualified-historical-resume-source/1",
                "source_map_sha256": resume_map_sha256,
                "source_checkpoint_sha256": source["checkpoint_sha256"],
                "source_sidecar_sha256": source["sidecar_sha256"],
                "source_batch_sha256": source["batch_record_sha256"],
                "source_commit_sha256": source["commit_sha256"],
                "training_seed": seed,
                "cursor": {"episodes_done": 4, "updates": 1, "optimizer_steps": 8},
            }
            state["round58_repair_attestation"] = deepcopy(identity)
            return controller, optimizer, state
        checked_path(resolved)
        require_sidecar(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require_payload(payload, identity, version=training_version)
        return load_controller_state(
            load0, path, freeze=_attested(freeze, identity, version=training_version))

    def load_read_only(path, *, freeze):
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            # External BC/reference checkpoints are read-only inputs, never R58 resume sources.
            return load_controller_state(load_ro0, path, freeze=freeze)
        require_sidecar(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require_payload(payload, identity, version=training_version)
        return load_controller_state(
            load_ro0, path, freeze=_attested(freeze, identity, version=training_version))

    def validate_current_committed_source(path, seed):
        resolved = Path(path).resolve()
        if resolved in continuation_by_path:
            source_seed, entry = continuation_by_path[resolved]
            if (int(seed) != source_seed
                    or _file_sha256(resolved) != entry["checkpoint_sha256"]
                    or _file_sha256(entry["commit_path"]) != entry["commit_sha256"]
                    or _file_sha256(entry["sidecar_path"]) != entry["sidecar_sha256"]):
                raise RuntimeError("R58 historical V25 continuation changed")
            return {"path": str(resolved), "sha256": entry["checkpoint_sha256"],
                    "batch": int(entry["cursor"]["updates"]) - 1,
                    "commit_sha256": entry["commit_sha256"],
                    "cursor": deepcopy(entry["cursor"])}
        resolved = checked_path(path)
        require_sidecar(resolved)
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        require_payload(payload, identity, version=training_version)
        state = payload["training_state"]
        updates = int(state.get("updates", -1))
        if (int(state.get("training_seed", -1)) != int(seed) or updates < 1
                or int(state.get("episodes_done", -1)) != 4 * updates):
            raise RuntimeError("R58 selected current checkpoint has a noncommitted cursor")
        receipt_path = Path(r40_train._batch_commit_path(seed, updates - 1))
        if not receipt_path.is_file():
            provenance = state.get("round58_resume_provenance")
            historical = approved_sources.get(int(seed))
            b0_boundary = (int(seed) in (22, 33) and updates == 1
                           and isinstance(provenance, dict)
                           and provenance.get("kind") == "round58-qualified-historical-resume-source/1"
                           and provenance.get("source_map_sha256") == resume_map_sha256
                           and provenance.get("source_checkpoint_sha256")
                           == (historical or {}).get("checkpoint_sha256"))
            b9_boundary = (int(seed) == 11 and updates == 10
                           and isinstance(provenance, dict)
                           and provenance.get("kind") == "round58-exact-committed-batch9-source/1"
                           and provenance.get("source_checkpoint_sha256")
                           == continuation.get("checkpoint_sha256")
                           and provenance.get("source_commit_sha256")
                           == continuation.get("commit_sha256"))
            boundary = continuation_by_seed.get(int(seed), {})
            migrated_boundary = (bool(boundary)
                                 and updates == int(boundary.get("cursor", {}).get("updates", -1))
                                 and isinstance(provenance, dict)
                                 and provenance.get("kind")
                                     == "round58-exact-committed-continuation-source/1"
                                 and provenance.get("source_checkpoint_sha256")
                                     == boundary.get("checkpoint_sha256")
                                 and provenance.get("source_commit_sha256")
                                     == boundary.get("commit_sha256"))
            if not (resolved == Path(r40_train._rolling_path(seed)).resolve()
                    and (b0_boundary or b9_boundary or migrated_boundary)):
                raise RuntimeError("R58 selected current checkpoint lacks its exact batch commit")
            return {"path": str(resolved), "sha256": _file_sha256(resolved),
                    "batch": updates - 1, "boundary_provenance": provenance,
                    "cursor": {"episodes_done": 4 * updates, "updates": updates,
                               "optimizer_steps": int(state["optimizer_steps"])}}
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        digest = _file_sha256(resolved)
        batch_record = root / str(receipt.get("batch_record", ""))
        if (receipt.get("kind") != "round40-batch-checkpoint-commit/1"
                or int(receipt.get("training_seed", -1)) != int(seed)
                or int(receipt.get("batch", -1)) != updates - 1
                or int(receipt.get("episodes_done", -1)) != 4 * updates
                or int(receipt.get("updates", -1)) != updates
                or int(receipt.get("optimizer_steps", -1))
                   != int(state.get("optimizer_steps", -2))
                or not batch_record.is_file()
                or _file_sha256(batch_record) != receipt.get("batch_record_sha256")
                or digest not in {receipt.get("rolling_checkpoint_sha256"),
                                  receipt.get("postupdate_checkpoint_sha256")}):
            raise RuntimeError("R58 selected current checkpoint is not bound to its batch/commit")
        return {"path": str(resolved), "sha256": digest,
                "batch": updates - 1, "commit_sha256": _file_sha256(receipt_path),
                "cursor": {"episodes_done": 4 * updates, "updates": updates,
                           "optimizer_steps": int(state["optimizer_steps"])}}

    def fresh_state(seed):
        return _attested(fresh0(seed), identity, version=training_version)

    def latest(seed, freeze):
        candidates = ([r40_train._rolling_path(seed)] if r40_train._rolling_path(seed).exists() else [])
        candidates.extend(sorted(Path(r40_train.common.CHECKPOINTS).glob(
            f"v2_train_s{int(seed)}_after_b*.pt")))
        # Reject every candidate before the frozen R40 selector can instantiate a controller.
        for path in candidates:
            validate_current_committed_source(path, seed)
        if candidates:
            return latest0(seed, _attested(freeze, identity, version=training_version))
        selected_continuation = continuation_by_seed.get(int(seed))
        if selected_continuation is not None:
            selected_path = Path(selected_continuation["path"]).resolve()
            if _file_sha256(selected_path) != selected_continuation["checkpoint_sha256"]:
                raise RuntimeError("R58 batch9 selector source changed")
            return selected_path
        source = approved_sources.get(int(seed))
        if source is None:
            raise RuntimeError(f"R58 has no exact qualified resume source for seed {seed}")
        path = Path(source["path"]).resolve()
        if path not in historical_path_seed or _file_sha256(path) != source["checkpoint_sha256"]:
            raise RuntimeError(f"R58 source selector refused unlisted or changed source for seed {seed}")
        return path

    recover_commit0 = r40_train._recover_final_batch_commit
    def recover_final_batch_commit(seed, state, controller, checkpoint):
        receipt = Path(r40_train._batch_commit_path(seed, 0))
        provenance = state.get("round58_resume_provenance")
        boundary = continuation_by_seed.get(int(seed), {})
        if (boundary
                and int(state.get("episodes_done", -1)) == int(boundary.get("cursor", {}).get("episodes_done", -1))
                and int(state.get("updates", -1)) == int(boundary.get("cursor", {}).get("updates", -1))
                and isinstance(provenance, dict)
                and provenance.get("kind") == "round58-exact-committed-continuation-source/1"
                and provenance.get("source_commit_sha256") == boundary.get("commit_sha256")):
            return None
        if (not receipt.exists() and int(state.get("episodes_done", -1)) == 4
                and int(state.get("updates", -1)) == 1
                and isinstance(provenance, dict)
                and provenance.get("kind") == "round58-qualified-historical-resume-source/1"
                and provenance.get("source_map_sha256") == resume_map_sha256
                and provenance.get("training_seed") == int(seed)
                and provenance.get("source_checkpoint_sha256")
                    == approved_sources[int(seed)]["checkpoint_sha256"]):
            return None
        return recover_commit0(seed, state, controller, checkpoint)

    def train_seed(seed, *, episodes, freeze, reference_cache):
        if r40_train.trainer is not trainer:
            raise RuntimeError("R58 live R40 trainer binding changed before training entry")
        actual_root = Path(r40_train.common.CHECKPOINTS).resolve()
        if actual_root != root:
            raise RuntimeError("R58 live checkpoint root changed before training entry")
        return train0(seed, episodes=episodes,
                      freeze=_attested(freeze, identity, version=training_version),
                      reference_cache=reference_cache)

    def preserving_checkpoint_path(path, freeze):
        resolved = Path(path).resolve()
        if resolved.is_file():
            require_sidecar(resolved)
        return preserve0(path, freeze)

    def write_immutable_snapshot(path, controller, optimizer, state, freeze):
        resolved = Path(path).resolve()
        if resolved.is_file():
            require_sidecar(resolved)
            return immutable0(path, controller, optimizer, state, freeze)
        result = immutable0(path, controller, optimizer, state, freeze)
        finish_renamed_checkpoint(path, "snapshot")
        return result

    def save_milestone(controller, optimizer, state, seed, episodes, freeze):
        path = r40_train._milestone_path(seed, episodes, freeze)
        existed = Path(path).is_file()
        if existed:
            require_sidecar(path)
        result = milestone0(controller, optimizer, state, seed, episodes, freeze)
        if not existed:
            finish_renamed_checkpoint(path, "milestone")
        return result

    def guarded_torch_load_cpu(path):
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            return run_torch_load_cpu(path)
        require_sidecar(resolved)
        return run_torch_load_cpu(path)

    trainer.TRAINING_VERSION = training_version
    if versions is not None:
        versions.TRAINING_VERSION = training_version
    trainer.save_checkpoint = save
    trainer.load_checkpoint = load
    trainer.load_checkpoint_read_only = load_read_only
    trainer.fresh_training_state = fresh_state
    r40_train._latest_resume_source = latest
    r40_train._recover_final_batch_commit = recover_final_batch_commit
    r40_train.train_seed = train_seed
    if preserve0 is not None:
        r40_train._preserving_checkpoint_path = preserving_checkpoint_path
    if immutable0 is not None:
        r40_train._write_immutable_snapshot = write_immutable_snapshot
    if milestone0 is not None:
        r40_train._save_milestone = save_milestone
    if r42_run is not None and callable(run_torch_load_cpu):
        r42_run.torch_load_cpu = guarded_torch_load_cpu

    def undo() -> None:
        for name, value in trainer_before.items():
            setattr(trainer, name, value)
        if versions_before is not None:
            versions_before["module"].TRAINING_VERSION = versions_before["TRAINING_VERSION"]
        for name, value in r40_before.items():
            setattr(r40_train, name, value)
        for name, value in optional_r40_before.items():
            setattr(r40_train, name, value)
        if r42_run is not None and run_torch_load_cpu is not None:
            r42_run.torch_load_cpu = run_torch_load_cpu

    return {"kind": "round58-r40-checkpoint-contract/1",
            "identity": deepcopy(identity), "training_version": training_version,
            "checkpoint_root": str(root), "undo": undo,
            "wrapped_save": trainer.save_checkpoint is save,
            "wrapped_load": trainer.load_checkpoint is load,
            "wrapped_r40_train": r40_train.train_seed is train_seed,
            "wrapped_candidate_selector": r40_train._latest_resume_source is latest,
            "wrapped_historical_commit_boundary":
                r40_train._recover_final_batch_commit is recover_final_batch_commit,
            "committed_batch9_source_bound": continuation_path is not None,
            "validate_current_source": validate_current_committed_source,
            "wrapped_checkpoint_sidecar": True,
            "wrapped_readonly_runtime_loader": r42_run is not None and callable(run_torch_load_cpu)}
