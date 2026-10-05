"""Validated constant-time completion index for ROUND44 fixed evaluation cells."""

from __future__ import annotations

import hashlib
import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

SEEDS = (11, 22, 33)
MODES = ("greedy", "sampled")
SCENES = tuple(range(1_301_000, 1_301_016))
INDEX_KIND = "round44-eval-artifact-receipt/2"
JSON_REPRESENTATION_VERSION = "round44-strict-json-value/1"
EVAL_LOGICAL_RE = re.compile(
    r"^model_eval_new_s(11|22|33)_(greedy|sampled)_scene(13010\d{2})$")
ARTIFACT_NAMES = ("row", "lifecycle", "transcript", "tape")


class EvalIndexError(RuntimeError):
    """A recorded evaluation cannot be safely resumed or counted."""


class _DuplicateJSONKey(ValueError):
    """Raised when a persisted JSON object repeats a field name."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON number is not allowed: {value}")


def _validate_json_value(value: Any, path: str = "$", *, allow_tuples: bool = True) -> None:
    """Accept only strict JSON values, optionally including tuple array inputs."""
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise EvalIndexError(f"non-finite JSON number at {path}")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise EvalIndexError(f"JSON object key at {path} must be a string")
            _validate_json_value(item, f"{path}.{key}", allow_tuples=allow_tuples)
        return
    if type(value) is list or (allow_tuples and type(value) is tuple):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]", allow_tuples=allow_tuples)
        return
    raise EvalIndexError(f"unsupported JSON value at {path}: {type(value).__name__}")


def _json_value_equal(left: Any, right: Any) -> bool:
    """Compare strict JSON values, treating Python tuple/list as the same JSON array."""
    _validate_json_value(left)
    _validate_json_value(right)

    def equal(a: Any, b: Any) -> bool:
        a_array = type(a) in (list, tuple)
        b_array = type(b) in (list, tuple)
        if a_array or b_array:
            return (a_array and b_array and len(a) == len(b)
                    and all(equal(x, y) for x, y in zip(a, b)))
        if type(a) is not type(b):
            return False
        if type(a) is dict:
            return (a.keys() == b.keys()
                    and all(equal(a[key], b[key]) for key in a))
        if type(a) is float:
            # hex() preserves exact binary value and distinguishes -0.0 from +0.0.
            return a.hex() == b.hex()
        return a == b

    return equal(left, right)


def expected_eval_keys() -> tuple[tuple[int, str, int], ...]:
    return tuple((seed, mode, scene) for seed in SEEDS for mode in MODES for scene in SCENES)


def evaluation_schedule() -> tuple[tuple[int, str, int], ...]:
    return expected_eval_keys()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise EvalIndexError(f"unreadable JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise EvalIndexError(f"JSON artifact must be an object: {path}")
    _validate_json_value(value, str(path), allow_tuples=False)
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_json_constant,
                )
                if not isinstance(value, dict):
                    raise EvalIndexError(f"JSONL record is not an object: {path}:{number}")
                _validate_json_value(value, f"{path}:{number}", allow_tuples=False)
                records.append(value)
    except (OSError, UnicodeError, ValueError) as error:
        raise EvalIndexError(f"unreadable JSONL artifact: {path}") from error
    return records


def _slug(logical_job: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", logical_job).strip("_.")


def _strict_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise EvalIndexError(f"{field} must be an integer")
    return value


def _resolve_in_run(run_dir: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EvalIndexError(f"{field} path is missing")
    path = Path(value).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as error:
        raise EvalIndexError(f"{field} path escapes the ROUND44 output: {path}") from error
    if not path.is_file():
        raise EvalIndexError(f"{field} artifact is missing: {path}")
    return path


def _key_from_claim(claim: dict[str, Any], path: Path) -> tuple[int, str, int] | None:
    logical = claim.get("logical_job")
    if not isinstance(logical, str) or not logical.startswith("model_eval_"):
        return None
    match = EVAL_LOGICAL_RE.fullmatch(logical)
    if not match:
        raise EvalIndexError(f"foreign or malformed evaluation claim: {logical}")
    seed, mode, scene = int(match.group(1)), match.group(2), int(match.group(3))
    if _strict_int(claim.get("attempt"), "claim.attempt") < 1:
        raise EvalIndexError(f"invalid evaluation attempt in {path}")
    metadata = claim.get("metadata")
    if not isinstance(metadata, dict) or not _json_value_equal(metadata, {
        "arm": "NEW", "training_seed": seed, "mode": mode, "scenario_seed": scene,
    }):
        raise EvalIndexError(f"evaluation claim metadata mismatch: {path}")
    return seed, mode, scene


def _key_from_row(row: dict[str, Any], path: Path) -> tuple[int, str, int] | None:
    if row.get("kind") == "train":
        if "eval_mode" in row:
            raise EvalIndexError(f"training row carries an evaluation label: {path}")
        return None
    looks_like_eval = (
        row.get("kind") == "eval" or "eval_mode" in row
        or str(row.get("label", "")).startswith("eval_")
    )
    if not looks_like_eval:
        return None
    if row.get("kind") != "eval" or row.get("model_arm") != "NEW":
        raise EvalIndexError(f"foreign evaluation row in ROUND44 output: {path}")
    try:
        key = (
            _strict_int(row.get("training_seed"), "row.training_seed"),
            row.get("eval_mode"),
            _strict_int(row.get("scenario_seed"), "row.scenario_seed"),
        )
    except TypeError as error:
        raise EvalIndexError(f"malformed evaluation identity: {path}") from error
    if key[0] not in SEEDS or key[1] not in MODES or key[2] not in SCENES:
        raise EvalIndexError(f"evaluation row is outside the frozen matrix: {path}")
    return key


def _expected_label(key: tuple[int, str, int]) -> str:
    seed, mode, scene = key
    return f"eval_new_s{seed}_{mode}_s{scene}_r44"


def _verify_artifact_receipt(
    run_dir: Path,
    claim_path: Path,
    claim: dict[str, Any],
    row_path: Path,
    row: dict[str, Any],
    key: tuple[int, str, int],
    *,
    training_version: str,
    reward_timing_version: str,
    runner_version: str,
    checkpoint_parameter_sha256: str,
) -> dict[str, Any]:
    if _key_from_row(row, row_path) != key:
        raise EvalIndexError(f"evaluation row identity does not match its claim key: {row_path}")
    attempt = _strict_int(row.get("attempt"), "row.attempt")
    if attempt < 1 or _strict_int(claim.get("attempt"), "claim.attempt") != attempt:
        raise EvalIndexError(f"row/claim attempt mismatch for {key}")
    expected_category = "normal" if attempt == 1 else "reserve"
    if claim.get("category") != expected_category:
        raise EvalIndexError(f"evaluation attempt used the wrong quota category: {claim_path}")
    if (claim.get("status") != "complete"
            or type(claim.get("round")) is not int or claim.get("round") != 44):
        raise EvalIndexError(f"evaluation claim is not complete ROUND44: {claim_path}")
    if claim.get("training_version") != training_version:
        raise EvalIndexError(f"evaluation claim training version mismatch: {claim_path}")
    if claim.get("runner_version") != runner_version:
        raise EvalIndexError(f"evaluation claim runner version mismatch: {claim_path}")
    if claim.get("comm_action_semantics") != "one-shot-send-skip/1":
        raise EvalIndexError(f"evaluation claim action semantics mismatch: {claim_path}")
    if claim.get("logical_job") != f"model_eval_new_s{key[0]}_{key[1]}_scene{key[2]}":
        raise EvalIndexError(f"evaluation claim logical key mismatch: {claim_path}")
    if any(_strict_int(claim.get(field), f"claim.{field}") != expected for field, expected in (
            ("environment_constructor_calls", 1), ("run_invocation_calls", 1),
            ("joint_run_method_calls", 1), ("run_return_calls", 1),
            ("run_failure_calls", 0), ("completed_steps", 1000))):
        raise EvalIndexError(f"evaluation claim run accounting is incomplete: {claim_path}")
    if row.get("training_version") != training_version \
            or row.get("reward_timing_version") != reward_timing_version \
            or row.get("runner_version") != runner_version \
            or row.get("comm_action_semantics") != "one-shot-send-skip/1":
        raise EvalIndexError(f"evaluation row version identity mismatch: {row_path}")
    if row.get("label") != _expected_label(key) or row.get("eval_block") != f"one_shot_{key[1]}":
        raise EvalIndexError(f"foreign or mismatched evaluation label: {row_path}")
    metadata = claim.get("metadata")
    if not isinstance(metadata, dict) or not _json_value_equal(metadata, {
        "arm": "NEW", "training_seed": key[0], "mode": key[1], "scenario_seed": key[2],
    }):
        raise EvalIndexError(f"evaluation claim metadata mismatch: {claim_path}")
    identity = row.get("checkpoint_parameter_identity")
    if not isinstance(identity, dict) or identity.get("parameter_sha256") != checkpoint_parameter_sha256:
        raise EvalIndexError(f"evaluation row checkpoint parameter identity mismatch: {row_path}")
    artifacts = row.get("artefacts")
    if not isinstance(artifacts, dict):
        raise EvalIndexError(f"evaluation row has no artifact map: {row_path}")
    resolved: dict[str, Path] = {"row": row_path.resolve()}
    for name in ("lifecycle", "transcript", "tape"):
        resolved[name] = _resolve_in_run(run_dir, artifacts.get(name), f"row.artefacts.{name}")
    declared_row = Path(str(artifacts.get("row", ""))).resolve()
    if declared_row != row_path.resolve():
        raise EvalIndexError(f"evaluation row path does not point to its discovered file: {row_path}")

    lifecycle = _read_jsonl(resolved["lifecycle"])
    transcript = _read_jsonl(resolved["transcript"])
    tape = _read_jsonl(resolved["tape"])
    collector = row.get("collector_summary")
    if not isinstance(collector, dict) or type(collector.get("decision_steps")) is not int:
        raise EvalIndexError(f"evaluation row collector summary is malformed: {row_path}")
    if len(tape) != collector["decision_steps"]:
        raise EvalIndexError(f"evaluation tape length does not match row summary: {row_path}")
    if _strict_int(row.get("steps"), "row.steps") != 1000:
        raise EvalIndexError(f"evaluation row did not complete 1000 steps: {row_path}")
    if len(lifecycle) == 0 or len(transcript) == 0:
        raise EvalIndexError(f"evaluation lifecycle/transcript is empty: {row_path}")

    receipt = claim.get("round44_eval_artifact_receipt")
    if not isinstance(receipt, dict) or receipt.get("kind") != INDEX_KIND:
        raise EvalIndexError(f"completed evaluation claim lacks ROUND44 artifact receipt: {claim_path}")
    if set(receipt) != {
            "kind", "json_representation", "eval_key", "attempt", "training_version",
            "runner_version", "checkpoint_parameter_sha256", "artifacts"}:
        raise EvalIndexError(f"claim artifact receipt has missing or unknown fields: {claim_path}")
    if not _json_value_equal(receipt.get("eval_key"), [key[0], key[1], key[2]]) \
            or _strict_int(receipt.get("attempt"), "receipt.attempt") != attempt \
            or receipt.get("json_representation") != JSON_REPRESENTATION_VERSION \
            or receipt.get("training_version") != training_version \
            or receipt.get("runner_version") != runner_version \
            or receipt.get("checkpoint_parameter_sha256") != checkpoint_parameter_sha256:
        raise EvalIndexError(f"claim artifact receipt identity mismatch: {claim_path}")
    declared_artifacts = receipt.get("artifacts")
    if not isinstance(declared_artifacts, dict) or set(declared_artifacts) != set(ARTIFACT_NAMES):
        raise EvalIndexError(f"claim artifact receipt is incomplete: {claim_path}")
    actual_artifacts: dict[str, Any] = {}
    for name, path in resolved.items():
        entry = declared_artifacts.get(name)
        actual = {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        if not _json_value_equal(entry, actual):
            raise EvalIndexError(f"{name} artifact hash/size differs from completion receipt: {path}")
        actual_artifacts[name] = actual
    return {
        "kind": INDEX_KIND,
        "json_representation": JSON_REPRESENTATION_VERSION,
        "eval_key": [key[0], key[1], key[2]],
        "attempt": attempt,
        "logical_job": claim["logical_job"],
        "row_path": str(row_path.resolve()),
        "row_sha256": actual_artifacts["row"]["sha256"],
        "claim_path": str(claim_path.resolve()),
        "claim_sha256": _sha256(claim_path),
        "artifacts": actual_artifacts,
        "checkpoint_parameter_sha256": checkpoint_parameter_sha256,
        "training_version": training_version,
        "runner_version": runner_version,
    }


def create_claim_artifact_receipt(
    run_dir: Path,
    row: dict[str, Any],
    *,
    training_version: str,
    runner_version: str,
    checkpoint_parameter_sha256: str,
) -> dict[str, Any]:
    """Create the durable hash receipt before the runner marks an eval claim complete."""
    run_dir = Path(run_dir).resolve()
    key = (
        _strict_int(row.get("training_seed"), "row.training_seed"),
        row.get("eval_mode"),
        _strict_int(row.get("scenario_seed"), "row.scenario_seed"),
    )
    if key[0] not in SEEDS or key[1] not in MODES or key[2] not in SCENES:
        raise EvalIndexError(f"artifact receipt key is outside the frozen matrix: {key}")
    if row.get("kind") != "eval" or row.get("model_arm") != "NEW":
        raise EvalIndexError("artifact receipt may only be created for a NEW evaluation row")
    if _key_from_row(row, Path("<in-memory-row>")) != key:
        raise EvalIndexError(f"new evaluation row identity does not match its key: {key}")
    if row.get("training_version") != training_version or row.get("runner_version") != runner_version:
        raise EvalIndexError("new evaluation row has wrong version before claim completion")
    if (row.get("label") != _expected_label(key)
            or row.get("eval_block") != f"one_shot_{key[1]}"
            or row.get("comm_action_semantics") != "one-shot-send-skip/1"):
        raise EvalIndexError(f"new evaluation has a foreign label: {row.get('label')!r}")
    if _strict_int(row.get("attempt"), "row.attempt") < 1:
        raise EvalIndexError("new evaluation attempt must be positive")
    identity = row.get("checkpoint_parameter_identity")
    if not isinstance(identity, dict) or identity.get("parameter_sha256") != checkpoint_parameter_sha256:
        raise EvalIndexError(f"new evaluation does not point at its frozen ep128 checkpoint: {key}")
    artifacts = row.get("artefacts")
    if not isinstance(artifacts, dict):
        raise EvalIndexError(f"new evaluation lacks an artifact map: {key}")
    resolved: dict[str, Path] = {}
    for name in ("row", "lifecycle", "transcript", "tape"):
        resolved[name] = _resolve_in_run(run_dir, artifacts.get(name), f"row.artefacts.{name}")
    _validate_json_value(row)
    stored_row = _read_json(resolved["row"])
    if not _json_value_equal(stored_row, row):
        raise EvalIndexError(f"persisted row differs from the completed in-memory row: {key}")
    lifecycle = _read_jsonl(resolved["lifecycle"])
    transcript = _read_jsonl(resolved["transcript"])
    tape = _read_jsonl(resolved["tape"])
    collector = row.get("collector_summary")
    if not isinstance(collector, dict) or type(collector.get("decision_steps")) is not int:
        raise EvalIndexError(f"collector summary is malformed for {key}")
    if (len(tape) != collector["decision_steps"] or not lifecycle or not transcript
            or _strict_int(row.get("steps"), "row.steps") != 1000):
        raise EvalIndexError(f"one or more eval artifacts are incomplete for {key}")
    return {
        "kind": INDEX_KIND,
        "json_representation": JSON_REPRESENTATION_VERSION,
        "eval_key": [key[0], key[1], key[2]],
        "attempt": _strict_int(row.get("attempt"), "row.attempt"),
        "training_version": training_version,
        "runner_version": runner_version,
        "checkpoint_parameter_sha256": checkpoint_parameter_sha256,
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in resolved.items()
        },
    }


@dataclass
class EvalCompletionIndex:
    run_dir: Path
    entries: dict[tuple[int, str, int], dict[str, Any]]
    row_cache: dict[tuple[int, str, int], dict[str, Any]]

    def get(self, seed: int, mode: str, scene: int) -> dict[str, Any] | None:
        """Return the already-validated in-memory row; performs no filesystem access."""
        key = (int(seed), str(mode), int(scene))
        row = self.row_cache.get(key)
        return copy.deepcopy(row) if row is not None else None

    @property
    def completed_count(self) -> int:
        return len(self.entries)

    @property
    def completed_keys(self) -> list[list[Any]]:
        return [[seed, mode, scene] for seed, mode, scene in evaluation_schedule()
                if (seed, mode, scene) in self.entries]

    def add_verified(self, entry: dict[str, Any], row: dict[str, Any]) -> None:
        raw_key = entry.get("eval_key")
        if not isinstance(raw_key, list) or len(raw_key) != 3:
            raise EvalIndexError("verified completion has malformed eval_key")
        key = (int(raw_key[0]), str(raw_key[1]), int(raw_key[2]))
        if key not in expected_eval_keys():
            raise EvalIndexError(f"verified completion is outside the fixed matrix: {key}")
        if key in self.entries:
            raise EvalIndexError(f"duplicate successful evaluation cell: {key}")
        if not isinstance(row, dict) or row.get("label") != _expected_label(key):
            raise EvalIndexError(f"incremental index row identity mismatch: {key}")
        self.entries[key] = entry
        self.row_cache[key] = copy.deepcopy(row)


def _collect_claims(run_dir: Path) -> tuple[dict[tuple[str, int], tuple[Path, dict[str, Any]]],
                                              dict[tuple[int, str, int], list[tuple[Path, dict[str, Any]]]]]:
    all_claims: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    by_key: dict[tuple[int, str, int], list[tuple[Path, dict[str, Any]]]] = {}
    claim_root = run_dir / "launch_claims"
    if not claim_root.exists():
        return all_claims, by_key
    for path in sorted(claim_root.glob("*/attempt_*.json")):
        claim = _read_json(path)
        logical = claim.get("logical_job")
        attempt = _strict_int(claim.get("attempt"), "claim.attempt")
        if not isinstance(logical, str) or attempt < 1:
            raise EvalIndexError(f"claim identity is malformed: {path}")
        claim_key = (logical, attempt)
        if claim_key in all_claims:
            raise EvalIndexError(f"duplicate claim identity: {claim_key}")
        all_claims[claim_key] = (path, claim)
        eval_key = _key_from_claim(claim, path)
        if eval_key is not None:
            expected_category = "normal" if attempt == 1 else "reserve"
            if claim.get("category") != expected_category:
                raise EvalIndexError(f"evaluation claim has the wrong quota category: {path}")
            by_key.setdefault(eval_key, []).append((path, claim))
    return all_claims, by_key


def build_eval_index(
    run_dir: Path,
    *,
    training_version: str,
    reward_timing_version: str,
    runner_version: str,
    checkpoint_parameter_sha256_by_seed: dict[int, str],
) -> EvalCompletionIndex:
    """Build once from durable rows/claims; reject ambiguous or incomplete successes."""
    run_dir = Path(run_dir).resolve()
    rows_dir = run_dir / "rows"
    claim_map, claims_by_key = _collect_claims(run_dir)
    rows_by_key: dict[tuple[int, str, int], list[tuple[Path, dict[str, Any]]]] = {}
    if rows_dir.exists():
        for path in sorted(rows_dir.glob("*.json")):
            row = _read_json(path)
            key = _key_from_row(row, path)
            if key is None:
                continue
            rows_by_key.setdefault(key, []).append((path, row))

    candidate_keys = set(rows_by_key) | set(claims_by_key)
    expected = set(expected_eval_keys())
    foreign = candidate_keys - expected
    if foreign:
        raise EvalIndexError(f"evaluation evidence contains foreign keys: {sorted(foreign)}")
    entries: dict[tuple[int, str, int], dict[str, Any]] = {}
    row_cache: dict[tuple[int, str, int], dict[str, Any]] = {}
    for key in sorted(candidate_keys, key=expected_eval_keys().index):
        rows = rows_by_key.get(key, [])
        claims = claims_by_key.get(key, [])
        unresolved_claims = [(path, claim) for path, claim in claims
                             if claim.get("status") not in {
                                 "complete", "failed_technical", "failed_non_technical",
                                 "failed_nontechnical"}]
        if unresolved_claims:
            raise EvalIndexError(
                f"evaluation has a nonterminal claim requiring inspection: "
                f"{unresolved_claims[0][0]}")
        claims_by_attempt = {
            _strict_int(claim.get("attempt"), "claim.attempt"): (path, claim)
            for path, claim in claims
        }
        completed_claims = [(path, claim) for path, claim in claims
                            if claim.get("status") == "complete"]
        successful_rows: list[tuple[Path, dict[str, Any]]] = []
        for row_path, row in rows:
            attempt = _strict_int(row.get("attempt"), "row.attempt")
            linked = claims_by_attempt.get(attempt)
            if linked is None:
                raise EvalIndexError(f"evaluation row has no matching attempt claim for {key}: {row_path}")
            claim_path, claim = linked
            if (_key_from_row(row, row_path) != key
                    or _strict_int(claim.get("attempt"), "claim.attempt") != attempt
                    or claim.get("logical_job")
                    != f"model_eval_new_s{key[0]}_{key[1]}_scene{key[2]}"):
                raise EvalIndexError(f"evaluation row is linked to the wrong attempt claim for {key}")
            artifacts = row.get("artefacts")
            if not isinstance(artifacts, dict) or not isinstance(artifacts.get("row"), str) \
                    or Path(artifacts["row"]).resolve() != row_path.resolve():
                raise EvalIndexError(f"evaluation row path does not match its attempt row: {row_path}")
            if claim.get("status") in {
                    "failed_technical", "failed_non_technical", "failed_nontechnical"}:
                if "round44_eval_artifact_receipt" in claim:
                    raise EvalIndexError(f"failed evaluation attempt carries a completion receipt: {claim_path}")
                continue
            successful_rows.append((row_path, row))
        if len(successful_rows) > 1 or len(completed_claims) > 1:
            raise EvalIndexError(f"duplicate successful evaluation evidence for {key}")
        if not successful_rows and not completed_claims:
            continue
        if len(successful_rows) != 1 or len(completed_claims) != 1:
            raise EvalIndexError(f"row/complete-claim pair is incomplete for {key}")
        row_path, row = successful_rows[0]
        claim_path, claim = completed_claims[0]
        attempt = _strict_int(row.get("attempt"), "row.attempt")
        maximum_attempt = max(_strict_int(item.get("attempt"), "claim.attempt")
                              for _, item in claims)
        if attempt != maximum_attempt:
            raise EvalIndexError(f"a later claim exists after the successful evaluation attempt for {key}")
        logical = f"model_eval_new_s{key[0]}_{key[1]}_scene{key[2]}"
        if (claim.get("logical_job") != logical
                or _strict_int(claim.get("attempt"), "claim.attempt") != attempt):
            raise EvalIndexError(f"row is linked to the wrong claim for {key}")
        checkpoint_sha = checkpoint_parameter_sha256_by_seed.get(key[0])
        if not isinstance(checkpoint_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha):
            raise EvalIndexError(f"missing frozen ep128 parameter identity for seed {key[0]}")
        entries[key] = _verify_artifact_receipt(
            run_dir, claim_path, claim, row_path, row, key,
            training_version=training_version,
            reward_timing_version=reward_timing_version,
            runner_version=runner_version,
            checkpoint_parameter_sha256=checkpoint_sha,
        )
        row_cache[key] = copy.deepcopy(row)
    return EvalCompletionIndex(run_dir=run_dir, entries=entries, row_cache=row_cache)


def verify_new_cell(
    run_dir: Path,
    *,
    seed: int,
    mode: str,
    scene: int,
    training_version: str,
    reward_timing_version: str,
    runner_version: str,
    checkpoint_parameter_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the just-written row/claim/artifacts before indexing or progress update."""
    run_dir = Path(run_dir).resolve()
    key = (int(seed), str(mode), int(scene))
    logical = f"model_eval_new_s{key[0]}_{key[1]}_scene{key[2]}"
    claim_dir = run_dir / "launch_claims" / _slug(logical)
    claims = []
    nonterminal = []
    for claim_path in sorted(claim_dir.glob("attempt_*.json")):
        claim = _read_json(claim_path)
        if claim.get("status") == "complete":
            claims.append((claim_path, claim))
        elif claim.get("status") not in {
                "failed_technical", "failed_non_technical", "failed_nontechnical"}:
            nonterminal.append(claim_path)
    if nonterminal:
        raise EvalIndexError(f"new evaluation has a nonterminal claim requiring inspection: {nonterminal[0]}")
    if len(claims) != 1:
        raise EvalIndexError(f"expected exactly one completed claim for new cell {key}")
    claim_path, claim = claims[0]
    attempt_numbers = [_strict_int(_read_json(path).get("attempt"), "claim.attempt")
                       for path in sorted(claim_dir.glob("attempt_*.json"))]
    attempt = _strict_int(claim.get("attempt"), "claim.attempt")
    if not attempt_numbers or attempt != max(attempt_numbers):
        raise EvalIndexError(f"a later claim exists after the successful evaluation attempt for {key}")
    row_path = run_dir / "rows" / f"v2_r44_eval_new_s{key[0]}_{key[1]}_s{key[2]}_a{attempt}.json"
    if not row_path.is_file():
        raise EvalIndexError(f"new evaluation row is missing for {key}: {row_path}")
    row = _read_json(row_path)
    if _key_from_row(row, row_path) != key:
        raise EvalIndexError(f"new evaluation row key does not match requested cell {key}")
    entry = _verify_artifact_receipt(
        run_dir, claim_path, claim, row_path, row, key,
        training_version=training_version,
        reward_timing_version=reward_timing_version,
        runner_version=runner_version,
        checkpoint_parameter_sha256=checkpoint_parameter_sha256,
    )
    return row, entry


def audit_fixed_matrix(
    run_dir: Path,
    *,
    training_version: str,
    reward_timing_version: str,
    runner_version: str,
    checkpoint_parameter_sha256_by_seed: dict[int, str],
) -> dict[str, Any]:
    """Independent final disk audit; it does not consume the live in-memory index."""
    fresh = build_eval_index(
        run_dir,
        training_version=training_version,
        reward_timing_version=reward_timing_version,
        runner_version=runner_version,
        checkpoint_parameter_sha256_by_seed=checkpoint_parameter_sha256_by_seed,
    )
    expected = set(expected_eval_keys())
    actual = set(fresh.entries)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise EvalIndexError(f"fixed evaluation matrix is incomplete: missing={missing}, extra={extra}")
    per_group = {f"s{seed}_{mode}": sum(
        (seed, mode, scene) in actual for scene in SCENES)
        for seed in SEEDS for mode in MODES}
    if set(per_group.values()) != {16}:
        raise EvalIndexError(f"fixed evaluation group counts are not 16 each: {per_group}")
    return {
        "kind": "round44-independent-eval-matrix-audit/1",
        "status": "passed",
        "cell_count": len(fresh.entries),
        "group_counts": per_group,
        "entries": [fresh.entries[key] for key in expected_eval_keys()],
    }
