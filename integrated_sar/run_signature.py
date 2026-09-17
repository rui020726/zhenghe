"""One fingerprint per run, so an artefact can never be read as an arm it did not run.

The phase before this one produced files named ``...sticky...`` that had been produced
without the sticky flag. Nothing in the pipeline noticed, because nothing recorded what a
run *was* -- only what it was called. A resume check that skips on the existence of a path,
and an aggregation that reads whatever is on disk, then quietly report the wrong arm.

A signature is written into both artefacts a run produces (the lifecycle log's
``run_config`` record and the decision trace's config record) and checked before anything is
aggregated. It carries the arm identity and everything that could make two runs differ
without differing in name: the flag, the code commit, the checkpoint digests, the seed, the
stream the run replayed, a hash of the configuration, and the trace schema version.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# Schema /2 adds the raw/forced/post-mask choice, the command identity, the reservation
# identity and the validity conditions to the decision record, and replaces the single
# ``honoured`` flag on ``held_reservation`` with the executed / raw-choice split.
TRACE_SCHEMA = "allocation-decision-trace/2"
SIGNATURE_FORMAT = "run-signature/1"

FIELDS = (
    "arm",
    "sticky_commitments",
    "code_commit",
    "checkpoints",
    "seed",
    "stream",
    "config_hash",
    "trace_schema",
)


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    target = Path(path)
    if not target.exists():
        return None
    return hashlib.sha256(target.read_bytes()).hexdigest()


def code_commit(repo: str | Path | None = None) -> str:
    """The commit the run was produced from, or an explicit marker when unknown.

    Never an empty string: a missing commit has to look missing, not match another missing
    one, or two runs from different trees would compare equal.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo) if repo is not None else None,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return "<no-git>"
    return result.stdout.strip() or "<no-git>"


def config_hash(configuration: Any) -> str:
    """A stable hash of the configuration, so a changed default cannot hide behind a label."""
    payload = configuration.as_dict() if hasattr(configuration, "as_dict") else dict(configuration)
    # The output paths are part of the invocation, not of the experiment: two arms differ in
    # their paths by construction, and hashing them would make every signature unique.
    payload = {
        key: value
        for key, value in payload.items()
        if key not in {"output", "decision_trace", "reservation_log"}
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build(configuration: Any, *, arm: str, repo: str | Path | None = None) -> dict[str, Any]:
    """The signature of one run, from the run's own configuration."""
    piloc = getattr(configuration, "piloc_checkpoint", None)
    ctde = getattr(configuration, "task_checkpoint", None)
    stream = getattr(configuration, "replay_stream", None)
    return {
        "format": SIGNATURE_FORMAT,
        "arm": arm,
        "sticky_commitments": bool(getattr(configuration, "sticky_commitments", False)),
        "code_commit": code_commit(repo),
        "checkpoints": {
            "piloc": sha256_file(piloc),
            "ctde": sha256_file(ctde),
        },
        "seed": int(getattr(configuration, "seed", -1)),
        "stream": {"path": str(stream) if stream else None, "sha256": sha256_file(stream)},
        "config_hash": config_hash(configuration),
        "trace_schema": TRACE_SCHEMA,
    }


def comparable(signature: Mapping[str, Any]) -> dict[str, Any]:
    """Only the fields that must match for two runs to be the same arm on the same data."""
    return {key: signature.get(key) for key in FIELDS}


def difference(expected: Mapping[str, Any], found: Mapping[str, Any]) -> list[str]:
    """The fields that disagree, named, so a failure says what is wrong and not just that."""
    left, right = comparable(expected), comparable(found)
    return [key for key in FIELDS if left.get(key) != right.get(key)]


# The two artefacts a run writes, and what "finished" means for each: a lifecycle log is
# complete when it closes with the run's own final record, a trace when it closes with its
# trace_final line. A run killed halfway leaves a file that parses but never says it ended,
# which is why existence alone may never be read as success.
LIFECYCLE = "lifecycle"
TRACE = "trace"


def _records(path: str | Path) -> list[dict[str, Any]]:
    """Every JSON object in the file. A truncated last line is dropped, not guessed at."""
    out: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    out.append(payload)
    except OSError:
        return []
    return out


def _first_and_last(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
    """The first record, the last record, and the line count.

    The trace files are large, so only two lines are decoded: the count is taken by
    iterating the raw bytes, and the last record comes from a short read at the end of the
    file. Nothing here needs the middle of the file.
    """
    first: dict[str, Any] | None = None
    lines = 0
    try:
        with open(path, "rb") as handle:
            for raw in handle:
                lines += 1
                if first is not None:
                    continue
                text = raw.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    first = payload

            tail_start = max(0, handle.seek(0, 2) - 65536)
            handle.seek(tail_start)
            tail = handle.read().decode("utf-8", "replace").strip().splitlines()
    except OSError:
        return None, None, 0
    last: dict[str, Any] | None = None
    for text in reversed(tail):
        text = text.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            last = payload
            break
    return first, last, lines


def _signature_of(record: Mapping[str, Any], kind: str) -> dict[str, Any] | None:
    if kind == LIFECYCLE:
        detail = record.get("detail")
        signature = detail.get("run_signature") if isinstance(detail, Mapping) else None
    else:
        meta = record.get("meta")
        signature = meta.get("run_signature") if isinstance(meta, Mapping) else None
    return dict(signature) if isinstance(signature, Mapping) else None


def _is_complete(records: Sequence[Mapping[str, Any]], kind: str) -> bool:
    if not records:
        return False
    last = records[-1]
    if kind == LIFECYCLE:
        return (
            last.get("event_type") == "final"
            and last.get("status") == "run_finished"
            and str(last.get("target_id")) == "__run__"
        )
    return last.get("event_type") == "trace_final"


@dataclass(frozen=True, slots=True)
class Artefact:
    """What one file on disk actually is, as opposed to what its name claims.

    ``problems`` is empty only for a file that exists, carries a signature and ended. Every
    other outcome is named, because the failure this exists to prevent was silence: a
    missing or half-written file simply not being counted, and an arm quietly reporting the
    results of a shorter batch.
    """

    path: str
    kind: str
    exists: bool
    signature: dict[str, Any] | None = None
    complete: bool = False
    records: int = 0
    problems: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.exists and self.signature is not None and self.complete


def inspect(path: str | Path, kind: str) -> Artefact:
    """Read one artefact and say whether it may be trusted, without judging its arm."""
    target = Path(path)
    if not target.exists():
        return Artefact(str(target), kind, exists=False, problems=("missing",))
    first, last, lines = _first_and_last(target)
    problems: list[str] = []
    if first is None:
        problems.append("unreadable")
    signature = _signature_of(first, kind) if first is not None else None
    if first is not None and signature is None:
        problems.append("unsigned")
    complete = _is_complete([last] if last is not None else [], kind)
    if not complete:
        problems.append("incomplete")
    return Artefact(
        path=str(target),
        kind=kind,
        exists=True,
        signature=signature,
        complete=complete,
        records=lines,
        problems=tuple(problems),
    )


def read_signature(path: str | Path, kind: str) -> dict[str, Any] | None:
    """Just the signature a file carries, without judging whether the file finished."""
    target = Path(path)
    if not target.exists():
        return None
    first, _, _ = _first_and_last(target)
    return _signature_of(first, kind) if first is not None else None


def detection_fingerprint(path: str | Path) -> str | None:
    """A digest of the detections a run processed, in order.

    Every arm of a split replays the same recorded stream, so this digest must agree across
    arms for one seed. It is taken from the lifecycle log rather than from the stream file
    because it is evidence about what the run did, not about what it was pointed at: a run
    that replayed the wrong stream, or dropped events, gets a different digest.

    A log that exists and finished but recorded no detections at all -- a seed the frozen
    search policy legitimately never fires on -- digests the empty sequence, so arms still
    agree. Only a missing or unreadable log yields ``None``.
    """
    target = Path(path)
    if not target.exists():
        return None
    rows: list[tuple[Any, ...]] = []
    for record in _records(target):
        if record.get("event_type") != "detection":
            continue
        detail = record.get("detail") or {}
        position = detail.get("grid_position")
        rows.append(
            (
                round(float(record.get("timestamp") or 0.0), 6),
                str(record.get("target_id")),
                str(record.get("source_event_id")),
                tuple(position) if isinstance(position, (list, tuple)) else None,
            )
        )
    canonical = json.dumps(rows, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
