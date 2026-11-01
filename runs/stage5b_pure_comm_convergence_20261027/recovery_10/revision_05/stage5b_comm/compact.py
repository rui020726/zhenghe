"""Bounded writers: exclusive JSON with a size cap, gzip JSONL for per-episode streams."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import time
from pathlib import Path

from . import constants as C


def dumps(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_bytes(payload):
    """The exact bytes a payload will occupy; callers hash these before writing."""
    return dumps(payload).encode("utf-8")


def write_bytes_exclusive(path, data, cap_bytes=None):
    """Write exact bytes (no newline translation) and refuse to replace a file."""
    cap = C.FILE_SIZE_CAP_BYTES if cap_bytes is None else int(cap_bytes)
    if len(data) > cap:
        raise RuntimeError("%s would be %d bytes, over the %d byte cap"
                           % (path, len(data), cap))
    with Path(path).open("xb") as handle:
        handle.write(data)
    return Path(path)


def read_bytes(path):
    return Path(path).read_bytes()


def write_json_exclusive(path, payload):
    return write_bytes_exclusive(path, json_bytes(payload))


def write_json_checked(path, payload):
    """Overwrite (only inside the run root) with the same size guard."""
    text = dumps(payload)
    if len(text.encode("utf-8")) > C.FILE_SIZE_CAP_BYTES:
        raise RuntimeError("%s would exceed the 10 MB cap" % (path,))
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return Path(path)


def write_gz_jsonl(path, rows):
    with gzip.open(str(path), "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return Path(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gzip_bytes(data):
    """Deterministic gzip bytes: no filename, mtime 0."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as handle:
        handle.write(data)
    return buffer.getvalue()


def archive_world_logs(layout, directory):
    """Gzip exactly the two new-world event logs, prove round-trip, then remove plaintext.

    Refuses anything outside this run's world_logs, refuses other file names, and refuses to
    touch an existing .gz or provenance file. On any failure the plaintext originals stay and
    the partial archives are removed; no shell command is used.
    """
    names = ("lifecycle.jsonl", "relay_transcript.jsonl")
    base = Path(layout.world_logs).resolve()
    target = Path(directory).resolve()
    if target != base and base not in target.parents:
        raise RuntimeError("%s is outside this run's world_logs %s" % (target, base))
    present = sorted(path.name for path in target.iterdir() if path.is_file())
    if present != sorted(names):
        raise RuntimeError("%s holds %r, expected exactly %r"
                           % (target, present, sorted(names)))
    record = {"schema": "stage5b-world-log-provenance/1", "directory": str(target),
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "files": [], "reversible": True,
              "note": "plaintext removed only after a verified byte-identical gzip round trip"}
    created = []
    try:
        for name in names:
            source = target / name
            if source.is_symlink() or not source.is_file():
                raise RuntimeError("%s is not a regular file" % (source,))
            data = source.read_bytes()
            packed = gzip_bytes(data)
            archive = target / (name + ".gz")
            if archive.exists():
                raise RuntimeError("refusing to overwrite %s" % (archive,))
            with archive.open("xb") as handle:
                handle.write(packed)
            created.append(archive)
            back = gzip.decompress(archive.read_bytes())
            if back != data:
                raise RuntimeError("%s does not round-trip" % (name,))
            source_sha = hashlib.sha256(data).hexdigest()
            packed_sha = hashlib.sha256(packed).hexdigest()
            if hashlib.sha256(back).hexdigest() != source_sha:
                raise RuntimeError("%s round-trip sha differs" % (name,))
            record["files"].append({
                "name": name,
                "original": {"path": str(source), "bytes": len(data), "sha256": source_sha},
                "gzip": {"path": str(archive), "bytes": len(packed), "sha256": packed_sha},
                "roundtrip": "BYTE_IDENTICAL",
            })
        provenance = target / "log_provenance.json"
        if provenance.exists():
            raise RuntimeError("refusing to overwrite %s" % (provenance,))
        with provenance.open("xb") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, indent=2,
                                    sort_keys=True).encode("utf-8") + b"\n")
        created.append(provenance)
        for name in names:
            (target / name).unlink()
    except BaseException:
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    record["provenance_path"] = str(target / "log_provenance.json")
    return record


def iter_rows(directory):
    """All compact per-episode rows, sorted by file name for stable aggregation."""
    for path in sorted(Path(directory).glob("*.json")):
        yield read_json(path)

