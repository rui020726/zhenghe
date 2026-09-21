"""Is 900000..900199 really unseen by this whole project? Audit before recording.

The claim under audit is the one round 3 got burned on: a block registered as fresh is only
fresh if nothing in the project has ever *run* one of its seeds. Round 1 registered
``500000..500399`` as an independent test and round 3 found seed 500168 inside the
membership-consistency work, so the label had to come off. This round's block is an
*engineering* block, not a learning test set, but the same rule applies: it must not overlap
anything already executed, declared or recorded.

Three sources of evidence, artefacts winning over declarations, because an executed episode
cannot be un-seen:

* every episode artefact on disk under ``runs/**`` (``rp_``/``tr_``/``ct_`` ... ``_s<seed>.jsonl``)
  and every recorded detection stream (``s<seed>.json``);
* every seed range any round's own scripts declare (``range(...)`` with a lower bound >= 1000;
  smaller ranges are loop counters);
* stream contents: two files named ``s<N>.json`` with different bytes are a naming collision,
  not a leak, and are reported rather than silently merged.

The verdict is mechanical: if the candidate block intersects any of the three, the audit fails
and the round stops -- the brief forbids swapping in a different block after looking.

    python block_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "runs"

#: This round's own directory. Its declarations are this round's pre-registration and its
#: artefacts do not exist yet when the audit runs; neither counts as contamination, but both
#: are reported so a reader can see the difference.
OWN_ROUND = HERE.name

#: The candidate engineering block: registered here, generated only after a clean audit.
CANDIDATE_BLOCK = (900000, 900200)

#: The sealed learning final test block. Registered, never generated, never read.
SEALED_BLOCK = (800000, 800400)

#: ``rp_``/``tr_``/``ct_``-prefixed lifecycle, decision-trace and transcript files end in
#: ``_s<seed>.jsonl``; recorded detection streams are ``s<seed>.json``.
ARTEFACT = re.compile(r"(?:_s|^s)(\d+)\.jsonl?$")
STREAM_FILE = re.compile(r"^s(\d+)\.jsonl?$")
RANGE_DECLARATION = re.compile(r"range\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def artefact_seeds() -> dict[int, set[str]]:
    """Every scenario seed with an executed artefact on disk, mapped to the rounds that ran it."""
    used: dict[int, set[str]] = defaultdict(set)
    for path in RUNS.rglob("*.json*"):
        if path.is_dir():
            continue
        match = ARTEFACT.search(path.name)
        if match is None:
            continue
        round_name = path.relative_to(RUNS).parts[0]
        used[int(match.group(1))].add(round_name)
    return used


def declared_blocks() -> dict[str, list[dict[str, Any]]]:
    """The seed ranges each round's own scripts name, straight from the source."""
    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for script in sorted(RUNS.glob("*/**/*.py")):
        round_name = script.relative_to(RUNS).parts[0]
        text = script.read_text(encoding="utf-8", errors="replace")
        for start, stop in RANGE_DECLARATION.findall(text):
            low = int(start)
            if low < 1000:
                # Small ranges are loop counters and index arithmetic, not scenario blocks.
                continue
            blocks[round_name].append(
                {
                    "start": low,
                    "stop": int(stop),
                    "file": str(script.relative_to(ROOT)),
                }
            )
    return blocks


def stream_files() -> dict[int, list[Path]]:
    """Every recorded detection stream on disk, by seed number."""
    found: dict[int, list[Path]] = defaultdict(list)
    for path in RUNS.rglob("*.json*"):
        if path.is_dir():
            continue
        match = STREAM_FILE.fullmatch(path.name)
        if match is None:
            continue
        found[int(match.group(1))].append(path)
    return found


def stream_content_collisions(
    seeds: set[int], streams: dict[int, list[Path]]
) -> dict[str, Any]:
    """Same-numbered stream files with different content: naming collisions, not leaks."""
    collisions: dict[str, Any] = {}
    for seed in sorted(seeds):
        paths = streams.get(seed, [])
        if len(paths) < 2:
            continue
        by_digest: dict[str, list[str]] = defaultdict(list)
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            by_digest[digest].append(str(path.relative_to(RUNS)))
        if len(by_digest) > 1:
            collisions[str(seed)] = {
                digest: locations for digest, locations in sorted(by_digest.items())
            }
    return collisions


def manifest_seed_mentions(low: int, high: int) -> list[dict[str, Any]]:
    """Seed numbers the candidate block's range mentions inside any manifest JSON.

    Declarations do not only live in code: rounds register their blocks in manifests and
    protocol documents' machine-readable twins. A six-digit number inside the candidate
    range appearing as a JSON *value* is therefore reported with its path, so a block that
    some round already promised is caught even before an episode exists. Only integers are
    collected -- hex digests and byte counts are not seed identity.
    """
    mentions: list[dict[str, Any]] = []

    def walk(node: Any, path: str, where: Path) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, int) and low <= node < high:
            mentions.append(
                {"file": str(where.relative_to(RUNS)), "path": path, "seed": node}
            )
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}", where)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]", where)

    for path in sorted(RUNS.rglob("*.json")):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        # This round's own documents name the candidate block by design; they are the
        # pre-registration, not a prior claim on the block.
        if path.relative_to(RUNS).parts[0] == OWN_ROUND:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        walk(payload, "$", path)
    return mentions


def audit(candidate: tuple[int, int]) -> dict[str, Any]:
    used = artefact_seeds()
    blocks = declared_blocks()
    streams = stream_files()
    candidate_block = set(range(*candidate))
    sealed_block = set(range(*SEALED_BLOCK))

    # 1. Executed artefacts inside the candidate block.
    artefact_hits = {
        seed: sorted(used[seed]) for seed in sorted(candidate_block & set(used))
    }
    # 2. Declared ranges that cover any candidate seed, from every round except this one.
    declaration_hits: list[dict[str, Any]] = []
    for round_name, entries in sorted(blocks.items()):
        if round_name == OWN_ROUND:
            continue
        for entry in entries:
            covered = sorted(candidate_block & set(range(entry["start"], entry["stop"])))
            if covered:
                declaration_hits.append(
                    {
                        "round": round_name,
                        "file": entry["file"],
                        "declared": [entry["start"], entry["stop"]],
                        "covered_candidate_seeds": [covered[0], covered[-1]],
                        "count": len(covered),
                    }
                )
    # 3. Recorded streams inside the candidate block.
    stream_hits = {
        seed: [str(path.relative_to(RUNS)) for path in streams[seed]]
        for seed in sorted(candidate_block & set(streams))
    }
    # 4. Manifest documents that already name a seed inside the candidate block.
    manifest_hits = manifest_seed_mentions(candidate[0], candidate[1])
    # Streams that merely share a number with the block but live elsewhere are the collision
    # case; only in-block files count as overlap, the rest is reported for transparency.
    collisions = stream_content_collisions(candidate_block, streams)

    sealed_artefacts = sorted(sealed_block & set(used))
    sealed_streams = sorted(sealed_block & set(streams))

    clean = not artefact_hits and not declaration_hits and not stream_hits and not manifest_hits
    return {
        "candidate_block": {
            "block": [candidate[0], candidate[1] - 1],
            "size": len(candidate_block),
            "purpose": (
                "engineering confirmation of locked communication rules only; it may never be "
                "called an unseen test set for a learned model"
            ),
        },
        "sources": {
            "artefact_seeds_on_disk": len(used),
            "episode_artefacts_on_disk": len(list(RUNS.rglob("rp_*.jsonl"))),
            "recorded_streams_on_disk": len(streams),
            "declared_rounds": sorted(blocks),
            "note": (
                "artefacts win over declarations: an executed episode cannot be un-seen. This "
                "round's own pre-registration is excluded from the contamination set and listed "
                "under own_round."
            ),
        },
        "overlap": {
            "executed_artefacts": artefact_hits,
            "declared_by_other_rounds": declaration_hits,
            "recorded_streams": stream_hits,
            "manifest_seed_mentions": manifest_hits,
            "stream_content_collisions": collisions,
        },
        "clean": clean,
        "verdict": (
            "zero overlap: the block may be registered and its 200 streams recorded"
            if clean
            else "OVERLAP FOUND: stop and report; do not swap in a different seed block"
        ),
        "sealed_learning_block": {
            "block": [SEALED_BLOCK[0], SEALED_BLOCK[1] - 1],
            "artefacts_on_disk": sealed_artefacts,
            "streams_on_disk": sealed_streams,
            "generated": bool(sealed_artefacts or sealed_streams),
            "read": False,
            "clean": not (sealed_artefacts or sealed_streams),
        },
        "own_round": {
            "name": OWN_ROUND,
            "declared_blocks": blocks.get(OWN_ROUND, []),
            "note": "this round's own pre-registration; not contamination",
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=CANDIDATE_BLOCK[0])
    parser.add_argument("--stop", type=int, default=CANDIDATE_BLOCK[1])
    args = parser.parse_args()
    result = audit((args.start, args.stop))
    path = HERE / "COMM_RULE_BLOCK_AUDIT.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"written: {path}")
    if not result["clean"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
