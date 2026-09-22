"""Have the four pressure blocks ever been touched, claimed or named anywhere in this project?

The claim under audit is stronger than round 7's. That round had one new block; this round has
four, one per task-pressure layer, and the brief forbids swapping a block after looking -- so
"clean" must be established from four independent sources *before* a single stream is recorded:

1. **Git**: tracked paths, tracked file contents at HEAD, and commit messages on any branch.
2. **Manifests / registrations**: every ``*.json`` under ``runs/`` read as text, keeping the
   integer tokens that fall inside a candidate block (a seed registered as a JSON value counts
   even when no episode artefact survived).
3. **On-disk artefacts**: every path under ``runs/`` whose *name* carries a scenario seed. The
   47 GB of per-episode JSONL is never opened -- it is named after its seed, so the name is the
   evidence. Two files that share a seed number but differ in bytes are reported as a naming
   collision, which is its own kind of conflict.
4. **Documentation claims**: every ``*.md`` in the repository, both as bare numbers and as
   declared ranges (``800000..800399``), because a range some round promised conflicts even if
   nothing was ever generated for it.

The verdict is mechanical: any conflict in sources 1-4 inside a candidate block fails the audit
and the round stops. This script never renumbers a block to make itself pass. One classification
is explicit: the brief itself names these four blocks, so a mention inside
``PRE_REGISTRATION_FILES`` is recorded as this round's own pre-registration rather than as a
conflict -- and that list is a named constant echoed into the output, never widened silently.

    python seed_audit.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "runs"

#: This round's own directory: its declarations are the pre-registration, not contamination.
OWN_ROUND = HERE.name

#: num_goals -> candidate episode seeds, from the fixtures this script does not import (the audit
#: must be readable on its own, before any frozen module is trusted).
BLOCKS: dict[int, tuple[int, ...]] = {
    6: tuple(range(1_000_000, 1_000_016)),
    8: tuple(range(1_000_100, 1_000_116)),
    10: tuple(range(1_000_200, 1_000_216)),
    12: tuple(range(1_000_300, 1_000_316)),
}

#: Blocks the project has already promised elsewhere, read out of the charter's own words. A
#: candidate that intersects one of these is not fresh even if the disk is empty, and the sealed
#: learning block must stay ungenerated.
DECLARED_BLOCKS: dict[str, tuple[int, int]] = {
    "pilot fitting": (3_000, 3_012),
    "independent test (round 1)": (500_000, 500_400),
    "live-system block": (600_000, 600_060),
    "communication training/dev block": (700_000, 700_072),
    "sealed learning test (never generated)": (800_000, 800_400),
    "engineering confirmation block (round 7, consumed)": (900_000, 900_200),
}

#: Scenario seeds are six digits or more in this project; smaller numbers are loop counters.
MIN_SEED_DIGITS = 6

#: The brief itself assigns these four blocks, so its own words are a pre-registration and not a
#: prior claim on the data. The list is explicit and every file in it is reported with its commit:
#: a mention of a candidate seed anywhere else is a conflict, and this list is never widened to
#: let one through.
PRE_REGISTRATION_FILES = ("COMM_TASK_PRESSURE_LEARNABILITY_NEXT_ROUND.md",)

SEED_ALTERNATION = "|".join(str(seed) for seeds in BLOCKS.values() for seed in seeds)
BARE_SEED = re.compile(rf"(?<!\d)(?:{SEED_ALTERNATION})(?!\d)")
SEED_IN_NAME = re.compile(r"(?:_s|^s)(\d{6,})\.jsonl?$")
STREAM_IN_NAME = re.compile(r"^s(\d{6,})\.json$")
RANGE_IN_TEXT = re.compile(r"(?<!\d)(\d{6,})\s*(?:\.\.\.|\.\.|-|,)\s*(\d{6,})(?!\d)")
RANGE_IN_CODE = re.compile(r"range\(\s*(\d{6,})\s*,\s*(\d{6,})\s*\)")


def git(*arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode not in (0, 1):  # git grep exits 1 for "no matches"
        raise RuntimeError(f"git {' '.join(arguments)} failed: {result.stderr[:200]}")
    return result.stdout


def own_candidate_seeds() -> set[int]:
    return {seed for seeds in BLOCKS.values() for seed in seeds}


# ---------------------------------------------------------------- source 1: git
def git_paths() -> list[dict[str, Any]]:
    """Tracked files whose *name* carries a candidate seed."""
    hits: list[dict[str, Any]] = []
    for path in git("ls-files").splitlines():
        match = SEED_IN_NAME.search(Path(path).name) or STREAM_IN_NAME.match(Path(path).name)
        if match and int(match.group(1)) in own_candidate_seeds():
            hits.append({"file": path, "seed": int(match.group(1))})
    return hits


def git_contents() -> list[dict[str, Any]]:
    """Tracked file contents that name a candidate seed.

    Searched in the working tree, which the code freeze reports field by field as identical to
    HEAD; searching the index instead would silently miss a tracked file that differs from it.
    """
    hits: list[dict[str, Any]] = []
    output = git(
        "grep",
        "-I",
        "-n",
        "-E",
        rf"(^|[^0-9])({SEED_ALTERNATION})([^0-9]|$)",
        "--",
        f":(exclude)runs/{OWN_ROUND}",
    )
    for line in output.splitlines():
        file_name, _, rest = line.partition(":")
        number, _, text = rest.partition(":")
        seeds = sorted({int(value) for value in BARE_SEED.findall(text)})
        if seeds:
            hits.append(
                {
                    "file": file_name,
                    "line": int(number) if number.isdigit() else None,
                    "seeds": seeds,
                    "excerpt": text.strip()[:160],
                }
            )
    return hits


def git_history() -> list[dict[str, Any]]:
    """Commit messages on any branch that promise a candidate seed."""
    hits: list[dict[str, Any]] = []
    for line in git("log", "--all", "--format=%H%x09%ad%x09%s", "--date=short").splitlines():
        sha, date, subject = line.split("\t", 2)
        seeds = sorted({int(value) for value in BARE_SEED.findall(subject)})
        if seeds:
            hits.append({"sha": sha[:9], "date": date, "seeds": seeds, "subject": subject[:160]})
    return hits


# ---------------------------------------------------------------- source 2: manifests
def manifest_contents(limit_bytes: int = 16_000_000) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Every ``*.json`` registration under ``runs/``, searched for candidate seeds as integers."""
    hits: list[dict[str, Any]] = []
    coverage = {"scanned": 0, "skipped_too_large": 0}
    for path in sorted(RUNS.rglob("*.json")):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        relative = path.relative_to(RUNS).parts[0] if path.relative_to(RUNS).parts else ""
        if relative == OWN_ROUND:
            continue
        size = path.stat().st_size
        if size > limit_bytes:
            coverage["skipped_too_large"] += 1
            continue
        coverage["scanned"] += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        seeds = sorted({int(value) for value in BARE_SEED.findall(text)})
        if seeds:
            hits.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "seeds": seeds,
                    "size_bytes": size,
                }
            )
    return hits, coverage


# ---------------------------------------------------------------- source 3: artefacts on disk
def artefact_names() -> tuple[dict[str, list[str]], dict[int, list[Path]], dict[str, int]]:
    """Seeds carried by file names under ``runs/`` -- the 47 GB of JSONL is never opened."""
    used: dict[int, set[str]] = defaultdict(set)
    streams: dict[int, list[Path]] = defaultdict(list)
    coverage = {"files_seen": 0}
    for path in RUNS.rglob("*"):
        if not path.is_file():
            continue
        coverage["files_seen"] += 1
        match = SEED_IN_NAME.search(path.name) or STREAM_IN_NAME.match(path.name)
        if match is None:
            continue
        seed = int(match.group(1))
        round_name = path.relative_to(RUNS).parts[0]
        used[seed].add(round_name)
        if STREAM_IN_NAME.match(path.name):
            streams[seed].append(path)
    by_round = {str(seed): sorted(rounds) for seed, rounds in sorted(used.items())}
    return by_round, dict(streams), coverage


def stream_content_collisions(seeds: set[int], streams: dict[int, list[Path]]) -> dict[str, Any]:
    """Same stream number, different bytes: a naming collision, not a leak, and still a stop."""
    import hashlib

    collisions: dict[str, Any] = {}
    for seed in sorted(seeds):
        paths = streams.get(seed, [])
        if len(paths) < 2:
            continue
        by_digest: dict[str, list[str]] = defaultdict(list)
        for path in paths:
            by_digest[hashlib.sha256(path.read_bytes()).hexdigest()[:16]].append(
                str(path.relative_to(ROOT))
            )
        if len(by_digest) > 1:
            collisions[str(seed)] = dict(sorted(by_digest.items()))
    return collisions


# ---------------------------------------------------------------- source 4: documentation
def documentation_claims() -> list[dict[str, Any]]:
    """Markdown that names a candidate seed or declares a range covering one."""
    hits: list[dict[str, Any]] = []
    candidates = own_candidate_seeds()
    for path in sorted(ROOT.rglob("*.md")):
        if OWN_ROUND in path.parts or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        seeds = {int(value) for value in BARE_SEED.findall(text)}
        for start, stop in RANGE_IN_TEXT.findall(text):
            low, high = sorted((int(start), int(stop)))
            if high - low > 500_000:
                continue  # a date range or a byte count, not a seed block
            seeds |= {seed for seed in candidates if low <= seed <= high}
        for start, stop in RANGE_IN_CODE.findall(text):
            seeds |= {seed for seed in candidates if int(start) <= seed < int(stop)}
        if seeds:
            hits.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "seeds": sorted(seeds)[:20],
                    "count": len(seeds),
                }
            )
    return hits


def declared_range_conflicts() -> list[dict[str, Any]]:
    """Blocks the project already owns, checked against the candidates arithmetically."""
    conflicts: list[dict[str, Any]] = []
    for goals, seeds in BLOCKS.items():
        for name, (low, high) in DECLARED_BLOCKS.items():
            overlap = sorted(set(seeds) & set(range(low, high)))
            if overlap:
                conflicts.append(
                    {
                        "goals": goals,
                        "declared_block": [low, high - 1],
                        "declared_as": name,
                        "overlapping_seeds": overlap[:10],
                        "count": len(overlap),
                    }
                )
    return conflicts


# ---------------------------------------------------------------- verdict
def audit() -> dict[str, Any]:
    blocks = BLOCKS
    global_candidates = own_candidate_seeds()
    selected = global_candidates

    paths = git_paths()
    contents = git_contents()
    history = git_history()
    manifests, manifest_coverage = manifest_contents()
    artefacts, artefact_streams, artefact_coverage = artefact_names()
    collisions = stream_content_collisions(selected, artefact_streams)
    documents = documentation_claims()
    declared = declared_range_conflicts()

    def in_block(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for hit in hits:
            seeds = set(hit.get("seeds") or [hit.get("seed")]) & global_candidates
            if not seeds:
                continue
            named = str(hit.get("file", hit.get("subject", "")))
            kept.append(
                {
                    **hit,
                    "seeds": sorted(seeds),
                    "classification": (
                        "own_pre_registration"
                        if Path(named).name in PRE_REGISTRATION_FILES
                        else "conflict"
                    ),
                }
            )
        return kept

    def conflicts(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [hit for hit in hits if hit["classification"] == "conflict"]

    artefact_hits = {
        seed: locations for seed, locations in artefacts.items() if int(seed) in global_candidates
    }
    stream_hits = {
        str(seed): [str(path.relative_to(ROOT)) for path in locations]
        for seed, locations in sorted(artefact_streams.items())
        if seed in global_candidates
    }

    sources = {
        "git": {
            "tracked_paths_with_seed": in_block(paths),
            "tracked_content_mentions": in_block(contents),
            "commit_message_mentions": in_block(history),
        },
        "manifests": {
            "json_registrations_mentioning_a_seed": in_block(manifests),
            "coverage": manifest_coverage,
        },
        "artefacts": {
            "executed_episode_artefacts": artefact_hits,
            "recorded_streams": stream_hits,
            "stream_content_collisions": collisions,
            "coverage": {**artefact_coverage, "note": "file names carry the seed; 47 GB of per-episode JSONL is never opened"},
        },
        "documentation": {
            "markdown_mentions_or_declares_a_candidate_seed": in_block(documents),
            "overlap_with_blocks_the_project_already_owns": declared,
        },
    }

    counted = {
        "git": (
            sources["git"]["tracked_paths_with_seed"]
            + sources["git"]["tracked_content_mentions"]
            + sources["git"]["commit_message_mentions"]
        ),
        "manifests": sources["manifests"]["json_registrations_mentioning_a_seed"],
        "documentation": sources["documentation"]["markdown_mentions_or_declares_a_candidate_seed"],
    }
    hits = {
        "git": len(conflicts(counted["git"])),
        "manifests": len(conflicts(counted["manifests"])),
        "artefacts": len(artefact_hits) + len(stream_hits) + len(collisions),
        "documentation": len(conflicts(counted["documentation"])) + len(declared),
    }
    pre_registration = {
        source: sorted({str(Path(str(hit.get("file", ""))).name) for hit in listed if hit["classification"] == "own_pre_registration"})
        for source, listed in counted.items()
    }
    clean = not any(hits.values())

    sealed = set(range(*DECLARED_BLOCKS["sealed learning test (never generated)"]))
    sealed_artefacts = sorted(seed for seed in sealed if str(seed) in artefacts)
    sealed_streams = sorted(seed for seed in sealed if str(seed) in artefact_streams)

    own = {
        "round": OWN_ROUND,
        "declared_blocks": {str(goals): [seeds[0], seeds[-1]] for goals, seeds in blocks.items()},
        "note": "this script and the protocol it ships with name the blocks by design",
    }

    return {
        "blocks_audited": {
            str(goals): {"seeds": [seeds[0], seeds[-1]], "count": len(seeds)}
            for goals, seeds in blocks.items()
        },
        "blocks_the_project_already_owns": {
            name: [low, high - 1] for name, (low, high) in DECLARED_BLOCKS.items()
        },
        "sources": sources,
        "hit_counts": hits,
        "classified_as_this_rounds_own_pre_registration": pre_registration,
        "pre_registration_files_allowed": list(PRE_REGISTRATION_FILES),
        "clean": clean,
        "verdict": (
            "zero overlap across all four sources: the blocks may be registered and recorded"
            if clean
            else "CONFLICT: stop, submit a protocol revision, do not renumber the block"
        ),
        "rule": (
            "an executed artefact cannot be un-seen, a registered seed cannot be un-claimed, and "
            "a declared range is a promise the project has already made. Any hit stops the round; "
            "the block is never moved to make the audit pass."
        ),
        "sealed_learning_block": {
            "block": [800_000, 800_399],
            "artefacts_on_disk": sealed_artefacts,
            "streams_on_disk": sealed_streams,
            "generated": bool(sealed_artefacts or sealed_streams),
            "read_by_this_round": False,
            "clean": not (sealed_artefacts or sealed_streams),
        },
        "own_round": own,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="SEED_COLLISION_AUDIT.json", help="where to write the audit"
    )
    args = parser.parse_args()
    result = audit()
    path = HERE / args.out
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "clean": result["clean"],
        "hit_counts": result["hit_counts"],
        "verdict": result["verdict"],
        "sealed_block_clean": result["sealed_learning_block"]["clean"],
        "written": args.out,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if result["clean"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
