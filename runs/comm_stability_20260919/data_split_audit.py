"""Which scenario seeds has this project already looked at, and what can still be called new?

The claim being audited is narrow and important: a block of seeds registered as an independent
final test is only independent if nothing in the project has ever *run* one of them. Round 1
registered ``500000..500399`` that way, and round 3 found the defect -- seed 500168 appears in
the membership-consistency work too, so the block is not clean and the label has to come off
before any result is read from it.

Two sources of evidence, because a manifest only records what someone remembered to write:

* artefacts on disk. Every episode this system has ever run left a file named
  ``rp_<label>_s<seed>.jsonl`` (and its transcript/trace siblings), so the filename list is the
  union of everything actually executed, whether or not a document mentions it;
* declared blocks in each round's own scripts, which is the pre-registration side -- what a
  round said it would use.

Where the two disagree the artefacts win: an executed episode cannot be un-seen. A declared
block with no artefacts is a plan, and is reported as such.
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

#: ``rp_``/``tr_``/``ct_`` prefixed lifecycle, decision-trace and communication transcripts all
#: end in ``_s<seed>.jsonl``; the recorded detection streams are ``s<seed>.json``.
ARTEFACT = re.compile(r"(?:_s|^s)(\d+)\.jsonl?$")
RANGE_DECLARATION = re.compile(r"range\(\s*(\d+)\s*,\s*(\d+)\s*\)")
#: The block round 1 registered as the independent final test, and the specific seed that
#: motivated this audit.
PREVIOUS_TEST_BLOCK = (500000, 500400)
SUSPECT_SEEDS = (500168,)
#: The replacement candidate, chosen by the arithmetic of the blocks already spoken for and not
#: by peeking: the lowest 100 000-aligned 400-seed block with no artefact and no declared
#: reference anywhere under runs/. Everything above seed 700071 is untouched, and the blocks
#: below it -- 10000s, 20000s, 30000s, 500000s, 600000s, 700000s -- are all spoken for.
CANDIDATE_BLOCK = (800000, 800400)


def artefact_seeds() -> dict[int, set[str]]:
    """Every scenario seed with an executed episode on disk, mapped to the rounds that ran it."""
    used: dict[int, set[str]] = defaultdict(set)
    for path in RUNS.rglob("*.json*"):
        match = ARTEFACT.search(path.name)
        if match is None or path.is_dir():
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
            blocks[round_name].append({"start": low, "stop": int(stop), "file": str(script.relative_to(ROOT))})
    return blocks


def _signatures(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The run signature and the stream stamp, wherever this round's format kept them."""
    detail = record.get("detail") if isinstance(record.get("detail"), dict) else {}
    signature = record.get("run_signature") or detail.get("run_signature") or {}
    stream = signature.get("stream") or {}
    configuration = record.get("configuration") or detail.get("configuration") or {}
    # The keys name the entry fields they fill in, so the caller can copy them straight across.
    return signature, {
        "arm_labels": signature.get("arm"),
        "code_commits": signature.get("code_commit"),
        "stream_sha256": stream.get("sha256"),
        "steps": configuration.get("steps"),
    }


def configurations_for(seeds: set[int]) -> dict[str, dict[str, Any]]:
    """For the seeds in question, what each executed episode actually replayed.

    The run identity is read from the first record, which is where the harness writes it -- older
    rounds nested it under ``detail``, newer ones beside it. What a reader needs is not a hash of
    a dictionary but the three things that make two episodes of "the same seed" different
    scenarios: which arm, which code, and which stream content.
    """
    seen: dict[str, dict[str, Any]] = {}
    for path in RUNS.rglob("rp_*.jsonl"):
        round_name = path.relative_to(RUNS).parts[0]
        match = re.search(r"_s(\d+)\.jsonl$", path.name)
        if match is None or int(match.group(1)) not in seeds:
            continue
        seed = int(match.group(1))
        key = f"{round_name}:{seed}"
        entry = seen.setdefault(
            key,
            {
                "round": round_name,
                "seed": seed,
                "episodes": 0,
                "arm_labels": set(),
                "code_commits": set(),
                "stream_sha256": set(),
                "steps": set(),
            },
        )
        entry["episodes"] += 1
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                signature, stamp = _signatures(json.loads(line))
                if stamp["arm_labels"] or stamp["stream_sha256"]:
                    for field, value in stamp.items():
                        if value is not None:
                            entry[field].add(str(value))
                    break
        if not entry["arm_labels"]:
            entry["arm_labels"].add("no-signature")
    return {
        key: {inner_key: sorted(value) for inner_key, value in sorted(entry.items()) if isinstance(value, set)}
        for key, entry in sorted(seen.items())
    }


def stream_contents() -> dict[str, Any]:
    """Every recorded stream file, grouped by seed: the same number need not be the same scenario."""
    by_seed: dict[int, dict[str, set[str]]] = defaultdict(dict)
    for path in RUNS.rglob("s*.json*"):
        match = re.fullmatch(r"s(\d+)\.jsonl?", path.name)
        if match is None:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        by_seed[int(match.group(1))].setdefault(digest, set()).add(str(path.relative_to(RUNS).parts[0]))
    shared = {
        str(seed): sorted(blocks.values())
        for seed, blocks in by_seed.items()
        if len(blocks) > 1
    }
    return {
        "seeds_with_recorded_streams": len(by_seed),
        "seeds_recorded_under_more_than_one_content": shared,
        "note": (
            "two files named s<N>.json are the same scenario only if their content matches. Where "
            "they do not, the seed number was reused across differently generated blocks and the "
            "collision is a naming collision, not a data leak."
        ),
    }


def audit(candidate: tuple[int, int], previous: tuple[int, int], *, with_contents: bool = False) -> dict[str, Any]:
    used = artefact_seeds()
    blocks = declared_blocks()
    previous_block = set(range(*previous))
    candidate_block = set(range(*candidate))
    contaminated = {seed: sorted(used[seed]) for seed in sorted(previous_block & set(used))}
    suspects = {
        str(seed): sorted(used.get(seed, [])) for seed in SUSPECT_SEEDS
    }
    return {
        "scope": {
            "walked": str(RUNS.relative_to(ROOT)) + "/**",
            "episodes_on_disk": sum(len(paths) for paths in [list(RUNS.rglob("rp_*.jsonl"))]),
            "distinct_scenario_seeds": len(used),
            "note": (
                "integrated-system scenario seeds only. PILOC and task_allocation training "
                "seeds index a different random process (map generation and network "
                "initialisation) and are not scenario identity."
            ),
        },
        "previously_registered_final_test": {
            "block": [previous[0], previous[1]],
            "declared_by": "round 1 (feat/communication-control-baselines)",
            "seeds_with_executed_episodes": len(contaminated),
            "size": len(previous_block),
            "verdict": (
                "not independent -- episodes exist on disk for the listed seeds, so a final test "
                "may not be scored on this block and call itself unseen"
            )
            if contaminated
            else "clean",
            "contaminated_seeds": contaminated,
            "suspect_seeds": suspects,
        },
        "declared_blocks_per_round": blocks,
        "stream_content_collisions": stream_contents() if with_contents else None,
        "replacement": {
            "block": [candidate[0], candidate[1]],
            "size": len(candidate_block),
            "used_seeds_inside_block": sorted(candidate_block & set(used)),
            "clean": not (candidate_block & set(used)),
            "selection_rule": (
                "the lowest 100 000-aligned 400-seed block with no artefact and no declared "
                "reference anywhere under runs/; chosen before any episode in it was run, and "
                "nothing in this round reads it"
            ),
            "status": "registered, not generated, not evaluated",
        },
        "splits_this_round_keeps": {
            "train": [700000, 700048],
            "validation": [700048, 700072],
            "note": (
                "both blocks keep their round-2 purpose. The 24 validation seeds may be "
                "diagnosed on as many times as needed; they are not an independent test, and "
                "calling them one after seeing results is the error this file exists to catch."
            ),
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-start", type=int, default=CANDIDATE_BLOCK[0])
    parser.add_argument("--candidate-stop", type=int, default=CANDIDATE_BLOCK[1])
    parser.add_argument("--with-configurations", action="store_true", help="read the config hash of every episode in the contaminated block")
    args = parser.parse_args(argv)
    result = audit(
        (args.candidate_start, args.candidate_stop), PREVIOUS_TEST_BLOCK, with_contents=args.with_configurations
    )
    if args.with_configurations:
        contaminated = {int(seed) for seed in result["previously_registered_final_test"]["contaminated_seeds"]}
        result["previously_registered_final_test"]["executed_configurations"] = configurations_for(contaminated)
    path = HERE / "COMM_DATA_SPLIT_AUDIT.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "declared_blocks_per_round"}, ensure_ascii=False, indent=1))
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
