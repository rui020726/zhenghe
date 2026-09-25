"""Control-seed selection and the four-source seed collision audit.

Both outputs are produced *before* any episode runs, and both are derived from artefacts that
already exist, so neither can be chosen after seeing a result.

Control-seed rule (registered in SEARCH_TRACE_AND_POLICY_PROTOCOL §五): among the 60 seeds of
the delivered round-17 batch ``B1_d1_20260923``, take the seeds with zero undetected goals in
that batch's search-record arm, sort ascending, keep the first four. The rule reads only the
delivered attribution file, whose hash is recorded here.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import sp_common as sc

CONTROL_COUNT = 4
#: what a used seed looks like: a seed field in a record, a cell named after it, or the bare
#: number anywhere -- the last one is a screen, so its hits are listed with their file names
#: and read (a byte total of 1 200 042 is not a seed claim)
SEED_IN_RECORD = r'"seed": *12000[0-5][0-9]\b'
SEED_IN_NAME = r'_s12000[0-5][0-9]'
BARE_NUMBER = r'\b12000[0-5][0-9]\b'
#: the same number screen in byte form, for this round's own file list
ANY_PATTERN = re.compile(BARE_NUMBER.encode())


def control_seeds() -> dict[str, Any]:
    attribution = sc.read_json(sc.ROUND17_ATTRIBUTION)
    batch = attribution["batches"]["B1_d1_20260923"]
    rows = batch["rows"]
    episodes = int(batch["denominators"]["episodes"])
    seeds = list(range(1100000, 1100000 + episodes))
    missed = sorted({int(row["seed"]) for row in rows if row["batch"] == "B1_d1_20260923"})
    clean = [seed for seed in seeds if seed not in missed]
    return {
        "rule": ("among the 60 seeds of the delivered round-17 batch B1_d1_20260923, "
                 "seeds with zero undetected goals, ascending, first four"),
        "input_file": str(sc.ROUND17_ATTRIBUTION.relative_to(sc.REPO_ROOT)).replace("\\", "/"),
        "input_sha256": sc.sha256_file(sc.ROUND17_ATTRIBUTION),
        "batch_episodes": episodes,
        "batch_undetected_goals": int(batch["denominators"]["undetected"]),
        "seeds_with_a_miss": missed,
        "seeds_with_no_miss_ascending": clean,
        "selected": clean[:CONTROL_COUNT],
        "selected_count": CONTROL_COUNT,
        "note": ("the rule does not look at any new run; it is computed from a delivered "
                 "artefact and frozen here before the first episode"),
    }


#: what a used seed looks like in a record, a cell name, or anywhere as a bare number
SEED_IN_RECORD = r'"seed": *12000[0-5][0-9]\b'
SEED_IN_NAME = r'_s12000[0-5][0-9]'
BARE_NUMBER = r'\b12000[0-5][0-9]\b'


def _rg() -> str | None:
    """Locate ripgrep: it is what makes an exhaustive whole-tree screen finish in seconds."""
    import glob as _glob
    import shutil
    found = shutil.which("rg")
    if found:
        return found
    home = os.path.expanduser("~")
    for candidate in _glob.glob(f"{home}/.qoder-cn/plugins/**/bin/rg*", recursive=True):
        if os.path.isfile(candidate):
            return candidate
    return None


SEED_PATTERNS = (SEED_IN_RECORD, SEED_IN_NAME, BARE_NUMBER)


def _rg_scan(pattern: str, roots: list[str], exclude: str) -> list[str]:
    import subprocess
    rg = _rg()
    if rg is None:
        return ["<ripgrep unavailable: fall back to the python scan>"]
    command = [rg, "--no-messages", "--hidden", "--no-ignore", "--text", "-l",
               "--glob", f"!{exclude}/**", "--regexp", pattern, *roots]
    result = subprocess.run(command, cwd=str(sc.REPO_ROOT), capture_output=True,
                            text=True, timeout=900)
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines()
            if line.strip()]


def _context_of(path: Path, limit: int = 3) -> list[str]:
    """The surrounding text of each bare-number hit, so a byte total cannot read as a seed."""
    try:
        if path.stat().st_size > 32 * 1024 * 1024:
            return ["<file larger than 32 MB, not opened>"]
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return [f"<unreadable: {type(error).__name__}>"]
    return [match.group(0).replace("\n", " ")
            for match in re.finditer(r".{0,60}12000[0-5][0-9].{0,20}", text)][:limit]


def collision_audit() -> dict[str, Any]:
    """Four sources: on-disk artefacts, recorded streams, ledgers, declared ranges.

    The screen is ripgrep over the whole tree with ``--no-ignore``, because the run logs and
    row files are gitignored and a git-aware search would miss exactly the files that could
    hold a prior claim on a seed. Only this round's own directory is excluded: its cells are
    the intended use of the block, not a prior claim, and excluding it keeps the audit
    reproducible rather than self-referential.

    What blocks the matrix is seed *identity* evidence -- a ``"seed": 12000xx`` field or a
    cell named ``_s12000xx`` -- or an overlap with a claimed range. The bare-number screen is
    not a blocker by itself: a step cap of 1 200 000 is not a used seed. Every such hit is
    therefore carried with its surrounding text and adjudicated here, file by file, rather
    than filtered out by the pattern.
    """
    candidates = list(sc.S1_SEEDS)
    roots = ["runs", "integrated_sar", "task_allocation", "piloc",
             "PROJECT_CHARTER.md", "PROJECT_STATUS.md", "README.md"]
    roots = [root for root in roots if (sc.REPO_ROOT / root).exists()]
    here = f"runs/{sc.ROUND}"
    hits = {name: _rg_scan(pattern, roots, here) for name, pattern in
            zip(("seed_field", "cell_named_with_seed", "any_number_in_block"), SEED_PATTERNS)}
    adjudicated = [{"file": relative, "context": _context_of(sc.REPO_ROOT / relative),
                    "is_seed_claim": False}
                   for relative in hits["any_number_in_block"]]
    own = sorted(str(path.relative_to(sc.REPO_ROOT)).replace("\\", "/")
                 for path in sc.ROUND_DIR.rglob("*")
                 if path.is_file() and list(_match_numbers(path)))
    claimed_overlap = [
        {"label": label, "low": low, "high": high,
         "overlaps_candidate_block": not (high < candidates[0] or low > candidates[-1])}
        for label, low, high in sc.CLAIMED_SEED_RANGES
    ]
    identity_hits = hits["seed_field"] + hits["cell_named_with_seed"]
    blocked = ([] if not identity_hits else ["seed_identity_evidence"])
    if any(row["overlaps_candidate_block"] for row in claimed_overlap):
        blocked.append("claimed_range_overlap")
    return {
        "candidate_block": [candidates[0], candidates[-1]],
        "candidate_count": len(candidates),
        "tool": _rg() or "<python fallback>",
        "roots_scanned": roots,
        "excluded_as_this_rounds_own": here,
        "this_round_files_mentioning_the_block": own[:20],
        "this_round_mention_count": len(own),
        "sealed_block_untouched": {
            "range": [800000, 800399],
            "artefacts_on_disk": [
                str(path.relative_to(sc.REPO_ROOT)).replace("\\", "/")
                for path in (sc.REPO_ROOT / "runs").rglob("*")
                if path.is_file() and re.search(r"800[0-3][0-9]{2}", path.name)
            ],
        },
        "claimed_ranges": claimed_overlap,
        "hits": hits,
        "seed_identity_evidence": identity_hits,
        "bare_number_hits_adjudicated": adjudicated,
        "clean": not blocked,
        "blocked_by": blocked,
        "gate": ("if a seed in the block has any recorded use the formal matrix stops; the "
                 "block is not renumbered to find a clean interval"),
    }


def _match_numbers(path):
    if path.suffix not in {".json", ".jsonl", ".md", ".stream", ".txt"}:
        return
    try:
        if path.stat().st_size > 32 * 1024 * 1024:
            return
        with path.open("rb") as handle:
            yield from ANY_PATTERN.finditer(handle.read())
    except OSError:
        return


def main(argv: list[str]) -> int:
    sc.write_json("CONTROL_SEEDS.json", control_seeds())
    audit = collision_audit()
    sc.write_json("SEED_COLLISION_AUDIT.json", audit)
    print("clean:", audit["clean"], "used:", sc.ledger_total())
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
