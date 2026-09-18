"""Write down which commit the corrected-environment block is frozen at, from git itself.

Nothing here is typed by hand: the value is ``git rev-parse HEAD`` taken while the production
tree is clean, on the branch the block belongs to. The block refuses to run unless
``integrated_sar`` and ``task_allocation`` still match that commit, so the freeze is a claim
about files, not about a label.

    D:/project_env/python.exe runs/corrected_env_final_20260918/freeze.py
    ... freeze.py --check        # what the block calls; prints and exits non-zero on drift
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TARGET = HERE / "FREEZE.json"
BRANCH = "fix/task-membership-consistency"
# The code that executes. The test directory is excluded: a test file cannot change what a run
# does, and freezing on it would block honest documentation commits for no safety at all.
PRODUCTION = ("integrated_sar", "task_allocation", ":(exclude)integrated_sar/tests")
STATUS_PATHS = ("integrated_sar", "task_allocation", ":(exclude)integrated_sar/tests")


def git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def tree_is_clean() -> tuple[bool, str]:
    """What must hold before any commit may be named as the freeze."""
    dirty = git("status", "--porcelain", *STATUS_PATHS)
    if dirty:
        return False, "uncommitted production changes: " + dirty.replace("\n", "; ")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != BRANCH:
        return False, f"on branch {branch}, expected {BRANCH}"
    return True, branch


def check() -> tuple[bool, str]:
    """Is the production tree still the frozen one? True with the commit it matches."""
    clean, detail = tree_is_clean()
    if not clean:
        return False, detail
    if not TARGET.exists():
        return False, "no FREEZE.json: the block has not been registered"
    commit = json.loads(TARGET.read_text(encoding="utf-8"))["code_commit"]
    drift = git("diff", commit, "--", *PRODUCTION)
    if drift:
        return False, f"production differs from {commit} by {len(drift.splitlines())} diff lines"
    return True, commit


def main(argv: list[str]) -> int:
    if "--check" in argv:
        ok, detail = check()
        print(("frozen at " if ok else "REFUSED: ") + detail)
        return 0 if ok else 1
    clean, detail = tree_is_clean()
    if not clean:
        print(f"refusing to register a freeze: {detail}")
        return 1
    head = git("rev-parse", "HEAD")
    payload = {
        "code_commit": head,
        "branch": BRANCH,
        "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": (
            "the freeze is the commit that was HEAD while integrated_sar and task_allocation "
            "were clean; the block re-checks it before building anything"
        ),
        "verified_production_paths": list(PRODUCTION),
        "head_at_registration": head,
    }
    part = TARGET.with_suffix(".json.part")
    part.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(part, TARGET)
    print(f"registered {payload['code_commit']}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
