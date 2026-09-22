"""Register the code freeze every training episode must be able to point at.

Provenance §2.3-2.5 in executable form. The freeze is not a tag or a promise in a document: it is a
commit SHA plus the blob id of every module the training entry actually imports, and every launch
re-derives that list and refuses to start if the working tree, HEAD, or the set of modules itself
has moved. That is the only way the mistake this round exists to close -- a hand-kept list of files
that named a module which has never existed -- stays closed: the list is produced by asking Python
what it loaded, so it cannot contain anything the run did not execute.

    python freeze.py --check            # report cleanliness, write nothing
    python freeze.py --register         # freeze HEAD and record every graph member's blob
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import retrain_common as common
import train_retrain  # noqa: F401  (importing it is what makes the graph this round's graph)


def graph() -> list[str]:
    """The executed modules plus this directory's own drivers, deduplicated and sorted."""
    modules = common.executed_modules()
    local = common.HERE.relative_to(common.ROOT).as_posix()
    return sorted(set(modules) | {f"{local}/{name}" for name in common.round_scripts()})


def register(argv: list[str]) -> int:
    sha = common.git_out("rev-parse", "HEAD")
    branch = common.git_out("rev-parse", "--abbrev-ref", "HEAD")
    paths = graph()
    orphan = common.untracked_code(paths)
    if orphan:
        raise SystemExit(
            "refusing to freeze: these members of the execution graph are not committed, so the "
            f"SHA would not describe them: {orphan}"
        )
    payload: dict[str, Any] = {
        "document": "CODE_FREEZE",
        "schema": common.__name__ + "/1",
        "branch": branch,
        "sha": sha,
        "short": sha[:9],
        "subject": common.git_out("log", "-1", "--format=%s"),
        "committed_at": common.git_out("log", "-1", "--format=%ad", "--date=iso"),
        "registered_at": common.git_out("log", "-1", "--format=%ct"),
        "trainer_identity": common.setting(),
        "configuration_hash": train_retrain.configuration_hash(),
        "derivation": (
            "sys.modules after importing train_retrain and the archived sampler/trainer entries, "
            "unioned with this directory's own scripts"
        ),
        "modules": {path: blob for path, blob in common.blob_identity(paths).items()},
        "module_count": len(paths),
        "production_diff_since_fix_head": common.production_diff(),
        "rule": (
            "every launch re-derives the graph and requires: no committed or working-tree "
            "difference against this SHA for any member, no untracked member, and no module in the "
            "graph that this freeze did not record"
        ),
    }
    missing = sorted(
        path for path, ids in payload["modules"].items() if not ids["in_head"]
    )
    if missing:
        raise SystemExit(f"refusing to freeze: graph members missing from HEAD: {missing}")
    common.write_json(common.FREEZE_FILE, payload)
    print(
        json.dumps(
            {
                "sha": payload["sha"],
                "modules": payload["module_count"],
                "production_empty": payload["production_diff_since_fix_head"]["empty"],
                "clean_after_registration": common.freeze_cleanliness()["clean"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def check(argv: list[str]) -> int:
    payload = common.freeze()
    if not payload.get("registered"):
        print(json.dumps({"registered": False, "clean": False}))
        return 1
    paths = graph()
    cleanliness = common.freeze_cleanliness(paths)
    print(
        json.dumps(
            {
                "freeze_sha": payload["sha"],
                "graph_members": len(paths),
                "clean": cleanliness["clean"],
                "detail": cleanliness,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0 if cleanliness["clean"] else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.register:
        return register(argv)
    return check(argv)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
