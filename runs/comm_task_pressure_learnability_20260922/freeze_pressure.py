"""Register and check the code freeze this layer's episodes must be produced by.

A round whose aggregation code changed while its episodes were on disk cannot say which code made
those numbers, so every launch records the freeze it ran under and the report refuses to pool rows
from two different freezes. The executed set is derived from ``sys.modules`` -- what actually got
imported -- not from a hand-written list, which is how round 10 caught a module in its own
manifest that neither exists nor is imported.

    python freeze_pressure.py --register
    python freeze_pressure.py --check
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pressure_common as common

#: The entry modules of this round: importing them pulls in everything the harness executes.
ENTRIES = (
    "pressure_common",
    "pressure_env",
    "record_layer",
    "pressure_matrix",
    "pressure_metrics",
    "judge_layers",
    "seed_audit",
)

#: Round scripts the fixtures import and therefore execute.
INHERITED = (
    "pilot_common",
    "pilot_env",
    "train_pilot",
    "protocol_common",
    "protocol_env",
    "tight_common",
    "attribution_metrics",
    "confirm_common",
    "confirm_env",
    "comm_metrics",
    "evaluate_pilot",
)

HERE_DIR = Path(__file__).resolve().parent


def executed_modules() -> dict[str, dict[str, Any]]:
    """Every module this round can execute, judged by git blob identity where it is tracked."""
    import os  # noqa: F401 - forces the stdlib entry into sys.modules before the scan

    for name in ENTRIES + INHERITED:
        importlib.import_module(name)
    root = str(common.ROOT)
    found: dict[str, dict[str, Any]] = {}
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        absolute = os.path.abspath(path)
        if "__pycache__" in absolute:
            continue
        if not absolute.startswith(root + os.sep):
            continue
        relative = os.path.relpath(absolute, root).replace("\\", "/")
        if relative.endswith(".py"):
            found[relative] = common.blob_identity(relative)
    return found


def graph() -> dict[str, Any]:
    modules = executed_modules()
    own_scripts = {
        str(path.relative_to(common.ROOT)).replace("\\", "/"): common.blob_identity(
            str(path.relative_to(common.ROOT)).replace("\\", "/")
        )
        for path in sorted(HERE_DIR.glob("*.py"))
    }
    members = {**modules, **own_scripts}
    dirty = sorted(
        relative
        for relative, identity in members.items()
        if identity.get("exists") and not identity.get("matches_head")
    )
    absent = sorted(relative for relative, identity in members.items() if not identity.get("in_head"))
    return {
        "branch": common.git("rev-parse", "--abbrev-ref", "HEAD"),
        "head_at_registration": common.git("rev-parse", "HEAD"),
        "modules": members,
        "module_count": len(members),
        "production_modules": sorted(
            relative for relative in members if relative.startswith("integrated_sar/")
        ),
        "dirty_relative_to_head": dirty,
        "not_in_head": absent,
        "clean": not dirty,
    }


def register() -> dict[str, Any]:
    payload = graph()
    if payload["not_in_head"]:
        raise SystemExit(
            "an executed module is not committed, so the freeze would name code that Git does "
            f"not have: {payload['not_in_head']}"
        )
    payload["registered"] = True
    common.write_json(common.json_path("CODE_FREEZE.json"), payload)
    return payload


def check() -> dict[str, Any]:
    frozen = json.loads(common.json_path("CODE_FREEZE.json").read_text(encoding="utf-8"))
    now = graph()
    moved = {
        relative: [frozen["modules"][relative].get("blob"), now["modules"].get(relative, {}).get("blob")]
        for relative in frozen["modules"]
        if relative not in now["modules"]
        or frozen["modules"][relative].get("blob") != now["modules"][relative].get("blob")
    }
    return {
        "frozen_at": frozen["head_at_registration"],
        "frozen_blob": frozen.get("execution_blob"),
        "registered_head": frozen["head_at_registration"],
        "current_head": now["head_at_registration"],
        "modules": len(frozen["modules"]),
        "still_executing_the_same_code": not moved,
        "moved": {key: value for key, value in sorted(moved.items())},
        "clean_now": now["clean"],
    }


def execution_blob(sha: str) -> str:
    """The blob of the freeze file itself, so a commit can be named by what it recorded."""
    return common.git("rev-parse", f"{sha}:runs/{common.HERE.name}/CODE_FREEZE.json")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("register", "check", "show"), default="show")
    args = parser.parse_args()
    if args.mode == "register":
        payload = register()
        print(
            json.dumps(
                {
                    "registered": payload["head_at_registration"],
                    "modules": payload["module_count"],
                    "dirty": payload["dirty_relative_to_head"],
                    "not_in_head": payload["not_in_head"],
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        return 0
    if args.mode == "check":
        report = check()
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0 if report["still_executing_the_same_code"] else 1
    payload = json.loads(common.json_path("CODE_FREEZE.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "registered": payload["head_at_registration"],
                "modules": payload["module_count"],
                "production_modules": payload["production_modules"],
                "dirty": payload["dirty_relative_to_head"],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
