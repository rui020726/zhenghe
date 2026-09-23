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
        # ``torch.ops`` and ``torch.classes`` are synthesised on attribute access with a bare
        # relative ``__file__`` ("_ops.py"), which resolves against whatever directory the round
        # happens to run from -- so a lazy loader can invent a file inside this round's own
        # directory. Only absolute origins describe code that exists.
        if not os.path.isabs(path):
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


def register(name: str = "CODE_FREEZE.json", goals: int | None = None) -> dict[str, Any]:
    """Write the freeze file a layer's rows will be stamped with.

    The episode-code identity is recorded next to the head, so a row can be checked against the
    freeze that claims to cover it without re-deriving the whole graph (`registered_freeze` reads
    it), and so an extension layer can prove it ran in the same world as the layers it extends.
    """
    import pressure_matrix

    payload = graph()
    if payload["not_in_head"]:
        raise SystemExit(
            "an executed module is not committed, so the freeze would name code that Git does "
            f"not have: {payload['not_in_head']}"
        )
    payload["registered"] = True
    payload["freeze_file"] = name
    payload.update(pressure_matrix.episode_code_identity())
    if goals is not None:
        payload["layer"] = common.launch_grant_for(goals)
        payload["expected_episode_code_identity"] = payload["episode_code_identity"]
    common.write_json(common.json_path(name), payload)
    return payload


def check(name: str = "CODE_FREEZE.json") -> dict[str, Any]:
    import pressure_matrix

    frozen = json.loads(common.json_path(name).read_text(encoding="utf-8"))
    now = graph()
    identity_now = pressure_matrix.episode_code_identity()["episode_code_identity"]
    moved = {
        relative: [frozen["modules"][relative].get("blob"), now["modules"].get(relative, {}).get("blob")]
        for relative in frozen["modules"]
        if relative not in now["modules"]
        or frozen["modules"][relative].get("blob") != now["modules"][relative].get("blob")
    }
    return {
        "freeze_file": name,
        "frozen_at": frozen["head_at_registration"],
        "frozen_blob": frozen.get("execution_blob"),
        "registered_head": frozen["head_at_registration"],
        "current_head": now["head_at_registration"],
        "modules": len(frozen["modules"]),
        "still_executing_the_same_code": not moved,
        "episode_code_identity_at_freeze": frozen.get("episode_code_identity"),
        "episode_code_identity_now": identity_now,
        "same_episode_code_state": frozen.get("episode_code_identity") == identity_now,
        "moved": {key: value for key, value in sorted(moved.items())},
        "clean_now": now["clean"],
    }


def execution_blob(sha: str) -> str:
    """The blob of the freeze file itself, so a commit can be named by what it recorded."""
    return common.git("rev-parse", f"{sha}:runs/{common.HERE.name}/CODE_FREEZE.json")


def register_start_ledger(goals: int, freeze_name: str) -> dict[str, Any]:
    """Snapshot the launch accounts and licences before the layer's first episode spends anything.

    The g12 extension is allowed 16 launches more than the brief's arithmetic, so the file that
    says so has to exist before the spending, not beside the bill afterwards.
    """
    audit = json.loads(common.json_path(common.audit_file(goals)).read_text(encoding="utf-8"))
    payload = {
        "layer": int(goals),
        "seeds": [common.LAYERS[goals][0], common.LAYERS[goals][-1]],
        "launch_ceiling": common.launch_grant_for(goals),
        "ledger_before": common.ledger_totals(),
        "launches_this_layer_would_need": common.STREAMS_PER_LAYER * (1 + len(common.ARM_ORDER)),
        "freeze_file": freeze_name,
        "seed_audit": {
            "file": common.audit_file(goals),
            "clean": audit["clean"],
            "hit_counts": audit["hit_counts"],
            "scope": audit["scope"],
        },
        "sealed_block": common.sealed_block_artefacts(),
        "production_defaults": common.production_defaults_ok(),
        "no_new_verification_episodes": True,
        "rule": "failures spend the ceiling; no seed replacement, no extra arms, no re-runs",
    }
    name = f"G{int(goals)}_START_LEDGER.json"
    common.write_json(common.json_path(name), payload)
    payload["file"] = name
    return payload


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("register", "check", "show"), default="show")
    parser.add_argument(
        "--name",
        default=None,
        help="freeze file; defaults to the file the layer's rows read it from",
    )
    parser.add_argument("--goals", type=int, default=None, choices=sorted(common.LAYERS))
    args = parser.parse_args()
    name = args.name or (common.freeze_file(args.goals) if args.goals else "CODE_FREEZE.json")
    if args.mode == "register":
        payload = register(name, args.goals)
        report = {
            "freeze_file": name,
            "registered": payload["head_at_registration"],
            "modules": payload["module_count"],
            "episode_code_identity": payload["episode_code_identity"],
            "run_episode_sha256": payload["run_episode_sha256"],
            "layer": payload.get("layer"),
            "dirty": payload["dirty_relative_to_head"],
            "not_in_head": payload["not_in_head"],
        }
        if args.goals == common.G12_GOALS:
            report["start_ledger"] = register_start_ledger(args.goals, name)["file"]
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0
    if args.mode == "check":
        report = check(name)
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0 if report["still_executing_the_same_code"] else 1
    payload = json.loads(common.json_path(name).read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "freeze_file": name,
                "registered": payload["head_at_registration"],
                "modules": payload["module_count"],
                "episode_code_identity": payload.get("episode_code_identity"),
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
