"""Run the nine locked arms over one pressure layer's 16 streams.

One cell is one episode: ``pressure_env.run_arm`` builds the frozen configuration, checks the
transport's own declaration of protocol, credit and channel before the first step, and records
the launch whether the cell succeeds or fails. Rows are appended to a per-shard JSONL so an
interrupted shard can resume -- the resume key is **(arm, seed)**, never arm alone: a check that
asks "have I run this arm?" would run the first stream of every arm and skip the rest, loudly
reporting a complete matrix.

    python pressure_matrix.py --goals 6 --shards 4 --shard 0
    python pressure_matrix.py --goals 6 --mode coverage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pressure_common as common
import pressure_env
import record_layer

#: Every artefact path a row carries is absolute; the report rewrites it under this marker.
ARTEFACT_KEY = "artefacts"


def registered_freeze(goals: int) -> dict[str, Any]:
    """The freeze this process executes under, stamped into every row it writes.

    Round 10 and this round both hit the same question: a layer whose rows were produced by two
    different code states cannot say which code produced which number. The stamp makes that
    checkable from the rows themselves instead of from a narrative in the report.

    The freeze file is chosen by layer, because the g12 extension registers its own next to the
    delivered one rather than replacing the evidence about which code made those 432 episodes.
    """
    name = common.freeze_file(goals)
    path = common.json_path(name)
    if not path.exists():
        raise SystemExit(
            f"{name} is missing: register the freeze before starting any cell, or the round cannot "
            "say what code ran"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = episode_code_identity()
    registered = str(payload["head_at_registration"])
    if payload.get("episode_code_identity") not in (None, identity["episode_code_identity"]):
        raise SystemExit(
            f"{name} was registered against episode code "
            f"{payload['episode_code_identity']}, this process runs "
            f"{identity['episode_code_identity']}: the world changed after the freeze, so the "
            "cells would mix two code states in one aggregate"
        )
    return {
        "head": registered,
        "freeze_file": name,
        "modules": int(payload["module_count"]),
        "freeze_file_sha256": common.sha256(path)[:16],
        "episode_code_identity": identity["episode_code_identity"],
        "run_episode_sha256": identity["run_episode_sha256"],
    }


def rows_path(goals: int, shard: int) -> Path:
    return common.layer_rows(goals) / f"rows_g{goals}_shard{shard}.jsonl"


def load_rows(goals: int) -> dict[tuple[str, int], dict[str, Any]]:
    """Read every shard's rows for a layer, de-duplicating by (arm, seed)."""
    seen: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(common.layer_rows(goals).glob(f"rows_g{goals}_shard*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (str(row["arm"]), int(row["seed"]))
                if key in seen and seen[key] != row:
                    raise RuntimeError(
                        f"{path.name}: cell {key} was written twice with different content; "
                        "the same seed cannot belong to two shards"
                    )
                seen[key] = row
    return seen


def cells(goals: int) -> list[tuple[str, int]]:
    return [(arm, seed) for seed in common.LAYERS[goals] for arm in common.ARM_ORDER]


def run(goals: int, shard: int, shards: int) -> dict[str, Any]:
    freeze = registered_freeze(goals)
    record_layer.require_clean_audit(goals)
    manifest = common.json_path(f"STREAM_MANIFEST_g{goals}.json")
    if not manifest.exists():
        raise SystemExit(
            f"STREAM_MANIFEST_g{goals}.json is missing: the layer's streams must be recorded and "
            "merged before any arm runs on them"
        )
    registered = json.loads(manifest.read_text(encoding="utf-8"))
    if registered["illegal_streams"]:
        raise SystemExit(
            f"layer {goals} has illegal streams and L0 cannot pass: {registered['illegal_streams']}"
        )
    by_seed = {int(row["seed"]): row for row in registered["streams"]}
    common.layer_rows(goals).mkdir(parents=True, exist_ok=True)
    common.layer_runs(goals).mkdir(parents=True, exist_ok=True)
    done = load_rows(goals)
    path = rows_path(goals, shard)
    path.touch(exist_ok=True)
    mine = common.shard_of(cells(goals), shards, shard)
    started = time.time()
    written = 0
    failures: list[str] = []
    with path.open("a", encoding="utf-8") as handle:
        for arm, seed in mine:
            if (arm, seed) in done:
                continue
            if seed not in by_seed:
                raise SystemExit(f"seed {seed} is in the matrix but not in the layer manifest")
            try:
                row = pressure_env.run_arm(goals, seed, arm)
            except Exception as error:  # noqa: BLE001 - a failed cell is reported, not skipped
                failures.append(f"{arm} s{seed}: {type(error).__name__}: {str(error)[:200]}")
                print(f"[FAIL g{goals} {arm} s{seed}] {error}", file=sys.stderr, flush=True)
                continue
            row["code_freeze"] = freeze
            row["stream_detected"] = int(by_seed[seed]["detected"])
            row["stream_placed"] = int(by_seed[seed]["placed"])
            row["stream_last_detection_step"] = by_seed[seed]["last_detection_step"]
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += 1
            done[(arm, seed)] = row
            print(
                f"[g{goals} {arm} s{seed}] on_time={row['on_time']} "
                f"bytes={row['spent_bytes']} credit={row['budget_used_fraction']} "
                f"comp={row['probe']['competition_points'] if row['probe'] else None} "
                f"{row['seconds']}s",
                flush=True,
            )
    return {
        "goals": goals,
        "shard": shard,
        "shards": shards,
        "cells_assigned": len(mine),
        "cells_written": written,
        "failures": failures,
        "seconds": round(time.time() - started, 1),
    }


def coverage(goals: int) -> dict[str, Any]:
    """What the layer actually has: every (arm, seed) present, per arm, plus the gaps."""
    done = load_rows(goals)
    expected = cells(goals)
    missing = [f"{arm} s{seed}" for arm, seed in expected if (arm, seed) not in done]
    by_arm: dict[str, int] = {arm: 0 for arm in common.ARM_ORDER}
    for (arm, _seed) in done:
        by_arm[arm] += 1
    seeds = sorted({seed for _arm, seed in done})
    return {
        "goals": goals,
        "cells_expected": len(expected),
        "cells_present": len(done),
        "per_arm": by_arm,
        "distinct_seeds": len(seeds),
        "missing_cells": missing[:20],
        "missing_count": len(missing),
        "complete": not missing,
        "ledger": common.ledger_totals(),
    }


#: Fields that legitimately differ between two passes over the same cell.
VOLATILE_ROW_FIELDS = ("seconds", "artefacts", "code_freeze")


def compare_rows(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> dict[str, Any]:
    """Did a re-run under a new freeze reproduce the same episodes?

    This is the check that turns "we re-ran it" into evidence: everything but timing, the artefact
    paths and the freeze stamp must be identical, cell by cell. A difference here is not noise to
    be averaged, it is the round's determinism claim failing.
    """
    def key(row: dict[str, Any]) -> tuple[str, int]:
        return (str(row["arm"]), int(row["seed"]))

    def strip(row: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in row.items() if k not in VOLATILE_ROW_FIELDS}

    left = {key(row): strip(row) for row in old}
    right = {key(row): strip(row) for row in new}
    only_old = sorted(set(left) - set(right))
    only_new = sorted(set(right) - set(left))
    differing: dict[str, list[str]] = {}
    for cell in sorted(set(left) & set(right)):
        fields = sorted(
            field
            for field in set(left[cell]) | set(right[cell])
            if left[cell].get(field) != right[cell].get(field)
        )
        if fields:
            differing[f"{cell[0]} s{cell[1]}"] = fields
    return {
        "compared_cells": len(set(left) & set(right)),
        "only_in_old": [f"{arm} s{seed}" for arm, seed in only_old][:10],
        "only_in_new": [f"{arm} s{seed}" for arm, seed in only_new][:10],
        "differing_fields": differing,
        "identical": not differing and not only_old and not only_new and bool(left),
    }


def episode_code_identity(at_commit: str | None = None) -> dict[str, Any]:
    """A hash of the code that can change an episode, as distinct from the code that reads it.

    ``integrated_sar/**`` is the simulation and ``run_episode`` is the harness around it; the probe
    that measures a candidate list sits beside that function and cannot reach the world. Those two
    facts are what let a layer whose cells started on either side of a probe-only fix still be one
    episode-generating code state -- and the claim is a hash of source text, not a sentence in a
    report. A freeze that moved the world changes this identity and forces a re-run.
    """
    import inspect

    import pressure_env

    #: Only commits are memoised: they name immutable content, and dozens of ``git rev-parse``
    #: calls per freeze version would otherwise dominate the report. A working-tree identity is
    #: recomputed every time, because the tree is exactly the thing a round must not edit mid-run.
    memo: dict[str, dict[str, Any]] = episode_code_identity.__dict__.setdefault("_memo", {})
    if at_commit is not None and at_commit in memo:
        return memo[at_commit]

    production = sorted(
        str(path.relative_to(common.ROOT)).replace("\\", "/")
        for path in (common.ROOT / "integrated_sar").rglob("*.py")
        if "tests" not in path.parts and "__pycache__" not in path.parts
    )
    if at_commit:
        blobs = {
            relative: common.git("rev-parse", f"{at_commit}:{relative}")
            for relative in production
        }
        source = common.git(
            "show", f"{at_commit}:runs/{common.HERE.name}/pressure_env.py"
        )
        start = source.index("def run_episode(")
        source = source[start : source.index("\ndef ", start + 1)]
    else:
        blobs = {relative: common.blob_identity(relative)["blob"] for relative in production}
        source = inspect.getsource(pressure_env.run_episode)
    digest = hashlib.sha256()
    for relative, blob in sorted(blobs.items()):
        digest.update(f"{relative}:{blob}\n".encode("utf-8"))
    digest.update(b"run_episode\n")
    digest.update(source.encode("utf-8"))
    identity = {
        "episode_code_identity": digest.hexdigest()[:16],
        "production_modules": len(blobs),
        "run_episode_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()[:16],
        "commit": at_commit or "working tree",
    }
    if at_commit is not None:
        memo[at_commit] = identity
    return identity


def freeze_history() -> dict[str, Any]:
    """Every registered freeze version in Git, with the episode identity each one names.

    Both freeze files are walked: the delivered layers stamped their rows against
    ``CODE_FREEZE.json`` and the g12 extension stamps against ``G12_FREEZE.json``, and a row from
    either one must be attributable to a named code state. Pooling only the first file would make
    every g12 row look unstamped.
    """
    history: dict[str, Any] = {}
    heads: dict[str, str] = {}
    for name in sorted({common.freeze_file(goals) for goals in common.LAYER_ORDER}):
        relative = f"runs/{common.HERE.name}/{name}"
        commits = [
            line.strip()
            for line in common.git("log", "--format=%H", "--", relative).splitlines()
            if line.strip()
        ]
        for commit in commits:
            payload = json.loads(common.git("show", f"{commit}:{relative}"))
            registered = str(payload["head_at_registration"])
            identity = episode_code_identity(registered)
            heads[registered[:9]] = registered[:9]
            history[registered[:9]] = {
                "registered_by_commit": commit[:9],
                "modules": int(payload["module_count"]),
                "freeze_file": name,
                **identity,
            }
    history["_heads"] = heads
    return history


def row_provenance(goals: int) -> dict[str, Any]:
    """Say which code state produced which cells of a layer, and whether more than one exists.

    Rows written before the freeze stamp existed are reported as un stamped rather than attributed
    by guesswork. What can still be decided is whether *any* code state that could have produced
    them differs: every freeze this round registered names an episode identity, so if those all
    agree, an un stamped row cannot have come from different episode code either.
    """
    rows = load_rows(goals)
    history = freeze_history()
    known = {
        head[:9]: entry["episode_code_identity"]
        for head, entry in history.items()
        if isinstance(entry, dict) and "episode_code_identity" in entry
    }
    identities = sorted(set(known.values()))
    stamped: dict[str, int] = {}
    stamped_identities: set[str] = set()
    unknown_stamps: dict[str, int] = {}
    unstamped = 0
    for row in rows.values():
        head = str((row.get("code_freeze") or {}).get("head", ""))[:9]
        if head:
            stamped[head] = stamped.get(head, 0) + 1
            if head in known:
                stamped_identities.add(known[head])
            else:
                unknown_stamps[head] = unknown_stamps.get(head, 0) + 1
        else:
            unstamped += 1
    return {
        "cells": len(rows),
        "stamped": stamped,
        "un_stamped": unstamped,
        "stamped_with_a_head_no_freeze_names": unknown_stamps,
        "freezes_registered_this_round": history,
        "distinct_episode_code_states": identities,
        "identities_named_by_stamped_rows": sorted(stamped_identities),
        "single_episode_code_state": len(identities) == 1
        and not unknown_stamps
        and (not stamped_identities or stamped_identities == set(identities)),
        "caveat": (
            f"{unstamped} of {len(rows)} cells carry no stamp because the stamp was added after they "
            "ran; their attribution comes from the freeze timeline above, not from the row"
        ),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", type=int, required=True, choices=sorted(common.LAYERS))
    parser.add_argument(
        "--mode", choices=("run", "coverage", "compare", "provenance"), default="run"
    )
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument(
        "--old-dir",
        default=None,
        help="row files to compare a re-run against (--mode compare); never rewritten",
    )
    args = parser.parse_args()
    if args.mode == "compare":
        if not args.old_dir:
            parser.error("--mode compare needs --old-dir")
        old_rows: list[dict[str, Any]] = []
        for path in sorted(Path(args.old_dir).glob("rows_g*_shard*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                old_rows.extend(json.loads(line) for line in handle if line.strip())
        new_rows = list(load_rows(args.goals).values())
        report = compare_rows(old_rows, new_rows)
        common.write_json(
            common.HERE / f"RERUN_DETERMINISM_g{args.goals}.json", report
        )
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0 if report["identical"] else 1
    if args.mode == "provenance":
        report = row_provenance(args.goals)
        common.write_json(common.HERE / f"ROW_PROVENANCE_g{args.goals}.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0 if report["single_episode_code_state"] else 1
    if args.mode == "coverage":
        report = coverage(args.goals)
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0 if report["complete"] else 1
    if args.shard >= args.shards:
        parser.error(f"--shard {args.shard} is outside --shards {args.shards}")
    report = run(args.goals, args.shard, args.shards)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
