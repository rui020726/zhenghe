"""ROUND66 independent, read-only evaluation eligibility auditor.

WHY THIS FILE EXISTS
--------------------
The decision side ruled the companion handling in r66_eval_fixed2.py (persist_result /
_receipt_is_sound) too broad: it records ANY absent companion as "declared_but_not_produced"
(acceptable) so long as at least one companion is present. Under that single rule a MODEL row
that never wrote its collector tape - or lost its lifecycle or transcript - would still be
admitted to the statistics. That is the exact class of silent drop the round cannot tolerate.

This module re-derives eligibility INDEPENDENTLY from the claims + disk, scoped by the REAL run
class of each row, and fails closed on any missing required evidence. It imports nothing from the
adapter (no r66_eval_fixed2, no r66_execute) and re-implements the tiny file/json helpers itself,
so it is a genuine second opinion rather than a restatement of _receipt_is_sound.

Only claims in the evaluation namespace (logical_job beginning "r66eval2_") are audited; training
claims live under a different namespace and their eligibility is proven by the first-batch gates,
not here. Out-of-scope complete claims are counted separately and never silently folded in.

RUN CLASS
---------
The run class is decided by the row itself:
  * collector_summary is a dict  -> "model": a collector ran, so a tape MUST exist and be closed
    by the collector: the tape file must be present, non-empty, and carry exactly
    collector_summary["decision_steps"] lines.
  * collector_summary is None     -> "rule": no collector was attached (immediate / late /
    always_hold rule arms, and the late_discovery_gate rows that carry policy=learned but no
    collector). A tape may be structurally absent, exactly as the 48 block-A references are.

lifecycle and transcript are written by EVERY run class, so they are required for both. A MISSING
file and a present-but-empty (zero-event / zero-byte) file are reported with DIFFERENT status and
are never conflated.

The auditor is strictly read-only: it never edits a claim, never writes a row, never re-runs a
cell whose evidence is already complete, and never mutates module state.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from collections import Counter
from pathlib import Path
from typing import Any

ARTEFACT_KEYS = ("lifecycle", "transcript", "tape")
EVAL_ROW_PREFIXES = ("model", "rule", "focal")
EVAL_NAMESPACE = "r66eval2_"
EXPECTED_CELLS = 678


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _line_count(path: Path) -> int:
    n = 0
    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            n += 1
    return n


def classify_run(row: dict) -> str:
    """model if a collector ran (collector_summary is a dict), else rule."""
    cs = row.get("collector_summary")
    if isinstance(cs, dict):
        return "model"
    if cs is None:
        return "rule"
    return "unknown"


def _artefact_status(artefacts: dict, key: str) -> tuple[str, Path | None]:
    """present / zero_byte / missing / not_declared. present-but-empty is distinct from absent."""
    p = artefacts.get(key)
    if not isinstance(p, str) or not p:
        return "not_declared", None
    pp = Path(p)
    if not pp.is_file():
        return "missing", pp
    if pp.stat().st_size == 0:
        return "zero_byte", pp
    return "present", pp


def check_row_evidence(row: dict) -> dict:
    """Run-class-scoped companion contract. Returns eligible flag + per-artifact status + reasons."""
    artefacts = row.get("artefacts") or {}
    run = classify_run(row)
    statuses: dict[str, str] = {}
    paths: dict[str, Path | None] = {}
    for key in ARTEFACT_KEYS:
        st, pp = _artefact_status(artefacts, key)
        statuses[key] = st
        paths[key] = pp
    reasons: list[str] = []

    # lifecycle and transcript are written by every run class.
    for key in ("lifecycle", "transcript"):
        st = statuses[key]
        if st == "present":
            continue
        where = f" at {paths[key]}" if paths[key] is not None else " (not declared)"
        reasons.append(f"{key}: {st}{where}")

    # tape: closed by the collector for model runs; structurally optional for rule runs.
    tape_st = statuses["tape"]
    if run == "model":
        cs = row.get("collector_summary") or {}
        ds = cs.get("decision_steps") if isinstance(cs, dict) else None
        if tape_st == "not_declared":
            reasons.append("tape: model run declares no tape path")
        elif tape_st == "missing":
            reasons.append(f"tape: model run tape missing at {paths['tape']}")
        elif tape_st == "zero_byte":
            reasons.append(f"tape: model run tape zero-byte at {paths['tape']}")
        else:
            if not isinstance(ds, int) or isinstance(ds, bool):
                reasons.append(f"tape: model run has no integer decision_steps ({ds!r})")
            else:
                lc = _line_count(paths["tape"])
                if lc != ds:
                    reasons.append(f"tape: model run tape lines {lc} != decision_steps {ds}")
    elif run == "rule":
        if tape_st == "zero_byte":
            reasons.append(f"tape: rule run tape present but zero-byte at {paths['tape']}")
        # missing / not_declared / present-nonempty are all acceptable for a rule run.
    else:
        reasons.append("run class unknown: collector_summary is "
                       f"{type(row.get('collector_summary')).__name__}")

    return {"run_class": run, "eligible": not reasons, "reasons": reasons,
            "artifacts": statuses}


def _parse_logical(logical: str) -> dict | None:
    m = re.match(r"^r66eval2_rule_(.+)_greedy_scene(\d+)$", logical)
    if m:
        return {"scope": "rule", "arm": m.group(1).upper(), "scene": int(m.group(2)),
                "mode": "greedy", "seed": None}
    m = re.match(r"^r66eval2_model_(.+)_s(\d+)_(greedy|sampled)_scene(\d+)$", logical)
    if m:
        return {"scope": "model", "arm": m.group(1).upper(), "scene": int(m.group(4)),
                "mode": m.group(3), "seed": int(m.group(2))}
    m = re.match(r"^r66eval2_focal_(.+)_s(\d+)_scene(\d+)$", logical)
    if m:
        return {"scope": "focal", "arm": m.group(1).upper(), "scene": int(m.group(3)),
                "mode": "greedy", "seed": int(m.group(2))}
    return None


def _identity_reason(row: dict, logical: str | None) -> str | None:
    """None when the row is internally consistent and (if given) matches its logical_job."""
    try:
        arm = str(row["model_arm"]).upper()
        scene = int(row["scenario_seed"])
        mode = row.get("eval_mode")
        seed = row.get("training_seed")
        steps = int(row["steps"])
        on_time = row.get("on_time")
    except Exception as exc:
        return f"row missing typed identity fields: {exc}"
    if steps <= 0:
        return f"row steps {steps} <= 0"
    if not isinstance(on_time, int) or isinstance(on_time, bool):
        return f"row on_time is not an int: {on_time!r}"
    if not logical:
        return None
    exp = _parse_logical(logical)
    if exp is None:
        return f"cannot parse logical_job {logical!r}"
    if arm != exp["arm"]:
        return f"row arm {arm} != logical {exp['arm']}"
    if scene != exp["scene"]:
        return f"row scene {scene} != logical {exp['scene']}"
    if exp["scope"] == "rule":
        if seed is not None:
            return f"rule row carries training_seed {seed}"
        if mode != "greedy":
            return f"rule row eval_mode {mode} != greedy"
    else:
        if seed is None or int(seed) != exp["seed"]:
            return f"row seed {seed} != logical {exp['seed']}"
        if mode != exp["mode"]:
            return f"row mode {mode} != logical {exp['mode']}"
    return None


def audit_receipt(claim: dict) -> dict:
    """Re-derive one claim's admissibility from disk. Fails closed on every problem."""
    logical = claim.get("logical_job")
    receipt = claim.get("result_receipt")
    if not isinstance(receipt, dict):
        return {"admissible": False, "reasons": ["no result_receipt object"],
                "run_class": None, "artifacts": {}, "row_path": None, "logical_job": logical}
    row_path = receipt.get("row_path")
    if not row_path:
        return {"admissible": False, "reasons": ["receipt names no row_path"],
                "run_class": None, "artifacts": {}, "row_path": None, "logical_job": logical}
    p = Path(row_path)
    if not p.is_file() or p.stat().st_size == 0:
        return {"admissible": False, "reasons": [f"row missing or empty at {row_path}"],
                "run_class": None, "artifacts": {}, "row_path": str(p), "logical_job": logical}
    try:
        first = _sha256_file(p)
        recorded = receipt.get("row_sha256")
        if recorded is not None and first != recorded:
            return {"admissible": False,
                    "reasons": [f"row sha256 {first} != receipt {recorded}"],
                    "run_class": None, "artifacts": {}, "row_path": str(p), "logical_job": logical}
        row = _read_json(p)
        if _sha256_file(p) != first:
            return {"admissible": False, "reasons": ["row hash not stable across re-read"],
                    "run_class": None, "artifacts": {}, "row_path": str(p), "logical_job": logical}
    except Exception as exc:
        return {"admissible": False, "reasons": [f"row read/hash failed: {exc}"],
                "run_class": None, "artifacts": {}, "row_path": str(p), "logical_job": logical}
    evidence = check_row_evidence(row)
    reasons = list(evidence["reasons"])
    idr = _identity_reason(row, logical)
    if idr:
        reasons.append(f"identity: {idr}")
    return {"admissible": not reasons, "reasons": reasons, "run_class": evidence["run_class"],
            "artifacts": evidence["artifacts"], "row_path": str(p), "logical_job": logical}


def audit_claims(raw_root: Path) -> dict:
    """Audit every complete EVAL claim, decide one admissible row per logical job, list orphans."""
    raw_root = Path(raw_root)
    claims_root = raw_root / "launch_claims"
    rows_dir = raw_root / "rows"
    per_job: dict[str, dict] = {}
    complete = 0
    out_of_scope = 0
    unreadable = 0
    if claims_root.is_dir():
        for d in sorted(p for p in claims_root.iterdir() if p.is_dir()):
            for path in sorted(d.glob("attempt_*.json")):
                try:
                    claim = _read_json(path)
                except Exception:
                    unreadable += 1
                    continue
                if claim.get("status") != "complete":
                    continue
                logical = claim.get("logical_job") or ""
                if not logical.startswith(EVAL_NAMESPACE):
                    out_of_scope += 1
                    continue
                complete += 1
                res = audit_receipt(claim)
                bucket = per_job.setdefault(logical, {"attempts": [], "admissible_paths": []})
                bucket["attempts"].append({"claim_path": str(path),
                                           "admissible": res["admissible"],
                                           "reasons": res["reasons"]})
                if res["admissible"]:
                    resolved = str(Path(res["row_path"]).resolve())
                    if resolved not in bucket["admissible_paths"]:
                        bucket["admissible_paths"].append(resolved)

    admissible_jobs: dict[str, str] = {}
    ambiguous: list[dict] = []
    excluded: list[str] = []
    for logical, bucket in per_job.items():
        adm = sorted(set(bucket["admissible_paths"]))
        if len(adm) == 1:
            admissible_jobs[logical] = adm[0]
        elif not adm:
            excluded.append(logical)
        else:
            ambiguous.append({"logical_job": logical, "rows": adm})

    on_disk: set[str] = set()
    if rows_dir.is_dir():
        for prefix in EVAL_ROW_PREFIXES:
            for q in rows_dir.glob(f"v2_{EVAL_NAMESPACE}{prefix}_*.json"):
                on_disk.add(str(q.resolve()))
    named = {str(Path(p).resolve()) for p in admissible_jobs.values()}
    orphans = sorted(on_disk - named)

    rc: Counter = Counter()
    for p in admissible_jobs.values():
        try:
            rc[classify_run(_read_json(p))] += 1
        except Exception:
            rc["unreadable"] += 1

    return {
        "summary": {
            "complete_eval_claims_audited": complete,
            "out_of_scope_complete_claims": out_of_scope,
            "unreadable_claims": unreadable,
            "admissible_logical_jobs": len(admissible_jobs),
            "excluded_logical_jobs": len(excluded),
            "ambiguous_logical_jobs": len(ambiguous),
            "orphan_row_files": len(orphans),
            "run_class_breakdown": dict(rc),
            "expected_cells": EXPECTED_CELLS,
            "all_expected_cells_admissible": (len(admissible_jobs) == EXPECTED_CELLS
                                              and not ambiguous and not excluded),
        },
        "admissible_jobs": admissible_jobs,
        "ambiguous": ambiguous,
        "excluded_logical_jobs": sorted(excluded),
        "orphan_rows": orphans,
        "per_job": per_job,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND66 independent eval eligibility auditor")
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    result = audit_claims(Path(args.raw_root))
    out = args.out or str(Path(args.raw_root) / "runtime" /
                          "ROUND66_EVAL_ELIGIBILITY_AUDIT.json")
    Path(out).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    s = result["summary"]
    print(json.dumps(s, indent=2, sort_keys=True))
    print(f"wrote {out}")
    ok = s["all_expected_cells_admissible"]
    print("VERDICT:", "ALL_CELLS_ADMISSIBLE" if ok else "INCOMPLETE_OR_NONADMISSIBLE")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
