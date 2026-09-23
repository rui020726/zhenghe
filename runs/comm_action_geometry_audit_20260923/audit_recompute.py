"""Re-judge the four delivered pressure layers from the stored rows and compare to what shipped.

Run:  python audit_recompute.py

The handoff makes this a gate: the audit may only proceed if an independent re-run of the frozen
judge reproduces every published verdict exactly. Nothing here writes into the source round --
``judge_layers.judge()`` and the ``pressure_metrics`` aggregators are called as functions, while the
command-line entry points that would overwrite ``GATES_g*.json`` are never used.

Published gates carry a *live* reading of the launch ledger (the file says how many launches had
happened when that layer was judged, which is smaller for g6 than for g12 by construction). Those
paths are exempted here, by name, before the comparison -- not after seeing the diff -- and every
other field must be byte-equal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import audit_common as A

CALLS = A.install_env_tripwire()
A.enter_source_round()

import judge_layers  # noqa: E402
import pressure_common as common  # noqa: E402
import pressure_metrics  # noqa: E402

#: Ledger readings that were true at the moment each layer was judged and cannot be true now that
#: the next layer has run. Keyed by JSON path prefix.
TIME_VARYING_PREFIXES = (
    "$.launch_ledger",
    "$.gates.L0.evidence.launches",
    "$.gates.L0.evidence.technical_failures_total",
)

#: Keys the g12 extension round added to the judge/aggregator *code* after the three delivered
#: layers shipped. Their absence in the published files is schema drift the later round caused,
#: not a data difference -- so they are allowed as `only-in-recomputed`, and only these ones.
#: g12 itself shipped under this newer code, and must therefore show zero differences at all.
LATER_ROUND_KEYS = (
    "$.gates.L0.evidence.harness_agreement_provenance",
    "$.gates.L0.evidence.technical_failures_this_layer",
    "$.gates.L2.evidence.band_edge_readings",
    "$.gates.L3.evidence.step_geometry_of_the_best_rule",
    "$.step_geometry",
)


def classify(finding: str) -> str:
    """Sort one diff line into the two pre-declared families, or into 'must stop the round'."""
    path = finding.split(":", 1)[0]
    if exempt(path):
        return "time_varying_ledger"
    if "only-in-recomputed" in finding and path in LATER_ROUND_KEYS:
        return "added_by_later_round"
    return "reproduction_failure"


def exempt(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[")
               for prefix in TIME_VARYING_PREFIXES)


def split(findings: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Separate the allowed readings from a real reproduction failure."""
    allowed: dict[str, list[str]] = {"time_varying_ledger": [], "added_by_later_round": []}
    failures: list[str] = []
    for finding in findings:
        kind = classify(finding)
        (failures if kind == "reproduction_failure" else allowed[kind]).append(finding)
    return allowed, failures


def leaf_count(payload: Any) -> int:
    """How many leaf values the comparison actually walked -- the size of the evidence base."""
    if isinstance(payload, dict):
        return sum(leaf_count(value) for value in payload.values()) or 1
    if isinstance(payload, list):
        return sum(leaf_count(value) for value in payload) or 1
    return 1


def compare_layer(layer: str, goals: int) -> dict[str, Any]:
    published_gates = A.read_json(A.SOURCE_ROUND / f"GATES_{layer}.json")
    published_matrix = A.read_json(A.SOURCE_ROUND / f"RULE_MATRIX_{layer}.json")
    published_metrics = A.read_json(A.SOURCE_ROUND / f"PRESSURE_METRICS_{layer}.json")

    recomputed_gates = judge_layers.judge(goals)
    recomputed_matrix = pressure_metrics.layer_report(goals)
    recomputed_metrics = pressure_metrics.search_pressure(goals)

    verdicts: dict[str, Any] = {}
    allowed_all: dict[str, Any] = {}
    failures_all: dict[str, Any] = {}
    for label, pub, rec in (
        ("GATES", published_gates, recomputed_gates),
        ("RULE_MATRIX", published_matrix, recomputed_matrix),
        ("PRESSURE_METRICS", published_metrics, recomputed_metrics),
    ):
        allowed, failures = split(A.deep_diff(pub, rec, limit=400))
        verdicts[f"{label}_leaves_compared"] = leaf_count(pub)
        verdicts[f"{label}_time_varying_ledger_readings"] = len(
            allowed["time_varying_ledger"]
        )
        verdicts[f"{label}_keys_added_by_later_round"] = len(allowed["added_by_later_round"])
        if "passed" in pub or "passed" in rec:
            verdicts[f"{label}_layer_passed_matches"] = pub.get("passed") == rec.get("passed")
        allowed_all[label] = allowed
        failures_all[label] = failures

    return {
        "layer": layer,
        "goals": goals,
        "verdicts": verdicts,
        "allowed_time_varying_differences": allowed_all,
        "reproduction_failures": failures_all,
        "reproduced": not any(failures_all.values()),
        "published_gate_booleans": {
            name: bool(gate["passed"]) for name, gate in published_gates["gates"].items()
        },
        "recomputed_gate_booleans": {
            name: bool(gate["passed"]) for name, gate in recomputed_gates["gates"].items()
        },
        "best_fixed_rule_published": published_gates["gates"]["L3"]["evidence"]["best_fixed_rule"],
        "best_fixed_rule_recomputed": recomputed_gates["gates"]["L3"]["evidence"]["best_fixed_rule"],
    }


def ledger_breakdown() -> dict[str, Any]:
    """Every launch on disk, split by layer, kind and success, to settle the ceiling arithmetic.

    The handoff flagged "the ledger had 144 left but g12 needed 160" as unresolved. The answer is
    in these counts, not in prose. Note that the two numbers a reader can get from these files
    disagree by construction: ``entries`` is the audit trail (one line per bookkeeping event) while
    ``starts`` is the launch count the ceiling is charged against, and a re-registration that
    salvaged an already-recorded stream adds an entry without adding a launch. Counting lines
    instead of reading ``starts`` is what makes the arithmetic look short.
    """
    rows: list[dict[str, Any]] = []
    per_file: list[dict[str, Any]] = []
    for path in sorted(A.SOURCE_ROUND.glob(common.LEDGER_PATTERN)):
        payload = A.read_json(path)
        entries = payload.get("entries", [])
        per_file.append(
            {
                "file": path.name,
                "starts": int(payload.get("starts", 0)),
                "failures": int(payload.get("failures", 0)),
                "entries": len(entries),
                "non_launch_entries": sum(
                    1 for e in entries if "re-registered without an episode" in str(e.get("detail"))
                ),
            }
        )
        for entry in entries:
            rows.append({**entry, "file": path.name})

    def _is_launch(entry: dict[str, Any]) -> bool:
        return "re-registered without an episode" not in str(entry.get("detail"))

    per_layer: dict[str, dict[str, Any]] = {}
    for row in rows:
        layer = f"g{row.get('goals')}"
        bucket = per_layer.setdefault(
            layer,
            {"entries": 0, "launches": 0, "non_launch_entries": 0, "ok": 0, "failed": 0,
             "data_launches": 0, "gate_launches": 0, "by_kind": {}},
        )
        bucket["entries"] += 1
        if not _is_launch(row):
            bucket["non_launch_entries"] += 1
            continue
        bucket["launches"] += 1
        bucket["ok" if row.get("ok") else "failed"] += 1
        kind = str(row.get("kind"))
        inner = bucket["by_kind"].setdefault(
            kind, {"launches": 0, "ok": 0, "failed": 0, "failure_details": []}
        )
        inner["launches"] += 1
        inner["ok" if row.get("ok") else "failed"] += 1
        if not row.get("ok"):
            inner["failure_details"].append(
                f"seed {row.get('seed')}: {str(row.get('detail'))[:110]}"
            )
        if kind in common.MATRIX_KINDS:
            bucket["data_launches"] += 1
        elif kind in common.GATE_KINDS:
            bucket["gate_launches"] += 1

    total_data = sum(bucket["data_launches"] for bucket in per_layer.values())
    return {
        "entries_total": len(rows),
        "launches_total": sum(1 for row in rows if _is_launch(row)),
        "non_launch_entries": sum(1 for row in rows if not _is_launch(row)),
        "data_launches": total_data,
        "gate_launches": sum(bucket["gate_launches"] for bucket in per_layer.values()),
        "failed_data_launches": sum(
            bucket["by_kind"].get(kind, {}).get("failed", 0)
            for bucket in per_layer.values()
            for kind in common.MATRIX_KINDS
        ),
        "per_layer": per_layer,
        "files_where_entries_exceed_starts": [
            item for item in per_file if item["entries"] != item["starts"]
        ],
        "ceiling": {
            "brief_cap": common.MATRIX_LAUNCH_CAP,
            "grant": common.G12_EXTENSION_GRANT,
            "effective_cap": common.MATRIX_LAUNCH_CAP_WITH_GRANT,
            "data_launches_before_g12": total_data
            - per_layer.get("g12", {}).get("data_launches", 0),
            "g12_needed": 160,
            "headroom_before_g12": common.MATRIX_LAUNCH_CAP
            - (total_data - per_layer.get("g12", {}).get("data_launches", 0)),
            "shortfall_before_grant": (total_data - per_layer.get("g12", {}).get("data_launches", 0))
            + 160
            - common.MATRIX_LAUNCH_CAP,
        },
    }


def main(argv: list[str]) -> int:
    result = {
        "layers": {layer: compare_layer(layer, A.GOALS_OF[layer]) for layer in A.LAYERS},
        "launch_ledger": ledger_breakdown(),
        "environment_tripwire_calls": list(CALLS),
        "exemptions_declared_before_comparing": {
            "live_ledger_readings": list(TIME_VARYING_PREFIXES),
            "keys_added_by_the_later_g12_round": list(LATER_ROUND_KEYS),
        },
    }
    #: g12 shipped under the current code, so it has no excuse for a single difference of any
    #: kind. If it does differ, the "later round added keys" story is wrong and the whole
    #: comparison of the other three layers becomes suspect.
    def _difference_count(entry: dict[str, Any]) -> int:
        return sum(len(v) for v in entry["reproduction_failures"].values()) + sum(
            len(items)
            for buckets in entry["allowed_time_varying_differences"].values()
            for items in buckets.values()
        )

    differences = {layer: _difference_count(entry) for layer, entry in result["layers"].items()}
    hard_failures = sum(
        len(v) for entry in result["layers"].values() for v in entry["reproduction_failures"].values()
    )
    result["differences_per_layer"] = differences
    result["g12_reproduces_with_zero_differences"] = differences["g12"] == 0
    result["all_layers_reproduced"] = hard_failures == 0 and differences["g12"] == 0
    A.write_json(A.HERE / "RECOMPUTE_L0_L4.json", result)
    print(json.dumps(
        {
            "reproduced": result["all_layers_reproduced"],
            "g12_clean": result["g12_reproduces_with_zero_differences"],
            "per_layer": {
                layer: {
                    "published": entry["published_gate_booleans"],
                    "recomputed": entry["recomputed_gate_booleans"],
                    "hard_failures": sum(len(v) for v in entry["reproduction_failures"].values()),
                    "differences": differences[layer],
                    "leaves_compared": {
                        key.split("_")[0]: value for key, value in entry["verdicts"].items()
                        if key.endswith("leaves_compared")
                    },
                    "allowed_by_category": {
                        category: sum(
                            len(buckets.get(category, []))
                            for buckets in entry["allowed_time_varying_differences"].values()
                        )
                        for category in ("time_varying_ledger", "added_by_later_round")
                    },
                }
                for layer, entry in result["layers"].items()
            },
            "launch_ledger": {
                key: value
                for key, value in result["launch_ledger"].items()
                if key in ("entries_total", "launches_total", "non_launch_entries",
                           "data_launches", "gate_launches", "failed_data_launches", "ceiling")
            },
            "per_layer_launches": {
                layer: {
                    "launches": bucket["launches"],
                    "data": bucket["data_launches"],
                    "failed": bucket["failed"],
                    "non_launch_entries": bucket["non_launch_entries"],
                }
                for layer, bucket in result["launch_ledger"]["per_layer"].items()
            },
            "tripwire_calls": result["environment_tripwire_calls"],
        },
        ensure_ascii=False,
        indent=1,
    ))
    return 0 if result["all_layers_reproduced"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
