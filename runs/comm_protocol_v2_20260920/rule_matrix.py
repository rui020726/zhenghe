"""Does the new protocol preserve the old behaviour, and which rule is the strongest one in it?

Two jobs, in the order §4 of the brief puts them.

``--mode pairs`` proves the equivalence claim on eight training streams at both credits. Three
pairs are compared: ``first_send_plus_bound_3_10`` against ``immediate``, ``bound_only_3_10``
against ``hold_to_bound``, and ``send_once`` against the bound-free ``immediate`` that round 4
called ``send_once_no_bound``. What has to agree is the behaviour: which packets went out, at which
step, in which position within the step, what they cost, what the receivers applied, and how the
tasks ended. What is *allowed* to differ is the label -- a send the arm asked for reads ``policy``
here even when a clock is what made it ask -- so each pair also reports the reason split and the
timer band recovered from the transcript.

``--mode matrix`` then runs the whole rule catalogue at both credits over the 48 training streams
and ``--mode lock`` chooses the primary comparator by the frozen rule: most on-time targets per
episode, ties broken by fewer bytes, further ties broken by a fixed lexicographic order. The
selection is made on training streams only, before any learned number exists, and the two
equivalence rules plus ``send_once`` are reported whatever they rank.

    python rule_matrix.py --mode pairs --threads 1
    python rule_matrix.py --mode matrix --shards 6 --shard 0 --threads 1
    python rule_matrix.py --mode lock
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import protocol_common as common
import protocol_env

from integrated_sar.comm import LEGACY_SHARED_BOUND, POLICY_OWNED_TIMING
from integrated_sar.comm_policy import parse_parameters

#: The equivalence pairs: (v2 arm, legacy arm, legacy bound setting). The third entry is the
#: round-4 name of the arm whose timing behaviour the new rule has to reproduce.
EQUIVALENCE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("first_send_plus_bound_3_10", "immediate", "bounded"),
    ("bound_only_3_10", "hold_to_bound", "bounded"),
    ("send_once", "send_once_no_bound", "unbounded"),
)

#: The step's emission list, the receivers' beliefs and the task outcome: everything a reader needs
#: in order to say "these two arms behaved alike" rather than "these two arms scored alike".
BEHAVIOUR_FIELDS = (
    "sends",
    "refusals",
    "airtime_bytes",
    "transmission_attempts",
    "packets_delivered",
    "deliveries_applied",
    "task",
    "generated_items",
    "coalesced_changes",
    "redundant_emissions",
    "unsent_change_steps",
    "belief",
)

#: Fixed tie-break order, so a tie cannot be resolved by whoever wrote the report.
TIE_BREAK_LEXICOGRAPHIC = True


def spec_for(table: str, arm: str) -> dict[str, Any]:
    spec = common.arm_spec(table, arm)
    spec["arm"] = arm
    spec["kind"] = "rule"
    return spec


def run_arm(spec: dict[str, Any], seed: int, *, tier: str, directory: Path) -> dict[str, Any]:
    """One rule arm, one stream, one credit, on the protocol its table says it belongs to."""
    legacy = spec.get("mode") is None or spec.get("mode") == LEGACY_SHARED_BOUND
    bounded = spec.get("bound") != "none"
    row = protocol_env.run_episode(
        seed,
        directory=directory,
        label=spec["arm"],
        policy=spec["policy"],
        policy_params=parse_parameters(spec.get("params", "")) or None,
        budget_bytes=None if spec.get("ceiling") else common.tier_bytes(tier),
        timing_mode=LEGACY_SHARED_BOUND if legacy else POLICY_OWNED_TIMING,
        max_wait_steps=common.BOUND_MAX_WAIT_STEPS if bounded else None,
        urgent_wait_steps=common.BOUND_URGENT_WAIT_STEPS if bounded else None,
        information="isolated",
        delivery_probability=1.0,
    )
    records = common.metrics.load_transcript(row["artefacts"]["transcript"])
    sends = [record for record in records if record["kind"] == "send"]
    positions: dict[int, int] = defaultdict(int)
    ordered: list[list[Any]] = []
    for record in sends:
        step = int(record["step"])
        ordered.append([step, str(record["target"]), positions[step]])
        positions[step] += 1
    rollup = common.attribution_metrics._per_reason_rollup(records)
    ledger = row["ledger"]
    closure = common.attribution_metrics._closure(ledger, rollup, spent_bytes=row["spent_bytes"])
    return {
        "arm": spec["arm"],
        "table": "legacy" if legacy else "v2",
        "mode": LEGACY_SHARED_BOUND if legacy else POLICY_OWNED_TIMING,
        "tier": tier,
        "seed": int(seed),
        "sends": ordered,
        # The refusal *tuple* is behaviour: when, which target, and which constraint said no. Its
        # send-reason label is not -- under the old protocol a takeover that could not be paid for
        # is labelled forced_maxwait, under the new one the identical refusal is the arm's own
        # request. Both readings are kept, and only the first is compared.
        "refusals": [
            [int(r["step"]), str(r["target"]), str(r["reason"])]
            for r in records
            if r["kind"] == "refused"
        ],
        "refusals_by_label": [
            [int(r["step"]), str(r["target"]), str(r["reason"]), str(r.get("send_reason", "policy"))]
            for r in records
            if r["kind"] == "refused"
        ],
        "airtime_bytes": int(row["spent_bytes"]),
        "credit_used_fraction": row["budget_used_fraction"],
        "transmission_attempts": int(ledger["transmission_attempts"]),
        "packets_delivered": int(ledger["packets_delivered"]),
        "deliveries_applied": int(ledger["deliveries_applied"]),
        "packets_lost": int(ledger["packets_lost"]),
        "task": dict(row["task"]),
        "generated_items": int(ledger["generated_items"]),
        "coalesced_changes": int(ledger["coalesced_changes"]),
        "redundant_emissions": int(ledger["redundant_emissions"]),
        "unsent_change_steps": int(ledger["unsent_change_steps"]),
        "max_backlog_age": int(ledger["max_backlog_age"]),
        "belief": [
            [str(agent), str(target), int(revision), float(progress)]
            for agent, target, revision, progress in _belief_order(records)
        ],
        "emissions_by_reason": {
            reason: int(bucket["emissions"]) for reason, bucket in rollup["by_reason"].items()
        },
        "bytes_by_reason": {
            reason: int(bucket["bytes"]) for reason, bucket in rollup["by_reason"].items()
        },
        "timer_band": _timer_band(sends),
        "belief_age_steps": float(row["belief"].get("belief_age_steps_mean") or 0.0),
        "residuals": {key: value for key, value in closure["residuals"].items() if value},
        "generation_identity_unspoken": int(
            closure["generated_balance"]["unspoken_or_uncounted_at_end"]
        ),
        "seconds": float(row["seconds"]),
        "config_hash": row["config_hash"],
        "arm_label": row["arm_label"],
    }


def _belief_order(records: list[dict[str, Any]]) -> list[tuple[str, str, int, float]]:
    """Every packet a receiver applied, in delivery order: (agent, target, revision, clock).

    ``clock`` is the relay's own time reading carried on the delivery record, so two arms that
    applied the same packets at the same instants produce the same list.
    """
    applied: list[tuple[str, str, int, float]] = []
    for record in records:
        if record["kind"] != "deliver" or not record["accepted"]:
            continue
        applied.append(
            (
                str(record["agent"]),
                str(record["target"]),
                int(record["revision"]),
                float(record.get("now") or 0.0),
            )
        )
    return applied


def _timer_band(sends: list[dict[str, Any]]) -> dict[str, int]:
    """How much of an arm's own traffic was driven by the declared 3/10 clock.

    Recovered from the transcript rather than from a label, because under the new protocol a timer
    request *is* a policy request: ``(step - ready_step)`` against the same bound predicate the
    transport applies, with the urgency window read off the packet's own deadline and clock.
    """
    urgent = common.BOUND_URGENT_WAIT_STEPS
    limit = common.BOUND_MAX_WAIT_STEPS
    window = common.BOUND_URGENCY_TIME
    counts: Counter[str] = Counter()
    for record in sends:
        age = int(record["step"]) - int(record["ready_step"])
        slack = float(record["deadline"]) - float(record["now"])
        bound = urgent if 0.0 <= slack <= window else limit
        first = bool(record.get("first"))
        counts["timer_driven" if age >= bound else "asked_ahead"] += 1
        counts["first_reports" if first else "refreshes"] += 1
        if age >= bound and not first:
            counts["timer_driven_refreshes"] += 1
    return dict(sorted(counts.items()))


def pairs_mode(args: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(max(1, int(args.threads)))
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"pairs": {}, "episodes": 0}
    for tier in args.tiers:
        for v2_arm, legacy_arm, bound in EQUIVALENCE_PAIRS:
            v2 = spec_for("v2", v2_arm)
            legacy = spec_for("legacy", legacy_arm)
            cells: list[dict[str, Any]] = []
            for seed in args.seeds:
                owned = run_arm(v2, seed, tier=tier, directory=directory / f"owned_{tier}")
                before = run_arm(legacy, seed, tier=tier, directory=directory / f"before_{tier}")
                drift = {
                    key: [owned.get(key), before.get(key)]
                    for key in BEHAVIOUR_FIELDS
                    if owned.get(key) != before.get(key)
                }
                cells.append(
                    {
                        "seed": int(seed),
                        "equal": not drift,
                        "drift": drift,
                        "emissions": len(owned["sends"]),
                        "airtime_bytes": owned["airtime_bytes"],
                        "reasons_owned": owned["emissions_by_reason"],
                        "reasons_legacy": before["emissions_by_reason"],
                        "refusal_labels_owned": _label_counts(owned["refusals_by_label"]),
                        "refusal_labels_legacy": _label_counts(before["refusals_by_label"]),
                        "timer_band_owned": owned["timer_band"],
                        "timer_band_legacy": before["timer_band"],
                    }
                )
            key = f"{v2_arm} vs {legacy_arm} @ {tier}"
            mismatched = [cell for cell in cells if not cell["equal"]]
            report["pairs"][key] = {
                "v2_arm": v2_arm,
                "legacy_arm": legacy_arm,
                "legacy_bound": bound,
                "episodes": len(cells),
                "all_equal": not mismatched,
                "mismatched_seeds": [cell["seed"] for cell in mismatched],
                "first_mismatch_fields": sorted(
                    {field for cell in mismatched for field in cell["drift"]}
                ),
                "cells": cells,
            }
            report["episodes"] += len(cells)
            print(
                f"{key}: {len(cells) - len(mismatched)}/{len(cells)} episodes identical; "
                f"emissions={sum(cell['emissions'] for cell in cells)} "
                f"bytes={sum(cell['airtime_bytes'] for cell in cells)} "
                f"timer_band_owned={sum(cell['timer_band_owned'].get('timer_driven', 0) for cell in cells)} "
                f"forced_legacy={sum(cell['reasons_legacy'].get('forced_maxwait', 0) for cell in cells)}",
                flush=True,
            )
    report["verdict"] = all(values["all_equal"] for values in report["pairs"].values())
    common.write_json(common.json_path("COMM_PROTOCOL_V2_EQUIVALENCE.json"), report)
    print(json.dumps({"verdict": report["verdict"], "pairs": len(report["pairs"])}, ensure_ascii=False))
    return 0 if report["verdict"] else 1


def _label_counts(refusals: list[list[Any]]) -> dict[str, int]:
    counted: dict[str, int] = {}
    for _, _, _, label in refusals:
        counted[label] = counted.get(label, 0) + 1
    return dict(sorted(counted.items()))


def matrix_run_mode(args: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(max(1, int(args.threads)))
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, dict[str, Any], int]] = []
    arms = list(common.V2_ARMS) + list(common.CEILING_ARMS)
    if args.arms:
        arms = [arm for arm in arms if arm in set(args.arms)]
    for tier in args.tiers:
        for arm in arms:
            table = "ceiling" if arm in common.CEILING_ARMS else "v2"
            spec = spec_for(table, arm)
            for seed in args.seeds:
                tasks.append((tier, spec, seed))
    written = 0
    for tier, spec, seed in tasks[args.shard :: args.shards]:
        path = directory / f"row_{tier}_{spec['arm']}_s{seed}.json"
        if path.exists() and not args.overwrite:
            continue
        row = run_arm(spec, seed, tier=tier, directory=directory / f"runs_{tier}")
        common.write_json(path, row)
        written += 1
        print(
            f"[{tier}] {spec['arm']} s{seed}: onT={row['task']['on_time']}/{row['task']['targets']} "
            f"bytes={row['airtime_bytes']} em={len(row['sends'])} refusals={len(row['refusals'])} "
            f"({row['seconds']}s)",
            flush=True,
        )
    print(json.dumps({"rows_written": written, "directory": str(directory)}, ensure_ascii=False))
    return 0


def load_rows(directory: Path, tier: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob(f"row_{tier}_*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def matrix_report_mode(args: argparse.Namespace) -> int:
    """Cost-effect table per tier on the training streams, then the frozen selection rule."""
    report: dict[str, Any] = {
        "selection_rule": (
            "highest mean on-time targets per episode; ties broken by fewer mean bytes; remaining "
            "ties broken by ascending arm name. Training streams only, learned arms excluded, "
            "out-of-protocol ceilings excluded."
        ),
        "tie_break": "lexicographic by arm name" if TIE_BREAK_LEXICOGRAPHIC else None,
        "primary_budget_tier": common.PRIMARY_TIER,
        "tiers": {},
    }
    for tier in args.tiers:
        rows = load_rows(args.output_dir, tier)
        if not rows:
            continue
        wanted = set(args.arms or (list(common.V2_ARMS) + list(common.CEILING_ARMS)))
        expected_arms = [arm for arm in common.V2_ARMS if arm in wanted] + [
            arm for arm in common.CEILING_ARMS if arm in wanted
        ]
        missing = _coverage_gaps(rows, expected_arms=expected_arms, expected_seeds=args.seeds)
        by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_arm[row["arm"]].append(row)
        table: dict[str, Any] = {}
        for arm, values in sorted(by_arm.items()):
            episodes = len(values)
            on_time = [float(row["task"]["on_time"]) for row in values]
            table[arm] = {
                "episodes": episodes,
                "on_time_targets_per_episode": round(float(np.mean(on_time)), 4),
                "on_time_per_episode_stdev": round(float(np.std(on_time)), 4),
                "detected_targets_total": int(sum(row["task"]["targets"] for row in values)),
                "expired_total": int(sum(row["task"]["expired"] for row in values)),
                "unresolved_total": int(sum(row["task"]["unresolved"] for row in values)),
                "airtime_bytes_mean": round(float(np.mean([row["airtime_bytes"] for row in values])), 1),
                "airtime_bytes_total": int(sum(row["airtime_bytes"] for row in values)),
                "credit_used_fraction_mean": round(
                    float(np.mean([row["credit_used_fraction"] or 0.0 for row in values])), 4
                ),
                "episodes_with_refusal": int(sum(1 for row in values if row["refusals"])),
                "refusals_total": int(sum(len(row["refusals"]) for row in values)),
                "emissions_per_episode": round(
                    float(np.mean([len(row["sends"]) for row in values])), 3
                ),
                "first_reports_total": int(
                    sum(row["timer_band"].get("first_reports", 0) for row in values)
                ),
                "timer_driven_refreshes_total": int(
                    sum(row["timer_band"].get("timer_driven_refreshes", 0) for row in values)
                ),
                "redundant_emissions_total": int(sum(row["redundant_emissions"] for row in values)),
                "deliveries_applied_total": int(sum(row["deliveries_applied"] for row in values)),
                "belief_age_steps_mean": round(
                    float(np.mean([row["belief_age_steps"] for row in values])), 3
                ),
                "max_backlog_age_mean": round(
                    float(np.mean([row["max_backlog_age"] for row in values])), 1
                ),
                "emissions_by_reason": {
                    reason: int(sum(row["emissions_by_reason"].get(reason, 0) for row in values))
                    for reason in sorted({key for row in values for key in row["emissions_by_reason"]})
                },
                "ledger_residuals_nonzero": sorted(
                    {key for row in values for key in row["residuals"]}
                ),
                "generation_identity_unspoken_total": int(
                    sum(row["generation_identity_unspoken"] for row in values)
                ),
                "selectable": arm in common.SELECTABLE_V2_ARMS,
                "ceiling_reference": arm in common.CEILING_ARMS,
            }
        ranked = sorted(
            (arm for arm, values in table.items() if values["selectable"]),
            key=lambda arm: (
                -table[arm]["on_time_targets_per_episode"],
                table[arm]["airtime_bytes_mean"],
                arm,
            ),
        )
        report["tiers"][tier] = {
            "budget_bytes": common.tier_bytes(tier),
            "rows": len(rows),
            "expected_rows": len(expected_arms) * len(args.seeds),
            "coverage_gaps": missing,
            "arms": table,
            "ranking": ranked,
            "selected_primary": ranked[0] if ranked else None,
            "always_reported": [
                "first_send_plus_bound_3_10",
                "bound_only_3_10",
                "send_once",
            ],
        }
        print(
            f"[{tier}] selected={report['tiers'][tier]['selected_primary']} ranking={ranked[:4]} "
            f"gaps={len(missing)}",
            flush=True,
        )
    report["round4_reproduction"] = round4_reproduction(report)
    if report["round4_reproduction"].get("mismatches"):
        print("ROUND-4 REPRODUCTION MISMATCHES: "
              + json.dumps(report["round4_reproduction"]["mismatches"], ensure_ascii=False)[:800])
    common.write_json(common.json_path("COMM_PROTOCOL_V2_RULE_SELECTION.json"), report)
    return 0


def round4_reproduction(report: dict[str, Any]) -> dict[str, Any]:
    """The two equivalent arms must reproduce round 4's 48-stream rule gate, value for value.

    That gate ran on the same streams, the same credits, the same transport in its legacy mode, and
    its aggregates are tier-qualified -- so it is the only archived read that can check the 75 %
    tier, whose per-stream transcripts round 4 overwrote. A difference here would mean this tree
    changed something about the old protocol, and nothing in this round is allowed to.
    """
    path = common.TIGHT_ROUND / "COMM_TIGHT_BUDGET_GATE.json"
    if not path.exists():
        return {"available": False, "reason": f"{path} is not on this machine"}
    gate = json.loads(path.read_text(encoding="utf-8"))
    comparisons = {
        "first_send_plus_bound_3_10": "immediate",
        "bound_only_3_10": "hold_to_bound",
        "send_once": "send_once_no_bound",
    }
    fields = (
        "on_time_targets_per_episode_mean",
        "airtime_bytes_mean",
        "belief_age_steps_mean",
        "episodes_with_any_refusal",
    )
    checks: dict[str, Any] = {}
    mismatches: dict[str, Any] = {}
    for tier, block in report["tiers"].items():
        for arm, legacy in comparisons.items():
            key = f"{tier}/{legacy}"
            if key not in gate["per_cell"] or arm not in block["arms"]:
                continue
            before, now = gate["per_cell"][key], block["arms"][arm]
            pair = {
                "round4": {field: before.get(field) for field in fields},
                "this_round": {
                    field: value
                    for field, value in zip(
                        fields,
                        (
                            now["on_time_targets_per_episode"],
                            now["airtime_bytes_mean"],
                            now["belief_age_steps_mean"],
                            now["episodes_with_refusal"],
                        ),
                    )
                },
            }
            checks[key] = pair
            if pair["round4"] != pair["this_round"]:
                mismatches[key] = pair
    return {
        "available": True,
        "source": str(path),
        "cells_compared": len(checks),
        "cells_agreeing": len(checks) - len(mismatches),
        "mismatches": mismatches,
        "fields": list(fields),
        "checks": checks,
    }


def _coverage_gaps(
    rows: list[dict[str, Any]], *, expected_arms: list[str], expected_seeds: list[int]
) -> list[str]:
    """Every (arm, stream) cell this matrix should hold and does not.

    A missing cell is not a zero: an arm whose expensive episodes were interrupted would otherwise
    be ranked on the cheap ones that survived, and look good for the wrong reason.
    """
    have = {(row["arm"], int(row["seed"])) for row in rows}
    return [
        f"{arm}:s{seed}"
        for arm in expected_arms
        for seed in expected_seeds
        if (arm, int(seed)) not in have
    ]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("pairs", "matrix", "report"), required=True)
    parser.add_argument(
        "--tiers", nargs="+", default=list(common.BUDGET_TIERS), choices=list(common.BUDGET_TIERS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=common.HERE / "rules")
    args = parser.parse_args(argv)
    if args.seeds is None:
        args.seeds = (
            list(common.GATE_STREAMS) if args.mode == "pairs" else list(common.TRAIN_SEEDS)
        )
    if args.shard >= args.shards:
        raise SystemExit(f"--shard {args.shard} is outside --shards {args.shards}")
    if args.mode == "pairs":
        return pairs_mode(args)
    if args.mode == "matrix":
        return matrix_run_mode(args)
    return matrix_report_mode(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
