"""§4: does the tight tier actually constrain the rules, on the 48 training streams?

Round 3 settled that at 75 % of the unbudgeted event-rule cost the credit never binds: the three
arms on the frontier used 0.40–0.45 of it, so "under budget" described all of them and none of
them at once. This file asks the same question of the 25 % tier -- 29,300 B, below the 35,084 B
that ``immediate`` spends inside the shared silence bound -- because the round's whole premise is
that a budget which does not bite cannot test budget-aware learning.

Two things are reported side by side because they are not the same question:

* the mean, which is what a tier was calibrated on, and
* the per-episode distribution, because a credit below an arm's average still fails to bind in an
  episode where the traffic is light, and an arm that is only sometimes constrained teaches a
  controller a different lesson from one that is always constrained.

The gate this file exists to answer, written before it was run: training on two tiers proceeds
only if the tight tier produces real refusals for the strong rules *and* leaves a legal choice
about what to say -- i.e. refusals that are not entirely the fixed forced queue being truncated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import tight_common as common

TIER_ORDER = common.BUDGET_TIERS


def row_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("rows_budget*.jsonl"))


def row_path(directory: Path, shard: str) -> Path:
    return directory / f"rows_budget{'.' + shard if shard else ''}.jsonl"


def probe_directory(args: argparse.Namespace) -> Path:
    """One directory per snapshot: the pre-fix run and the post-fix run must not overwrite."""
    return common.HERE / str(getattr(args, "output_subdir", "probe") or "probe")


def done_keys(paths: list[Path]) -> set[tuple[str, int, str]]:
    finished: set[tuple[str, int, str]] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    finished.add((str(record["arm"]), int(record["seed"]), str(record["tier"])))
    return finished


def run(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(1, int(args.threads)))
    directory = probe_directory(args)
    directory.mkdir(parents=True, exist_ok=True)
    out = row_path(directory, args.shard)
    finished = done_keys(row_files(directory))
    streams = list(common.TRAIN_SEEDS)
    if args.seed_modulus:
        streams = [seed for seed in streams if seed % args.seed_modulus == args.seed_remainder]
    started = time.time()
    written = 0
    for tier in args.tiers:
        budget = common.BUDGETS_BYTES[tier]
        for spec in common.RULE_ARMS:
            if args.arms and spec["arm"] not in args.arms:
                continue
            ceiling = bool(spec.get("ceiling"))
            for seed in streams:
                key = (spec["arm"], int(seed), tier)
                if key in finished or (ceiling and tier != TIER_ORDER[0]):
                    # An unbudgeted ceiling does not depend on the tier; run it once.
                    continue
                result = common.run_rule(
                    spec, seed, budget_bytes=None if ceiling else budget, directory=directory
                )
                row = dict(common.episode_row(result), tier=tier)
                with out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
                if row["residuals"]:
                    print(f"!! {tier}/{spec['arm']} s{seed}: residuals {row['residuals']}", flush=True)
                print(
                    f"[{tier:<6}{spec['arm']:<18} s{seed}] on_time={row['on_time_targets_per_episode']}/"
                    f"{row['detected_targets_this_episode']} bytes={row['airtime_bytes']:>6} "
                    f"use={row['credit_used_fraction']} refusals={row['refusals']['refusals']:>3} "
                    f"forced={row['emissions_forced']:>3}/{row['emissions']} "
                    f"({time.time() - started:.0f}s)",
                    flush=True,
                )
    print(f"shard finished: {written} new episodes", flush=True)
    return 0


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen: set[tuple[str, int, str]] = set()
    doubled = set()
    for row in rows:
        key = (str(row["arm"]), int(row["seed"]), str(row["tier"]))
        if key in seen:
            doubled.add(key)
        seen.add(key)
    if doubled:
        raise SystemExit(f"duplicate rows, refusing to pool: {sorted(doubled)[:5]}")
    return rows


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(float(np.mean(values)), 4),
        "min": round(float(np.min(values)), 4),
        "p25": round(float(np.percentile(values, 25)), 4),
        "median": round(float(np.median(values)), 4),
        "p75": round(float(np.percentile(values, 75)), 4),
        "max": round(float(np.max(values)), 4),
    }


def aggregate(args: argparse.Namespace) -> int:
    rows = read_rows(row_files(probe_directory(args)))
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[(str(row["arm"]), str(row["tier"]))].append(row)
    table: dict[str, Any] = {}
    for (arm, tier), cell in sorted(by_cell.items()):
        refusals = [int(row["refusals"]["refusals"]) for row in cell]
        bound_rows = [row for row in cell if int(row["refusals"]["refusals"]) > 0]
        table[f"{tier}/{arm}"] = {
            "episodes": len(cell),
            "on_time_targets_per_episode_mean": round(
                float(np.mean([int(row["on_time_targets_per_episode"]) for row in cell])), 4
            ),
            "on_time_rate_per_episode_mean": round(
                float(np.mean([row["on_time_rate_within_detected_this_episode"] or 0.0 for row in cell])), 5
            ),
            "expired_per_episode_mean": round(
                float(np.mean([int(row["expired_targets_per_episode"]) for row in cell])), 4
            ),
            "airtime_bytes_mean": round(float(np.mean([int(row["airtime_bytes"]) for row in cell])), 1),
            "credit_used_fraction": _distribution(
                [float(row["credit_used_fraction"] or 0.0) for row in cell]
            ),
            "episodes_at_or_over_credit": sum(
                1 for row in cell if float(row["credit_used_fraction"] or 0.0) >= 0.999
            ),
            "episodes_with_any_refusal": len(bound_rows),
            "refusals_per_episode": _distribution([float(value) for value in refusals]),
            "first_refusal_step": _distribution(
                [
                    float(row["refusals"]["first_refusal_step"])
                    for row in cell
                    if row["refusals"]["first_refusal_step"] is not None
                ]
            ),
            "refusal_classes": {
                reason: sum(int(row["refusals"]["refusal_classes"].get(reason, 0)) for row in cell)
                for reason in sorted({reason for row in cell for reason in row["refusals"]["refusal_classes"]})
            },
            "refused_send_reasons": {
                reason: sum(int(row["refusals"]["refused_as"].get(reason, 0)) for row in cell)
                for reason in sorted({reason for row in cell for reason in row["refusals"]["refused_as"]})
            },
            "refused_message_types": {
                kind: sum(int(row["refusals"]["refused_message_types"].get(kind, 0)) for row in cell)
                for kind in sorted({kind for row in cell for kind in row["refusals"]["refused_message_types"]})
            },
            "bytes_refused_total": sum(int(row["refusals"]["bytes_refused"]) for row in cell),
            "optional_emissions_share_mean": round(
                float(np.mean([row["optional_emissions_share"] or 0.0 for row in cell])), 4
            ),
            "autonomous_share_of_bytes_mean": round(
                float(np.mean([row["autonomous_share_of_bytes"] or 0.0 for row in cell])), 4
            ),
            "forced_bytes_mean": round(float(np.mean([int(row["forced_bytes"]) for row in cell])), 1),
            "autonomous_bytes_mean": round(float(np.mean([int(row["autonomous_bytes"]) for row in cell])), 1),
            "belief_age_steps_mean": round(
                float(np.mean([float(row["belief_age_steps"]) for row in cell])), 3
            ),
            "unknown_to_everyone_steps_mean": round(
                float(np.mean([float(row["unknown_to_everyone_steps"]) for row in cell])), 3
            ),
            "generation_balance_total": sum(int(row["generation_balance"]) for row in cell),
            "generation_balance_negative_episodes": sum(
                1 for row in cell if int(row["generation_balance"]) < 0
            ),
            "ledger_residual_rows": sum(1 for row in cell if row["residuals"]),
        }
    payload = {
        "setting": {"information": "isolated", "delivery_probability": 1.0, "streams": "train"},
        "budgets_bytes": common.BUDGETS_BYTES,
        "tiers": list(args.tiers),
        "episodes": len(rows),
        "per_cell": table,
        "gate": _gate(table, common.BUDGETS_BYTES),
        "note": (
            "credit_used_fraction is measured against the episode's own credit; 'episodes with any "
            "refusal' is the honest binding test, because the executor never spends past the credit "
            "and so cannot report over-100% use"
        ),
    }
    common.write_json(common.json_path("COMM_TIGHT_BUDGET_GATE.json"), payload)
    _print(table, payload["gate"])
    return 0


def _gate(table: dict[str, Any], budgets: dict[str, int]) -> dict[str, Any]:
    """The §4 decision, from the frozen numbers rather than from a reading of them.

    Two conditions, both about the tight tier, both evaluated on arms that share the protocol (so
    the no-bound and no-credit diagnostics are excluded):

    1. the credit actually refuses something the arm wanted to say -- real constraint;
    2. some of what got refused was a *policy* send rather than the bound's forced queue -- a real
       choice remains for a controller to make.
    """
    tier = common.BUDGET_TIERS[1]
    protocol_arms = [
        key for key, value in table.items()
        if key.startswith(f"{tier}/") and not key.endswith(("send_once_no_bound", "event_unbudgeted"))
    ]
    per_arm: dict[str, Any] = {}
    for key in protocol_arms:
        values = table[key]
        arm = key.split("/", 1)[1]
        refusals = sum(int(count) for count in values["refusal_classes"].values())
        optional = int(values["refused_send_reasons"].get("policy", 0))
        per_arm[arm] = {
            "refusals": refusals,
            "policy_refusals": optional,
            "episodes_with_any_refusal": values["episodes_with_any_refusal"],
            "episodes": values["episodes"],
            "binds": bool(refusals > 0),
            "leaves_a_choice": bool(optional > 0),
        }
    binding = [arm for arm, value in per_arm.items() if value["binds"] and value["leaves_a_choice"]]
    return {
        "tier": tier,
        "budget_bytes": budgets[tier],
        "per_arm": per_arm,
        "arms_bound_with_a_remaining_choice": sorted(binding),
        "proceed_to_two_tier_training": bool(binding),
        "rule": (
            "training on both tiers proceeds only if at least one protocol arm at the tight tier "
            "is refused something it asked to say; a tier that only truncates the bound's forced "
            "queue leaves the controller nothing to learn about"
        ),
    }


def _print(table: dict[str, Any], gate: dict[str, Any]) -> None:
    header = (
        f"{'cell':<30}{'eps':>4}{'onT/ep':>8}{'rate/ep':>9}{'bytes':>9}{'use>99%':>8}"
        f"{'boundEps':>9}{'refusals':>9}{'policyNo':>9}{'optional':>9}{'age':>8}"
    )
    print(header)
    print("-" * len(header))
    for key in sorted(table):
        values = table[key]
        print(
            f"{key:<30}{values['episodes']:>4}"
            f"{values['on_time_targets_per_episode_mean']:>8}"
            f"{values['on_time_rate_per_episode_mean']:>9}"
            f"{values['airtime_bytes_mean']:>9}"
            f"{values['episodes_at_or_over_credit']:>8}"
            f"{values['episodes_with_any_refusal']:>9}"
            f"{sum(int(count) for count in values['refusal_classes'].values()):>9}"
            f"{int(values['refused_send_reasons'].get('policy', 0)):>9}"
            f"{values['optional_emissions_share_mean']:>9}"
            f"{values['belief_age_steps_mean']:>8}"
        )
    print(json.dumps(gate, ensure_ascii=False, indent=1))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "aggregate"), default="run")
    parser.add_argument("--tiers", nargs="*", default=list(common.BUDGET_TIERS))
    parser.add_argument("--arms", nargs="*", default=[])
    parser.add_argument("--shard", default="")
    parser.add_argument("--seed-modulus", type=int, default=0)
    parser.add_argument("--seed-remainder", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-subdir", default="probe")
    args = parser.parse_args(argv)
    unknown = [tier for tier in args.tiers if tier not in common.BUDGETS_BYTES]
    if unknown:
        raise SystemExit(f"unknown budget tier(s): {unknown}; locked tiers are {list(common.BUDGETS_BYTES)}")
    if args.mode == "aggregate":
        return aggregate(args)
    return run(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
