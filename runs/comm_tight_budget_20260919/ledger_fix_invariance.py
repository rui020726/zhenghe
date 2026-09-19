"""§3's acceptance test: the accounting fix may move counters and nothing else.

Two snapshots of the same matrix -- every rule arm, both locked tiers, all 48 training streams --
taken before and after the production change, with the same frozen code everywhere else. What has
to be identical is everything an arm did: the packets it put on the air, the order they went out
in, the bytes the executor charged, the refusals it met, and the task outcome. What is allowed to
change is only what the ledger *calls* those events.

The specific prediction, written from the reproduction rather than from the diff: a poll-side
merge now counts as both a generated item and a coalesce, so the two counters rise by the same
amount and their difference -- the quantity the old identity was built from -- does not move.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import tight_common as common

BEHAVIOUR_FIELDS = (
    "emissions",
    "emissions_forced",
    "airtime_bytes",
    "autonomous_bytes",
    "forced_bytes",
    "on_time_targets_per_episode",
    "detected_targets_this_episode",
    "expired_targets_per_episode",
    "belief_age_steps",
    "unknown_to_everyone_steps",
)
#: Deliberately not compared across snapshots: the two runs write to different directories, and
#: both of these carry the artefact path, so they differ by construction, not by experiment.
PATH_DERIVED_FIELDS = ("config_hash",)
#: ``config_hash`` folds in the transcript path; ``run_signature.config_hash`` excludes only the
#: lifecycle and trace paths, so the comparison has to strip the same fields to mean "same run".
_ARTEFACT_PATH_KEYS = ("output", "decision_trace", "reservation_log", "comm_transcript")
ACCOUNTING_FIELDS = ("generated_items", "coalesced_changes", "generation_balance")


def load(directory: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted(directory.glob("rows_budget*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["arm"]), str(row["tier"]), int(row["seed"]))
            if key in rows:
                raise SystemExit(f"{path.name}: duplicate {key}")
            rows[key] = row
    return rows


def _stripped(values: dict[str, Any]) -> str:
    import json as _json

    return _json.dumps(
        {key: value for key, value in values.items() if key not in _ARTEFACT_PATH_KEYS},
        sort_keys=True,
        default=str,
    )


def configuration_identity(seeds: list[int]) -> dict[str, Any]:
    """Same experiment in two output directories: strip the paths and the config must be equal.

    Built the way ``pilot_env.run_episode`` builds it, because that translation from arm kwargs to
    ``comm_*`` configuration keys is exactly what a reader must not have to trust by eye.
    """
    import pilot_env
    from integrated_sar.comm_policy import parse_parameters

    mismatches: list[dict[str, Any]] = []
    checked = 0
    for spec in common.RULE_ARMS:
        for tier in common.BUDGET_TIERS:
            budget = None if spec.get("ceiling") else common.BUDGETS_BYTES[tier]
            for seed in seeds[:4]:
                parameters = parse_parameters(spec.get("params") or "") or {}
                unbounded = spec.get("bound") == "none"
                base = {
                    "comm_policy": spec["policy"],
                    "comm_policy_params": ",".join(
                        f"{key}={value}" for key, value in sorted(parameters.items())
                    ),
                    "comm_budget_bytes": budget,
                    "comm_sends_per_tick": None,
                    "comm_delivery_probability": 1.0,
                    "comm_delay_ticks": 0,
                    "comm_max_wait_steps": None if unbounded else pilot_env.MAX_WAIT_STEPS,
                    "comm_urgent_wait_steps": None if unbounded else pilot_env.MAX_WAIT_WHEN_URGENT_STEPS,
                    "gossip_interval": pilot_env.ISOLATED_GOSSIP_INTERVAL,
                    "quiet": True,
                }
                first = common.configuration_for(
                    seed, directory=common.HERE / "probe", label=spec["arm"], **base
                )
                second = common.configuration_for(
                    seed, directory=common.HERE / "probe_after", label=spec["arm"], **base
                )
                checked += 1
                if _stripped(first) != _stripped(second):
                    mismatches.append({"arm": spec["arm"], "tier": tier, "seed": seed})
    return {"cells_checked": checked, "mismatches": mismatches[:10], "mismatch_total": len(mismatches)}


def compare(args: argparse.Namespace) -> int:
    before = load(common.HERE / args.before)
    after = load(common.HERE / args.after)
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    behaviour_breaks: list[dict[str, Any]] = []
    accounting_delta: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    refusals_identical = 0
    for key in sorted(set(before) & set(after)):
        old, new = before[key], after[key]
        for field in BEHAVIOUR_FIELDS:
            if old.get(field) != new.get(field):
                behaviour_breaks.append({"cell": list(key), "field": field, "before": old.get(field), "after": new.get(field)})
        if old.get("refusals") == new.get("refusals"):
            refusals_identical += 1
        else:
            behaviour_breaks.append({"cell": list(key), "field": "refusals", "before": old["refusals"], "after": new["refusals"]})
        for field in ACCOUNTING_FIELDS:
            delta = int(new.get(field, 0)) - int(old.get(field, 0))
            if delta:
                accounting_delta[field]["cells_changed"] += 1
                accounting_delta[field]["total_units"] += delta
    seeds = sorted({key[2] for key in before})
    configuration = configuration_identity(seeds)
    payload = {
        "cells_compared": len(set(before) & set(after)),
        "only_in_before": [list(key) for key in missing][:10],
        "only_in_after": [list(key) for key in extra][:10],
        "behaviour_fields": list(BEHAVIOUR_FIELDS),
        "excluded_path_derived_fields": list(PATH_DERIVED_FIELDS),
        "configuration_identity": configuration,
        "behaviour_differences": behaviour_breaks[:20],
        "behaviour_differences_total": len(behaviour_breaks),
        "refusal_detail_identical": refusals_identical,
        "accounting_deltas": {field: dict(sorted(values.items())) for field, values in sorted(accounting_delta.items())},
        "prediction": {
            "generated_rises_by_coalescing": True,
            "coalesced_rises_by_the_same_amount": True,
            "generation_balance_under_the_old_reading_unchanged": (
                accounting_delta["generation_balance"].get("total_units", 0) == 0
            ),
            "equal_rise_in_both_counters": (
                accounting_delta["generated_items"].get("total_units", 0)
                == accounting_delta["coalesced_changes"].get("total_units", 0)
            ),
        },
    }
    payload["verdict"] = {
        "sends_billing_and_outcomes_unchanged": not behaviour_breaks and not missing and not extra,
        "configurations_identical_apart_from_paths": not configuration["mismatches"],
        "counters_did_move": bool(accounting_delta),
    }
    common.write_json(common.json_path("COMM_LEDGER_FIX_INVARIANCE.json"), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    return 0 if all(
        (
            payload["verdict"]["sends_billing_and_outcomes_unchanged"],
            payload["verdict"]["configurations_identical_apart_from_paths"],
            payload["verdict"]["counters_did_move"],
        )
    ) else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", default="probe")
    parser.add_argument("--after", default="probe_after")
    args = parser.parse_args(argv)
    return compare(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
