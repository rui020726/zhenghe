"""Judge stage one: does turning B0 on change anything, and is the timing mode inert?

This is the gate the round's whole data stage depends on, so it compares *records*, not
impressions: the lifecycle logs of the same arm and seed with the probe off and on have to
agree everywhere except in the identity fields that name where artefacts were written.
"""

from __future__ import annotations

import json
from typing import Any

import d1_common as dc

from integrated_sar.info_probe import behaviour_view
from integrated_sar.lifecycle_metrics import load_records


def rows_for(tag: str, arm: str, seed: int) -> dict[str, Any] | None:
    path = dc.ROWS / f"{tag}_{arm}_s{seed}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def behaviour_records(path) -> list[dict[str, Any]]:
    return [behaviour_view(record) for record in load_records(path)]


#: Counters the transport accumulated. Dropped from the record view because that block mixes
#: them with declarations, so they are compared here, field by field, by a caller that knows
#: a same-arm/same-seed pair may not move any of them.
COUNTER_FIELDS = ("airtime_bytes", "emissions", "first_emissions", "refresh_emissions",
                  "transmission_attempts", "packets_delivered", "packets_lost",
                  "delivery_refused", "budget_refused_sends", "capacity_refused_sends",
                  "events_accepted", "status_counts")


def counters_equal(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """The names that differ; empty means the two cells did the same amount of everything."""
    return [field for field in COUNTER_FIELDS if left.get(field) != right.get(field)]


def compare_pairs(seeds: list[int], arms: list[str]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in arms:
            off, on = rows_for("b0off", arm, seed), rows_for("b0on", arm, seed)
            offrep, onrep = rows_for("b0offrep", arm, seed), rows_for("b0onrep", arm, seed)
            missing = [row is None for row in (off, on, offrep, onrep)]
            if any(missing):
                checks.append({"arm": arm, "seed": seed, "complete": False,
                               "missing": [name for name, flag in
                                          zip(("b0off", "b0on", "b0offrep", "b0onrep"), missing)
                                          if flag]})
                continue
            lifecycle = dc.cell_paths("b0off", arm, seed)["lifecycle"]
            probe_lifecycle = dc.cell_paths("b0on", arm, seed)["lifecycle"]
            same_records = behaviour_records(lifecycle) == behaviour_records(probe_lifecycle)
            counter_differences = counters_equal(off, on)
            checks.append({
                "arm": arm, "seed": seed, "complete": True,
                "behaviour_records_identical": same_records,
                "detection_sequence_equal": off["detection_sequence_sha"] ==
                on["detection_sequence_sha"],
                "emission_sequence_equal": off["emission_sequence_sha"] ==
                on["emission_sequence_sha"],
                "charge_equal": off["airtime_bytes"] == on["airtime_bytes"],
                "outcome_equal": (off["on_time"], off["expired"], off["discovered"]) ==
                                 (on["on_time"], on["expired"], on["discovered"]),
                "counter_differences": counter_differences,
                "counters_identical": not counter_differences,
                "repeat_off_deterministic": behaviour_records(
                    dc.cell_paths("b0off", arm, seed)["lifecycle"]) == behaviour_records(
                    dc.cell_paths("b0offrep", arm, seed)["lifecycle"])
                    and not counters_equal(off, offrep),
                "repeat_on_deterministic": behaviour_records(
                    dc.cell_paths("b0on", arm, seed)["lifecycle"]) == behaviour_records(
                    dc.cell_paths("b0onrep", arm, seed)["lifecycle"])
                    and not counters_equal(on, onrep),
                "probe_records_present": bool(on["probe_packets_arrived"] is not None),
                "probe_anomalies": on["probe_anomalies"],
            })
    return {"pairs": checks,
            "all_identical": all(
                check.get("behaviour_records_identical")
                and check.get("detection_sequence_equal") and check.get("emission_sequence_equal")
                and check.get("charge_equal") and check.get("outcome_equal")
                and check.get("counters_identical")
                and check.get("repeat_off_deterministic") and check.get("repeat_on_deterministic")
                for check in checks),
            "deterministic": all(check.get("repeat_off_deterministic")
                                 and check.get("repeat_on_deterministic")
                                 for check in checks if check.get("complete"))}


def compare_timing_mode(seeds: list[int]) -> dict[str, Any]:
    """I6: with no max-wait configured, the timing mode must not change anything at all."""
    checks = []
    for seed in seeds:
        legacy, owned = rows_for("modelegacy", "A1", seed), rows_for("modeowned", "A1", seed)
        if legacy is None or owned is None:
            checks.append({"seed": seed, "complete": False})
            continue
        records_legacy = behaviour_records(dc.cell_paths("modelegacy", "A1", seed)["lifecycle"])
        records_owned = behaviour_records(dc.cell_paths("modeowned", "A1", seed)["lifecycle"])
        checks.append({
            "seed": seed, "complete": True,
            "behaviour_records_identical": records_legacy == records_owned,
            "differing_record_count": sum(1 for left, right in zip(records_legacy, records_owned)
                                          if left != right),
            "record_count": len(records_legacy),
            "emission_sequence_equal": legacy["emission_sequence_sha"] ==
            owned["emission_sequence_sha"],
            "charge_equal": legacy["airtime_bytes"] == owned["airtime_bytes"],
            "outcome_equal": legacy["on_time"] == owned["on_time"],
            "counter_differences": counters_equal(legacy, owned),
            "counters_identical": not counters_equal(legacy, owned),
            "airtime_bytes": legacy["airtime_bytes"],
        })
    return {"pairs": checks,
            "inert": all(check.get("behaviour_records_identical") and check.get("charge_equal")
                         and check.get("outcome_equal") and check.get("counters_identical")
                         and check.get("emission_sequence_equal")
                         for check in checks if check.get("complete"))
            and all(check.get("complete") for check in checks)}


def run() -> int:
    seeds = [1100000, 1100001]
    b0 = compare_pairs(seeds, ["A1", "A3"])
    mode = compare_timing_mode(seeds)
    payload = {
        "round": dc.ROUND,
        "stage": "b0_paired_replay_and_timing_mode",
        "launches_booked_so_far": dc.ledger_report(),
        "b0_replay": b0,
        "i6_timing_mode_inert": mode,
        "passed": b0["all_identical"] and b0["deterministic"] and mode["inert"],
        "what_this_proves": (
            "turning B0 on leaves the detection records, the task outcomes, the emission "
            "sequence and the charged bytes identical, and repeating a cell reproduces it — "
            "so the probe cannot be the reason for anything the data stage measures. The "
            "timing-mode pair additionally shows the transport does not speak for any arm "
            "when no max-wait is configured, which is what lets A1 and A2 differ in exactly "
            "one field."
        ),
        "stop_rule": "if passed is false, the data arms must not be run",
    }
    dc.write_json("B0_PAIRED_REPLAY.json", payload)
    print(json.dumps({"b0_all_identical": b0["all_identical"], "deterministic": b0["deterministic"],
                      "mode_inert": mode["inert"], "passed": payload["passed"],
                      "pairs": len(b0["pairs"])}, ensure_ascii=False))
    for check in b0["pairs"]:
        if not check.get("behaviour_records_identical"):
            print("FAILED PAIR", check)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
