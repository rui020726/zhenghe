"""ROUND18_ARCHIVE_CLOSURE.json: round 18 archived, and its counts closed from its own logs.

Zero new episodes: every number here is re-read from the artefacts round 18 already delivered
(``runs/search_trace_and_policy_20260925/``). The point is twofold. First, an archival check — the
120 matrix cells must partition consistently (placed = detected + undetected) and their
communication cost must be reconstructable packet by packet from the transcripts. Second, the
quantity round 19 actually needs from round 18: how much *task-level* slack the deployed path
leaves at the frozen difficulty, because a controller that minimises bytes subject to a
task-effect constraint can only be judged where task failures exist.

Run:  /d/project_env/python.exe runs/comm_cost_objective_design_20260926/c19_closure_r18.py
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import Any

import c19_common as c19

sys.path.insert(0, str(c19.REPO_ROOT))

from integrated_sar.lifecycle_metrics import window_metrics  # noqa: E402  (stdlib only)

EPSILON_S = 1e-6


def cells() -> list[dict[str, Any]]:
    out = []
    for path in sorted((c19.R18 / "rows").glob("data_*.json")):
        row = c19.read_json(path)
        if row.get("ok"):
            out.append(row)
    return out


def lifecycle_view(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute the task counts straight from the ``rp`` log, ignoring the row's own numbers."""
    records = list(c19.iter_jsonl(c19.REPO_ROOT / row["artefacts"]["lifecycle"]))
    window = float(row["steps"])
    metrics = window_metrics(records, window)
    detected_steps: dict[str, float] = {}
    deadlines: dict[str, float] = {}
    for record in records:
        if record.get("event_type") == "detection":
            target = str(record["target_id"])
            stamp = float(record.get("timestamp") or 0.0)
            detected_steps.setdefault(target, stamp)
            deadlines.setdefault(target, stamp + float(
                (record.get("detail") or {}).get("deadline_offset", 0.0)))
    return {"recomputed": {key: getattr(metrics, key) for key in (
                "discovered", "completed", "expired", "on_time", "unassigned", "active")},
            "log_complete": metrics.log_complete,
            "detected_steps": detected_steps,
            "deadlines": deadlines}


def transcript_view(row: dict[str, Any]) -> dict[str, Any]:
    """Every billed packet of one cell, and whether its price reconstructs."""
    sends, delivers = [], Counter()
    for record in c19.iter_jsonl(c19.REPO_ROOT / row["artefacts"]["transcript"]):
        kind = record.get("kind")
        if kind == "send":
            sends.append(record)
        elif kind == "deliver":
            delivers[str(record.get("reason"))] += 1
    billed = sum(int(record["cost_bytes"]) for record in sends)
    mismatched_price = [record["target"] for record in sends
                        if int(record["cost_bytes"])
                        != int(record["payload_bytes"]) * int(record["subscribers"])]
    return {"emissions": len(sends),
            "first_emissions": sum(1 for record in sends if record.get("first")),
            "refresh_emissions": sum(1 for record in sends
                                     if int(record.get("revision") or 1) > 1),
            "billed_bytes_reconstructed": billed,
            "payload_bytes_total": sum(int(r["payload_bytes"]) for r in sends),
            "subscribers_observed": sorted({int(r["subscribers"]) for r in sends}),
            "packets": sum(int(r["packets"]) for r in sends),
            "packets_lost": sum(int(r["packets_lost"]) for r in sends),
            "price_formula_mismatches": mismatched_price,
            "deliver_reasons": dict(delivers),
            "distinct_targets_sent": sorted({str(r["target"]) for r in sends})}


def probe_view(row: dict[str, Any]) -> dict[str, Any]:
    """The step at which the environment itself deleted each target (B1 side file)."""
    first: dict[str, int] = {}
    steps = 0
    for record in c19.iter_jsonl(c19.REPO_ROOT / row["artefacts"]["search_probe"]):
        if record.get("kind") != "step":
            continue
        steps += 1
        for target in record.get("env_deleted_target_ids") or ():
            first.setdefault(str(target), int(record["step"]))
    return {"probe_steps": steps, "deleted_first_step": first}


def main() -> int:
    rows = cells()
    checks: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    per_cell: dict[str, list[dict[str, Any]]] = {arm: [] for arm in c19.ARMS_R18}

    for row in rows:
        arm, seed = row["arm"], int(row["seed"])
        label = f"{arm}/{seed}"
        probe_block = row["probe"]
        life = lifecycle_view(row)
        trans = transcript_view(row)
        probe = probe_view(row)

        placed = int(row["goals_placed"])
        detected = [str(t) for t in row["detected_target_ids"]]
        undetected = [str(t) for t in row["undetected_ids"]]
        probe_placed = [str(t) for t in probe_block["probe_placed_ids"]]

        def note(name: str, ok: bool, detail: Any = None) -> None:
            checks[name] += 1 if ok else 0
            if not ok:
                failures.append({"cell": label, "check": name, "detail": detail})

        note("placed_is_six", placed == 6, placed)
        note("probe_placed_equals_row_placed",
             int(probe_block["probe_placed"]) == placed, probe_block["probe_placed"])
        note("detected_plus_undetected_equals_placed",
             len(detected) + len(undetected) == placed,
             [len(detected), len(undetected), placed])
        note("partition_is_exact",
             sorted(detected + undetected) == sorted(probe_placed)
             and not (set(detected) & set(undetected)),
             [sorted(set(detected) & set(undetected))])
        note("undetected_count_matches_ids",
             int(row["undetected_count"]) == len(undetected))
        note("discovered_field_equals_detected_ids",
             int(row["discovered"]) == len(detected),
             [row["discovered"], len(detected)])
        # the task side: the registry only ever learns a target through a DetectionEvent
        note("lifecycle_discovered_equals_detected",
             life["recomputed"]["discovered"] == len(detected),
             [life["recomputed"]["discovered"], len(detected)])
        note("undetected_targets_never_enter_lifecycle",
             not (set(undetected) & set(life["detected_steps"])),
             sorted(set(undetected) & set(life["detected_steps"])))
        note("on_time_never_exceeds_discovered",
             life["recomputed"]["on_time"] <= life["recomputed"]["discovered"],
             [life["recomputed"]["on_time"], life["recomputed"]["discovered"]])
        for field, key in (("on_time", "on_time"), ("completed", "completed"),
                           ("expired", "expired"), ("discovered", "discovered"),
                           ("unassigned", "unassigned"), ("active", "active")):
            note(f"row_{field}_equals_recompute",
                 int(row[field]) == life["recomputed"][key],
                 [row[field], life["recomputed"][key]])
        note("lifecycle_log_complete", life["log_complete"])
        # cost side
        note("airtime_bytes_reconstructed",
             int(row["airtime_bytes"]) == trans["billed_bytes_reconstructed"],
             [row["airtime_bytes"], trans["billed_bytes_reconstructed"]])
        note("every_price_is_payload_times_subscribers",
             not trans["price_formula_mismatches"], trans["price_formula_mismatches"])
        note("subscribers_constant_six", trans["subscribers_observed"] == [6],
             trans["subscribers_observed"])
        note("emissions_equal_detected_targets",
             trans["emissions"] == len(detected),
             [trans["emissions"], len(detected)])
        note("every_emission_is_a_first_report",
             trans["refresh_emissions"] == 0, trans["refresh_emissions"])
        note("packets_lost_is_zero_in_r18", int(row["packets_lost"]) == 0,
             row["packets_lost"])
        # timing side: probe deletion step vs the lifecycle detection instant (dt = 1.0 s)
        agree = all(abs(float(step) - life["detected_steps"].get(target, -1e9)) <= 1.0
                    for target, step in probe["deleted_first_step"].items())
        note("probe_step_matches_lifecycle_timestamp", agree,
             [probe["deleted_first_step"],
              {k: v for k, v in life["detected_steps"].items()}])
        note("probe_reports_no_anomalies",
             int(probe_block["probe_final"]["anomaly_count"]) == 0
             and int(probe_block["probe_final"]["disagreement_steps"]) == 0)

        per_cell.setdefault(arm, []).append({
            "seed": seed,
            "placed": placed,
            "detected": len(detected),
            "undetected": len(undetected),
            "on_time": life["recomputed"]["on_time"],
            "expired": life["recomputed"]["expired"],
            "missed_after_detection": (life["recomputed"]["discovered"]
                                       - life["recomputed"]["on_time"]),
            "airtime_bytes": int(row["airtime_bytes"]),
            "emissions": trans["emissions"],
            "refresh_emissions": trans["refresh_emissions"],
            "payload_bytes": trans["payload_bytes_total"],
            "packets": trans["packets"],
            "packets_lost": trans["packets_lost"],
            "detected_ids": detected,
            "undetected_ids": undetected,
            "detected_steps": {k: life["detected_steps"][k] for k in detected
                               if k in life["detected_steps"]},
            "deadlines": {k: life["deadlines"][k] for k in detected
                          if k in life["deadlines"]},
        })

    delivered = c19.read_json(c19.R18 / "SEARCH_POLICY_COMPARISON_RESULTS.json")
    primary = delivered["primary_result"]
    secondary = delivered["secondary_results"]

    def agrees(recomputed: float | None, delivered_value: float | None) -> Any:
        """Round-18 means were rounded to 5 decimals, this recompute keeps 6.

        Comparing the two with ``==`` reports a disagreement that is only a formatting
        difference, so the tolerance is registered here rather than in each field.
        """
        if recomputed is None or delivered_value is None:
            return None
        return {"equal": abs(recomputed - delivered_value) <= 1e-5,
                "recomputed": recomputed, "delivered": delivered_value}

    summary: dict[str, Any] = {}
    crosscheck: dict[str, Any] = {}
    for arm in c19.ARMS_R18:
        block = per_cell[arm]
        n = len(block)
        placed = sum(cell["placed"] for cell in block)
        detected = sum(cell["detected"] for cell in block)
        undetected = sum(cell["undetected"] for cell in block)
        on_time = sum(cell["on_time"] for cell in block)
        expired = sum(cell["expired"] for cell in block)
        missed = sum(cell["missed_after_detection"] for cell in block)
        bytes_total = sum(cell["airtime_bytes"] for cell in block)
        summary[arm] = {
            "cells": n, "placed": placed, "detected": detected, "undetected": undetected,
            "on_time": on_time, "expired": expired,
            "detected_but_not_on_time": missed,
            "cells_with_any_detected_but_not_on_time": sum(
                1 for cell in block if cell["missed_after_detection"] > 0),
            "closure": {"detected_plus_undetected": detected + undetected,
                        "equals_placed": detected + undetected == placed},
            "per_episode": {
                "placed": c19.ratio(placed, n), "detected": c19.ratio(detected, n),
                "undetected": c19.ratio(undetected, n), "on_time": c19.ratio(on_time, n),
                "expired": c19.ratio(expired, n),
                "detected_but_not_on_time": c19.ratio(missed, n),
                "airtime_bytes": c19.ratio(bytes_total, n),
                "emissions": c19.ratio(sum(c["emissions"] for c in block), n),
                "refresh_emissions": c19.ratio(
                    sum(c["refresh_emissions"] for c in block), n)},
            "bytes_per_detection": c19.ratio(bytes_total, detected),
            "bytes_per_on_time_task": c19.ratio(bytes_total, on_time),
        }
        crosscheck[arm] = {
            "undetected_mean": agrees(summary[arm]["per_episode"]["undetected"],
                                      primary[f"{arm}_mean_per_episode"]),
            "airtime_mean": agrees(summary[arm]["per_episode"]["airtime_bytes"],
                                   secondary[f"airtime_bytes_per_episode_{arm}"]["mean"]),
            "detections_mean": agrees(summary[arm]["per_episode"]["detected"],
                                      secondary[f"detections_per_episode_{arm}"]["mean"]),
            "on_time_mean": agrees(summary[arm]["per_episode"]["on_time"],
                                   secondary[f"on_time_tasks_per_episode_{arm}"]["mean"]),
        }

    payload = {
        "round": "19", "phase": "step 1 — archive round 18, no new episode",
        "sources": {
            "rows": "runs/search_trace_and_policy_20260925/rows/data_*.json",
            "lifecycle": "*_rp.jsonl (recomputed with integrated_sar/lifecycle_metrics.py)",
            "transcript": "*_ct.jsonl", "search_probe": "*_sp.jsonl",
            "crosscheck": "runs/search_trace_and_policy_20260925/"
                          "SEARCH_POLICY_COMPARISON_RESULTS.json"},
        "cells_expected": {"per_arm": 60, "total": 120,
                           "seeds": [c19.SEEDS_R18[0], c19.SEEDS_R18[-1]]},
        "checks": {name: checks[name] for name in sorted(checks)},
        "checks_total_per_arm": 60 * 2,
        "failed_checks": failures,
        "all_checks_pass": not failures,
        "batch_summary": summary,
        "crosscheck_against_delivered_document": crosscheck,
        "pairing": {
            "seeds_paired": len({c["seed"] for c in per_cell["A"]}
                                & {c["seed"] for c in per_cell["B"]}),
            "on_time_difference_A_minus_B_total": sum(
                a["on_time"] - b["on_time"] for a, b in
                zip(sorted(per_cell["A"], key=lambda c: c["seed"]),
                    sorted(per_cell["B"], key=lambda c: c["seed"]))),
            "airtime_difference_A_minus_B_total_mean": c19.ratio(
                sum(a["airtime_bytes"] - b["airtime_bytes"] for a, b in
                    zip(sorted(per_cell["A"], key=lambda c: c["seed"]),
                        sorted(per_cell["B"], key=lambda c: c["seed"]))),
                len(per_cell["A"]))},
        "read_for_round_19": {
            "task_level_slack_in_deployed_path": {
                "A": summary["A"]["detected_but_not_on_time"],
                "B": summary["B"]["detected_but_not_on_time"],
                "cells_with_any": summary["A"]["cells_with_any_detected_but_not_on_time"]
                + summary["B"]["cells_with_any_detected_but_not_on_time"],
                "note": ("detected − on_time over 120 episodes. The frozen six-goal deployed "
                         "difficulty produces almost no post-detection failure, so a byte "
                         "constraint judged against on-time tasks has no denominator here — "
                         "the same vacated judgement round 16 recorded as `undecidable`.")},
            "traffic_shape_under_the_frozen_defaults": {
                "emissions_equal_detections": all(
                    cell["emissions"] == cell["detected"]
                    for arm in c19.ARMS_R18 for cell in per_cell[arm]),
                "refresh_emissions_total": sum(
                    cell["refresh_emissions"] for arm in c19.ARMS_R18 for cell in per_cell[arm]),
                "subscribers": 6,
                "packets_lost_total": sum(cell["packets_lost"] for arm in c19.ARMS_R18
                                          for cell in per_cell[arm]),
                "note": ("`immediate` with no max-wait bound emits exactly one first report per "
                         "detection and never refreshes: the deployed default traffic is already "
                         "one packet-set per discovery, at payload x 6 receivers.")},
            "bytes_per_on_time_task": {
                "A": summary["A"]["bytes_per_on_time_task"],
                "B": summary["B"]["bytes_per_on_time_task"]}},
        "per_cell": {arm: sorted(per_cell[arm], key=lambda cell: cell["seed"])
                     for arm in c19.ARMS_R18},
    }
    c19.write_json("ROUND18_ARCHIVE_CLOSURE.json", payload)
    import json
    print(json.dumps({
        "cells": len(rows), "checks": sum(checks.values()),
        "failed": len(failures),
        "A": {k: summary["A"]["per_episode"][k] for k in
              ("detected", "undetected", "on_time", "airtime_bytes")},
        "B": {k: summary["B"]["per_episode"][k] for k in
              ("detected", "undetected", "on_time", "airtime_bytes")},
        "slack_detected_but_not_on_time": {
            "A": summary["A"]["detected_but_not_on_time"],
            "B": summary["B"]["detected_but_not_on_time"]},
        "crosscheck_all": all(entry["equal"] for block in crosscheck.values()
                              for entry in block.values())
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
