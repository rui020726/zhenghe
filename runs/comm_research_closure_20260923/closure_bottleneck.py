"""Assemble the joint-system bottleneck verdict from the reconstructed chains.

Two rules shape this file:

* every recomputed quantity is checked against a published one, and the check is
  reported whether or not it agrees — an audit that only prints its own numbers is
  indistinguishable from a guess;
* a bucket that the logs cannot attribute is printed as ``not attributable`` with the
  reason, rather than being folded into the neighbouring bucket, because folding is
  exactly how "the allocation side loses 33 goals" would turn into a false finding.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import closure_common as cc

REPO = cc.REPO_ROOT

INFORMATION_CLASSES = ("c2_info_never_arrived", "c2_info_arrived_after_deadline")
EXECUTION_CLASSES = ("c3_known_never_dispatched", "c3_dispatched_never_worked",
                     "c3_worked_but_incomplete", "c5_unresolved_at_window_end",
                     "c4_complete_late")

PATH_LABELS = {
    "P1_deployed_live_search": "deployed path, live PILOC search (coalition_validation systemloop)",
    "P2_deployed_replayed_search": "deployed path, replayed detections (joint_eval pilot)",
    "P3_isolated_budgeted_comm": "isolated path, hard 29 300 B budget (task-pressure layers)",
    "P4_lossy_confirmation": "deployed and isolated lossy channels (S3 rule confirmation)",
}


def load(name: str) -> dict[str, Any]:
    return json.loads((cc.ROUND_DIR / name).read_text(encoding="utf-8"))


def decompose(arm: dict[str, Any], information_attributable: bool) -> dict[str, Any]:
    classes = arm["per_goal_classes"]
    execution = sum(count for key, count in classes.items() if key in EXECUTION_CLASSES)
    information = sum(count for key, count in classes.items() if key in INFORMATION_CLASSES)
    return {
        "goals_placed": arm["goals_placed"],
        "1_search_miss": arm["not_discovered"],
        "2_information_loss": information if information_attributable else None,
        "2_information_loss_note": None if information_attributable else (
            "this path has no per-recipient knowledge record and its uplink was configured "
            "lossless with zero delay, so 'information did not arrive' is not a measurable "
            "quantity here; the loss is reported where it can be seen, not assumed to be zero"
        ),
        "3_allocation_or_execution_loss": execution,
        "on_time": arm["on_time_total"],
        "loss_share_of_placed": {
            "search_miss": cc.ratio(arm["not_discovered"], arm["goals_placed"]),
            "information": cc.ratio(information, arm["goals_placed"]) if information_attributable
            else None,
            "allocation_or_execution": cc.ratio(execution, arm["goals_placed"]),
        },
        "per_goal_class_comm_facts": arm.get("per_goal_class_comm_facts", {}),
    }


def belief_source_recomputation() -> dict[str, Any]:
    """Re-derive the 'how much of first knowing bypasses the uplink' figure from raw rows."""
    path = REPO / "runs/comm_control_20260919/TRAFFIC_PROBE.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    agg: dict[str, Counter] = {}
    for row in document["rows"]:
        key = ("isolated" if row["isolated"] else "deployed") + \
              ("+silent_uplink" if row["muted"] else "+uplink_on")
        bucket = agg.setdefault(key, Counter())
        bucket["episodes"] += 1
        for source, count in (row.get("first_belief_source_by_pair") or {}).items():
            bucket[f"first_knowing_{source}"] += count
        bucket["belief_writes"] += sum((row.get("belief_writes_by_source") or {}).values())
        bucket["targets_known"] += int(row.get("distinct_targets_known") or 0)
    out: dict[str, Any] = {}
    for key, bucket in sorted(agg.items()):
        first_total = sum(value for name, value in bucket.items()
                          if name.startswith("first_knowing_"))
        via_uplink = bucket.get("first_knowing_uplink", 0)
        out[key] = {**dict(sorted(bucket.items())), "first_knowing_total": first_total,
                    "share_via_uplink": cc.ratio(via_uplink, first_total),
                    "share_bypassing_uplink": cc.ratio(first_total - via_uplink, first_total)
                    if first_total else None}
    return out


CHANNEL_FIELDS = ("transmission_attempts", "packets_delivered", "packets_lost", "emissions",
                  "first_emissions")
OUTCOME_FIELDS = ("discovered_per_episode", "on_time_per_episode", "expired_per_episode",
                  "unresolved_per_episode", "belief_age_steps", "total_bytes",
                  "detection_to_assignment", "detection_to_work")


def setting_contrast(aggregate_path: Path) -> dict[str, Any]:
    """Same arm, same 200 streams, only the self-sensing bypass toggled.

    Pairing is asserted, not assumed: the contrast is reported per arm together with
    whether that arm's forward-channel ledger is identical across the two settings.
    Where the ledger differs, the two cells are not a paired comparison and are marked so.
    """
    document = json.loads(aggregate_path.read_text(encoding="utf-8"))["settings"]
    out: dict[str, Any] = {"normal_lossy_vs_isolated_lossy": {},
                           "isolated_lossy_vs_lossless": {}}
    for arm, normal in sorted(document["normal-lossy"].items()):
        isolated = document["isolated-lossy"][arm]
        lossless = document["isolated-lossless"][arm]
        identical = all(normal[key] == isolated[key] for key in CHANNEL_FIELDS)
        out["normal_lossy_vs_isolated_lossy"][arm] = {
            "channel_ledger_identical": identical,
            "channel": {key: normal[key] for key in CHANNEL_FIELDS},
            "normal": {key: normal[key] for key in OUTCOME_FIELDS},
            "isolated": {key: isolated[key] for key in OUTCOME_FIELDS},
            "difference_normal_minus_isolated": {
                key: round(normal[key] - isolated[key], 4) for key in OUTCOME_FIELDS[:4]},
        }
        out["isolated_lossy_vs_lossless"][arm] = {
            "on_time_lossless": lossless["on_time_per_episode"],
            "on_time_lossy": isolated["on_time_per_episode"],
            "cost_of_channel_loss": round(lossless["on_time_per_episode"]
                                          - isolated["on_time_per_episode"], 4),
        }
    return out


def published_targets() -> list[dict[str, Any]]:
    chain = load("CHAIN_RECONSTRUCTION.json")
    manifest = load("CLOSURE_INPUT_MANIFEST.json")
    geometry = json.loads((REPO / "runs/comm_action_geometry_audit_20260923/"
                           "COMM_ACTION_GEOMETRY_AUDIT.json").read_text(encoding="utf-8"))
    loop = json.loads((REPO / "runs/coalition_validation_20260918/SYSTEM_LOOP.json")
                      .read_text(encoding="utf-8"))
    attribution = json.loads((REPO / "runs/joint_eval_20260916/expired_attribution.json")
                             .read_text(encoding="utf-8"))
    gates = {layer: json.loads((REPO / "runs/comm_task_pressure_learnability_20260922/"
                                f"GATES_{layer}.json").read_text(encoding="utf-8"))["gates"]
             for layer in ("g6", "g8", "g10", "g12")}
    checks: list[dict[str, Any]] = []

    def check(quantity: str, published: Any, recomputed: Any) -> None:
        checks.append({"quantity": quantity, "published": published, "recomputed": recomputed,
                       "agrees": published == recomputed})

    p1 = chain["paths"]["P1_deployed_live_search"]["groups"]["systemloop"]["arms"]
    for arm, key in (("baseline", "baseline"), ("candidate", "candidate")):
        for field, our in (("goals_placed", "goals_placed"),
                           ("discovered", "goals_discovered"),
                           ("on_time_completed", "on_time_total")):
            check(f"P1 {arm} {field}", loop["arms"][key][field], p1[arm][our])
        check(f"P1 {arm} expired", loop["arms"][key]["expired"],
              p1[arm]["registry_class_counts"]["expired"])

    p2 = chain["paths"]["P2_deployed_replayed_search"]["groups"]["joint_eval"]["arms"]
    for arm in ("nocoord", "rs0.25", "rs0.5"):
        published = attribution["arms"][arm]
        counts = published["counts"]
        check(f"P2 {arm} completed", published["latency"]["completed"]["n"],
              p2[arm]["registry_class_counts"]["completed"])
        check(f"P2 {arm} expired", len(published["expired"]),
              p2[arm]["registry_class_counts"]["expired"])
        if "never_assigned" in counts:
            check(f"P2 {arm} never_assigned", counts["never_assigned"],
                  p2[arm]["per_goal_classes"].get("c3_known_never_dispatched", 0))

    per_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell in geometry["per_cell"]:
        per_cell.setdefault((cell["layer"], cell["arm"]), []).append(cell)
    mismatches: list[str] = []
    for (layer, arm), cells in per_cell.items():
        ours = chain["paths"]["P3_isolated_budgeted_comm"]["groups"][layer]["arms"].get(arm)
        if ours is None:
            mismatches.append(f"{layer}/{arm} missing from this scan")
            continue
        for field, our_key in (("on_time", "on_time_total"), ("detected", "goals_discovered"),
                               ("placed", "goals_placed")):
            if sum(float(cell[field] or 0) for cell in cells) != ours[our_key]:
                mismatches.append(f"{layer}/{arm} {field}: geometry "
                                  f"{sum(float(cell[field] or 0) for cell in cells)} vs "
                                  f"chain {ours[our_key]}")
    check("P3 all 36 layer/arm cells (placed, detected, on_time) vs the geometry audit",
          "0 disagreements", mismatches or "0 disagreements")

    for layer in ("g6", "g8", "g10", "g12"):
        evidence = gates[layer]["L2"]["evidence"]
        arm = evidence["best_fixed_rule"]
        rates = [cell["on_time"] / cell["detected"] for cell in per_cell[(layer, arm)]
                 if cell["detected"]]
        rate = round(sum(rates) / len(rates), 5)
        check(f"{layer} best-rule on-time/detected (mean of per-stream rates)",
              evidence["best_on_time_rate_of_detected"], rate)
        check(f"{layer} best-rule on-time total vs the geometry audit cells",
              sum(int(cell["on_time"]) for cell in per_cell[(layer, arm)]),
              chain["paths"]["P3_isolated_budgeted_comm"]["groups"][layer]["arms"][arm]
              ["on_time_total"])

    never_reported = {"send_once": 8, "periodic_20": 43, "bound_only_3_10": 48,
                      "periodic_5": 66, "event": 73}
    for arm, published in never_reported.items():
        ours = chain["paths"]["P3_isolated_budgeted_comm"]["groups"]["g12"]["arms"][arm]
        check(f"g12 {arm} goals detected but never reported",
              published, ours["per_goal_classes"].get("c2_info_never_arrived", 0))

    probe = belief_source_recomputation()
    deployed_muted = probe.get("deployed+silent_uplink", {})
    check("deployed first-knowing bypassing the uplink (390 of 408)", 390,
          deployed_muted.get("first_knowing_total"))
    check("of those, via line-of-sight refresh", 388, deployed_muted.get("first_knowing_los_refresh"))
    check("of those, via one-hop gossip", 2, deployed_muted.get("first_knowing_gossip"))
    check("isolated path first-knowing with the uplink silenced", 0,
          probe.get("isolated+silent_uplink", {}).get("first_knowing_total", 0))
    check("pressure-round episode launches recorded in the ledgers", 666,
          manifest["pressure_launch_ledger"]["starts"])
    check("sealed block 800000..800399 artefacts on disk", 0,
          manifest["sealed_block_800000_800399_artefacts_on_disk"])

    confirm = json.loads((REPO / "runs/comm_rule_confirm_20260921/"
                          "COMM_RULE_CONFIRMATION_RESULTS.json").read_text(encoding="utf-8"))
    lossy = confirm["settings"]["isolated-lossy"]
    check("isolated-lossy ack_retry_2 minus noack_send_once, on-time per episode",
          0.835, round(lossy["ack_retry_2"]["on_time_per_episode"]
                       - lossy["noack_send_once"]["on_time_per_episode"], 4))
    check("isolated-lossy total bytes difference of the same pair (the eighth round's "
          "canonical +26 354.1 B/ep)",
          26354.1, round(lossy["ack_retry_2"]["total_bytes"]
                         - lossy["noack_send_once"]["total_bytes"], 1))
    normal = confirm["settings"]["normal-lossy"]
    check("normal-lossy: the same pair is indistinguishable on task effect",
          0.0, round(normal["ack_retry_2"]["on_time_per_episode"]
                     - normal["noack_send_once"]["on_time_per_episode"], 4))
    return checks


def ranking(chain: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for name, path in sorted(chain["paths"].items()):
        rows = []
        if path.get("configuration", {}).get("mode") == "published_aggregate":
            continue
        for group, data in sorted(path["groups"].items()):
            for arm, values in sorted(data["arms"].items()):
                information_attributable = bool(
                    path["configuration"]["channel"].get("transcript")) and not bool(
                    path["configuration"]["channel"].get("line_of_sight_refresh", False))
                parts = {
                    "search_miss": values["not_discovered"],
                    "information": sum(count for key, count in
                                       values["per_goal_classes"].items()
                                       if key in INFORMATION_CLASSES)
                    if information_attributable else None,
                    "allocation_or_execution": sum(
                        count for key, count in values["per_goal_classes"].items()
                        if key in EXECUTION_CLASSES),
                }
                scored = {k: v for k, v in parts.items() if v is not None}
                out.append({
                    "path": name, "path_label": PATH_LABELS.get(name, name),
                    "group": group, "arm": arm, "buckets": parts,
                    "largest_attributable_loss": max(scored, key=scored.get) if scored else None,
                    "information_bucket_attributable": information_attributable,
                })
    return out


def run() -> int:
    cc.install_env_tripwire()
    chain = load("CHAIN_RECONSTRUCTION.json")
    stages = load("STAGE_CAPABILITY.json")
    checks = published_targets()
    paths: dict[str, Any] = {}
    for name, path in sorted(chain["paths"].items()):
        configuration = path["configuration"]
        entry: dict[str, Any] = {"label": PATH_LABELS.get(name, name),
                                 "channel": configuration["channel"],
                                 "search_backend": configuration["search_backend"],
                                 "pairing": configuration["pairing"]}
        if configuration.get("mode") == "published_aggregate":
            entry["published_aggregate"] = path["published_aggregate"]
            entry["transcript_spot_checks"] = path["transcript_spot_checks"]
            entry["note"] = configuration.get("why_not_rescanned")
        else:
            information_attributable = bool(configuration["channel"].get("transcript")) and \
                not bool(configuration["channel"].get("line_of_sight_refresh", False))
            entry["groups"] = {
                group: {arm: decompose(values, information_attributable)
                        for arm, values in sorted(data["arms"].items())}
                for group, data in sorted(path["groups"].items())}
            entry["paired_contrasts"] = path["paired_contrasts"]
        paths[name] = entry
    agreements = sum(1 for item in checks if item["agrees"])
    payload = {
        "round": cc.ROUND,
        "status": ("read-only inventory of archived runs; 0 episodes, 0 new streams, "
                   "0 optimizer.step, 0 checkpoint loads, no production parameter changed"),
        "eight_stages": [f"{item['stage']}_{item['key']}" for item in stages["stages"]],
        "stage_capability_headline": stages["headline"],
        "lifecycle_event_type_set": stages["lifecycle_event_type_set"],
        "absent_record_checks": stages["absent_record_checks"],
        "paths": paths,
        "loss_ranking": ranking(chain),
        "setting_contrast_same_arm": setting_contrast(
            REPO / "runs/comm_rule_confirm_20260921/COMM_RULE_CONFIRMATION_RESULTS.json"),
        "missing_attribution": [
            "stage 1 (placed): no record of a goal that nothing ever detected, so 'never "
            "discovered' is counted against the configured num_goals, not against a log",
            "stage 3 (delivered): no per-recipient loss record; attempts = arrived + lost closes "
            "per send, but which robot was left uninformed is unrecoverable",
            "stage 4 (robot knew): belief writes are not an event type; the only per-(goal, robot) "
            "record is a transcript deliver row, and three non-message channels also write belief",
            "stage 6 (arrived) and 7 (started work): no event type; both are proxies from "
            "fingerprint-changed progress rows, so they can postdate or predate each other (one "
            "hop mean in this file is negative for exactly that reason)",
            "no archived run has live search, a non-trivial channel and per-goal execution "
            "attribution at once: the deployed runs are lossless, the lossy runs replay "
            "detections, and the isolated runs switch the environment's own sensing off",
        ],
        "not_claimed": [
            "no bucket in this file is an intervention's causal gain: the deployed-path "
            "information bucket is reported as not attributable rather than as zero",
            "arm-to-arm differences within a path are designed contrasts on shared scenarios; "
            "cross-path differences (a P1 rate against a P3 rate, for example) are not",
            "the setting contrast compares identical transmission ledgers with the sensing "
            "bypass toggled, which measures the value of the environment's own sensing, not the "
            "value of any communication policy",
            "nothing here reopens the learning question or proposes a change to a production "
            "default",
        ],
        "belief_source_recomputation": belief_source_recomputation(),
        "reconciliation_with_published": checks,
        "reconciliation_summary": {
            "checks": len(checks), "agreeing": agreements, "disagreeing": len(checks) - agreements,
            "disagreements": [item for item in checks if not item["agrees"]],
        },
        "env_tripwire": cc.tripwire_report(),
    }
    cc.write_json("JOINT_SYSTEM_BOTTLENECK_AUDIT.json", payload)
    print(json.dumps(payload["reconciliation_summary"], ensure_ascii=False, indent=1)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
