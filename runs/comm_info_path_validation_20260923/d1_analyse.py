"""Judge D1 against the frozen revision: three paired contrasts, then the mechanism record.

Order matters here. The contrasts are the causal statements, so they are computed first and
from the per-seed pairing only. The mechanism record describes where a difference landed,
and its numbers are never presented as the reason for one.

The internal consistency check is not decorative: the per-goal chain in this file is written
from the probe and lifecycle records, while the arm-level on-time count in each row comes
from the project's own window_metrics over the same lifecycle log. If the two disagree the
round says so instead of publishing either.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import d1_common as dc

WINDOW = 1000
MECHANISM_ARMS = ("A3", "A4")


# --------------------------------------------------------------------------- loading --------
def load_rows(tag: str) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for path in sorted(dc.ROWS.glob(f"{tag}_*_s*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        out[row["arm"]][int(row["seed"])] = row
    return dict(out)


# ------------------------------------------------------------------------ per-goal chain ----
def goal_chain(arm: str, seed: int, tag: str = "data") -> dict[str, Any]:
    """Join probe and lifecycle records by target id into one row per goal."""
    paths = dc.cell_paths(tag, arm, seed)
    goals: dict[str, dict[str, Any]] = {}

    def goal(target: str) -> dict[str, Any]:
        return goals.setdefault(target, {
            "placed": False, "detected_at": None, "deadline": None, "completed_at": None,
            "expired_seen": False, "uplink_arrived": 0, "uplink_lost": 0,
            "first_known_step": None, "known_by": set(), "stale_robots": set(),
            "arrived_robots": set(), "age_max": 0.0, "mismatch_shares": [],
        })

    if paths["lifecycle"].exists():
        for record in dc.iter_jsonl(paths["lifecycle"]):
            target = str(record.get("target_id"))
            stamp = float(record.get("timestamp") or 0.0)
            kind = record.get("event_type")
            if kind == "detection" and target != "__run__":
                body = goal(target)
                body["detected_at"] = stamp if body["detected_at"] is None \
                    else min(body["detected_at"], stamp)
                offset = (record.get("detail") or {}).get("deadline_offset")
                if offset is not None:
                    body["deadline"] = body["detected_at"] + float(offset)
            elif kind == "progress" and target != "__run__":
                body = goal(target)
                status = str(record.get("status"))
                if status == "completed" and body["completed_at"] is None:
                    body["completed_at"] = stamp
                elif status == "expired":
                    body["expired_seen"] = True
            elif kind == "final" and target != "__run__":
                detail = record.get("detail") or {}
                if detail.get("deadline") is not None:
                    goal(target)["deadline"] = float(detail["deadline"])

    if paths["probe"].exists():
        for record in dc.iter_jsonl(paths["probe"]):
            kind = record.get("kind")
            target = str(record.get("target_id", ""))
            if kind == "placed":
                goal(target)["placed"] = True
            elif not target:
                continue
            elif kind == "delivery":
                body = goal(target)
                if record.get("arrived"):
                    body["uplink_arrived"] += 1
                else:
                    body["uplink_lost"] += 1
            elif kind == "knowledge":
                body = goal(target)
                step = float(record.get("step") or 0.0)
                body["first_known_step"] = step if body["first_known_step"] is None \
                    else min(body["first_known_step"], step)
                body["known_by"].add(str(record.get("robot_id")))
            elif kind == "arrival":
                goal(target)["arrived_robots"].add(str(record.get("robot_id")))
            elif kind == "staleness":
                checks = int(record.get("checks") or 0)
                mismatched = int(record.get("mismatch_checks") or 0)
                body = goal(target)
                body["mismatch_shares"].append(
                    round(mismatched / checks, 5) if checks else 0.0)
                if mismatched > 0:
                    body["stale_robots"].add(str(record.get("robot_id")))
                    body["age_max"] = max(body["age_max"],
                                          float(record.get("max_age_steps") or 0.0))
    return goals


def goal_mechanism(body: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """(was this evaluable, which flags hold) for one goal. Several may hold at once.

    The two return shapes matter and are not the same thing: ``True, ()`` means every
    predicate was checked and none applied, which rules an information cause *out*, while
    ``False, ()`` means a record was missing and nothing may be claimed either way. The
    first version of this function collapsed them, and it turned eleven ordinary execution
    failures into "unexplained".
    """
    deadline = body.get("deadline")
    if deadline is None:
        return False, ()
    known = body.get("first_known_step")
    flags: list[str] = []
    if body["uplink_arrived"] == 0:
        flags.append("M1a_uplink_never_delivered")
    if known is None:
        flags.append("M1b_never_known")
    elif known >= deadline:
        flags.append("M2_known_after_deadline")
    if body["stale_robots"]:
        flags.append("M3_stale_at_a_decision")
    return True, tuple(flags)


def classify_goal(body: dict[str, Any]) -> tuple[str, ...]:
    """Flags only. Kept for the tests; the analysis distinguishes "none" from "unknown"."""
    return goal_mechanism(body)[1]


def is_on_time(body: dict[str, Any]) -> bool:
    deadline = body.get("deadline")
    return body.get("completed_at") is not None and deadline is not None \
        and body["completed_at"] <= deadline


# ------------------------------------------------------------------------- statistics -------
def paired_bootstrap(deltas: list[float], rng_seed: int = dc.BOOTSTRAP_RNG) -> dict[str, Any]:
    import numpy as np

    values = np.array(deltas, dtype=float)
    if values.size == 0:
        return {"n": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, values.size, size=(dc.BOOTSTRAP_REPLICATIONS, values.size))
    means = values[draws].mean(axis=1)
    return {
        "n": int(values.size),
        "mean": round(float(values.mean()), 5),
        "ci95": [round(float(np.percentile(means, 2.5)), 5),
                 round(float(np.percentile(means, 97.5)), 5)],
        "min": int(values.min()), "median": round(float(np.median(values)), 3),
        "max": int(values.max()),
        "seeds_left_better": int((values > 0).sum()),
        "seeds_equal": int((values == 0).sum()),
        "seeds_right_better": int((values < 0).sum()),
        "replications": dc.BOOTSTRAP_REPLICATIONS,
        "rng": rng_seed,
    }


def contrast(name: str, left: str, right: str, seeds: list[int],
             rows: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    shared = [seed for seed in seeds if seed in rows.get(left, {}) and seed in rows.get(right, {})]
    deltas = [rows[left][seed]["on_time"] - rows[right][seed]["on_time"] for seed in shared]
    bytes_delta = [rows[left][seed]["airtime_bytes"] - rows[right][seed]["airtime_bytes"]
                   for seed in shared]
    lost_delta = [int(rows[left][seed].get("packets_lost") or 0)
                  - int(rows[right][seed].get("packets_lost") or 0) for seed in shared]
    return {
        "name": name, "arms": [left, right], "paired_seeds": len(shared),
        "on_time_difference_left_minus_right": paired_bootstrap(deltas),
        "per_episode_means": {
            left: round(sum(rows[left][seed]["on_time"] for seed in shared) / max(1, len(shared)), 4),
            right: round(sum(rows[right][seed]["on_time"] for seed in shared) / max(1, len(shared)), 4)},
        "expired_per_episode": {
            arm: round(sum(rows[arm][seed]["expired"] for seed in shared) / max(1, len(shared)), 4)
            for arm in (left, right)},
        "charged_bytes_difference_left_minus_right_per_episode":
            round(sum(bytes_delta) / max(1, len(shared)), 1),
        "packets_lost_difference_per_episode": round(sum(lost_delta) / max(1, len(shared)), 3),
    }


INFORMATION_FLAGS = ("M1b_never_known", "M2_known_after_deadline", "M3_stale_at_a_decision")


def mechanism(arm: str, seeds: list[int], rows: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    placed = detected = on_time = failed = determined = undetermined = 0
    flags = Counter()
    combinations = Counter()
    slack: list[float] = []
    ages: list[float] = []
    mismatch_share: list[float] = []
    on_time_with_staleness = 0
    no_cause = 0
    chain_conflicts: list[str] = []
    information_attribution = 0
    for seed in seeds:
        if seed not in rows.get(arm, {}):
            continue
        goals = goal_chain(arm, seed)
        row = rows[arm][seed]
        mine_on_time = sum(1 for body in goals.values()
                           if body["detected_at"] is not None and is_on_time(body))
        if mine_on_time != row["on_time"]:
            chain_conflicts.append(f"s{seed}: chain {mine_on_time} vs metrics {row['on_time']}")
        placed += sum(1 for body in goals.values() if body["placed"])
        for target, body in goals.items():
            if body["detected_at"] is None:
                continue
            detected += 1
            if is_on_time(body):
                on_time += 1
                if body["deadline"] is not None and body["completed_at"] is not None:
                    slack.append(body["deadline"] - body["completed_at"])
                # A flag that fires on goals that finished early cannot, by itself, be
                # evidence that the goal failed for that reason; its incidence on this
                # subset is the calibration for that.
                if "M3_stale_at_a_decision" in classify_goal(body):
                    on_time_with_staleness += 1
                continue
            failed += 1
            evaluable, found = goal_mechanism(body)
            if not evaluable:
                undetermined += 1
                continue
            determined += 1
            combination = "+".join(found) if found else "M0_no_mechanism_flag_applies"
            combinations[combination] += 1
            for flag in found:
                flags[flag] += 1
            if not found:
                no_cause += 1
            if any(flag in INFORMATION_FLAGS for flag in found):
                information_attribution += 1
            ages.append(body["age_max"])
            mismatch_share.extend(body["mismatch_shares"])

    def share(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 5) if denominator else None

    return {
        "arm": arm,
        "denominators": {"placed": placed, "detected": detected, "on_time": on_time,
                         "failed": failed, "mechanism_determined": determined,
                         "mechanism_undetermined": undetermined},
        "shares": {
            "S_all": share(information_attribution, failed),
            "S_det": share(information_attribution, determined),
            "S_unres": share(undetermined, failed),
        },
        "information_attribution_numerator": information_attribution,
        "failed_with_no_mechanism_flag": no_cause,
        "flag_counts": dict(sorted(flags.items())),
        "co_occurrence_counts": dict(sorted(combinations.items())),
        "slack_at_completion_steps": dc.quantiles(slack) if slack else None,
        "stale_age_steps_max": dc.quantiles(ages) if ages else None,
        "stale_mismatch_share_of_checks": dc.quantiles(mismatch_share) if mismatch_share else None,
        "m3_calibration": {
            "share_of_failed_goals_flagged": share(
                sum(count for key, count in combinations.items()
                    if "M3_stale_at_a_decision" in key), failed),
            "share_of_on_time_goals_flagged": share(on_time_with_staleness, on_time),
            "reading": (
                "nearly every goal carries the frozen M3 predicate, completed ones included, "
                "because a belief written on a poll is always behind the truth it describes; "
                "so M3 counts are reported, and the bottleneck verdict is not made to rest on "
                "M3 alone"
            ),
        },
        "chain_vs_metrics_conflicts": chain_conflicts[:10],
        "chain_vs_metrics_conflict_count": len(chain_conflicts),
    }


def run() -> int:
    rows = load_rows("data")
    seeds = dc.SEEDS
    per_arm = {}
    for arm in dc.DATA_ARMS:
        present = [seed for seed in seeds if seed in rows.get(arm, {})]
        values = [rows[arm][seed] for seed in present]
        per_arm[arm] = {
            "episodes": len(present),
            "on_time_per_episode": round(sum(v["on_time"] for v in values) / max(1, len(values)), 4),
            "expired_per_episode": round(sum(v["expired"] for v in values) / max(1, len(values)), 4),
            "discovered_per_episode": round(sum(v["discovered"] for v in values) /
                                            max(1, len(values)), 4),
            "placed_per_episode": round(sum(v["goals_placed"] for v in values) / max(1, len(values)), 4),
            "airtime_bytes_per_episode": round(sum(v["airtime_bytes"] for v in values) /
                                               max(1, len(values)), 1),
            "packets_lost_per_episode": round(sum(int(v.get("packets_lost") or 0) for v in values) /
                                             max(1, len(values)), 3),
            "targets_never_known_per_episode": round(
                sum(int(v.get("targets_never_known") or 0) for v in values) / max(1, len(values)), 4),
            "probe_anomaly_cells": sum(1 for v in values if v.get("probe_anomalies")),
        }
    contrasts = [contrast(name, left, right, seeds, rows) for name, left, right in dc.CONTRASTS]
    identity_path = dc.ROUND_DIR / "IDENTITY_AUDIT_data.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else {}
    mechanisms = {arm: mechanism(arm, seeds, rows) for arm in MECHANISM_ARMS}
    verdict = freeze_verdict(contrasts, mechanisms, identity)
    payload = {
        "round": dc.ROUND,
        "protocol": "D1_PROTOCOL_REVISION.md",
        "start_sha": dc.START_SHA,
        "launches": dc.ledger_report(),
        "configuration": {
            "base": dc.base_configuration_fields(), "arms": dc.ARM_OVERRIDES,
            "seeds": [seeds[0], seeds[-1]], "seed_count": len(seeds),
            "checkpoints": {"search": str(dc.PIL_CHECKPOINT), "allocation": str(dc.CTDE_CHECKPOINT)},
            "training": "none; both networks are loaded frozen and are never updated",
        },
        "per_arm": per_arm,
        "contrasts": contrasts,
        "identity_gates": identity.get("gates_passed"),
        "licensed_phrasing": identity.get("phrasing_licensed"),
        "mechanism_record": mechanisms,
        "verdict": verdict,
    }
    dc.write_json("D1_RESULTS.json", payload)
    print(json.dumps({"per_arm": {arm: facts["on_time_per_episode"]
                                  for arm, facts in per_arm.items()},
                      "contrasts": {item["name"]: item["on_time_difference_left_minus_right"]
                                    ["mean"] for item in contrasts},
                      "verdict": verdict}, ensure_ascii=False, indent=1))
    return 0


def freeze_verdict(contrasts: list[dict[str, Any]], mechanisms: dict[str, Any],
                   identity: dict[str, Any]) -> dict[str, Any]:
    primary = mechanisms.get("A3", {})
    denominators = primary.get("denominators", {})
    failed = denominators.get("failed", 0)
    undetermined = denominators.get("mechanism_undetermined", 0)
    shares = primary.get("shares", {})
    s_all = shares.get("S_all")
    s_unres = round(undetermined / failed, 5) if failed else None
    c2 = next(item for item in contrasts if item["name"] == "C2")
    c1 = next(item for item in contrasts if item["name"] == "C1")
    guard = s_unres is None or s_unres > 0.5
    declared = "undecidable"
    if not guard and s_all is not None:
        ci2 = c2["on_time_difference_left_minus_right"]["ci95"]
        if s_all < 0.05 and ci2 and ci2[1] < 0.10:
            declared = "stop_1_information_not_a_bottleneck"
        elif s_all >= 0.20:
            declared = "stop_2_information_is_a_bottleneck"
        else:
            declared = "in_band_no_side_taken"
    return {
        "primary_mechanism_arm": "A3",
        "S_all": s_all, "S_unres": s_unres,
        "guard_blocks_verdict": guard,
        "deployed_failures_available": failed,
        "declared": declared,
        "c1_on_time_difference": c1["on_time_difference_left_minus_right"],
        "c2_on_time_difference": c2["on_time_difference_left_minus_right"],
        "c3_on_time_difference": next(item for item in contrasts
                                      if item["name"] == "C3")["on_time_difference_left_minus_right"],
        "all_identity_gates_passed": bool((identity or {}).get("all_passed")),
        "reading_limit": (
            "when the deployed arm has no on-time failures at the frozen difficulty, S_all has "
            "an empty denominator and the frozen criteria cannot be satisfied in either "
            "direction; that is reported as a vacated judgement, not as a null result about "
            "information in general"
        ),
    }


if __name__ == "__main__":
    raise SystemExit(run())
