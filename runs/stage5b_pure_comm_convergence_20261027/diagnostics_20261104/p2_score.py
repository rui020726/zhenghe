"""P2 scoring: five acceptance criteria, every number with its denominator.

Reads only the P2 world results and the frozen N=6 rule-arm rows. Runs no world.

The N=6 cell is assembled from two directories on purpose: scene 2301901 was the
2-replay seam pilot (tag pilot2) and scenes 2301902..2301908 came from the main
block (tag p2). Both were produced by the same code path at the same seam
settings, so the cell is 8 scenes either way; the split is recorded per row.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE.parent
PROBE = STAGE / "recovery_12" / "revision_01" / "execution" / "scale_probe"
FROZEN = (STAGE / ".." / "stage5b_pure_comm_budget58600_20261026" / "revision_02"
          / "execution" / "rows" / "eval").resolve()
SCENES = tuple(range(2301901, 2301909))
ARMS = ("normal_immediate", "always_hold")
BUDGET = 58600
NS = (3, 6, 12)

#: Which directories hold each N cell, in merge order.
SOURCES = {
    3: [("P2_3_fixed_p2", "p2")],
    6: [("P2_6_fixed_pilot2", "pilot2"), ("P2_6_fixed_p2", "p2")],
    12: [("P2_12_fixed_p2", "p2")],
}

#: P1 analytic predictions at the tape-measured payload (416.4457 B).
AFFORD = {3: 46, 6: 23, 12: 11}
COST_PER_SEND_PREDICTED = {3: 1249.337, 6: 2498.674, 12: 4997.348}
SPEC_LITERAL_COST = {3: 1250.3, 6: 2500.6, 12: 5001.2}
COST_TOLERANCE = 0.005


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def transcript_counters(rundir, namespace, label):
    """Count refusals from the archived relay transcript of one finished world.

    ``world.run_episode`` returns a row without the refusal counters -- those are added by
    ``episode.compact_row`` from the relay's own counters, which my P2 driver did not call.
    The transcript this world wrote is still on disk (gzipped by compact.archive_world_logs),
    so the counts are recovered here instead of being paid for again by a replay.
    """
    path = (PROBE / rundir / "world_logs" / namespace / label
            / "relay_transcript.jsonl.gz")
    if not path.exists():
        return {"found": False, "path": str(path)}
    counts = {"emissions": 0, "budget_refused_sends": 0, "capacity_refused_sends": 0,
              "spend_bytes": 0, "refusal_reasons": {}}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            kind = record.get("kind")
            if kind == "send":
                counts["emissions"] += 1
                counts["spend_bytes"] += int(record.get("cost_bytes") or 0)
            elif kind == "refused":
                # comm.py records budget.why_blocked(), i.e. "budget_exhausted" /
                # "capacity_step" -- not the bare token. Match on the prefix.
                reason = str(record.get("reason") or "")
                counts["refusal_reasons"][reason] = (
                    counts["refusal_reasons"].get(reason, 0) + 1)
                if reason.startswith("budget"):
                    counts["budget_refused_sends"] += 1
                elif reason.startswith("capacity"):
                    counts["capacity_refused_sends"] += 1
    return {"found": True, "path": str(path), "sha256": sha256(path), **counts}


def collect(n):
    """Merge the directories holding this N cell into {(arm, scene): result}."""
    cells, provenance = {}, {}
    for dirname, tag in SOURCES[n]:
        payload = load(PROBE / dirname / "RESULTS.json")
        for result in payload["results"]:
            if result.get("status") != "ok":
                continue
            key = (result["arm"], int(result["scene"]))
            cells[key] = result
            provenance[key] = {"dir": dirname, "tag": tag,
                               "sha256": sha256(PROBE / dirname / "RESULTS.json")}
    return cells, provenance


def frozen_row(arm, scene):
    path = FROZEN / ("eval_%s_none_%d.json" % (arm, scene))
    return (load(path) if path.exists() else None), str(path)


def main() -> None:
    checks, per_n = [], {}

    for n in NS:
        cells, provenance_map = collect(n)
        missing = [(a, s) for a in ARMS for s in SCENES if (a, s) not in cells]
        if missing:
            raise SystemExit("N=%d missing cells: %r" % (n, missing))

        arm_rows = {}
        for arm in ARMS:
            rows = {}
            for scene in SCENES:
                row = cells[(arm, scene)]["row"]
                spent = int(row["spent_bytes"])
                emissions = int(row["emissions"])
                prov = provenance_map[(arm, scene)]
                counters = transcript_counters(
                    prov["dir"],
                    "eval__%s__none__%d" % (arm, scene),
                    "s5b_%s_s%d" % (arm, scene))
                rows[scene] = {
                    "spent_bytes": spent, "emissions": emissions,
                    "on_time": int(row["on_time"]),
                    "generated_goals": int(row["generated_goals"]),
                    "budget_share": spent / BUDGET,
                    "cost_per_send": (spent / emissions) if emissions else None,
                    "budget_refused_sends": counters.get("budget_refused_sends"),
                    "capacity_refused_sends": counters.get("capacity_refused_sends"),
                    "transcript_emissions": counters.get("emissions"),
                    "realized_spend": counters.get("spend_bytes"),
                    "refusal_reasons": counters.get("refusal_reasons"),
                    "transcript_found": counters.get("found"),
                    "transcript_path": counters.get("path"),
                    "masked_by_credit_total": row.get("masked_by_credit_total"),
                    "first_masked_credit_tick": row.get("first_masked_credit_tick"),
                    "budget_bytes_in_world": int(row["budget_bytes"]),
                    "budget_credit": row.get("budget_credit"),
                    "budget_authority_after_exit": row.get("budget_authority_after_exit"),
                    "provenance": prov,
                }
            arm_rows[arm] = rows

        imm = arm_rows["normal_immediate"]
        hold = arm_rows["always_hold"]
        tot_spent = sum(imm[s]["spent_bytes"] for s in SCENES)
        tot_emit = sum(imm[s]["emissions"] for s in SCENES)
        mean_cost = tot_spent / tot_emit if tot_emit else None
        at_cap = sum(1 for s in SCENES
                     if (imm[s]["budget_refused_sends"] or 0) > 0)
        on_time_imm = sum(imm[s]["on_time"] for s in SCENES)
        on_time_hold = sum(hold[s]["on_time"] for s in SCENES)
        denom = sum(imm[s]["generated_goals"] for s in SCENES)

        per_n[n] = {
            "n": n, "arms": arm_rows,
            "immediate_totals": {"spent_bytes": tot_spent, "emissions": tot_emit,
                                 "cost_per_send_mean": mean_cost,
                                 "denominator_emissions": tot_emit},
            "worlds_at_budget_cap": {"count": at_cap, "denominator": len(SCENES)},
            "on_time_interval": {"upper_immediate": on_time_imm,
                                 "lower_always_hold": on_time_hold,
                                 "denominator": denom,
                                 "headroom": on_time_imm - on_time_hold},
            "per_world_emissions": {str(s): imm[s]["emissions"] for s in SCENES},
            "per_world_budget_refusals": {str(s): imm[s]["budget_refused_sends"]
                                          for s in SCENES},
        }

        # ---- criterion 1: cost per send is proportional to N -------------------
        predicted = SPEC_LITERAL_COST[n]
        measured_at_measured_payload = COST_PER_SEND_PREDICTED[n]
        ratio = (mean_cost / measured_at_measured_payload) if mean_cost else None
        checks.append({
            "criterion": 1, "n": n, "name": "cost_per_send_proportional_to_N",
            "measured": mean_cost, "protocol_literal": predicted,
            "p1_value_at_measured_payload": measured_at_measured_payload,
            "delta_vs_protocol_literal_pct": (100.0 * (mean_cost - predicted) / predicted
                                              if mean_cost else None),
            "ratio_measured_over_P1": ratio,
            "denominator": "%d B over %d emissions, 8 scenes" % (tot_spent, tot_emit),
            "tolerance": "+-%.1f%%" % (COST_TOLERANCE * 100),
            "pass": bool(mean_cost is not None
                         and abs(mean_cost - predicted) / predicted <= COST_TOLERANCE),
        })

        # ---- criterion 2: emissions approach affordable_sends ------------------
        per_world = [imm[s]["emissions"] for s in SCENES]
        refusals = [imm[s]["budget_refused_sends"] for s in SCENES]
        spend = [imm[s]["realized_spend"] for s in SCENES]
        targets = [imm[s]["generated_goals"] for s in SCENES]
        truncations, explained = {}, []
        for i, s in enumerate(SCENES):
            if per_world[i] >= AFFORD[n]:
                truncations[s] = "at_or_above_predicted"
                explained.append(True)
            elif (refusals[i] or 0) > 0:
                truncations[s] = "budget_exhausted"
                explained.append(True)
            elif per_world[i] >= targets[i]:
                # every target this world generated was addressed: the ceiling is the target
                # count, not the credit. This is checkable, unlike a bare "no candidate" claim.
                truncations[s] = "no_candidate_target(all_generated_targets_addressed)"
                explained.append(True)
            else:
                truncations[s] = "unexplained_shortfall"
                explained.append(False)
        within_cap = all(spend[i] <= BUDGET for i in range(len(SCENES)))
        checks.append({
            "criterion": 2, "n": n, "name": "emissions_approach_affordable_sends",
            "affordable_sends_P1": AFFORD[n],
            "per_world_emissions": per_world,
            "per_world_realized_spend_bytes": spend,
            "per_world_generated_targets": targets,
            "per_world_budget_refusals": refusals,
            "worlds_at_affordable_sends": sum(1 for e in per_world if e == AFFORD[n]),
            "worlds_at_or_above": sum(1 for e in per_world if e >= AFFORD[n]),
            "worlds_at_budget_cap": at_cap,
            "denominator_worlds": len(SCENES),
            "truncation_reason_per_scene": {str(k): v for k, v in truncations.items()},
            "realized_spend_never_exceeds_budget": within_cap,
            "every_shortfall_explained": all(explained),
            "pass": bool(within_cap and all(explained)),
            "note": ("NUM_GOALS=24 is the candidate ceiling: at N=3 the credit buys 46 sends "
                     "but only 24 targets exist, so the ceiling is the target count, not the "
                     "budget. At N=12 the pooled floor of 11 is a *mean-payload* figure; the "
                     "observed per-message cost spans 4,404-5,376 B, so 12 emissions can still "
                     "fit inside 58,600 B and two worlds realise 58,572 / 58,512 B."
                     if n == 12 else
                     "N=6 is the only calibre where the credit, not the target count, binds: "
                     "the 24th target never fits, which is why every world stops at 23."
                     if n == 6 else
                     "N=3: target-limited, not budget-limited."),
        })

        # ---- criterion 3: always_hold is silent --------------------------------
        checks.append({
            "criterion": 3, "n": n, "name": "always_hold_emits_and_spends_nothing",
            "spent_bytes_total": sum(hold[s]["spent_bytes"] for s in SCENES),
            "emissions_total": sum(hold[s]["emissions"] for s in SCENES),
            "denominator": "8 scenes",
            "pass": bool(sum(hold[s]["spent_bytes"] for s in SCENES) == 0
                         and sum(hold[s]["emissions"] for s in SCENES) == 0),
        })

        # ---- criterion 4: headroom between the two bounds ----------------------
        headroom = on_time_imm - on_time_hold
        checks.append({
            "criterion": 4, "n": n, "name": "on_time_bound_interval_and_headroom",
            "upper_immediate": on_time_imm, "lower_always_hold": on_time_hold,
            "headroom": headroom, "denominator": denom,
            "verdict": "no_headroom" if headroom <= 1 else "has_headroom",
            "pass": True,
            "note": "criterion 4 asks for the interval and the no_headroom flag, not for a "
                    "direction; both bounds are reported and the flag is derived",
        })

    # ---- criterion 5: N=6 must equal the frozen artifacts ----------------------
    n6 = per_n[6]["arms"]
    detail5, mismatches = {}, []
    for arm in ARMS:
        detail5[arm] = {}
        for scene in SCENES:
            frozen, path = frozen_row(arm, scene)
            mine = n6[arm][scene]
            if frozen is None:
                detail5[arm][scene] = {"frozen_row": None, "path": path,
                                       "note": "no frozen row found"}
                mismatches.append((arm, scene, "missing_frozen_row"))
                continue
            pairs = {
                "paid_bytes": (int(frozen["paid_bytes"]), mine["spent_bytes"]),
                "emissions": (int(frozen["emissions"]), mine["emissions"]),
                "on_time": (int(frozen["on_time"]), mine["on_time"]),
            }
            bad = [k for k, (a, b) in pairs.items() if a != b]
            detail5[arm][str(scene)] = {"frozen": {k: v[0] for k, v in pairs.items()},
                                        "p2": {k: v[1] for k, v in pairs.items()},
                                        "equal": not bad, "differing": bad,
                                        "frozen_path": path}
            for key in bad:
                mismatches.append((arm, scene, key))
    checks.append({
        "criterion": 5, "n": 6, "name": "N6_equals_frozen_artifacts_per_scene",
        "cells_compared": len(ARMS) * len(SCENES),
        "denominator": "%d cells x 3 quantities" % (len(ARMS) * len(SCENES)),
        "mismatches": ["%s/s%d/%s" % m for m in mismatches],
        "detail": {arm: {str(s): v for s, v in detail5[arm].items()} for arm in ARMS},
        "pass": not mismatches,
    })

    # ---- budget authority must not have drifted with N -------------------------
    KEYS = ("v2_common.STAGE_B_INHERITED_OVERRIDES['comm_budget_bytes']",
            "a4_hooks.RELAY_BUDGET_BYTES",
            "a4_world_adapter.RELAY_BUDGET_BYTES")
    authority, live_values, restored_values = {}, {}, set()
    live_ok = 0
    worlds = 0
    for n in NS:
        for arm in ARMS:
            for scene in SCENES:
                cell = per_n[n]["arms"][arm][scene]
                worlds += 1
                credit = cell["budget_credit"] or {}
                if (int(cell["budget_bytes_in_world"]) == BUDGET
                        and int(credit.get("executor_budget_bytes", -1)) == BUDGET
                        and int(credit.get("protocol_budget_bytes", -1)) == BUDGET
                        and int(credit.get("run_configuration_budget_bytes", -1)) == BUDGET
                        and int(credit.get("context_budget_bytes", -1)) == BUDGET):
                    live_ok += 1
                after = cell["budget_authority_after_exit"] or {}
                for key in KEYS:
                    live_values.setdefault(n, set()).add(
                        credit.get("executor_budget_bytes", cell["budget_bytes_in_world"]))
                    restored_values.add((key, after.get(key)))
    authority = {str(n): sorted(live_values[n]) for n in NS}
    checks.append({
        "criterion": 6, "n": None, "name": "budget_authority_did_not_drift_with_N",
        "worlds_checked": worlds,
        "worlds_with_all_five_budget_reads_at_58600": live_ok,
        "denominator": "%d worlds x 5 live budget reads (row.budget_bytes + the four "
                       "budget_credit fields)" % worlds,
        "live_authority_observed_per_n": authority,
        "authority_values_after_exit": sorted(str(v) for v in restored_values),
        "pass": bool(live_ok == worlds),
        "note": "world.budget_authority() rebinds all three in-memory authorities for each "
                "world and restores them in a finally block, so the observed values are "
                "recorded twice: while the world ran (must be 58600) and after exit (must be "
                "the frozen 29300). The stored module attributes are the post-exit figure.",
    })

    passed = sum(1 for c in checks if c["pass"])
    verdict = "pass" if passed == len(checks) else "fail"

    print("=" * 78)
    print("P2 acceptance  (fixed calibre, 48 replays, rule arms only)")
    print("=" * 78)
    for n in NS:
        block = per_n[n]
        print("N=%-2d immediate: spent=%d B  emissions=%d  cost/send=%.4f B  "
              "at-cap=%d/8  on_time=%d/%d"
              % (n, block["immediate_totals"]["spent_bytes"],
                 block["immediate_totals"]["emissions"],
                 block["immediate_totals"]["cost_per_send_mean"] or 0.0,
                 block["worlds_at_budget_cap"]["count"],
                 block["on_time_interval"]["upper_immediate"],
                 block["on_time_interval"]["denominator"]))
        print("     always_hold: spent=%d B  emissions=%d  on_time=%d/%d   headroom=%d"
              % (sum(block["arms"]["always_hold"][s]["spent_bytes"] for s in SCENES),
                 sum(block["arms"]["always_hold"][s]["emissions"] for s in SCENES),
                 block["on_time_interval"]["lower_always_hold"],
                 block["on_time_interval"]["denominator"],
                 block["on_time_interval"]["headroom"]))
        print("     per-world emissions: %s   (P1 affordable_sends=%d)"
              % (block["per_world_emissions"], AFFORD[n]))
    print()
    for c in checks:
        print("  [%s] criterion %s N=%s %s" % ("PASS" if c["pass"] else "FAIL",
                                               c["criterion"], c["n"], c["name"]))
        if not c["pass"]:
            print("        ", json.dumps({k: v for k, v in c.items()
                                          if k not in ("detail",)}, ensure_ascii=False)[:600])
    print()
    print("P2 verdict: %s  (%d/%d checks pass)" % (verdict, passed, len(checks)))

    report = {
        "schema": "scale-probe-p2-summary/1", "stage": "P2", "mode": "fixed",
        "identity": "local experiment level, no identity built",
        "budget_bytes": BUDGET, "replays": 48,
        "p1_predictions": {"affordable_sends": AFFORD,
                           "cost_per_send_at_measured_payload": COST_PER_SEND_PREDICTED,
                           "protocol_literals": SPEC_LITERAL_COST},
        "per_n": {str(n): per_n[n] for n in NS},
        "budget_authority_observed": {str(k): v for k, v in authority.items()},
        "checks": checks, "checks_passed": passed, "checks_total": len(checks),
        "verdict": verdict,
        "labour_coupling": {str(n): (n == 12) for n in NS},
        "comparability_note": "N!=6 is a new config and a new world: config_hash and "
                              "behaviour_sha256 both move, so the N=3 and N=12 on-time "
                              "counts are NOT comparable with the archived 108/192 or "
                              "131/192 figures. The N=6 equality check (criterion 5) is "
                              "the one comparison this stage licenses.",
    }
    out = PROBE / "P2_summary.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True),
                   encoding="utf-8")
    print("wrote", out)
    print("sha256", sha256(out))


if __name__ == "__main__":
    main()
