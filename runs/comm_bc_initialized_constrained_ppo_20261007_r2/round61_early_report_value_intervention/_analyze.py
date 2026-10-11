"""ROUND61 paired analysis: for each frozen case, FORCE_SEND vs FORCE_SKIP target and whole-system
outcomes, cost, emission/assignment chain, plus the legal-input aliasing observation. Pure
computation over the recorded cell results."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
freeze = json.loads((HERE / "ROUND61_CASE_FREEZE.json").read_text(encoding="utf-8"))
cases = freeze["cases"]
res = json.loads((RAW / "ROUND61_CELL_RESULTS.json").read_text(encoding="utf-8"))

def tshort(t):
    return t.replace("piloc-e1-", "")

def cell(arm, case):
    return res[f"{arm}:{case['scene']}:{case['target']}"]

paired = []
for case in cases:
    s = cell("FORCE_SEND", case)
    k = cell("FORCE_SKIP", case)
    s_ids, k_ids = set(s["on_time_ids"]), set(k["on_time_ids"])
    tgt = case["target"]
    y_send = 1 if tgt in s_ids else 0
    y_skip = 1 if tgt in k_ids else 0
    new_ids = sorted(s_ids - k_ids)
    lost_ids = sorted(k_ids - s_ids)
    rec = {
        "role": case["role"], "scene": case["scene"], "target": tgt, "ready_step": case["ready_step"],
        "baseline_action": case["baseline_action"], "distance_to_focus": case["distance_to_focus"],
        "pre_action_features_19d": case["features_19d"],
        "cols_13_14_15_18": case["cols_13_14_15_18"],
        "target_on_time": {"Y_SEND": y_send, "Y_SKIP": y_skip, "delta": y_send - y_skip},
        "whole_system_on_time": {"SEND": len(s_ids), "SKIP": len(k_ids), "delta": len(s_ids) - len(k_ids)},
        "new_targets_under_SEND": new_ids, "lost_targets_under_SEND": lost_ids,
        "cost": {"SEND_bytes": s["spent_bytes"], "SKIP_bytes": k["spent_bytes"],
                 "delta_bytes": s["spent_bytes"] - k["spent_bytes"]},
        "emissions": {"SEND": s["emission_sequence"], "SKIP": k["emission_sequence"]},
        "intervene_record": {"SEND": s["iv_record"], "SKIP": k["iv_record"]},
        "on_time_ids": {"SEND": s["on_time_ids"], "SKIP": k["on_time_ids"]},
    }
    paired.append(rec)

# aliasing observation: identical float32 vectors with different intervention benefit
by_vec = {}
for case in cases:
    key = tuple(case["features_19d"])
    by_vec.setdefault(key, []).append(case)
aliasing = []
for key, group in by_vec.items():
    if len(group) > 1:
        deltas = []
        for c in group:
            s = cell("FORCE_SEND", c); k = cell("FORCE_SKIP", c)
            deltas.append((len(set(s["on_time_ids"])) - len(set(k["on_time_ids"])),
                           s["spent_bytes"] - k["spent_bytes"]))
        if len(set(deltas)) > 1:
            aliasing.append({"vector": list(key), "cases": [(c["scene"], c["target"]) for c in group],
                             "send_minus_skip": deltas})

summary = {
    "focus": paired[0],
    "controls": paired[1:],
    "focus_target_outcome_changed": paired[0]["target_on_time"]["delta"] != 0 or bool(paired[0]["new_targets_under_SEND"] or paired[0]["lost_targets_under_SEND"]),
    "controls_with_whole_system_change": [p["scene"] for p in paired[1:] if p["whole_system_on_time"]["delta"] != 0 or p["new_targets_under_SEND"] or p["lost_targets_under_SEND"]],
    "identical_vector_conflicts": aliasing,
    "guard_snapshot": json.loads((RAW / "ROUND61_GUARD_SNAPSHOT.json").read_text(encoding="utf-8")),
}
(RAW / "ROUND61_PAIRED_ANALYSIS.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("FOCUS", paired[0]["scene"], paired[0]["target"],
      "Y_SEND", paired[0]["target_on_time"]["Y_SEND"], "Y_SKIP", paired[0]["target_on_time"]["Y_SKIP"],
      "whole", paired[0]["whole_system_on_time"], "cost_delta", paired[0]["cost"]["delta_bytes"],
      "new", paired[0]["new_targets_under_SEND"], "lost", paired[0]["lost_targets_under_SEND"])
for p in paired[1:]:
    print(f"CTRL {p['scene']} {p['target']} Yd{p['target_on_time']['delta']} whole_d{p['whole_system_on_time']['delta']} cost_d{p['cost']['delta_bytes']} new{p['new_targets_under_SEND']} lost{p['lost_targets_under_SEND']}")
print("identical_vector_conflicts:", len(aliasing))
print("guard:", summary["guard_snapshot"])
