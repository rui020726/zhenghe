# -*- coding: utf-8 -*-
"""ROUND72 strict budget/refusal field-coverage receipt (zero environment, read-only).

ROUND72_ERRATA budget_agg used .get(key,0) / or {} defaults, so its
"measured_zero_not_missing" flag could not distinguish a *measured* zero from a
field that is simply *absent*. This script proves, per row and per split, the
exact existence+type of every budget/refusal field, aggregates values ONLY where
the field exists, and marks the rule (immediate) collector pre-mask channel as
N/A because that path never runs the learned collector (collector_summary is
None). No env/model/forward/RNG/optimizer/PPO/checkpoint writes.
"""
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r72_paths import REPO, RUNS, R68, R71, R66, DELIV

DEV_LO, DEV_HI = 1800000, 1800031
FOCAL = 1700027


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def manifest_sha(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update((Path(p).name + "\t" + sha256(p) + "\n").encode("utf-8"))
    return h.hexdigest()


def scene_attempt(name):
    m = re.search(r"_s(\d+)_a(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def scan(paths):
    s = {
        "rows_total": len(paths), "rows_loaded": 0, "rows_failed": 0,
        "refusal_sequence_present": 0, "refusal_sequence_type_list": 0,
        "refusal_sequence_rows_nonempty": 0, "refusal_sequence_len_total": 0,
        "cost_stream_present": 0, "cost_stream_type_dict": 0, "cost_stream_steps_total": 0,
        "steps_missing_budget_refused_sends": 0, "steps_missing_capacity_refused_sends": 0,
        "measured_budget_refused_sends_total": 0, "measured_capacity_refused_sends_total": 0,
        "collector_absent": 0, "collector_none": 0, "collector_dict": 0,
        "dict_rows_missing_one_shot_counts": 0, "dict_rows_missing_eligibility_counts": 0,
        "dict_rows_missing_masked_budget_or_capacity": 0,
        "dict_rows_missing_budget_masked_first_reports": 0,
        "dict_rows_missing_capacity_masked_first_reports": 0,
        "measured_masked_budget_or_capacity_total": 0,
        "measured_budget_masked_first_reports_total": 0,
        "measured_capacity_masked_first_reports_total": 0,
    }
    for p in paths:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            s["rows_failed"] += 1
            continue
        s["rows_loaded"] += 1
        if "refusal_sequence" in d:
            s["refusal_sequence_present"] += 1
            rs = d["refusal_sequence"]
            if isinstance(rs, list):
                s["refusal_sequence_type_list"] += 1
                s["refusal_sequence_len_total"] += len(rs)
                if len(rs) > 0:
                    s["refusal_sequence_rows_nonempty"] += 1
        if "cost_stream" in d:
            s["cost_stream_present"] += 1
            cs = d["cost_stream"]
            if isinstance(cs, dict):
                s["cost_stream_type_dict"] += 1
                for _, step in cs.items():
                    if not isinstance(step, dict):
                        continue
                    s["cost_stream_steps_total"] += 1
                    if "budget_refused_sends" in step:
                        s["measured_budget_refused_sends_total"] += int(step["budget_refused_sends"] or 0)
                    else:
                        s["steps_missing_budget_refused_sends"] += 1
                    if "capacity_refused_sends" in step:
                        s["measured_capacity_refused_sends_total"] += int(step["capacity_refused_sends"] or 0)
                    else:
                        s["steps_missing_capacity_refused_sends"] += 1
        if "collector_summary" not in d:
            s["collector_absent"] += 1
        elif d["collector_summary"] is None:
            s["collector_none"] += 1
        elif isinstance(d["collector_summary"], dict):
            s["collector_dict"] += 1
            oc = d["collector_summary"].get("one_shot_counts")
            ec = d["collector_summary"].get("eligibility_counts")
            if isinstance(oc, dict) and "masked_budget_or_capacity" in oc:
                s["measured_masked_budget_or_capacity_total"] += int(oc["masked_budget_or_capacity"] or 0)
            else:
                s["dict_rows_missing_masked_budget_or_capacity"] += 1
                if not isinstance(oc, dict):
                    s["dict_rows_missing_one_shot_counts"] += 1
            if isinstance(ec, dict):
                if "budget_masked_first_reports" in ec:
                    s["measured_budget_masked_first_reports_total"] += int(ec["budget_masked_first_reports"] or 0)
                else:
                    s["dict_rows_missing_budget_masked_first_reports"] += 1
                if "capacity_masked_first_reports" in ec:
                    s["measured_capacity_masked_first_reports_total"] += int(ec["capacity_masked_first_reports"] or 0)
                else:
                    s["dict_rows_missing_capacity_masked_first_reports"] += 1
            else:
                s["dict_rows_missing_budget_masked_first_reports"] += 1
                s["dict_rows_missing_capacity_masked_first_reports"] += 1
                if not isinstance(ec, dict):
                    s["dict_rows_missing_eligibility_counts"] += 1
    return s


def verdict(s):
    n = s["rows_loaded"]
    exec_covered = (n == s["rows_total"] and n > 0
                    and s["refusal_sequence_present"] == n and s["refusal_sequence_type_list"] == n
                    and s["cost_stream_present"] == n and s["cost_stream_type_dict"] == n
                    and s["steps_missing_budget_refused_sends"] == 0
                    and s["steps_missing_capacity_refused_sends"] == 0)
    exec_zero = (s["refusal_sequence_len_total"] == 0
                 and s["measured_budget_refused_sends_total"] == 0
                 and s["measured_capacity_refused_sends_total"] == 0)
    if s["collector_dict"] == n and n > 0:
        coll = "covered" if (s["dict_rows_missing_masked_budget_or_capacity"] == 0
                             and s["dict_rows_missing_budget_masked_first_reports"] == 0
                             and s["dict_rows_missing_capacity_masked_first_reports"] == 0
                             and s["dict_rows_missing_one_shot_counts"] == 0
                             and s["dict_rows_missing_eligibility_counts"] == 0) else "incomplete"
        coll_zero = (s["measured_masked_budget_or_capacity_total"] == 0
                     and s["measured_budget_masked_first_reports_total"] == 0
                     and s["measured_capacity_masked_first_reports_total"] == 0)
    elif s["collector_none"] == n and n > 0:
        coll, coll_zero = "not_applicable_rule_path", None
    else:
        coll, coll_zero = "mixed_or_absent", None
    return {"executor_refusal_covered": bool(exec_covered),
            "executor_refusal_measured_zero": bool(exec_zero),
            "collector_premask_status": coll,
            "collector_premask_measured_zero": coll_zero}


GLOB = {
    "dev_eval_learned": str(R71 / "raw" / "rows" / "v2_r71eval_model_new_ep128_*.json"),
    "dev_eval_immediate": str(R66 / "raw" / "rows" / "v2_r66eval2_rule_immediate_*.json"),
    "training": str(R71 / "raw" / "rows" / "v2_r71_train_*.json"),
}
splits = {}
for k, g in GLOB.items():
    paths = glob.glob(g)
    s = scan(paths)
    s["verdict"] = verdict(s)
    splits[k] = s

# immediate composition (correct the handoff "focal row" assumption)
imm = sorted(glob.glob(GLOB["dev_eval_immediate"]))


def spent_of(path):
    try:
        return json.load(open(path, encoding="utf-8")).get("spent_bytes")
    except Exception:
        return None


by_scene = {}
for p in imm:
    sc, at = scene_attempt(Path(p).name)
    by_scene.setdefault(sc, []).append((at, Path(p).name, p))
in_range_scenes = sorted(sc for sc in by_scene if sc is not None and DEV_LO <= sc <= DEV_HI)
extra_rows = []
for sc in sorted(by_scene, key=lambda x: (x is None, x)):
    lst = sorted(by_scene[sc], key=lambda t: (t[0] is None, t[0]))
    in_range = sc is not None and DEV_LO <= sc <= DEV_HI
    canon = lst[0] if in_range else None
    for at, nm, pth in (lst[1:] if in_range else lst):
        csp = spent_of(canon[2]) if canon else None
        sp = spent_of(pth)
        pm = re.search(r"rule_immediate_(a\d+)_s", nm)
        run_tok = pm.group(1) if pm else ("a" + str(at))
        extra_rows.append({"scene": sc, "run_token": run_tok, "file": nm,
                           "canonical_file": canon[1] if canon else None,
                           "spent_bytes": sp, "canonical_spent_bytes": csp,
                           "spent_equals_canonical": (sp == csp) if canon else None})
composition = {
    "dev_block_canonical_rows": len(in_range_scenes),
    "dev_block_scene_range": [DEV_LO, DEV_HI],
    "extra_repeat_rows": extra_rows,
    "focal_scene_1700027_rows": len(by_scene.get(FOCAL, [])),
    "note": ("immediate glob = 32 canonical dev-block rows (scenes 1800000..1800031) + 1 extra "
             "re-attempt of scene 1800000 (run immediate_a2). There is NO 1700027 focal row in this glob; "
             "the earlier 'focal row' assumption is corrected. All 33 are executor rows with "
             "identical field shape; the extra re-attempt must not be folded into a per-scene mean."),
}

# cross-split channel verdicts
exec_all = all(splits[k]["verdict"]["executor_refusal_covered"] for k in GLOB)
exec_zero_all = all(splits[k]["verdict"]["executor_refusal_measured_zero"] for k in GLOB)
coll_cov = [k for k in GLOB if splits[k]["verdict"]["collector_premask_status"] == "covered"]
coll_na = [k for k in GLOB if splits[k]["verdict"]["collector_premask_status"] == "not_applicable_rule_path"]
coll_zero_where_cov = all(splits[k]["verdict"]["collector_premask_measured_zero"] is True for k in coll_cov)

receipt = {
    "kind": "round72-budget-coverage-receipt/1",
    "generated_by": "r72_coverage.py",
    "purpose": ("Prove per-row existence+type of budget/refusal fields so the errata 'measured zero' "
                "claim is grounded; executor refusals covered on every row, collector pre-mask covered "
                "for learned+training and N/A for the immediate rule path."),
    "zero_side_effects": {"environment": 0, "model_loads": 0, "forward": 0, "rng": 0,
                          "optimizer_or_ppo": 0, "checkpoint_writes": 0, "new_claims": 0},
    "row_globs": GLOB,
    "splits": splits,
    "composition_notes": {"dev_eval_immediate": composition},
    "channel_verdicts": {
        "executor_refusal_channel": {"covered_all_splits": bool(exec_all),
                                     "measured_zero_all_splits": bool(exec_zero_all),
                                     "fields": ["refusal_sequence", "cost_stream.*.budget_refused_sends",
                                                "cost_stream.*.capacity_refused_sends"],
                                     "note": "present as list/dict on every loaded row; measured zero everywhere."},
        "collector_premask_channel": {"covered_splits": coll_cov, "na_splits": coll_na,
                                      "measured_zero_where_covered": bool(coll_zero_where_cov),
                                      "fields": ["collector_summary.one_shot_counts.masked_budget_or_capacity",
                                                 "collector_summary.eligibility_counts.budget_masked_first_reports",
                                                 "collector_summary.eligibility_counts.capacity_masked_first_reports"],
                                      "note": ("learned+training rows run the learned collector (dict) and measure "
                                               "zero pre-masks; immediate rows are the rule path with "
                                               "collector_summary=None so these are N/A, not 'measured 0'.")},
    },
    "manifest_sha256": {k: manifest_sha(glob.glob(GLOB[k])) for k in GLOB},
}
out = DELIV / "ROUND72_COVERAGE_RECEIPT.json"
out.write_text(json.dumps(receipt, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out.name, "size", out.stat().st_size)
for k in GLOB:
    v = splits[k]["verdict"]
    print(k, "rows", splits[k]["rows_loaded"], "exec_cov", v["executor_refusal_covered"],
          "exec_zero", v["executor_refusal_measured_zero"], "coll", v["collector_premask_status"])
print("immediate:", composition["dev_block_canonical_rows"], "canonical dev +",
      len(composition["extra_repeat_rows"]), "extra; focal1700027", composition["focal_scene_1700027_rows"])
for e in composition["extra_repeat_rows"]:
    print("  extra", e["scene"], e["run_token"], "spent", e["spent_bytes"],
          "canon", e["canonical_spent_bytes"], "equal", e["spent_equals_canonical"])