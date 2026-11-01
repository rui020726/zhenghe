# -*- coding: utf-8 -*-
"""ROUND72 final-acceptance errata (zero environment, read-only).

Emits ROUND72_ERRATA.json (concise machine acceptance package) and
ROUND72_ERRATA.md from persisted artifacts. No env/model/forward/RNG/optimizer.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r72_paths import REPO, RUNS, R68, R71, R66, DELIV

FROZEN_SRC = RUNS / "round51_frozen_source_gradient_close" / "frozen_sources"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


boundary = load(DELIV / "ROUND72_BOUNDARY_AUDIT.json")
r71_stats = load(R71 / "raw" / "ROUND71_STATISTICS.json")
r68_rule = load(R68 / "raw" / "ROUND68_RULE_CONTEXT.json")
r71_ck = load(R71 / "ROUND71_CHECKPOINT_INDEX.json")
r71_led = load(R71 / "ROUND71_RESOURCE_LEDGER.json")
psm = boundary["comparisons"]["PRIMARY_R68_TRAINED_VS_R71_NEW_EP128"]["per_seed_mode"]


def mode_reduce(mode):
    cells = steps = feats = acts = nf = early = late = aflip = argf = lchg = 0
    emax = lmax = enc = 0.0
    for k, v in psm.items():
        if not k.endswith(mode):
            continue
        idn, s = v["identity"], v["summary"]
        m = s["margin"]
        cells += v["cells"]; steps += idn["steps_equal"]; feats += idn["features_equal"]; acts += idn["actions_equal"]
        nf += s["n_free"]; early += s["early"]["n"]; late += s["late"]["n"]
        aflip += s["action_flips"]; argf += m["argmax_flips"]; lchg += m["late_n_dp_nonzero"]
        emax = max(emax, m["early_max_abs_dp"]); lmax = max(lmax, m["late_max_abs_dp"])
        enc = max(enc, m["late_max_encroachment_ratio"] or 0.0)
    return {"cells": cells, "steps_equal": steps, "features_equal": feats, "actions_equal": acts,
            "n_free": nf, "early": early, "late": late, "action_flips": aflip, "argmax_flips": argf,
            "late_changed": lchg, "early_max_abs_dp": emax, "late_max_abs_dp": round(lmax, 4),
            "late_max_encroachment_ratio": round(enc, 4)}


greedy = mode_reduce("greedy")
sampled = mode_reduce("sampled")

# ---- budget aggregation from real episode rows (measured, not inferred) ----
R71R = str(R71 / "raw" / "rows")
R66R = str(R66 / "raw" / "rows")


def budget_agg(name, paths):
    n = 0; spent = []; bset = set(); uf = []
    bref = cref = refseq = mbc = mb = mc = 0; miss = set()
    for p in paths:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        n += 1
        if "spent_bytes" in d:
            spent.append(d["spent_bytes"])
        else:
            miss.add("spent_bytes")
        if "budget_bytes" in d:
            bset.add(d["budget_bytes"])
        else:
            miss.add("budget_bytes")
        if "budget_used_fraction" in d:
            uf.append(d["budget_used_fraction"])
        else:
            miss.add("budget_used_fraction")
        if d.get("refusal_sequence"):
            refseq += 1
        for _, v in (d.get("cost_stream") or {}).items():
            if isinstance(v, dict):
                bref += int(v.get("budget_refused_sends", 0) or 0)
                cref += int(v.get("capacity_refused_sends", 0) or 0)
        oc = (d.get("collector_summary") or {}).get("one_shot_counts") or {}
        ec = (d.get("collector_summary") or {}).get("eligibility_counts") or {}
        mbc += int(oc.get("masked_budget_or_capacity", 0) or 0)
        mb += int(ec.get("budget_masked_first_reports", 0) or 0)
        mc += int(ec.get("capacity_masked_first_reports", 0) or 0)
    return {"rows": n, "budget_bytes": sorted(bset),
            "spent_min": min(spent) if spent else None, "spent_max": max(spent) if spent else None,
            "spent_mean": round(sum(spent) / len(spent), 1) if spent else None,
            "budget_used_fraction_max": max(uf) if uf else None,
            "budget_refused_sends": bref, "capacity_refused_sends": cref,
            "rows_with_refusal_sequence": refseq,
            "collector_masked_budget_or_capacity": mbc,
            "budget_masked_first_reports": mb, "capacity_masked_first_reports": mc,
            "missing_fields": sorted(miss)}


budget = {
    "dev_eval_learned": budget_agg("dev_eval_learned", glob.glob(R71R + "/v2_r71eval_model_new_ep128_*.json")),
    "dev_eval_immediate": budget_agg("dev_eval_immediate", glob.glob(R66R + "/v2_r66eval2_rule_immediate_*.json")),
    "training": budget_agg("training", glob.glob(R71R + "/v2_r71_train_*.json")),
}

# ---- strict per-row coverage receipt (grounds the measured-zero claim) ----
cov = load(DELIV / "ROUND72_COVERAGE_RECEIPT.json")

# ---- per-seed primary task/cost ----
def prim(cell):
    p = r71_stats["per_seed_mode"][cell]
    q = p["primary"]
    return {"on_time": p["NEW_EP128_on_time_total"], "bytes_per_scene": p["NEW_EP128_bytes_per_scene"],
            "task_delta": q["task_delta_NEW_EP128_minus_R68_TRAINED_ALPHA_mean"],
            "cost_delta_B_per_scene": q["cost_delta_NEW_EP128_minus_R68_TRAINED_ALPHA_B_per_scene"],
            "cost_ci95": q["cost_ci95_NEW_EP128_minus_R68_TRAINED_ALPHA_B_per_scene"],
            "targets_lost": q["targets_lost_NEW_EP128_vs_R68_TRAINED_ALPHA_count"]}


per_seed = {"greedy": {c: prim(c) for c in ("11/greedy", "22/greedy", "33/greedy")},
            "sampled": {c: prim(c) for c in ("11/sampled", "22/sampled", "33/sampled")}}

ra = r68_rule["rule_absolute"]
ts11 = r11 = r71_ck["seeds"]["s11"]["trained_ep128"]["training_state"]

# ---- locked assertions (pure) ----
assert greedy["n_free"] == 549 and greedy["early"] == 402 and greedy["late"] == 147
assert sampled["n_free"] == 549 and sampled["early"] == 402 and sampled["late"] == 147
assert greedy["cells"] == 96 and greedy["features_equal"] == 96 and greedy["steps_equal"] == 96
assert greedy["action_flips"] == 0 and greedy["argmax_flips"] == 0
assert sampled["cells"] == 96 and sampled["features_equal"] < 96 and sampled["action_flips"] == 14
assert sampled["argmax_flips"] == 0
for sp in budget.values():
    assert sp["budget_bytes"] == [29300]
    assert sp["budget_refused_sends"] == 0 and sp["capacity_refused_sends"] == 0
    assert sp["rows_with_refusal_sequence"] == 0
    assert sp["collector_masked_budget_or_capacity"] == 0
    assert sp["missing_fields"] == []
# coverage-receipt grounded assertions (strict per-row existence+type)
assert cov["channel_verdicts"]["executor_refusal_channel"]["covered_all_splits"] is True
assert cov["channel_verdicts"]["executor_refusal_channel"]["measured_zero_all_splits"] is True
assert cov["channel_verdicts"]["collector_premask_channel"]["measured_zero_where_covered"] is True
assert cov["channel_verdicts"]["collector_premask_channel"]["na_splits"] == ["dev_eval_immediate"]
assert cov["composition_notes"]["dev_eval_immediate"]["focal_scene_1700027_rows"] == 0
assert ts11["gamma"] == 1.0 and ts11["gae_lambda"] == 1.0
assert r71_led["offline_forward_ledger"]["rows"] == 70693

sources = {
    "ROUND72_BOUNDARY_AUDIT.json": str(DELIV / "ROUND72_BOUNDARY_AUDIT.json"),
    "ROUND72_VALIDATION.json": str(DELIV / "ROUND72_VALIDATION.json"),
    "ROUND72_TRAINING_SIGNAL_AUDIT.json": str(DELIV / "ROUND72_TRAINING_SIGNAL_AUDIT.json"),
    "ROUND71_STATISTICS.json": str(R71 / "raw" / "ROUND71_STATISTICS.json"),
    "ROUND68_RULE_CONTEXT.json": str(R68 / "raw" / "ROUND68_RULE_CONTEXT.json"),
    "ROUND71_CHECKPOINT_INDEX.json": str(R71 / "ROUND71_CHECKPOINT_INDEX.json"),
    "ROUND71_RESOURCE_LEDGER.json": str(R71 / "ROUND71_RESOURCE_LEDGER.json"),
    "r42_one_shot.py": str(R71 / "raw" / "r42_one_shot.py"),
    "r40_train_resume_adapter.py": str(R71 / "raw" / "r40_train_resume_adapter.py"),
    "r66_frozen.py": str(R66 / "r66_frozen.py"),
    "canonical_v2_trainer.py": str(FROZEN_SRC / "canonical_v2_trainer.py"),
    "comm_budget.py": str(REPO / "integrated_sar" / "comm_budget.py"),
    "comm_learning.py": str(REPO / "integrated_sar" / "comm_learning.py"),
    "ROUND72_COVERAGE_RECEIPT.json": str(DELIV / "ROUND72_COVERAGE_RECEIPT.json"),
}
source_hashes = {name: {"path": path, "sha256": sha256(path)} for name, path in sources.items()}

errata = {
    "kind": "round72-errata/1",
    "supersedes": "specific claims in ROUND72_REPORT.md and ROUND72_MACHINE_VERDICT.json at commit 6d28f40; the original delivery is preserved unchanged",
    "evidence_head": "fe53d476efb56baaae194d68d7f749a546b14389",
    "correction_1_mode_split": {
        "note": "PRIMARY 1098 free decisions = greedy(549) + sampled(549); it is NOT 'greedy 1098'. 192 is the total eval matrix (96 greedy + 96 sampled); sampled is genuinely different.",
        "greedy": greedy, "sampled": sampled,
        "corrected": [
            "greedy alone has 549 free decisions (early 402, late 147) over 96 cells that are fully identical (steps/features/actions 96/96)",
            "greedy task/cost is identical to R68 on its 96 greedy cells",
            "sampled differs: 83/96 features_equal, 90/96 steps_equal, 14 action flips",
            "sampled features are not all equal, so sampled dp is not purely same-state parameter change and sampling variance is not quantitatively excluded",
        ],
        "preserved": [
            "probabilities did change on 147/147 visited late states in each mode",
            "sampled behaviour did differ: 14 late action flips",
        ],
    },
    "correction_2_budget": {
        "budget_bytes": 29300,
        "evidence_basis": "measured from real episode rows: spent_bytes, budget_bytes, budget_used_fraction, refusal_sequence, cost_stream.budget_refused_sends/capacity_refused_sends, collector_summary mask counts; both the executor-refusal channel and the collector pre-mask channel were checked. Strict per-row existence+type is proven in ROUND72_COVERAGE_RECEIPT.json.",
        "splits": budget,
        "downgrade": "the previous 'NOT binding' inferred from spent<29300 is replaced by MEASURED zero refusals with the fields PRESENT on every row (executor channel, all 609 rows across the three splits), and MEASURED zero collector pre-masks where the learned collector runs (learned 192 + training 384); for the immediate rule arm the collector pre-mask is N/A (collector_summary is None), NOT 'measured 0'.",
        "measured_zero_not_missing": {
            "executor_refusal_channel": "measured_zero_and_field_present_on_every_row",
            "collector_premask_channel_learned": "measured_zero_and_field_present",
            "collector_premask_channel_training": "measured_zero_and_field_present",
            "collector_premask_channel_immediate": "not_applicable_rule_path_collector_summary_is_None"
        },
        "immediate_collector_note": "In the budget_agg 'splits' the immediate collector mask value 0 is a .get-default because collector_summary is None; the coverage receipt reclassifies it as N/A (rule path), NOT a measured 0. The immediate executor-refusal zeros ARE measured (refusal_sequence and cost_stream refusal fields are present on all 33 rows).",
        "immediate_row_composition": cov["composition_notes"]["dev_eval_immediate"],
        "coverage_receipt": "ROUND72_COVERAGE_RECEIPT.json",
        "training_block_note": "training rows (block 1300000..1300047) were located via r72_paths and the frozen jobs index; they DO carry the budget/refusal fields, so the training block is now evidenced, not asserted",
    },
    "correction_3_teacher_billing": {
        "offline_billing_rows": {"value": r71_led["offline_forward_ledger"]["rows"],
                                 "meaning": "R71 offline-fit forward billing total (includes gate/diagnostic forwards), cap 72000; NOT the teacher/fit dataset"},
        "teacher_original_scale": {"joint": 32046, "target": 58928,
                                   "source": "delegation-provided; not independently re-derived from a located numeric field"},
        "attribution_gap": {"value": 11765, "meaning": "historical per-call attribution gap preserved as-is; not fabricated"},
        "r70_fit_source": "1300000..1300047",
    },
    "correction_4_margin_framing": {
        "corrected": "'the mechanism is margin, not saturation' is too strong",
        "restated": "this actual update did not cross p=0.5 (a boundary-margin description); by itself it does not rule out saturation, gradient noise, or observation limits; late probabilities reach as high as ~0.99",
        "prior": "beta=1.00 prior is confirmed only in the training loss; inference has no additional prior logit (real path)",
    },
    "r71_per_seed_primary_task_cost": per_seed,
    "rule_arms": {a: {"on_time": ra[a]["on_time_total"], "bytes_per_scene_mean": ra[a]["spent_bytes_per_scene_mean"]}
                  for a in ("IMMEDIATE", "LATE_DISCOVERY_GATE", "ALWAYS_HOLD")},
    "consumption_this_session": {"environment": 0, "model_loads": 0, "forward": 0, "rng": 0,
                                 "optimizer_or_ppo": 0, "checkpoint_writes": 0, "new_claims": 0},
    "source_hashes": source_hashes,
}

(DELIV / "ROUND72_ERRATA.json").write_text(json.dumps(errata, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote ROUND72_ERRATA.json size", (DELIV / "ROUND72_ERRATA.json").stat().st_size)
print("greedy", greedy["n_free"], "feat_eq", greedy["features_equal"], "flips", greedy["action_flips"])
print("sampled", sampled["n_free"], "feat_eq", sampled["features_equal"], "flips", sampled["action_flips"])
print("budget splits rows", {k: v["rows"] for k, v in budget.items()})