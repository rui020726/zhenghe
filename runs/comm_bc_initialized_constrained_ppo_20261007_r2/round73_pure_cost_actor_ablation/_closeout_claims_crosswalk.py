# ROUND73 read-only close-out: claim/slot/permit/row crosswalk (stdlib only, no writes except OUT).
import json, os, sys, collections, hashlib, time

ROOT = sys.argv[1]
OUT = sys.argv[2]
RAW = os.path.join(ROOT, "raw")
SLOTS = os.path.join(RAW, "launch_slots")
CLAIMS = os.path.join(RAW, "launch_claims")
ROWS = os.path.join(RAW, "rows")
LOGS = os.path.join(RAW, "logs")

def jload(p):
    with open(p, "rb") as f:
        return json.loads(f.read().decode("utf-8"))

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

slots = []
for cat in sorted(os.listdir(SLOTS)):
    d = os.path.join(SLOTS, cat)
    if not os.path.isdir(d):
        continue
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            slots.append(jload(os.path.join(d, name)))

claims = []
for job in sorted(os.listdir(CLAIMS)):
    d = os.path.join(CLAIMS, job)
    if not os.path.isdir(d):
        continue
    for name in sorted(os.listdir(d)):
        if name.startswith("attempt_") and name.endswith(".json"):
            r = jload(os.path.join(d, name))
            r["_dir"] = d
            claims.append(r)

permits = []
for c in claims:
    pd = os.path.join(c["_dir"], "permits")
    if not os.path.isdir(pd):
        continue
    for name in sorted(os.listdir(pd)):
        if name.endswith(".json"):
            r = jload(os.path.join(pd, name))
            permits.append({"job": c["logical_job"], "file": name, "rec": r})

rows = []
for name in sorted(os.listdir(ROWS)):
    if name.endswith(".json"):
        rows.append((name, jload(os.path.join(ROWS, name))))

def scope_of(job):
    if job.startswith("r73_new_training_"):
        return "train"
    if job.startswith("r73eval_model_new_ep128_"):
        return "model_eval"
    return "other"

def row_label_of_claim(job):
    if job.startswith("r73_new_training_"):
        return job.replace("r73_new_training_", "r73_train_", 1)
    if job.startswith("r73eval_model_new_ep128_"):
        parts = job.split("_scene")
        return parts[0]
    return job

rep = {}
rep["slots"] = {
    "total": len(slots),
    "categories": dict(collections.Counter(s.get("category") for s in slots)),
    "slot_numbers_min": min(s.get("slot") for s in slots),
    "slot_numbers_max": max(s.get("slot") for s in slots),
    "contiguous_1_to_max": sorted(s.get("slot") for s in slots) == list(range(1, max(s.get("slot") for s in slots) + 1)),
    "claimed_at_unix_min": min(s.get("claimed_at_unix") for s in slots),
    "claimed_at_unix_max": max(s.get("claimed_at_unix") for s in slots),
}
rep["claims"] = {
    "total": len(claims),
    "distinct_logical_jobs": len(set(c["logical_job"] for c in claims)),
    "attempts": dict(collections.Counter(c.get("attempt") for c in claims)),
    "statuses": dict(collections.Counter(c.get("status") for c in claims)),
    "scopes": dict(collections.Counter(scope_of(c["logical_job"]) for c in claims)),
    "scope_status": dict(collections.Counter("%s|%s" % (scope_of(c["logical_job"]), c.get("status")) for c in claims)),
    "claimed_at_unix_min": min(c.get("claimed_at_unix") for c in claims),
    "claimed_at_unix_max": max(c.get("claimed_at_unix") for c in claims),
}
for f in ["environment_constructor_calls", "joint_run_constructed_calls", "run_invocation_calls",
          "joint_run_method_calls", "run_return_calls", "run_failure_calls",
          "joint_run_build_failures", "completed_steps"]:
    vals = [c.get(f) for c in claims]
    rep["claims"]["field_" + f] = {
        "present": sum(1 for v in vals if v is not None),
        "missing": sum(1 for v in vals if v is None),
        "types": dict(collections.Counter(type(v).__name__ for v in vals)),
        "distinct_values": dict(collections.Counter(str(v) for v in vals)),
        "sum": sum(v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)),
    }
rep["permits"] = {
    "total_files": len(permits),
    "kinds": dict(collections.Counter(p["rec"].get("kind") for p in permits)),
    "states": dict(collections.Counter(p["rec"].get("state") for p in permits)),
    "constructor_returned": dict(collections.Counter(str(p["rec"].get("constructor_returned")) for p in permits)),
    "jobs_with_consumed_permit": len(set(p["job"] for p in permits if p["rec"].get("state") == "consumed")),
    "claims_without_any_permit_file": [c["logical_job"] for c in claims
                                        if not os.path.isdir(os.path.join(c["_dir"], "permits"))],
}

# booked vs proven constructors
booked = sum(c.get("environment_constructor_calls") or 0 for c in claims)
proven = 0
proven_jobs = []
for c in claims:
    ev = []
    pd = os.path.join(c["_dir"], "permits")
    if os.path.isdir(pd):
        for name in os.listdir(pd):
            if name.endswith(".json") and not name.endswith(".issued.json"):
                r = jload(os.path.join(pd, name))
                if r.get("state") == "consumed" and r.get("constructor_returned") is True:
                    ev.append("permit_consumed")
    if c.get("constructor_called_at_unix"):
        ev.append("constructor_timestamp")
    if (c.get("completed_steps") or 0) > 0:
        ev.append("steps")
    if ev:
        proven += 1
        proven_jobs.append({"job": c["logical_job"], "evidence": ev})
rep["constructor_accounting"] = {
    "booked_environment_constructor_calls": booked,
    "booked_joint_run_constructed_calls": sum(c.get("joint_run_constructed_calls") or 0 for c in claims),
    "claims_with_any_construction_evidence": proven,
    "claims_with_row": 0,
    "note": "booked = the two claim counter fields; proven = permit consumed+returned, constructor timestamp, or >0 completed steps",
}

# row crosswalk with corrected label mapping
row_by_key = {}
dup = []
for name, r in rows:
    key = (r.get("label"), r.get("scenario_seed"), r.get("attempt"))
    if key in row_by_key:
        dup.append([name, row_by_key[key], list(key)])
    row_by_key[key] = name
matched = 0
unmatched_complete = []
for c in claims:
    if c.get("status") != "complete":
        continue
    lbl = row_label_of_claim(c["logical_job"])
    md = c.get("metadata") or {}
    key = (lbl, md.get("scenario_seed"), c.get("attempt"))
    if key in row_by_key:
        matched += 1
    else:
        unmatched_complete.append({"job": c["logical_job"], "looked_for": list(key)})
rep["row_crosswalk"] = {
    "rows_on_disk": len(rows),
    "duplicate_row_keys": dup,
    "complete_claims_matched_to_a_row": matched,
    "complete_claims_without_row": unmatched_complete,
    "rows_without_complete_claim": [n for (lbl, sc, att), n in row_by_key.items()
                                     if lbl not in set(row_label_of_claim(c["logical_job"]) for c in claims)],
    "row_kinds": dict(collections.Counter(r.get("kind") for _, r in rows)),
}
rep["constructor_accounting"]["claims_with_row"] = matched

# eval coverage detail
eval_cov = collections.defaultdict(lambda: {"claims": set(), "rows": set()})
for c in claims:
    if scope_of(c["logical_job"]) != "model_eval":
        continue
    parts = c["logical_job"].split("_")
    seed, mode = parts[4], parts[5]
    md = c.get("metadata") or {}
    sc = md.get("scenario_seed")
    eval_cov[(seed, mode)]["claims"].add(sc)
for name, r in rows:
    if r.get("kind") != "model_eval":
        continue
    lbl = r.get("label") or ""
    if lbl.startswith("r73eval_model_new_ep128_"):
        parts = lbl.split("_")
        seed, mode = parts[4], parts[5]
        eval_cov[(seed, mode)]["rows"].add(r.get("scenario_seed"))
rep["eval_coverage"] = {}
for k in sorted(eval_cov):
    v = eval_cov[k]
    rep["eval_coverage"]["%s|%s" % k] = {
        "claimed_scenes": len(v["claims"]),
        "row_scenes": len(v["rows"]),
        "claimed_not_row": sorted(v["claims"] - v["rows"]),
        "row_not_claimed": sorted(v["rows"] - v["claims"]),
        "missing_from_1800000_1800031": sorted(set(range(1800000, 1800032)) - v["claims"]),
    }

# train coverage detail
tr = collections.defaultdict(lambda: {"claims": set(), "rows": set()})
for c in claims:
    if scope_of(c["logical_job"]) != "train":
        continue
    md = c.get("metadata") or {}
    tr[md.get("training_seed")]["claims"].add(md.get("episode_index"))
for name, r in rows:
    if r.get("kind") != "train":
        continue
    tr[r.get("training_seed")]["rows"].add(r.get("episode_index"))
rep["train_coverage"] = {str(k): {"claimed_episodes": len(v["claims"]), "row_episodes": len(v["rows"]),
                                   "contiguous_0_127": sorted(v["claims"]) == list(range(128)) and sorted(v["rows"]) == list(range(128))}
                          for k, v in sorted(tr.items(), key=lambda kv: str(kv[0]))}

# same-timepoint snapshot
snap = {"read_at_unix": time.time()}
for rel in ["ROUND73_LEDGER.json", "ROUND73_FREEZE.json", "ROUND73_LICENCE.json", "ROUND73_MACHINE_VERDICT.json"]:
    p = os.path.join(ROOT, rel)
    snap[rel] = {"exists": os.path.exists(p), "sha256": sha256(p) if os.path.exists(p) else None,
                 "bytes": os.path.getsize(p) if os.path.exists(p) else None}
for rel in ["runtime/ROUND73_LEDGER.json", "runtime/r73_forward_ledger.json",
            "runtime/synthetic_gates/r73_synthetic_optimizer_steps.json",
            "runtime/synthetic_gates/R73_QUALIFICATION_SUITE_RUN.json",
            "ROUND73_ACCOUNTING_AUDIT.json"]:
    p = os.path.join(RAW, rel)
    snap[rel] = {"exists": os.path.exists(p), "sha256": sha256(p) if os.path.exists(p) else None,
                 "bytes": os.path.getsize(p) if os.path.exists(p) else None}
snap["counts"] = {
    "launch_slot_files": sum(len([n for n in os.listdir(os.path.join(SLOTS, c)) if n.endswith(".json")])
                              for c in os.listdir(SLOTS) if os.path.isdir(os.path.join(SLOTS, c))),
    "claim_dirs": len([d for d in os.listdir(CLAIMS) if os.path.isdir(os.path.join(CLAIMS, d))]),
    "attempt_files": len([n for d in os.listdir(CLAIMS) if os.path.isdir(os.path.join(CLAIMS, d))
                           for n in os.listdir(os.path.join(CLAIMS, d)) if n.startswith("attempt_")]),
    "row_files": len([n for n in os.listdir(ROWS) if n.endswith(".json")]),
    "log_files": len([n for n in os.listdir(LOGS) if n.endswith(".jsonl")]),
}
rep["same_timepoint_snapshot"] = snap

def _sk(o):
    if isinstance(o, dict):
        return {("null" if k is None else str(k)): _sk(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sk(v) for v in o]
    return o

with open(OUT, "wb") as f:
    f.write((json.dumps(_sk(rep), indent=1, sort_keys=True) + "\n").encode("utf-8"))
print("wrote", OUT)