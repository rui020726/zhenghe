# ROUND73 read-only close-out helper: claim / slot / permit audit.
# stdlib only. Parses existing files; writes nothing except the report path given on argv.
import json, os, sys, collections

ROOT = sys.argv[1]
OUT = sys.argv[2]
SLOTS = os.path.join(ROOT, "launch_slots")
CLAIMS = os.path.join(ROOT, "launch_claims")
ROWS = os.path.join(ROOT, "rows")

def jload(p):
    with open(p, "rb") as f:
        return json.loads(f.read().decode("utf-8"))

slots = {}
for cat in sorted(os.listdir(SLOTS)):
    d = os.path.join(SLOTS, cat)
    if not os.path.isdir(d):
        continue
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        r = jload(os.path.join(d, name))
        slots[(r.get("category"), r.get("slot"))] = {"file": os.path.join(d, name), "rec": r}

claims = {}
for job in sorted(os.listdir(CLAIMS)):
    d = os.path.join(CLAIMS, job)
    if not os.path.isdir(d):
        continue
    for name in sorted(os.listdir(d)):
        if not (name.startswith("attempt_") and name.endswith(".json")):
            continue
        r = jload(os.path.join(d, name))
        key = (r.get("logical_job"), r.get("attempt"))
        claims[key] = {"file": os.path.join(d, name), "rec": r}

CONSTRUCTOR_FIELDS = ["environment_constructor_calls", "joint_run_constructed_calls"]
INT_FIELDS = ["run_invocation_calls", "joint_run_method_calls", "run_return_calls",
              "run_failure_calls", "joint_run_build_failures", "completed_steps"]

def field_stats(recs, field):
    present = 0
    missing = []
    types = collections.Counter()
    values = collections.Counter()
    total = 0
    for k, v in recs.items():
        r = v["rec"]
        if field not in r:
            missing.append(k[0])
            continue
        present += 1
        val = r[field]
        types[type(val).__name__] += 1
        if isinstance(val, bool):
            values[str(val)] += 1
        elif isinstance(val, (int, float)):
            values[val] += 1
            total += val
        else:
            values[str(val)] += 1
    return {"field": field, "present": present, "missing_count": len(missing),
            "missing_jobs": sorted(set(missing)), "types": dict(types),
            "distinct_values": {str(k): v for k, v in sorted(values.items(), key=lambda kv: str(kv[0]))},
            "sum": total}

claim_list = list(claims.items())
summary = {
    "slots_total": len(slots),
    "slots_by_category": dict(collections.Counter(k[0] for k in slots)),
    "claims_total": len(claims),
    "claim_keys": len(set(k[0] for k in claims)),
    "claim_attempts": dict(collections.Counter(k[1] for k in claims)),
    "constructor_field_stats": [field_stats(claims, f) for f in CONSTRUCTOR_FIELDS],
    "int_field_stats": [field_stats(claims, f) for f in INT_FIELDS],
}

# slot <-> claim cross-check
slot_to_claim = {}
unmatched_claims = []
for (job, att), v in claims.items():
    r = v["rec"]
    key = (r.get("category"), r.get("launch_slot"))
    if key in slot_to_claim:
        unmatched_claims.append({"job": job, "reason": "slot claimed twice", "slot": key})
    slot_to_claim[key] = job
slots_without_claim = [k for k in slots if k not in slot_to_claim]
claims_without_slot = [k for k in claims if (claims[k]["rec"].get("category"), claims[k]["rec"].get("launch_slot")) not in slots]
summary["slot_claim_crosscheck"] = {
    "distinct_slots_claimed": len(slot_to_claim),
    "slots_without_claim": [list(k) for k in slots_without_claim],
    "claims_without_slot": [list(k) for k in claims_without_slot],
    "duplicate_slot_claims": unmatched_claims,
}

# statuses and scopes
def scope_of(job):
    if job.startswith("r73_new_training_"):
        return "train"
    if job.startswith("r73eval_model_new_ep128_"):
        return "model_eval"
    if job.startswith("r73eval_focal_") or "focal" in job:
        return "focal"
    return "other:" + job

status_counter = collections.Counter()
scope_counter = collections.Counter()
by_scope_status = collections.Counter()
zero_constructor = []
nonzero_but_not_complete = []
complete_but_zero_steps = []
for (job, att), v in claims.items():
    r = v["rec"]
    st = r.get("status")
    sc = scope_of(job)
    status_counter[st] += 1
    scope_counter[sc] += 1
    by_scope_status[(sc, st)] += 1
    ec = r.get("environment_constructor_calls")
    jc = r.get("joint_run_constructed_calls")
    cs = r.get("completed_steps")
    ss = r.get("simulation_started")
    if ec == 0 and jc == 0:
        zero_constructor.append(job)
    if (ec or 0) > 0 and st != "complete":
        nonzero_but_not_complete.append({"job": job, "status": st, "ec": ec, "jc": jc,
                                          "steps": cs, "sim_started": ss})
    if st == "complete" and (cs or 0) == 0:
        complete_but_zero_steps.append(job)
summary["status_counts"] = dict(status_counter)
summary["scope_counts"] = dict(scope_counter)
summary["scope_status_counts"] = {("%s|%s" % k): v for k, v in sorted(by_scope_status.items())}
summary["zero_constructor_claims"] = sorted(zero_constructor)
summary["constructor_positive_but_not_complete"] = nonzero_but_not_complete
summary["complete_but_zero_steps"] = sorted(complete_but_zero_steps)

# rows on disk vs claims
rows = sorted(n for n in os.listdir(ROWS) if n.endswith(".json"))
summary["rows_on_disk"] = len(rows)
row_labels = collections.Counter()
for n in rows:
    r = jload(os.path.join(ROWS, n))
    row_labels[r.get("label")] += 1
summary["row_label_counts"] = dict(row_labels)
summary["row_labels_not_in_claims"] = sorted(set(row_labels) - set(k[0] for k in claims))
summary["complete_claims_without_row"] = sorted(
    set(k[0] for k, v in claims.items() if v["rec"].get("status") == "complete") - set(row_labels))

# the interrupted cell detail
interrupted = {}
for (job, att), v in claims.items():
    r = v["rec"]
    if r.get("status") != "complete" and (r.get("environment_constructor_calls") or 0) > 0:
        interrupted[job] = r
summary["interrupted_or_incomplete_with_constructor"] = interrupted

# per-seed / per-scene coverage for train
train_cov = collections.defaultdict(set)
for (job, att), v in claims.items():
    if scope_of(job) != "train":
        continue
    md = v["rec"].get("metadata") or {}
    train_cov[md.get("training_seed")].add(md.get("episode_index"))
summary["train_episode_coverage"] = {str(k): {"episodes": len(v), "min": min(v), "max": max(v),
                                              "contiguous_0_127": sorted(v) == list(range(128))}
                                     for k, v in sorted(train_cov.items(), key=lambda kv: str(kv[0]))}

eval_cov = collections.defaultdict(set)
for (job, att), v in claims.items():
    if scope_of(job) != "model_eval":
        continue
    parts = job.split("_")
    # r73eval_model_new_ep128_s33_sampled_scene1800029
    seed = parts[4]; mode = parts[5]; scene = parts[6].replace("scene", "")
    eval_cov[(seed, mode)].add(int(scene))
summary["model_eval_scene_coverage"] = {("%s|%s" % k): {"scenes": len(v), "min": min(v), "max": max(v),
                                                       "contiguous_1800000_1800031": sorted(v) == list(range(1800000, 1800032))}
                                        for k, v in sorted(eval_cov.items(), key=lambda kv: str(kv[0]))}

with open(OUT, "wb") as f:
    f.write((json.dumps(summary, indent=1, sort_keys=True) + "\n").encode("utf-8"))
print("wrote", OUT)