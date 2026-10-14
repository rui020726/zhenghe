import json, os, collections, hashlib
BASE = os.path.join("D:" + os.sep, "code", "\u6574\u5408", "zhenghe-round42", "runs",
                    "comm_bc_initialized_constrained_ppo_20261007_r2")
RAW = os.path.join(BASE, "round73_pure_cost_actor_ablation", "raw")
RT = os.path.join(RAW, "runtime")
def lines(p):
    with open(p, "rb") as f:
        return [l for l in f.read().split(b"\n") if l.strip()]
out = {}
for name in ("ppo_update_calls.jsonl", "optimizer_step_calls.jsonl", "checkpoint_writes.jsonl"):
    p = os.path.join(RT, name)
    ls = lines(p)
    recs = []
    bad = 0
    for l in ls:
        try:
            recs.append(json.loads(l.decode("utf-8")))
        except Exception:
            bad += 1
    out[name] = {"lines": len(ls), "parsed": len(recs), "unparseable": bad,
                 "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                 "bytes": os.path.getsize(p)}
    if name == "ppo_update_calls.jsonl":
        out[name]["seeds"] = dict(collections.Counter(str(r.get("training_seed") or r.get("seed")) for r in recs))
        out[name]["optimizer_steps_sum"] = sum(int(r.get("optimizer_steps", 0)) for r in recs)
        out[name]["optimizer_steps_values"] = dict(collections.Counter(str(r.get("optimizer_steps")) for r in recs))
        out[name]["structures"] = sorted(set(str(r.get("structure")) for r in recs))
        out[name]["first"] = recs[0] if recs else None
        out[name]["last"] = recs[-1] if recs else None
    if name == "optimizer_step_calls.jsonl":
        out[name]["keys"] = sorted(set(k for r in recs for k in r.keys()))
        out[name]["first"] = recs[0] if recs else None
print(json.dumps(out, indent=1)[:4000])