import json, os, hashlib, collections
BASE = os.path.join("D:" + os.sep, "code", "\u6574\u5408", "zhenghe-round42", "runs",
                    "comm_bc_initialized_constrained_ppo_20261007_r2")
RAW = os.path.join(BASE, "round73_pure_cost_actor_ablation", "raw")
arch = os.path.join(RAW, "synthetic_test_fixtures", "diverted_real_journal_lines",
                    "ppo_update_calls.jsonl.synthetic.jsonl")
with open(arch, "rb") as f:
    data = f.read()
recs = [json.loads(l.decode("utf-8")) for l in data.split(b"\n") if l.strip()]
def steps(r):
    return int((r.get("line") or {}).get("optimizer_steps", 0))
first = "gap2 task/lambda invariance, arm A"
starts = [i for i, r in enumerate(recs) if r.get("what") == first]
bounds = starts + [len(recs)]
runs = []
for i in range(len(starts)):
    seg = recs[bounds[i]:bounds[i + 1]]
    runs.append({"run": i + 1, "calls": len(seg),
                 "physical_steps": sum(steps(r) for r in seg),
                 "first_what": seg[0].get("what"), "last_what": seg[-1].get("what")})
prefix = data[:20818]
prec = [json.loads(l.decode("utf-8")) for l in prefix.split(b"\n") if l.strip()]
out = {
    "archive_total_lines": len(recs),
    "archive_total_bytes": len(data),
    "optimizer_steps_values": dict(collections.Counter(str(steps(r)) for r in recs)),
    "optimizer_steps_sum_all": sum(steps(r) for r in recs),
    "inferred_runs": runs,
    "inferred_totals": {"runs": len(runs), "calls": sum(r["calls"] for r in runs),
                        "physical_steps": sum(r["physical_steps"] for r in runs)},
    "prefix_20818": {"lines": len(prec), "calls": len(prec),
                     "physical_steps": sum(steps(r) for r in prec),
                     "sha256": hashlib.sha256(prefix).hexdigest()},
    "suffix_after_boundary": {"lines": len(recs) - len(prec),
                              "physical_steps": sum(steps(r) for r in recs[len(prec):])},
}
with open(os.path.join(RAW, "_closeout_archive_runs.json"), "wb") as f:
    f.write((json.dumps(out, indent=1, sort_keys=True) + "\n").encode("utf-8"))
print(json.dumps(out, indent=1))