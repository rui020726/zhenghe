import json, os, hashlib, collections
BASE = os.path.join("D:" + os.sep, "code", "\u6574\u5408", "zhenghe-round42", "runs",
                    "comm_bc_initialized_constrained_ppo_20261007_r2")
R = os.path.join(BASE, "round73_pure_cost_actor_ablation")
RAW = os.path.join(R, "raw")
FX = os.path.join(RAW, "synthetic_test_fixtures")
def jload(p):
    with open(p, "rb") as f:
        return json.loads(f.read().decode("utf-8"))
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

out = {}
# ---- 1. diverted archive: line count, optimizer_steps sum, labels in order, byte boundary
arch = os.path.join(FX, "diverted_real_journal_lines", "ppo_update_calls.jsonl.synthetic.jsonl")
with open(arch, "rb") as f:
    data = f.read()
lines = [l for l in data.split(b"\n") if l.strip()]
recs = [json.loads(l.decode("utf-8")) for l in lines]
boundary = 20818
prefix = data[:boundary]
out["archive"] = {
    "path": arch.replace(BASE, "<BASE>"),
    "total_bytes": len(data),
    "total_lines": len(recs),
    "sha256_whole_file": sha256_bytes(data),
    "boundary_bytes": boundary,
    "boundary_lines": len([l for l in prefix.split(b"\n") if l.strip()]),
    "sha256_first_20818_bytes": sha256_bytes(prefix),
    "matches_receipt_recorded_prefix_sha": sha256_bytes(prefix) == "950f380746ef7808ba4cdd227b4079af0b37b25828d546d5892222dab6b9d7a9",
    "optimizer_steps_field_present": sum(1 for r in recs if "optimizer_steps" in r),
    "optimizer_steps_sum": sum(int(r.get("optimizer_steps", 0)) for r in recs),
    "optimizer_steps_distinct": dict(collections.Counter(str(r.get("optimizer_steps")) for r in recs)),
    "labels_in_order": [str(r.get("what") or r.get("label") or r.get("kind")) for r in recs],
    "keys_seen": sorted(set(k for r in recs for k in r.keys())),
}
# run segmentation: a new run starts at the alphabetical-first label
first_label = "gap2 task/lambda invariance, arm A"
starts = [i for i, l in enumerate(out["archive"]["labels_in_order"]) if l == first_label]
runs = []
bounds = starts + [len(recs)]
for i in range(len(starts)):
    seg = recs[bounds[i]:bounds[i + 1]]
    runs.append({"run": i + 1, "calls": len(seg),
                 "physical_steps": sum(int(r.get("optimizer_steps", 0)) for r in seg),
                 "labels": [str(r.get("what") or r.get("label")) for r in seg]})
out["archive"]["inferred_runs"] = runs
out["archive"]["inferred_run_count"] = len(runs)
out["archive"]["inferred_totals"] = {
    "runs": len(runs),
    "ppo_update_calls": sum(r["calls"] for r in runs),
    "physical_optimizer_steps": sum(r["physical_steps"] for r in runs),
}
# ---- 2. receipts
rec = jload(os.path.join(FX, "R73_SYNTHETIC_PPO_JOURNAL_RECEIPT.json"))
charges = rec.get("gate_forward_charges") or []
single = [c for c in charges if "one 19-D row" in str(c.get("what"))]
agg = [c for c in charges if c not in single]
out["receipt_reconstructed"] = {
    "path": "raw/synthetic_test_fixtures/R73_SYNTHETIC_PPO_JOURNAL_RECEIPT.json",
    "reconstructed_flag": rec.get("reconstructed"),
    "live_update_count": rec.get("live_update_count"),
    "physical_optimizer_steps": rec.get("physical_optimizer_steps"),
    "forward_rows": rec.get("forward_rows"),
    "charge_entries": len(charges),
    "charge_rows_sum": sum(int(c.get("rows", 0)) for c in charges),
    "single_row_forwards_recorded": len(single),
    "single_row_total": sum(int(c.get("rows", 0)) for c in single),
    "aggregate_precharges": [{"what": c.get("what"), "rows": c.get("rows")} for c in agg],
    "aggregate_precharge_total": sum(int(c.get("rows", 0)) for c in agg),
    "derivation_note": rec.get("reconstruction_note"),
}
scr = jload(os.path.join(FX, "R73_SYNTHETIC_PPO_JOURNAL_RECEIPT_SCRATCH_OVERWRITE.json"))
scharges = scr.get("gate_forward_charges") or []
ssingle = [c for c in scharges if "one 19-D row" in str(c.get("what"))]
out["receipt_scratch_copy"] = {
    "path": "raw/synthetic_test_fixtures/R73_SYNTHETIC_PPO_JOURNAL_RECEIPT_SCRATCH_OVERWRITE.json",
    "reconstructed_flag": scr.get("reconstructed"),
    "live_update_count": scr.get("live_update_count"),
    "physical_optimizer_steps": scr.get("physical_optimizer_steps"),
    "forward_rows": scr.get("forward_rows"),
    "charge_entries": len(scharges),
    "charge_rows_sum": sum(int(c.get("rows", 0)) for c in scharges),
    "single_row_forwards_recorded": len(ssingle),
    "single_row_total": sum(int(c.get("rows", 0)) for c in ssingle),
    "aggregate_precharge_total": sum(int(c.get("rows", 0)) for c in scharges if c not in ssingle),
}
# ---- 3. ledgers
def led(p):
    return jload(p)
out["forward_ledgers"] = {
    "real": led(os.path.join(RAW, "runtime", "r73_forward_ledger.json")),
    "debug_scratch_copy": led(os.path.join(FX, "debug_scratch", "forward_rows.json")),
    "cap_probe_fixture": led(os.path.join(FX, "cap_probe", "forward_rows.json")),
}
out["synthetic_step_guards"] = {
    "real": led(os.path.join(RAW, "runtime", "synthetic_gates", "r73_synthetic_optimizer_steps.json")),
    "debug_scratch_copy": led(os.path.join(FX, "debug_scratch", "synthetic_optimizer_steps.json")),
    "cap_probe_fixture": led(os.path.join(FX, "cap_probe", "steps.json")),
    "contention_fixture": led(os.path.join(FX, "contention", "steps.json")),
}
with open(os.path.join(RAW, "_closeout_offline_accounting.json"), "wb") as f:
    f.write((json.dumps(out, indent=1, sort_keys=True) + "\n").encode("utf-8"))
print(json.dumps({k: out[k] for k in ("archive",)}, indent=1)[:200])
print("real ledger:", json.dumps(out["forward_ledgers"]["real"]))
print("scratch ledger:", json.dumps(out["forward_ledgers"]["debug_scratch_copy"]))
print("receipt(real-path) charges:", out["receipt_reconstructed"]["charge_entries"],
      "rows:", out["receipt_reconstructed"]["charge_rows_sum"],
      "single:", out["receipt_reconstructed"]["single_row_forwards_recorded"],
      "agg:", out["receipt_reconstructed"]["aggregate_precharge_total"])
print("receipt(scratch copy) charges:", out["receipt_scratch_copy"]["charge_entries"],
      "rows:", out["receipt_scratch_copy"]["charge_rows_sum"],
      "single:", out["receipt_scratch_copy"]["single_row_forwards_recorded"],
      "agg:", out["receipt_scratch_copy"]["aggregate_precharge_total"])