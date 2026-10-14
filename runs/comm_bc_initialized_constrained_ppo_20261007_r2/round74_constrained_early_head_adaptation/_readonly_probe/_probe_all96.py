import json, glob, sys, re, os
d = sys.argv[1]
pat = re.compile(r"r40_batch_s(\d+)_b(\d+)_attempt(\d+)\.json$")
files = []
for p in glob.glob(d + "\\r40_batch_*.json"):
    m = pat.search(os.path.basename(p))
    if m:
        files.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), p))
files.sort()
print("total files:", len(files))
for seed in (11, 22, 33):
    rs = [f for f in files if f[0] == seed]
    print("--- seed", seed, "batches:", len(rs), "---")
    tot_s = tot_m = tot_st = 0
    for s, b, a, p in rs:
        o = json.load(open(p, encoding="utf-8"))
        u = o["update"]
        tot_s += u["samples"]; tot_m += u["minibatches"]; tot_st += u["optimizer_steps"]
        print("  b%-2d samples=%-4d minibatches=%d steps=%d epochs=%d early=%s late=%s resumed=%s lambda=%.3f eps_done=%d opt_after=%d" % (
            b, u["samples"], u["minibatches"], u["optimizer_steps"], u["epochs"], u.get("early_free"), u.get("late_free"), o.get("resumed_batch"), o["lambda_at_rollout"], o["episodes_done_after"], o["optimizer_steps_after"]))
    print("  SUM samples=%d SUM minibatches=%d SUM optimizer_steps=%d" % (tot_s, tot_m, tot_st))