import json, sys, glob

def scan(path):
    o = json.load(open(path, encoding="utf-8"))
    u = o["update"]
    rec = {
        "file": path.split("\\")[-1],
        "seed": o["seed"], "batch": o["batch"],
        "episodes": o["episodes"], "scenarios": o["scenarios"],
        "lambda_at_rollout": o["lambda_at_rollout"],
        "samples": u["samples"], "minibatches": u["minibatches"],
        "optimizer_steps": u["optimizer_steps"], "epochs": u["epochs"],
        "early_free": u.get("early_free"), "late_free": u.get("late_free"),
        "gate_rows": u.get("gate_rows"),
        "trainable_objs": u.get("trainable_parameter_object_count"),
        "frozen_objs": u.get("frozen_parameter_object_count"),
        "dual_eta": o["dual"].get("eta"),
        "lambda_before": o["dual"].get("lambda_before"),
        "lambda_next": o["dual"].get("lambda_next"),
        "resumed_batch": o.get("resumed_batch"),
        "structure": o["structure"],
        "before_sha": o["parameters_before_sha"][:12],
        "after_sha": o["parameters_after_sha"][:12],
    }
    # per-sample rollout payload presence
    blob = json.dumps(o)
    for t in ["log_prob", "behavior_log_prob", "features", "observation", "sequential_mask", "ALL_ROWS", "beta", "advantage", "value", "reward"]:
        rec["has_" + t] = blob.count(t)
    # numeric arrays with len>=8 anywhere
    arrs = {}
    def walk(x, pth):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, pth + "." + k)
        elif isinstance(x, list):
            if x and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x) and len(x) >= 8:
                arrs.setdefault(pth, len(x))
            for v in x:
                walk(v, pth + "[]")
    walk(o, "")
    rec["numeric_arrays"] = arrs
    return rec

d = sys.argv[1]
files = sorted(glob.glob(d + "\\r40_batch_s11_b*.json"))
print("n files:", len(files))
for f in files[:3]:
    r = scan(f)
    print(json.dumps(r, ensure_ascii=False, indent=1))