import json, os
BASE = os.path.join("D:" + os.sep, "code", "\u6574\u5408", "zhenghe-round42", "runs",
                    "comm_bc_initialized_constrained_ppo_20261007_r2")
R = os.path.join(BASE, "round73_pure_cost_actor_ablation")
RAW = os.path.join(R, "raw")
CK = os.path.join(RAW, "checkpoints")
def jload(p):
    with open(p, "rb") as f:
        return json.loads(f.read().decode("utf-8"))
freeze = jload(os.path.join(R, "ROUND73_FREEZE.json"))
init = freeze["init_source_per_seed"]
out = {"seeds": {}, "notes": []}
for seed, sk in ((11, "s11"), (22, "s22"), (33, "s33")):
    recs = {}
    for b in range(32):
        p = os.path.join(CK, "r40_batch_%s_b%d_attempt01.json" % (sk, b))
        if not os.path.exists(p):
            out["notes"].append("missing batch record %s b%d" % (sk, b))
            continue
        recs[b] = jload(p)
    problems = []
    if 0 in recs:
        if recs[0].get("parameters_before_sha") != init[sk]["parameter_sha256"]:
            problems.append("batch0_before_sha_!=_registered_INIT_param_sha")
    for b in range(1, 32):
        if b in recs and (b - 1) in recs:
            if recs[b].get("parameters_before_sha") != recs[b - 1].get("parameters_after_sha"):
                problems.append("chain_break_between_batch_%d_and_%d" % (b - 1, b))
    last = recs.get(31, {})
    ev = jload(os.path.join(RAW, "rows", "v2_r73eval_model_new_ep128_s%d_greedy_s1800000_a1.json" % seed))
    ev_sha = ev["checkpoint_load_identity"]["parameter_identity"]["parameter_sha256"]
    if last.get("parameters_after_sha") != ev_sha:
        problems.append("batch31_after_sha_!=_eval_param_sha")
    if last.get("episodes_done_after") != 128:
        problems.append("episodes_done_after_!=_128")
    steps = sorted(set(r.get("optimizer_steps_after") for r in recs.values()))
    lambdas = [r.get("dual", {}).get("lambda_next") for r in recs.values()]
    out["seeds"][sk] = {
        "batch_records": len(recs),
        "batch0_before_sha": recs.get(0, {}).get("parameters_before_sha"),
        "registered_init_param_sha": init[sk]["parameter_sha256"],
        "batch0_before_matches_registered_init": recs.get(0, {}).get("parameters_before_sha") == init[sk]["parameter_sha256"],
        "chain_contiguous_0_to_31": all(
            recs[b].get("parameters_before_sha") == recs[b - 1].get("parameters_after_sha")
            for b in range(1, 32) if b in recs and (b - 1) in recs),
        "batch31_after_sha": last.get("parameters_after_sha"),
        "eval_param_sha": ev_sha,
        "batch31_after_matches_eval": last.get("parameters_after_sha") == ev_sha,
        "episodes_done_after_final": last.get("episodes_done_after"),
        "optimizer_steps_after_final": last.get("optimizer_steps_after"),
        "optimizer_steps_after_set": steps,
        "lambda_start": recs.get(0, {}).get("dual", {}).get("lambda_before"),
        "lambda_final": last.get("dual", {}).get("lambda_next"),
        "lambda_path": lambdas,
        "actor_objective": last.get("actor_objective"),
        "structure": last.get("structure"),
        "problems": problems,
    }
with open(os.path.join(RAW, "_closeout_training_chain.json"), "wb") as f:
    f.write((json.dumps(out, indent=1, sort_keys=True) + "\n").encode("utf-8"))
print(json.dumps(out, indent=1))