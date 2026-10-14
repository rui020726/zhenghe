import json, os, collections
BASE = os.path.join("D:" + os.sep, "code", "\u6574\u5408", "zhenghe-round42", "runs",
                    "comm_bc_initialized_constrained_ppo_20261007_r2")
R73 = os.path.join(BASE, "round73_pure_cost_actor_ablation", "raw")
def jload(p):
    with open(p, "rb") as f:
        return json.loads(f.read().decode("utf-8"))
out = {}
# 1) train e127 row param sha vs eval param sha
chain = {}
for seed in (11, 22, 33):
    cands = [n for n in os.listdir(os.path.join(R73, "rows"))
             if n.startswith("v2_r73_train_s%d_e127_" % seed)]
    if len(cands) != 1:
        chain[str(seed)] = {"issue": "row count %d" % len(cands)}
        continue
    row = jload(os.path.join(R73, "rows", cands[0]))
    ev = jload(os.path.join(R73, "rows", "v2_r73eval_model_new_ep128_s%d_greedy_s1800000_a1.json" % seed))
    chain[str(seed)] = {
        "train_e127_row": cands[0],
        "train_e127_param_sha": (row.get("controller_parameters_after") or {}).get("parameter_sha256"),
        "eval_param_sha": (ev.get("checkpoint_load_identity") or {}).get("parameter_identity", {}).get("parameter_sha256"),
        "train_e127_lambda": row.get("lambda"),
        "train_e127_optimizer_steps": row.get("optimizer_steps"),
        "train_e127_episode_index": row.get("episode_index"),
        "train_e127_training_seed": row.get("training_seed"),
    }
    chain[str(seed)]["match"] = chain[str(seed)]["train_e127_param_sha"] == chain[str(seed)]["eval_param_sha"]
out["train_e127_to_eval_chain"] = chain
# 2) discovered / task ceiling per scene (R73 greedy s33)
disc = {}
for sc in range(1800000, 1800032):
    row = jload(os.path.join(R73, "rows", "v2_r73eval_model_new_ep128_s33_greedy_s%d_a1.json" % sc))
    r71 = jload(os.path.join(BASE, "round71_critic_initialization_calibration", "raw", "rows",
                             "v2_r71eval_model_new_ep128_s33_greedy_a1_s%d_a1.json" % sc))
    disc[str(sc)] = {"r73_discovered": row.get("discovered"), "r73_on_time": row.get("on_time"),
                     "r73_completed": row.get("completed"), "r73_expired": row.get("expired"),
                     "r73_active": row.get("active"), "r73_unassigned": row.get("unassigned"),
                     "r71_discovered": r71.get("discovered"), "r71_on_time": r71.get("on_time"),
                     "same_discovered": row.get("discovered") == r71.get("discovered")}
out["discovered_closure_s33"] = disc
# 3) claim metadata for the 96 greedy cells
claims = {}
for job in sorted(os.listdir(os.path.join(R73, "launch_claims"))):
    if not job.startswith("r73eval_model_new_ep128_") or "_greedy_" not in job:
        continue
    d = os.path.join(R73, "launch_claims", job)
    for n in sorted(os.listdir(d)):
        if n.startswith("attempt_"):
            r = jload(os.path.join(d, n))
            claims[job] = r
bad = []
for job, r in claims.items():
    md = r.get("metadata") or {}
    if md.get("arm") != "NEW_EP128" or md.get("mode") != "greedy":
        bad.append({"job": job, "issue": "metadata"})
    if not str(md.get("eval_version", "")).endswith("+r73-pure-cost-actor-objective/1+r73-eval/1"):
        bad.append({"job": job, "issue": "eval_version"})
    if r.get("status") != "complete":
        bad.append({"job": job, "issue": "status " + str(r.get("status"))})
out["greedy_claim_metadata"] = {"claims": len(claims), "problems": bad,
                                "statuses": dict(collections.Counter(r.get("status") for r in claims.values()))}
# 4) same check for sampled
sclaims = {}
for job in sorted(os.listdir(os.path.join(R73, "launch_claims"))):
    if not job.startswith("r73eval_model_new_ep128_") or "_sampled_" not in job:
        continue
    d = os.path.join(R73, "launch_claims", job)
    for n in sorted(os.listdir(d)):
        if n.startswith("attempt_"):
            sclaims[job] = jload(os.path.join(d, n))
out["sampled_claim_status"] = dict(collections.Counter(r.get("status") for r in sclaims.values()))
out["sampled_claims_total"] = len(sclaims)
with open(os.path.join(R73, "_closeout_chain_check.json"), "wb") as f:
    f.write((json.dumps(out, indent=1, sort_keys=True) + "\n").encode("utf-8"))
print(json.dumps({"chain": {k: v.get("match") for k, v in chain.items()},
                  "greedy_claims": out["greedy_claim_metadata"]["claims"],
                  "greedy_claim_problems": out["greedy_claim_metadata"]["problems"],
                  "sampled_claims": len(sclaims),
                  "sampled_status": out["sampled_claim_status"]}, indent=1))