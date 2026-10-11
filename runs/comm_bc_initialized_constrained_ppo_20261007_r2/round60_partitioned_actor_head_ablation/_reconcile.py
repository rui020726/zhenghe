"""ROUND60 zero-environment reconciliation (read-only). No environment construction, simulation,
claim, optimizer update, PPO replay, RNG restore, checkpoint write, or new evaluation. Only
read-only parsing + pure-computation model-state verification."""
import json, hashlib
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
QUAR = HERE / "_quarantine_attempt1_criticdetect_missing"

def param_sha_from_payload(payload):
    sd = payload["controller"]
    order = [("trunk", "trunk"), ("early_head", "early_head"), ("late_head", "late_head"),
             ("value_task_head", "value_task_head"), ("value_cost_head", "value_cost_head")]
    tensors = {}
    for key, prefix in order:
        for name, t in sd[key].items():
            tensors[f"{prefix}.{name}"] = t.detach().cpu()
    h = hashlib.sha256()
    for name in sorted(tensors):
        h.update(name.encode()); h.update(tensors[name].contiguous().numpy().tobytes())
    return h.hexdigest()

print("== compliant ep128 provenance (read-only) ==")
for s in (11, 22, 33):
    pl = torch.load(RAW / "checkpoints" / f"v2_train_s{s}_ep128.pt", map_location="cpu", weights_only=False)
    sha = param_sha_from_payload(pl)
    st = pl["training_state"]
    b31 = json.loads((RAW / "checkpoints" / f"r40_batch_s{s}_b31_attempt01.json").read_text(encoding="utf-8"))
    print(f"seed {s}: ep128 param_sha={sha[:16]} episodes_done={st['episodes_done']} updates={st['updates']} "
          f"opt_steps={st['optimizer_steps']} lambda={st['lambda']} structure={pl.get('structure')} "
          f"| b31_params_after={b31['parameters_after_sha'][:16]} match={sha==b31['parameters_after_sha']}")

print("== eval binding to compliant ep128 ==")
import glob, os
for s in (11, 22, 33):
    p = glob.glob(str(RAW / "rows" / f"v2_r60_eval_new_s{s}_greedy_s1700000_a1.json"))[0]
    r = json.load(open(p, encoding="utf-8"))
    bind = r["checkpoint_load_identity"]["parameter_identity"]["parameter_sha256"][:16]
    pl = torch.load(RAW / "checkpoints" / f"v2_train_s{s}_ep128.pt", map_location="cpu", weights_only=False)
    print(f"seed {s}: eval_bind={bind} ep128={param_sha_from_payload(pl)[:16]} match={bind==param_sha_from_payload(pl)[:16]}")

print("== consumption reconciliation ==")
probe_env = 1
quar_claims = len(list((QUAR / "launch_claims").glob("*/*.json")))
quar_complete = sum(1 for p in (QUAR / "launch_claims").glob("*/*.json")
                    if json.load(open(p, encoding="utf-8"))["status"] == "complete")
quar_opt = sum(int(json.load(open(p, encoding="utf-8"))["update"]["optimizer_steps"])
               for p in (QUAR / "checkpoints").glob("r40_batch_s11_b*.json"))
quar_batches = len(list((QUAR / "checkpoints").glob("r40_batch_s11_b*.json")))
raw_claims = len(list((RAW / "launch_claims").glob("*/*.json")))
raw_complete = sum(1 for p in (RAW / "launch_claims").glob("*/*.json")
                   if json.load(open(p, encoding="utf-8"))["status"] == "complete")
raw_opt = sum(int(json.load(open(p, encoding="utf-8"))["update"]["optimizer_steps"])
              for p in (RAW / "checkpoints").glob("r40_batch_s*_b*.json"))
raw_batches = len(list((RAW / "checkpoints").glob("r40_batch_s*_b*.json")))
total_env = probe_env + quar_claims + raw_claims
total_opt = quar_opt + raw_opt
print(f"probe: env={probe_env}")
print(f"non-compliant(quarantine): env_constructions={quar_claims} completed_episodes={quar_complete} "
      f"PPO_updates={quar_batches} optimizer_steps={quar_opt}")
print(f"compliant(raw): claims={raw_claims} complete={raw_complete} PPO_updates={raw_batches} optimizer_steps={raw_opt}")
print(f"TOTAL real env constructions = {probe_env}+{quar_claims}+{raw_claims} = {total_env}")
print(f"TOTAL real optimizer steps = {quar_opt}+{raw_opt} = {total_opt}")
print(f"hard cap = 600 -> headroom = {600 - total_env}  ({'OVER by '+str(total_env-600) if total_env>600 else 'within'})")
print(f"report claimed cumulative 577 / headroom 23 -> undercounted by {total_env - 577}")
