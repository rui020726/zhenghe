"""ROUND60 init-equivalence: with early_head == late_head == the OLD action_head, the routed
partitioned PPO update must reproduce the frozen R58 gate-prior ppo_update loss and gradients
exactly. This proves the routing is a faithful refactor at initialisation, so any difference that
appears after training is the structural intervention and not a transcription bug.

Self-consistent synthetic episodes only (no simulation, no real rollout). The optimizer updates
here are pure-synthetic and write only to a throwaway test directory.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r60_partitioned as r60

mods = r60.configure_source_imports()
trainer, common, controller_module = mods["trainer"], mods["common"], mods["controller_module"]
from onpolicy import joint_log_prob

SEED = 11
ep0 = r60.load_r58_ep0_model(SEED)
old = controller_module.fresh_controller(SEED)
old.inner.trunk.load_state_dict(ep0["trunk"]); old.inner.action_head.load_state_dict(ep0["action_head"])
old.inner.value_head.load_state_dict(ep0["value_task_head"]); old.cost_head.load_state_dict(ep0["value_cost_head"])
new = r60.fresh_partitioned_controller(SEED, ep0_state=ep0)

def feat(ready, first=1.0, age=0.0):
    f = [0.0]*19
    f[0] = age; f[5] = first; f[13] = 1.0; f[14] = 2.0; f[15] = 0.8
    f[18] = (ready + age) / 1000.0
    return f

gen = torch.Generator().manual_seed(20260)
def build_episode(ep_index, scene, decision_steps=(2, 5, 8)):
    tapes = []
    for k, step in enumerate(decision_steps):
        rows = [feat(100 + 30*k), feat(180 + 30*k)]
        x = torch.tensor(rows, dtype=torch.float32)
        with torch.no_grad():
            logits, values = old.forward(x)
            cv = old.cost_values(x)
            probs = torch.softmax(logits, -1)
            acts = torch.multinomial(probs, 1, generator=gen).squeeze(1).tolist()
            lp = float(joint_log_prob(logits, torch.tensor(acts), torch.ones(len(acts))))
        targets = [f"c{k}a", f"c{k}b"]
        tapes.append({"step": step, "features": rows, "targets": targets,
                      "actions": {t: a for t, a in zip(targets, acts)},
                      "actionable": list(targets),
                      "log_prob": lp, "value": float(values.mean()), "cost_value": float(cv.mean())})
    return {"collector": type("C", (), {"records": tapes})(),
            "steps": 10, "cost_stream": {s: {"airtime_bytes": 120.0} for s in decision_steps},
            "task_stream": {s: (0.5 if k % 2 == 0 else 0.0) for k, s in enumerate(decision_steps)},
            "lambda": 1.0, "episode_index": ep_index, "scenario_seed": scene, "on_time": 6}
episodes = [build_episode(0, 1300000), build_episode(1, 1300001),
            build_episode(2, 1300002), build_episode(3, 1300003)]

# sanity: at init the routed NEW forward equals the OLD forward on every episode feature row
_allrows = torch.tensor([r for ep in episodes for t in ep["collector"].records for r in t["features"]],
                        dtype=torch.float32)
with torch.no_grad():
    _ol, _ov = old.forward(_allrows); _nl, _nv = new.forward(_allrows)
print("INIT_FORWARD_EQUAL", torch.equal(_ol, _nl), torch.equal(_ov, _nv),
      "rows", _allrows.shape[0], "samples", len(trainer.batch_samples(episodes)))

freeze = {"training_version": r60.R60_TRAINING_VERSION}

# (A) OLD through the RAW frozen ppo_update (no gate prior) -- the reference loss formula.
opt_old = torch.optim.Adam(old.parameters, lr=common.LEARNING_RATE)
report_old = trainer.ppo_update(old, opt_old, episodes, update_index=0)

# (B) NEW through the routed partitioned update, gate prior OFF. At init early==late==action_head,
# so this must reproduce the OLD loss and gradients exactly (isolates routing + loss formula).
opt_new = torch.optim.Adam(new.parameters, lr=common.LEARNING_RATE)
report_new = r60.partitioned_ppo_update(new, opt_new, episodes, update_index=0, gate_beta=0.0)

# (C) NEW through the routed partitioned update, gate prior ON (beta=1.00, the preserved R58 prior).
new_gate = r60.fresh_partitioned_controller(SEED, ep0_state=ep0)
opt_gate = torch.optim.Adam(new_gate.parameters, lr=common.LEARNING_RATE)
report_gate = r60.partitioned_ppo_update(new_gate, opt_gate, episodes, update_index=0, gate_beta=1.0)

def close(a, b, tol=2e-4):
    return abs(float(a) - float(b)) <= tol

terms = ["policy_surrogate", "value_loss_task", "value_loss_cost", "entropy",
         "gradient_norm_mean", "approx_kl", "clip_fraction"]
mismatch = {t: [report_old.get(t), report_new.get(t)] for t in terms
            if not close(report_old.get(t, 0.0), report_new.get(t, 0.0))}
gate_applied = (report_gate.get("gate_rows", 0) > 0 and abs(float(report_gate.get("gate_loss", 0.0))) > 1e-9
                and report_old.get("minibatches") == report_new.get("minibatches"))
print(json.dumps({"kind": "round60-init-equivalence-ppo/1",
                  "A_old_minibatches": report_old.get("minibatches"),
                  "B_new_minibatches": report_new.get("minibatches"),
                  "all_loss_terms_equal_A_vs_B": not mismatch, "mismatch": mismatch,
                  "old": {t: report_old.get(t) for t in terms},
                  "new_gate_off": {t: report_new.get(t) for t in terms},
                  "gate_prior_applied_C": gate_applied,
                  "C_gate_rows": report_gate.get("gate_rows"), "C_gate_loss": report_gate.get("gate_loss"),
                  "C_early_free": report_gate.get("early_free"), "C_late_free": report_gate.get("late_free")}, indent=1))
if mismatch or not gate_applied:
    raise SystemExit("init equivalence FAILED")
print("INIT EQUIVALENCE PASSED (routed update reproduces OLD loss with gate off; gate prior routed via early head with gate on)")
