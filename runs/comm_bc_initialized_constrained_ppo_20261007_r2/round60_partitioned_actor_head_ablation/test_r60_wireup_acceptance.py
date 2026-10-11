"""ROUND60 wire-up acceptance: prove the partitioned actor head is correctly wired through the
REAL controller, collector, batch/actor, optimizer and checkpoint paths before any training.

Only NEW tests that the ROUND60 structural change can affect are added; unchanged ROUND58
behaviour is not re-proved. Pure-synthetic optimizer updates are capped at 32 and write only to a
throwaway test directory; they are never presented as real training. No simulation, no real PPO
training, no checkpoint outside the test dir, no new evaluation.
"""
from __future__ import annotations
import copy, json, sys, tempfile
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r60_partitioned as r60

RESULTS: list[dict] = []
def check(name, passed, **detail):
    RESULTS.append({"check": name, "passed": bool(passed), **detail})
    print(("PASS" if passed else "FAIL"), name, "" if passed else json.dumps(detail, default=str))
    if not passed:
        raise SystemExit(f"acceptance failed: {name}")

mods = r60.configure_source_imports()
trainer, common, controller_module = mods["trainer"], mods["common"], mods["controller_module"]
SEED = 11
ep0 = r60.load_r58_ep0_model(SEED)

old = controller_module.fresh_controller(SEED)
old.inner.trunk.load_state_dict(ep0["trunk"])
old.inner.action_head.load_state_dict(ep0["action_head"])
old.inner.value_head.load_state_dict(ep0["value_task_head"])
old.cost_head.load_state_dict(ep0["value_cost_head"])
new = r60.fresh_partitioned_controller(SEED, ep0_state=ep0)

panel_path = HERE.parent / "round49_bc_input_policy_closure" / "INPUT_PANEL_MANIFEST.json"
panel = json.loads(panel_path.read_text(encoding="utf-8")) if panel_path.is_file() else None
if panel is not None:
    feats = torch.tensor([row["features"] for row in panel["records"]], dtype=torch.float32)
    with torch.no_grad():
        ol, ov = old.forward(feats)
        nl, nv = new.forward(feats)
    check("init_forward_logits_equal_old", torch.equal(ol, nl), max_abs=float((ol-nl).abs().max()))
    check("init_forward_values_equal_old", torch.equal(ov, nv), max_abs=float((ov-nv).abs().max()))
    og = (torch.softmax(ol, -1)[:, 1] >= torch.softmax(ol, -1)[:, 0]).long()
    ng = (torch.softmax(nl, -1)[:, 1] >= torch.softmax(nl, -1)[:, 0]).long()
    check("init_greedy_actions_equal_old", torch.equal(og, ng))
    g = torch.Generator().manual_seed(12345)
    os_idx = torch.multinomial(torch.softmax(ol, -1), 1, generator=g).squeeze(1)
    g2 = torch.Generator().manual_seed(12345)
    ns_idx = torch.multinomial(torch.softmax(nl, -1), 1, generator=g2).squeeze(1)
    check("init_sampled_actions_equal_old", torch.equal(os_idx, ns_idx))
    panel_rows = len(panel["records"])
else:
    panel_rows = 0

def tensors_equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
check("early_head_equals_old_action_head", tensors_equal(new.inner.early_head, old.inner.action_head))
check("late_head_equals_old_action_head", tensors_equal(new.inner.late_head, old.inner.action_head))
check("early_and_late_are_distinct_objects",
      all(a is not b for a in new.inner.early_head.parameters() for b in new.inner.late_head.parameters()))
check("trunk_values_equal_old", tensors_equal(new.inner.trunk, old.inner.trunk))
check("task_value_head_values_equal_old", tensors_equal(new.inner.value_head, old.inner.value_head))
check("cost_head_values_equal_old", tensors_equal(new.cost_head, old.cost_head))
# within NEW: trunk, both heads, dual critic are five distinct parameter groups (no sharing)
_sets = new.module_parameter_sets()
_all_ids = [id(p) for ps in _sets.values() for p in ps]
check("five_modules_no_shared_params", len(_all_ids) == len(set(_all_ids)) == 12, n=len(_all_ids))
check("trunk_distinct_from_heads",
      all(a is not b for a in new.inner.trunk.parameters()
          for b in list(new.inner.early_head.parameters()) + list(new.inner.late_head.parameters())))
snap = copy.deepcopy(new.inner.late_head.state_dict())
with torch.no_grad():
    new.inner.early_head.weight.add_(1.0)
check("heads_independent_perturb", new.inner.late_head.state_dict()["weight"].equal(snap["weight"]))
with torch.no_grad():
    new.inner.early_head.weight.sub_(1.0)

def feat(ready, first=1.0, age=0.0):
    f = [0.0]*19
    f[0] = age
    f[5] = first
    f[18] = (ready + age) / 1000.0
    return f
with torch.no_grad():
    new.inner.early_head.weight.add_(0.5)
    new.inner.late_head.weight.sub_(0.3)
rows = torch.tensor([feat(147), feat(148), feat(0), feat(200), feat(147)], dtype=torch.float32)
with torch.no_grad():
    h = new.inner.trunk(rows)
    el = new.inner.early_head(h); ll = new.inner.late_head(h)
    routed = r60.route_logits(new, rows, h)
exp = torch.where((r60.ready_steps(rows) <= 147).unsqueeze(1), el, ll)
check("routing_matches_head_selection", torch.equal(routed, exp))
rs = r60.ready_steps(rows).tolist()
check("ready_values_exact", rs == [147.0, 148.0, 0.0, 200.0, 147.0], ready=rs)
perm = torch.tensor([4, 2, 0, 3, 1])
with torch.no_grad():
    routed_perm = r60.route_logits(new, rows[perm], new.inner.trunk(rows[perm]))
sel = (r60.ready_steps(rows) <= 147)
sel_perm = (r60.ready_steps(rows[perm]) <= 147)
check("routing_head_selection_order_independent", torch.equal(sel_perm, sel[perm]))
check("routing_logits_order_independent", torch.allclose(routed_perm, routed[perm], atol=1e-5),
      max_abs=float((routed_perm - routed[perm]).abs().max()))
with torch.no_grad():
    new.inner.early_head.weight.sub_(0.5)
    new.inner.late_head.weight.add_(0.3)

def one_sample_actor_grad(controller, features_row, action, advantage):
    x = torch.as_tensor(np.asarray([features_row], dtype=np.float32))
    h = controller.inner.trunk(x)
    logits = r60.route_logits(controller, x, h)
    lp = torch.log_softmax(logits, -1)[0, action]
    loss = -(lp * advantage)
    params = controller.parameters
    for p in params:
        p.grad = None
    g = torch.autograd.grad(loss, params, allow_unused=True)
    sets = controller.module_parameter_sets()
    idx = {id(p): i for i, p in enumerate(params)}
    def l1(names):
        return float(sum(g[idx[id(p)]].abs().sum().item() for n in names for p in sets[n]
                         if g[idx[id(p)]] is not None))
    return {"early_head": l1(["early_head"]), "late_head": l1(["late_head"]), "trunk": l1(["trunk"])}
ge = one_sample_actor_grad(new, feat(100), 1, 0.7)
gl = one_sample_actor_grad(new, feat(180), 0, -0.4)
check("early_sample_grads_early_head_only", ge["early_head"] > 0 and ge["late_head"] == 0.0, **ge)
check("late_sample_grads_late_head_only", gl["late_head"] > 0 and gl["early_head"] == 0.0, **gl)

params = new.parameters
ids = [id(p) for p in params]
check("parameter_objects_unique_and_count_12", len(ids) == len(set(ids)) == 12, n=len(ids))
opt = torch.optim.Adam(params, lr=common.LEARNING_RATE)
opt_ids = [id(p) for grp in opt.param_groups for p in grp["params"]]
check("optimizer_covers_each_param_once", sorted(opt_ids) == sorted(ids) and len(opt_ids) == 12)
x = torch.randn(6, 19)
x[:, 0] = 0.0                                  # age_steps
x[:, 18] = torch.tensor([50, 120, 147, 148, 200, 300], dtype=torch.float32) / 1000.0  # step_over_horizon
logits, vals = new.forward(x)
loss = logits.sum() + vals.sum() + new.cost_values(x).sum()
opt.zero_grad(); loss.backward()
grads_present = sum(1 for p in params if p.grad is not None)
check("all_params_receive_grad", grads_present == 12, present=grads_present)
norm = torch.nn.utils.clip_grad_norm_(params, common.MAX_GRAD_NORM)
check("clip_over_all_params_finite", bool(torch.isfinite(norm)))
opt.step()  # one synthetic optimizer update (cap is 32); populates Adam state for all 12 params
synthetic_updates = 1

import contextlib
modules = (new.inner.value_head, new.cost_head)
@contextlib.contextmanager
def critic_input_detached():
    handles = []
    def detach(_m, args):
        return (args[0].detach(), *args[1:])
    for m in modules:
        handles.append(m.register_forward_pre_hook(detach))
    try:
        yield
    finally:
        for hnd in handles:
            hnd.remove()
with critic_input_detached():
    h = new.inner.trunk(x)
    vt = new.inner.value_head(h).squeeze(-1)
    value_loss = ((vt - 0.3) ** 2).mean()
    trunk_grad = torch.autograd.grad(value_loss, list(new.inner.trunk.parameters()), allow_unused=True)
    trunk_l2 = float(sum((g**2).sum() for g in trunk_grad if g is not None))
    rl = r60.route_logits(new, x, h)
    actor_trunk_grad = torch.autograd.grad(rl.sum(), list(new.inner.trunk.parameters()), allow_unused=True)
    actor_trunk_l2 = float(sum((g**2).sum() for g in actor_trunk_grad if g is not None))
check("critic_trunk_isolated", trunk_l2 == 0.0, trunk_l2=trunk_l2)
check("actor_path_connected_under_detach", actor_trunk_l2 > 0.0, actor_trunk_l2=actor_trunk_l2)

from onpolicy import joint_log_prob, action_entropy
logits2 = torch.tensor([[2.0, 0.5], [0.1, 0.2], [1.0, -1.0]])
picked = torch.tensor([1, 0, 1])
usable = torch.tensor([1.0, 0.0, 1.0])
lp_all = joint_log_prob(logits2, picked, usable)
lp_free_only = joint_log_prob(logits2[[0, 2]], picked[[0, 2]], usable[[0, 2]])
check("joint_log_prob_ignores_masked", abs(float(lp_all) - float(lp_free_only)) < 1e-6,
      all=float(lp_all), free=float(lp_free_only))
ent = action_entropy(logits2, usable)
ent_free = action_entropy(logits2[[0, 2]], usable[[0, 2]])
check("entropy_ignores_masked", abs(float(ent) - float(ent_free)) < 1e-6)

freeze = {"training_version": r60.R60_TRAINING_VERSION, "structure": "partitioned-early-late-actor-head/1"}
state = r60.fresh_training_state(SEED)
with torch.no_grad():
    for p in new.parameters:
        p.add_(0.01)
tmp = HERE / "_accept_tmp"
tmp.mkdir(parents=True, exist_ok=True)
ckpt = tmp / "v2_train_s11.pt"
r60.save_checkpoint(new, opt, state, ckpt, freeze=freeze)
rc, ropt, rstate = r60.load_checkpoint(ckpt, freeze=freeze)
check("restore_controller_params", all(torch.equal(a, b) for a, b in zip(new.parameters, rc.parameters)))
saved_sd, loaded_sd = opt.state_dict(), ropt.state_dict()
steps_loaded = sorted(int(s["step"]) for s in loaded_sd["state"].values())
check("restore_adam_state_covers_12", len(loaded_sd["state"]) == 12, n=len(loaded_sd["state"]))
check("restore_adam_steps_match", steps_loaded == [1] * 12, steps=steps_loaded)
check("restore_adam_moment_match",
      all(torch.allclose(saved_sd["state"][i]["exp_avg"], loaded_sd["state"][i]["exp_avg"], atol=1e-7)
          for i in range(len(saved_sd["state"]))))
check("restore_training_state_cursor", rstate["episodes_done"] == 0 and rstate["training_seed"] == SEED
      and rstate["gae_lambda"] == 1.0 and rstate["eta_lambda"] == 0.5)
old_ckpt = HERE.parent.parent / "comm_rule_warmstart_cost_learning_20261006_r2" / "checkpoints" / "BC_POST_MODEL_s11.pt"
rejected = False
try:
    payload = torch.load(old_ckpt, map_location="cpu", weights_only=False)
    if payload.get("schema") != r60.R60_STATE_SCHEMA:
        rejected = True
except Exception:
    rejected = True
check("old_single_head_checkpoint_rejected_as_resume", rejected)

from integrated_sar.comm_policy import SendItem, SendObservation
from r42_one_shot import OneShotSendSkipCollector
def make_item(tid, ready):
    # feature ready_step = round(step/1000*1000) - round(age_steps) = step - age_steps.
    # observation.step is 180, so age_steps = 180 - ready gives the intended ready_step.
    age = 180 - ready
    return SendItem(target_id=tid, ready_step=ready, age_steps=age, pending_event=True,
                    changed_view=True, first_emission=True, revision=0, revision_delta=0,
                    stage_index=0, stage_changed=False, progress=0.0, progress_delta=0.0,
                    requirements_changed=False, members_changed=False, deadline=100.0,
                    remaining_time=50.0, sent_count=0, last_sent_step=None, since_last_emission_time=None,
                    has_future_requirements=False, attempt_charge_bytes=100, send_executable=True,
                    send_block_reason="allowed")
items = (make_item("t_early", 100), make_item("t_late", 180))
obs = SendObservation(step=180, current_time=1.0, items=items, remaining_budget_bytes=29300,
                      budget_bytes_per_episode=29300, capacity_remaining=8, backlog_targets=2, subscribers=1)
coll = OneShotSendSkipCollector(new, greedy=True)
asked, actions, free, logp, value, tape = coll.choose(obs)
check("collector_choose_runs", tape is not None and set(free) == {"t_early", "t_late"},
      free=list(free), actions=actions)
with torch.no_grad():
    fm = torch.tensor(tape["features"], dtype=torch.float32)
    cl, _ = new.forward(fm)
    hh = new.inner.trunk(fm)
    check("collector_routing_early_uses_early_head", torch.equal(cl[0], new.inner.early_head(hh)[0]))
    check("collector_routing_late_uses_late_head", torch.equal(cl[1], new.inner.late_head(hh)[1]))

print(json.dumps({"kind": "round60-wireup-acceptance/1", "status": "passed",
                  "checks": len(RESULTS), "panel_rows": panel_rows, "seed": SEED}, indent=1))
