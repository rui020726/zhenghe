"""ROUND66 wire-up acceptance: prove the frozen early computation path is correctly wired through
the REAL controller, collector, batch/actor, optimizer and checkpoint paths before any training.

Only NEW tests that the ROUND66 structural change can affect are added; unchanged ROUND60/R58
behaviour (routing, joint likelihood, mask semantics, GAE, ALL_ROWS, dual update) is not re-proved
beyond the checks the freeze can break. Pure-synthetic optimizer updates are capped at 32 and write
only to a throwaway test directory; they are never presented as real training. No simulation, no
new environment, no checkpoint outside the test dir, no new evaluation.

What this file deliberately does NOT claim:
  * the frozen tensors receiving no gradient is a consequence of ``requires_grad=False`` and says
    nothing about whether the R53 critic->trunk isolation is correct;
  * a zero reading on the late head for a pure-early minibatch is a routing fact, not a bug;
  * initial forward equivalence with the ROUND60 ep0 model does not imply equal gradient norms,
    equal global clip coefficients or an equal Adam trajectory.
"""
from __future__ import annotations
import copy, json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RUN_ROOT / "round53_critic_isolation_trial_readiness"))
import r66_frozen as r66
from round53_trial_adapter import critic_input_detached

RESULTS: list[dict] = []
def check(name, passed, **detail):
    RESULTS.append({"check": name, "passed": bool(passed), **detail})
    print(("PASS" if passed else "FAIL"), name, "" if passed else json.dumps(detail, default=str))
    if not passed:
        raise SystemExit(f"acceptance failed: {name}")

mods = r66_frozen_mods = r66.configure_source_imports()
trainer, common, controller_module = mods["trainer"], mods["common"], mods["controller_module"]
SEED = 11
SYNTHETIC_UPDATE_CAP = 32
synthetic_updates = 0

# ---------------------------------------------------------------- 1. initial equivalence ----
# NEW is initialised model-only from the compliant ROUND60 ep0. Everything the early path computes
# must be bit-identical to that ROUND60 ep0 model on the frozen 576-row input panel.
ep0_source = r66.load_r60_model(SEED, "ep0")
old, old_identity = r66.load_r60_checkpoint_read_only(SEED, "ep0")
new, new_ep0, new_frozen_identity, new_summary = r66.build_frozen_controller(
    controller_module.fresh_controller(SEED), SEED)

panel_path = HERE.parent / "round49_bc_input_policy_closure" / "INPUT_PANEL_MANIFEST.json"
panel = json.loads(panel_path.read_text(encoding="utf-8"))
feats = torch.tensor([row["features"] for row in panel["records"]], dtype=torch.float32)
with torch.no_grad():
    old_logits, old_values = old.forward(feats)
    new_logits, new_values = new.forward(feats)
    old_cost = old.cost_values(feats)
    new_cost = new.cost_values(feats)
check("init_logits_equal_r60_ep0", torch.equal(old_logits, new_logits),
      max_abs=float((old_logits - new_logits).abs().max()))
check("init_task_values_equal_r60_ep0", torch.equal(old_values, new_values),
      max_abs=float((old_values - new_values).abs().max()))
check("init_cost_values_equal_r60_ep0", torch.equal(old_cost, new_cost),
      max_abs=float((old_cost - new_cost).abs().max()))
old_greedy = (torch.softmax(old_logits, -1)[:, 1] >= torch.softmax(old_logits, -1)[:, 0]).long()
new_greedy = (torch.softmax(new_logits, -1)[:, 1] >= torch.softmax(new_logits, -1)[:, 0]).long()
check("init_greedy_actions_equal_r60_ep0", torch.equal(old_greedy, new_greedy))
g_old = torch.Generator().manual_seed(12345)
g_new = torch.Generator().manual_seed(12345)
old_sampled = torch.multinomial(torch.softmax(old_logits, -1), 1, generator=g_old).squeeze(1)
new_sampled = torch.multinomial(torch.softmax(new_logits, -1), 1, generator=g_new).squeeze(1)
check("init_sampled_actions_equal_r60_ep0", torch.equal(old_sampled, new_sampled))
check("init_parameter_sha_equals_r60_ep0_source",
      new_summary["parameter_sha256"] == ep0_source["parameter_sha256"],
      got=new_summary["parameter_sha256"], expected=ep0_source["parameter_sha256"])

# ------------------------------------------------------- 2. the frozen/trainable partition --
frozen, trainable = r66.partitioned_parameters(new)
all_params = list(new.parameters)
all_ids = [id(p) for p in all_params]
check("parameter_objects_12", len(all_ids) == 12, n=len(all_ids))
check("parameter_objects_unique", len(all_ids) == len(set(all_ids)))
check("frozen_objects_6", len(frozen) == 6, n=len(frozen))
check("trainable_objects_6", len(trainable) == 6, n=len(trainable))
check("partition_is_complete", len(frozen) + len(trainable) == len(all_ids))
check("frozen_scalars_5570", sum(p.numel() for p in frozen) == 5570,
      n=sum(p.numel() for p in frozen))
check("trainable_scalars_260", sum(p.numel() for p in trainable) == 260,
      n=sum(p.numel() for p in trainable))
check("frozen_requires_grad_false", all(not p.requires_grad for p in frozen))
check("trainable_requires_grad_true", all(p.requires_grad for p in trainable))
sets = new.module_parameter_sets()
check("frozen_modules_are_trunk_and_early_head",
      sorted(sets["trunk"] + sets["early_head"], key=id) == sorted(frozen, key=id))
check("trainable_modules_are_late_and_dual_critic",
      sorted(sets["late_head"] + sets["task_value_head"] + sets["cost_head"], key=id)
      == sorted(trainable, key=id))
check("early_and_late_are_distinct_objects",
      all(a is not b for a in new.inner.early_head.parameters()
          for b in new.inner.late_head.parameters()))
# ROUND60's ep0 early_head is itself the exact copy of the OLD R58 action_head (proved in the
# ROUND60 wire-up acceptance), so equality here is equality with the OLD shared head by transitivity.
check("early_head_equals_r60_ep0_early_head",
      all(torch.equal(a, b) for a, b in zip(new.inner.early_head.parameters(),
                                            old.inner.early_head.parameters())))
check("late_head_equals_r60_ep0_late_head",
      all(torch.equal(a, b) for a, b in zip(new.inner.late_head.parameters(),
                                            old.inner.late_head.parameters())))
check("early_and_late_heads_start_equal",
      all(torch.equal(a, b) for a, b in zip(new.inner.early_head.parameters(),
                                            new.inner.late_head.parameters())))
early_snapshot = copy.deepcopy(new.inner.early_head.state_dict())
late_snapshot = copy.deepcopy(new.inner.late_head.state_dict())
with torch.no_grad():
    new.inner.early_head.weight.add_(1.0)
check("heads_independent_perturb",
      new.inner.late_head.state_dict()["weight"].equal(late_snapshot["weight"]))
new.inner.early_head.load_state_dict(early_snapshot)
check("early_head_restored_after_perturb",
      all(torch.equal(a, b) for a, b in zip(new.inner.early_head.parameters(),
                                            old.inner.early_head.parameters())))

# --------------------------------------------------------------------- 3. routing is intact --
def feat(ready, first=1.0, age=0.0):
    f = [0.0] * 19
    f[0] = age
    f[5] = first
    f[18] = (ready + age) / 1000.0
    return f

rows = torch.tensor([feat(147), feat(148), feat(0), feat(200), feat(147)], dtype=torch.float32)
with torch.no_grad():
    hidden = new.inner.trunk(rows)
    early_logits = new.inner.early_head(hidden)
    late_logits = new.inner.late_head(hidden)
    routed = r66.route_logits(new, rows, hidden)
expected = torch.where((r66.ready_steps(rows) <= 147).unsqueeze(1), early_logits, late_logits)
check("routing_matches_head_selection", torch.equal(routed, expected))
check("ready_values_exact", r66.ready_steps(rows).tolist() == [147.0, 148.0, 0.0, 200.0, 147.0],
      ready=r66.ready_steps(rows).tolist())
perm = torch.tensor([4, 2, 0, 3, 1])
with torch.no_grad():
    routed_perm = r66.route_logits(new, rows[perm], new.inner.trunk(rows[perm]))
sel = r66.ready_steps(rows) <= 147
check("routing_head_selection_order_independent",
      torch.equal((r66.ready_steps(rows[perm]) <= 147), sel[perm]))
check("routing_logits_order_independent",
      torch.allclose(routed_perm, routed[perm], atol=1e-6),
      max_abs=float((routed_perm - routed[perm]).abs().max()))

# ------------------------------------------- 4. per-sample gradient: frozen gets nothing -----
def sample_grads(controller, feature_row, action, advantage):
    x = torch.as_tensor(np.asarray([feature_row], dtype=np.float32))
    hidden = controller.inner.trunk(x)
    logits = r66.route_logits(controller, x, hidden)
    values = controller.inner.value_head(hidden).squeeze(-1)
    cost = controller.cost_head(hidden).squeeze(-1)
    loss = (-(torch.log_softmax(logits, -1)[0, action]) * advantage
            + values.sum() + cost.sum())
    for p in controller.parameters:
        p.grad = None
    loss.backward()
    frozen_now, trainable_now = r66.partitioned_parameters(controller)
    def l1(params):
        return float(sum(p.grad.abs().sum().item() for p in params if p.grad is not None))
    def any_grad(params):
        return any(p.grad is not None for p in params)
    return {"frozen_any_grad": any_grad(frozen_now), "frozen_l1": l1(frozen_now),
            "early_head_l1": l1(list(controller.inner.early_head.parameters())),
            "late_head_l1": l1(list(controller.inner.late_head.parameters())),
            "critic_l1": l1(list(controller.inner.value_head.parameters())
                            + list(controller.cost_head.parameters()))}

ge = sample_grads(new, feat(100), 1, 0.7)
check("early_sample_frozen_params_get_no_grad", not ge["frozen_any_grad"], **ge)
check("early_sample_early_head_no_grad", ge["early_head_l1"] == 0.0, **ge)
check("early_sample_late_head_no_grad", ge["early_head_l1"] == 0.0 and ge["late_head_l1"] == 0.0, **ge)
check("early_sample_critic_still_trains", ge["critic_l1"] > 0.0, **ge)
gl = sample_grads(new, feat(180), 0, -0.4)
check("late_sample_frozen_params_get_no_grad", not gl["frozen_any_grad"], **gl)
check("late_sample_late_head_gets_grad", gl["late_head_l1"] > 0.0, **gl)
check("late_sample_early_head_no_grad", gl["early_head_l1"] == 0.0, **gl)
check("late_sample_critic_still_trains", gl["critic_l1"] > 0.0, **gl)
for p in new.parameters:
    p.grad = None

# ------------------------------------------------------- 5. optimizer owns exactly six -------
opt = r66.build_optimizer(new, common.LEARNING_RATE)
opt_ids = [id(p) for group in opt.param_groups for p in group["params"]]
check("optimizer_covers_six_trainable_once", sorted(opt_ids) == sorted(id(p) for p in trainable),
      n=len(opt_ids))
check("optimizer_excludes_frozen", not (set(opt_ids) & set(id(p) for p in frozen)))
check("optimizer_lr_is_frozen_living_value", float(opt.param_groups[0]["lr"]) == float(common.LEARNING_RATE),
      lr=float(opt.param_groups[0]["lr"]))

# ------------------------------- 6. synthetic update: frozen bit-unchanged, trainable moves ---
def identities(controller):
    f, t = r66.partitioned_parameters(controller)
    return (r66._tensor_identity(f)["sha256"], r66._tensor_identity(t)["sha256"])

frozen_before, trainable_before = identities(new)
x = torch.randn(8, 19)
x[:, 0] = 0.0
x[:, 18] = torch.tensor([50, 120, 147, 148, 200, 300, 30, 260], dtype=torch.float32) / 1000.0
logits, values = new.forward(x)
cost_values = new.cost_values(x)
loss = logits.sum() + values.sum() + cost_values.sum()
opt.zero_grad()
loss.backward()
check("synthetic_update_frozen_have_no_grad", all(p.grad is None for p in frozen))
check("synthetic_update_trainable_have_grad", all(p.grad is not None for p in trainable))
norm = torch.nn.utils.clip_grad_norm_(trainable, common.MAX_GRAD_NORM)
check("clip_over_trainable_finite", bool(torch.isfinite(norm)))
opt.step()
synthetic_updates += 1
frozen_after, trainable_after = identities(new)
check("synthetic_update_frozen_bit_unchanged", frozen_before == frozen_after,
      before=frozen_before, after=frozen_after)
check("synthetic_update_trainable_moved", trainable_before != trainable_after)
check("synthetic_update_within_cap", synthetic_updates <= SYNTHETIC_UPDATE_CAP,
      synthetic_updates=synthetic_updates)

# ------------------------------- 7. pure-early minibatch fabricates no late actor gradient ----
pure_early = torch.tensor([feat(10), feat(60), feat(147), feat(1)], dtype=torch.float32)
hidden = new.inner.trunk(pure_early)
early_logits = new.inner.early_head(hidden)
late_logits = new.inner.late_head(hidden)
routed = r66.route_logits(new, pure_early, hidden)
actor_loss = -(torch.log_softmax(routed, -1)[:, 1]).sum()
for p in new.parameters:
    p.grad = None
actor_loss.backward()
late_l1 = float(sum(p.grad.abs().sum().item() for p in new.inner.late_head.parameters()
                    if p.grad is not None))
check("pure_early_minibatch_late_head_grad_is_zero", late_l1 == 0.0, late_head_l1=late_l1)
check("pure_early_minibatch_early_head_has_no_grad_at_all",
      all(p.grad is None for p in new.inner.early_head.parameters()))
check("pure_early_minibatch_frozen_have_no_grad", all(p.grad is None for p in frozen))
for p in new.parameters:
    p.grad = None
# ... and the mirror image, so the routing is not simply always-zero on one branch.
pure_late = torch.tensor([feat(148), feat(300), feat(200)], dtype=torch.float32)
hidden = new.inner.trunk(pure_late)
routed = r66.route_logits(new, pure_late, hidden)
late_loss = -(torch.log_softmax(routed, -1)[:, 1]).sum()
late_loss.backward()
late_l1_late = float(sum(p.grad.abs().sum().item() for p in new.inner.late_head.parameters()
                         if p.grad is not None))
check("pure_late_minibatch_late_head_grad_nonzero", late_l1_late > 0.0, late_head_l1=late_l1_late)
for p in new.parameters:
    p.grad = None

# --------------------------------- 8. the R53 critic->trunk isolation context is still live ---
probe = torch.randn(3, 19, requires_grad=True)
with critic_input_detached(new):
    hidden_probe = new.inner.trunk(probe)
    value_out = new.inner.value_head(hidden_probe).sum()
    cost_out = new.cost_head(hidden_probe).sum()
    actor_out = r66.route_logits(new, probe, hidden_probe).sum()
grad_to_probe_from_critic = torch.autograd.grad(value_out + cost_out, probe, allow_unused=True)[0]
check("critic_input_detached_blocks_gradient_to_trunk_output",
      grad_to_probe_from_critic is None or float(grad_to_probe_from_critic.abs().sum()) == 0.0)
grad_to_probe_from_actor = torch.autograd.grad(actor_out, probe, allow_unused=True)[0]
check("actor_path_connected_through_trunk_output",
      grad_to_probe_from_actor is not None and float(grad_to_probe_from_actor.abs().sum()) > 0.0)
check("critic_hooks_removed_after_context",
      not new.inner.value_head._forward_pre_hooks and not new.cost_head._forward_pre_hooks)
# The actor gradient reaches the trunk OUTPUT but the trunk PARAMETERS stay frozen: that is the
# whole point of the intervention and it is recorded as a parameter-level fact, not a repair.
check("trunk_parameters_still_frozen_after_isolation_probe",
      all(not p.requires_grad for p in new.inner.trunk.parameters()))

# ------------------------------------- 9. joint likelihood / entropy mask semantics unchanged --
from onpolicy import joint_log_prob, action_entropy
logits2 = torch.tensor([[2.0, 0.5], [0.1, 0.2], [1.0, -1.0]])
picked = torch.tensor([1, 0, 1])
usable = torch.tensor([1.0, 0.0, 1.0])
lp_all = joint_log_prob(logits2, picked, usable)
lp_free = joint_log_prob(logits2[[0, 2]], picked[[0, 2]], usable[[0, 2]])
check("joint_log_prob_ignores_masked", abs(float(lp_all) - float(lp_free)) < 1e-6,
      all=float(lp_all), free=float(lp_free))
ent = action_entropy(logits2, usable)
ent_free = action_entropy(logits2[[0, 2]], usable[[0, 2]])
check("entropy_ignores_masked", abs(float(ent) - float(ent_free)) < 1e-6)

# ------------------------- 10. a REAL PPO update over real reloaded ROUND60 episode artefacts --
# Read-only reuse of ROUND60's own saved training rows/tapes: no environment is constructed, no
# new episode is produced, and the update runs on a throwaway controller in a throwaway directory.
def reload_episode(seed, ep):
    row_matches = sorted((r66.R60_DIR / "raw" / "rows").glob(f"v2_r60_train_s{seed}_e{ep}_s*_a1.json"))
    if not row_matches:
        return None
    row_path = row_matches[0]
    scene = int(row_path.name.rsplit("_s", 1)[-1].split("_a1")[0])
    tape_path = r66.R60_DIR / "raw" / "logs" / f"v2_r60_train_s{seed}_e{ep}_s{scene}_a1_tape.jsonl"
    if not tape_path.is_file():
        return None
    row = json.loads(row_path.read_text(encoding="utf-8"))
    tapes = [json.loads(line) for line in tape_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    row["collector"] = SimpleNamespace(records=tapes)
    row["training_seed"] = int(seed)
    row["episode_index"] = int(ep)
    row["scenario_seed"] = scene
    row["lambda"] = 1.0
    return row

episodes = [reload_episode(SEED, ep) for ep in range(4)]
episodes = [e for e in episodes if e is not None]
if len(episodes) == 4:
    ppo_controller, _ep0b, _fid, _sum = r66.build_frozen_controller(
        controller_module.fresh_controller(SEED), SEED)
    ppo_opt = r66.build_optimizer(ppo_controller, common.LEARNING_RATE)
    f_before, t_before = identities(ppo_controller)
    report = r66.frozen_early_ppo_update(ppo_controller, ppo_opt, episodes, update_index=0,
                                         diag_updates=(0,))
    synthetic_updates += 1
    f_after, t_after = identities(ppo_controller)
    check("real_ppo_update_structure_is_r66", report["structure"] == r66.R66_STRUCTURE,
          structure=report["structure"])
    check("real_ppo_update_frozen_bit_unchanged", f_before == f_after)
    check("real_ppo_update_trainable_moved", t_before != t_after)
    check("real_ppo_update_frozen_object_count", report["frozen_parameter_object_count"] == 6)
    check("real_ppo_update_trainable_object_count", report["trainable_parameter_object_count"] == 6)
    check("real_ppo_update_likelihood_alignment_passed",
          bool(report["likelihood_alignment"]["passed"]), alignment=report["likelihood_alignment"])
    means = report["module_gradient_l1_mean"]
    check("real_ppo_update_frozen_modules_recorded_none",
          means.get("trunk") is None and means.get("early_head") is None, means=means)
    check("real_ppo_update_trainable_modules_recorded_numbers",
          isinstance(means.get("late_head"), float) and isinstance(means.get("cost_head"), float),
          means=means)
    diag = report.get("gate_diag") or {}
    check("real_ppo_update_gate_diag_present", bool(diag), diag=diag)
    check("real_ppo_update_gate_diag_does_not_claim_frozen_zero",
          diag.get("trunk") == "frozen_requires_grad_false"
          and diag.get("early_head") == "frozen_requires_grad_false", diag=diag)
    check("real_ppo_update_gate_diag_reports_graph_existence",
          "base_output_requires_grad" in diag and "prior_output_requires_grad" in diag, diag=diag)
    if diag.get("prior_output_requires_grad"):
        check("real_ppo_update_prior_has_no_trainable_path",
              diag.get("prior_trainable_grad_l1") == 0.0, diag=diag)
    else:
        check("real_ppo_update_prior_absent_is_labelled",
              diag.get("prior_gradient") == "absent_no_eligible_gate_rows", diag=diag)
    check("real_ppo_update_late_actor_grad_recorded",
          "late_actor_grad_l1" in diag and "critic_grad_l1" in diag, diag=diag)
    check("real_ppo_update_within_cap", synthetic_updates <= SYNTHETIC_UPDATE_CAP,
          synthetic_updates=synthetic_updates)
else:
    check("real_r60_episode_artefacts_available", False, found=len(episodes))

# ------------------------------------------ 11. checkpoint save / restore / rejection --------
state = r66.fresh_training_state(SEED)
state["episodes_done"] = 4
state["updates"] = 1
state["optimizer_steps"] = int(report["optimizer_steps"]) if len(episodes) == 4 else 0
tmp = HERE / "_acceptance_scratch"
tmp.mkdir(parents=True, exist_ok=True)
ckpt = tmp / "v2_train_s11.pt"
r66.save_checkpoint(new, opt, state, ckpt, freeze={"kind": "round66-acceptance-freeze/1"})
rc, ropt, rstate = r66.load_checkpoint(ckpt, freeze={"kind": "round66-acceptance-freeze/1"})
check("restore_controller_params", all(torch.equal(a, b) for a, b in zip(new.parameters, rc.parameters)))
loaded = ropt.state_dict()
check("restore_adam_state_covers_six", len(loaded["state"]) == 6, n=len(loaded["state"]))
check("restore_adam_param_groups_six",
      [len(g["params"]) for g in loaded["param_groups"]] == [6])
check("restore_adam_steps_match",
      sorted(int(s["step"]) for s in loaded["state"].values()) == [1] * 6,
      steps=sorted(int(s["step"]) for s in loaded["state"].values()))
check("restore_training_state_cursor",
      rstate["episodes_done"] == 4 and rstate["training_seed"] == SEED
      and rstate["gae_lambda"] == 1.0 and rstate["eta_lambda"] == 0.5
      and rstate["loss_version"] == r66.R66_LOSS_VERSION
      and rstate["optimizer_schema"] == r66.R66_OPTIMIZER_SCHEMA)
check("restore_reestablishes_frozen_mask",
      all(not p.requires_grad for p in r66.frozen_parameters(rc)))
check("restore_keeps_trainable_trainable",
      all(p.requires_grad for p in r66.trainable_parameters(rc)))
check("restore_frozen_sha_matches", r66._tensor_identity(r66.frozen_parameters(rc))["sha256"]
      == r66._tensor_identity(frozen)["sha256"])
ro_controller, ro_state, ro_identity = r66.load_checkpoint_read_only(
    ckpt, freeze={"kind": "round66-acceptance-freeze/1"})
check("read_only_load_restores_no_optimizer_or_rng",
      ro_identity["optimizer_restored"] is False and ro_identity["rng_restored"] is False)
check("read_only_load_identity_has_frozen_mask", bool(ro_identity.get("frozen_mask")))

for seed, role in ((SEED, "ep0"), (SEED, "ep128"), (22, "ep0")):
    rejected = False
    try:
        r66.load_checkpoint(r66.r60_model_path(seed, role), freeze={"kind": "x"})
    except RuntimeError as error:
        rejected = "ROUND60" in str(error)
    except Exception:
        rejected = True
    check(f"r60_{role}_checkpoint_rejected_as_r66_resume", rejected)

quarantine_like = (r66.R60_DIR / "_quarantine_attempt1_criticdetect_missing" / "raw"
                   / "checkpoints" / "v2_train_s11_before_b0.pt")
rejected = False
try:
    r66._require_compliant_r60_path(quarantine_like)
except RuntimeError:
    rejected = True
check("quarantined_r60_source_rejected", rejected)

saved = r66.R60_EXPECTED_IDENTITY[SEED]["ep0"]["file_sha256"]
r66.R60_EXPECTED_IDENTITY[SEED]["ep0"]["file_sha256"] = "0" * 64
rejected = False
try:
    r66.load_r60_model(SEED, "ep0")
except RuntimeError:
    rejected = True
finally:
    r66.R60_EXPECTED_IDENTITY[SEED]["ep0"]["file_sha256"] = saved
check("r60_source_byte_drift_rejected", rejected)

saved_cursor = r66.R60_EXPECTED_IDENTITY[SEED]["ep0"]["episodes_done"]
r66.R60_EXPECTED_IDENTITY[SEED]["ep0"]["episodes_done"] = 128
rejected = False
try:
    r66.load_r60_model(SEED, "ep0")
except RuntimeError:
    rejected = True
finally:
    r66.R60_EXPECTED_IDENTITY[SEED]["ep0"]["episodes_done"] = saved_cursor
check("r60_source_cursor_mismatch_rejected", rejected)

# ---------------------------------------------------- 12. collector still routes correctly ---
from integrated_sar.comm_policy import SendItem, SendObservation
from r42_one_shot import OneShotSendSkipCollector

def make_item(tid, ready):
    age = 180 - ready
    return SendItem(target_id=tid, ready_step=ready, age_steps=age, pending_event=True,
                    changed_view=True, first_emission=True, revision=0, revision_delta=0,
                    stage_index=0, stage_changed=False, progress=0.0, progress_delta=0.0,
                    requirements_changed=False, members_changed=False, deadline=100.0,
                    remaining_time=50.0, sent_count=0, last_sent_step=None,
                    since_last_emission_time=None, has_future_requirements=False,
                    attempt_charge_bytes=100, send_executable=True, send_block_reason="allowed")

items = (make_item("t_early", 100), make_item("t_late", 180))
obs = SendObservation(step=180, current_time=1.0, items=items, remaining_budget_bytes=29300,
                      budget_bytes_per_episode=29300, capacity_remaining=8, backlog_targets=2,
                      subscribers=1)
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

print(json.dumps({"kind": "round66-wireup-acceptance/1", "status": "passed",
                  "checks": len(RESULTS), "panel_rows": len(panel["records"]), "seed": SEED,
                  "synthetic_optimizer_updates": synthetic_updates,
                  "synthetic_update_cap": SYNTHETIC_UPDATE_CAP,
                  "frozen_scalar_count": new_summary["frozen_scalar_count"],
                  "trainable_scalar_count": new_summary["trainable_scalar_count"],
                  "r60_ep0_file_sha256": ep0_source["file_sha256"],
                  "r60_ep0_parameter_sha256": ep0_source["parameter_sha256"]}, indent=1))
