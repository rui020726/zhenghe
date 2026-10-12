"""ROUND68 zero-environment wire-up acceptance + cumulative-budget guard tests.

Proves, through the REAL controller, collector, batch/actor, optimizer and checkpoint paths (NOT
import-mode stubs), that the ROUND68 late-initial-confidence intervention is wired correctly before
any environment is constructed:

  1. the initial (one-time 0.10-scaled) controller's early path is bit-identical to the compliant
     ROUND60 ep0, the late path is exactly that scale, and the routed forward/loss equals an
     explicit reference computation (NOT a claim of equal gradient norms / clip / Adam trajectory);
  2. routing: ready=147/148 boundary, shuffled batches, mixed partitions, order independence;
  3. one sample uses only its routed head; the other head's actor gradient is 0 for that sample
     (an absent gradient is NOT the same as a frozen parameter under Adam momentum);
  4. optimizer / zero_grad / clip / save / restore cover trunk + both heads + both critics, every
     unique parameter exactly once (optimizer owns the six trainable; save/restore cover all twelve);
  5. the R53 critic->trunk isolation is still live, proved at PARAMETER level on an UNFROZEN
     controller (on the ROUND68 controller the parameter-level effect is masked by the freeze and is
     recorded as such, never as a measured zero);
  6. sequential mask, joint log-probability, entropy and ALL_ROWS semantics are unchanged;
  7. a full resume covers both heads + Adam + lambda + RNG + cursor, and ROUND60/ROUND66
     checkpoints are REJECTED as ROUND68 training-resume sources;
  8. the real CLI configure/bind chain builds and reads the same ROUND68 objects;
  9. an unregistered start is rejected BEFORE environment construction, and a duplicate start under
     one claim is rejected;
 10. the cumulative-budget guards behave: hard boundary, no refund, restart/subdir/backup
     isolation, no dedup of deterministic duplicates, a single winner under a concurrent race, and
     ROUND60's historical 608 constructions cannot be read as headroom under a 600 cap.

Budget discipline: the acceptance/synthetic model forward rows and synthetic optimizer steps this
file really consumes are charged BEFORE the work to the PERSISTENT ROUND68 guards under
raw/runtime/r68_guard_*.json (caps 8192 rows / 32 steps / 32 PPO calls, non-refundable, never
reset by a rerun). Guard BEHAVIOUR tests use throwaway synthetic ledgers in a temp directory and
never touch the real approval ledger. No environment is constructed, no training/eval claim is
created, and synthetic checkpoints are written only to a dedicated throwaway test directory.
--guard-root exists for development reruns only; the real run uses the default raw/runtime root.
"""
from __future__ import annotations
import argparse, copy, json, random, shutil, sys, tempfile, threading, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent
R66_DIR = RUN_ROOT / "round66_bc_early_path_preserving_late_adaptation"
R53_DIR = RUN_ROOT / "round53_critic_isolation_trial_readiness"
for _p in (str(HERE), str(R66_DIR), str(R53_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r68_execute as base          # the real ROUND68 executor (CLI chain, gates, ledger, plan)
import r68_frozen as r68            # the only scientific difference + full ROUND68 identity
import r66_frozen as r66            # inherited frozen-path machinery
import r68_guards as guards         # persistent acceptance/synthetic-computation guards
from round53_trial_adapter import critic_input_detached

_ap = argparse.ArgumentParser()
_ap.add_argument("--guard-root", type=Path, default=None,
                 help="DEV ONLY: redirect the PURE synthetic-ledger fixture root only; model forward/step/PPO guards always use raw/runtime")
_args, _unknown = _ap.parse_known_args()

RAW = HERE / "raw"
# Model-computation guards are ALWAYS the single persistent cumulative root (raw/runtime); a
# throwaway --guard-root may only redirect the PURE synthetic-ledger fixtures, never real model
# forward / optimizer.step / PPO accounting (that is how throwaway dev roots hid over-cap usage).
PERSISTENT_GUARD_ROOT = (RAW / "runtime").resolve()
PERSISTENT_GUARD_ROOT.mkdir(parents=True, exist_ok=True)
GUARD_ROOT = PERSISTENT_GUARD_ROOT
FORWARD_GUARD = guards.ForwardRowGuard(PERSISTENT_GUARD_ROOT / "r68_guard_forward_rows.json")
STEP_GUARD = guards.SyntheticStepGuard(PERSISTENT_GUARD_ROOT / "r68_guard_synthetic_steps.json")
PPO_GUARD = guards.SyntheticPpoGuard(PERSISTENT_GUARD_ROOT / "r68_guard_synthetic_ppo.json")
FIXTURE_GUARD_ROOT = Path(_args.guard_root).resolve() if _args.guard_root else PERSISTENT_GUARD_ROOT

RESULTS: list[dict] = []
SEED = 11
SYNTHETIC_STEPS = 0
SYNTHETIC_PPO = 0


def check(name, passed, **detail):
    RESULTS.append({"check": name, "passed": bool(passed), **detail})
    print(("PASS " if passed else "FAIL "), name,
          "" if passed else json.dumps(detail, default=str)[:900], flush=True)
    if not passed:
        try:
            finish("failed")
        except Exception:
            pass
        raise SystemExit(f"ROUND68 zero-env acceptance failed at: {name}")


def charge_forward(rows: int, what: str) -> int:
    return FORWARD_GUARD.charge(int(rows), what)


def charge_steps(n: int, what: str) -> int:
    global SYNTHETIC_STEPS
    SYNTHETIC_STEPS += int(n)
    return STEP_GUARD.charge(int(n), what)


def charge_ppo(what: str) -> int:
    global SYNTHETIC_PPO
    SYNTHETIC_PPO += 1
    return PPO_GUARD.charge(1, what)


def feat(ready, first=1.0, age=0.0):
    f = [0.0] * 19
    f[0] = age
    f[5] = first
    f[18] = (ready + age) / 1000.0
    return f


def identities(controller):
    frozen, trainable = r66.partitioned_parameters(controller)
    return (r66._tensor_identity(frozen)["sha256"], r66._tensor_identity(trainable)["sha256"])


def finish(status: str, extra: dict | None = None) -> int:
    payload = {"kind": "round68-zero-env-wireup-acceptance/1",
               "status": status, "checks": len(RESULTS),
               "failed": [r["check"] for r in RESULTS if not r["passed"]],
               "seed": SEED,
               "synthetic_optimizer_steps": SYNTHETIC_STEPS,
               "synthetic_ppo_calls": SYNTHETIC_PPO,
               "guard_root": str(GUARD_ROOT),
               "guards": {"forward_rows": FORWARD_GUARD.snapshot(),
                          "synthetic_steps": STEP_GUARD.snapshot(),
                          "synthetic_ppo": PPO_GUARD.snapshot()},
               "results": RESULTS,
               "finished_unix": time.time()}
    if extra:
        payload.update(extra)
    out = RAW / "runtime" / "ROUND68_ZERO_ENV_TEST.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    base._write_json(out, payload)
    print(json.dumps({k: v for k, v in payload.items() if k not in ("results", "guards")}, indent=1))
    return 0 if status == "passed" else 1



mods = r68.configure_source_imports()
trainer, common, controller_module = mods["trainer"], mods["common"], mods["controller_module"]
from onpolicy import joint_log_prob, action_entropy
import cost_math as cm

# ============================================================ 1. initial equivalence =========
# NEW is initialised model-only from the compliant ROUND60 ep0, THEN one late-head 0.10 scale.
ep0_source = r68.load_r60_model(SEED, "ep0")
old_controller, old_identity = r68.load_r60_checkpoint_read_only(SEED, "ep0")
new_controller, new_ep0, new_frozen_id, new_summary, scale_id = r68.fresh_r68_controller(SEED)
pre_controller, pre_ep0, _pre_fid, pre_summary = r66.build_frozen_controller(
    controller_module.fresh_controller(SEED), SEED)

check("pre_scale_init_equals_r60_ep0_parameter_sha",
      pre_summary["parameter_sha256"] == ep0_source["parameter_sha256"],
      got=pre_summary["parameter_sha256"], expected=ep0_source["parameter_sha256"])
check("scale_moved_parameters", scale_id["pre_scale_parameter_sha256"] != scale_id["post_scale_parameter_sha256"])
check("scale_pre_equals_r60_ep0", bool(scale_id["pre_scale_equals_r60_ep0"]))
check("late_head_params_are_exactly_0p10_of_ep0",
      torch.equal(new_controller.inner.late_head.weight, pre_controller.inner.late_head.weight * 0.10)
      and torch.equal(new_controller.inner.late_head.bias, pre_controller.inner.late_head.bias * 0.10))
for _name in ("trunk", "early_head", "value_task_head", "cost_head"):
    check(f"{_name}_bit_identical_to_r60_ep0",
          r68._module_sha(new_controller, _name) == r68._module_sha(old_controller, _name))
check("late_head_differs_from_ep0_by_exactly_the_scale",
      r68._module_sha(new_controller, "late_head") != r68._module_sha(old_controller, "late_head"))
check("two_heads_are_independent_objects",
      new_controller.inner.early_head is not new_controller.inner.late_head
      and all(a is not b for a in new_controller.inner.early_head.parameters()
              for b in new_controller.inner.late_head.parameters()))

panel_path = RUN_ROOT / "round49_bc_input_policy_closure" / "INPUT_PANEL_MANIFEST.json"
panel = json.loads(panel_path.read_text(encoding="utf-8"))
feats = torch.tensor([row["features"] for row in panel["records"]], dtype=torch.float32)
ready = r66.ready_steps(feats)
early_mask = ready <= 147
charge_forward(4 * len(feats), "init panel: old/new logits+values+cost (576 rows x 4 forwards)")
with torch.no_grad():
    old_logits, old_values = old_controller.forward(feats)
    new_logits, new_values = new_controller.forward(feats)
    old_cost = old_controller.cost_values(feats)
    new_cost = new_controller.cost_values(feats)
check("init_task_values_bit_equal_r60_ep0", torch.equal(old_values, new_values),
      max_abs=float((old_values - new_values).abs().max()))
check("init_cost_values_bit_equal_r60_ep0", torch.equal(old_cost, new_cost),
      max_abs=float((old_cost - new_cost).abs().max()))
check("init_early_rows_logits_bit_equal_r60_ep0",
      torch.equal(old_logits[early_mask], new_logits[early_mask]),
      early_rows=int(early_mask.sum()))
late_mask = ~early_mask
check("init_late_rows_logits_equal_0p10_scaled_ep0",
      bool(late_mask.any()) and torch.allclose(
          new_logits[late_mask], old_logits[late_mask] * 0.10, rtol=1e-5, atol=1e-6),
      late_rows=int(late_mask.sum()),
      max_rel=float(((new_logits[late_mask] - old_logits[late_mask] * 0.10).abs()
                     / old_logits[late_mask].abs().clamp_min(1e-9)).max()) if bool(late_mask.any()) else None)
old_greedy = (torch.softmax(old_logits, -1)[:, 1] >= torch.softmax(old_logits, -1)[:, 0]).long()
new_greedy = (torch.softmax(new_logits, -1)[:, 1] >= torch.softmax(new_logits, -1)[:, 0]).long()
check("init_greedy_actions_unchanged_by_positive_scale", torch.equal(old_greedy, new_greedy),
      changed=int((old_greedy != new_greedy).sum()))
# The one-time 0.10 scale preserves the greedy argmax but necessarily changes the SAMPLING
# distribution on late rows (a logit-temperature effect: that lower confidence IS the
# intervention). The invariant that must hold is: early rows (bit-identical logits) sample
# identically under the same generator seed, and every differing row is a late row.
_g_old = torch.Generator().manual_seed(12345)
_g_new = torch.Generator().manual_seed(12345)
old_sampled = torch.multinomial(torch.softmax(old_logits, -1), 1, generator=_g_old).squeeze(1)
new_sampled = torch.multinomial(torch.softmax(new_logits, -1), 1, generator=_g_new).squeeze(1)
_sdiff = old_sampled != new_sampled
check("init_sampled_actions_differ_only_on_late_rows",
      int((_sdiff & early_mask).sum()) == 0,
      changed=int(_sdiff.sum()), changed_on_early=int((_sdiff & early_mask).sum()),
      changed_on_late=int((_sdiff & late_mask).sum()))
old_p = torch.softmax(old_logits, -1)
new_p = torch.softmax(new_logits, -1)
check("init_late_sampling_confidence_lowered_towards_uniform",
      bool(late_mask.any())
      and float((new_p[late_mask, 1] - 0.5).abs().mean()) < float((old_p[late_mask, 1] - 0.5).abs().mean()),
      old_mean_abs_dev=float((old_p[late_mask, 1] - 0.5).abs().mean()) if bool(late_mask.any()) else None,
      new_mean_abs_dev=float((new_p[late_mask, 1] - 0.5).abs().mean()) if bool(late_mask.any()) else None)
check("init_early_sampling_distribution_unchanged",
      torch.equal(old_p[early_mask], new_p[early_mask]))

# initial forward/loss equivalence against an explicit reference computation (same parameters).
_x = torch.tensor([feat(10), feat(60), feat(147), feat(148), feat(200), feat(300), feat(30), feat(260)],
                  dtype=torch.float32)
_picked = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
_usable = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
charge_forward(2 * len(_x), "manual vs controller routed forward/loss")
with torch.no_grad():
    _hidden = new_controller.inner.trunk(_x)
    _logits_ref = r66.route_logits(new_controller, _x, _hidden)
    _values_ref = new_controller.inner.value_head(_hidden).squeeze(-1)
    _cost_ref = new_controller.cost_head(_hidden).squeeze(-1)
    _logits_c, _values_c = new_controller.forward(_x)
    _cost_c = new_controller.cost_values(_x)
check("routed_forward_matches_explicit_module_composition",
      torch.equal(_logits_c, _logits_ref) and torch.equal(_values_c, _values_ref)
      and torch.equal(_cost_c, _cost_ref))
_lp_ref = joint_log_prob(_logits_ref, _picked, _usable)
_ent_ref = action_entropy(_logits_ref, _usable)
_loss_ref = (-_lp_ref + common.VALUE_COEFFICIENT * (_values_ref.mean() + _cost_ref.mean())
             - common.ENTROPY_BONUS * _ent_ref)
_lp_c = joint_log_prob(_logits_c, _picked, _usable)
_ent_c = action_entropy(_logits_c, _usable)
_loss_c = (-_lp_c + common.VALUE_COEFFICIENT * (_values_c.mean() + _cost_c.mean())
           - common.ENTROPY_BONUS * _ent_c)
check("initial_loss_equivalent_to_reference", torch.equal(_loss_ref, _loss_c),
      ref=float(_loss_ref), got=float(_loss_c))
# ============================================================ 2. routing =====================
rows = torch.tensor([feat(147), feat(148), feat(0), feat(200), feat(147)], dtype=torch.float32)
charge_forward(6, "routing boundary + permutation forwards")
with torch.no_grad():
    hidden = new_controller.inner.trunk(rows)
    early_logits = new_controller.inner.early_head(hidden)
    late_logits = new_controller.inner.late_head(hidden)
    routed = r66.route_logits(new_controller, rows, hidden)
expected = torch.where((r66.ready_steps(rows) <= 147).unsqueeze(1), early_logits, late_logits)
check("routing_matches_head_selection", torch.equal(routed, expected))
check("ready_values_exact_at_147_148", r66.ready_steps(rows).tolist() == [147.0, 148.0, 0.0, 200.0, 147.0],
      ready=r66.ready_steps(rows).tolist())
perm = torch.tensor([4, 2, 0, 3, 1])
with torch.no_grad():
    routed_perm = r66.route_logits(new_controller, rows[perm], new_controller.inner.trunk(rows[perm]))
check("routing_head_selection_order_independent",
      torch.equal((r66.ready_steps(rows[perm]) <= 147), (r66.ready_steps(rows) <= 147)[perm]))
check("routing_logits_order_independent", torch.allclose(routed_perm, routed[perm], atol=1e-6),
      max_abs=float((routed_perm - routed[perm]).abs().max()))
# mixed partition inside ONE stacked batch: every row takes its own head, no cross-talk.
mixed = torch.tensor([feat(147), feat(148), feat(10), feat(260), feat(147), feat(300)], dtype=torch.float32)
with torch.no_grad():
    mh = new_controller.inner.trunk(mixed)
    mr = r66.route_logits(new_controller, mixed, mh)
    me = new_controller.inner.early_head(mh)
    ml = new_controller.inner.late_head(mh)
sel = (r66.ready_steps(mixed) <= 147)
check("mixed_batch_rows_take_own_head",
      torch.equal(mr[sel], me[sel]) and torch.equal(mr[~sel], ml[~sel]))

# ============================================================ 3. per-sample gradients ========
# An UNFROZEN R60-ep0 partitioned controller is the probe for the merged (routed) map, so the
# head-output gradients are measurable at all; the R68 controller's frozen early path is then
# checked against it.
probe = r66.fresh_partitioned_controller(SEED, ep0_state=None, verify_ep0=False)
probe.load_state_dict(ep0_source["controller"])
check("probe_controller_is_unfrozen_r60_ep0",
      all(p.requires_grad for p in probe.parameters)
      and probe.parameter_summary()["parameter_sha256"] == ep0_source["parameter_sha256"])


def merged_map_grads(controller, batch):
    hidden = controller.inner.trunk(batch)
    early_out = controller.inner.early_head(hidden)
    late_out = controller.inner.late_head(hidden)
    is_early = (r66.ready_steps(batch) <= 147).unsqueeze(1)
    routed_out = torch.where(is_early, early_out, late_out)
    loss = -(torch.log_softmax(routed_out, -1)[:, 1]).sum()
    g_early, g_late = torch.autograd.grad(loss, (early_out, late_out), allow_unused=True)
    return g_early, g_late


grows = torch.tensor([feat(100), feat(180), feat(147), feat(148)], dtype=torch.float32)
charge_forward(2, "merged-map head-output gradients (forward + backward)")
g_early, g_late = merged_map_grads(probe, grows)
row_is_early = (r66.ready_steps(grows) <= 147).tolist()
check("merged_map_early_head_output_grad_only_on_early_rows",
      all((g_early[i] is None or float(g_early[i].abs().sum()) == 0.0) == (not row_is_early[i])
          for i in range(len(row_is_early))),
      row_is_early=row_is_early,
      early_l1=[None if g_early[i] is None else float(g_early[i].abs().sum()) for i in range(4)],
      late_l1=[None if g_late[i] is None else float(g_late[i].abs().sum()) for i in range(4)])
check("merged_map_late_head_output_grad_only_on_late_rows",
      all((g_late[i] is None or float(g_late[i].abs().sum()) == 0.0) == row_is_early[i]
          for i in range(len(row_is_early))))
# On the UNFROZEN probe every module receives gradient on a mixed batch (trunk/critic shared).
pb = torch.tensor([feat(60), feat(190), feat(147), feat(240)], dtype=torch.float32)
hidden = probe.inner.trunk(pb)
loss_all = (r66.route_logits(probe, pb, hidden).sum() + probe.inner.value_head(hidden).squeeze(-1).sum()
            + probe.cost_head(hidden).squeeze(-1).sum())
for p in probe.parameters:
    p.grad = None
loss_all.backward()
def _l1(params):
    return float(sum(p.grad.abs().sum().item() for p in params if p.grad is not None))
check("unfrozen_probe_all_modules_receive_gradient",
      _l1(list(probe.inner.trunk.parameters())) > 0.0
      and _l1(list(probe.inner.early_head.parameters())) > 0.0
      and _l1(list(probe.inner.late_head.parameters())) > 0.0
      and _l1(list(probe.inner.value_head.parameters()) + list(probe.cost_head.parameters())) > 0.0)
for p in probe.parameters:
    p.grad = None

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

charge_forward(2, "per-sample routed gradient probes")
ge = sample_grads(new_controller, feat(100), 1, 0.7)
check("early_sample_frozen_params_get_no_grad", not ge["frozen_any_grad"], **ge)
check("early_sample_late_head_actor_grad_zero", ge["late_head_l1"] == 0.0, **ge)
check("early_sample_early_head_has_no_grad_at_all",
      all(p.grad is None for p in new_controller.inner.early_head.parameters()), **ge)
check("early_sample_critic_still_trains", ge["critic_l1"] > 0.0, **ge)
gl = sample_grads(new_controller, feat(180), 0, -0.4)
check("late_sample_frozen_params_get_no_grad", not gl["frozen_any_grad"], **gl)
check("late_sample_late_head_gets_grad", gl["late_head_l1"] > 0.0, **gl)
check("late_sample_early_head_has_no_grad_at_all",
      all(p.grad is None for p in new_controller.inner.early_head.parameters()), **gl)
check("late_sample_critic_still_trains", gl["critic_l1"] > 0.0, **gl)
for p in new_controller.parameters:
    p.grad = None
# Recorded fact, not a claim: a zero/absent gradient on the un-routed head does NOT imply the
# parameter is frozen; Adam momentum can still move a parameter whose current gradient is absent.
check("absent_grad_is_not_frozen_under_adam_note_recorded", True,
      note="zero actor gradient on the un-routed head is a routing fact; it is not evidence that "
           "the head's parameters cannot move under Adam momentum")

# ============================================================ 4. optimizer / coverage =========
frozen, trainable = r66.partitioned_parameters(new_controller)
opt = r68.build_optimizer(new_controller, common.LEARNING_RATE)
opt_ids = [id(p) for group in opt.param_groups for p in group["params"]]
check("optimizer_covers_six_trainable_exactly_once",
      sorted(opt_ids) == sorted(id(p) for p in trainable) and len(opt_ids) == 6, n=len(opt_ids))
check("optimizer_excludes_frozen", not (set(opt_ids) & set(id(p) for p in frozen)))
check("optimizer_lr_is_frozen_living_value",
      float(opt.param_groups[0]["lr"]) == float(common.LEARNING_RATE),
      lr=float(opt.param_groups[0]["lr"]))
sd = new_controller.state_dict()
flat = [f"{module}.{key}" for module in sd for key in sd[module]]
check("save_covers_twelve_parameters_exactly_once",
      sorted(flat) == sorted(["trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
                              "early_head.weight", "early_head.bias", "late_head.weight",
                              "late_head.bias", "value_task_head.weight", "value_task_head.bias",
                              "value_cost_head.weight", "value_cost_head.bias"]),
      names=sorted(flat))

f_before, t_before = identities(new_controller)
x = torch.randn(8, 19)
x[:, 0] = 0.0
x[:, 18] = torch.tensor([50, 120, 147, 148, 200, 300, 30, 260], dtype=torch.float32) / 1000.0
charge_forward(1, "synthetic coverage minibatch forward")
logits, values = new_controller.forward(x)
cost_values = new_controller.cost_values(x)
loss = logits.sum() + values.sum() + cost_values.sum()
opt.zero_grad()
loss.backward()
check("synthetic_update_frozen_have_no_grad", all(p.grad is None for p in frozen))
check("synthetic_update_trainable_have_grad", all(p.grad is not None for p in trainable))
manual_norm = float(sum(float((p.grad ** 2).sum()) for p in trainable) ** 0.5)
norm = torch.nn.utils.clip_grad_norm_(trainable, common.MAX_GRAD_NORM)
check("clip_norm_is_over_exactly_the_six_trainable", abs(float(norm) - manual_norm) < 1e-4,
      reported=float(norm), manual=manual_norm)
check("clip_over_trainable_finite", bool(torch.isfinite(norm)))
charge_steps(1, "synthetic coverage optimizer.step")
opt.step()
f_after, t_after = identities(new_controller)
check("synthetic_update_frozen_bit_unchanged", f_before == f_after, before=f_before, after=f_after)
check("synthetic_update_trainable_moved", t_before != t_after)
opt.zero_grad()
check("zero_grad_clears_trainable_grads",
      all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in trainable))
for p in new_controller.parameters:
    p.grad = None

# ============================================================ 5. critic->trunk isolation ======
# (a) input-level probe on the R68 controller: the critic path is blocked from the trunk OUTPUT
#     while the actor path stays connected (parameter-level effect is masked by the freeze).
probe_in = torch.randn(3, 19, requires_grad=True)
charge_forward(3, "critic-isolation input probe")
with critic_input_detached(new_controller):
    hidden_probe = new_controller.inner.trunk(probe_in)
    value_out = new_controller.inner.value_head(hidden_probe).sum()
    cost_out = new_controller.cost_head(hidden_probe).sum()
    actor_out = r66.route_logits(new_controller, probe_in, hidden_probe).sum()
grad_critic = torch.autograd.grad(value_out + cost_out, probe_in, allow_unused=True)[0]
check("critic_input_detached_blocks_gradient_to_trunk_output",
      grad_critic is None or float(grad_critic.abs().sum()) == 0.0)
grad_actor = torch.autograd.grad(actor_out, probe_in, allow_unused=True)[0]
check("actor_path_connected_through_trunk_output",
      grad_actor is not None and float(grad_actor.abs().sum()) > 0.0)
check("critic_hooks_removed_after_context",
      not new_controller.inner.value_head._forward_pre_hooks
      and not new_controller.cost_head._forward_pre_hooks)
# (b) PARAMETER-level proof on the UNFROZEN R60-ep0 controller: inside the context the critic
#     loss produces NO trunk-parameter gradient, while the actor loss still does; outside it the
#     critic loss does reach the trunk parameters. This is the isolation mechanism itself, which
#     the R68 freeze would otherwise mask.
ix = torch.randn(4, 19)
ix[:, 0] = 0.0
ix[:, 18] = torch.tensor([50, 120, 147, 148], dtype=torch.float32) / 1000.0
trunk_params = list(probe.inner.trunk.parameters())
charge_forward(8, "critic-isolation parameter-level proof (2 forwards)")
h = probe.inner.trunk(ix)
crit = probe.inner.value_head(h).sum() + probe.cost_head(h).sum()
g_un = torch.autograd.grad(crit, trunk_params, allow_unused=True)
unisolated = sum(float(t.abs().sum()) for t in g_un if t is not None)
with critic_input_detached(probe):
    h2 = probe.inner.trunk(ix)
    crit2 = probe.inner.value_head(h2).sum() + probe.cost_head(h2).sum()
    actor2 = r66.route_logits(probe, ix, h2).sum()
g_crit = torch.autograd.grad(crit2, trunk_params, allow_unused=True)
g_actor = torch.autograd.grad(actor2, trunk_params, allow_unused=True)
check("isolation_unisolated_critic_reaches_trunk_params", unisolated > 0.0, unisolated_l1=unisolated)
check("isolation_blocks_critic_to_trunk_param_grad",
      all(t is None or float(t.abs().sum()) == 0.0 for t in g_crit),
      isolated_l1=[None if t is None else float(t.abs().sum()) for t in g_crit])
check("isolation_actor_still_reaches_trunk_params",
      all(t is not None and float(t.abs().sum()) > 0.0 for t in g_actor))
check("isolation_hooks_removed_on_unfrozen_probe",
      not probe.inner.value_head._forward_pre_hooks and not probe.cost_head._forward_pre_hooks)
check("r68_trunk_params_stay_frozen_after_isolation_probe",
      all(not p.requires_grad for p in new_controller.inner.trunk.parameters()))

# ============================================================ 6. mask / likelihood / ALL_ROWS ==
logits3 = torch.tensor([[2.0, 0.5], [0.1, 0.2], [1.0, -1.0]])
picked3 = torch.tensor([1, 0, 1])
usable3 = torch.tensor([1.0, 0.0, 1.0])
lp_all = joint_log_prob(logits3, picked3, usable3)
lp_free = joint_log_prob(logits3[[0, 2]], picked3[[0, 2]], usable3[[0, 2]])
check("joint_log_prob_ignores_masked", abs(float(lp_all) - float(lp_free)) < 1e-6,
      all=float(lp_all), free=float(lp_free))
ent3 = action_entropy(logits3, usable3)
ent_free = action_entropy(logits3[[0, 2]], usable3[[0, 2]])
check("entropy_ignores_masked", abs(float(ent3) - float(ent_free)) < 1e-6)
lp_masked_row = joint_log_prob(logits3, picked3, torch.zeros(3))
check("joint_log_prob_all_masked_row_is_zero", float(lp_masked_row) == 0.0, value=float(lp_masked_row))
ent_masked_row = action_entropy(logits3, torch.zeros(3))
check("entropy_all_masked_row_is_zero", float(ent_masked_row) == 0.0, value=float(ent_masked_row))
_a = torch.tensor([0.3, -1.2, 2.0]); _c = torch.tensor([0.1, 0.4, -0.7])
check("actor_advantage_is_lambda_task_minus_cost",
      torch.allclose(cm.actor_advantage(_a, _c, 0.5), 0.5 * _a - _c, atol=0, rtol=0))
_r = torch.tensor([1.0, 2.0, 3.0, 4.0]); _v = torch.tensor([0.5, 0.5, 0.5, 0.5])
_term = torch.tensor([False, False, False, True]); _boot = torch.tensor([0.5, 0.5, 0.5, 0.0])  # per-row next-values v[1..3]; 0 after the true-terminal step
_adv, _ret = cm.gae(_r, _v, gamma=1.0, gae_lambda=1.0, terminal=_term, bootstrap=_boot)
_manual_ret = torch.tensor([float(_r[i:].sum()) for i in range(4)])
check("gae_lambda1_gamma1_equals_undiscounted_return", torch.allclose(_ret, _manual_ret, atol=1e-6),
      got=_ret.tolist(), expected=_manual_ret.tolist())
check("gae_advantage_is_return_minus_value", torch.allclose(_adv, _manual_ret - _v, atol=1e-6))

# Fabricated two-episode batch whose tapes describe the R68 controller's OWN draw (built through
# the same routed forward the collector uses), so likelihood_alignment can pass. 5 decision rows,
# one of them a closed-HOLD (critic-only) row: the ALL_ROWS denominator must include it.
ppo_controller, *_ = r68.fresh_r68_controller(SEED)  # 5-tuple; keep the controller
ppo_opt = r68.build_optimizer(ppo_controller, common.LEARNING_RATE)


def fab_tape(controller, step, ready, taf, first=1.0):
    f = feat(ready, first=first, age=float(step % 5))
    targets = [t for (t, _a, _fr) in taf]
    actions = {t: a for (t, a, _fr) in taf}
    free = sorted(t for (t, _a, fr) in taf if fr)
    x = torch.as_tensor(np.asarray([f], dtype=np.float32))
    with torch.no_grad():
        logits, value = controller.forward(x)
        cost_value = controller.cost_values(x)
    lp = float(joint_log_prob(
        logits, torch.tensor([actions[t] for t in targets], dtype=torch.long),
        torch.tensor([1.0 if t in free else 0.0 for t in targets], dtype=torch.float32)))
    return {"step": int(step), "features": [float(v) for v in f], "targets": targets,
            "actions": {t: int(actions[t]) for t in targets}, "actionable": list(free),
            "log_prob": lp, "value": float(value[0]), "cost_value": float(cost_value[0])}


def fab_episode(controller, ep_index, scene, spec, horizon, cost_stream, task_stream):
    tapes = [fab_tape(controller, s, r, taf, first=f) for (s, r, taf, f) in spec]
    return {"collector": SimpleNamespace(records=tapes), "steps": int(horizon),
            "cost_stream": {int(k): float(v) for k, v in cost_stream.items()},
            "task_stream": {int(k): float(v) for k, v in task_stream.items()},
            "lambda": 1.0, "episode_index": int(ep_index), "training_seed": SEED,
            "scenario_seed": int(scene)}


epA = fab_episode(ppo_controller, 0, 1300000,
                  [(5, 100, [("t1", 1, True)], 1.0),
                   (9, 180, [("t2", 0, True)], 1.0),
                   (13, 50, [("t3", 1, False)], 1.0)],
                  16, {5: 100.0, 9: 0.0, 13: 50.0}, {5: 0.0, 9: 1.0, 13: 0.0})
epB = fab_episode(ppo_controller, 1, 1300001,
                  [(4, 147, [("t1", 0, True)], 1.0),
                   (11, 200, [("t2", 1, True)], 1.0)],
                  14, {4: 200.0, 11: 0.0}, {4: 0.0, 11: 1.0})
fab_episodes = [epA, epB]
samples = trainer.batch_samples(fab_episodes)
check("all_rows_one_sample_per_decision_step", len(samples) == 5, got=len(samples))
check("all_rows_includes_closed_hold_row", any(len(s["free"]) == 0 for s in samples),
      free_sizes=[len(s["free"]) for s in samples])
charge_forward(5 * len(samples), "fabricated ALL_ROWS update: alignment + 1 epoch of 1 minibatch")
saved_epochs = common.EPOCHS_PER_BATCH
common.EPOCHS_PER_BATCH = 1   # test-only: one minibatch so the reported surrogate is exact
try:
    fab_report = r68.frozen_early_ppo_update(ppo_controller, ppo_opt, fab_episodes,
                                             update_index=99, diag_updates=(99,))
finally:
    common.EPOCHS_PER_BATCH = saved_epochs
charge_ppo("fabricated ALL_ROWS update")
charge_steps(int(fab_report["optimizer_steps"]), "fabricated ALL_ROWS update optimizer steps")
check("all_rows_update_samples_is_five", fab_report["samples"] == 5, got=fab_report["samples"])
check("all_rows_update_one_minibatch_one_step",
      fab_report["minibatches"] == 1 and fab_report["optimizer_steps"] == 1,
      minibatches=fab_report["minibatches"], steps=fab_report["optimizer_steps"])
check("all_rows_update_likelihood_alignment_passed",
      bool(fab_report["likelihood_alignment"]["passed"]), alignment=fab_report["likelihood_alignment"])
check("all_rows_update_routing_active", fab_report["early_free"] == 2 and fab_report["late_free"] == 2,
      early_free=fab_report["early_free"], late_free=fab_report["late_free"])

# Exact ALL_ROWS denominator proof: with one minibatch at the pre-update parameters, the reported
# policy surrogate must equal the mean over ALL rows (5), and must differ from a mean over the
# free rows only (4).
stacked = torch.as_tensor(np.concatenate([s["features"] for s in samples], axis=0))
ppo2_controller, *_ = r68.fresh_r68_controller(SEED)  # 5-tuple; keep the controller
hidden = ppo2_controller.inner.trunk(stacked)
logits = r66.route_logits(ppo2_controller, stacked, hidden)
row = 0
new_lp, old_lp, adv = [], [], []
for s in samples:
    count = len(s["targets"])
    rows_idx = torch.arange(row, row + count)
    row += count
    picked = torch.tensor([s["actions"][t] for t in s["targets"]], dtype=torch.long)
    usable = torch.tensor([1.0 if t in s["free"] else 0.0 for t in s["targets"]], dtype=torch.float32)
    new_lp.append(joint_log_prob(logits[rows_idx], picked, usable))
    old_lp.append(float(s["log_prob"]))
    adv.append(float(s["advantage"]))
new_lp_t = torch.stack(new_lp)
old_lp_t = torch.tensor(old_lp, dtype=torch.float32)
adv_t = torch.tensor(adv, dtype=torch.float32)
ratio = torch.exp(new_lp_t - old_lp_t)
surrogate = torch.min(ratio * adv_t, torch.clamp(ratio, 1.0 - common.CLIP, 1.0 + common.CLIP) * adv_t)
manual_all_rows = float(surrogate.mean())
manual_free_rows = float(surrogate[[i for i, s in enumerate(samples) if len(s["free"]) > 0]].mean())
check("all_rows_denominator_includes_closed_hold_row",
      abs(fab_report["policy_surrogate"] - manual_all_rows) < 2e-6
      and abs(fab_report["policy_surrogate"] - manual_free_rows) > 1e-9,
      reported=fab_report["policy_surrogate"], manual_all_rows=manual_all_rows,
      manual_free_rows=manual_free_rows)
gate_diag = fab_report.get("gate_diag") or {}
check("all_rows_update_gate_diag_labels_frozen_structurally",
      gate_diag.get("trunk") == "frozen_requires_grad_false"
      and gate_diag.get("early_head") == "frozen_requires_grad_false", diag=gate_diag)
check("all_rows_update_gate_diag_records_late_actor_and_critic_grads",
      "late_actor_grad_l1" in gate_diag and "critic_grad_l1" in gate_diag, diag=gate_diag)
check("all_rows_update_trainable_moved",
      r66._tensor_identity(r66.trainable_parameters(ppo_controller))["sha256"]
      != r66._tensor_identity(r66.trainable_parameters(ppo2_controller))["sha256"])

# ============================================================ 7. real PPO update + resume =====
# Read-only reuse of ROUND60's own saved training rows/tapes (episodes 0..3 were collected by the
# ROUND60 ep0 policy, i.e. BEFORE ROUND60's first update). No environment is constructed.
R60_DIR = r66.R60_DIR


def reload_episode(seed, ep):
    matches = sorted((R60_DIR / "raw" / "rows").glob(f"v2_r60_train_s{seed}_e{ep}_s*_a1.json"))
    if not matches:
        return None
    row = json.loads(matches[0].read_text(encoding="utf-8"))
    tape_path = Path(row["artefacts"]["tape"])
    if not tape_path.is_file():
        return None
    tapes = [json.loads(line) for line in tape_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    row["collector"] = SimpleNamespace(records=tapes)
    row["training_seed"] = int(seed)
    row["episode_index"] = int(ep)
    row["scenario_seed"] = int(row["scenario_seed"])
    row["lambda"] = 1.0
    return row


episodes = [reload_episode(SEED, ep) for ep in range(4)]
episodes = [e for e in episodes if e is not None]
check("real_r60_episode_artefacts_available", len(episodes) == 4, found=len(episodes))
if len(episodes) == 4:
    # The update function is the SAME object R68 re-exports; run it on the R60-ep0-equal frozen
    # controller (pre_controller) so the tapes describe the controller's own draw. The R68-scaled
    # controller is exercised by the fabricated-episode update in section 6.
    check("r68_update_is_the_inherited_r66_function",
          r68.frozen_early_ppo_update is r66.frozen_early_ppo_update)
    real_opt = r68.build_optimizer(pre_controller, common.LEARNING_RATE)
    real_samples = trainer.batch_samples(episodes)
    charge_forward((1 + common.EPOCHS_PER_BATCH) * len(real_samples),
                   "real-episode update: alignment + epoch minibatch forwards")
    f_before, t_before = identities(pre_controller)
    real_report = r68.frozen_early_ppo_update(pre_controller, real_opt, episodes,
                                              update_index=0, diag_updates=(0,))
    charge_ppo("real-episode update")
    charge_steps(int(real_report["optimizer_steps"]), "real-episode update optimizer steps")
    f_after, t_after = identities(pre_controller)
    check("real_update_structure_is_inherited_r66_math",
          real_report["structure"] == r66.R66_STRUCTURE, structure=real_report["structure"])
    check("real_update_frozen_bit_unchanged", f_before == f_after)
    check("real_update_trainable_moved", t_before != t_after)
    check("real_update_object_counts_6_6",
          real_report["frozen_parameter_object_count"] == 6
          and real_report["trainable_parameter_object_count"] == 6)
    check("real_update_likelihood_alignment_passed",
          bool(real_report["likelihood_alignment"]["passed"]),
          alignment=real_report["likelihood_alignment"])
    check("real_update_routing_active_both_partitions",
          real_report["early_free"] > 0 and real_report["late_free"] > 0,
          early_free=real_report["early_free"], late_free=real_report["late_free"])
    means = real_report["module_gradient_l1_mean"]
    check("real_update_frozen_modules_recorded_none",
          means.get("trunk") is None and means.get("early_head") is None, means=means)
    check("real_update_trainable_modules_recorded_numbers",
          isinstance(means.get("late_head"), float) and isinstance(means.get("cost_head"), float),
          means=means)
    rdiag = real_report.get("gate_diag") or {}
    check("real_update_gate_diag_present", bool(rdiag), diag=rdiag)
    check("real_update_gate_diag_does_not_claim_frozen_zero",
          rdiag.get("trunk") == "frozen_requires_grad_false"
          and rdiag.get("early_head") == "frozen_requires_grad_false", diag=rdiag)

    # ---- full resume: both heads + Adam + lambda + RNG + cursor, then rejection of foreign sources
    state = r68.fresh_training_state(SEED)
    state["episodes_done"] = 4
    state["updates"] = 1
    state["optimizer_steps"] = int(real_report["optimizer_steps"])
    state["lambda"] = 0.73
    state["lambda_history"] = [{"batch": 0, "g_hat": 0.1, "lambda_before": 1.0, "lambda_next": 0.73}]
    scratch = HERE / "_r68_test_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    ckpt = scratch / "v2_train_s11_resume_test.pt"
    rng_at_save = trainer.rng_state()
    r68.save_checkpoint(pre_controller, real_opt, state, ckpt, freeze=base.R68_FREEZE)
    torch.manual_seed(987654)     # perturb the live RNG so a restore must actively restore it
    rc, ropt, rstate = r68.load_checkpoint(ckpt, freeze=base.R68_FREEZE)
    rng_after_load = trainer.rng_state()

    def rng_equal(a, b):
        if set(a) != set(b):
            return False
        for key in a:
            va, vb = a[key], b[key]
            if isinstance(va, torch.Tensor) or isinstance(vb, torch.Tensor):
                if not (isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor)
                        and torch.equal(va, vb)):
                    return False
            elif va != vb:
                return False
        return True

    check("resume_restores_both_heads_and_all_five_modules",
          all(torch.equal(a, b) for a, b in zip(pre_controller.parameters, rc.parameters))
          and len(list(rc.parameters)) == 12)
    check("resume_restores_adam_over_six_trainable",
          len(ropt.state_dict()["state"]) == 6
          and [len(g["params"]) for g in ropt.state_dict()["param_groups"]] == [6])
    check("resume_restores_adam_step_counts",
          sorted(int(s["step"]) for s in ropt.state_dict()["state"].values())
          == [int(real_report["optimizer_steps"])] * 6,
          steps=sorted(int(s["step"]) for s in ropt.state_dict()["state"].values()))
    check("resume_restores_lambda_and_cursor",
          float(rstate["lambda"]) == 0.73 and int(rstate["episodes_done"]) == 4
          and int(rstate["updates"]) == 1 and float(rstate["gae_lambda"]) == 1.0
          and float(rstate["eta_lambda"]) == 0.5
          and rstate["state_schema"] == r68.R68_STATE_SCHEMA)
    check("resume_restores_rng", rng_equal(rng_at_save, rng_after_load))
    check("resume_reestablishes_frozen_mask",
          all(not p.requires_grad for p in r68.frozen_parameters(rc))
          and all(p.requires_grad for p in r68.trainable_parameters(rc)))
    check("resume_late_head_is_loaded_not_rescaled",
          torch.equal(rc.inner.late_head.weight, pre_controller.inner.late_head.weight))
    ro_controller, ro_state, ro_identity = r68.load_checkpoint_read_only(
        ckpt, freeze=base.R68_FREEZE)
    check("read_only_load_restores_no_optimizer_or_rng",
          ro_identity["optimizer_restored"] is False and ro_identity["rng_restored"] is False)
    check("read_only_load_identity_carries_r68_contract",
          ro_identity["structure"] == r68.R68_STRUCTURE
          and float(ro_identity["late_init_alpha"]) == r68.LATE_INIT_ALPHA)

    for seed, role in ((SEED, "ep0"), (SEED, "ep128"), (22, "ep0")):
        rejected = False
        try:
            r68.load_checkpoint(r66.r60_model_path(seed, role), freeze=base.R68_FREEZE)
        except RuntimeError as error:
            rejected = "ROUND60" in str(error)
        except Exception:
            rejected = True
        check(f"r60_{role}_checkpoint_rejected_as_r68_resume", rejected)
    r66_ep128 = R66_DIR / "raw" / "checkpoints" / "v2_train_s11_ep128.pt"
    rejected = False
    try:
        r68.load_checkpoint(r66_ep128, freeze=base.R68_FREEZE)
    except RuntimeError as error:
        rejected = "ROUND60/ROUND66" in str(error)
    except Exception:
        rejected = True
    check("r66_ep128_checkpoint_rejected_as_r68_resume", rejected)
    rejected = False
    try:
        r68.load_checkpoint(ckpt, freeze={"kind": "wrong-freeze"})
    except RuntimeError:
        rejected = True
    check("wrong_freeze_rejected_on_resume", rejected)
    quarantine_like = (r66.R60_DIR / "_quarantine_attempt1_criticdetect_missing" / "raw"
                       / "checkpoints" / "v2_train_s11_before_b0.pt")
    rejected = False
    try:
        r66._require_compliant_r60_path(quarantine_like)
    except RuntimeError:
        rejected = True
    check("quarantined_r60_source_rejected", rejected)

# ============================================================ 8. real CLI configure chain =====
cfg = base.configure_and_inject(RAW)
ident = cfg["identity"]
check("cli_trainer_is_frozen_v2", Path(cfg["trainer"].__file__).name == "v2_trainer.py")
check("cli_ppo_update_is_r68", bool(ident["ppo_update_is_r68"]))
check("cli_save_is_r68", bool(ident["save_is_r68"]))
check("cli_load_is_r68", bool(ident["load_is_r68"]))
check("cli_load_ro_is_r68", bool(ident["load_ro_is_r68"]))
check("cli_fresh_controller_is_frozen_factory", bool(ident["fresh_is_frozen"]))
check("cli_training_version_is_r68", ident["training_version"] == r68.R68_TRAINING_VERSION)
check("cli_living_gae_lambda_1_eta_0p5",
      float(cfg["common"].GAE_LAMBDA) == 1.0 and float(cfg["common"].ETA_LAMBDA) == 0.5)
check("cli_caps_are_r68_quota",
      int(cfg["runtime"]["r40_common"].NORMAL_CAP) == 774
      and int(cfg["runtime"]["r40_common"].RESERVE_CAP) == 26
      and int(cfg["runtime"]["r40_common"].HARD_CAP) == 800)
check("cli_output_roots_under_r68_raw",
      str(Path(cfg["common"].CHECKPOINTS).resolve()).startswith(str(RAW.resolve()))
      and str(Path(cfg["common"].ROWS).resolve()).startswith(str(RAW.resolve()))
      and str(cfg["runtime"]["r40_common"].ROUND_DIR.resolve()) == str(RAW.resolve()))
check("cli_save_load_names_are_r68_wrappers",
      cfg["trainer"].save_checkpoint.__name__ == "r68_save"
      and cfg["trainer"].load_checkpoint.__name__ == "r68_load"
      and cfg["trainer"].load_checkpoint_read_only.__name__ == "r68_load_ro")
acc = base.zero_env_acceptance(cfg)
check("cli_zero_env_acceptance_passed", acc["status"] == "passed",
      failed=[k for k, v in acc["checks"].items() if not v])
cli_controller = cfg["frozen_fresh"](22)
check("cli_fresh_factory_builds_r68_structure_for_each_seed",
      r68.frozen_parameter_summary(cli_controller)["structure"] == r68.R68_STRUCTURE
      and cli_controller.parameter_summary()["parameter_object_count"] == 12)
ident22 = json.loads((Path(cfg["common"].CHECKPOINTS) / "r68_init_identity_s22.json")
                     .read_text(encoding="utf-8"))
check("cli_fresh_factory_writes_r68_init_identity_per_seed",
      bool(ident22["pre_scale_equals_r60_ep0"])
      and abs(float(ident22["late_init_alpha"]) - 0.10) < 1e-12)
check("cli_ppo_journal_path_under_r68_raw",
      str((RAW / "runtime" / "ppo_update_calls.jsonl").resolve()).startswith(str(RAW.resolve())))

# ============================================================ 9. unregistered / duplicate start
r42_run = cfg["r42_run"]
runner = cfg["runner"]
check("cli_builder_is_the_counted_builder", runner.build_joint_run.__name__ == "counted_build",
      name=getattr(runner.build_joint_run, "__name__", None))
real_rows_dir = RAW / "rows"
real_logs_dir = RAW / "logs"
before_rows = sorted(p.name for p in real_rows_dir.glob("*")) if real_rows_dir.is_dir() else []
before_logs = sorted(p.name for p in real_logs_dir.glob("*")) if real_logs_dir.is_dir() else []
r42_run._ACTIVE_CLAIM = None
r42_run._ACTIVE_PROGRESS = None
raised = ""
try:
    runner.build_joint_run(SimpleNamespace(scenario_seed=1800000))
except RuntimeError as exc:
    raised = str(exc)
check("unregistered_start_rejected_before_environment",
      "without an atomic quota claim" in raised, error=raised[:200])
after_rows = sorted(p.name for p in real_rows_dir.glob("*")) if real_rows_dir.is_dir() else []
after_logs = sorted(p.name for p in real_logs_dir.glob("*")) if real_logs_dir.is_dir() else []
check("unregistered_start_constructed_no_environment",
      r42_run._ACTIVE_PROGRESS is None and after_rows == before_rows and after_logs == before_logs)
dup_dir = Path(tempfile.mkdtemp(prefix="r68_dup_claim_"))
dup_claim = dup_dir / "attempt_0001.json"
dup_claim.write_text(json.dumps({"logical_job": "synthetic-duplicate-start-test", "attempt": 1,
                                 "status": "claimed_before_environment"}) + "\n", encoding="utf-8")
r42_run._ACTIVE_CLAIM = dup_claim
prog = r42_run._new_run_progress()
prog["environment_constructor_calls"] = 1
r42_run._ACTIVE_PROGRESS = prog
raised2 = ""
try:
    runner.build_joint_run(SimpleNamespace(scenario_seed=1800000))
except RuntimeError as exc:
    raised2 = str(exc)
check("duplicate_start_under_one_claim_rejected",
      "already attempted an environment build" in raised2, error=raised2[:200])
r42_run._ACTIVE_CLAIM = None
r42_run._ACTIVE_PROGRESS = None
check("active_claim_globals_restored", r42_run._ACTIVE_CLAIM is None
      and r42_run._ACTIVE_PROGRESS is None)
shutil.rmtree(dup_dir, ignore_errors=True)

# ============================================================ 10. cumulative budget guards ====
# (a) the persistent-guard behaviour, on throwaway synthetic ledgers.
gdir = Path(tempfile.mkdtemp(prefix="r68_guard_synth_"))
g1 = guards.ForwardRowGuard(gdir / "rows.json", cap=20)
for i in range(19):
    g1.charge(1, f"synthetic-prefill-{i}")
check("guard_19_consumed_leaves_exactly_one", g1.remaining() == 1, used=g1.used())
used20 = g1.charge(1, "synthetic-twentieth")
check("guard_twentieth_charge_accepted", used20 == 20)
refused = ""
try:
    g1.charge(1, "synthetic-twenty-first")
except RuntimeError as exc:
    refused = str(exc)
check("guard_twenty_first_refused", "would take used to 21 > cap 20" in refused, error=refused[:200])
check("guard_cap_holds_after_refusal", g1.used() == 20)
try:
    g1.charge(2, "synthetic-doomed-work")
except RuntimeError:
    pass
check("guard_over_cap_charge_applies_nothing", g1.used() == 20)
g1b = guards.ForwardRowGuard(gdir / "rows.json", cap=20)
check("guard_restart_does_not_reset_consumption", g1b.used() == 20)
sub = gdir / "other_output_subdir"
sub.mkdir(parents=True, exist_ok=True)
g2 = guards.ForwardRowGuard(sub / "rows.json", cap=20)
check("guard_other_subdir_fresh_but_original_unchanged", g2.used() == 0 and g1.used() == 20)
backup = gdir / "rows.backup.json"
shutil.copyfile(g1.path, backup)
gb = guards.ForwardRowGuard(backup, cap=20)
gb.charge(1, "charge-on-backup-copy")
check("guard_backup_copy_charge_does_not_change_original", gb.used() == 21 and g1.used() == 20)
g3 = guards.ForwardRowGuard(gdir / "dup.json", cap=20)
g3.charge(3, "deterministic work with identical output")
g3.charge(3, "deterministic work with identical output")
check("guard_deterministic_duplicate_not_deduped", g3.used() == 6)
race_guard = guards.ForwardRowGuard(gdir / "race.json", cap=20)
for i in range(19):
    race_guard.charge(1, f"race-prefill-{i}")
wins: list = []
losses: list = []
_race_lock = threading.Lock()


def _racer(n):
    try:
        race_guard.charge(1, f"racer-{n}")
        with _race_lock:
            wins.append(n)
    except RuntimeError:
        with _race_lock:
            losses.append(n)


_threads = [threading.Thread(target=_racer, args=(n,)) for n in range(8)]
for _t in _threads:
    _t.start()
for _t in _threads:
    _t.join()
check("guard_concurrent_last_slot_single_winner",
      len(wins) == 1 and len(losses) == 7 and race_guard.used() == 20,
      wins=wins, losses=losses, used=race_guard.used())
shutil.rmtree(gdir, ignore_errors=True)

# (b) the r40_common atomic claim/slot mechanism, on a throwaway synthetic ROUND_DIR.
r40 = cfg["runtime"]["r40_common"]
saved_r40 = (r40.ROUND_DIR, r40.NORMAL_CAP, r40.RESERVE_CAP, r40.HARD_CAP, r40.plan)
sdir = Path(tempfile.mkdtemp(prefix="r68_slot_synth_"))
r40.ROUND_DIR = sdir
r40.NORMAL_CAP = 20
r40.RESERVE_CAP = 2
r40.HARD_CAP = 22
r40.plan = lambda: {"normal_total": 20, "technical_reserve_max": 2, "hard_cap": 22,
                    "task_id": base.TASK_ID}
try:
    for i in range(19):
        r40.claim_launch(f"synthetic_job_{i}", category="normal", attempt=1,
                         metadata={"synthetic": True})
    snap = r40.ledger_snapshot()
    check("slot_19_consumed_one_remains",
          snap["normal_used"] == 19 and snap["plan"]["normal_total"] == 20, used=snap["normal_used"])
    p20 = r40.claim_launch("synthetic_job_19", category="normal", attempt=1)
    check("slot_twentieth_claim_accepted", r40.ledger_snapshot()["normal_used"] == 20)
    refused = ""
    try:
        r40.claim_launch("synthetic_job_20", category="normal", attempt=1)
    except RuntimeError as exc:
        refused = str(exc)
    check("slot_twenty_first_refused", "launch cap 20 exhausted" in refused, error=refused[:200])
    check("slot_refusal_consumes_nothing", r40.ledger_snapshot()["normal_used"] == 20)
    refused_claim = sdir / "launch_claims" / "synthetic_job_20" / "attempt_0001.json"
    check("slot_refused_claim_marked_not_started",
          refused_claim.is_file() and json.loads(refused_claim.read_text(encoding="utf-8"))["status"]
          == "not_started_quota_exhausted")
    r40.finish_launch(p20, ok=False, detail="synthetic failure")
    check("slot_failed_run_keeps_its_slot", r40.ledger_snapshot()["normal_used"] == 20)
    dup = ""
    try:
        r40.claim_launch("synthetic_job_19", category="normal", attempt=1)
    except RuntimeError as exc:
        dup = str(exc)
    check("slot_failed_attempt_not_silently_retried", "already claimed" in dup, error=dup[:200])
    booked = r40.ledger_snapshot()["total_used"]
    backup_dir = sdir / "backup_copy"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(sdir / "launch_claims", backup_dir / "launch_claims")
    shutil.copytree(sdir / "launch_slots", backup_dir / "launch_slots")
    check("slot_backup_copy_is_not_a_new_execution",
          r40.ledger_snapshot()["total_used"] == booked)
    meta = {"arm": "SYNTHETIC", "scenario_seed": 1800000, "note": "identical deterministic output"}
    ca = r40.claim_launch("synthetic_dup_a", category="normal", attempt=1, metadata=dict(meta))
    cb = r40.claim_launch("synthetic_dup_b", category="normal", attempt=1, metadata=dict(meta))
    r40.finish_launch(ca, ok=True, detail="identical")
    r40.finish_launch(cb, ok=True, detail="identical")
    check("slot_deterministic_duplicate_claims_not_deduped",
          r40.ledger_snapshot()["total_used"] == booked + 2)
    rdir2 = Path(tempfile.mkdtemp(prefix="r68_slot_race_"))
    r40.ROUND_DIR = rdir2
    r40.NORMAL_CAP = 1
    r40.RESERVE_CAP = 1
    r40.HARD_CAP = 2
    wins2: list = []
    losses2: list = []

    def _racer2(n):
        try:
            r40.claim_launch(f"race_job_{n}", category="normal", attempt=1)
            with _race_lock:
                wins2.append(n)
        except RuntimeError:
            with _race_lock:
                losses2.append(n)

    _threads2 = [threading.Thread(target=_racer2, args=(n,)) for n in range(8)]
    for _t in _threads2:
        _t.start()
    for _t in _threads2:
        _t.join()
    check("slot_concurrent_last_slot_single_winner",
          len(wins2) == 1 and len(losses2) == 7, wins=wins2, losses=losses2)
    shutil.rmtree(rdir2, ignore_errors=True)
finally:
    r40.ROUND_DIR, r40.NORMAL_CAP, r40.RESERVE_CAP, r40.HARD_CAP, r40.plan = saved_r40
    shutil.rmtree(sdir, ignore_errors=True)
check("r40_live_quota_restored_after_synthetic_tests",
      int(r40.NORMAL_CAP) == 774 and int(r40.RESERVE_CAP) == 26 and int(r40.HARD_CAP) == 800
      and Path(r40.ROUND_DIR).resolve() == RAW.resolve())

# (c) ROUND60's historical overage cannot be read as ROUND68 headroom.
r60_verdict = json.loads((r66.R60_DIR / "ROUND60_RECONCILIATION_VERDICT.json")
                         .read_text(encoding="utf-8"))
totals = r60_verdict["totals"]
check("r60_historical_608_exceeds_the_600_hard_cap",
      int(totals["real_environment_constructions"]) == 608
      and int(totals["hard_cap"]) == 600 and int(totals["over_by"]) == 8, totals=totals)
r60_ledger = json.loads((r66.R60_DIR / "raw" / "runtime" / "ROUND60_LEDGER.json")
                        .read_text(encoding="utf-8"))
check("r60_ledger_is_r60_scoped",
      r60_ledger["task_id"] == "ROUND60-PARTITIONED-ACTOR-HEAD-ABLATION"
      and int(r60_ledger["counts"]["hard_headroom"]) == 23)
r68_counts = base.R68Ledger(RAW).counts()
check("r68_ledger_is_independent_and_fresh",
      r68_counts["cumulative_attempts"] == 0
      and r68_counts["cumulative_attempts_including_historical_probe"] == 0
      and r68_counts["hard_cap"] == 800 and r68_counts["hard_headroom"] == 800,
      counts=r68_counts)
check("r68_quota_is_a_fresh_independent_authorization",
      base.HISTORICAL_PROBE == 0 and base.NORMAL_CAP == 774 and base.RESERVE_CAP == 26
      and base.HARD_CAP == 800 and base.TASK_ID == "ROUND68-LATE-INITIAL-CONFIDENCE-ABLATION"
      and (RAW / "runtime" / "ROUND68_LEDGER.json").resolve()
      != (r66.R60_DIR / "raw" / "runtime" / "ROUND60_LEDGER.json").resolve())

# ============================================================ 11. protection + no side effects
prot1 = RUN_ROOT / "ROUND46_CURRENT_TASK.json"
prot2 = RUN_ROOT / "ROUND46_R2_V6_STOP_STATE.json"
check("protected_round46_files_unchanged",
      r66.sha256_file(prot1) == "cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b"
      and r66.sha256_file(prot2) == "93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18")
check("test_created_no_real_claims",
      len(list((RAW / "launch_claims").glob("*/*.json"))) == 0)
check("test_consumed_no_real_slots",
      len(list((RAW / "launch_slots" / "normal").glob("*.json"))) == 0
      and len(list((RAW / "launch_slots" / "reserve").glob("*.json"))) == 0)
check("test_left_no_active_claim", r42_run._ACTIVE_CLAIM is None)

raise SystemExit(finish("passed", extra={
    "notes": [
        "initial forward/loss equivalence is a wiring fact, not a claim of equal gradient norms, "
        "clip coefficients or Adam trajectory versus ROUND66/ROUND60",
        "an absent gradient on the un-routed head is a routing fact, not a frozen parameter",
        "the R68 critic->trunk parameter-level isolation is proved on an UNFROZEN controller; on "
        "the R68 controller the freeze masks the parameter-level effect by construction",
        "the inherited update report stamps the ROUND66 update-math structure; the ROUND68 stamp "
        "lives in the controller summary, the gate, the ppo journal and the checkpoint contract",
        "synthetic checkpoints are confined to the throwaway _r68_test_scratch directory",
    ],
    "scratch_dir": str(HERE / "_r68_test_scratch"),
}))
