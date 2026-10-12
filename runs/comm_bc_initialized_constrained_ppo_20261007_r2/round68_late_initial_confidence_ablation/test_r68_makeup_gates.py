"""ROUND68 make-up acceptance gates for the forgone zero-environment tail (ZERO env constructions).

Context (append-only): pre-launch zero-env acceptance was developed against four throwaway guard
roots and consolidated to 9319 model forward rows against the 8192 cap (EXCEEDED by 1127;
preserved, never refunded). A future-only make-up grant (raw/runtime/ROUND68_MAKEUP_GRANT.json)
authorises <=2048 ADDITIONAL forward rows, +<=16 physical synthetic optimizer steps charged to the
SHARED cap-32 steps guard (the total stays <=32), and synthetic PPO calls counted separately. The
grant does NOT retroactively approve the old overage and changes no research configuration.

This file runs ONLY the forgone items, through the real controller / batch / optimizer /
checkpoint / CLI paths (no import-mode stubs, no rerun of the sections that already passed):

  M0  preserved guard state, living-config-vs-freeze values, and the pure mask / likelihood /
      ALL_ROWS / GAE math re-verified in its corrected form (the previously failed GAE check);
  M1  a real PPO update through the inherited r66 update function on the ROUND68-scaled
      controller, driven by the controller's OWN synthetic behaviour record (multi-target bundles
      with mixed early/late routing inside one bundle, masked targets, closed-HOLD rows, both SEND
      and SKIP actions), plus a test-scoped lr=0 optimizer instance that makes the ALL_ROWS
      denominator exactly checkable (the frozen EPOCHS=4 / MINIBATCH=256 living values are NOT
      modified anywhere in this file);
  M2  a real 4-episode PPO update, full five-module + six-parameter Adam + lambda + RNG + cursor
      checkpoint save/restore, a deterministic post-resume update that must match the unbroken
      continuation, read-only load semantics, and rejection of ROUND60 / ROUND66 / wrong-freeze /
      quarantine sources;
  M3  the real CLI configure_and_inject identity chain, stopping before any episode provider;
  M4  unregistered-start and duplicate-start rejection BEFORE environment construction;
  M5  pure cumulative-guard semantics (hard boundary, no refund, restart/subdir/backup isolation,
      no dedup of deterministic duplicates, single winner under a concurrent race, the make-up
      grant is not expandable, ROUND60's historical 608 constructions are not ROUND68 headroom);
  M6  protected-file byte identity and no-side-effects assertions.

Forward-row accounting: a hook on controller.inner.trunk counts each top-level trunk entry as
prod(shape[:-1]) rows when the last dimension is 19 (a 1-D length-19 input counts as 1 row);
masked/frozen rows count; internal layers (route_logits / value_head / cost_head on a precomputed
hidden) never re-enter the trunk and are not double-counted. The make-up guard is charged
atomically BEFORE the forward runs and is never refunded. Before every real update a projection
computed from the actual sample features (alignment rows + EPOCHS x candidate rows) must fit the
remaining grant, or the phase stops instead of overrunning. 265 decision records are NOT assumed
to be 265 input rows: one decision record carries one 19-dim row PER candidate target.

Data provenance: ROUND60's historical tapes are NOT used here and are NOT modified. Every tape in
this file is a synthetic behaviour record drawn from the ROUND68 controller under test through the
same routed forward the collector uses, so the logged probabilities describe that controller's own
draw; nothing historical is presented as ROUND68 alpha=0.10 on-policy data.

No environment is constructed, no training/eval claim is created, no committed PPO update is
replayed, and synthetic checkpoints are written only to the dedicated _r68_makeup_scratch
directory. This file does not resume training; it only re-establishes eligibility to do so.
"""
from __future__ import annotations
import copy, json, random, shutil, sys, tempfile, threading, time
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

import r68_execute as base          # the real ROUND68 executor (CLI chain, ledger, plan)
import r68_frozen as r68            # the only scientific difference + full ROUND68 identity
import r66_frozen as r66            # inherited frozen-path machinery
import r68_guards as guards         # persistent acceptance/synthetic-computation guards

RAW = HERE / "raw"
RT = RAW / "runtime"
OLD_FWD = guards.ForwardRowGuard(RT / "r68_guard_forward_rows.json")
MAKEUP_FWD = guards.ForwardRowGuard(RT / "r68_makeup_forward_rows.json", cap=2048)
STEP_GUARD = guards.SyntheticStepGuard(RT / "r68_guard_synthetic_steps.json", cap=32)
PPO_GUARD = guards.SyntheticPpoGuard(RT / "r68_guard_synthetic_ppo.json", cap=32)

RESULTS: list[dict] = []
HOOK_LOG: list[dict] = []
PROJECTIONS: list[dict] = []
NOTES: list[str] = []
SEED = 11
SYNTH_STEPS = 0
SYNTH_PPO = 0
_SCRATCH = HERE / "_r68_makeup_scratch"
_CLAIMS_BEFORE = {p.name: p.read_text(encoding="utf-8")
                  for p in sorted((RAW / "launch_claims").glob("*/*.json"))}
_SLOTS_BEFORE = sorted(p.name for p in (RAW / "launch_slots" / "normal").glob("*.json"))
_ROWS_BEFORE = sorted(p.name for p in (RAW / "rows").glob("*")) if (RAW / "rows").is_dir() else []
_LOGS_BEFORE = sorted(p.name for p in (RAW / "logs").glob("*")) if (RAW / "logs").is_dir() else []


def check(name, passed, **detail):
    RESULTS.append({"check": name, "passed": bool(passed), **detail})
    print(("PASS " if passed else "FAIL ") + name +
          ("" if passed else "  " + json.dumps(detail, default=str)[:900]), flush=True)
    if not passed:
        try:
            finish("failed")
        except Exception:
            pass
        raise SystemExit(f"ROUND68 make-up gates failed at: {name}")


def finish(status: str, extra: dict | None = None) -> int:
    counted = [e for e in HOOK_LOG if e["counted"]]
    payload = {"kind": "round68-makeup-gates/1",
               "task_id": base.TASK_ID, "branch": base.BRANCH,
               "status": status, "checks": len(RESULTS),
               "failed": [r["check"] for r in RESULTS if not r["passed"]],
               "synthetic_optimizer_steps": SYNTH_STEPS, "synthetic_ppo_calls": SYNTH_PPO,
               "trunk_hook": {
                   "rule": ("rows = prod(shape[:-1]) when the last dimension is 19 (a 1-D "
                            "length-19 input counts as 1 row); masked/frozen rows count; internal "
                            "layers never re-enter the trunk and are not double-counted; charged "
                            "atomically BEFORE the forward, never refunded"),
                   "calls": len(HOOK_LOG), "counted_calls": len(counted),
                   "counted_rows": int(sum(e["rows"] for e in counted)),
                   "not_counted_entries": [e for e in HOOK_LOG if not e["counted"]]},
               "projections": PROJECTIONS,
               "guards": {"old_forward_rows_preserved": OLD_FWD.snapshot(),
                          "makeup_forward_rows": MAKEUP_FWD.snapshot(),
                          "synthetic_steps": STEP_GUARD.snapshot(),
                          "synthetic_ppo": PPO_GUARD.snapshot()},
               "results": RESULTS, "notes": NOTES, "finished_unix": time.time()}
    if extra:
        payload.update(extra)
    out = RT / "ROUND68_MAKEUP_GATES.json"
    base._write_json(out, payload)
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("results", "guards", "projections")}, indent=1))
    return 0 if status == "passed" else 1


def _rows_of(x):
    if isinstance(x, torch.Tensor):
        shape = tuple(int(d) for d in x.shape)
        if shape and shape[-1] == 19:
            n = 1
            for d in shape[:-1]:
                n *= int(d)
            return int(n)
    return None


def install_trunk_hook(controller, label):
    """Count every top-level trunk entry (rows where the last dim is 19) and charge the make-up
    guard BEFORE the forward runs. Internal layers are not wrapped: they never re-enter the trunk."""
    trunk = controller.inner.trunk
    orig = trunk.forward

    def hooked(x, *a, **k):
        rows = _rows_of(x)
        shape = list(x.shape) if isinstance(x, torch.Tensor) else str(type(x))
        if rows is None:
            HOOK_LOG.append({"controller": label, "shape": shape, "counted": False})
        else:
            MAKEUP_FWD.charge(rows, f"{label}: trunk forward rows={rows} shape={shape}")
            HOOK_LOG.append({"controller": label, "shape": shape, "rows": rows, "counted": True})
        return orig(x, *a, **k)

    trunk.forward = hooked
    return trunk


def hook_rows(label=None):
    return sum(e.get("rows", 0) for e in HOOK_LOG
               if e["counted"] and (label is None or e["controller"] == label))


def project_update_rows(samples, what):
    """Projected forward rows for one update: one alignment forward per sample (all its candidate
    rows) + EPOCHS full passes over the same candidate rows (minibatching does not change the
    total). Refuses to start when the projection does not fit the remaining make-up grant."""
    per = int(sum(len(s["features"]) for s in samples))
    total = per * (1 + int(common.EPOCHS_PER_BATCH))
    PROJECTIONS.append({"what": what, "samples": len(samples), "candidate_rows": per,
                        "alignment_rows": per, "epoch_rows": per * int(common.EPOCHS_PER_BATCH),
                        "projected_total_rows": total,
                        "makeup_remaining_before": MAKEUP_FWD.remaining()})
    if total > MAKEUP_FWD.remaining():
        raise RuntimeError(f"{what}: projected {total} forward rows exceed the make-up grant "
                           f"remaining {MAKEUP_FWD.remaining()}; refusing to start")
    return total


def charge_steps(n, what):
    global SYNTH_STEPS
    SYNTH_STEPS += int(n)
    return STEP_GUARD.charge(int(n), what)


def charge_ppo(what):
    global SYNTH_PPO
    SYNTH_PPO += 1
    return PPO_GUARD.charge(1, what)


def feat(ready, first=1.0, age=0.0):
    f = [0.0] * 19
    f[0] = float(age)
    f[5] = float(first)
    f[18] = (float(ready) + float(age)) / 1000.0
    return f


def identities(controller):
    frozen, trainable = r66.partitioned_parameters(controller)
    return (r66._tensor_identity(frozen)["sha256"], r66._tensor_identity(trainable)["sha256"])


def build_episode(controller, ep_index, scene, spec):
    records = [draw_record(controller, st, targets) for (st, targets) in spec["records"]]
    return {"collector": SimpleNamespace(records=records), "steps": int(spec["steps"]),
            "cost_stream": {int(k): float(v) for k, v in spec["cost"].items()},
            "task_stream": {int(k): float(v) for k, v in spec["task"].items()},
            "lambda": 1.0, "episode_index": int(ep_index), "training_seed": SEED,
            "scenario_seed": int(scene)}


mods = r68.configure_source_imports()
trainer, common, controller_module = mods["trainer"], mods["common"], mods["controller_module"]
from onpolicy import joint_log_prob, action_entropy  # noqa: E402
import cost_math as cm  # noqa: E402

_GEN = torch.Generator().manual_seed(20260927)


def draw_record(controller, step, spec):
    """One synthetic behaviour record: per-target 19-dim rows, actions drawn from the
    controller's OWN routed distribution (masked targets take SKIP), log_prob/value/cost_value
    read from the same forward the collector uses."""
    rows = [feat(r, first=fi, age=float(int(step) % 5)) for (_t, r, fi, _fr) in spec]
    x = torch.as_tensor(np.asarray(rows, dtype=np.float32))
    with torch.no_grad():
        logits, value = controller.forward(x)
        cost_value = controller.cost_values(x)
    probs = torch.softmax(logits, dim=-1)
    targets, actions, free = [], {}, []
    for i, (t, _r, _fi, fr) in enumerate(spec):
        targets.append(t)
        if fr:
            actions[t] = int(torch.multinomial(probs[i], 1, generator=_GEN).item())
            free.append(t)
        else:
            actions[t] = int(r66.SKIP_INDEX)
    usable = [1.0 if t in free else 0.0 for t in targets]
    lp = float(joint_log_prob(logits,
                              torch.tensor([actions[t] for t in targets], dtype=torch.long),
                              torch.tensor(usable, dtype=torch.float32)))
    return {"step": int(step), "features": rows, "targets": targets, "actions": actions,
            "actionable": list(free), "log_prob": lp,
            "value": float(value.mean()), "cost_value": float(cost_value.mean())}


# ============================================================ M0. preserved state + pure math =
old = OLD_FWD.snapshot()
check("old_forward_overage_preserved_9319_over_8192",
      old["used"] == 9319 and old["cap"] == 8192, snapshot={k: old[k] for k in ("cap", "used")})
refused = ""
try:
    OLD_FWD.charge(1, "make-up phase: any further charge must still be refused")
except RuntimeError as exc:
    refused = str(exc)
check("old_forward_guard_still_refuses_further_charge",
      "would take used to 9320 > cap 8192" in refused and OLD_FWD.used() == 9319, error=refused[:200])
check("makeup_grant_fresh_zero_used", MAKEUP_FWD.used() == 0 and MAKEUP_FWD.remaining() == 2048)
check("shared_steps_guard_used_is_2", STEP_GUARD.used() == 2, used=STEP_GUARD.used())
check("shared_ppo_guard_used_is_0", PPO_GUARD.used() == 0, used=PPO_GUARD.used())

live = {"epochs_per_batch": int(common.EPOCHS_PER_BATCH), "minibatch_steps": int(common.MINIBATCH_STEPS),
        "learning_rate": float(common.LEARNING_RATE), "data_order_seed": int(common.DATA_ORDER_SEED),
        "gae_lambda": float(common.GAE_LAMBDA), "eta_lambda": float(common.ETA_LAMBDA),
        "gamma": float(common.GAMMA), "log_prob_tolerance": float(common.LOG_PROB_TOLERANCE),
        "b0_bytes": float(common.B0_BYTES), "clip": float(common.CLIP),
        "entropy_bonus": float(common.ENTROPY_BONUS), "value_coefficient": float(common.VALUE_COEFFICIENT),
        "max_grad_norm": float(common.MAX_GRAD_NORM)}
expected_live = {"epochs_per_batch": 4, "minibatch_steps": 256, "learning_rate": 0.0003,
                 "data_order_seed": 20260927, "gae_lambda": 1.0, "eta_lambda": 0.5, "gamma": 1.0,
                 "log_prob_tolerance": 1e-06, "b0_bytes": 10000.0, "clip": 0.2,
                 "entropy_bonus": 0.01, "value_coefficient": 0.5, "max_grad_norm": 0.5}
check("living_config_matches_frozen_values", live == expected_live, live=live, expected=expected_live)
check("frozen_constants_cut147_horizon1000_beta1_skip0",
      float(r66.READY_CUT) == 147.0 and float(r66.HORIZON) == 1000.0
      and float(r66.GATE_BETA) == 1.00 and int(r66.SKIP_INDEX) == 0 and int(r66.SEND_INDEX) == 1)

_r = torch.tensor([1.0, 2.0, 3.0, 4.0]); _v = torch.tensor([0.5, 0.5, 0.5, 0.5])
_term = torch.tensor([False, False, False, True]); _boot = torch.tensor([0.5, 0.5, 0.5, 0.0])
_adv, _ret = cm.gae(_r, _v, gamma=1.0, gae_lambda=1.0, terminal=_term, bootstrap=_boot)
_manual = torch.tensor([float(_r[i:].sum()) for i in range(4)])
check("gae_lambda1_gamma1_equals_undiscounted_return", torch.allclose(_ret, _manual, atol=1e-6),
      got=_ret.tolist(), expected=_manual.tolist())
check("gae_advantage_is_return_minus_value", torch.allclose(_adv, _manual - _v, atol=1e-6))
_a = torch.tensor([0.3, -1.2, 2.0]); _c = torch.tensor([0.1, 0.4, -0.7])
check("actor_advantage_is_lambda_task_minus_cost",
      torch.allclose(cm.actor_advantage(_a, _c, 0.5), 0.5 * _a - _c, atol=0, rtol=0))
logits3 = torch.tensor([[2.0, 0.5], [0.1, 0.2], [1.0, -1.0]])
picked3 = torch.tensor([1, 0, 1]); usable3 = torch.tensor([1.0, 0.0, 1.0])
check("joint_log_prob_ignores_masked",
      abs(float(joint_log_prob(logits3, picked3, usable3))
          - float(joint_log_prob(logits3[[0, 2]], picked3[[0, 2]], usable3[[0, 2]]))) < 1e-6)
check("entropy_ignores_masked",
      abs(float(action_entropy(logits3, usable3))
          - float(action_entropy(logits3[[0, 2]], usable3[[0, 2]]))) < 1e-6)
check("joint_log_prob_all_masked_row_is_zero",
      float(joint_log_prob(logits3, picked3, torch.zeros(3))) == 0.0)
check("entropy_all_masked_row_is_zero", float(action_entropy(logits3, torch.zeros(3))) == 0.0)

# ============================================================ M1. real PPO update (own draw) ==
M1_SPEC = {
    0: {"scene": 1300000, "steps": 1000,
        "records": [
            (5, [("t1", 100, 1.0, True), ("t2", 140, 1.0, True), ("t3", 200, 0.0, False)]),
            (9, [("t1", 180, 1.0, True), ("t2", 181, 1.0, True)]),
            (13, [("t1", 50, 1.0, False)]),
            (20, [("t1", 147, 0.0, True), ("t2", 147, 1.0, True)]),
        ],
        "cost": {5: 120.0, 9: 0.0, 13: 80.0, 20: 40.0, 30: 60.0},
        "task": {5: 0.0, 9: 1.0, 13: 0.0, 20: 0.5, 30: 0.0}},
    1: {"scene": 1300001, "steps": 1000,
        "records": [
            (4, [("t1", 148, 1.0, True), ("t2", 300, 1.0, True)]),
            (11, [("t1", 200, 1.0, True)]),
            (16, [("t1", 90, 1.0, True), ("t2", 95, 1.0, False)]),
            (22, [("t1", 260, 1.0, False)]),
        ],
        "cost": {4: 200.0, 11: 0.0, 16: 90.0, 22: 30.0, 33: 55.0},
        "task": {4: 0.0, 11: 1.0, 16: 0.0, 22: 0.5, 33: 0.0}},
}

m1_ctrl, *_ = r68.fresh_r68_controller(SEED)
install_trunk_hook(m1_ctrl, "m1")
m1_episodes = [build_episode(m1_ctrl, 0, M1_SPEC[0]["scene"], M1_SPEC[0]),
               build_episode(m1_ctrl, 1, M1_SPEC[1]["scene"], M1_SPEC[1])]
m1_samples = trainer.batch_samples(m1_episodes)
check("m1_one_sample_per_decision_step", len(m1_samples) == 8, got=len(m1_samples))
check("m1_closed_hold_rows_present", sum(1 for s in m1_samples if len(s["free"]) == 0) == 2)
check("m1_masked_targets_inside_bundles_present",
      any(len(s["targets"]) > len(s["free"]) > 0 for s in m1_samples))
def _recompute_log_prob(controller, sample):
    x = torch.as_tensor(np.asarray(sample["features"], dtype=np.float32))
    with torch.no_grad():
        logits, _ = controller.forward(x)
    return float(joint_log_prob(
        logits,
        torch.tensor([sample["actions"][t] for t in sample["targets"]], dtype=torch.long),
        torch.tensor([1.0 if t in sample["free"] else 0.0 for t in sample["targets"]],
                     dtype=torch.float32)))


check("m1_own_draw_alignment_is_exact",
      max(abs(float(s["log_prob"]) - _recompute_log_prob(m1_ctrl, s))
          for s in m1_samples) < 1e-9)
# ---- M1a: real PPO update through the inherited update function (real lr, real forwards) ----
m1_opt = r68.build_optimizer(m1_ctrl, common.LEARNING_RATE)
project_update_rows(m1_samples, "M1 real-lr update (inherited r66 update math, R68-scaled controller)")
f_before, t_before = identities(m1_ctrl)
m1_report = r68.frozen_early_ppo_update(m1_ctrl, m1_opt, m1_episodes,
                                        update_index=9001, diag_updates=(9001,))
charge_ppo("M1 real-lr update")
charge_steps(int(m1_report["optimizer_steps"]), "M1 real-lr update optimizer steps")
f_after, t_after = identities(m1_ctrl)
check("m1_update_structure_is_inherited_r66_math", m1_report["structure"] == r66.R66_STRUCTURE,
      structure=m1_report["structure"])
check("m1_update_frozen_bit_unchanged", f_before == f_after)
check("m1_update_trainable_moved", t_before != t_after)
check("m1_update_object_counts_6_6",
      m1_report["frozen_parameter_object_count"] == 6
      and m1_report["trainable_parameter_object_count"] == 6)
check("m1_update_alignment_passed", bool(m1_report["likelihood_alignment"]["passed"]),
      alignment=m1_report["likelihood_alignment"])
check("m1_update_routing_active_both_partitions",
      m1_report["early_free"] > 0 and m1_report["late_free"] > 0,
      early_free=m1_report["early_free"], late_free=m1_report["late_free"])
means = m1_report["module_gradient_l1_mean"]
check("m1_frozen_modules_recorded_none",
      means.get("trunk") is None and means.get("early_head") is None, means=means)
check("m1_trainable_modules_recorded_numbers",
      isinstance(means.get("late_head"), float) and isinstance(means.get("task_value_head"), float)
      and isinstance(means.get("cost_head"), float), means=means)
d1 = m1_report.get("gate_diag") or {}
check("m1_gate_diag_labels_frozen_structurally",
      d1.get("trunk") == "frozen_requires_grad_false"
      and d1.get("early_head") == "frozen_requires_grad_false", diag=d1)
check("m1_late_actor_received_gradient", float(d1.get("late_actor_grad_l1", -1.0) or 0.0) > 0.0, diag=d1)
check("m1_critic_received_gradient", float(d1.get("critic_grad_l1", -1.0) or 0.0) > 0.0, diag=d1)
check("m1_value_losses_finite",
      bool(np.isfinite([m1_report.get("value_loss_task", 0.0), m1_report.get("value_loss_cost", 0.0)])))
NOTES.append("the beta=1.00 gate prior acts on early first-emission free candidates, which route to "
             "the FROZEN early_head, so on the ROUND68 controller the prior cannot move any "
             "trainable parameter; that is the documented ROUND66 effect and is recorded as "
             "absent/zero here, never as a measured training effect")

# ---- M1b: lr=0 test-scoped optimizer: ALL_ROWS denominator exactly checkable ----------------
m1x_ctrl, *_ = r68.fresh_r68_controller(SEED)
install_trunk_hook(m1x_ctrl, "m1_exact")
m1x_opt = r68.build_optimizer(m1x_ctrl, 0.0)
m1x_samples = trainer.batch_samples(m1_episodes)
adv_all = float(np.mean([s["advantage"] for s in m1x_samples]))
adv_free = float(np.mean([s["advantage"] for s in m1x_samples if len(s["free"]) > 0]))
check("m1_fixture_separates_all_rows_from_free_rows", abs(adv_all - adv_free) > 1e-3,
      adv_all=adv_all, adv_free=adv_free)
project_update_rows(m1x_samples, "M1 lr=0 exactness update (ALL_ROWS denominator)")
fx_before, tx_before = identities(m1x_ctrl)
m1x_report = r68.frozen_early_ppo_update(m1x_ctrl, m1x_opt, m1_episodes,
                                         update_index=9002, diag_updates=(9002,))
charge_ppo("M1 lr=0 exactness update")
charge_steps(int(m1x_report["optimizer_steps"]), "M1 lr=0 exactness update optimizer steps")
check("m1_lr0_update_left_parameters_bit_unchanged", identities(m1x_ctrl) == (fx_before, tx_before))
check("m1_all_rows_denominator_includes_closed_hold_row",
      abs(float(m1x_report["policy_surrogate"]) - adv_all) < 5e-6
      and abs(float(m1x_report["policy_surrogate"]) - adv_free) > 1e-3,
      reported=m1x_report["policy_surrogate"], adv_all=adv_all, adv_free=adv_free)
rows_all = np.concatenate([s["features"] for s in m1x_samples], axis=0)
with torch.no_grad():
    _lg, _ = m1x_ctrl.forward(torch.as_tensor(rows_all))
exp_ent, _row = 0.0, 0
for s in m1x_samples:
    n = len(s["targets"])
    usable = torch.tensor([1.0 if t in s["free"] else 0.0 for t in s["targets"]],
                          dtype=torch.float32)
    exp_ent += float(action_entropy(_lg[_row:_row + n], usable))
    _row += n
exp_ent /= len(m1x_samples)
check("m1_entropy_all_rows_mean_matches_reference",
      abs(float(m1x_report["entropy"]) - exp_ent) < 5e-6,
      reported=m1x_report["entropy"], expected=exp_ent)
check("m1_hook_charged_only_nineteen_dim_rows",
      all(e["counted"] for e in HOOK_LOG) and hook_rows("m1") > 0 and hook_rows("m1_exact") > 0,
      m1_rows=hook_rows("m1"), m1_exact_rows=hook_rows("m1_exact"))

# ============================================================ M2. real 4-episode update + resume
def build_m2_episode(controller, ep_index, scene, n_records=12):
    rng = random.Random(20260927 + 1000 * int(ep_index))
    steps = sorted(rng.sample(range(3, 960), n_records))
    closed_step = steps[len(steps) // 2]
    records, cost, task = [], {}, {}
    for st in steps:
        spec = []
        for j in range(rng.choice([1, 2, 2, 3, 4])):
            early = rng.random() < 0.5
            ready = rng.randint(30, 147) if early else rng.randint(148, 420)
            first = 1.0 if rng.random() < 0.6 else 0.0
            free = (st != closed_step) and (rng.random() < 0.78)
            spec.append((f"t{j + 1}", ready, first, free))
        records.append(draw_record(controller, st, spec))
        cost[st] = float(rng.choice([0.0, 40.0, 90.0, 150.0, 220.0]))
        task[st] = float(rng.choice([0.0, 0.5, 1.0]))
    cost[1000] = 60.0
    task[1000] = 0.0
    return {"collector": SimpleNamespace(records=records), "steps": 1000,
            "cost_stream": {int(k): float(v) for k, v in cost.items()},
            "task_stream": {int(k): float(v) for k, v in task.items()},
            "lambda": 1.0, "episode_index": int(ep_index), "training_seed": SEED,
            "scenario_seed": int(scene)}


m2_ctrl, *_ = r68.fresh_r68_controller(SEED)
install_trunk_hook(m2_ctrl, "m2")
m2_episodes = [build_m2_episode(m2_ctrl, e, 1300010 + e) for e in range(4)]
m2_samples = trainer.batch_samples(m2_episodes)
m2_rows = int(sum(len(s["features"]) for s in m2_samples))
check("m2_four_episode_batch_of_forty_eight_records",
      len(m2_episodes) == 4 and len(m2_samples) == 48, samples=len(m2_samples))
check("m2_records_carry_more_rows_than_records", m2_rows > len(m2_samples),
      records=len(m2_samples), candidate_rows=m2_rows)
def _free_readies(s):
    out = []
    for i, t in enumerate(s["targets"]):
        if t in s["free"]:
            f = s["features"][i]
            out.append(int(round(float(f[18]) * 1000) - round(float(f[0]))))
    return out


_all_free_readies = [r for s in m2_samples for r in _free_readies(s)]
check("m2_has_closed_hold_early_and_late_free_rows",
      any(not s["free"] for s in m2_samples)
      and any(r <= 147 for r in _all_free_readies) and any(r > 147 for r in _all_free_readies),
      closed=sum(1 for s in m2_samples if not s["free"]),
      early_free_rows=sum(1 for r in _all_free_readies if r <= 147),
      late_free_rows=sum(1 for r in _all_free_readies if r > 147))
m2_opt = r68.build_optimizer(m2_ctrl, common.LEARNING_RATE)
project_update_rows(m2_samples, "M2 update 0 (real lr, four-episode batch)")
f2_before, t2_before = identities(m2_ctrl)
m2_r0 = r68.frozen_early_ppo_update(m2_ctrl, m2_opt, m2_episodes,
                                    update_index=9003, diag_updates=(9003,))
charge_ppo("M2 update 0")
charge_steps(int(m2_r0["optimizer_steps"]), "M2 update 0 optimizer steps")
check("m2_update0_structure_is_inherited_r66_math", m2_r0["structure"] == r66.R66_STRUCTURE)
check("m2_update0_frozen_bit_unchanged_trainable_moved",
      identities(m2_ctrl)[0] == f2_before and identities(m2_ctrl)[1] != t2_before)
check("m2_update0_alignment_passed", bool(m2_r0["likelihood_alignment"]["passed"]),
      alignment=m2_r0["likelihood_alignment"])
check("m2_update0_routing_active_both_partitions",
      m2_r0["early_free"] > 0 and m2_r0["late_free"] > 0,
      early_free=m2_r0["early_free"], late_free=m2_r0["late_free"])
means2 = m2_r0["module_gradient_l1_mean"]
check("m2_update0_frozen_modules_recorded_none",
      means2.get("trunk") is None and means2.get("early_head") is None, means=means2)
d2 = m2_r0.get("gate_diag") or {}
check("m2_update0_late_actor_and_critic_received_gradient",
      float(d2.get("late_actor_grad_l1", -1.0) or 0.0) > 0.0
      and float(d2.get("critic_grad_l1", -1.0) or 0.0) > 0.0, diag=d2)

# ---- full resume: five modules + six-parameter Adam + lambda + RNG + cursor ------------------
state = r68.fresh_training_state(SEED)
state["episodes_done"] = 4
state["updates"] = 1
state["optimizer_steps"] = int(m2_r0["optimizer_steps"])
state["lambda"] = 0.73
state["lambda_history"] = [{"batch": 0, "g_hat": 0.1, "lambda_before": 1.0, "lambda_next": 0.73}]
_SCRATCH.mkdir(parents=True, exist_ok=True)
ckpt = _SCRATCH / "v2_train_s11_makeup_resume_test.pt"
rng_at_save = trainer.rng_state()
r68.save_checkpoint(m2_ctrl, m2_opt, state, ckpt, freeze=base.R68_FREEZE)
saved_params = {k: v.detach().clone() for k, v in m2_ctrl.state_dict().items()}
saved_opt = copy.deepcopy(m2_opt.state_dict())
project_update_rows(m2_samples, "M2 update 1 unbroken continuation (real lr)")
u1_unbroken = r68.frozen_early_ppo_update(m2_ctrl, m2_opt, m2_episodes,
                                          update_index=9004, diag_updates=(9004,))
charge_ppo("M2 update 1 unbroken continuation")
charge_steps(int(u1_unbroken["optimizer_steps"]), "M2 update 1 unbroken continuation steps")
sha_after_unbroken = m2_ctrl.parameter_summary()["parameter_sha256"]

random.seed(13579)
np.random.seed(13579)
torch.manual_seed(13579)
rc, ropt, rstate = r68.load_checkpoint(ckpt, freeze=base.R68_FREEZE)
restored_params = {k: v.detach().clone() for k, v in rc.state_dict().items()}
check("resume_restores_all_twelve_parameters_bit_exact",
      set(restored_params) == set(saved_params)
      and all(torch.equal(restored_params[k], saved_params[k]) for k in saved_params),
      keys=sorted(restored_params))


def adam_state_equal(a, b):
    if set(a["state"]) != set(b["state"]):
        return False
    for k in a["state"]:
        sa, sb = a["state"][k], b["state"][k]
        if set(sa) != set(sb):
            return False
        for kk in sa:
            va, vb = sa[kk], sb[kk]
            if isinstance(va, torch.Tensor) or isinstance(vb, torch.Tensor):
                if not (isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor)
                        and torch.equal(va, vb)):
                    return False
            elif va != vb:
                return False
    return ([len(g["params"]) for g in a["param_groups"]]
            == [len(g["params"]) for g in b["param_groups"]])


check("resume_restores_six_parameter_adam_state_bit_exact",
      adam_state_equal(ropt.state_dict(), saved_opt)
      and [len(g["params"]) for g in ropt.state_dict()["param_groups"]] == [6])
check("resume_restores_lambda_history_and_cursor",
      float(rstate["lambda"]) == 0.73 and int(rstate["episodes_done"]) == 4
      and int(rstate["updates"]) == 1 and int(rstate["optimizer_steps"]) == int(m2_r0["optimizer_steps"])
      and rstate["structure"] == r68.R68_STRUCTURE
      and abs(float(rstate["late_init_alpha"]) - r68.LATE_INIT_ALPHA) < 1e-12
      and len(rstate["lambda_history"]) == 1, state={k: rstate.get(k) for k in
                                                    ("lambda", "episodes_done", "updates",
                                                     "optimizer_steps", "structure",
                                                     "late_init_alpha")})
check("resume_restores_all_three_rng_streams", trainer.rng_state() == rng_at_save)
check("resume_reestablished_frozen_mask",
      not any(p.requires_grad for p in r68.frozen_parameters(rc))
      and all(p.requires_grad for p in r68.trainable_parameters(rc)))

project_update_rows(m2_samples, "M2 update 1 resumed continuation (real lr)")
u1_resumed = r68.frozen_early_ppo_update(rc, ropt, m2_episodes,
                                         update_index=9004, diag_updates=(9004,))
charge_ppo("M2 update 1 resumed continuation")
charge_steps(int(u1_resumed["optimizer_steps"]), "M2 update 1 resumed continuation steps")
_cmp = ("policy_surrogate", "value_loss_task", "value_loss_cost", "entropy", "approx_kl",
        "clip_fraction", "gradient_norm_mean", "gate_loss", "early_free", "late_free",
        "optimizer_steps", "samples", "minibatches")
check("resumed_update_matches_unbroken_continuation",
      all(float(u1_unbroken.get(k, -1e18)) == float(u1_resumed.get(k, -1e18)) for k in _cmp),
      unbroken={k: u1_unbroken.get(k) for k in _cmp}, resumed={k: u1_resumed.get(k) for k in _cmp})
check("resumed_final_parameters_match_unbroken",
      rc.parameter_summary()["parameter_sha256"] == sha_after_unbroken)

ro_ctrl, ro_state, ro_ident = r68.load_checkpoint_read_only(ckpt, freeze=base.R68_FREEZE)
check("read_only_load_restores_no_optimizer_or_rng",
      ro_ident["optimizer_restored"] is False and ro_ident["rng_restored"] is False)
check("read_only_load_identity_carries_r68_contract",
      ro_ident["structure"] == r68.R68_STRUCTURE
      and float(ro_ident["late_init_alpha"]) == r68.LATE_INIT_ALPHA)
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
# ============================================================ M3. real CLI configure chain =====
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
      failed=[k for k, v in acc["checks"].items() if not v], num_checks=len(acc["checks"]))
base._write_json(RT / "ROUND68_MAKEUP_CLI_ACCEPTANCE.json", acc)
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
check("cli_no_model_forward_ran_in_this_phase",
      hook_rows("m1") + hook_rows("m1_exact") + hook_rows("m2")
      == sum(e["rows"] for e in HOOK_LOG if e["counted"]))

# ============================================================ M4. unregistered / duplicate start
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
check("active_claim_globals_restored",
      r42_run._ACTIVE_CLAIM is None and r42_run._ACTIVE_PROGRESS is None)
shutil.rmtree(dup_dir, ignore_errors=True)

# ============================================================ M5. pure cumulative-guard semantics
gdir = Path(tempfile.mkdtemp(prefix="r68_makeup_guard_synth_"))
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
check("guard_overcap_charge_applies_nothing", g1.used() == 20)
g1b = guards.ForwardRowGuard(gdir / "rows.json", cap=20)
check("guard_restart_does_not_reset", g1b.used() == 20)
sub = gdir / "other_subdir"
sub.mkdir(parents=True, exist_ok=True)
g1c = guards.ForwardRowGuard(sub / "rows.json", cap=20)
check("guard_new_subdir_is_a_separate_ledger_not_a_reset",
      g1c.used() == 0 and g1b.used() == 20)
backup = gdir / "backup_copy"
backup.mkdir(parents=True, exist_ok=True)
shutil.copy2(gdir / "rows.json", backup / "rows.json")
check("guard_backup_copy_is_not_a_new_execution", g1b.used() == 20)
gA = guards.ForwardRowGuard(gdir / "dup_a.json", cap=20)
gB = guards.ForwardRowGuard(gdir / "dup_b.json", cap=20)
gA.charge(3, "identical deterministic work")
gB.charge(3, "identical deterministic work")
check("guard_deterministic_duplicate_work_not_deduped",
      gA.used() == 3 and gB.used() == 3)
_race_lock = threading.Lock()
_wins: list = []
_losses: list = []
race_dir = Path(tempfile.mkdtemp(prefix="r68_makeup_race_"))
race_guard = guards.ForwardRowGuard(race_dir / "rows.json", cap=20)
for i in range(19):
    race_guard.charge(1, f"race-prefill-{i}")


def _racer(n):
    try:
        race_guard.charge(1, f"race-{n}")
        with _race_lock:
            _wins.append(n)
    except RuntimeError:
        with _race_lock:
            _losses.append(n)


_threads = [threading.Thread(target=_racer, args=(n,)) for n in range(8)]
for _t in _threads:
    _t.start()
for _t in _threads:
    _t.join()
check("guard_concurrent_last_slot_single_winner",
      len(_wins) == 1 and len(_losses) == 7 and race_guard.used() == 20,
      wins=_wins, losses=_losses, used=race_guard.used())
shutil.rmtree(race_dir, ignore_errors=True)
shutil.rmtree(gdir, ignore_errors=True)
refused_grant = ""
try:
    MAKEUP_FWD.charge(10 ** 9, "make-up phase: the 2048 grant must not be expandable")
except RuntimeError as exc:
    refused_grant = str(exc)
check("makeup_grant_is_not_expandable",
      "would take used to" in refused_grant and MAKEUP_FWD.used() <= 2048,
      error=refused_grant[:200], used=MAKEUP_FWD.used())

r40 = cfg["runtime"]["r40_common"]
saved_r40 = (r40.ROUND_DIR, r40.NORMAL_CAP, r40.RESERVE_CAP, r40.HARD_CAP, r40.plan)
sdir = Path(tempfile.mkdtemp(prefix="r68_makeup_slot_synth_"))
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
    rdir2 = Path(tempfile.mkdtemp(prefix="r68_makeup_slot_race_"))
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
check("r68_ledger_records_exactly_the_two_real_launches",
      r68_counts["normal_claims"] == 2 and r68_counts["reserve_claims"] == 0
      and r68_counts["complete"] == 1 and r68_counts["cumulative_attempts"] == 2
      and r68_counts["hard_cap"] == 800 and r68_counts["hard_headroom"] == 798,
      counts=r68_counts)

# ============================================================ M6. protection + no side effects
prot1 = RUN_ROOT / "ROUND46_CURRENT_TASK.json"
prot2 = RUN_ROOT / "ROUND46_R2_V6_STOP_STATE.json"
check("protected_round46_files_unchanged",
      r66.sha256_file(prot1) == "cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b"
      and r66.sha256_file(prot2) == "93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18")
claims_after = {p.name: p.read_text(encoding="utf-8")
                for p in sorted((RAW / "launch_claims").glob("*/*.json"))}
check("makeup_gates_created_no_new_claims", claims_after == _CLAIMS_BEFORE,
      before=sorted(_CLAIMS_BEFORE), after=sorted(claims_after))
check("makeup_gates_consumed_no_slots",
      sorted(p.name for p in (RAW / "launch_slots" / "normal").glob("*.json")) == _SLOTS_BEFORE
      and len(list((RAW / "launch_slots" / "reserve").glob("*.json"))) == 0)
rows_after = sorted(p.name for p in real_rows_dir.glob("*")) if real_rows_dir.is_dir() else []
logs_after = sorted(p.name for p in real_logs_dir.glob("*")) if real_logs_dir.is_dir() else []
check("makeup_gates_wrote_no_rows_or_logs",
      rows_after == _ROWS_BEFORE and logs_after == _LOGS_BEFORE)
check("synthetic_checkpoint_confined_to_scratch_dir",
      ckpt.is_file() and str(ckpt.resolve()).startswith(str(_SCRATCH.resolve()))
      and not any(str(p.resolve()).startswith(str((RAW / "checkpoints").resolve()))
                  for p in _SCRATCH.glob("*.pt") if p != ckpt))
check("trunk_hook_total_matches_charged_rows",
      sum(e["rows"] for e in HOOK_LOG if e["counted"]) == MAKEUP_FWD.used()
      and MAKEUP_FWD.used() <= 2048, charged=MAKEUP_FWD.used())
check("synthetic_step_total_within_shared_cap", STEP_GUARD.used() <= 32, used=STEP_GUARD.used())
check("synthetic_ppo_total_within_shared_cap", PPO_GUARD.used() <= 32, used=PPO_GUARD.used())

NOTES.extend([
    "the 18 pure rechecks and the 28/28 CLI metadata acceptance are NOT represented as real PPO / "
    "full checkpoint recovery / old-version rejection passes; this file is the first real pass of "
    "those forgone items",
    "ROUND60's historical tapes were not used and not modified; every tape here is a synthetic "
    "behaviour record drawn from the ROUND68 controller under test",
    "the lr=0 optimizer in M1b is a test-scoped instance that makes the ALL_ROWS arithmetic "
    "exactly checkable; the frozen living lr 3e-4, EPOCHS=4 and MINIBATCH=256 are unchanged",
    "the inherited update report stamps the ROUND66 update-math structure because the update "
    "function IS ROUND66's; the ROUND68 stamp lives in the controller summary, the gate, the ppo "
    "journal and the checkpoint contract",
    "this file constructs no environment, resumes no training, replays no committed update and "
    "creates no claim or slot; it only re-establishes eligibility",
])

raise SystemExit(finish("passed", extra={
    "sections": {"M0": "preserved guards + living config + pure math",
                 "M1": "real PPO update on the R68-scaled controller from its own synthetic draw "
                       "(real lr) + lr=0 exactness of the ALL_ROWS denominator",
                 "M2": "real 4-episode update + full resume + deterministic continuation + "
                       "read-only load + foreign-source rejection",
                 "M3": "real CLI configure_and_inject identity chain (stop before provider)",
                 "M4": "unregistered / duplicate start rejection before environment construction",
                 "M5": "pure cumulative-guard semantics + grant non-expandability + R60 608",
                 "M6": "protection + no side effects"},
    "scratch_dir": str(_SCRATCH),
    "hook_rows_by_controller": {label: hook_rows(label)
                                for label in sorted({e["controller"] for e in HOOK_LOG})},
}))