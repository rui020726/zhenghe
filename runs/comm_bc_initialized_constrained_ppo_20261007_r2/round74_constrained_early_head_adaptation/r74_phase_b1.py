"""ROUND74 PHASE B1: the real model/controller/batch/PPO qualification gate runner.

Zero environment constructions. It qualifies the ONE R74 structural change (early_head becomes
trainable; the shared trunk + late_head + dual critic keep their ROUND71 role) through the R74
production resource entry (r74_guard_prod.R74ResourcesProd). Every real optimizer step is charged
to the synthetic-step guard BEFORE it runs (make_charged_step around the pristine Adam.step); every
real trunk forward is metered at the trunk entry (register_forward_pre_hook -> offline row guard).

It does NOT run the full multi-epoch ppo_update, does NOT construct an environment, does NOT write
the R43 training journal, and does NOT resume any R60/R66/R68/R71/R73 full checkpoint. It runs the
faithful R74 minibatch loss (r74_model.minibatch_loss, verified a faithful transcription) for the
controlled single-step gates. PHASE B (the 384/192/focal matrix) is NOT started here.

Usage:
  python r74_phase_b1.py --mode fixture --fixture-root <throwaway>   # simulated, 0 production charge
  python r74_phase_b1.py --mode production                            # real charge to canonical counters
Exit code 0 iff all five gates pass, env constructions stay 0, and no gate raised.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
R53_DIR = HERE.parent / "round53_critic_isolation_trial_readiness"
if str(R53_DIR) not in sys.path:
    sys.path.insert(0, str(R53_DIR))

import torch
_PRISTINE_ADAM_STEP = torch.optim.Adam.step
_ADAM_PRISTINE_AT_IMPORT = not hasattr(torch.optim.Adam.step, "__wrapped__")

import numpy as np
import r74_model as M
from r74_guard_prod import R74ResourcesProd, make_charged_step
from r74_guard import R74_CANONICAL_ROOT

REGISTERED_INIT_SHA = {
    11: "6f6581cff959dc027dfe5e771d963ef4b27b5a2f9d6847bf3d1bc6328c5de32f",
    22: "29dbecdb9eaeae5ba346d8a68d03a13ca4ebb56680e28b4fff288032acfd1aa9",
    33: "3685f4aa30367276e2cc4f9cb6cad23773350fa084476cd249ae1f11be62f0ca",
}
REGISTERED_INIT_FILE_SHA = {
    11: "fc25ead19bce38d00ae3bf619971ab717084711bdd2b5eb0431d1dda720a952f",
    22: "d95b9a27d5c28c8cd0b80c63b3539c78763bbaf53b79734a522e8e5a59cd8f81",
    33: "27baa262443a5ca6c6fb15bd60e0b294f6313324fe84cf386a2ac24377a874a9",
}
CONTROL_COMMIT = "fe53d476efb56baaae194d68d7f749a546b14389"
LIVE_KEYS = ("GAMMA", "GAE_LAMBDA", "ETA_LAMBDA", "CLIP", "VALUE_COEFFICIENT", "ENTROPY_BONUS",
             "LEARNING_RATE", "MAX_GRAD_NORM", "EPOCHS_PER_BATCH", "MINIBATCH_STEPS",
             "READY_CUT", "GATE_BETA")


def _adam_digest(opt):
    sd = opt.state_dict()["state"]
    h = hashlib.sha256()
    for k in sorted(sd, key=str):
        st = sd[k]
        for f in ("step", "exp_avg", "exp_avg_sq"):
            v = st[f]
            arr = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
            h.update(f.encode()); h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def run_all(mode, fixture_root=None):
    cfg = M.configure()
    live = cfg["live"]
    from onpolicy import joint_log_prob
    import cost_math as cm
    import v2_trainer as trainer
    from round53_trial_adapter import critic_input_detached

    if mode == "production":
        res = R74ResourcesProd.production()
        receipts = HERE / "raw" / "runtime" / "round74_phase_b1"
    else:
        res = R74ResourcesProd.fixture(fixture_root)
        receipts = Path(fixture_root) / "round74_phase_b1_receipts"
    receipts.mkdir(parents=True, exist_ok=True)

    verdict = {
        "kind": "round74-phase-b1-verdict/1", "mode": mode,
        "canonical_root": str(R74_CANONICAL_ROOT),
        "adam_pristine_at_import": bool(_ADAM_PRISTINE_AT_IMPORT),
        "adam_pristine_at_run": bool(torch.optim.Adam.step is _PRISTINE_ADAM_STEP),
        "control_commit": CONTROL_COMMIT,
        "live": {k: live[k] for k in LIVE_KEYS},
        "before": res.snapshots(), "gates": {}, "counterexamples": {}, "errors": [],
    }

    def rows(): return res.offline_forward.used()
    def steps(): return res.synth_steps.used()

    def make_meter(controller, label):
        def hook(module, inputs):
            res.offline_forward.charge_trunk_forward(inputs[0], label)  # None -> input unchanged
        return controller.inner.trunk.register_forward_pre_hook(hook)

    def mkrow(ready, first=1.0):
        f = np.zeros(19, dtype=np.float32); f[18] = ready / 1000.0; f[5] = first
        return f

    def make_sample(controller, readies, actions, adv=0.7):
        feats = np.stack([mkrow(r) for r in readies], axis=0)
        tg = list(range(len(readies)))
        with torch.no_grad():
            lg, _, _, _ = M.forward_logits_values(controller, torch.as_tensor(feats))
            lp = float(joint_log_prob(lg, torch.tensor(actions, dtype=torch.long), torch.ones(len(readies))))
        return {"features": feats, "targets": tg, "actions": {t: a for t, a in zip(tg, actions)},
                "free": set(tg), "log_prob": lp, "advantage": adv,
                "return_task": 1.0, "return_cost": 0.2}

    results = {}
    meter_c = None
    meter_c2 = None
    early_s = late_s = None

    # ---------------- B1-G1 identity + mask (3 seeds) ----------------
    g = {"id": "B1-G1", "name": "model_identity_and_mask", "seeds": {}}
    r0, s0 = rows(), steps()
    try:
        for seed in (11, 22, 33):
            init = M.load_init(seed, expected_param_sha256=REGISTERED_INIT_SHA[seed])
            c = init["controller"]; mask = M.assert_mask(c); ident = init["identity"]
            file_sha = hashlib.sha256(M.r71_init_path(seed).read_bytes()).hexdigest()
            checks = {
                "param_sha_matches_registered": ident["init_parameter_sha256"] == REGISTERED_INIT_SHA[seed],
                "file_sha_matches_registered": file_sha == REGISTERED_INIT_FILE_SHA[seed],
                "from_trained_ep128_is_false": ident["from_trained_ep128"] is False,
                "optimizer_fresh": ident["optimizer_fresh"] is True,
                "mask_8_trainable_4_frozen_12_total": (mask["trainable_parameter_object_count"] == 8
                    and mask["frozen_parameter_object_count"] == 4 and mask["total_parameter_object_count"] == 12),
                "scalars_390_5440_5830": (mask["trainable_scalar_count"] == 390
                    and mask["frozen_scalar_count"] == 5440 and mask["total_scalar_count"] == 5830),
                "early_head_trainable": bool(c.inner.early_head.weight.requires_grad),
                "trunk_frozen": not bool(next(c.inner.trunk.parameters()).requires_grad),
                "structure_is_r74": M.R74_STRUCTURE == "partitioned-early-late-actor-head/1+early-head-trainable-late-adaptation/1",
            }
            g["seeds"][f"s{seed}"] = {"pass": all(checks.values()), "checks": checks,
                                      "init_parameter_sha256": ident["init_parameter_sha256"]}
        g["status"] = "passed" if all(v["pass"] for v in g["seeds"].values()) else "failed"
    except Exception as e:
        g["status"] = "failed"; g["error"] = f"{type(e).__name__}: {e}"
        verdict["errors"].append(f"B1-G1: {type(e).__name__}: {e}")
    g["offline_rows"] = rows() - r0; g["physical_steps"] = steps() - s0
    results["B1-G1"] = g
    verdict["counterexamples"]["B1-G1"] = ("an old four-module/ep128 INIT would fail param_sha_matches_registered "
        "or from_trained_ep128_is_false; a still-frozen early_head fails early_head_trainable")

    # ---------------- s11 gate controller + trunk meter ----------------
    c = M.load_init(11, expected_param_sha256=REGISTERED_INIT_SHA[11])["controller"]
    meter_c = make_meter(c, "b1_s11_gate_trunk_forward")

    # ---------------- B1-G2 routed forward boundary ----------------
    g = {"id": "B1-G2", "name": "routed_forward_boundary"}
    r0, s0 = rows(), steps()
    try:
        readies = [100, 147, 148, 200, 50, 300, 147, 148]
        feats = np.stack([mkrow(x) for x in readies], axis=0).astype(np.float32)
        logits, vt, vc, hidden = M.forward_logits_values(c, torch.as_tensor(feats))
        with torch.no_grad():
            early = c.inner.early_head(hidden); late = c.inner.late_head(hidden)
        per_row = {}; all_ok = True
        for i, rd in enumerate(readies):
            got = M.ready_step_of(feats[i]); exp = early[i] if got <= 147 else late[i]
            ok = bool(torch.allclose(logits[i].detach(), exp.detach(), atol=0, rtol=0))
            per_row[str(i)] = {"ready": got, "route": "early" if got <= 147 else "late", "bit_exact": ok}
            all_ok = all_ok and ok and (got == rd)
        checks = {"every_row_bit_exact": all_ok,
                  "boundary_147_early_148_late": per_row["1"]["route"] == "early" and per_row["2"]["route"] == "late"}
        g["checks"] = checks; g["per_row"] = per_row
        g["status"] = "passed" if all(checks.values()) else "failed"
    except Exception as e:
        g["status"] = "failed"; g["error"] = f"{type(e).__name__}: {e}"
        verdict["errors"].append(f"B1-G2: {type(e).__name__}: {e}")
    g["offline_rows"] = rows() - r0; g["physical_steps"] = steps() - s0
    results["B1-G2"] = g
    verdict["counterexamples"]["B1-G2"] = "the ready=148 rows must route to late_head; a single shared head would match early rows but the 147/148 pair pins the boundary"

    # ---------------- B1-G3 gradient routing + critic isolation ----------------
    g = {"id": "B1-G3", "name": "gradient_routing_and_critic_isolation"}
    r0, s0 = rows(), steps()
    try:
        from types import SimpleNamespace
        torch.manual_seed(20261013)
        trunk = torch.nn.Linear(3, 4); vh = torch.nn.Linear(4, 1); ch = torch.nn.Linear(4, 1)
        synth = SimpleNamespace(inner=SimpleNamespace(value_head=vh), cost_head=ch)
        x = torch.randn(8, 3)
        def _cb():
            for p in trunk.parameters(): p.grad = None
            hidden = trunk(x); loss = vh(hidden).sum() + ch(hidden).sum(); loss.backward()
            return trunk.weight.grad
        with critic_input_detached(synth):
            iso_grad = _cb()
        plain_grad = _cb()
        iso = {"isolated_trunk_grad_none_or_zero": (iso_grad is None or float(iso_grad.abs().max()) == 0.0),
               "plain_trunk_grad_nonzero": (plain_grad is not None and float(plain_grad.abs().max()) > 0.0),
               "temporary_hooks_removed": (len(vh._forward_pre_hooks) == 0 and len(ch._forward_pre_hooks) == 0)}
        early_s = make_sample(c, [100, 120], [1, 0]); late_s = make_sample(c, [200, 260], [1, 0])
        _, ie = M.minibatch_loss(c, [early_s], beta=live["GATE_BETA"], common=live, want_grads=True)
        _, il = M.minibatch_loss(c, [late_s], beta=live["GATE_BETA"], common=live, want_grads=True)
        route = {"early_sample_early_grad_pos": ie["grad_l1_early_head"] > 0,
                 "early_sample_late_grad_zero": ie["grad_l1_late_head"] == 0.0,
                 "early_sample_critics_grad_pos": ie["grad_l1_value_task_head"] > 0 and ie["grad_l1_value_cost_head"] > 0,
                 "late_sample_late_grad_pos": il["grad_l1_late_head"] > 0,
                 "late_sample_early_grad_zero": il["grad_l1_early_head"] == 0.0,
                 "late_sample_critics_grad_pos": il["grad_l1_value_task_head"] > 0 and il["grad_l1_value_cost_head"] > 0}
        adv = {f"At{at:+.1f}_Ac{ac:+.1f}": abs(float(cm.actor_advantage(at, ac, 1.0)) - (at - ac)) < 1e-9
               for at, ac in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (-1.0, 0.5)]}
        info_keys = ("early_free", "late_free", "grad_l1_early_head", "grad_l1_late_head",
                     "grad_l1_value_task_head", "grad_l1_value_cost_head", "preclip_total_norm")
        g["critic_isolation"] = iso; g["gradient_routing"] = route; g["advantage_sign"] = adv
        g["early_info"] = {k: ie[k] for k in info_keys}; g["late_info"] = {k: il[k] for k in info_keys}
        g["status"] = "passed" if (all(iso.values()) and all(route.values()) and all(adv.values())) else "failed"
    except Exception as e:
        g["status"] = "failed"; g["error"] = f"{type(e).__name__}: {e}"
        verdict["errors"].append(f"B1-G3: {type(e).__name__}: {e}")
    g["offline_rows"] = rows() - r0; g["physical_steps"] = steps() - s0
    results["B1-G3"] = g
    verdict["counterexamples"]["B1-G3"] = "a LATE sample with early_grad>0 would mean the heads are not partitioned; critic_input_detached removing the plain trunk grad proves critic->trunk isolation"

    # ---------------- B1-G4 controlled real Adam step ----------------
    g = {"id": "B1-G4", "name": "controlled_real_adam_step"}
    r0, s0 = rows(), steps()
    try:
        assert torch.optim.Adam.step is _PRISTINE_ADAM_STEP, "Adam.step is not pristine"
        if early_s is None or late_s is None:
            raise RuntimeError("G3 samples unavailable")
        opt = M.build_optimizer(c, live["LEARNING_RATE"])
        n_opt = sum(len(gp["params"]) for gp in opt.param_groups)
        before = M.module_parameter_digest(c)
        loss, _ = M.minibatch_loss(c, [early_s, late_s], beta=live["GATE_BETA"], common=live, want_grads=False)
        opt.zero_grad(set_to_none=True); loss.backward()
        trunk_params = list(c.inner.trunk.parameters()); train_params = M.trainable_parameters(c)
        trunk_none = all(p.grad is None for p in trunk_params)
        train_present = all(p.grad is not None for p in train_params)
        clip_norm = float(torch.nn.utils.clip_grad_norm_(train_params, live["MAX_GRAD_NORM"]))
        make_charged_step(_PRISTINE_ADAM_STEP, res.synth_steps, what="b1_g4_establish_nonempty_adam")(opt)
        opt.zero_grad(set_to_none=True)
        after = M.module_parameter_digest(c)
        checks = {"optimizer_owns_exactly_8_trainable": n_opt == 8,
                  "adam_nonempty_after_step": len(opt.state) > 0,
                  "trunk_digest_unchanged": before["trunk"] == after["trunk"],
                  "early_head_moved": before["early_head"] != after["early_head"],
                  "late_head_moved": before["late_head"] != after["late_head"],
                  "task_head_moved": before["value_task_head"] != after["value_task_head"],
                  "cost_head_moved": before["value_cost_head"] != after["value_cost_head"],
                  "trunk_params_grad_none": trunk_none, "trainable_params_grad_present": train_present}
        g["checks"] = checks; g["clip_norm"] = clip_norm; g["adam_state_entries"] = len(opt.state)
        g["status"] = "passed" if all(checks.values()) else "failed"
    except Exception as e:
        g["status"] = "failed"; g["error"] = f"{type(e).__name__}: {e}"
        verdict["errors"].append(f"B1-G4: {type(e).__name__}: {e}")
    g["offline_rows"] = rows() - r0; g["physical_steps"] = steps() - s0
    results["B1-G4"] = g

    # ---------------- B1-G5 full restore + resume contract ----------------
    g = {"id": "B1-G5", "name": "full_restore_and_resume_contract"}
    r0, s0 = rows(), steps()
    ckpt_dir = receipts / "gate_checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "r74_b1_gate_s11.pt"
    try:
        if "opt" not in dir():
            raise RuntimeError("G4 optimizer unavailable")
        state = M.fresh_training_state(11, live)
        M.save_checkpoint(c, opt, state, ckpt, live=live)
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        save = {"schema_is_r74": payload.get("schema") == M.R74_STATE_SCHEMA,
                "trainer_tag_is_r74": payload.get("trainer_tag") == M.R74_TRAINER_TAG,
                "structure_is_r74": payload.get("structure") == M.R74_STRUCTURE,
                "optimizer_param_groups_2222": payload.get("optimizer_param_groups") == [2, 2, 2, 2],
                "frozen_mask_frozen_scalar_5440": payload.get("frozen_mask", {}).get("frozen_scalar_count") == 5440}
        src_param = M.module_parameter_digest(c); src_adam = _adam_digest(opt); src_rng = trainer.rng_state()
        c2, opt2, ts2 = M.load_checkpoint(ckpt, live=live)
        meter_c2 = make_meter(c2, "b1_s11_restored_trunk_forward")
        restore = {"param_match": M.module_parameter_digest(c2) == src_param,
                   "adam_match": _adam_digest(opt2) == src_adam, "rng_match": trainer.rng_state() == src_rng,
                   "lambda_is_1": float(ts2["lambda"]) == 1.0,
                   "state_schema_r74": ts2["state_schema"] == M.R74_STATE_SCHEMA,
                   "restored_optimizer_owns_8": sum(len(gp["params"]) for gp in opt2.param_groups) == 8}
        def _one(controller, optimizer, what):
            loss, _ = M.minibatch_loss(controller, [early_s, late_s], beta=live["GATE_BETA"], common=live, want_grads=False)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(M.trainable_parameters(controller), live["MAX_GRAD_NORM"])
            make_charged_step(_PRISTINE_ADAM_STEP, res.synth_steps, what=what)(optimizer)
            optimizer.zero_grad(set_to_none=True)
        _one(c, opt, "b1_g5_source_next_update"); _one(c2, opt2, "b1_g5_restore_next_update")
        det = {"same_update_param_identical": M.module_parameter_digest(c) == M.module_parameter_digest(c2),
               "same_update_adam_identical": _adam_digest(opt) == _adam_digest(opt2)}
        refused_exc = ""
        try:
            M.load_checkpoint(M.r71_init_path(11), live=live); refused = False
        except Exception as e:
            refused = True; refused_exc = type(e).__name__
        resume = {"foreign_r71_init_full_resume_refused": refused is True, "self_r74_accepted": True}
        g["save_checks"] = save; g["restore_checks"] = restore; g["determinism_checks"] = det
        g["resume_checks"] = resume; g["foreign_refusal_exception"] = refused_exc
        allc = {}; [allc.update(d) for d in (save, restore, det, resume)]
        g["status"] = "passed" if all(allc.values()) else "failed"
    except Exception as e:
        g["status"] = "failed"; g["error"] = f"{type(e).__name__}: {e}"
        verdict["errors"].append(f"B1-G5: {type(e).__name__}: {e}")
    g["offline_rows"] = rows() - r0; g["physical_steps"] = steps() - s0
    results["B1-G5"] = g
    verdict["counterexamples"]["B1-G5"] = "loading the R71 calibration INIT as a full-resume source must raise; a divergent source-vs-restore param/adam digest would mean the checkpoint does not carry full state"

    for h in (meter_c, meter_c2):
        if h is not None:
            try: h.remove()
            except Exception: pass

    after = res.snapshots()
    verdict["gates"] = results; verdict["after"] = after
    verdict["env_constructions"] = after["env_slots"]["used"]
    verdict["synthetic_steps_charged_this_run"] = after["synthetic_physical_steps"]["used"] - verdict["before"]["synthetic_physical_steps"]["used"]
    verdict["offline_rows_charged_this_run"] = after["offline_forward_target_rows"]["used"] - verdict["before"]["offline_forward_target_rows"]["used"]
    verdict["all_gates_passed"] = all(results[k].get("status") == "passed" for k in results)
    verdict["env_is_zero"] = (after["env_slots"]["used"] == 0)
    verdict["exit_code"] = 0 if (verdict["all_gates_passed"] and verdict["env_is_zero"] and not verdict["errors"]) else 1
    (receipts / "ROUND74_PHASE_B1_VERDICT.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixture", "production"], required=True)
    ap.add_argument("--fixture-root", default=None)
    args = ap.parse_args()
    v = run_all(args.mode, args.fixture_root)
    print("MODE", v["mode"], "EXIT", v["exit_code"], "ALL_PASS", v["all_gates_passed"],
          "ENV", v["env_constructions"], "SYNTH_STEPS", v["synthetic_steps_charged_this_run"],
          "OFFLINE_ROWS", v["offline_rows_charged_this_run"])
    for k in sorted(v["gates"]):
        print("  ", k, v["gates"][k].get("status"),
              "steps=", v["gates"][k].get("physical_steps"), "rows=", v["gates"][k].get("offline_rows"))
    sys.exit(v["exit_code"])


if __name__ == "__main__":
    main()