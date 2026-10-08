"""ROUND57 read-only attribution analysis. Zero env/sim/optimizer/checkpoint ops."""
from __future__ import annotations
import json, sys, hashlib
from pathlib import Path
from typing import Any
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
R53 = REPO / "runs/comm_bc_initialized_constrained_ppo_20261007_r2/round53_critic_isolation_trial_readiness"
R56 = REPO / "runs/comm_bc_initialized_constrained_ppo_20261007_r2/round56_soft_early_report_prior"
R53RAW, R56RAW = R53 / "raw", R56 / "raw"
SEEDS = (11, 22, 33)
SCENES = tuple(range(1_301_000, 1_301_016))
LATE_CUT, HORIZON = 147.0, 1000.0
ZERO_OPS = {"environment_constructor_calls": 0, "simulation_runs": 0, "new_flows": 0,
            "optimizer_steps": 0, "checkpoint_writes": 0, "new_episode_rows": 0,
            "cpu_model_forwards": 0, "autograd_grad_diagnoses": 0}
sys.path.insert(0, str(R56))
sys.path.insert(0, str(R53))
sys.path.insert(0, str(REPO / "runs/comm_rule_warmstart_cost_learning_20261006_r2"))
import r45_bc  # noqa: E402
r45_bc.configure_source_imports()
import v2_controller  # noqa: E402
from r53_execute import read_json, verify_baseline_reuse  # noqa: E402
from r53_contracts import sha256_file  # noqa: E402
import r56_contracts  # noqa: E402
TV56 = r56_contracts.TRAINING_VERSION
TV53 = TV56.replace("+r56-soft-early-report-prior-beta0p10/1", "")

def metrics(row):
    ids = sorted({e["task_id"] for e in row.get("task_reward_events", []) if e.get("on_time") and e.get("credited")})
    firsts = int(sum(1 for it in row.get("emission_sequence", []) if len(it) >= 6 and it[5] is True))
    sent = {str(it[1]): int(it[3]) for it in row.get("emission_sequence", []) if len(it) >= 6 and it[5] is True}
    return {"on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
            "first_reports": firsts, "ids": ids, "sent": sent}

def load_row(entry):
    p = entry["path"] if isinstance(entry, dict) else entry
    return read_json(p)

def main():
    # ---------- Task A: input identity ----------
    r56_rows, r53_rows = {}, {}
    for path in sorted((R56RAW / "rows").glob("v2_r44_eval_new_s*_a1.json")):
        row = read_json(path)
        assert row.get("training_version") == TV56, path
        r56_rows.setdefault((int(row["training_seed"]), str(row["eval_mode"])), {})[int(row["scenario_seed"])] = metrics(row)
    for path in sorted((R53RAW / "rows").glob("v2_r44_eval_new_s*_a1.json")):
        row = read_json(path)
        assert row.get("training_version") == TV53, path
        r53_rows.setdefault((int(row["training_seed"]), str(row["eval_mode"])), {})[int(row["scenario_seed"])] = metrics(row)
    identity_ok = (len(r56_rows) == 6 and len(r53_rows) == 6 and
                   all(set(v) == set(SCENES) for m in (r56_rows, r53_rows) for v in m.values()))
    ep128_ids = {}
    for seed in SEEDS:
        s = read_json(R56RAW / f"ROUND46_TRAINING_s{seed}.json")
        ep128_ids[str(seed)] = s["ep128_checkpoint_sha256"]

    # ---------- Task B: decomposition R56-R53 and R56-late ----------
    baseline = verify_baseline_reuse()
    arms = {}
    for label, name in (("immediate_dev", "immediate"), ("late_dev", "late")):
        for entry in baseline["development_references"][label]:
            arms.setdefault((name, 0, "greedy"), {})[int(entry["scene"])] = metrics(load_row(entry))
    def decomp(armA, armB, seed, mode):
        per_scene, totals = {}, {"A": 0, "B": 0, "C": 0, "delta": 0}
        for scene in SCENES:
            a, b = armA[scene]["sent"], armB[scene]["sent"]
            A = sum(v for k, v in a.items() if k not in b)
            B = sum(v for k, v in b.items() if k not in a)
            C = sum(a[k] - b[k] for k in a.keys() & b.keys())
            delta = armA[scene]["spent_bytes"] - armB[scene]["spent_bytes"]
            assert A - B + C == delta
            per_scene[str(scene)] = {"delta": delta, "A": A, "B": B, "C": C,
                "A_targets": {k: v for k, v in a.items() if k not in b},
                "B_targets": {k: v for k, v in b.items() if k not in a},
                "C_targets": {k: a[k] - b[k] for k in a.keys() & b.keys() if a[k] != b[k]}}
            for k in ("A", "B", "C", "delta"): totals[k] += per_scene[str(scene)][k]
        return {"totals": totals, "per_scene": per_scene}
    decompositions = {"r56_minus_r53": {}, "r56_minus_late": {}}
    for seed in SEEDS:
        for mode in ("greedy", "sampled"):
            decompositions["r56_minus_r53"][f"s{seed}/{mode}"] = decomp(
                r56_rows[(seed, mode)], r53_rows[(seed, mode)], seed, mode)
            decompositions["r56_minus_late"][f"s{seed}/{mode}"] = decomp(
                r56_rows[(seed, mode)], arms[("late", 0, "greedy")], seed, mode)

    # ---------- Task C: same-input probe ----------
    sys.path.insert(0, str(R56.parent / "round53_critic_isolation_trial_readiness"))
    import r45_bc  # noqa: E402
    r45_bc.configure_source_imports()
    import v2_controller  # noqa: E402
    models = {}
    for rnd, rawdir, tv in (("r53", R53RAW, TV53), ("r56", R56RAW, TV56)):
        for seed in SEEDS:
            payload = torch.load(rawdir / "checkpoints" / f"v2_train_s{seed}_ep128.pt",
                                 map_location="cpu", weights_only=False)
            assert payload.get("training_version") == tv
            c = v2_controller.fresh_controller(seed)
            sd = payload["controller"]
            c.inner.trunk.load_state_dict(sd["trunk"])
            c.inner.action_head.load_state_dict(sd["action_head"])
            c.inner.value_head.load_state_dict(sd["value_task_head"])
            c.cost_head.load_state_dict(saved := sd["value_cost_head"])
            models[(rnd, seed)] = c
    def p_send(model, feat):
        ZERO_OPS["cpu_model_forwards"] += 1
        with torch.no_grad():
            logits = model.inner.action_head(model.inner.trunk(
                torch.as_tensor([feat], dtype=torch.float32)))
        return float(torch.softmax(logits, -1)[0][1])
    probe, self_check_max = {}, 0.0
    decision_count = 0
    for rnd, rawdir, rows in (("r53", R53RAW, r53_rows), ("r56", R56RAW, r56_rows)):
        for seed in SEEDS:
            for mode in ("greedy", "sampled"):
                pts = []
                for scene in SCENES:
                    src = rawdir / "rows" / f"v2_r44_eval_new_s{seed}_{mode}_s{scene}_a1.json"
                    erow = read_json(src)
                    tape = Path(erow["artefacts"]["tape"])
                    for line in tape.read_text(encoding="utf-8").strip().splitlines():
                        t = json.loads(line)
                        for target in t.get("free_one_shot_choices", []):
                            i = list(t["actions"].keys()).index(target)
                            f = [float(v) for v in t["features"][i]]
                            ready = int(round(float(f[18]) * 1000.0)) - int(round(float(f[0])))
                            saved = float(t["send_probabilities"][target])
                            own = models[(rnd, seed)]
                            recomputed = p_send(own, f)
                            self_check_max = max(self_check_max, abs(recomputed - saved))
                            pts.append({"scene": scene, "target": target,
                                        "step": int(t["step"]), "ready": ready,
                                        "bucket": "early" if ready <= LATE_CUT else "late",
                                        "p_own": saved, "p_own_recomputed": recomputed,
                                        "vec": json.dumps(f, separators=(",", ":"))})
                            decision_count += 1
                probe[f"{rnd}/s{seed}/{mode}"] = pts
    # cross-model comparison on deduped vectors per source panel
    probe_out = {"kind": "round57-same-input-policy-probe/1", "self_check_max_abs_error": self_check_max,
                 "decision_points": decision_count, "panels": {}}
    for srckey, pts in probe.items():
        rnd = srckey.split("/")[0]
        seed = int(srckey.split("/")[1][1:])
        other = "r56" if rnd == "r53" else "r53"
        seen, uniq = {}, []
        for p in pts:
            if p["vec"] in seen:
                continue
            seen[p["vec"]] = True
            uniq.append(p)
        out_rows = []
        for p in uniq:
            p53 = p_send(models[("r53", seed)], json.loads(p["vec"]))
            p56 = p_send(models[("r56", seed)], json.loads(p["vec"]))
            a53 = int(p53 > 0.5); a56 = int(p56 > 0.5)
            out_rows.append({**{k: p[k] for k in ("scene", "target", "step", "ready", "bucket")},
                             "p53_send": p53, "p56_send": p56, "delta": p56 - p53,
                             "action_r53": a53, "action_r56": a56,
                             "transition": ("SEND->SKIP" if (a53, a56) == (1, 0)
                                            else "SKIP->SEND" if (a53, a56) == (0, 1) else "unchanged")})
        early = [r for r in out_rows if r["bucket"] == "early"]
        late = [r for r in out_rows if r["bucket"] == "late"]
        probe_out["panels"][srckey] = {
            "recorded": len(pts), "deduped": len(uniq),
            "early": {"n": len(early),
                      "mean_p53": float(np.mean([r["p53_send"] for r in early])) if early else None,
                      "mean_p56": float(np.mean([r["p56_send"] for r in early])) if early else None,
                      "mean_delta": float(np.mean([r["delta"] for r in early])) if early else None,
                      "SEND->SKIP": sum(1 for r in early if r["transition"] == "SEND->SKIP"),
                      "SKIP->SEND": sum(1 for r in early if r["transition"] == "SKIP->SEND"),
                      "unchanged": sum(1 for r in early if r["transition"] == "unchanged")},
            "late": {"n": len(late),
                     "mean_p53": float(np.mean([r["p53_send"] for r in late])) if late else None,
                     "mean_p56": float(np.mean([r["p56_send"] for r in late])) if late else None,
                     "mean_delta": float(np.mean([r["delta"] for r in late])) if late else None,
                     "SEND->SKIP": sum(1 for r in late if r["transition"] == "SEND->SKIP"),
                     "SKIP->SEND": sum(1 for r in late if r["transition"] == "SKIP->SEND"),
                     "unchanged": sum(1 for r in late if r["transition"] == "unchanged")},
            "rows": out_rows}
        ZERO_OPS["cpu_model_forwards"] += 2 * len(uniq)

    # ---------- Task D: prior exposure from real batch records ----------
    exposure = {"kind": "round56-prior-exposure/1", "per_batch": [], "missing": []}
    for seed in SEEDS:
        for batch in range(32):
            rec = read_json(R56RAW / "checkpoints" / f"r40_batch_s{seed}_b{batch}_attempt01.json")
            upd = rec["update"]
            entry = {"seed": seed, "batch": batch, "gate_loss": upd.get("gate_loss"),
                     "samples": upd.get("samples"), "optimizer_steps": upd.get("optimizer_steps")}
            exposure["per_batch"].append(entry)
    gl = [e["gate_loss"] for e in exposure["per_batch"]]
    exposure["gate_loss_stats"] = {"n": len(gl), "nonzero": sum(1 for v in gl if v > 0),
                                   "max": max(gl), "mean": float(np.mean(gl))}
    exposure["missing"] = ["per-minibatch composition (order/advantages) not persisted -> base-vs-prior gradient separation at ep0 is 不可判",
                           "receiver-side message state (accepted/superseded) not persisted"]
    # prior gradient norm at ep0 on batch-0 gate rows (pure computation, no step)
    grad_diag = []
    for seed in SEEDS:
        payload = torch.load(R56RAW / "checkpoints" / f"v2_train_s{seed}_ep0.pt",
                             map_location="cpu", weights_only=False)
        c = v2_controller.fresh_controller(seed)
        sd = payload["controller"]
        c.inner.trunk.load_state_dict(sd["trunk"]); c.inner.action_head.load_state_dict(sd["action_head"])
        c.inner.value_head.load_state_dict(sd["value_task_head"]); c.cost_head.load_state_dict(sd["value_cost_head"])
        gate_terms = []
        for batch in range(1):
            rec = read_json(R56RAW / "checkpoints" / f"r40_batch_s{seed}_b{batch}_attempt01.json")
            for art in rec["episode_artifacts"]:
                tape = Path(art["tape_path"])
                for line in tape.read_text(encoding="utf-8").strip().splitlines():
                    t = json.loads(line)
                    for target in t.get("free_one_shot_choices", []):
                        i = list(t["actions"].keys()).index(target)
                        f = [float(v) for v in t["features"][i]]
                        ready = int(round(float(f[18]) * 1000.0)) - int(round(float(f[0])))
                        if ready <= LATE_CUT:
                            x = torch.as_tensor([f], dtype=torch.float32)
                            logits = c.inner.action_head(c.inner.trunk(x))
                            gate_terms.append(-torch.log_softmax(logits, -1)[0][0])
        if gate_terms:
            loss = torch.stack(gate_terms).sum() / max(1, len(gate_terms))
            grads = torch.autograd.grad(loss, list(c.inner.trunk.parameters()) +
                                        list(c.inner.action_head.parameters()), retain_graph=False,
                                        allow_unused=True)
            gnorm = float(sum(float(g.abs().sum()) for g in grads if g is not None))
        else:
            gnorm = 0.0
        ZERO_OPS["autograd_grad_diagnoses"] += 1
        grad_diag.append({"seed": seed, "batch0_gate_rows": len(gate_terms),
                          "prior_grad_l1_at_ep0": gnorm})
    exposure["prior_grad_l1_at_ep0_batch0"] = grad_diag

    # ---------- write deliverables ----------
    result = {"identity_ok": identity_ok, "ep128_ids": ep128_ids,
              "decompositions": decompositions, "probe": probe_out,
              "exposure": exposure, "zero_ops": ZERO_OPS}
    (HERE / "INPUT_IDENTITY.json").write_text(json.dumps(
        {"kind": "round57-input-identity/1", "rows_verified": identity_ok,
         "r56_training_version": TV56, "r53_training_version": TV53,
         "ep128_checkpoint_sha256": ep128_ids}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "COST_DECOMPOSITION.json").write_text(json.dumps(
        {"kind": "round57-cost-decomposition/1", **decompositions}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "SAME_INPUT_POLICY_PROBE.json").write_text(json.dumps(probe_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "PRIOR_EXPOSURE_AND_SIGNAL.json").write_text(json.dumps(exposure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"identity_ok": identity_ok, "decisions": decision_count,
                      "self_check": self_check_max, "panels": {k: v["panels"] if False else k for k in []},
                      "gate_loss_max": max(gl), "gate_nonzero_batches": sum(1 for v in gl if v > 0),
                      "prior_grad": grad_diag}, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
