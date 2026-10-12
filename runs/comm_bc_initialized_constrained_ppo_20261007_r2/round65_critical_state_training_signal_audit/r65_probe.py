"""ROUND65 focal policy forward probe: read-only CPU forward at the focal 19-D vector for the
three compliant R60 seeds at ep0 / ep64 / ep128. Exactly 9 batched model forwards total.

Zero-environment and zero-training-side-effect: it reuses the REAL R60 partitioned controller and
routing (r60_partitioned.configure_source_imports + fresh_partitioned_controller + load_state_dict
of the checkpoint's controller module), never a hand-written network. It does NOT call environment
setup, a training/eval launcher, or restore optimizer / training RNG / lambda / cursor. Model-only
parameter loads + torch.no_grad inference only.
"""
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r65_audit as A
import r65_common as C

RUN = HERE.parent
R60 = RUN / "round60_partitioned_actor_head_ablation"
CK = R60 / "raw" / "checkpoints"
SEEDS = [11, 22, 33]
MILESTONES = [("ep0", "before_b0"), ("ep64", "after_b15"), ("ep128", "ep128")]
GROUND_TRUTH = {"seed": 33, "milestone": "ep128", "p_send": 0.19370390474796295}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_identity_sha(seed, milestone, batches):
    """Expected controller parameter SHA from the compliant identity chain (never guessed)."""
    if milestone == "ep0":
        idn = json.load(open(CK / f"r60_ep0_identity_s{seed}.json", encoding="utf-8"))
        return idn["parameter_sha256"], "r60_ep0_identity file (R58-ep0-derived NEW init)"
    if milestone == "ep64":
        return batches[(seed, 15)]["parameters_after_sha"], "batch b15 parameters_after_sha"
    if milestone == "ep128":
        return batches[(seed, 31)]["parameters_after_sha"], "batch b31 parameters_after_sha"
    raise ValueError(milestone)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args(); out = Path(args.out)

    # Authoritative focal parsed from the R64 tape (validated against the constant, abort on drift).
    focal32, focal_bytes, prov = A.parse_focal()

    # Real R60 partitioned controller + routing, canonical modules SHA-verified.
    sys.path.insert(0, str(R60))
    import r60_partitioned as r60
    binding = r60.configure_source_imports()
    import v2_controller as _vc
    canonical = {
        "v2_controller_file": getattr(_vc, "__file__", None),
        "v2_trainer_sha256": r60.EXPECTED_TRAINER_SHA256,
        "v2_common_sha256": r60.EXPECTED_COMMON_SHA256,
        "v2_controller_sha256": r60.EXPECTED_CONTROLLER_SHA256,
    }

    _, batches, _ = A.build_index()
    batch = torch.tensor(focal32.reshape(1, -1), dtype=torch.float32)

    results, forwards_used, gt_checked = [], 0, None
    for seed in SEEDS:
        for milestone, tag in MILESTONES:
            path = CK / f"v2_train_s{seed}_{tag}.pt"
            rec = {"seed": seed, "milestone": milestone, "checkpoint_file": path.name,
                   "checkpoint_file_sha256": sha256_file(path) if path.exists() else None,
                   "status": "pending"}
            if not path.exists():
                rec["status"] = "missing"; results.append(rec); continue
            exp_sha, exp_src = expected_identity_sha(seed, milestone, batches)
            rec["expected_param_sha256"] = exp_sha; rec["expected_identity_source"] = exp_src
            payload = torch.load(path, map_location="cpu", weights_only=False)
            ctrl = r60.fresh_partitioned_controller(seed, ep0_state=None, verify_ep0=False)
            ctrl.load_state_dict(payload["controller"])
            summ = ctrl.parameter_summary()
            got_sha = summ["parameter_sha256"]
            rec["computed_param_sha256"] = got_sha
            rec["param_identity_match"] = (got_sha == exp_sha)
            rec["structure"] = payload.get("structure")
            # One read-only no_grad forward (the real routed controller).
            with torch.no_grad():
                logits, task_value = ctrl.forward(batch)
                cost_value = ctrl.cost_values(batch)
            forwards_used += 1
            probs = torch.softmax(logits[0], dim=-1)
            p_send = float(probs[1].item()); p_skip = float(probs[0].item())
            greedy = int(torch.argmax(logits[0]).item())
            rec.update({"status": "verified" if rec["param_identity_match"] else "identity_mismatch",
                        "logits": [float(x) for x in logits[0].tolist()],
                        "p_send": p_send, "p_skip": p_skip,
                        "greedy_action": "SEND" if greedy == 1 else "SKIP",
                        "task_value": float(task_value[0].item()),
                        "cost_value": float(cost_value[0].item())})
            if seed == GROUND_TRUTH["seed"] and milestone == GROUND_TRUTH["milestone"]:
                gt_checked = {"expected_p_send": GROUND_TRUTH["p_send"], "got_p_send": p_send,
                              "abs_gap": abs(p_send - GROUND_TRUTH["p_send"]),
                              "match_within_1e-6": abs(p_send - GROUND_TRUTH["p_send"]) < 1e-6}
            results.append(rec)

    # per-seed ep0 -> ep128 change
    by = {(r["seed"], r["milestone"]): r for r in results}
    change = {}
    for seed in SEEDS:
        e0, e128 = by.get((seed, "ep0")), by.get((seed, "ep128"))
        if e0 and e128 and e0.get("p_send") is not None and e128.get("p_send") is not None:
            change[str(seed)] = {"p_send_ep0": e0["p_send"], "p_send_ep128": e128["p_send"],
                                 "delta": e128["p_send"] - e0["p_send"],
                                 "greedy_ep0": e0["greedy_action"], "greedy_ep128": e128["greedy_action"]}

    probe = {"kind": "round65-focal-policy-probe/1",
        "focal": prov,
        "canonical_binding": canonical,
        "forward_budget": 9, "forwards_used": forwards_used,
        "ground_truth_check_s33_ep128": gt_checked,
        "note": ("V(s) is a state return, not the marginal value of SEND-SKIP; equal V for the same "
                 "input does not prove sending has no effect. This read-only probe is not a "
                 "trajectory counterfactual and does not establish loss causality."),
        "results": results, "ep0_to_ep128_p_change": change}
    json.dump(probe, open(out / "FOCAL_POLICY_PROBE.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("canonical v2_controller:", canonical["v2_controller_file"])
    print("forwards_used:", forwards_used)
    print("ground truth s33 ep128:", json.dumps(gt_checked, ensure_ascii=False))
    for r in results:
        print(f"  s{r['seed']} {r['milestone']}: {r['status']} p_send={r.get('p_send')} greedy={r.get('greedy_action')} V={r.get('task_value')} Vcost={r.get('cost_value')} idmatch={r.get('param_identity_match')}")
    print("ep0->ep128 change:", json.dumps(change, ensure_ascii=False))


if __name__ == "__main__":
    main()
