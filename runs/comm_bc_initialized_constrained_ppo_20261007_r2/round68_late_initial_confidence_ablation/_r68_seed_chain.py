#!/usr/bin/env python
"""ROUND68 per-seed training-chain eligibility, read-only.

For each seed, proves from the immutable artefacts on disk that the 32 committed batches form one
unbroken chain: parameters_before[b] -> parameters_after[b] -> parameters_before[b+1], that batch 0
starts from the scaled ROUND60-ep0 initialisation, that batch 31 lands exactly on the evaluated
ep128, that the episode cursor covers 0..127 exactly once, that the actor used the pre-update dual
lambda, and that the frozen trunk / early head are bit-identical to ROUND60 ep0 at ep128.

Read-only: it loads checkpoints and the ROUND60 ep0 reference on CPU for identity comparison only.
No model forward, no PPO update, no optimizer step, no environment construction, no checkpoint write.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r68_execute as base   # noqa: E402
import r68_frozen as r68     # noqa: E402

SEEDS = (11, 22, 33)


def verify_seed(seed: int, raw: Path, ck: Path) -> dict:
    recs = []
    for b in range(32):
        p = ck / f"r40_batch_s{seed}_b{b}_attempt01.json"
        if not p.is_file():
            raise RuntimeError(f"seed {seed}: missing batch record b{b}")
        recs.append(json.loads(p.read_text(encoding="utf-8")))
    init = json.loads((ck / f"r68_init_identity_s{seed}.json").read_text(encoding="utf-8"))
    ctrl, state, ident = r68.load_checkpoint_read_only(
        ck / f"v2_train_s{seed}_ep128.pt", freeze=base.R68_FREEZE)

    checks: dict[str, Any] = {}

    # --- episode cursor -------------------------------------------------------------
    seen = [e for r in recs for e in r["episodes"]]
    checks["episodes_cover_0_127_once"] = sorted(seen) == list(range(128))
    checks["episodes_done_after_monotone_4x"] = all(
        r["episodes_done_after"] == 4 * (i + 1) for i, r in enumerate(recs))
    checks["scenarios_per_batch_4"] = all(len(r["scenarios"]) == 4 for r in recs)

    # --- parameter chain ------------------------------------------------------------
    chain = all(recs[i]["parameters_after_sha"] == recs[i + 1]["parameters_before_sha"]
                for i in range(31))
    checks["chain_before_after_contiguous_32"] = chain
    checks["b0_before_is_scaled_init"] = (
        recs[0]["parameters_before_sha"] == init["post_scale_parameter_sha256"])
    checks["init_pre_scale_equals_r60_ep0"] = bool(init["pre_scale_equals_r60_ep0"])
    checks["init_alpha_0p10"] = abs(float(init["late_init_alpha"]) - r68.LATE_INIT_ALPHA) < 1e-12
    checks["init_late_head_scaled_once"] = (
        init["late_head_sha_before_scale"] != init["late_head_sha_after_scale"])
    ep128_sha = ctrl.parameter_summary()["parameter_sha256"]
    checks["b31_after_is_ep128"] = recs[31]["parameters_after_sha"] == ep128_sha
    checks["late_head_moved_from_init"] = (
        r68._module_sha(ctrl, "late_head") != init["late_head_sha_after_scale"])

    # --- dual lambda chain: the actor must see the PRE-update lambda -----------------
    checks["lambda_at_rollout_equals_dual_before"] = all(
        r["lambda_at_rollout"] == r["dual"]["lambda_before"] for r in recs)
    checks["lambda_at_rollout_0_is_init_1p0"] = recs[0]["lambda_at_rollout"] == 1.0
    checks["lambda_chain_rollout_b_equals_next_b_minus_1"] = all(
        recs[i]["lambda_at_rollout"] == recs[i - 1]["dual"]["lambda_next"] for i in range(1, 32))
    lam = [float(r["dual"]["lambda_next"]) for r in recs]
    checks["lambda_monotone_nondecreasing"] = all(lam[i] <= lam[i + 1] for i in range(31))
    checks["lambda_final_matches_training_state"] = lam[-1] == float(state["lambda"])
    checks["lambda_history_len_32"] = len(state.get("lambda_history", [])) == 32
    checks["eta_lambda_0p5"] = float(recs[0]["dual"]["eta"]) == 0.5

    # --- frozen modules bit-identical to ROUND60 ep0 --------------------------------
    r60 = r68.load_r60_model(seed, "ep0")
    ref = r68.fresh_partitioned_controller(seed, ep0_state=None, verify_ep0=False)
    ref.load_state_dict(r60["controller"])
    checks["trunk_ep128_equals_r60_ep0"] = (
        r68._module_sha(ctrl, "trunk") == r68._module_sha(ref, "trunk"))
    checks["early_head_ep128_equals_r60_ep0"] = (
        r68._module_sha(ctrl, "early_head") == r68._module_sha(ref, "early_head"))
    checks["critics_moved_from_r60_ep0"] = (
        r68._module_sha(ctrl, "value_task_head") != r68._module_sha(ref, "value_task_head"))

    # --- optimizer step accounting ---------------------------------------------------
    steps = [int(r["optimizer_steps_after"]) for r in recs]
    deltas = [steps[0]] + [steps[i] - steps[i - 1] for i in range(1, 32)]
    checks["optimizer_steps_positive_each_batch"] = all(d > 0 for d in deltas)
    checks["optimizer_steps_sum_matches_after"] = sum(deltas) == steps[-1]
    checks["episodes_done_128"] = int(state["episodes_done"]) == 128
    checks["updates_32"] = int(state["updates"]) == 32

    return {
        "seed": seed,
        "checks": checks,
        "checks_failed": sorted(k for k, v in checks.items() if v is not True),
        "facts": {
            "ep128_parameter_sha256": ep128_sha,
            "b0_parameters_before_sha": recs[0]["parameters_before_sha"],
            "b31_parameters_after_sha": recs[31]["parameters_after_sha"],
            "init_post_scale_parameter_sha256": init["post_scale_parameter_sha256"],
            "init_pre_scale_parameter_sha256": init["pre_scale_parameter_sha256"],
            "r60_ep0_parameter_sha256": init["r60_ep0_parameter_sha256"],
            "optimizer_steps_after_b31": steps[-1],
            "optimizer_steps_deltas": deltas,
            "lambda_final": lam[-1],
            "lambda_first5": lam[:5],
            "trunk_sha_ep128": r68._module_sha(ctrl, "trunk"),
            "trunk_sha_r60_ep0": r68._module_sha(ref, "trunk"),
            "early_head_sha_ep128": r68._module_sha(ctrl, "early_head"),
            "early_head_sha_r60_ep0": r68._module_sha(ref, "early_head"),
            "late_head_sha_ep128": r68._module_sha(ctrl, "late_head"),
            "late_head_sha_after_scale": init["late_head_sha_after_scale"],
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND68 per-seed chain eligibility")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    raw = Path(args.raw_root).resolve()
    ck = raw / "checkpoints"
    base.configure_and_inject(raw)

    ppo = [json.loads(l) for l in
           (raw / "runtime" / "ppo_update_calls.jsonl").read_text(encoding="utf-8").splitlines()
           if l.strip()]
    opt = [json.loads(l) for l in
           (raw / "runtime" / "optimizer_step_calls.jsonl").read_text(encoding="utf-8").splitlines()
           if l.strip()]
    idx = [x["update_index"] for x in ppo]
    journals = {
        "ppo_update_calls_lines": len(ppo),
        "ppo_update_index_each_0_31_exactly_3_times":
            sorted(idx) == sorted(list(range(32)) * 3),
        "ppo_update_steps_sum": sum(int(x["optimizer_steps"]) for x in ppo),
        "ppo_update_all_r68_structure": all(
            x.get("structure") == r68.R68_STRUCTURE for x in ppo),
        "optimizer_step_calls_lines": len(opt),
        "optimizer_step_all_invoked": all(
            x.get("call_status") == "step_invoked" for x in opt),
        "optimizer_step_all_adam": all(x.get("optimizer_class") == "Adam" for x in opt),
        "optimizer_step_call_index_contiguous": (
            sorted(int(x["call_index"]) for x in opt) == list(range(1, len(opt) + 1))),
    }
    journals["ppo_steps_sum_equals_optimizer_lines"] = (
        journals["ppo_update_steps_sum"] == journals["optimizer_step_calls_lines"])

    # The checkpoint-write journal is NOT instrumented anywhere in this round: no code path in
    # r68_execute.py / r68_frozen.py / r68_guards.py or the v2 modules appends to it. Reporting
    # "0 checkpoint writes" would therefore be a false claim; it is recorded as not-instrumented.
    cw = raw / "runtime" / "checkpoint_writes.jsonl"
    journals["checkpoint_write_journal"] = {
        "path": str(cw),
        "lines": sum(1 for _ in open(cw, encoding="utf-8")) if cw.is_file() else None,
        "status": "NOT_INSTRUMENTED",
        "note": ("no code path in this round appends to raw/runtime/checkpoint_writes.jsonl, so its "
                 "line count is NOT evidence about how many training checkpoints were written. The "
                 "actual training checkpoint writes are unknown from this journal and are counted "
                 "instead from the artefacts on disk (see below); they are not claimed to be 0."),
        "actual_training_checkpoint_files_on_disk": {
            f"s{seed}": {
                "before_b": len(list(ck.glob(f"v2_train_s{seed}_before_b*.pt"))),
                "after_b": len(list(ck.glob(f"v2_train_s{seed}_after_b*.pt"))),
                "rolling": len(list(ck.glob(f"v2_train_s{seed}.pt"))),
                "ep128": len(list(ck.glob(f"v2_train_s{seed}_ep128.pt"))),
            } for seed in SEEDS},
    }

    out = {"kind": "round68-per-seed-chain-eligibility/1",
           "task_id": base.TASK_ID,
           "branch": "exp/round68-late-initial-confidence-ablation",
           "training_version": r68.R68_TRAINING_VERSION,
           "seeds": {str(s): verify_seed(s, raw, ck) for s in SEEDS},
           "journals": journals,
           "environment_constructions": 0, "model_forwards": 0, "ppo_calls": 0,
           "optimizer_steps": 0, "checkpoint_writes": 0,
           "read_only_model_loads_for_identity": 3 * 2}
    all_failed = {s: out["seeds"][s]["checks_failed"] for s in out["seeds"]}
    out["all_seeds_verified"] = all(not v for v in all_failed.values())
    out["checks_failed_by_seed"] = all_failed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    tmp.replace(args.out)
    for s, e in out["seeds"].items():
        print(f"s{s}: {'VERIFIED' if not e['checks_failed'] else 'FAILED ' + str(e['checks_failed'])}"
              f"  b0_before={e['facts']['b0_parameters_before_sha'][:12]}"
              f"  ep128={e['facts']['ep128_parameter_sha256'][:12]}"
              f"  opt_steps={e['facts']['optimizer_steps_after_b31']}"
              f"  lambda_final={e['facts']['lambda_final']}")
    print("journals:", json.dumps({k: v for k, v in journals.items()
                                   if k != "checkpoint_write_journal"}))
    print("all_seeds_verified:", out["all_seeds_verified"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())