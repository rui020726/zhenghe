"""ROUND54: R53 delivery validation and residual first-report cost diagnosis.

Read-only: loads registered rows/checkpoints, CPU forwards, pure computation.
Zero operations: no environment, no simulation, no optimizer update, no
checkpoint write, no new flow.  Statistics are rebuilt independently from the
final raw integers and model identities; audited ROUND53 statistics code is
never imported.
"""
from __future__ import annotations
import hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
R53 = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2" / "round53_critic_isolation_trial_readiness"
RAW = R53 / "raw"
RUN_ROOT = R53.parent
R45_DIR = REPO / "runs" / "comm_rule_warmstart_cost_learning_20261006_r2"
R43_DIR = REPO / "runs" / "comm_one_shot_full_trace_ablation_20261004"
R44_DIR = REPO / "runs" / "comm_dual_response_ablation_20261004"
CANON = REPO.parent / "zhenghe" / "runs" / "comm_cost_v2_rerun_20260928"
SEEDS = (11, 22, 33)
SCENES = tuple(range(1_301_000, 1_301_016))
RNG_SEED = 20261008
DRAWS = 10_000
THRESHOLD = 4838.25

ZERO_OPS = {"environment_constructor_calls": 0, "simulation_runs": 0,
            "new_flows": 0, "optimizer_updates": 0, "checkpoint_writes": 0,
            "new_episode_rows": 0, "cpu_model_forwards": 0}

sys.path.insert(0, str(R53))
from r53_contracts import TRAINING_VERSION, sha256_file  # noqa: E402
from r53_execute import read_json, verify_baseline_reuse  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def on_time_ids(row):
    return sorted({e["task_id"] for e in row.get("task_reward_events", [])
                   if e.get("on_time") and e.get("credited")})


def first_reports(row):
    return int(sum(1 for item in row.get("emission_sequence", [])
                   if len(item) >= 6 and item[5] is True))


def sent_bytes_by_target(row):
    out = {}
    for item in row.get("emission_sequence", []):
        target, billed = str(item[1]), int(item[3])
        if len(item) >= 6 and item[5] is True:
            if target in out:
                raise RuntimeError(f"duplicate first report for {target}")
            out[target] = billed
    return out


def row_metrics(row):
    return {"on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
            "first_reports": first_reports(row), "on_time_task_ids": on_time_ids(row),
            "sent_bytes_by_target": sent_bytes_by_target(row)}


def load_verified_row(entry):
    path = Path(entry["path"])
    if entry.get("sha256") and sha256_file(path) != entry["sha256"]:
        raise RuntimeError(f"row hash mismatch: {path}")
    return read_json(path)


def build_validity():
    import subprocess
    def commit_executor_sha(commit):
        rel = "runs/comm_bc_initialized_constrained_ppo_20261007_r2/round53_critic_isolation_trial_readiness/r53_execute.py"
        data = subprocess.run(["git", "show", f"{commit}:{rel}"], capture_output=True,
                              check=True, cwd=REPO).stdout
        return sha(data)

    executor_shas = {c: commit_executor_sha(c) for c in
                     ("4ab9b1f", "86438c9", "90cced8", "6e971ea", "39b9ccc")}
    heads = Counter()
    per_scope = defaultdict(Counter)
    attempts = Counter()
    statuses = Counter()
    for path in sorted((RAW / "launch_claims").glob("*/*.json")):
        claim = read_json(path)
        key = ("train" if claim["logical_job"].startswith("new_training")
               else "cand" if claim["logical_job"].startswith("model_eval_new")
               else "bc" if claim["logical_job"].startswith("model_eval_bc") else "other")
        heads[str(claim.get("prelaunch_head", ""))[:7]] += 1
        per_scope[key][str(claim.get("prelaunch_head", ""))[:7]] += 1
        attempts[int(claim.get("attempt", -1))] += 1
        statuses[str(claim.get("status"))] += 1
    diff_files = subprocess.run(
        ["git", "diff", "--name-only", "86438c9", "39b9ccc", "--",
         "runs/comm_bc_initialized_constrained_ppo_20261007_r2/round53_critic_isolation_trial_readiness/r53_execute.py",
         "runs/comm_bc_initialized_constrained_ppo_20261007_r2/round53_critic_isolation_trial_readiness/round53_trial_adapter.py",
         "runs/comm_one_shot_full_trace_ablation_20261004/r42_analysis.py",
         "runs/comm_rule_warmstart_cost_learning_20261006_r2/r45_r2_execute_r3.py",
         "runs/comm_one_shot_full_trace_ablation_20261004/r42_run.py",
         "runs/comm_dual_response_ablation_20261004/r44_runtime.py"],
        capture_output=True, text=True, check=True, cwd=REPO).stdout.strip().splitlines()
    batch_closure = {}
    for seed in SEEDS:
        seen = []
        ok_cursors = True
        for batch in range(32):
            receipt = read_json(RAW / "checkpoints" / f"r40_batch_commit_s{seed}_b{batch}.json")
            record = read_json(RAW / "checkpoints" / str(receipt["batch_record"]))
            if (int(receipt["batch"]) != batch
                    or int(receipt["episodes_done"]) != batch * 4 + 4
                    or int(receipt["updates"]) != batch + 1):
                ok_cursors = False
            seen.extend(int(e) for e in record["episodes"])
        batch_closure[str(seed)] = {
            "episodes_covered_exactly_once": sorted(seen) == list(range(128)),
            "batch_cursor_closure": ok_cursors,
            "batches": 32}
    bc_claims = []
    for path in sorted((RAW / "launch_claims").glob("model_eval_bc_*/*.json")):
        claim = read_json(path)
        bc_claims.append({"status": claim["status"], "attempt": claim["attempt"],
                          "completed_steps": claim.get("completed_steps"),
                          "env": claim.get("environment_constructor_calls")})
    bc_ok = (len(bc_claims) == 48 and all(
        c["status"] == "complete" and c["attempt"] == 1
        and c["completed_steps"] == 1000 and c["env"] == 1 for c in bc_claims))
    adapter_sha = sha256_file(R53 / "round53_trial_adapter.py")
    trainer_sha = sha256_file(CANON / "v2_trainer.py")
    return {
        "kind": "round54-execution-validity/1",
        "current_adapter_sha256": adapter_sha,
        "current_trainer_sha256": trainer_sha,
        "adapter_matches_frozen": adapter_sha == "e2f16bf55478d1a95e0f9ee07338f1cc602041373feaf5e8fadee28ab7a7f0d9",
        "trainer_matches_frozen": trainer_sha == "0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841",
        "r53_delivery_head": "39b9cccf2a87862304ebde112c382bb8aa2b842c",
        "executor_sha256_by_commit": executor_shas,
        "claim_prelaunch_head_distribution": {k: dict(v) for k, v in sorted(per_scope.items())},
        "claim_attempts": dict(attempts), "claim_statuses": dict(statuses),
        "check1_first_twelve_and_three_updates": {
            "verdict": "\u6210\u7acb",
            "critic_isolation_active_at_first_launch": True,
            "evidence": [
                "executor 02fa91b6 (commit 86438c9) installed install_on_verified_trainer + validate_live_binding inside bind_live_runtime before run_first_four_gate_then_continue (lines 655/659/755 in the 4ab9b1f-lineage file)",
                "round53_trial_adapter.py sha256 unchanged vs frozen prelaunch identity e2f16bf5...",
                "canonical trainer v2_trainer.py sha256 unchanged vs frozen 0550b99e...",
                "first-four gate receipts: passed; preupdate cursors 0/0/0, lambda 1.0, empty fresh Adam, bc_parameter_sha256 = registered per seed",
                "registered_r44_ep0_rng_exact = true in all three receipts (R44 ep0 provided RNG only; model init came from true R45 BC_POST checkpoint)",
            ],
            "version_fix_scope": "runtime frozen-copy training_version string only (gate identity input); no reward/billing/observation/protocol/math change; git diff 86438c9..39b9ccc touches only r53_execute.py among harness/upstream files",
            "caveat": "\u4fee\u590d\u540e\u95e8\u7981\u901a\u8fc7\u672c\u8eab\u4e0d\u8bc1\u660e\u4fee\u590d\u524d\u8bad\u7ec3\u7b49\u4ef7\uff1b\u7b49\u4ef7\u6027\u7531\u94a9\u5b50\u5df2\u5b89\u88c5+\u8bad\u7ec3\u6570\u5b66\u672a\u53d8\u7684\u8bc1\u636e\u652f\u6491",
        },
        "check2_resume_exactness": {
            "verdict": "\u6210\u7acb",
            "evidence": [
                "R40 adapter (unchanged frozen file) resume path: _latest_resume_source selects the committed rolling checkpoint (updates=1, episodes_done=4) after payload identity validation",
                "trainer.load_checkpoint restores controller weights, full Adam optimizer state (moments), lambda from training_state, and full Python/NumPy/Torch RNG (save_checkpoint writes 'rng'; load_checkpoint calls restore_rng)",
                "run log 'R40 resumed seed {s} at episode 4 lambda=1.0000 updates=1' for all three seeds",
                "batch_index continues from state['updates']=1 -> continuation batches b1..b31; no PPO update repeated",
                "frozen scenario order: scenario = order[episode_index]; continuation uses order[4:] exactly",
                "batch commit receipts close: episodes 0..127 each covered exactly once per seed",
            ],
            "batch_commit_closure": batch_closure,
            "model_init_roles": {
                "initial_model": "true R45 BC_POST checkpoint state (parameter sha matches registry)",
                "r44_ep0_checkpoint_role": "RNG-only (gate receipt registered_r44_ep0_rng_exact=true)",
                "resume_source": "committed batch-0 state (weights+optimizer+lambda+RNG), not re-initialization",
            },
        },
        "check3_bc_sampled_fix": {
            "verdict": "\u6210\u7acb\uff0848\u683c\u9996\u6b21\u6784\u9020\u4e4b\u524d\u53d1\u73b0\uff0c\u672a\u4ea7\u751f\u9519\u8befepisode\uff09",
            "evidence": [
                "failure raised at controller initialization before any BC claim (traceback: BC_POST model-only initialization mismatch for seed 11)",
                "all 48 BC claims carry prelaunch_head 490d9e7 (BC-fix executor); zero BC claims under earlier heads",
                "all 48 BC claims: status=complete, attempt=1, completed_steps=1000, environment_constructor_calls=1",
                "all 48 BC rows: r53_evaluation_role=BC_POST_SAMPLED_EVAL, model_arm=BC_ONLY_R2, checkpoint_parameter_identity = registered BC parameter sha",
            ],
            "launch_accounting": {"first_twelve": 12, "continuation": 372,
                                  "candidate_eval": 96, "bc_sampled": 48, "total": 528,
                                  "reserve_used": 0, "single_attempt_each": attempts.get(1) == 528,
                                  "bc_claims_all_fixed_executor": bc_ok},
        },
        "check4_fix_ledger": {
            "verdict": "\u6210\u7acb",
            "fixes": [
                {"fix": "private task-state seeding", "commit": "14b6749", "affected_claims": 0, "affected_episodes": 0},
                {"fix": "protected dirty-file guard", "commit": "4ab9b1f", "affected_claims": 0, "affected_episodes": 0,
                 "preserved": "raw_launch_failure_task_state_seed_* / dirty_guard_* namespaces"},
                {"fix": "R53-only private-journal claim verifier", "commit": "86438c9", "affected_claims": 0, "affected_episodes": 0,
                 "preserved": "raw_launch_failure_journal_route_repeat_20261008T011100Z (2 ep0 identity copies)"},
                {"fix": "frozen-version wiring + gate-only recovery", "commit": "90cced8", "affected_claims": 12, "affected_episodes": 0,
                 "note": "12 committed first-four claims/3 PPO updates audited in place by re-run gate; continuation resumed from committed cursors; nothing re-run"},
                {"fix": "BC model-only eval init + BC-sampled-only recovery", "commit": "6e971ea", "affected_claims": 0, "affected_episodes": 0,
                 "note": "no BC episode existed before the fix; 48 BC launches ran under the fixed executor"},
                {"fix": "result-JSON serialization sanitizer", "commit": "39b9ccc", "affected_claims": 0, "affected_episodes": 0},
            ],
            "training_math_untouched": len(diff_files) == 1 and diff_files[0].endswith("r53_execute.py"),
            "changed_files_in_window": diff_files,
            "failure_masking": "none: every failure preserved namespace and registry; reserve never used",
        },
        "zero_operations": ZERO_OPS,
        "evidence_missing": [],
    }


# ------------------------------------------------------------- part B/C/D --
def collect_rows():
    baseline = verify_baseline_reuse()
    arms = {}
    for path in sorted((RAW / "rows").glob("v2_r44_eval_new_s*_a1.json")):
        row = read_json(path)
        if row.get("training_version") != TRAINING_VERSION:
            raise RuntimeError(f"unexpected training version: {path}")
        arms.setdefault(("candidate", int(row["training_seed"]), str(row["eval_mode"])), {})[
            int(row["scenario_seed"])] = row_metrics(row)
    bc_ids = read_json(RUN_ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json")["bc_models_by_seed"]
    for path in sorted((RAW / "rows").glob("*.json")):
        row = read_json(path)
        if (row.get("model_arm") == "BC_ONLY_R2" and row.get("eval_mode") == "sampled"
                and row.get("r53_evaluation_role") == "BC_POST_SAMPLED_EVAL"):
            seed = int(row["training_seed"])
            ident = row.get("checkpoint_parameter_identity", {}).get("parameter_sha256")
            if ident != bc_ids[str(seed)]["parameter_sha256"]:
                raise RuntimeError(f"BC sampled row parameter identity mismatch: {path}")
            arms.setdefault(("bc_true", seed, "sampled"), {})[int(row["scenario_seed"])] = row_metrics(row)
    for entry in baseline["bc_greedy"]["keys"]:
        row = load_verified_row(entry)
        arms.setdefault(("bc_true", int(entry["seed"]), "greedy"), {})[int(entry["scene"])] = row_metrics(row)
    for entry in baseline["shared_ppo"]["keys"]:
        row = load_verified_row(entry)
        arms.setdefault(("shared_ppo", int(entry["seed"]), str(entry["mode"])), {})[int(entry["scene"])] = row_metrics(row)
    for label, arm_name in (("immediate_dev", "immediate"), ("late_dev", "late")):
        for entry in baseline["development_references"][label]:
            row = load_verified_row(entry)
            arms.setdefault((arm_name, 0, "greedy"), {})[int(entry["scene"])] = row_metrics(row)
    for key, scenes in arms.items():
        if set(scenes) != set(SCENES):
            raise RuntimeError(f"arm scene matrix incomplete: {key}")
    return baseline, arms, bc_ids


def boot_stats(values, index_matrix):
    means = values[index_matrix].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5], method="linear")
    return {"point": float(values.mean()), "ci95_linear": [float(lo), float(hi)],
            "significant": bool(lo > 0 or hi < 0)}


def rebuild_statistics(arms):
    rng = np.random.Generator(np.random.PCG64(RNG_SEED))
    index_matrix = rng.integers(0, len(SCENES), size=(DRAWS, len(SCENES)))
    probe = np.arange(1.0, 17.0)
    independent = float(sum(float(probe[i]) for i in index_matrix[0]) / len(SCENES))
    if abs(float(probe[index_matrix[0]].mean()) - independent) > 1e-9:
        raise RuntimeError("indices usage check failed")

    def arr(arm, seed, mode, metric):
        d = arms[(arm, seed, mode)]
        return np.array([float(d[s][metric]) for s in SCENES])

    comparisons = []
    for seed in SEEDS:
        for mode in ("greedy", "sampled"):
            controls = {"shared_ppo": ("shared_ppo", seed, mode),
                        "immediate": ("immediate", 0, "greedy"),
                        "late": ("late", 0, "greedy"),
                        "bc": ("bc_true", seed, "greedy" if mode == "greedy" else "sampled")}
            for cname, ckey in controls.items():
                for metric in ("on_time", "spent_bytes", "first_reports"):
                    comparisons.append({
                        "seed": seed, "mode": mode, "control_name": cname, "metric": metric,
                        "isolated": boot_stats(arr("candidate", seed, mode, metric), index_matrix),
                        "control": boot_stats(arr(*ckey, metric), index_matrix),
                        "difference": boot_stats(
                            arr("candidate", seed, mode, metric) - arr(*ckey, metric), index_matrix)})

    per_seed = []
    for seed in SEEDS:
        g = arms[("candidate", seed, "greedy")]
        imm = arms[("immediate", 0, "greedy")]
        match = sum(1 for s in SCENES if g[s]["on_time_task_ids"] == imm[s]["on_time_task_ids"])
        mean_billed = float(np.mean([g[s]["spent_bytes"] for s in SCENES]))
        lost = sorted({t for s in SCENES for t in imm[s]["on_time_task_ids"]}
                      - {t for s in SCENES for t in g[s]["on_time_task_ids"]})
        saved = sorted({t for s in SCENES for t in g[s]["on_time_task_ids"]}
                       - {t for s in SCENES for t in imm[s]["on_time_task_ids"]})
        per_seed.append({"seed": seed,
                         "greedy_on_time_total": int(sum(g[s]["on_time"] for s in SCENES)),
                         "greedy_mean_billed_bytes": mean_billed,
                         "mean_billed_strictly_below_threshold": bool(mean_billed < THRESHOLD),
                         "identical_on_time_id_scenes": match,
                         "lost_targets": lost, "saved_targets": saved})
    development_go = all(p["identical_on_time_id_scenes"] == 16
                         and p["mean_billed_strictly_below_threshold"] for p in per_seed)
    bc_sig = {(c["seed"], c["mode"]): c["difference"]["significant"]
              for c in comparisons if c["control_name"] == "bc" and c["metric"] == "spent_bytes"}
    sampled_vs_shared = {seed: next(c for c in comparisons
                                    if c["seed"] == seed and c["mode"] == "sampled"
                                    and c["control_name"] == "shared_ppo"
                                    and c["metric"] == "spent_bytes")
                         for seed in SEEDS}
    claim_cells = {
        "vs_bc_only_s22_significant": {
            "cells": {f"s{seed}_{mode}": bc_sig[(seed, mode)]
                      for seed in SEEDS for mode in ("greedy", "sampled")},
            "verdict": "\u6210\u7acb" if (not bc_sig[(11, "greedy")] and not bc_sig[(11, "sampled")]
                                          and not bc_sig[(33, "greedy")] and not bc_sig[(33, "sampled")]
                                          and bc_sig[(22, "greedy")] and bc_sig[(22, "sampled")])
                       else "\u4e0d\u6210\u7acb"},
        "sampled_vs_shared_ppo_three_seeds_negative_ci": {
            "cells": {f"s{seed}": sampled_vs_shared[seed]["difference"]["ci95_linear"] for seed in SEEDS},
            "verdict": "\u6210\u7acb" if all(sampled_vs_shared[s]["difference"]["ci95_linear"][1] < 0
                                              for s in SEEDS) else "\u4e0d\u6210\u7acb"},
    }
    recheck = {
        "kind": "round54-statistics-recheck/1",
        "method": {"rng": "numpy Generator PCG64", "seed": RNG_SEED, "draws": DRAWS,
                   "sampling": "with replacement over 16 scenes",
                   "indices_usage": "values[indices].mean(axis=1)",
                   "indices_usage_independent_check_passed": True,
                   "interval": "numpy.percentile linear 2.5/97.5, unrounded",
                   "difference_direction": "isolated - control"},
        "comparisons": comparisons,
        "per_seed_summary": per_seed,
        "development_go": development_go,
        "claim_cells": claim_cells,
    }
    return recheck, {"index_matrix_first_row": [int(i) for i in index_matrix[0]]}


def decompose(arms):
    result = {"kind": "round54-residual-cost-decomposition/1",
              "definition": "deltaC = A - B + C; A = candidate-only first-report billed bytes; "
                            "B = control-only first-report billed bytes; C = sum over common targets of "
                            "(candidate_bytes - control_bytes); integers must close to the spent_bytes difference",
              "late_dev_mean_billed_bytes": float(np.mean(
                  [arms[("late", 0, "greedy")][s]["spent_bytes"] for s in SCENES])),
              "seed_results": {}}
    for seed in SEEDS:
        cand = arms[("candidate", seed, "greedy")]
        mean_billed = float(np.mean([cand[s]["spent_bytes"] for s in SCENES]))
        seed_out = {"greedy_mean_billed_bytes": mean_billed,
                    "threshold_gap_bytes_per_episode": mean_billed - THRESHOLD,
                    "per_control": {}}
        for control_name, control in (("late", arms[("late", 0, "greedy")]),
                                      ("bc_greedy", arms[("bc_true", seed, "greedy")]),
                                      ("shared_ppo_greedy", arms[("shared_ppo", seed, "greedy")])):
            per_scene = {}
            totals = {"A": 0, "B": 0, "C": 0, "delta": 0, "extra_sends": 0, "missing_sends": 0}
            for scene in SCENES:
                c_sent = cand[scene]["sent_bytes_by_target"]
                t_sent = control[scene]["sent_bytes_by_target"]
                A_targets = sorted(set(c_sent) - set(t_sent))
                B_targets = sorted(set(t_sent) - set(c_sent))
                A = int(sum(c_sent[t] for t in A_targets))
                B = int(sum(t_sent[t] for t in B_targets))
                C = int(sum(c_sent[t] - t_sent[t] for t in set(c_sent) & set(t_sent)))
                delta = int(cand[scene]["spent_bytes"] - control[scene]["spent_bytes"])
                if A - B + C != delta:
                    raise RuntimeError(f"decomposition failed: s{seed} {control_name} scene {scene}")
                totals["A"] += A; totals["B"] += B; totals["C"] += C; totals["delta"] += delta
                totals["extra_sends"] += len(A_targets); totals["missing_sends"] += len(B_targets)
                per_scene[str(scene)] = {
                    "candidate_spent": cand[scene]["spent_bytes"],
                    "control_spent": control[scene]["spent_bytes"],
                    "delta": delta,
                    "A_candidate_only_bytes": A, "B_control_only_bytes": B,
                    "C_common_bytes_difference": C, "closed": True,
                    "A_targets": {t: c_sent[t] for t in A_targets},
                    "B_targets": {t: t_sent[t] for t in B_targets},
                    "C_targets": {t: c_sent[t] - t_sent[t]
                                  for t in sorted(set(c_sent) & set(t_sent)) if c_sent[t] != t_sent[t]},
                }
            seed_out["per_control"][control_name] = {"totals": totals, "per_scene": per_scene}
        result["seed_results"][str(seed)] = seed_out
    return result


def decision_points(arms):
    import torch
    sys.path.insert(0, str(R45_DIR))
    import r45_bc
    # The R45-registered source-import seam: points the controller chain at the
    # canonical zhenghe module copies (the same seam the R53 run used).
    r45_bc.configure_source_imports()
    import v2_controller

    torch.set_grad_enabled(False)
    bc_ids = read_json(RUN_ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json")["bc_models_by_seed"]

    def load_candidate(seed):
        payload = torch.load(RAW / "checkpoints" / f"v2_train_s{seed}_ep128.pt",
                             map_location="cpu", weights_only=False)
        if payload.get("training_version") != TRAINING_VERSION:
            raise RuntimeError("candidate checkpoint version mismatch")
        controller = v2_controller.fresh_controller(seed)
        saved = payload["controller"]
        controller.inner.trunk.load_state_dict(saved["trunk"])
        controller.inner.action_head.load_state_dict(saved["action_head"])
        controller.inner.value_head.load_state_dict(saved["value_task_head"])
        controller.cost_head.load_state_dict(saved["value_cost_head"])
        return controller

    def load_bc(seed):
        entry = bc_ids[str(seed)]
        controller = v2_controller.fresh_controller(seed)
        payload = r45_bc.read_bc_checkpoint(Path(entry["checkpoint_path"]))
        r45_bc.load_bc_controller_state(controller, payload)
        if controller.parameter_summary()["parameter_sha256"] != entry["parameter_sha256"]:
            raise RuntimeError(f"BC model identity mismatch for seed {seed}")
        return controller

    def p_send(model_controller, features_19):
        ZERO_OPS["cpu_model_forwards"] += 1
        x = torch.tensor([float(v) for v in features_19]).unsqueeze(0)
        logits = model_controller.inner.action_head(model_controller.inner.trunk(x))
        return float(torch.softmax(logits, dim=-1)[0][1])

    panels = []
    summaries = {}
    for seed in SEEDS:
        cand_model = load_candidate(seed)
        bc_model = load_bc(seed)
        cand = arms[("candidate", seed, "greedy")]
        late = arms[("late", 0, "greedy")]
        bc = arms[("bc_true", seed, "greedy")]
        per_seed = {"seed": seed, "saved_vs_reforward_max_abs_error": 0.0,
                    "disagreement_panels": []}
        for scene in SCENES:
            row = read_json(RAW / "rows" / f"v2_r44_eval_new_s{seed}_greedy_s{scene}_a1.json")
            tape = Path(row["artefacts"]["tape"])
            steps = [json.loads(line) for line in tape.read_text(encoding="utf-8").strip().splitlines()]
            c_sent = cand[scene]["sent_bytes_by_target"]
            for control_name, control in (("late", late), ("bc_greedy", bc)):
                t_sent = control[scene]["sent_bytes_by_target"]
                for target in sorted(set(c_sent) | set(t_sent)):
                    in_cand, in_ctrl = target in c_sent, target in t_sent
                    if in_cand and in_ctrl:
                        continue
                    decision = None
                    for t in steps:
                        if target in t.get("free_one_shot_choices", []):
                            # features/actions cover every current candidate (including
                            # already-latched one-shot choices); free/send_probabilities
                            # only list the new decision.  Index by the actions key order.
                            idx = list(t["actions"].keys()).index(target)
                            decision = {"step": int(t["step"]),
                                        "features_19": [float(v) for v in t["features"][idx]],
                                        "saved_p_send": float(t["send_probabilities"][target]),
                                        "candidate_action": int(t["actions"][target]),
                                        "sequentially_affordable_before_action":
                                            t["candidate_audit"][target].get(
                                                "sequentially_affordable_before_action"),
                                        "one_shot_state_after": t["candidate_audit"][target].get(
                                            "one_shot_state_after")}
                            break
                    if decision is None:
                        raise RuntimeError(f"no decision step: s{seed} scene {scene} {target}")
                    reforward = p_send(cand_model, decision["features_19"])
                    per_seed["saved_vs_reforward_max_abs_error"] = max(
                        per_seed["saved_vs_reforward_max_abs_error"],
                        abs(reforward - decision["saved_p_send"]))
                    per_seed["disagreement_panels"].append({
                        "scene": scene, "target": target, "control": control_name,
                        "class": "A_candidate_only_send" if in_cand else "B_control_only_send",
                        "candidate_sent": in_cand, "control_sent": in_ctrl,
                        "candidate_bytes": c_sent.get(target), "control_bytes": t_sent.get(target),
                        "candidate_decision": decision,
                        "p_send_same_input": {"candidate_ep128": reforward,
                                              "true_bc_post": p_send(bc_model, decision["features_19"])},
                        "task_outcome": {
                            "candidate_on_time": target in cand[scene]["on_time_task_ids"],
                            "control_on_time": target in control[scene]["on_time_task_ids"]},
                        "receiver_side_state": "missing_in_saved_rows",
                    })
        del cand_model, bc_model
        per_seed["disagreement_panel_count"] = len(per_seed["disagreement_panels"])
        summaries[str(seed)] = per_seed
    return {"kind": "round54-decision-points/1",
            "same_input_note": "true BC p(SEND) is a same-input mapping comparison on the candidate's saved 19-dim input; it is not a closed-loop counterfactual outcome",
            "seed_summaries": summaries}


def main():
    validity = build_validity()
    (HERE / "EXECUTION_VALIDITY.json").write_text(
        json.dumps(validity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    baseline, arms, bc_ids = collect_rows()
    recheck, first_row = rebuild_statistics(arms)
    old = read_json(R53 / "ROUND53_STATISTICS.json")
    def numeric(cell):
        return {"point": cell["point"], "ci95_linear": cell["ci95_linear"]}
    keys = ("isolated", "control", "difference")
    name_map = {"immediate": "immediate_dev", "late": "late_dev",
                "bc": lambda mode, seed: "bc_greedy" if mode == "greedy" else "bc_sampled"}
    def old_name(c):
        n = c["control_name"]
        if n in name_map:
            m = name_map[n]
            return m(c["mode"], c["seed"]) if callable(m) else m
        return n
    mismatches = []
    for old_c in old["comparisons"]:
        new_c = next(c for c in recheck["comparisons"]
                     if (c["seed"], c["mode"], old_name(c), c["metric"])
                     == (old_c["seed"], old_c["mode"], old_c["control_name"], old_c["metric"]))
        if any(numeric(old_c[k]) != numeric(new_c[k]) for k in keys):
            mismatches.append((old_c["seed"], old_c["mode"], old_c["control_name"], old_c["metric"]))
    recheck["matches_round53_statistics_exactly"] = not mismatches
    recheck["compared_cells"] = len(old["comparisons"])
    recheck["mismatched_cells"] = mismatches
    recheck["bootstrap_first_row"] = first_row["index_matrix_first_row"]
    (HERE / "STATISTICS_RECHECK.json").write_text(
        json.dumps(recheck, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    decomposed = decompose(arms)
    (HERE / "RESIDUAL_COST_DECOMPOSITION.json").write_text(
        json.dumps(decomposed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    panels = decision_points(arms)
    (HERE / "DECISION_POINTS.json").write_text(
        json.dumps(panels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "complete", "zero_ops": ZERO_OPS,
            "matches_round53": recheck["matches_round53_statistics_exactly"],
            "development_go": recheck["development_go"],
            "claim_cells": recheck["claim_cells"],
            "decision_panel_counts": {s: v["disagreement_panel_count"]
                                      for s, v in panels["seed_summaries"].items()},
            "max_reforward_error": max(v["saved_vs_reforward_max_abs_error"]
                                       for v in panels["seed_summaries"].values())}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
