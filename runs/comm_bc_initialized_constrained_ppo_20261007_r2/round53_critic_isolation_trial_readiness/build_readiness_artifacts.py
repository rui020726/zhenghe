"""Build ROUND53 static readiness receipts only; this never loads project code."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
REPO = ROOT.parents[1]
CANON = REPO.parent / "zhenghe" / "runs" / "comm_cost_v2_rerun_20260928"
R52 = ROOT / "round52_critic_gradient_isolation_prototype"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def put(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

contract = read(R52 / "PROTOTYPE_CONTRACT.json")
round51 = ROOT / "round51_frozen_source_gradient_close" / "GRADIENT_COMPONENTS.json"
trainer = CANON / "v2_trainer.py"
r46 = ROOT / "r46_execute_r2.py"
v10 = ROOT / "r46_v10_eval_final_close.py"
freeze = read(ROOT / "ROUND46_R2_V6_FROZEN_CONFIG.json")
keyplan = read(ROOT / "ROUND46_R2_V6_KEY_PLAN.json")
identity = read(ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json")
reuse_path = REPO / "runs" / "comm_dual_response_ablation_20261004" / "ROUND44_BASELINE_REUSE.json"
reuse = read(reuse_path)
v10_root = ROOT / "v10_eval_final_close"
matrix_path = v10_root / "ROUND46_R2_V10_MACHINE_VERDICT.json"
matrix = read(matrix_path)
input_manifest = v10_root / "ROUND46_R2_V10_ANALYSIS_INPUT_MANIFEST.json"
results = v10_root / "ROUND46_R2_V10_FINAL_RESULTS.json"

def scene_map(items, predicate=lambda x: True):
    return {str(x["scene"]): x["sha256"] for x in items
            if int(x["scene"]) in range(1301000, 1301016) and predicate(x)}

immediate = scene_map(reuse["round42_immediate_references"])
late = scene_map(reuse["round42_fixed_rules"], lambda x: x.get("arm") == "late_discovery_gate")
bc_index = {"path": str(ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json"),
            "sha256": sha(ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json"),
            "rows": 48, "scope": "BC greedy: 3 seeds x 16 development scenes"}
ppo_index = {"path": str(input_manifest), "sha256": sha(input_manifest),
             "results_path": str(results), "results_sha256": sha(results),
             "machine_verdict_path": str(matrix_path), "machine_verdict_sha256": sha(matrix_path),
             "accepted_effective_rows": matrix.get("effective_matrix", {}).get("effective_rows"),
             "accepted_row_keys_sha256": matrix.get("effective_matrix", {}).get("row_keys_sha256"),
             "scope": "final accepted 3 seeds x greedy/sampled x 16 scenes"}

sources = [trainer, r46, v10, R52 / "PROTOTYPE_CONTRACT.json",
           R52 / "FORWARD_AND_GRADIENT_COMPARISON.json", R52 / "PURE_TEST_EVIDENCE.json",
           round51, ROOT / "ROUND46_R2_V6_FROZEN_CONFIG.json",
           ROOT / "ROUND46_R2_V6_KEY_PLAN.json", ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json",
           input_manifest, results, matrix_path, reuse_path]
put("SOURCE_BINDING.json", {
    "kind": "round53-source-binding/1", "status": "source_hashes_captured_read_only",
    "task_id": "ROUND53-ZERO-LAUNCH-ISOLATED-TRIAL-READINESS",
    "initial_head": "40adea4e16c5ccd6ee454f86462e4c959149da8a",
    "branch": "feat/round53-critic-isolation-trial-readiness",
    "source_files": [{"path": str(p), "sha256": sha(p)} for p in sources],
    "canonical_trainer_sha256_expected": "0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841",
    "canonical_trainer_sha256_observed": sha(trainer),
    "scientific_change": "task/cost critic-head input representation detached from shared trunk for critic-loss gradient only",
    "preserved": ["actor/entropy gradient route", "forward values", "losses and coefficients", "mask and pooling", "data order", "fresh optimizer/dual/RNG contract", "BC_POST four-module initializer identity"],
    "no_claimed_runtime_binding": True
})

put("BASELINE_REUSE_CHECK.json", {
    "kind": "round53-baseline-reuse-check/1", "status": "partial_reuse_verified_missing_one_arm",
    "bc_greedy": bc_index,
    "ppo_final_matrix": ppo_index,
    "immediate_reference": {"manifest": str(reuse_path), "sha256": sha(reuse_path), "scenes": len(immediate), "row_sha256_by_scene": immediate,
                            "training_g_hat_reuse": "not_yet_verified_against all 48 training scenario keys/order"},
    "late_reference": {"manifest": str(reuse_path), "sha256": sha(reuse_path), "scenes": len(late), "row_sha256_by_scene": late},
    "missing": ["BC sampled comparator: 48 rows; cannot be reused from registered index"],
    "reuse_gate": "Each reused row still requires per-row configuration/model/protocol/hash validation; this readiness pass does not open or reinterpret old claim/ledger history.",
    "old_round46_quota_transfer": False
})

normal = {"candidate_training": 384, "candidate_greedy_eval": 48,
          "candidate_sampled_eval": 48, "BC_sampled_comparator_eval": 48}
assert sum(normal.values()) == 528
put("TRIAL_PLAN.json", {
    "kind": "round53-isolated-critic-trial-plan/1", "status": "proposed_not_authorized",
    "authorization": "PENDING_USER_APPROVAL", "development_go": False,
    "single_variable": contract["version"],
    "normal_claims": normal, "normal_total": 528,
    "new_technical_failure_reserve": 24, "hard_cap": 552,
    "quota_identity": "fresh ROUND53 isolated claim/slot/ledger namespace; historical ROUND46 504 cap is not carried over",
    "seeds": [11, 22, 33], "training_episodes_per_seed": 128,
    "evaluation_scenes": list(range(1301000, 1301016)),
    "reuse_only_after_row_identity_check": True,
    "sealed_confirmation": "not opened",
    "fresh_state": ["BC_POST four-module parameters per seed", "fresh Adam", "fresh dual state per frozen contract", "fresh RNG/data-order state"],
    "resume_policy": "No previous PPO optimizer/RNG resume. Route-version freeze identity must differ so old checkpoints reject; implementation and test of this checkpoint-version barrier remain required before READY.",
    "zero_launch_this_turn": True
})

put("READY_CHECK.json", {
    "kind": "round53-readiness-check/1", "status": "NOT_READY", "ready": False,
    "authorization": "PENDING_USER_APPROVAL", "authorization_is_not_implied_by_readiness": True,
    "development_go": False,
    "passed_static": ["ROUND52 mathematical prototype/source identity available", "critic stop-gradient context and trainer method wrapper have pure mechanics tests", "candidate resource arithmetic closes: 528 + 24 = 552", "BC greedy 48 and PPO final 96 indices identified", "immediate/late 16-scene row hash maps available"],
    "blocking_gaps": [
        "No ROUND53 private execution binder exists: current ROUND46 _seed_v6_output_namespace creates/writes v6 paths and seeds a journal from the family journal; no proven adapter redirects every trainer, runner, claim, row, log, checkpoint and journal path to a fresh R53 namespace without touching prior artifacts.",
        "The current V10 entry is evaluation-only and hard-closes a 96-row matrix; no isolated end-to-end training-plus-144-row evaluation closure exists for the extra BC sampled comparator.",
        "BC sampled comparator rows are absent (48 required).",
        "The route-version freeze/checkpoint incompatibility barrier is specified but not implemented/tested in the actual save/load path.",
        "Exact 48 training g_hat reuse has not been verified against the frozen training scenario key/order; baseline reuse remains conditional per row."
    ],
    "must_not_launch": True,
    "required_before_any_future_run": "A separate explicit execution authorization and quota plus closure of every blocking gap; this report grants neither."
})

zero = {k: 0 for k in ("environment_construction", "simulation", "new_scenes_or_flows",
                        "new_evaluation", "real_training_updates", "synthetic_optimizer_steps",
                        "checkpoint_writes")}
put("ZERO_LAUNCH_TEST_EVIDENCE.json", {
    "kind": "round53-zero-launch-evidence/1", "status": "pure_tests_passed_no_project_runtime_loaded",
    "pure_test_count": 7, "pure_test_failures": 0,
    "operations": zero, "test_command": "D:\\project_env\\python.exe -m unittest test_round53_pure.py -v",
    "scope": "adapter hook lifecycle on fake heads; source/role fail-closed mechanics; in-memory quota/claim uniqueness fixture; authorization refusal; pure path containment; zero-counter fixture. Does not test project runtime booking/ledger integration, actual path binding, environment, optimizer, or checkpoint path.",
    "project_runtime_imported": False, "environment_builder_called": False,
    "optimizer_step_called": False, "checkpoint_written": False
})

report = f'''# ROUND53 readiness report\n\n## Decision\n\n**NOT READY; no launch was performed.** The route is a plausible single-variable gradient intervention, but the future runner is not yet isolated from ROUND46/V6 output namespaces and the registered comparison is incomplete. Authorization remains `PENDING_USER_APPROVAL`; this local preparation is not authorization. `development_go=false` remains unchanged.\n\n## Frozen candidate\n\nThe sole proposed change is detaching the shared representation at the task-value and cost-value critic-head inputs during PPO loss computation. Actor and entropy retain the ordinary trunk path. ROUND52's exact source-bound gradient prototype reports unchanged scalar losses/predictions and actor/entropy/head gradients, with critic-to-trunk gradients cut; its tests were limited to its declared source/panel/prototype scope. Canonical trainer SHA256: `{sha(trainer)}`. No trainer update or checkpoint operation occurred here.\n\n## Resource proposal (not authorized)\n\n384 training + 48 greedy + 48 sampled + 48 BC sampled = **528 normal** claims, plus a **new 24** technical-failure reserve, cap **552**. Historical ROUND46's 504 ceiling/reserve is not transferable. The sealed confirmation block remains unopened.\n\n## Reusable evidence and gaps\n\nBC greedy index: `{bc_index['sha256']}` (48 rows). ROUND46 final shared PPO manifest: `{ppo_index['sha256']}` (96 accepted rows). Immediate/late registered references each include {len(immediate)}/{len(late)} development scenes. Reuse is conditional on per-row identity checking. BC sampled is absent (48 needed), and immediate training-cache mapping is not closed against all 48 frozen training keys.\n\nThe key implementation blocker is output isolation: ROUND46's `_seed_v6_output_namespace` creates V6 directories and initializes the V6 checkpoint journal from the family journal; simply changing a top-level run directory is insufficient. V10 evaluation also returns a fixed 96-row matrix, not the proposed 144 rows. The route-version checkpoint incompatibility barrier is not implemented/tested in the actual save/load path. Accordingly the current runtime cannot be represented as a safe, isolated executable trial.\n\n## Tests and zero-operation evidence\n\nSeven pure tests exercise fake-head hook lifecycle, source/runtime role guards, wrapper restoration, in-memory slot/claim closure, authorization refusal, path containment, and zero counters. They do not invoke project runtime booking/ledger integration, environment, optimizer, training or checkpoint code. Seven operation counters are all zero: {', '.join(f'{k}=0' for k in zero)}.\n\n## Next action (single, finite)\n\nPrepare and statically test one ROUND53-private runner/binder that redirects and verifies every output path and journal, adds the route-version resume barrier, and closes the 144-row matrix plus exact 48-key training-cache reuse; do not run it. Only after that passes should a separate explicit execution authorization be considered. This distinguishes a sound gradient-isolation intervention from an apparent effect caused by stale checkpoint resume, misbound trainer, or output contamination.\n\n## Provenance\n\nInitial HEAD: `40adea4e16c5ccd6ee454f86462e4c959149da8a`; branch: `feat/round53-critic-isolation-trial-readiness`. Full committed HEAD is recorded after local commit. Protected ROUND46 dirty files remain untouched and unstaged.\n'''
(OUT / "ROUND53_READINESS_REPORT.md").write_text(report, encoding="utf-8")
print(json.dumps({"files_written": 6, "source_hashes": len(sources), "normal": sum(normal.values()), "cap": 552}, indent=2))
