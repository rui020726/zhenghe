"""One-shot generator for ROUND75_S1_RUN_PLAN.json (deleted after use; the plan is the artifact)."""
import hashlib, json, re, ast
from pathlib import Path

RUN_ROOT = Path("runs/comm_bc_initialized_constrained_ppo_20261007_r2")
R75 = RUN_ROOT / "round75_native_ppo_qualification"
R66_DIR = RUN_ROOT / "round66_bc_early_path_preserving_late_adaptation"
R74 = RUN_ROOT / "round74_constrained_early_head_adaptation"
R66_FILE = R66_DIR / "r66_frozen.py"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def size(p):
    return Path(p).stat().st_size

r66_src = R66_FILE.read_text(encoding="utf-8")
tree = ast.parse(r66_src)
consts = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        try:
            consts[node.targets[0].id] = ast.literal_eval(node.value)
        except Exception:
            pass

# R58_EP0_MODELS[11]: the dict holds Path expressions, so literal_eval cannot read it.
registry = re.search(r"R58_EP0_MODELS = \{(.*?)\n\}", r66_src, re.S).group(1)
entry = re.search(r"\n\s*11:\s*\{(.*?)\},\n", registry, re.S).group(1)
run_root = R66_FILE.resolve().parents[1]

def _field(entry_text, key):
    found = re.search(r'"' + key + r'"\s*:\s*"([^"]+)"', entry_text)
    assert found, key
    return found.group(1)

path_expr = re.search(r'"path"\s*:\s*(.*?),\n', entry, re.S).group(1)
path_parts = [part for part in re.findall(r'"([^"]+)"', path_expr) if part != ".."]
model_path = (run_root / "..").joinpath(*path_parts).resolve()
ep0 = {"checkpoint_sha256": _field(entry, "checkpoint_sha256"),
       "parameter_sha256": _field(entry, "parameter_sha256"),
       "action_head_sha256": _field(entry, "action_head_sha256")}

seg = None
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CANON" for t in node.targets):
        seg = ast.get_source_segment(r66_src, node.value)
CANON = run_root.parents[1].parent
for part in re.findall(r'"([^"]+)"', seg):
    CANON = CANON / part
assert (CANON / "v2_trainer.py").is_file(), CANON
assert model_path.is_file(), model_path

row_path = RUN_ROOT / "round58_gen2" / "raw" / "rows" / "v2_r44_train_s11_e0_s1300027_a1.json"
tape_path = RUN_ROOT / "round58_gen2" / "raw" / "logs" / "v2_r44_train_s11_e0_s1300027_a1_tape.jsonl"
row = json.loads(row_path.read_text(encoding="utf-8"))
ledger_dir = R75 / "raw" / "runtime"
ledger_files = sorted(p.name for p in ledger_dir.glob("*") if p.is_file())
ledger_text = (ledger_dir / "r75_total_ledger.jsonl").read_text(encoding="utf-8")

plan = {
  "kind": "round75-s1-run-plan/1",
  "request_id": "R75_S0_FINAL_PREP_20261014",
  "status": "PREPARED_NOT_AUTHORIZED_NOT_EXECUTED",
  "meaning": ("The complete real S1 driver and its freeze precheck are prepared and frozen here. "
              "Nothing in this plan has been executed: zero torch import, zero model load or "
              "construction, zero forward/backward, zero optimizer or PPO step, zero environment, "
              "zero sampling, zero checkpoint write. S1 requires a separate decision-side "
              "execution permission."),
  "branch": "exp/round75-native-ppo-qualification",
  "head_at_writing": "c16ce9362f93993192cf9a1ec3330cba90d25f30",
  "parent_head": "0f8574accae16676a75a248da180be9d49e9fa04",
  "future_command": {
    "precheck_only": "D:\\project_env\\python.exe runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/r75_s1_precheck.py",
    "full_s1": "D:\\project_env\\python.exe runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/r75_s1_runner.py --run-s1",
    "note": "the precheck is pure-file and zero-charge; --run-s1 is the paid path and is NOT run now",
  },
  "entry_points": {
    "precheck": "r75_s1_precheck.py (pure stdlib, no torch, no r75_frozen import, read-only)",
    "runner": "r75_s1_runner.py (lazy torch / r75_frozen import inside the run path only)",
  },
  "input_identity": {
    "row": {"path": row_path.as_posix(), "sha256": sha(row_path), "bytes": size(row_path)},
    "tape": {"path": tape_path.as_posix(), "sha256": sha(tape_path), "bytes": size(tape_path)},
    "r58_ep0_model": {"path": model_path.as_posix(), "sha256": sha(model_path),
                      "bytes": size(model_path),
                      "registry_checkpoint_sha256": ep0["checkpoint_sha256"],
                      "registry_parameter_sha256": ep0["parameter_sha256"],
                      "registry_action_head_sha256": ep0["action_head_sha256"]},
    "behaviour_parameters": {"row_field": "controller_parameters_after.parameter_sha256",
                             "value": row["controller_parameters_after"]["parameter_sha256"],
                             "parameter_count": row["controller_parameters_after"]["parameter_count"],
                             "matches_registry": row["controller_parameters_after"]["parameter_sha256"] == ep0["parameter_sha256"]},
    "batch_facts": {"seed": 11, "episode_index": row["episode_index"], "scenario_seed": row["scenario_seed"],
                    "training_seed": row["training_seed"], "steps": row["steps"],
                    "optimizer_steps": row["optimizer_steps"], "lambda": row["lambda"],
                    "lambda_at_rollout": row["lambda_at_rollout"], "eta_lambda": row["eta_lambda"],
                    "budget_bytes": row["budget_bytes"], "b0_bytes": row["b0_bytes"],
                    "config_hash": row["config_hash"], "training_version": row["training_version"],
                    "substituted_fields_classified": len(row["substituted_fields"])},
    "required_batch_shape": {"joint_records": 66, "candidate_rows": 168, "feature_dim": 19,
                             "horizon_ticks": 1000, "targets": 6, "free_early": 5, "free_late": 1},
    "adapter": ("r75_replay_adapter.load_qualification_batch(row, tape) - the existing qualified adapter; "
                "no new collector, no environment, no episode/GAE switch, no field back-filling, "
                "no rewriting of the recorded log-probs"),
  },
  "source_identity": {
    "canonical_trainer": {"path": (CANON / "v2_trainer.py").as_posix(), "sha256": sha(CANON / "v2_trainer.py"),
                          "expected": consts["EXPECTED_TRAINER_SHA256"]},
    "canonical_common": {"path": (CANON / "v2_common.py").as_posix(), "sha256": sha(CANON / "v2_common.py"),
                         "expected": consts["EXPECTED_COMMON_SHA256"]},
    "canonical_controller": {"path": (CANON / "v2_controller.py").as_posix(), "sha256": sha(CANON / "v2_controller.py"),
                             "expected": consts["EXPECTED_CONTROLLER_SHA256"]},
    "r66_frozen": {"path": R66_FILE.as_posix(), "sha256": sha(R66_FILE),
                   "expected": "a22425729c9327ffcb1d57f4820f3b38f2bb7e35a6d6d81890fe64a75a246b10"},
    "r75_guard": {"path": (R75 / "r75_guard.py").as_posix(), "sha256": sha(R75 / "r75_guard.py"),
                  "expected": "4d730f71fb34ad411043c65141b6bcbb3f7d2f3c4668a6b53054e6400e35d55f",
                  "note": "the verified mechanisms are NOT touched by this revision"},
    "r75_frozen": {"path": (R75 / "r75_frozen.py").as_posix(), "sha256": sha(R75 / "r75_frozen.py"),
                   "revision": ("final-prep: the returned approx_kl reads the real kl accumulator; "
                                "the epoch meter phase differences cumulative snapshots")},
    "r75_replay_adapter": {"path": (R75 / "r75_replay_adapter.py").as_posix(),
                           "sha256": sha(R75 / "r75_replay_adapter.py")},
    "reused_verified_sources": {
      "r74_guard_prod.py": {"path": (R74 / "r74_guard_prod.py").as_posix(), "sha256": sha(R74 / "r74_guard_prod.py"),
                            "expected": "623b9ec6cd243d27321e8c745a0247c97126c9028141cb0ed4760dd90c0796db"},
      "r74_guard.py": {"path": (R74 / "r74_guard.py").as_posix(), "sha256": sha(R74 / "r74_guard.py"),
                       "expected": "9a6b06374fc5b56b6d8077dc6aa332444533198aca0967cb36b867795af614d7"},
      "r43_call_counter_hooks.py": {"path": (RUN_ROOT / "r43_call_counter_hooks.py").as_posix(),
                                    "sha256": sha(RUN_ROOT / "r43_call_counter_hooks.py"),
                                    "expected": "02f80072a14d7305b619edd0db9617a7b0a0f31315b3265aebf1e694ac7398f7"},
    },
    "gae_default_distinction": {
      "v2_common_default": "GAE_LAMBDA = 0.95, ETA_LAMBDA = 0.05 (module defaults, NOT used)",
      "r66_living_override": "configure_source_imports() sets common.GAE_LAMBDA = 1.0 and common.ETA_LAMBDA = 0.5",
      "requirement": ("the S1 run must report the LIVE values 1.0 / 0.5 and must never silently fall back "
                      "to the 0.95 / 0.05 defaults"),
    },
  },
  "gate_only_controller": {
    "construction": ("r75_frozen.build_gate_only_controller(seed=11) -> r66.load_r58_ep0_model + "
                     "r66.fresh_partitioned_controller(ep0_state=...) + freeze_trunk_only"),
    "two_head_equality": ("both independent heads are set to the OLD single action_head values, so "
                          "route_logits reproduces the single-head behaviour by construction"),
    "no_late_scaling": "no late x .10 or any other head scaling is applied",
    "alignment": ("r66.likelihood_alignment must pass within common.LOG_PROB_TOLERANCE = 1e-6; the gate "
                  "prior never enters the forward"),
    "qualification_only": ("these parameters NEVER become the R71 research INIT and NEVER become an effect "
                           "training checkpoint"),
    "r71_research_config_unchanged": ["lambda*A_task - A_cost combined advantage", "trunk frozen only",
                                      "8 trainable objects (early_head, late_head, task_value_head, cost_head; 390 scalars)",
                                      "12 objects / 5830 scalars total", "beta=1.00 gate prior is a training-loss term only",
                                      "19-dim observation, ready_cut 147", "gamma 1.0", "live GAE lambda 1.0",
                                      "dual eta 0.5", "lambda_init 1.0", "B0 10000",
                                      "budget 29300 / 6 targets / ACK off"],
  },
  "candidates": {
    "A1_nonempty_adam_critic_only_control_step": {
      "physical_optimizer_steps": 1, "offline_trunk_target_rows": 0, "joint_records": 0,
      "design": ("Adam(lr=3e-4) over the SAME eight trainable objects as the native path. ONE synthetic "
                 "physical step in which only the two critic modules carry a legal controllable non-zero "
                 "gradient; both actor heads and the trunk receive none, so the actor/trunk parameters do "
                 "not move and the original behaviour probabilities are preserved. The step is pre-charged "
                 "to the central ledger before optimizer.step() and recorded at the real entry."),
      "truthful_records": ["for each of the eight objects: grad is None / non-None",
                           "for each of the eight objects: Adam exp_avg / exp_avg_sq present / absent",
                           "an absent state is recorded as absent and NEVER fabricated as non-empty"],
      "full_state_saved": ["five module state_dicts", "all Adam state values with the eight-object name mapping",
                           "lambda (dual) and lambda_at_rollout", "python random state",
                           "numpy random state", "torch RNG state", "update cursor 0",
                           "data identity (row/tape sha256)", "frozen configuration"],
    },
    "A2_source_actual_update": {
      "physical_optimizer_steps": 4, "offline_trunk_target_rows": 840, "joint_records": 1,
      "design": ("From the SAME A1 state, call r75_native_ppo_update(update_index=0) once through the real "
                 "path: trainer.batch_samples + GAE, the metered r66.likelihood_alignment pass (168 candidate "
                 "rows over 66 calls), the DATA_ORDER_SEED shuffle, 4 epochs x 1 minibatch, the beta=1.00 / "
                 "ready_cut=147 gate prior, the clipped surrogate + dual critic + entropy + gate loss, the "
                 "global clip over the eight trainable objects, optimizer.step."),
    },
    "A3_full_state_save_restore_same_update": {
      "physical_optimizer_steps": 4, "offline_trunk_target_rows": 840, "joint_records": 1,
      "design": ("Restore the exact pre-update A1 state into FRESH objects and re-execute the SAME update "
                 "(update_index=0). The re-run starts from the restored pre-state and does not advance the "
                 "scientific cursor; it replaces the trial, it is not a second committed update."),
    },
  },
  "accounting": {
    "planned_totals": {"physical_optimizer_steps": 9, "offline_trunk_target_rows": 1680,
                       "joint_records": 2, "environment_constructions": 0},
    "per_native_update": {"physical_optimizer_steps": 4, "trunk_entry_calls": 70,
                          "trunk_entry_rows": 840, "largest_single_row_charge": 168,
                          "alignment_calls": 66, "alignment_rows": 168,
                          "epoch_calls": 4, "epoch_rows": 672},
    "both_native_updates": {"trunk_entry_calls": 140, "trunk_entry_rows": 1680},
    "record_exposures": {"unique_records": 66, "batch_reads": 2, "record_exposures": 132},
    "joint_record_semantics": ("the central unit joint_record counts BATCH-READ events, not original joint "
                               "records and not environment episodes: 2 batch-read events re-expose the same "
                               "66 unique records 132 times; the environment guard stays 0/0/0"),
    "caps": {"physical_optimizer_step": "normal 12 + technical 4 / hard 16",
             "offline_trunk_target_row": "normal 5000 + technical 1000 / hard 6000, per-charge 1024",
             "environment_construction": "0 / 0 / 0 - forbidden by source"},
    "headroom_after_plan": {"physical_optimizer_steps": 7, "offline_trunk_target_rows": 4320},
    "no_quota_borrowing": ("the plan may not convert headroom into extra steps; no extra forward "
                           "verification, parameter/state comparison does not forward"),
  },
  "metering_contract": {
    "attachment": "every real forward attaches ONLY to R75Resources.production() (the canonical durable root)",
    "installation": ("resources.ensure_trunk_meter(controller, ...) is installed BEFORE any possible "
                     "forward, for BOTH controllers (A2 and A3)"),
    "recorded_per_controller": ["calls_attempted", "calls_charged", "rows_charged", "calls_completed",
                                "rows_completed", "calls_refused", "calls_failed"],
    "phase_split": {"alignment": "cumulative snapshot at the end of r66.likelihood_alignment minus pre_meter",
                    "epochs": "final cumulative snapshot minus the alignment-end snapshot",
                    "total": "final cumulative snapshot minus pre_meter",
                    "invariant": "alignment + epochs == total for EVERY component"},
    "no_dodge": ("the meter is never uninstalled to escape accounting; no aggregate replaces the entry; "
                 "no double charge"),
    "reconciliation": ("every Adam.step is pre-charged and reconciled item-by-item against the durable "
                       "counter; resources.verify_or_fail() runs before any work"),
  },
  "comparison_contract": {
    "compared_value_by_value": ["five module parameters", "all Adam state values with the eight-object name mapping",
                                "lambda / lambda_at_rollout", "python / numpy / torch RNG state",
                                "update cursor", "scientific numeric returns of the update record"],
    "excluded_in_advance_and_accounted_separately": [
      "wall-clock timestamps", "central cumulative resource counters and ledger event sequence numbers",
      "temporary file paths and process ids",
      "the A3 trunk-entry meter absolute cumulative readings (A3 is a second controller, so its absolute "
      "counts are expected to differ and are reconciled, not diffed)"],
    "forbidden_exclusions": ("no real scientific difference may be excluded; the exclusion list is fixed "
                             "here before the run"),
  },
  "stop_conditions": ["any source, data or ledger byte-identity mismatch",
                      "the R58 ep0 state-dict key mapping into the five-module controller fails",
                      "likelihood_alignment exceeds 1e-6 (data is never rewritten, alignment never disabled)",
                      "any Adam object whose recorded grad/state presence contradicts the measurement",
                      "the A3 re-run record differs from A2 in any compared field",
                      "any ledger refusal, any fixture/simulated attachment, any environment construction attempt",
                      "non-finite loss or gradient norm",
                      "any checkpoint save that omits the optimizer state or the RNG state"],
  "execution_discipline": {
    "attempts": "ONE real execution attempt",
    "no_auto_retry": "an anomaly keeps the failure and the consumed originals and stops immediately",
    "no_technical_reserve": "ordinary exhaustion is never auto-upgraded to the technical reserve",
    "no_refund": "a failed step is never refunded",
    "no_scratch": "no charged scratch, dev, preheat, small trial or extra probe",
    "history_preserved": ("the R74 >=16 physical-step lower bound and every earlier gap stay permanently "
                          "recorded"),
  },
  "not_covered": ["torch import, model load/construction and the state-dict mapping outcome",
                  "the real likelihood_alignment outcome",
                  "any real or synthetic optimizer step, PPO update or replay",
                  "the real collector, real episodes, closed-loop simulation, any evaluation block",
                  "full multi-batch training and any research checkpoint",
                  "any communication-policy effect, cost or task-outcome conclusion"],
  "canonical_ledger_frozen_snapshot": {
    "root": ledger_dir.as_posix(),
    "files": {name: {"sha256": sha(ledger_dir / name), "bytes": size(ledger_dir / name)}
              for name in ledger_files},
    "total_ledger_events": len(ledger_text.splitlines()),
    "charged_events": sum(1 for line in ledger_text.splitlines() if '"charged"' in line),
    "rule": ("the precheck must reproduce this snapshot byte-for-byte; no bootstrap append and no rebuild "
             "is permitted"),
  },
}

out = R75 / "ROUND75_S1_RUN_PLAN.json"
out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
print("wrote", out, out.stat().st_size, "bytes")
print("CANON:", CANON)
print("model:", model_path, size(model_path), "bytes")
print("row sha:", plan["input_identity"]["row"]["sha256"][:16],
      "behaviour match:", plan["input_identity"]["behaviour_parameters"]["matches_registry"])
print("ledger files:", len(ledger_files), "events:", plan["canonical_ledger_frozen_snapshot"]["total_ledger_events"],
      "charged:", plan["canonical_ledger_frozen_snapshot"]["charged_events"])
ok = (plan["source_identity"]["canonical_trainer"]["sha256"] == consts["EXPECTED_TRAINER_SHA256"]
      and plan["source_identity"]["canonical_common"]["sha256"] == consts["EXPECTED_COMMON_SHA256"]
      and plan["source_identity"]["canonical_controller"]["sha256"] == consts["EXPECTED_CONTROLLER_SHA256"]
      and plan["source_identity"]["r66_frozen"]["sha256"] == plan["source_identity"]["r66_frozen"]["expected"]
      and plan["input_identity"]["r58_ep0_model"]["sha256"] == ep0["checkpoint_sha256"])
print("ALL SOURCE/DATA IDENTITIES MATCH:", ok)