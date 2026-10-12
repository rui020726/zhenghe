import json, time, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r68_guards as guards

RAW = HERE / "raw" / "runtime"
# New-phase make-up forward-row guard (future-only grant). Atomic, charge-before-work, no refund.
fwd = guards.ForwardRowGuard(RAW / "r68_makeup_forward_rows.json", cap=2048)
steps = guards.SyntheticStepGuard(RAW / "r68_guard_synthetic_steps.json", cap=32)
ppo = guards.SyntheticPpoGuard(RAW / "r68_guard_synthetic_ppo.json", cap=32)

# Read current old-guard state to register the preserved violation verbatim.
old_fwd = json.loads((RAW / "r68_guard_forward_rows.json").read_text(encoding="utf-8"))

reg = {
  "kind": "round68-makeup-grant/1",
  "task_id": "ROUND68-LATE-INITIAL-CONFIDENCE-ABLATION",
  "branch": "exp/round68-late-initial-confidence-ablation",
  "recorded_unix": time.time(),
  "authorization": ("future-only make-up grant for the forgone acceptance tail, within the user's "
                    "existing remote research/training authorization. Does NOT retroactively approve "
                    "the old 8192 forward-row overage. Does NOT change env 774+26/800, alpha0.10, or "
                    "any research config. Constructs NO environment; resumes NO training."),
  "old_violation_preserved": {
    "acceptance_forward_row_guard": {"cap": old_fwd["cap"], "used": old_fwd["used"],
                                     "over_by": old_fwd["used"] - old_fwd["cap"], "status": "EXCEEDED"},
    "synthetic_optimizer_step_guard": {"cap": 32, "used": steps.used(), "status": "within_cap"},
    "synthetic_ppo_call_guard": {"cap": 32, "used": ppo.used(), "status": "within_cap"},
    "throwaway_dev_roots_retained": {"r68_dev_guards": 2304, "r68_dev_guards_b": 2331,
                                     "r68_dev_guards_c": 2342, "r68_dev_guards_d": 2342},
    "note": "old 9319 forward rows and all temp dev roots preserved unchanged; not refunded or zeroed."
  },
  "new_grant": {
    "additional_forward_rows_cap": 2048,
    "additional_physical_synthetic_steps_cap": 16,
    "total_physical_synthetic_steps_cap_unchanged": 32,
    "existing_steps_counted": steps.used(),
    "ppo_calls_counted_separately": True,
    "note": ("total step cap not raised; existing steps count; any unknown historical steps must be "
             "closed (no zeroing); the +16 tail steps are charged to the shared cap-32 guard so the "
             "total stays <= 32.")
  },
  "accounting_model": {
    "unit": ("actual model input rows per top-level forward / cost_values / direct trunk probe; "
             "internal layers (route_logits / value_head / cost_head on a precomputed hidden) are "
             "not double-counted because they do not re-enter the trunk."),
    "charge_timing": "atomic, BEFORE the forward/step runs; no refund across failure/repeat/process death.",
    "lower_bound": ("if any manual counter missed a call, record the actual lower bound and the "
                    "missing source; do not rely on counter equality for global compliance.")
  },
  "new_phase_guards": {
    "forward": "raw/runtime/r68_makeup_forward_rows.json (cap 2048, atomic, no refund)",
    "steps": "reuse raw/runtime/r68_guard_synthetic_steps.json (cap 32; enforces total<=32)",
    "ppo": "reuse raw/runtime/r68_guard_synthetic_ppo.json (cap 32; counted for the record)"
  },
  "scope": {
    "passed_by_identity_reuse": ["1 init equivalence", "2 routing 147/148+perm+mixed",
                                 "3 per-sample gradients", "4 optimizer/coverage",
                                 "5 critic->trunk isolation", "6 pure GAE/mask/likelihood/ALL_ROWS",
                                 "CLI metadata acceptance 28/28", "pure recheck 18/18"],
    "forgone_to_make_up": ["6 fabricated-episode PPO update (real forwards)",
                           "7 real R60-episode PPO update + full 5-module/Adam/lambda/RNG/cursor resume + deterministic subsequent update + R60/R66/wrong-freeze/quarantine rejection",
                           "8 CLI configure_and_inject identity in-test",
                           "9 unregistered/duplicate-start claim guard"],
    "explicitly_not_claimed": ("the 18 pure recheck and CLI metadata acceptance must NOT be "
                               "represented as real PPO / full checkpoint recovery / old-version "
                               "rejection gate passes.")
  },
  "relation_to_800_env_budget": ("the 8192/32/32 and new 2048/16 caps are ACCEPTANCE/SYNTHETIC-TEST "
                                 "computation caps, separate from the 800 env-construction budget.")
}
(RAW / "ROUND68_MAKEUP_GRANT.json").write_text(json.dumps(reg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("makeup_forward_guard:", json.dumps(fwd.snapshot()))
print("steps_guard used/cap:", steps.used(), "/", steps.snapshot()["cap"])
print("ppo_guard used/cap:", ppo.used(), "/", ppo.snapshot()["cap"])
print("registration:", RAW / "ROUND68_MAKEUP_GRANT.json")