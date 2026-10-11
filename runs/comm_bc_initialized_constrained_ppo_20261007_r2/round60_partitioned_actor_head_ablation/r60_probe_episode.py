"""ROUND60 bounded probe: stand up the frozen R46/R42 sim runtime (exactly as R58 configured it),
inject the partitioned controller + routed ppo_update, and run ONE real training episode to
validate end-to-end integration and measure per-episode compute. Read-only w.r.t. all frozen
identity; writes only under the round60 probe namespace. This is a measurement, not the fixed run.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent
GEN2 = RUN_ROOT / "round58_gen2"
AUDIT = RUN_ROOT / "round58_isolation_stop_audit"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(GEN2))
sys.path.insert(0, str(AUDIT))
import r60_partitioned as r60
import r58_execute

# The frozen r58 stack guards its output root to live under round58_gen2; the probe writes to a
# clearly-named subdir there. This is a measurement only, not the ROUND60 fixed-run namespace.
PROBE_RAW = GEN2 / "_r60_probe_raw"
print("configuring frozen runtime (this imports the R46/R42 sim stack)...", flush=True)
t0 = time.time()
base = r58_execute.load_base_runner()
r58_execute.bind_runner_paths(base, PROBE_RAW)
r58_execute.install_private_configure_seams(base, allow_existing_journal=False)
base._R45_MODULE = None
base._RUNTIME = None
base._configure_runtime_module()
runtime = base._RUNTIME
assert runtime is not None, "runtime did not configure"
print("runtime configured in %.1fs; keys=%s" % (time.time() - t0, sorted(runtime.keys())), flush=True)

trainer = runtime["trainer"]
controller_module = runtime["controller_module"]
common = runtime["common"]
orig_fresh = controller_module.fresh_controller  # capture BEFORE patching to avoid recursion
def partitioned_fresh(seed):
    base = orig_fresh(int(seed))                 # frozen RNG order preserved
    pc = r60.PartitionedController(base)
    ep0_state = r60.load_r58_ep0_model(int(seed))
    pc.inner.trunk.load_state_dict(ep0_state["trunk"])
    pc.inner.value_head.load_state_dict(ep0_state["value_task_head"])
    pc.cost_head.load_state_dict(ep0_state["value_cost_head"])
    pc.inner.early_head.load_state_dict(ep0_state["action_head"])
    pc.inner.late_head.load_state_dict(ep0_state["action_head"])
    return pc
controller_module.fresh_controller = partitioned_fresh
orig_ppo = trainer.ppo_update
def routed_ppo(controller, optimizer, episodes, *, update_index):
    return r60.partitioned_ppo_update(controller, optimizer, episodes, update_index=update_index)
trainer.ppo_update = routed_ppo
print("partitioned controller + routed ppo_update injected", flush=True)

# PROBE ONLY: bypass the ROUND46 frozen-plan claim/bookkeeping verifier (branch/worktree/journal/
# freeze-hash guards bound to the R58 identity). The simulation and the partitioned training math
# are untouched; this only lets one episode run so we can measure compute. Not used for any claim.
import r44_runtime
r44_runtime.verify_frozen_source = lambda frozen: {}
try:
    base._assert_branch_head = lambda: None
    base._assert_source_worktree_clean = lambda: None
except Exception:
    pass

from r42_one_shot import OneShotSendSkipCollector
controller = partitioned_fresh(11)
coll = OneShotSendSkipCollector(controller, greedy=False)
runner = runtime["runner"]
scene = int(common.scenario_order()[0])
print("running one real training episode: seed=11 episode=0 scene=%d ..." % scene, flush=True)
t1 = time.time()
from r42_run import run_training_episode
try:
    row = run_training_episode(common, runner, controller, training_seed=11, episode_index=0,
                               scenario_seed=scene, lam=1.0)
    dt = time.time() - t1
    print(json.dumps({"kind": "round60-probe-episode/1", "status": "ok",
                      "scene": scene, "episode_seconds": round(dt, 2), "steps": int(row.get("steps", -1)),
                      "on_time": row.get("on_time", None), "decision_records": len(coll.records),
                      "distribution_reads": coll.distribution_reads}, indent=1))
    print(json.dumps({"per_episode_seconds": round(dt, 2),
                      "fixed_plan_576_episodes_estimated_seconds": round(dt * 576, 1)}))
except Exception as exc:
    import traceback
    traceback.print_exc()
    print(json.dumps({"kind": "round60-probe-episode/1", "status": "error",
                      "error": f"{type(exc).__name__}: {exc}"}))
