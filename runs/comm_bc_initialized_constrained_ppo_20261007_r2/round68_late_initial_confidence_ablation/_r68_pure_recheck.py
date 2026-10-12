import sys, os, json, hashlib, shutil, tempfile
from pathlib import Path
R68 = Path("runs/comm_bc_initialized_constrained_ppo_20261007_r2/round68_late_initial_confidence_ablation")
sys.path.insert(0, str(R68.resolve()))
import torch
import r68_guards as guards
import r68_frozen as r68
r68.configure_source_imports()
import cost_math as cm
import r68_execute as base
import r66_frozen as r66
RAW = (R68 / "raw").resolve()
results = []
def check(name, passed, **detail):
    results.append((name, bool(passed)))
    print(("PASS " if passed else "FAIL ") + name + ("" if passed else "  " + json.dumps(detail, default=str)[:300]))
_r = torch.tensor([1.0,2.0,3.0,4.0]); _v = torch.tensor([0.5,0.5,0.5,0.5])
_term = torch.tensor([False,False,False,True]); _boot = torch.tensor([0.5,0.5,0.5,0.0])
_adv,_ret = cm.gae(_r,_v,gamma=1.0,gae_lambda=1.0,terminal=_term,bootstrap=_boot)
_manual = torch.tensor([10.0,9.0,7.0,4.0])
check("gae_nextvalue_bootstrap_telescopes_to_undiscounted_return", torch.allclose(_ret,_manual,atol=1e-6), got=_ret.tolist())
check("gae_advantage_is_return_minus_value", torch.allclose(_adv,_manual-_v,atol=1e-6), got=_adv.tolist())
_a=torch.tensor([0.3,-1.2,2.0]); _c=torch.tensor([0.1,0.4,-0.7])
check("actor_advantage_is_lambda_task_minus_cost", torch.allclose(cm.actor_advantage(_a,_c,0.5),0.5*_a-_c,atol=0,rtol=0))
_adv0,_ret0 = cm.gae(_r,_v,gamma=1.0,gae_lambda=1.0,terminal=_term,bootstrap=torch.zeros(4))
check("gae_allzero_bootstrap_is_prefix_short_as_documented", torch.allclose(_ret0,torch.tensor([8.5,8.0,6.5,4.0]),atol=1e-6), got=_ret0.tolist())
gdir = RAW / "rejected" / "_pure_guard_fixture"
gdir.mkdir(parents=True, exist_ok=True)
g = guards.ForwardRowGuard(gdir/"rows.json", cap=20)
for i in range(19): g.charge(1, f"pre-{i}")
check("guard_19_leaves_one", g.remaining()==1)
check("guard_20th_accepted", g.charge(1,"20th")==20)
refused=""
try: g.charge(1,"21st")
except RuntimeError as e: refused=str(e)
check("guard_21st_refused", "would take used to 21 > cap 20" in refused, err=refused[:160])
try: g.charge(2,"doomed")
except RuntimeError: pass
check("guard_overcap_applies_nothing", g.used()==20)
gb = guards.ForwardRowGuard(gdir/"rows.json", cap=20)
check("guard_restart_no_reset", gb.used()==20)
shutil.rmtree(gdir, ignore_errors=True)
persist = guards.ForwardRowGuard(RAW/"runtime"/"r68_guard_forward_rows.json")
check("persistent_forward_used_is_9319", persist.used()==9319, used=persist.used())
blocked=""
try: persist.charge(1, "post-reconciliation attempt")
except RuntimeError as e: blocked=str(e)
check("persistent_forward_further_charge_blocked", "would take used to 9320 > cap 8192" in blocked, err=blocked[:160])
check("persistent_forward_unchanged_after_block", persist.used()==9319)
pstep = guards.SyntheticStepGuard(RAW/"runtime"/"r68_guard_synthetic_steps.json")
check("persistent_steps_used_is_2", pstep.used()==2, used=pstep.used())
for path,want in [("runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_CURRENT_TASK.json","cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b"),
                  ("runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_R2_V6_STOP_STATE.json","93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18")]:
    h=hashlib.sha256(open(path,"rb").read()).hexdigest()
    check("protected_"+os.path.basename(path), h==want, got=h)
counts = base.R68Ledger(RAW).counts()
check("r68_ledger_fresh_zero_attempts", counts["cumulative_attempts"]==0 and counts["hard_cap"]==800 and counts["hard_headroom"]==800, counts={k:counts[k] for k in ("cumulative_attempts","hard_cap","hard_headroom")})
check("r68_quota_fresh_independent", base.HISTORICAL_PROBE==0 and base.NORMAL_CAP==774 and base.RESERVE_CAP==26 and base.HARD_CAP==800)
rv = json.loads((r66.R60_DIR/"ROUND60_RECONCILIATION_VERDICT.json").read_text(encoding="utf-8"))
tot = rv["totals"]
check("r60_608_over_600_disclosed_not_r68_headroom", int(tot["real_environment_constructions"])==608 and int(tot["hard_cap"])==600 and int(tot["over_by"])==8, totals=tot)
npass = sum(1 for _,p in results if p); n=len(results)
print(f"\nPURE RECHECK: {npass}/{n} passed")
sys.exit(0 if npass==n else 1)