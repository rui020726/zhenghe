import sys, json, hashlib
from pathlib import Path
R68 = Path(r"runs/comm_bc_initialized_constrained_ppo_20261007_r2/round68_late_initial_confidence_ablation")
sys.path.insert(0, str(R68))
import torch
import r68_frozen as r68
import r66_frozen as r66

r68.configure_source_imports()
import v2_controller as vc

def mod_sha(controller, name):
    m = {"trunk": controller.inner.trunk, "early_head": controller.inner.early_head,
         "late_head": controller.inner.late_head, "value_task_head": controller.inner.value_head,
         "cost_head": controller.cost_head}[name]
    d = hashlib.sha256()
    for n,t in sorted(m.named_parameters()):
        d.update(n.encode()); d.update(t.detach().cpu().contiguous().numpy().tobytes())
    return d.hexdigest()

seed = 11
b = vc.fresh_controller(seed)
ctrl, ep0, frozen_id, summary, scale_id = r68.build_frozen_controller(b, seed)
print("scale identity:", json.dumps({k:scale_id[k] for k in ("late_init_alpha","pre_scale_equals_r60_ep0")}, ensure_ascii=False))

# Reference: R60 ep0 controller, unscaled
ep0m = r66.load_r60_model(seed, "ep0")
ref = r66.fresh_partitioned_controller(seed, ep0_state=None, verify_ep0=False)
ref.load_state_dict(ep0m["controller"])

# 1) trunk/early/critics byte-identical to R60 ep0
for name in ("trunk","early_head","value_task_head","cost_head"):
    same = mod_sha(ctrl, name) == mod_sha(ref, name)
    print(f"  {name} == R60 ep0:", same)
    assert same, name

# 2) late_head == 0.10 * R60 ep0 late_head (weight and bias)
lw = ctrl.inner.late_head.weight.detach(); lb = ctrl.inner.late_head.bias.detach()
rw = ref.inner.late_head.weight.detach(); rb = ref.inner.late_head.bias.detach()
wok = torch.allclose(lw, rw*0.10, atol=0, rtol=0); bok = torch.allclose(lb, rb*0.10, atol=0, rtol=0)
print("  late_head.weight == 0.10*ep0:", bool(wok), " bias == 0.10*ep0:", bool(bok))
assert wok and bok

# 3) frozen mask: trunk+early requires_grad False; late+critics True
frozen = r68.frozen_parameters(ctrl); trainable = r68.trainable_parameters(ctrl)
print("  frozen objs:", len(frozen), "trainable objs:", len(trainable),
      "frozen_rg_false:", all(not p.requires_grad for p in frozen),
      "trainable_rg_true:", all(p.requires_grad for p in trainable))
assert len(frozen)==6 and len(trainable)==6

# 4) checkpoint save/load round-trip under a TEST freeze dict (R68 identity)
freeze = {"kind": r68.R68_FREEZE_KIND, "late_init_alpha": 0.10, "structure": r68.R68_STRUCTURE,
          "test": True}
opt = r68.build_optimizer(ctrl, 3e-4)
state = r68.fresh_training_state(seed)
testdir = R68 / "raw" / "rejected" / "_smoke"
testdir.mkdir(parents=True, exist_ok=True)
p = testdir / "smoke_ep0.pt"
r68.save_checkpoint(ctrl, opt, state, p, freeze=freeze)
c2, o2, s2 = r68.load_checkpoint(p, freeze=freeze)
print("  round-trip late == 0.10*ep0:", bool(torch.allclose(c2.inner.late_head.weight.detach(), rw*0.10, atol=0, rtol=0)))
print("  round-trip late_init_alpha in state:", s2.get("late_init_alpha"), "schema:", s2.get("state_schema")==r68.R68_STATE_SCHEMA)
assert torch.allclose(c2.inner.late_head.weight.detach(), rw*0.10, atol=0, rtol=0)

# 5) reject R60 checkpoint as R68 resume source
r60path = r66.r60_model_path(seed, "ep0")
try:
    r68.load_checkpoint(r60path, freeze=freeze)
    print("  ERROR: R60 checkpoint was NOT rejected"); raise SystemExit(2)
except RuntimeError as e:
    print("  R60 resume rejected:", str(e)[:70])

# 6) reject wrong freeze on the R68 checkpoint
try:
    r68.load_checkpoint(p, freeze={"kind":"wrong"})
    print("  ERROR: wrong freeze NOT rejected"); raise SystemExit(3)
except RuntimeError as e:
    print("  wrong-freeze rejected:", str(e)[:60])

# cleanup smoke artifacts (test-only dir; not a real checkpoint)
import shutil; shutil.rmtree(testdir, ignore_errors=True)
print("SMOKE OK")
