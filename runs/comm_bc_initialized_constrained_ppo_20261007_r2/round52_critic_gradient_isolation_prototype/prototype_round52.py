from __future__ import annotations
import ast, hashlib, importlib.util, json, random, sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent                         # ROUND46 run root
REPO=ROOT.parents[1]
SEEDS=(11,22,33)
TRAINER_SHA='0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841'
TOL=1e-6
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(n,x): (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PANEL=read(ROOT/'round49_bc_input_policy_closure'/'INPUT_PANEL_MANIFEST.json')
SEND=int(PANEL['action_mapping']['SEND']); SKIP=int(PANEL['action_mapping']['SKIP'])
assert SEND!=SKIP and set(PANEL['action_mapping'])=={'SEND','SKIP'}

CANON=REPO.parent/'zhenghe'/'runs'/'comm_cost_v2_rerun_20260928'
TRAINER=CANON/'v2_trainer.py'
assert sha(TRAINER)==TRAINER_SHA
sys.path[:0]=[str(x) for x in [CANON,REPO.parent/'zhenghe'/'runs'/'comm_cost_stage_b_20260927',REPO.parent/'zhenghe'/'runs'/'comm_tight_budget_20260919',REPO.parent/'zhenghe'/'runs'/'comm_budget_pilot_20260919',REPO.parent/'zhenghe-round40'/'runs'/'comm_first_report_training_ablation_20261003']]
import v2_trainer as trainer
import v2_controller
R48P=ROOT/'round48_policy_drift_probe'/'r48_probe.py'
sp=importlib.util.spec_from_file_location('r48_r52_readonly',R48P); r48=importlib.util.module_from_spec(sp); sp.loader.exec_module(r48)

# Apply only the already-frozen R43 data-contract override. Do not call runtime setup.
r43=read(REPO/'runs'/'comm_one_shot_full_trace_ablation_20261004'/'ROUND43_FROZEN_CONFIG.json')['round43']
assert trainer.common.GAE_LAMBDA==r43['gae_lambda_old'] and float(r43['gae_lambda'])==1.0
trainer.common.GAE_LAMBDA=float(r43['gae_lambda'])
CLIP_NORM=float(trainer.common.MAX_GRAD_NORM)   # read from exact source; do not assume a value
assert SEND==1 and SKIP==0

@contextmanager
def critic_input_detached(controller):
    """Temporary head-input route; values are unchanged, only critic-to-trunk autograd is cut."""
    modules=(controller.inner.value_head,controller.cost_head)
    before=[set(m._forward_pre_hooks) for m in modules]
    handles=[]
    def detach_first(_module,args):
        if not args: raise RuntimeError('critic head received no shared representation')
        return (args[0].detach(),*args[1:])
    try:
        for module in modules: handles.append(module.register_forward_pre_hook(detach_first))
        yield
    finally:
        for handle in reversed(handles): handle.remove()
        if any(set(m._forward_pre_hooks)!=old for m,old in zip(modules,before)):
            raise AssertionError('temporary critic hooks were not fully removed')

# Extract, from the exact frozen trainer AST, the statements that produce its batch loss.
source=TRAINER.read_text(encoding='utf-8'); tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='ppo_update')
outer=next(n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='start')
body=[]; started=False
for stmt in outer.body:
    if isinstance(stmt,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='stacked' for t in stmt.targets): started=True
    if not started: continue
    if isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Call) and ast.unparse(stmt.value.func)=='optimizer.zero_grad': break
    body.append(stmt)
calls=[ast.unparse(n.func) for st in body for n in ast.walk(st) if isinstance(n,ast.Call)]
assert any(isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in s.targets) for s in body)
assert not any(any(x in c for x in ('step','backward','zero_grad','clip_grad_norm')) for c in calls)
ret=ast.Return(value=ast.Dict(keys=[ast.Constant(k) for k in ('loss','surrogate','entropy_t','value_task_loss','value_cost_loss','logits','candidate_values_task','candidate_values_cost','stacked')],values=[ast.Name(id=k,ctx=ast.Load()) for k in ('loss','surrogate','entropy_t','value_task_loss','value_cost_loss','logits','candidate_values_task','candidate_values_cost','stacked')]))
helper=ast.FunctionDef(name='r52_exact_batch_loss',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='controller'),ast.arg(arg='chunk')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body+[ret],decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[helper],type_ignores=[])),str(TRAINER),'exec'),trainer.__dict__)
exact_loss=trainer.r52_exact_batch_loss

def params_by_group(c):
    groups={'trunk':list(c.inner.trunk.named_parameters()),'action_head':list(c.inner.action_head.named_parameters()),'task_value_head':list(c.inner.value_head.named_parameters()),'cost_value_head':list(c.cost_head.named_parameters())}
    ids=[id(p) for named in groups.values() for _,p in named]
    assert len(ids)==len(set(ids)) and len(ids)==10
    assert set(ids)=={id(p) for p in c.parameters}  # cross-check frozen clipping parameter set, not used to form groups
    flat=[p for named in groups.values() for _,p in named]
    return groups,flat
def module_digest(c):
    h=hashlib.sha256()
    for role,sd in sorted(c.state_dict().items()):
        for name,t in sorted(sd.items()): h.update((role+'.'+name).encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def read_controller(seed,path,expected_file,expected_params):
    assert sha(path)==expected_file
    py=random.getstate(); ns=np.random.get_state(); ts=torch.random.get_rng_state()
    try:
        c=v2_controller.fresh_controller(seed)
        state=r48.load_controller(path) # extracts controller only; no optimizer/RNG restoration
        c.inner.trunk.load_state_dict(state['trunk']); c.inner.action_head.load_state_dict(state['action_head']); c.inner.value_head.load_state_dict(state['value_task_head']); c.cost_head.load_state_dict(state['value_cost_head'])
        for m in (c.inner.trunk,c.inner.action_head,c.inner.value_head,c.cost_head): m.eval()
    finally: random.setstate(py); np.random.set_state(ns); torch.random.set_rng_state(ts)
    assert module_digest(c)==expected_params
    return c
def episodes_for(seed):
    record_path=ROOT/'v6'/'checkpoints'/f'r40_batch_s{seed}_b0_attempt01.json'; record=read(record_path)
    episodes=[]; evidence=[]
    for art in record['episode_artifacts']:
        stem=f"v2_r44_train_s{seed}_e{art['episode_index']}_s{art['scenario_seed']}_a{art['attempt']}"
        rowp=ROOT/'v6'/'rows'/f'{stem}.json'; tapep=ROOT/'v6'/'logs'/f'{stem}_tape.jsonl'
        assert sha(rowp)==art['row_sha256'] and sha(tapep)==art['tape_sha256']
        row=read(rowp); tapes=[json.loads(s) for s in tapep.read_text(encoding='utf-8').splitlines() if s.strip()]
        episodes.append({'collector':SimpleNamespace(records=tapes),'steps':row['steps'],'cost_stream':row['cost_stream'],'task_stream':row['task_stream'],'lambda':float(row['lambda']),'episode_index':int(art['episode_index']),'scenario_seed':int(art['scenario_seed'])})
        evidence.append({'scene':art['scenario_seed'],'row_sha256':sha(rowp),'tape_sha256':sha(tapep),'records':len(tapes)})
    samples=trainer.batch_samples(episodes); assert len(samples)==265
    role=read(ROOT/'round48_policy_drift_probe'/'MODEL_ROLE_IDENTITY.json')['roles']['ep0_training_init'][str(seed)]
    replay=read_controller(seed,Path(role['round46_initializer_file']),role['round46_initializer_file_sha256'],role['round46_initializer_parameter_sha256'])
    assert trainer.likelihood_alignment(replay,samples)['passed']
    return record,samples,evidence

def components(out):
    return {'actor_surrogate':-out['surrogate'].mean(),'entropy':-trainer.common.ENTROPY_BONUS*out['entropy_t'].mean(),'task_value':trainer.common.VALUE_COEFFICIENT*out['value_task_loss'],'cost_value':trainer.common.VALUE_COEFFICIENT*out['value_cost_loss']}
def grads_for(c,samples,detached):
    groups,ps=params_by_group(c)
    ctx=critic_input_detached(c) if detached else _nullcontext()
    with ctx: out=exact_loss(c,samples)
    comp=components(out)
    all_losses={**comp,'total_loss':out['loss']}
    gs={}; values={}; group_tables={}
    for name,loss in all_losses.items():
        raw=torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True)
        by_group={}; piece=[]; at=0
        for group,named in groups.items():
            cur=[]
            for _,p in named:
                z=raw[at]; at+=1
                cur.append(None if z is None else z.detach())
            by_group[group]=cur
            piece.extend([torch.zeros_like(p).reshape(-1) if z is None else z.reshape(-1) for (_,p),z in zip(named,cur)])
        flat=torch.cat(piece)
        gs[name]={'flat':flat,'groups':by_group}
        values[name]=float(loss.detach())
        group_tables[name]={g:{'l1':float(torch.cat([torch.zeros_like(p).reshape(-1) if z is None else z.reshape(-1) for (_,p),z in zip(groups[g],by_group[g])]).abs().sum()),'l2':float(torch.linalg.vector_norm(torch.cat([torch.zeros_like(p).reshape(-1) if z is None else z.reshape(-1) for (_,p),z in zip(groups[g],by_group[g])]))),'none_parameters':[n for (n,_),z in zip(groups[g],by_group[g]) if z is None]} for g in groups}
    assert all(p.grad is None for p in ps)
    return out,comp,gs,values,group_tables

class _nullcontext:
    def __enter__(self): return None
    def __exit__(self,*exc): return False

def snapshot(c,samples):
    x=torch.as_tensor(np.concatenate([s['features'] for s in samples],axis=0),dtype=torch.float32)
    with torch.no_grad():
        h=c.inner.trunk(x); z=c.inner.action_head(h); vt=c.inner.value_head(h).squeeze(-1); vc=c.cost_head(h).squeeze(-1)
    return (z.clone(),vt.clone(),vc.clone())
def flat_metrics(x): return {'l1':float(x.abs().sum()),'l2':float(torch.linalg.vector_norm(x)),'max_abs':float(x.abs().max())}

ROUND51=read(ROOT/'round51_frozen_source_gradient_close'/'GRADIENT_COMPONENTS.json')
ROLE=read(ROOT/'round48_policy_drift_probe'/'MODEL_ROLE_IDENTITY.json')
assert len(PANEL['records'])==576 and PANEL['action_mapping']=={'SEND':SEND,'SKIP':SKIP}
assert ROLE['roles']['ep0_training_init'] and ROLE['roles']['ep128_final']
r43_role=ROLE['roles']['ep0_training_init']['11']['r43_era_ep0_file_not_the_round46_init']
try:
    bc_role=ROLE['roles']['ep0_training_init']['11']
    read_controller(11,Path(bc_role['round46_initializer_file']),bc_role['round46_initializer_file_sha256'],r43_role['four_module_parameter_sha256'])
    wrong_role_rejected=False
except AssertionError:
    wrong_role_rejected=True
assert wrong_role_rejected
all_results={}; directions=[]; checkpoint_hashes={}; params_unchanged={}; hook_cleanup=True; baseline_ok=True
for seed in SEEDS:
    batch,samples,artifact_evidence=episodes_for(seed)
    ep0=ROLE['roles']['ep0_training_init'][str(seed)]
    path=Path(ep0['round46_initializer_file']); c=read_controller(seed,path,ep0['round46_initializer_file_sha256'],ep0['round46_initializer_parameter_sha256'])
    checkpoint_hashes[str(path)]=ep0['round46_initializer_file_sha256']; initial_hash=module_digest(c)
    base_snapshot=snapshot(c,samples)
    oldout,oldcomp,oldg,oldvalues,oldgroups=grads_for(c,samples,False)
    prehook={id(m):set(m._forward_pre_hooks) for m in (c.inner.value_head,c.cost_head)}
    try:
        with critic_input_detached(c):
            new_snapshot=snapshot(c,samples)
            newout,newcomp,newg,newvalues,newgroups=grads_for(c,samples,False)
    finally:
        hook_cleanup &= all(set(m._forward_pre_hooks)==prehook[id(m)] for m in (c.inner.value_head,c.cost_head))
    assert hook_cleanup
    for a,b in zip(base_snapshot,new_snapshot): assert torch.equal(a,b)
    for k in oldvalues: assert abs(oldvalues[k]-newvalues[k])<=1e-7
    # Round51 exact-source baseline reproduction, both scalar terms and module gradients.
    ref=ROUND51['per_seed'][str(seed)]
    for k in ref['loss_terms']: assert abs(oldvalues[k]-ref['loss_terms'][k])<=1e-6,(seed,k,oldvalues[k],ref['loss_terms'][k])
    for k in ('actor_surrogate','entropy','task_value','cost_value','total_loss'):
        for group in ('trunk','action_head','task_value_head','cost_value_head'):
            round51_name={'trunk':'shared_trunk','cost_value_head':'cost_value_head'}.get(group,group)
            a=oldgroups[k][group]['l2']; b=ref['gradient_by_parameter_group'][k][round51_name]['l2']
            assert abs(a-b)<1e-5,(seed,k,group,a,b)
    for component in ('actor_surrogate','entropy'):
        assert torch.allclose(oldg[component]['flat'],newg[component]['flat'],atol=1e-7,rtol=0)
    for component,head in (('task_value','task_value_head'),('cost_value','cost_value_head')):
        assert newgroups[component]['trunk']['l2']==0.0
        gcheck,_=params_by_group(c); cut=sum(p.numel() for _,p in gcheck['trunk'])+sum(p.numel() for _,p in gcheck['action_head'])
        assert torch.allclose(oldg[component]['flat'][cut:],newg[component]['flat'][cut:],atol=1e-7,rtol=0)
    oldtotal=oldg['total_loss']['flat']; newtotal=newg['total_loss']['flat']
    # Independent sum: actor+entropy and critic-head gradients, explicitly zero trunk critic slices.
    group,ps=params_by_group(c)
    assembled=oldg['actor_surrogate']['flat']+oldg['entropy']['flat']+oldg['task_value']['flat']+oldg['cost_value']['flat']
    assert torch.allclose(assembled,oldtotal,atol=2e-6,rtol=0)
    assert torch.allclose(newg['actor_surrogate']['flat']+newg['entropy']['flat']+newg['task_value']['flat']+newg['cost_value']['flat'],newtotal,atol=2e-6,rtol=0)
    assert module_digest(c)==initial_hash and all(p.grad is None for p in ps) and sha(path)==ep0['round46_initializer_file_sha256']
    params_unchanged[str(seed)]={'parameter_sha256':initial_hash,'checkpoint_sha256':sha(path)}
    oldnorm=float(torch.linalg.vector_norm(oldtotal)); newnorm=float(torch.linalg.vector_norm(newtotal))
    oldscale=min(1.0,CLIP_NORM/oldnorm) if oldnorm else 1.0; newscale=min(1.0,CLIP_NORM/newnorm) if newnorm else 1.0
    by_panel=[]
    # Cache Jacobian only for identical float32 vectors, while preserving all source rows.
    jac_cache={}; panel_forward_checks=0
    for row in PANEL['records']:
        key=row['features_sha256']
        x=torch.tensor(row['features'],dtype=torch.float32)
        with torch.no_grad():
            h=c.inner.trunk(x); base_logits=c.inner.action_head(h); base_task=c.inner.value_head(h).squeeze(-1); base_cost=c.cost_head(h).squeeze(-1)
        with critic_input_detached(c),torch.no_grad():
            h2=c.inner.trunk(x); new_logits=c.inner.action_head(h2); new_task=c.inner.value_head(h2).squeeze(-1); new_cost=c.cost_head(h2).squeeze(-1)
        assert torch.equal(base_logits,new_logits) and torch.equal(base_task,new_task) and torch.equal(base_cost,new_cost)
        panel_forward_checks+=1
        if key not in jac_cache:
            logits,_=c.forward(x)
            logodds=logits[...,SEND]-logits[...,SKIP]
            jac=torch.autograd.grad(logodds.sum(),ps,retain_graph=False,allow_unused=True)
            jac_cache[key]=torch.cat([torch.zeros_like(p).reshape(-1) if q is None else q.detach().reshape(-1) for p,q in zip(ps,jac)])
        jv=jac_cache[key]
        od=float(torch.dot(jv,-oldtotal)); nd=float(torch.dot(jv,-newtotal))
        directions.append({'seed':seed,'panel':row['panel'],'source_seed':row['source_seed'],'scene':row['scene'],'target':row['target'],'tick':row['tick'],'features_sha256':key,'unclipped_old':od,'unclipped_isolated':nd,'clipped_old':od*oldscale,'clipped_isolated':nd*newscale,'old_clip_scale':oldscale,'isolated_clip_scale':newscale})
    all_results[str(seed)]={'batch_record_sha256':sha(ROOT/'v6'/'checkpoints'/f'r40_batch_s{seed}_b0_attempt01.json'),'episode_inputs':artifact_evidence,'samples':len(samples),'target_rows':sum(len(x['targets']) for x in samples),'free_rows':sum(len(x['free']) for x in samples),'loss_terms_old':oldvalues,'loss_terms_isolated':newvalues,'loss_numeric_equal':True,'gradient_groups_old':oldgroups,'gradient_groups_isolated':newgroups,'total_old':flat_metrics(oldtotal),'total_isolated':flat_metrics(newtotal),'old_clip_scale':oldscale,'isolated_clip_scale':newscale,'clip_norm_source_value':CLIP_NORM,'gradient_sum_closed':True,'critic_trunk_isolated':True,'actor_entropy_gradients_equal':True,'predictions_exactly_equal':True,'panel_forward_checks':panel_forward_checks,'panel_unique_vectors_cached':len(jac_cache),'panel_rows_preserved':len(PANEL['records'])}

assert len(directions)==3*len(PANEL['records']) and hook_cleanup
assert all(sha(p)==h for p,h in checkpoint_hashes.items())
comparison={'kind':'round52-forward-gradient-comparison/1','status':'passed','source_trainer_sha256':sha(TRAINER),'round51_gradient_json_sha256':sha(ROOT/'round51_frozen_source_gradient_close'/'GRADIENT_COMPONENTS.json'),'panel_sha256':sha(ROOT/'round49_bc_input_policy_closure'/'INPUT_PANEL_MANIFEST.json'),'action_mapping':{'SEND':SEND,'SKIP':SKIP},'loss_reduction_and_mask_contract':'exact frozen trainer AST; all decision rows retained; same pooling, mask, target, denominator, gamma/GAE/advantage','temporary_hook_contract':'forward_pre_hook on task/cost head input only; detached tensor preserves values; all handles removed in finally','per_seed':all_results,'per_input_directions':directions,'input_rows':len(PANEL['records']),'direction_rows':len(directions),'unique_vector_cache_is_only_computation_optimization':True,'checkpoints_unchanged':True,'parameters_unchanged':params_unchanged,'hook_cleanup_verified':hook_cleanup,'optimizer_steps':0,'parameter_writes':0}
dump('FORWARD_AND_GRADIENT_COMPARISON.json',comparison)

# Simple analytic model traverses the same temporary route, with no update operation.
class TinyInner:
    def __init__(self): self.trunk=torch.nn.Linear(1,1,bias=False); self.action_head=torch.nn.Linear(1,2,bias=False); self.value_head=torch.nn.Linear(1,1,bias=False)
    def forward(self,x):
        h=self.trunk(x); return self.action_head(h),self.value_head(h).squeeze(-1)
class TinyController:
    def __init__(self): self.inner=TinyInner(); self.cost_head=torch.nn.Linear(1,1,bias=False)
    def forward(self,x): return self.inner.forward(x)
tiny=TinyController()
with torch.no_grad(): tiny.inner.trunk.weight.fill_(2); tiny.inner.action_head.weight.copy_(torch.tensor([[0.],[4.]])); tiny.inner.value_head.weight.fill_(5); tiny.cost_head.weight.fill_(5)
x=torch.tensor([[3.]])
logit0,tv0=tiny.forward(x); cv0=tiny.cost_head(tiny.inner.trunk(x))
tl=.5*(tv0-0).square().mean(); base=torch.autograd.grad(tl,(tiny.inner.trunk.weight,tiny.inner.value_head.weight),retain_graph=True,allow_unused=True)
assert base[0] is not None and abs(float(base[0])-450)<1e-5
try:
    with critic_input_detached(tiny):
        logit1,tv1=tiny.forward(x); cv1=tiny.cost_head(tiny.inner.trunk(x)); tl1=.5*(tv1-0).square().mean()
        new=torch.autograd.grad(tl1,(tiny.inner.trunk.weight,tiny.inner.value_head.weight),allow_unused=True)
        assert torch.equal(logit0,logit1) and torch.equal(tv0,tv1) and torch.equal(cv0,cv1)
        assert new[0] is None and new[1] is not None and abs(float(new[1])-180)<1e-5
        # Actor path remains connected while the critic head's representation is detached.
        actor_grad=torch.autograd.grad(logit1[...,SEND].sum(),tiny.inner.trunk.weight)[0]
        assert abs(float(actor_grad)-12)<1e-5
        raise LookupError('synthetic exception cleanup test')
except LookupError: pass
assert len(tiny.inner.value_head._forward_pre_hooks)==0 and len(tiny.cost_head._forward_pre_hooks)==0
for _ in range(3):
    with critic_input_detached(tiny): pass
assert len(tiny.inner.value_head._forward_pre_hooks)==0 and len(tiny.cost_head._forward_pre_hooks)==0
dump('PURE_TEST_EVIDENCE.json',{'kind':'round52-pure-test-evidence/1','status':'passed','batch_baseline_matches_round51':True,'scalar_loss_terms_and_predictions_equal':True,'actor_entropy_gradient_equal':True,'critic_heads_gradient_equal':True,'critic_shared_trunk_gradient_none_or_zero':True,'independent_module_parameter_enumeration':True,'isolated_total_gradient_sum_closed':True,'critic_head_detach_preserves_actor_path':True,'exception_hooks_removed':True,'repeated_calls_do_not_accumulate_hooks':True,'wrong_model_role_refusal':wrong_role_rejected,'round49_r43_role_mismatch_source':r43_role,'optimizer_steps':0,'parameter_writes':0,'checkpoint_writes':0,'environment_construction':0,'simulation':0,'new_scenes_or_flows':0,'new_evaluation':0,'real_training_updates':0,'synthetic_optimizer_steps':0,'test_scope':'only source-bound forward/gradient, exact 576-row support, and scalar closed-form prototype seam'})
print(json.dumps({'seed_norms':{s:{'old':v['total_old']['l2'],'isolated':v['total_isolated']['l2'],'old_clip':v['old_clip_scale'],'new_clip':v['isolated_clip_scale']} for s,v in all_results.items()},'direction_rows':len(directions),'clip_norm':CLIP_NORM,'analytic':{'old_trunk_grad':float(base[0]),'new_trunk_grad':'None','head_grad':float(new[1])}}))
