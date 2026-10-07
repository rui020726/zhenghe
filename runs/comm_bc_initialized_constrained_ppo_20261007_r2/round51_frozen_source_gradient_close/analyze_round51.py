from __future__ import annotations
import ast, hashlib, json, random, sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[1]
OUT = Path(__file__).resolve().parent
SEEDS=(11,22,33)
EXPECTED_TRAINER='0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def jload(p): return json.loads(Path(p).read_text(encoding='utf-8'))

CANON=REPO.parent/'zhenghe'/'runs'/'comm_cost_v2_rerun_20260928'
TRAINER=CANON/'v2_trainer.py'
assert sha(TRAINER)==EXPECTED_TRAINER
SIB=REPO.parent/'zhenghe'
sys.path[:0]=[str(x) for x in [CANON,SIB/'runs'/'comm_cost_stage_b_20260927',SIB/'runs'/'comm_tight_budget_20260919',SIB/'runs'/'comm_budget_pilot_20260919',REPO.parent/'zhenghe-round40'/'runs'/'comm_first_report_training_ablation_20261003']]
import v2_trainer as trainer
import v2_controller
import importlib.util
spec=importlib.util.spec_from_file_location('r48round51',RUN/'round48_policy_drift_probe'/'r48_probe.py')
r48=importlib.util.module_from_spec(spec); spec.loader.exec_module(r48)

# The R43 frozen override is the active runtime route; do not call runtime setup.
r43=jload(REPO/'runs'/'comm_one_shot_full_trace_ablation_20261004'/'ROUND43_FROZEN_CONFIG.json')['round43']
assert trainer.common.GAE_LAMBDA==r43['gae_lambda_old'] and r43['gae_lambda']==1.0
trainer.common.GAE_LAMBDA=float(r43['gae_lambda'])

# Extract the exact loss-producing AST statements from the frozen ppo_update loop.
src=TRAINER.read_text(encoding='utf-8'); tree=ast.parse(src)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='ppo_update')
outer=next(n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='start')
body=[]; started=False
for s in outer.body:
    if isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='stacked' for t in s.targets): started=True
    if not started: continue
    if isinstance(s,ast.Expr) and isinstance(s.value,ast.Call) and ast.unparse(s.value.func)=='optimizer.zero_grad': break
    body.append(s)
assert any(isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in s.targets) for s in body)
assert not any(isinstance(n,ast.Call) and ast.unparse(n.func).endswith(('backward','step','zero_grad','clip_grad_norm_')) for s in body for n in ast.walk(s))
ret=ast.Return(value=ast.Dict(keys=[ast.Constant(k) for k in ['loss','surrogate','entropy_t','value_task_loss','value_cost_loss','new_log_prob_t','stacked']],values=[ast.Name(id=k,ctx=ast.Load()) for k in ['loss','surrogate','entropy_t','value_task_loss','value_cost_loss','new_log_prob_t','stacked']]))
helper=ast.FunctionDef(name='exact_batch_loss',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='controller'),ast.arg(arg='chunk')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body+[ret],decorator_list=[])
mod=ast.fix_missing_locations(ast.Module(body=[helper],type_ignores=[])); ns=trainer.__dict__; exec(compile(mod,str(TRAINER),'exec'),ns); exact_batch_loss=ns['exact_batch_loss']

panel_path=RUN/'round49_bc_input_policy_closure'/'INPUT_PANEL_MANIFEST.json'
panel=jload(panel_path); records=panel['records']
assert len(records)==576
cost=jload(RUN/'round47_cost_decomposition'/'PER_SCENE_COST_DECOMPOSITION.json')
def category(seed,scene,target):
    sc=cost['per_seed'][str(seed)]['vs_bc']['scenes'][str(scene)]
    if target in sc['A_model_only_targets']: return 'PPO_only_vs_BC'
    if target in sc['B_bench_only_targets']: return 'BC_only_vs_PPO'
    if target in sc['common_targets']: return 'common_send'
    return 'neither_sends'
def norm_params(c): return list(c.inner.trunk.parameters())+list(c.inner.action_head.parameters())+list(c.inner.value_head.parameters())+list(c.cost_head.parameters())
def module_hash(c):
    state=c.state_dict(); h=hashlib.sha256()
    for role in sorted(state):
        for k,v in sorted(state[role].items()): h.update((role+'.'+k).encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def controller_for(seed,path,expected):
    py=random.getstate(); nr=np.random.get_state(); tr=torch.random.get_rng_state()
    try:
        c=v2_controller.fresh_controller(seed)
        payload=torch.load(path,map_location='cpu',weights_only=False)
        st=payload['controller']
        c.inner.trunk.load_state_dict(st['trunk']); c.inner.action_head.load_state_dict(st['action_head']); c.inner.value_head.load_state_dict(st['value_task_head']); c.cost_head.load_state_dict(st['value_cost_head'])
        c.inner.trunk.eval(); c.inner.action_head.eval(); c.inner.value_head.eval(); c.cost_head.eval()
    finally: random.setstate(py); np.random.set_state(nr); torch.random.set_rng_state(tr)
    got=module_hash(c); assert got==expected,(seed,got,expected)
    return c

def ckpt_identity(seed,role):
    gate=jload(RUN/'v6'/f'ROUND45_R2_FIRST_BATCH_GATE_R3_s{seed}.json')
    if role=='ep0':
        path=REPO/'runs'/'comm_rule_warmstart_cost_learning_20261006_r2'/'checkpoints'/f'BC_POST_MODEL_s{seed}.pt'
        exp=gate['bc_parameter_sha256']
        filehash=gate['bc_checkpoint_sha256']
    else:
        path=RUN/'v6'/'checkpoints'/f'v2_train_s{seed}_ep128.pt'
        ids=jload(RUN/'round48_policy_drift_probe'/'MODEL_ROLE_IDENTITY.json')
        exp=ids['roles']['ep128_final'][str(seed)]['parameter_sha256_recomputed']
        filehash=ids['roles']['ep128_final'][str(seed)]['file_sha256']
    assert sha(path)==filehash,(path,sha(path),filehash)
    return path,exp,filehash

# Exact source batch reconstruction from the four indexed episode artifacts per seed.
batch_out={}; grad_results={}; panel_sens=[]; file_pre={}; param_pre={}
for seed in SEEDS:
    recpath=RUN/'v6'/'checkpoints'/f'r40_batch_s{seed}_b0_attempt01.json'; rec=jload(recpath)
    episodes=[]; srcs=[]
    for art in rec['episode_artifacts']:
        base=f"v2_r44_train_s{seed}_e{art['episode_index']}_s{art['scenario_seed']}_a{art['attempt']}"
        rowp=RUN/'v6'/'rows'/f'{base}.json'; tapep=RUN/'v6'/'logs'/f'{base}_tape.jsonl'
        assert sha(rowp)==art['row_sha256'] and sha(tapep)==art['tape_sha256']
        row=jload(rowp); tapes=[json.loads(x) for x in tapep.read_text(encoding='utf-8').splitlines() if x.strip()]
        episodes.append({'collector':SimpleNamespace(records=tapes),'steps':row['steps'],'cost_stream':row['cost_stream'],'task_stream':row['task_stream'],'lambda':float(row['lambda']),'episode_index':int(art['episode_index']),'scenario_seed':int(art['scenario_seed'])})
        srcs.append({'episode':art['episode_index'],'scene':art['scenario_seed'],'row':str(rowp),'row_sha256':sha(rowp),'tape':str(tapep),'tape_sha256':sha(tapep),'tape_rows':len(tapes)})
    samples=trainer.batch_samples(episodes); assert len(samples)==265
    path0,exp0,fh0=ckpt_identity(seed,'ep0'); model=controller_for(seed,path0,exp0)
    assert trainer.likelihood_alignment(model,samples)['passed']
    params=norm_params(model); assert sum(p.numel() for p in params)==5700
    groups={'shared_trunk':list(model.inner.trunk.parameters()),'action_head':list(model.inner.action_head.parameters()),'task_value_head':list(model.inner.value_head.parameters()),'cost_value_head':list(model.cost_head.parameters())}
    assert sum(len(v) for v in groups.values())==len(params) and len({id(p) for ps in groups.values() for p in ps})==len(params)
    pred=exact_batch_loss(model,samples)
    # Decompose directly from exact source-produced tensors/constants.
    comps={'actor_surrogate':-pred['surrogate'].mean(),'entropy':-trainer.common.ENTROPY_BONUS*pred['entropy_t'].mean(),'task_value':trainer.common.VALUE_COEFFICIENT*pred['value_task_loss'],'cost_value':trainer.common.VALUE_COEFFICIENT*pred['value_cost_loss']}
    assert abs(float(sum(comps.values()).detach()-pred['loss'].detach()))<1e-6
    gcomp={}; metrics={}
    for name,loss in [*comps.items(),('total_loss',pred['loss'])]:
        gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        flat=torch.cat([(g.detach().reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)) for p,g in zip(params,gs)])
        gcomp[name]=flat
        metrics[name]={'l1':float(flat.abs().sum()),'l2':float(torch.linalg.vector_norm(flat)),'max_abs':float(flat.abs().max())}
    closure=sum(gcomp[k] for k in comps)-gcomp['total_loss']
    norm=float(torch.linalg.vector_norm(gcomp['total_loss'])); clip=min(1.0,0.5/norm) if norm else 1.0
    group_spans={}; offset=0
    for gn,ps in groups.items():
        n=sum(p.numel() for p in ps); group_spans[gn]=(offset,offset+n); offset+=n
    grouped={}
    for cn,vec in gcomp.items():
        grouped[cn]={}
        for gn,(lo,hi) in group_spans.items():
            gv=vec[lo:hi]
            grouped[cn][gn]={'l1':float(gv.abs().sum()),'l2':float(torch.linalg.vector_norm(gv)),'max_abs':float(gv.abs().max())}
    def cosine(a,b):
        aa=gcomp[a][group_spans['shared_trunk'][0]:group_spans['shared_trunk'][1]]; bb=gcomp[b][group_spans['shared_trunk'][0]:group_spans['shared_trunk'][1]]
        den=float(torch.linalg.vector_norm(aa)*torch.linalg.vector_norm(bb))
        return None if den==0 else float(torch.dot(aa,bb)/den)
    path128,exp128,fh128=ckpt_identity(seed,'ep128'); final=controller_for(seed,path128,exp128)
    assert sha(path128)==fh128 and module_hash(final)==exp128
    grad_results[str(seed)]={'samples':len(samples),'target_rows':sum(len(s['targets']) for s in samples),'free_rows':sum(len(s['free']) for s in samples),'masked_rows':sum(len(s['targets'])-len(s['free']) for s in samples),'episodes':srcs,'loss_terms':{k:float(v.detach()) for k,v in comps.items()},'total_loss':float(pred['loss'].detach()),'gradients':metrics,'gradient_by_parameter_group':grouped,'shared_trunk_cosine':{'actor_vs_task_value':cosine('actor_surrogate','task_value'),'actor_vs_cost_value':cosine('actor_surrogate','cost_value'),'actor_vs_entropy':cosine('actor_surrogate','entropy'),'task_vs_cost_value':cosine('task_value','cost_value')},'gradient_closure':{'l2':float(torch.linalg.vector_norm(closure)),'max_abs':float(closure.abs().max())},'hypothetical_global_clip_scale':clip,'actual_minibatches':rec['update']['minibatches'],'full_batch_is_not_replayed_minibatch':True,'ep128_identity_verified':{'path':str(path128),'file_sha256':fh128,'parameter_sha256':module_hash(final)}}
    # Parameter-direction logit sensitivity on every Round49 panel row; Jacobian-vector products are diagnostic only.
    for group_name,gg in [('actor_surrogate',gcomp['actor_surrogate']),('entropy',gcomp['entropy']),('task_value',gcomp['task_value']),('cost_value',gcomp['cost_value']),('total_loss',gcomp['total_loss'])]:
        offset=0; gmap={}
        for ps in groups.values():
            for p in ps:
                n=p.numel(); gmap[id(p)]=gg[offset:offset+n].reshape_as(p); offset+=n
        for row in records:
            x=torch.tensor(row['features'],dtype=torch.float32)
            logits,_=model.forward(x); z=logits[...,1]-logits[...,0]
            jv=torch.autograd.grad(z.sum(),params,retain_graph=True,allow_unused=True)
            direction=sum(((j.detach()*(-gmap[id(p)])).sum() for p,j in zip(params,jv) if j is not None),torch.tensor(0.0))
            panel_sens.append({'seed':seed,'panel':row['panel'],'scene':row['scene'],'target_id':row['target'],'feature_sha256':row['features_sha256'],'cost_class_metadata_only':category(seed,row['scene'],row['target']),'component':group_name,'directional_logodds_change_per_unit_sgd':float(direction)})
    assert module_hash(model)==exp0 and all(p.grad is None for p in params)
    file_pre[str(path0)]=fh0; file_pre[str(path128)]=fh128
    param_pre[str(seed)]={'ep0':module_hash(model),'ep0_expected':exp0}
    batch_out[str(seed)]={'checkpoint_ep0':str(path0),'checkpoint_ep0_sha256':fh0,'parameter_sha256':exp0,'batch_record':str(recpath),'batch_record_sha256':sha(recpath),'samples':len(samples),'gae_lambda':trainer.common.GAE_LAMBDA}

# First-order sensitivities from autograd above; verify no source input changed.
for p,h in file_pre.items(): assert sha(p)==h
dump('GRADIENT_COMPONENTS.json',{'kind':'round51-gradient-components/2','status':'computed_exact_frozen_source','source_sha256':sha(TRAINER),'source_path':str(TRAINER),'source_ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),'batch':batch_out,'per_seed':grad_results,'operations':{'optimizer_step':0,'backward':0,'checkpoint_write':0,'environment':0}})
dump('LOGIT_DIRECTIONAL_SENSITIVITY.json',{'kind':'round51-logit-directional-sensitivity/2','status':'descriptive_first_order_only','panel_sha256':sha(panel_path),'records':panel_sens,'unique_panel_rows':len(records),'note':'autograd Jacobian-vector projection on exact batch gradients; not Adam, not a training replay, not a causal attribution'})
dump('PURE_TEST_EVIDENCE.json',{'kind':'round51-pure-test-evidence/2','source_ast_extracts_loss_without_update_calls':True,'gradient_component_closure_max_abs':max(v['gradient_closure']['max_abs'] for v in grad_results.values()),'all_checkpoint_hashes_unchanged':True,'all_loaded_parameter_hashes_verified':True,'environment':0,'simulation':0,'new_flow':0,'new_evaluation':0,'real_training_update':0,'synthetic_optimizer_step':0,'checkpoint_write':0})
print(json.dumps({'seeds':{k:{'loss':v['total_loss'],'grad':v['gradients']['total_loss'],'clip':v['hypothetical_global_clip_scale']} for k,v in grad_results.items()},'sensitivity_rows':len(panel_sens)}))
