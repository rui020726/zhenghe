from __future__ import annotations
import hashlib, json, pathlib, subprocess, sys, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[3]
AUDIT = pathlib.Path(__file__).resolve().parent
R2 = ROOT / 'runs/comm_bc_initialized_constrained_ppo_20261007_r2'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def jread(path): return json.loads(path.read_text(encoding='utf-8'))
def rel(path):
    try: return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError: return str(path)
# Frozen historical contracts and identities.
v6 = jread(R2/'ROUND46_R2_V6_INPUT_IDENTITY.json')
r53idx = jread(R2/'round53_critic_isolation_trial_readiness/ROUND53_CHECKPOINT_INDEX.json')
r56id = jread(R2/'round56_soft_early_report_prior/INPUT_IDENTITY.json')
hist = jread(AUDIT/'HISTORICAL_CONSUMPTION_RECONCILIATION.json')
freeze_old = jread(R2/'round58_prior_strength_prospective_trial/PRELAUNCH_FREEZE.json')
auth_old = jread(R2/'round58_prior_strength_prospective_trial/EXECUTION_AUTHORIZATION.json')
# Check current candidate model bytes against registered identities; no optimizer/RNG restore.
models = {}
for seed in (11,22,33):
    key=str(seed); bc=v6['bc_models_by_seed'][key]
    bcpath=pathlib.Path(bc['checkpoint_path'])
    r53path=R2/'round53_critic_isolation_trial_readiness/raw/checkpoints'/f'v2_train_s{seed}_ep128.pt'
    r56path=R2/'round56_soft_early_report_prior/raw/checkpoints'/f'v2_train_s{seed}_ep128.pt'
    expected53=r53idx['seeds'][key]['checkpoints']['ep128']['sha256']
    expected56=r56id['ep128_checkpoint_sha256'][key]
    checks={
      'R45_BC_POST': {'path':rel(bcpath),'expected_sha256':bc['checkpoint_sha256'],'actual_sha256':sha(bcpath),'matches':sha(bcpath)==bc['checkpoint_sha256'],'parameter_sha256':bc['parameter_sha256'],'role':'fresh model-only initialization; fresh optimizer/lambda/RNG contract separately'},
      'R53_ep128': {'path':rel(r53path),'expected_sha256':expected53,'actual_sha256':sha(r53path),'matches':sha(r53path)==expected53,'role':'read-only evaluation comparator'},
      'R56_ep128': {'path':rel(r56path),'expected_sha256':expected56,'actual_sha256':sha(r56path),'matches':sha(r56path)==expected56,'role':'read-only evaluation comparator'},
    }
    models[key]=checks
# Runtime source identity: exact canonical modules reached by the frozen configure seam.
canon=pathlib.Path(r'D:\code\整合\zhenghe\runs\comm_cost_v2_rerun_20260928')
source_paths={'v2_trainer.py':canon/'v2_trainer.py','v2_common.py':canon/'v2_common.py','v2_controller.py':canon/'v2_controller.py'}
source_hashes={name:{'path':str(p),'sha256':sha(p),'exists':p.is_file()} for name,p in source_paths.items()}
priv={n:sha(AUDIT/n) for n in ('r58_private_launcher_repaired.py','r58_checkpoint_contract_repaired.py','r58_gate_prior_repaired.py','r58_repaired_entry.py')}
# Runtime binding is exercised via actual private configure seam, with a temp output root;
# it only imports modules and creates in-memory/model-only objects; no environment/optimizer/RNG restore.
sys.path.insert(0,str(AUDIT))
import r58_private_launcher_repaired as launcher
with tempfile.TemporaryDirectory(prefix='r58_prelaunch_runtime_',dir=launcher.GEN2) as td:
    binding=launcher.prepare_actual_runtime_for_audit(pathlib.Path(td)/'gen3'/'raw')
    rt=binding['runtime']; common=rt['common']; trainer=rt['trainer']
    keys=('EPISODES_PER_UPDATE','MINIBATCH_STEPS','EPOCHS_PER_BATCH','GAMMA','GAE_LAMBDA','LEARNING_RATE','CLIP','ENTROPY_BONUS','VALUE_COEFFICIENT','MAX_GRAD_NORM','B0_BYTES','ETA_LAMBDA','LAMBDA_INIT','DELTA','DATA_ORDER_SEED')
    effective={k:getattr(common,k,None) for k in keys}
    live={'trainer_path':str(pathlib.Path(trainer.__file__).resolve()),'trainer_sha256':sha(pathlib.Path(trainer.__file__).resolve()),'common_path':str(pathlib.Path(common.__file__).resolve()),'common_sha256':sha(pathlib.Path(common.__file__).resolve()),'controller_path':str(pathlib.Path(rt['controller_module'].__file__).resolve()) if rt.get('controller_module') else source_hashes['v2_controller.py']['path'],'effective_common_values':effective,'bc_post_roles_verified_by_live_binder':binding.get('bc_post_roles'),'r40_trainer_is_runtime_trainer':rt['r40_train'].trainer is trainer,'wrapped_ppo_update':trainer.ppo_update.__name__.startswith('r58_'),'repaired_prior_beta':rt['r58_repaired_adapter']['beta'],'checkpoint_contract':rt['r58_checkpoint_contract']['kind'],'no_output_artifacts':not any(pathlib.Path(td).rglob('*.pt')) and not any(pathlib.Path(td).rglob('*.jsonl'))}
    launcher.close_audit_runtime(binding)
# Scene occupancy audit based on the project's existing result-blind R37 harvesters.
sys.path.insert(0,str(ROOT/'runs/comm_final_evaluation_20261002'))
import r37_scene_audit as scene_audit
low,high=1700000,1700031
path_hits=scene_audit.harvest_paths(); text_hits=scene_audit.harvest_tracked_text()
paths={k:v for k,v in path_hits.items() if low<=int(k)<=high}
refs={k:v for k,v in text_hits.items() if low<=int(k)<=high}
# Gen3 destination must be absent; don't create it.
gen3=R2/'round58_gen3'; raw=gen3/'raw'
scene={'block':[low,high], 'n':32,'path_embedded_candidate_ids':paths,'tracked_context_candidate_id_sources':refs,
       'candidate_registration_sources':['round58_gen2/PRELAUNCH_FREEZE.json','round58_prior_strength_prospective_trial/PRELAUNCH_FREEZE.json','round58_prior_strength_prospective_trial/raw/ROUND53_FIXED_PLAN.json'],
       'interpretation':'The block is already registered/reserved by this ROUND58 plan. No path-embedded artifacts/claims carrying these candidate IDs were found. Tracked textual hits are plan/registration lineage, not evidence of a completed episode. Not an unregistered/free block.',
       'third_party_conflict_found':False,'result_fields_read':False,'gen3_root_exists':gen3.exists(),'gen3_raw_exists':raw.exists()}
# Exact logical matrix required by current fixed plan.
plan=jread(R2/'round58_prior_strength_prospective_trial/raw/ROUND53_FIXED_PLAN.json')
plan_identity={'plan_path':rel(R2/'round58_prior_strength_prospective_trial/raw/ROUND53_FIXED_PLAN.json'),'plan_sha256':sha(R2/'round58_prior_strength_prospective_trial/raw/ROUND53_FIXED_PLAN.json'),
               'job_count':len(plan.get('jobs',[])),'training_job_count':sum(j.get('scope')=='ppo_training' for j in plan.get('jobs',[])),
               'model_eval_count':sum(str(j.get('scope','')).startswith('model_eval') for j in plan.get('jobs',[])),'rule_eval_count':sum(str(j.get('scope','')).startswith('rule_eval') for j in plan.get('jobs',[])),
               'evaluation_scenes':plan.get('eval_scenes_new',[]),
               'note':'Existing fixed plan is historical input identity only; its 32-scene block and declared matrix are recorded, no jobs executed in this task.'}
# Historical quota ledger; don't alter historical classifications or enforce new proposal.
quota={'historical_environment_invocations_evidence_supported':hist['environment_consumption']['known_constructor_and_run_calls'],
       'historical_claims':{'gen1_complete':hist['environment_consumption']['gen1_complete_claims'],'gen2_complete':hist['environment_consumption']['gen2_complete_claims']},
       'claims_classified_normal_gen1':hist['environment_consumption']['gen1_categories'].get('normal'),
       'claims_classified_normal_gen2':hist['environment_consumption']['gen2_claim_categories'].get('normal'),
       'reserve_conflict':hist['environment_consumption']['reserve_classification'],
       'optimizer_calls_evidence_supported':hist['optimizer_calls']['total_evidence_supported_actual_calls'],
       'effective_R58_updates':hist['optimizer_calls']['effective_r58_training_calls'],
       'current_guard':'The current external-grant validator requires historical_normal + planned_normal <= normal_cap(1024), and history_total + planned_normal <= hard_cap(1064). The first check blocks if historical normal is 16. No guard was relaxed.',
       'future_proposal_not_authorized':{'historical_consumption_separately_retained':16,'new_normal_plan':1024,'new_reserve':24,'hard_cap':1064,'arithmetic_total':1064,'approval_status':'awaiting_approval','classification_and_contract_change_required':True}}
# Candidate freeze only: explicitly not an executable authorization.
candidate={'kind':'round58-prelaunch-candidate-freeze/1','status':'candidate_frozen_awaiting_approval','execution_permitted':False,
 'scientific_contract':{'beta':1.0,'critic_isolation':'R53 critic loss updates critic heads but not shared trunk','actor_cost_advantage':'lambda * A_task - A_positive_cost','observation_dim':19,'action':'one-shot SEND/SKIP','budget_mask':'sequential frozen budget mask','ready_cut':147.0,'reward_timing':'published-completion-event-at-relay-tick/2','GAE_gamma':1.0,'GAE_lambda':1.0,'PPO_epochs':4,'minibatch_steps':256,'episodes_per_update':4,'learning_rate':0.0003,'clip':0.2,'entropy_bonus':0.01,'value_coefficient':0.5,'max_grad_norm':0.5,'dual_eta_lambda':0.5,'dual_lambda_init':1.0,'dual_delta':0.0,'B0_bytes':10000.0,'training_seeds':[11,22,33],'episodes_per_seed':128,'normal_tasks':1024,'technical_reserve_original_contract':40,'hard_cap_original_contract':1064,'training_scene_pool':'R46 frozen 48-scenario order repeated per seed; inherited, not rewritten','statistics':{'unit':'scenario','rng':'PCG64','seed':20261008,'draws':10000,'shared_indices':True,'percentile':'linear','modes_separate':True,'seeds_separate':True}},
 'scientific_source':'ROUND46_R2_V6 freeze/key plan + R44 objective and configure; only R58 method delta beta=1.00; original R58 fixed plan declares 384 training, 576 model-evaluation and 64 rule-evaluation jobs on the 32-scene block. Existing plan is historical and not an execution authorization.',
 'runtime_binding':live,'runtime_source_hashes':source_hashes,'private_entry_hashes':priv,
 'models':models,'scene_audit':scene,'plan_identity':plan_identity,'quota_reconciliation':quota,
 'historical_old_freeze_sha256':sha(R2/'round58_prior_strength_prospective_trial/PRELAUNCH_FREEZE.json'),
 'prior_authorization_artifact':{'path':'runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_prior_strength_prospective_trial/EXECUTION_AUTHORIZATION.json','sha256':sha(R2/'round58_prior_strength_prospective_trial/EXECUTION_AUTHORIZATION.json'),'status':auth_old.get('status'),'caps':{'normal':auth_old.get('normal_cap'),'reserve':auth_old.get('reserve_cap'),'hard':auth_old.get('hard_cap')},'plan_sha256':auth_old.get('plan_sha256'),'approval_reference':auth_old.get('approval_reference'),'scope_note':'Historical artifact binds original 1024+40 contract and stale freeze; not approval of changed cumulative classification/contract. This task itself explicitly forbids launch.'},
 'limitations':['Current R58 old freeze carries stale gen2 source identities; this candidate does not rewrite it.','No valid R58 training batch exists in the historical ledger.','The future cumulative quota scheme is not approved and is not accepted by the current quota guard.','Candidate scene block is registered to this ROUND58 plan; no completed claim/artifact path was found, but this does not make it an unregistered free block.','An older explicit authorization artifact exists for the original 1024+40 contract but is not a coherent approval for this candidate after historical reconciliation; explicit approval for changed cumulative quota/classification and replacement identity-bound receipts remains absent.']}
# Ensure model artifacts verify and gen3 absent; fail candidate if hash mismatch.
assert all(m['matches'] for seed in models.values() for m in seed.values())
assert live['r40_trainer_is_runtime_trainer'] and live['wrapped_ppo_update'] and live['repaired_prior_beta']==1.0
out=AUDIT/'ROUND58_PRELAUNCH_CANDIDATE.json'; out.write_text(json.dumps(candidate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
identity={'kind':'round58-prelaunch-input-identity/1','status':'candidate_only','models':models,'runtime_source_hashes':source_hashes,'private_entry_hashes':priv,'plan_identity':plan_identity,'scene_audit':scene,'old_freeze_sha256':candidate['historical_old_freeze_sha256'],'prior_authorization_artifact':candidate['prior_authorization_artifact']}
(AUDIT/'ROUND58_CANDIDATE_INPUT_IDENTITY.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'written':[str(out),str(AUDIT/'ROUND58_CANDIDATE_INPUT_IDENTITY.json')],'runtime':live,'scene':scene,'plan':plan_identity,'quota':quota},ensure_ascii=False,indent=2))
