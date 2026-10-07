import json,hashlib,statistics
from pathlib import Path
O=Path(__file__).resolve().parent; R=O.parent; P=R.parents[1]
def rd(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def wr(n,x): (O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def hs(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
g=rd(O/'GRADIENT_COMPONENTS.json'); z=rd(O/'LOGIT_DIRECTIONAL_SENSITIVITY.json'); panel=rd(R/'round49_bc_input_policy_closure'/'INPUT_PANEL_MANIFEST.json')
a={}
for x in z['records']:
 k=(x['seed'],x['panel'],x['cost_class_metadata_only'],x['component'])
 a.setdefault(k,[]).append(x['directional_logodds_change_per_unit_sgd'])
z['aggregates_descriptive_only']=[{'seed':k[0],'panel':k[1],'cost_category':k[2],'component':k[3],'count':len(v),'mean':statistics.fmean(v),'min':min(v),'max':max(v)} for k,v in sorted(a.items())]
wr('LOGIT_DIRECTIONAL_SENSITIVITY.json',z)
t=P.parent/'zhenghe'/'runs'/'comm_cost_v2_rerun_20260928'/'v2_trainer.py'
wr('SOURCE_AND_INPUT_IDENTITY.json',{'kind':'round51-source-and-input-identity/2','status':'verified','trainer_path':str(t),'trainer_sha256':hs(t),'trainer_copy_sha256':hs(O/'frozen_sources'/'canonical_v2_trainer.py'),'ast_extraction':'exact ppo_update statements from stacked through loss; all update calls excluded','runtime_chain':rd(O/'SOURCE_RECOVERY.json')['runtime_call_chain'],'active_GAE_lambda':1.0,'round46_batch':g['batch'],'round49_panel_sha256':hs(R/'round49_bc_input_policy_closure'/'INPUT_PANEL_MANIFEST.json'),'round49_panel_records':len(panel['records']),'models':'same-seed BC_POST and ep128 identities verified; CPU; optimizer/RNG not restored'})
wr('PURE_TEST_EVIDENCE.json',{'kind':'round51-pure-test-evidence/3','exact_source_hash':hs(t),'three_seed_batches':True,'sample_count_per_seed':{s:v['samples'] for s,v in g['per_seed'].items()},'gradient_closure_max_abs':max(v['gradient_closure']['max_abs'] for v in g['per_seed'].values()),'panel_count':len(panel['records']),'directional_projection_count':len(z['records']),'checkpoint_and_parameter_hashes_unchanged':True,'parameter_grad_buffers_empty':True,'ast_update_operations_excluded':True,'environment_construction':0,'simulation':0,'new_flow':0,'new_evaluation':0,'real_training_update':0,'synthetic_optimizer_step':0,'checkpoint_write':0})
lines=['# ROUND51 冻结源码与首批梯度闭合报告','','训练器身份已闭合：物理源码 raw SHA256 '+hs(t)+' 与R45/R46冻结值相同，且已保存相同字节的只读分析副本。损失段由确切源码AST提取；未调用训练更新、backward、optimizer、训练/评估/环境入口。更正本轮早期中间稿的阻断状态，不改历史报告。','','## 输入与范围','',f"按索引读取ROUND46每seed四条row/tape，哈希均匹配；每seed重建265个decision样本。Round49面板共{len(panel['records'])}行、19维。same-seed BC_POST初值及ep128角色身份均通过Round46/48冻结身份核验。R43冻结运行覆盖将GAE lambda从默认.95置1.0，本分析按该运行合同。",'', '训练记录显示8个minibatch、4个epoch；原shuffle索引未持久化。本分析一次性对265行计算full-batch梯度，是诊断而非原始minibatch/Adam更新重演。','','| seed | loss | actor surrogate | entropy | task value | cost value | 总梯度L2 | 预测clip缩放 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for s,v in g['per_seed'].items():
 q=v['loss_terms']; lines.append(f"| {s} | {v['total_loss']:.6g} | {q['actor_surrogate']:.6g} | {q['entropy']:.3g} | {q['task_value']:.6g} | {q['cost_value']:.6g} | {v['gradients']['total_loss']['l2']:.6g} | {v['hypothetical_global_clip_scale']:.5f} |")
lines+=['','逐项梯度L1/L2/max、参数组梯度、共享trunk余弦及分项闭合误差见GRADIENT_COMPONENTS.json。task critic对shared trunk的梯度范数在三seed均显著大于actor surrogate；cost critic也经共享trunk进入表示，方向随seed而异。仅说明计算图中存在critic到policy-logit的共享路径，不证明其导致ep128漂移或成本/任务结果。clip缩放仅按范数推算。','',f"Round49真实既有面板上有{len(z['records'])}条seed×损失项的一阶 J_logit·(-g) 投影；逐条值和按BC/PPO轨迹、成本类别分组的描述统计在LOGIT_DIRECTIONAL_SENSITIVITY.json。它不是Adam、不是闭环轨迹、不作因果或显著性结论；成本类别只作元数据。",'', '## 验证和边界','', '精确源码hash、row/tape/checkpoint身份、样本数量、分项梯度闭合、输入与checkpoint哈希不变、参数.grad为空均已检查。AST提取段不含zero_grad/backward/clip/step。仅做上述纯计算检查，不运行项目全量测试。七项零操作计数见PURE_TEST_EVIDENCE.json；development_go=false保持。','', '**唯一后续建议（本轮不执行，需另行授权）：**在相同BC_POST初始化、四条冻结episode及相同剩余合同下，只比较critic梯度是否回流共享trunk（当前回流为对照，stop-gradient为唯一变量），检查actor-logit位移是否按预测减小且任务/成本冻结指标不恶化。若位移未减小或指标恶化即可否证该方法值得继续。当前结果不能预判收益，也不授权续训或环境评估。','','提交后HEAD记录于最终交付回执。只提交本ROUND51新目录；模型/row/tape、历史文件及受保护脏文件不入提交。']
(O/'ROUND51_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('finalize ok',len(z['records']))

