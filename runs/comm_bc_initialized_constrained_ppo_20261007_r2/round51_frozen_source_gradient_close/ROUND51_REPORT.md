# ROUND51 冻结源码与首批梯度闭合报告

训练器身份已闭合：物理源码 raw SHA256 0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841 与R45/R46冻结值相同，且已保存相同字节的只读分析副本。损失段由确切源码AST提取；未调用训练更新、backward、optimizer、训练/评估/环境入口。更正本轮早期中间稿的阻断状态，不改历史报告。

## 输入与范围

按索引读取ROUND46每seed四条row/tape，哈希均匹配；每seed重建265个decision样本。Round49面板共576行、19维。same-seed BC_POST初值及ep128角色身份均通过Round46/48冻结身份核验。R43冻结运行覆盖将GAE lambda从默认.95置1.0，本分析按该运行合同。

训练记录显示8个minibatch、4个epoch；原shuffle索引未持久化。本分析一次性对265行计算full-batch梯度，是诊断而非原始minibatch/Adam更新重演。

| seed | loss | actor surrogate | entropy | task value | cost value | 总梯度L2 | 预测clip缩放 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | 3.85468 | -3.19909 | -2.17e-07 | 6.99422 | 0.0595487 | 10.5863 | 0.04723 |
| 22 | 5.82626 | -4.07173 | -3.51e-07 | 9.84469 | 0.053304 | 25.4647 | 0.01964 |
| 33 | 4.89626 | -2.89494 | -2.39e-07 | 7.57494 | 0.216265 | 18.8699 | 0.02650 |

逐项梯度L1/L2/max、参数组梯度、共享trunk余弦及分项闭合误差见GRADIENT_COMPONENTS.json。task critic对shared trunk的梯度范数在三seed均显著大于actor surrogate；cost critic也经共享trunk进入表示，方向随seed而异。仅说明计算图中存在critic到policy-logit的共享路径，不证明其导致ep128漂移或成本/任务结果。clip缩放仅按范数推算。

Round49真实既有面板上有8640条seed×损失项的一阶 J_logit·(-g) 投影；逐条值和按BC/PPO轨迹、成本类别分组的描述统计在LOGIT_DIRECTIONAL_SENSITIVITY.json。它不是Adam、不是闭环轨迹、不作因果或显著性结论；成本类别只作元数据。

## 验证和边界

精确源码hash、row/tape/checkpoint身份、样本数量、分项梯度闭合、输入与checkpoint哈希不变、参数.grad为空均已检查。AST提取段不含zero_grad/backward/clip/step。仅做上述纯计算检查，不运行项目全量测试。七项零操作计数见PURE_TEST_EVIDENCE.json；development_go=false保持。

**唯一后续建议（本轮不执行，需另行授权）：**在相同BC_POST初始化、四条冻结episode及相同剩余合同下，只比较critic梯度是否回流共享trunk（当前回流为对照，stop-gradient为唯一变量），检查actor-logit位移是否按预测减小且任务/成本冻结指标不恶化。若位移未减小或指标恶化即可否证该方法值得继续。当前结果不能预判收益，也不授权续训或环境评估。

提交后HEAD记录于最终交付回执。只提交本ROUND51新目录；模型/row/tape、历史文件及受保护脏文件不入提交。
