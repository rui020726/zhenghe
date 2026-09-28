# ROUND25 判定：固定 seed33 greedy 候选在 64 个新场景上的复验

- run_id: `comm_cost_greedy_replication_20260928`
- 问题：开发集事后选定的 seed33 最终 greedy 模型，在新场景上能否比固定规则少花已计量成本，同时未观察到按时任务数损失？

## 判定字段

- `primary_cost_evidence`: **lower_cost_supported**
- `observed_task_count_status`: **task_count_tradeoff_observed**
- `constraint_guarantee`: not_established
- `training_seed_stability`: not_established

## 主成本比较（seed33 greedy 对 late_discovery_gate_147）

- dC 均值 -696.3 B/场景，95% CI [-43626.6, -419.4]
- 成本计数：更省 34 / 相同 7 / 更高 23
- 任务计数（同一比较）：改善 4 / 相同 60 / 退化 0

## 任务比较（对 immediate 与对 late）

- 对 immediate：dY 均值 -0.0312，CI [-1.9375, -0.0312]；dC 均值 -11893.1 B，CI [-429606.0, -319740.2]

## 陈述

- 观察到任务数权衡：存在逐场景净按时任务数退化，逐场景公开，不用其他场景的改善抵消。

## 限定

- 候选是看过 ROUND24 16 个开发场景后从事后选出的固定模型；本结果检验此固定候选的泛化，不能消除模型筛选事实，也不能估计随机训练一次的成功率
- 64 场景是本轮资源约束下的固定探索性复验规模，不是已论证足以检验 delta=0 的功效样本
- 任务样本全打平不证明 delta=0 总体非劣；即使 64/64 无退化，1-0.05^(1/64)≈4.57% 也只是i.i.d. 场景假设下发生净退化场景概率的单侧 95% 上限，不是期望任务损失上限，更不是部署保证
- bootstrap 描述此固定候选在指定生成分布上的场景不确定性，不含训练初始化不确定性
- 训练 sampled 对偶约束不能转移为 greedy 保证；约束保证始终 not_established
- seed11/22 与 sampled 的 ROUND24 差异只作历史背景，不用一个候选成功宣称 3/3 成功
- 费用口径仅为已计量载荷尝试成本（丢包照计、拒付计0并单列），不是全系统通信量或真实无线能耗
- 192 次启动不是 192 个独立场景；未池化任何旧开发场景

## 发布状态

- remote_verification: NOT_PERFORMED
- publication_status: PENDING_NETWORK
- local execution complete; remote publication pending; no network attempt was made this session and no unpublished state is claimed as published
