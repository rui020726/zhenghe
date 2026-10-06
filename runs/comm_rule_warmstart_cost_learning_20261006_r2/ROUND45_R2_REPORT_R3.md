# ROUND45-R2 执行报告

- 分支：`exp/comm-rule-warmstart-cost-learning-r2`
- 当前 HEAD：`2446499b80d0d96096cdcba499a7314fc1fb9b38`
- 执行阶段：stopped with preserved R2 evidence
- 运行器版本：`counted-jointrun-proxy/1+eval-key-index/1+round45-bc-warmstart/1+round45-bc-nested-controller-state/2+round45-postbc-readback/1+round45-attempt/2+r2-durable-key-consumer/1+round45-r2-four-module-bc-eval/1+round45-r2-bc-receipt-adoption/4+round45-r2-empty-transcript-zero-closure/3+round45-r2-r3-prelaunch-identity/2+round45-r2-adopted-ledger-snapshot/1`
- 训练语义版本：`constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/dual-eta-lambda-0p5/1/rule-bc-late147-init/1/published-completion-event-at-relay-tick/2`

## 计数

- BC：本次 3000 次真实更新；含旧失败共 4000 次。没有重跑 BC。
- 环境构造：0 次；run 调用：0 次；成功返回：0 次。
- 正常 claim：1/528；技术 reserve：0/24；总 claim：1/552。
- PPO 更新：0；Adam step：0。

## 门禁与结果

- 执行在门禁/运行错误后停止，保留所有 claim、row、日志和检查点。
- 错误：`RuntimeError: R2 frozen tracked source changed: ['M runs/comm_rule_warmstart_cost_learning_20261006_r2/ROUND45_CURRENT_TASK.json']`

## 限制

- BC 控制通过只代表固定规则初始化满足进入训练的预设门槛；不代表 RL 增量价值。
- R43 结果仅作历史记录；主要模型对照是固定 R44 NEW 矩阵。
- 16 个开发场景不是独立确认集。未调参、未选 seed、未追加场景或训练。
- 未推送、合并或部署。
