# ROUND53 执行报告 — critic 梯度隔离固定训练与开发评估

状态：**技术执行完整闭合；development_go = false**。

## 执行完整性

- 正常任务 528/528 全部 complete（训练 384 + 候选 greedy/sampled 评估 96 + 真 BC_POST sampled 评估 48）；技术储备 0/24 使用；总启动 528 ≤ 552 硬上限。
- 实际环境构造 528 次、成功 episode 528、完成步数 528,000、训练批 96、PPO 更新 96、optimizer step 608、checkpoint 写入 297。失败重试 0。
- 首 12 集（每 seed 首 4 集）真实首批动态门通过：BC 初值身份、fresh optimizer/λ=1、登记 R44 ep0 RNG、critic→trunk 切断、critic head 训练保留、mask/likelihood/正成本时序、GAE/组合优势/g_hat/λ 更新、checkpoint 版本与游标恢复。
- claim 与 528 个 durable slot 严格闭合（strict=true）；派生快照与核心 claim 分离。

## 逐 seed 结果（16 个开发场景，1301000..1301015）

所有臂、所有 seed、所有场景 on-time 任务数均为 6/6（任务上限饱和），greedy 与 immediate 的按时目标 ID 集合逐场景完全一致（丢失目标 0、救回目标 0）。

平均已计量字节（B/集，greedy）：

| seed | 隔离 PPO | vs 阈值 4838.25 | vs BC greedy（点估计差） | vs 共享 PPO（点估计差） | vs immediate | vs late |
|---|---|---|---|---|---|---|
| 11 | 4995.4 | 未达标（超 157.1） | −156.8，CI95 [−470.2, 0.0] | −782.6，CI95 [−1721.2, 0.0] | −9943.9* | +157.1*（更差） |
| 22 | 3923.2 | 达标 | −1228.9*，CI95 [−1988.3, −482.2] | −771.0*，CI95 [−1372.1, −288.4] | −11016.0* | −915.0* |
| 33 | 4851.0 | 未达标（超 12.8） | −301.1，CI95 [−746.6, 0.0] | −770.6，CI95 [−1709.2, 0.0] | −10088.2* | +12.8，CI95 [−433.1, 471.4] |

sampled 模式方向一致且幅度更大（详见 ROUND53_STATISTICS.json；例如 vs 共享 PPO：s11 −1710.8*、s22 −603.0*、s33 −2044.1*，均显著为负）。
`*` = 95% 描述性区间不含 0。差值方向均为“隔离候选 − 对照”。

## 判读

- **开发判据未通过**：三 seed greedy 臂虽然全部 16/16 场景与 immediate 具有相同按时目标 ID 集合，但 seed 11（4995.38）与 seed 33（4851.00）的平均计费高于 4838.25 B/集，仅 seed 22（3923.25）达标。development_go=false。
- 任务上限饱和（全臂 6/6）使 on-time 维度无差异信号；成本维度上，隔离候选相对 BC 与共享 PPO 的点估计在全部 6 个 seed×mode 组合均为负（下降），但相对 BC 的下降仅在 seed 22 达到区间显著；相对共享 PPO 在 seed 22 两种模式与 seed 11/33 的 sampled 显著，greedy 的 seed 11/33 区间触及 0。
- 相对 late 规则：seed 11 greedy 显著更差（+157.1），seed 22 显著更好，seed 33 无差异——跨 seed 不一致。
- critic 隔离是否按合同生效：是（training_version 含 `+r53-critic-representation-stop-gradient/1`，活体绑定与首批门在执行中核验）。该方法相对 BC/共享 PPO 是否建立跨 seed 一致的附加收益：**未建立**。

## 未建立的主张

独立确认、正式非劣、总体零损失、泛化、部署证据均未建立；CI 含 0 不写为等效。本轮仅为已使用开发块上的候选结果。

## 偏差与更正登记（全部无影响科学性的静默热修；登记文件在本目录）

1. `LAUNCH_FAILURE_TASK_STATE_REGISTRY.json`：私有 task-state 播种缺失（0 episode，修复 14b6749）。
2. `LAUNCH_FAILURE_DIRTY_GUARD_REGISTRY.json`：受保护脏文件守卫（0 episode，修复 4ab9b1f）。
3. `LAUNCH_FAILURE_JOURNAL_ROUTE_REGISTRY.json` + `LAUNCH_FAILURE_JOURNAL_ROUTE_REPEAT_REGISTRY.json`：R46 历史 journal 身份校验无法用于 R53 私有路由；登记的修复首次未落实，第二次复现后实现 R53-only claim verifier（两次均 0 episode/0 claim；第二次 2 个 ep0 身份拷贝已保留于 raw_launch_failure_journal_route_repeat_20261008T011100Z）。
4. `LAUNCH_FAILURE_FIRST_BATCH_GATE_VERSION_REGISTRY.json`：首批门禁 training-version 接缝（R45 版本边界合同漏项）。12 个首四集 claim/3 次 PPO 更新为修复前执行器产出但通过修复后门禁审计；恢复仅重跑纯审计门禁并经 R40 resume 续训，未重复任何已提交更新（现场保留 raw_launch_failure_first_batch_gate_version_20261008T014500Z，118 文件 SHA 校验一致）。
5. `LAUNCH_FAILURE_BC_EVAL_INIT_REGISTRY.json`：BC 评估分支漏载 BC checkpoint 状态（R45/R46 既有合同）；修复镜像 R46 model-only preflight，恢复仅补跑 48 格 BC sampled。480 个已完成 claim 未重跑（raw 533.6MB 活动命名空间原位保留）。
6. 最终 result JSON 序列化遇到描述性字段中的函数对象，加透明替换器并记录路径（0 新 episode）。
7. 沙箱限制：冻结 checkpoint 的 `Path.resolve(strict=True)` 在沙箱内对合法可读路径产生虚假 PermissionError，恢复执行在用户批准下以提升权限运行（仅本地读算，无网络）。

## 交付物

- `ROUND53_EXECUTION_RESULT.json`（raw/ 内，status=complete）
- `ROUND53_STATISTICS.json`、`ROUND53_MACHINE_VERDICT.json`、`ROUND53_CHECKPOINT_INDEX.json`
- `ROUND53_GIT_EXTERNAL_MANIFEST.json`（Git 外原件 3875 项，含路径/大小/SHA256）
- 授权回执 `EXECUTION_AUTHORIZATION.json`、prelaunch 冻结 `PRELAUNCH_FREEZE.json`、执行依据 `ROUND53_EXECUTION_BASIS_USER_PROMPT.md`
- 检查点索引覆盖 ep0/ep64(after_b15)/ep128/rolling 与逐批 λ 链（每 seed 32 批）
