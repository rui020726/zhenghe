# ROUND45-R2 执行停止报告：BC-only receipt 接缝

## 停止原因

新 runner 已通过 zero-environment seam/container 回归并完成新冻结。第一格
`r45_bc_control_s11_scene1301000` 在一次 1000 步真实运行返回后，进入 R44 的
evaluation receipt writer。R44 receipt 索引只为 `model_arm="NEW"` 的固定 ep128 矩阵行生成
receipt；本格身份是 `model_arm="BC_ONLY_R2"`，因此抛出：

```text
EvalIndexError: artifact receipt may only be created for a NEW evaluation row
```

这是运行后身份接线错误。它不说明 BC 策略的科学门槛通过或失败。该 row 与原始文件完整保留，
但两种 receipt 都未写入，claim 状态为 `failed_technical`、`completed_logical_jobs=0`，
所以没有把这格纳入 BC control gate。

## 实际计数

- 本轮 BC：3000 次真实更新；含旧失败累计 4000。没有重跑 BC。
- 正常 claim：1/528；技术 reserve：0/24；总 claim：1/552。
- 环境构造：1；run 调用：1；run 返回：1；完成步数：1000。
- 物理 row 中可读到 6 个按时任务、7284 B，但因 receipt 缺失且 claim 未完成，不作为有效控制结果。
- PPO 更新、Adam step、checkpoint 写入：均为 0。
- 有效控制格：0；其余 47 个控制格、384 个训练 episode、96 个终评格均未启动。
- 未消耗 reserve，没有重试。已完成 1000 步的 key 不满足冻结 retry guard；不重写原 claim 为 complete。

行、lifecycle、transcript、tape 与 claim 的 SHA256/大小见
`ROUND45_R2_POST_LAUNCH_STOP_V1.json`。BC 权重和 optimizer 原件仍留在本机，Git 跟踪已在此前
本地提交中移除；没有推送。

## 后续授权边界

完整执行依据允许一次最小版本化机械修复；该次修复已用于 StageCController 四模块 eval 接缝。
本次暴露的是第二个、不同的 R44/R2 receipt 接缝。依执行依据停止，不追加第二次修复、不热改冻结
runner，也不重新运行该已完成 key。需要决策者另行决定是否授权：版本化地让 BC-only 行使用 R2 自身
receipt 路径，同时定义如何严谨处置这个已运行但未注册完成的 key。
