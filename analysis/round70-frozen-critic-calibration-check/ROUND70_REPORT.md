# ROUND70 报告 — 冻结表示上的 critic 校准检验（只读诊断，零环境）

- 任务：`ROUND70-FROZEN-CRITIC-CALIBRATION-CHECK`
- 分支：`analysis/round70-frozen-critic-calibration-check`（建于已核验 HEAD `4f66ae783bd6e49afdba777210a0cef28c8ebe04`，即 R69 收口头）
- 复用轮次：ROUND68（既有 384 tape/row、96 batch、已核验 per-seed 模型身份）；R69（字段时点/单位映射与闭合：GAE、奖励归集、路由、lambda_before）
- 机器结果：`analysis/round70-frozen-critic-calibration-check/ROUND70_FROZEN_CRITIC_CALIBRATION_CHECK.json`
- 分析脚本：`r70_frozen_critic_calibration_check.py`；窄纯算术测试：`test_r70_pure_arithmetic.py`（9/9 通过）
- 执行依据（提示词原文存档）：`ROUND70_EXECUTION_BASIS.md`

**只检验预测能力。** 本轮不证明 critic 错误、不证明 baseline 能提高策略、不证明消息的因果价值、不认定 clip 根因。

## 1. 接手核验

HEAD 核验为 `4f66ae7`（R69 更正pass收口头，`analysis/round69-late-actor-signal-audit` 尖端）。工作树 `zhenghe-round42` 即该 HEAD 且含 R68 运行原件；无 ROUND70 同目标任务在运行（未见 round70 分支/文件）。在其上新建独立 analysis 分支，未覆盖任何既有分支，未推送/合并/部署。R68/R69 原件、旧失败、消费记录全部只读未动。

## 2. 数据来源勘误（追加登记，保留 R69 原文）

R69 报告正文（第 6 节及更正块）把被审计块统称“开发块 1800000..1800031”。实测：本审计训练统计来自**真实训练场景 1300000..1300047**（batch `scenarios` 与 row `scenario_seed` 均为 13xxxx，48 个唯一场景，384 episode，32046 决策步行）。1800000..1800031 是 R68 的**评估块**（见 ROUND68_EVAL_PLAN；R69 仅在 `context_inputs/.../r68_eval_primary...` 作背景引用）。**更正：R69 的“180块”称谓是来源误标；本轮训练拟合全部只用 1300000..1300047，180 评估结果仅作背景，不加入任何拟合或选模型。R69 原文一字未改。**

## 3. 方法

- **真正 critic 输入**：tape 的 per-decision-step、per-target 19-D `features`（采集器原样入 trunk）。决策步 critic 值 = `head(mean over targets of trunk(features))`；因 head 仿射、mean 线性，决策步值恰为该 mean-embedding 的仿射函数。
- **只读重建**：从已验证 `state_dict` 只取 5 模块重建 per-seed 冻结 trunk + 双 critic 头（CPU/eval/no_grad）；未调用训练 resume/load_checkpoint，不恢复 optimizer/λ/RNG。
- **仿射头判定**：`value_task_head`、`value_cost_head` 均为 `Linear(64,1)`，即 64-D 冻结表示上的**仿射头**；双输出仿射基线与真实 critic 同类可比，**适用**（非“头非仿射→不适用停止”分支），未改模型类硬套。
- **trunk 逐位冻结资格**：每 seed 一条 trunk（requires_grad=False），before_bN/after_bN 逐位相同；seed 11/22/33 三条互异。32,046 行 embedding 只提取一次，多折复用。
- **接线核验**：before_b0/b15/b31（seed 11）新鲜重算 value/cost 与保存 rollout 前值最大差 ~2e-7（float32 精度），全部通过；未用 ep128 critic 值顶替。
- **CV（拟合前锁定）**：按训练 scene 排序，独立 `PCG64(20261013)` 一次置乱、`array_split` 5 折；三 seed 共享同一 scene→折映射，同 scene 所有重复 episode/行同折。固定 `ridge=1e-3`；训练折内中心/尺度归一化（仅用于统计拟合），常量列尺度设 1、截距不惩罚。每折拟合**双输出仿射 ridge** + **训练折均值截距-only 对照**；每 seed 5+5，共 15+15=30 个拟合，全部仅内存，不保存拟合权重、不写 checkpoint。事后诊断，不称独立确认；未扫折数/正则/特征，未选最好 seed。

## 4. 结果（held-out / 仅 OOF 进入主表；残差 = 回报 − 预测）

**晚期自由行（R69 的 1.786 vs 0.262 那批，n=513，任务回报）**

| 预测器 | bias | MAE | RMSE | n |
| --- | ---: | ---: | ---: | ---: |
| 原 rollout critic | +1.523 | 1.566 | 1.829 | 513 |
| OOF 仿射基线 | +0.233 | 0.817 | 1.012 | 513 |
| OOF 截距基线 | -0.836 | 1.067 | 1.248 | 513 |

**晚期自由行，λ_before·任务 − 正成本 组合残差**

| 预测器 | bias | MAE | RMSE | n |
| --- | ---: | ---: | ---: | ---: |
| 原 rollout critic | +1.633 | 1.707 | 2.010 | 513 |
| OOF 仿射基线 | +0.244 | 0.898 | 1.120 | 513 |
| OOF 截距基线 | -1.271 | 1.418 | 1.663 | 513 |

**全部行（ALL_ROWS，n=32046，任务回报）**

| 预测器 | bias | MAE | RMSE | n |
| --- | ---: | ---: | ---: | ---: |
| 原 rollout critic | +2.253 | 2.500 | 3.230 | 32046 |
| OOF 仿射基线 | -0.425 | 1.012 | 2.071 | 32046 |
| OOF 截距基线 | -0.116 | 1.998 | 2.271 | 32046 |

**每 seed 晚期自由行任务 RMSE（稳健性：仿射 < 截距 < critic）**

| seed | 原 critic | 截距基线 | OOF 仿射 |
| --- | ---: | ---: | ---: |
| 11 | 1.908 | 1.170 | 0.988 |
| 22 | 1.950 | 1.394 | 0.986 |
| 33 | 1.610 | 1.165 | 1.060 |

（成本侧同样：晚期行 cost 残差 bias 由 critic +0.308 → 仿射 +0.056；全部行三者在 cost 上接近，因成本尺度小。）

## 5. 结论

- **同一冻结表示能否更好预测这些保存回报？能。** 在**同一** frozen trunk embedding 上，OOF 双输出仿射基线在 held-out 行上把任务 RMSE 从原 critic 的 1.83 降到 1.01（晚期行，约减半）、从 3.23 降到 2.07（全部行），并显著压低 bias；晚期行上三 seed 一致。
- **constant（截距）贡献**：仅取训练折均值就已消去原 critic 的巨大系统性欠预测偏差（全部行 task bias +2.25、晚期行 +1.52 → 近 0/负）。即原 critic 的主要失准是“整体电平标定”，一个常数即可纠正大半。
- **表示（embedding）贡献**：在常数之上，64-D 表示仍带来真实增益——晚期行 task RMSE 截距 1.25 → 仿射 1.01，且修正了截距在晚期行的过冲 bias（−0.84 → +0.23）；组合残差 bias 由 +1.63 → +0.24。表示增益在晚期行三 seed 稳健；在全部行上较小且跨 seed 不均（seed 22 全部行仿射略逊于截距）。
- 因此：**这些残差不是冻结表示无法预测的；同一表示上存在更校准的预测基线**，其校准增益可分解为“常数电平纠正（主）+ 表示残差结构（次、晚期行更明显）”。

## 6. 仍不能推论

- Does not prove the critic is wrong, only whether a better affine predictor exists on this representation.
- Does not show a baseline would improve the policy.
- Does not establish the causal value of a message.
- Does not identify the root cause of any clipping.
- 另有：仿射基线是就回报拟合的，其更低误差部分来自“为这些回报优化”，而 critic 是冻结的、另训的映射；此为预测能力对照，不是同训练预算的公平算法竞赛。

## 7. 资源消费（如实分列）

- 神经前向（权威口径=决策步行）：**33158 / 上限 36000**（未超限）；目标级 trunk 评测 60800（mean-over-targets 内部展开，另行披露）；前向调用 1171 次；每批 ≤ 1024。
- 只读模型加载：9（per-seed before_b0/b15/b31 共 9 个 state_dict）。
- 统计参数拟合：ridge 15 + 截距 15 = 30（纯内存）。
- 环境/episode/claim/真实或合成 optimizer.step/PPO 重放/autograd/训练 RNG 恢复/checkpoint 写入：全部 0。
- 拟合权重、原模型、tape、log 均不写入 Git。

**本轮 HEAD**：提交于已核验 `4f66ae7` 之上的新 analysis 分支；commit SHA 以提交信息与最终答复为准。
