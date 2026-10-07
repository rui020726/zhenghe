# ROUND49 BC 输入与策略映射闭合报告

任务：`ROUND49-BC-INPUT-POLICY-CLOSURE`。仅对已存 BC/PPO tape 与现有模型做 CPU、eval、no_grad 前向。开发块 `1301000..1301015` 已多轮使用。

## 1. 覆盖与回放门禁

- BC：索引解引用 48 行、48 seed-scene；纳入 288 个自由首报输入。PPO：ROUND48 同面板 288 条。

- 排除项：BC tape 内 5607 个非入选目标记录，类别 {"fixed_hold_one_shot_closed": 5607}；PPO tape 5602 个非自由/关闭记录。

- BC 自身输入由同 seed 真实训练初值复现；PPO 输入由同 seed ep128 复现。概率容差 1e-06，动作映射 SEND=1/SKIP=0，平局归 SEND。总体回放状态：True。

- 每条记录的来源 tape/row、hash、特征、动作、概率与 mask/候选合法性在 `INPUT_PANEL_MANIFEST.json`。相同向量仅缓存 forward，目标记录没有去重。

- ROUND48 概率探针发现源角色接线错误：其身份构建记录了正确 BC_POST 初值，但前向函数实际装载 `REGISTERED_EP0` 指向的 R43 旧模型。ROUND48 ep0 概率/翻转表因此不对应真实训练初值；ep128 的 tape 概率回放仍有效。本报告用对应 seed 的 R45 BC_POST 初值重算，并在 `STATEMENT_CORRECTIONS.md` 留下更正。

- ROUND48曾报告的 s11/s22/s33 ep0均值约 `.542/.375/.504` 以及相关边界翻转，不用于ROUND46初值解释；真实 BC_POST 初值在各自面板的重算均值约 `.343/.345/.345`。受影响统计字段逐项列在勘误。

- 实际加载证据：旧文件的 forward 参数摘要、真实初值摘要及本轮六个 forward 文件/参数 SHA256 均逐模型列在下一节；旧 R43 文件在新的加载门禁处因参数摘要不匹配被拒。

## 2. BC 轨迹与 PPO 轨迹输入上的概率

每个来源面板、每个 seed 均列 ep0 与 ep128；这是同一批已存输入上的策略映射，不是对另一策略闭环轨迹的预测。

| 输入来源 | seed | ep0 p均值[范围] | ep128 p均值[范围] | Δp均值 | 距0.5平均绝对距离 ep0→ep128 | greedy发送数 ep0→ep128 | 同seed转换 SS/SK/KS/KK |
|---|---:|---:|---:|---:|---:|---:|---:|
| BC_trajectory | 11 | 0.343448 [0.000000, 1.000000] | 0.463338 [0.034769, 1.000000] | +0.119890 | 0.488133→0.412336 | 33→37 | 33 / 0 / 4 / 59 |
| BC_trajectory | 22 | 0.344603 [0.000000, 1.000000] | 0.304183 [0.000004, 1.000000] | -0.040419 | 0.487544→0.476104 | 33→30 | 30 / 3 / 0 / 63 |
| BC_trajectory | 33 | 0.344653 [0.000000, 1.000000] | 0.463674 [0.029545, 1.000000] | +0.119021 | 0.488157→0.408935 | 33→36 | 33 / 0 / 3 / 60 |
| PPO_trajectory | 11 | 0.343439 [0.000000, 1.000000] | 0.463334 [0.034769, 1.000000] | +0.119895 | 0.488060→0.412331 | 33→37 | 33 / 0 / 4 / 59 |
| PPO_trajectory | 22 | 0.344603 [0.000000, 1.000000] | 0.304183 [0.000004, 1.000000] | -0.040419 | 0.487544→0.476104 | 33→30 | 30 / 3 / 0 / 63 |
| PPO_trajectory | 33 | 0.342356 [0.000000, 1.000000] | 0.463582 [0.029545, 1.000000] | +0.121225 | 0.488164→0.408843 | 33→36 | 33 / 0 / 3 / 60 |

## 3. 控制路径、概率回放和特征覆盖

| 模型角色 | seed | 实际 forward 文件 SHA256 | 实际 forward 参数 SHA256 |
|---|---:|---|---|
| ep0 | 11 | `76ffc86c4aca26fb3b27380d5195a0cbd07822f7b85d1f1070a5550776dd458a` | `37303c494cd997894fba1441cf9d2cde86cf2135e71f443a4ed1bcb9ec7f0330` |
| ep128 | 11 | `0f161663aaf963552f8f02d6363f3c52e6f67119217f835c501cb2d0705bfc10` | `8b87f4d2591a3a0c573b71a08ec8444db1e3f91850edbf2f2b958dddb31e5670` |
| ep0 | 22 | `3cdb83785c51f9e40005d66a1d024d06a5ba6940616e22f3b01a9bf65b5e609d` | `e8531e0aff95ed207bc70026247d45869656d266aa974838a7ec2cfe352479b0` |
| ep128 | 22 | `aae4aa1b24b7e7ffadffeb7a09d45935f44970ac62a8cfd4344630248472cc62` | `b88a6b377d9e18c3cef17ab5205f7652b1394fcfbc6b28e5642140025028caf4` |
| ep0 | 33 | `9329ef048dfc5c1805af5756c7ea0962eedfc8d01aa26841e35af7001616803a` | `10bae22f3896adc507f05c5d868aad43b0441d19f754f2ef4dc373fc964e57a5` |
| ep128 | 33 | `10f3fc8d2c2a788b7e48e1f9021b5b234c39a00706bfe486e6f2bab84a0fb9b9` | `dc2903397865c853c59ed631e7b0a699fcb2d55b7dee8956101c53984b36b821` |

ROUND48 的错误加载文件及其真实 forward 参数 SHA256：

| seed | 错误加载路径 | 实际参数 SHA256 | 正确 BC_POST 初值参数 SHA256 |
|---:|---|---|---|
| 11 | `D:\code\整合\zhenghe-round42\runs\comm_one_shot_gate_resume_20261003_v2\checkpoints\v2_train_s11_ep0.pt` | `7eff0f40203fdae63437bb07a16204ab0289bf938e9c263f781b3adfc40d5eff` | `37303c494cd997894fba1441cf9d2cde86cf2135e71f443a4ed1bcb9ec7f0330` |
| 22 | `D:\code\整合\zhenghe-round42\runs\comm_one_shot_gate_resume_20261003_v2\checkpoints\v2_train_s22_ep0.pt` | `b74c10c85c148a4b46b4513a6b04660446c31a778a6537a8f13e22ffc78e8095` | `e8531e0aff95ed207bc70026247d45869656d266aa974838a7ec2cfe352479b0` |
| 33 | `D:\code\整合\zhenghe-round42\runs\comm_one_shot_gate_resume_20261003_v2\checkpoints\v2_train_s33_ep0.pt` | `0ddebe7dc04019ebb079e25b9b4191b654e59167b928c416112393fbc6d0d3aa` | `10bae22f3896adc507f05c5d868aad43b0441d19f754f2ef4dc373fc964e57a5` |

- BC 索引控制臂标记 `BC_ONLY_R2`，其 checkpoint parameter identity 与 ROUND46 同 seed `BC_POST_MODEL` 初始化一致；BC tape 记录 one-shot schema、free choice、顺序预算 mask 审计及 SEND 概率。BC 的动作按原 tape 复现。身份相同本身不足以证明路径相同；本轮另核验 tape 资格、mask 顺序、候选次序与概率/动作重放。详见 `BC_PATH_REPLAY.json`。

- 每 seed/来源的向量计数（相同输入记录仍逐条保留）：

| 来源 | seed | 输入记录 | 唯一向量 | 重复向量组 | 重复组内记录 | 跨 scene-target 共享组 |
|---|---:|---:|---:|---:|---:|---:|
| BC_trajectory | 11 | 96 | 69 | 13 | 40 | 13 |
| BC_trajectory | 22 | 96 | 69 | 13 | 40 | 13 |
| BC_trajectory | 33 | 96 | 69 | 13 | 40 | 13 |
| PPO_trajectory | 11 | 96 | 69 | 13 | 40 | 13 |
| PPO_trajectory | 22 | 96 | 69 | 13 | 40 | 13 |
| PPO_trajectory | 33 | 96 | 69 | 13 | 40 | 13 |
| 两来源合并 | 11 | 192 | 75 | 63 | 180 | 13 |
| 两来源合并 | 22 | 192 | 72 | 66 | 186 | 13 |
| 两来源合并 | 33 | 192 | 74 | 64 | 182 | 13 |
| 两臂全seed合并 | — | 576 | 78 | 74 | 572 | 13 |

- 两来源向量支持集：{"BC_unique_vectors": 69, "PPO_unique_vectors": 76, "intersection_unique_vectors": 67, "BC_only_unique_vectors": 2, "PPO_only_unique_vectors": 9}。这是输入支持覆盖，不表示独立样本或行为因果。

- ROUND48 历史范围复核：{"per_seed_records": 96, "pooled_records": 288, "pooled_distinct_feature_vectors": 76, "pooled_vector_hash_groups_repeated": 69, "meaning_note": "96 is the per-seed panel size (288 pooled). 76 is the number of unique feature hashes across the pooled PPO records. 69 counts those hashes occurring more than once, not independent scenes or necessarily distinct targets."}。

- 本 pooled 面板中恒定列为 0–12、16、17；变化列为 13 `has_future_requirements`、14 `queue_backlog_targets`、15 `budget_remaining_fraction`、18 `step_over_horizon`。列范围和端点占比见机读 `feature_observations`。恒定列不能分开这些记录；变化列与 p/动作的关联不能确定漂移由哪一维导致。

## 4. 成本类别交叉观察

ROUND47 vs BC 的 PPO-only / BC-only / both / neither 首报标签是按同 seed、scene、target 联结；标签只作事后分析元数据，不进入 actor。各组 n、ep0 与 ep128 SEND 数和 mean Δp见下表；它描述输入上的概率映射变化，不代表真实发送/任务结果。

| 来源 | seed | PPO-only发送 n / Δp | BC-only发送 n / Δp | 双方发送 n / Δp | 双方均不发 n / Δp |
|---|---:|---:|---:|---:|---:|
| BC_trajectory | 11 | 4 / +0.746 | — | 33 / +0.018 | 59 / +0.135 |
| BC_trajectory | 22 | — | 3 / -0.639 | 30 / -0.048 | 63 / -0.008 |
| BC_trajectory | 33 | 3 / +0.763 | — | 33 / +0.015 | 60 / +0.144 |
| PPO_trajectory | 11 | 4 / +0.745 | — | 33 / +0.018 | 59 / +0.135 |
| PPO_trajectory | 22 | — | 3 / -0.639 | 30 / -0.048 | 63 / -0.008 |
| PPO_trajectory | 33 | 3 / +0.798 | — | 33 / +0.019 | 60 / +0.144 |

这些输入差异不能拆分参数效应与闭环轨迹变化；同一输入的 ep0→ep128 才是固定输入上的映射差。
## 5. 结论与方法依据

- BC 控制与真实训练初值：**可对应**，同 seed 参数身份和源模型概率/动作重放闭合；因此两条来源面板的同输入 ep0→ep128 对比有效。

- BC 自身输入是否可分：可分。正确 BC 初值在每 seed BC 面板上的概率均值约 0.344，概率覆盖接近 0 到 1，greedy 动作 33 SEND / 63 SKIP；这反驳“全局固定近0.5”的描述，但只说明已观测开发输入支持上存在区分，不说明可泛化。

- Round47 配对 CI 未变：s11 `[0,1406.625]`、s22 `[-928.434375,0]`、s33 `[0,1249.5]` B/集，端点精确含0；均不支持严格排除零的 seed 成本差。

- 同一输入上的 ep0 与 ep128 概率/动作差异是模型参数映射差异；BC与PPO轨迹样本分布、动作数和成本标签不同，还可能由闭环轨迹选择变化造成。

- 正确初值到 ep128 的可复现边界变化在 BC 与 PPO 两种来源面板上均很少：s11/s33 为少数 SKIP→SEND，s22 为少数 SEND→SKIP。ROUND48报告的几十次边界翻转是错误旧初值导致，不能用来解释seed成本差异。转换顺序为 ep0→ep128，`SS/SK/KS/KK`分别指 SEND→SEND、SEND→SKIP、SKIP→SEND、SKIP→SKIP。

- 证据不足以支持改训练器、改特征、换阈值或删除首报。按时任务结果与成本的既有结论不升级，`development_go=false`。

## 6. 唯一有限下一步建议

若继续研究，唯一建议是另行登记一次未用场景块上的固定策略配对确认：假设当前 BC→PPO 成本差只由开发轨迹输入分布造成；唯一变量为策略 checkpoint（真实 BC 初值 vs ep128），同场景同配置配对，其他编码、greedy 解码与任务/成本判据固定。可证伪结果是新块仍出现可重复的同方向按时目标集合与计费成本差；若差异消失或跨 seed 方向不稳，则开发块的映射变化不足以支持稳定方法收益。该确认未授权、未执行。

## 7. 执行计数与限制

| 项目 | 本轮新增 |
|---|---:|
| 环境构造 | 0 |
| 仿真启动 | 0 |
| 新流 | 0 |
| 真实训练更新 | 0 |
| 合成 optimizer.step | 0 |
| checkpoint 写入 | 0 |

仅运行本轮纯计算测试；未运行环境/训练入口或全量环境套件。完整交付 HEAD 在提交后记录于最终交接。
