# ROUND50：共享主干梯度机制只读分析

## 结论摘要

本轮未能做出实际PPO损失分量的数值梯度归因。ROUND46冻结清单要求的 `canonical_v2_trainer.py` SHA256 为 `0550b99…9a841`，当前唯一可见候选 `runs/comm_cost_v2_rerun_20260928/v2_trainer.py` 的SHA256为 `ffb695…05a6`，不匹配；检查当前HEAD的694个受跟踪Python文件也未找到冻结字节。不能用现有代码重写/猜测损失后冒充冻结训练器。因此不输出actor、entropy、task-value、cost-value分量梯度，不输出总梯度闭合或 `J_logit·(-g_component)`。这是明确的输入缺口，不是数值为零。

Round49的来源身份和策略回放可以复用：BC轨迹、PPO轨迹三seed均通过原动作/概率复现，SEND索引为1、SKIP为0，贪心平局归SEND；真实训练初值身份对应同seed BC_POST，R43旧ep0文件不匹配，不能替代。Round49面板为576条合法真实输入、19维float32。本轮未对模型做CPU forward，因为缺失的正是需要配对的冻结梯度；所有记录仍在索引清单中，但没有伪造敏感性结果。

## ROUND46首批原始记录中可直接确认的内容

以下是原批次记录的指标（非本轮重新训练/更新）。三seed各包含4个episode、265个样本、8个minibatch、8次已记录optimizer step，lambda=1；gate和likelihood对齐均通过。`module_gradient_l1_mean`是当时各minibatch反向后、裁剪前按模块汇总的L1均值，并非损失分量梯度，也不含方向/符号。

| seed | trunk L1均值 | action head L1均值 | task head L1均值 | cost head L1均值 | total grad norm均值 |
|---:|---:|---:|---:|---:|---:|
| 11 | 417.691176 | 7.4956e-05 | 47.0226387 | 3.53954748 | 11.2798 |
| 22 | 465.851917 | 0.001690426 | 164.436663 | 3.09187498 | 26.0257 |
| 33 | 373.722412 | 0.000664713 | 127.520251 | 22.778818 | 18.7925 |

这些值说明首批记录里共享主干和value heads确实参与了反向传播汇总，且action-head L1远小于trunk L1；但由于缺少actor/entropy/task critic/cost critic分别求梯度的原始向量，无法由L1大小确定哪项造成主干变化、方向是否抵消、critic梯度投影到SEND-vs-SKIP logit后为正还是负。也不能把“trunk梯度大”直接解释为critic干扰。

记录显示每seed有8个minibatch/step，但检查到的索引批记录不含实际shuffle后的minibatch成员顺序。故即使恢复源文件，本报告也只计划做完整batch0聚合梯度，不能声称复刻任一当时实际minibatch更新。

## 问题逐项回答

1. **首批actor方向：不可判。** 已有模块总梯度摘要不含actor surrogate独立梯度向量/符号；PPO训练后的策略漂移不能替代首批更新方向。
2. **critic是否经共享主干影响策略：结构上可以，实际本批影响方向/大小未核实。** 纯计算闭式测试用共享标量主干验证：critic对actor专属head的直接导数为0，但可经共享参数使action logit一阶变化非零。这是架构性质，不是实现错误，也非本项目本批数值证据。
3. **是否支持只隔离critic→actor主干梯度的单变量消融：当前证据不足。** 未能算出实际component gradient vectors、夹角/投影、component和总梯度闭合或logit敏感性。不能只基于value-head/trunk L1提出“修复”。不主张训练器已错，也不声称共享梯度是成本/漂移根因。
4. **成本、边界和动作类别：** 当前没有将ROUND49成本标签并入actor输入；没有新的同输入forward，因而也没有合法证据把首批梯度同独有/共同首报成本差逐类对应。Round49的策略映射变化是同输入反事实结果，不是另一策略闭环轨迹。
5. **更广科学结论：** 不影响`development_go=false`，不改变成本CI含0的既有更正，不支持保任务降本、显著性或部署主张。

## 唯一建议的后续有限动作（本轮未执行）

先从项目冻结归档恢复并验证SHA256=`0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841`的训练器源码，并定位批0实际loss/归约实现；若有既存minibatch排列一并按原索引绑定，否则明确仅做完整批。随后仅在三seed首批真实rollout样本及真实BC_POST参数上执行CPU `autograd.grad`，分别分解actor、entropy、task critic、cost critic，核对参数组梯度和闭合，再计算Round49真实19维输入上的logit负梯度方向敏感性。

这项动作要区分的解释是：首批主干更新主要由actor目标驱动，还是由一个/两个critic或entropy经共享主干显著旋转；只有分量方向、投影和总梯度闭合支持后一解释，才值得提出“仅detach critic到共享主干”的方法消融。变量、环境配额、场景数均为0（离线证据恢复与计算）；若冻结源码不能按hash恢复，则停止机制归因，不转而猜实现。

## 操作范围与测试

本轮读取冻结配置/identity、Round49策略面板和路径回放、三个seed按索引指定的first-batch gate与batch记录；未扫描训练日志全集，未加载模型checkpoint，未运行环境/评估/训练入口。纯测试6项通过，测试覆盖见`PURE_TEST_EVIDENCE.json`。

| 本轮副作用计数 | 次数 |
|---|---:|
| environment_construction | 0 |
| simulation_start | 0 |
| new_episode_or_flow | 0 |
| real_training_update | 0 |
| synthetic_optimizer_step | 0 |
| checkpoint_write | 0 |
| evaluation_runner_start | 0 |

`development_go=false`保持不变。
