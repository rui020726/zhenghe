# ROUND58：现场核验与隔离适配修复（非实验完成报告）

## 状态

本轮在隔离目录完成了只读现场核验、最小适配器诊断修复及16项窄测试；**没有启动新执行代次、替代四集、训练或评估**。这不是ROUND58固定计划完成交付。

修复提交包括：

- 5fd6e0fa… — 计数与被动诊断修复；
- d2bd3b7… — 运行时对象绑定和旧状态恢复fail-closed；
- da87c377… — 双计数修复；
- 97ffdc71… — R53 critic隔离接线修复。

完整HEAD见最终回复。旧停止报告保留为当时快照；本报告补充并更正对更新总数及首四集输入资格的解释，不改历史原件。

## 现场条件与停止

历史材料不满足“只有原seed11四次消费、seed22/33尚未开始”的Path C前提：gen1有seed11 e0–e3四条complete claim；gen2又有三个seed各e0–e3的12条complete claim。gen2的12条claim均记normal，而代次登记称从前代reserve消耗4次；账本仍报normal=4、reserve=0、optimizer=0。分类无法闭合，不能把矛盾改写为reserve已确认。

因此本轮不创建新claim，也不使用reserve。当前有效R58训练游标仍为零；gen1、gen2的complete claim均作为历史消费保留，不计入有效R58训练集。

## 计数及科学接线

真实活体配置读取确认：265个样本、minibatch=256、epochs=4，真实PPO循环应是每epoch两个minibatch，即8个optimizer调用。旧适配器在canonical trainer已递增后又追加一次minibatches计数，导致报告16、记录索引1,3,…,15，且按双计数分母计算暴露统计。修复只移除重复计数；没有改变PPO epoch、minibatch、loss或更新次数。

更严重地，旧R58源码补丁编译后直接替换ppo_update，没有保留R53 wrapper在critic representation上的detach context。对真实controller的纯梯度探针确认：旧可调用路径中task/cost critic可回传到shared trunk；R53要求二者不回传到trunk但仍训练各自critic head，actor仍训练trunk与action head。隔离适配器现将整个patched PPO调用置于同一R53 context。历史更新因此不能作为R53隔离条件下的有效R58更新；这不是beta梯度计算错误的证据。

旧诊断接缝还会在缺少hashlib时触发NameError，并可能让诊断或记录IO错误阻断调用方的核心提交。新隔离adapter补入hashlib，并将诊断/被动写入异常作为报告错误单独返回；核心PPO错误仍原样传播。该adapter未接入历史R58 launcher，因此不能据此声称当前固定计划入口已修好。

## 首四集数据资格（只读分类）

四条gen1 row/tape文件的claim receipt SHA均匹配。四个row报告同一controller参数SHA，且与ROUND45 BC初始化回执中的参数SHA相同；每个tape均保存action、log probability、features和budget-sequential-mask/1，row保存每集action RNG seed、lambda=1及reward timing版本。

这说明critic隔离失效不会自动追溯性改变更新前已落盘的rollout字节；原四集可作为有完整性证据的候选数据保存。它们未被本轮送入batch_samples、GAE或PPO，也没有重建权重。

但它们尚不能认证为可复现/可复用的训练批：row列出12项替代默认字段，尚未与冻结环境契约逐值闭合；完整Python/NumPy/Torch RNG状态及原minibatch排列缺失；canonical batch_samples/GAE对象没有独立哈希封存。本轮因此不判“数据已无效”，也不判“已足够重建”。机器分类见INPUT_REUSE_QUALIFICATION.json。

## 修复代码边界

r58_gate_prior_repaired.py是隔离副本；r58_repaired_entry.py仅做活体module identity/config核验、对象绑定、model-only读取与拒绝旧训练恢复。测试证明它绑定到ppo_update实际globals中的common对象及R40 trainer对象，并拒绝缺少repair attestation的旧R58 checkpoint。

历史round58_gen2/r58_execute.py仍导入旧适配器。未修改其冻结身份，也未以新冻结身份替代。故现在没有可运行且已门禁的固定计划执行器。

本轮故障注入覆盖的是适配器诊断抛错/记录器写错时的返回路径，以及核心PPO异常仍传播；没有穿过真实批checkpoint与成功回执的原子事务接缝。因此，核心checkpoint提交/回执的端到端事务安全仍未被本轮证明，不能将测试结果外推到训练事务已安全。

## 测试与零操作

16项窄测试全部通过，退出码0。真实torch.optim.Optimizer.step为0；测试中的104个SpyOptimizer回调仅捕获梯度，不更新参数。环境构造/运行、checkpoint写入、训练tape重放均为0。详见NARROW_TEST_RESULTS.json。

## 结论

1. ROUND58历史运行链不具备接受为有效R58科学训练链的证据：关键critic隔离接线失效，且claims/ledger/journal/batch记录冲突。
2. 计数修复表明冻结契约对应8步；不能回写历史16成8，也不能把历史报告修成8步。
3. 原四集rollout字节有候选复用价值，但当前证据不足以认证训练批等价性。
4. 隔离adapter修复和测试不等于已修复并冻结生产执行器，也不等于满足重新启动条件。
5. 固定计划尚未完成：没有有效训练集、640评估格或统计/verdict；无新环境、真实更新或checkpoint写入。

后续若要重启，需先形成独立、审查通过的执行身份和真实launcher接线，并解决当前代次claim/额度分类及rollout可复用性证据缺口。当前授权仅用于本轮零环境修复与资格分类；本轮停止。
