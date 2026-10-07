# ROUND53 后补实现与连接复核

**此前 conditional readiness 已撤回：R53 v2 使用了冻结契约未注册的 hash-seeded RNG 路径。源码已最小修正，但 first-four 动态门仍未执行/验证；因此当前 `ready_for_authorized_execution=false`。552额度仍 `PENDING_USER_APPROVAL`，未生成授权回执或启动运行。** `development_go=false` 不变。首次交付的 READY_CHECK /2 原件已逐字节保存在 `ROUND53_READY_CHECK_V2_AS_DELIVERED.json`；其关于RNG路径“bypassed / fresh hash RNG”的陈述是历史事实记录，但现已确认与冻结契约不符并被本复核撤回。

## RNG合同复核与修复

ROUND46训练清单把唯一科学变化限定为 critic gradient route；它明示 `initializer=ROUND45 BC_POST_MODEL four modules, model-only, same seed`、fresh Adam、dual λ=1、`r44_ep0_rng_only=true`。冻结输入身份/ROUND44配置注册的ep0文件是 s11 `70fe64c18f61b33d17b6aa528cf830b43919a89b22210b11d7edb732bd75f1fd`、s22 `96a09f2f4796069a10f77e5472a4fa8c09ddbf1407e44a7102c022c43c5790c5`、s33 `9030be21c945a490802692569e0fb5b4608602394e43233666ff83adcc2cfe8d`。三者文件SHA均与输入身份和ROUND44冻结登记一致；Python/NumPy/Torch状态分量摘要详见 `ROUND53_FINAL_CONNECTION_REVIEW.json`。

冻结真实次序为：R45 `_call_training_seed` 临时装入同seed BC四模块；R42 `train_seed` 仅读取注册ep0 payload的 `rng`；`_registered_ep0_rng_after_init` 在同seed `fresh_controller`（含BC参数覆盖）返回后调用 `trainer.restore_rng(rng_state)`；之后R40 fresh adapter建立空Adam、λ=1并写本轮ep0状态/开始首集。R42 ep0的模型权重和optimizer没有恢复。ROUND53原先在R46→R45→R42路径外自行SHA-seed Python/NumPy/Torch并直调R40，构成额外随机过程因子；不能证明与注册状态逐字节相同。已删除该支路，现binder要求真实R46 factory-scope→R45 `_call_training_seed`→R42 `train_seed` 源路径和三seed冻结RNG路径/hash精确相符。训练续跑只允许私有ROUND53同版本checkpoint；BC仅模型初始化。

## Claim/预算与基线连接

严格claim检查改为两遍、与目录枚举顺序无关：reserve attempt 2 必须对应同key真实持久化的 normal attempt 1=`failed_technical`，不信任自报 `original_status`；normal必须精确闭合528，reserve上限24而非必须耗尽。新增slot文件与claim逐项身份交叉闭合，strict final门直接执行，不被nonfatal派生快照warning吞掉。纯反例覆盖少用reserve、伪报失败、错slot、坏strict关闭。

BC 48行已核BC_ONLY_R2角色、真实BC_POST checkpoint及参数SHA（包含 `bc_parameter_sha256`）、scene/mode/steps、world/config与行为指纹、成本和冻结通信协议。PPO 96行逐行核NEW、seed/scene/mode、ep128参数SHA；并核三份真实ep128 checkpoint摘要、SHA及训练summary。成本校验涵盖整数spent/on_time、预算、B0、1000步、one-shot/first-report、reward timing、ACK关闭及policy-owned timing。开发参照与训练immediate缓存按索引复用，未做全历史重扫，也未重放forward或评估。

## 当前 `r53_execute.py` 函数清单

`read_json, canonical_sha, plan_from_frozen_sources, verify_baseline_reuse, load_base_runner, bind_runner_paths, install_private_configure_seams, bind_live_runtime, execute_authorized`。关键实际函数分段源码见 `ROUND53_ACTUAL_PATH_EXCERPTS.md`，每段不超过120行。

## 测试范围及未覆盖项

窄测试直接报告 **21项，exit 0，OK**。隔离fixture实接了实际runtime配置器/输出绑定，但把真正train/env/eval/claim/save入口全部拒绝；optimizer守卫覆盖 `Optimizer.step`、`Adam.step`、`SGD.step`，checkpoint写守卫覆盖 `torch.save` 和trainer.save入口；实际观测七项操作计数均为0。实际first-four任务门、任何环境、模型更新、评估和最终真实ledger没有执行；获批后仍须验证这些动态性质。本复核不恢复条件就绪；它不代表科学成功或实验批准。

## 交付

`ROUND53_FINAL_CONNECTION_REVIEW.json`给出风险项逐条状态和RNG字节摘要；`ROUND53_ACTUAL_PATH_EXCERPTS.md`按每段≤120源码行列出未来训练/评估/RNG/init/save/load/budget关键接线。完整代码、纯反例和基线证据保持在本目录。旧 `/2`原件保留，只更新当前 `READY_CHECK.json`为`/3`。
