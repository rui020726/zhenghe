# ROUND75 S0 收口报告：原生 PPO 资格接线（零收费实现轮）

- 分支：`exp/round75-native-ppo-qualification`
- HEAD：`c1a228e1629c3e501ed0def121f0d11ba43d2cdc`
- 运行目录：`runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/`
- 结论：**S0（实现 + 只读冻结准备）完成，全部零收费；S1 付费执行未获授权，停在决策端核验。**

## 一、本轮做了什么

把 R71 科学目标（early 在任务约束下可学）背后的**资格接线**落成可审计代码，但不运行任何神经网络：

1. `r75_frozen.py`：对 r66 真实 `frozen_early_ppo_update` 的忠实全函数移植 `r75_native_ppo_update`。唯一结构改动是冻结集合——R66 冻结 `("trunk","early_head")`，R75 只冻结 `("trunk",)`，可训练/optimizer/裁剪参数对象从 6 个增至 8 个（early_head + late_head + 双 critic，390 scalars；trunk 4 对象 5440 scalars；合计 12 对象 5830 scalars）。损失、门先验（beta=1.00 / ready_cut=147）、shuffle、epoch×minibatch 循环、归约、裁剪算法、optimizer 协议、返回契约逐字保留。函数级哈希：旧 `0a891308…`、新 `07f502fa…`，均 237 行，diff 194 行，只含许可改动。
2. `r75_guard.py`：R75 唯一中央生产账本（纯 stdlib）。生产模式唯一，`simulated()` 直接抛错；接线层源码拒绝 None 或任何非生产对象。预扣语义（先扣再干）、失败不退额、分类上限（physical step 12+4/16；offline trunk 行 5000+1000/6000，单次 ≤1024；环境构造 0/0/0 全禁）、拒绝不消费、重启不清零、只读对账、锁。S0 新增公开 `bootstrap()`：在零起点落盘零值计数器与 `ledger-opened` 标记，消费非零后拒绝再 bootstrap。
3. `r75_replay_adapter.py`：torch-free 的 row+tape→episode 适配器，fail-closed 校验。
4. `test_r75_s0_spy.py`：20/20 通过，全程不 import torch。覆盖真实 JSON 适配器、假对象账本、AST 级接线源码核验、纯算术资源核算。

## 二、资格数据与行为模型身份（三方闭合）

冻结批次：seed 11 / round58_gen2 / e0 / scenario 1300027。row `88271178…`、tape `6d4c4363…`（S0 实地复哈希一致）。66 条记录（≤256）、特征全 19 维、step 1..192、cost_stream 1000 ticks、task_stream 6 目标、lambda=1.0、候选行 168、free early 5 / late 1、gate-eligible early 5。

行为模型：e0 row 的 `controller_parameters_after.parameter_sha256=37303c49…`（5700 参数、单头）＝ R58 gen2 ep0 ＝ `BC_POST_MODEL_s11.pt`（文件 SHA `76ffc86c…`，S0 实地复哈希一致；action_head `aad35fc6…`）。R58→双头转换走既有证据路径 `r66.load_r58_ep0_model` + `r66.fresh_partitioned_controller(ep0_state=…)`，两头均置为原 action_head——路由前向按构造复现单头行为。12 个 substituted_fields 已逐条分类：comm_* mask/timing 字段解析值与冻结配置一致（预算 29300、ACK off、无发送率上限、无等待上限、policy-owned timing），未触发停止条件；残留（archived 原值不可恢复、依赖 row 解析 + config_hash）已登记。

## 三、活体配置绑定

以 R66 冻结件 `living_config_verified` 为准并实地复核：gamma=1.0、**GAE_LAMBDA=1.0**、eta=0.5、delta=0、lambda_init=1、DATA_ORDER_SEED=20260927、B0=10000、lr=3e-4、clip=0.2、entropy=0.01、value_coef=0.5、max_grad_norm=0.5、epochs=4、minibatch=256、log_prob_tolerance=1e-6、gate_beta=1.00、ready_cut=147。**导入默认 GAE_LAMBDA=0.95 / eta=0.05 是旧值，禁用**；`r66_frozen.configure_source_imports()` 在 import 时覆写 `common.GAE_LAMBDA=1.0 / common.ETA_LAMBDA=0.5`。规范模块 trainer/common/controller 字节身份 S0 实地复核一致（`0550b99e…` / `66d019a3…` / `9b2362ee…`）。

## 四、S1 候选门禁设计（未执行、未授权）

对同一冻结批次，保守上界：每次更新 4 个物理 optimizer 步（66 样本 → 4×ceil(66/256)）、840 行 offline trunk 目标行（168×(4+1)）。候选合计 ≤9 物理步（normal 12 / hard 16）、≤1680 行（normal 5000 / hard 6000），最大单次行计费 168（≤1024）：

- A1 非空 Adam 的 critic-only 控制步（≤1 步 / 0 行）：只有双 critic 带梯度走一步，证明 Adam 状态非空、无梯度对象不污染他人、八对象各一次、trunk 从不进 optimizer。
- A2 源实际更新（4 步 / 840 行）：真实路径跑一次 `r75_native_ppo_update`；alignment 在 1e-6 内复现记录 log-probs，失败即停（不改写数据、不禁用对齐）。
- A3 完整恢复同一更新（4 步 / 840 行）：更新前存全量状态（五模块参数、八对象 Adam、lambda、训练 RNG、游标），恢复后重跑同一 update_index=0；记录必须逐位相等，否则停。重跑自恢复前状态起、不推进科学游标，不是第二次已提交更新。

环境构造 0/0/0；joint record 另列无计划。R60 历史 608 次环境构造不搬入本账本、不构成余量。

## 五、零收费计数与账本状态

本轮：模型加载/构造 0、forward 0、backward 0、optimizer 更新（真实/合成）0、PPO 更新/重放 0、环境构造 0、采样 0、checkpoint 写入 0、训练恢复 0、RNG 恢复 0、torch import 0。中央账本 `raw/runtime` 已 bootstrap 为零值身份（4 个零计数器 + 4 条 ledger-opened 标记，无计费事件，锁已释放），consistency_check 一致。R74 历史（物理步下界 ≥16 等）不搬入 R75 账本。

## 六、保护与边界

两份 ROUND46 脏文件 S0 实地复哈希一致、未动未暂存；R59 分析日志保持未跟踪、永不暂存；不推送、不合并、不部署。本报告与机器 verdict 只支持“资格接线已实现且身份可追溯”这一技术结论；任何关于通信策略效果、成本或任务结局的科学结论都必须等 S1 付费执行及其门禁通过之后另行交付。

## 七、交付物

代码：`r75_frozen.py`、`r75_guard.py`、`r75_replay_adapter.py`、`test_r75_s0_spy.py`；文档：`ROUND75_NATIVE_WIRING_DIFF.md`、`ROUND75_NATIVE_WIRING_IDENTITY.json`、`ROUND75_FREEZE.json`、`ROUND75_S0_PLAN.json`、`ROUND75_S0_VERDICT.json`、`ROUND75_S0_REPORT.md`、`ROUND75_GIT_EXTERNAL_MANIFEST.json`。账本原件：`raw/runtime/` 下四个零计数器与 `r75_total_ledger.jsonl`。