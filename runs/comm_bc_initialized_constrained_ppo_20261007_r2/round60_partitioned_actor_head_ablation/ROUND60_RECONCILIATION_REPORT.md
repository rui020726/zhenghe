# ROUND60 最终有效性与累计消费收口（零环境核验）

本次为零环境核验：环境构造、仿真、新 claim、真实/合成 optimizer 更新、PPO 重放、RNG 恢复、checkpoint 写入、新评估全部为 **0**；仅只读解析、纯计算模型状态核验。HEAD 核验 `3a8bae8ef35b1fbc99847c4b8f347f02d615cdac`，工作区干净，无 python 进程。所有旧原件、报告、verdict 未改写；本文件与 `ROUND60_RECONCILIATION_VERDICT.json` 为追加。

## 1. 全部实际消费及 600 上限核算

按原始执行事件与来源逐项重建（非合规尝试是真实消耗，不因科学样本被排除而记 0；备份/确定性重复不去重）：

| 类别 | 环境构造 | 完成 episode | PPO 更新 | optimizer step | checkpoint |
|---|---:|---:|---:|---:|---|
| 历史非计划 probe | 1 | 1 | 0 | 0 | 0 |
| 非合规真实训练（隔离） | 31 | 30 | 7 | 44 | 7 after_b + 8 before_b + rolling + 7 batch record + ep0 |
| 最终合规训练 | 384 | 384 | 96 | 556 | ep128×3 + after_b×96 + before_b×96 + rolling×3 + batch record×96 + ep0×3 |
| 最终评估 | 192 | — | 0 | 0 | — |
| 技术失败（reserve 重试） | 0 | — | 0 | 0 | — |
| 纯合成测试 | 0 | — | — | 合成（见下） | 合成（测试目录） |

- 非合规训练定位：seed 11，quarantine `_quarantine_attempt1_criticdetect_missing`，batch 0–6 提交（episode 0–27）+ batch 7 部分（episode 28–30，episode 30 环境已构造但 0 步即中断）；执行版本为修复 critic→trunk 隔离之前的 `train_seed → partitioned_ppo_update`（绕过 detach）。stop 因发现该缺陷；修复后登记新身份、重新验收、从 ep0 全新重跑（合规 raw）。
- **真实环境构造合计 = 1 + 31 + 384 + 192 = 608**；**真实 optimizer step 合计 = 44 + 556 = 600**。
- **硬上限 600 → 实际 608，超限 8**。超限量全部来自非合规重跑的 31 次环境构造（计划/有效消耗为 576 合规 + 1 probe = 577，本在限额内）。不把非合规训练改判为 reserve 来掩盖超限；不追加批准。

## 2. 三 seed 最终训练资格（全部 verified）

三个 ep128 均来自完整合规链（只读核验，param_sha 前缀）：

- seed 11：ep0 身份 `fb82ad80`（= R58 ep0 trunk/双 critic + early=late=action_head）；b0 params_before=`fb82ad80`（全新起点）；32 批（episode 0–127）；b31 params_after == ep128 `809f35f7`；opt=184、λ=1.0。
- seed 22：同构；b31 == ep128 `286e2f84`；opt=180、λ=1.0。
- seed 33：同构；b31 == ep128 `bce71a0e`；opt=192、λ=1.125。
- **未继承非合规权重/Adam/λ**：合规 b0 params_after `998d57db` ≠ 非合规 b0 `a95b6010` ≠ 非合规 b6 `ff70d4ae`；合规 raw 从空命名空间全新构造，未复用非合规 tape。
- **critic→trunk 隔离确已启用**（非仅 gate 名称）：执行版本调用路径为 `train_seed → trainer.ppo_update(routed_ppo) → with critic_input_detached → partitioned_ppo_update`（代码行 432/210/214）；且合规 b0 与非合规 b0 在同一 ep0、同一 episode、同一 action RNG 下 params_after 不同，差异只在共享 trunk 梯度，与 detach 生效一致。
- **192 格评估绑定合规链**：三个 seed 的 eval 行 `checkpoint_load_identity` param_sha 分别 == 各自合规 ep128（`809f35f7/286e2f84/bce71a0e`），load_mode=read_only，optimizer/rng 未恢复。

结论：三 seed 均 **verified**，无 unverified；可据合规链保留描述性结构消融结果。

## 3. 原 577/556 等数字是否需要更正

- **576（有效合规结果）= 正确**：384 训练 + 192 评估，全部 complete，来自合规链。
- **556（optimizer step）= 合规链正确**；但**真实 optimizer step 合计为 600**（556 合规 + 44 非合规）。
- **577（累计含 probe）= 需更正为 608**；**headroom 23 = 需更正为超限 8**。原累计少计 31（漏计非合规重跑的环境构造）。

## 4. 可保留与必须暂缓的结论

可保留（来自 verified 合规链，限于该开发块 1700000..1700031 的描述性结果）：NEW−R58 / NEW−immediate / NEW−late 的任务与费用对照；seed 11 与 R58 在评估输出上一致、seed 22 保任务且费用降（CI 含 0）、seed 33 greedy 丢失 scene 1700027/`piloc-e1-target-5`、sampled 保任务；early 关键目标随 seed/模式而异（vs R58 仅 seed 33 greedy 丢失；vs immediate seed 11/33 greedy 丢失；vs late seed 22/33 sampled 救回）。bootstrap 索引与 R58 冻结索引一致。

必须暂缓/不宣称（原报告亦未宣称）：跨 seed 一致的费用下降（所有费用 CI 含 0、seed 11 持平，故不成立）；参数共享干扰为唯一根因；泛化；正式非劣；部署。

措辞限定：“s11 逐比特相同”仅指**评估输出**（逐场景 on_time、spent_bytes、emission 序列），**不扩大**到训练参数或更新轨迹（ep128 参数 `809f35f7` 与 R58 ep128 不同，更新轨迹亦不同）。分头前向/loss 初值等价不表示梯度范数、全局裁剪、Adam 轨迹相同——该更正保留。

治理披露：实际累计 608 超硬上限 600 共 8，均源于非合规重跑；这是簿记/配额超限的如实记录，不改变合规链的科学有效性，也不追加批准或改分类。

## 5. 完整 HEAD 与本次零副作用计数

HEAD `3a8bae8ef35b1fbc99847c4b8f347f02d615cdac`（本次收口在此基础上追加报告/verdict 提交）。本次零环境核验副作用：环境构造 0、仿真 0、新 claim 0、真实 optimizer 更新 0、合成 optimizer 更新 0、PPO 重放 0、RNG 恢复 0、checkpoint 写入 0、新评估 0。两份 ROUND46 脏文件 SHA256 维持 `cf3ad198…`、`93a2f69f…`，未修改、未暂存。

合成更新附注（单独计数，不占 600 硬上限）：上一会话验收报告 1 次合成 optimizer 更新；init-equivalence 正确性检查每次运行对合成 episode 跑 3 次 partitioned_ppo_update（12 optimizer step/次），未入 journal，跨多次重跑的确切累计**未知**（下界 ≥ 报告 1 + 至少一次 equivalence 的 12）；本轮 0。合成 checkpoint 写入仅发生在验收的独立测试目录。

本地显式点名提交本报告、verdict 与 `_reconcile.py`；不提交权重、checkpoint、tape、原始仿真日志；不推送、不合并、不部署。完成后停止，不开启下一实验。
