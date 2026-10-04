# ROUND44 停止报告：seed11 首批比较接缝异常

## 状态

ROUND44 **未完成**。首个正式运行在 seed11 的四集首批和一个 PPO batch 后，执行器因 `NameError: name 'run_module' is not defined` 退出。异常发生在 `_compare_first_batch` 读取 R43 post-update optimizer 状态时，首批 R43/R44 比较回执没有生成。按冻结依据中“真实运行失败需保留 claim 并停止受影响工作、不得热修冻结代码”的约束，本次没有重放四集或继续后续 seed。

这不是策略效果差或科学结论；目前没有三 seed 结果，也没有可用的 ROUND44 任务保持/强规则判定。

## 已发生的执行

| 项目 | 实际值 |
|---|---:|
| seed11 完成 episode | 4 / 128 |
| 完成场景 | 1300027, 1300014, 1300028, 1300044 |
| 环境构造 / run 调用 | 4 / 4 |
| 完成环境步 | 4,000 |
| PPO batch | 1 |
| optimizer step | 12 |
| checkpoint 写入调用 | 4 |
| 正常额度已用 | 4 / 480 |
| 技术储备已用 | 0 / 24 |
| 固定评估完成 | 0 / 96 |

四集合计为 **22 个按时任务、37,098 载荷尝试字节**。batch0 记录为 631 行，`g_hat=+0.25`，`lambda_dual` 从 1.0000 更新到 1.1250。seed11 四集本地门禁记录为 `first_four_gate_ready`，after-b0 checkpoint 已提交；但与 R43 首批逐字段比较的执行器接缝失败，所以不把 seed11 认定为“首批全部门禁通过”。

当前 after-b0 checkpoint：`checkpoints/v2_train_s11_after_b0.pt`，SHA256 `42a5160ead497ef3ce5dcb2e9783fe094b30f6a1f245630149bdde2c75946d25`；恢复状态为 episode 4、update 1、optimizer step 12、lambda 1.125、eta 0.5，含 Python/NumPy/Torch RNG。该精确状态作为证据保留；本报告不据此授权继续或重放。

## 身份修订和门禁

- 原 R44 冻结 HEAD：`597d02aed42b7b332c604c61630e461233c74868`。其冻结 JSON 的字节副本保存在 `runtime/ROUND44_FROZEN_CONFIG_597d02a.json`，SHA256 `354b2d4441ec1d4ce8d4aa6ba6f79b29d4cba17c82d525f2a566a68c4e5190ed`。首次预启动失败原因和输出摘要保存在 `runtime/ROUND44_PRESERVED_PRELAUNCH_FAILURE_597d02a.json`。
- 后续提交 `2353c4616746cf0a7c9fcb3a519a7042a28ba176` 只把 R43 已登记的 report-only finalizer 身份与科研执行身份分列；R43 其余 13 个执行源码哈希保持一致。finalizer blob、`ROUND43_STATUS`、最终报告哈希均与 R43 交付身份匹配。
- 用实际校验入口 `verify_r43_source_identity` 执行临时夹具回归：准确登记的最终 report hash 通过；未登记报告字节、把任意报告哈希写入本地标签、以及 `r42_run.py` 源码漂移均被拒绝。0 环境、0 run、0 optimizer step、0 checkpoint 写入。详情：`runtime/ROUND44_SOURCE_IDENTITY_REGRESSION.json`。
- 此后 R44 才生成匹配 HEAD `2353c46` 的预启动身份收据。首批异常发生在该身份下，不存在后续热改。

## 失败定位与对账

异常调用栈显示 `r44_execute.py:_compare_first_batch` 引用了未定义的 `run_module` 名称。四个 episode claim 都是 `complete`，各 1,000 步；账本无技术失败 claim，但总执行因 post-update comparison exception 停止。原始 ledger、progress、行、tape、环境日志、claim、checkpoint 与批提交均保持在本地，不覆盖。

运行 progress 仍保留原值 `running`，没有篡改成 complete 或 stopped；本文件及 `ROUND44_STOP_STATE.json` 是独立的停止事实记录。当前机器在退出后未检测到 Python 实验进程。原始文件路径、大小与 SHA256 列在 `ROUND44_STOP_ARTIFACT_INVENTORY.json`。

## 额度和科学判读

4 个正常 claim 已消耗，不退款；技术储备仍为 24，硬上限仍为 504。本轮未启动 seed22、seed33，未执行 96 格固定评估，未完成 paired analysis。因此不报告本轮性能胜负、seed 稳定性或研究目标达成。

继续需要单独处理执行器故障并建立新的机械执行身份，同时遵循精确恢复纪律，不能从 episode 0 或 batch0 重新训练以覆盖已提交更新。当前停止状态提交只包含报告和清单，不含模型、tape、环境日志或原始 episode 行。
