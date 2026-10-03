# ROUND42-R 固定计划续执行复核

日期：2026-10-03
分支：`fix/comm-one-shot-runner-counting`
复核时 HEAD：`2825069eaf9acf0fff87a9a02e69ae2f8e88c785`

## 状态

固定计划**未完成**。本次按续执行依据完成零启动只读核验，没有重新运行环境、恢复训练或调用优化器。现有执行在 seed 11 首批 gate 处以失败状态停止；本报告保留该状态，没有把静态路径核对改写成 gate 通过。

## 身份与代码核验

- 继续使用 ROUND42-R 分支，工作区干净；没有重置或覆盖文件。
- 预启动 HEAD 为 `f46fe4703aa03ce6400f63c601c3203ef741dd9d`，当前 HEAD 为 `2825069eaf9acf0fff87a9a02e69ae2f8e88c785`。预启动 HEAD 是当前 HEAD 的祖先。
- 两个 HEAD 之间只增加了已停止运行的账本、停止状态、库存和诊断报告；`r42_run.py`、`r42_analysis.py`、`r42_execute.py`、`r42_one_shot.py`、`r42_versions.py` 等执行代码与预启动版本逐字节一致。
- 重新按冻结配置中的真实 worktree 和 dependency 路径核验了 84 项代码/依赖 SHA256，全部匹配；canonical 输入、ROUND40、ROUND41 的 HEAD 也与冻结身份相符。ROUND42-R carryover、pure-compute gate 和 runner 回归证据的冻结哈希均匹配。
- 原冻结身份仍记录 `counted-jointrun-proxy/1`；训练语义仍为 `one-shot-send-skip/1`，训练版本仍为 `constrained-cost-v2/one-shot-send-skip/1/published-completion-event-at-relay-tick/2`。冻结配置哈希为 `8054593ee60b6a7a78bd18f72f6316fe90ab346b3550f4393b1b81d9ae555fa2`。
- 未发现 `python`/`pythonw` 进程；本次只读检查了项目文件与 checkpoint 元数据。

续执行依据原文保存在 [CONTINUATION_BASIS_FIXED_PLAN_20261003.md](CONTINUATION_BASIS_FIXED_PLAN_20261003.md)，SHA256：`ee3f66902e53489acc2a8bf38fbda0fbc498a34b4bc68306f9fa8a50834d3095`。

## 已发生的运行与额度

当前账本记载：

| 项目 | 累计 |
|---|---:|
| normal claims（含 ROUND42 原失败） | 68 |
| reserve claims | 1 |
| claims / builder 调用 / JointRun 构造 | 69 / 69 / 69 |
| `run()` 调用 / 底层方法调用 / 正常返回 | 68 / 68 / 68 |
| 完成仿真步数 | 68,000 |
| 已完成逻辑任务 | 68 |
| PPO 更新 / Adam steps | 1 / 12 |
| checkpoint 写入调用 | 4 |
| 新增技术失败 | 0 |
| 剩余 reserve / 总硬上限 | 27 / 700 |

64 个 immediate 参考 episode 已完成并通过原参考重试门禁，其中 scene 1300000 的 reserve attempt 2 完成 1000 步。seed 11 的 episode 0–3 已实际运行并提交一次 PPO 更新，但首批 gate 没有通过，因此这四集和更新不作为已验证训练结果。seed 11 后续 episode、seed 22/33、192 个模型评估格、32 个规则评估格均未执行；还剩 604 个计划逻辑任务。

原 ROUND42 的失败 claim 仍为失败：1 次 normal claim、1 次构造、0 次 `run()`、0 步。它没有被改写为成功，也没有退还额度。

## 首批 gate 的失败原因与复核

已保存的停止异常是：

```text
RuntimeError: seed 11: frozen PILOC/CTDE checkpoint path set changed
```

异常发生在首批 gate 最后的冻结 checkpoint 路径集合比较。训练批记录使用 Windows 反斜杠路径；gate 以 `Path.as_posix()` 生成正斜杠路径。只在内存中将记录键的 `\\` 规范化为 `/` 后，路径集合与两个冻结配置路径相同。逐文件重算的结果也一致：

| checkpoint | 批记录 SHA256 | 冻结配置 SHA256 | 当前文件 SHA256 |
|---|---|---|---|
| PILOC `runs/cur_v2rep05_s2/best/shared_mappo_model_21_3100.pth` | `d57aed25b3d46dffb2bd269ea87140953828ba7f51a960bb01995ec054dfca33` | 同左 | 同左 |
| CTDE `runs/joint_eval_20260916/ctde_final_7_1.pth` | `c7c256da0fc20451c1104996ce494148f567e4f83acf265d9baae8f0f053500d` | 同左 | 同左 |

批记录的原文件 SHA256 为 `85f2192e18838506a198198e4f88fa090168c4aeaeb754f42e268b89c22b10b0`。traceback 表明该函数此前的首报自由动作、likelihood、奖励时序、GAE、对偶更新、梯度、参数链及计数断言均已通过；由于路径集合断言抛错，原函数没有继续执行后续逐文件路径值断言。上表是本次独立只读重算的证据，**不是新的 gate 通过记录**。原 `ROUND42` gate 失败状态保持不变。

seed 11 的 `v2_train_s11_after_b0.pt` 与 rolling checkpoint `v2_train_s11.pt` 字节完全相同，SHA256 均为 `a85722e0864456ea3eb735195c7ffcde4be9d42301bdf2bc0781db6747a851ef`。只读反序列化确认其训练版本、冻结身份相符，状态包含 controller、optimizer、Python/NumPy/Torch RNG；游标为 episode 4、update 1、optimizer steps 12、lambda 1.0125。这说明已保存完整恢复状态，但本次未加载该状态继续训练，也未重复这次更新。

## 停止依据

现有冻结执行器明确拒绝已存在运行产物或 progress 文件的隐式续跑；冻结源检查还要求当前 HEAD 与 `ROUND42_PRELAUNCH_IDENTITY.json` 中的预启动 HEAD 完全相同。当前代码虽未变化，当前 HEAD 因停止交付提交而不同。直接重启原入口会触发其拒绝条件；忽略 gate、改写旧 prelaunch 记录、人工置为通过或重复执行 seed 11 的 PPO 更新都会绕过原冻结/恢复契约。

因此本次按提示词保留原 gate、checkpoint、stop state 和账本并停止。若决策者要求继续，需先形成独立、版本化的恢复执行身份与只读 gate 复核，并证明能够从上述完整状态精确恢复；不得重放已经提交的 PPO 更新或重跑已完成场景。本地报告提交不推送远端。

## 本次检查边界

- 新环境启动：0
- 新流：0
- 真实训练更新：0
- 合成 optimizer 更新：0
- checkpoint 写回：0
- 旧失败原件或停止账本改写：0
