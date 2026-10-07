# ROUND49 更正与口径补充

## ROUND48 ep0 源模型接线更正

ROUND48 的 `r48_probe.py` 在 `build_model_identity()` 中正确记录 ROUND46 初值为 R45 `BC_POST_MODEL_s{seed}.pt`，并把 `REGISTERED_EP0` 路径标成 R43 旧件、明确说明它不是 ROUND46 初值。但 `forward_panel()` 随后实际用 `REGISTERED_EP0[seed]["path"]` 加载前向。因此 ROUND48 保存的 ep0 概率、边界翻转、动作过渡和依赖这些值的解释对应的是 R43 旧 ep0 模型，不是本轮训练实际初值；这些概率探针结论由本轮同 seed R45 BC_POST 参数重算替代。ROUND48 ep128 源模型复现检查仍成立。旧报告和结果文件保留，不覆写。

受影响项：ROUND48 每 seed ep0 概率均值/范围、ep0→ep128 平均漂移、0.5边界穿越和四类 greedy 转换，以及按成本类别汇总的Δp解释全部以旧模型为基线，现由 ROUND49 的正确同 seed 初值输出取代。ROUND48 tape选择、96条/seed的自由输入纳入、特征统计、ep128前向概率与原 tape复现、ROUND47成本结果及冻结CI未因此失效。

## 口径补充

- ROUND47 的 seed-wise PPO−BC 配对区间端点精确包含 0；ROUND48 的统计更正仍有效。不得称任一 seed 差异已由该 CI 排除零。
- 未舍入冻结端点：s11 `[0, 1406.625]` B/集；s22 `[-928.434375, 0]` B/集；s33 `[0, 1249.5]` B/集。保持原共享场景 bootstrap，不新增单侧检验。
- Greedy 是确定性 argmax；p(SEND) 接近 0.5 不表示实际随机掷硬币。
- ROUND48 的 96/76/69 分别是**每 seed 记录数**、**合并 288 条 PPO 轨迹的唯一向量数**、**合并面板中出现多次的不同向量哈希组数**。ROUND48 原字段名“shared_by_multiple_targets”实际上只检查同一向量哈希是否出现多次，没有判别是否为不同目标；它不是独立目标数或场景数。
- 按实际动作/成本类别观察的 p 变化只描述策略映射与轨迹标签的共同出现，不证明 PPO 更新机制、成本差根因或通信因果。同一输入 ep0→ep128 的变化比较的是策略映射；BC/PPO 各自轨迹输入分布还包含闭环路径改变。
- 本轮未开展新的统计检验；开发块仍是已用开发块，`development_go=false`。

## 源参数摘要对照

ROUND48错误加载位于 `r48_probe.py:283-287`；错误源登记在 `:45-55`，而身份构建另记真实初值与旧文件摘要在 `:134-171`。

| seed | ROUND48加载的R43文件 | R43文件SHA256 | R43实际参数SHA256 | BC_POST初值参数SHA256 | 参数匹配 |
| 11 | `D:\code\整合\zhenghe-round42\runs\comm_one_shot_gate_resume_20261003_v2\checkpoints\v2_train_s11_ep0.pt` | `70fe64c18f61b33d17b6aa528cf830b43919a89b22210b11d7edb732bd75f1fd` | `7eff0f40203fdae63437bb07a16204ab0289bf938e9c263f781b3adfc40d5eff` | `37303c494cd997894fba1441cf9d2cde86cf2135e71f443a4ed1bcb9ec7f0330` | False |
| 22 | `D:\code\整合\zhenghe-round42\runs\comm_one_shot_gate_resume_20261003_v2\checkpoints\v2_train_s22_ep0.pt` | `96a09f2f4796069a10f77e5472a4fa8c09ddbf1407e44a7102c022c43c5790c5` | `b74c10c85c148a4b46b4513a6b04660446c31a778a6537a8f13e22ffc78e8095` | `e8531e0aff95ed207bc70026247d45869656d266aa974838a7ec2cfe352479b0` | False |
| 33 | `D:\code\整合\zhenghe-round42\runs\comm_one_shot_gate_resume_20261003_v2\checkpoints\v2_train_s33_ep0.pt` | `9030be21c945a490802692569e0fb5b4608602394e43233666ff83adcc2cfe8d` | `0ddebe7dc04019ebb079e25b9b4191b654e59167b928c416112393fbc6d0d3aa` | `10bae22f3896adc507f05c5d868aad43b0441d19f754f2ef4dc373fc964e57a5` | False |

R43文件路径与参数摘要位于 `BC_PATH_REPLAY.json`。ROUND48静态 REGISTERED_EP0 参数值与真实初值相同，但其被指向文件反算出的参数摘要不同；这正是角色标签没有保护住真实加载的原因。
