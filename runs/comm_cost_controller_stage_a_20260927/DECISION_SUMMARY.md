# 一页决策摘要 —— 第二十轮 Stage A

**结论：Stage A 通过并已提交。** 0 环境启动、0 新流、0 optimizer.step、0 生产代码改动、
未加载真实 checkpoint、未读取封存集。27 项合成测试全绿（unittest 自身计数），
生产三棵树相对本轮起点 `efca2a5` 的 diff 为空，导入审计只带入 torch/numpy 与标准库。
证据：`STAGE_A_TEST_RESULTS.json` + `MANIFEST.json`（在 `runs/comm_cost_controller_stage_a_20260927/`）。

**原 spec 已离线取得并核对**：bundle SHA256 与指令一致，`git bundle verify` 通过，
导入的文档提交 `72c4c3f71e07339bf6533fbd771aef3b2682103c` 与要求的完整 SHA 逐字符相同，
父提交为第十七轮头、只新增两文件；在 `feat/comm-cost-stage-a`（起点 `efca2a5`）上
cherry-pick 为 `c539711`，两份 spec 的工作区 sha256 已登记。
`DESIGN_SPEC_COMPLIANCE.md` §四已**从占位改为逐项差异**（遵守项 20 条、偏差 7 条）。

**推荐方案（已按原 spec 固定符号，全部数学已有测试）**：

```text
c = 已计量尝试字节 / B0        g = E[Y_ref] − E[Y_policy] − δ
L_max = −E[c] − λ·g, λ ≥ 0     λ_next = max(0, λ + η·ĝ)
```

成本进**目标**（cost critic + 组合优势 `A_task − λ·A_cost`），任务效果进**约束**
（配对统计 + bootstrap CI），预算是**硬执行栏杆**而非成本目标，拒付**不算省**。
双 critic 与 λ 已登记进训练状态版本（`constrained-cost-v1` / `constrained-lagrangian/1`），
旧 checkpoint **fail closed**。

**必须一并告诉你的三件修正**：
1. **B/C 账本原本不可闭合**：`STEPS_PER_EPISODE=1000`、`EPISODES_PER_UPDATE=4`、单 seed 400k 步
   ⇒ 3 seed 仅训练就 1 200 次 episode 启动，与旧"C ≤ 488 / 硬上限 520"冲突。
   两个可闭合选项见 `STAGE_BC_RESOURCE_AND_STATISTICS.md`，**推荐小训练量档**
   （3 × 100k 步 = 300 训练 + 360 评估 + 12 smoke = 672 正常，储备 28，硬上限 700）；
   保持 400k/seed 的备选需把上限抬到 1 600。**不靠缩短 horizon 修补账本。**
2. **统计论证已修订**：旧"δ=0.10 + n=120 可判定"是不充分论证（正态近似、无真实配对分布与功效）。
   新口径：逐 seed 配对差 + 配对 bootstrap CI（B=10 000）+ **事前**功效模拟；
   检不出就事先标探索性，不承诺确定性验收；不按中间显著性扩样。
3. **第十九轮 spec 的三个未核对项已补齐**（`SPEC_ITEM_VERIFICATION.json`）：
   "120 000 步" = 120 矩阵格 × 1000 步（与 14 条历史复现、32 项被动性检查是不同账）；
   未四舍五入整数闭合 A 351/9/350/1、B 291/69/290/0，**差出的每个目标**是
   `1200025/piloc-e1-target-5`（A）与 `1200010/piloc-e1-target-3`（B），
   都是 detected 但未完成；35/20/5 由逐 seed 直方图验证通过。

**最小下一步 = Stage B（未启动，需批准）**：≤ 24 次启动、0  optimizer.step，
做真实 driver 上的接线与强规则检查——λ=0 且强制全发时必须**逐字节复现 `immediate`**，
成本归因恒等式在真实运行上成立，并测量 K1′/K2。
Stage C（三 seed 学习）另行批准；δ=0.10、降本门槛 ρ、B0=30 000 B、训练量、
启动上限 700 **全部仍是建议、待批准**。

**本地完整 SHA**：分支 `feat/comm-cost-stage-a`，起点 `efca2a50c8d06f6fa7062c204d94c98b137206a4`，
导入 `c539711002d51335a97925a4653a837288a1c629`，Stage A 提交
`a14853bbe74a121eb92e2105534ae013b086587d`；远端未核验
（`github.com:443` 不通，本会话成功远端接触 0 次）。
