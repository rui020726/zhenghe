# Stage C 判定：首轮有界成本学习 pilot（不部署、非正式非劣验收）

日期：2026-09-28。分支 `feat/comm-cost-stage-c`，起点 `6ebefbe4e7a81182281807333d64d60cadf03198`
（已核验 Stage B 头）。执行依据 `ROUND22_COMM_COST_LEARNING_PILOT.md`
（`docs/comm-cost-learning-pilot@f73ebdf8cc3ab60a6a866fdbe0e299b122245cf7`，git fetch 取得、
字节级核验、cherry-pick 为 `39a3da5`）。协议与冻结件：`STAGE_C_PROTOCOL.md` +
`CONFIG_AND_MODEL_MANIFEST.json` + `SCENE_REGISTRY.json` + `START_LEDGER.json`。

## 正式结论：**pilot_completed**（计划跑完；非技术失败；非"学习优于规则"）

672 次计划启动全部完成，无未解决门禁失败，无追加训练/调参/扩样。
**本轮不得据此声称**：学习收益、成本学习有效、任务非劣、胜过规则、δ=0 约束成立、
排除欠训练。16 个开发场景只支持探索性 pilot；C 之后的确认实验/部署另行批准。

## 启动账本（`START_LEDGER.json`）

| 用途 | 计划 | 实际 |
|---|---:|---:|
| 3 通信训练 seed × 128 集 | 384 | 384 |
| immediate reference（48 训练 + 16 开发场景） | 64 | 64（另 2 次启动前技术失败 + 1 次重复 reference） |
| 学习模型开发评估（3 seed × 3 时点 × 16 场景） | 144 | 145（1 次评估崩溃后重跑） |
| 五条规则开发评估（5 × 16 场景） | 80 | 80 |
| **合计** | **672** | **677 / 700 硬上限** |

技术失败 5 次（全部留痕、不扣回）：2 次配置继承 bug 拦截 + 2 次后台启动被掐断 +
1 次评估 summary 崩溃；每次的修复都作为新代码身份再冻结，训练始终在单一冻结身份
（`ffca51cc2a054f01`）下完成。首 4 集正确性门禁（计入 384）**G1–G7 全部通过**
（`FIRST4_GATE_s11.json`）：optimizer 只含通信控制器参数（形状/计数核验）、冻结 checkpoint
文件哈希不变、log_prob 用登记初值重算最大差 2.38e-07（≤1e-6）、reward/GAE/组合优势/g_hat
从落盘件独立重算闭合、λ 更新公式与方向正确（batch 0：g_hat=+0.25 ⇒ λ 1.0→1.0125）、
双 critic 预测/loss 有限、全 mask 无 actor 伪样本、checkpoint 纯计算进程恢复前向一致。

## 训练过程（`TRAINING_CURVES.json`）

三个 seed 各 128 集（每 seed ≤128 000 环境步）、32 批更新、464 optimizer.step/seed；
λ 从 1.0 单调升至 ≈1.49（每批 g_hat ∈ [+0.25, +0.5]，约束始终被违背 ⇒ 对偶变量持续上调，
方向符合公式）。采样 rollout 全程 ≈28.2k B/ep（预算的 96%）、≈11 次发射、≈5.6 个按时任务，
四个 32 集窗口内无可见趋势。贪心评估（冻结权重与 λ）：

| seed | 初值(0 集) | 64 集 | 128 集（最终） |
|---|---|---|---|
| 11 | 5.50 按时 / 28 198.9 B | 同左 | 同左 |
| 22 | 5.75 / 4 975.5 B | 5.50 / **0 B** | 5.50 / **0 B** |
| 33 | 5.50 / 28 189.5 B | 5.8125 / 28 148.3 B | 5.8125 / 28 148.3 B |

（开发场景 immediate 参考：16/16 场景 6 按时、均值 14 939.2 B/ep。）

## 最终比较（`COMPARISON.json`，固定 128 集 checkpoint 对 immediate，16 开发场景逐场景配对）

| 臂 | 按时任务差/ep | 字节差/ep | 改善/打平/退化（场景） |
|---|---:|---:|---|
| 学习 seed 11 | −0.5000 | +13 259.6 | 0 / 0 / 16 |
| 学习 seed 22 | −0.5000 | −14 939.2 | 11 / 0 / 5 |
| 学习 seed 33 | −0.1875 | +13 209.0 | 0 / 0 / 16 |
| always_hold | −0.5000 | −14 939.2 | 11 / 0 / 5 |
| **late_discovery_gate** | **0.0000** | **−10 101.0** | **16 / 0 / 0** |
| periodic_20 | 0.0000 | −356.2 | 8 / 0 / 8 |
| event | −0.4375 | +13 123.1 | 0 / 0 / 16 |
| random_send | −0.2500 | +13 105.5 | 0 / 0 / 16 |

读法（spec §8 纪律）：**优于 immediate ≠ 优于强规则**。本轮最强的对照是
`late_discovery_gate`（仅晚发现首报、阈值 147 固定不搜索）：在全部 16 个开发场景上任务数与
immediate 相同、每集少 10 101.0 B。三个学习 seed 均未接近它：seed 22 贪心策略退化为
"几乎全 HOLD"（0 B/ep，任务 −0.5，与 always_hold 同档），seed 11/33 花费仍高于 immediate
且任务更差。**没有 seed 表现出一致的成本-任务优势，不选最佳 seed，不池化三个 seed。**

## 诊断与方法限制

- λ 是诊断量：三 seed 均升至 ≈1.49 仍在 0 边界以上，说明约束在整个 128 集内未被满足；
  不得据此宣称"约束机制无效"或"已满足"，也不得在运行中放宽 δ。
- 采样（p≈0.5/候选）与贪心行为差异大（seed 22 贪心全 HOLD、采样却花 96% 预算），
  训练信号与部署发射器之间的差距是本轮可见的方法限制。
- 128k 步未收敛不能宣称排除欠训练；不得因此自动续训。
- 16 开发场景只支持探索性结论；零观测损失（late_discovery_gate）**不构成**总体非劣/等效证明；
  无已批准 ρ，候选价值判断留待后续（含确认实验）另行批准。
- 费用口径是当前计费模型的载荷尝试成本（B0=10 000 归一化），不是无线空口或全系统成本。

## 交付件与产物位置

机读件（已提交）：`FIRST4_GATE_s11.json`、`COMPARISON.json`、`TRAINING_CURVES.json`、
`CHECKPOINT_INDEX.json`、`START_LEDGER.json`、`CONFIG_AND_MODEL_MANIFEST.json`、
`SCENE_REGISTRY.json`、`GATE_RECHECK.json`（比较件沿用名）。
原始日志/tape/权重不入 Git：`logs/`、`rows/`、`checkpoints/`（checkpoint 路径与
SHA-256 见 `CHECKPOINT_INDEX.json`，含 freeze 身份 `ffca51cc2a054f01`）。
生产三棵树相对分支起点 `6ebefbe` **零 diff**（只训练通信控制器；搜索/分配权重来自两个
冻结 checkpoint，逐批核对文件哈希）。

**本地 SHA**：分支 `feat/comm-cost-stage-c`（链见 `git log`）。**远端**：本轮授权普通推送
Stage A/A1/B/C 分支；不 force；分叉则停。网络失败有限重试，不影响本地交付。
