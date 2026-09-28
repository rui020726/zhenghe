# ROUND25 §1 只读整理完成报告：发布阻断，未启动下一轮

日期：2026-09-28。分支 `exp/comm-cost-greedy-replication`（起点 = 已核验 ROUND24 执行头
`f920c55`，仅导入本 spec 单文档提交 `e2822f1`）。

## §1.1 起点核验（git 只读）

| 项 | 实际 |
|---|---|
| ROUND24 HEAD | `f920c555b7bfe7d01296b9099635f673c2e5dcb6`（verdict 报告的短 SHA `f920c55`，一致） |
| HEAD tree | `2543ab8314013d501c03441b2d9ec8d8beb88802` |
| 提交链（4） | `5be3f1e`(spec) → `70d0dc1`(冻结) → `e318345`(执行完毕) → `f920c55`(推送状态) |
| 工作区 | tracked 零 diff；其他会话未跟踪件（`architecture/`、旧 rows 等）原样保留 |

## §1.2-§1.3 ROUND24 交付机械复核（`ROUND24_SOURCE_RECHECK.json`，全部从落盘件重算）

- **672 行 rows 齐全**：train 384 / eval 144 / reference 64 / rule 80；账本 live 672，
  **技术失败 0**；成本账本、任务 tap、ACK-off 三项闭合违例均为 0。
- **checkpoint 身份**：seed 11/22/33 各有 ep0/64/128 三个里程碑；ep0 与新鲜
  `Controller(种子)` **逐参数字节相等**（确实从登记初值训练）；ep64=16 批、ep128=32 批，
  λ 终值 1.5000 / 1.2375 / 1.4750；五件套 schema（trainer/loss/state/gae/optimizer/reward
  timing）齐全。批记录中 **v1 checkpoint 读入 0 次**。
- **seed11 参数链补偿证据覆盖每一批**：32/32 批记录连续；参数链 31/31 环一致；
  cost_head 权重 32/32 批均变化且每批预裁剪梯度 L1 > 0；likelihood 每批 ≤ 4.8e-07；
  seed22/33 同样通过。即不是"一条警告已解释"，而是逐批复核。
- 交付件哈希清单已登记（verdict/manifest/index/ledger/gate/artefact manifest 共 23 项）。

## §1.4 发布：阻断

两条已授权发布路径本会话均不可用：

1. **普通 git push**：`git push -u origin exp/comm-cost-v2-rerun` 2 次尝试均
   `Failed to connect to github.com port 443`（spec 允许的重试次数已用尽）。
2. **已授权 GitHub 接口**：本会话中该 MCP 服务器已下线，不可用。

未声称任何推送成功；远端 `exp/comm-cost-v2-rerun` 状态本会话无法核验。按 spec §1.4
停止规则：**只完成本节只读整理，报告发布阻断，不启动下一轮**（§2 诊断及其后的 192 次
评估均未开始，0 新环境启动）。

## §1.5-§1.6 分支与原件核对

- 新分支 `exp/comm-cost-v2-rerun` 之外另建 `exp/comm-cost-greedy-replication`，仅含
  spec 单文档提交；未重复导入修复链，未回退历史。
- spec 来源登记：用户转交全文；声称来源 `docs/comm-cost-greedy-replication@2fd8d9a9…`，
   本会话**未能**按固定提交逐字节复核（所有远端通道不可用），已在 `START_IDENTITY.json`
   标注 `remote_verification: NOT_PERFORMED`；三条弱一致信号（f920c55 与报告一致、
   86cff022 与 ROUND24 字节核验过的计划提交一致、起草时远端无该执行分支）已登记，
   不构成字节级核验。
- §1.6 的"三个 checkpoint 文件 SHA256/内部 seed/128 标签/schema/解码来源 + 搜索分配权重
  锁定"依赖 §2 的诊断与冻结件，属下一节工作，未开始。

## 恢复条件（任一满足并经用户确认后继续）

1. 任一已授权发布通道恢复 → 先推送 ROUND24（4 提交）与本分支，字节核验 spec blob，
   登记树等价证据，然后从 §2 继续；
2. 用户明确授权跳过/延后发布 → 从 §2 继续，发布阻断单独挂账。
