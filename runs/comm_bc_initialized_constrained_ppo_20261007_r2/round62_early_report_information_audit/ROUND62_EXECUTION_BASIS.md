# ROUND62 执行依据

本文件记录用户在 2026-10-11 提供的 ROUND62 授权与范围，作为本地执行依据；不另立研究 spec。

## 授权与边界

- 仓库 `D:\code\整合\zhenghe-round42`；ROUND61 起点 `e14a7fe7ad24b888e81fcf4485f576b9268661c18`；分析分支 `analysis/round62-early-report-information-audit`。
- 仅进行零环境只读分析、纯计算窄测试、必要的报告与本地提交。不得构造环境、仿真、新 claim、新 episode、optimizer 更新、PPO 重放、训练/RNG 恢复、checkpoint 写入或新评估。不得使用 ROUND61 剩余额度。
- 保留全部历史原件与其他会话文件；不 reset、不覆盖历史输出、不推送、不合并、不部署。
- ROUND60 的 608 次环境构造和超限 8 次是已登记历史偏差，不回改、不退额、不借用额度。

## 核验对象

- 冻结模型：ROUND60 合规 seed33 ep128 greedy 五模块参数，只读；路径、文件与参数身份依正式 R61 冻结件核验。
- ROUND61 固定 8 案例及来源；L/L0 门禁；每个 FORCE_SEND/FORCE_SKIP 的前缀、覆盖机会、任务集合、成本、发送/投递/分配链；slot、builder、环境、runner、步数与资格分别对账。
- 输入分层：A 为动作前 19 维 actor 输入；B 为动作时发送端合法字段；C 为接收端、分配器与结局事后字段。不得把 B/C 伪装成 A 或用于案例选择。
- 可只读扫描 R60 seed33 greedy 的 32 个已存场景首报 tape，不产生新增结果标签。
- 读取 R29/R30 的既有报告与机器结果，避免重复推荐相同的 deadline 编码训练。

## 结论纪律

结论限于固定模型、8 个事后案例、每场景一次随机实现的 SEND/SKIP 局部闭环差异。L/L0 仅为门禁，不算独立样本；accepted 不等于任务收益；同结局不等于安全删除。相近向量不证明不可分，无精确碰撞不证明 19 维充分。不得作总体必要性、安全跳过、非劣、泛化或部署主张，不做小样本显著性包装。

保护 `ROUND46_CURRENT_TASK.json` 与 `ROUND46_R2_V6_STOP_STATE.json`，预期 SHA256 分别为 `cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b`、`93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18`；不得修改或暂存。

## 要求的交付

追加 `ROUND62_REPORT.md`、`ROUND62_VERDICT.json`、`ROUND61_EVIDENCE_RECHECK.json`、`DECISION_INFORMATION_PANEL.json`、纯计算分析代码与反例窄测试。训练及仿真副作用必须为 0；明确未知值，不用 0 填补缺失。提交只包含显式点名的代码、测试、报告与机器结果，不含权重、tape 或原始日志。
