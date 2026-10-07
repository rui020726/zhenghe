# ROUND49 执行依据

任务标识：`ROUND49-BC-INPUT-POLICY-CLOSURE`
起点：`analysis/round48-policy-drift-probe`，完整 HEAD `b85a9f61921b176ba77c7909a9e25da687fcf7b0`。
目标分支：`analysis/round49-bc-input-policy-closure`。

## 目标与范围

补齐 ROUND48 未纳入的 48 条 BC 控制 tape（3 seed × 16 个已使用开发场景），核实控制臂模型/19 维编码/动作解码/候选顺序和顺序预算 mask；把 BC 自身轨迹与 ROUND48 的 PPO greedy 输入并列，用各 seed 实际训练初值和 ep128 六个策略做 CPU、eval、no_grad 同输入前向比较。只分析已保存记录，不生成新场景或轨迹。

项目任务是多机器人搜救通信控制，PILOC 搜索与 CTDE 任务分配冻结。当前动作是一目标一次首报 SEND/SKIP；SKIP 不会等待后补发。greedy 由真实动作索引的 argmax 解码，平局归 SEND。开发块 `1301000..1301015` 已用于多轮选择，不能作为独立确认块。现有指标：immediate 6.0 个按时任务/集、14939.25 B/集；late_discovery_gate 6.0、4838.25 B/集；BC 三 seed 6.0、5152.125 B/集；PPO ep128 s11/s22/s33 分别 5778/4694.25/5621.625 B/集，均逐场景保持 immediate 的按时目标 ID 集合。`development_go=false` 保持。

ROUND48 已确认 ROUND46 ep0 actor/共享主干来源为同 seed 的 R45 `BC_POST_MODEL_s{seed}.pt`；ROUND42/R43 时代旧 ep0 不属于本轮初值。ROUND48 的主要面板来自 PPO 最终 greedy tape。本轮不重复 ROUND48，不覆盖历史件。

## 输入与身份

- ROUND47/48 的输入身份未变时复用，不重扫仓库历史账本。
- BC 48 行以 `ROUND46_R2_V6_INPUT_IDENTITY.json` 的 `round45_control_rows` 精确解引用；每行的 tape 由该行 `artefacts.tape` 解引用，禁止宽泛 glob。
- PPO 面板沿 ROUND48 的最终矩阵、V7/V8/V9 历史接纳来源和 tape 索引逻辑解引用。
- 读取 ROUND48 报告、机读结果、模型身份、输入 manifest 与 ROUND47 成本分解及统计更正。
- 核对文件与参数身份；同名、其他轮或其他 seed 的模型不能代用。

## 硬边界

环境构造 0；仿真启动 0；新流 0；真实训练更新 0；合成 `optimizer.step` 0；checkpoint 写入 0。
允许读取代码、索引、JSON/JSONL/tape/checkpoint；CPU/eval/no_grad 只读 forward；添加隔离分析脚本、纯计算测试、报告和本地提交。

禁止恢复 optimizer/RNG；修改模型、编码、阈值、解码器或默认值；拟合替代策略；生成场景；运行环境构造套件；安装依赖；联网、推送、合并、部署。保留旧报告、原件和失败记录。受保护脏文件及其他会话文件不修改、不暂存。

## 分析纪律

1. 纳入所有保存且合法、动作前、首报资格成立、按原顺序 mask 后仍可执行 SEND 与 SKIP 的记录；只依据来源、schema、语义和 mask 资格筛选，不依成本、结果或动作筛选。特征缺失时仅可从已保存合法动作前字段按冻结真实编码恢复；不从后续状态/结局/接收端信息重建，不填零。
2. 每条记录保留场景、目标、来源 seed/arm、tick、候选身份、完整 19 维输入和 hash、原动作/概率（若有）、合法性证据。不同目标的相同向量记录均保留。
3. 先验证 BC tape 在其来源 seed 初值上的概率与 greedy 动作可复现；PPO tape 在来源 seed ep128 上概率与动作可复现。概率容差固定 `1e-6`。核对 dtype/维度顺序、SEND/SKIP 索引与 mask 顺序。关键身份或回放失败的来源不得混入通过结论，并如实定位。
4. 同 seed ep0→ep128 是主要配对；跨 seed 仅辅助。概率分布、动作切换和输入群组均为描述性统计，不增加显著性检验。
5. ROUND47 的 CI 下端/上端精确包含零的更正继续有效。greedy 是确定性的；p 接近 0.5 不表示随机掷硬币。按实际最终动作分组看到的概率变化不证明 PPO 更新机制或成本因果。不同实际轨迹的差异不能全归因参数；同输入反事实不代表另一策略的真实闭环轨迹。
6. 按每 seed 与 pooled 范围分别闭合记录数、唯一向量数、重复向量组、重复组记录数、跨目标共享及跨 seed 重复。ROUND48 的 96、76、69 必须注明统计范围，禁止混用。
7. 成本类别仅作为离线分析标签，不进入 actor。不得由相关性或权重改变量归因某一特征；无法定位时标不可判。

## 交付与结束条件

交付 `INPUT_PANEL_MANIFEST.json`、`BC_PATH_REPLAY.json`、`COMMON_INPUT_POLICY_COMPARISON.json`、`STATEMENT_CORRECTIONS.md`、`ROUND49_POLICY_CLOSURE_REPORT.md`、分析脚本和纯计算测试。报告说明模型和控制路径是否可对应、BC 输入是否可分、策略映射变化与轨迹变化分别能支持什么结论、当前是否支持具体方法改动；只给一个有依据的有限下一步，写明待区分假设、唯一变量、对照、可证伪结果，不执行。

提交只含本轮新分析文件，不含模型、大日志、旧件、受保护脏状态或其他会话文件。本地提交，不推送。报告列完整 HEAD、覆盖/缺口、实际测试范围和六项零操作计数，然后停止。
