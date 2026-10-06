当前任务=ROUND46；不是ROUND43，不恢复失败的ROUND45；384训练+96评估，储备24，hard504。

# ROUND46 执行依据

本文件按当前用户给出的 ROUND46 完整任务卡及本轮身份纠偏整理，作为本轮冻结依据。旧 ROUND43 的完成件只作为历史输入；它的 480 次启动不计入 ROUND46。ROUND45 的 BC-only 控制门禁继续保持 failed，原报告、stop 状态、claims、rows、RNG、训练和账本均不改写。本轮不调用 ROUND45 的固定计划入口，也不把 ROUND45 门禁改成通过。

## 研究与训练定义

- 研究对象是冻结 BC 四模块初始化后的 constrained PPO，训练 seed 为 11、22、33；每 seed 固定 128 集、32 个四集 batch。
- 每 seed 只从相同 seed 的 ROUND45 R2 `BC_POST_MODEL_s{seed}.pt` 读取 trunk、action head、task value head、positive-cost value head。BC optimizer 和 BC RNG 不恢复。
- 每 seed 创建完整新 Adam，含10个唯一参数，初始 Adam state 为空；训练游标、update、optimizer step 均为0，`lambda_dual=1`。
- 只从对应 ROUND44 ep0 状态恢复已登记的 Python、NumPy、Torch rollout RNG；不恢复 ROUND44 模型或 optimizer。BC 权重载入不得改变 ambient RNG。
- 训练版本沿用已冻结的 `constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/dual-eta-lambda-0p5/1/rule-bc-late147-init/1/published-completion-event-at-relay-tick/2`。本轮单独登记 ROUND46 runner 和恢复身份。
- 固定训练场景为 ROUND44 原始128项序列逐 seed 重用，范围 1300000–1300047；开发场景 1301000–1301015。冻结文件保留完整序列，并记录 SHA-256 的序列化算法；冻结前对序列逐项比较。
- 保持原协议、PILOC/CTDE 权重、19维观测、mask、奖励时序、成本定义、gamma=1、GAE lambda=1、ALL_ROWS、B0=10000 B、delta=0、`eta_lambda=0.5`、`lambda_next=max(0,lambda+0.5*g_hat)`、PPO/双 critic/optimizer 参数、预算、ACK 关闭、1000步及场景参数。不得调参、改变策略或增加训练。

## 固定任务与额度

- 新 ROUND46 正常任务：384 个训练 episode + 96 个 ep128 最终评估格（3 seed × greedy/sample × 16 开发场景）=480 次启动。
- 技术储备最多24次，ROUND46 硬上限504。失败不退额；真实环境构造、run 调用、完成步数、训练更新都分别入账。
- ROUND45 已发生的48个环境注册/启动单独保留，不并入 ROUND46 的480正常任务；联合 family 上限为552（48历史 + ROUND46 504）。
- 环境 claim 必须先原子提交，场景键不可重放。科学表现差、全 SEND、全 SKIP、任务损失都不是技术失败。
- 可复用身份未变的 ROUND42/44/45 参考、规则和 BC greedy 控制数据；不重跑参考，不扫描无关历史实验。训练所需64个已登记 immediate 参考只按其冻结清单构造缓存并逐行核验。

## 运行门禁与顺序

1. 首次环境前，冻结并本地提交本执行依据、ROUND46 runner、配置差异、完整场景/逻辑键计划、训练版本与输入身份。
2. 运行零环境、零 optimizer.step 的 preflight：核验三个 BC checkpoint 字节和四模块参数摘要、R44 ep0 RNG 来源、CPU 单线程、空 Adam、10参数集合、实际模块/版本/预算接线、参考缓存身份与恢复边界。ROUND45 旧 failed gate 仅读取并校验其原 SHA。
3. 依次对 seed11、22、33 各运行首4集并核验其真实 batch0。三个 seed 的门禁全部通过后，才按 seed11、22、33 各自完成剩余124集。任何正确性问题立即停在受影响位置并保留证据。
4. 完成三个 seed 固定 ep128 checkpoint 后，只运行96个 NEW greedy/sample 评估格；不选 checkpoint、不扫阈值、不调参、不增场景。
5. 统计使用16场景为单位的共享 bootstrap 索引：PCG64(20261007)，10000×16，均值按 `values[index].mean(axis=1)`，线性分位数。

## 判读与停止

greedy 为主，sampled 为辅，逐 seed 报绝对任务数、计费字节、首报与 SEND/SKIP、逐目标集合、相对 BC greedy、immediate、late、always_hold 的差值和配对区间。BC sampled 只在存在已核验的冻结旧格时配对；不得为补齐对照增加授权外启动。

仅当三个 seed 的 greedy 目标身份集合各自与 immediate 完全一致，且每 seed 计费均严格低于 late 规则的固定开发块均值，才报告“开发块联合改善”。否则如实报告任务保持/成本/跨 seed 判定；CI 含0不称统计优势。开发块已经用于开发，不能写成正式非劣、泛化或总体零损失，也不能声称已证明失败唯一源自信用分配。

模型、tape、日志与运行账本留在 Git 外；版本化报告、门禁摘要、清单及 runner 只放入 `runs/comm_bc_initialized_constrained_ppo_20261007/`。完成后只本地提交并停止，不推送、不部署、不追加实验。
