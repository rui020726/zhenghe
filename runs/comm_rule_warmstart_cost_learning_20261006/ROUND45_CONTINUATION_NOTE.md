续接说明（2026-10-06）：
- 本轮当前分支为 exp/comm-rule-warmstart-cost-learning，标签前分割提交为 4cbd3df6b6b7c20453bf2b7fddcc720ca0348a47，祖先为 ROUND44 最终交付 cbd437afad6ddc3a7a68c8b7e10c0cd572a778dd。先核当前状态并沿该分支继续，不新建、不重置、不切回起点。
- canonical teacher 为只读路径 D:\code\整合\zhenghe\runs\comm_cost_v2_rerun_20260928\v2_rules.py，SHA256 a7ce36de2d465cadaba90bd70b36539b5e865a73be8c4e075b9faf5a86c53a57；它匹配 R44 冻结身份。ROUND42 同名副本 e307bdd1…不是本轮 teacher 来源。
- 已确认 R44 384 训练 tape 的文件哈希和大小全部匹配原清单，三 seed 各128。还必须逐样本验证动作前特征、标签与自由机会，不能由文件哈希推定标签完整。
- 尚无已知真实 BC 更新或 ROUND45 新环境启动证据。继续之前核对本轮账本与 checkpoint，不能按旧摘要猜计数。
- EXECUTION_BASIS.md 是摘要时，保留其原件与提交；本文件保存完整原始 ROUND45 提示词，续接说明另存 ROUND45_CONTINUATION_NOTE.md。此处不新建研究 spec。
- 本轮额度528正常+24技术 reserve=552总上限；准确已用量从账本取。门禁通过后自动继续，不逐阶段请示。不推送、不部署、不启动ROUND46。
- 旧 ROUND43 附件和 R44 结果均已完成，不作为当前任务或新的实验授权。

复核续接记录（2026-10-06）：
- R45 主 OLD 比较基线严格使用 ROUND44 的 NEW ep128 模型及其最终96格评估矩阵；ROUND43 ep128 的96行只留在 R44 baseline-reuse 历史归档中，不能改标为 R44 OLD 或用作 R45 主比较。
- 已通过只读 R44 门禁：反序列化 seed 11/22/33 的 ep128 checkpoint，逐个核对文件字节 SHA、Raw Manifest、训练状态/version、全部参数 SHA 与四个模块参数 SHA；参数摘要与 R44 checkpoint index 及参数链吻合。
- 使用冻结 R44 `r44_eval_index.audit_fixed_matrix` 对磁盘 claims/rows/receipts/artifacts 重建矩阵，结果与交付 `ROUND44_EVAL_MATRIX_AUDIT.json` 完全一致：96格、6组各16格。s11/greedy/1301000 只选 attempt 2；失败 attempt 1 保留在历史中且不入科学矩阵。此次只读操作不创建环境、不调用仿真、不训练。
- R45 greedy 诊断按实际 `OneShotSendSkipCollector` 规则对 float32 softmax 概率比较，`p_SEND >= p_HOLD` 时发 SEND；等 logits 反例已通过，0环境、0 optimizer step。BC optimizer seam 测试使用实际梯度准备/裁剪路径核对真实 trunk 与 action 参数身份；不再把零梯度/裁剪的自比较循环称为独立验证。
- 当前计数：真实 BC 更新0；PPO/optimizer 更新0；ROUND45环境构造0；合成 optimizer step累计2（已保存历史结果）；额度 claim 0。冻结源提交完成之前不得启动任何实际 BC 更新。
