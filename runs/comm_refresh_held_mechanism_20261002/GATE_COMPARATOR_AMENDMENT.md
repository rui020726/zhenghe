# 首场景复现门禁比较器勘误

ROUND39 预启动提交 `e8f3e76f5555e1da271e6d8d560c32fadf151ba2` 后，首场景九格全部完成。第一次门禁执行在 seed11 原臂处停止：行为行仅 `config_hash` 与 ROUND37 不同。停止后没有再启动环境。

只读核查了六个原臂/规则对照（三个 original greedy、immediate、late_147、always_hold）：

- 排除 `config_hash` 后，行内模型身份、任务/通信行为、字节流、奖励流、`behaviour_sha256` 等字段全部相同；
- 从生命周期首行完整读取 `RunConfiguration` 和场景详情，排除 `output`、`comm_transcript`、`decision_trace`、`reservation_log` 后，配置与场景逐字段相同；
- `run_signature` 的 arm、checkpoint、seed、stream、schema 等行为身份相同；记录的 Git commit 因 ROUND39 冻结提交不同，且 17 个 ROUND37 冻结源文件哈希全部匹配；
- 原始 `config_hash` 差异来自每轮不同的 `comm_transcript` 输出路径。`integrated_sar/run_signature.py` 的哈希函数排除了 `output`，但没有排除 `comm_transcript`；它属于本轮输出路径元数据。

第一次门禁还把生命周期首条 `run_config` 记录纳入执行前缀比较，该记录包含输出路径和 Git commit。勘误后的前缀比较仅覆盖实际任务/生命周期事件，配置身份另按上述字段独立比较。

经预先授权的“排除本轮路径等非行为元数据”规则进行门禁复核后，六个历史对照全匹配；首发、预算顺序、权重不变、refresh 抑制和 s22 负对照门禁均通过。复核是只读的，环境启动仍为 9，技术失败与重试均为 0。未修改 `r39_collector.py`、`v2_run.py` 或生产目录中的行为代码；改动仅用于门禁比较和接续调度。

更新后的门禁结果见 `FIRST_SCENE_GATE.json`。剩余 135 格由独立接续脚本按同一已冻结任务清单和已有计数继续。
