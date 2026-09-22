# target-scoped 边界可行性结果

启动：24 / 28

## T0 实现可信

- 合成测试 + 冒烟：通过（详见 INTERFACE_CHECK）

## T1 起点可执行

- 通过：[{'seed': 700010, 'start_send_ok': True, 'detail': 'ok'}, {'seed': 700018, 'start_send_ok': True, 'detail': 'ok'}, {'seed': 700020, 'start_send_ok': True, 'detail': 'ok'}, {'seed': 700022, 'start_send_ok': True, 'detail': 'ok'}]

## T2 边界可解析（按 H）

| H | 可解析 pair | 阈值 | 判定 |
|---:|---:|---|---|
| 1 | 2/4 | 3/4 | 未过 |
| 5 | 0/4 | 3/4 | 未过 |
| 20 | 3/4 | 3/4 | 通过 |

pair 可解析 = 两分支均为 `flushed_latest` 或 `terminal_before_boundary`。

### 判读

- 仅当某 H 的 T2 通过，才可建议以后对该 H 跑完整 episode（本轮不执行）。
- H=20 不过而短 H 通过 ⇒ 只报告较短延迟可执行，不自动缩短研究问题。
- 全部不过 ⇒ 下一步只能单独讨论 credit reservation 新协议。
- 通过也不证明任务收益、观测可预测优劣或学习优于规则。

