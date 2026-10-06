# ROUND45-R2 首次入口失败记录

首次修复后执行在首个 BC-only 控制格的 claim 之前停止。实际异常来自
`_load_bc_eval_controller` 对 `StageCController` 容器调用 `.eval()`；该对象是控制器容器，
推理模式应只设置到它持有的四个 `torch.nn.Module`：actor trunk、action head、task value head
和 cost value head。

```text
AttributeError: 'StageCController' object has no attribute 'eval'
```

该次执行没有取得 normal/reserve claim，没有构造或运行环境，没有成功 episode、PPO/Adam 更新，
也没有写入训练 checkpoint。BC 不重跑。原运行账本、报告、冻结件和 prelaunch 收据 SHA256
记录在 `ROUND45_R2_PRE_ENV_ENTRY_FAILURE_V1.json`；旧冻结文件继续保留。

原因是运行器调用对象类型错误。它不构成科学结果，不计入固定计划，也不消耗环境额度。
