# ROUND75 S1B 真实资格执行收口

请求 ID：R75_S1B_20261015
分支：exp/round75-native-ppo-qualification
HEAD：5ee9f2d9effb3bb6773688bf2f51fa33522bb346（parent 3803dad0468a24e0c78589c557aace00085e2b60）
输出根：runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/
attempt 目录：raw/s1_revised/R75_S1B_20261015/
结果：FAILED at A3_restore。A1 与 A2 已完成；A3 在任何更新前崩溃。

## 1. 实际消费（真实付费，不退额）
- physical_optimizer_step：5（A1 = 1 critic-only；A2 = 4 native update0；A3 = 0）。guard used_normal=5, used_reserve=0, refusals=0。
- offline_trunk_target_row：840（70 次 trunk 调用，每次 12 行）。guard used_normal=840。
- joint_record（batch-read）：1。guard used_normal=1。
- environment_construction：0。
- controller 构造 2/2、optimizer 构造 2/2、checkpoint 文件 load 3/3（两个 R58 ep0 + 一个 A1 bundle）。全部在预算内。
- 总台账：88 事件，SHA fff8dc1f3d9222ed280a5dc9139f3f82de4c5537ccbc69894d340b44e605475d，bytes 19980。

## 2. 失败点
- phase：A3_restore
- 错误：RuntimeError: Boolean value of Tensor with more than one value is ambiguous
- 位置：r75_s1_runner.py:167，_rng_payload_differences 内 if not _raw_equal(live_torch, expected_torch)
- 路径：restore_full_state -> verify_restore -> _rng_payload_differences
- 定性：A3 已完成 fresh controller 构造、第 2 个 optimizer 构造与 A1 bundle 的磁盘 load（第 3 次 checkpoint load），在 restore 起点门禁的 RNG 载荷比较处崩溃。live torch RNG 状态在 A2 更新后是含多于一个元素的张量（Philox offset），_raw_equal 对其做布尔求值触发歧义异常。这是 runner 恢复验证代码的真实缺陷，非测试缺陷、非预算超限。

## 3. 成功部分（原件路径）
- child 内置预检：74/74 通过，零成本。
- r66_frozen 模块加载成功；canonical 生产台账以零起点打开并核对一致。
- A1：1 次 critic-only 非空 Adam 步，bundle 已写盘：r75_s1_A1_full_state.pt（54708 B，SHA 23f34508357d0bfaa7b51a6f0ec34dc6462dfab603f402477c813f6e60596de0）。
- A2：1 次完整 native update0（4 physical steps），70 trunk calls，840 rows charged；meter A2 calls_attempted=70, calls_charged=70, rows_charged=840, calls_completed=70, rows_completed=840, refused=0, failed=0。
- 逐步观察：A1_A2 五步全部 completed，durable_used_before/after 逐项一致，每步恰好一个新 permit，delegated=true，step_charged_nothing=true。

## 4. 边界与结论
- 本结果只支持"此 R58 固定回放上的 A1/A2 路径在真实 torch/Adam/原生 PPO 下可执行并记账"，不证明真实 collector、完整训练恢复或策略效果。
- A3 序列化恢复的资格未被证明：崩溃发生在恢复验证门禁，恢复后的起点比较与 A3 native update0 均未执行。
- 未修代码、未重跑、未挑通过；失败原件保留，真实 counter 不填 0。
- S1 继续未放行其他用途；本报告不构成任何新的执行许可。

## 5. Git 外原件（不提交）
- r75_s1_A1_full_state.pt（付费权重）
- r75_s1_attempt_counters.jsonl / r75_s1_failure_receipts.jsonl
- r75_s1_outer_v2_attempt_marker_R75_S1B_20261015.json / r75_s1_outer_v2_license_consumed_R75_S1B_20261015.json
- r75_s1_outer_v2_prepared.json / r75_s1_outer_v2_outcome.json
- r75_s1_outer_v2_stdout.bin / r75_s1_outer_v2_stderr.bin（权威）/ .txt（有损副本）
