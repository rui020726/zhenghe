# ROUND58 恢复门禁通过后停止报告（V17）

## 状态

- **本地 HEAD：** `b60bafd864b61c26c32de6de25ffe5b257817e3d`
- **分支：** `exp/round58-prior-strength-prospective-trial`
- **状态：** 三 seed 的只读真实恢复门禁通过；正式续执行在 seed11 episode4 的环境配置阶段遭遇 `sp_common` 导入错误。依冻结纪律已停止，没有在 V16 身份下热修或再次启动。
- **V16 执行身份提交：** `b60bafd864b61c26c32de6de25ffe5b257817e3d`
- **train_pilot 导入修复来源提交：** `85a1cfade0cd1659004c4d8b500b417ba8982a3c`

## 恢复门禁

V17 门禁实际执行了私有 launcher 的 `__main__` 路线、V16 授权/冻结/额度/场景校验、迁移证明、configure/bind 和真实 checkpoint loader。三个 seed 均恢复了获准源；controller 张量、完整 Adam state/moments、参数组、λ/history、episode4/update1/optimizer_steps8 游标、RNG 与批准源闭合。每个 seed 的 Adam 为 8 steps、10 个状态项、1 个参数组。历史 checkpoint 中的旧 repair attestation 与新执行 freeze attestation 分别验证，没有把身份字段简单忽略。

`train_pilot` 实际加载为 `D:\code\整合\zhenghe\runs\comm_budget_pilot_20260919\train_pilot.py`，SHA256 `87dcf2e5251c99fc533ba52480a43da89902a3a0c371f5b9cfec6240d202ae81`。门禁在 episode provider 前受控停止。记录：45 项窄测试、退出码0；V17 restore gate 退出码0；torch load 9、RNG restore 9、torch save 0，环境/claim/PPO/optimizer/checkpoint payload/evaluation 均为0。V16 曾有一份由审计比较错误造成的失败报告，原样保留；V17 以明确区分 source attestation 与新执行 attestation 的核验通过。

## 正式续执行及停止原因

第一次 CLI 调用因传入相对授权路径，在真实入口前置校验处退出（exit 1）；未产生 claim/额度消费。随后使用绝对路径的 CLI 确实通过授权和恢复，日志显示 `R40 resumed seed 11 at episode 4 lambda=1.0000 updates=1`。接着冻结的 `v2_common.py:373` 在 `stage_b_inherited_config` 执行 `import sp_common as sc` 时抛出 `ModuleNotFoundError: No module named 'sp_common'`。这是 train_pilot 恢复导入之外的新运行时导入上下文问题。没有通过增加搜索路径或热修当前身份来绕过。

执行器为 episode `new_training_s11_e4` 自动生成并保留了两次失败尝试：normal slot 13 与 reserve slot 1。两份 claim/failure 原件分别记录：环境构造0、JointRun构造0、run invocation 0、simulation_started=false、completed_steps0、optimizer step0、checkpoint写入0；partial tape 均为空 SHA256 `e3b0…b855`。因此没有完成 episode，也没有计入任何训练更新；本次消耗了两个启动账本 slot，但没有新增环境构造。

## 额度和任务账本

- 历史实际环境消费：16（保留原分类/争议）。
- 本代此前完成：12 个 episode启动。
- 本次新增账本：1 normal + 1 reserve；两者失败且不退额。
- 累计启动 slot：30/1064；normal 13/1024，剩余 normal 容量1011；reserve 1/24，剩余23；累计硬上限余量1034。
- 按已知环境构造事件，本次增加0，累计仍为28；与启动slot账本分栏，不混称。
- 固定计划：本次完成0训练、0评估；372训练 episode与640评估格仍未完成（1012项逻辑计划未闭合）。seed11 episode4尝试失败不算完成。

故不能报告原计划完成，也不能在本轮继续。任何后续必须先解决新发现的导入来源问题并建立独立、版本化的执行身份；还需明确处理已经消耗的normal/reserve slot及失败 episode 的续行方式。此报告不是新的运行批准。

## 保留与交付

- V10及此前失败、V16审计失败、V17门禁通过、旧checkpoint/claim/tape/journal，以及本次两次claim/slot/failure原件均保留。
- 两个受保护文件 SHA256 核验一致：
  - `ROUND46_CURRENT_TASK.json`: `cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b`
  - `ROUND46_R2_V6_STOP_STATE.json`: `93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18`
- 机器报告：`ROUND58_RESUME_EXECUTION_STOP_AFTER_V17.json`，SHA256 `66837014cf073f7dabdb083917dd58f95ba8cc8cfe0b310e6ab9b1e252a07ba6`。
- 只提交了私有恢复适配、测试、V16 freeze/auth/map、V17 proof 与门禁审计材料；权重、tape、运行日志和本次失败原件不入 Git。保护文件仍为接手时的 dirty 状态，未暂存。
