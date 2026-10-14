# ROUND75 S0 启动握手修复收口

请求 ID：R75_S0_STARTUP_HANDSHAKE_FIX_20261015
分支：exp/round75-native-ppo-qualification
parent_head：3803dad0468a24e0c78589c557aace00085e2b60
输出根：runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/
取代只读取证请求：R75_S0_COMMITTED_EVIDENCE_READ_20261015

本报告追加，不改写任何历史原件。范围为源码、stdlib 假接缝测试、冻结与本地收口。零环境：未导入 Torch/NumPy 实际 NN 路径，未做模型加载/构造、forward/backward、optimizer/PPO、环境、采样、checkpoint 写入、训练或 RNG 恢复。S1 仍未放行，本报告不构成执行许可。最终 HEAD 于提交后由 git 实读，本文件不内嵌自称。

## 1. 四个启动接缝缺陷（已修）
1. argv 双前缀：run() 与 launch_once() 都曾前置 [python, runner]。现由 build_child_argv() 只拼一次，launch_once() 原样 exec 完整向量，并以 argv[1] 必须等于 runner 路径做交叉校验，真实命令不可能携带两次 python/runner。
2. 父标记自拒：register_attempt() 先写预启动 marker，子进程 _bind_execution_identity() 又按同 id 拒绝，必拒父刚建的合法 marker。现为父到子一次许可承接：marker 记 license_state=issued 与真实 argv；子用 O_EXCL 写一次性消费记录接管；缺、不匹配、已消费、旧 id、历史目录、跨目录重放全部拒绝。
3. 测试发现：unittest.main 原在第 653 行，位于后 12 个方法之前，只跑了 30。入口已移到文件末尾；后 12 现在执行；旧 Ran-30 stdout 记录保留，并追加"后 12 曾未执行"的更正。
4. 身份/目录纪律：每次 attempt 独立目录 base/<request-id>；按 request id 全局去重；拒绝覆盖；prepare 不发许可、不启动 child。

## 2. 许可承接契约（采样/greedy/似然/评估之外，仅路径接线）
- 父 register_attempt 写 issued marker（含 argv）；
- 子 _bind_execution_identity 只接管 fresh 许可，O_EXCL 写消费记录一次；
- 中央生产台账仍在 raw/runtime，attempt 目录不成为第二台账；
- 该契约不触及训练数学、数据、RNG、PPO 矩阵，也不放行 S1。

## 3. 测试与预检结果
- test_r75_s0_import_bootstrap.py：声明=发现=执行=49，OK，exit 0。
  - 原 30（未动）；曾未执行的后 12（RunnerWiringAstTest 5 + RunnerIdentityRoutingTest 7，已按真实许可契约校正）；本轮新增 7（ParentChildLicenseSeamTest 5 + WrapperBehaviourTest 增 2）。
  - 桥接用例用真实 wrapper 登记、真实 runner 解析/绑定，paid 路径 run_s1 与 Popen 以 spy 拦截；fresh 一次到达 spy、second/跨目录/旧 id/已消费文件均拒绝；最终 argv 的 python 与 script 各出现一次。
- r75_s1_precheck.py：74 项检查全过，0 失败，exit 0，零成本；被 pin 的冻结源（r75_frozen/r66/canonical/guard/adapter）未改。

## 4. 本轮开发失败（如实披露）
首次套件运行 1 failure + 5 errors，均为测试自身缺陷，已修：
- 5 errors：ParentChildLicenseSeamTest 在调 register_attempt 前未建 attempt 目录，触发 FileNotFoundError；修法为 child_argv 先 mkdir，与 issue_fresh_license 一致。
- 1 failure：RunnerWiringAstTest.test_the_identity_is_bound_before_the_paid_path 用 ast.walk（BFS 序）判断调用先后，把 run_s1 当作索引 0；修法为按 node.lineno 源序比较。
六项皆非驱动或付费路径缺陷；修复后 49 OK exit 0；无一用于掩盖代码缺陷。

## 5. 未重跑
旧 40/34/31/20/7 套件未重跑，其记录保留。本轮仅运行此受影响纯 suite 与更新 pin 后的纯 precheck 各一次。

## 6. 冻结身份（当前盘上 SHA256）
- r75_s1_outer_exec_v2.py：c448d465c7b841419427fcf0c0357fe012720b723fc88ef697010fcf0ba0622e
- r75_s1_runner.py：3ba6ea272942a7f77d65ff6d5922de13395e7149d800972730ca83a8cbd2309d（wrapper 的 EXPECT_CODE 正好 pin 此值，一致）
- test_r75_s0_import_bootstrap.py：48ed8cec62f07c6e08f72f7022764b06ff24b1b2342bd0c8665cb19ad663ec05

## 7. 保护与边界
ROUND46_CURRENT_TASK.json（cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b）与 ROUND46_R2_V6_STOP_STATE.json（93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18）均未修改、未暂存，哈希复核一致。raw/s1 九件原件、原 wrapper r75_s1_outer_exec.py、旧输出、R59 分析日志全部保留。不 stage 权重/raw/runtime/保护件/__pycache__/临时日志；不 push/merge/deploy。S1 未授权、未执行。
