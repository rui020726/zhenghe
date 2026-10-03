# ROUND42-R 固定计划完成报告

## 结论

- 固定计划完成：新增续执行604项，累计成功逻辑任务672/672。
- 额度闭合：normal 672 + reserve 1 = 673次claim/构造，未超过700上限，reserve余27。
- 未取得跨seed、跨模式的稳定保任务降本结论。NEW在seed11 sampled相对OLD出现一组描述性结果：任务数相同、少计费4566字节；但NEW相对immediate没有任何seed/mode组合保持全部任务数，且其他组合出现任务损失。
- 开发块为既有16场景，不支持泛化、总体零损失或统计非劣结论。

## 执行身份

- 分支：fix/comm-one-shot-gate-resume
- 本次执行源码HEAD：0369408dec2cfabffb623e0445174725f9dde0d5
- 原ROUND42 prelaunch HEAD：f46fe4703aa03ce6400f63c601c3203ef741dd9d
- 已知复核HEAD：2825069eaf9acf0fff87a9a02e69ae2f8e88c785
- 修复前最近交付HEAD：5cf41b5edd1b5e13f41b15abc08ae5876da4cd11
- 恢复执行版本：round42-gate-and-exact-resume/2；路径门禁：round42-frozen-checkpoint-path-map/2
- 训练语义、动作语义、runner核心保持原值：constrained-cost-v2/one-shot-send-skip/1/published-completion-event-at-relay-tick/2、one-shot-send-skip/1、counted-jointrun-proxy/1。
- v2执行时长：7795.11秒。源代码冻结身份在HEAD 0369408；本报告及结果索引作为后续独立本地提交保存。

ROUND42_REPORT.md由冻结的ROUND42报告入口生成。其初始身份栏将执行HEAD与原prelaunch HEAD混为一项，并沿用了原分支名；现已更正为上列原prelaunch、执行HEAD和恢复分支。科学结果表未更改。

## 门禁修复与精确恢复

实际首批gate失败原因是Windows反斜杠与正斜杠使同一冻结checkpoint路径集合比较失败。修复保留盘符、完整目录和文件名，拒绝规范化冲突或根目录外路径，并对两个文件重新计算冻结SHA256。路径回归9项通过；适配器AST等价回归2项通过。

v2使用ROUND40训练器的独立副本。唯一代码变更将parameter_identity导入移到train_seed函数作用域；移除该导入后，适配器和ROUND40冻结源码的整个模块AST相同。ROUND40外部冻结源和科研/训练数学均未改动。

实际完整seed11 batch0 gate通过；随后seed11、seed22、seed33首四集gate均通过。episode4恢复源SHA256为a85722e0864456ea3eb735195c7ffcde4be9d42301bdf2bc0781db6747a851ef，游标episode4、update1、Adam step12、lambda1.0125。四个模型模块、10个Adam状态及step/moments、lambda/历史/批游标、Python/NumPy/Torch CPU RNG及下一次随机输出全部精确匹配。gate和恢复检查合计0环境、0 optimizer step、0 checkpoint写入。

attempt1因parameter_identity导入作用域错误在新claim之前停止，claim、环境、step和写入均为0；停止报告、traceback、gate及恢复记录保存在原v1目录，并有8份字节一致副本。attempt1不占额度。新v2单独冻结并继续；未覆盖旧stop、旧失败gate或原运行材料。

## 固定计划和累计账本

| 工作块 | 新完成项 |
|---|---:|
| seed11 episode4–127 | 124 |
| seed22训练128集 | 128 |
| seed33训练128集 | 128 |
| OLD/NEW × 3 seed × greedy/sampled × 16场景 | 192 |
| late_discovery_gate / always_hold × 16场景 | 32 |
| 合计 | 604 |

| 累计计数 | 结果 |
|---|---:|
| 成功逻辑任务 | 672/672 |
| normal / reserve / total claims | 672 / 1 / 673 |
| 环境构造 / JointRun构造 | 673 / 673 |
| run调用 / 成功episode步数 | 672 / 672,000 |
| 历史技术失败 / 本轮新技术失败 | 1 / 0 |
| PPO更新 / Adam steps | 96 / 1,092 |
| checkpoint写入 | 297 |
| reserve剩余 / 总硬上限 | 27 / 700 |

历史技术失败是最初immediate scene1300000构造失败；其reserve重试成功且计入既有总额度。attempt1恢复启动前停止，不增加claim。失败样本不进入科学任务结果。

### 训练汇总

| Seed | 按时任务/128 | 计费字节 | 首报 | Free SEND | Free SKIP | 最终λ | Adam steps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 739 | 762,114 | 305 | 305 | 457 | 1.2875 | 352 |
| 22 | 732 | 449,154 | 180 | 180 | 582 | 1.3750 | 400 |
| 33 | 741 | 737,754 | 296 | 296 | 466 | 1.2625 | 340 |

### 开发块配对结果

每格为同一16场景开发块；首报数是relay接受的首报发射数。负的字节差表示NEW比OLD少。

| Seed / 模式 | OLD 任务/字节/首报 | NEW 任务/字节/首报 | NEW−OLD任务/字节 | 丢失/救回目标 |
|---|---:|---:|---:|---:|
| 11 / greedy | 88 / 0 / 0 | 88 / 0 / 0 | 0 / 0 | 0 / 0 |
| 11 / sampled | 94 / 99,276 / 40 | 94 / 94,710 / 38 | 0 / −4,566 | 0 / 0 |
| 22 / greedy | 88 / 0 / 0 | 88 / 0 / 0 | 0 / 0 | 0 / 0 |
| 22 / sampled | 95 / 110,304 / 44 | 93 / 55,896 / 22 | −2 / −54,408 | 2 / 0 |
| 33 / greedy | 96 / 239,028 / 96 | 88 / 0 / 0 | −8 / −239,028 | 8 / 0 |
| 33 / sampled | 95 / 151,374 / 61 | 92 / 50,250 / 20 | −3 / −101,124 | 3 / 0 |

immediate开发基线为96个按时任务、239,028字节。两个greedy NEW seed呈全SKIP且低于immediate任务数；seed11 sampled相对OLD任务持平并少4566字节，但相对immediate少2任务。seed22 sampled、seed33 greedy/sample分别损失2、8、3个任务。按目标的丢失、救回、集合变化和逐场景详情见ROUND42_PAIRED_RESULTS.json。

### 规则基线

| 规则 | 按时任务/16场景 | 字节 | 首报 | 成功送达包 | 丢包 |
|---|---:|---:|---:|---:|---:|
| late_discovery_gate | 96 | 77,412 | 31 | 125 | 61 |
| always_hold | 88 | 0 | 0 | 0 | 0 |

## 原件、checkpoint与复现

原停机运行370个纳入清单的文件哈希全部不变，库存摘要为5756513d565efbafcbfd65a1187f3262222e2dcb1e8a877309ffe269156adbce。新运行raw artifact manifest记录4,349个文件；ROUND42_CHECKPOINT_INDEX.json记录三个seed的ep0/64/128、逐批参数链，以及只读OLD ep128检查。原始大文件保存在本机Git忽略目录，清单与报告纳入本地结果提交。

运行环境：Python 3.10.19、NumPy 2.2.6、PyTorch 2.9.0+cu128；runner按冻结配置使用CPU。

源码门禁回归命令：

    D:\project_env\python.exe -m unittest discover -s runs/comm_one_shot_gate_resume_20261003_v2 -p test_r42_gate_resume.py -v
    D:\project_env\python.exe -m unittest discover -s runs/comm_one_shot_gate_resume_20261003_v2 -p test_r42_resume_adapter.py -v

本轮实际执行命令：

    D:\project_env\python.exe -u runs/comm_one_shot_gate_resume_20261003_v2/r42_resume.py

命令使用的执行HEAD为0369408dec2cfabffb623e0445174725f9dde0d5。完成目录保留start marker并拒绝重复运行；不要删除输出后再次执行。此次没有推送、部署或新增实验。

## 机器可读交付

- ROUND42R_COMPLETION_AUDIT.json：本报告关键指标、身份、门禁、恢复、额度和验证结果。
- ROUND42_RESUME_FINAL_RESULT.json：完整结构化结果。
- ROUND42R_EXECUTION_LEDGER.json 与 ROUND42R_LOGICAL_JOB_LEDGER.json：累计额度和逻辑任务状态。
- ROUND42_FIRST_BATCH_GATES.json、runtime/ROUND42_EXACT_RESUME_VERIFICATION.json：真实门禁与精确恢复证据。
- ROUND42_PAIRED_RESULTS.json、ROUND42_CHECKPOINT_INDEX.json、ROUND42_RAW_ARTIFACT_MANIFEST.json：配对结果、checkpoint和原始文件索引。
- ROUND42_VERDICT.json：固定开发块上的结论标记。
