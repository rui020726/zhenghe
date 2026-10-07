# ROUND52 执行依据

任务：ROUND52-CRITIC-GRADIENT-ISOLATION-PROTOTYPE。

接手 HEAD：e51bcb82a50b5959165f94fea1de02120ebec44d；分支 analysis/round52-critic-gradient-isolation-prototype。

范围仅限 critic-to-shared-trunk stop-gradient 的隔离原型、真实已记录 batch0 的纯计算对照、闭式测试、报告及本地提交。严格不调用训练器更新，不运行环境/评估，不调用 optimizer.step/backward，不写 checkpoint。6项零操作计数：环境构造、仿真、新场景/新流、新评估、真实训练更新、合成 optimizer.step、checkpoint写回均为0（合计7类）。

依据：ROUND51已核验物理训练器SHA256 0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841；三seed batch0真实265 decision样本；ROUND49既有576行真实面板；实际SEND=1/SKIP=0。使用同seed BC_POST初值。development_go=false，所有历史及受保护脏文件只读。

唯一候选变量：task/cost critic loss 对共享 trunk 的梯度回流被切断；actor/entropy/前向/头部梯度与所有其余配置保持不变。原型仅使用本新目录临时 module hook/context，不更改生产代码/训练器/checkpoint。
