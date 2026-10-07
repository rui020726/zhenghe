# ROUND53 执行依据

任务：ROUND53-ZERO-LAUNCH-ISOLATED-TRIAL-READINESS。

接手 HEAD：40adea4e16c5ccd6ee454f86462e4c959149da8a；当前分支 feat/round53-critic-isolation-trial-readiness。

本轮只准备 future trial 接线、零启动预检和资源/基线核验。552次仍等待明确批准；当前 authorization 必须 PENDING_USER_APPROVAL。绝不启动环境、仿真、新流/评估/训练，不创建真实 claim/ledger/checkpoint。

唯一因子是ROUND52验证过的 task/cost critic shared-trunk stop-gradient。复用物理冻结trainer SHA 0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841、同seed R45 BC_POST四模块初始化、fresh Adam/dual/RNG/数据顺序合同。不得误用R43旧ep0，不得改变科研配置/数学。

资源提案：384训练+48 candidate greedy +48 candidate sampled +48 BC sampled=528 normal，24全新技术失败储备，cap552；全标 proposed_not_authorized，不借旧额度。

保护两份ROUND46既有脏文件和所有旧源/数据；仅新目录及必要最小白名单修改。本轮7类操作计数均须为0：环境构造、仿真、新场景/新流、新评估、真实训练更新、合成optimizer.step、checkpoint写回。

允许纯CPU forward/autograd.grad、slots stub、fake booking/ledger fixture、AST及小型纯计算单测。不调用configure/build/run/评估/训练入口，不恢复optimizer/RNG，不写.pt，不联网/推送/合并/部署。

READY_CHECK为条件执行就绪而非批准。批准必须来自后续独立明确执行依据及额度，且先核对科学/运行代码与输入绑定；单纯文档提交HEAD变化不得阻断匹配代码执行。当前开发判定development_go=false保持。
