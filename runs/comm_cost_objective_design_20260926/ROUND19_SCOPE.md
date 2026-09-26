# 第十九轮范围与方法：收束搜索支线，返回学习型通信控制主线

日期：2026-09-26。分支 `docs/comm-cost-objective-design`（本轮新建）。
本轮**只做分析与设计**：0 新 episode、0 新流、0 训练、0 生产代码改动、0 默认值改动、
搜索/分配权重一字不动、封存集（`800000..800399`、`900000..900199`）未开启、未启动 pilot。

## 一、起点核验（先写失败，再写成功）

| 项 | 内容 | 结果 |
|---|---|---|
| 本地第十八轮头 | `git rev-parse HEAD`（在建立工作分支之前） | **`4c71fead0047298101869200a8518059afc42ccc`**，与要求的完整 SHA 逐字符相同 |
| 本地工作树 | `git status --porcelain` 中被跟踪文件的差异 | **0 项**；未跟踪项只有其他会话的产物（`architecture/`、`runs/comm_send_actionability_20260922/rows_s*.json`、`dev_summary.json` 等），本轮**一个都不动、一个都不提交** |
| 工作分支 | `git switch -c docs/comm-cost-objective-design 4c71fea…` | 成功，父提交即第十八轮头，**未切回旧代码执行** |
| 远端 `docs/comm-cost-objective-plan` | `git fetch origin refs/heads/docs/comm-cost-objective-plan:…` | **失败**：`Recv failure: Connection was reset`（1 次）；随后 `git ls-remote` **失败 3 次**：`Failed to connect to github.com port 443 after ~21 s` |
| 要求的远端完整 SHA `72c4c3f71e07339bf6533fbd771aef3b2682103c` | 核对 | **未能核对**。`git cat-file -t` 该对象：本地不存在。全库搜索（`--all --grep`、`reflog`、`fsck --dangling` 的 1 个游离提交 + 4 个游离 blob 逐个查看、`find -iname *ROUND19*`）**均无该提交及其文件** |
| 替代通道 | `curl https://api.github.com/repos/rui020726/zhenghe` → **404**（仓库私有，未认证 API 读不到，且本轮不注入凭据）；`github.com` 443 超时；`raw.githubusercontent.com` TLS 握手失败；`pypi.org` 200 ⇒ 出站网络活着，**只有 github.com:443 不通** | 不能替代 `git fetch`：拿不到 commit/tree 对象，也就拿不到可核验的 SHA |
| cherry-pick 该文档提交 | — | **未执行**（对象不在本地，且祖先关系无法核验；按要求"若已有后续改动，先核对祖先关系"，此处连远端都未核对，故不猜） |
| 两份指定文档 `ROUND19_COMM_MAINLINE_RETURN.md`、`ROUND19_CONSTRAINED_COMM_CONTROLLER_DESIGN.md` | 完整读取 | **未读到**（只在远端分支上）⇒ 本轮所有产出都只依据仓库内已跟踪的原件，见 §四"缺证据" |
| 最后**实际核验过**的远端状态 | `refs/remotes/origin/analysis/real-search-undetected-attribution` | `80af16ba4b90ce0c2d5456dd8b8e57068b4513e4`（第十八轮登记）。第十八轮分支 `analysis/search-trace-and-policy` 的远端 SHA **至今未核验** |

已读并作为约束使用的：`AGENTS.md`、`PROJECT_CHARTER.md`（v4.1 §1、§6、§7、§8、§9、§10、§11）、
`PROJECT_STATUS.md`、`ARCHITECTURE.md`、第十八轮全套产物、第十六轮 D1 产物、第十五轮收束件。
**没有读到的那两份文档不影响"哪些话不能说"**：那部分纪律来自仓库内已跟踪的总纲与收束件。

## 二、为什么本轮仍然可以做完整的分析

要求的五步里，①②只依赖磁盘上已有的日志；③④是设计件，其约束条件全部写在仓库里；
⑤里只有"推送并核对远端 SHA"一项被网络挡住。因此本轮照原顺序交付，把不能核实的部分
逐项登记为缺证据，而不是用推断补全。

## 三、产物索引

| 文件 | 内容 | 机读件 |
|---|---|---|
| `ROUND19_SCOPE.md` | 本文：起点、网络失败、范围与禁止 | — |
| `ROUND18_ARCHIVE_CLOSURE.md` | 步骤①：第十八轮归档 + 发现数/按时完成数闭合 | `ROUND18_ARCHIVE_CLOSURE.json`（脚本 `c19_closure_r18.py`） |
| `COMM_COST_COVERAGE.md` | 步骤②：公平强规则基线、成本覆盖范围、冗余分解、deadline 余量 | `COMM_COST_COVERAGE.json`（`c19_cost_coverage.py`）、`LEARNABILITY_PROBE_OFFLINE.json`（`c19_learnability.py`） |
| `CONSTRAINED_COMM_COST_DESIGN.md` | 步骤③：显式优化通信成本、同时约束任务效果的最小学习控制器设计 | — |
| `CONSTRAINED_COMM_COST_PILOT.md` | 步骤④：**待批准**的 pilot 方案、容忍度、资源上限、停止条件 | — |
| `ROUND19_VERIFICATION.json`、`MANIFEST.json` | 步骤⑤：本轮自检（生产代码零改动、0 episode、数字一致性） | `c19_verify.py` |
| `rejected/` | 本轮内部被替换的三份旧产物（改一次登记一次，不覆盖不删除） | — |

## 四、缺证据登记（本轮不补推断）

1. **远端 SHA `72c4c3f7…` 未核对**，第十八轮分支与本轮分支的远端状态**未核验**；网络恢复后的
   恢复命令写在 `CONSTRAINED_COMM_COST_PILOT.md` §七。
2. **两份第十九轮指定文档未读**：若其中已冻结了基线定义、容忍度或资源上限，本轮设计件
   一律让位于它——本轮交付的是**决策包**，不是既成事实。
3. **没有任何一个部署路径上的中间成本点**：磁盘上部署配置只有 0 B 与 14 573.3 B/ep 两个成本
   端点（第十六轮 A2 与 A1），中间只能靠"关掉一部分报告"去实测，不能离线算出来。
4. **一次性把视距刷新与一跳 gossip 分开的臂不存在**：两者在同一个函数、同一个间隔开关下
   （`task_allocation/old_task_env.py:477-512`、`:1107-1110`），所以"冗余知识来自哪条免费信道"
   这个问题**无证据可答**。
5. **19 维网络输入里，首报的 8 个变化检测维度在部署默认路径上是常量**（见 `COMM_COST_COVERAGE.md`
   §五）——这是实测，不是推断；但"因此学习器没有额外可学的东西"这句**不是**本轮结论，
   它需要在有按时失败的运行点上测。
6. 阈值规则（§`LEARNABILITY_PROBE_OFFLINE.json`）是在**同一批 60 seed 上读出来的**，
   未经未见 seed 验证；它的数字只能作为**上界与基线定义**引用，不得写成"可省 78 %"。
