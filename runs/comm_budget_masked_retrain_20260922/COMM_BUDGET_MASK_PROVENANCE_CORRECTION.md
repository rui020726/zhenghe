# 上一轮 manifest 的 provenance 更正（不新增 episode）

日期：2026-09-22。本轮 `comm_budget_masked_retrain_20260922`，分支 `exp/comm-budget-masked-retrain`，执行指令 `COMM_BUDGET_MASKED_RETRAIN_NEXT_ROUND.md` §2。上一轮：`fix/comm-budget-action-mask`，产物目录 `runs/comm_budget_action_mask_fix_20260922/`。

**结论先说**：两处都是**身份口径**缺陷，不触及任何被测数量；M0–M3 的判定不因此改变。但正式训练会把权重与结论挂在一个 SHA 上，所以先把「哪段代码产生了这些数字」说清楚。

## 一、幽灵模块：`integrated_sar/comm_transcript.py`

- 盘上存在：`False`；Git 历史上任何一次提交加入过它：`False`（`(no commit ever added this path)`）；在派生 import 图里：`False`。
- 它所指的那个类真实定义在 `integrated_sar/comm.py`（模块 `integrated_sar.comm.py`），由 `joint_driver.py` 从 `.comm` 导入，从来不存在独立模块文件。
- 为什么它显示成 `dirty_modules` 而不是 `absent`：`git rev-parse HEAD:<不存在路径>` **不报错**，它把参数原样写到 stdout（实测 stdout=`'HEAD:integrated_sar/comm_transcript.py'`，returncode=128；`git cat-file -e` 才返回非零 128）。旧 builder 取前 16 字符当 blob id，于是幽灵条目得到一个假 blob 和 `matches_head: false`：

```json
{
 "head_blob": "HEAD:integrated_",
 "matches_head": false,
 "working_blob": null
}
```

- 旧 manifest 的 `dirty_modules` = `['integrated_sar/comm_transcript.py']`，其中真正存在且与 HEAD 一致的模块 5 个：`['integrated_sar/comm.py', 'integrated_sar/comm_ack.py', 'integrated_sar/comm_budget.py', 'integrated_sar/comm_learning.py', 'integrated_sar/comm_policy.py']`。
- the only entry in dirty_modules was a path that does not exist and was never executed; no executed module was dirty, so M0-M3's measurements stand, but the manifest's identity column did not

## 二、两个头：证据头 `7f9929c` 与交付头 `e692607`

- 证据头：`7f9929c83cd80185830ca1ece4ec6899629605cf|2026-09-22 16:04:19 +0800|docs(comm): record the budget-action mask fix, its gates and the corrected next-step boundary`；旧 manifest 的 `executed_code.head` = `7f9929c83`，旧 RESULTS 的 `head` = `7f9929c83`。
- 交付头：`e692607212fcc67fc82d045a8a53d44c79bde4d7|2026-09-22 16:11:06 +0800|test(comm): prove an old checkpoint is forward-only and keep the gates re-runnable`。
- 两者之间改动过的文件：`['runs/comm_budget_action_mask_fix_20260922/COMM_BUDGET_ACTION_MASK_MANIFEST.json', 'runs/comm_budget_action_mask_fix_20260922/COMM_BUDGET_ACTION_MASK_RESULTS.json', 'runs/comm_budget_action_mask_fix_20260922/COMM_BUDGET_ACTION_MASK_RESULTS.md', 'runs/comm_budget_action_mask_fix_20260922/test_budget_action_mask.py']`；其中属于执行图的：`[]`（空即表示交付头没有动过任何执行代码）。
- 派生执行图（49 个模块）在两头的差异：已提交 `[]`、工作区 `[]` → clean=`True`。
- M0-M3 were measured with the executed graph of 7f9929c; e692607 changed only a test file and the regenerated result documents, and the derived graph is identical at both heads (clean=True)
- a reader who re-runs the gates at e692607 gets the same code under test, but a reader who cites 'delivered at e692607' as the SHA of the measurement would be quoting a commit that only documents it

## 三、builder 怎么修，旧数字怎么留

- 修的是 `runs/comm_budget_action_mask_fix_20260922/mask_manifest.py`：模块清单改为**由 import 派生**（`derives_from_import_graph=True`），并且先用 `git cat-file -e` 判定「在不在 HEAD」再去 hash（`resolves_absence_before_hashing=True`）。
- 修好后重建的 builder 仍把幽灵算进清单吗？声明式契约里 `False`、派生图里 `False`；builder 现在对该路径的回答是`{"in_head": false, "on_disk": false, "head_blob": null, "working_blob": null, "matches_head": false}`——即 `in_head: false` 的缺失项，而不是脏模块。

**旧 smoke 数字原样保留（未重跑、未改写）：**

| 量 | 值 |
|---|---:|
| `candidates` | 20655 |
| `initial_executable_candidates` | 197 |
| `actionable_candidates` | 181 |
| `requested_by_policy` | 86 |
| `executed_policy_emissions` | 86 |
| `policy_refusals` | 0 |
| `max_abs_log_prob_gap` | 0 |
| `all_gates_passed` | True |

- 旧 smoke 闭合：requested==executed `True`、policy 拒付 0 `True`、log-prob 最大差 `0.0`、optimizer.step `0`。
- 本轮更正**启动的 episode 数**：`0`（纯身份核算，不产生新测量）。

## 四、本轮据此建立的冻结规则

- 做法：the executed graph is derived by importing the training entry, every member's blob is recorded against a registered code-freeze SHA, and any launch checks that the working tree and HEAD still match it
- 失败即停：no freeze is registered
- 失败即停：a graph member differs from the freeze in HEAD or in the working tree
- 失败即停：a graph member or a script in this directory is untracked
- 失败即停：a module appears in the graph that the freeze never recorded

## 五、旧文件未被改动（sha256 前 16 位）

- `runs/comm_budget_action_mask_fix_20260922/COMM_BUDGET_ACTION_MASK_MANIFEST.json` — `30d0a79fee00599c`
- `runs/comm_budget_action_mask_fix_20260922/COMM_BUDGET_ACTION_MASK_RESULTS.json` — `7e9ee79184d5e59a`
- `runs/comm_budget_action_mask_fix_20260922/COMM_BUDGET_ACTION_MASK_RESULTS.md` — `1e5d19dc37be7508`
- `runs/comm_budget_action_mask_fix_20260922/COMM_BUDGET_ACTION_MASK_SMOKE.json` — `2e7d449099540e87`
- `runs/comm_budget_action_mask_fix_20260922/COMM_BUDGET_ACTION_MASK_PROTOCOL.md` — `b35626688f36cb85`

