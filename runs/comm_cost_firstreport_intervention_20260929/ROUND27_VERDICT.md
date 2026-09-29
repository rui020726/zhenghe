# ROUND27 判定：固定失败场景的首报提前干预

场景 `1400031`（ROUND26 事后选定的失败案例，特权诊断场景）。

## 启动账本

正常 5/5，储备 0/3 未动，硬上限 8；失败 0、重试 0、optimizer.step 0、训练 0、阈值调整 0。

## 五路径对照表

| 路径 | 覆盖目标 | on_time | expired | 已计量字节 | 发射 |
| --- | --- | --- | --- | --- | --- |
| L | — | 4 | 2 | 10578 | 4 |
| L0 | piloc-e1-target-0, piloc-e1-target-5 | 4 | 2 | 10578 | 4 |
| E0 | piloc-e1-target-0 | 4 | 2 | 10578 | 4 |
| E5 | piloc-e1-target-5 | 6 | 0 | 7764 | 3 |
| E05 | piloc-e1-target-0, piloc-e1-target-5 | 6 | 0 | 10578 | 4 |

## 干预有效性（相对 L）

| 比较 | 干预动作 | 接收 | 任务变化 | 成本变化 |
| --- | --- | --- | --- | --- |
| E0-L | 0@[19]; 5:未覆盖 | 0:5/6 accepted; 5:未覆盖 | on_time +0（获得 ，损失 ） | +0 B |
| E5-L | 0:未覆盖; 5@[1] | 0:未覆盖; 5:5/6 accepted | on_time +2（获得 0,5，损失 ） | -2814 B |
| E05-L | 0@[19]; 5@[1] | 0:4/6 accepted; 5:5/6 accepted | on_time +2（获得 0,5，损失 ） | +0 B |

## 机制（描述性，单场景）

- 决定性干预：E5。
- E0 单独：an accepted first report for target-0 did not move the allocation chain: the first assignment of target-0 still waited for step 31 and the work pattern was already the same segments as L。
- E5：the acceptance at step 1 seeded the belief of five robots; the assignment chain for target-5 started at step 1 and the work segments ran 2-21 with five then two robots; target-0's chain restarted at steps 31-60 with five robots and finished at 70.0。
- E05：both named targets on time (60.0 and 66.0), whole scene 6/6；组合相互作用：the combination did not show a task effect beyond E5's; the two targets' work chains interleaved differently but the deadline outcomes coincide。

## 判读纪律

- in this one post-hoc selected scene and this random implementation, advancing the first report of target-5 changed the failing outcome; that is all this run supports
- it is not established that early sending is generally effective, that all tasks should be announced early, or that the actor needs an ACK channel
- one scene, one RNG implementation: no significance, no bootstrap, no non-inferiority claim, no new seeds, channel repeats or rule searches
- different emissions change the channel RNG consumption, so the post-override paths are not the same per-message loss batch; outcome changes are not attributable to the single accepted message
- E0 is a real intervention with an accepted message that changed no outcome: it is evidence that an accepted first report is not sufficient here, not evidence that early sending is ineffective

## 发布状态

remote_verification = NOT_PERFORMED；publication_status = PENDING_NETWORK；本会话网络重试 0 次。
