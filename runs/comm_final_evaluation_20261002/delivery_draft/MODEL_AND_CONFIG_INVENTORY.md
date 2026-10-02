# 模型、配置与版本清单

本文件由 `r37_delivery.py` 直接从冻结件与磁盘重算生成，不从报告正文抄数字。

## 1. 被评估的三个模型（全部评估、全部报告）

| 训练 seed | checkpoint（Git 外） | 字节 SHA256 | whole-controller 参数摘要 | episodes / 更新批 | λ 终值 | 版本字段 |
|---|---|---|---|---|---|---|
| 11 | `runs/comm_final_train_20261001/seed11/checkpoints/v2_train_s11_ep128.pt` | `032c2b77f60f1519…` | `553d02dd1de285a7…` | 128 / 32 | 1.5000 | 两处齐全 |
| 22 | `runs/comm_final_train_20261001/seed22/checkpoints/v2_train_s22_ep128.pt` | `de47f3c90789a8bd…` | `6a84ab76e206a511…` | 128 / 32 | 1.2625 | 两处齐全 |
| 33 | `runs/comm_final_train_20261001/seed33/checkpoints/v2_train_s33_ep128.pt` | `98bb6e74e25fa7de…` | `cd0498693f7636ae…` | 128 / 32 | 1.4750 | 两处齐全 |

- 装载方式：`load_checkpoint_read_only`，`load_mode=read_only、optimizer_restored=False、rng_restored=False`（不建 optimizer、不动 RNG）
- 采集语义标签：`versioned_capture_semantics`（缺版本字段的旧件会标 `legacy_unversioned_capture_semantics` 并禁止续训）
- 中间里程碑 ep64 与 rolling 文件**未参与评估**，只作审计存在

## 2. 版本与代码身份

| 项 | 值 |
|---|---|
| training_version | `constrained-cost-v2/published-completion-event-at-relay-tick/2` |
| reward_timing_version | `published-completion-event-at-relay-tick/2` |
| trainer_tag | `constrained-cost-v2` |
| loss_version | `constrained-lagrangian/2` |
| state_schema | `constrained-cost-state/2` |
| gae_schema | `per-decision-next-value/1` |
| optimizer_schema | `stage-c-four-module-parameter-set/1` |
| reward_timing_schema | `post-action-most-recent-decision/1` |
| 评估分支 / 头 | `eval/comm-final-fixed-evaluation` @ `e2a272c0303f580d957e778f7e222a56d265372f` |
| 生产件（integrated_sar） | 8 个哈希登记 |
| Stage C harness | 9 个哈希登记 |

## 3. 搜索 / 分配模型哈希（冻结，不改）

- `runs/cur_v2rep05_s2/best/shared_mappo_model_21_3100.pth` — `d57aed25b3d46dffb2bd269ea87140953828ba7f51a960bb01995ec054dfca33`
- `runs/joint_eval_20260916/ctde_final_7_1.pth` — `c7c256da0fc20451c1104996ce494148f567e4f83acf265d9baae8f0f053500d`

## 4. 运行环境与完整配置快照

| 字段 | 值 |
|---|---|
| interpreter | `D:\project_env\python.exe` |
| python | `3.10.19` |
| torch | `2.9.0+cu128` |
| numpy | `2.2.6` |
| steps_per_episode | `1000` |
| budget_bytes | `29300` |
| b0_bytes | `10000.0` |
| delta | `0.0` |
| delivery_probability | `0.7` |
| ack_enabled | `False` |
| replay_or_online | `每个 cell 独立在线运行一集（1000 步），不做回放；与训练块同一运行方式` |

- 继承的世界字段共 48 项，全部登记在 `EVALUATION_IDENTITY.json`
- 传输覆盖字段：`{"comm_ack": false, "comm_budget_bytes": 29300, "comm_delay_ticks": 0, "comm_delivery_probability": 0.7, "comm_max_wait_steps": null, "comm_range": 0.3, "comm_sends_per_tick": null, "comm_subscribers": "all", "comm_timing_mode": "policy_owned_timing", "comm_urgent_wait_steps": null, "decision_trace": null, "device": "cpu", "gossip_interval": 2, "info_probe": null, "mock": false, "num_goals": 6, "quiet": true, "record_stream": null, "replay_stream": null, "search_probe": null, "steps": 1000}`

## 5. 场景块与九臂

- 场景：1600000..1600063（64 个），占用依据 `SCENE_USAGE_AUDIT.json`（不看结局选块）
- 臂：`model_s11_greedy, model_s11_sampled, model_s22_greedy, model_s22_sampled, model_s33_greedy, model_s33_sampled, immediate, late_discovery_gate, always_hold`
- 额度：正常 576 / 录制储备 64（判定不需要）/ 失败储备 32 / 硬上限 672

## 6. 产物位置（Git 外，逐件哈希在本轮登记表内）

- `checkpoints/*.pt`：0 件
- `claims/*.claim`：576 件
- `logs/*.jsonl`：1536 件
- `rows/*.json`：576 件
- `shards/*.json`：5 件
- 合计 2693 件、580,222,303 字节

## 7. 门禁与对账状态

- 技术门禁：`all_passed=True`，核对项 7 项，计入正常评估 9 次
- 账本四方闭合：`all_passed=True`，行 576 / claim 576 / 账本完成 575 / 真实 ledger 576
- 技术失败 0 次、复用行 0 次、独立流录制 0 次
- 统计矩阵完整性：True（cells=576）
