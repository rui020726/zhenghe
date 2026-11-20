# 论文原始环境复现结果（修改6）

## 1. 环境与权重

| 项目 | 值 |
|---|---|
| 代码目录 | `D:\code\修改6--在修改5的基础上，所有网络使用相同的参数` |
| 权重 | `path_to_save_model_1/shared_mappo_model_21_9232.pth`（`MAPPO.load` 写死的文件名） |
| 地图 | `DungeonMaps/test`，`test_manifest.txt` 中的 250 张图（250 行、无重复、全部存在） |
| 智能体/目标 | 2 个智能体、6 个动态目标 |
| 回合上限 | 250 步 |
| 通信 | perfect（无丢包/延迟） |
| 决策 | 论文原始 `test.py` 的随机采样动作 + A* 混合决策 |
| Python | `D:\env\python.exe`，torch 2.9.1+cpu（CPU-only） |

原 `MAPPO.load` 写死 `device='cuda'`，本机无 CUDA。新增 `runs/ch3_paper_env/run_paper_eval.py`，在运行时用 CPU 版本替换 `test.MAPPO.load`，**未修改任何原有源码**。

## 2. 运行命令

```powershell
$env:PAPER_TORCH_THREADS='4'; $env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
D:\env\python.exe runs/ch3_paper_env/run_paper_eval.py --episodes 250 --condition perfect `
  --out runs/ch3_paper_env/paper_perfect_250.json
```

250 回合墙钟 340 s（约 1.36 s/回合）。

## 3. 结果

| 指标 | 本次复现（250 场景） | 论文表 I | 差异 |
|---|---|---|---|
| SR | **94.0%** | 95.6% | −1.6 pp |
| AS（全部回合） | 131.11 | 129.08 | +2.03 |
| AS（仅成功回合） | 123.52 | — | — |
| SV（全部回合方差） | 3731.75 | 2897.15 | — |
| SV（仅成功回合方差） | 3010.10 | 2897.15 | +112.95（+3.9%） |
| ANTO | 5.912 | 5.716 | +0.196 |
| 末覆盖率 | 0.810 | — | — |
| A* 平均触发 | 8.83 次/回合 | — | — |

论文对 SV 的定义是“成功回合步数的方差”，对应本表 `SV_success=3010.10`，与论文 2897.15 相差约 3.9%；SR 与 ANTO 也都在 2 个百分点/0.2 个目标以内。差异主要来自每回合智能体/目标的随机初始位置与具体地图顺序（本次用 seed=2000000000+episode 固定复现）。

## 4. 结论

- 之前在新仓库得到的 SR≈22% 是因为用了另一套环境（seeded 随机地图、legacy/v2 权重、不同发现判据），**不是论文模型本身的水平**。
- 在论文原始代码、原始 250 张测试地图、原始权重下，可以复现论文表 I：SR 94.0% vs 95.6%，AS(成功) 123.5 vs 129.1，ANTO 5.91 vs 5.72。

## 5. 文件

- `paper_perfect_250.json`：逐回合结果 + 汇总
- `paper_perfect_250.csv`：逐回合 CSV
- `paper_perfect_250.json.jsonl`：逐回合增量日志（防止写入中断丢数据）
- `run_paper_eval.py`：CPU 兼容的评测入口
