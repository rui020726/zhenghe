# ROUND63：Early 首报的匹配可行性审计

## 结论

**焦点没有跨场景共同支持，当前材料不足以把 ready 时机与 deadline slack 的关联分开。** R60 seed33 greedy 的焦点 `1700027/piloc-e1-target-5` 为 ready=134、backlog=1。146个合格 early 候选中，只有焦点自身落在精确 `(ready=134, backlog=1)` 组；其他场景精确支持数为0。不能放宽匹配条件凑样本，也不能把不同场景间的 slack 差解释成因果效应。

本轮只审计已有输入和已完成的 R61 干预原件；没有新环境、样本、模型推理或干预。

## 输入与提取

读取已核验 R60 seed33 greedy 的32个场景 tape 及其配对生命周期原件。每个 tape 的 SHA256、每个候选的 tape/lifecycle 路径和 SHA256 均在 [MATCHED_PANEL.json](MATCHED_PANEL.json) 中。模型身份沿用 R62 已核验的 R60 seed33 ep128 greedy 文件 SHA256 `573e4c26a4fee6f01405d4386ca73b518eeb1adb9cd8409bc83661e4113f6c5a`；本轮没有加载模型。

筛选规则为已有的自由首报、ready≤147、first-emission、真实顺序预算下可付、初始发送可执行且阻断原因 `allowed`。ready 按冻结公式 `round(f[18]×1000)-round(f[0])` 重算并核对决策步，完整保留19维。共146个唯一 `(scene,target)` 候选，无重复项、无缺失 deadline 原件，且146个 baseline 动作全部为 SKIP。

deadline slack 从同场景生命周期 detection timestamp 与 `deadline_offset` 推得，公式为 `deadline - (decision_step - 1)`；缺失值保持 null，不填0。R61 原始选择距离只使用特征列13、14、15、18按候选范围缩放的平方差和，恒定列贡献0；它不是完整19维欧氏距离。

## 支持与近邻

146例形成66个精确 `(ready_step, backlog)` 组，其中20组至少跨两个场景且组内 remaining time 有变化。再加入 future-requirements 和 budget-fraction 的精确匹配后为80组，其中23组跨场景。它们说明其他 covariate strata 中存在描述性 slack 变异，但不提供焦点的共同支持，也没有这些组的配对干预标签。

按冻结的协变量词典序，焦点最近的异场景候选为 `1700012/piloc-e1-target-1`：ready=87（差47步）、backlog=3（差2）；这不是匹配控制。焦点的 exact group 只有自身，故不报告任何估计的时机或 slack 效应。

### R61 原7个控制：相对焦点差值

| Scene / target | ready | |Δ ready| | |Δ backlog| | Δ remaining time | R61 scaled distance | target Δ | cost Δ (B) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1700015 / target-0 | 75 | 59 | 0 | -0.817 | 0.196789 | 0 | +2496 |
| 1700012 / target-4 | 79 | 55 | 1 | +59.651 | 0.211010 | 0 | +2496 |
| 1700000 / target-2 | 41 | 93 | 0 | +18.453 | 0.488948 | 0 | +2688 |
| 1700009 / target-0 | 29 | 105 | 1 | -7.316 | 0.663269 | 0 | +2496 |
| 1700016 / target-0 | 17 | 117 | 1 | +5.633 | 0.813871 | 0 | +2682 |
| 1700018 / target-3 | 14 | 120 | 1 | -20.758 | 0.854065 | 0 | +378 |
| 1700005 / target-3 | 11 | 123 | 1 | -30.918 | 0.895277 | 0 | +2484 |

Δ remaining is control minus focus (focus slack 98.850). All seven controls had target and whole-system task delta 0 under R61 SEND versus SKIP. These outcomes are linked from R61 raw paired data after ranking; they do not turn these cases into matched controls. No outcomes are assigned to the other 138 candidates.

For focus `1700027/target-5`, the R61 paired result is target delta +1 and cost delta +2688 B (SEND rescues the target in that realization). This is an observed case effect, not an estimate of a general early-report policy effect.

## Interpretation limits

The covariate ranking is outcome-blind, and a pure-computation test mutates outcome fields without changing the ranking. Still, the focus has no exact cross-scene ready/backlog support. Twenty multi-scene groups with varying slack are not interventions and cannot isolate slack from scene differences. R61's seven controls were selected by the four-column scaled distance, not by ready/backlog matching; nearest in that metric is not synonymous with common support.

These eight historical, post-hoc cases from one frozen model and one random realization cannot establish message safety, general necessity, separability of timing and slack, or sufficiency of the 19D observation. No classifier, threshold, or new rule was fit.

## Verification and provenance

- Analysis script: [round63_analysis.py](round63_analysis.py)
- Narrow tests: [test_round63_analysis.py](test_round63_analysis.py), 5 passed, exit 0.
- Machine-readable panel: [MATCHED_PANEL.json](MATCHED_PANEL.json)
- Machine verdict: [ROUND63_VERDICT.json](ROUND63_VERDICT.json)
- Execution basis: [ROUND63_EXECUTION_BASIS.md](ROUND63_EXECUTION_BASIS.md)
- Starting/source HEAD: `62dcb2419f323461b6aa07bb79ce39e3d17a4ac7`.
- R61 raw effect source SHA256: `4b303efa39563a92c5bb8fab8f74ccd84e6bec7e0eab26b09f1745c9d5446bbd`.
- Protected ROUND46 hashes were rechecked and unchanged. The pre-existing untracked R59 log was not modified or staged.

ROUND63 side-effect counts: environment builds/runs/steps 0; claims/slots 0; model loads/forwards 0; RNG restores 0; optimizer/PPO updates 0; checkpoint writes 0; new evaluations 0. Only read-only parsing, deterministic arithmetic, five synthetic unit tests, and local ROUND63 report artifact writes occurred. No push, merge, deployment, or follow-on experiment.
