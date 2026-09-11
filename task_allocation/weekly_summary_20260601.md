# 周报：CTDE 异构 MRTA 系统优化（2026.05.28 — 06.01）

## 一、背景与目标

旧系统（ckpt 5/25）使用 priority-based reward，存在 deadline 成功率低（Small 19% / Medium 35% / Large 56%）、过期率高（75%/59%/39%）、agent 规模不合理（7/14/22，无梯度）等问题。本周目标是优化 reward 设计、调整实验规模、提升 DL-SR。

## 二、完成的修改

### 1. Reward 重构（old_task_env.py）
- 过期任务：-0.5 惩罚 → +0.05×decay 引导式激励
- 按时/过期梯度放大：过期完成系数 0.2 → 0.1（10× 差距）
- Episode reward 系数翻倍：A 300→600，lam 80→120，omega 200→400

### 2. Agent 规模（方案B）
- 测试规模：Small (2,3)×(3,5)→(3,4)×(4,5)，avg 7→16；Medium 14→21；Large 不变
- 训练规模：per_species (2,5)→(4,6)，species (3,4)→(3,5)，avg 10.5→18

### 3. Deadline 放宽
- Curriculum Stage 4/5 下限：0.95 → 1.0
- 测试 DL_TIGHTNESS：(1.0,1.5) → (1.2,1.8)

### 4. 工程优化
- Gossip 通信降频参数化（硬编码 → gossip_interval 可配置）
- env.step() ThreadPoolExecutor 并行化（8 envs）
- 断点续训（RESUME + 自动查找最新 checkpoint）
- 训练周期 20000 → 15000，LR 衰减 10000→15000

## 三、关键发现：Reward 改动导致严重退化

### 训练结果（ckpt 5/28，15000 轮）

| | Small | Medium | Large |
|:--|:-----:|:-----:|:-----:|
| DL-SR | 10-12% | 13-15% | 14-17% |
| Expired | 82-84% | 79-82% | 77-81% |

所有指标比旧模型（ckpt 5/25）全面退化且失去规模梯度。

### 对照实验（06.01）：隔离变量定位根因

用旧 ckpt 5/25 在新测试条件下（方案B 规模 + deadline 1.2-1.8）评测：

| | Small | Medium | Large |
|:--|:-----:|:-----:|:-----:|
| DL-SR | **39.1%** | **59.2%** | **67.8%** |
| Expired | 54.8% | 34.6% | 26.1% |
| Reward | +74 | +277 | +364 |

### 诊断结论

- **方案B + deadline 放宽有效**：旧模型仅换测试条件，DL-SR 提升至 Small 39%/Medium 59%/Large 68%
- **Reward 改动是唯一退化原因**：相同测试条件，ckpt 5/28 比 ckpt 5/25 DL-SR 低 27-51 个百分点
- **退化机制**：过期任务 incentive（+0.05）替代 penalty（-0.5）→ agent 不再区分任务紧急程度 → 利用率翻倍但选择质量暴跌 → 全部过期

## 四、后续计划

1. 回退 reward 至 ckpt 5/25 版本
2. 保留方案B、deadline 放宽、gossip 参数化等非 reward 优化
3. 从 ckpt 5/25 热启动，仅增量训练 5000 轮
4. 验证 DL-SR 能否在 39%/59%/68% 基础上进一步改善
