# w/o A* 推理期「合法动作掩码 + 低温采样」实验

## 1. 改动范围

- **新增** `runs/ch3_paper_env/run_mask_eval.py`：独立评测入口。
- **未修改**：`test.py`、`train.py`、`src/**`、任何原有权重
  `path_to_save_model_1/shared_mappo_model_21_9232.pth`。
- 训练逻辑完全不动；只在推理选动作阶段替换采样方式。

### 1.1 动作索引（已核对）

`src/Map/timemap.py`：

```python
action_list = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)
```

即 `0=上(-1,0)`、`1=下(+1,0)`、`2=左(0,-1)`、`3=右(0,+1)`。

### 1.2 合法动作掩码

对每个智能体，用已知障碍通道 `two_channelmap[agent]["obstacle_map"]`
（`1=障碍`、`0=已知空地`、`-1=未知`）与地图边界构造 4 维 mask：

```python
def valid_mask(agents_pos, agent_index, obstacle_map):
    x, y = agents_pos[agent_index]
    h, w = obstacle_map.shape
    mask = np.zeros(4, dtype=np.float32)
    for i, (dx, dy) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        nx, ny = int(x) + dx, int(y) + dy
        if 0 <= nx < h and 0 <= ny < w and obstacle_map[nx, ny] != 1:
            mask[i] = 1.0
    return mask
```

- 只屏蔽**已知**障碍与越界；未知格视为可走（避免过度保守）。
- 若 4 个方向全被挡，回退到原分布，避免 NaN。

### 1.3 温度采样

```python
probs = actor_probs(...)                 # 与原 Actor 输出一致
probs = probs * mask                     # 掩码
q = probs.clamp_min(1e-12) ** (1.0 / T)  # 温度缩放
q = q / q.sum()
action = torch.multinomial(q, 1)
```

- `T <= 0.05` 退化为 `argmax`；`T = 1.0` 保持原分布。
- 新增 CLI：`--action-temperature`（默认 0.5）、`--no-action-mask`；
  另保留 `--no-rescue/--rescue`，默认关闭 A*。

## 2. 正确性校验

1. **前向一致性**：同一观测下，新脚本的 `actor_probs` 与
   `policy.agents.get_action(...)[1]` 逐元素 `allclose(atol=1e-6)`。
2. **局部信息素一致性**：自实现的 11×11 窗口与 `test.get_local_ph` 完全相等。
3. **掩码方向正确**：示例中 agent_1 上方为已知障碍，mask 为 `[0,1,1,1]`。
4. **控制臂**：新采样器 + 无掩码 + T=1.0 得 SR 48.4%，与原基线 52.0%
   相差 3.6 pp。原因是原 `get_action` 内部会先做一次被丢弃的 `multinomial`
   抽样，新采样器不再消耗这次 RNG；两者动作分布相同、随机流不同。
   因此消融以**控制臂**为参照更严谨。

## 3. 结果（250 张 `DungeonMaps/test`，同 seed 协议，A* 关闭）

| Arm | Mask | T | SR | AS | ANTO | Coverage |
|---|---|---|---|---|---|---|
| 原始基线（原 test.select_action） | 否 | 1.0 | 52.0% | 186.60 | 5.060 | 0.7262 |
| 控制臂（新采样器） | 否 | 1.0 | 48.4% | 194.20 | 5.080 | 0.7280 |
| 仅温度 | 否 | 0.5 | 50.8% | 189.87 | 5.076 | 0.7152 |
| 仅掩码 | 是 | 1.0 | 62.8% | 170.68 | 5.436 | 0.7634 |
| 掩码 + 温度 | 是 | 0.3 | 65.2% | 164.50 | 5.392 | 0.7637 |
| **掩码 + 温度（推荐）** | **是** | **0.5** | **68.0%** | **162.06** | **5.452** | **0.7669** |

相对控制臂的同 seed 配对（McNemar 精确检验）：

| Arm | SR 变化 | 仅改善/仅退化 | p | ΔAS | ΔANTO |
|---|---|---|---|---|---|
| 仅温度 T=0.5 | +2.4 pp | 46 / 40 | 0.59 | −4.3 | −0.004 |
| 仅掩码 T=1.0 | +14.4 pp | 60 / 24 | 1.07e-4 | −23.5 | +0.356 |
| 掩码 + T=0.3 | +16.8 pp | 68 / 26 | 1.73e-5 | −29.7 | +0.312 |
| **掩码 + T=0.5** | **+19.6 pp** | **76 / 27** | **1.45e-6** | **−32.1** | **+0.372** |

相对原始基线：掩码 + T=0.5 为 +16.0 pp（69 / 29，p=6.57e-5），
ΔAS −24.5，ΔANTO +0.392。

## 4. 可复现命令

工作目录：`D:\code\修改6--在修改5的基础上，所有网络使用相同的参数`

```powershell
$env:PAPER_TORCH_THREADS='4'; $env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'

# 推荐配置：w/o A* + 合法动作掩码 + T=0.5
D:\env\python.exe runs/ch3_paper_env/run_mask_eval.py --episodes 250 `
  --action-temperature 0.5 --out runs/ch3_paper_env/mask/mask_t0.5.json

# 控制臂：同采样器、无掩码、T=1.0
D:\env\python.exe runs/ch3_paper_env/run_mask_eval.py --episodes 250 `
  --action-temperature 1.0 --no-action-mask --out runs/ch3_paper_env/mask/raw_t1.0.json

# 仅掩码
D:\env\python.exe runs/ch3_paper_env/run_mask_eval.py --episodes 250 `
  --action-temperature 1.0 --out runs/ch3_paper_env/mask/mask_t1.0.json

# 仅温度
D:\env\python.exe runs/ch3_paper_env/run_mask_eval.py --episodes 250 `
  --action-temperature 0.5 --no-action-mask --out runs/ch3_paper_env/mask/notmask_t0.5.json
```

每臂 250 场景约 7.5–8.7 分钟（CPU，4 线程）。

## 5. 结论与局限

- 增益主要来自**合法动作掩码**（相对控制臂 +14.4 pp，p=1.1e-4）；
  低温采样在其上再贡献约 +5 pp，合计 SR 68.0%（+19.6 pp，p=1.45e-6）。
- 同时 AS 从 194.2 降到 162.1，ANTO 从 5.08 升到 5.45，覆盖率从 0.728 升到 0.767。
- T=0.3 略低于 T=0.5（65.2% vs 68.0%），说明过度低温会损失探索；0.5 是当前最优。
- 掩码只使用**已知**障碍；未知区域不屏蔽，因此不会把“可能可走”的格子误判为非法。
- 所有对比均为 250 张固定地图、`seed=2000000000+episode`、A* 关闭；未改训练与权重。
- 该增益是否会与 A* 叠加尚未测试（本实验针对 w/o A* 场景）。
