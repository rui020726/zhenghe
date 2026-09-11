from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List


class RevisitCoverageMeter:
    """
    复访率与覆盖指标采集器（感知覆盖 + 多重计数口径）
    - 访问判定：本步内若格点落入任一体的观测半径 v 且（可选）视线无遮挡，则该格被“访问”。
    - 计数口径：多重计数。同一步同一格被多名体覆盖，visits 计多次；仅首次将该格从“未访问”变为“已访问”的一次计入 u，其余计入 revisits。
    - 仅统计自由格（free_mask=True 的格）。
    """
    def __init__(
        self,
        free_mask: np.ndarray,
        *,
        radius: int = 5,
        window: int = 100,
        use_los: bool = True,
    ):
        """
        free_mask: (H,W) 布尔矩阵，自由格为 True
        radius: 观测半径（L2圆盘）
        window: 窗口复访率的窗口长度
        use_los: 是否启用视线遮挡判定（调用 env._is_line_of_sight_blocked）
        """
        self.free_mask = free_mask.astype(bool)
        self.H, self.W = self.free_mask.shape
        self.radius = int(radius)
        self.window = int(window)
        self.use_los = bool(use_los)

        self.visited = np.zeros_like(self.free_mask, dtype=bool)  # 唯一覆盖集合
        self.F = int(self.free_mask.sum()) if int(self.free_mask.sum()) > 0 else 1

        # 时间序列指标
        self.U: List[float] = []
        self.u: List[int] = []
        self.visits: List[int] = []
        self.revisits: List[int] = []
        self.r_inst: List[float] = []
        self.r_cum: List[float] = []
        self.r_win: List[float] = []
        self.min_dist: List[float] = []   # 每步最小对间距（安全代理）
        self.energy: List[float] = []     # 每步“能量”（位移范数平方的和）

        # 时鲜度（未刷新时长）统计
        self.staleness = np.zeros_like(self.free_mask, dtype=np.int32)

        # 上一步位姿（用于能量计算）
        self._prev_agents_pos: Optional[np.ndarray] = None

    @staticmethod
    def _disk_indices(i: int, j: int, v: int, H: int, W: int):
        """
        返回以(i,j)为中心、半径v（L2圆盘）内的所有栅格坐标（裁剪到地图内）。
        修复说明：使用 meshgrid 生成坐标，再用布尔掩模筛选，避免 ogrid 布尔索引形状不匹配问题。
        """
        di = np.arange(-v, v + 1, dtype=int)
        dj = np.arange(-v, v + 1, dtype=int)
        II, JJ = np.meshgrid(di, dj, indexing="ij")  # (2v+1, 2v+1)
        mask = (II * II + JJ * JJ) <= v * v
        I = (i + II[mask]).astype(int)
        J = (j + JJ[mask]).astype(int)
        valid = (I >= 0) & (I < H) & (J >= 0) & (J < W)
        return I[valid], J[valid]

    def _pairwise_min_dist(self, agents_pos: np.ndarray) -> float:
        if agents_pos.shape[0] < 2:
            return float("inf")
        mind = float("inf")
        for a in range(agents_pos.shape[0]):
            for b in range(a + 1, agents_pos.shape[0]):
                d = np.linalg.norm(agents_pos[a] - agents_pos[b], ord=2)
                if d < mind:
                    mind = d
        return mind

    def update(self, agents_pos: np.ndarray, env) -> None:
        """
        在每一步环境更新后调用。
        agents_pos: (N,2) int，当前各体格坐标
        env: 需提供 env.map (障碍=1)，以及 env._is_line_of_sight_blocked(x1,y1,x2,y2, env.map) 方法
             注意：env._is_line_of_sight_blocked 返回 True 表示可见（未被阻挡），False 表示被阻挡。
        """
        # 安全与能量代理
        self.min_dist.append(self._pairwise_min_dist(agents_pos))
        if self._prev_agents_pos is None:
            self.energy.append(0.0)
        else:
            disp = agents_pos - self._prev_agents_pos  # (N,2)
            step_energy = float((disp ** 2).sum())     # 所有体位移范数平方之和
            self.energy.append(step_energy)
        self._prev_agents_pos = agents_pos.copy()

        # 生成当步覆盖“多重集合”
        covered_cells: List[Tuple[int, int]] = []
        for (ai, aj) in agents_pos:
            I, J = self._disk_indices(int(ai), int(aj), self.radius, self.H, self.W)
            if self.use_los:
                for i, j in zip(I, J):
                    # 仅统计自由格；障碍格跳过
                    if not self.free_mask[i, j]:
                        continue
                    if env._is_line_of_sight_blocked(int(ai), int(aj), int(i), int(j), env.map):
                        covered_cells.append((int(i), int(j)))
            else:
                for i, j in zip(I, J):
                    if self.free_mask[i, j]:
                        covered_cells.append((int(i), int(j)))

        # 当步访问次数（多重计数）
        vcount = len(covered_cells)
        self.visits.append(vcount)

        # 用于唯一覆盖的集合：对“格”去重
        ucount = 0
        if vcount > 0:
            uniq = np.array(list(set(covered_cells)), dtype=int)
            newly = ~self.visited[uniq[:, 0], uniq[:, 1]]
            ucount = int(newly.sum())
            # 更新唯一覆盖集合
            self.visited[uniq[:, 0], uniq[:, 1]] = True

            # 更新时鲜度：覆盖到的自由格清零，其余自由格+1
            self.staleness[self.free_mask] += 1
            self.staleness[uniq[:, 0], uniq[:, 1]] = 0
        else:
            # 即便没有覆盖，也要推进时鲜度（自由格 +1）
            self.staleness[self.free_mask] += 1

        self.u.append(ucount)
        rv = vcount - ucount
        self.revisits.append(rv)

        # 覆盖曲线
        U_t = float(self.visited.sum()) / float(self.F)
        self.U.append(U_t)

        # 复访率
        r_inst = rv / max(1, vcount)
        self.r_inst.append(r_inst)

        tot_v = max(1, np.sum(self.visits))
        tot_r = np.sum(self.revisits)
        self.r_cum.append(float(tot_r) / float(tot_v))

        t = len(self.visits)
        t0 = max(0, t - self.window)
        wv = np.sum(self.visits[t0:t])
        wr = np.sum(self.revisits[t0:t])
        self.r_win.append(float(wr) / float(max(1, wv)))

    def t_at_coverage(self, thr: float = 0.9) -> Optional[int]:
        U = np.asarray(self.U, dtype=float)
        idx = np.where(U >= thr)[0]
        return int(idx[0]) if idx.size > 0 else None

    def summarize(self) -> Dict[str, float]:
        U = np.asarray(self.U, dtype=float)
        r_cum = np.asarray(self.r_cum, dtype=float)
        r_win = np.asarray(self.r_win, dtype=float)
        energy = np.asarray(self.energy, dtype=float)
        min_dist = np.asarray(self.min_dist, dtype=float)

        return dict(
            U_T=float(U[-1]) if U.size > 0 else 0.0,
            t90=self.t_at_coverage(0.9),
            r_cum_final=float(r_cum[-1]) if r_cum.size > 0 else 0.0,
            r_win_mean=float(r_win.mean()) if r_win.size > 0 else 0.0,
            energy_mean=float(energy.mean()) if energy.size > 0 else 0.0,
            min_dist_median=float(np.median(min_dist)) if min_dist.size > 0 else float("inf"),
            staleness_max=int(self.staleness[self.free_mask].max()) if self.F > 0 else 0,
        )

    def save_plots(self, out_dir: str, prefix: str = "ep") -> None:
        os.makedirs(out_dir, exist_ok=True)

        # 覆盖 U(t)
        plt.figure(figsize=(6, 4))
        plt.plot(self.U, label="U(t)")
        t90 = self.t_at_coverage(0.9)
        if t90 is not None:
            plt.axvline(t90, color="r", ls="--", label=f"t90={t90}")
        plt.ylim(0, 1)
        plt.xlabel("t")
        plt.ylabel("U")
        plt.title("唯一覆盖 U(t)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_coverage_U.png"))
        plt.close()

        # 复访率（累计 + 窗口）
        plt.figure(figsize=(6, 4))
        plt.plot(self.r_cum, label="累计复访率 r_cum")
        plt.plot(self.r_win, label=f"窗口复访率 r_win (w={self.window})")
        plt.ylim(0, 1)
        plt.xlabel("t")
        plt.ylabel("rate")
        plt.title("复访率曲线")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_revisit_rates.png"))
        plt.close()

        # 安全与能量
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(self.min_dist)
        ax[0].set_title("最小对间距（安全代理）")
        ax[0].set_xlabel("t")
        ax[0].set_ylabel("min distance")

        ax[1].plot(self.energy)
        ax[1].set_title("控制能量（步位移平方和）")
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("energy")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_safety_energy.png"))
        plt.close()

        # 时鲜度分布（终值）
        stale_vals = self.staleness[self.free_mask]
        if stale_vals.size > 0:
            plt.figure(figsize=(6, 4))
            plt.hist(stale_vals.flatten(), bins=30, color="tab:orange")
            plt.title("终局时鲜度（未刷新时长）分布")
            plt.xlabel("steps since last observed")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}_staleness_hist.png"))
            plt.close()

    def save_summary_csv(self, out_csv: str, ep_idx: int) -> None:
        summ = self.summarize()
        header_needed = not os.path.exists(out_csv)
        import csv
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header_needed:
                w.writerow(
                    [
                        "episode",
                        "U_T",
                        "t90",
                        "r_cum_final",
                        "r_win_mean",
                        "energy_mean",
                        "min_dist_median",
                        "staleness_max",
                    ]
                )
            w.writerow(
                [
                    ep_idx,
                    summ["U_T"],
                    summ["t90"],
                    summ["r_cum_final"],
                    summ["r_win_mean"],
                    summ["energy_mean"],
                    summ["min_dist_median"],
                    summ["staleness_max"],
                ]
            )