import numpy as np
import os
import tqdm
from src.Map.Map import load_map

# 阴影图用于表示Agent出现在某个点时的视野范围，出去被障碍物遮挡的部分
# 函数实现了 Bresenham 直线算法，用于在二维网格中找到从 (x0, y0) 到 (x1, y1) 的直线路径，
# 并在路径上标记是否被障碍物阻挡。
def bresenham(x0, y0, x1, y1, obstacles, shadow_map):
    x_dist = abs(x0 - x1)
    y_dist = -abs(y0 - y1)
    x_step = 1 if x1 > x0 else -1
    y_step = 1 if y1 > y0 else -1

    error = x_dist + y_dist

    # shadowed = False
    shadow_map[y0, x0] = False

    while x0 != x1 or y0 != y1:
        # 根据误差决定是进行水平步进还是垂直步进。
        if 2 * error - y_dist > x_dist - 2 * error:
            # horizontal step
            error += y_dist
            x0 += x_step
        else:
            # vertical step
            error += x_dist
            y0 += y_step

        if obstacles[y0, x0]:
            # shadowed = True
            return

        # if shadowed:
        shadow_map[y0, x0] = False


# 函数用于计算整个地图的阴影图。
def calculate_shadowing(map_path, save_as):
    print("Calculating shadowing maps")
    total_map = load_map(map_path)
    obstacles = total_map.obstacles
    size = total_map.obstacles.shape[0]
    total = size * size

    total_shadow_map = np.ones((size, size, size, size), dtype=bool)
    #  使用 tqdm 包来显示进度条，表示计算每个点的阴影信息的进度
    with tqdm.tqdm(total=total) as pbar:
        # 遍历地图中所有的位置 (i, j)，对每个位置计算它的阴影图
        for i, j in np.ndindex(total_map.obstacles.shape):
            # 创建一个布尔数组 shadow_map，初始时表示所有位置都可以看到（True）。
            # 随后会更新这个数组，根据障碍物的情况修改视野信息。
            shadow_map = np.ones((size, size), dtype=bool)

            for x in range(size):
                bresenham(i, j, x, 0, obstacles, shadow_map)
                bresenham(i, j, x, size - 1, obstacles, shadow_map)
                bresenham(i, j, 0, x, obstacles, shadow_map)
                bresenham(i, j, size - 1, x, obstacles, shadow_map)

            total_shadow_map[j, i] = shadow_map
            pbar.update(1)

    np.save(save_as, total_shadow_map)
    return total_shadow_map


# 检查阴影图文件是否已存在，如果存在则加载，
# 若不存在则调用 calculate_shadowing 函数生成新的阴影图。
def load_or_create_shadowing(map_path):
    shadow_file_name = os.path.splitext(map_path)[0] + "_shadowing.npy"
    if os.path.exists(shadow_file_name):
        return np.load(shadow_file_name)
    else:
        return calculate_shadowing(map_path, shadow_file_name)
