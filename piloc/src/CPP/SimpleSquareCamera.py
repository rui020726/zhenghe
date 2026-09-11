import os
import numpy as np

from src.Map.Shadowing import load_or_create_shadowing
from src.Map.Map import load_map


class SimpleSquareCameraParams:
    def __init__(self):
        self.half_length = 2    # 摄像头视野的半边长度
        self.map_path = "res/downtown.png"


class SimpleSquareCamera:

    def __init__(self, params: SimpleSquareCameraParams):
        self.params = params
        total_map = load_map(self.params.map_path)
        self.obstacles = total_map.obstacles
        self.size = self.obstacles.shape[:2]    # 获取障碍物地图的大小
        self.obstruction_map = load_or_create_shadowing(self.params.map_path)   # 加载或创建一个遮挡地图

    # 计算并返回摄像头视野中的可视区域，考虑障碍物和遮挡。
    def computeView(self, position, attitude):
        view = np.zeros(self.size, dtype=bool)
        camera_width = self.params.half_length
        x_pos, y_pos = position[0], position[1]
        x_size, y_size = self.size[0], self.size[1]
        view[max(0, y_pos - camera_width):min(y_size, y_pos + camera_width + 1),
             max(0, x_pos - camera_width):min(x_size, x_pos + camera_width + 1)] = True  # 确保视野不越界

        view &= ~self.obstacles  # 从视野中移除障碍物
        view &= ~self.obstruction_map[y_pos, x_pos]  # 将当前位置的遮挡区域从视野中移除
        return view
