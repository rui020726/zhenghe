import numpy as np

from src.Map.Map import Map
from src.StateUtils import pad_centered
from src.base.BaseState import BaseState


class CPPScenario:
    def __init__(self):
        self.target_path = ""
        self.position_idx = 0
        self.movement_budget = 100


class CPPState(BaseState):
    def __init__(self, map_init: Map):
        super().__init__(map_init)
        self.target = None
        self.position = None
        self.movement_budget = None
        self.landed = False
        self.terminal = False

        self.initial_movement_budget = None
        self.initial_target_cell_count = 0
        self.coverage = None

    def reset_target(self, target):
        self.target = target
        self.initial_target_cell_count = np.sum(target)
        self.coverage = np.zeros(self.target.shape, dtype=bool)

    def get_remaining_cells(self):   # 计算并返回当前目标区域中剩余的单元格数
        return np.sum(self.target)

    def get_total_cells(self):
        return self.initial_target_cell_count

    # 计算当前覆盖率的比例
    def get_coverage_ratio(self):
        return 1.0 - float(np.sum(self.get_remaining_cells())) / float(self.initial_target_cell_count)

    def get_scalars(self):
        return np.array([self.movement_budget])

    def get_num_scalars(self):
        return 1

    # 返回一个布尔类型的地图，通常用于表示不同区域的状态，如禁飞区、障碍物、着陆区等。
    def get_boolean_map(self):
        padded_red = pad_centered(self, np.concatenate([np.expand_dims(self.no_fly_zone, -1),
                                                        np.expand_dims(self.obstacles, -1)], axis=-1), 1)
        padded_rest = pad_centered(self, np.concatenate([np.expand_dims(self.landing_zone, -1),
                                                         np.expand_dims(self.target, -1)], axis=-1), 0)
        return np.concatenate([padded_red, padded_rest], axis=-1)

    def get_boolean_map_shape(self):
        return self.get_boolean_map().shape

    # 返回一个与布尔地图形状相同的浮动类型地图
    def get_float_map(self):
        shape = list(self.get_boolean_map().shape)
        shape[2] = 0    # 将地图的第三个维度为 0
        float_map = np.zeros(tuple(shape), dtype=float)
        return float_map

    def get_float_map_shape(self):
        return self.get_float_map().shape

    # 检查当前的位置是否在着陆区
    def is_in_landing_zone(self):
        return self.landing_zone[self.position[1]][self.position[0]]

    # 检查当前位置是否在禁飞区
    def is_in_no_fly_zone(self):
        # 越界也是禁飞区
        if 0 <= self.position[1] < self.no_fly_zone.shape[0] and 0 <= self.position[0] < self.no_fly_zone.shape[1]:
            return self.no_fly_zone[self.position[1], self.position[0]]
        return True

    # 更新目标区域和覆盖区域，表示已探索的区域
    def add_explored(self, view):
        self.target &= ~view    # 将 view 视为已探索区域，并从 target 中移除这些区域。
        self.coverage |= view   # 将 view 视为已探索区域，并将其添加到 coverage 中

    def set_terminal(self, terminal):
        self.terminal = terminal

    def set_landed(self, landed):
        self.landed = landed

    def set_position(self, position):
        self.position = position

    def decrement_movement_budget(self):
        self.movement_budget -= 1

    def is_terminal(self):
        return self.terminal
