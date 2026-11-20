from src.base.GridActions import GridActions


class GridPhysics:
    def __init__(self):
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = None

    def movement_step(self, action: GridActions):
        old_position = self.state.position
        x, y = old_position

        if action == GridActions.NORTH:
            y += 1
        elif action == GridActions.SOUTH:
            y -= 1
        elif action == GridActions.WEST:
            x -= 1
        elif action == GridActions.EAST:
            x += 1
        elif action == GridActions.LAND:  # 降落动作
            self.landing_attempts += 1
            if self.state.is_in_landing_zone():  # 是降落区域则降落成功
                self.state.set_landed(True)

        self.state.set_position([x, y])
        if self.state.is_in_no_fly_zone():  # 飞到禁飞区
            # Reset state
            self.boundary_counter += 1
            x, y = old_position
            self.state.set_position([x, y])

        self.state.decrement_movement_budget()   # 行动预算-1
        self.state.set_terminal(self.state.landed or (self.state.movement_budget == 0))  # 如果降落或移动预算=0则终止

        return x, y

    def reset(self, state):  # 重置代理的状态和计数器
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = state
