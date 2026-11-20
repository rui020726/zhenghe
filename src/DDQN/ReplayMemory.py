import numpy as np


def shape(exp):
    if type(exp) is np.ndarray:
        return list(exp.shape)
    else:
        return []


def type_of(exp):
    if type(exp) is np.ndarray:
        return exp.dtype
    else:
        return type(exp)


class ReplayMemory(object):
    """
    Replay memory class for RL
    """

    def __init__(self, size):
        self.k = 0  # 当前存储位置
        self.head = -1  # 最新的经验位置
        self.full = False  # 是否已满
        self.size = size
        self.memory = None  # 用来存储经验的数组

    def initialize(self, experience):    # 初始化回放记忆，经验的形状和类型根据输入确定
        self.memory = [np.zeros(shape=[self.size] + shape(exp), dtype=type_of(exp)) for exp in experience]

    def store(self, experience):     # 存储一个新的经验
        if self.memory is None:
            self.initialize(experience)
        if len(experience) != len(self.memory):
            raise Exception('经验的大小与记忆的大小不匹配', len(experience), '!=', len(self.memory))

        for e, mem in zip(experience, self.memory):
            mem[self.k] = e  # 将经验存储到当前的位置

        self.head = self.k  # 更新最新的经验位置
        self.k += 1
        if self.k >= self.size:
            self.k = 0
            self.full = True  # 记忆已满，开始覆盖最旧的经验

    def sample(self, batch_size):    # 从记忆中随机采样一批经验
        r = self.size
        if not self.full:
            r = self.k  # 如果未满，采样的范围是当前的存储位置
        random_idx = np.random.choice(r, size=batch_size, replace=False)
        random_idx[0] = self.head  # 保证最新的经验一定包含在采样中

        return [mem[random_idx] for mem in self.memory]

    def get(self, start, length):    # 从记忆中获取一段连续的经验
        return [mem[start:start + length] for mem in self.memory]

    def get_size(self):  # 返回当前已存储的经验数量
        if self.full:
            return self.size
        return self.k

    def get_max_size(self):  # 返回最大存储容量
        return self.size

    def reset(self):  # 重置回放记忆
        self.k = 0
        self.head = -1
        self.full = False

    def shuffle(self):   # 打乱经验
        self.memory = self.sample(self.get_size())
