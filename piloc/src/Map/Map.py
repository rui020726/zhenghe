import numpy as np
from skimage import io


class Map:
    def __init__(self, map_data):
        self.start_land_zone = map_data[:, :, 2].astype(bool)     # 地图数据第三层表示起点或降落区域 (start_land_zone)，通过 .astype(bool) 转换为布尔数组。
        self.nfz = map_data[:, :, 0].astype(bool)   # 地图数据第一层表示禁飞区 (nfz)。
        self.obstacles = map_data[:, :, 1].astype(bool)     # 地图数据第二层表示障碍物 (obstacles)

    def get_starting_vector(self):   # 获取所有起点/降落区域的坐标
        similar = np.where(self.start_land_zone)     # 找到 start_land_zone 中布尔值为 True 的索引。
        return list(zip(similar[1], similar[0]))     # 将坐标（行和列）转换为 (x, y) 格式。

    def get_free_space_vector(self):     # 获取所有自由空间的坐标
        free_space = np.logical_not(
            np.logical_or(self.obstacles, self.start_land_zone))        # 找到障碍物或起点区域的联合区域; 取反，标记自由空间为 True。
        free_idcs = np.where(free_space)     # 将自由空间的坐标转换为 (x, y) 格式。
        return list(zip(free_idcs[1], free_idcs[0]))

    def get_size(self):
        return self.start_land_zone.shape[:2]    # 提取地图的行数（高度）和列数（宽度）。

# 加载图像，并将其转换为布尔值数组。
def load_image(path):
    if type(path) is not str:    #  检查路径是否是字符串类型。
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=True)     # 加载图像并转换为灰度模式。
    return np.array(data, dtype=bool)     # 将灰度图像转换为布尔数组（非零像素为 True，零像素为 False）。

# 保存布尔或非布尔图像到文件
def save_image(path, image):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    if image.dtype == bool:     # 如果图像是布尔数组，将其转换为 8 位整数（0 或 255）
        io.imsave(path, image * np.uint8(255))      # 使用 scikit-image 的 imsave 函数保存图像
    else:
        io.imsave(path, image)

# 加载 3D 地图数据，并返回 Map 对象
def load_map(path):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=False)     # 以彩色模式加载图像。
    return Map(data)

# 加载目标区域的布尔图像，并排除障碍物。
def load_target(path, obstacles=None):
    if type(path) is not str:
        raise TypeError('path needs to be a string')

    data = np.array(io.imread(path, as_gray=True), dtype=bool)      # 加载目标图像并转换为布尔数组。
    if obstacles is not None:       # 如果提供了障碍物数组，将目标区域与障碍物取反逻辑与（排除障碍物区域）。
        data = data & ~obstacles
    return data
