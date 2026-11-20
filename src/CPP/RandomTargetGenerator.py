import numpy as np
from skimage.draw import random_shapes
import logging


class RandomTargetGeneratorParams:
    def __init__(self):
        self.coveragfe_range = (0.2, 0.8)
        self.shape_range = (1, 5)


class RandomTargetGenerator:

    def __init__(self, params: RandomTargetGeneratorParams, shape):
        self.params = params
        self.shape = shape

    # 随机生成区域
    def generate_target(self, obstacles):

        area = np.product(self.shape)

        target = self.__generate_random_shapes_area(
            self.params.shape_range[0],
            self.params.shape_range[1],
            area * self.params.coverage_range[0],
            area * self.params.coverage_range[1]
        )

        return target & ~obstacles

    # 这个方法使用 random_shapes 函数从 skimage.draw 库生成随机形状的图像，并将其转换为一个布尔数组
    def __generate_random_shapes(self, min_shapes, max_shapes):
        img, _ = random_shapes(self.shape, max_shapes, min_shapes=min_shapes, multichannel=False,
                               allow_overlap=True, random_seed=np.random.randint(2**16 - 1))
        # Numpy random usage for random seed unifies random seed which can be set for repeatability
        attempt = np.array(img != 255, dtype=bool)
        return attempt, np.sum(attempt)

    # 这是主要的形状生成方法，尝试生成一定数量的随机形状，并确保它们的总面积在指定范围内。
    # 如果无法在指定的尝试次数内生成符合要求的形状，它会发出警告并返回一个随机的形状。
    def __generate_random_shapes_area(self, min_shapes, max_shapes, min_area, max_area, retry=100):
        for attemptno in range(retry):
            attempt, area = self.__generate_random_shapes(min_shapes, max_shapes)
            if min_area is not None and min_area > area:
                continue
            if max_area is not None and max_area < area:
                continue
            return attempt
        print("Here")
        logging.warning("Was not able to generate shapes with given area constraint in allowed number of tries."
                        " Randomly returning next attempt.")
        attempt, area = self.__generate_random_shapes(min_shapes, max_shapes)
        logging.warning("Size is: ", area)
        return attempt

    # 这个方法用于生成一个随机形状的区域，并确保这些形状不与给定的排除区域重叠。
    def __generate_exclusive_shapes(self, exclusion, min_shapes, max_shapes):
        attempt, area = self.__generate_random_shapes(min_shapes, max_shapes)
        attempt = attempt & (~exclusion)  # 排除给定的区域
        area = np.sum(attempt)
        return attempt, area

    # 这个方法尝试多次生成一个满足面积要求的随机形状区域，并且确保这些形状区域不与 exclusion 区域重叠
    def __generate_exclusive_shapes_area(self, exclusion, min_shapes, max_shapes, min_area, max_area, retry=100):
        for attemptno in range(retry):
            attempt, area = self.__generate_exclusive_shapes(exclusion, min_shapes, max_shapes)
            if min_area is not None and min_area > area:  # 如果区域面积小于最小要求，则继续尝试
                continue
            if max_area is not None and max_area < area:  # 如果区域面积大于最大要求，则继续尝试
                continue
            return attempt  # 如果面积符合要求，则返回当前生成的区域

        logging.warning("Was not able to generate shapes with given area constraint in allowed number of tries."
                        " Randomly returning next attempt.")
        attempt, area = self.__generate_exclusive_shapes(exclusion, min_shapes, max_shapes)
        logging.warning("Size is: ", area)
        return attempt
