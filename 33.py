import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.DDQN.Agent import MAPPO
import src.Map.timemap
import configs
from src.Map.timemap import Environment
from torch.distributions import Categorical
from src.DDQN.rescue import RescueAlgorithm
import random
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import imageio
import matplotlib
matplotlib.use('Agg')  # 使用非交互模式
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import imageio.v2 as imageio
from PIL import Image
import os
import re
def create_gif_from_images(image_folder, gif_filename='timestamp_map_simulation.gif', duration=0.1):
    images = []

    # 获取所有保存的图片文件路径，并确保按文件名顺序加载
    def natural_sort_key(filename):
        """通过提取文件名中的数字进行自然排序"""
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', filename)]

    # 获取图片路径并按数字顺序排序
    for filename in sorted(os.listdir(image_folder), key=natural_sort_key):
        if filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            images.append(imageio.imread(image_path))  # 使用 imageio.v2.imread

    # 如果没有图像，则跳过创建 GIF
    if len(images) == 0:
        print("No images found in the folder, skipping GIF creation.")
        return

    # 使用 imageio 将图像列表保存为 gif 文件
    imageio.mimsave(gif_filename, images, duration=duration)
    print(f"GIF saved to: {gif_filename}")
temp_image_folder = 'framess'
temp_image_folder_2 = 'frame'
# 创建 GIF - 时间戳地图的 GIF
create_gif_from_images(temp_image_folder_2, gif_filename='timestamp_map_simulation.gif')
create_gif_from_images(temp_image_folder, gif_filename='global_map_simulation.gif')