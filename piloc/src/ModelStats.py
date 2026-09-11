import collections
import datetime
import os
import shutil

import tensorflow as tf
import numpy as np
import progressbar
import distutils.util


class ModelStatsParams:
    def __init__(self,
                 save_model='models/save_model',
                 moving_average_length=50):  # 计算移动平均的长度，默认为 50。
        self.save_model = save_model
        self.moving_average_length = moving_average_length
        self.log_file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 当前时间戳生成一个日志文件的名称
        self.training_images = False


class ModelStats:

    def __init__(self, params: ModelStatsParams, display, force_override=False):
        self.params = params
        self.display = display

        self.evaluation_value_callback = None   # 存储评估值回调函数，用于定期评估模型性能。
        self.env_map_callback = None    # 用于获取环境地图的回调函数。
        self.log_value_callbacks = []   # 存储日志值的回调函数。
        self.trajectory = []    # 存储训练过程中或测试过程中模型轨迹的列表。

        self.log_dir = "logs/training/" + params.log_file_name  # 创建训练日志的目录路径。
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                                   histogram_freq=100)  # 允许在训练过程中记录并可视化各种训练数据
        self.model = None

        if os.path.isdir(self.log_dir):
            if force_override:  # 强制覆盖
                shutil.rmtree(self.log_dir)
            else:
                print(self.log_dir, 'already exists.')
                resp = input('Override log file? [Y/n]\n')
                if resp == '' or distutils.util.strtobool(resp):
                    print('Deleting old log dir')
                    shutil.rmtree(self.log_dir)
                else:
                    raise AttributeError('Okay bye')

        self.training_log_writer = tf.summary.create_file_writer(self.log_dir + '/training')
        self.testing_log_writer = tf.summary.create_file_writer(self.log_dir + '/test')

        self.evaluation_deque = collections.deque(maxlen=params.moving_average_length)   # 存储最近的评估结果，用于计算滑动平均。
        self.eval_best = -float('inf')
        self.bar = None

    # 设置一个回调函数 callback，用于计算评估值。
    def set_evaluation_value_callback(self, callback: callable):
        self.evaluation_value_callback = callback

    def add_experience(self, experience):
        self.trajectory.append(experience)

    def set_model(self, model):
        self.tensorboard_callback.set_model(model)
        self.model = model

    def set_env_map_callback(self, callback: callable):
        self.env_map_callback = callback

    def add_log_data_callback(self, name: str, callback: callable):
        self.log_value_callbacks.append((name, callback))

    def log_training_data(self, step):

        with self.training_log_writer.as_default():
            self.log_data(step, self.params.training_images)

    def log_testing_data(self, step):
        with self.testing_log_writer.as_default():
            self.log_data(step)

        if self.evaluation_value_callback:
            self.evaluation_deque.append(self.evaluation_value_callback())

    # 负责记录训练过程中的标量数据（如损失、准确率等）和图像数据（如轨迹图像），并将它们写入 TensorBoard。
    def log_data(self, step, images=True):

        for callback in self.log_value_callbacks:
            tf.summary.scalar(callback[0], callback[1](), step=step)
        if images:
            trajectory = self.display.display_episode(self.env_map_callback(), trajectory=self.trajectory)
            tf.summary.image('trajectory', trajectory,
                             step=step)

    # 该方法用于判断是否保存当前的模型，如果当前的评估值（滑动平均）比历史上的最佳评估值要好，就保存模型。
    def save_if_best(self):
        if len(self.evaluation_deque) < self.params.moving_average_length:
            return

        eval_mean = np.mean(self.evaluation_deque)
        if eval_mean > self.eval_best:
            self.eval_best = eval_mean
            if self.params.save_model != '':
                print('Saving best with:', eval_mean)
                self.model.save_weights(self.params.save_model + '_best')

    def get_log_dir(self):
        return self.log_dir

    def training_ended(self):

        if self.params.save_model != '':
            self.model.save_weights(self.params.save_model + '_unfinished')
            print('Model saved as', self.params.save_model + '_unfinished')

    def save_episode(self, save_path):
        f = open(save_path + ".txt", "w")

        for callback in self.log_value_callbacks:
            f.write(callback[0] + ' ' + str(callback[1]()) + '\n')
        f.close()

    def on_episode_begin(self, episode_count):
        self.tensorboard_callback.on_epoch_begin(episode_count)
        self.trajectory = []

    def on_episode_end(self, episode_count):
        self.tensorboard_callback.on_epoch_end(episode_count)

