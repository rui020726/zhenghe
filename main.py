import argparse
import os

from src.CPP.Environment import CPPEnvironmentParams, CPPEnvironment


from utils import *


def main_cpp(p):
    env = CPPEnvironment(p)

    env.run()



if __name__ == "__main__":
    # 手动设置参数
    class Args:
        gpu = True  # 激活GPU使用
        generate_config = False  # 不生成默认配置文件
        config = "config/cpp.json"  # 配置文件路径
        id = "custom_id"  # 覆盖日志文件名和保存名
        params = None  # 额外的参数
        cpp = True  # 运行覆盖路径规划

    args = Args()

    if args.generate_config:
        if args.cpp:
            generate_config(CPPEnvironmentParams(), "config/cpp.json")
        else:
            print("Specify which config to generate")
        exit(0)

    if args.config is None:
        print("Config file needed!")
        exit(1)

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    params = read_config(args.config)

    if args.params is not None:
        params = override_params(params, args.params)

    if args.id is not None:
        params.model_stats_params.save_model = "models/" + args.id
        params.model_stats_params.log_file_name = args.id

    if args.cpp:
        main_cpp(params)

