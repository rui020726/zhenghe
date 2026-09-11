import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)


def extract_tensorboard_data(log_dir):
    """
    从TensorBoard日志文件中提取所有训练数据

    Args:
        log_dir: TensorBoard日志目录路径

    Returns:
        dict: {metric_name: [(step, value), ...]}
    """
    # 查找日志文件
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))

    if not event_files:
        raise FileNotFoundError(f"No TensorBoard log files found in {log_dir}")

    print(f"Found {len(event_files)} TensorBoard log file(s)")

    # 提取数据
    all_data = {}

    for event_file in event_files:
        print(f"Processing: {event_file}")
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        # 获取所有标量指标
        tags = ea.Tags()['scalars']

        for tag in tags:
            events = ea.Scalars(tag)
            data = [(e.step, e.value) for e in events]

            if tag not in all_data:
                all_data[tag] = []
            all_data[tag].extend(data)

    # 按step排序并去重
    for tag in all_data:
        all_data[tag] = sorted(set(all_data[tag]), key=lambda x: x[0])

    return all_data


def plot_training_curves(data, save_dir='training_plots'):
    """
    绘制所有训练曲线

    Args:
        data: 从TensorBoard提取的数据
        save_dir: 保存图片的目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 分组绘制
    loss_metrics = [k for k in data.keys() if 'Loss' in k]
    perf_metrics = [k for k in data.keys() if 'Perf' in k]

    # ===== 1. 绘制Loss指标 =====
    if loss_metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Loss Metrics', fontsize=16, fontweight='bold')

        for idx, metric in enumerate(loss_metrics):
            if idx >= 6:
                break

            row = idx // 3
            col = idx % 3

            steps, values = zip(*data[metric])
            axes[row, col].plot(steps, values, linewidth=2, alpha=0.7)
            axes[row, col].set_title(metric.replace('Loss/', ''))
            axes[row, col].set_xlabel('Episode')
            axes[row, col].set_ylabel('Value')
            axes[row, col].grid(True, alpha=0.3)

            # 添加移动平均线
            if len(values) > 10:
                window = min(50, len(values) // 10)
                moving_avg = pd.Series(values).rolling(window=window, min_periods=1).mean()
                axes[row, col].plot(steps, moving_avg, 'r-', linewidth=2, label=f'MA({window})')
                axes[row, col].legend()

        # 隐藏多余的子图
        for idx in range(len(loss_metrics), 6):
            row = idx // 3
            col = idx % 3
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_metrics.png'), dpi=300, bbox_inches='tight')
        print(f"✅ Loss metrics saved to {save_dir}/loss_metrics.png")
        plt.close()

    # ===== 2. 绘制Performance指标 =====
    if perf_metrics:
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle('Performance Metrics', fontsize=16, fontweight='bold')

        for idx, metric in enumerate(perf_metrics):
            if idx >= 8:
                break

            row = idx // 4
            col = idx % 4

            steps, values = zip(*data[metric])
            axes[row, col].plot(steps, values, linewidth=2, alpha=0.7, color='green')
            axes[row, col].set_title(metric.replace('Perf/', ''))
            axes[row, col].set_xlabel('Episode')
            axes[row, col].set_ylabel('Value')
            axes[row, col].grid(True, alpha=0.3)

            # 添加移动平均线
            if len(values) > 10:
                window = min(50, len(values) // 10)
                moving_avg = pd.Series(values).rolling(window=window, min_periods=1).mean()
                axes[row, col].plot(steps, moving_avg, 'r-', linewidth=2, label=f'MA({window})')
                axes[row, col].legend()

        # 隐藏多余的子图
        for idx in range(len(perf_metrics), 8):
            row = idx // 4
            col = idx % 4
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        print(f"✅ Performance metrics saved to {save_dir}/performance_metrics.png")
        plt.close()

    # ===== 3. 绘制关键指标对比 =====
    key_metrics = [
        'Perf/Total Success Rate',
        'Perf/High Priority Success Rate',
        'Loss/Reward',
        'Perf/Makespan'
    ]

    available_keys = [k for k in key_metrics if k in data]

    if available_keys:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Key Performance Indicators', fontsize=16, fontweight='bold')

        for idx, metric in enumerate(available_keys):
            if idx >= 4:
                break

            row = idx // 2
            col = idx % 2

            steps, values = zip(*data[metric])
            axes[row, col].plot(steps, values, linewidth=1.5, alpha=0.5, label='Raw')

            # 移动平均
            if len(values) > 10:
                window = min(100, len(values) // 10)
                moving_avg = pd.Series(values).rolling(window=window, min_periods=1).mean()
                axes[row, col].plot(steps, moving_avg, 'r-', linewidth=3, label=f'MA({window})')

            axes[row, col].set_title(metric.replace('Loss/', '').replace('Perf/', ''), fontsize=14, fontweight='bold')
            axes[row, col].set_xlabel('Episode', fontsize=12)
            axes[row, col].set_ylabel('Value', fontsize=12)
            axes[row, col].legend(fontsize=10)
            axes[row, col].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'key_metrics.png'), dpi=300, bbox_inches='tight')
        print(f"✅ Key metrics saved to {save_dir}/key_metrics.png")
        plt.close()

    # ===== 4. 保存数据为CSV =====
    for metric_name, metric_data in data.items():
        steps, values = zip(*metric_data)
        df = pd.DataFrame({'episode': steps, 'value': values})
        csv_path = os.path.join(save_dir, f"{metric_name.replace('/', '_')}.csv")
        df.to_csv(csv_path, index=False)

    print(f"\n✅ All metrics data saved to {save_dir}/")


def print_training_summary(data):
    """打印训练摘要统计"""
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)

    # Success Rate
    if 'Perf/Total Success Rate' in data:
        steps, values = zip(*data['Perf/Total Success Rate'])
        print(f"\n📊 Total Success Rate:")
        print(f"  Initial: {values[0]:.2%}")
        print(f"  Final: {values[-1]:.2%}")
        print(f"  Best: {max(values):.2%} (Episode {steps[values.index(max(values))]})")
        print(f"  Average (last 100): {np.mean(values[-100:]):.2%}")

    # High Priority Success
    if 'Perf/High Priority Success Rate' in data:
        steps, values = zip(*data['Perf/High Priority Success Rate'])
        print(f"\n🎯 High Priority Success Rate:")
        print(f"  Initial: {values[0]:.2%}")
        print(f"  Final: {values[-1]:.2%}")
        print(f"  Best: {max(values):.2%} (Episode {steps[values.index(max(values))]})")
        print(f"  Average (last 100): {np.mean(values[-100:]):.2%}")

    # Reward
    if 'Loss/Reward' in data:
        steps, values = zip(*data['Loss/Reward'])
        print(f"\n💰 Reward:")
        print(f"  Initial: {values[0]:.2f}")
        print(f"  Final: {values[-1]:.2f}")
        print(f"  Best: {max(values):.2f} (Episode {steps[values.index(max(values))]})")
        print(f"  Average (last 100): {np.mean(values[-100:]):.2f}")

    # Makespan
    if 'Perf/Makespan' in data:
        steps, values = zip(*data['Perf/Makespan'])
        print(f"\n⏱️  Makespan:")
        print(f"  Initial: {values[0]:.2f}")
        print(f"  Final: {values[-1]:.2f}")
        print(f"  Best: {min(values):.2f} (Episode {steps[values.index(min(values))]})")
        print(f"  Average (last 100): {np.mean(values[-100:]):.2f}")

    print("=" * 80 + "\n")


def main():
    """主函数"""
    # 配置路径（根据你的parameters.py中的设置调整）
    LOG_DIR =r' D:\代码\task-01\train\save_2\events.out.tfevents.1763555040.CHINAMI-N7MVRIO.45784.0'  # 或 './logs/train'

    # 检查路径是否存在
    if not os.path.exists(LOG_DIR):
        print(f"❌ Log directory not found: {LOG_DIR}")
        print("\nPlease check your TensorBoard log directory.")
        print("Common locations:")
        print("  - ./runs")
        print("  - ./logs/train")
        print("  - ./model/train")
        return

    try:
        # 提取数据
        print("Extracting data from TensorBoard logs...")
        data = extract_tensorboard_data(LOG_DIR)

        print(f"\n✅ Extracted {len(data)} metrics:")
        for metric_name in sorted(data.keys()):
            num_points = len(data[metric_name])
            print(f"  - {metric_name}: {num_points} data points")

        # 打印统计摘要
        print_training_summary(data)

        # 绘制曲线
        print("\nGenerating plots...")
        plot_training_curves(data)

        print("\n✅ All training curves have been saved!")
        print("You can also view them in TensorBoard by running:")
        print(f"  tensorboard --logdir={LOG_DIR}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()