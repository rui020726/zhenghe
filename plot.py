import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_comparison_coverage_curves(csv_path_1, csv_path_2, output_path='coverage_comparison.png'):
    """
    绘制两个方法的覆盖率对比图
    :param csv_path_1: PILOC-ph的CSV数据路径
    :param csv_path_2: PILOC的CSV数据路径
    :param output_path: 输出图像路径
    """
    # 读取两个CSV文件
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制PILOC的曲线（红色，先绘制使其在底层）
    ax.plot(df2['step'], df2['mean_coverage'], 'r-', linewidth=2.5, label='PILOC (Mean Coverage)', zorder=2)
    ax.fill_between(df2['step'],
                    df2['mean_coverage'] - df2['std_coverage'],
                    df2['mean_coverage'] + df2['std_coverage'],
                    alpha=0.3,
                    color='red',
                    label='PILOC (±1 Std Dev)',
                    zorder=1)

    # 绘制PILOC-ph的曲线（蓝色，后绘制使其在上层）
    ax.plot(df1['step'], df1['mean_coverage'], 'b-', linewidth=2.5, label='PILOC-ph (Mean Coverage)', zorder=4)
    ax.fill_between(df1['step'],
                    df1['mean_coverage'] - df1['std_coverage'],
                    df1['mean_coverage'] + df1['std_coverage'],
                    alpha=0.3,
                    color='blue',
                    label='PILOC-ph (±1 Std Dev)',
                    zorder=3)

    # 设置图表属性
    ax.set_xlabel('Time Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coverage Rate', fontsize=14, fontweight='bold')
    ax.set_title('Coverage Rate Comparison: PILOC-ph vs PILOC', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 1.0)

    # 添加最终覆盖率标注（放在曲线右侧）
    final_piloc_ph = df1['mean_coverage'].iloc[-1]
    final_piloc = df2['mean_coverage'].iloc[-1]

    # 调整标注位置避免重叠
    ax.text(252, final_piloc_ph, f'{final_piloc_ph:.3f}',
            ha='left', va='center', fontsize=11, color='blue', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', alpha=0.8))
    ax.text(252, final_piloc, f'{final_piloc:.3f}',
            ha='left', va='center', fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))

    # 添加关键时间点的垂直参考线
    key_steps = [50, 100, 150, 200]
    for step in key_steps:
        ax.axvline(x=step, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✅ 对比图已保存到: {output_path}')

    # 打印详细统计信息
    print("\n" + "=" * 70)
    print("📊 覆盖率对比统计:")
    print("=" * 70)
    print(f"PILOC-ph 最终平均覆盖率: {final_piloc_ph:.4f} ± {df1['std_coverage'].iloc[-1]:.4f}")
    print(f"PILOC    最终平均覆盖率: {final_piloc:.4f} ± {df2['std_coverage'].iloc[-1]:.4f}")
    improvement = (final_piloc_ph - final_piloc) / final_piloc * 100
    print(f"\n相对改进幅度: {improvement:.2f}%")
    print(f"绝对改进值:   {(final_piloc_ph - final_piloc):.4f}")

    # 计算在不同时间点的覆盖率差异
    print("\n⏱️  关键时间点对比:")
    print("-" * 70)
    for step in [50, 100, 150, 200, 250]:
        idx = step - 1
        cov_ph = df1['mean_coverage'].iloc[idx]
        cov_piloc = df2['mean_coverage'].iloc[idx]
        diff = cov_ph - cov_piloc
        print(f"Step {step:3d}: PILOC-ph={cov_ph:.4f}, PILOC={cov_piloc:.4f}, Δ={diff:+.4f}")
    print("=" * 70)


def plot_side_by_side_subplots(csv_path_1, csv_path_2, output_path='coverage_side_by_side.png'):
    """
    绘制两个方法的覆盖率子图对比（左右并排）
    :param csv_path_1: PILOC-ph的CSV数据路径
    :param csv_path_2: PILOC的CSV数据路径
    :param output_path: 输出图像路径
    """
    # 读取两个CSV文件
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：PILOC-ph
    ax1.plot(df1['step'], df1['mean_coverage'], 'b-', linewidth=2.5, label='Mean Coverage')
    ax1.fill_between(df1['step'],
                     df1['mean_coverage'] - df1['std_coverage'],
                     df1['mean_coverage'] + df1['std_coverage'],
                     alpha=0.3,
                     color='blue',
                     label='±1 Std Dev')
    ax1.set_xlabel('Time Steps', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Coverage Rate', fontsize=13, fontweight='bold')
    ax1.set_title('PILOC-ph', fontsize=15, fontweight='bold', color='blue')
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 250)
    ax1.set_ylim(0, 1.0)

    # 添加最终覆盖率标注
    final_ph = df1['mean_coverage'].iloc[-1]
    ax1.text(125, 0.92, f'Final: {final_ph:.4f}',
             ha='center', fontsize=12, color='blue', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='blue', alpha=0.7))

    # 右图：PILOC
    ax2.plot(df2['step'], df2['mean_coverage'], 'r-', linewidth=2.5, label='Mean Coverage')
    ax2.fill_between(df2['step'],
                     df2['mean_coverage'] - df2['std_coverage'],
                     df2['mean_coverage'] + df2['std_coverage'],
                     alpha=0.3,
                     color='red',
                     label='±1 Std Dev')
    ax2.set_xlabel('Time Steps', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Coverage Rate', fontsize=13, fontweight='bold')
    ax2.set_title('PILOC', fontsize=15, fontweight='bold', color='red')
    ax2.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 250)
    ax2.set_ylim(0, 1.0)

    # 添加最终覆盖率标注
    final_piloc = df2['mean_coverage'].iloc[-1]
    ax2.text(125, 0.92, f'Final: {final_piloc:.4f}',
             ha='center', fontsize=12, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', edgecolor='red', alpha=0.7))

    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✅ 并排对比图已保存到: {output_path}')


def plot_difference_curve(csv_path_1, csv_path_2, output_path='coverage_difference.png'):
    """
    绘制两个方法覆盖率的差值曲线
    :param csv_path_1: PILOC-ph的CSV数据路径
    :param csv_path_2: PILOC的CSV数据路径
    :param output_path: 输出图像路径
    """
    # 读取两个CSV文件
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    # 计算差值
    diff = df1['mean_coverage'] - df2['mean_coverage']

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制差值曲线
    ax.plot(df1['step'], diff, 'g-', linewidth=2.5, label='Coverage Difference (PILOC-ph - PILOC)')
    ax.fill_between(df1['step'], 0, diff, where=(diff >= 0), alpha=0.3, color='green', label='PILOC-ph Better')
    ax.fill_between(df1['step'], 0, diff, where=(diff < 0), alpha=0.3, color='orange', label='PILOC Better')

    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

    # 设置图表属性
    ax.set_xlabel('Time Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coverage Rate Difference', fontsize=14, fontweight='bold')
    ax.set_title('Coverage Rate Difference: PILOC-ph - PILOC', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 250)

    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✅ 差值曲线图已保存到: {output_path}')


# 使用示例
if __name__ == '__main__':
    # 替换为你的实际CSV文件路径
    csv_piloc_ph = r'D:/代码/修改6--在修改5的基础上，所有网络使用相同的参数/Abl_del_ph/metrics_out_ph/average_coverage_data.csv'
    csv_piloc = r'D:/代码/修改6--在修改5的基础上，所有网络使用相同的参数/metrics_out/average_coverage_data.csv'

    print("🚀 开始生成覆盖率对比图...\n")

    try:
        # 生成叠加对比图
        plot_comparison_coverage_curves(csv_piloc_ph, csv_piloc, 'coverage_comparison_overlay.png')

        # 生成并排对比图
        plot_side_by_side_subplots(csv_piloc_ph, csv_piloc, 'coverage_comparison_side_by_side.png')

        # 新增：生成差值曲线图
        plot_difference_curve(csv_piloc_ph, csv_piloc, 'coverage_difference.png')

        print("\n✅ 所有图表生成完成！")

    except FileNotFoundError as e:
        print(f"❌ 错误：找不到CSV文件")
        print(f"   {e}")
        print("   请检查文件路径是否正确。")
    except Exception as e:
        print(f"❌ 发生错误：{e}")