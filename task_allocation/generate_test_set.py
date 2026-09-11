import os
import pickle
from env.task_env import TaskEnv

# 创建测试集目录（在当前目录下）
test_set_dir = 'RALTestSet-3'
os.makedirs(test_set_dir, exist_ok=True)

print(f"Generating test set in {test_set_dir}...")
print(f"Full path: {os.path.abspath(test_set_dir)}")

# 生成50个测试环境
for i in range(50):
    try:
        # 使用固定的随机种子确保测试集可复现
        env = TaskEnv(
            prio_range=(0.3, 1.0),
            per_species_range=(2, 6),
            species_range=(2, 6),
            tasks_range=(5, 30),
            traits_dim=5,
            max_task_size=3,
            duration_scale=4,
            seed=i
        )

        # 保存环境
        env_path = os.path.join(test_set_dir, f'env_{i}.pkl')
        with open(env_path, 'wb') as f:
            pickle.dump(env, f)

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/50 environments")

    except Exception as e:
        print(f"Error generating env_{i}: {e}")
        continue

print(f"\n✅ Test set generated successfully!")
print(f"Location: {os.path.abspath(test_set_dir)}")

# 验证文件
env_files = [f for f in os.listdir(test_set_dir) if f.endswith('.pkl')]
print(f"Total files: {len(env_files)}")