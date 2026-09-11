class EnvParams:
    # ===== 🔥 确保这些参数正确 =====
    SPECIES_AGENTS_RANGE = (4, 6)  # 每个物种4-6个智能体
    SPECIES_RANGE = (4, 6)         # 4-6个物种
    TASKS_RANGE = (20, 30)         # 20-30个任务
    MAX_TIME = 200
    DEADLINE_TIGHTNESS = (0.5, 2.0)


class TrainParams:
    USE_GPU = True
    USE_GPU_GLOBAL = True
    NUM_GPU = 1
    NUM_META_AGENT = 1
    LR = 1e-5
    GAMMA = 1
    DECAY_STEP = 2e3
    RESET_OPT = False
    EVALUATE = True
    EVALUATION_SAMPLES = 256
    RESET_RAY = False
    INCREASE_DIFFICULTY = 20000
    SUMMARY_WINDOW = 8
    DEMON_RATE = 0.5
    IL_DECAY = -1e-5  # -1e-6 700k decay 0.5, -1e-5 70k decay 0.5, -1e-4 7k decay 0.5
    BATCH_SIZE = 1024
    AGENT_INPUT_DIM = 13
    TASK_INPUT_DIM = 18
    EMBEDDING_DIM = 128
    SAMPLE_SIZE = 200
    PADDING_SIZE = 50
    POMO_SIZE = 10
    FORCE_MAX_OPEN_TASK = False
    MAX_EPISODES = 100000
    ENTROPY_WEIGHT = 0.15 # 熵权重（鼓励探索）

class SaverParams:
    FOLDER_NAME = 'save_12'
    # 12修改了训练过程，帮助收敛
    # 11修改了训练过程，帮助收敛
    # 8是边决策边更新
    # 7是修改奖励函数
    # 6是修改奖励函数
    # save_5的奖励函数中，性能最好
    MODEL_PATH = f'model/{FOLDER_NAME}'
    TRAIN_PATH = f'train/{FOLDER_NAME}'
    GIFS_PATH = f'gifs/{FOLDER_NAME}'
    LOAD_MODEL = False
    LOAD_FROM = 'current'  # 'best'
    SAVE = True
    SAVE_IMG = True
    SAVE_IMG_GAP = 1000

