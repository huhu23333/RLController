# config.py - 所有配置参数
import os

# ==================== 路径 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "runs")

# ==================== 仿真参数 ====================
DT_ENV = 1e-4                     # 环境积分步长
DT_CTRL = 1e-2                    # 控制周期
STEPS_PER_CTRL = int(DT_CTRL / DT_ENV)  # 100

# 环境物理参数 (与 test2.py 一致)
J = 0.01
TAU_C = 0.2
TAU_S = 0.3
OMEGA_S = 0.01
B = 0.01
ANGLE_LIMIT = None
DISTURBANCE_TORQUE = 0.0

# 控制限幅
U_MIN = -2.0
U_MAX = 2.0

# ==================== 神经网络参数 ====================
H = 20                            # 预测/输出时域 (控制步数, H*DT_CTRL=0.2s)
STATE_DIM = 1                     # omega (速度), 不再输入当前角度
TARGET_DIM = H                    # 目标角度与当前角度的连续化差值序列 y_i
INPUT_DIM = STATE_DIM + TARGET_DIM  # 21
ACTION_DIM = 1
HIDDEN_DIM = 256

# 数据增强参数
NOISE_STD_MIN = 0.0               # 训练噪音标准差下限 (弧度)
NOISE_STD_MAX = 0.3               # 训练噪音标准差上限 (弧度)
SCALE_FACTOR_MIN = 0.7            # 随机尺度因子下限
SCALE_FACTOR_MAX = 1.3            # 随机尺度因子上限

# ==================== SAC 训练参数 ====================
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
WD_ACTOR = 1e-4
WD_CRITIC = 1e-4
GAMMA = 0.85
TAU = 0.005                      # 目标网络软更新系数
BATCH_SIZE = 256
REPLAY_SIZE = 500000
TARGET_ENTROPY = -ACTION_DIM     # 目标熵 = -dim(A)

# 训练循环
TOTAL_EPISODES = 50000            # 总训练 episode 数
STEPS_PER_EPISODE = 2000          # 每个 episode 的控制步数 (20s)
MIN_SEGMENT_LENGTH = 1000        # 单段最小长度 (控制步数, 10s)
GRADIENT_STEPS = 50              # 每个 episode 的梯度更新步数
START_TRAIN_AFTER = 10000         # 经验回放有足够数据后才开始训练
SAVE_INTERVAL = 100              # 每 N 个 episode 保存一次模型
LOG_INTERVAL = 10                # 每 N 个 episode 记录一次日志

# 奖励权重
REWARD_TRACK_W = 1.0             # 跟踪误差权重
REWARD_TORQUE_W = 0.0 # 0.0001           # 力矩惩罚权重

# ==================== 数据采集参数 ====================
RECORD_KEY = "space"             # 录制开关按键 (pygame key name)
SAVE_KEY = "s"                   # 保存数据按键
LOAD_KEY = "l"                   # 读取数据按键
