# distill.py - 知识蒸馏脚本
# 从预训练的 SAC Actor (教师网络) 中蒸馏知识到轻量级学生网络
# 使用与 train_sac.py 相同的输入构建函数和环境交互方式

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import math
import random
from torch.utils.tensorboard import SummaryWriter

from config import (
    DATA_DIR, MODEL_DIR, LOG_DIR,
    DT_CTRL,
    U_MIN, U_MAX, H,
    STATE_DIM, TARGET_DIM, INPUT_DIM, ACTION_DIM, HIDDEN_DIM,
    NOISE_STD_MIN, NOISE_STD_MAX, SCALE_FACTOR_MIN, SCALE_FACTOR_MAX,
    MIN_SEGMENT_LENGTH, STEPS_PER_EPISODE,
)
from networks import Actor, SmallActor


# ---------- 角度工具函数 (与 train_sac.py 保持一致) ----------
def wrap_angle(a):
    """将角度标准化到 [-π, π]"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def angle_error(a, b):
    """计算两个角度之间的最小误差 (弧度, [-π, π])"""
    err = a - b
    return (err + math.pi) % (2 * math.pi) - math.pi


# ---------- 数据加载 (与 train_sac.py 保持一致) ----------
def load_all_target_sequences():
    """加载 DATA_DIR 下所有录制数据文件"""
    if not os.path.exists(DATA_DIR):
        print(f"数据目录不存在: {DATA_DIR}")
        return []

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.npz')])
    if not files:
        print(f"数据目录为空: {DATA_DIR}")
        return []

    all_targets = []
    for f in files:
        filepath = os.path.join(DATA_DIR, f)
        try:
            data = np.load(filepath)
            targets = data['target'].astype(np.float32)
            if len(targets) > MIN_SEGMENT_LENGTH:
                all_targets.append(targets)
                print(f"  加载 {f}: {len(targets)} 步 ({len(targets)*DT_CTRL:.1f}s)")
        except Exception as e:
            print(f"  加载 {f} 失败: {e}")

    print(f"共加载 {len(all_targets)} 条目标序列")
    return all_targets


def sample_target_episode(all_targets, total_len, min_segment_len, add_offset=True):
    """从目标序列库中随机采样片段并拼接"""
    segments = []
    remaining = total_len

    while remaining > 0:
        seq_idx = random.randint(0, len(all_targets) - 1)
        seq = all_targets[seq_idx]

        max_seg_len = min(len(seq), remaining)
        if remaining < min_segment_len:
            min_len = remaining
        else:
            min_len = min(min_segment_len, max_seg_len)

        if max_seg_len < min_len:
            continue

        seg_len = random.randint(min_len, max_seg_len)
        start = random.randint(0, len(seq) - seg_len)
        segment = seq[start:start + seg_len].copy()
        segments.append(segment)
        remaining -= seg_len

    target_seq = np.concatenate(segments)[:total_len]

    if add_offset:
        noise_std = random.uniform(NOISE_STD_MIN, NOISE_STD_MAX)
        if noise_std > 0:
            target_seq = target_seq + np.random.normal(0, noise_std, size=target_seq.shape).astype(np.float32)

        scale = random.uniform(SCALE_FACTOR_MIN, SCALE_FACTOR_MAX)
        target_seq = target_seq * scale

        direction = 1 if random.random() < 0.5 else -1
        target_seq = target_seq * direction

        offset = random.uniform(-math.pi, math.pi)
        target_seq = target_seq + offset

    return target_seq.astype(np.float32)


# ---------- 状态构建函数 (与 train_sac.py 的 build_state 保持一致) ----------
def build_state(theta, omega, target_seq, idx, H_val):
    """
    构建 SAC 的输入状态向量 - 使用连续化差值序列.
    (与 train_sac.py 中的 build_state 完全一致)

    Args:
        theta, omega: 当前状态
        target_seq:   完整目标序列
        idx:          当前步索引 (从0开始)
        H_val:        预测时域

    Returns:
        state: np.array [STATE_DIM + H_val]
        结构: [ω, y_0, y_1, ..., y_{H-1}]
    """
    state_vec = np.zeros(STATE_DIM + H_val, dtype=np.float32)
    state_vec[0] = omega

    for k in range(H_val):
        tgt_idx = idx + k + 1
        if tgt_idx < len(target_seq):
            tgt = float(target_seq[tgt_idx])
        else:
            tgt = float(target_seq[-1])

        raw_y = wrap_angle(tgt - theta)

        if k == 0:
            y_cont = raw_y
        else:
            y_prev = state_vec[STATE_DIM + k - 1]
            delta = wrap_angle(raw_y - y_prev)
            y_cont = y_prev + delta

        state_vec[STATE_DIM + k] = y_cont

    return state_vec


# ---------- 随机采样参数 ----------
# 速度采样参数 (正态分布, 弧度/秒)
OMEGA_MEAN = 0.0     # 速度均值
OMEGA_STD = 10.0      # 速度标准差
OMEGA_NEAR_0_PROB = 0.2      # 速度采样换为另一个接近0的分布的概率
OMEGA_NEAR_0_STD = 0.1      # 接近0的分布速度标准差
OMEGA_0_PROB = 0.2      # 在上述基础上速度固定为0概率

D_THETA_NEAR_0_PROB = 0.1  # 当前位置与第一个目标位置相近的概率
D_THETA_NEAR_0_STD = 0.02      # 上述的接近分布方差


# ---------- 蒸馏配置 ----------
# 学生网络结构参数
STUDENT_HIDDEN_DIM = 64           # 学生网络隐藏层维度
STUDENT_NUM_BLOCKS = 3            # 学生网络 ResidualFCBlock 数量

# 蒸馏训练参数
DISTILL_LR = 1e-3                 # 学生网络学习率
DISTILL_EPOCHS = 100              # 训练轮数 (每轮遍历所有数据一遍)
DISTILL_BATCH_SIZE = 256          # 批次大小
DISTILL_WD = 1e-5                 # 权重衰减
DISTILL_LOG_INTERVAL = 10         # 日志间隔 (epoch)
DISTILL_SAVE_INTERVAL = 50        # 保存间隔 (epoch)
DISTILL_EPISODES_PER_EPOCH = 20   # 每轮采样的 episode 数

# 损失权重
LOSS_ACTION_MSE_W = 1.0           # 动作 MSE 损失权重
LOSS_DISTRIBUTION_W = 0.0         # 分布匹配损失权重 (KL散度, 默认关闭)


def distill():
    """
    主蒸馏训练函数.

    流程:
    1. 加载教师网络 (预训练的 SAC Actor)
    2. 初始化学生网络 (轻量级 SmallActor)
    3. 从录制数据中采样目标序列
    4. 在仿真环境中交互, 教师网络输出作为监督信号
    5. 训练学生网络模仿教师网络的输出
    6. 保存蒸馏后的学生模型
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print("=" * 60)
    print("SAC Actor 知识蒸馏")
    print("=" * 60)

    # ---- 1. 加载教师网络 ----
    teacher_path = os.path.join(MODEL_DIR, "actor_best.pth")
    if not os.path.exists(teacher_path):
        # 尝试查找最新的 actor 模型
        actors = sorted([f for f in os.listdir(MODEL_DIR)
                         if f.startswith('actor_') and f.endswith('.pth')])
        if not actors:
            print(f"错误: 在 {MODEL_DIR} 未找到任何 actor 模型文件!")
            print("请先运行 train_sac.py 训练教师网络.")
            return
        teacher_path = os.path.join(MODEL_DIR, actors[-1])
        print(f"未找到 actor_best.pth, 自动选择: {teacher_path}")

    teacher = Actor(INPUT_DIM, ACTION_DIM, HIDDEN_DIM, U_MIN, U_MAX).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()
    # 冻结教师网络参数
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"教师网络已加载: {teacher_path}")
    teacher_param_count = sum(p.numel() for p in teacher.parameters())
    print(f"教师网络参数量: {teacher_param_count:,}")

    # ---- 2. 初始化学生网络 ----
    student = SmallActor(
        INPUT_DIM, ACTION_DIM,
        hidden_dim=STUDENT_HIDDEN_DIM,
        num_blocks=STUDENT_NUM_BLOCKS,
        u_min=U_MIN, u_max=U_MAX
    ).to(device)
    student.train()

    student_param_count = sum(p.numel() for p in student.parameters())
    print(f"学生网络参数量: {student_param_count:,}")
    print(f"压缩比: {teacher_param_count / student_param_count:.1f}x")
    print(f"学生网络结构: hidden_dim={STUDENT_HIDDEN_DIM}, num_blocks={STUDENT_NUM_BLOCKS}")

    # ---- 3. 加载训练数据 ----
    print("\n加载目标序列数据...")
    all_targets = load_all_target_sequences()
    if not all_targets:
        print("错误: 没有找到训练数据! 请先运行 collect_data.py 采集数据.")
        return

    # ---- 4. 配置优化器和日志 ----
    optimizer = optim.Adam(student.parameters(), lr=DISTILL_LR, weight_decay=DISTILL_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DISTILL_EPOCHS)
    mse_loss = nn.MSELoss()

    distill_log_dir = os.path.join(LOG_DIR, "distill")
    os.makedirs(distill_log_dir, exist_ok=True)
    writer = SummaryWriter(distill_log_dir)

    global_step = 0
    best_loss = float('inf')

    print(f"\n开始蒸馏训练: {DISTILL_EPOCHS} epochs, "
          f"每 epoch {DISTILL_EPISODES_PER_EPOCH} episodes")

    # ---- 5. 蒸馏训练循环 ----
    for epoch in range(1, DISTILL_EPOCHS + 1):
        epoch_action_loss = 0.0
        epoch_total_steps = 0

        for ep in range(DISTILL_EPISODES_PER_EPOCH):
            # 采样目标序列 (需要 STEPS_PER_EPISODE + H 个目标)
            target_seq = sample_target_episode(
                all_targets,
                total_len=STEPS_PER_EPISODE + H,
                min_segment_len=MIN_SEGMENT_LENGTH,
                add_offset=True
            )

            # 收集当前 episode 的 (状态, 教师动作) 对
            states_batch = []
            teacher_actions_batch = []

            for step in range(STEPS_PER_EPISODE):
                # 随机采样当前状态 (不使用仿真环境)
                if random.random() > D_THETA_NEAR_0_PROB:
                    theta = random.uniform(-math.pi, math.pi)
                else:
                    theta = random.gauss(float(target_seq[step + 1]), D_THETA_NEAR_0_STD)
                if random.random() > OMEGA_NEAR_0_PROB:
                    omega = random.gauss(OMEGA_MEAN, OMEGA_STD)
                else:
                    if random.random() > OMEGA_0_PROB:
                        omega = random.gauss(OMEGA_MEAN, OMEGA_NEAR_0_STD)
                    else:
                        omega = 0.0

                # 构建状态 (与 SAC 训练完全一致)
                state = build_state(theta, omega, target_seq, step, H)

                # 教师网络前向 (确定性推理)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    teacher_action_t, _, _ = teacher(
                        state_tensor, deterministic=True, with_log_prob=False
                    )
                teacher_action = teacher_action_t.cpu().numpy()[0]

                # 保存 (状态, 教师动作) 对
                states_batch.append(state)
                teacher_actions_batch.append(teacher_action)

            # ---- 在此 episode 的数据上训练学生网络 ----
            # 将数据转换为张量
            states_tensor = torch.FloatTensor(np.array(states_batch)).to(device)
            teacher_actions_tensor = torch.FloatTensor(np.array(teacher_actions_batch)).to(device)

            # 小批量训练
            n_samples = len(states_batch)
            indices = list(range(n_samples))
            random.shuffle(indices)

            for start_idx in range(0, n_samples, DISTILL_BATCH_SIZE):
                batch_indices = indices[start_idx:start_idx + DISTILL_BATCH_SIZE]
                batch_states = states_tensor[batch_indices]
                batch_teacher_actions = teacher_actions_tensor[batch_indices]

                # 学生网络前向
                student_actions = student(batch_states)

                # 动作 MSE Loss: 让学生输出接近教师输出
                loss_action = mse_loss(student_actions, batch_teacher_actions)

                # 总损失
                total_loss = LOSS_ACTION_MSE_W * loss_action

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

                epoch_action_loss += loss_action.item() * len(batch_indices)
                epoch_total_steps += len(batch_indices)
                global_step += 1

        # ---- epoch 结束 ----
        scheduler.step()
        avg_action_loss = epoch_action_loss / max(epoch_total_steps, 1)
        current_lr = scheduler.get_last_lr()[0]

        # 记录日志
        writer.add_scalar('Distill/Action_MSE_Loss', avg_action_loss, epoch)
        writer.add_scalar('Distill/LR', current_lr, epoch)

        # 打印进度
        if epoch % DISTILL_LOG_INTERVAL == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{DISTILL_EPOCHS} | "
                  f"Action MSE: {avg_action_loss:.6e} | "
                  f"LR: {current_lr:.2e}")

        # 保存最佳模型
        if avg_action_loss < best_loss:
            best_loss = avg_action_loss
            student_save_path = os.path.join(MODEL_DIR, "student_best.pth")
            torch.save(student.state_dict(), student_save_path)
            print(f"  >>> 保存最佳学生模型: {student_save_path} (loss={avg_action_loss:.6e})")

        # 定期保存
        if epoch % DISTILL_SAVE_INTERVAL == 0:
            student_save_path = os.path.join(MODEL_DIR, f"student_epoch{epoch}.pth")
            torch.save(student.state_dict(), student_save_path)
            print(f"  >>> 检查点保存: {student_save_path}")

    # ---- 6. 训练结束 ----
    student_save_path = os.path.join(MODEL_DIR, "student_final.pth")
    torch.save(student.state_dict(), student_save_path)
    print(f"\n{'=' * 60}")
    print(f"蒸馏训练完成!")
    print(f"最佳损失: {best_loss:.6e}")
    print(f"学生模型已保存到: {student_save_path}")
    print(f"{'=' * 60}")

    # 评估: 比较教师和学生输出
    print("\n运行评估: 比较教师和学生网络输出...")
    evaluate(student, teacher, all_targets, device)

    writer.close()


def evaluate(student, teacher, all_targets, device):
    """
    评估学生网络与教师网络的输出差异.

    在随机采样的 episode 上对比两个网络的输出动作.
    """
    student.eval()
    n_eval_episodes = 10
    total_action_mse = 0.0
    total_action_mae = 0.0
    total_steps = 0

    for ep in range(n_eval_episodes):
        target_seq = sample_target_episode(
            all_targets,
            total_len=STEPS_PER_EPISODE + H,
            min_segment_len=MIN_SEGMENT_LENGTH,
            add_offset=True
        )

        for step in range(STEPS_PER_EPISODE):
            # 随机采样当前状态 (不使用仿真环境)
            theta = random.uniform(-math.pi, math.pi)
            omega = random.gauss(OMEGA_MEAN, OMEGA_STD)

            state = build_state(theta, omega, target_seq, step, H)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                teacher_action_t, _, _ = teacher(
                    state_tensor, deterministic=True, with_log_prob=False
                )
                student_action_t = student(state_tensor)

            # 直接比较 (不经过环境)
            diff = student_action_t - teacher_action_t
            total_action_mse += (diff ** 2).sum().item()
            total_action_mae += diff.abs().sum().item()
            total_steps += 1

    avg_mse = total_action_mse / max(total_steps, 1)
    avg_mae = total_action_mae / max(total_steps, 1)
    print(f"  评估 {n_eval_episodes} episodes, {total_steps} 步")
    print(f"  学生 vs 教师 动作 MSE: {avg_mse:.6e}")
    print(f"  学生 vs 教师 动作 MAE: {avg_mae:.6e}")

    student.train()


if __name__ == "__main__":
    distill()
