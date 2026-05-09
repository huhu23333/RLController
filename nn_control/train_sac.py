# train_sac.py - SAC 强化学习训练脚本
# 从录制数据中采样目标序列片段拼接, 训练神经网络替代 MPC 控制器
# 使用 TensorBoard 记录训练过程

import torch
import numpy as np
import os
import sys
import math
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'test'))
from SimEnv import SimpleYawSimEnv

from config import (
    DATA_DIR, MODEL_DIR, LOG_DIR,
    DT_ENV, DT_CTRL, STEPS_PER_CTRL,
    J, TAU_C, TAU_S, OMEGA_S, B, ANGLE_LIMIT, DISTURBANCE_TORQUE,
    U_MIN, U_MAX, H,
    STATE_DIM, TARGET_DIM, INPUT_DIM, ACTION_DIM, HIDDEN_DIM,
    LR_ACTOR, LR_CRITIC, LR_ALPHA,
    GAMMA, TAU, BATCH_SIZE, REPLAY_SIZE, TARGET_ENTROPY,
    TOTAL_EPISODES, STEPS_PER_EPISODE, MIN_SEGMENT_LENGTH,
    GRADIENT_STEPS, START_TRAIN_AFTER, SAVE_INTERVAL, LOG_INTERVAL,
    REWARD_TRACK_W, REWARD_TORQUE_W,
)
from sac_agent import SACAgent
from replay_buffer import ReplayBuffer


def angle_error(a, b):
    """计算两个角度之间的最小误差 (弧度, [-π, π])"""
    err = a - b
    return (err + math.pi) % (2 * math.pi) - math.pi


def wrap_angle(a):
    """将角度标准化到 [-π, π]"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def load_all_target_sequences():
    """
    加载 DATA_DIR 下所有录制数据文件, 返回目标序列列表.
    每个序列是一个 numpy 数组, shape = (N,).
    """
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
                # 只保留足够长的序列
                all_targets.append(targets)
                print(f"  加载 {f}: {len(targets)} 步 ({len(targets)*DT_CTRL:.1f}s)")
        except Exception as e:
            print(f"  加载 {f} 失败: {e}")

    print(f"共加载 {len(all_targets)} 条目标序列")
    return all_targets


def sample_target_episode(all_targets, total_len, min_segment_len, add_offset=True):
    """
    从目标序列库中随机采样片段并拼接, 构成一个训练 episode 的目标序列.

    Args:
        all_targets: 目标序列列表
        total_len:   拼接后的总长度 (控制步数)
        min_segment_len: 每个片段的最小长度
        add_offset:  是否添加随机角度偏移

    Returns:
        target_seq: np.array [total_len] 目标角度序列
    """
    # 随机采样片段直到达到所需长度
    segments = []
    remaining = total_len

    while remaining > 0:
        # 随机选择一条序列
        seq_idx = random.randint(0, len(all_targets) - 1)
        seq = all_targets[seq_idx]

        # 从该序列中随机采样一段, 长度至少 min_segment_len (但不超过剩余长度)
        max_seg_len = min(len(seq), remaining)
        # 确保最小片段长度, 但也要考虑这是否是最后一段
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

    # 拼接所有片段
    target_seq = np.concatenate(segments)[:total_len]

    # 添加随机角度偏移
    if add_offset:
        offset = random.uniform(-math.pi, math.pi)
        target_seq = target_seq + offset

    return target_seq.astype(np.float32)


def build_state(theta, omega, target_seq, idx, H_val):
    """
    构建 SAC 的输入状态向量.

    Args:
        theta, omega: 当前状态
        target_seq:   完整目标序列
        idx:          当前步索引 (从0开始), idx 对应的时间步为 t
        H_val:        预测时域

    Returns:
        state: np.array [STATE_DIM + 2*H_val]
        结构: [sin(θ), cos(θ), ω, sin(r_{t+1}), cos(r_{t+1}), ..., sin(r_{t+H}), cos(r_{t+H})]
    """
    # 当前状态部分
    state_vec = np.zeros(STATE_DIM + 2 * H_val, dtype=np.float32)
    state_vec[0] = math.sin(theta)
    state_vec[1] = math.cos(theta)
    state_vec[2] = omega

    # 未来目标序列 (sin/cos 对)
    for k in range(H_val):
        tgt_idx = idx + k + 1  # t+1, t+2, ..., t+H
        if tgt_idx < len(target_seq):
            tgt = float(target_seq[tgt_idx])
        else:
            tgt = float(target_seq[-1])  # 超出时使用最后一个目标

        state_vec[STATE_DIM + 2 * k]     = math.sin(tgt)
        state_vec[STATE_DIM + 2 * k + 1] = math.cos(tgt)

    return state_vec


class TrainingEnv:
    """用于训练的仿真环境封装"""
    def __init__(self):
        self.env = SimpleYawSimEnv(
            dt=DT_ENV, J=J, tau_c=TAU_C, tau_s=TAU_S,
            omega_s=OMEGA_S, b=B,
            angle_limit=ANGLE_LIMIT,
            disturbance_torque=DISTURBANCE_TORQUE
        )

    def reset(self, theta=0.0, omega=0.0):
        self.env.theta = theta
        self.env.omega = omega

    def step_n(self, torques, n_steps):
        """
        执行 n 个控制步 (每个控制步包含 STEPS_PER_CTRL 个 sub-steps).
        torques: 长度为 n_steps 的力矩数组
        返回: (theta_sequence, omega_sequence)
        """
        thetas = []
        omegas = []
        for i in range(n_steps):
            tau = float(torques[i])
            # 每个控制步内进行 STEPS_PER_CTRL 次环境积分
            for _ in range(STEPS_PER_CTRL):
                self.env.step(np.array([tau]))
            thetas.append(self.env.theta)
            omegas.append(self.env.omega)
        return np.array(thetas), np.array(omegas)

    def step_one(self, torque):
        """执行 1 个控制步"""
        for _ in range(STEPS_PER_CTRL):
            self.env.step(np.array([float(torque)]))
        return self.env.theta, self.env.omega


def train():
    """主训练循环"""
    # 检查 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载目标序列
    print("加载目标序列数据...")
    all_targets = load_all_target_sequences()
    if not all_targets:
        print("错误: 没有找到训练数据! 请先运行 collect_data.py 采集数据.")
        return

    # 初始化
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    agent = SACAgent(
        input_dim=INPUT_DIM, action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, lr_alpha=LR_ALPHA,
        gamma=GAMMA, tau=TAU, target_entropy=TARGET_ENTROPY,
        u_min=U_MIN, u_max=U_MAX, device=device
    )

    replay_buffer = ReplayBuffer(REPLAY_SIZE)
    train_env = TrainingEnv()
    writer = SummaryWriter(LOG_DIR)

    total_transitions = 0
    best_episode_reward = -float('inf')

    print(f"开始训练: {TOTAL_EPISODES} episodes, "
          f"每 episode {STEPS_PER_EPISODE} 步, H={H}")
    print(f"输入维度: {INPUT_DIM}, 动作维度: {ACTION_DIM}")

    for episode in range(1, TOTAL_EPISODES + 1):
        # ---- 采样目标序列 ----
        # 需要 STEPS_PER_EPISODE + H 个目标 (最后 H 个用于 lookahead)
        target_seq = sample_target_episode(
            all_targets,
            total_len=STEPS_PER_EPISODE + H,
            min_segment_len=MIN_SEGMENT_LENGTH,
            add_offset=True
        )

        # ---- 初始化环境 ----
        # 初始状态: 追踪第一个目标
        init_theta = wrap_angle(float(target_seq[0]) + random.uniform(-0.5, 0.5))
        init_omega = random.uniform(-0.1, 0.1)
        train_env.reset(init_theta, init_omega)
        theta = init_theta
        omega = init_omega

        episode_reward = 0.0
        episode_errors = []
        episode_torques = []

        # ---- Episode 循环 ----
        for step in range(STEPS_PER_EPISODE):
            # 构建状态
            state = build_state(theta, omega, target_seq, step, H)

            # 选择动作
            action = agent.select_action(state, deterministic=False)

            # 应用第一个力矩, 仿真 1 个控制步
            torque = float(action[0])
            theta_new, omega_new = train_env.step_one(torque)
            theta_new = wrap_angle(theta_new)

            # 计算奖励
            target_current = float(target_seq[step + 1])  # 当前步的目标
            err = angle_error(theta_new, target_current)
            reward = -(REWARD_TRACK_W * err ** 2) - (REWARD_TORQUE_W * torque ** 2)

            # 构建下一状态 (用下一时刻的位置看未来 H 步目标)
            next_state = build_state(theta_new, omega_new, target_seq, step + 1, H)

            # 判断是否结束 (超出范围或达到末尾)
            done = (step >= STEPS_PER_EPISODE - 1)

            # 存入经验回放
            replay_buffer.add(state, action, reward, next_state, done)

            # 更新状态
            theta = theta_new
            omega = omega_new
            episode_reward += reward
            episode_errors.append(abs(err))
            episode_torques.append(torque)

            total_transitions += 1

        # ---- SAC 更新 ----
        if len(replay_buffer) >= START_TRAIN_AFTER:
            update_info_list = []
            for _ in range(GRADIENT_STEPS):
                info = agent.update(replay_buffer, BATCH_SIZE)
                if info is not None:
                    update_info_list.append(info)

            # 记录日志
            if update_info_list and episode % LOG_INTERVAL == 0:
                # 取平均值
                avg_info = {k: np.mean([d[k] for d in update_info_list])
                           for k in update_info_list[0].keys()}

                writer.add_scalar('Loss/Critic1', avg_info['critic1_loss'], episode)
                writer.add_scalar('Loss/Critic2', avg_info['critic2_loss'], episode)
                writer.add_scalar('Loss/Actor', avg_info['actor_loss'], episode)
                writer.add_scalar('Loss/Alpha', avg_info['alpha_loss'], episode)
                writer.add_scalar('Params/Alpha', avg_info['alpha'], episode)
                writer.add_scalar('Params/Q1_mean', avg_info['q1_mean'], episode)
                writer.add_scalar('Params/Q2_mean', avg_info['q2_mean'], episode)

        # ---- 记录 episode 统计 ----
        mean_error = np.mean(episode_errors) if episode_errors else 0.0
        mean_torque = np.mean(np.abs(episode_torques)) if episode_torques else 0.0

        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/MeanError_rad', mean_error, episode)
        writer.add_scalar('Episode/MeanTorque', mean_torque, episode)
        writer.add_scalar('Episode/BufferSize', len(replay_buffer), episode)

        # ---- 打印进度 ----
        if episode % LOG_INTERVAL == 0:
            print(f"Ep {episode:5d}/{TOTAL_EPISODES} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Err: {mean_error:.4f} rad | "
                  f"Torque: {mean_torque:.4f} | "
                  f"Alpha: {agent.alpha:.3f} | "
                  f"Buffer: {len(replay_buffer)}")

        # ---- 保存最佳模型 ----
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            agent.save(MODEL_DIR, 'best')

        # ---- 定期保存 ----
        if episode % SAVE_INTERVAL == 0:
            agent.save(MODEL_DIR, episode)

    # ---- 训练结束 ----
    agent.save(MODEL_DIR, 'final')
    writer.close()
    print(f"\n训练完成! 最佳 episode 奖励: {best_episode_reward:.2f}")
    print(f"模型已保存到 {MODEL_DIR}")


if __name__ == "__main__":
    train()
