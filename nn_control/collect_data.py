# collect_data.py - 数据采集脚本 (MPC控制 + 手动鼠标输入)
# 功能: 录制目标序列并保存, 或从文件读取目标序列回放
# 操作: 鼠标控制目标方向, 空格键开始/停止录制, S 保存, L 加载文件回放, ESC 退出

import pygame
import sys
import math
import numpy as np
import os
import time
from collections import deque

# 导入父目录 test 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'test'))
from SimEnv import SimpleYawSimEnv
from mpc_cpp.mpc_wrapper import MPCController
from PID import PIDController

from config import (
    DATA_DIR, DT_ENV, DT_CTRL, STEPS_PER_CTRL,
    J, TAU_C, TAU_S, OMEGA_S, B, ANGLE_LIMIT, DISTURBANCE_TORQUE,
    U_MIN, U_MAX, H,
)

# ================== 窗口参数 (与 test2.py 一致) ==================
ORIGIN_WIDTH, ORIGIN_HEIGHT = 1600, 1200
CURVE_WIDTH = 800
WIDTH = ORIGIN_WIDTH + CURVE_WIDTH
HEIGHT = ORIGIN_HEIGHT
CENTER_X = ORIGIN_WIDTH // 2
CENTER_Y = HEIGHT // 2
B_ARROW = 300
ARROW_LEN = 500
CURVE_RECT = pygame.Rect(ORIGIN_WIDTH, 0, CURVE_WIDTH, HEIGHT)

DELAY_TIME = 0.1
DELAY_STEPS = int(DELAY_TIME / DT_CTRL)


class CurvePlotter:
    """曲线绘图器 (与 test2.py 完全相同)"""
    def __init__(self, screen, rect, dt, init_time_range=5.0, init_angle_range=3.14):
        self.screen = screen
        self.rect = rect
        self.dt = dt
        self.time_range = init_time_range
        self.angle_range = init_angle_range
        max_len = int(init_time_range / dt) + 100
        self.angles = deque(maxlen=max_len)
        self.targets = deque(maxlen=max_len)
        self.errors = deque(maxlen=max_len)
        self.font = pygame.font.SysFont("Consolas", 16)

    def add_point(self, angle_wrapped, target_angle):
        error = target_angle - angle_wrapped
        error = (error + math.pi) % (2 * math.pi) - math.pi
        self.angles.append(angle_wrapped)
        self.targets.append(target_angle)
        self.errors.append(error)
        needed_len = int(self.time_range / self.dt) + 10
        if self.angles.maxlen < needed_len:
            self.angles = deque(self.angles, maxlen=needed_len)
            self.targets = deque(self.targets, maxlen=needed_len)
            self.errors = deque(self.errors, maxlen=needed_len)

    def modify_time_range(self, delta):
        new_range = self.time_range + delta
        if 0.5 <= new_range <= 20.0:
            self.time_range = new_range
            new_len = int(self.time_range / self.dt) + 10
            self.angles = deque(self.angles, maxlen=new_len)
            self.targets = deque(self.targets, maxlen=new_len)
            self.errors = deque(self.errors, maxlen=new_len)

    def modify_angle_range(self, delta):
        new_range = self.angle_range + delta
        if 0.2 <= new_range <= math.pi:
            self.angle_range = new_range

    def draw(self):
        pygame.draw.rect(self.screen, (30, 30, 40), self.rect)
        pygame.draw.rect(self.screen, (100, 100, 120), self.rect, 2)
        plot_rect = self.rect.inflate(-40, -40)
        plot_rect.x += 20
        plot_rect.y += 20
        if plot_rect.width <= 0 or plot_rect.height <= 0:
            return
        n_h_lines = 5
        for i in range(n_h_lines + 1):
            y_ratio = i / n_h_lines
            y = plot_rect.bottom - y_ratio * plot_rect.height
            angle_val = -self.angle_range + 2 * self.angle_range * y_ratio
            pygame.draw.line(self.screen, (60, 60, 70),
                             (plot_rect.left, y), (plot_rect.right, y), 1)
            label = self.font.render(f"{angle_val:.1f}", True, (180, 180, 200))
            self.screen.blit(label, (plot_rect.left - 35, y - 5))
        n_v_lines = 6
        for i in range(n_v_lines + 1):
            x_ratio = i / n_v_lines
            x = plot_rect.left + x_ratio * plot_rect.width
            t_val = self.time_range * (1 - x_ratio)
            pygame.draw.line(self.screen, (60, 60, 70),
                             (x, plot_rect.top), (x, plot_rect.bottom), 1)
            if i % 2 == 0:
                label = self.font.render(f"{t_val:.1f}s", True, (180, 180, 200))
                self.screen.blit(label, (x - 20, plot_rect.bottom + 5))
        n_points = int(self.time_range / self.dt)
        angles_list = list(self.angles)[-n_points:]
        targets_list = list(self.targets)[-n_points:]
        errors_list = list(self.errors)[-n_points:]
        if len(angles_list) < 2:
            return

        def angle_to_y(angle):
            ratio = (angle + self.angle_range) / (2 * self.angle_range)
            ratio = max(0.0, min(1.0, ratio))
            return plot_rect.bottom - ratio * plot_rect.height

        def index_to_x(idx):
            ratio = idx / (len(angles_list) - 1)
            return plot_rect.left + ratio * plot_rect.width

        points_angle = [(index_to_x(i), angle_to_y(angles_list[i]))
                        for i in range(len(angles_list))]
        pygame.draw.lines(self.screen, (0, 200, 0), False, points_angle, 2)
        points_target = [(index_to_x(i), angle_to_y(targets_list[i]))
                         for i in range(len(targets_list))]
        pygame.draw.lines(self.screen, (200, 50, 50), False, points_target, 2)
        points_error = [(index_to_x(i), angle_to_y(errors_list[i]))
                        for i in range(len(errors_list))]
        pygame.draw.lines(self.screen, (240, 220, 60), False, points_error, 2)
        legend_y = self.rect.top + 10
        for color, text in [((0, 200, 0), "Actual"), ((200, 50, 50), "Target"),
                             ((240, 220, 60), "Error")]:
            pygame.draw.rect(self.screen, color, (self.rect.right - 70, legend_y, 12, 12))
            label = self.font.render(text, True, (220, 220, 220))
            self.screen.blit(label, (self.rect.right - 55, legend_y - 2))
            legend_y += 18
        help_text = self.font.render(
            f"TimeRange:{self.time_range:.1f}s  AngleRange:{self.angle_range:.1f}rad",
            True, (200, 200, 200))
        self.screen.blit(help_text, (self.rect.x + 10, self.rect.bottom - 45))
        help2 = self.font.render("Keys: +/- : TimeRange  [ / ] : AngleRange", True, (150, 150, 150))
        self.screen.blit(help2, (self.rect.x + 10, self.rect.bottom - 30))


def draw_arrow(screen, font, angle, delayed_target, total_time):
    """绘制箭靶和文字"""
    line_y = CENTER_Y - B_ARROW
    pygame.draw.line(screen, (180, 180, 180), (0, line_y), (ORIGIN_WIDTH, line_y), 1)
    end_x = CENTER_X - ARROW_LEN * math.sin(angle)
    end_y = CENTER_Y - ARROW_LEN * math.cos(angle)
    pygame.draw.line(screen, (0, 255, 0), (CENTER_X, CENTER_Y), (end_x, end_y), 3)
    head_len = 12
    head_angle = math.pi / 7
    ang1 = angle + math.pi - head_angle
    ang2 = angle + math.pi + head_angle
    p1 = (end_x - head_len * math.sin(ang1), end_y - head_len * math.cos(ang1))
    p2 = (end_x - head_len * math.sin(ang2), end_y - head_len * math.cos(ang2))
    pygame.draw.polygon(screen, (0, 255, 0), [p1, (end_x, end_y), p2])
    target_x = CENTER_X - ARROW_LEN * math.sin(delayed_target)
    target_y = CENTER_Y - ARROW_LEN * math.cos(delayed_target)
    pygame.draw.circle(screen, (255, 60, 60), (int(target_x), int(target_y)), 6)
    time_surf = font.render(f"Time: {total_time:.4f} s", True, (255, 255, 255))
    angle_surf = font.render(
        f"Angle: {angle:.3f}  Delayed Target: {delayed_target:.3f}", True, (255, 255, 255))
    screen.blit(time_surf, (10, 10))
    screen.blit(angle_surf, (10, 50))


def create_mpc():
    """创建 MPC 控制器 (与 test2.py 参数一致)"""
    return MPCController(
        dt_control=DT_CTRL, dt_sim=DT_ENV,
        J=J, tau_c=TAU_C, tau_s=TAU_S, omega_s=OMEGA_S, b=B,
        angle_limit=ANGLE_LIMIT, disturbance_torque=DISTURBANCE_TORQUE,
        N=DELAY_STEPS, Q=5.0, R=0.01, Rd=0.1,
        u_min=U_MIN, u_max=U_MAX, max_iter=30, delay_par_n=3
    )


def save_recorded_data(filename, data):
    """保存录制数据到 .npz 文件"""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, filename)
    np.savez_compressed(filepath,
                        theta=data['theta'],
                        omega=data['omega'],
                        target=data['target'],
                        torque=data['torque'],
                        timestamps=data['timestamps'])
    print(f"数据已保存到 {filepath} ({len(data['timestamps'])} 条记录)")


def load_recorded_data(filename):
    """从 .npz 文件加载数据"""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return None, None
    data = np.load(filepath)
    target_seq = data['target']
    # 构建时间序列 -- 控制步数 * DT_CTRL 的间隔
    timestamps = np.arange(len(target_seq)) * DT_CTRL
    print(f"已从 {filepath} 加载数据 ({len(target_seq)} 条记录, "
          f"时长 {timestamps[-1]:.2f}s)")
    return target_seq, timestamps


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Data Collection - SPACE:Record  S:Save  L:Load  ESC:Quit")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)
    small_font = pygame.font.Font(None, 24)

    env = SimpleYawSimEnv(dt=DT_ENV)
    mpc = create_mpc()    
    pid = PIDController(
        Kp=5.0, Ki=50.0, Kd=0.3,
        integral_limit=None,
        integral_sep_threshold=0.2,
        derivative_filter_tc=None,
        output_limit=2.0,
        deadband=0.0
    )
    curve_plotter = CurvePlotter(screen, CURVE_RECT, DT_CTRL,
                                 init_time_range=5.0, init_angle_range=3.14)

    total_time = 0.0
    next_ctrl_time = DT_CTRL
    action = np.array([0.0])
    running = True

    # 目标延迟缓冲
    target_buffer = deque()
    delayed_target = 0.0
    last_valid_real_target = 0.0

    # ---- 录制状态 ----
    is_recording = False
    recorded_data = {'theta': [], 'omega': [], 'target': [], 'torque': [], 'timestamps': []}

    # ---- 回放状态 (从文件加载目标序列) ----
    playback_targets = None       # 加载的目标序列
    playback_timestamps = None    # 对应的时间戳
    playback_idx = 0              # 当前回放位置
    is_playback = False

    last_ctrl_call_time = time.time()
    sleep_time = 0.0

    while running:
        # ----- 事件处理 -----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # 空格键切换录制状态
                    if is_playback:
                        # 回放模式下, 空格键不控制录制
                        pass
                    else:
                        is_recording = not is_recording
                        if is_recording:
                            recorded_data = {'theta': [], 'omega': [], 'target': [],
                                            'torque': [], 'timestamps': []}
                            print(">>> 开始录制数据 <<<")
                        else:
                            print(f">>> 停止录制 (已录制 {len(recorded_data['timestamps'])} 条) <<<")
                elif event.key == pygame.K_s:
                    # S 键保存当前录制的数据
                    if len(recorded_data['timestamps']) > 0:
                        filename = f"recording_{int(time.time())}.npz"
                        save_recorded_data(filename, recorded_data)
                    else:
                        print("没有录制数据可保存")
                elif event.key == pygame.K_l:
                    # L 键加载文件进行回放
                    # 列出可用文件
                    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npz')] \
                            if os.path.exists(DATA_DIR) else []
                    if files:
                        print("可用数据文件:")
                        for i, f in enumerate(files):
                            print(f"  [{i}] {f}")
                        # 加载第一个文件 (简化: 始终加载最新的文件)
                        latest_file = sorted(files)[-1]
                        print(f"自动加载: {latest_file}")
                        targets, timestamps = load_recorded_data(latest_file)
                        if targets is not None:
                            playback_targets = targets
                            playback_timestamps = timestamps
                            playback_idx = 0
                            is_playback = True
                            is_recording = False
                            # 重置环境
                            env = SimpleYawSimEnv(dt=DT_ENV)
                            total_time = 0.0
                            next_ctrl_time = DT_CTRL
                            target_buffer.clear()
                            print(">>> 回放模式已启动 <<<")
                    else:
                        print(f"没有找到数据文件 (请先在 {DATA_DIR} 目录放置 .npz 文件)")
                elif event.key == pygame.K_r:
                    # R 键退出回放模式
                    if is_playback:
                        is_playback = False
                        playback_targets = None
                        env = SimpleYawSimEnv(dt=DT_ENV)
                        total_time = 0.0
                        next_ctrl_time = DT_CTRL
                        target_buffer.clear()
                        print(">>> 已退出回放模式 <<<")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    curve_plotter.modify_time_range(0.5)
                elif event.key == pygame.K_MINUS:
                    curve_plotter.modify_time_range(-0.5)
                elif event.key == pygame.K_LEFTBRACKET:
                    curve_plotter.modify_angle_range(-0.2)
                elif event.key == pygame.K_RIGHTBRACKET:
                    curve_plotter.modify_angle_range(0.2)

        # ----- 获取目标 -----
        if is_playback and playback_targets is not None:
            # 回放模式: 从文件读取目标
            idx = int(total_time / DT_CTRL) if total_time >= 0 else 0
            if idx < len(playback_targets):
                real_target = float(playback_targets[idx])
            else:
                real_target = last_valid_real_target
        else:
            # 手动模式: 鼠标输入
            mouse_x, _ = pygame.mouse.get_pos()
            if 0 <= mouse_x < ORIGIN_WIDTH:
                dx = mouse_x - CENTER_X
                real_target = math.atan2(-dx, B_ARROW)
                last_valid_real_target = real_target
            else:
                real_target = last_valid_real_target

        # ----- 环境步进 -----
        total_time += DT_ENV
        env.step(action)

        # ----- 控制周期 -----
        if total_time >= next_ctrl_time - 1e-12:
            current_time = total_time

            # 目标延迟缓冲
            target_buffer.append((current_time, real_target))
            while len(target_buffer) > mpc.N + 1:
                target_buffer.popleft()

            if target_buffer:
                target_time_ref = current_time - DELAY_TIME
                delayed_target = target_buffer[0][1]
                for t, val in target_buffer:
                    if t <= target_time_ref:
                        delayed_target = val
                    else:
                        break
            else:
                delayed_target = 0.0

            # MPC 控制
            # state = (env.theta, env.omega)
            # ref_list = [0.0] * (mpc.N - len(target_buffer) + 1) + \
            #            [target_buffer[-(i+1)][1] for i in reversed(range(len(target_buffer)))]
            # tau_opt = mpc.step(state, ref_list)
            # action = np.array([tau_opt])
            # PID 控制 (替代 MPC)
            error = delayed_target - env.theta
            error = (error + math.pi) % (2 * math.pi) - math.pi
            tau_opt = pid.step(np.array([error]), DT_CTRL)
            action = np.array([tau_opt])

            # 录制数据
            if is_recording and not is_playback:
                recorded_data['theta'].append(env.theta)
                recorded_data['omega'].append(env.omega)
                recorded_data['target'].append(real_target)
                recorded_data['torque'].append(tau_opt)
                recorded_data['timestamps'].append(current_time)

            # 曲线数据
            theta_wrapped = math.atan2(math.sin(env.theta), math.cos(env.theta))
            curve_plotter.add_point(theta_wrapped, delayed_target)

            # 渲染
            screen.fill((20, 20, 20))
            draw_arrow(screen, font, env.theta, delayed_target, total_time)
            curve_plotter.draw()

            # 状态提示
            status_lines = []
            status_lines.append(
                f"Sleep: {sleep_time:.4f}s/frame ({(sleep_time/DT_CTRL)*100:.1f}% Free)")
            if is_recording:
                status_lines.append(
                    f"[REC] Recording... ({len(recorded_data['timestamps'])} steps)")
            if is_playback:
                status_lines.append(
                    f"[PLAYBACK] Step {playback_idx}/{len(playback_targets)}")
            status_lines.append("Keys: SPACE=Record  S=Save  L=Load  R=StopPlayback  ESC=Quit")
            for i, line in enumerate(status_lines):
                color = (255, 100, 100) if "REC" in line else (255, 255, 255)
                surf = small_font.render(line, True, color)
                screen.blit(surf, (10, 90 + i * 22))

            pygame.display.flip()
            next_ctrl_time += DT_CTRL

            # 速率控制
            now = time.time()
            elapsed = now - last_ctrl_call_time
            if elapsed < DT_CTRL:
                sleep_time = DT_CTRL - elapsed
                time.sleep(DT_CTRL - elapsed)
            else:
                sleep_time = DT_CTRL - elapsed
            last_ctrl_call_time = time.time()

        clock.tick(0)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
