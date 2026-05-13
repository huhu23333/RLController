# control_nn.py - 使用训练好的神经网络进行控制 (替换 MPC)
# 界面与 test2.py 完全一致, 仅将 MPC 替换为神经网络推理

import pygame
import sys
import math
import numpy as np
import os
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'test'))
from SimEnv import SimpleYawSimEnv

from config import (
    MODEL_DIR, DT_ENV, DT_CTRL, STEPS_PER_CTRL,
    J, TAU_C, TAU_S, OMEGA_S, B, ANGLE_LIMIT, DISTURBANCE_TORQUE,
    U_MIN, U_MAX, H,
    STATE_DIM, TARGET_DIM, INPUT_DIM, ACTION_DIM, HIDDEN_DIM,
)
from networks import Actor, SmallActor


# ================== 窗口参数 (与 test2.py 完全一致) ==================
ORIGIN_WIDTH, ORIGIN_HEIGHT = 1600, 1200
CURVE_WIDTH = 800
WIDTH = ORIGIN_WIDTH + CURVE_WIDTH
HEIGHT = ORIGIN_HEIGHT
CENTER_X = ORIGIN_WIDTH // 2
CENTER_Y = HEIGHT // 2
B_ARROW = 300
ARROW_LEN = 500
CURVE_RECT = pygame.Rect(ORIGIN_WIDTH, 0, CURVE_WIDTH, HEIGHT)

DELAY_TIME = 0.2
DELAY_STEPS = int(DELAY_TIME / DT_CTRL)


class CurvePlotter:
    """曲线绘图器 (与 test2.py 相同)"""
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

        pygame.draw.lines(self.screen, (0, 200, 0), False,
                          [(index_to_x(i), angle_to_y(angles_list[i]))
                           for i in range(len(angles_list))], 2)
        pygame.draw.lines(self.screen, (200, 50, 50), False,
                          [(index_to_x(i), angle_to_y(targets_list[i]))
                           for i in range(len(targets_list))], 2)
        pygame.draw.lines(self.screen, (240, 220, 60), False,
                          [(index_to_x(i), angle_to_y(errors_list[i]))
                           for i in range(len(errors_list))], 2)
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


def load_actor(model_path=None):
    """加载训练好的 Actor 网络 (教师网络)"""
    import torch
    if model_path is None:
        # 查找最新的模型文件
        if os.path.exists(MODEL_DIR):
            actors = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith('actor_')])
            if actors:
                model_path = os.path.join(MODEL_DIR, actors[-1])
                print(f"自动选择模型: {model_path}")
            else:
                print(f"错误: 在 {MODEL_DIR} 未找到模型文件")
                return None
        else:
            print(f"错误: 模型目录不存在 {MODEL_DIR}")
            return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actor = Actor(INPUT_DIM, ACTION_DIM, HIDDEN_DIM, U_MIN, U_MAX).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()
    print(f"教师模型已加载: {model_path} (设备: {device})")
    print(f"参数量: {sum(p.numel() for p in actor.parameters()):,}")
    return actor


def load_student(model_path=None):
    """加载蒸馏后的学生网络 (SmallActor)"""
    import torch
    if model_path is None:
        # 查找最佳学生模型
        if os.path.exists(MODEL_DIR):
            candidates = [f for f in os.listdir(MODEL_DIR)
                          if f.startswith('student_') and f.endswith('.pth')]
            # 优先选择 student_best.pth
            if 'student_best.pth' in candidates:
                model_path = os.path.join(MODEL_DIR, 'student_best.pth')
            elif candidates:
                candidates.sort()
                model_path = os.path.join(MODEL_DIR, candidates[-1])
                print(f"未找到 student_best.pth, 自动选择: {model_path}")
            else:
                print(f"错误: 在 {MODEL_DIR} 未找到学生模型文件")
                return None
        else:
            print(f"错误: 模型目录不存在 {MODEL_DIR}")
            return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    student = SmallActor(INPUT_DIM, ACTION_DIM, u_min=U_MIN, u_max=U_MAX).to(device)
    student.load_state_dict(torch.load(model_path, map_location=device))
    student.eval()
    print(f"学生模型已加载: {model_path} (设备: {device})")
    print(f"参数量: {sum(p.numel() for p in student.parameters()):,}")
    return student


def get_chinese_font(size):
    """获取支持中文的字体，如果找不到则回退到默认字体"""
    # 常见中文字体路径 (按优先级排序)
    font_candidates = [
        # Noto Sans CJK (Linux)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        # WenQuanYi (Linux)
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        # Windows 常见中文字体
        "C:\\Windows\\Fonts\\msyh.ttc",        # Microsoft YaHei
        "C:\\Windows\\Fonts\\simhei.ttf",      # SimHei
        "C:\\Windows\\Fonts\\simsun.ttc",      # SimSun
        # macOS 常见中文字体
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                font = pygame.font.Font(font_path, size)
                # 验证是否能渲染中文
                test_surf = font.render("中文测试", True, (255, 255, 255))
                if test_surf.get_width() > 0:
                    return font
            except Exception:
                continue

    # 如果都不行，尝试用 SysFont 按字体名称查找
    chinese_font_names = [
        "Noto Sans CJK SC", "Noto Sans CJK",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Microsoft YaHei", "SimHei", "SimSun",
        "PingFang SC", "Heiti SC", "STHeiti",
    ]
    for name in chinese_font_names:
        try:
            font = pygame.font.SysFont(name, size)
            test_surf = font.render("中文测试", True, (255, 255, 255))
            if test_surf.get_width() > 10:  # 宽度 > 10 说明确实渲染出了文字
                return font
        except Exception:
            continue

    # 最后的回退：使用默认字体（可能无法显示中文，但至少程序不会崩溃）
    print("警告: 未找到支持中文的字体，中文可能显示为方框")
    return pygame.font.Font(None, size)


def inference_with_model(model, nn_state, is_student=False):
    """
    统一推理接口.
    教师网络使用 get_action(state, deterministic=True)
    学生网络使用 get_action(state) (确定性输出)
    """
    if is_student:
        return model.get_action(nn_state)
    else:
        return model.get_action(nn_state, deterministic=True)


def build_nn_state(theta, omega, target_buffer, H_val):
    """
    构建神经网络的输入状态向量 - 使用连续化差值序列.

    使用 target_buffer 中最近的目标序列作为未来 H 步的预测.

    Returns:
        state: np.array [STATE_DIM + H_val]
        结构: [ω, y_0, y_1, ..., y_{H-1}]
    """
    state = np.zeros(STATE_DIM + H_val, dtype=np.float32)
    state[0] = omega

    # 从 target_buffer 提取目标值 (按时间排序, 取最近的一段)
    if len(target_buffer) > 0:
        # target_buffer 按时间先后排列, 提取目标值列表
        targets = [tgt for _, tgt in target_buffer]
        # 取最近的 min(len, H) 个目标
        recent_targets = targets[-H_val:] if len(targets) > H_val else targets
    else:
        recent_targets = [0.0]
        targets = [0.0]

    # 计算连续化差值序列
    for k in range(H_val):
        if k < len(recent_targets):
            tgt = recent_targets[k]
        elif len(recent_targets) > 0:
            tgt = recent_targets[-1]
        else:
            tgt = 0.0

        raw_y = (tgt - theta + math.pi) % (2 * math.pi) - math.pi

        if k == 0:
            y_cont = raw_y
        else:
            y_prev = state[STATE_DIM + k - 1]
            delta = (raw_y - y_prev + math.pi) % (2 * math.pi) - math.pi
            y_cont = y_prev + delta

        state[STATE_DIM + k] = y_cont

    return state


def main():
    import torch
    import argparse

    # ---- 命令行参数解析 ----
    parser = argparse.ArgumentParser(description="神经网络控制界面 (支持模型切换)")
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='教师模型路径 (默认自动选择 actor_best.pth)')
    parser.add_argument('--student', '-s', type=str, default=None,
                        help='学生模型路径 (默认自动选择 student_best.pth)')
    parser.add_argument('--use-student', action='store_true', default=False,
                        help='启动时使用学生模型 (默认使用教师模型)')
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Neural Network Control (SAC) - 支持模型切换")
    clock = pygame.time.Clock()
    font = get_chinese_font(30)
    small_font = get_chinese_font(24)

    # 加载教师网络
    actor = load_actor(args.model)
    if actor is None:
        print("无法加载教师模型, 退出.")
        return
    device = next(actor.parameters()).device

    # 加载学生网络 (蒸馏模型)
    student = load_student(args.student)

    # 当前使用的模型 (默认教师, 或根据命令行参数选择)
    use_student = args.use_student and student is not None
    current_model = student if use_student else actor
    is_student_mode = use_student

    # 初始化仿真环境
    env = SimpleYawSimEnv(dt=DT_ENV, J=0.01)
    curve_plotter = CurvePlotter(screen, CURVE_RECT, DT_CTRL,
                                 init_time_range=5.0, init_angle_range=3.14)

    total_time = 0.0
    next_ctrl_time = DT_CTRL
    action = np.array([0.0])
    running = True

    # 目标延迟缓冲 (与 test2.py 相同)
    target_buffer = deque()
    delayed_target = 0.0
    last_valid_real_target = 0.0

    last_ctrl_call_time = time.time()
    sleep_time = 0.0
    nn_inference_time = 0.0  # 神经网络推理耗时

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    curve_plotter.modify_time_range(0.5)
                elif event.key == pygame.K_MINUS:
                    curve_plotter.modify_time_range(-0.5)
                elif event.key == pygame.K_LEFTBRACKET:
                    curve_plotter.modify_angle_range(-0.2)
                elif event.key == pygame.K_RIGHTBRACKET:
                    curve_plotter.modify_angle_range(0.2)
                elif event.key == pygame.K_d:
                    # 按 'd' 键切换教师/学生模型
                    if student is not None:
                        is_student_mode = not is_student_mode
                        current_model = student if is_student_mode else actor
                        mode_name = "学生模型 (蒸馏)" if is_student_mode else "教师模型 (原始SAC)"
                        print(f"\n>>> 已切换到 {mode_name}")
                        if is_student_mode:
                            param_count = sum(p.numel() for p in student.parameters())
                        else:
                            param_count = sum(p.numel() for p in actor.parameters())
                        print(f"    参数量: {param_count:,}")
                    else:
                        print("\n学生模型未加载, 无法切换. 请先运行 distill.py 进行蒸馏训练.")

        # ----- 获取鼠标目标 -----
        mouse_x, _ = pygame.mouse.get_pos()
        if 0 <= mouse_x < ORIGIN_WIDTH:
            dx = mouse_x - CENTER_X
            real_target = -dx/B_ARROW*1.5 # math.atan2(-dx, B_ARROW)
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
            while len(target_buffer) > DELAY_STEPS + 1:
                target_buffer.popleft()
                #delayed_target = target_buffer.popleft()[1]

            # 计算延迟目标
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

            # ---- 神经网络控制 (可切换教师/学生) ----
            nn_state = build_nn_state(env.theta, env.omega, target_buffer, H)

            t_infer_start = time.perf_counter()
            action_np = inference_with_model(current_model, nn_state, is_student_mode)
            nn_inference_time = time.perf_counter() - t_infer_start

            # 应用第一个力矩
            tau_nn = float(action_np[0])
            action = np.array([np.clip(tau_nn, U_MIN, U_MAX)])

            # 曲线数据
            theta_wrapped = math.atan2(math.sin(env.theta), math.cos(env.theta))
            curve_plotter.add_point(theta_wrapped, delayed_target)

            # 渲染
            screen.fill((20, 20, 20))
            draw_arrow(screen, font, env.theta, delayed_target, total_time)
            curve_plotter.draw()

            # 状态文字
            model_name = "学生模型 (蒸馏)" if is_student_mode else "教师模型 (原始SAC)"
            param_count = sum(p.numel() for p in current_model.parameters())
            status_lines = [
                f"Control: {model_name}  [按 D 键切换]",
                f"参数量: {param_count:,}  H={H} (horizon: {H*DT_CTRL:.2f}s)",
                f"Torque: {tau_nn:.4f}  NN Inference: {nn_inference_time*1000:.2f}ms",
                f"Sleep: {sleep_time:.4f}s/frame ({(sleep_time/DT_CTRL)*100:.1f}% Free)",
            ]
            for i, line in enumerate(status_lines):
                color = (100, 200, 255) if i == 0 else (255, 255, 255)
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
