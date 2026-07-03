# test3.py
import pygame
import sys
import math
import numpy as np
from collections import deque
from SimEnv import SimpleYawSimEnv
# from MPC import MPCController
from mpc_cpp_simpler.mpc_wrapper import MPCController
import time

# ================== 窗口与区域参数 (与 test1 相同) ==================
ORIGIN_WIDTH, ORIGIN_HEIGHT = 1600, 1200
CURVE_WIDTH = 800
WIDTH = ORIGIN_WIDTH + CURVE_WIDTH
HEIGHT = ORIGIN_HEIGHT
CENTER_X = ORIGIN_WIDTH // 2
CENTER_Y = HEIGHT // 2
B = 300
ARROW_LEN = 500

CURVE_RECT = pygame.Rect(ORIGIN_WIDTH, 0, CURVE_WIDTH, HEIGHT)

# ================== 仿真与控制参数 ==================
DT_ENV = 1e-4                     # 环境步长
DT_MPC_SIM = 1e-3                 # MPC仿真步长
DT_CTRL = 1e-2                    # MPC控制周期 (与 test1 的 PID 周期相同)
DELAY_TIME = 0.1                  # 目标延迟时间 (秒)
DELAY_STEPS = int(DELAY_TIME / DT_CTRL)   # 延迟对应的控制周期数

# ================== 曲线绘图器类 (与 test1 相同) ==================
class CurvePlotter:
    """负责存储数据、管理显示范围、绘制三条动态曲线"""
    def __init__(self, screen, rect, dt_pid, init_time_range=5.0, init_angle_range=3.14):
        self.screen = screen
        self.rect = rect
        self.dt = dt_pid
        self.time_range = init_time_range
        self.angle_range = init_angle_range

        max_len = int(init_time_range / dt_pid) + 100
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

        # 水平网格
        n_h_lines = 5
        for i in range(n_h_lines + 1):
            y_ratio = i / n_h_lines
            y = plot_rect.bottom - y_ratio * plot_rect.height
            angle_val = -self.angle_range + 2 * self.angle_range * y_ratio
            pygame.draw.line(self.screen, (60, 60, 70),
                             (plot_rect.left, y), (plot_rect.right, y), 1)
            label = self.font.render(f"{angle_val:.1f}", True, (180, 180, 200))
            self.screen.blit(label, (plot_rect.left - 35, y - 5))

        # 垂直网格
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

        # 获取显示点
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

        # 实际角度 (绿色)
        points_angle = [(index_to_x(i), angle_to_y(angles_list[i]))
                        for i in range(len(angles_list))]
        pygame.draw.lines(self.screen, (0, 200, 0), False, points_angle, 2)

        # 目标角度 (红色)
        points_target = [(index_to_x(i), angle_to_y(targets_list[i]))
                         for i in range(len(targets_list))]
        pygame.draw.lines(self.screen, (200, 50, 50), False, points_target, 2)

        # 误差 (黄色)
        points_error = [(index_to_x(i), angle_to_y(errors_list[i]))
                        for i in range(len(errors_list))]
        pygame.draw.lines(self.screen, (240, 220, 60), False, points_error, 2)

        # 图例
        legend_y = self.rect.top + 10
        for color, text in [((0,200,0), "Actual"), ((200,50,50), "Target"), ((240,220,60), "Error")]:
            pygame.draw.rect(self.screen, color, (self.rect.right - 70, legend_y, 12, 12))
            label = self.font.render(text, True, (220, 220, 220))
            self.screen.blit(label, (self.rect.right - 55, legend_y - 2))
            legend_y += 18

        help_text = self.font.render(f"TimeRange:{self.time_range:.1f}s  AngleRange:{self.angle_range:.1f}rad", True, (200,200,200))
        self.screen.blit(help_text, (self.rect.x + 10, self.rect.bottom - 45))
        help2 = self.font.render("Keys: +/- : TimeRange  [ / ] : AngleRange", True, (150,150,150))
        self.screen.blit(help2, (self.rect.x + 10, self.rect.bottom - 30))


# ================== 左侧绘图函数 (显示延迟后的目标) ==================
def draw_original(screen, font, angle, delayed_target, total_time):
    """绘制箭靶和文字，目标角度使用延迟后的目标"""
    line_y = CENTER_Y - B
    pygame.draw.line(screen, (180, 180, 180), (0, line_y), (ORIGIN_WIDTH, line_y), 1)

    # 当前角度箭头 (绿色)
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

    # 目标方向指示点 (红色) - 使用延迟目标
    target_x = CENTER_X - ARROW_LEN * math.sin(delayed_target)
    target_y = CENTER_Y - ARROW_LEN * math.cos(delayed_target)
    pygame.draw.circle(screen, (255, 60, 60), (int(target_x), int(target_y)), 6)

    time_surf = font.render(f"Time: {total_time:.4f} s", True, (255, 255, 255))
    angle_surf = font.render(f"Angle: {angle:.3f}  Delayed Target: {delayed_target:.3f}", True, (255, 255, 255))
    screen.blit(time_surf, (10, 10))
    screen.blit(angle_surf, (10, 50))


# ================== 主函数 ==================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Yaw Control with MPC and 0.2s Target Delay")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)

    # 初始化仿真环境 (参数与 test1 相同)
    env = SimpleYawSimEnv(dt=DT_ENV)

    # 初始化 MPC 控制器 (使用环境相同的物理参数)
    mpc = MPCController(
        dt_control=DT_CTRL,
        dt_sim=DT_MPC_SIM,
        J=0.010092,#env.J,
        tau_c=0.207288,#env.tau_c,
        b=0.009696,#env.b,
        angle_limit=env.angle_limit,
        disturbance_torque=env.disturbance_torque,
        N=DELAY_STEPS,           # 预测时域 20步 = 0.2秒
        Q=5.0,          # 跟踪权重
        R=0.01,         # 控制量惩罚
        Rd=0.1,         # 增量惩罚
        u_min=-2.0,
        u_max=2.0,
        max_iter=30,
        delay_par_n=2
    )
    # mpc = MPCController(
    #     dt_control=DT_CTRL,
    #     dt_sim=DT_MPC_SIM,
    #     J=0.013384,#env.J,
    #     tau_c=0.011547,#env.tau_c,
    #     tau_s=0.214934,#env.tau_s,
    #     omega_s=0.029070,#,env.omega_s,
    #     b=0.012533,#env.b,
    #     angle_limit=env.angle_limit,
    #     disturbance_torque=-0.002348,#env.disturbance_torque,
    #     N=DELAY_STEPS,           # 预测时域 20步 = 0.2秒
    #     Q=5.0,          # 跟踪权重
    #     R=0.01,         # 控制量惩罚
    #     Rd=0.1,         # 增量惩罚
    #     u_min=-2.0,
    #     u_max=2.0,
    #     max_iter=30,
    #     delay_par_n=3
    # )

    # 曲线绘图器
    curve_plotter = CurvePlotter(screen, CURVE_RECT, DT_CTRL,
                                 init_time_range=5.0, init_angle_range=3.14)

    total_time = 0.0
    next_ctrl_time = DT_CTRL
    action = np.array([0.0])
    running = True

    # 目标延迟队列: 存储 (时间戳, 实时鼠标目标)
    target_buffer = deque()
    delayed_target = 0.0           # 当前延迟后的目标
    last_valid_real_target = 0.0   # 当鼠标移出左边区域时保持上一个有效实时目标

    # 主循环
    last_ctrl_call_time = time.time()
    sleep_time = 0.0

    while running:
        # ----- 事件处理 -----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    curve_plotter.modify_time_range(0.5)
                elif event.key == pygame.K_MINUS:
                    curve_plotter.modify_time_range(-0.5)
                elif event.key == pygame.K_LEFTBRACKET:
                    curve_plotter.modify_angle_range(-0.2)
                elif event.key == pygame.K_RIGHTBRACKET:
                    curve_plotter.modify_angle_range(0.2)

        # ----- 获取实时鼠标目标 (仅当鼠标位于左边区域时更新) -----
        mouse_x, _ = pygame.mouse.get_pos()
        if 0 <= mouse_x < ORIGIN_WIDTH:
            dx = mouse_x - CENTER_X
            real_target = -dx/B*1.5 # math.atan2(-dx, B)
            last_valid_real_target = real_target
        else:
            real_target = last_valid_real_target

        # ----- 环境步进 (以 DT_ENV 频率) -----
        total_time += DT_ENV
        env.step(action)

        # ----- 每个控制周期执行一次 MPC 计算和渲染 -----
        if total_time >= next_ctrl_time - 1e-12:
            current_time = total_time

            # 1. 将当前实时目标存入缓冲区 (带时间戳)
            target_buffer.append((current_time, real_target))

            # 2. 移除时间戳早于 current_time - DELAY_TIME 的旧数据
            while len(target_buffer) > mpc.N + 1:
                target_buffer.popleft()

            # 3. 获取延迟目标: 取缓冲区中时间最接近 current_time - DELAY_TIME 的值
            #    (通常就是第一个元素，因为缓冲区按时间排序)
            if target_buffer:
                # 找到第一个时间戳 >= 当前延迟时刻的元素，取前一个
                target_time = current_time - DELAY_TIME
                # 由于缓冲区是顺序的，直接取最后一个时间戳 ≤ target_time 的值
                delayed_target = target_buffer[0][1]  # 默认取最旧的
                for t, val in target_buffer:
                    if t <= target_time:
                        delayed_target = val
                    else:
                        break
            else:
                delayed_target = 0.0

            # 4. 获取当前状态
            state = (env.theta, env.omega)

            # 5. 调用 MPC 控制器 (使用延迟目标作为参考)
            tau_opt = mpc.step(state, 
                               ([0.0]*(mpc.N-len(target_buffer)+1))+[target_buffer[-(i+1)][1] for i in reversed(range(len(target_buffer)))]
                               )
            action = np.array([tau_opt])

            # 6. 为曲线添加数据点 (角度需 wrap 到 [-π, π])
            theta_wrapped = math.atan2(math.sin(env.theta), math.cos(env.theta))
            curve_plotter.add_point(theta_wrapped, delayed_target)

            # 7. 渲染
            screen.fill((20, 20, 20))
            draw_original(screen, font, env.theta, delayed_target, total_time)
            curve_plotter.draw()
            sleep_surf = font.render(f"Sleep Time: {sleep_time:.4f}s/frame ({(sleep_time/DT_CTRL)*100.0:.2f}% Free Time)", True, (255, 255, 255))
            screen.blit(sleep_surf, (10, 90))
            pygame.display.flip()

            next_ctrl_time += DT_CTRL

            # 简易速率控制 (尽量保证控制周期稳定)
            now = time.time()
            elapsed = now - last_ctrl_call_time
            if elapsed < DT_CTRL:
                sleep_time = DT_CTRL - elapsed
                time.sleep(DT_CTRL - elapsed)
            else:
                sleep_time = DT_CTRL - elapsed
            last_ctrl_call_time = time.time()

        # 限制 CPU 占用 (事件响应)
        clock.tick(0)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
