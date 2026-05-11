# test1.py
import pygame
import sys
import math
import numpy as np
from collections import deque
from SimEnv import SimpleYawSimEnv
from PID import PIDController
import time

# ================== 窗口与区域参数 ==================
ORIGIN_WIDTH, ORIGIN_HEIGHT = 1600, 1200          # 原图形区域大小
CURVE_WIDTH = 800                               # 曲线区域宽度
WIDTH = ORIGIN_WIDTH + CURVE_WIDTH              # 总宽度 = 1200
HEIGHT = ORIGIN_HEIGHT                          # 高度不变
CENTER_X = ORIGIN_WIDTH // 2                    # 原区域中心 x
CENTER_Y = HEIGHT // 2
B = 300                                         # 横线位置（像素）
ARROW_LEN = 500                                 # 箭头长度

# 曲线绘制区域（窗口右侧）
CURVE_RECT = pygame.Rect(ORIGIN_WIDTH, 0, CURVE_WIDTH, HEIGHT)

# ================== 仿真与控制参数 ==================
DT_ENV = 1e-2                                   # 环境步长
DT_PID = 1e-2                                   # PID 控制步长

# ================== 曲线绘图器类 ==================
class CurvePlotter:
    """负责存储数据、管理显示范围、绘制三条动态曲线"""
    def __init__(self, screen, rect, dt_pid, init_time_range=5.0, init_angle_range=3.14):
        """
        参数:
            screen: pygame.Surface 主窗口
            rect: pygame.Rect 曲线区域的位置和大小
            dt_pid: 数据点的时间间隔 (s)
            init_time_range: 初始横轴时间范围 (s)
            init_angle_range: 初始纵轴角度范围 (rad)，对称范围 [-range, range]
        """
        self.screen = screen
        self.rect = rect
        self.dt = dt_pid
        self.time_range = init_time_range      # 横轴显示的时间长度 (s)
        self.angle_range = init_angle_range    # 纵轴范围绝对值 (rad)

        # 数据缓存 (双端队列，自动丢弃旧数据)
        max_len = int(init_time_range / dt_pid) + 100   # 预留多余空间
        self.angles = deque(maxlen=max_len)            # 实际角度 (归一化到 [-π,π])
        self.targets = deque(maxlen=max_len)           # 目标角度 (已归一化)
        self.errors = deque(maxlen=max_len)            # 角度误差 (归一化)

        # 字体 (用于坐标轴数值)
        self.font = pygame.font.SysFont("Consolas", 16)

    def add_point(self, angle_wrapped, target_angle):
        """添加一个数据点 (角度已归一化到 [-π,π])"""
        # 计算误差并归一化
        error = target_angle - angle_wrapped
        error = (error + math.pi) % (2 * math.pi) - math.pi

        self.angles.append(angle_wrapped)
        self.targets.append(target_angle)
        self.errors.append(error)

        # 动态调整队列最大长度以适应新的 time_range
        needed_len = int(self.time_range / self.dt) + 10
        if self.angles.maxlen < needed_len:
            self.angles = deque(self.angles, maxlen=needed_len)
            self.targets = deque(self.targets, maxlen=needed_len)
            self.errors = deque(self.errors, maxlen=needed_len)

    def modify_time_range(self, delta):
        """调节横轴时间范围 (s)"""
        new_range = self.time_range + delta
        if 0.5 <= new_range <= 20.0:
            self.time_range = new_range
            # 调整队列最大长度
            new_len = int(self.time_range / self.dt) + 10
            self.angles = deque(self.angles, maxlen=new_len)
            self.targets = deque(self.targets, maxlen=new_len)
            self.errors = deque(self.errors, maxlen=new_len)

    def modify_angle_range(self, delta):
        """调节纵轴角度范围 (rad)"""
        new_range = self.angle_range + delta
        if 0.2 <= new_range <= math.pi:
            self.angle_range = new_range

    def draw(self):
        """清除曲线区域并绘制坐标轴、网格、三条曲线"""
        # 1. 绘制背景
        pygame.draw.rect(self.screen, (30, 30, 40), self.rect)
        # 2. 绘制外框
        pygame.draw.rect(self.screen, (100, 100, 120), self.rect, 2)

        # 获取绘图区域内部矩形 (留出边距给坐标轴文字)
        plot_rect = self.rect.inflate(-40, -40)
        plot_rect.x += 20
        plot_rect.y += 20
        if plot_rect.width <= 0 or plot_rect.height <= 0:
            return

        # 3. 绘制网格线
        # 水平网格 (角度)
        n_h_lines = 5
        for i in range(n_h_lines + 1):
            y_ratio = i / n_h_lines
            y = plot_rect.bottom - y_ratio * plot_rect.height
            angle_val = -self.angle_range + 2 * self.angle_range * y_ratio
            pygame.draw.line(self.screen, (60, 60, 70),
                             (plot_rect.left, y), (plot_rect.right, y), 1)
            # 标注数值
            label = self.font.render(f"{angle_val:.1f}", True, (180, 180, 200))
            self.screen.blit(label, (plot_rect.left - 35, y - 5))

        # 垂直网格 (时间)
        n_v_lines = 6
        for i in range(n_v_lines + 1):
            x_ratio = i / n_v_lines
            x = plot_rect.left + x_ratio * plot_rect.width
            t_val = self.time_range * (1 - x_ratio)   # 右侧为当前时刻
            pygame.draw.line(self.screen, (60, 60, 70),
                             (x, plot_rect.top), (x, plot_rect.bottom), 1)
            if i % 2 == 0:
                label = self.font.render(f"{t_val:.1f}s", True, (180, 180, 200))
                self.screen.blit(label, (x - 20, plot_rect.bottom + 5))

        # 4. 绘制三条曲线
        # 获取需要显示的点数 (最近 N 个点)
        n_points = int(self.time_range / self.dt)
        angles_list = list(self.angles)[-n_points:]
        targets_list = list(self.targets)[-n_points:]
        errors_list = list(self.errors)[-n_points:]

        if len(angles_list) < 2:
            return

        # 辅助函数：将角度值转换为屏幕 y 坐标
        def angle_to_y(angle):
            # angle 范围 [-angle_range, angle_range]
            ratio = (angle + self.angle_range) / (2 * self.angle_range)
            ratio = max(0.0, min(1.0, ratio))   # 截断超出范围的值
            return plot_rect.bottom - ratio * plot_rect.height

        # 辅助函数：将时间索引转换为屏幕 x 坐标 (最右侧为最新点)
        def index_to_x(idx):
            # idx: 0 对应最旧点, len-1 对应最新点
            ratio = idx / (len(angles_list) - 1)
            return plot_rect.left + ratio * plot_rect.width

        # 绘制实际角度 (绿色)
        points_angle = [(index_to_x(i), angle_to_y(angles_list[i]))
                        for i in range(len(angles_list))]
        pygame.draw.lines(self.screen, (0, 200, 0), False, points_angle, 2)

        # 绘制目标角度 (红色)
        points_target = [(index_to_x(i), angle_to_y(targets_list[i]))
                         for i in range(len(targets_list))]
        pygame.draw.lines(self.screen, (200, 50, 50), False, points_target, 2)

        # 绘制误差 (黄色)
        points_error = [(index_to_x(i), angle_to_y(errors_list[i]))
                        for i in range(len(errors_list))]
        pygame.draw.lines(self.screen, (240, 220, 60), False, points_error, 2)

        # 5. 图例
        legend_y = self.rect.top + 10
        for color, text in [((0,200,0), "Actual"), ((200,50,50), "Target"), ((240,220,60), "Error")]:
            pygame.draw.rect(self.screen, color, (self.rect.right - 70, legend_y, 12, 12))
            label = self.font.render(text, True, (220, 220, 220))
            self.screen.blit(label, (self.rect.right - 55, legend_y - 2))
            legend_y += 18

        # 6. 坐标轴文字提示
        help_text = self.font.render(f"TimeRange:{self.time_range:.1f}s  AngleRange:{self.angle_range:.1f}rad", True, (200,200,200))
        self.screen.blit(help_text, (self.rect.x + 10, self.rect.bottom - 45))
        help2 = self.font.render("Keys: +/- : TimeRange  [ / ] : AngleRange", True, (150,150,150))
        self.screen.blit(help2, (self.rect.x + 10, self.rect.bottom - 30))

# ================== 原有绘图函数 (仅左边区域) ==================
def draw_original(screen, font, angle, target_angle, total_time):
    """绘制箭靶和文字 (保持原有逻辑)"""
    # 全屏背景由调用者统一填充，这里只画左边区域的内容
    # 横线
    line_y = CENTER_Y - B
    pygame.draw.line(screen, (180, 180, 180), (0, line_y), (ORIGIN_WIDTH, line_y), 1)

    # 当前角度箭头 (绿色)
    end_x = CENTER_X - ARROW_LEN * math.sin(angle)
    end_y = CENTER_Y - ARROW_LEN * math.cos(angle)
    pygame.draw.line(screen, (0, 255, 0), (CENTER_X, CENTER_Y), (end_x, end_y), 3)
    # 箭头头部
    head_len = 12
    head_angle = math.pi / 7
    ang1 = angle + math.pi - head_angle
    ang2 = angle + math.pi + head_angle
    p1 = (end_x - head_len * math.sin(ang1), end_y - head_len * math.cos(ang1))
    p2 = (end_x - head_len * math.sin(ang2), end_y - head_len * math.cos(ang2))
    pygame.draw.polygon(screen, (0, 255, 0), [p1, (end_x, end_y), p2])

    # 目标方向指示点 (红色)
    target_x = CENTER_X - ARROW_LEN * math.sin(target_angle)
    target_y = CENTER_Y - ARROW_LEN * math.cos(target_angle)
    pygame.draw.circle(screen, (255, 60, 60), (int(target_x), int(target_y)), 6)

    # 显示时间和角度数值
    time_surf = font.render(f"Time: {total_time:.4f} s", True, (255, 255, 255))
    angle_surf = font.render(f"Angle: {angle:.3f}  Target: {target_angle:.3f}", True, (255, 255, 255))
    screen.blit(time_surf, (10, 10))
    screen.blit(angle_surf, (10, 50))

# ================== 主函数 ==================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Yaw Control with Real-time Curves")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)

    # 初始化环境与控制器
    env = SimpleYawSimEnv(dt=DT_ENV)
    pid = PIDController(
        Kp=5.0, Ki=50.0, Kd=0.3,
        integral_limit=None,
        integral_sep_threshold=0.2,
        derivative_filter_tc=None,
        output_limit=2.0,
        deadband=0.0
    )

    # 曲线绘图器
    curve_plotter = CurvePlotter(screen, CURVE_RECT, DT_PID,
                                 init_time_range=5.0, init_angle_range=3.14)

    total_time = 0.0
    next_pid_time = DT_PID
    action = np.array([0.0])
    running = True

    # 目标角度 (初始值)
    target_angle = 0.0

    # 上次鼠标位置 (用于限制有效区域)
    last_valid_target = 0.0

    # 简单的帧率控制 (保持 PID 速率)
    last_pid_time = time.time()

    while running:
        # ----- 事件处理 -----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # 调节横轴范围 (+ / -)
                if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    curve_plotter.modify_time_range(0.5)
                elif event.key == pygame.K_MINUS:
                    curve_plotter.modify_time_range(-0.5)
                # 调节纵轴范围 ( [ / ] )
                elif event.key == pygame.K_LEFTBRACKET:
                    curve_plotter.modify_angle_range(-0.2)
                elif event.key == pygame.K_RIGHTBRACKET:
                    curve_plotter.modify_angle_range(0.2)

        # ----- 获取目标角度 (仅当鼠标位于左边区域时更新) -----
        mouse_x, _ = pygame.mouse.get_pos()
        if 0 <= mouse_x < ORIGIN_WIDTH:
            dx = mouse_x - CENTER_X
            target_angle = -dx/B*1.5 # math.atan2(-dx, B)      # 与题目公式一致
            last_valid_target = target_angle
        else:
            target_angle = last_valid_target       # 鼠标移出左侧则保持原角度

        # ----- 环境步进 -----
        total_time += DT_ENV
        env.step(action)

        # ----- PID 控制与渲染 (每个 PID 周期执行一次) -----
        if total_time >= next_pid_time - 1e-12:
            theta = env.theta
            # 角度误差归一化 (用于控制)
            error = target_angle - theta
            error = (error + math.pi) % (2 * math.pi) - math.pi
            action = pid.step(np.array([error]), DT_PID)

            # 为曲线准备归一化后的实际角度 (映射到 [-π,π])
            theta_wrapped = math.atan2(math.sin(theta), math.cos(theta))

            # 更新曲线数据
            curve_plotter.add_point(theta_wrapped, target_angle)

            # 渲染整个窗口
            screen.fill((20, 20, 20))                     # 全屏背景
            draw_original(screen, font, theta, target_angle, total_time)   # 左区域内容
            curve_plotter.draw()                          # 右区域曲线

            pygame.display.flip()

            next_pid_time += DT_PID

            # 简易限频 (保持 PID 周期稳定)
            now = time.time()
            elapsed = now - last_pid_time
            if elapsed < DT_PID:
                time.sleep(DT_PID - elapsed)
            last_pid_time = time.time()

        # 限制 CPU 占用，同时保证事件响应流畅
        clock.tick(0)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
