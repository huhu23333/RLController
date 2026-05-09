import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd,
                 integral_limit=None,
                 integral_sep_threshold=None,
                 derivative_filter_tc=None,
                 output_limit=None,
                 deadband=0.0):
        """
        初始化 PID 控制器。

        参数:
            Kp, Ki, Kd : float
                比例、积分、微分系数。
            integral_limit : float or tuple/list of two floats, optional
                积分限幅。若为 float，表示对称限幅 [-limit, limit]；
                若为 (low, high)，则分别指定下限和上限；None 表示不限幅。
            integral_sep_threshold : float, optional
                积分分离阈值。当 |error| > threshold 时，积分项停止累加。
            derivative_filter_tc : float, optional
                微分项指数平滑时间常数 (秒)。若为 None 或 0，则不滤波。
            output_limit : float or tuple/list of two floats, optional
                输出限幅。格式同 integral_limit。
            deadband : float, optional
                死区阈值。当 |error| < deadband 时，将误差视为 0。
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.deadband = deadband
        self.integral_sep_threshold = integral_sep_threshold

        # 积分限幅处理
        if integral_limit is None:
            self.integral_limit_low = -np.inf
            self.integral_limit_high = np.inf
        elif isinstance(integral_limit, (int, float)):
            limit = abs(integral_limit)
            self.integral_limit_low = -limit
            self.integral_limit_high = limit
        else:
            self.integral_limit_low = float(integral_limit[0])
            self.integral_limit_high = float(integral_limit[1])

        # 输出限幅处理
        if output_limit is None:
            self.output_limit_low = -np.inf
            self.output_limit_high = np.inf
        elif isinstance(output_limit, (int, float)):
            limit = abs(output_limit)
            self.output_limit_low = -limit
            self.output_limit_high = limit
        else:
            self.output_limit_low = float(output_limit[0])
            self.output_limit_high = float(output_limit[1])

        self.derivative_filter_tc = derivative_filter_tc

        # 状态变量
        self.integral_sum = 0.0
        self.prev_error = None          # 上一次误差 (None 表示尚未有历史)
        self.prev_derivative_filtered = 0.0

    def step(self, error_array, dt):
        """
        单步计算控制输出。

        参数:
            error_array : numpy.ndarray of shape (1,)
                当前时刻的误差信号 (期望值 - 实际值)，以单个元素的数组形式给出。
            dt : float
                两次调用之间的时间间隔 (秒)。

        返回:
            numpy.ndarray of shape (1,)
                当前时刻的控制输出。
        """
        # 提取标量误差
        error = error_array.item()

        # 死区处理
        if abs(error) < self.deadband:
            error = 0.0

        # 比例项
        P = self.Kp * error

        # 积分项 (带积分分离和限幅)
        if self.Ki != 0.0:
            # 积分分离：若设置了阈值且误差绝对值超过阈值，则不累加积分
            if (self.integral_sep_threshold is None or
                abs(error) <= self.integral_sep_threshold):
                self.integral_sum += error * dt
                # 积分限幅
                if self.integral_sum > self.integral_limit_high:
                    self.integral_sum = self.integral_limit_high
                elif self.integral_sum < self.integral_limit_low:
                    self.integral_sum = self.integral_limit_low
            I = self.integral_sum * self.Ki
        else:
            I = 0.0

        # 微分项 (可选指数平滑滤波)
        if self.Kd != 0.0:
            # 原始微分 (第一次调用时无历史，设为 0)
            if self.prev_error is None:
                derivative_raw = 0.0
            else:
                derivative_raw = (error - self.prev_error) / dt

            # 一阶低通滤波 (指数平滑)，使用精确系数 alpha = 1 - exp(-dt/tau)
            if (self.derivative_filter_tc is not None and
                self.derivative_filter_tc > 0.0):
                alpha = 1.0 - np.exp(-dt / self.derivative_filter_tc)
                self.prev_derivative_filtered = alpha * derivative_raw + (1.0 - alpha) * self.prev_derivative_filtered
                D = self.prev_derivative_filtered * self.Kd
            else:
                D = derivative_raw * self.Kd
        else:
            D = 0.0

        # 总输出
        output = P + I + D

        # 输出限幅
        if output > self.output_limit_high:
            output = self.output_limit_high
        elif output < self.output_limit_low:
            output = self.output_limit_low

        # 更新误差历史
        self.prev_error = error

        # 返回单个元素的数组
        return np.array([output])
