import numpy as np

class SimpleYawSimEnv:
    """
    单轴偏航云台仿真环境，采用正确的 Stribeck 摩擦模型。
    动力学方程: J * dω/dt = τ_motor - τ_friction(ω) - τ_disturbance
    状态: [角度 θ (rad), 角速度 ω (rad/s)]
    """

    def __init__(self,
                 dt: float = 0.001,
                 J: float = 0.01,
                 tau_c: float = 0.2,
                 tau_s: float = 0.3,
                 omega_s: float = 0.01,
                 b: float = 0.01,
                 initial_angle: float = 0.0,
                 initial_omega: float = 0.0,
                 angle_limit: float = None,
                 disturbance_torque: float = 0.0):
        """
        参数:
            dt: 仿真步长 (s)
            J: 绕 yaw 轴的转动惯量 (kg·m²)
            tau_c: 库仑摩擦力矩 (Nm)
            tau_s: 最大静摩擦力矩 (Nm) (tau_s > tau_c)
            omega_s: Stribeck 速度 (rad/s) (摩擦下降的特征速度)
            b: 粘滞摩擦系数 (Nm·s/rad)
            initial_angle: 初始角度 (rad)
            initial_omega: 初始角速度 (rad/s)
            angle_limit: 角度限制 (rad)，None 表示无限制（连续旋转）
            disturbance_torque: 恒定的外部扰动扭矩 (Nm)
        """
        self.dt = dt
        self.J = J
        self.tau_c = tau_c
        self.tau_s = tau_s
        self.omega_s = omega_s
        self.b = b
        self.angle_limit = angle_limit
        self.disturbance_torque = disturbance_torque

        # 状态变量
        self.theta = initial_angle
        self.omega = initial_omega

        # 保存初始状态用于 reset
        self.initial_theta = initial_angle
        self.initial_omega = initial_omega

        # 速度零点判断阈值 (rad/s)
        self.eps_omega = 1e-8

    def _friction_torque(self, tau_motor: float, omega: float) -> float:
        """
        计算摩擦力矩 (修正版)
        返回值: 摩擦力矩 (Nm)
        """
        if abs(omega) < self.eps_omega:
            # 静止 / 几乎静止
            if abs(tau_motor) <= self.tau_s:
                # 未克服静摩擦，合力矩为零，保持静止
                return tau_motor
            else:
                # 克服静摩擦，开始运动；摩擦跳变为库仑摩擦（方向与电机力矩相同）
                return np.sign(tau_motor) * self.tau_c
        else:
            # 运动中：经典的 Stribeck 模型（使用角速度）
            sign_omega = np.sign(omega)
            tau_f = (self.tau_c + (self.tau_s - self.tau_c) * np.exp(-abs(omega) / self.omega_s)) * sign_omega
            tau_f += self.b * omega
            return tau_f

    def _dynamics(self, tau_motor: float):
        """欧拉法更新状态"""
        tau_f = self._friction_torque(tau_motor, self.omega)
        tau_net = tau_motor - tau_f - self.disturbance_torque
        alpha = tau_net / self.J

        self.omega += alpha * self.dt
        self.theta += self.omega * self.dt

        # 可选角度限幅（到达限位时速度置零）
        if self.angle_limit is not None:
            if self.theta > self.angle_limit:
                self.theta = self.angle_limit
                self.omega = 0.0
            elif self.theta < -self.angle_limit:
                self.theta = -self.angle_limit
                self.omega = 0.0

    def reset(self) -> np.ndarray:
        """重置环境，返回初始角度 (弧度) 作为一维 numpy 数组"""
        self.theta = self.initial_theta
        self.omega = self.initial_omega
        return np.array([self.theta])

    def step(self, action: np.ndarray) -> np.ndarray:
        """
        执行一步仿真
        参数:
            action: numpy 数组，包含控制力矩 (Nm)，例如 np.array([0.1])
        返回:
            obs: numpy 数组，当前角度 (rad)
        """
        tau_motor = float(action.item()) if action.size == 1 else float(action[0])
        self._dynamics(tau_motor)
        return np.array([self.theta])
