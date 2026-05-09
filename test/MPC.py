# MPC.py
import numpy as np
from scipy.optimize import minimize
from collections import deque

class MPCController:
    """
    基于完整非线性动力学的模型预测控制器。
    使用scipy.optimize.minimize在线求解最优控制序列。
    """
    def __init__(self,
                 dt_control,      # 控制周期 (s)
                 dt_sim,          # 仿真积分步长 (s)
                 J, tau_c, tau_s, omega_s, b,
                 angle_limit=None,
                 disturbance_torque=0.0,
                 N=20,            # 预测时域 (步数)
                 Q=1.0,           # 角度跟踪权重
                 R=0.01,          # 控制量惩罚权重
                 Rd=0.1,          # 控制增量惩罚权重
                 u_min=-2.0,      # 控制力矩下限
                 u_max=2.0,       # 控制力矩上限
                 max_iter=50):    # 优化最大迭代次数
        """
        dt_control: MPC决策周期 (s)，通常等于控制周期
        dt_sim:     模型积分步长 (s)，应与环境步长一致
        N:          预测时域步数
        Q,R,Rd:     代价函数权重
        u_min,u_max: 控制力矩限幅
        """
        self.dt_control = dt_control
        self.dt_sim = dt_sim
        self.N = N
        self.Q = Q
        self.R = R
        self.Rd = Rd
        self.u_min = u_min
        self.u_max = u_max
        self.max_iter = max_iter

        # 模型参数
        self.J = J
        self.tau_c = tau_c
        self.tau_s = tau_s
        self.omega_s = omega_s
        self.b = b
        self.angle_limit = angle_limit
        self.disturbance_torque = disturbance_torque

        # 积分步数 (每个控制周期内的细积分步数)
        self.steps_per_control = int(round(dt_control / dt_sim))
        if abs(self.steps_per_control * dt_sim - dt_control) > 1e-9:
            raise ValueError("dt_control 必须是 dt_sim 的整数倍")

        # 上一次最优控制序列 (用于 warm start)
        self.prev_u_seq = np.random.random(N) # np.zeros(N) # np.random.random(N)

        # 微小速度阈值 (与仿真环境保持一致)
        self.eps_omega = 1e-8

    # ---------- 摩擦模型 (与 SimEnv 完全一致) ----------
    def _friction_torque(self, tau_motor, omega):
        """计算摩擦力矩，与 SimpleYawSimEnv._friction_torque 相同"""
        if abs(omega) < self.eps_omega:
            if abs(tau_motor) <= self.tau_s:
                return tau_motor
            else:
                return np.sign(tau_motor) * self.tau_c
        else:
            sign_omega = np.sign(omega)
            tau_f = (self.tau_c + (self.tau_s - self.tau_c) * np.exp(-abs(omega) / self.omega_s)) * sign_omega
            tau_f += self.b * omega
            return tau_f

    # ---------- 单步欧拉积分 (与 SimEnv 相同) ----------
    def _dynamics_step(self, theta, omega, tau_motor, dt):
        """使用欧拉法积分一步，返回 (新角度, 新角速度)"""
        tau_f = self._friction_torque(tau_motor, omega)
        tau_net = tau_motor - tau_f - self.disturbance_torque
        alpha = tau_net / self.J
        omega_new = omega + alpha * dt
        theta_new = theta + omega_new * dt

        # 角度限位处理 (与 SimEnv 一致)
        if self.angle_limit is not None:
            if theta_new > self.angle_limit:
                theta_new = self.angle_limit
                omega_new = 0.0
            elif theta_new < -self.angle_limit:
                theta_new = -self.angle_limit
                omega_new = 0.0
        return theta_new, omega_new

    # ---------- 预测轨迹 ----------
    def _predict_trajectory(self, x0, u_seq):
        """
        给定初始状态 x0 = [theta, omega] 和控制序列 u_seq (长度 N)，
        返回预测的角度序列 (长度 N+1，包含初始状态)
        """
        theta, omega = x0
        theta_pred = [theta]
        for k in range(self.N):
            tau = u_seq[k]
            # 在一个控制周期内积分多次
            for _ in range(self.steps_per_control):
                theta, omega = self._dynamics_step(theta, omega, tau, self.dt_sim)
            theta_pred.append(theta)
        return np.array(theta_pred)

    # ---------- 代价函数 (用于 scipy.optimize) ----------
    def _cost_function(self, u_flat, x0, theta_ref):
        """
        代价函数:
        Σ_{k=1}^{N} Q * (wrap_error(θ_k - θ_ref))^2
        + Σ_{k=0}^{N-1} R * u_k^2
        + Σ_{k=1}^{N-1} Rd * (u_k - u_{k-1})^2
        """
        u_seq = u_flat.reshape(self.N)
        theta_pred = self._predict_trajectory(x0, u_seq)

        cost = 0.0
        # 跟踪误差
        for k in range(1, self.N + 1):
            err = theta_pred[k] - theta_ref[k]
            # 角度误差归一化到 [-pi, pi]
            err = (err + np.pi) % (2 * np.pi) - np.pi
            cost += self.Q * np.sum(err ** 2)

        # 控制量惩罚
        cost += self.R * np.sum(u_seq ** 2)

        # 控制增量惩罚
        if self.N > 1:
            du = np.diff(u_seq)
            cost += self.Rd * np.sum(du ** 2)

        return cost

    # ---------- 求解最优控制 ----------
    def step(self, state, theta_ref):
        """
        输入:
            state: [theta, omega] 当前状态
            theta_ref: 当前参考目标 (延迟后的目标)
        返回:
            最优控制力矩 (标量)
        """
        x0 = np.array([state[0], state[1]])

        # 初始猜测: 平移上一次的最优序列 (warm start)
        u0 = np.roll(self.prev_u_seq, -1)
        u0[-1] = u0[-2] if self.N > 1 else 0.0
        # 确保初始猜测在边界内
        u0 = np.clip(u0, self.u_min, self.u_max)

        # 定义边界 (每个控制量都要约束)
        bounds = [(self.u_min, self.u_max) for _ in range(self.N)]

        # 调用 scipy 优化器
        result = minimize(
            fun=self._cost_function,
            x0=u0,
            args=(x0, theta_ref),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': self.max_iter, 'disp': False}
        )

        if not result.success:
            # 若优化失败，使用上一次序列的第一个值或零
            u_opt = self.prev_u_seq[0] if self.prev_u_seq is not None else 0.0
        else:
            u_opt_seq = result.x
            # 保存本次最优序列供下次使用
            self.prev_u_seq = u_opt_seq
            u_opt = u_opt_seq[0]

        return np.clip(u_opt, self.u_min, self.u_max)
