"""
基于PyTorch的可微分参数辨识脚本
- DifferentiableYawSimEnv: 使用PyTorch实现的Stribeck摩擦模型（sigmoid门控）
- 使用torch优化器进行梯度下降
- 数据源: 与train_sac.py相同的.npz文件，3s短序列，PID控制器生成力矩
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import math
import matplotlib.pyplot as plt

# 导入PID控制器和原始仿真环境(用于数据生成)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nn_control'))
from config import DATA_DIR, DT_ENV, DT_CTRL, STEPS_PER_CTRL, J, TAU_C, TAU_S, OMEGA_S, B

plt.rcParams['font.family'] = 'SimHei'


class StraightThroughGate(torch.autograd.Function):
    """前向：离散门控（|x|<eps → 1，否则0）；反向：梯度方向取反（近似 ∂gate/∂x ≈ -1）"""
    @staticmethod
    def forward(ctx, x, eps):
        ctx.save_for_backward(x)
        ctx.eps = eps
        return (x.abs() < eps).float()

    @staticmethod
    def backward(ctx, grad_output):
        # 将梯度取反，使门控随 |x| 增大而减小
        return -grad_output*1.0, None

# ==================== 可微分仿真环境 ====================
class DifferentiableYawSimEnv:
    """
    可微分单轴偏航云台仿真环境
    使用PyTorch张量运算，sigmoid门控替代摩擦力的if分支
    动力学: J * dω/dt = τ_motor - τ_friction(ω) - τ_disturbance
    
    注意: 该环境使用100Hz采样率（dt=0.01），内部使用欧拉法积分
    """
    
    def __init__(self, dt: float = 0.01):
        """
        参数:
            dt: 仿真步长(s)，默认0.01（100Hz）
        """
        self.dt = dt
        
        # 可学习参数（将在外部设置）
        self.J = None
        self.tau_c = None
        self.tau_s = None
        self.omega_s = None
        self.b = None
        self.tau_d = None
    
    def set_params(self, J, tau_c, tau_s, omega_s, b, tau_d):
        """设置可学习参数"""
        self.J = J
        self.tau_c = tau_c
        self.tau_s = tau_s
        self.omega_s = omega_s
        self.b = b
        self.tau_d = tau_d
    
    def friction_torque(self, tau_motor, omega):
        """
        使用sigmoid门控的可微分摩擦模型
        
        门控逻辑:
        - gate ≈ 1 当 |omega| 接近0（静止状态）
        - gate ≈ 0 当 |omega| 较大（运动状态）
        
        静止时: tau_f = gate * tau_motor_clamped + (1-gate) * tau_moving
        其中 tau_motor_clamped 将电机力矩限制在静摩擦力范围内
        
        运动时: 使用Stribeck模型
        """
        self.eps_omega = 1e-6
        # beta = 5.0*(1/self.eps_omega)  # sigmoid倍率，越大门控越陡峭
        
        # # 门控值: 当 |omega| 很小时 gate ≈ 1
        # gate = torch.sigmoid(beta * (self.eps_omega - omega.abs()))

        # gate = 1 if omega.abs() < self.eps_omega else 0
        
        gate = StraightThroughGate.apply(omega, self.eps_omega)
        
        # ---- 静止分支: 静摩擦 ----
        # 电机力矩被钳制在 [-tau_s, tau_s] 内
        tau_motor_clamped = torch.clamp(tau_motor, -self.tau_s, self.tau_s)
        
        # ---- 运动分支: Stribeck模型 ----
        sign_omega = torch.sign(omega)
        # Stribeck指数项
        stribeck_term = self.tau_c + (self.tau_s - self.tau_c) * torch.exp(-omega.abs() / self.omega_s)
        tau_moving = stribeck_term * sign_omega + self.b * omega
        
        # 门控混合
        tau_f = gate * tau_motor_clamped + (1.0 - gate) * tau_moving
        
        return tau_f
    
    def simulate(self, tau_seq, theta_init, omega_init):
        """
        仿真整条力矩序列
        
        参数:
            tau_seq: Tensor [N] 力矩序列
            theta_init: float 初始角度
            omega_init: float 初始角速度
        
        返回:
            theta_seq: Tensor [N] 角度序列
            omega_seq: Tensor [N] 角速度序列
        """
        N = tau_seq.shape[0]
        
        theta_list = []
        omega_list = []
        
        theta = torch.tensor(theta_init, dtype=tau_seq.dtype, device=tau_seq.device)
        omega = torch.tensor(omega_init, dtype=tau_seq.dtype, device=tau_seq.device)
        
        for i in range(N):
            tau_motor = tau_seq[i]
            tau_f = self.friction_torque(tau_motor, omega)
            tau_net = tau_motor - tau_f - self.tau_d
            alpha = tau_net / self.J
            
            omega = omega + alpha * self.dt
            theta = theta + omega * self.dt
            
            theta_list.append(theta)
            omega_list.append(omega)
        
        theta_seq = torch.stack(theta_list)
        omega_seq = torch.stack(omega_list)
        
        return theta_seq, omega_seq


# ==================== 数据生成 ====================
def load_recording_data(filepath=None):
    """
    加载录制数据，与train_sac.py使用相同的数据源
    """
    if filepath is None:
        files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.npz')])
        if not files:
            raise FileNotFoundError(f"在 {DATA_DIR} 中未找到.npz文件")
        filepath = os.path.join(DATA_DIR, files[-1])
    
    data = np.load(filepath)
    return data['theta'].astype(np.float64), data['omega'].astype(np.float64), \
           data['target'].astype(np.float64), data['torque'].astype(np.float64), \
           data['timestamps'].astype(np.float64)


def generate_training_data(data_theta, data_omega, data_target, data_torque, data_timestamps,
                           dt_env=DT_ENV, dt_ctrl=DT_CTRL, steps_per_ctrl=STEPS_PER_CTRL,
                           seq_len=300, tau_s_estimate=None):
    """
    从录制数据中随机采样短序列，或生成恒定力矩激励数据。

    参数：
        data_*: 录制数据
        seq_len: 序列长度（控制步数）
        tau_s_estimate: 当前估计的静摩擦力矩，若为 None 则强制使用正常采样

    返回：
        tau_seq, theta_true, omega_true, theta_init, omega_init
    """
    # 决定生成模式
    if tau_s_estimate is not None and tau_s_estimate > 0:
        r = np.random.rand()
        if r < 0.1:          # 标准差 = tau_s_estimate
            std = tau_s_estimate
        elif r < 0.2:        # 标准差 = 0.1 * tau_s_estimate
            std = 0.1 * tau_s_estimate
        else:                # 正常采样
            std = None
    else:
        std = None

    # ===== 特殊模式：恒定力矩 =====
    if std is not None:
        # 采样恒定力矩（均值为0）
        tau_const = np.random.normal(0, std)
        tau_const = np.clip(tau_const, -2.0, 2.0)

        from SimEnv import SimpleYawSimEnv
        env = SimpleYawSimEnv(dt=dt_env, J=J, tau_c=TAU_C, tau_s=TAU_S,
                              omega_s=OMEGA_S, b=B)
        env.theta = 0.0
        env.omega = 0.0

        tau_seq = []
        theta_true = []
        omega_true = []
        for _ in range(seq_len):
            env.step(np.array([tau_const]))
            tau_seq.append(tau_const)
            theta_true.append(env.theta)
            omega_true.append(env.omega)

        return (np.array(tau_seq, dtype=np.float32),
                np.array(theta_true, dtype=np.float32),
                np.array(omega_true, dtype=np.float32),
                0.0, 0.0)   # 初始状态均为0

    # ===== 正常采样模式（原有逻辑） =====
    from PID import PIDController

    total_ctrl_steps = len(data_torque)
    if total_ctrl_steps < seq_len:
        raise ValueError(f"录制数据长度({total_ctrl_steps})小于要求序列长度({seq_len})")

    max_start = total_ctrl_steps - seq_len - 1
    start_idx = np.random.randint(0, max_start)

    theta_segment = data_theta[start_idx:start_idx + seq_len]
    omega_segment = data_omega[start_idx:start_idx + seq_len]
    target_segment = data_target[start_idx:start_idx + seq_len]

    theta_init = float(theta_segment[0])
    omega_init = float(omega_segment[0])

    pid = PIDController(
        Kp=5.0, Ki=50.0, Kd=0.3,
        integral_limit=None,
        integral_sep_threshold=0.2,
        derivative_filter_tc=None,
        output_limit=2.0,
        deadband=0.0
    )

    from SimEnv import SimpleYawSimEnv
    env = SimpleYawSimEnv(dt=dt_env, J=J, tau_c=TAU_C, tau_s=TAU_S,
                          omega_s=OMEGA_S, b=B)
    env.theta = theta_init
    env.omega = omega_init

    tau_seq = []
    theta_true = []
    omega_true = []
    for i in range(seq_len):
        error = target_segment[i] - env.theta
        error = (error + math.pi) % (2 * math.pi) - math.pi
        tau = pid.step(np.array([error]), dt_ctrl)[0]
        tau = np.clip(tau, -2.0, 2.0)

        for _ in range(steps_per_ctrl):
            env.step(np.array([tau]))

        tau_seq.append(tau)
        theta_true.append(env.theta)
        omega_true.append(env.omega)

    return (np.array(tau_seq, dtype=np.float32),
            np.array(theta_true, dtype=np.float32),
            np.array(omega_true, dtype=np.float32),
            theta_init, omega_init)


# ==================== 优化主流程 ====================
def optimize_parameters(
    data_theta, data_omega, data_target, data_torque, data_timestamps,
    num_epochs=2000, lr=0.001, seq_len=300, num_sequences_per_epoch=5,
    device='cpu'
):
    """
    使用PyTorch优化器进行参数辨识
    
    参数:
        data_*: 录制数据
        num_epochs: 优化轮数
        lr: 学习率
        seq_len: 每段序列长度（步数，对应3s@100Hz）
        num_sequences_per_epoch: 每轮采样的序列数量
        device: 计算设备
    """
    # 创建可微分仿真环境
    diff_env = DifferentiableYawSimEnv(dt=0.01)  # 100Hz

    # true_params = [J, TAU_C, TAU_S, OMEGA_S, B, 0.0]
    
    # 定义可学习参数（转为可学习参数）
    # 初始值使用合理猜测
    log_J = nn.Parameter(torch.tensor(math.log(0.05), device=device))
    log_tau_c = nn.Parameter(torch.tensor(math.log(0.5), device=device))
    log_tau_s = nn.Parameter(torch.tensor(math.log(0.8), device=device))
    log_omega_s = nn.Parameter(torch.tensor(math.log(0.05), device=device))
    log_b = nn.Parameter(torch.tensor(math.log(0.03), device=device))
    tau_d = torch.tensor(0.0, device=device) # nn.Parameter(torch.tensor(0.0, device=device))
    
    # 使用log空间编码正约束参数，并使tau_s > tau_c
    # 实际参数通过变换得到
    def get_params():
        J_val = torch.exp(log_J)
        tau_c_val = torch.exp(log_tau_c)
        # tau_s = tau_c + exp(log_diff) 保证 tau_s > tau_c
        tau_s_val = tau_c_val + torch.exp(log_tau_s)
        omega_s_val = torch.exp(log_omega_s)
        b_val = torch.exp(log_b)
        return J_val, tau_c_val, tau_s_val, omega_s_val, b_val, tau_d
    
    # 优化器
    optimizer = torch.optim.SGD(
        [log_J, log_tau_c, log_tau_s, log_omega_s, log_b, tau_d],
        lr=lr
    )
    
    # 学习率调度器
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 损失记录
    loss_history = []
    param_history = []
    
    true_params = [J, TAU_C, TAU_S, OMEGA_S, B, 0.0]  # 真实参数
    
    print(f"开始优化: {num_epochs} epochs, lr={lr}, seq_len={seq_len}")
    print(f"真实参数: J={J}, tau_c={TAU_C}, tau_s={TAU_S}, omega_s={OMEGA_S}, b={B}, tau_d=0.0")
    print(f"设备: {device}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for _ in range(num_sequences_per_epoch):
            # 生成训练数据（每次采样不同的片段）# 在优化循环内部，生成数据之前
            with torch.no_grad():
                _, _, tau_s_val, _, _, _ = get_params()
                tau_s_est = tau_s_val.item()

            tau_seq_np, theta_true_np, omega_true_np, theta_init, omega_init = \
                generate_training_data(
                    data_theta, data_omega, data_target, data_torque, data_timestamps,
                    seq_len=seq_len,
                    tau_s_estimate=tau_s_est   # 传入当前估计值
                )
            
            # 转为张量
            tau_seq = torch.tensor(tau_seq_np, dtype=torch.float32, device=device)
            theta_true = torch.tensor(theta_true_np, dtype=torch.float32, device=device)
            omega_true = torch.tensor(omega_true_np, dtype=torch.float32, device=device)
            
            # 获取当前参数并设置到环境
            J_val, tau_c_val, tau_s_val, omega_s_val, b_val, tau_d_val = get_params()
            diff_env.set_params(J_val, tau_c_val, tau_s_val, omega_s_val, b_val, tau_d_val)
            
            # 可微分仿真
            theta_sim, omega_sim = diff_env.simulate(tau_seq, theta_init, omega_init)
            
            # 损失: 位置和速度的均方误差
            pos_loss = torch.mean((theta_sim - theta_true) ** 2)
            vel_loss = torch.mean((omega_sim - omega_true) ** 2)
            loss = pos_loss + 1.0 * vel_loss  # 速度损失权重

            # print(loss)
            
            optimizer.zero_grad()
            loss.backward()
            # print(log_J.grad, log_tau_c.grad, log_tau_s.grad, log_omega_s.grad, log_b.grad, tau_d.grad)
            torch.nn.utils.clip_grad_value_(
                [log_J, log_tau_c, log_tau_s, log_omega_s, log_b, tau_d],
                2.0
            )
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= num_sequences_per_epoch
        # scheduler.step()
        
        loss_history.append(epoch_loss)
        
        # 记录参数
        with torch.no_grad():
            J_v, tc_v, ts_v, os_v, b_v, td_v = get_params()
            param_history.append([
                J_v.item(), tc_v.item(), ts_v.item(), os_v.item(), b_v.item(), td_v.item()
            ])
        
        # 打印日志
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                J_v, tc_v, ts_v, os_v, b_v, td_v = get_params()
                print(f"Epoch {epoch:5d}/{num_epochs} | Loss: {epoch_loss:.6e}")
                print(f"  J={J_v.item():.6f} (真:{true_params[0]:.4f})  "
                      f"tau_c={tc_v.item():.6f} (真:{true_params[1]:.4f})  "
                      f"tau_s={ts_v.item():.6f} (真:{true_params[2]:.4f})")
                print(f"  omega_s={os_v.item():.6f} (真:{true_params[3]:.4f})  "
                      f"b={b_v.item():.6f} (真:{true_params[4]:.4f})  "
                      f"tau_d={td_v.item():.6f} (真:{true_params[5]:.4f})")
    
    # 最终参数
    with torch.no_grad():
        final_params = [p.item() for p in get_params()]
    
    return final_params, loss_history, param_history


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载录制数据（与train_sac.py相同的数据源）
    print("加载录制数据...")
    try:
        data_theta, data_omega, data_target, data_torque, data_timestamps = load_recording_data()
        print(f"加载成功: {len(data_torque)} 条记录, 总时长 {data_timestamps[-1]:.2f}s")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 collect_data.py 采集数据")
        sys.exit(1)
    
    # 优化
    final_params, loss_history, param_history = optimize_parameters(
        data_theta, data_omega, data_target, data_torque, data_timestamps,
        num_epochs=200000,
        lr=1e-3,
        seq_len=3,
        num_sequences_per_epoch=3,
        device=device
    )
    
    # 打印最终结果
    param_names = ['J', 'tau_c', 'tau_s', 'omega_s', 'b', 'tau_d']
    true_params = [J, TAU_C, TAU_S, OMEGA_S, B, 0.0]
    
    print("\n" + "=" * 60)
    print("辨识结果:")
    print(f"{'参数':<10} {'估计值':<14} {'真实值':<12} {'绝对误差':<12} {'相对误差%':<12}")
    print("-" * 60)
    for name, est, true in zip(param_names, final_params, true_params):
        abs_err = abs(est - true)
        rel_err = abs_err / abs(true) * 100 if abs(true) > 1e-10 else 0.0
        print(f"{name:<10} {est:<14.6f} {true:<12.6f} {abs_err:<12.6f} {rel_err:<12.2f}")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # 损失曲线
    ax = axes[0, 0]
    ax.plot(loss_history)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('损失曲线')
    ax.grid(True)
    
    # 将参数历史转为 numpy 数组
    param_history = np.array(param_history)

    # 参数收敛曲线（放在第一行第二列开始）
    param_start_idx = 1  # 跳过 (0,0)
    for i, (name, true_val) in enumerate(zip(param_names, true_params)):
        idx = param_start_idx + i
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        ax.plot(param_history[:, i], label='估计值')
        ax.axhline(y=true_val, color='r', linestyle='--', label='真实值')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(name)
        ax.set_title(f'{name} 收敛曲线')
        ax.legend()
        ax.grid(True)

    # 仿真对比图（放在最后剩余位置，例如 (2,2)）
    ax = axes[2, 2]
    
    # 使用最终参数进行仿真对比
    diff_env = DifferentiableYawSimEnv(dt=0.01)
    with torch.no_grad():
        J_t = torch.tensor(final_params[0], device=device)
        tc_t = torch.tensor(final_params[1], device=device)
        ts_t = torch.tensor(final_params[2], device=device)
        os_t = torch.tensor(final_params[3], device=device)
        b_t = torch.tensor(final_params[4], device=device)
        td_t = torch.tensor(final_params[5], device=device)
        diff_env.set_params(J_t, tc_t, ts_t, os_t, b_t, td_t)
    
    # 采样一段数据做仿真对比
    tau_seq_np, theta_true_np, omega_true_np, theta_init, omega_init = \
        generate_training_data(
            data_theta, data_omega, data_target, data_torque, data_timestamps,
            seq_len=300
        )
    tau_seq_t = torch.tensor(tau_seq_np, dtype=torch.float32, device=device)
    theta_sim_t, omega_sim_t = diff_env.simulate(tau_seq_t, theta_init, omega_init)
    
    ax.plot(theta_true_np[:300], label='真实角度', alpha=0.7)
    ax.plot(theta_sim_t.cpu().numpy()[:300], '--', label='仿真角度', alpha=0.7)
    ax.set_xlabel('步数')
    ax.set_ylabel('角度 (rad)')
    ax.set_title('角度对比 (最终参数)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()