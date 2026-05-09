import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from SimEnv import SimpleYawSimEnv
from PID import PIDController

plt.rcParams['font.family'] = 'SimHei'

# ---------------------------- 数据生成（增强激励 + 绘图） ----------------------------
def generate_data(dt=0.001, seed=42):
    """
    生成包含丰富动态特性的力矩序列与角度响应，并在返回前显示曲线。
    总时长 100 秒。
    """
    np.random.seed(seed)
    env = SimpleYawSimEnv(dt=dt)          # 默认参数：真实模型
    total_time = 200.0
    N = int(total_time / dt)
    tau_seq = np.zeros(N)
    theta_seq = np.zeros(N)

    # 时间区间边界（秒）
    t_end_step = 10.0
    t_end_free = 20.0
    t_end_pid  = 25.0
    t_end_ramp = 40.0
    t_end_sweep = 200.0

    # 1. 阶跃段 (0~10s) : 每1s跳变一次，幅值序列（绝对值 ≤2）
    step_amplitudes = [0.0, 1.0, -2.0, 2.0, -1.5, 1.5, -1.0, 1.0, -0.5, 0.5]
    step_duration = 1.0
    step_samples = int(step_duration / dt)
    for i, amp in enumerate(step_amplitudes):
        start = int(i * step_samples)
        end = start + step_samples
        tau_seq[start:end] = amp

    # 2. 自由运动段 (10~20s) : 力矩=0
    free_start = int(t_end_step / dt)
    free_end   = int(t_end_free / dt)
    tau_seq[free_start:free_end] = 0.0

    # 将环境重置并运行前20秒开环，填充角度
    env.reset()
    for i in range(free_end):
        tau = tau_seq[i]
        theta = env.step(np.array([tau]))
        theta_seq[i] = theta[0]

    # 3. PID 位置控制段 (20~25s) : 将位置稳定在0
    pid = PIDController(Kp=5.0, Ki=50.0, Kd=0.3,
                        integral_limit=None,
                        integral_sep_threshold=0.2,
                        derivative_filter_tc=None,
                        output_limit=2.0,
                        deadband=0.0)
    pid_start = free_end
    pid_end   = int(t_end_pid / dt)
    for i in range(pid_start, pid_end):
        error = np.array([0.0 - env.theta])   # 期望位置0
        tau = pid.step(error, dt)
        tau = np.clip(tau, -2.0, 2.0)         # 确保不超过2
        tau_seq[i] = tau[0]
        theta = env.step(tau)
        theta_seq[i] = theta[0]

    # 4. 斜坡段 (25~40s) : 力矩从0线性增加至2.0
    ramp_start = pid_end
    ramp_end   = int(t_end_ramp / dt)
    ramp_len = ramp_end - ramp_start
    ramp_tau = np.linspace(0.0, 2.0, ramp_len)
    for i, tau_val in enumerate(ramp_tau):
        idx = ramp_start + i
        tau_seq[idx] = tau_val
        theta = env.step(np.array([tau_val]))
        theta_seq[idx] = theta[0]

    # 5. 扫频段 (40~200s) : 对数间隔频率，每20s一段，正弦幅值2.0
    sweep_start = ramp_end
    sweep_end   = int(t_end_sweep / dt)
    freq_list = np.logspace(np.log10(0.1), np.log10(3.0), 8)
    segment_duration = 20.0
    segment_samples = int(segment_duration / dt)
    t_local = np.arange(segment_samples) * dt
    for seg_idx, f in enumerate(freq_list):
        start_idx = sweep_start + seg_idx * segment_samples
        end_idx = start_idx + segment_samples
        if end_idx > sweep_end:
            end_idx = sweep_end
        tau_seg = 2.0 * np.sin(2 * np.pi * f * t_local[:end_idx-start_idx])
        tau_seq[start_idx:end_idx] = tau_seg
        for j, tau_val in enumerate(tau_seg):
            idx = start_idx + j
            if idx >= N:
                break
            theta = env.step(np.array([tau_val]))
            theta_seq[idx] = theta[0]

    # 力矩限幅（保证绝对值不超过2）
    tau_seq = np.clip(tau_seq, -2.0, 2.0)

    # ================== 显示力矩及位移曲线 ==================
    t_axis = np.arange(N) * dt
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t_axis, tau_seq, linewidth=0.8)
    plt.ylabel('力矩 (Nm)')
    plt.title('输入力矩序列')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t_axis, theta_seq, linewidth=0.8, color='orange')
    plt.xlabel('时间 (s)')
    plt.ylabel('角度 (rad)')
    plt.title('角度响应序列')
    plt.grid(True)
    plt.tight_layout()
    plt.show()   # 显示后继续执行

    return tau_seq, theta_seq, dt

# ---------------------------- 仿真预测函数 ----------------------------
def simulate_with_params(tau_seq, dt, params, init_theta, init_omega):
    J, tau_c, tau_s, omega_s, b, tau_d = params
    env = SimpleYawSimEnv(
        dt=dt, J=J, tau_c=tau_c, tau_s=tau_s, omega_s=omega_s, b=b,
        initial_angle=init_theta, initial_omega=init_omega,
        angle_limit=None, disturbance_torque=tau_d
    )
    theta_sim = []
    for tau in tau_seq:
        theta = env.step(np.array([tau]))
        theta_sim.append(theta[0])
    return np.array(theta_sim)

# ---------------------------- 损失函数 ----------------------------
loss_function_call_time = 0
def loss_function(params, tau_seq, theta_meas, dt, init_theta, init_omega):
    global loss_function_call_time
    loss_function_call_time += 1
    print(f"{loss_function_call_time=}")
    try:
        theta_sim = simulate_with_params(tau_seq, dt, params, init_theta, init_omega)
        mse = np.mean((theta_meas - theta_sim) ** 2)
        if params[2] <= params[1]:   # tau_s <= tau_c
            mse += 1e3 * (params[1] - params[2]) ** 2
        return mse
    except:
        return 1e12

# ---------------------------- 参数估计主函数 ----------------------------
def estimate_parameters(tau_seq, theta_meas, dt,
                        bounds=None, use_global_first=True):
    init_theta = theta_meas[0]
    init_omega = (theta_meas[1] - theta_meas[0]) / dt if len(theta_meas) > 1 else 0.0

    if bounds is None:
        bounds = [
            (0.001, 0.1),    # J
            (0.01, 2.0),     # tau_c
            (0.02, 3.0),     # tau_s
            (0.001, 0.2),    # omega_s
            (0.0, 0.5),      # b
            (-1.0, 1.0)      # tau_d
        ]

    x0 = np.array([0.05, 0.5, 0.8, 0.05, 0.03, 0.2])

    if use_global_first:
        print("执行全局优化（差分进化）...")
        result_global = differential_evolution(
            loss_function, bounds,
            args=(tau_seq, theta_meas, dt, init_theta, init_omega),
            maxiter=30, popsize=15, seed=42, disp=False
        )
        x0 = result_global.x
        print(f"全局优化完成，初始损失: {result_global.fun:.6e}")

    print("执行局部优化（L-BFGS-B）...")
    result_local = minimize(
        loss_function, x0,
        args=(tau_seq, theta_meas, dt, init_theta, init_omega),
        method='L-BFGS-B', bounds=bounds,
        options={'ftol': 1e-8, 'gtol': 1e-8}
    )
    return result_local.x, result_local.fun

# ---------------------------- 主程序 ----------------------------
if __name__ == "__main__":
    print("生成数据（含阶跃、自由、PID稳速、斜坡、扫频）...")
    tau_seq, theta_meas, dt = generate_data()   # 使用默认 dt=0.001

    print("开始参数辨识...")
    est_params, final_loss = estimate_parameters(tau_seq, theta_meas, dt)

    param_names = ['J', 'tau_c', 'tau_s', 'omega_s', 'b', 'tau_d']
    true_params = [0.01, 0.2, 0.3, 0.01, 0.01, 0.0]  # SimEnv 默认值

    print("\n辨识结果：")
    print(f"最终损失 (MSE): {final_loss:.6e}")
    print("参数名      估计值       真实值       绝对误差")
    for name, est, true in zip(param_names, est_params, true_params):
        print(f"{name:8s}  {est:10.6f}  {true:10.6f}  {abs(est-true):10.6f}")

    # 可选：再用估计参数仿真对比曲线
    init_theta = theta_meas[0]
    init_omega = (theta_meas[1] - theta_meas[0]) / dt
    theta_sim = simulate_with_params(tau_seq, dt, est_params, init_theta, init_omega)
    plt.figure(figsize=(12, 6))
    plt.plot(theta_meas[:], label='实测角度', alpha=0.7)
    plt.plot(theta_sim[:], '--', label='仿真角度 (辨识参数)', alpha=0.7)
    plt.xlabel('步数')
    plt.ylabel('角度 (rad)')
    plt.legend()
    plt.title('角度响应对比')
    plt.grid(True)
    plt.show()
