import ctypes
import numpy as np
from pathlib import Path

class MPCController:
    """Python wrapper for C++ MPC controller using ctypes"""
    def __init__(self,
                 dt_control, dt_sim,
                 J, tau_c, tau_s, omega_s, b,
                 angle_limit=None,
                 disturbance_torque=0.0,
                 N=20,
                 Q=1.0,
                 R=0.01,
                 Rd=0.1,
                 u_min=-2.0,
                 u_max=2.0,
                 max_iter=50,
                 delay_par_n=3,
                 lib_path=None):
        """
        参数含义与原始 Python 版本完全一致，新增 delay_par_n
        """
        self.dt_control = dt_control
        self.dt_sim = dt_sim
        self.N = N
        self.angle_limit = angle_limit if angle_limit is not None else -1.0  # -1 表示无限制
        self.disturbance_torque = disturbance_torque
        self.delay_par_n = delay_par_n

        # 加载共享库
        if lib_path is None:
            lib_path = Path(__file__).parent / "libmpc_controller.so"
        self.lib = ctypes.CDLL(str(lib_path))

        # 定义函数原型（注意增加 delay_par_n 参数）
        self.lib.mpc_create.argtypes = [
            ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int  # delay_par_n
        ]
        self.lib.mpc_create.restype = ctypes.c_void_p

        self.lib.mpc_destroy.argtypes = [ctypes.c_void_p]
        self.lib.mpc_destroy.restype = None

        self.lib.mpc_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int
        ]
        self.lib.mpc_step.restype = ctypes.c_double

        # 创建 C++ 实例
        self.handle = self.lib.mpc_create(
            dt_control, dt_sim,
            J, tau_c, tau_s, omega_s, b,
            self.angle_limit, disturbance_torque,
            N, Q, R, Rd,
            u_min, u_max, max_iter,
            delay_par_n
        )
        if self.handle is None:
            raise RuntimeError("Failed to create MPC controller")

    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, 'handle') and self.handle:
            self.lib.mpc_destroy(self.handle)

    def step(self, state, theta_ref):
        """
        state: [theta, omega] 当前状态
        theta_ref: 未来参考轨迹数组，长度至少为 N+1
        注意：返回的是 delay_par_n 步之前求解得到的最优控制序列的第 delay_par_n 步，
        前 delay_par_n 次调用直接返回 0.0。
        """
        theta, omega = state
        if isinstance(theta_ref, (list, tuple, np.ndarray)):
            ref_list = [float(x) for x in theta_ref]
        else:
            ref_list = [float(theta_ref)] * (self.N + 1)

        if len(ref_list) < self.N + 1:
            raise ValueError(f"theta_ref length must be at least {self.N+1}, got {len(ref_list)}")

        arr_type = ctypes.c_double * (self.N + 1)
        ref_arr = arr_type(*ref_list[:self.N+1])

        u_opt = self.lib.mpc_step(self.handle, theta, omega, ref_arr, self.N + 1)
        return u_opt
    