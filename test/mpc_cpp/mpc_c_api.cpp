#include "mpc_controller.h"
#include <vector>

extern "C" {

// 不透明指针类型
typedef void* MPCHandle;

MPCHandle mpc_create(double dt_control, double dt_sim,
                     double J, double tau_c, double tau_s, double omega_s, double b,
                     double angle_limit, double disturbance_torque,
                     int N, double Q, double R, double Rd,
                     double u_min, double u_max, int max_iter,
                     int delay_par_n) {
    try {
        MPCController* mpc = new MPCController(dt_control, dt_sim,
                                                J, tau_c, tau_s, omega_s, b,
                                                angle_limit, disturbance_torque,
                                                N, Q, R, Rd,
                                                u_min, u_max, max_iter,
                                                delay_par_n);
        return static_cast<MPCHandle>(mpc);
    } catch (...) {
        return nullptr;
    }
}

void mpc_destroy(MPCHandle handle) {
    if (handle) {
        delete static_cast<MPCController*>(handle);
    }
}

double mpc_step(MPCHandle handle, double theta, double omega, const double* theta_ref, int ref_len) {
    if (!handle) return 0.0;
    MPCController* mpc = static_cast<MPCController*>(handle);
    std::vector<double> ref(theta_ref, theta_ref + ref_len);
    try {
        return mpc->step(theta, omega, ref);
    } catch (...) {
        return 0.0;
    }
}

} // extern "C"
