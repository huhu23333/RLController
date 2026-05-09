#ifndef MPC_CONTROLLER_H
#define MPC_CONTROLLER_H

#include <vector>
#include <utility>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>
#include <atomic>
#include <map>
#include <memory>
#include <limits>

class MPCController {
public:
    MPCController(double dt_control, double dt_sim,
                  double J, double tau_c, double tau_s, double omega_s, double b,
                  double angle_limit, double disturbance_torque,
                  int N, double Q, double R, double Rd,
                  double u_min, double u_max, int max_iter,
                  int delay_par_n = 3);

    ~MPCController();

    double step(double theta, double omega, const std::vector<double>& theta_ref);

    // 公有方法，供 CostFunctor 调用
    std::vector<double> predictTrajectory(double theta0, double omega0, const std::vector<double>& u_seq) const;
    static double normalizeAngle(double angle);

private:
    // 模型参数
    double dt_control_, dt_sim_;
    int steps_per_control_;
    double J_, tau_c_, tau_s_, omega_s_, b_;
    double angle_limit_, disturbance_torque_;
    int N_;
    double Q_, R_, Rd_;
    double u_min_, u_max_;
    int max_iter_;

    // 延迟步数
    int delay_par_n_;

    // 上一次最优序列 (由优化任务读取/更新，需加锁)
    mutable std::mutex prev_u_mutex_;
    std::vector<double> prev_u_seq_;

    // 随机数
    mutable std::mt19937 rng_;
    double noise_amplitude_;

    static constexpr double eps_omega_ = 1e-8;

    // 摩擦模型和动力学
    double frictionTorque(double tau_motor, double omega) const;
    std::pair<double, double> dynamicsStep(double theta, double omega, double tau_motor, double dt) const;

    // 线程池相关
    void workerLoop();
    void submitTask(std::function<void()> task);
    void shutdownThreadPool();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    bool stop_;

    // 每次 step 的子任务状态
    struct StepTaskState {
        uint64_t step_index = 0;
        int num_subtasks = 0;
        int completed_subtasks = 0;
        bool all_done = false;
        bool best_valid = false;
        std::vector<double> best_u_seq;
        double best_cost = std::numeric_limits<double>::max();
        std::mutex mtx;
        std::condition_variable cv;
    };

    // 保存各 step 的任务状态
    std::mutex results_mutex_;
    std::condition_variable results_cv_;
    std::map<uint64_t, std::shared_ptr<StepTaskState>> step_states_;

    // 步数计数器
    std::atomic<uint64_t> step_counter_{0};
};

#endif // MPC_CONTROLLER_H
