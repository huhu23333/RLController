#include "mpc_controller.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <ceres/ceres.h>

// ------------------------------------------------------------
// 辅助函数 (公有化)
// ------------------------------------------------------------
double MPCController::normalizeAngle(double angle) {
    angle = std::fmod(angle + M_PI, 2.0 * M_PI);
    if (angle < 0) angle += 2.0 * M_PI;
    return angle - M_PI;
}

double MPCController::frictionTorque(double tau_motor, double omega) const {
    if (std::abs(omega) < eps_omega_) {
        if (std::abs(tau_motor) <= tau_s_)
            return tau_motor;
        else
            return (tau_motor > 0 ? tau_c_ : -tau_c_);
    } else {
        double sign_omega = (omega > 0) ? 1.0 : -1.0;
        double tau_f = (tau_c_ + (tau_s_ - tau_c_) * std::exp(-std::abs(omega) / omega_s_)) * sign_omega;
        tau_f += b_ * omega;
        return tau_f;
    }
}

std::pair<double, double> MPCController::dynamicsStep(double theta, double omega, double tau_motor, double dt) const {
    double tau_f = frictionTorque(tau_motor, omega);
    double tau_net = tau_motor - tau_f - disturbance_torque_;
    double alpha = tau_net / J_;
    double omega_new = omega + alpha * dt;
    double theta_new = theta + omega_new * dt;

    if (angle_limit_ > 0.0) {
        if (theta_new > angle_limit_) {
            theta_new = angle_limit_;
            omega_new = 0.0;
        } else if (theta_new < -angle_limit_) {
            theta_new = -angle_limit_;
            omega_new = 0.0;
        }
    }
    return {theta_new, omega_new};
}

std::vector<double> MPCController::predictTrajectory(double theta0, double omega0, const std::vector<double>& u_seq) const {
    std::vector<double> theta_pred;
    theta_pred.reserve(N_ + 1);
    double theta = theta0, omega = omega0;
    theta_pred.push_back(theta);
    for (int k = 0; k < N_; ++k) {
        double tau = u_seq[k];
        for (int step = 0; step < steps_per_control_; ++step) {
            auto [theta_new, omega_new] = dynamicsStep(theta, omega, tau, dt_sim_);
            theta = theta_new;
            omega = omega_new;
        }
        theta_pred.push_back(theta);
    }
    return theta_pred;
}

// ------------------------------------------------------------
// Ceres 代价函数
// ------------------------------------------------------------
class MPCCostFunctor {
public:
    MPCCostFunctor(const MPCController* mpc,
                   double theta0, double omega0,
                   const std::vector<double>& theta_ref,
                   int N, double Q, double R, double Rd)
        : mpc_(mpc), theta0_(theta0), omega0_(omega0),
          theta_ref_(theta_ref), N_(N), Q_(Q), R_(R), Rd_(Rd) {}

    bool operator()(double const* const* parameters, double* residuals) const {
        const double* u = parameters[0];
        std::vector<double> u_seq(u, u + N_);
        std::vector<double> theta_pred = mpc_->predictTrajectory(theta0_, omega0_, u_seq);
        int idx = 0;
        for (int k = 1; k <= N_; ++k) {
            double err = theta_pred[k] - theta_ref_[k];
            err = MPCController::normalizeAngle(err);
            residuals[idx++] = std::sqrt(Q_) * err;
        }
        for (int k = 0; k < N_; ++k) {
            residuals[idx++] = std::sqrt(R_) * u_seq[k];
        }
        for (int k = 1; k < N_; ++k) {
            residuals[idx++] = std::sqrt(Rd_) * (u_seq[k] - u_seq[k-1]);
        }
        return true;
    }

private:
    const MPCController* mpc_;
    double theta0_, omega0_;
    const std::vector<double> theta_ref_;
    int N_;
    double Q_, R_, Rd_;
};

// ------------------------------------------------------------
// 构造函数与析构函数
// ------------------------------------------------------------
MPCController::MPCController(double dt_control, double dt_sim,
                             double J, double tau_c, double tau_s, double omega_s, double b,
                             double angle_limit, double disturbance_torque,
                             int N, double Q, double R, double Rd,
                             double u_min, double u_max, int max_iter,
                             int delay_par_n)
    : dt_control_(dt_control), dt_sim_(dt_sim),
      J_(J), tau_c_(tau_c), tau_s_(tau_s), omega_s_(omega_s), b_(b),
      angle_limit_(angle_limit), disturbance_torque_(disturbance_torque),
      N_(N), Q_(Q), R_(R), Rd_(Rd), u_min_(u_min), u_max_(u_max), max_iter_(max_iter),
      delay_par_n_(delay_par_n),
      rng_(std::random_device{}()), noise_amplitude_(0.1 * (u_max - u_min)),
      stop_(false)
{
    double ratio = dt_control_ / dt_sim_;
    steps_per_control_ = static_cast<int>(std::round(ratio));
    if (std::abs(steps_per_control_ * dt_sim_ - dt_control_) > 1e-9) {
        throw std::invalid_argument("dt_control must be an integer multiple of dt_sim");
    }
    prev_u_seq_.assign(N_, 0.0);

    // 启动线程池（硬件并发数）
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 2;
    workers_.reserve(num_threads);
    for (unsigned int i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&MPCController::workerLoop, this);
    }
    std::cout << num_threads << " threads, delay_par_n = " << delay_par_n_ << std::endl;
}

MPCController::~MPCController() {
    shutdownThreadPool();
}

void MPCController::workerLoop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

void MPCController::submitTask(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        tasks_.push(std::move(task));
    }
    cv_.notify_one();
}

void MPCController::shutdownThreadPool() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
}

// ------------------------------------------------------------
// 核心 step 方法（异步提交，延迟返回历史结果）
// ------------------------------------------------------------
double MPCController::step(double theta, double omega, const std::vector<double>& theta_ref) {
    if (theta_ref.size() < static_cast<size_t>(N_ + 1)) {
        throw std::invalid_argument("theta_ref length must be at least N+1");
    }

    // 1. 当前步序号
    uint64_t current_step = step_counter_.fetch_add(1);

    // 2. 生成候选初始猜测（使用加锁读取 prev_u_seq_）
    std::vector<double> prev_u;
    {
        std::lock_guard<std::mutex> lock(prev_u_mutex_);
        prev_u = prev_u_seq_;
    }

    std::vector<std::vector<double>> candidates;
    std::uniform_real_distribution<double> noise_dist(-noise_amplitude_, noise_amplitude_);
    std::uniform_real_distribution<double> rand_dist(u_min_, u_max_);

    // 2.1 平移上一次最优序列
    std::vector<double> u_shift(N_);
    if (!prev_u.empty()) {
        for (int i = 0; i < N_-1; ++i) u_shift[i] = prev_u[i+1];
        u_shift[N_-1] = prev_u[N_-1];
    } else {
        std::fill(u_shift.begin(), u_shift.end(), 0.0);
    }
    candidates.push_back(u_shift);

    // 2.2 平移 + 随机噪声 (2组)
    for (int k = 0; k < 2; ++k) {
        std::vector<double> u_noisy = u_shift;
        for (int i = 0; i < N_; ++i) {
            u_noisy[i] += noise_dist(rng_);
            u_noisy[i] = std::clamp(u_noisy[i], u_min_, u_max_);
        }
        candidates.push_back(u_noisy);
    }

    // 2.3 完全随机序列 (2组)
    for (int k = 0; k < 2; ++k) {
        std::vector<double> u_rand(N_);
        for (int i = 0; i < N_; ++i) {
            u_rand[i] = rand_dist(rng_);
        }
        candidates.push_back(u_rand);
    }

    // 3. 创建本次 step 的状态，并注册到全局 map
    auto state = std::make_shared<StepTaskState>();
    state->step_index = current_step;
    state->num_subtasks = candidates.size();
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        step_states_[current_step] = state;
    }

    // 4. 将每个初始猜测的优化作为子任务提交到线程池
    //    需要复制 theta_ref 以确保异步安全
    std::vector<double> ref_copy = theta_ref;  // 拷贝一份
    for (size_t i = 0; i < candidates.size(); ++i) {
        std::vector<double> u0 = candidates[i];
        submitTask([this, state, theta, omega, ref_copy, u0, i]() {
            std::vector<double> u_opt = u0;
            ceres::Problem problem;
            MPCCostFunctor* cost_functor = new MPCCostFunctor(this, theta, omega, ref_copy, N_, Q_, R_, Rd_);
            int num_residuals = N_ + N_ + (N_ - 1);
            auto* cost_function = new ceres::DynamicNumericDiffCostFunction<MPCCostFunctor>(cost_functor, ceres::TAKE_OWNERSHIP);
            cost_function->SetNumResiduals(num_residuals);
            cost_function->AddParameterBlock(N_);
            problem.AddResidualBlock(cost_function, nullptr, u_opt.data());

            for (int j = 0; j < N_; ++j) {
                problem.SetParameterLowerBound(u_opt.data(), j, u_min_);
                problem.SetParameterUpperBound(u_opt.data(), j, u_max_);
            }

            ceres::Solver::Options options;
            options.max_num_iterations = max_iter_;
            options.function_tolerance = 1e-8;
            options.parameter_tolerance = 1e-8;
            options.minimizer_progress_to_stdout = false;
            options.num_threads = 1;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            bool ok = summary.IsSolutionUsable();
            double cost = ok ? summary.final_cost : std::numeric_limits<double>::max();

            // 更新 state 中的全局最佳结果
            {
                std::lock_guard<std::mutex> lock(state->mtx);
                if (ok && cost < state->best_cost) {
                    state->best_cost = cost;
                    state->best_u_seq = std::move(u_opt);
                    state->best_valid = true;
                }
                state->completed_subtasks++;
                if (state->completed_subtasks == state->num_subtasks) {
                    state->all_done = true;
                    if (state->best_valid) {
                        // 更新 prev_u_seq_ 供后续 step 使用
                        std::lock_guard<std::mutex> lock_prev(prev_u_mutex_);
                        prev_u_seq_ = state->best_u_seq;
                    }
                    state->cv.notify_all();
                }
            }

            // 通知可能在等待该步完成的 step 线程
            {
                std::lock_guard<std::mutex> lock(results_mutex_);
                results_cv_.notify_all();
            }
        });
    }

    // 5. 如果还没有足够的历史步，直接返回 0.0
    if (current_step < static_cast<uint64_t>(delay_par_n_)) {
        return 0.0;
    }
    uint64_t target_step = current_step - delay_par_n_;

    // 6. 等待 target_step 的任务全部完成
    std::shared_ptr<StepTaskState> target_state;
    {
        std::unique_lock<std::mutex> lock(results_mutex_);
        results_cv_.wait(lock, [this, target_step] {
            auto it = step_states_.find(target_step);
            if (it != step_states_.end()) {
                std::lock_guard<std::mutex> lk(it->second->mtx);
                return it->second->all_done;
            }
            return false;
        });
        target_state = step_states_[target_step];

        // 7. 清理更旧的状态（保留 target_step 及之后）
        auto it = step_states_.begin();
        while (it != step_states_.end() && it->first < target_step) {
            it = step_states_.erase(it);
        }
    }

    // 8. 从历史结果中取出第 delay_par_n_ 步的控制量
    double u_result = 0.0;
    {
        std::lock_guard<std::mutex> lock(target_state->mtx);
        if (target_state->best_valid && !target_state->best_u_seq.empty()) {
            int idx = std::min(delay_par_n_, static_cast<int>(target_state->best_u_seq.size()) - 1);
            u_result = target_state->best_u_seq[idx];
        }
    }
    return u_result;
}
