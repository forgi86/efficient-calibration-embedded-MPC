import GPyOpt
import idwgopt.idwgopt
import numpy as np
from pendulum_MPC_sim import simulate_pendulum_MPC, get_parameter, get_default_parameters
import matplotlib.pyplot as plt
from objective_function import dict_to_x, f_x, get_simoptions_x
import pickle
import time
import numba as nb

if __name__ == '__main__':

    np.random.seed(0)     # initialize seed for reproducibility

    # optimization parameters
    # Run the optimization
    # n_init = 10 changed to 2*n_var
    max_iter = 500  # function evaluation budget
    max_time = np.inf  # time budget
    eps = 0.0  # Minimum allows distance between the last two observations (For Bayesian Optimization)
    eps_calc = 1.0 # Simulate results on a machine eps_calc times slower

    # method = 'BO'
    method = "IDWGOPT"
    machine = 'PC'

    dict_x0 = {
        'Qy_scale': 0.9,
        'Qu_scale': 0.0,
        'QDu_scale': 0.1,
        'Qy11': 0.1,
        'Qy22': 0.9,
        'Np': 40,
        'Nc_perc': 1.0,
        'Ts_MPC': 25e-3,
        'QP_eps_abs_log': -3,
        'QP_eps_rel_log': -2,
        'Q_kal_scale': 0.5,
        'Q_kal_11': 0.1,
        'Q_kal_22': 0.4,
        'Q_kal_33': 0.1,
        'Q_kal_44': 0.4,
        'R_kal_scale': 0.5,
        'R_kal_11': 0.5,
        'R_kal_22': 0.5
    }

    dict_context = {}

    n_var = len(dict_x0)
    n_init = 2 * n_var

    x0 = dict_to_x(dict_x0)

    bounds = [
        {'name': 'Qy_scale', 'type': 'continuous', 'domain': (1e-6, 1)},  # 0
        {'name': 'Qu_scale', 'type': 'continuous', 'domain': (0, 1e-12)},  # 1
        {'name': 'QDu_scale', 'type': 'continuous', 'domain': (1e-6, 1)},  # 2
        {'name': 'Qy11', 'type': 'continuous', 'domain': (0, 1)},  # 3
        {'name': 'Qy22', 'type': 'continuous', 'domain': (0, 1)},  # 4
        {'name': 'Np', 'type': 'continuous', 'domain': (5, 300)},  # 5
        {'name': 'Nc_perc', 'type': 'continuous', 'domain': (0.3, 1)},  # 6
        {'name': 'Ts_MPC', 'type': 'continuous', 'domain': (1e-3, 50e-3)},  # 7
        {'name': 'QP_eps_abs_log', 'type': 'continuous', 'domain': (-7, -1)},  # 8
        {'name': 'QP_eps_rel_log', 'type': 'continuous', 'domain': (-7, -1)},  # 9
        {'name': 'Q_kal_scale', 'type': 'continuous', 'domain': (0.01, 1)},  # 10
        {'name': 'R_kal_scale', 'type': 'continuous', 'domain': (0.01, 1)},  # 11
        {'name': 'Q_kal_11', 'type': 'continuous', 'domain': (1e-6, 1)},  # 12
        {'name': 'Q_kal_22', 'type': 'continuous', 'domain': (1e-6, 1)},  # 13
        {'name': 'Q_kal_33', 'type': 'continuous', 'domain': (1e-6, 1)},  # 14
        {'name': 'Q_kal_44', 'type': 'continuous', 'domain': (1e-6, 1)},  # 15
        {'name': 'R_kal_11', 'type': 'continuous', 'domain': (1e-6, 1)},  # 16
        {'name': 'R_kal_22', 'type': 'continuous', 'domain': (1e-6, 1)},  # 17

    ]

    constraints = [
        {'name': 'min_time', 'constraint': '-x[:,5]*x[:,7] + 0.1'},
        # {'name': 'MPC_scale_norm_1', 'constraint': 'x[:,0] + x[:,1] + x[:,2] -1.1'},
        # {'name': 'MPC_scale_norm_2', 'constraint': '-x[:,0] -x[:,1] -x[:,2] +0.9'},
        # {'name': 'Qy_sum_1', 'constraint': 'x[:,3] + x[:,4] - 1.1'},
        # {'name': 'Qy_sum_2', 'constraint': '-x[:,3] -x[:,4] +0.9'},
        # {'name': 'KAL_scale_norm_1', 'constraint': 'x[:,10] + x[:,11]  -1.1'},
        # {'name': 'KAL_scale_norm_2', 'constraint': '-x[:,10] -x[:,11]  +0.9'},
        # {'name': 'Q_KAL_sum_1', 'constraint': 'x[:,12] + x[:,13] + x[:,14] + x[:,15]  -1.1'},
        # {'name': 'Q_KAL_sum_2', 'constraint': '-x[:,12] -x[:,13] -x[:,14] -x[:,15]  +0.9'},
        # {'name': 'R_KAL_sum_1', 'constraint': 'x[:,16] + x[:,17]  -1.1'},
        # {'name': 'R_KAL_sum_2', 'constraint': '-x[:,16] -x[:,17]  +0.9'},
    ]


    def f_x_calc(x):
        return f_x(x, eps_calc)


    feasible_region = GPyOpt.Design_space(space=bounds,
                                          constraints=constraints)  # , constraints=constraints_context)
    X_init = GPyOpt.experiment_design.initial_design('random', feasible_region, n_init)

    time_optimization_start = time.perf_counter()
    if method == "BO":
        myBopt = GPyOpt.methods.BayesianOptimization(f_x_calc,
                                                     X=X_init,
                                                     domain=bounds,
                                                     model_type='GP',
                                                     acquisition_type='EI',
                                                     normalize_Y=True,
                                                     exact_feval=False)

        myBopt.run_optimization(max_iter=max_iter - n_init, max_time=max_time, eps=eps, verbosity=False)

        x_opt = myBopt.x_opt
        J_opt = myBopt.fx_opt
        X_sample = myBopt.X
        J_sample = myBopt.Y
        idx_opt = np.argmin(J_sample)

    if method == "IDWGOPT":

        # IDWGOPT initialization
        nvars = len(bounds)
        lb = np.zeros((nvars, 1)).flatten("c")
        ub = lb.copy()
        for i in range(0, nvars):
            lb[i] = bounds[i]['domain'][0]
            ub[i] = bounds[i]['domain'][1]

        problem = idwgopt.idwgopt.default(nvars)
        problem["nsamp"] = n_init
        problem["maxevals"] = max_iter
        problem["g"] = lambda x: np.array([-x[5] * x[7] + 0.1])
        problem["lb"] = lb
        problem["ub"] = ub
        problem["f"] = f_x_calc
        problem["useRBF"] = 1  # use Radial Basis Functions
        # problem["useRBF"] = 0 # Inverse Distance Weighting
        if problem["useRBF"]:
            epsil = .5


            @nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])], '(n),(n)->()')
            def fun_rbf(x1, x2, res):
                res[0] = 1 / (1 + epsil ** 2 * np.sum((x1 - x2) ** 2))


            problem['rbf'] = fun_rbf

        problem["alpha"] = 1
        problem["delta"] = .5

        problem["svdtol"] = 1e-6
        # problem["globoptsol"] = "direct"
        problem["globoptsol"] = "pswarm"
        problem["display"] = 0

        problem["scalevars"] = 1

        problem["constraint_penalty"] = 1e3
        problem["feasible_sampling"] = False

        problem["shrink_range"] = 0  # 0 = don't shrink lb/ub

        tic = time.perf_counter()
        out = idwgopt.idwgopt.solve(problem)
        toc = time.perf_counter()

        x_opt = out["xopt"]
        J_opt = out["fopt"]
        J_sample = out["F"]
        X_sample = out["X"]
        idx_opt = np.argmin(J_sample, axis=0)

    time_optimization = time.perf_counter() - time_optimization_start

    print(f"J_best_val: {J_opt:.3f}")

    # In[Re-simulate with the optimal point]

    simopt = get_simoptions_x(x_opt)
    simout = simulate_pendulum_MPC(simopt)

    tsim = simout['t']
    xsim = simout['x']
    usim = simout['u']

    x_ref = simout['x_ref']
    uref = get_parameter({}, 'uref')

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(tsim, xsim[:, 0], "k", label='p')
    axes[0].plot(tsim, x_ref[:, 0], "r--", label="p_ref")
    axes[0].set_title("Position (m)")

    axes[1].plot(tsim, xsim[:, 2] * 360 / 2 / np.pi, label="phi")
    axes[1].plot(tsim, x_ref[:, 2] * 360 / 2 / np.pi, "r--", label="phi_ref")
    axes[1].set_title("Angle (deg)")

    axes[2].plot(tsim, usim[:, 0], label="u")
    axes[2].plot(tsim, uref * np.ones(np.shape(tsim)), "r--", label="u_ref")
    axes[2].set_title("Force (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    J_best_curr = np.zeros(np.shape(J_sample))
    J_best_val = J_sample[0]
    iter_best_val = 0

    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    axes = [axes]
    for i in range(len(J_best_curr)):
        if J_sample[i] < J_best_val:
            J_best_val = J_sample[i]
            iter_best_val = i
        J_best_curr[i] = J_best_val

    N = len(J_sample)
    iter = np.arange(0, N, dtype=np.int)
    axes[0].plot(iter, J_sample, 'k*', label='Current test boint')
    axes[0].plot(iter, J_best_curr, 'g', label='Current best point')
    axes[0].plot(iter_best_val, J_best_val, 's', label='Overall best point')

    for ax in axes:
        ax.grid(True)
        ax.legend()

    # In[Re-evaluate optimal controller]
    J_opt = f_x(x_opt)
    print(J_opt)

    plt.show()

    # In[Store results]

    result = {}
    if method == 'BO':
        myBopt.f = None  # hack to solve the issues pickling the myBopt object
        myBopt.objective = None

        results = {'X_sample': myBopt.X, 'J_sample': myBopt.Y,
                   'idx_opt': idx_opt, 'x_opt': x_opt, 'J_opt': J_opt,
                   'eps_calc': eps_calc,
                   'time_iter': myBopt.time_iter, 'time_f_eval': myBopt.time_f_eval,
                   'time_opt_acquisition': myBopt.time_opt_acquisition, 'time_fit_surrogate': myBopt.time_fit_surrogate,
                   'myBopt': myBopt, 'method': method
                   }

    if method == 'IDWGOPT':
        results = {'X_sample': X_sample, 'J_sample': J_sample,
                   'idx_opt': idx_opt, 'x_opt': x_opt, 'J_opt': J_opt,
                   'eps_calc': eps_calc,
                   'time_iter': out['time_iter'], 'time_f_eval': out['time_f_eval'],
                   'time_opt_acquisition': out['time_opt_acquisition'], 'time_fit_surrogate': out['time_fit_surrogate'],
                   'out': out, 'method': method
                   }

    res_filename = f"res_slower{eps_calc:.0f}_{max_iter:.0f}iter_{method}_{machine}.pkl"

    with open(res_filename, "wb") as file:
        pickle.dump(results, file)
