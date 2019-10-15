import numpy as np
from pendulum_MPC_sim import simulate_pendulum_MPC, get_parameter
from numpy.random import seed
import matplotlib.pyplot as plt
from objective_function import f_x, get_simoptions_x
from pendulum_model import RAD_TO_DEG
import pickle
import os
from scipy.interpolate import interp1d
if __name__ == '__main__':

    algo = 'IDWGOPT'

    machine = 'PI'#'PC'
    eps_calc = 1.0
    iter_max_plot = 500

    plt.close('all')
    res_filename = f"res_slower{eps_calc:.0f}_500iter_{algo}_{machine}.pkl"
    results = pickle.load(open(res_filename, "rb"))


    # In[]
    FIG_FOLDER = 'fig'
    if not os.path.isdir(FIG_FOLDER):
        os.makedirs(FIG_FOLDER)

    # In[Re-simulate]

    ## Re-simulate with the optimal point
    x_opt = results['x_opt']

    simopt = get_simoptions_x(x_opt)
    t_ref_vec = np.array([0.0, 5.0, 10.0,  13.0,   20.0, 22.0,  25.0, 30.0, 35.0,  40.0, 100.0])
    p_ref_vec = np.array([0.0, 0.4,  0.0,   0.9,    0.9,  0.4,   0.4,  0.4,  0.0,   0.0, 0.0])
    rp_fun = interp1d(t_ref_vec, p_ref_vec, kind='linear')
    def xref_fun_def(t):
        return np.array([rp_fun(t), 0.0, 0.0, 0.0])
    simopt['xref_fun'] = xref_fun_def


    simout = simulate_pendulum_MPC(simopt)

    t = simout['t']
    x = simout['x']
    u = simout['u']
    y = simout['y']
    y_meas = simout['y_meas']
    x_ref = simout['x_ref']
    x_MPC_pred = simout['x_MPC_pred']
    x_fast = simout['x_fast']
    x_ref_fast = simout['x_ref_fast']
    y_ref = x_ref[:, [0, 2]]  # on-line predictions from the Kalman Filter
    uref = get_parameter({}, 'uref')
    u_fast = simout['u_fast']

    t_int = simout['t_int_fast']
    t_fast = simout['t_fast']
    t_calc = simout['t_calc']

    fig, axes = plt.subplots(3, 1, figsize=(8, 6))
    #    axes[0].plot(t, y_meas[:, 0], "r", label='p_meas')
    axes[0].plot(t_fast, x_fast[:, 0], "k", label='$p$')
    axes[0].plot(t, y_ref[:, 0], "r--", label="$p^{ref}$", linewidth=2)
    axes[0].set_ylim(-0.2, 1.0)
    axes[0].set_ylabel("Position (m)")

    #    axes[1].plot(t, y_meas[:, 1] * RAD_TO_DEG, "r", label='phi_meas')
    axes[1].plot(t_fast, x_fast[:, 2] * RAD_TO_DEG, 'k', label="$\phi$")
    idx_pred = 0
    axes[1].set_ylim(-12, 12)
    axes[1].set_ylabel("Angle (deg)")

    axes[2].plot(t, u[:, 0], 'k', label="u")
    # axes[2].plot(t, uref * np.ones(np.shape(t)), "r--", label="u_ref")
    axes[2].set_ylim(-8, 8)
    axes[2].set_ylabel("Force (N)")
    axes[2].set_xlabel("Simulation time (s)")

    for ax in axes:
        ax.grid(True)
        ax.legend(loc='upper right')

    fig_name = f"BEST_{algo}_{machine}.pdf"
    fig_path = os.path.join(FIG_FOLDER, fig_name)
    fig.savefig(fig_path, bbox_inches='tight')

    # MPC time check
    # In[MPC computation time ]
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(t, y_meas[:, 0], "r", label='p_meas')
    axes[0].plot(t_fast, x_fast[:, 0], "k", label='p')
    axes[0].step(t, y_ref[:, 0], "k--", where='post', label="p_ref")
    axes[0].set_ylim(-0.2, 1.0)
    axes[0].set_xlabel("Simulation time (s)")
    axes[0].set_ylabel("Position (m)")

    axes[1].step(t, t_calc[:, 0] * 1e3, "b", where='post', label='T_MPC')
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("MPC time (ms)")
    axes[1].set_ylim(0, 4)
    axes[2].step(t_fast[1:], t_int[1:, 0] * 1e3, "b", where='post', label='T_ODE')
    axes[2].set_xlabel("Simulation time (s)")
    axes[2].set_ylabel("ODE time (ms)")
    axes[2].set_ylim(0, 0.3)
    axes[3].step(t, u[:, 0], where='post', label="F")
    axes[3].step(t_fast, u_fast[:, 0], where='post', label="F_d")
    axes[3].set_xlabel("Simulation time (s)")
    axes[3].set_ylabel("Force (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    # In[Iteration plot]

    Y = results['J_sample']
    Ts_MPC = simout['Ts_MPC']

    Y_best_curr = np.zeros(np.shape(Y))
    Y_best_val = Y[0]
    iter_best_val = 0

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    axes = [axes]
    for i in range(len(Y_best_curr)):
        if Y[i] < Y_best_val:
            Y_best_val = Y[i]
            iter_best_val = i
        Y_best_curr[i] = Y_best_val

    N = len(Y)
    iter = np.arange(1, N + 1, dtype=np.int)
    axes[0].plot(iter, Y, 'k*', label='Current test point')
#    axes[0].plot(iter, Y_best_curr, 'r', label='Current best point')
    axes[0].plot(iter, Y_best_val*np.ones(Y.shape), '-', label='Overall best point', color='red')
    axes[0].set_xlabel("Iteration index n (-)")
    axes[0].set_ylabel(r"Performance cost $\tilde {J}^{\mathrm{cl}}$")

    for ax in axes:
        ax.grid(True)
        ax.legend(loc='upper right')

    axes[0].set_xlim((0, iter_max_plot))
    axes[0].set_ylim((-1, 19))

    fig_name = f"ITER_{algo}_{machine}.pdf"
    fig_path = os.path.join(FIG_FOLDER, fig_name)
    fig.savefig(fig_path, bbox_inches='tight')

    # In[Recompute optimum]
    J_opt = f_x(x_opt, eps_calc=results['eps_calc'])
    print(J_opt)

    # In[Optimization computation time]

    t_unknown = results['time_iter'] - (
                results['time_f_eval'] + results['time_opt_acquisition'] + results['time_fit_surrogate'])
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.step(iter, results['time_iter'], 'k', where='post', label='Total')
    ax.step(iter, results['time_f_eval'], 'r', where='post', label='Eval')
    ax.step(iter, results['time_opt_acquisition'], 'y', where='post', label='Opt')
    ax.step(iter, results['time_fit_surrogate'], 'g', where='post', label='Fit')
    ax.grid(True)
    ax.legend()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.step(iter, np.cumsum(results['time_iter']), 'k', where='post', label='Total')
    ax.step(iter, np.cumsum(results['time_f_eval']), 'r', where='post', label='Function evaluation')
    ax.step(iter, np.cumsum(results['time_fit_surrogate']), 'g', where='post', label='Surrogate fitting')
    ax.step(iter, np.cumsum(results['time_opt_acquisition']), 'y', where='post', label='Surrogate optimization')
    # ax.step(iter, np.cumsum(t_unknown), 'g', where='post', label='Unknown')
    ax.set_xlabel("Iteration index i (-)")
    ax.set_ylabel("Comulative computational time (s)")
    ax.grid(True)
    ax.legend()

    fig_name = f"COMPUTATION_{algo}_{machine}.pdf"
    fig_path = os.path.join(FIG_FOLDER, fig_name)
    fig.savefig(fig_path, bbox_inches='tight')

    residual_time = np.sum(results['time_iter']) - np.sum(results['time_f_eval']) - np.sum(
        results['time_opt_acquisition']) - np.sum(results['time_fit_surrogate'])
