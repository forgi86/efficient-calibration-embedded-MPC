import numpy as np
from pendulum_MPC_sim import simulate_pendulum_MPC, get_parameter, Ts_fast, get_default_parameters
from scipy import sparse
from pendulum_model import *

N_eval = 0

VAR_NAMES = ['QDu_scale',
             'Qy11',
             'Qy22',
             'Np',
             'Nc_perc',
             'Ts_MPC',
             'QP_eps_abs_log',
             'QP_eps_rel_log',
             'Q_kal_11',
             'Q_kal_22',
             'Q_kal_33',
             'Q_kal_44',
             'R_kal_11',
             'R_kal_22',
             ]


def dict_to_x(dict_x):
    N_vars1 = len(dict_x)
    N_vars2 = len(VAR_NAMES)
    assert (N_vars1 == N_vars2)
    N_vars = N_vars1

    x = np.zeros(N_vars)
    for var_idx in range(N_vars):
        x[var_idx] = dict_x[VAR_NAMES[var_idx]]
    return x


def x_to_dict(x):
    if len(x.shape) == 2:
        x = x[0]
    N_vars1 = len(x)
    N_vars2 = len(VAR_NAMES)
    assert (N_vars1 == N_vars2)
    N_vars = N_vars1

    dict_x = {}
    for var_idx in range(N_vars):
        dict_x[VAR_NAMES[var_idx]] = x[var_idx]
    return dict_x


def get_simoptions_x(x):
    so = x_to_dict(x)  # simopt

    # MPC cost: weight matrices
    Qx = sparse.diags([so['Qy11'], 0, so['Qy22'], 0])  # /sum_MPC_Qy   # Quadratic cost for states x0, x1, ..., x_N-1
    QxN = Qx
    QDu = so['QDu_scale'] * sparse.eye(1)  # Quadratic cost for Du0, Du1, ...., Du_N-1

    so['Qx'] = Qx
    so['QxN'] = QxN
    so['Qu'] = 0.0 * sparse.eye(1)
    so['QDu'] = QDu

    # MPC cost: prediction and control horizon
    so['Np'] = np.int(round(so['Np']))
    so['Nc'] = np.int(round(so['Np'] * so['Nc_perc']))

    # MPC cost: sample time
    # Make Ts_MPC a multiple of Ts_fast
    Ts_MPC = so['Ts_MPC']
    Ts_MPC = ((Ts_MPC // Ts_fast)) * Ts_fast  # make Ts_MPC an integer multiple of Ts_fast
    so['Ts_MPC'] = Ts_MPC

    # MPC: solver settings
    so['QP_eps_abs'] = 10 ** so['QP_eps_abs_log']
    so['QP_eps_rel'] = 10 ** so['QP_eps_rel_log']

    # Kalman filter: matrices
    Q_kal = np.diag([so['Q_kal_11'], so['Q_kal_22'], so['Q_kal_33'], so['Q_kal_44']])
    R_kal = np.diag([so['R_kal_11'], so['R_kal_22']])

    so['Q_kal'] = Q_kal
    so['R_kal'] = R_kal

    # Fixed simulation settings
    so['std_nphi'] = 0.01
    so['std_npos'] = 0.02

    so['std_dF'] = 0.1
    so['w_F'] = 5

    return so


def f_x(x, eps_calc=1.0, seed_val=None):
    global N_eval

    if seed_val is None:
        seed_val = N_eval

    simoptions = get_simoptions_x(x)
    simoptions['seed_val'] = seed_val

    sim_failed = False
    try:
        simout = simulate_pendulum_MPC(simoptions)
    except ValueError as e:
        print(e)
        sim_failed = True

    if not sim_failed:
        t = simout['t']
        y_meas = simout['y_meas']
        x_ref = simout['x_ref']
        p_meas = y_meas[:, 0]
        phi_meas = y_meas[:, 1]

        p_ref = x_ref[:, 0]
        phi_ref = x_ref[:, 2]

        J_perf = 10 * np.mean(np.abs(p_ref - p_meas)) + 0.0 * np.max(np.abs(p_ref - p_meas)) + \
                 30 * np.mean(np.abs(np.abs(phi_ref - phi_meas)))  # + 15*np.max(np.abs(np.abs(phi_ref - phi_meas)))

        # Computation of the barrier function
        t_calc = simout['t_calc']
        eps_margin = 0.8

        t_calc = eps_calc * t_calc
        t_calc_wc = np.max(t_calc)  # worst-case computational cost (max computational time)

        Ts_MPC = simout['Ts_MPC']
        t_available = Ts_MPC * eps_margin

        delay_wc = (t_calc_wc - t_available)
        delay_wc = delay_wc * (delay_wc >= 0)
        J_calc = (delay_wc / t_available) * 1e3

        emergency = simout['emergency_fast']
        emergency_time, _ = np.where(emergency > 0)
        if len(emergency_time) > 0:
            J_emergency = (len(emergency) - emergency_time[0]) / len(emergency) * 1e3
        else:
            J_emergency = 0.0

    else:
        J_perf = 1e3
        J_calc = 1e3
        J_emergency = 1e3  # (len(emergency) - emergency_time[0]) / len(emergency) * 1e3
        # J_perf = 2e1
        # J_fit = 2e1

    print(f"N_eval: {N_eval}, J_perf:{J_perf:.2f}, J_calc:{J_calc:.2f}, J_emergency:{J_emergency:.2f}")
    N_eval += 1

    return np.log(J_perf) + np.log(1 + J_calc + J_emergency)  # + J_fit


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    dict_x0 = {

        'QDu_scale': 0.001,
        'Qy11': 0.1,
        'Qy22': 0.5,
        'Np': 100,
        'Nc_perc': 0.5,
        'Ts_MPC': 10e-3,
        'QP_eps_abs_log': -3,
        'QP_eps_rel_log': -3,
        'Q_kal_11': 0.1,
        'Q_kal_22': 0.9,
        'Q_kal_33': 0.1,
        'Q_kal_44': 0.9,
        'R_kal_11': 0.5,
        'R_kal_22': 0.5
    }
    x0 = dict_to_x(dict_x0)

    f_x0 = x_to_dict(x0)
    J_tot = f_x(x0)

    simopt = get_simoptions_x(x0)
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

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(t, y_meas[:, 0], "b", label='p_meas')
    axes[0].plot(t, y[:, 0], "k", label='p')
    axes[0].plot(t, y_ref[:, 0], "k--", label="p_ref")
    axes[0].set_title("Position (m)")

    axes[1].plot(t, y_meas[:, 1] * RAD_TO_DEG, "b", label='phi_meas')
    axes[1].plot(t, y[:, 1] * RAD_TO_DEG, 'k', label="phi")
    axes[1].plot(t, y_ref[:, 1] * RAD_TO_DEG, "k--", label="phi_ref")
    axes[1].set_title("Angle (deg)")

    axes[2].plot(t, u[:, 0], label="u")
    axes[2].plot(t, uref * np.ones(np.shape(t)), "r--", label="u_ref")
    axes[2].set_title("Force (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    default = get_default_parameters(simopt)

    t_calc = simout['t_calc']
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.hist(t_calc * 1e3)
    plt.title("Computation time (ms)")

    t_int = simout['t_int_fast']
    t_fast = simout['t_fast']
    u_fast = simout['u_fast']

    # MPC time check
    # In[MPC computation time ]
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(t, y_meas[:, 0], "b", label='p_meas')
    axes[0].plot(t_fast, x_fast[:, 0], "k", label='p')
    axes[0].step(t, y_ref[:, 0], "k--", where='post', label="p_ref")
    axes[0].set_ylim(-0.2, 1.0)
    axes[0].set_xlabel("Simulation time (s)")
    axes[0].set_ylabel("Position (m)")

    axes[1].step(t, t_calc[:, 0] * 1e3, "b", where='post', label='T_MPC')
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("MPC time (ms)")
    axes[1].set_ylim(0, 4)
    axes[2].step(t_fast[1:], t_int[1:, 0] * 1e3, "b", where='post', label='T_ODE')  # why is 1st slow? check numba
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
