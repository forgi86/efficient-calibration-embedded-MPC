import numpy as np
import scipy.sparse as sparse
from pyMPC.mpc import MPCController
from kalman import kalman_design_simple, LinearStateEstimator
from pendulum_model import *
from scipy.integrate import ode
from scipy.interpolate import interp1d
import time
import control
import control.matlab
import numpy.random

Ts_fast = 1e-3

Ac_def = np.array([[0, 1, 0, 0],
               [0, -b / M, -(g * m) / M, (ftheta * m) / M],
               [0, 0, 0, 1],
               [0, b / (M * l), (M * g + g * m) / (M * l), -(M * ftheta + ftheta * m) / (M * l)]])

Bc_def = np.array([
    [0.0],
    [1.0 / M],
    [0.0],
    [-1 / (M * l)]
])

# Reference input and states
#t_ref_vec = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
#p_ref_vec = np.array([0.0, 0.8, 0.8, 0.0, 0.0])
#rp_fun = interp1d(t_ref_vec, p_ref_vec, kind='zero')
t_ref_vec = np.array([0.0, 5.0, 10.0, 20.0, 25.0, 30.0, 40.0, 100.0])
p_ref_vec = np.array([0.0, 0.0,  0.8, 0.8,  0.0,  0.0,  0.8, 0.8])
rp_fun = interp1d(t_ref_vec, p_ref_vec, kind='linear')


def xref_fun_def(t):
    return np.array([rp_fun(t), 0.0, 0.0, 0.0])


#Qx_def = 0.9 * sparse.diags([0.1, 0, 0.9, 0])   # Quadratic cost for states x0, x1, ..., x_N-1
#QxN_def = Qx_def
#Qu_def = 0.0 * sparse.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
#QDu_def = 0.01 * sparse.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

Ts_MPC_def = 10e-3
Qx_def = 1.0 * sparse.diags([1.0, 0, 5.0, 0])   # Quadratic cost for states x0, x1, ..., x_N-1
QxN_def = Qx_def

Qu_def = 0.0 * sparse.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
QDu_def = 1e-5/(Ts_MPC_def**2) * sparse.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

DEFAULTS_PENDULUM_MPC = {
    'xref_fun': xref_fun_def,
    'uref':  np.array([0.0]), # N
    'std_npos': 0.02,  # m
    'std_nphi': 0.01,  # rad
    'std_dF': 0.1,  # N
    'w_F': 10,  # rad
    'len_sim': 40, #s

    'Ac': Ac_def,
    'Bc': Bc_def,
    'Ts_MPC': Ts_MPC_def,
    'Q_kal':  np.diag([0.1, 10, 0.1, 10]),
    'R_kal': 1*np.eye(2),

    'Np': 100,
    'Nc': 50,
    'Qx': Qx_def,
    'QxN': QxN_def,
    'Qu': Qu_def,
    'QDu': QDu_def,
    'QP_eps_abs': 1e-3,
    'QP_eps_rel': 1e-3,
    'seed_val': None

}


def get_parameter(sim_options, par_name):
    return sim_options.get(par_name, DEFAULTS_PENDULUM_MPC[par_name])


def get_default_parameters(sim_options):
    """ Which parameters are left to default ??"""
    default_keys = [key for key in DEFAULTS_PENDULUM_MPC if key not in sim_options]
    return default_keys


def simulate_pendulum_MPC(sim_options):

    seed_val = get_parameter(sim_options,'seed_val')
    if seed_val is not None:
        np.random.seed(seed_val)

    Ac = get_parameter(sim_options, 'Ac')
    Bc = get_parameter(sim_options, 'Bc')

    Cc = np.array([[1., 0., 0., 0.],
                   [0., 0., 1., 0.]])

    Dc = np.zeros((2, 1))

    [nx, nu] = Bc.shape  # number of states and number or inputs
    ny = np.shape(Cc)[0]

    Ts_MPC = get_parameter(sim_options, 'Ts_MPC')
    ratio_Ts = int(Ts_MPC // Ts_fast)
    Ts_MPC = ((Ts_MPC // Ts_fast)) * Ts_fast # make Ts_MPC an integer multiple of Ts_fast

    # Brutal forward euler discretization
    Ad = np.eye(nx) + Ac*Ts_MPC
    Bd = Bc*Ts_MPC
    Cd = Cc
    Dd = Dc

    # Standard deviation of the measurement noise on position and angle

    std_npos = get_parameter(sim_options, 'std_npos')
    std_nphi = get_parameter(sim_options, 'std_nphi')

    # Force disturbance
    std_dF = get_parameter(sim_options, 'std_dF')

    # disturbance power spectrum
    w_F = get_parameter(sim_options, 'w_F') # bandwidth of the force disturbance
    tau_F = 1 / w_F
    Hu = control.TransferFunction([1], [1 / w_F, 1])
    Hu = Hu * Hu
    Hud = control.matlab.c2d(Hu, Ts_fast)
    N_sim_imp = tau_F / Ts_fast * 20
    t_imp = np.arange(N_sim_imp) * Ts_fast
    t, y = control.impulse_response(Hud, t_imp)
    y = y[0]
    std_tmp = np.sqrt(np.sum(y ** 2))  # np.sqrt(trapz(y**2,t))
    Hu = Hu / (std_tmp) * std_dF


    N_skip = int(20 * tau_F // Ts_fast) # skip initial samples to get a regime sample of d
    t_sim_d = get_parameter(sim_options, 'len_sim')  # simulation length (s)
    N_sim_d = int(t_sim_d // Ts_fast)
    N_sim_d = N_sim_d + N_skip
    e = np.random.randn(N_sim_d)
    te = np.arange(N_sim_d) * Ts_fast
    _, d, _ = control.forced_response(Hu, te, e)
    d_fast = d[N_skip:]
    #td = np.arange(len(d)) * Ts_fast
    

    Np = get_parameter(sim_options, 'Np')
    Nc = get_parameter(sim_options, 'Nc')

    Qx = get_parameter(sim_options, 'Qx')
    QxN = get_parameter(sim_options, 'QxN')
    Qu = get_parameter(sim_options, 'Qu')
    QDu = get_parameter(sim_options, 'QDu')

    # Reference input and states
    xref_fun = get_parameter(sim_options, 'xref_fun') # reference state
    xref_fun_v = np.vectorize(xref_fun, signature='()->(n)')

    t0 = 0
    xref_MPC = xref_fun(t0)
    uref = get_parameter(sim_options, 'uref')
    uminus1 = np.array([0.0])     # input at time step negative one - used to penalize the first delta u at time instant 0. Could be the same as uref.

    # Constraints
    xmin = np.array([-1.5, -100, -100, -100])
    xmax = np.array([1.5,   100.0, 100, 100])

    umin = np.array([-10])
    umax = np.array([10])

    Dumin = np.array([-100*Ts_MPC])
    Dumax = np.array([100*Ts_MPC])

    # Initialize simulation system
    phi0 = 10*2*np.pi/360
    x0 = np.array([0, 0, phi0, 0]) # initial state
    system_dyn = ode(f_ODE_wrapped).set_integrator('vode', method='bdf') #    dopri5
#    system_dyn = ode(f_ODE_wrapped).set_integrator('dopri5')
    system_dyn.set_initial_value(x0, t0)
    system_dyn.set_f_params(0.0)

    QP_eps_rel = get_parameter(sim_options, 'QP_eps_rel')
    QP_eps_abs = get_parameter(sim_options, 'QP_eps_abs')

    # Emergency exit conditions
    
    EMERGENCY_STOP = False
    EMERGENCY_POS = 2.0
    EMERGENCY_ANGLE = 30*DEG_TO_RAD

    K = MPCController(Ad,Bd,Np=Np,Nc=Nc,x0=x0,xref=xref_MPC,uminus1=uminus1,
                      Qx=Qx, QxN=QxN, Qu=Qu,QDu=QDu,
                      xmin=xmin,xmax=xmax,umin=umin,umax=umax,Dumin=Dumin,Dumax=Dumax,
                      eps_feas = 1e3, eps_rel= QP_eps_rel, eps_abs=QP_eps_abs)

    try:
        K.setup(solve=True) # setup initial problem and also solve it
    except:
        EMERGENCY_STOP = True
    
    if not EMERGENCY_STOP:
        if K.res.info.status != 'solved':
            EMERGENCY_STOP = True
        
    # Basic Kalman filter design
    Q_kal =  get_parameter(sim_options, 'Q_kal')
    R_kal = get_parameter(sim_options, 'R_kal')
    L, P, W = kalman_design_simple(Ad, Bd, Cd, Dd, Q_kal, R_kal, type='predictor')
    x0_est = x0
    KF = LinearStateEstimator(x0_est, Ad, Bd, Cd, Dd,L)

    # Simulate in closed loop
    len_sim = get_parameter(sim_options, 'len_sim')  # simulation length (s)
    nsim = int(np.ceil(len_sim / Ts_MPC))  # simulation length(timesteps) # watch out! +1 added, is it correct?
    t_vec = np.zeros((nsim, 1))
    t_calc_vec = np.zeros((nsim,1)) # computational time to get MPC solution (+ estimator)
    status_vec = np.zeros((nsim,1))
    x_vec = np.zeros((nsim, nx))
    x_ref_vec = np.zeros((nsim, nx))
    y_vec = np.zeros((nsim, ny))
    y_meas_vec = np.zeros((nsim, ny))
    y_est_vec = np.zeros((nsim, ny))
    x_est_vec = np.zeros((nsim, nx))
    u_vec = np.zeros((nsim, nu))
    x_MPC_pred = np.zeros((nsim, Np+1, nx)) # on-line predictions from the Kalman Filter

    nsim_fast = int(len_sim // Ts_fast)
    t_vec_fast = np.zeros((nsim_fast, 1))
    x_vec_fast = np.zeros((nsim_fast, nx)) # finer integration grid for performance evaluation
    x_ref_vec_fast = np.zeros((nsim_fast, nx)) # finer integration grid for performance evaluatio
    u_vec_fast = np.zeros((nsim_fast, nu)) # finer integration grid for performance evaluatio
    Fd_vec_fast = np.zeros((nsim_fast, nu))  #
    t_int_vec_fast = np.zeros((nsim_fast, 1))
    emergency_vec_fast = np.zeros((nsim_fast, 1))  #

    t_pred_all = t0 + np.arange(nsim + Np + 1) * Ts_MPC
    Xref_MPC_all = xref_fun_v(t_pred_all)

    t_step = t0
    x_step = x0
    u_MPC = None
    for idx_fast in range(nsim_fast):

        ## Determine step type: fast simulation only or MPC step
        idx_MPC = idx_fast // ratio_Ts
        run_MPC = (idx_fast % ratio_Ts) == 0

        # Output for step i
        # Ts_MPC outputs
        if run_MPC: # it is also a step of the simulation at rate Ts_MPC
            t_vec[idx_MPC, :] = t_step
            x_vec[idx_MPC, :] = x_step#system_dyn.y
            xref_MPC = xref_fun(t_step)  # reference state
            t_pred = t_step + np.arange(Np + 1) * Ts_MPC
            x_ref_vec[idx_MPC,:] = xref_MPC

            if not EMERGENCY_STOP:
#                u_MPC, info_MPC = K.output(return_x_seq=True, return_status=True)  # u[i] = k(\hat x[i]) possibly computed at time instant -1
                u_MPC, info_MPC = K.output(return_status=True)  # u[i] = k(\hat x[i]) possibly computed at time instant -1
            else:
                u_MPC = np.zeros(nu)
                
            if not EMERGENCY_STOP:
                if info_MPC['status'] != 'solved':
                    EMERGENCY_STOP = True
                    
            if not EMERGENCY_STOP:
                pass
                #x_MPC_pred[idx_MPC, :, :] = info_MPC['x_seq']  # x_MPC_pred[i,i+1,...| possibly computed at time instant -1]
            u_vec[idx_MPC, :] = u_MPC

            y_step = Cd.dot(x_step)  # y[i] from the system
            ymeas_step = np.copy(y_step)
            ymeas_step[0] += std_npos * np.random.randn()
            ymeas_step[1] += std_nphi * np.random.randn()
            y_vec[idx_MPC,:] = y_step
            y_meas_vec[idx_MPC,:] = ymeas_step
            if not EMERGENCY_STOP:
                status_vec[idx_MPC,:] = (info_MPC['status'] != 'solved')
                
            if np.abs(ymeas_step[0]) > EMERGENCY_POS or np.abs(ymeas_step[1]) > EMERGENCY_ANGLE:
                EMERGENCY_STOP = True
            

        # Ts_fast outputs
        t_vec_fast[idx_fast,:] = t_step
        x_vec_fast[idx_fast, :] = x_step #system_dyn.y
        x_ref_vec_fast[idx_fast, :] = xref_MPC
        u_fast = u_MPC + d_fast[idx_fast]
        u_vec_fast[idx_fast,:] = u_fast
        Fd_vec_fast[idx_fast,:] = d_fast[idx_fast]
        emergency_vec_fast[idx_fast,:] = EMERGENCY_STOP

        ## Update to step i+1

        # Controller simulation step at rate Ts_MPC
        if run_MPC:
            time_calc_start = time.perf_counter()
            # Kalman filter: update and predict
            #KF.update(ymeas_step) # \hat x[i|i]
            #KF.predict(u_MPC)    # \hat x[i+1|i]
            KF.predict_update(u_MPC, ymeas_step)
            # MPC update
            if not EMERGENCY_STOP:
                Xref_MPC = Xref_MPC_all[idx_MPC:idx_MPC + Np + 1]
                K.update(KF.x, u_MPC, xref=Xref_MPC) # update with measurement and reference
                #K.update(KF.x, u_MPC, xref=xref_MPC)  # update with measurement and reference
            t_calc_vec[idx_MPC,:] = time.perf_counter() - time_calc_start
            if t_calc_vec[idx_MPC,:] > 2 * Ts_MPC:
                EMERGENCY_STOP = True

        # System simulation step at rate Ts_fast
        time_integrate_start = time.perf_counter()
        system_dyn.set_f_params(u_fast)
        system_dyn.integrate(t_step + Ts_fast)
        x_step = system_dyn.y
        #x_step = x_step + f_ODE_jit(t_step, x_step, u_fast)*Ts_fast
        #x_step = x_step + f_ODE(0.0, x_step, u_fast) * Ts_fast
        t_int_vec_fast[idx_fast,:] = time.perf_counter() - time_integrate_start

        # Time update
        t_step += Ts_fast

    simout = {'t': t_vec, 'x': x_vec, 'u': u_vec, 'y': y_vec, 'y_meas': y_meas_vec, 'x_ref': x_ref_vec, 'x_MPC_pred': x_MPC_pred, 'status': status_vec, 'Fd_fast': Fd_vec_fast,
              't_fast': t_vec_fast, 'x_fast': x_vec_fast, 'x_ref_fast': x_ref_vec_fast, 'u_fast': u_vec_fast, 'emergency_fast': emergency_vec_fast,
              'KF': KF, 'K': K, 'nsim': nsim, 'Ts_MPC': Ts_MPC, 't_calc': t_calc_vec,
              't_int_fast': t_int_vec_fast
              }

    return simout


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import matplotlib


    plt.close('all')
    
    simopt = DEFAULTS_PENDULUM_MPC

    time_sim_start = time.perf_counter()
    simout = simulate_pendulum_MPC(simopt)
    time_sim = time.perf_counter() - time_sim_start

    t = simout['t']
    x = simout['x']
    u = simout['u']
    y = simout['y']
    y_meas = simout['y_meas']
    x_ref = simout['x_ref']
    x_MPC_pred = simout['x_MPC_pred']
    x_fast = simout['x_fast']
    u_fast = simout['u_fast']

    t_fast = simout['t_fast']
    x_ref_fast = simout['x_ref_fast']
    Fd_fast = simout['Fd_fast']
    KF = simout['KF']
    status = simout['status']

    uref = get_parameter(simopt, 'uref')
    Np = get_parameter(simopt, 'Np')
    nsim = len(t)
    nx = x.shape[1]
    ny = y.shape[1]

    y_ref = x_ref[:, [0, 2]]
    y_OL_pred = np.zeros((nsim-Np-1, Np+1, ny)) # on-line predictions from the Kalman Filter
    y_MPC_pred = x_MPC_pred[:, :, [0, 2]] # how to vectorize C * x_MPC_pred??
    y_MPC_err = np.zeros(np.shape(y_OL_pred))
    y_OL_err = np.zeros(np.shape(y_OL_pred))
    for i in range(nsim-Np-1):
        u_init = u[i:i+Np+1, :]
        x_init = x[i,:]
        y_OL_pred[i,:,:] = KF.sim(u_init,x_init)
        y_OL_err[i, :, :] = y_OL_pred[i, :, :] - y_meas[i:i + Np + 1]
        y_MPC_err[i, :, :] = y_MPC_pred[i, :, :] - y_meas[i:i + Np + 1]

    fig,axes = plt.subplots(3,1, figsize=(10,10))
    axes[0].plot(t, y_meas[:, 0], "b", label='p_meas')
    axes[0].plot(t_fast, x_fast[:, 0], "k", label='p')
    axes[0].plot(t, y_ref[:,0], "k--", label="p_ref")
    idx_pred = 0
    axes[0].plot(t[idx_pred:idx_pred+Np+1], y_OL_pred[0, :, 0], 'r', label='Off-line k-step prediction')
    axes[0].plot(t[idx_pred:idx_pred+Np+1], y_MPC_pred[0, :, 0], 'c', label='MPC k-step prediction' )
    axes[0].plot(t[idx_pred:idx_pred+Np+1], y_OL_err[0, :, 0], 'r--', label='Off-line prediction error')
    axes[0].plot(t[idx_pred:idx_pred+Np+1], y_MPC_err[0, :, 0], 'c--', label='MPC prediction error')
    axes[0].set_ylim(-0.2,1.0)
    axes[0].set_title("Position (m)")


    axes[1].plot(t, y_meas[:, 1]*RAD_TO_DEG, "b", label='phi_meas')
    axes[1].plot(t_fast, x_fast[:, 2]*RAD_TO_DEG, 'k', label="phi")
    idx_pred = 0
    axes[1].plot(t[idx_pred:idx_pred+Np+1], y_OL_pred[0, :, 1]*RAD_TO_DEG, 'r', label='Off-line k-step prediction')
    axes[1].plot(t[idx_pred:idx_pred+Np+1], y_MPC_pred[0, :, 1]*RAD_TO_DEG, 'c',label='MPC k-step prediction' )
    axes[1].plot(t[idx_pred:idx_pred+Np+1], y_OL_err[0, :, 1]*RAD_TO_DEG, 'r--', label='Off-line prediction error' )
    axes[1].plot(t[idx_pred:idx_pred+Np+1], y_MPC_err[0, :, 1]*RAD_TO_DEG, 'c--', label='MPC prediction error')
    axes[1].set_ylim(-20,20)
    axes[1].set_title("Angle (deg)")

    axes[2].plot(t, u[:,0], label="u")
    axes[2].plot(t, uref*np.ones(np.shape(t)), "r--", label="u_ref")
    axes[2].set_ylim(-8,8)
    axes[2].set_title("Force (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    # In[MPC computational time]
    t_calc = simout['t_calc']
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist(t_calc * 1000, bins=100)
    ax.set_title('MPC computation time)')
    ax.grid(True)
    ax.set_xlabel('Time (ms)')

    # In[ODE integration time]
    t_int = simout['t_int_fast']
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist(t_int * 1e3, bins=100)
    ax.set_title('ODE integration time)')
    ax.grid(True)
    ax.set_xlabel('Time (ms)')

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
    axes[1].set_ylim(0, 4.0)
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

    time_integrate = np.sum(simout['t_int_fast'])
    time_MPC = np.sum(simout['t_calc'])
    print(f'Total time: {time_sim:.2f}, Integration time: {time_integrate:.2f}, MPC time:{time_MPC:.2f}')

    Ts_MPC = simout['Ts_MPC']
