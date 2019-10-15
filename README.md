# pyMPC_BO

Data-driven MPC tuning of:

 * MPC weight matrices
 * MPC sampling time T\_{MPC}
 * Prediction and control horizon
 * Kalman filter matrices
 * MPC solver relative and absolute tolerances,

under the constraint that T\_{calc,max} < T\_{MPC} where T\_{calc,max} is the maximum MPC time spent to solve the optimization problem in the simulation.
This constraints means that the controller can run in real-time.

## Usage 

The main script to be executed for MPC calibration is

`` BO_IDW_main.py``

The results of the Bayesian Optimization are saved in the results_*.pkl file
 on the disk and are read by the script

``BO_analysis.py``

that generates relevant plots
## Other files:
 * ``pendulum_model.py``: dynamic equations of the pendulum 
 * ``pendulum_MPC_sim``: performs a single closed-loop MPC simulation
 * ``objective_function.py``: objective function
 * ``kalman.py``: implements a kalman filter

## Othe folders:
 * ``pyMPC``: containts the pyMPC MPC library. Copied from the repository <https://github.com/forgi86/pyMPC.git>, branch dev-BO
 * ``idwgopt``: contains the idwgopt package version 1.2, copied from <http://cse.lab.imtlucca.it/~bemporad/idwgopt/> 
## Dependencies:

Simulations performed on a Python 3.6 conda environment with

 * numpy
 * scipy
 * matplotlib
 * OSQP (a QP solver used by the MPC controller)
 * python-control (used to solve the DARE of the Kalman Filter)
 * GPyOpt (for Bayesian Optimization, optional) 

These dependencies may be installed through the commands:
```
conda install numpy scipy matplotlib
pip install osqp
pip install control
pip install gpyopt
```
