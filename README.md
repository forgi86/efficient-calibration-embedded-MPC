# Efficient Calibration for Embedded MPC

This code performs an efficient data-driven MPC calibration by tuning:

 * MPC weight matrices
 * MPC sampling time <a href="https://www.codecogs.com/eqnedit.php?latex=T_{\mathrm{s}}^{\mathrm{MPC}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_{\mathrm{s}}^{\mathrm{MPC}}" title="T_{\mathrm{s}}^{\mathrm{MPC}}" /></a>
 * Prediction and control horizon
 * Kalman filter matrices
 * QP solver relative and absolute tolerances

to optimize a closed-loop objective function <a href="https://www.codecogs.com/eqnedit.php?latex=J^{cl}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J^{cl}" title="J^{cl}" /></a>, under the constraint that <a href="https://www.codecogs.com/eqnedit.php?latex=T_{\mathrm{s}}^{\mathrm{MPC}}&space;\leq&space;\eta&space;T_{\mathrm{s}}^{\mathrm{MPC}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_{\mathrm{calc}}^{\mathrm{MPC}}&space;\leq&space;\eta&space;T_{\mathrm{s}}^{\mathrm{MPC}}" title="T_{\mathrm{s}}^{\mathrm{MPC}} \leq \eta T_{\mathrm{s}}^{\mathrm{MPC}}" /></a> where <a href="https://www.codecogs.com/eqnedit.php?latex=T_{\mathrm{calc}}^{\mathrm{MPC}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_{\mathrm{calc}}^{\mathrm{MPC}}" title="T_{\mathrm{calc}}^{\mathrm{MPC}}" /></a> is the (worst-case) time required to compute the MPC
control low. This constraints guarantees that the controller can run in real-time.

## Main scripts: 

The main script to be executed for MPC calibration is

`` GLIS_BO_main.py``

The results of the MPC calibration are saved in the results_*.pkl file
 on the disk and are read by the script

``GLIS_BO_analysis.py``

that produces the relevant plots.
## Other files:
 * ``pendulum_model.py``: dynamic equations of the pendulum 
 * ``pendulum_MPC_sim``: performs a single closed-loop MPC simulation
 * ``objective_function.py``: objective function
 * ``kalman.py``: implements a kalman filter

## Included dependencies:
 * ``pyMPC``: containts the pyMPC library for Model Predictive Control. Copied from branch dev-BO of my repository <https://github.com/forgi86/pyMPC.git>, 
 * ``idwgopt``: contains the python version of the GLIS package version 1.1. Copied from <http://cse.lab.imtlucca.it/~bemporad/glis/> 
## Other dependencies:

Simulations were performed on a Python 3.6 conda environment with

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
