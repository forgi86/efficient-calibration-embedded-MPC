import numpy as np

# Constants #
M = 0.5
m = 0.2
b = 0.1
ftheta = 0.1
l = 0.3
g = 9.81
RAD_TO_DEG = 360.0/2.0/np.pi
DEG_TO_RAD = 1.0/RAD_TO_DEG

# Nonlinear dynamics ODE:
# \dot x = f_ODE(x,u)
from numba import jit

@jit("float64[:](float64,float64[:],float64[:])", nopython=True, cache=True)
def f_ODE_jit(t, x, u):
    F = u[0]
    v = x[1]
    theta = x[2]
    omega = x[3]
    der = np.zeros(x.shape)
    der[0] = v
    der[1] = (m * l * np.sin(theta) * omega ** 2 - m * g * np.sin(theta) * np.cos(theta) + m * ftheta * np.cos(
        theta) * omega + F - b * v) / (M + m * (1 - np.cos(theta) ** 2))
    der[2] = omega
    der[3] = ((M + m) * (g * np.sin(theta) - ftheta * omega) - m * l * omega ** 2 * np.sin(theta) * np.cos(
        theta) - (
                      F - b * v) * np.cos(theta)) / (l * (M + m * (1 - np.cos(theta) ** 2)))
    return der


def f_ODE_wrapped(t,x,u):
    return f_ODE_jit(t,x,u)

def f_ODE(t, x, u):
    F = u
    v = x[1]
    theta = x[2]
    omega = x[3]
    der = np.zeros(x.shape)
    der[0] = v
    der[1] = (m * l * np.sin(theta) * omega ** 2 - m * g * np.sin(theta) * np.cos(theta) + m * ftheta * np.cos(
        theta) * omega + F - b * v) / (M + m * (1 - np.cos(theta) ** 2))
    der[2] = omega
    der[3] = ((M + m) * (g * np.sin(theta) - ftheta * omega) - m * l * omega ** 2 * np.sin(theta) * np.cos(
        theta) - (
                      F - b * v) * np.cos(theta)) / (l * (M + m * (1 - np.cos(theta) ** 2)))
    return der


if __name__ == '__main__':
    f_ODE_jit(0.1, np.array([0.3, 0.2, 0.1, 0.4]), 3.1)
