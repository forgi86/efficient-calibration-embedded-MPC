import time
import numpy as np
import numba as nb

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])], '(n),(n)->()')
def fast_dist(X1, X2, res):
    res[0] = 0.0
    n = X1.shape[0]
    for i in range(n):
        res[0] += (X1[i] - X2[i])**2

def solve(prob):
    """
    Solve global optimization problem using inverse distance
    weighting and radial basis functions.

    (C) 2019 A. Bemporad

    sol = idwgopt.solve(prob) solves the global optimization problem

    min  f(x)
    s.t. lb <= x <=ub, A*x <=b, g(x)<=0

    using the global optimization algorithm described in [1]. The approach is
    particularly useful when f(x) is time-consuming to evaluate, as it
    attempts at minimizing the number of function evaluations.

    The default problem structure is

    prob = idwgopt.default(nvars)

    where nvars = dimension of optimization vector x. See function idwgopt_default
    for a description of all available options.

    The output argument 'out' is a structure reporting the following information:

    out.X:    trace of all samples x at which f(x) has been evaluated
    out.F:    trace of all function evaluations f(x)
    out.W:    final set of weights (only meaningful for RBFs)
    out.xopt: best sample found during search
    out.fopt: best value found during search, fopt=f(xopt)


    Required Python packages:
        pyDOE:   https://pythonhosted.org/pyDOE/
        nlopt:   https://nlopt.readthedocs.io (required only if DIRECT solver is used)
        pyswarm: https://pythonhosted.org/pyswarm/ (required only if PSO solver is used)

    [1] A. Bemporad, "Global optimization via inverse weighting and radial
    basis functions", arXiv:1906.06498v1, June 15, 2019.
    https://arxiv.org/pdf/1906.06498.pdf

    """

    import idwgopt.idwgopt_init as idwgopt_init
    from pyswarm import pso  # https://pythonhosted.org/pyswarm/

    from numpy.linalg import svd

    from numpy import zeros, ones, diag, inf
    from numpy import where, maximum, exp
    from math import sqrt, atan, pi
    import contextlib
    import io

    def get_rbf_weights(M, F, NX, svdtol):
        # Solve M*W = F using SVD

        U, dS, V = svd(M[0:NX, 0:NX])
        ii = where(dS >= svdtol)
        ns = max(ii[0]) + 1
        W = (V[0:ns, ].T).dot(diag(1 / dS[0:ns].flatten('C')).dot((U[:, 0:ns].T).dot(F[0:NX])))

        return W

    def facquisition(xx, X, F, N, alpha, delta, dF, W, rbf, useRBF):
        # Acquisition function to minimize to get next sample

        #d = sum(((X[0:N, ] - (ones((N, 1)) * xx)) ** 2).T)

        #d = np.sum((X[0:N, ] - xx) ** 2, axis=-1)
        d = fast_dist(X[0:N,:], xx)

        ii = where(d < 1e-12)
        if ii[0].size > 0:
            fhat = F[ii[0]][0]
            dhat = 0
        else:
            w = exp(-d) / d
            sw = np.sum(w)

            if useRBF:
                v = rbf(X[0:N,:],xx)
                fhat = v.ravel().dot(W.ravel())
#                v = zeros((N, 1))
#                for j in range(N):
#                    v[j] = rbf(X[j,].T, xx)
#                fhat = sum(v * W)
            else:
                fhat = np.sum(F[0:N, ] * w) / sw

#            dhat = (delta * atan(1 / np.sum(1 / d)) * 2 / pi * dF +
#                    alpha * sqrt(np.sum(w * (F[0:N, ] - fhat).flatten("c") ** 2) / sw))
            dhat = delta * atan(1 / np.sum(1 / d)) * 2 / pi * dF
            f_err = (F[0:N, ] - fhat).flatten("c")
            f_err_w = w * f_err**2
            dhat += alpha * sqrt(np.sum(f_err_w) / sw)

        f = fhat - dhat

        return f

    (f, lb, ub, nvar, Aineq, bineq, g, isLinConstrained, isNLConstrained,
     X, F, z, nsamp, maxevals, epsDeltaF, alpha, delta, rhoC, display, svdtol,
     dd, d0, useRBF, rbf, M, scalevars, globoptsol, DIRECTopt,
     PSOiters, PSOswarmsize) = idwgopt_init.init(prob)


    time_iter = []
    time_f_eval = []
    time_opt_acquisition = []
    time_fit_surrogate = []

    for i in range(nsamp):
        time_fun_eval_start = time.perf_counter()
        F[i] = f(X[i,].T)
        time_fun_eval_i = time.perf_counter() - time_fun_eval_start
        time_iter.append(time_fun_eval_i)
        time_f_eval.append(time_fun_eval_i)
        time_opt_acquisition.append(0.0)
        time_fit_surrogate.append(0.0)

    if useRBF:
        W = get_rbf_weights(M, F, nsamp, svdtol)
    else:
        W = []

    fbest = inf
    zbest = zeros((nsamp, 1))
    for i in range(nsamp):
        isfeas = True
        if isLinConstrained:
            isfeas = isfeas and all(Aineq.dot(X[i,].T) <= bineq.flatten("c"))
        if isNLConstrained:
            isfeas = isfeas and all(g(X[i,]) <= 0)
        if isfeas and fbest > F[i]:
            fbest = F[i]
            zbest = X[i,]

    Fmax = max(F[0:nsamp])
    Fmin = min(F[0:nsamp])

    N = nsamp

    while N < maxevals:

        time_iter_start = time.perf_counter()

        dF = Fmax - Fmin

        if isLinConstrained or isNLConstrained:
            penalty = rhoC * dF
        if isLinConstrained and isNLConstrained:
            constrpenalty = lambda x: (penalty * (sum(maximum((Aineq.dot(x) - bineq).flatten("c"), 0) ** 2)
                                                  + sum(maximum(g(x), 0) ** 2)))
        elif isLinConstrained and not isNLConstrained:
            constrpenalty = lambda x: penalty * (sum(maximum((Aineq.dot(x) - bineq).flatten("c"), 0) ** 2))
        elif not isLinConstrained and isNLConstrained:
            constrpenalty = lambda x: penalty * sum(maximum(g(x), 0) ** 2)
        else:
            constrpenalty = lambda x: 0

        acquisition = lambda x: (facquisition(x, X, F, N, alpha, delta, dF, W, rbf, useRBF)
                                 + constrpenalty(x))

        time_opt_acq_start = time.perf_counter()
        if globoptsol == "pswarm":
            # pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
            #    swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-8,
            #    minfunc=1e-8, debug=False)
            if display == 0:
                with contextlib.redirect_stdout(io.StringIO()):
                    z, cost = pso(acquisition, lb, ub, swarmsize=PSOswarmsize,
                                  minfunc=dF * 1e-8, maxiter=PSOiters)
            else:
                z, cost = pso(acquisition, lb, ub, swarmsize=PSOswarmsize,
                              minfunc=dF * 1e-8, maxiter=PSOiters)

        elif globoptsol == "direct":
            DIRECTopt.set_min_objective(lambda x, grad: acquisition(x)[0])
            z = DIRECTopt.optimize(z.flatten("c"))
        time_opt_acquisition.append(time.perf_counter() - time_opt_acq_start)

        time_fun_eval_start = time.perf_counter()
        fz = f(z)  # function evaluation
        time_f_eval.append(time.perf_counter() - time_fun_eval_start)

        N = N + 1

        X[N - 1,] = z.T
        F[N - 1] = fz

        Fmax = max(Fmax, fz)
        Fmin = min(Fmin, fz)

        time_fit_surrogate_start = time.perf_counter()
        if useRBF:
            # Just update last row and column of M
            for h in range(N):
                mij = rbf(X[h,], X[N - 1,])
                M[h, N - 1] = mij
                M[N - 1, h] = mij

            W = get_rbf_weights(M, F, N, svdtol)
        time_fit_surrogate.append(time.perf_counter() - time_fit_surrogate_start)

        if fbest > fz:
            fbest = fz.copy()
            zbest = z.copy()

        if display > 0:

            print("N = %4d, cost = %8g, best = %8g" % (N, fz, fbest))

            string = ""
            for j in range(nvar):
                aux = zbest[j]
                if scalevars:
                    aux = aux * dd[j] + d0[j]

                string = string + " x" + str(j + 1) + " = " + str(aux)
            print(string)

        time_iter.append(time.perf_counter() - time_iter_start)

    # end while

    xopt = zbest.copy()
    if scalevars:
        # Scale variables back
        xopt = xopt * dd + d0
        X = X * (ones((N, 1)) * dd) + ones((N, 1)) * d0

    fopt = fbest.copy()

    if not useRBF:
        W = []

    out = {"xopt": xopt,
           "fopt": fopt,
           "X": X,
           "F": F,
           "W": W,
           'time_iter': np.array(time_iter),
           'time_opt_acquisition': np.array(time_opt_acquisition),
           'time_fit_surrogate': np.array(time_fit_surrogate),
           'time_f_eval': np.array(time_f_eval)}

    return out


def default(nvars):
    """ Generate default problem structure for IDW-RBF Global Optimization.

     problem=idwgopt.default(n) generate a default problem structure for a
     an optimization with n variables.

     (C) 2019 by A. Bemporad.
    """

    import idwgopt.idwgopt_default as idwgopt_default
    problem = idwgopt_default.set(nvars)

    return problem
