def set(nvars):
    """
    Generate default problem structure for IDW-RBF Global Optimization.
    
    problem = idwgopt_default.set(n) generate a default problem structure for a
    an optimization with n variables.
     
    (C) 2019 A. Bemporad, July 6, 2019
    """

    from numpy import zeros, ones
    from numpy import sum as vecsum
    
    problem = {
        "f": "[set cost function here]", # cost function to minimize
        "lb": -1 * ones((nvars, 1)),  # lower bounds on the optimization variables
        "ub": 1 * ones((nvars, 1)),  # upper bounds on the optimization variables
        "maxevals": 20,  # maximum number of function evaluations
        "alpha": 1, # weight on function uncertainty variance measured by IDW
        "delta": 0.5, # weight on distance from previous samples
        "nsamp": 2*nvars,       # number of initial samples
        "useRBF": 1, # 1 = use RBFs, 0 = use IDW interpolation
        "rbf": lambda x1,x2: 1/(1+0.25*vecsum((x1-x2)**2)), # inverse quadratic 
        #                       RBF function (only used if useRBF=1)
        "scalevars": 1,  # scale problem variables
        "svdtol": 1e-6,   # tolerance used to discard small singular values
        "Aineq": zeros((0,nvars)),  # matrix A defining linear inequality constraints 
        "bineq": zeros((0,1)),     # right hand side of constraints A*x <= b
        "g": 0,          # constraint function. Example: problem.g = lambda x: x[0]**2+x[1]**2-1
        "shrink_range": 1, # flag. If 0, disable shrinking lb and ub to bounding box of feasible set
        "constraint_penalty": 1000, # penalty term on violation of linear inequality
        #                             and nonlinear constraint
        "feasible_sampling": False, # if True, initial samples are forced to be feasible
        "globoptsol": "direct", # nonlinear solver used during acquisition.
        # interfaced solvers are:
        #   "direct" DIRECT from NLopt tool (nlopt.readthedocs.io)
        #   "pswarm" PySwarm solver (pythonhosted.org/pyswarm/)
        "display": 0, # verbosity level (0=minimum)
        "PSOiters": 500, # number of iterations in PSO solver
        "PSOswarmsize": 20, # swarm size in PSO solver
        "epsDeltaF": 1e-4 # minimum value used to scale the IDW distance function
    }

    return problem
