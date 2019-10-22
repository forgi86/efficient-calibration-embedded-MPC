def init(prob):
    """
    Init function for idwgopt.py
    
    (C) 2019 A. Bemporad, July 6, 2019
    """
    
    from pyDOE import lhs #https://pythonhosted.org/pyDOE/
    # import nlopt # https://nlopt.readthedocs.io
    from pyswarm import pso # https://pythonhosted.org/pyswarm/

    from scipy.optimize import linprog as linprog

    from numpy import size, zeros, ones, diag
    from numpy import where, maximum
    from math import ceil
    import sys
    import contextlib
    import io

    
    # input arguments
    f0 = prob["f"]
    lb = prob["lb"].copy()
    ub = prob["ub"].copy()
    maxevals = prob["maxevals"]
    alpha = prob["alpha"] 
    delta = prob["delta"]
    nsamp = prob["nsamp"]      
    useRBF = prob["useRBF"] 
    rbf = prob["rbf"]       
    scalevars = prob["scalevars"] 
    svdtol = prob["svdtol"]   
    Aineq = prob["Aineq"].copy()   
    bineq = prob["bineq"].copy()
    g0 = prob["g"]          
    shrink_range = prob["shrink_range"]
    rhoC = prob["constraint_penalty"]
    feasible_sampling = prob["feasible_sampling"]
    globoptsol = prob["globoptsol"]
    display = prob["display"] 
    PSOiters = prob["PSOiters"]
    PSOswarmsize = prob["PSOswarmsize"]
    epsDeltaF = prob["epsDeltaF"]

    nvar = size(lb) # number of optimization variables

    isLinConstrained = (size(bineq) > 0)
    isNLConstrained = (g0 != 0)
    if not isLinConstrained and not isNLConstrained:
        feasible_sampling = False

    f = f0
    g = g0
    if scalevars:
        # Rescale problem variables in [-1,1]
        dd = (ub-lb)/2
        d0 = (ub+lb)/2
        f = lambda x: f0(x*dd+d0)
    
        lb = -ones(nvar)
        ub = ones(nvar)
    
        if isLinConstrained:
            bineq = bineq-Aineq.dot(d0)
            Aineq = Aineq.dot(diag(dd.flatten('C')))
    
        if isNLConstrained:
            g = lambda x: g0(x*dd+d0)
        
    # set solver options
    if globoptsol=="pswarm":
        # nothing to do
        pass
        DIRECTopt = []

    elif globoptsol=="direct":
        #DIRECTopt = nlopt.opt(nlopt.GN_DIRECT_L, 2)
        DIRECTopt = nlopt.opt(nlopt.GN_DIRECT, 2)
        DIRECTopt.set_lower_bounds(lb.flatten("C"))
        DIRECTopt.set_upper_bounds(ub.flatten("C"))
        DIRECTopt.set_ftol_abs(1e-5)
        DIRECTopt.set_maxeval(2000)
        DIRECTopt.set_xtol_rel(1e-5)
        
    else: 
        print("Unknown solver")
        sys.exit(1)


    if shrink_range == 1:
        # possibly shrink lb,ub to constraints
        if not isNLConstrained and isLinConstrained:
            flin=zeros((nvar,1))
        
            for i in range(nvar):
                flin[i]=1
                res=linprog(flin, Aineq, bineq, bounds=(None,None))
                aux=max(lb[i],res.fun)
                lb[i]=aux
                flin[i]=-1
                res=linprog(flin, Aineq, bineq, bounds=(None,None))
                aux=min(ub[i],-res.fun)
                ub[i]=aux
                flin[i]=0
        
        elif isNLConstrained:
            NLpenaltyfun = lambda x: sum(maximum(g(x),0)**2)
            if isLinConstrained:
                LINpenaltyfun = lambda x: sum(maximum((Aineq.dot(x)-bineq).flatten("c"),0)**2)
            else:
                LINpenaltyfun = lambda x: 0
            
            for i in range(0,nvar):
                obj_fun = lambda x: x[i] + 1.0e4*(NLpenaltyfun(x) + LINpenaltyfun(x))
                if globoptsol=="pswarm":
                    if display == 0:
                        with contextlib.redirect_stdout(io.StringIO()):
                            z, cost = pso(obj_fun, lb, ub, swarmsize=30,
                                        minfunc=1e-8, maxiter=2000)
                    else:
                        z, cost = pso(obj_fun, lb, ub, swarmsize=30,
                                    minfunc=1e-8, maxiter=2000)
                else: # globoptsol=="direct":
                    DIRECTopt.set_min_objective(lambda x,grad: obj_fun(x)[0])
                    z = DIRECTopt.optimize(z.flatten("c"))
                lb[i] = max(lb[i],z[i])
                
                obj_fun = lambda x: -x[i] + 1.0e4*(NLpenaltyfun(x) + LINpenaltyfun(x))
                
                if globoptsol=="pswarm":
                    if display == 0:
                        with contextlib.redirect_stdout(io.StringIO()):
                            z, cost = pso(obj_fun, lb, ub, swarmsize=30, 
                                        minfunc=1e-8, maxiter=2000)
                    else:
                        z, cost = pso(obj_fun, lb, ub, swarmsize=30, 
                                    minfunc=1e-8, maxiter=2000)
                else: # globoptsol=="direct":
                    DIRECTopt.set_min_objective(lambda x,grad: obj_fun(x)[0])
                    z = DIRECTopt.optimize(z.flatten("c"))
                ub[i] = min(ub[i],z[i])
    
    if maxevals<nsamp:
        errstr = "Max number of function evaluations is too low. You specified"
        errstr = errstr + " maxevals = " + str(maxevals) + " and nsamp = " + str(nsamp)
        print(errstr)
        sys.exit(1)

    X = zeros((maxevals,nvar))
    F = zeros((maxevals,1))
    z = (lb+ub)/2
    
    if not feasible_sampling:
        X[0:nsamp,] = lhs(nvar,nsamp,"m")
        X[0:nsamp,] = X[0:nsamp,]*(ones((nsamp,1))*(ub-lb)) + ones((nsamp,1))*lb
    else:
        nn = nsamp
        nk = 0
        while (nk<nsamp):
            XX = lhs(nvar,nn,"m")
            XX = XX*(ones((nn,1))*(ub-lb)) + ones((nn,1))*lb
        
            ii = ones((nn,1)).flatten("C")
            for i in range(nn):
                if isLinConstrained:
                    ii[i]=all(Aineq.dot(XX[i,].T) <= bineq.flatten("c"))
                if isNLConstrained:
                    ii[i]=ii[i] and all(g(XX[i,])<=0)
            
            nk = sum(ii)
            if (nk==0):
                nn = 20*nn
            elif (nk<nsamp):
                nn = ceil(min(20,1.1*nsamp/nk)*nn)

        ii = where(ii)
        X[0:nsamp,]=XX[ii[0][0:nsamp],]
        
    if useRBF:
        M = zeros((maxevals+nsamp,maxevals+nsamp)) # preallocate the entire matrix
        for i in range(nsamp):
            for j in range(i,nsamp):
                mij=rbf(X[i,],X[j,])
                M[i,j]=mij
                M[j,i]=mij
    else:
        M = []

    return (f,lb,ub,nvar,Aineq,bineq,g,isLinConstrained,isNLConstrained,
    X,F,z,nsamp,maxevals,epsDeltaF,alpha,delta,rhoC,display,svdtol,
    dd,d0,useRBF,rbf,M,scalevars,globoptsol,DIRECTopt,PSOiters,PSOswarmsize)
