Global Optimization via Inverse Distance Weighting

(C) 2019 by A. Bemporad, June 15, 2019

This package contains a MATLAB and a Python implementation of the global
optimization method described in the paper:

A. Bemporad, "Global optimization via inverse weighting and radial 
basis functions", arXiv:1906.06498v1, June 15, 2019.
https://arxiv.org/pdf/1906.06498.pdf

The MATLAB version requires the Particle Swarm Optimization solver
PSwarm (http://www.norg.uminho.pt/aivaz/pswarm) and/or the DIRECT solver
available from the NLopt library (https://github.com/stevengj/nlopt).
Comparisons with Bayesian Optimization require function bayesopt
from the Statistics and Machine Learning Toolbox for MATLAB.

The Python version requires the Particle Swarm Optimization solver
pyswarm (https://pythonhosted.org/pyswarm) and/or the DIRECT solver
available from the NLopt library (https://github.com/stevengj/nlopt).
Comparisons with Bayesian Optimization require the GPyOpt package 
(https://github.com/SheffieldML/GPyOpt).


Version tracking:

v1.1     (August 3, 2019) Python code largely optimized (by Marco Forgione)

v1.0.2   (July 6, 2019) Moved init and default functions to external files 
         idwgopt_init.py and idwgopt_default.py, respectively

v1.0.1   (July 4, 2019) Added option "shrink_range" to disable shrinking
         lb and ub to bounding box of feasible set. Fixed small bug in 
         computing initial best value and initial range of F. 

v1.0     (June 15, 2019) Initial version

