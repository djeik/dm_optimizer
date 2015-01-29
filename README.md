Difference Map Optimizer
========================

DMO is a reasearch project aimed at developing a new method for the
unconstrained global minimization of multivariate scalar functions. It is based
on the Divide and Concur method for solving high-dimensional constraint
satisfaction problems, which is itself using the Difference Map method.

In brief, DMO is an iterative method, performing a local minimization of the
objective function at each step. The landscape of discovered minima is used to
determine the next location at which a minimization should take place.

The project is still in the research phase, and not ready for use. My goal is
to release it as a package in both Mathematica and Python.
