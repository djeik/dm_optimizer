#!/usr/bin/env mscript

Needs["DifferenceMapOptimizer`"];
Needs["DMTestFunctions`"];

settings = Import["!cat", "JSON"];

startpoints = "startpoints" /. settings;
niter = "niter" /. settings;
tol = "tolerance" /. settings;

dim = Length[startpoints[[1]]];
vars = Table[xx[i], {i, 1, dim}];

result = DifferenceMapOptimizer[griewankN @ vars, vars, niter, tol,  startpoint -> startpoints, verbosity -> 0]

Export["!cat", result, "JSON"];
