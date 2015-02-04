#!/usr/bin/env mscript

Needs["DifferenceMapOptimizer`"];
Needs["DMTestFunctions`"];

functions = {
    {"griewank", griewankN},
    {"rosenbrock", rosenbrockN},
    {"schwefel", schwefelN},
    {"schaffer", schafferN}
};

dim = 4; (* Do everything with the same number of dimensions. *)
niter = 100; (* Number of iterations for each optimization. *)
tolerance = 10^-8; (* Tolerance for distinguishing reals. *)
runCount = 50; (* Do each function 50 times. *)
range = {-250, 250}; (* The range for sampling values x_i for the startpoints *)
vars = Table[xx[i], {i, 1, dim}];

test[{fname_, f_}] :=
    Module[{starts},
        Write[Streams["stderr"], "Beginning test: ", fname];
        (* Each run requires 2 startpoints, each startpoint has 4 components, there are runCount runs. *)
        starts = Table[Table[Table[RandomReal[range], {dim}], {2}], {runCount}];

        run[starters_] := DifferenceMapOptimizer[f @ vars, vars, niter, tolerance, startpoint -> starters];
        testResults = Map[run, starts];
        Return[fname -> testResults];
    ];

results = Map[test, functions];
Export["!cat", results, "JSON"];
