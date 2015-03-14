#!/usr/bin/env mscript

Needs["DifferenceMapOptimizer`"];
Needs["DMTestFunctions`"];
Needs["DMUtils`"];

functions = {
    {"griewank", griewankN},
    {"rosenbrock", rosenbrockN},
    {"schwefel", schwefelN},
    {"schaffer", schafferN}
};

(* Do everything with the same number of dimensions. *)
dim = 4;
(* Number of iterations for each optimization. *)
niter = 100;
(* Tolerance for distinguishing reals. *)
tolerance = 10^-8;
(* Do each function 50 times. *)
runCount = Environment["DM_RUNCOUNT"];
If[runCount == $Failed, runCount = 50];

(* Maximum number of iterations used internally by DM's local solver. *)
innerNiter = Environment["DM_INNERNITER"];
If[innerNiter === $Failed,
    innerNiter = 100,
    innerNiter = ParseNumber[innerNiter]
];

(* The range for sampling the first value values x_i *)
range = {-250, 250};
(* The distance that the second starting point is going to be from the first
startpoint. *)
secondPointDistance = 5;
(* The names of the variables we're operating on (not really important) *)
vars = Table[xx[i], {i, 1, dim}];

solvers = {
    {"dm", Module[{r = DifferenceMapOptimizer[
            #1 @ vars, vars, niter, tolerance, startpoint -> #2,
            LocalMaxIterations -> innerNiter]}, r]&},
    {"sa",
        (* shiftAmount takes just the first part of the second slot because the
        passed in value is a pair of points, since DM requires a pair, whereas
        SA requires just a single one. *)
        Module[{r, t, nfev, fun = #1, shiftAmount = #2[[1]]},
            nfev = 0;
            r = NMinimize[ShiftOverBy[shiftAmount, fun][vars], vars,
                EvaluationMonitor -> Hold[nfev = nfev + 1],
                MaxIterations -> niter];
            { "fun" -> r[[1]], "x" -> vars /. r[[2]], "nfev" -> nfev
            }] &}
};

test[mysolver_, f_] :=
    With[{
        (*
            Define the startpoints. The first startpoint is sampled randomly
            from the range global on each of its components, of which there are
            dim. The second startpoint is a point a random direction away from
            the first but with some fixed distance (see
            DMUtils`RandomAwayFrom).
        *)
        starts = Table[
            With[{firstPoint = Table[RandomReal[range], {dim}]},
                {firstPoint, RandomAwayFrom[firstPoint, secondPointDistance]}],
            {runCount}],
        (* Define the function that does just one run. *)
        run = mysolver[f, #] &},
        (* Compute the run result for each pair of starting points. *)
        Map[run, starts]
    ];

resultsPerSolver[solver_] := MapThread[
    Module[{fname = #1, fun = #2, r},
        Write[Streams["stderr"], "Beginning test: ", fname];
        r = test[solver, fun];
        fname -> r
    ] &,
    Transpose[functions]
];

results = MapThread[
    Module[{solverName = #1, solver = #2},
        Write[Streams["stderr"], "### Solver: ", solverName];
        solverName -> resultsPerSolver[solver]
    ] &,
    Transpose[solvers]
];

Export["!cat", results, "JSON"];

(*
 vim: set filetype=mma: *)
