#!/usr/bin/env mscript

Needs["DifferenceMapOptimizer`"];
Needs["DMTestFunctions`"];
Needs["DMUtils`"];

(* Do everything with the same number of dimensions. *)
dim = Environment["DM_DIMENSIONS"];
If[dim === $Failed,
    dim = 4,
    dim = ParseNumber[dim]
];

(* Number of iterations for each optimization. *)
niter = Environment["DM_NITER"];
If[niter === $Failed,
    niter = 100,
    niter = ParseNumber[niter]
];

(* Tolerance for distinguishing reals. *)
tolerance = Environment["DM_TOLERANCE"];
If[tolerance === $Failed,
    tolerance = 10^-8,
    tolerance = ParseNumber[tolerance]
];

(* Number of times to run optimize each function. *)
runCount = Environment["DM_RUNCOUNT"];
If[runCount === $Failed,
    runCount = 50,
    runCount = ParseNumber[runCount]
];

(* Maximum number of iterations used internally by DM's local solver. *)
innerNiter = Environment["DM_INNERNITER"];
If[innerNiter === $Failed,
    innerNiter = 100,
    innerNiter = ParseNumber[innerNiter]
];

(* The range for sampling the first value values x_i *)
range = Environment["DM_RANGE"];
If[range === $Failed,
    range = {-1000, 1000},
    range = ParseNumber /@ StringSplit[range, ","]
];

(* The distance that the second starting point is going to be from the first
startpoint. *)
secondPointDistance = Environment["DM_STARTDISTANCE"];
If[secondPointDistance === $Failed,
    secondPointDistance = 5.0,
    secondPointDistance = ParseNumber[secondPointDistance]
];

(* The solvers to collect data for. *)
solversToDo = Environment["DM_SOLVERS"];
If[solversToDo === $Failed,
    solversToDo = {"dm", "sa", "de", "nm", "rs"},
    solversToDo = StringSplit[solversToDo, ","]
];

(* The names of the variables we're operating on (not really important),
provided we're consistent.*)
vars = Table[xx[i], {i, 1, dim}];

makeBuiltinSolver[solverType_] :=
    (* shiftAmount takes just the first part of the second slot because the
    passed in value is a pair of points, since DM requires a pair, whereas SA
    requires just a single one. *)
    Module[{r, t, nfev, fun = #1, shiftAmount = #2[[1]]},
        nfev = 0;
        {{fun, argmin}, {steps}} = Reap[Quiet[NMinimize[
            ShiftOverBy[shiftAmount, fun][vars], vars,
            EvaluationMonitor -> Hold[nfev = nfev + 1 ; Sow[vars]],
            MaxIterations -> niter,
            Method -> solverType]]];
        {
            "fun" -> fun,
            "x" -> vars /. argmin,
            "nfev" -> nfev,
            "iterate" -> steps
        }
    ] &;

solvers = {
    {"dm", Quiet[DifferenceMapOptimizer[
                #1 @ vars, vars, niter, tolerance, startpoint -> #2,
                LocalMaxIterations -> innerNiter]] &
    },
    {"auto", makeBuiltinSolver[Automatic]},
    {"sa",
        makeBuiltinSolver["SimulatedAnnealing"]
    },
    {"de", makeBuiltinSolver["DifferentialEvolution"]},
    {"nm", makeBuiltinSolver["NelderMead"]},
    {"rs", makeBuiltinSolver["RandomSearch"]}
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
            With[{firstPoint = Table[RandomReal["range" /. f], {dim}]},
                {firstPoint, RandomAwayFrom[firstPoint, secondPointDistance]}],
            {runCount}],
        (* Define the function that does just one run. *)
        run = mysolver["function" /. f, #] &},
        (* Compute the run result for each pair of starting points. *)
        Map[run, starts]
    ];

resultsPerSolver[solver_] := Map[
    Module[{f = asRules[#], r},
        Write[Streams["stderr"], "Beginning test: ", "name" /. f];
        r = test[solver, f];
        ("name" /. f) -> r
    ] &,
    testFunctions
];

results := MapThread[
    Module[{solverName = #1, solver = #2},
        Write[Streams["stderr"], "### Solver: ", solverName];
        solverName -> resultsPerSolver[solver]
    ] &,
    Transpose[Select[solvers, MemberQ[solversToDo, #[[1]]] &]]
];

(*
 vim: set filetype=mma: *)
