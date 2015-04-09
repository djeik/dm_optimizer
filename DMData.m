#!/usr/bin/env mscript

Needs["DifferenceMapOptimizer`"];
Needs["DMTestFunctions`"];
Needs["DMUtils`"];

makeResults[settings_] := Module[{solvers, results, makeBuiltinSolver, test,
    resultsPerSolver},

    (* The names of the variables we're operating on (not really important),
    provided we're consistent.*)
    vars = Table[xx[i], {i, 1, settings["dim"]}];

    makeBuiltinSolver[solverType_] :=
        (* shiftAmount takes just the first part of the second slot because the
        passed in value is a pair of points, since DM requires a pair, whereas SA
        requires just a single one. *)
        Module[{r, t, nfev, function = #1, shiftAmount = #2[[1]]},
            {{fun, argmin}, {steps}} = Reap[Quiet[NMinimize[
                ShiftOverBy[shiftAmount, function][vars], vars,
                EvaluationMonitor -> Hold[Sow[vars]],
                MaxIterations -> settings["niter"],
                Method -> solverType]]];
            <|
                "fun" -> fun,
                "x" -> vars /. argmin,
                "nfev" -> Length[steps],
                "iterate" -> steps
            |>
        ] &;

    solvers = {
        {"dm", Quiet[DifferenceMapOptimizer[
                    #1 @ vars, vars, settings["niter"], settings["tolerance"], startpoint -> #2,
                    LocalMaxIterations -> settings["innerNiter"]]] &
        },
        {"auto", makeBuiltinSolver[Automatic]},
        {"sa",
        (* For Simulated Annealing, implement a system of random restarts to
        ensure that the desired number of function evaluations take place.
        *)
        Function[{function, startpoints},
            Module[{innerSolver, newStart, nfev = 0, s, ss = {}},
                innerSolver = makeBuiltinSolver["SimulatedAnnealing"];
                newStart = startpoints[[1]];
                While[nfev < settings["maxnfev"],
                    s = innerSolver[function, {newStart, Null}];
                    nfev += s["nfev"];
                    AppendTo[ss, s];
                    newStart = RandomReal[{-10, 10}, Length[startpoints[[1]]]];
                    (* Rescale newStart to have the same length as the original
                    startpoint. *)
                    newStart = (Norm[startpoints[[1]]] / Norm[newStart]) newStart;
                    Print[nfev];
                ];
                s = MinimalBy[ss, #["fun"]][[1]];
                s["nfev"] = nfev;
                Return[s];
            ]
        ]},
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
                With[{firstPoint = Table[RandomReal["range" /. f], {settings["dim"]}]},
                    {firstPoint, RandomAwayFrom[
                        firstPoint, settings["secondPointDistance"]
                    ]}],
                {settings["runCount"]}],
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
        Transpose[Select[solvers, MemberQ[settings["solversToDo"], #[[1]]] &]]
    ];

    Return[results];
];

(*
 vim: set filetype=mma: *)
