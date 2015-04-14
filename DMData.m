#!/usr/bin/env mscript

Get["DifferenceMapOptimizer`"];
Get["DMTestFunctions`"];
Get["DMUtils`"];

makeResults[settings_] := Module[{solvers, results, makeBuiltinSolver, test,
    resultsPerSolver, randomRestartStrategy},

    makeBuiltinSolver[solverType_] :=
        (* shiftAmount takes just the first part of the second slot because the
        passed in value is a pair of points, since DM requires a pair, whereas SA
        requires just a single one. *)
        Function[{function, startpoints},
            Module[
                {r, t, nfev, shiftAmount = startpoints[[1]],
                range = "range" /. function,
                tracker,
                vars = Table[xx[i], {i, 1, "dim" /. function}]
                },
                (* Create the tracker function that will count the function evaluations in earnest. *)
                tracker = TrackExpr[
                    ShiftOverBy[shiftAmount, "function" /. function],
                    Hold[Sow[1, 1]]];
                {time, {{{fun, argmin}, {steps}}, {nfev}}} = AbsoluteTiming[
                    Reap[
                        Reap[
                            NMinimize[
                                Hold[tracker @ vars],
                                Map[{#, range[[1]], range[[2]]} &, vars],
                                EvaluationMonitor :> Sow[vars, 0],
                                MaxIterations -> settings["niter"],
                                Method -> solverType],
                            0
                        ],
                        1
                    ]
                ];
                <|
                    "fun" -> fun,
                    "x" -> vars /. argmin,
                    "nfev" -> Length[nfev],
                    "iterate" -> steps,
                    "time" -> time
                |>
            ]
        ];

    (* Construct a builtin solver of the give type, and if it performs less than
        the maximum allowed function evaluations as specified in the settings,
        then it is retried with a different RandomSeed.  *)
    builtinRandomRestartStrategy[solverType_] :=
        Function[{function, startpoints},
            Module[{nfev = 0, seed = RandomInteger[1000000], result, results, solver},
                {result, {results}} = Reap[
                    While[nfev < settings[["maxnfev"]],
                        solver = makeBuiltinSolver[{solverType, "RandomSeed" -> seed}];
                        result = solver[function, startpoints];
                        nfev += result[["nfev"]];
                        seed += 1;
                        Sow[result];
                        If[Not[settings[["randomrestart"]]],
                            Break[];
                        ];
                    ]
                ];
                result = MinimalBy[results, #[["fun"]] &];
                result[[1, "nfev"]] = nfev;
                result[[1]]
            ]
        ];

    solvers = {
        {"dm", Function[{function, startpoints},
            Module[{time, r, vars},
                vars = Table[xx[i], {i, 1, "dim" /. function}];
                {time, r} = AbsoluteTiming[Quiet[DifferenceMapOptimizer[
                ("function" /. function) @ vars, vars, settings["niter"],
                settings["tolerance"], startpoint -> startpoints,
                LocalMaxIterations -> settings["innerNiter"]]]];
                r[["time"]] = time;
                r
            ]
        ]},
        {"de", builtinRandomRestartStrategy["DifferentialEvolution"]},
        {"sa", builtinRandomRestartStrategy["SimulatedAnnealing"]},
        {"nm", builtinRandomRestartStrategy["NelderMead"]},
        {"rs", builtinRandomRestartStrategy["RandomSearch"]}
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
                With[{firstPoint = Table[RandomReal["range" /. f], {"dim" /. f}]},
                    {firstPoint, RandomAwayFrom[
                        firstPoint, settings["secondPointDistance"]
                    ]}],
                {settings["runCount"]}],
            (* Define the function that does just one run. *)
            run = mysolver[f, #] &},
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
            Print["### Solver: ", solverName];
            solverName -> resultsPerSolver[solver]
        ] &,
        Transpose[Select[solvers, MemberQ[settings["solversToDo"], #[[1]]] &]]
    ];

    Return[results];
];

(* vim: set filetype=mma: *)
