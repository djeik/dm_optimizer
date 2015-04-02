#!/usr/bin/env mscript

Needs["DMData`"];

Module[{dim, niter, tolerance, runCount, innerNiter, range,
    secondPointDistance, solversToDo},

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

    settings = {
        "dim" -> dim,
        "niter" -> niter,
        "tolerance" -> tolerance,
        "runCount" -> runCount,
        "innerNiter" -> innerNiter,
        "secondPointDistance" -> secondPointDistance,
        "solversToDo" -> solversToDo
    };

    results = makeResults[Association[settings]];

    Export[
        "!cat", {
            "results" -> results,
            "settings" -> {
                "dimensions" -> dim,
                "iterations" -> niter,
                "tolerance" -> tolerance,
                "run_count" -> runCount,
                "inner_iterations" -> innerNiter,
                "range" -> range,
                "second_point_distance" -> secondPointDistance
            }
        },
        "JSON"
    ];
];
