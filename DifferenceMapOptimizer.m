(* ::Package:: *)

BeginPackage["DifferenceMapOptimizer`"];

ClearAll[DifferenceMapOptimizer];
DifferenceMapOptimizer::usage = "Optimize a given expression of some given" <>
        "variables, for a given number of iterations using a given tolerance.";

Begin["Private`"];

pseudo = 0.0001;
firsttargetratio = 0.9;
scal = 0.05; (* The greediness in updating target *)

(* This function follows the mathematica syntax. expr is an expression that
depends on variables vars. The search will run for niter generations, tol is
the minimum distance between neighboring minima to be considered distinct. *)
DifferenceMapOptimizer[expr_, vars_, iterationCount_, tol_, OptionsPattern[]] :=
    Module[{x0, x1, val1, iterate, sol1, localMinimum, delta, target, nnear,
            near, deltan, fnear, step, dim, iterationNumber, maxit,
            bestMinimum, pastMinima, iteratePositions, steps, messages,
            verboseLevel, nfev},
        (* Convenience function for printing things out depending on how
        verbose we want to be, governed by a global verbosity level. *)
        verboseLevel = OptionValue[verbosity];
        (* By default, we don't want to see any messages except errors, whose
        priority levels should be negative. *)

        PrintLog[priority_, messages__] :=
            If[priority <= verboseLevel,
                Print[messages]];

        (* Solutions produced by FindMinimum are of the form {functionValue,
        variableAssociation}, where the variableAssociation is a list like
        {x[[1]] -> 3, x[[2]] -> 5, etc.}
        solToVec converts this association into a vector like {3, 5, etc.}
        based on the variables to optimize on passed in through `vars`. *)
        solToVec[sol_] :=
            {sol[[1]], vars /. sol[[2]]};

        (* vecToRep converts a list of variables and a list of values into a
        list of rules associating those variables to those values. For example,
        vecToRep[{x,y,z},{1,2,3}] -> {x -> 1, y -> 2, z -> 3}. In other words,
        this is a zip with the association operator `->`. *)
        vecToRep[variables_, values_] :=
            Table[variables[[i]] -> values[[i]], {i, 1, Length[values]}];

        newtarget[vals_, best_] := best + scal * (best - Min[vals]);

        refreshtarget[vals_] :=
            Module[{best, second},
                {best, second} = Ordering[vals, 2];
                Return[vals[[best]] + scal * (vals[[best]] - vals[[second]])];
            ];

        maxit = OptionValue[LocalMaxIterations];
        autoTarget = OptionValue[Target] === Automatic;
        If[Not[autoTarget],
            target = OptionValue[Target];
        ];
        dim = Length[vars];
        steps = {};
        messages = {};
        nfev = 0;

        PrintLog[3, "Dimensions: ", dim, "; variables: ", vars,
            "; iterations: ", iterationCount];

        (* Fetch the startpoint option, and if it's set to Automatic, use a
        rewrite-rule to make some random vectors in the range [0, 2). *)
        {x0, x1} = OptionValue[startpoint] /.
            Automatic -> Table[RandomReal[2, dim], {i, 2}];

        val1 = Quiet[
            FindMinimum[expr, Table[{vars[[i]], x0[[i]]}, {i, 1, dim}],
                MaxIterations -> maxit,
                EvaluationMonitor -> Hold[nfev = nfev + 1]]
        ];

        PrintLog[3, "Initial local minimum: ", val1];

        (* Set two past minima: one being the startpoint for the first local
        minimization, the other actually being the minimum found starting
        from there. *)
        pastMinima = {{N[expr /. vecToRep[vars, x0]], x0}, solToVec[val1]};
        PrintLog[2, pastMinima];
        If[autoTarget,
            target = val1[[1]] - firsttargetratio * Abs[pastMinima[[1]][[1]] - val1[[1]]];
            PrintLog[3, "Initial target value: ", target];
        ];

        iterate = x1;
        iteratePositions = {iterate};
        For[iterationNumber = 1, iterationNumber <= iterationCount,
                iterationNumber++,
            PrintLog[3, "Iteration #", iterationNumber];
            PrintLog[3, expr, " ",
                Table[{vars[[i]], iterate[[i]]}, {i, 1, dim}]];
            PrintLog[2, "Iterate: ", iterate];

            (* localMinimum[[1]] is the minimum, localMinimum[[2]] is the arg
            min. *)
            localMinimum = Quiet[
                FindMinimum[ReleaseHold[expr], Table[{vars[[i]], iterate[[i]]}, 
                    {i, 1, dim}], MaxIterations -> maxit,
                    EvaluationMonitor -> Hold[nfev = nfev + 1]]
            ] // solToVec;
            PrintLog[2, localMinimum];

            (* how far is the current local minimum from the target *)
            delta = localMinimum[[1]] - target;

            If [delta < 0, (* found solution, update target *)
                    (* The given target was beaten. Thus we can exit. *)
                If[autoTarget,
                    target = newtarget[pastMinima[[1;;, 1]], localMinimum[[1]]];
                    delta = localMinimum[[1]] - target, (* Update *)
                (* else we beat the user-given target and can quit *)
                    AppendTo[messages, "user-defined target achieved"];
                    Break[];
                ];
            ];

            (* Find the nearest past local minimum. *)
            near = Nearest[pastMinima[[1;;, 2]], localMinimum[[2]], 2];

            (* If the true nearest one is too near, then take the
            second-nearest one. *)
            If [Norm[(near[[1]] - localMinimum[[2]])] < tol,
                nnear = near[[2]],
                nnear = near[[1]]
            ];

            (* Calculate the function-value of the nearest past minimum. *)
            fnear = ReleaseHold[expr] /. vecToRep[vars, nnear];
            (* See how far away it is from the target. *)
            deltan = fnear - target;

            PrintLog[2,"delta: ", delta, "; deltan: ", deltan,
                "; pmin: ", localMinimum[[2]], "; nnear: ", nnear, ";"];

            (* If the past minimum is too close in y-value to the current past
            minimum, then we're in a flat landscape; to avoid jumping too far
            away, we use the pseudo value instead of the difference between
            these deltas. *)
            If[(delta - deltan)^2 < pseudo^2,
                PrintLog[1,"Past minimum too close in y. Using pseudo."];
                step =- (delta / pseudo)
                    * (localMinimum[[2]] - nnear),
                step =- (delta / (delta - deltan))
                    * (localMinimum[[2]] - nnear)
            ];

            iterate = iterate + step; (* Take the step. *)

            (* Keep track of steps taken by the iterate. *)
            AppendTo[steps, step];
            (* Keep track of previously discovered local minima. *)
            AppendTo[pastMinima, localMinimum];
            (* Keep track of the iterate position over time. *)
            AppendTo[iteratePositions, iterate];

            If[Norm[step] < tol,
                Module[{oldtarget},
                    oldtarget = target;
                    refreshtarget[pastMinima[[1 ;;, 1]]];
                    If[(oldtarget - target)^2 < tol^2,
                        AppendTo[messages, "Stuck"];
                        Break[];
                    ];
                ];
            ];

            If[Not[autoTarget] &&
                    Divisible[iterationNumber, OptionValue[refreshrate]],
                Module[{oldtarget},
                    (* Update target *)
                    oldtarget = target;
                    target = refreshtarget[pastMinima[[1 ;;, 1]]];
                    If [((oldtarget - target) / target)^2 > 0.1^2,
                    PrintLog[1, "refreshed target to ", target]];
                ];
            ];
        ];

        If[iterationNumber > iterationCount,
            AppendTo[messages, "Requested number of iterations completed."]
            PrintLog[1, "Returning best value: ",
                "requested number of iterations completed."];
        ]

        (* a nasty hack to force pastMinima to be fully evaluated. *)
        pastMinima;

        (* Return the best value so far *)
        bestMinimum = Sort[pastMinima, #1[[1]] < #2[[1]] &][[1]];
        Return[<|
            "iterate" -> iteratePositions,
            "minima" -> pastMinima,
            "x" -> bestMinimum[[2]],
            "fun" -> bestMinimum[[1]],
            "steps" -> steps,
            "messages" -> messages,
            "nfev" -> nfev
        |>];
    ]; (* End of DifferenceMapOptimizer function *)

(* Set up the default arguments for the DifferenceMapOptimizer function. *)
Options[DifferenceMapOptimizer] = {
        startpoint -> Automatic,
        refreshrate -> 10,
        verbosity -> 0,
        LocalMaxIterations -> 100,
        Target -> Automatic
};

End[];

EndPackage[];
