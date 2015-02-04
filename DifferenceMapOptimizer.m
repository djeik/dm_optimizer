(* ::Package:: *)

BeginPackage["DifferenceMapOptimizer`"];

DifferenceMapOptimizer::usage = "Optimize a given expression of some given variables, for a given number of iterations using a given tolerance.";

Begin["Private`"];

pseudo = 0.0001;
firsttargetratio = 0.9;
scal = 0.05; (* The greediness in updating target *)

(* This function follows the mathematica syntax. expr is an expression that
depends on variables vars. The search will run for niter generations, tol is
the minimum distance between neighboring minima to be considered distinct. *)
DifferenceMapOptimizer[expr_, vars_, iterationCount_, tol_, OptionsPattern[]] :=
    Module[{x0, x1, val1, iterate, sol1, localMinimum, delta, target, nnear, near,
            deltan, fnear, step, dim, iterationNumber, maxit, bestMinima, pastMinima, iteratePositions,
            steps, messages, verboseLevel},
        (* Convenience function for printing things out depending on how verbose we want to be, governed by a global verbosity level. *)
        verboseLevel = OptionValue[verbosity]; (* By default, we don't want to see any messages except errors, whose priority levels should be negative. *)

        PrintLog[priority_, messages__] :=
            If[priority <= verboseLevel,
                Write[Streams["stderr"], messages]];

        (* Solutions produced by FindMinimum are of the form {functionValue,
        variableAssociation}, where the variableAssociation is a list like
        {x[[1]] -> 3, x[[2]] -> 5, etc.}
        solToVec converts this association into a vector like {3, 5, etc.}
        *)
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

        maxit = 20; (* Maximum number of iterations for the local minimizer. *)
        dim = Length[vars];
        steps = {};
        messages = {};

        PrintLog[3, "Dimensions: ", dim, "; variables: ", vars, "; iterations: ", iterationCount];

        (* Fetch the startpoint option, and if it's set to Automatic, use a
        rewrite-rule to make some random vectors. *)
        {x0, x1} = OptionValue[startpoint] /. Automatic -> Table[RandomReal[2, dim], {i, 2}];

        val1 = Quiet[
            FindMinimum[expr, Table[{vars[[i]], x0[[i]]}, {i, 1, dim}], MaxIterations -> maxit]
        ];

        PrintLog[3, "Initial local minimum: ", val1];

        (* Set two past minima: one being the startpoint for the first local minimization, the
         * other actually being the minimum found starting from there. *)
        pastMinima = {{N[expr /. vecToRep[vars, x0]], x0}, solToVec[val1]};
        PrintLog[2, pastMinima];
        target = firsttargetratio * val1[[1]];
        PrintLog[3, "Initial target value: ", target];

        iterate = x1;
        iteratePositions = {iterate};
        For[iterationNumber = 1, iterationNumber <= iterationCount, iterationNumber++,
            PrintLog[3, "Iteration #", iterationNumber];
            PrintLog[3, expr, " ", Table[{vars[[i]], iterate[[i]]}, {i, 1, dim}]];
            PrintLog[2, "Iterate: ", iterate];

            localMinimum = Quiet[
                FindMinimum[ReleaseHold[expr], Table[{vars[[i]], iterate[[i]]}, {i, 1, dim}], MaxIterations -> maxit]
            ] // solToVec;
            PrintLog[2, localMinimum];

            delta = localMinimum[[1]] - target; (* how far is the current local minimum from the target *)

            If [delta < 0, (* found solution, update target *)
                target = newtarget[pastMinima[[1;;, 1]], localMinimum[[1]]];
                delta = localMinimum[[1]]-target (* Update *)
            ];

            near = Nearest[pastMinima[[1;;, 2]], localMinimum[[2]], 2];

            (* If the true nearest one is too near, then take the second-nearest one. *)
            If [Norm[(near[[1]] - localMinimum[[2]])] < tol,
                nnear = near[[2]],
                nnear = near[[1]]
            ];

            fnear = ReleaseHold[expr] /. vecToRep[vars, nnear];
            deltan = fnear - target;

            PrintLog[2,"delta: ", delta, "; deltan: ", deltan, "; pmin: ", localMinimum[[2]], "; nnear: ", nnear, ";"];
            If[(delta - deltan)^2 < pseudo^2,
                PrintLog[1,"Past minimum too close in y. Using pseudo."];
                step =- (delta / pseudo) * (localMinimum[[2]] - nnear),
                step =- (delta / (delta - deltan)) * (localMinimum[[2]] - nnear)
            ];

            iterate = iterate + step; (* Take the step. *)

            AppendTo[steps, step]; (* Keep track of steps taken by the iterate. *)
            AppendTo[pastMinima, localMinimum]; (* Keep track of previously discovered local minima. *)
            AppendTo[iteratePositions, iterate]; (* Keep track of the iterate position over time. *)

            If[Norm[step]<tol,
                AppendTo[messages, "Fixed-point found."];
                PrintLog[1,"Step size really small (fixed-point). Returning current value."];
                Break[];
            ];

            If[iterationNumber ~Mod~ OptionValue[refreshrate] == 0,
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
            PrintLog[1, "Returning best value; requested number of iterations completed."];
        ]

        (* Return the best value so far *)
        bestMinima = Sort[pastMinima, #1[[1]] < #2[[1]] &][[{1, 2}]];
        Return[{
            "iterate" -> iteratePositions,
            "minima" -> pastMinima,
            "x" -> bestMinima[[2]],
            "fun" -> bestMinima[[1]],
            "steps" -> steps,
            "messages" -> messages
        }];
    ]; (* End of downvars function *)

(* Set up the default arguments for the downvars function. *)
Options[DifferenceMapOptimizer] = {startpoint -> Automatic, refreshrate -> 10, verbosity -> 0};

End[];

EndPackage[];
