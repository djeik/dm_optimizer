BeginPackage["DMTestFunctions`"];

ClearAll[griewankN];
griewankN[v_List /; And @@ Map[NumberQ, v]] :=
    With[{N = Length[v]},
        1 / 4000 v . v - Product[Cos[v[[i]] / Sqrt[i]], {i, 1, N}] + 1
    ];

ClearAll[rosenbrockN];
rosenbrockN[v_List /; And @@ Map[NumberQ, v]] :=
    Sum[(1 - v[[i]])^2 + 100 (v[[i+1]] - v[[i]]^2)^2, {i, 1, Length[v] - 1}];

ClearAll[schwefelN];
schwefelN[v_List /; And @@ Map[NumberQ, v]] :=
    418.9828872724339 Length[v]
        - Sum[v[[i]] Sin[Sqrt[Abs[v[[i]]]]], {i, 1, Length[v]}];

ClearAll[schafferN];
schafferN[v_List /; And @@ Map[NumberQ, v]] :=
    Block[{i},
        Sum[(v[[i]]^2 + v[[i+1]]^2)^0.25
            * (Sin[50 (v[[i]]^2 + v[[i+1]]^2)^0.10]^2 + 1),
            {i, 1, Length[v] - 1}]
    ];

ClearAll[rastriginN];
rastriginN[v_List /; And @@ Map[NumberQ, v]] :=
    With[{N = Length[v]},
        10.0 N Sum[v[[i]]^2.0 - 10.0 Cos[2.0 Pi v[[i]]], {i, 1, N}]
    ];

ClearAll[ackleyN];
ackleyN[v_List /; And @@ Map[NumberQ, v]] :=
    With[{N = Length[v]},
        20.0 -
        20.0 Exp[-0.2 Sqrt[(1/N) Sum[v[[i]]^2, {i, 1, N}]]] +
        Exp[1] -
        Exp[(1.0/N) Sum[Cos[2 Pi v[[i]]], {i, 1, N}]]
    ];

ClearAll[testFunctions];
testFunctions = {
    {"griewank", griewankN, {-1000.0, 1000.0}, 0.0},
    {"rosenbrock", rosenbrockN, {-1000.0, 1000.0}, 0.0},
    {"schwefel", schwefelN, {-500.0, 500.0}, 0.0},
    {"schaffer", schafferN, {-100.0, 100.0}, 0.0},
    {"ackley", ackleyN, {-15.0, 30.0}, 0.0}
};

ClearAll[testFunctionLabels];
testFunctionLabels = {"name", "function", "range", "optimum"};

ClearAll[asRules];
asRules[fd_] := MapThread[#1 -> #2 &, {testFunctionLabels, fd}];

EndPackage[];

(* vim: set filetype=mma: *)
