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

ClearAll[goldsteinPrice2];
goldsteinPrice2[v_List /; And @@ Map[NumberQ, v]] :=
    With[{
        u = 1 + (v[[1]] + v[[2]] + 1)^2 * (19 - 14*v[[1]] + 3*v[[1]]^2 -
            14*v[[2]] + 6*v[[1]]*v[[2]] + 3 * v[[2]]^2),
        v = 30 + (2*v[[1]] - 3*v[[2]])^2 * (18 - 32*v[[1]] + 12*v[[1]]^2 +
            48*v[[2]] - 36*v[[1]]*v[[2]] + 27*v[[2]]^2)
        },
        v * u
    ];

ClearAll[f1];
f1[v_List /; And @@ Map[NumberQ, v]] := 0.2 * (v[[1]]^2 + v[[2]]^2) / 2 + 10 * Sin[v[[1]] + v[[2]]]^2 + 10 * Sin[100 * (v[[1]] - v[[2]])]^2

ClearAll[h4];
h4 = hN[4];

ClearAll[testFunctions];
testFunctions = {
    {"griewank", griewankN, {-1200.0, 800.0}, 0.0, 4},
    {"rosenbrock", rosenbrockN, {-1200.0, 800.0}, 0.0, 4},
    {"schaffer", schafferN, {-175.0, 75.0}, 0.0, 4},
    {"ackley", ackleyN, {-15.0, 30.0}, 0.0, 4},
    {"h4", h4, {-100.0, 150.0}, 0.0, 4},
    {"f1", f1, {-100.0, 100.0}, 0.0, 2}
};

ClearAll[testFunctionLabels];
testFunctionLabels = {"name", "function", "range", "optimum", "dim"};

ClearAll[asRules];
asRules[fd_] := MapThread[#1 -> #2 &, {testFunctionLabels, fd}];

Begin["Private`"];

ClearAll[hN];
hN[n_] := With[{random = RandomReal[{0, 1}, n]},
    Function[x,
        .2 Plus @@ (random*(x)^2) + 10 Sin[Dot[random*x^2, x]]^2 + 10 Sin[10 Dot[random*x, x]]^2
    ]
];

End[];

EndPackage[];

(* vim: set filetype=mma: *)
