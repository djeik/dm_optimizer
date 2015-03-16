BeginPackage["DMTestFunctions`"];

griewankN[v_List /; And @@ Map[NumberQ, v]] := 1 / 4000 v . v
            - Times @@ MapIndexed[Cos[#1 / Sqrt[#2[[1]]]] &, v]
            + 1;

rosenbrockN[v_List /; And @@ Map[NumberQ, v]] :=
    Sum[(1 - v[[i]])^2 + 100 (v[[i+1]] - v[[i]]^2)^2, {i, 1, Length[v] - 1}];

schwefelN[v_List /; And @@ Map[NumberQ, v]] :=
    418.9828872724339 Length[v]
        - Sum[v[[i]] Sin[Sqrt[Abs[v[[i]]]]], {i, 1, Length[v]}];

schafferN[v_List /; And @@ Map[NumberQ, v]] :=
    Block[{i},
        Sum[(v[[i]]^2 + v[[i+1]]^2)^0.25
            * (Sin[50 (v[[i]]^2 + v[[i+1]]^2)^0.10]^2 + 1),
            {i, 1, Length[v] - 1}]
    ];

rastriginN[v_List /; And @@ Map[NumberQ, v]] :=
    With[{N = Length[v]},
        10.0 N Sum[v[[i]]^2.0 - 10.0 Cos[2.0 Pi v[[i]]], {i, 1, N}]
    ];

ackleyN[v_List /; And @@ Map[NumberQ, v]] :=
    With[{N = Length[v]},
        20.0 -
        20.0 Exp[-0.2 Sqrt[(1/N) Sum[v[[i]]^2, {i, 1, N}]]] +
        Exp[1] -
        Exp[(1.0/N) Sum[Cos[2 Pi v[[i]]], {i, 1, N}]]
    ];

EndPackage[];

(* vim: set filetype=mma: *)
