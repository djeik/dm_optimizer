BeginPackage["DMTestFunctions`"];

griewankN[v_List /; And @@ Map[NumberQ, v]] :=
    Module[{a},
        a = 1 / 4000 v . v - Times @@ MapIndexed[Cos[#1 / Sqrt[#2[[1]]]] &, v] + 1;
        Sow[{evalN, a}];
        evalN += 1;
        a
    ];

rosenbrockN[v_List /; And @@ Map[NumberQ, v]] :=
    Sum[(1 - v[[i]])^2 + 100 (v[[i+1]] - v[[i]]^2)^2, {i, 1, Length[v] - 1}];

EndPackage[];
