BeginPackage["DMTestFunctions`"];

griewankN[v_List /; And @@ Map[NumberQ, v]] :=
    Module[{a},
        a = 1 / 4000 v . v - Times @@ MapIndexed[Cos[#1 / Sqrt[#2[[1]]]] &, v] + 1;
        Sow[{evalN, a}];
        evalN += 1;
        a
    ];

EndPackage[];
