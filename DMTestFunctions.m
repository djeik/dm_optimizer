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



EndPackage[];

(* vim: set filetype=mma: *)
