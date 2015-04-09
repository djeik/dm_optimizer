BeginPackage["DMUtils`"];

RandomAwayFrom[point_, distance_] :=
    With[{rvec = Map[RandomReal[{-1, 1}] &, point]},
        point + (distance / Norm[rvec]) rvec
    ];

ShiftOverBy[vect_, fun_] := fun[# - vect] &;

ParseNumber[s_] := With[{ss = StringToStream[s]},
    n = Read[ss, Number];
    Close[ss];
    Return[n];
];

ToAssociations[rules_] :=
    Replace[rules, x : {__Rule} :> Association[x], {0, Infinity}];

EndPackage[];

(* vim: set filetype=mma: *)
