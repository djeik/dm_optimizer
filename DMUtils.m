BeginPackage["DMUtils`"];

ClearAll[RandomAwayFrom];
RandomAwayFrom[point_, distance_] :=
    With[{rvec = Map[RandomReal[{-1, 1}] &, point]},
        point + (distance / Norm[rvec]) rvec
    ];

ClearAll[ShiftOverBy];
ShiftOverBy[vect_, fun_] := fun[# - vect] &;

ClearAll[ParseNumber];
ParseNumber[s_] := With[{ss = StringToStream[s]},
    n = Read[ss, Number];
    Close[ss];
    Return[n];
];

ClearAll[ToAssociations];
ToAssociations[rules_] :=
    Replace[rules, x : {__Rule} :> Association[x], {0, Infinity}];

EndPackage[];

(* vim: set filetype=mma: *)
