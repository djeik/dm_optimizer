(* ::Package:: *)

BeginPackage["DMUtils`"];

ClearAll[RandomAwayFrom];
RandomAwayFrom::usage = "Take a random point a given distance away from " <>
        "another";

ClearAll[ShiftOverBy];
ShiftOverBy::usage = "Move a function over in its codomain by a given vector.";

ClearAll[ParseNumber];
ParseNumber::usage = "Convert a string to a number.";

ClearAll[TrackExpr];
TrackExpr::usage = "Higher-order function. Given a function and an " <>
        "expression, evaluate the expression " <>
        "before every function evaluation.";

ClearAll[ToAssociations];
ToAssociations::usage = "Convert a nested list-of-rules structure into "
        "nested associations.";

Begin["Private`"];

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

TrackExpr[fun_, expr_] := With[{},
    ReleaseHold[expr];
    fun @ ##
] &;

End[];

EndPackage[];

(* vim: set filetype=mma: *)
