BeginPackage["DMUtils`"];

RandomAwayFrom[point_, distance_] := 
    With[{rvec = Map[RandomReal[{-1, 1}] &, point]},
        point + (distance / Norm[rvec]) rvec
    ];

ShiftOverBy[vect_, fun_] := fun[# - vect] &; 

EndPackage[];

(* vim: set filetype=mma: *)
