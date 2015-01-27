#!/usr/bin/env mscript

Needs["DMTestFunctions`"];

settings = Import["!cat", "JSON"];

x = "point" /. settings;
dim = "dim" /. settings;

vars = Table[xx[i], {i, dim}];

minimum = Quiet[FindMinimum[
    griewankN @ vars,
    Table[{vars[[i]], x[[i]]}, {i, 1, dim}],
    (* Careful: this constant needs to agree with the one used in in DifferenceMapOptimizer.m *)
    MaxIterations -> 20 
]];

Export["!cat", {"fun" -> minimum[[1]], "x" -> vars /. minimum[[2]]}, "JSON"]
