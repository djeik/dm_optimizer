#!/usr/bin/env mscript

Needs["DMData`"];

Export[
    "!cat", {
        "results" -> results,
        "settings" -> {
            "dimensions" -> dim,
            "iterations" -> niter,
            "tolerance" -> tolerance,
            "run_count" -> runCount,
            "inner_iterations" -> innerNiter,
            "range" -> range,
            "second_point_distance" -> secondPointDistance
        }
    },
    "JSON"
];
