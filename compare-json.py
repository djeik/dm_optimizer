#!/usr/bin/env python

""" For two JSON files A and B structured like
    {
        function_name_1: [
            run_results_1,
            run_results_2,
            ...,
            run_results_n
        ],
        function_name_2: ...
    }
    and each representing a solver or version of a solver, construct a JSON file
    structured like
    {
        "statistic_names": [
            statistic_name_1,
            statistic_name_2,
            ...,
            statistic_name_n
        ],
        "statistics": {
            function_name_1: {
                A : {
                    statistic_name_1: statistic_value_A_1,
                    statistic_name_2: statistic_value_A_2,
                    ...,
                    statistic_name_n: statistic_value_A_n
                },
                B : {
                    statistic_name_1: statistic_value_B_1,
                    statistic_name_2: statistic_value_B_2,
                    ...,
                    statistic_name_n: statistic_value_B_n
                }
            },
            function_name_2: ...
        }
    }
    representing the comparison of those two solvers.
"""

from __future__ import print_function

import sys
import json
import os

import numpy as np
import matplotlib.pyplot as plt

# The command line interface for this comparison generator is to list the
# solver name, then the filename, e.g.
# $ ./compare-json.py python.json Python mathematica.json Mathematica
# The switch --infer-name will make compare-json.py use the filename minus any
# file extension as the solver name.

class CLIError(Exception):
    pass

if __name__ == "__main__":
    infer_name = False
    collected_args = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        n = lambda: sys.argv[i+1]
        if arg == "--infer-name":
            infer_name = True
        else:
            collected_args.append(arg)
        i += 1

    if len(collected_args) % 2 != 0:
        raise CLIError()

    data = [] # A listified dict

    if infer_name: # Then all args are paths
        for arg in collected_args:
            basename = os.path.basename(arg)
            if basename.endswith(".json"): # strip '.json' if present
                basename = basename[:-5]
            with open(arg, 'r') as f:
                data.append( (basename, json.load(f)) )

    # Get a list of the functions tested
    tested_functions = data[0][1].keys()

    # create a dict: function -> solver -> average solution
    results = {}

    for tested_fun in tested_functions:
        results[tested_fun] = {}
        for solver_name, run_data in data:
            results[tested_fun][solver_name] = \
                    sum(r['fun'] for r in run_data[tested_fun]) / len(run_data[tested_fun])

    print(json.dumps(results, indent=1))

