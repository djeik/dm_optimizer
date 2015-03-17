#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import test_functions as tf

import json
import sys

from collections import defaultdict

def collect(runs, projection, reducer):
    return reducer(runs, projection)

collect_average = lambda runs, projection: sum(
        projection(r) for r in runs) / float(len(runs))

collect_average_c = lambda s: lambda runs: collect_average(
        runs, lambda r: r[s])

collect_average_solution = collect_average_c('fun')
collect_average_nfev = collect_average_c('nfev')

collect_minimum = lambda runs, projection: min(projection(r) for r in runs)

collect_minimum_c = lambda s: lambda runs: collect_minimum(
        runs, lambda r: r[s])

#
collect_minimum_solution = collect_minimum_c('fun')

# Filter the runs according a projection function Run -> Bool
collect_filter = lambda runs, projection: filter(projection, runs)

# Count the runs passing a filter Run -> Bool
collect_filter_count = lambda runs, p: len(collect_filter(runs, p))

# Count the runs whose value is at most a given `t`.
collect_filter_count_success = lambda t, runs: \
        collect_filter_count(runs, lambda r:
                r['fun']**2 <= t**2)

# Given a threshold `t`, construct a function to compute the fraction of runs
# that are successful.
collect_frac_success = lambda t: lambda runs: \
        collect_filter_count_success(t, runs) / float(len(runs))

make_collectors = lambda (t,): [
        ('average_solution', collect_average_solution),
        ('minimum_solution', collect_minimum_solution),
        ('average_nfev', collect_average_nfev),
        ('frac_success', collect_frac_success(t))
    ]

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)

    solvers = data.keys()
    functions = data['dm'].keys()

    # For each function, construct its collector, and store it into a dict
    # Function Name -> Collector
    collectors = dict(map(
            lambda fn: (fn, make_collectors( (
                tf.tests_dict[fn]['optimum'] + tf.SUCCESS_EPSILON ,) )),
            functions))

    results = defaultdict(lambda: defaultdict(dict))

    for solver_name, solver_data in data.items():
        for function_name, function_runs in solver_data.items():
            results[solver_name][function_name] = dict(map(
                lambda (n, f): (n, f(function_runs)),
                collectors[function_name]))

    print json.dumps(results, indent=2)
